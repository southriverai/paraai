"""Map builder: estimate climb strength map from elevation using a trained CNN."""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from paraai.map.map_builder_base import MapBuilderBase
from paraai.map.map_estimate_net import MapEstimateNet
from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_datasets import RepositoryDatasets
from paraai.repository.repository_terrain import RepositoryTerrain

logger = logging.getLogger(__name__)


def _latlon_to_bbox(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """Bounding box around (lat, lon) with radius_m in meters."""
    radius_km = radius_m / 1000.0
    deg_lat = radius_km / 111.0
    deg_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat


def _extract_elevation_patch(
    elevation: np.ndarray,
    transform: rasterio.Affine,
    lat: float,
    lon: float,
    radius_m: float,
    size: int,
) -> torch.Tensor:
    """Extract elevation patch centered on (lat, lon), resize to (1, size, size), normalize to [0,1]."""
    lon_min, lat_min, lon_max, lat_max = _latlon_to_bbox(lat, lon, radius_m)
    from rasterio.windows import from_bounds

    win = from_bounds(lon_min, lat_min, lon_max, lat_max, transform)
    h, w = elevation.shape
    r0 = int(max(0, win.row_off))
    c0 = int(max(0, win.col_off))
    r1 = int(min(h, win.row_off + win.height))
    c1 = int(min(w, win.col_off + win.width))
    patch = elevation[r0:r1, c0:c1].astype(np.float32)
    valid = ~np.isnan(patch) & (patch > -500)
    if np.any(valid):
        lo, hi = np.percentile(patch[valid], [2, 98])
        patch = np.clip((patch - lo) / (hi - lo + 1e-10), 0, 1)
    else:
        patch = np.zeros_like(patch)
    patch = np.nan_to_num(patch, nan=0.0).astype(np.float32)
    if patch.size == 0:
        return torch.zeros(1, size, size, dtype=torch.float32)
    t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(torch.float32)
    t = nn.functional.interpolate(t, size=(size, size), mode="bilinear").squeeze(0)
    return t


class _MapDataset(Dataset):
    """Dataset of (input_map, target_map) pairs as tensors."""

    def __init__(
        self,
        input_maps: list[torch.Tensor],
        target_maps: list[torch.Tensor],
    ) -> None:
        self.input_maps = input_maps
        self.target_maps = target_maps
        assert len(self.input_maps) == len(self.target_maps)

    def __len__(self) -> int:
        return len(self.input_maps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_maps[idx], self.target_maps[idx]


class MapBuilderEstimateNet(MapBuilderBase):
    """
    Map builder that trains a CNN to estimate strength from elevation patches.

    Uses strength from the DataFrame at each point as target. Takes a DataFrame with lat, lon, strength.
    """

    def __init__(
        self,
        patch_size_m: float = 500.0,
        image_size: int = 64,
        grid_stride: int = 16,
        *,
        epochs: int = 5,
        batch_size: int = 8,
        lr: float = 1e-3,
        return_model: str = "lowest_test",
        model_path: Path | str | None = None,
    ) -> None:
        super().__init__(name="MapBuilderEstimateNet", output_map_names=["strength"])
        self.patch_size_m = patch_size_m
        self.image_size = image_size
        self.grid_stride = grid_stride
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.return_model = return_model
        self.model_path = Path(model_path) if model_path else None

    def get_cache_params(self) -> dict:
        return {
            "patch_size_m": self.patch_size_m,
            "image_size": self.image_size,
            "grid_stride": self.grid_stride,
        }

    def get_model_cache_params(self) -> dict:
        """Params for model cache key (excludes split params)."""
        return {
            "patch_size_m": self.patch_size_m,
            "image_size": self.image_size,
            "grid_stride": self.grid_stride,
        }

    def _build_dataset(
        self,
        bounding_box: BoundingBox,
        df_training: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[float],
        list[float],
        list[float],
        list[float],
        np.ndarray,
        rasterio.Affine,
        float,
        float,
    ]:
        """Build (input_maps_train, target_maps_train, input_maps_test, target_maps_test) from df_training and df_test."""
        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"]
        transform = terrain["transform"]

        strength_vals = df_training["strength"].to_numpy(dtype=np.float64)
        valid = strength_vals[~np.isnan(strength_vals) & (strength_vals > 0)]
        if valid.size > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            strength_lo, strength_hi = float(p2), float(p98)
        else:
            strength_lo, strength_hi = 0.0, 1.0
        if strength_hi <= strength_lo:
            strength_hi = strength_lo + 1e-6

        def _build_patches(
            df: pd.DataFrame,
        ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float], list[float]]:
            inputs: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            lats: list[float] = []
            lons: list[float] = []
            for _, row in df.iterrows():
                lat_val = float(row["lat"])
                lon_val = float(row["lon"])
                raw_val = float(row["strength"])
                patch = _extract_elevation_patch(elevation, transform, lat_val, lon_val, self.patch_size_m, self.image_size)
                norm_val = float(np.clip((raw_val - strength_lo) / (strength_hi - strength_lo), 0, 1))
                target = torch.full((1, self.image_size, self.image_size), norm_val, dtype=torch.float32)
                inputs.append(patch)
                targets.append(target)
                lats.append(lat_val)
                lons.append(lon_val)
            return inputs, targets, lats, lons

        input_maps_train, target_maps_train, lats_train, lons_train = _build_patches(df_training)
        input_maps_test, target_maps_test, lats_test, lons_test = _build_patches(df_test)

        return (
            input_maps_train,
            target_maps_train,
            input_maps_test,
            target_maps_test,
            lats_train,
            lons_train,
            lats_test,
            lons_test,
            elevation,
            transform,
            strength_lo,
            strength_hi,
        )

    def build_dataset(
        self,
        bounding_box: BoundingBox,
        df_training: pd.DataFrame,
        df_test: pd.DataFrame,
        *,
        ignore_cache: bool = False,
    ) -> dict:
        """
        Build dataset from df_training and df_test (split is made by the caller).

        Returns dict with: input_maps_train, target_maps_train, input_maps_test, target_maps_test,
        metadata (strength_lo, strength_hi, image_size, patch_size_m, grid_stride, bounding_box).
        """
        for df, name in [(df_training, "df_training"), (df_test, "df_test")]:
            if "lat" not in df.columns or "lon" not in df.columns:
                raise ValueError(f"{name} must have columns 'lat' and 'lon'")
            if "strength" not in df.columns:
                raise ValueError(f"{name} must have column 'strength'")

        if not ignore_cache:
            repo = RepositoryDatasets.get_instance()
            data = repo.get_dataset(
                self.name,
                bounding_box,
                **self.get_cache_params(),
            )
            if data is not None:
                n_train = len(data["input_maps_train"])
                n_test = len(data["input_maps_test"])
                logger.info("Loaded dataset from cache (%s train, %s test patches)", n_train, n_test)
                return data

        (
            input_maps_train,
            target_maps_train,
            input_maps_test,
            target_maps_test,
            lats_train,
            lons_train,
            lats_test,
            lons_test,
            elevation,
            transform,
            strength_lo,
            strength_hi,
        ) = self._build_dataset(bounding_box, df_training, df_test)

        data = {
            "input_maps_train": input_maps_train,
            "target_maps_train": target_maps_train,
            "input_maps_test": input_maps_test,
            "target_maps_test": target_maps_test,
            "lats_train": lats_train,
            "lons_train": lons_train,
            "lats_test": lats_test,
            "lons_test": lons_test,
            "metadata": {
                "strength_lo": strength_lo,
                "strength_hi": strength_hi,
                "image_size": self.image_size,
                "patch_size_m": self.patch_size_m,
                "grid_stride": self.grid_stride,
                "bounding_box": bounding_box.model_dump(),
            },
        }

        repo = RepositoryDatasets.get_instance()
        repo.save_dataset(data, self.name, bounding_box, **self.get_cache_params())
        logger.info(
            "Cached dataset (%s train, %s test patches) to repository",
            len(input_maps_train),
            len(input_maps_test),
        )
        return data

    def _train_model(
        self,
        input_maps_train: list[torch.Tensor],
        target_maps_train: list[torch.Tensor],
        input_maps_test: list[torch.Tensor],
        target_maps_test: list[torch.Tensor],
    ) -> MapEstimateNet:
        """Train CNN on pre-split train/test data. Returns best model."""
        train_ds = _MapDataset(input_maps_train, target_maps_train)
        test_ds = _MapDataset(input_maps_test, target_maps_test)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        in_c = input_maps_train[0].shape[0]
        out_c = target_maps_train[0].shape[0]
        model = MapEstimateNet(in_channels=in_c, out_channels=out_c, size=self.image_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training on %s", device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_test_loss = float("inf")
        best_test_state: dict | None = None

        chunk_size = 10
        for chunk_start in range(0, self.epochs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.epochs)
            n_chunk = chunk_end - chunk_start
            pbar = tqdm(range(n_chunk), desc=f"Epochs {chunk_start + 1}-{chunk_end}", unit="epoch")
            for _epoch in pbar:
                model.train()
                train_loss = 0.0
                n_train_batches = 0
                for inp_batch, tgt_batch in train_loader:
                    inp_dev = inp_batch.to(device)
                    tgt_dev = tgt_batch.to(device)
                    optimizer.zero_grad()
                    pred = model(inp_dev)
                    if pred.shape[2:] != tgt_dev.shape[2:]:
                        pred = nn.functional.interpolate(pred, size=tgt_dev.shape[2:], mode="bilinear")
                    loss = criterion(pred, tgt_dev)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    n_train_batches += 1
                train_loss /= n_train_batches if n_train_batches else 1

                model.eval()
                with torch.no_grad():
                    test_loss = 0.0
                    n_test_batches = 0
                    for inp_batch, tgt_batch in test_loader:
                        inp_dev = inp_batch.to(device)
                        tgt_dev = tgt_batch.to(device)
                        pred = model(inp_dev)
                        if pred.shape[2:] != tgt_dev.shape[2:]:
                            pred = nn.functional.interpolate(pred, size=tgt_dev.shape[2:], mode="bilinear")
                        test_loss += criterion(pred, tgt_dev).item()
                        n_test_batches += 1
                    test_loss /= n_test_batches if n_test_batches else 1

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_test_state = copy.deepcopy(model.state_dict())
                    if self.model_path:
                        self.model_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(best_test_state, self.model_path)
                        logger.info("Saved best model (test_loss=%.6f) to %s", best_test_loss, self.model_path)

                pbar.set_postfix(train_loss=f"{train_loss:.4f}", test_loss=f"{test_loss:.4f}")

        if self.return_model == "lowest_test" and best_test_state is not None:
            model.load_state_dict(best_test_state)
            logger.info("Loaded model with lowest test loss (%.6f)", best_test_loss)

        return model

    def _run_inference(
        self,
        model: MapEstimateNet,
        elevation: np.ndarray,
        transform: rasterio.Affine,
        strength_lo: float,
        strength_hi: float,
    ) -> np.ndarray:
        """Run model inference on full grid. Returns denormalized strength array."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        h, w = elevation.shape
        stride = self.grid_stride
        n_rows = (h + stride - 1) // stride
        n_cols = (w + stride - 1) // stride
        pred_sparse = np.zeros((n_rows, n_cols), dtype=np.float32)
        with torch.no_grad():
            for ri, r in enumerate(tqdm(range(0, h, stride), desc="Inference", unit="row")):
                for ci, c in enumerate(range(0, w, stride)):
                    lon, lat = rasterio.transform.xy(transform, r, c)
                    patch = _extract_elevation_patch(elevation, transform, float(lat), float(lon), self.patch_size_m, self.image_size)
                    patch_batch = patch.unsqueeze(0).to(device)
                    pred = model(patch_batch)
                    val = pred[0, 0, self.image_size // 2, self.image_size // 2].item()
                    pred_sparse[ri, ci] = val
        pred_t = torch.from_numpy(pred_sparse).unsqueeze(0).unsqueeze(0)
        pred_grid = (
            nn.functional.interpolate(pred_t, size=(h, w), mode="bilinear", align_corners=False).squeeze().numpy().astype(np.float32)
        )
        pred_grid = np.clip(pred_grid, 0, 1)
        return pred_grid * (strength_hi - strength_lo) + strength_lo

    def _build_impl(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame | None = None,
    ) -> dict[str, VectorMapArray]:
        """Build strength map by loading model from repository and running inference."""
        from paraai.repository.repository_models import RepositoryModels

        repo_models = RepositoryModels.get_instance()
        model_data = repo_models.get_model(self.name, **self.get_model_cache_params())
        if model_data is None:
            raise FileNotFoundError(f"Model not found in repository for {self.name}. Run train mode first to train and save the model.")

        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"]
        transform = terrain["transform"]

        # strength_lo, strength_hi for denormalization (from df if available)
        if not df.empty and "strength" in df.columns:
            strength_vals = df["strength"].to_numpy(dtype=np.float64)
            valid = strength_vals[~np.isnan(strength_vals) & (strength_vals > 0)]
            if valid.size > 0:
                p2, p98 = np.percentile(valid, [2, 98])
                strength_lo, strength_hi = float(p2), float(p98)
            else:
                strength_lo, strength_hi = 0.0, 1.0
        else:
            strength_lo, strength_hi = 0.0, 1.0
        if strength_hi <= strength_lo:
            strength_hi = strength_lo + 1e-6

        model = MapEstimateNet(
            in_channels=model_data["in_channels"],
            out_channels=model_data["out_channels"],
            size=model_data["image_size"],
        )
        model.load_state_dict(model_data["state_dict"])

        logger.info("Loaded model from repository, running inference")
        strength_arr = self._run_inference(model, elevation, transform, strength_lo, strength_hi)

        return {
            "strength": VectorMapArray(
                "strength",
                bounding_box,
                strength_arr,
                transform=transform,
            ),
        }
