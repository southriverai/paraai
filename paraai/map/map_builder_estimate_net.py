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
from paraai.map.map_builder_convolution import MapBuilderConvolution
from paraai.map.map_estimate_net import MapEstimateNet
from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
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

    Uses MapBuilderConvolution as target. Takes a DataFrame with lat, lon, count, strength.
    """

    def __init__(
        self,
        kernel_size_m: float = 200.0,
        patch_size_m: float = 500.0,
        image_size: int = 64,
        grid_stride: int = 16,
        *,
        epochs: int = 5,
        batch_size: int = 8,
        lr: float = 1e-3,
        test_frac: float = 0.2,
        split_seed: int = 42,
        return_model: str = "lowest_test",
        model_path: Path | str | None = None,
    ) -> None:
        super().__init__(name="MapBuilderEstimateNet", output_map_names=["strength"])
        self.kernel_size_m = kernel_size_m
        self.patch_size_m = patch_size_m
        self.image_size = image_size
        self.grid_stride = grid_stride
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.test_frac = test_frac
        self.split_seed = split_seed
        self.return_model = return_model
        self.model_path = Path(model_path) if model_path else None
        self._last_model: MapEstimateNet | None = None
        self._last_input_maps: list[torch.Tensor] | None = None
        self._last_target_maps: list[torch.Tensor] | None = None
        self._last_test_indices: list[int] | None = None
        self._last_viz_data: dict | None = None

    def get_cache_params(self) -> dict:
        return {
            "kernel_size_m": self.kernel_size_m,
            "patch_size_m": self.patch_size_m,
            "image_size": self.image_size,
            "grid_stride": self.grid_stride,
        }

    def _build_dataset(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], np.ndarray, rasterio.Affine, float, float, np.ndarray]:
        """Build (input_maps, target_maps) from df. Returns also elevation, transform, strength_lo, strength_hi, strength_arr."""
        target_builder = MapBuilderConvolution(kernel_size_m=self.kernel_size_m)
        maps = target_builder.build(bounding_box, df)
        strength_vma = maps["strength"]

        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"]
        transform = terrain["transform"]
        h, w = elevation.shape

        strength_arr = strength_vma.array
        valid = strength_arr[strength_arr > 0]
        if valid.size > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            strength_lo, strength_hi = float(p2), float(p98)
        else:
            strength_lo, strength_hi = 0.0, 1.0
        if strength_hi <= strength_lo:
            strength_hi = strength_lo + 1e-6

        input_maps: list[torch.Tensor] = []
        target_maps: list[torch.Tensor] = []

        for r in range(0, h, self.grid_stride):
            for c in range(0, w, self.grid_stride):
                lon, lat = rasterio.transform.xy(transform, r, c)
                lon_val, lat_val = float(lon), float(lat)
                patch = _extract_elevation_patch(
                    elevation, transform, lat_val, lon_val, self.patch_size_m, self.image_size
                )
                raw_val = strength_vma.get_value(lat_val, lon_val)
                norm_val = float(np.clip((raw_val - strength_lo) / (strength_hi - strength_lo), 0, 1))
                target = torch.full((1, self.image_size, self.image_size), norm_val, dtype=torch.float32)
                input_maps.append(patch)
                target_maps.append(target)

        return input_maps, target_maps, elevation, transform, strength_lo, strength_hi, strength_arr.copy()

    def build_dataset(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame,
    ) -> dict:
        """
        Build dataset (input_maps, target_maps) from df for training.

        Returns dict with: input_maps, target_maps, metadata (strength_lo, strength_hi,
        image_size, patch_size_m, grid_stride, kernel_size_m, bounding_box).
        """
        input_maps, target_maps, elevation, transform, strength_lo, strength_hi, strength_arr = (
            self._build_dataset(bounding_box, df)
        )
        return {
            "input_maps": input_maps,
            "target_maps": target_maps,
            "metadata": {
                "strength_lo": strength_lo,
                "strength_hi": strength_hi,
                "image_size": self.image_size,
                "patch_size_m": self.patch_size_m,
                "grid_stride": self.grid_stride,
                "kernel_size_m": self.kernel_size_m,
                "bounding_box": bounding_box.model_dump(),
            },
        }

    def _train_model(
        self,
        input_maps: list[torch.Tensor],
        target_maps: list[torch.Tensor],
    ) -> tuple[MapEstimateNet, list[int]]:
        """Train CNN and return (best model, test_indices)."""
        dataset = _MapDataset(input_maps, target_maps)
        n_total = len(dataset)
        n_test = max(1, int(n_total * self.test_frac))
        n_train = n_total - n_test
        train_ds, test_ds = torch.utils.data.random_split(
            dataset, [n_train, n_test], generator=torch.Generator().manual_seed(self.split_seed)
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        in_c = input_maps[0].shape[0]
        out_c = target_maps[0].shape[0]
        model = MapEstimateNet(in_channels=in_c, out_channels=out_c, size=self.image_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training on %s", device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_test_loss = float("inf")
        best_test_state: dict | None = None

        pbar = tqdm(range(self.epochs), desc="Training", unit="epoch")
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

        return model, list(test_ds.indices)

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
            for ri, r in enumerate(range(0, h, stride)):
                for ci, c in enumerate(range(0, w, stride)):
                    lon, lat = rasterio.transform.xy(transform, r, c)
                    patch = _extract_elevation_patch(
                        elevation, transform, float(lat), float(lon), self.patch_size_m, self.image_size
                    )
                    patch_batch = patch.unsqueeze(0).to(device)
                    pred = model(patch_batch)
                    val = pred[0, 0, self.image_size // 2, self.image_size // 2].item()
                    pred_sparse[ri, ci] = val
        pred_t = torch.from_numpy(pred_sparse).unsqueeze(0).unsqueeze(0)
        pred_grid = (
            nn.functional.interpolate(pred_t, size=(h, w), mode="bilinear", align_corners=False)
            .squeeze()
            .numpy()
            .astype(np.float32)
        )
        pred_grid = np.clip(pred_grid, 0, 1)
        return pred_grid * (strength_hi - strength_lo) + strength_lo

    def _build_impl(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame,
    ) -> dict[str, VectorMapArray]:
        """Build strength map by training CNN on elevation patches with MapBuilderConvolution targets."""
        if df.empty:
            raise ValueError("No points provided")
        if "lat" not in df.columns or "lon" not in df.columns:
            raise ValueError("DataFrame must have columns 'lat' and 'lon'")

        input_maps, target_maps, elevation, transform, strength_lo, strength_hi, strength_arr_raw = (
            self._build_dataset(bounding_box, df)
        )
        if not input_maps:
            raise ValueError("No patches extracted from region")

        logger.info("Dataset: %s patches", len(input_maps))
        model, test_indices = self._train_model(input_maps, target_maps)
        strength_arr = self._run_inference(model, elevation, transform, strength_lo, strength_hi)

        true_grid = np.clip(
            (strength_arr_raw.astype(np.float64) - strength_lo) / (strength_hi - strength_lo),
            0,
            1,
        ).astype(np.float32)

        self._last_model = model
        self._last_input_maps = input_maps
        self._last_target_maps = target_maps
        self._last_test_indices = test_indices
        self._last_viz_data = {
            "elevation": elevation,
            "transform": transform,
            "lon_min": bounding_box.lon_min,
            "lat_min": bounding_box.lat_min,
            "lon_max": bounding_box.lon_max,
            "lat_max": bounding_box.lat_max,
            "patch_size_m": self.patch_size_m,
            "image_size": self.image_size,
            "true_grid": true_grid,
            "strength_lo": strength_lo,
            "strength_hi": strength_hi,
        }

        return {
            "strength": VectorMapArray(
                "strength",
                bounding_box,
                strength_arr,
                transform=transform,
            ),
        }
