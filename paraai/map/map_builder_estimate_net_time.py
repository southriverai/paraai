"""Map builder: estimate climb strength map from elevation using a trained CNN."""

from __future__ import annotations

import asyncio
import copy
import logging

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from paraai.map.map_builder_base import MapBuilderBase, MapEvaluateResult
from paraai.map.map_estimate_net import MapEstimateNet
from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_datasets import RepositoryDatasets
from paraai.repository.repository_models import RepositoryModels
from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.tools_datasets import split_dataframe

logger = logging.getLogger(__name__)


def _extract_elevation_patch(
    elevation: np.ndarray,
    transform: rasterio.Affine,
    lat: float,
    lon: float,
    radius_m: float,
    size: int,
) -> torch.Tensor:
    """Extract elevation patch centered on (lat, lon), resize to (1, size, size), normalize to [0,1]."""
    bbox = BoundingBox.from_latlon_radius(lat, lon, radius_m)
    lon_min, lat_min, lon_max, lat_max = bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max
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
    ) -> None:
        super().__init__(name="MapBuilderEstimateNet", output_map_names=["strength"])
        self.patch_size_m = patch_size_m
        self.image_size = image_size
        self.grid_stride = grid_stride
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.return_model = return_model

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

    def get_dataset_cache_params(self, test_frac: float, split_seed: int) -> dict:
        """Params for dataset cache key (includes split params)."""
        return {
            **self.get_cache_params(),
            "test_frac": test_frac,
            "split_seed": split_seed,
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
            desc: str = "Building patches",
        ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float], list[float]]:
            inputs: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            lats: list[float] = []
            lons: list[float] = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, unit="patch"):
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

        input_maps_train, target_maps_train, lats_train, lons_train = _build_patches(df_training, desc="Train patches")
        input_maps_test, target_maps_test, lats_test, lons_test = _build_patches(df_test, desc="Test patches")

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
        test_frac: float | None = None,
        split_seed: int | None = None,
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
            if "time_of_day_h" not in df.columns:
                raise ValueError(f"{name} must have column 'time_of_day_h'")
            if "time_of_year_d" not in df.columns:
                raise ValueError(f"{name} must have column 'time_of_year_d'")

        cache_params = (
            self.get_dataset_cache_params(test_frac, split_seed)
            if test_frac is not None and split_seed is not None
            else self.get_cache_params()
        )

        if not ignore_cache:
            repo = RepositoryDatasets.get_instance()
            data = repo.get_dataset(self.name, bounding_box, **cache_params)
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
        repo.save_dataset(data, self.name, bounding_box, **cache_params)
        logger.info(
            "Cached dataset (%s train, %s test patches) to repository",
            len(input_maps_train),
            len(input_maps_test),
        )
        return data

    def get_or_build_dataset(
        self,
        bounding_box: BoundingBox,
        test_frac: float = 0.2,
        split_seed: int = 42,
        *,
        ignore_cache: bool = False,
    ) -> dict:
        """Load dataset from cache or build from region. Returns dataset dict."""
        cache_params = self.get_dataset_cache_params(test_frac, split_seed)
        if not ignore_cache:
            repo = RepositoryDatasets.get_instance()
            data = repo.get_dataset(self.name, bounding_box, **cache_params)
            if data is not None:
                n_train = len(data["input_maps_train"])
                n_test = len(data["input_maps_test"])
                logger.info("Loaded dataset from cache (%s train, %s test patches)", n_train, n_test)
                return data

        logger.info("Dataset not in cache, building from region")
        climb_df = asyncio.run(RepositorySimpleClimb.get_instance().get_climb_dataframe(bounding_box))
        train_df, test_df = split_dataframe(climb_df, test_frac, split_seed)
        logger.info("Split dataset into train/test (%s train, %s test)", len(train_df), len(test_df))
        return self.build_dataset(bounding_box, train_df, test_df, ignore_cache=True, test_frac=test_frac, split_seed=split_seed)

    def _train_model(
        self,
        input_maps_train: list[torch.Tensor],
        target_maps_train: list[torch.Tensor],
        input_maps_test: list[torch.Tensor],
        target_maps_test: list[torch.Tensor],
    ) -> MapEstimateNet:
        """Train CNN on pre-split train/test data. Returns best model."""
        in_c = input_maps_train[0].shape[0]
        out_c = target_maps_train[0].shape[0]
        model = MapEstimateNet(in_channels=in_c, out_channels=out_c, size=self.image_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training on %s", device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Load all data to GPU once and keep there during training
        train_inp = torch.stack(input_maps_train).to(device)
        train_tgt = torch.stack(target_maps_train).to(device)
        test_inp = torch.stack(input_maps_test).to(device)
        test_tgt = torch.stack(target_maps_test).to(device)

        best_test_loss = float("inf")
        best_test_state: dict | None = None

        pbar = tqdm(range(self.epochs), desc="Training", unit="epoch")
        for _epoch in pbar:
            model.train()
            optimizer.zero_grad()
            pred = model(train_inp)
            if pred.shape[2:] != train_tgt.shape[2:]:
                pred = nn.functional.interpolate(pred, size=train_tgt.shape[2:], mode="bilinear")
            loss = criterion(pred, train_tgt)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            model.eval()
            with torch.no_grad():
                pred = model(test_inp)
                if pred.shape[2:] != test_tgt.shape[2:]:
                    pred = nn.functional.interpolate(pred, size=test_tgt.shape[2:], mode="bilinear")
                test_loss = criterion(pred, test_tgt).item()

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_state = copy.deepcopy(model.state_dict())

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

    def _run_inference_on_points(
        self,
        model: MapEstimateNet,
        elevation: np.ndarray,
        transform: rasterio.Affine,
        strength_lo: float,
        strength_hi: float,
        lats: np.ndarray,
        lons: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Run model inference on (lat, lon) points. Returns denormalized strength array."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        patches = [
            _extract_elevation_patch(elevation, transform, float(lat), float(lon), self.patch_size_m, self.image_size)
            for lat, lon in zip(lats, lons, strict=True)
        ]
        inp = torch.stack(patches).to(device)
        pred_values: list[float] = []
        with torch.no_grad():
            for i in range(0, len(inp), batch_size):
                batch = inp[i : i + batch_size]
                pred = model(batch)
                for j in range(pred.shape[0]):
                    val = pred[j, 0, self.image_size // 2, self.image_size // 2].item()
                    val = np.clip(val, 0, 1)
                    pred_values.append(float(val * (strength_hi - strength_lo) + strength_lo))
        return np.array(pred_values, dtype=np.float64)

    def _build_impl(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame | None = None,
    ) -> dict[str, VectorMapArray]:
        """Build strength map by loading model from repository and running inference."""

        repo_models = RepositoryModels.get_instance()
        model_data = repo_models.get_model(self.name, **self.get_model_cache_params())
        if model_data is None:
            raise FileNotFoundError(f"Model not found in repository for {self.name}. Run train mode first to train and save the model.")

        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"]
        transform = terrain["transform"]

        strength_lo = model_data.get("strength_lo")
        strength_hi = model_data.get("strength_hi")
        if strength_lo is None or strength_hi is None:
            raise ValueError("strength_lo and strength_hi must be set in model data")
        if strength_hi <= strength_lo:
            raise ValueError("strength_hi must be greater than strength_lo")

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

    def evaluate(
        self,
        vector_map: VectorMapArray,
        evaluate_df: pd.DataFrame,
        column_name: str,
        n_train: int,
    ) -> MapEvaluateResult:
        """Evaluate estimated climb maps on held-out points. Builds dataset from points and runs inference directly."""
        if len(evaluate_df) == 0:
            raise ValueError("evaluate_df is empty")
        if "lat" not in evaluate_df.columns or "lon" not in evaluate_df.columns or column_name not in evaluate_df.columns:
            raise ValueError("evaluate_df must have columns 'lat', 'lon', and 'column_name'")
        lats = evaluate_df["lat"].values
        lons = evaluate_df["lon"].values
        true_values = evaluate_df[column_name].values
        bounding_box = vector_map.bounding_box
        repo_models = RepositoryModels.get_instance()
        model_data = repo_models.get_model(self.name, **self.get_model_cache_params())
        if model_data is None:
            raise FileNotFoundError(f"Model not found in repository for {self.name}. Run train mode first.")
        strength_lo = model_data.get("strength_lo")
        strength_hi = model_data.get("strength_hi")
        if strength_lo is None or strength_hi is None:
            raise ValueError("strength_lo and strength_hi must be set in model data")
        if strength_hi <= strength_lo:
            raise ValueError("strength_hi must be greater than strength_lo")

        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"]
        transform = terrain["transform"]

        model = MapEstimateNet(
            in_channels=model_data["in_channels"],
            out_channels=model_data["out_channels"],
            size=model_data["image_size"],
        )
        model.load_state_dict(model_data["state_dict"])

        pred_values = self._run_inference_on_points(model, elevation, transform, strength_lo, strength_hi, lats, lons)
        errors = np.abs(pred_values - true_values)

        return MapEvaluateResult(
            strength_mae=float(np.mean(errors)),
            strength_rmse=float(np.sqrt(np.mean(errors**2))),
            n_train=n_train,
            n_holdout=len(evaluate_df),
            true_values=true_values,
            pred_values=pred_values,
        )
