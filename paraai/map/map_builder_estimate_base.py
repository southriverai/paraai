"""Map builder: estimate climb strength map from elevation using a trained CNN."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
from tqdm import tqdm

from paraai.map.map_builder_base import MapBuilderBase, MapEvaluateResult
from paraai.map.map_estimate_net import MapEstimateNetSimple
from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_datasets import RepositoryDatasets
from paraai.repository.repository_terrain import RepositoryTerrain

logger = logging.getLogger(__name__)


def _compute_dataset_cache_hash(
    builder_name: str,
    bounding_box: BoundingBox,
    params: dict,
) -> str:
    """Hash for dataset cache key: builder + bbox + params."""
    data = {
        "builder": builder_name,
        "bbox": [bounding_box.lat_min, bounding_box.lat_max, bounding_box.lon_min, bounding_box.lon_max],
        "params": dict(sorted(params.items())),
    }
    s = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


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


class MapBuilderEstimateBase(MapBuilderBase):
    """
    Map builder that trains a CNN to estimate strength from elevation patches.

    Uses strength from the DataFrame at each point as target. Takes a DataFrame with lat, lon, strength.
    """

    def __init__(
        self,
        name: str,
        patch_size_m: float = 500.0,
        image_size: int = 64,
        grid_stride: int = 16,
        *,
        epochs: int = 5,
        batch_size: int = 8,
        lr: float = 1e-3,
        return_model: str = "lowest_test",
    ) -> None:
        super().__init__(name=name, output_map_names=["strength"])
        self.patch_size_m = patch_size_m
        self.image_size = image_size
        self.grid_stride = grid_stride
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.return_model = return_model

    def get_builder_id(self) -> str:
        """Builder identity string for cache keying."""
        params = {
            "patch_size_m": self.patch_size_m,
            "image_size": self.image_size,
            "grid_stride": self.grid_stride,
        }
        s = json.dumps(params, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def get_model_id(self) -> dict:
        """Params for cache key. Uses builder_id for estimate builders."""
        return {"builder_id": self.get_builder_id()}

    def get_cache_params(self) -> dict:
        """Params for cache key. Delegates to get_model_id."""
        return self.get_model_id()

    def get_model_cache_params(self) -> dict:
        """Params for model cache key (excludes split params)."""
        return {
            "patch_size_m": self.patch_size_m,
            "image_size": self.image_size,
            "grid_stride": self.grid_stride,
        }

    def _get_model_class(self) -> type[MapEstimateNetSimple]:
        raise NotImplementedError("Subclasses must implement _get_model_class")

    def get_dataset_cache_id(
        self,
        bounding_box: BoundingBox,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> str:
        """Compute dataset cache ID hash from builder params and train/test data."""
        params = {
            "builder_id": self.get_builder_id(),
            "train_df": train_df.to_dict(),
            "test_df": test_df.to_dict(),
        }
        return _compute_dataset_cache_hash(self.name, bounding_box, params)

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

        cache_id = self.get_dataset_cache_id(bounding_box, df_training, df_test)

        if not ignore_cache:
            repo = RepositoryDatasets.get_instance()
            data = repo.get_dataset(self.name, bounding_box, dataset_cache_id=cache_id)
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
        repo.save_dataset(data, self.name, bounding_box, dataset_cache_id=cache_id)
        logger.info(
            "Cached dataset (%s train, %s test patches) to repository",
            len(input_maps_train),
            len(input_maps_test),
        )
        return data

    def get_or_build_dataset(
        self,
        bounding_box: BoundingBox,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        ignore_cache: bool = False,
    ) -> dict:
        """Load dataset from cache or build from train_df and test_df. Returns dataset dict."""
        cache_id = self.get_dataset_cache_id(bounding_box, train_df, test_df)
        if not ignore_cache:
            repo = RepositoryDatasets.get_instance()
            data = repo.get_dataset(self.name, bounding_box, dataset_cache_id=cache_id)
            if data is not None:
                n_train = len(data["input_maps_train"])
                n_test = len(data["input_maps_test"])
                logger.info("Loaded dataset from cache (%s train, %s test patches)", n_train, n_test)
                return data

        logger.info("Dataset not in cache, building from region")
        return self.build_dataset(bounding_box, train_df, test_df, ignore_cache=True)

    def _train_model(
        self,
        input_maps_train: list[torch.Tensor],
        target_maps_train: list[torch.Tensor],
        input_maps_test: list[torch.Tensor],
        target_maps_test: list[torch.Tensor],
    ) -> MapEstimateNetSimple:
        """Train CNN on pre-split train/test data. Returns best model."""
        in_c = input_maps_train[0].shape[0]
        out_c = target_maps_train[0].shape[0]
        model = self._get_model_class()(in_channels=in_c, out_channels=out_c, size=self.image_size)
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

        t_start = time.perf_counter()
        for epoch in range(self.epochs):
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

            elapsed = time.perf_counter() - t_start
            eta_sec = (elapsed / (epoch + 1)) * (self.epochs - epoch - 1) if epoch < self.epochs - 1 else 0
            eta_str = f", ETA {eta_sec:.0f}s" if eta_sec > 0 else ""
            logger.info(
                "Epoch %d/%d: train_loss=%.4f test_loss=%.4f (%.1fs)%s",
                epoch + 1,
                self.epochs,
                train_loss,
                test_loss,
                elapsed,
                eta_str,
            )

        if self.return_model == "lowest_test" and best_test_state is not None:
            model.load_state_dict(best_test_state)
            logger.info("Loaded model with lowest test loss (%.6f)", best_test_loss)

        return model

    def _model_forward(
        self,
        model: MapEstimateNetSimple,
        batch: torch.Tensor,
        time_day_norm: torch.Tensor | None = None,
        time_year_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Override in subclasses for models with different forward signatures. Default: model(batch)."""
        return model(batch)

    @abstractmethod
    def _run_inference_on_points(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _run_inference_on_grid(
        self,
        bounding_box: BoundingBox,
        inference_params: dict,
    ) -> tuple[np.ndarray, rasterio.Affine]:
        pass

    def _build_impl(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame | None = None,
    ) -> dict[str, VectorMapArray]:
        """Build strength map by loading model from repository and running inference."""
        logger.info("Running inference on grid")
        strength_arr, transform = self._run_inference_on_grid(
            bounding_box,
            inference_params={
                "time_of_day_h": 12.0,
                "time_of_year_d": 182.5,
            },
        )

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
        bounding_box: BoundingBox,
        evaluate_df: pd.DataFrame,
    ) -> MapEvaluateResult:
        """Evaluate on held-out points. Runs inference on points directly (no map building).
        Provide vector_map (for its bounding_box) or bounding_box."""
        if len(evaluate_df) == 0:
            raise ValueError("evaluate_df is empty")
        if "lat" not in evaluate_df.columns or "lon" not in evaluate_df.columns or "strength" not in evaluate_df.columns:
            raise ValueError("evaluate_df must have columns 'lat', 'lon', and 'strength'")

        true_values = evaluate_df["strength"].to_numpy(dtype=np.float64)
        pred_values = self._run_inference_on_points(
            bounding_box,
            evaluate_df,
        )
        errors = np.abs(pred_values - true_values)

        return MapEvaluateResult(
            strength_mae=float(np.mean(errors)),
            strength_rmse=float(np.sqrt(np.mean(errors**2))),
            n_holdout=len(evaluate_df),
            true_values=true_values,
            pred_values=pred_values,
        )
