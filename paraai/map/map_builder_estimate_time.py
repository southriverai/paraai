"""Map builder: estimate climb strength from elevation patches (with time_of_day, time_of_year)."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
from tqdm import tqdm

from paraai.map.map_builder_estimate_base import (
    MapBuilderEstimateBase,
    _extract_elevation_patch,
)
from paraai.map.map_estimate_net import MapEstimateNetTime
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_datasets import RepositoryDatasets
from paraai.repository.repository_models import RepositoryModels
from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.tools_datasets import split_dataframe

logger = logging.getLogger(__name__)


class MapBuilderEstimateTime(MapBuilderEstimateBase):
    """
    Map builder that trains a CNN to estimate strength from elevation patches.

    Requires time_of_day_h and time_of_year_d in the DataFrame (for future time-aware models).
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
        super().__init__(
            name="MapBuilderEstimateTime",
            patch_size_m=patch_size_m,
            image_size=image_size,
            grid_stride=grid_stride,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            return_model=return_model,
        )

    def _get_model_class(self) -> type[MapEstimateNetTime]:
        """Use time-aware architecture: conv encoder + dense with time -> scalar."""
        return MapEstimateNetTime

    def _model_forward(
        self,
        model: MapEstimateNetTime,
        batch: torch.Tensor,
        time_day_norm: torch.Tensor | None = None,
        time_year_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pass time into dense layer. Requires time_day_norm and time_year_norm."""
        if time_day_norm is None or time_year_norm is None:
            raise ValueError("MapEstimateNetTime requires time_of_day and time_of_year")
        return model(batch, time_day_norm, time_year_norm)

    def _build_dataset(
        self,
        bounding_box: BoundingBox,
        df_training: pd.DataFrame,
        df_test: pd.DataFrame,
    ):
        """Build dataset: elevation patches (1ch), time arrays, scalar targets."""
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
        ):
            inputs: list[torch.Tensor] = []
            time_day: list[float] = []
            time_year: list[float] = []
            targets: list[torch.Tensor] = []
            lats: list[float] = []
            lons: list[float] = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, unit="patch"):
                lat_val = float(row["lat"])
                lon_val = float(row["lon"])
                raw_val = float(row["strength"])
                time_of_day_h = float(row["time_of_day_h"])
                time_of_year_d = float(row["time_of_year_d"])
                patch = _extract_elevation_patch(elevation, transform, lat_val, lon_val, self.patch_size_m, self.image_size)
                time_day_norm = float(np.clip(time_of_day_h / 24.0, 0.0, 1.0))
                time_year_norm = float(np.clip(time_of_year_d / 365.0, 0.0, 1.0))
                norm_val = float(np.clip((raw_val - strength_lo) / (strength_hi - strength_lo), 0, 1))
                inputs.append(patch)
                time_day.append(time_day_norm)
                time_year.append(time_year_norm)
                targets.append(torch.tensor([norm_val], dtype=torch.float32))
                lats.append(lat_val)
                lons.append(lon_val)
            return inputs, time_day, time_year, targets, lats, lons

        (
            input_maps_train,
            time_day_train,
            time_year_train,
            target_maps_train,
            lats_train,
            lons_train,
        ) = _build_patches(df_training, desc="Train patches")
        (
            input_maps_test,
            time_day_test,
            time_year_test,
            target_maps_test,
            lats_test,
            lons_test,
        ) = _build_patches(df_test, desc="Test patches")

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
            time_day_train,
            time_year_train,
            time_day_test,
            time_year_test,
        )

    def get_dataset_cache_params(self, test_frac: float, split_seed: int) -> dict:
        """Params for dataset cache key (includes split params)."""
        return {
            **self.get_cache_params(),
            "test_frac": test_frac,
            "split_seed": split_seed,
            "time_model_v2": True,  # scalar output + time in dense layer; invalidates old cache
        }

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

        Requires time_of_day_h and time_of_year_d columns. Uses test_frac/split_seed for cache key.
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
            if data is not None and "time_of_day_train" in data:
                n_train = len(data["input_maps_train"])
                n_test = len(data["input_maps_test"])
                logger.info("Loaded dataset from cache (%s train, %s test patches)", n_train, n_test)
                return data

        result = self._build_dataset(bounding_box, df_training, df_test)
        (
            input_maps_train,
            target_maps_train,
            input_maps_test,
            target_maps_test,
            lats_train,
            lons_train,
            lats_test,
            lons_test,
            _elevation,
            _transform,
            strength_lo,
            strength_hi,
            time_day_train,
            time_year_train,
            time_day_test,
            time_year_test,
        ) = result

        data = {
            "input_maps_train": input_maps_train,
            "target_maps_train": target_maps_train,
            "input_maps_test": input_maps_test,
            "target_maps_test": target_maps_test,
            "lats_train": lats_train,
            "lons_train": lons_train,
            "lats_test": lats_test,
            "lons_test": lons_test,
            "time_of_day_train": time_day_train,
            "time_of_year_train": time_year_train,
            "time_of_day_test": time_day_test,
            "time_of_year_test": time_year_test,
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
        train_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
        *,
        test_frac: float = 0.2,
        split_seed: int = 42,
        ignore_cache: bool = False,
    ) -> dict:
        """Load dataset from cache or build from region. Pass train_df/test_df when split is done by caller."""
        if train_df is not None and test_df is not None:
            return self.build_dataset(
                bounding_box,
                train_df,
                test_df,
                ignore_cache=ignore_cache,
                test_frac=test_frac,
                split_seed=split_seed,
            )
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

    def _run_inference_on_points(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Run model inference on (lat, lon) points. Requires time_of_day_h and time_of_year_d in df."""
        if "time_of_day_h" not in df.columns or "time_of_year_d" not in df.columns:
            raise ValueError("df must have columns 'time_of_day_h' and 'time_of_year_d'")
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
        model = self._get_model_class()(
            in_channels=model_data["in_channels"],
            out_channels=model_data["out_channels"],
            size=model_data["image_size"],
        )
        model.load_state_dict(model_data["state_dict"])

        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Running inference on %s", device)
        model = model.to(device)
        model.eval()

        patches = [
            _extract_elevation_patch(
                terrain["elevation"],
                terrain["transform"],
                float(lat),
                float(lon),
                self.patch_size_m,
                self.image_size,
            )
            for lat, lon in zip(df["lat"], df["lon"], strict=True)
        ]
        inp = torch.stack(patches).to(device)
        td_norms = np.clip(df["time_of_day_h"].to_numpy() / 24.0, 0.0, 1.0).astype(np.float32)
        ty_norms = np.clip(df["time_of_year_d"].to_numpy() / 365.0, 0.0, 1.0).astype(np.float32)
        td_t = torch.from_numpy(td_norms).to(device)
        ty_t = torch.from_numpy(ty_norms).to(device)

        with torch.no_grad():
            pred = self._model_forward(model, inp, td_t, ty_t)
        pred_np = pred.cpu().numpy().flatten()
        return np.clip(pred_np, 0, 1) * (strength_hi - strength_lo) + strength_lo

    def _run_inference_on_grid(
        self,
        bounding_box: BoundingBox,
        inference_params: dict,
    ) -> np.ndarray:
        """Run model inference on grid. inference_params may include time_of_day_h and time_of_year_d (default 12, 182.5)."""
        time_of_day_h = inference_params.get("time_of_day_h", 12.0)
        time_of_year_d = inference_params.get("time_of_year_d", 182.5)
        time_day_norm = float(np.clip(time_of_day_h / 24.0, 0.0, 1.0))
        time_year_norm = float(np.clip(time_of_year_d / 365.0, 0.0, 1.0))

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
        model = self._get_model_class()(
            in_channels=model_data["in_channels"],
            out_channels=model_data["out_channels"],
            size=model_data["image_size"],
        )
        model.load_state_dict(model_data["state_dict"])

        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"]
        transform = terrain["transform"]
        h, w = elevation.shape
        stride = self.grid_stride
        n_rows = (h + stride - 1) // stride
        n_cols = (w + stride - 1) // stride
        pred_sparse = np.zeros((n_rows, n_cols), dtype=np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        td_t = torch.tensor([time_day_norm], dtype=torch.float32, device=device)
        ty_t = torch.tensor([time_year_norm], dtype=torch.float32, device=device)

        with torch.no_grad():
            for ri, r in enumerate(tqdm(range(0, h, stride), desc="Inference", unit="row")):
                for ci, c in enumerate(range(0, w, stride)):
                    lon, lat = rasterio.transform.xy(transform, r, c)
                    patch = _extract_elevation_patch(
                        elevation,
                        transform,
                        float(lat),
                        float(lon),
                        self.patch_size_m,
                        self.image_size,
                    )
                    patch_batch = patch.unsqueeze(0).to(device)
                    pred = self._model_forward(model, patch_batch, td_t, ty_t)
                    val = pred[0, 0].item()
                    pred_sparse[ri, ci] = val

        pred_t = torch.from_numpy(pred_sparse).unsqueeze(0).unsqueeze(0)
        pred_grid = (
            nn.functional.interpolate(pred_t, size=(h, w), mode="bilinear", align_corners=False).squeeze().numpy().astype(np.float32)
        )
        # pred_grid = np.clip(pred_grid, 0, 1)
        pred_grid = pred_grid * (strength_hi - strength_lo) + strength_lo
        return pred_grid, transform
