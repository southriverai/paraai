"""Map builder: estimate climb strength from elevation patches (with time_of_day, time_of_year)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
from tqdm import tqdm

from paraai.map.dataset_builder import extract_elevation_patch
from paraai.map.map_builder_estimate_base import MapBuilderEstimateBase
from paraai.map.map_estimate_net import MapEstimateNetTime
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_models import RepositoryModels
from paraai.repository.repository_terrain import RepositoryTerrain

logger = logging.getLogger(__name__)


class MapBuilderEstimateTime(MapBuilderEstimateBase):
    """
    Map builder that trains a CNN to estimate strength from elevation patches.

    Requires time_of_day_h and time_of_year_d in the DataFrame (for future time-aware models).
    """

    def __init__(
        self,
        epochs: int = 5,
        batch_size: int = 8,
        lr: float = 1e-3,
        return_model: str = "lowest_test",
    ) -> None:
        super().__init__(
            name="MapBuilderEstimateTime",
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
        ground_alt_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pass time and ground altitude into dense layer."""
        if time_day_norm is None or time_year_norm is None or ground_alt_norm is None:
            raise ValueError("MapEstimateNetTime requires time and ground altitude")
        return model(batch, time_day_norm, time_year_norm, ground_alt_norm)

    def _run_inference_on_points(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame,
        model_id: str,
    ) -> np.ndarray:
        """Run model inference on (lat, lon) points. Requires time and ground_alt_norm columns in df."""
        required = ["time_of_day_h", "time_of_year_d", "ground_alt_norm"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"df must have column '{col}'")
        repo_models = RepositoryModels.get_instance()
        model_data = repo_models.get_model(self.name, model_id)
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

        patch_size_m = model_data["patch_size_m"]
        image_size = model_data["image_size"]
        patches = [
            extract_elevation_patch(
                terrain["elevation"],
                terrain["transform"],
                float(lat),
                float(lon),
                patch_size_m,
                image_size,
            )
            for lat, lon in zip(df["lat"], df["lon"], strict=True)
        ]
        inp = torch.stack(patches).to(device)
        td_t = torch.from_numpy(np.clip(df["time_of_day_h"].to_numpy() / 24.0, 0.0, 1.0).astype(np.float32)).to(device)
        ty_t = torch.from_numpy(np.clip(df["time_of_year_d"].to_numpy() / 365.0, 0.0, 1.0).astype(np.float32)).to(device)
        ga_t = torch.from_numpy(df["ground_alt_norm"].to_numpy().astype(np.float32)).to(device)

        with torch.no_grad():
            pred = self._model_forward(model, inp, td_t, ty_t, ga_t)
        pred_np = pred.cpu().numpy().flatten()
        return np.clip(pred_np, 0, 1) * (strength_hi - strength_lo) + strength_lo

    def _run_inference_on_grid(
        self,
        bounding_box: BoundingBox,
        inference_params: dict,
        model_id: str,
    ) -> np.ndarray:
        """Run model inference on grid. inference_params may include time_of_day_h and time_of_year_d (default 12, 182.5)."""
        time_of_day_h = inference_params.get("time_of_day_h", 12.0)
        time_of_year_d = inference_params.get("time_of_year_d", 182.5)
        time_day_norm = float(np.clip(time_of_day_h / 24.0, 0.0, 1.0))
        time_year_norm = float(np.clip(time_of_year_d / 365.0, 0.0, 1.0))
        repo_models = RepositoryModels.get_instance()
        model_data = repo_models.get_model(self.name, model_id)
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

        patch_size_m = model_data["patch_size_m"]
        image_size = model_data["image_size"]

        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"]
        transform = terrain["transform"]
        h, w = elevation.shape
        stride = model_data["grid_stride"]
        n_rows = (h + stride - 1) // stride
        n_cols = (w + stride - 1) // stride
        pred_sparse = np.zeros((n_rows, n_cols), dtype=np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        td_t = torch.tensor([time_day_norm], dtype=torch.float32, device=device)
        ty_t = torch.tensor([time_year_norm], dtype=torch.float32, device=device)

        def _alt_norm(alt_m: float) -> float:
            return float(np.clip(alt_m / 10000.0, 0.0, 1.0))

        with torch.no_grad():
            for ri, r in enumerate(tqdm(range(0, h, stride), desc="Inference", unit="row")):
                for ci, c in enumerate(range(0, w, stride)):
                    lon, lat = rasterio.transform.xy(transform, r, c)
                    elev_val = float(np.nan_to_num(elevation[r, c], nan=0.0))
                    ga_t = torch.tensor([_alt_norm(elev_val)], dtype=torch.float32, device=device)
                    patch = extract_elevation_patch(
                        elevation,
                        transform,
                        float(lat),
                        float(lon),
                        patch_size_m,
                        image_size,
                    )
                    patch_batch = patch.unsqueeze(0).to(device)
                    pred = self._model_forward(model, patch_batch, td_t, ty_t, ga_t)
                    val = pred[0, 0].item()
                    pred_sparse[ri, ci] = val

        pred_t = torch.from_numpy(pred_sparse).unsqueeze(0).unsqueeze(0)
        pred_grid = (
            nn.functional.interpolate(pred_t, size=(h, w), mode="bilinear", align_corners=False).squeeze().numpy().astype(np.float32)
        )
        # pred_grid = np.clip(pred_grid, 0, 1)
        pred_grid = pred_grid * (strength_hi - strength_lo) + strength_lo
        return pred_grid, transform
