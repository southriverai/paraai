"""Map builder: estimate climb strength from elevation patches (simple, no time)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
from tqdm import tqdm

from paraai.map.map_builder_estimate_base import MapBuilderEstimateBase, _extract_elevation_patch
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_models import RepositoryModels
from paraai.repository.repository_terrain import RepositoryTerrain


class MapBuilderEstimateSimple(MapBuilderEstimateBase):
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
        super().__init__(
            name="MapBuilderEstimateSimple",
            patch_size_m=patch_size_m,
            image_size=image_size,
            grid_stride=grid_stride,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            return_model=return_model,
        )

    def _run_inference_on_points(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Run model inference on (lat, lon) points. Returns denormalized strength array."""
        # load model from repository
        repo_models = RepositoryModels.get_instance()
        model_data = repo_models.get_model(self.name, **self.get_model_cache_params())
        if model_data is None:
            raise FileNotFoundError(f"Model not found in repository for {self.name}. Run train mode first to train and save the model.")
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

        # load terrain from repository
        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running inference on {device}")
        model = model.to(device)
        model.eval()

        patches = [
            _extract_elevation_patch(terrain["elevation"], terrain["transform"], float(lat), float(lon), self.patch_size_m, self.image_size)
            for lat, lon in zip(df["lat"], df["lon"], strict=True)
        ]
        inp = torch.stack(patches).to(device)
        pred = self._model_forward(model, inp)
        return pred.cpu().numpy().flatten() * (strength_hi - strength_lo) + strength_lo

    def _run_inference_on_grid(
        self,
        bounding_box: BoundingBox,
        inference_params: dict,
    ) -> np.ndarray:
        """Run model inference on grid. Returns denormalized strength array."""
        # load model from repository
        repo_models = RepositoryModels.get_instance()
        model_data = repo_models.get_model(self.name, **self.get_model_cache_params())
        if model_data is None:
            raise FileNotFoundError(f"Model not found in repository for {self.name}. Run train mode first to train and save the model.")
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

        # load terrain from repository
        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"]
        transform = terrain["transform"]
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
                    patch_batch = patch.unsqueeze(0)
                    pred = self._model_forward(model, patch_batch)
                    if pred.dim() == 2:
                        val = pred[0, 0].item()
                    else:
                        val = pred[0, 0, self.image_size // 2, self.image_size // 2].item()
                    pred_sparse[ri, ci] = val
        pred_t = torch.from_numpy(pred_sparse).unsqueeze(0).unsqueeze(0)
        pred_grid = (
            nn.functional.interpolate(pred_t, size=(h, w), mode="bilinear", align_corners=False).squeeze().numpy().astype(np.float32)
        )
        pred_grid = np.clip(pred_grid, 0, 1)
        return pred_grid * (strength_hi - strength_lo) + strength_lo
