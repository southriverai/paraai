"""Map builder: estimate climb strength map from elevation using a trained CNN."""

from __future__ import annotations

import hashlib
import json
import logging
from abc import abstractmethod

import numpy as np
import pandas as pd
import rasterio
import torch

from paraai.map.dataset_builder import extract_elevation_patch
from paraai.map.map_builder_base import MapBuilderBase, MapEvaluateResult
from paraai.map.map_estimate_net import MapEstimateNetSimple
from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox

logger = logging.getLogger(__name__)

_extract_elevation_patch = extract_elevation_patch  # backward compat


class MapBuilderEstimateBase(MapBuilderBase):
    """
    Map builder that trains a CNN to estimate strength from elevation patches.

    Uses strength from the DataFrame at each point as target. Takes a DataFrame with lat, lon, strength.
    """

    def __init__(
        self,
        name: str,
        *,
        epochs: int = 5,
        batch_size: int = 8,
        lr: float = 1e-3,
        return_model: str = "lowest_test",
    ) -> None:
        super().__init__(name=name, output_map_names=["strength"])
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.return_model = return_model

    def get_builder_id(self) -> str:
        """Builder identity string for cache keying."""
        params = {}
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

    def _model_forward(
        self,
        model: MapEstimateNetSimple,
        batch: torch.Tensor,
        time_day_norm: torch.Tensor | None = None,
        time_year_norm: torch.Tensor | None = None,
        ground_alt_norm: torch.Tensor | None = None,
        start_alt_norm: torch.Tensor | None = None,
        end_alt_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Override in subclasses for models with different forward signatures."""
        if time_day_norm is None or time_year_norm is None or ground_alt_norm is None:
            raise ValueError("Model requires time and altitude")
        return model(batch, time_day_norm, time_year_norm, ground_alt_norm, start_alt_norm, end_alt_norm)

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
