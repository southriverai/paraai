"""Map builder: estimate climb maps from points using Gaussian convolution."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

from paraai.map.map_builder_base import MapBuilderBase
from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
from paraai.tool_spacetime import build_gaussian_kernel_meters

logger = logging.getLogger(__name__)


class MapBuilderConvolutionWeighted(MapBuilderBase):
    def __init__(
        self,
        kernel_size_m: float = 100,
        holdout_ratio: float = 0.1,
        flat_gradient_threshold_m: float = 1.0,
        output_map_names: list[str] | None = None,
    ):
        super().__init__(name="MapBuilderConvolutionWeighted", output_map_names=output_map_names)
        self.kernel_size_m = kernel_size_m
        self.holdout_ratio = holdout_ratio
        self.flat_gradient_threshold_m = flat_gradient_threshold_m

    def get_cache_params(self) -> dict:
        return {
            "kernel_size_m": self.kernel_size_m,
            "holdout_ratio": self.holdout_ratio,
            "flat_gradient_threshold_m": self.flat_gradient_threshold_m,
        }

    def _build_impl(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, VectorMapArray]:
        """Build estimated climb map from DataFrame with columns lat and lon (and optionally count, strength) using Gaussian convolution."""
        if df.empty:
            raise ValueError("No points provided")
        if "lat" not in df.columns or "lon" not in df.columns:
            raise ValueError("DataFrame must have columns 'lat' and 'lon'")

        # Build count and strength grids using the average builder
        from paraai.map.map_builder_average import MapBuilderAverage

        builder = MapBuilderAverage()
        maps = builder.build(bounding_box, df)
        count_grid = maps["count"].array
        strength_grid = maps["strength"].array

        # Convolve with Gaussian kernel, weighted by count
        center_lat = (bounding_box.lat_min + bounding_box.lat_max) / 2
        center_lon = (bounding_box.lon_min + bounding_box.lon_max) / 2
        kernel = build_gaussian_kernel_meters(self.kernel_size_m, center_lat, center_lon)
        logger.info("Convolution kernel: %s", kernel.shape)

        # Count: kernel-weighted sum of counts
        estimated_count = convolve(count_grid, kernel, mode="constant", cval=0)

        # Strength: count-weighted average. sum(strength * count * kernel) / sum(count * kernel)
        strength_times_count = strength_grid * count_grid
        strength_weighted_sum = convolve(strength_times_count, kernel, mode="constant", cval=0)
        count_weighted_sum = convolve(count_grid.astype(np.float64), kernel, mode="constant", cval=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            estimated_strength = np.where(
                count_weighted_sum > 0,
                strength_weighted_sum / count_weighted_sum,
                0.0,
            ).astype(np.float32)

        transform = maps["count"].transform
        return {
            "strength": VectorMapArray(
                "strength",
                bounding_box,
                estimated_strength.astype(np.float32),
                transform=transform,
            ),
            "count": VectorMapArray(
                "count",
                bounding_box,
                estimated_count.astype(np.float32),
                transform=transform,
            ),
        }
