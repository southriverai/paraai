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


class MapBuilderConvolution(MapBuilderBase):
    def __init__(
        self,
        kernel_size_m: float = 100,
    ):
        super().__init__(name="MapBuilderConvolution", output_map_names=["strength", "count"])
        self.kernel_size_m = kernel_size_m

    def get_cache_params(self) -> dict:
        return {
            "kernel_size_m": self.kernel_size_m,
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

        # Convolve with Gaussian kernel
        center_lat = (bounding_box.lat_min + bounding_box.lat_max) / 2
        center_lon = (bounding_box.lon_min + bounding_box.lon_max) / 2
        kernel = build_gaussian_kernel_meters(self.kernel_size_m, center_lat, center_lon)
        logger.info("Convolution kernel: %s", kernel.shape)

        estimated_count = convolve(count_grid, kernel, mode="constant", cval=0)
        estimated_strength = convolve(strength_grid, kernel, mode="constant", cval=0)

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
