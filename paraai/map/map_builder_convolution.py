"""Map builder: estimate climb maps from points using Gaussian convolution."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import rasterio.transform
from scipy.ndimage import convolve

from paraai.map.map_builder_base import MapBuilderBase
from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.tool_spacetime import build_gaussian_kernel_meters

logger = logging.getLogger(__name__)


class MapBuilderConvolution(MapBuilderBase):
    def __init__(
        self,
        kernel_size_m: float = 100,
        output_map_names: list[str] | None = None,
    ):
        super().__init__(name="MapBuilderConvolution", output_map_names=output_map_names)
        self.kernel_size_m = kernel_size_m

    def get_cache_params(self) -> dict:
        return {
            "kernel_size_m": self.kernel_size_m,
        }

    def _build_impl(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame,
    ) -> dict[str, VectorMapArray]:
        """Build estimated climb map from DataFrame with columns lat and lon (and optionally count, strength) using Gaussian convolution."""
        if df.empty:
            raise ValueError("No points provided")
        if "lat" not in df.columns or "lon" not in df.columns:
            raise ValueError("DataFrame must have columns 'lat' and 'lon'")
        count_col = "count" if "count" in df.columns else None
        strength_col = "strength" if "strength" in df.columns else None

        # Load DEM
        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"]
        transform = terrain["transform"]

        # Build count and strength grids
        count_grid = np.zeros(elevation.shape, dtype=np.float32)
        strength_grid = np.zeros(elevation.shape, dtype=np.float32)
        for _, row in df.iterrows():
            lat, lon = row["lat"], row["lon"]
            count = row[count_col] if count_col else 1.0
            strength = row[strength_col] if strength_col else 0.0
            row_idx, col_idx = rasterio.transform.rowcol(transform, [lon], [lat])
            r, c = int(row_idx[0]), int(col_idx[0])
            if 0 <= r < count_grid.shape[0] and 0 <= c < count_grid.shape[1]:
                count_grid[r, c] = count
                strength_grid[r, c] = strength

        # Convolve with Gaussian kernel
        center_lat = (bounding_box.lat_min + bounding_box.lat_max) / 2
        center_lon = (bounding_box.lon_min + bounding_box.lon_max) / 2
        kernel = build_gaussian_kernel_meters(self.kernel_size_m, center_lat, center_lon)
        logger.info("Convolution kernel: %s", kernel.shape)

        estimated_count = convolve(count_grid, kernel, mode="constant", cval=0)
        estimated_strength = convolve(strength_grid, kernel, mode="constant", cval=0)

        return {
            "strength": VectorMapArray(
                "strength",
                bounding_box,
                estimated_strength.astype(np.float32),
            ),
            "count": VectorMapArray(
                "count",
                bounding_box,
                estimated_count.astype(np.float32),
            ),
        }
