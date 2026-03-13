"""Map builder: estimate climb maps from points using Gaussian convolution."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import rasterio.transform
from scipy.ndimage import convolve

from paraai.map.map_builder_base import MapBuilderBase
from paraai.map.vectror_map_array import VectorMapArray
from paraai.repository.repository_terrain import RepositoryTerrain
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
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
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
        terrain = repo_terrain.get_elevation(lon_min, lat_min, lon_max, lat_max)
        elevation = terrain["elevation"]
        transform = terrain["transform"]

        # Remove flat pixels: compute gradient magnitude, exclude points in flat cells
        grad_y, grad_x = np.gradient(elevation)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_m_mag = grad_mag * 30  # 1 arc-sec ≈ 30m per pixel
        is_flat = grad_m_mag < self.flat_gradient_threshold_m

        # Build count and strength grids
        count_grid = np.zeros(elevation.shape, dtype=np.float32)
        strength_grid = np.zeros(elevation.shape, dtype=np.float32)
        for _, row in df.iterrows():
            lat, lon = row["lat"], row["lon"]
            count = row[count_col] if count_col else 1.0
            strength = row[strength_col] if strength_col else 0.0
            row_idx, col_idx = rasterio.transform.rowcol(transform, [lon], [lat])
            r, c = int(row_idx[0]), int(col_idx[0])
            if 0 <= r < elevation.shape[0] and 0 <= c < elevation.shape[1] and is_flat[r, c]:
                continue
            if 0 <= r < count_grid.shape[0] and 0 <= c < count_grid.shape[1]:
                count_grid[r, c] = count
                strength_grid[r, c] = strength

        # Convolve with Gaussian kernel, weighted by count
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
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

        return {
            "strength": VectorMapArray(
                "strength",
                lat_min,
                lat_max,
                lon_min,
                lon_max,
                estimated_strength.astype(np.float32),
            ),
            "count": VectorMapArray(
                "count",
                lat_min,
                lat_max,
                lon_min,
                lon_max,
                estimated_count.astype(np.float32),
            ),
        }
