"""Map builder: estimate climb maps from points using convolution."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import rasterio.transform

from paraai.map.map_builder_base import MapBuilderBase
from paraai.map.vectror_map_array import VectorMapArray
from paraai.repository.repository_terrain import RepositoryTerrain

logger = logging.getLogger(__name__)


class MapBuilderAverage(MapBuilderBase):
    def __init__(self, output_map_names: list[str] | None = None):
        super().__init__(name="MapBuilderAverage", output_map_names=output_map_names)

    def _build_impl(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        df: pd.DataFrame,
    ) -> dict[str, VectorMapArray]:
        """Build estimated climb map from DataFrame with columns lat and lon (and optionally count, strength)."""
        if "lat" not in df.columns or "lon" not in df.columns:
            raise ValueError("DataFrame must have columns 'lat' and 'lon'")
        count_col = "count" if "count" in df.columns else None
        strength_col = "strength" if "strength" in df.columns else None

        # Load DEM
        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(lon_min, lat_min, lon_max, lat_max)
        elevation = terrain["elevation"]
        transform = terrain["transform"]
        dem_shape = elevation.shape
        # TODO do this without getting the terrain

        # Build count and strength grids from training points only
        strength_grid = np.zeros(dem_shape, dtype=np.float32)
        count_grid = np.zeros(dem_shape, dtype=np.float32)
        for _, row in df.iterrows():
            lat, lon = row["lat"], row["lon"]
            count = row[count_col] if count_col else 1.0
            strength = row[strength_col] if strength_col else 0.0
            row_idx, col_idx = rasterio.transform.rowcol(transform, [lon], [lat])
            r, c = int(row_idx[0]), int(col_idx[0])
            if 0 <= r < strength_grid.shape[0] and 0 <= c < strength_grid.shape[1]:
                strength_grid[r, c] = strength
                count_grid[r, c] = count

        return {
            "strength": VectorMapArray(
                "strength",
                lat_min,
                lat_max,
                lon_min,
                lon_max,
                strength_grid,
            ),
            "count": VectorMapArray(
                "count",
                lat_min,
                lat_max,
                lon_min,
                lon_max,
                count_grid,
            ),
        }
