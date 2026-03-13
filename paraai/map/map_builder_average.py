"""Map builder: estimate climb maps from points using convolution."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import rasterio.transform

from paraai.map.map_builder_base import MapBuilderBase
from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_terrain import RepositoryTerrain

logger = logging.getLogger(__name__)


class MapBuilderAverage(MapBuilderBase):
    def __init__(self, output_map_names: list[str] | None = None):
        super().__init__(name="MapBuilderAverage", output_map_names=output_map_names)

    def _build_impl(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame,
    ) -> dict[str, VectorMapArray]:
        """Build estimated climb map from DataFrame with columns lat and lon (and optionally count, strength)."""
        if "lat" not in df.columns or "lon" not in df.columns:
            raise ValueError("DataFrame must have columns 'lat' and 'lon'")
        if "strength" not in df.columns:
            raise ValueError("DataFrame must have column 'strength'")

        # Load DEM
        repo_terrain = RepositoryTerrain.get_instance()
        terrain = repo_terrain.get_elevation(bounding_box)
        elevation = terrain["elevation"]
        transform = terrain["transform"]
        dem_shape = elevation.shape
        # TODO do this without getting the terrain

        # Build count and strength grids from training points only
        strength_grid = np.zeros(dem_shape, dtype=np.float32)
        count_grid = np.zeros(dem_shape, dtype=np.float32)
        for _, row in df.iterrows():
            lat, lon = row["lat"], row["lon"]
            row_idx, col_idx = rasterio.transform.rowcol(transform, [lon], [lat])
            r, c = int(row_idx[0]), int(col_idx[0])
            if 0 <= r < strength_grid.shape[0] and 0 <= c < strength_grid.shape[1]:
                strength_grid[r, c] += row["strength"]
                count_grid[r, c] += 1

        np.divide(strength_grid, count_grid, out=strength_grid, where=count_grid > 0)

        return {
            "strength": VectorMapArray(
                "strength",
                bounding_box,
                strength_grid,
            ),
            "count": VectorMapArray(
                "count",
                bounding_box,
                count_grid,
            ),
        }
