"""Map builder: estimate climb maps from points using convolution."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import rasterio.transform

from paraai.map.map_builder_base import MapBuilderBase
from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_terrain import RepositoryTerrain

logger = logging.getLogger(__name__)


class MapBuilderAverage(MapBuilderBase):
    def __init__(
        self,
    ):
        super().__init__(name="MapBuilderAverage", output_map_names=["count", "strength"])

    def _build_impl(
        self,
        bounding_box: BoundingBox,
        df: pd.DataFrame,
        model_id: str | None = None,
        **kwargs: Any,
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
        lat_min, lat_max = df["lat"].min(), df["lat"].max()
        lon_min, lon_max = df["lon"].min(), df["lon"].max()
        logger.info("Dataset: lat_min=%.4f, lat_max=%.4f, lon_min=%.4f, lon_max=%.4f", lat_min, lat_max, lon_min, lon_max)

        # Log DEM extent vs bbox: DEM may cover larger area (tile-aligned)
        bounds = rasterio.transform.array_bounds(dem_shape[0], dem_shape[1], transform)
        logger.info(
            "DEM extent: lon=%.4f-%.4f, lat=%.4f-%.4f (bbox requested: lon=%.4f-%.4f, lat=%.4f-%.4f)",
            bounds[0],
            bounds[2],
            bounds[1],
            bounds[3],
            bounding_box.lon_min,
            bounding_box.lon_max,
            bounding_box.lat_min,
            bounding_box.lat_max,
        )

        strength_grid = np.zeros(dem_shape, dtype=np.float32)
        count_grid = np.zeros(dem_shape, dtype=np.float32)
        n_in, n_out = 0, 0

        for _, row in df.iterrows():
            lat, lon = row["lat"], row["lon"]
            row_idx, col_idx = rasterio.transform.rowcol(transform, [lon], [lat])
            r, c = int(row_idx[0]), int(col_idx[0])
            if 0 <= r < strength_grid.shape[0] and 0 <= c < strength_grid.shape[1]:
                strength_grid[r, c] += row["strength"]
                count_grid[r, c] += 1
                n_in += 1
            else:
                n_out += 1

        logger.info("Points in raster: %d, out of bounds: %d (raster shape %s)", n_in, n_out, dem_shape)

        np.divide(strength_grid, count_grid, out=strength_grid, where=count_grid > 0)

        return {
            "strength": VectorMapArray(
                "strength",
                bounding_box,
                strength_grid,
                transform=transform,
            ),
            "count": VectorMapArray(
                "count",
                bounding_box,
                count_grid,
                transform=transform,
            ),
        }
