"""Map builder: estimate climb maps and exploration density from points."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import rasterio.transform
from scipy.ndimage import convolve

from paraai.map.map_builder_base import MapBuilderBase
from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.tool_spacetime import dem_pixel_size_m

logger = logging.getLogger(__name__)


class MapBuilderDensity(MapBuilderBase):
    def __init__(self, list_radius_m: list[float] | None = None):
        list_radius_m = list_radius_m or [200.0]
        output_names = ["strength", "count", "data_density"] + [f"count_within_{r}m" for r in list_radius_m]
        super().__init__(name="MapBuilderDensity", output_map_names=output_names)
        self.list_radius_m = list_radius_m

    def get_cache_params(self) -> dict:
        return {"list_radius_m": self.list_radius_m}

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

        # data_density: sum of counts within radius; high = explored, 0 = unexplored
        # Pixels with count=0 but data_density>0 = explored but yielded no points
        center_lat = (bounding_box.lat_min + bounding_box.lat_max) / 2
        width_m, height_m = dem_pixel_size_m(center_lat)

        result: dict[str, VectorMapArray] = {
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

        for radius_m in self.list_radius_m:
            radius_x = int(math.ceil(radius_m / width_m))
            radius_y = int(math.ceil(radius_m / height_m))
            size_x, size_y = 2 * radius_x + 1, 2 * radius_y + 1
            kernel = np.ones((size_y, size_x), dtype=np.float64)
            density = convolve(count_grid.astype(np.float64), kernel, mode="constant", cval=0).astype(np.float32)
            name = f"count_within_{radius_m}m"
            result[name] = VectorMapArray(name, bounding_box, density, transform=transform)

        # data_density: exploration density (same as count_within for first radius)
        first_density = result[f"count_within_{self.list_radius_m[0]}m"].array
        result["data_density"] = VectorMapArray(
            "data_density",
            bounding_box,
            first_density,
            transform=transform,
        )

        return result
