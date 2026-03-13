from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rasterio
import rasterio.transform

if TYPE_CHECKING:
    from paraai.model.boundingbox import BoundingBox


class VectorMapArray:
    def __init__(
        self,
        map_name: str,
        bounding_box: BoundingBox,
        array: np.ndarray,
    ):
        self.map_name = map_name
        self.bounding_box = bounding_box
        self.array = array
        # Rasterio: row 0 = north. from_bounds(west, south, east, north, width, height)
        self.transform = rasterio.transform.from_bounds(
            bounding_box.lon_min,
            bounding_box.lat_min,
            bounding_box.lon_max,
            bounding_box.lat_max,
            array.shape[1],
            array.shape[0],
        )

    @property
    def profile(self) -> dict:
        """Rasterio profile for writing to GeoTIFF."""
        return {
            "driver": "GTiff",
            "width": self.array.shape[1],
            "height": self.array.shape[0],
            "count": 1,
            "dtype": self.array.dtype,
            "transform": self.transform,
            "crs": rasterio.CRS.from_epsg(4326),
        }

    def get_array(self) -> np.ndarray:
        return self.array

    def get_transform(self) -> rasterio.transform.Transform:
        return self.transform

    def get_profile(self) -> dict:
        return self.profile

    def get_value(self, lat: float, lon: float) -> float:
        return self.get_values([lat], [lon])[0]

    def get_values(self, lats: list[float], lons: list[float]) -> np.ndarray:
        rows, cols = rasterio.transform.rowcol(self.transform, lons, lats)
        rows = np.clip(rows, 0, self.array.shape[0] - 1)
        cols = np.clip(cols, 0, self.array.shape[1] - 1)
        return self.array[rows, cols]
