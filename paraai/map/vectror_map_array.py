from __future__ import annotations

import numpy as np
import rasterio
import rasterio.transform


class VectorMapArray:
    def __init__(
        self,
        map_name: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        array: np.ndarray,
    ):
        self.map_name = map_name
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.array = array
        # Rasterio: row 0 = north. from_bounds(west, south, east, north, width, height)
        self.transform = rasterio.transform.from_bounds(
            lon_min, lat_min, lon_max, lat_max, array.shape[1], array.shape[0]
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

    def sample(self, lat: float, lon: float) -> float:
        row, col = rasterio.transform.rowcol(self.transform, [lon], [lat])
        r, c = int(row[0]), int(col[0])
        if 0 <= r < self.array.shape[0] and 0 <= c < self.array.shape[1]:
            return self.array[r, c]
        return 0.0
