"""Repository for terrain data: elevation and satellite imagery for bounding boxes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from paraai.tools_terrain import load_elevation, load_terrain


class RepositoryTerrain:
    """Load terrain (elevation, satellite) for bounding boxes. Caches DEM and full terrain."""

    instance: Optional[RepositoryTerrain] = None

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def initialize(cache_dir: Path) -> RepositoryTerrain:
        if RepositoryTerrain.instance is not None:
            raise ValueError("RepositoryTerrain already initialized")
        RepositoryTerrain.instance = RepositoryTerrain(cache_dir)
        return RepositoryTerrain.instance

    @staticmethod
    def get_instance() -> RepositoryTerrain:
        if not hasattr(RepositoryTerrain, "instance") or RepositoryTerrain.instance is None:
            raise ValueError("RepositoryTerrain not initialized")
        return RepositoryTerrain.instance

    def get_elevation(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        *,
        dem_resolution: int = 30,
    ) -> dict:
        """Load elevation (DEM) only for bbox. Returns dict with elevation, transform, crs, lat_min, lat_max, lon_min, lon_max."""
        return load_elevation(
            lon_min,
            lat_min,
            lon_max,
            lat_max,
            dem_resolution=dem_resolution,
            cache_dir=self.cache_dir,
        )

    def get_terrain(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        *,
        dem_resolution: int = 30,
        datetime_range: str = "2023-01/2023-12",
        cloud_cover_max: float = 20.0,
    ) -> dict:
        """Load elevation, RGB, and NDVI for bbox. Returns dict with elevation, rgb, ndvi, transform, crs."""
        return load_terrain(
            lon_min,
            lat_min,
            lon_max,
            lat_max,
            dem_resolution=dem_resolution,
            datetime_range=datetime_range,
            cloud_cover_max=cloud_cover_max,
            cache_dir=self.cache_dir,
        )
