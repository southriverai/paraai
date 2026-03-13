"""Repository for terrain data: elevation and satellite imagery for bounding boxes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from paraai.tools_terrain import load_elevation, load_terrain

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paraai.model.boundingbox import BoundingBox


def _bbox_hash(bbox: Sequence[float], resolution: int) -> str:
    """Stable hash for cache key from bbox and resolution."""
    key = f"{tuple(round(x, 6) for x in bbox)}{resolution}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


class RepositoryTerrain:
    """Load terrain (elevation, satellite) for bounding boxes. Caches DEM and full terrain."""

    instance: RepositoryTerrain | None = None

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
        bounding_box: BoundingBox,
        *,
        dem_resolution: int = 30,
    ) -> dict:
        """Load elevation (DEM) only for bbox. Returns dict with elevation, transform, crs, lat_min, lat_max, lon_min, lon_max."""
        return load_elevation(
            bounding_box.lon_min,
            bounding_box.lat_min,
            bounding_box.lon_max,
            bounding_box.lat_max,
            dem_resolution=dem_resolution,
            cache_dir=self.cache_dir,
        )

    def get_terrain(
        self,
        bounding_box: BoundingBox,
        *,
        dem_resolution: int = 30,
        datetime_range: str = "2023-01/2023-12",
        cloud_cover_max: float = 20.0,
    ) -> dict:
        """Load elevation, RGB, and NDVI for bbox. Returns dict with elevation, rgb, ndvi, transform, crs."""
        return load_terrain(
            bounding_box.lon_min,
            bounding_box.lat_min,
            bounding_box.lon_max,
            bounding_box.lat_max,
            dem_resolution=dem_resolution,
            datetime_range=datetime_range,
            cloud_cover_max=cloud_cover_max,
            cache_dir=self.cache_dir,
        )

    def get_dem_cache_path(self, bbox: Sequence[float], resolution: int) -> Path:
        """Return the cache path for a DEM. Creates cache dir if needed."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        h = _bbox_hash(bbox, resolution)
        return self.cache_dir / f"dem_{h}.tif"
