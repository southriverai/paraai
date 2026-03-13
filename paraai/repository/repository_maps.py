"""Repository for cached map outputs from various map generators."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from paraai.map.vectror_map_array import VectorMapArray
from paraai.tools_map import load_map, save_map


class RepositoryMaps:
    """Cache for map outputs. Supports different generators (Convolution, Flatland, etc.) and map types."""

    instance: Optional[RepositoryMaps] = None

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def initialize(cache_dir: Path) -> RepositoryMaps:
        if RepositoryMaps.instance is not None:
            raise ValueError("RepositoryMaps already initialized")
        RepositoryMaps.instance = RepositoryMaps(cache_dir)
        return RepositoryMaps.instance

    @staticmethod
    def get_instance() -> RepositoryMaps:
        if not hasattr(RepositoryMaps, "instance") or RepositoryMaps.instance is None:
            raise ValueError("RepositoryMaps not initialized")
        return RepositoryMaps.instance

    def get_map(
        self,
        generator_name: str,
        map_name: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        **params: object,
    ) -> VectorMapArray | None:
        """Load map from cache if it exists."""
        return load_map(
            self.cache_dir,
            generator_name,
            map_name,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            **params,
        )

    def save_map(
        self,
        vma: VectorMapArray,
        generator_name: str,
        map_name: str | None = None,
        lat_min: float | None = None,
        lat_max: float | None = None,
        lon_min: float | None = None,
        lon_max: float | None = None,
        **params: object,
    ) -> Path:
        """Save map to cache. Returns path to saved file."""
        return save_map(
            vma,
            self.cache_dir,
            generator_name,
            map_name=map_name,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            **params,
        )
