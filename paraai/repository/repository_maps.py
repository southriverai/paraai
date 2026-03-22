"""Repository for cached map outputs from various map generators."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox
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
        bounding_box: BoundingBox,
        builder_id: str,
    ) -> VectorMapArray | None:
        """Load map from cache if it exists."""
        return load_map(
            self.cache_dir,
            generator_name,
            map_name,
            bounding_box,
            builder_id,
        )

    def save_map(
        self,
        vma: VectorMapArray,
        generator_name: str,
        map_name: str | None = None,
        bounding_box: BoundingBox | None = None,
        **params: object,
    ) -> Path:
        """Save map to cache. Returns path to saved file."""
        return save_map(
            vma,
            self.cache_dir,
            generator_name,
            map_name=map_name,
            bounding_box=bounding_box,
            **params,
        )
