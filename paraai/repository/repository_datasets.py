"""Repository for cached datasets from map builders."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from paraai.model.boundingbox import BoundingBox
from paraai.tools_datasets import load_dataset, save_dataset


class RepositoryDatasets:
    """Cache for map-builder datasets. Keyed by builder name, bounding_box, and params."""

    instance: Optional[RepositoryDatasets] = None

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def initialize(cache_dir: Path) -> RepositoryDatasets:
        if RepositoryDatasets.instance is not None:
            raise ValueError("RepositoryDatasets already initialized")
        RepositoryDatasets.instance = RepositoryDatasets(cache_dir)
        return RepositoryDatasets.instance

    @staticmethod
    def get_instance() -> RepositoryDatasets:
        if not hasattr(RepositoryDatasets, "instance") or RepositoryDatasets.instance is None:
            raise ValueError("RepositoryDatasets not initialized")
        return RepositoryDatasets.instance

    def get_dataset(
        self,
        builder_name: str,
        bounding_box: BoundingBox,
        **params: object,
    ) -> dict | None:
        """Load dataset from cache if it exists."""
        return load_dataset(
            self.cache_dir,
            builder_name,
            bounding_box,
            **params,
        )

    def save_dataset(
        self,
        data: dict,
        builder_name: str,
        bounding_box: BoundingBox,
        **params: object,
    ) -> Path:
        """Save dataset to cache. Returns path to saved file."""
        return save_dataset(
            data,
            self.cache_dir,
            builder_name,
            bounding_box,
            **params,
        )
