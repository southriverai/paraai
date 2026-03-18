"""Repository for cached datasets from map builders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


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
        dataset_id: str,
    ) -> dict:
        """Load dataset from cache if exists. raises ValueError if not cached."""
        path = self.cache_dir / builder_name / f"{dataset_id}.pt"

        if not path.exists():
            raise ValueError(f"Dataset not found in cache: {path}")

        data = torch.load(path, weights_only=False)
        logger.debug("Loaded dataset from cache %s", path)
        return data

    def save_dataset(
        self,
        builder_name: str,
        dataset_id: str,
        data: dict,
    ) -> None:
        """Save dataset dict to cache."""
        safe_builder = str(builder_name).replace(" ", "_").replace(":", "_")
        path = self.cache_dir / safe_builder / f"{dataset_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        logger.debug("Saved dataset to %s", path)
        return path
