"""Repository for cached datasets from map builders."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import torch

from paraai.model.boundingbox import BoundingBox

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

    def get_dataset_cache_id(
        self,
        builder_name: str,
        bounding_box: BoundingBox,
        **params: object,
    ) -> str:
        """Compute dataset cache ID hash from builder name, bounding box, and params."""
        data = {
            "builder": builder_name,
            "bbox": [bounding_box.lat_min, bounding_box.lat_max, bounding_box.lon_min, bounding_box.lon_max],
            "params": dict(sorted(params.items())),
        }
        s = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def get_dataset(
        self,
        builder_name: str,
        dataset_cache_id: str,
    ) -> dict:
        """Load dataset from cache if exists. raises ValueError if not cached."""
        safe_builder = str(builder_name).replace(" ", "_").replace(":", "_")
        path = self.cache_dir / safe_builder / f"{dataset_cache_id}.pt"

        if not path.exists():
            raise ValueError(f"Dataset not found in cache: {path}")

        data = torch.load(path, weights_only=False)
        logger.debug("Loaded dataset from cache %s", path)
        return data

    def save_dataset(
        self,
        data: dict,
        builder_name: str,
        dataset_cache_id: str,
    ) -> None:
        """Save dataset dict to cache."""
        safe_builder = str(builder_name).replace(" ", "_").replace(":", "_")
        path = self.cache_dir / safe_builder / f"{dataset_cache_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        logger.debug("Saved dataset to %s", path)
        return path
