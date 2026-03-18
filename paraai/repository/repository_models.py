"""Repository for cached models from map builders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class RepositoryModels:
    """Cache for map-builder models. Keyed by builder name and params."""

    instance: Optional[RepositoryModels] = None

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def initialize(cache_dir: Path) -> RepositoryModels:
        if RepositoryModels.instance is not None:
            raise ValueError("RepositoryModels already initialized")
        RepositoryModels.instance = RepositoryModels(cache_dir)
        return RepositoryModels.instance

    @staticmethod
    def get_instance() -> RepositoryModels:
        if not hasattr(RepositoryModels, "instance") or RepositoryModels.instance is None:
            raise ValueError("RepositoryModels not initialized")
        return RepositoryModels.instance

    def _path(self, builder_name: str, model_id: str) -> Path:
        safe_builder = str(builder_name).replace(" ", "_").replace(":", "_")
        return self.cache_dir / safe_builder / f"{model_id}.pt"

    def save_model(
        self,
        builder_name: str,
        model_id: str,
        state_dict: dict,
        in_channels: int,
        out_channels: int,
        image_size: int,
        patch_size_m: float,
        grid_stride: int,
        strength_lo: float,
        strength_hi: float,
    ) -> Path:
        """Save model to cache. Returns path to saved file."""
        path = self._path(builder_name, model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "state_dict": state_dict,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "image_size": image_size,
            "patch_size_m": patch_size_m,
            "grid_stride": grid_stride,
            "strength_lo": strength_lo,
            "strength_hi": strength_hi,
        }
        torch.save(data, path)
        logger.debug("Saved model to %s", path)
        return path

    def load_model(
        self,
        builder_name: str,
        model_id: str,
    ) -> dict | None:
        """Load model from cache if exists. Returns None if not cached."""
        path = self._path(builder_name, model_id)
        if not path.exists():
            return None
        data = torch.load(path, weights_only=False)
        logger.debug("Loaded model from cache %s", path)
        return data

    def get_model(self, builder_name: str, model_id: str) -> dict | None:
        """Load model from cache if it exists."""
        path = self._path(builder_name, model_id)
        if not path.exists():
            return None
        data = torch.load(path, weights_only=False)
        logger.debug("Loaded model from cache %s", path)
        return data
