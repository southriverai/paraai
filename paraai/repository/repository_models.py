"""Repository for cached models from map builders."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from paraai.tools_models import load_model, save_model


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

    def get_model(self, builder_name: str, **params: object) -> dict | None:
        """Load model from cache if it exists."""
        return load_model(self.cache_dir, builder_name, **params)

    def save_model(
        self,
        state_dict: dict,
        in_channels: int,
        out_channels: int,
        image_size: int,
        builder_name: str,
        *,
        strength_lo: float = 0.0,
        strength_hi: float = 1.0,
        **params: object,
    ) -> Path:
        """Save model to cache. Returns path to saved file."""
        params_for_hash = dict(params)
        params_for_hash["image_size"] = image_size  # use positional, override if in params
        return save_model(
            state_dict,
            in_channels,
            out_channels,
            self.cache_dir,
            builder_name,
            strength_lo=strength_lo,
            strength_hi=strength_hi,
            **params_for_hash,
        )
