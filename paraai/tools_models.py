"""Save and load map-builder models to/from cache with keying by builder + params."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _model_cache_hash(builder_name: str, params: dict) -> str:
    """Hash for cache key: builder + params."""
    data = {
        "builder": builder_name,
        "params": dict(sorted(params.items())),
    }
    s = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def save_model(
    state_dict: dict,
    in_channels: int,
    out_channels: int,
    cache_dir: Path,
    builder_name: str,
    **params: object,
) -> Path:
    """Save model to cache. Returns path to saved file. image_size must be in params for cache key."""
    if "image_size" not in params:
        raise ValueError("params must include 'image_size' for cache key")
    h = _model_cache_hash(builder_name, params)
    cache_dir = Path(cache_dir)
    safe_builder = str(builder_name).replace(" ", "_").replace(":", "_")
    path = cache_dir / safe_builder / f"{h}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "state_dict": state_dict,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "image_size": params["image_size"],
    }
    torch.save(data, path)
    logger.debug("Saved model to %s", path)
    return path


def load_model(
    cache_dir: Path,
    builder_name: str,
    **params: object,
) -> dict | None:
    """Load model from cache if exists. Returns None if not cached."""
    h = _model_cache_hash(builder_name, params)
    cache_dir = Path(cache_dir)
    safe_builder = str(builder_name).replace(" ", "_").replace(":", "_")
    path = cache_dir / safe_builder / f"{h}.pt"

    if not path.exists():
        return None

    data = torch.load(path, weights_only=False)
    logger.debug("Loaded model from cache %s", path)
    return data
