"""Save and load map-builder datasets to/from cache with keying by builder + bbox + params."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

from paraai.model.boundingbox import BoundingBox

logger = logging.getLogger(__name__)


def _dataset_cache_hash(
    builder_name: str,
    bounding_box: BoundingBox,
    params: dict,
) -> str:
    """Hash for cache key: builder + bbox + params."""
    data = {
        "builder": builder_name,
        "bbox": [bounding_box.lat_min, bounding_box.lat_max, bounding_box.lon_min, bounding_box.lon_max],
        "params": dict(sorted(params.items())),
    }
    s = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def save_dataset(
    data: dict,
    cache_dir: Path,
    builder_name: str,
    bounding_box: BoundingBox,
    **params: object,
) -> Path:
    """Save dataset dict to cache. Returns path to saved file."""
    h = _dataset_cache_hash(builder_name, bounding_box, params)
    cache_dir = Path(cache_dir)
    safe_builder = str(builder_name).replace(" ", "_").replace(":", "_")
    path = cache_dir / safe_builder / f"{h}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    logger.debug("Saved dataset to %s", path)
    return path


def load_dataset(
    cache_dir: Path,
    builder_name: str,
    bounding_box: BoundingBox,
    **params: object,
) -> dict | None:
    """Load dataset from cache if exists. Returns None if not cached."""
    h = _dataset_cache_hash(builder_name, bounding_box, params)
    cache_dir = Path(cache_dir)
    safe_builder = str(builder_name).replace(" ", "_").replace(":", "_")
    path = cache_dir / safe_builder / f"{h}.pt"

    if not path.exists():
        return None

    data = torch.load(path, weights_only=False)
    logger.debug("Loaded dataset from cache %s", path)
    return data
