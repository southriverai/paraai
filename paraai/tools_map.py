"""Save and load map arrays to/from GeoTIFF with cache keying."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import rasterio

from paraai.map.vectror_map_array import VectorMapArray
from paraai.model.boundingbox import BoundingBox

logger = logging.getLogger(__name__)


def _map_cache_hash(
    generator_name: str,
    map_name: str,
    bounding_box: BoundingBox,
    params: dict,
) -> str:
    """Hash for cache key: generator + map_name + bbox + params."""
    data = {
        "generator": generator_name,
        "map": map_name,
        "bbox": [bounding_box.lat_min, bounding_box.lat_max, bounding_box.lon_min, bounding_box.lon_max],
        "params": dict(sorted(params.items())),
    }
    s = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def save_map(
    vma: VectorMapArray,
    cache_dir: Path,
    generator_name: str,
    map_name: str | None = None,
    bounding_box: BoundingBox | None = None,
    **params: object,
) -> Path:
    """Save VectorMapArray to cache. Returns path to saved file."""
    bbox = bounding_box if bounding_box is not None else vma.bounding_box
    map_name = map_name or vma.map_name

    h = _map_cache_hash(generator_name, map_name, bbox, params)
    cache_dir = Path(cache_dir)
    safe_gen = str(generator_name).replace(" ", "_").replace(":", "_")
    safe_map = str(map_name).replace(" ", "_")
    path = cache_dir / safe_gen / safe_map / f"{h}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(path, "w", **vma.profile) as dst:
        dst.write(vma.array, 1)

    logger.debug("Saved map to %s", path)
    return path


def load_map(
    cache_dir: Path,
    generator_name: str,
    map_name: str,
    bounding_box: BoundingBox,
    **params: object,
) -> VectorMapArray | None:
    """Load VectorMapArray from cache if exists. Returns None if not cached."""
    h = _map_cache_hash(generator_name, map_name, bounding_box, params)
    cache_dir = Path(cache_dir)
    safe_gen = str(generator_name).replace(" ", "_").replace(":", "_")
    safe_map = str(map_name).replace(" ", "_")
    path = cache_dir / safe_gen / safe_map / f"{h}.tif"

    if not path.exists():
        return None

    with rasterio.open(path) as src:
        array = src.read(1)
        transform = src.transform

    vma = VectorMapArray(
        map_name=map_name,
        bounding_box=bounding_box,
        array=array.astype(np.float32),
    )
    logger.debug("Loaded map from cache %s", path)
    return vma
