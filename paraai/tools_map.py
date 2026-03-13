"""Save and load map arrays to/from GeoTIFF with cache keying."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import rasterio

from paraai.map.vectror_map_array import VectorMapArray

logger = logging.getLogger(__name__)


def _map_cache_hash(
    generator_name: str,
    map_name: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    params: dict,
) -> str:
    """Hash for cache key: generator + map_name + bbox + params."""
    data = {
        "generator": generator_name,
        "map": map_name,
        "bbox": [lat_min, lat_max, lon_min, lon_max],
        "params": dict(sorted(params.items())),
    }
    s = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def save_map(
    vma: VectorMapArray,
    cache_dir: Path,
    generator_name: str,
    map_name: str | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    **params: object,
) -> Path:
    """Save VectorMapArray to cache. Returns path to saved file."""
    lat_min = lat_min if lat_min is not None else vma.lat_min
    lat_max = lat_max if lat_max is not None else vma.lat_max
    lon_min = lon_min if lon_min is not None else vma.lon_min
    lon_max = lon_max if lon_max is not None else vma.lon_max
    map_name = map_name or vma.map_name

    h = _map_cache_hash(generator_name, map_name, lat_min, lat_max, lon_min, lon_max, params)
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
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    **params: object,
) -> VectorMapArray | None:
    """Load VectorMapArray from cache if exists. Returns None if not cached."""
    h = _map_cache_hash(generator_name, map_name, lat_min, lat_max, lon_min, lon_max, params)
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
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        array=array.astype(np.float32),
    )
    logger.debug("Loaded map from cache %s", path)
    return vma
