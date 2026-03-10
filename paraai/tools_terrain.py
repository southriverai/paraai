"""Load terrain elevation and vegetation/satellite data for a bounding box."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio

if TYPE_CHECKING:
    from collections.abc import Sequence

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
SENTINEL2_COLLECTION = "sentinel-2-l2a"


def load_terrain(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    *,
    dem_resolution: int = 30,
    datetime_range: str = "2023-01/2023-12",
    cloud_cover_max: float = 20.0,
    cache_dir: Path | None = None,
) -> dict:
    """
    Load elevation, RGB satellite imagery, and NDVI for a bounding box.

    Returns terrain dict with elevation, red, green, blue, nir, rgb, ndvi, transform, crs.
    """
    bbox = [lon_min, lat_min, lon_max, lat_max]
    cache = Path(cache_dir) if cache_dir else None

    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        h = _terrain_cache_hash(bbox, dem_resolution, datetime_range, cloud_cover_max)
        cache_path = cache / f"terrain_{h}.npz"
        cached = _load_terrain_cache(cache_path)
        if cached is not None:
            print(f"Terrain: Using cached {cache_path.name} for bbox lon={lon_min:.4f}-{lon_max:.4f}, lat={lat_min:.4f}-{lat_max:.4f}")
            return cached

    dem_path = _download_dem(bbox, dem_resolution, cache)

    with rasterio.open(dem_path) as dem_src:
        elevation = dem_src.read(1)
        transform = dem_src.transform
        crs = dem_src.crs

    red, green, blue, nir = _download_sentinel2(
        bbox, datetime_range, cloud_cover_max, dem_path
    )

    rgb = _build_rgb(red, green, blue)
    ndvi = _compute_ndvi(red, nir)

    terrain = {
        "elevation": elevation.astype(np.float32),
        "red": red,
        "green": green,
        "blue": blue,
        "nir": nir,
        "rgb": rgb,
        "ndvi": ndvi.astype(np.float32),
        "transform": transform,
        "crs": crs,
    }

    if cache:
        h = _terrain_cache_hash(bbox, dem_resolution, datetime_range, cloud_cover_max)
        cache_path = cache / f"terrain_{h}.npz"
        _save_terrain_cache(cache_path, terrain)

    return terrain


def _bbox_hash(bbox: Sequence[float], resolution: int) -> str:
    """Stable hash for cache key from bbox and resolution."""
    key = f"{tuple(round(x, 6) for x in bbox)}{resolution}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _terrain_cache_hash(
    bbox: Sequence[float],
    dem_resolution: int,
    datetime_range: str,
    cloud_cover_max: float,
) -> str:
    """Stable hash for full terrain cache key."""
    key = (
        f"{tuple(round(x, 6) for x in bbox)}"
        f"{dem_resolution}{datetime_range}{cloud_cover_max}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _load_terrain_cache(cache_path: Path) -> dict | None:
    """Load terrain from cache file. Returns None if missing or invalid."""
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        transform = rasterio.Affine(*data["transform"])
        crs = rasterio.crs.CRS.from_string(str(data["crs"]))
        return {
            "elevation": data["elevation"].astype(np.float32),
            "red": data["red"],
            "green": data["green"],
            "blue": data["blue"],
            "nir": data["nir"],
            "rgb": data["rgb"],
            "ndvi": data["ndvi"].astype(np.float32),
            "transform": transform,
            "crs": crs,
        }
    except (OSError, KeyError, ValueError):
        return None


def _save_terrain_cache(cache_path: Path, terrain: dict) -> None:
    """Save terrain to cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    transform = terrain["transform"]
    transform_arr = [
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    ]
    np.savez_compressed(
        cache_path,
        elevation=terrain["elevation"],
        red=terrain["red"],
        green=terrain["green"],
        blue=terrain["blue"],
        nir=terrain["nir"],
        rgb=terrain["rgb"],
        ndvi=terrain["ndvi"],
        transform=np.array(transform_arr),
        crs=np.array(str(terrain["crs"])),
    )


def _download_dem(
    bbox: Sequence[float],
    resolution: int,
    cache_dir: Path | None,
) -> Path:
    """Download Copernicus DEM for bbox, return path to GeoTIFF."""
    from demloader.download import from_aws
    from demloader.prefixes import get_from_aoi

    left, bottom, right, top = bbox
    aoi = [left, bottom, right, top]
    bbox_str = f"lon={left:.4f}-{right:.4f}, lat={bottom:.4f}-{top:.4f}"

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        h = _bbox_hash(bbox, resolution)
        out_path = cache_dir / f"dem_{h}.tif"
        if out_path.exists():
            print(f"DEM: Using cached {out_path.name} for bbox ({bbox_str}), {resolution}m resolution")
            return out_path
    else:
        h = _bbox_hash(bbox, resolution)
        out_path = Path(f"dem_{h}.tif")

    print(f"DEM: Downloading Copernicus DEM for bbox ({bbox_str}), {resolution}m resolution -> {out_path}")
    prefixes = get_from_aoi(aoi, resolution=resolution)
    print(f"DEM: Fetching from AWS ({len(prefixes)} tile(s))...")
    from_aws(prefixes, resolution=resolution, out_path=str(out_path))
    print(f"DEM: Saved to {out_path}")

    return out_path


def _download_sentinel2(
    bbox: Sequence[float],
    datetime_range: str,
    cloud_cover_max: float,
    dem_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download Sentinel-2 bands B02, B03, B04, B08 for bbox, aligned to DEM grid."""
    from odc.geo.geobox import GeoBox
    from odc.stac import load
    from pystac_client import Client

    left, bottom, right, top = bbox
    bbox_str = f"lon={left:.4f}-{right:.4f}, lat={bottom:.4f}-{top:.4f}"
    print(f"Sentinel-2: Searching {SENTINEL2_COLLECTION} for bbox ({bbox_str}), datetime={datetime_range}, cloud<{cloud_cover_max}%")

    catalog = Client.open(EARTH_SEARCH_URL)
    search = catalog.search(
        collections=[SENTINEL2_COLLECTION],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
    )
    items = list(search.items())

    if not items:
        print(f"Sentinel-2: No scenes with cloud<{cloud_cover_max}%, retrying without cloud filter...")
        search_any = catalog.search(
            collections=[SENTINEL2_COLLECTION],
            bbox=bbox,
            datetime=datetime_range,
        )
        items = list(search_any.items())
        if items:
            warnings.warn(
                f"No scenes with cloud cover < {cloud_cover_max}%; using least cloudy scene.",
                UserWarning,
                stacklevel=2,
            )

    if not items:
        raise ValueError(
            f"No Sentinel-2 imagery found for bbox {bbox} and datetime {datetime_range}"
        )

    items_sorted = sorted(
        items,
        key=lambda i: i.properties.get("eo:cloud_cover", 100),
    )
    best = items_sorted[0]
    scene_id = best.id
    cloud = best.properties.get("eo:cloud_cover", "?")
    date = best.properties.get("datetime", best.properties.get("start_datetime", "?"))
    print(f"Sentinel-2: Using scene {scene_id} (cloud={cloud}%, date={date})")

    with rasterio.open(dem_path) as dem_src:
        dem_geobox = GeoBox.from_rio(dem_src)

    bands_needed = ["blue", "green", "red", "nir"]
    print(f"Sentinel-2: Downloading bands {bands_needed} for bbox ({bbox_str}), aligned to DEM grid...")
    try:
        ds = load(
            [best],
            bands=bands_needed,
            geobox=dem_geobox,
            dtype="uint16",
            nodata=0,
        )
    except ValueError as e:
        if "No such band" in str(e):
            available = list(best.assets.keys())
            msg = (
                f"{e}\n\nAvailable bands/assets for {SENTINEL2_COLLECTION}: "
                f"{sorted(available)}"
            )
            raise ValueError(msg) from e
        raise

    red = ds.red.isel(time=0).values
    green = ds.green.isel(time=0).values
    blue = ds.blue.isel(time=0).values
    nir = ds.nir.isel(time=0).values

    print(f"Sentinel-2: Loaded bands {bands_needed}, shape {red.shape}")
    return red, green, blue, nir


def _compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI = (NIR - Red) / (NIR + Red)."""
    r = red.astype(np.float64)
    n = nir.astype(np.float64)
    denom = n + r + 1e-10
    ndvi = (n - r) / denom
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def _build_rgb(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    *,
    percentile: tuple[float, float] = (2, 98),
) -> np.ndarray:
    """Stack B04, B03, B02 into RGB (H, W, 3) uint8 with percentile stretch."""
    stacked = np.stack([red, green, blue], axis=-1)
    valid = stacked[stacked > 0]
    if valid.size > 0:
        p_low, p_high = np.percentile(valid, percentile)
        stretched = np.clip((stacked - p_low) / (p_high - p_low + 1e-10), 0, 1)
    else:
        stretched = np.clip(stacked.astype(np.float64) / 3000.0, 0, 1)
    return (stretched * 255).astype(np.uint8)
