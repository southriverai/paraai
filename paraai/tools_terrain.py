"""Load terrain elevation and vegetation/satellite data for a bounding box."""

from __future__ import annotations

import hashlib
import logging
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from scipy.ndimage import map_coordinates

if TYPE_CHECKING:
    from collections.abc import Sequence

TILE_SIZE = 256

logger = logging.getLogger(__name__)

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
SENTINEL2_COLLECTION = "sentinel-2-l2a"


def load_elevation(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    *,
    dem_resolution: int = 30,
    cache_dir: Path | None = None,
) -> dict:
    """
    Load elevation (DEM) only for a bounding box. Lighter than load_terrain (no Sentinel).

    Returns dict with elevation, transform, crs.
    """
    bbox = [lon_min, lat_min, lon_max, lat_max]
    cache = Path(cache_dir) if cache_dir else Path("data", "cache", "terrain")
    cache.mkdir(parents=True, exist_ok=True)
    dem_path = _download_dem(bbox, dem_resolution, cache)
    with rasterio.open(dem_path) as dem_src:
        elevation = dem_src.read(1)
        transform = dem_src.transform
        crs = dem_src.crs
    return {
        "elevation": elevation.astype(np.float32),
        "transform": transform,
        "crs": crs,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }


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

    red, green, blue, nir = _download_sentinel2(bbox, datetime_range, cloud_cover_max, dem_path)

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
    key = f"{tuple(round(x, 6) for x in bbox)}" f"{dem_resolution}{datetime_range}{cloud_cover_max}"
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
        raise ValueError(f"No Sentinel-2 imagery found for bbox {bbox} and datetime {datetime_range}")

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
            msg = f"{e}\n\nAvailable bands/assets for {SENTINEL2_COLLECTION}: " f"{sorted(available)}"
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


def _lon_to_tile_x(lon_deg: float, zoom: int) -> float:
    """Convert longitude to tile x (continuous)."""
    n = 2**zoom
    return (lon_deg + 180) / 360 * n


def _lat_to_tile_y(lat_deg: float, zoom: int) -> float:
    """Convert latitude to tile y (continuous)."""
    n = 2**zoom
    lat_rad = math.radians(lat_deg)
    return (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n


def _tile_x_to_lon(x: int, zoom: int) -> float:
    """Left edge of tile x in degrees."""
    n = 2**zoom
    return 360 * x / n - 180


def _tile_y_to_lat(y: int, zoom: int) -> float:
    """Top edge of tile y in degrees (y=0 is north)."""
    n = 2**zoom
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(lat_rad)


def image_to_slippy_tiles(
    image: np.ndarray,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    slippytilename: str,
    max_zoom: int = 11,  # Default zoom level for slippy lower than this is beyond what dem 30 likes
    *,
    output_dir: Path | None = None,
    cmap: str = "gray",
    verbose: bool = False,
) -> Path:
    """Convert a georeferenced image to slippy map PNG tiles.

    Image extent is (lon_min, lon_max, lat_min, lat_max). Origin: row 0 = lat_min, col 0 = lon_min.
    Tiles are written to output_dir/slippytilename/z/x/y.png, limited to the extent.

    Args:
        image: 2D array (H, W), values 0-1 or will be normalized. Or (H, W, 3) RGB uint8.
        lon_min, lat_min, lon_max, lat_max: Bounding box in degrees.
        slippytilename: Name of the tile layer (e.g. 'climb_strength').
        max_zoom: Maximum zoom level (0 to max_zoom inclusive).
        output_dir: Base directory. Default: data/slippy_tile.
        cmap: Matplotlib colormap name for 2D arrays. Ignored if image is RGB.
        verbose: Log progress per zoom level.

    Returns:
        Path to the tile directory (output_dir/slippytilename).
    """
    from matplotlib import cm as mpl_cm
    from matplotlib.colors import Normalize

    output_dir = output_dir or Path("data", "slippy_tile")
    tile_dir = output_dir / slippytilename
    tile_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info("Building slippy tiles: %s (zoom 0-%d)", slippytilename, max_zoom)

    h, w = image.shape[:2]
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    # Convert 2D to RGB if needed
    if image.ndim == 2:
        valid = image[~np.isnan(image)] if np.any(~np.isnan(image)) else np.array([0.0])
        vmin = float(np.min(valid)) if valid.size else 0.0
        vmax = float(np.max(valid)) if valid.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-10
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = mpl_cm.get_cmap(cmap)
        image_rgb = (cmap_obj(norm(np.nan_to_num(image, nan=vmin)))[:, :, :3] * 255).astype(np.uint8)
    else:
        image_rgb = np.asarray(image, dtype=np.uint8)
        if image_rgb.shape[2] == 4:
            image_rgb = image_rgb[:, :, :3]

    total_tiles = 0
    for zoom in range(max_zoom + 1):
        n = 2**zoom
        x_min = max(0, int(math.floor(_lon_to_tile_x(lon_min, zoom))))
        x_max = min(n - 1, int(math.floor(_lon_to_tile_x(lon_max, zoom))))
        y_min = max(0, int(math.floor(_lat_to_tile_y(lat_max, zoom))))  # lat_max = north = small y
        y_max = min(n - 1, int(math.floor(_lat_to_tile_y(lat_min, zoom))))  # lat_min = south = large y

        for tx in range(x_min, x_max + 1):
            for ty in range(y_min, y_max + 1):
                tile_lon_min = _tile_x_to_lon(tx, zoom)
                tile_lon_max = _tile_x_to_lon(tx + 1, zoom)
                tile_lat_max = _tile_y_to_lat(ty, zoom)
                tile_lat_min = _tile_y_to_lat(ty + 1, zoom)

                # Sample image: map tile pixel (i,j) to image coords
                jj, ii = np.meshgrid(np.arange(TILE_SIZE), np.arange(TILE_SIZE), indexing="xy")
                lon = tile_lon_min + (jj + 0.5) / TILE_SIZE * (tile_lon_max - tile_lon_min)
                lat = tile_lat_max - (ii + 0.5) / TILE_SIZE * (tile_lat_max - tile_lat_min)

                col = (lon - lon_min) / lon_range * (w - 1) if lon_range > 0 else np.full_like(lon, w / 2)
                row = (lat - lat_min) / lat_range * (h - 1) if lat_range > 0 else np.full_like(lat, h / 2)

                coords = np.array([row, col], dtype=np.float64)
                tile_data = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                for c in range(3):
                    tile_data[:, :, c] = map_coordinates(image_rgb[:, :, c], coords, order=1, mode="constant", cval=0).astype(np.uint8)

                out_path = tile_dir / str(zoom) / str(tx)
                out_path.mkdir(parents=True, exist_ok=True)
                png_path = out_path / f"{ty}.png"
                _save_png(tile_data, png_path)
                total_tiles += 1

    if verbose:
        logger.info("Slippy tiles complete: %s (%d tiles)", tile_dir, total_tiles)

    return tile_dir


def _save_png(rgb: np.ndarray, path: Path) -> None:
    """Save RGB uint8 array to PNG."""
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
    except ImportError:
        import matplotlib.pyplot as plt

        plt.imsave(path, rgb)
