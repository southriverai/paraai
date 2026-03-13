"""
Build SimpleClimbPixels from SimpleClimbs in an area.

Aggregates climbs by DEM pixel (ground point snapped to 1 arc-second). Multiple climbs
in the same pixel are merged into one SimpleClimbPixel with mean climb strength and count.

Example:
  poetry run python script/climb/ingest_simple_climb_pixels.py --region bassano
  poetry run python script/climb/ingest_simple_climb_pixels.py --lat 59.91 --lon 10.75 --radius-km 50
  poetry run python script/climb/ingest_simple_climb_pixels.py --lat-min 45.5 --lat-max 46 --lon-min 11.4 --lon-max 12.1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from pathlib import Path

from paraai.model.simple_climb import SimpleClimb
from paraai.model.simple_climb_pixel import SimpleClimbPixel
from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.repository.repository_simple_climb_pixel import RepositorySimpleClimbPixel
from paraai.setup import setup
from paraai.tool_spacetime import REGION_BOUNDS, haversine_m

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data", "cache", "simple_climb_bbox")


def _cache_path(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> Path:
    key = f"{lat_min:.6f}_{lat_max:.6f}_{lon_min:.6f}_{lon_max:.6f}.json"
    return CACHE_DIR / key


async def _get_climbs_in_bbox(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> list[SimpleClimb]:
    """Load climbs in bounding box, from cache if available."""
    path = _cache_path(lat_min, lat_max, lon_min, lon_max)
    if path.exists():
        logger.info("Loading climbs from cache: %s", path)
        data = json.loads(path.read_text())
        return [SimpleClimb.model_validate(d) for d in data]
    repo_climb = RepositorySimpleClimb.get_instance()
    climbs = await repo_climb.get_all_in_bounding_box_by_ground(lat_min, lat_max, lon_min, lon_max, verbose=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.model_dump() for c in climbs]))
    logger.info("Cached %s climbs to %s", len(climbs), path)
    return climbs


def _climbs_to_pixels(climbs: list[SimpleClimb]) -> list[SimpleClimbPixel]:
    """Aggregate climbs by DEM pixel. Returns list of SimpleClimbPixel with unique pixels."""
    by_pixel: dict[str, SimpleClimbPixel] = {}
    for climb in climbs:
        if climb.climb_strength_m_s <= 0:
            continue
        pixel = SimpleClimbPixel.from_simple_climb(climb)
        if pixel.simple_climb_pixel_id in by_pixel:
            by_pixel[pixel.simple_climb_pixel_id].add_climb(climb)
        else:
            by_pixel[pixel.simple_climb_pixel_id] = pixel
    return list(by_pixel.values())


async def build_from_area(
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float | None = None,
    clear: bool = False,
) -> int:
    """Load SimpleClimbs in area, aggregate to pixels, insert. Returns pixel count."""
    repo_pixel = RepositorySimpleClimbPixel.get_instance()

    if lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
        logger.info("Loading climbs: bbox ground_lat=%s-%s, ground_lon=%s-%s", lat_min, lat_max, lon_min, lon_max)
        climbs = await _get_climbs_in_bbox(lat_min, lat_max, lon_min, lon_max)
        logger.info("Loaded %s climbs", len(climbs))
    elif lat is not None and lon is not None and radius_km is not None:
        logger.info("Loading climbs: radius %.0f km around (%.4f, %.4f)", radius_km, lat, lon)
        radius_m = radius_km * 1000
        deg_lat = radius_m / 111_000
        deg_lon = radius_m / (111_000 * math.cos(math.radians(lat)))
        bbox_climbs = await _get_climbs_in_bbox(lat - deg_lat, lat + deg_lat, lon - deg_lon, lon + deg_lon)
        climbs = [c for c in bbox_climbs if haversine_m(c.ground_lat, c.ground_lon, lat, lon) <= radius_m]
        logger.info("Loaded %s climbs (filtered from bbox)", len(climbs))
    else:
        raise ValueError("Specify --lat/--lon/--radius-km, or --lat-min/--lat-max/--lon-min/--lon-max, or --region <name>")

    if not climbs:
        logger.info("No climbs in area -> 0 pixels")
        return 0

    pixels = _climbs_to_pixels(climbs)
    logger.info("Aggregated to %s pixels", len(pixels))

    if clear:
        repo_pixel.clear_all()

    repo_pixel.upsert_many(pixels)
    logger.info("Upserted %s pixels", len(pixels))
    return len(pixels)


async def build_from_region(region_name: str, clear: bool = False) -> int:
    """Build pixels for a named region. Returns pixel count."""
    bounds = REGION_BOUNDS.get(region_name.lower())
    if bounds is None:
        raise ValueError(f"Unknown region '{region_name}'. Available: {list(REGION_BOUNDS)}")
    lat_min, lat_max, lon_min, lon_max = bounds
    return await build_from_area(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, clear=clear)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SimpleClimbPixels from SimpleClimbs in area")
    parser.add_argument("--lat", type=float, help="Center latitude (use with --lon, --radius-km)")
    parser.add_argument("--lon", type=float, help="Center longitude")
    parser.add_argument("--radius-km", type=float, help="Radius in km")

    parser.add_argument("--lat-min", type=float, help="Bounding box (all four required)")
    parser.add_argument("--lat-max", type=float)
    parser.add_argument("--lon-min", type=float)
    parser.add_argument("--lon-max", type=float)

    parser.add_argument("--region", type=str, help="Named region (e.g. bassano, sopot, europe)")

    parser.add_argument("--clear", action="store_true", help="Clear existing pixels before insert")
    args = parser.parse_args()

    if not any(
        [
            args.region,
            (args.lat and args.lon and args.radius_km),
            all(x is not None for x in (args.lat_min, args.lat_max, args.lon_min, args.lon_max)),
        ]
    ):
        parser.error("Specify --lat/--lon/--radius-km, or --lat-min/--lat-max/--lon-min/--lon-max, or --region <name>")
    return args


def main() -> None:
    args = parse_args()

    if args.region:
        asyncio.run(build_from_region(args.region, clear=args.clear))
    elif args.lat and args.lon and args.radius_km:
        asyncio.run(build_from_area(lat=args.lat, lon=args.lon, radius_km=args.radius_km, clear=args.clear))
    else:
        asyncio.run(
            build_from_area(
                lat_min=args.lat_min,
                lat_max=args.lat_max,
                lon_min=args.lon_min,
                lon_max=args.lon_max,
                clear=args.clear,
            )
        )


if __name__ == "__main__":
    setup()
    main()

# Run for specific regions:
#   poetry run python script/climb/ingest_simple_climb_pixels.py --region bassano
#   poetry run python script/climb/ingest_simple_climb_pixels.py --region sopot
#   poetry run python script/climb/ingest_simple_climb_pixels.py --region bansko
#   poetry run python script/climb/ingest_simple_climb_pixels.py --region europe
# Add --clear to replace existing pixels.
