"""
Query SimpleClimbPixels in a region and show them over a height map.

Example:
  poetry run python script/simple_climb_pixel/show_simple_climb_pixels.py --region bassano
  poetry run python script/simple_climb_pixel/show_simple_climb_pixels.py --region bassano --gaussian-sigma-m 100
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import rasterio.transform
from scipy.ndimage import convolve

from paraai.map.show_climb_map import show_climb_map
from paraai.repository.repository_simple_climb_pixel import RepositorySimpleClimbPixel
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.setup import setup
from paraai.tool_spacetime import REGION_BOUNDS, build_gaussian_kernel_meters

logger = logging.getLogger(__name__)


def show_region(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    region_name: str | None = None,
    gaussian_sigma_m: float = 0,
    save_slippy: str | None = None,
) -> None:
    """Load pixels in bbox, load elevation, plot together."""
    repo_pixel = RepositorySimpleClimbPixel.get_instance()
    repo_terrain = RepositoryTerrain.get_instance()
    pixels = repo_pixel.get_all_in_bounding_box(lat_min, lat_max, lon_min, lon_max)
    logger.info("Loaded %s SimpleClimbPixels in region", len(pixels))
    terrain = repo_terrain.get_elevation(lon_min, lat_min, lon_max, lat_max)
    elevation = terrain["elevation"]
    transform = terrain["transform"]

    # Use exact raster bounds so both charts share identical extent and resolution
    bounds = rasterio.transform.array_bounds(elevation.shape[0], elevation.shape[1], transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]  # left, right, bottom, top

    # Build count and strength grids: same shape as elevation
    count_grid = np.zeros(elevation.shape, dtype=np.float32)
    strength_grid = np.zeros(elevation.shape, dtype=np.float32)
    for p in pixels:
        row, col = rasterio.transform.rowcol(transform, [p.lon], [p.lat])
        r, c = int(row[0]), int(col[0])
        if 0 <= r < count_grid.shape[0] and 0 <= c < count_grid.shape[1]:
            count_grid[r, c] = p.climb_count
            strength_grid[r, c] = p.mean_climb_strength_m_s

    if gaussian_sigma_m > 0:
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        kernel = build_gaussian_kernel_meters(gaussian_sigma_m, center_lat, center_lon)
        logger.info("Gaussian kernel: %s", kernel.shape)
        count_grid = convolve(count_grid, kernel, mode="constant", cval=0)
        strength_grid = convolve(strength_grid, kernel, mode="constant", cval=0)

    count_grid = np.flipud(count_grid)
    strength_grid = np.flipud(strength_grid)

    region_title = f"SimpleClimbPixels in {region_name}" if region_name else "SimpleClimbPixels"
    show_climb_map(
        elevation,
        extent,
        count_grid,
        strength_grid,
        title=f"{region_title} (n={len(pixels)})",
        save_slippy=save_slippy,
        region_name=region_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show SimpleClimbPixels over height map")
    parser.add_argument("--region", type=str, help="Named region (bassano, sopot, bansko, europe)")
    parser.add_argument("--save-slippy", type=str, metavar="NAME", help="Export to slippy tiles (e.g. bassano)")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    bounds = REGION_BOUNDS.get(args.region.lower())
    if bounds is None:
        raise ValueError(f"Unknown region '{args.region}'. Available: {list(REGION_BOUNDS)}")
    lat_min, lat_max, lon_min, lon_max = bounds
    gaussian_sigma_m = 100
    show_region(
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        region_name=args.region,
        gaussian_sigma_m=gaussian_sigma_m,
        save_slippy=args.save_slippy,
    )


if __name__ == "__main__":
    setup()
    main()
