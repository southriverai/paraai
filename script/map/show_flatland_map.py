"""
View MapBuilderFlatland result (std and planarity) for an arbitrary region.

Example:
  poetry run python script/map/show_flatland_map.py --region bassano
  poetry run python script/map/show_flatland_map.py --region sopot --radius-m 300
  poetry run python script/map/show_flatland_map.py --region sopot --torch  # GPU version
  poetry run python script/map/show_flatland_map.py --bbox 45.5 46 11.4 12.1
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import rasterio.transform

from paraai.map.map_builder_flatland_torch import MapBuilderFlatlandTorch
from paraai.map.show_climb_map import show_flatland_map
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.setup import setup
from paraai.tool_spacetime import REGION_BOUNDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View flatland map (std and planarity) for a region")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--region", type=str, help="Named region (bassano, sopot, bansko, europe)")
    group.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Bounding box: lat_min lat_max lon_min lon_max",
    )
    parser.add_argument(
        "--radius-m",
        type=float,
        default=200,
        help="Radius in meters for std/planarity (default 200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.region:
        bounds = REGION_BOUNDS.get(args.region.lower())
        if bounds is None:
            raise ValueError(f"Unknown region '{args.region}'. Available: {list(REGION_BOUNDS)}")
        lat_min, lat_max, lon_min, lon_max = bounds
        region_name = args.region
    else:
        lat_min, lat_max, lon_min, lon_max = args.bbox
        region_name = "custom"

    builder = MapBuilderFlatlandTorch(radius_m=args.radius_m)
    logger.info("Building flatland map for %s", region_name)
    maps = builder.build(lat_min, lat_max, lon_min, lon_max, pd.DataFrame())
    logger.info("Built flatland map for %s", region_name)

    repo_terrain = RepositoryTerrain.get_instance()
    terrain = repo_terrain.get_elevation(lon_min, lat_min, lon_max, lat_max)
    elevation = terrain["elevation"]
    transform = terrain["transform"]
    bounds_arr = rasterio.transform.array_bounds(elevation.shape[0], elevation.shape[1], transform)
    extent = [bounds_arr[0], bounds_arr[2], bounds_arr[1], bounds_arr[3]]

    std_grid = np.flipud(maps["std"].array)
    planarity_grid = np.flipud(maps["planarity"].array)

    show_flatland_map(
        elevation,
        extent,
        std_grid,
        planarity_grid,
        title=f"Flatland map: {region_name} (radius={args.radius_m}m)",
        radius_m=args.radius_m,
    )


if __name__ == "__main__":
    setup()
    main()
