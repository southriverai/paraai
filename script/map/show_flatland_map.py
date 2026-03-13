"""
View MapBuilderFlatland result (std and planarity) for an arbitrary region.

Loads terrain (GDAL) before torch to avoid Windows access violation.

Example:
  poetry run python script/map/show_flatland_map.py --region bassano
  poetry run python script/map/show_flatland_map.py --region sopot --radius-m 300
  poetry run python script/map/show_flatland_map.py --region sopot --torch  # GPU version
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import rasterio.transform

from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_simple_climb_pixel import RepositorySimpleClimbPixel
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.setup import setup
from paraai.tool_spacetime import REGION_BOUNDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View flatland map (std and planarity)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--region", type=str, help="Named region (bassano, sopot, bansko, europe)")
    parser.add_argument("--radius-m", type=float, default=200, help="Radius for std/planarity (default 200)")
    parser.add_argument("--torch", action="store_true", help="Use PyTorch GPU (faster)")
    return parser.parse_args()


def main(region_name: str, bounding_box: BoundingBox) -> None:
    # CRITICAL: Load pixels and terrain (GDAL) before any torch import.
    # Torch + GDAL load order causes access violation on Windows if torch is first.
    logger.info("Loading pixels for %s", region_name)
    repo_pixel = RepositorySimpleClimbPixel.get_instance()
    _ = repo_pixel.get_all_in_bounding_box(
        bounding_box.lat_min, bounding_box.lat_max, bounding_box.lon_min, bounding_box.lon_max
    )

    repo_terrain = RepositoryTerrain.get_instance()
    logger.info("Loading terrain for %s", region_name)
    terrain = repo_terrain.get_elevation(bounding_box)
    elevation = terrain["elevation"]
    transform = terrain["transform"]

    bounds_arr = rasterio.transform.array_bounds(elevation.shape[0], elevation.shape[1], transform)
    extent = [bounds_arr[0], bounds_arr[2], bounds_arr[1], bounds_arr[3]]

    # Import torch only after GDAL has run
    if args.torch:
        logger.info("Using PyTorch GPU for flatland map")
        from paraai.map.map_builder_flatland_torch import MapBuilderFlatlandTorch

        builder = MapBuilderFlatlandTorch(radius_m=args.radius_m)
    else:
        logger.info("Using CPU for flatland map")
        from paraai.map.map_builder_flatland import MapBuilderFlatland

        builder = MapBuilderFlatland(radius_m=args.radius_m)

    logger.info("Building flatland map for %s (%s)", region_name, builder.name)
    maps = builder.build(bounding_box, pd.DataFrame())

    std_grid = np.flipud(maps["std"].array)
    planarity_grid = np.flipud(maps["planarity"].array)

    from paraai.map.show_climb_map import show_flatland_map

    show_flatland_map(
        elevation,
        extent,
        std_grid,
        planarity_grid,
        title=f"Flatland map: {region_name} (radius={args.radius_m}m)",
        radius_m=args.radius_m,
    )


if __name__ == "__main__":
    args = parse_args()
    bounding_box = REGION_BOUNDS.get(args.region.lower())
    if bounding_box is None:
        raise ValueError(f"Unknown region '{args.region}'. Available: {list(REGION_BOUNDS)}")
    region_name = args.region
    setup()
    main(region_name, bounding_box)
