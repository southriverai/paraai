"""Build slippy tiles for a region using a trained map-builder model.

Usage examples:
  python script/map/build_slippy_tiles.py sopot 614b2e75e0e05813 16
  python script/map/build_slippy_tiles.py bassano 614b2e75e0e05813 8 --estimator-type time --max-zoom 11
"""

from __future__ import annotations

import argparse
import logging

import rasterio.transform

from paraai.map.map_builder_estimate_simple import MapBuilderEstimateSimple
from paraai.map.map_builder_estimate_time import MapBuilderEstimateTime
from paraai.setup import setup
from paraai.tool_spacetime import get_bounding_box
from paraai.tools_terrain import image_to_slippy_tiles

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build slippy tiles for a region/model/stride.")
    parser.add_argument("region_name", type=str, help="Region name, e.g. sopot, bassano, bansko, europe")
    parser.add_argument("model_id", type=str, help="Trained model id from RepositoryModels")
    parser.add_argument("stride", type=int, help="Grid stride for inference (pixels)")
    parser.add_argument(
        "--estimator-type",
        choices=["simple", "time"],
        default="time",
        help="Estimator implementation that owns the model id",
    )
    parser.add_argument(
        "--max-zoom",
        type=int,
        default=11,
        help="Maximum slippy zoom level",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.stride < 1:
        parser.error("stride must be >= 1")
    if args.max_zoom < 0:
        parser.error("max-zoom must be >= 0")
    return args


def create_map_builder(estimator_type: str) -> MapBuilderEstimateSimple | MapBuilderEstimateTime:
    if estimator_type == "simple":
        return MapBuilderEstimateSimple()
    if estimator_type == "time":
        return MapBuilderEstimateTime()
    raise ValueError(f"Unknown estimator type: {estimator_type}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    setup()

    bounding_box = get_bounding_box(args.region_name)
    map_builder = create_map_builder(args.estimator_type)
    maps = map_builder.build(
        bounding_box,
        ignore_cache=True,
        model_id=args.model_id,
        grid_stride=args.stride,
    )
    strength_map = maps["strength"]
    bounds = rasterio.transform.array_bounds(
        strength_map.array.shape[0],
        strength_map.array.shape[1],
        strength_map.transform,
    )

    layer_name = f"{args.model_id}_strength_s{args.stride}"
    path = image_to_slippy_tiles(
        strength_map.array.astype("float64"),
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        slippytilename=layer_name,
        max_zoom=args.max_zoom,
        cmap="coolwarm",
        verbose=True,
    )
    logger.info("Slippy tiles saved to %s", path)


if __name__ == "__main__":
    main()
