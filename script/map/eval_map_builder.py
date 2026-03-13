"""
Test all map builders and show the best result.

Runs: Average, Convolution (100m, 200m), ConvolutionWeighted (100m, 200m).
Uses the same pixel split for all. Prints eval results. Shows chart for the best.

Usage examples:
  poetry run python script/map/eval_map_builder.py --region sopot
  poetry run python script/map/eval_map_builder.py --region bassano
  poetry run python script/map/eval_map_builder.py --region bansko --holdout-ratio 0.2
  poetry run python script/map/eval_map_builder.py --region europe --holdout-ratio 0.1 --seed 123
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random

import numpy as np
import pandas as pd
import rasterio.transform

from paraai.map.map_builder_average import MapBuilderAverage
from paraai.map.map_builder_convolution import MapBuilderConvolution
from paraai.map.map_builder_flatland_torch import MapBuilderFlatlandTorch
from paraai.map.show_climb_map import show_climb_map
from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.setup import setup
from paraai.tool_spacetime import REGION_BOUNDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test all map builders and show best result")
    parser.add_argument("--region", type=str, default="bassano", help="Region (bassano, sopot, bansko, europe)")
    parser.add_argument("--holdout-ratio", type=float, default=0.1, help="Fraction of pixels held out for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for holdout split")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    bounds = REGION_BOUNDS.get(args.region.lower())
    if bounds is None:
        raise ValueError(f"Unknown region '{args.region}'. Available: {list(REGION_BOUNDS)}")
    lat_min, lat_max, lon_min, lon_max = bounds

    repo_climb = RepositorySimpleClimb.get_instance()
    climbs = await repo_climb.get_all_in_bounding_box(lat_min, lat_max, lon_min, lon_max, verbose=True)
    # build a dataframe from the climbs
    climbs_df = pd.DataFrame(
        [(p.ground_lat, p.ground_lon, p.climb_strength_m_s) for p in climbs],
        columns=["lat", "lon", "strength"],
    )
    logger.info("Loaded %d SimpleClimbs in %s", len(climbs), args.region)

    if len(climbs) < 1:
        logger.warning("No pixels in region")
        return

    # build flatland map
    builder_flatland = MapBuilderFlatlandTorch()
    flatland_map = builder_flatland.build(
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    )

    # add the planarity to the dataframe
    climbs_df["planarity"] = flatland_map["planarity"].array[climbs_df["lat"].values, climbs_df["lon"].values]

    # remove all simple climbs from the dataframe that are in land with planarity < 0.5
    climbs_df = climbs_df[climbs_df["planarity"] >= 0.5]

    logger.info("Removed %d SimpleClimbs in land with planarity < 0.5", len(pixels) - len(pixels))

    # create average map with the remaining pixels
    builder_average = MapBuilderAverage()
    average_map = builder_average.build(lat_min, lat_max, lon_min, lon_max, climbs_df)

    # show the strength and count map
    show_climb_map(
        elevation,
        extent,
        average_map["count"].array,
        average_map["strength"].array,
        title=f"Average map: {args.region} (n_train={len(train_pixels)}, n_holdout={len(holdout_pixels)})",
        count_title="Climb count by pixel (Average)",
        strength_title="Mean climb strength by pixel (Average)",
    )

    # split the pixels into train and holdout
    rng = random.Random(args.seed)
    shuffled = pixels.copy()
    rng.shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * args.holdout_ratio))
    holdout_pixels = shuffled[:n_holdout]
    train_pixels = shuffled[n_holdout:]

    # create the train and holdout dataframes
    train_df = pd.DataFrame(
        [(p.lat, p.lon, p.climb_count, p.mean_climb_strength_m_s) for p in train_pixels],
        columns=["lat", "lon", "count", "strength"],
    )
    holdout_df = pd.DataFrame(
        [(p.lat, p.lon, p.climb_count, p.mean_climb_strength_m_s) for p in holdout_pixels],
        columns=["lat", "lon", "count", "strength"],
    )

    # show the train and holdout dataframes
    logger.info("Train dataframe: %s", train_df.head())
    # Same pixel split for all builders
    rng = random.Random(args.seed)
    shuffled = pixels.copy()
    rng.shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * args.holdout_ratio))
    holdout_pixels = shuffled[:n_holdout]
    train_pixels = shuffled[n_holdout:]

    train_df = pd.DataFrame(
        [(p.lat, p.lon, p.climb_count, p.mean_climb_strength_m_s) for p in train_pixels],
        columns=["lat", "lon", "count", "strength"],
    )
    holdout_df = pd.DataFrame(
        [(p.lat, p.lon, p.climb_count, p.mean_climb_strength_m_s) for p in holdout_pixels],
        columns=["lat", "lon", "count", "strength"],
    )

    # All builder configs: (display_name, builder_instance)
    configs = [
        ("Average", MapBuilderAverage()),
        ("Convolution 100m", MapBuilderConvolution(kernel_size_m=100)),
        ("Convolution 200m", MapBuilderConvolution(kernel_size_m=200)),
        # ("ConvolutionWeighted 100m", MapBuilderConvolutionWeighted(kernel_size_m=100)),
        # ("ConvolutionWeighted 200m", MapBuilderConvolutionWeighted(kernel_size_m=200)),
    ]

    results: list[tuple[str, object, dict]] = []

    for display_name, builder in configs:
        maps = builder.build(lat_min, lat_max, lon_min, lon_max, train_df)
        count_vma = maps["count"]
        strength_vma = maps["strength"]
        eval_result = builder.evaluate(
            count_vma,
            strength_vma,
            holdout_df,
            n_train=len(train_df),
        )
        results.append((display_name, eval_result, {"count_vma": count_vma, "strength_vma": strength_vma}))

    # Print all eval results
    print("\n" + "=" * 60)
    print(f"Map builder evaluation results (same split, n_train={len(train_pixels)}, n_holdout={len(holdout_pixels)})")
    print("=" * 60)
    for display_name, eval_result, _ in results:
        print(f"\n--- {display_name} ---")
        print(f"  count_mae:    {eval_result.count_mae:.4f}")
        print(f"  count_rmse:   {eval_result.count_rmse:.4f}")
        print(f"  strength_mae: {eval_result.strength_mae:.4f} m/s")
        print(f"  strength_rmse: {eval_result.strength_rmse:.4f} m/s")
    print("\n" + "=" * 60)

    # Best by strength_mae (lower is better)
    best_idx = min(range(len(results)), key=lambda i: results[i][1].strength_mae)
    best_name, best_eval, best_maps = results[best_idx]

    print(f"\nBest: {best_name} (strength_mae={best_eval.strength_mae:.4f} m/s)\n")

    # Load terrain for chart
    repo_terrain = RepositoryTerrain.get_instance()
    terrain = repo_terrain.get_elevation(lon_min, lat_min, lon_max, lat_max)
    elevation = terrain["elevation"]
    transform = terrain["transform"]
    bounds_arr = rasterio.transform.array_bounds(elevation.shape[0], elevation.shape[1], transform)
    extent = [bounds_arr[0], bounds_arr[2], bounds_arr[1], bounds_arr[3]]

    count_grid = np.flipud(best_maps["count_vma"].array)
    strength_grid = np.flipud(best_maps["strength_vma"].array)

    show_climb_map(
        elevation,
        extent,
        count_grid,
        strength_grid,
        title=f"{best_name}: {args.region} (n_train={len(train_pixels)}, n_holdout={len(holdout_pixels)})",
        count_title=f"Climb count by pixel ({best_name})",
        strength_title=f"Mean climb strength by pixel ({best_name})",
    )


if __name__ == "__main__":
    setup()
    asyncio.run(main())
