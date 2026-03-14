"""
Test all map builders and show the best result.

Runs: Average, Convolution (100m, 200m), ConvolutionWeighted (100m, 200m).
Uses the same pixel split for all. Prints eval results. Shows chart for the best.

Usage examples:
  poetry run python script/map/eval_map_builder.py --region sopot
  poetry run python script/map/eval_map_builder.py --region bassano
  poetry run python script/map/eval_map_builder.py --region bansko --holdout-ratio 0.2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from paraai.map.map_builder_average import MapBuilderAverage
from paraai.map.map_builder_flatland_torch import MapBuilderFlatlandTorch
from paraai.map.show_climb_map import show_map_eval
from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.setup import setup
from paraai.tool_spacetime import REGION_BOUNDS

if TYPE_CHECKING:
    from paraai.model.boundingbox import BoundingBox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def partition_dataframe(
    df: pd.DataFrame,
    holdout_ratio: float,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and holdout by holdout_ratio. Uses seed for reproducibility."""
    if len(df) == 0:
        return df.copy(), df.copy()
    rng = random.Random(seed)
    indices = df.index.tolist()
    rng.shuffle(indices)
    n_holdout = max(1, int(len(indices) * holdout_ratio))
    holdout_idx = indices[:n_holdout]
    train_idx = indices[n_holdout:]
    return df.loc[train_idx].reset_index(drop=True), df.loc[holdout_idx].reset_index(drop=True)


async def main(region_name: str, bounding_box: BoundingBox) -> None:
    logger.info(f"Evaluating map builders for {region_name} with bounding box {bounding_box}")

    repo_climb = RepositorySimpleClimb.get_instance()
    repo_terrain = RepositoryTerrain.get_instance()

    climbs = await repo_climb.get_all_in_bounding_box_by_ground(
        bounding_box,
        verbose=True,
    )
    # build a dataframe from the climbs
    climbs_df = pd.DataFrame(
        [(p.ground_lat, p.ground_lon, p.climb_strength_m_s) for p in climbs],
        columns=["lat", "lon", "strength"],
    )
    logger.info("Loaded %d SimpleClimbs in %s", len(climbs), region_name)

    # build flatland map
    builder_flatland = MapBuilderFlatlandTorch()
    flatland_maps = builder_flatland.build(bounding_box, None)

    terrain = repo_terrain.get_elevation(bounding_box)
    elevation = terrain["elevation"].astype(np.float32)
    elevation = np.nan_to_num(elevation, nan=0.0)

    # add the planarity to the dataframe
    climbs_df["planarity"] = flatland_maps["planarity"].get_values(climbs_df["lat"].values, climbs_df["lon"].values)

    # remove all simple climbs from the dataframe that are in land with planarity > 0.5
    climbs_df_no_flatland = climbs_df[climbs_df["planarity"] <= 0.5]

    removed = len(climbs_df) - len(climbs_df_no_flatland)
    pct = 100.0 * removed / len(climbs_df) if len(climbs_df) > 0 else 0.0
    logger.info("Removed %d SimpleClimbs in land with planarity < 0.5 (%.1f%%)", removed, pct)

    # partition the dataframe into train and holdout (use flatland-filtered climbs)
    train_df, holdout_df = partition_dataframe(climbs_df_no_flatland, args.holdout_ratio, seed=args.seed)

    # show the train and holdout dataframes
    logger.info("Train dataframe: \n%s", train_df.head())

    # All builder configs: (display_name, builder_instance)
    from paraai.map.map_builder_convolution import MapBuilderConvolution

    configs = [
        ("Average", MapBuilderAverage()),
        ("Convolution 100m", MapBuilderConvolution(kernel_size_m=100)),
        ("Convolution 200m", MapBuilderConvolution(kernel_size_m=200)),
        # ("ConvolutionWeighted 100m", MapBuilderConvolutionWeighted(kernel_size_m=100)),
        # ("ConvolutionWeighted 200m", MapBuilderConvolutionWeighted(kernel_size_m=200)),
    ]

    results: list[tuple[str, object, dict]] = []

    for display_name, builder in configs:
        maps = builder.build(
            bounding_box,
            train_df,
            ignore_cache=True,
        )
        count_vma = maps["count"]
        strength_vma = maps["strength"]
        eval_result = builder.evaluate(
            strength_vma,
            holdout_df,
            column_name="strength",
            n_train=len(train_df),
        )
        results.append((display_name, eval_result, {"count_vma": count_vma, "strength_vma": strength_vma}))

    # Print all eval results
    print("\n" + "=" * 60)
    print(f"Map builder evaluation results (same split, n_train={len(train_df)}, n_holdout={len(holdout_df)})")
    print("=" * 60)
    for display_name, eval_result, _ in results:
        print(display_name)
        print(f"  strength_mae={eval_result.strength_mae:.4f} m/s, strength_rmse={eval_result.strength_rmse:.4f} m/s")
    print("\n" + "=" * 60)

    # Best by strength_mae (lower is better)
    best_idx = min(range(len(results)), key=lambda i: results[i][1].strength_mae)
    best_name, best_eval, best_maps = results[best_idx]

    print(f"\nBest: {best_name} (strength_mae={best_eval.strength_mae:.4f} m/s)\n")

    show_map_eval(
        best_maps["strength_vma"],
        holdout_df,
        column_name="strength",
        title=f"{best_name}: {region_name}",
        elevation=elevation,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test all map builders and show best result")
    parser.add_argument("--region", type=str, default="bassano", help="Region (bassano, sopot, bansko, europe)")
    parser.add_argument("--holdout-ratio", type=float, default=0.1, help="Fraction of pixels held out for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for holdout split")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    region_name = args.region
    bounding_box = REGION_BOUNDS.get(region_name.lower())
    if bounding_box is None:
        raise ValueError(f"Unknown region '{region_name}'. Available: {list(REGION_BOUNDS)}")
    setup()
    asyncio.run(main(region_name, bounding_box))
