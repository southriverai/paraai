"""Train a CNN to estimate MapBuilderConvolution(200) strength map from elevation patches.

Modes:
  dataset    - Build (elevation patches, target maps) from region and save to RepositoryDatasets.
  train      - Load dataset from RepositoryDatasets (by builder + region + params), train the CNN, save model.
  estimate   - Load trained model, run inference on region, produce strength map (no evaluation).
  eval       - Evaluate on holdout data, print MAE/RMSE to console (requires trained model).
  train_eval - Train model, then evaluate on holdout (inference on test patches only, no map building).
  eval_show  - Evaluate on holdout data, print scores and show map chart.
  show_patch - Show two random elevation patches from the training set.

Usage examples (dataset-type and estimator-type always first):

  # 1. Build dataset for region (saves to RepositoryDatasets cache)
  python script/map/train_map_builder.py --dataset-type climb --estimator-type simple --mode dataset --region sopot

  # 2. Train model (loads dataset from RepositoryDatasets by builder + region + params)
  python script/map/train_map_builder.py --dataset-type climb --estimator-type simple --mode train --region sopot --epochs 10

  # 3. Estimate strength map (default mode, cached by MapBuilderBase)
  python script/map/train_map_builder.py --dataset-type climb --estimator-type simple --region sopot

  # 4. Evaluate on holdout data (print scores only)
  python script/map/train_map_builder.py --dataset-type climb --estimator-type simple --mode eval --region sopot

  # 5. Train then evaluate (no pre-trained model needed)
  python script/map/train_map_builder.py --dataset-type climb --estimator-type time --mode train_eval --region sopot

  # 6. Evaluate and show map chart
  python script/map/train_map_builder.py --dataset-type climb --estimator-type simple --mode eval_show --region sopot

  # 7. Show two random elevation patches from the training set
  python script/map/train_map_builder.py --dataset-type climb --estimator-type simple --mode show_patch --region sopot
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn

from paraai.map.dataset_builder import DatasetBuilder
from paraai.map.map_builder_estimate_base import MapBuilderEstimateBase
from paraai.map.map_builder_estimate_simple import MapBuilderEstimateSimple
from paraai.map.map_builder_estimate_time import MapBuilderEstimateTime
from paraai.model.boundingbox import BoundingBox
from paraai.model.train_log import TrainLog
from paraai.repository.repository_models import RepositoryModels
from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.repository.repository_train_logs import RepositoryTrainLogs
from paraai.setup import setup
from paraai.tool_spacetime import get_bounding_box

if TYPE_CHECKING:
    from paraai.model.simple_climb import SimpleClimb

logger = logging.getLogger(__name__)

BATCH_SIZE = 1024  # TODO


def _alt_norm(alt_m: float) -> float:
    """Rescale altitude to 0-1 by dividing by 10000."""
    return float(np.clip(alt_m / 10000.0, 0.0, 1.0))


def _climbs_to_eval_df(climbs: list[SimpleClimb]) -> pd.DataFrame:
    """Convert climbs to DataFrame with lat, lon, strength, time, altitude for evaluation."""
    return pd.DataFrame(
        [
            {
                "lat": c.ground_lat,
                "lon": c.ground_lon,
                "strength": c.climb_strength_m_s,
                "time_of_day_h": c.time_of_day_h,
                "time_of_year_d": c.time_of_year_d,
                "ground_alt_norm": _alt_norm(c.ground_alt_m),
                "start_alt_norm": _alt_norm(c.start_alt_m),
                "end_alt_norm": _alt_norm(c.end_alt_m),
            }
            for c in climbs
        ],
    )


async def run_dataset_mode(
    region: str,
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
) -> None:
    """Build dataset from region and save to RepositoryDatasets cache."""
    bounding_box = get_bounding_box(region)

    data = await dataset_builder.build_dataset(
        bounding_box,
        ignore_cache=True,
    )

    n_train = len(data["input_maps_train"])
    n_test = len(data["input_maps_test"])
    logger.info("Dataset (%s train, %s test patches) built and cached in RepositoryDatasets", n_train, n_test)


def _plot_all_training_results(current_log_path: Path | None = None) -> None:
    """Load all train logs from repository and plot train/test scores. Highlight current run if given."""
    from paraai.repository.repository_train_logs import RepositoryTrainLogs

    repo = RepositoryTrainLogs.get_instance()
    log_paths = repo.list_train_logs()
    if not log_paths:
        logger.warning("No training result files found in repository")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    for fp in log_paths:
        try:
            with fp.open(encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load %s: %s", fp, e)
            continue
        epochs_data = data.get("epochs", [])
        if not epochs_data:
            continue
        epochs = [e["epoch"] for e in epochs_data]
        train_scores = [e["train_score"] for e in epochs_data]
        test_scores = [e["test_score"] for e in epochs_data]
        region = data.get("region", "")
        builder = data.get("builder", "")
        label = f"{region}_{builder}" if region and builder else fp.stem
        is_current = current_log_path is not None and fp.resolve() == current_log_path.resolve()
        if is_current:
            ax1.plot(epochs, train_scores, label=f"{label} (current)", linewidth=3, color="red")
            ax2.plot(epochs, test_scores, label=f"{label} (current)", linewidth=3, color="red")
        else:
            ax1.plot(epochs, train_scores, label=label, alpha=0.6)
            ax2.plot(epochs, test_scores, label=label, alpha=0.6)
    ax1.set_ylabel("Train loss")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.set_ylabel("Test loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.suptitle("Training results (all runs)")
    plt.tight_layout()
    plt.show()


async def run_train_mode(
    region: str,
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
) -> None:
    """Load dataset from RepositoryDatasets (or build from region if missing), train model, save model."""
    repo_models = RepositoryModels.get_instance()
    bounding_box = get_bounding_box(region)
    data = await dataset_builder.build_dataset(
        bounding_box,
    )
    dataset_id = data["dataset_id"]
    model_id = map_builder.get_model_id(dataset_id)

    input_maps_train = data["input_maps_train"]
    target_maps_train = data["target_maps_train"]
    input_maps_test = data["input_maps_test"]
    target_maps_test = data["target_maps_test"]
    meta = data["metadata"]
    in_c = input_maps_train[0].shape[0]
    out_c = target_maps_train[0].shape[0]
    image_size = meta["image_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    use_time = "time_of_day_train" in data

    # Preload all data to GPU once (avoids per-batch transfer overhead)
    logger.info("Loading data to %s...", device)
    train_inp = torch.stack(input_maps_train).to(device)
    train_tgt = torch.stack(target_maps_train).to(device)
    test_inp = torch.stack(input_maps_test).to(device)
    test_tgt = torch.stack(target_maps_test).to(device)
    if use_time:
        time_day_train = data["time_of_day_train"]
        time_year_train = data["time_of_year_train"]
        time_day_test = data["time_of_day_test"]
        time_year_test = data["time_of_year_test"]
        ground_alt_train = data["ground_alt_train"]
        start_alt_train = data["start_alt_train"]
        end_alt_train = data["end_alt_train"]
        ground_alt_test = data["ground_alt_test"]
        start_alt_test = data["start_alt_test"]
        end_alt_test = data["end_alt_test"]
        train_td = torch.tensor(time_day_train, dtype=torch.float32, device=device)
        train_ty = torch.tensor(time_year_train, dtype=torch.float32, device=device)
        test_td = torch.tensor(time_day_test, dtype=torch.float32, device=device)
        test_ty = torch.tensor(time_year_test, dtype=torch.float32, device=device)
        train_ga = torch.tensor(ground_alt_train, dtype=torch.float32, device=device)
        train_sa = torch.tensor(start_alt_train, dtype=torch.float32, device=device)
        train_ea = torch.tensor(end_alt_train, dtype=torch.float32, device=device)
        test_ga = torch.tensor(ground_alt_test, dtype=torch.float32, device=device)
        test_sa = torch.tensor(start_alt_test, dtype=torch.float32, device=device)
        test_ea = torch.tensor(end_alt_test, dtype=torch.float32, device=device)

    model = map_builder._get_model_class()(in_channels=in_c, out_channels=out_c, size=image_size).to(device)
    # print model architecture
    logger.info("Model class: %s", map_builder._get_model_class())
    logger.info("Model architecture: %s", model)
    optimizer = torch.optim.Adam(model.parameters(), lr=map_builder.lr)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    best_test_loss = float("inf")
    best_state: dict | None = None

    n_train = len(train_inp)
    n_test = len(test_inp)

    repo_train_logs = RepositoryTrainLogs.get_instance()
    training_metrics: list[dict] = []

    interrupted = False

    def _sigint_handler(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        logger.info("Interrupt received. Finishing current epoch...")

    signal.signal(signal.SIGINT, _sigint_handler)

    t_start = time.perf_counter()
    for epoch in range(map_builder.epochs):
        epoch_start = time.perf_counter()
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = 0.0
        n_b = 0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            batch_inp = train_inp[idx]
            batch_tgt = train_tgt[idx]
            if use_time:
                batch_td = train_td[idx]
                batch_ty = train_ty[idx]
                batch_ga = train_ga[idx]
                batch_sa = train_sa[idx]
                batch_ea = train_ea[idx]
                pred = model(batch_inp, batch_td, batch_ty, batch_ga, batch_sa, batch_ea)
            else:
                pred = model(batch_inp)
                if pred.shape[2:] != batch_tgt.shape[2:]:
                    pred = nn.functional.interpolate(pred, size=batch_tgt.shape[2:], mode="bilinear")
            loss = criterion(pred, batch_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_b += 1
        train_loss /= n_b if n_b else 1

        model.eval()
        test_loss = 0.0
        n_b = 0
        with torch.no_grad():
            for i in range(0, n_test, BATCH_SIZE):
                batch_inp = test_inp[i : i + BATCH_SIZE]
                batch_tgt = test_tgt[i : i + BATCH_SIZE]
                if use_time:
                    batch_td = test_td[i : i + BATCH_SIZE]
                    batch_ty = test_ty[i : i + BATCH_SIZE]
                    batch_ga = test_ga[i : i + BATCH_SIZE]
                    batch_sa = test_sa[i : i + BATCH_SIZE]
                    batch_ea = test_ea[i : i + BATCH_SIZE]
                    pred = model(batch_inp, batch_td, batch_ty, batch_ga, batch_sa, batch_ea)
                else:
                    pred = model(batch_inp)
                test_loss += criterion(pred, batch_tgt).item()
                n_b += 1
        test_loss /= n_b if n_b else 1

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.perf_counter() - t_start
        epoch_duration = time.perf_counter() - epoch_start
        training_metrics.append(
            {
                "epoch": epoch + 1,
                "train_score": float(train_loss),
                "test_score": float(test_loss),
                "epoch_duration_s": float(epoch_duration),
                "total_time_s": float(elapsed),
            }
        )
        train_log = TrainLog(region=region, builder_name=map_builder.name, epochs=training_metrics)
        log_path = repo_train_logs.save_train_log(map_builder.name, model_id, train_log)

        eta_sec = (elapsed / (epoch + 1)) * (map_builder.epochs - epoch - 1) if epoch < map_builder.epochs - 1 else 0
        eta_str = f", ETA {eta_sec:.0f}s" if eta_sec > 0 else ""
        logger.info(
            "Epoch %d/%d: train_loss=%.4f test_loss=%.4f (%.1fs)%s",
            epoch + 1,
            map_builder.epochs,
            train_loss,
            test_loss,
            elapsed,
            eta_str,
        )

        if interrupted:
            logger.info("Training interrupted by user.")
            _plot_all_training_results(log_path)
            return

    if best_state is not None:
        model.load_state_dict(best_state)

    train_log = TrainLog(region=region, builder_name=map_builder.name, epochs=training_metrics)
    log_path = repo_train_logs.save_train_log(map_builder.name, model_id, train_log)

    repo_models.save_model(
        map_builder.name,
        model_id,
        in_channels=in_c,
        out_channels=out_c,
        image_size=meta["image_size"],
        patch_size_m=meta["patch_size_m"],
        grid_stride=meta["grid_stride"],
        state_dict=model.state_dict(),
        strength_lo=meta["strength_lo"],
        strength_hi=meta["strength_hi"],
    )
    logger.info("Saved model to RepositoryModels")


async def run_show_patch_mode(
    region: str,
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
) -> None:
    """Show elevation map with patch locations, and two random patches in separate charts."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    bounding_box = get_bounding_box(region)
    data = await dataset_builder.build_dataset(
        bounding_box,
    )

    input_maps_train = data["input_maps_train"]
    target_maps_train = data["target_maps_train"]
    lats_train = data["lats_train"]
    lons_train = data["lons_train"]
    meta = data["metadata"]
    patch_size_m = meta["patch_size_m"]

    n = len(input_maps_train)
    if n < 2:
        raise ValueError(f"Need at least 2 training patches, got {n}")

    rng = np.random.default_rng(dataset_builder.split_seed)
    indices = rng.choice(n, size=2, replace=False)

    # Load elevation for the full map
    repo_terrain = RepositoryTerrain.get_instance()
    terrain = repo_terrain.get_elevation(bounding_box)
    elevation = terrain["elevation"]
    transform = terrain["transform"]
    bounds = rasterio.transform.array_bounds(elevation.shape[0], elevation.shape[1], transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]  # left, right, bottom, top

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)

    # Left: full elevation map with dotted patch outlines
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.imshow(elevation, extent=extent, cmap="terrain", aspect="auto")
    colors = ["red", "blue"]
    for i, idx in enumerate(indices):
        lat, lon = lats_train[idx], lons_train[idx]
        bbox = BoundingBox.from_latlon_radius(lat, lon, patch_size_m)
        lon_min, lat_min, lon_max, lat_max = bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max
        rect = mpatches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=2,
            edgecolor=colors[i],
            facecolor="none",
            linestyle=":",
        )
        ax_map.add_patch(rect)
        ax_map.plot(lon, lat, "o", color=colors[i], markersize=4)
    ax_map.set_xlim(extent[0], extent[1])
    ax_map.set_ylim(extent[2], extent[3])
    ax_map.set_title("Elevation map (patch locations)")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_aspect("equal")

    # Right: two patches in separate charts
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(gs[i, 1])
        patch = input_maps_train[idx]  # (1, H, W)
        target = target_maps_train[idx]
        strength_val = target[0].item() if target.dim() == 1 else target[0, 0, 0].item()
        img = patch[0].numpy()
        ax.imshow(img, cmap="terrain", vmin=0, vmax=1)
        ax.set_title(f"Patch {idx + 1} (strength={strength_val:.3f})")
        ax.axis("off")
    plt.suptitle("Two random elevation patches from training set")
    plt.tight_layout()
    plt.show()


async def run_estimate_mode(
    region: str,
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
) -> None:
    """Load trained model from RepositoryModels, run inference on region, cache strength map."""

    bounding_box = get_bounding_box(region)
    repo_models = RepositoryModels.get_instance()
    dataset_id = dataset_builder.get_dataset_id(bounding_box)
    model_id = map_builder.get_model_id(dataset_id)
    model_data = repo_models.get_model(map_builder.name, model_id)
    if model_data is None:
        raise FileNotFoundError(f"Model not found in RepositoryModels for {map_builder.name}. Run train mode first.")

    map_builder.build(bounding_box, ignore_cache=True, model_id=model_id)
    logger.info("Built and cached strength map")


async def run_train_eval_mode(
    region: str,
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
) -> None:
    """Train model, then evaluate on holdout. No map building - inference on test patches only."""
    await run_train_mode(region, dataset_builder, map_builder)
    bounding_box = get_bounding_box(region)
    dataset_id = dataset_builder.get_dataset_id(bounding_box)
    model_id = map_builder.get_model_id(dataset_id)
    repo_simple_climb = RepositorySimpleClimb.get_instance()
    climbs = await repo_simple_climb.get_all_in_bounding_box_by_ground(bounding_box, verbose=True)
    train_climbs, holdout_climbs = dataset_builder.split_climbs(climbs)
    holdout_df = _climbs_to_eval_df(holdout_climbs)
    eval_result = map_builder.evaluate(
        bounding_box=bounding_box,
        evaluate_df=holdout_df,
        model_id=model_id,
    )
    print("\n" + "=" * 60)
    print(f"Map evaluation (n_train={len(train_climbs)}, n_holdout={len(holdout_climbs)})")
    print("=" * 60)
    print(f"strength_mae={eval_result.strength_mae:.4f} m/s, strength_rmse={eval_result.strength_rmse:.4f} m/s")
    print("=" * 60 + "\n")


async def run_eval_mode(
    region: str,
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
) -> None:
    """Evaluate on holdout data, print MAE/RMSE to console (requires trained model)."""

    bounding_box = get_bounding_box(region)
    repo_models = RepositoryModels.get_instance()
    model_data = repo_models.get_model(map_builder.name, **map_builder.get_model_cache_params())
    if model_data is None:
        raise FileNotFoundError(f"Model not found in RepositoryModels for {map_builder.name}. Run train mode first.")

    repo_simple_climb = RepositorySimpleClimb.get_instance()
    climbs = await repo_simple_climb.get_all_in_bounding_box_by_ground(bounding_box, verbose=True)
    train_climbs, holdout_climbs = dataset_builder.split_climbs(climbs)
    holdout_df = _climbs_to_eval_df(holdout_climbs)
    eval_result = map_builder.evaluate(bounding_box, holdout_df)
    print("\n" + "=" * 60)
    print(f"Map evaluation (n_train={len(train_climbs)}, n_holdout={len(holdout_climbs)})")
    print("=" * 60)
    print(f"strength_mae={eval_result.strength_mae:.4f} m/s, strength_rmse={eval_result.strength_rmse:.4f} m/s")
    print("=" * 60 + "\n")


async def run_eval_show_mode(
    region: str,
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
) -> None:
    """Evaluate MapBuilderEstimateNet on holdout data, print scores and show map chart."""

    bounding_box = get_bounding_box(region)
    repo_models = RepositoryModels.get_instance()
    model_data = repo_models.get_model(map_builder.name, **map_builder.get_model_cache_params())
    if model_data is None:
        raise FileNotFoundError(f"Model not found in RepositoryModels for {map_builder.name}. Run train mode first.")

    repo_simple_climb = RepositorySimpleClimb.get_instance()
    climbs = await repo_simple_climb.get_all_in_bounding_box_by_ground(bounding_box, verbose=True)
    train_climbs, holdout_climbs = dataset_builder.split_climbs(climbs)
    holdout_df = _climbs_to_eval_df(holdout_climbs)
    maps = map_builder.build(bounding_box, ignore_cache=True)
    vma = maps["strength"]
    eval_result = map_builder.evaluate(bounding_box, holdout_df)
    print("\n" + "=" * 60)
    print(f"Map evaluation (n_train={len(train_climbs)}, n_holdout={len(holdout_climbs)})")
    print("=" * 60)
    print(f"strength_mae={eval_result.strength_mae:.4f} m/s, strength_rmse={eval_result.strength_rmse:.4f} m/s")
    print("=" * 60 + "\n")

    from paraai.map.show_climb_map import show_map_eval

    terrain = RepositoryTerrain.get_instance().get_elevation(bounding_box)
    elevation = terrain["elevation"].astype(np.float32)
    elevation = np.nan_to_num(elevation, nan=0.0)
    show_map_eval(
        vma,
        holdout_df,
        column_name="strength",
        title=f"MapBuilderEstimate: {region}",
        elevation=elevation,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN to predict MapBuilderConvolution strength from elevation patches.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["dataset", "train", "estimate", "eval", "train_eval", "eval_show", "show_patch"],
        default="estimate",
        help="dataset: build and save dataset; train: load dataset, train, save model; estimate: load model, produce map (default); eval: evaluate on holdout (requires trained model); train_eval: train then evaluate on test patches (no map); eval_show: evaluate and show map chart; show_patch: show two random patches from training set",
    )
    parser.add_argument("--region", type=str, default="sopot", help="Region (bassano, sopot, bansko, europe)")
    parser.add_argument(
        "--estimator-type",
        choices=["simple", "time"],
        default="simple",
        help="Map builder: simple (elevation only) or time (elevation + time_of_day, time_of_year)",
    )
    parser.add_argument("--patch-size-m", type=float, default=500.0, help="Patch size in meters")
    parser.add_argument("--image-size", type=int, default=64, help="Patch resolution")
    parser.add_argument("--grid-stride", type=int, default=16, help="Stride for sampling grid")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fraction for test set")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--dataset-type",
        choices=["climb", "centre"],
        default="climb",
        help="climb: one point per climb; centre: add zero-climb points between consecutive climbs in same tracklog",
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def create_dataset_builder(
    dataset_builder_type: Literal["climb", "centre"],
    *,
    patch_size_m: float = 500.0,
    image_size: int = 64,
    grid_stride: int = 16,
    test_frac: float = 0.2,
    split_seed: int = 42,
    builder_name: str = "MapEstimateDataset",
) -> DatasetBuilder:
    """Create dataset builder from type and common params."""
    return DatasetBuilder(
        dataset_builder_type=dataset_builder_type,
        patch_size_m=patch_size_m,
        image_size=image_size,
        grid_stride=grid_stride,
        test_frac=test_frac,
        split_seed=split_seed,
        builder_name=builder_name,
    )


def create_map_builder(
    map_builder_type: Literal["simple", "time"],
) -> MapBuilderEstimateSimple | MapBuilderEstimateTime:
    """Create map builder from type and common params."""
    if map_builder_type == "simple":
        return MapBuilderEstimateSimple()
    if map_builder_type == "time":
        return MapBuilderEstimateTime()
    raise ValueError(f"Unknown map builder type: {map_builder_type}")


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    setup()
    region = args.region

    map_builder = create_map_builder(args.estimator_type)
    dataset_builder = create_dataset_builder(
        args.dataset_type,
        patch_size_m=args.patch_size_m,
        image_size=args.image_size,
        grid_stride=args.grid_stride,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
        builder_name=map_builder.name,
    )
    if args.mode == "show_patch":
        asyncio.run(run_show_patch_mode(region, dataset_builder, map_builder))
    elif args.mode == "dataset":
        asyncio.run(run_dataset_mode(region, dataset_builder, map_builder))
    elif args.mode == "train":
        asyncio.run(run_train_mode(region, dataset_builder, map_builder))
    elif args.mode == "estimate":
        asyncio.run(run_estimate_mode(region, dataset_builder, map_builder))
    elif args.mode == "eval":
        asyncio.run(run_eval_mode(region, dataset_builder, map_builder))
    elif args.mode == "train_eval":
        asyncio.run(run_train_eval_mode(region, dataset_builder, map_builder))
    elif args.mode == "eval_show":
        asyncio.run(run_eval_show_mode(region, dataset_builder, map_builder))
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
