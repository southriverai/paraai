"""Train a CNN to estimate MapBuilderConvolution(200) strength map from elevation patches.

Modes:
  dataset    - Build (elevation patches, target maps) from regions and save to RepositoryDatasets.
  train      - Load dataset from RepositoryDatasets (by builder + regions + params), train the CNN, save model.
  estimate   - Load trained model, run inference on regions, produce strength map (no evaluation).
  eval       - Evaluate on holdout data, print MAE/RMSE to console (requires trained model).
  train_eval - Train model, then evaluate on holdout (inference on test patches only, no map building).
  eval_show  - Evaluate on holdout data, print scores and show map chart.
  show_patch - Show two random elevation patches from the training set.

Usage examples (dataset-type and estimator-type always first):

  # 1. Build dataset for region(s) (saves to RepositoryDatasets cache)
  python script/map/train_map_builder.py --dataset-type climb --estimator-type simple --mode dataset --regions sopot

  # 2. Build dataset from multiple regions (climbs from all regions combined)
  python script/map/train_map_builder.py --dataset-type climb --mode dataset --regions sopot,bassano

  # 3. Train model (loads dataset from RepositoryDatasets by builder + regions + params)
  python script/map/train_map_builder.py --dataset-type climb --estimator-type simple --mode train --regions sopot --epochs 10

  # 4. Estimate strength map (default mode, cached by MapBuilderBase)
  python script/map/train_map_builder.py --dataset-type climb --estimator-type simple --regions sopot

  # 5. Evaluate on holdout data (print scores only)
  python script/map/train_map_builder.py --dataset-type climb --estimator-type simple --mode eval --regions sopot

  # 6. Train then evaluate (no pre-trained model needed)
  python script/map/train_map_builder.py --dataset-type climb --estimator-type time --mode train_eval --regions sopot

  # 7. Evaluate and show map chart
  python script/map/train_map_builder.py --dataset-type climb --estimator-type simple --mode eval_show --regions sopot

  # 8. Show two random elevation patches from the training set
  python script/map/train_map_builder.py --dataset-type climb --estimator-type simple --mode show_patch --regions sopot

  # 9. Show all training curves from repository
  python script/map/train_map_builder.py --mode show_plots

  # 10. Build dataset with flat_seed (remove climbs in flatlands, add 0-climb from flatlands)
  python script/map/train_map_builder.py --dataset-type flat_seed --estimator-type time --mode dataset --regions sopot
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

# BATCH_SIZE = 1024  # TODO


BATCH_SIZE = 4096  # TODO


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
    regions: list[str],
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
) -> None:
    """Build dataset from multiple regions (climbs fetched per bbox, no union)."""
    bounding_boxes = [get_bounding_box(r) for r in regions]
    data = await dataset_builder.build_dataset(
        bounding_boxes,
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
    regions: list[str],
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
) -> str:
    """Load dataset from RepositoryDatasets (or build from regions if missing), train model, save model."""
    repo_models = RepositoryModels.get_instance()
    bounding_boxes = [get_bounding_box(r) for r in regions]
    data = await dataset_builder.build_dataset(
        bounding_boxes,
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
        ground_alt_test = data["ground_alt_test"]
        train_td = torch.tensor(time_day_train, dtype=torch.float32, device=device)
        train_ty = torch.tensor(time_year_train, dtype=torch.float32, device=device)
        test_td = torch.tensor(time_day_test, dtype=torch.float32, device=device)
        test_ty = torch.tensor(time_year_test, dtype=torch.float32, device=device)
        train_ga = torch.tensor(ground_alt_train, dtype=torch.float32, device=device)
        test_ga = torch.tensor(ground_alt_test, dtype=torch.float32, device=device)

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
                pred = model(batch_inp, batch_td, batch_ty, batch_ga)
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
                    pred = model(batch_inp, batch_td, batch_ty, batch_ga)
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
        train_log = TrainLog(region=",".join(regions), builder_name=map_builder.name, epochs=training_metrics)
        log_path = repo_train_logs.save_train_log(map_builder.name, model_id, train_log)

        eta_sec = (elapsed / (epoch + 1)) * (map_builder.epochs - epoch - 1) if epoch < map_builder.epochs - 1 else 0
        eta_str = f", ETA {eta_sec:.0f}s" if eta_sec > 0 else ""
        logger.info(
            "Epoch %d/%d: train_loss=%.4f test_loss=%.4f (%.1fs/epoch)%s",
            epoch + 1,
            map_builder.epochs,
            train_loss,
            test_loss,
            epoch_duration,
            eta_str,
        )

        if interrupted:
            logger.info("Training interrupted by user.")
            if best_state is not None:
                model.load_state_dict(best_state)
            train_log = TrainLog(region=",".join(regions), builder_name=map_builder.name, epochs=training_metrics)
            log_path = repo_train_logs.save_train_log(map_builder.name, model_id, train_log)
            repo_models.save_model(
                map_builder.name,
                model_id,
                in_channels=in_c,
                out_channels=out_c,
                image_size=meta["image_size"],
                patch_size_m=meta["patch_size_m"],
                state_dict=model.state_dict(),
                strength_lo=meta["strength_lo"],
                strength_hi=meta["strength_hi"],
            )
            logger.info("Saved best model so far to RepositoryModels")
            _plot_all_training_results(log_path)
            return model_id

    if best_state is not None:
        model.load_state_dict(best_state)

    train_log = TrainLog(region=",".join(regions), builder_name=map_builder.name, epochs=training_metrics)
    log_path = repo_train_logs.save_train_log(map_builder.name, model_id, train_log)

    repo_models.save_model(
        map_builder.name,
        model_id,
        in_channels=in_c,
        out_channels=out_c,
        image_size=meta["image_size"],
        patch_size_m=meta["patch_size_m"],
        state_dict=model.state_dict(),
        strength_lo=meta["strength_lo"],
        strength_hi=meta["strength_hi"],
    )
    logger.info("Saved model to RepositoryModels")
    _plot_all_training_results(log_path)
    return model_id


async def run_show_patch_mode(
    regions: list[str],
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
) -> None:
    """Show elevation map with patch locations, and two random patches in separate charts. Uses first region."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    bounding_boxes = [get_bounding_box(r) for r in regions]
    bounding_box = bounding_boxes[0]
    data = await dataset_builder.build_dataset(
        bounding_boxes,
    )

    input_maps_train = data["input_maps_train"]
    target_maps_train = data["target_maps_train"]
    lats_train = data["lats_train"]
    lons_train = data["lons_train"]
    meta = data["metadata"]
    patch_size_m = meta["patch_size_m"]

    n = len(input_maps_train)
    in_first_region = [i for i in range(n) if bounding_box.is_in(lats_train[i], lons_train[i])]
    if len(in_first_region) < 2:
        raise ValueError(
            f"Need at least 2 training patches in first region for show_patch, got {len(in_first_region)}"
        )
    rng = np.random.default_rng(dataset_builder.split_seed)
    indices = rng.choice(in_first_region, size=2, replace=False)

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
    regions: list[str],
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
    *,
    grid_stride: int = 16,
) -> None:
    """Load trained model from RepositoryModels, run inference per region, cache strength map."""

    bounding_boxes = [get_bounding_box(r) for r in regions]
    repo_models = RepositoryModels.get_instance()
    dataset_id = dataset_builder.get_dataset_id(bounding_boxes)
    model_id = map_builder.get_model_id(dataset_id)
    model_data = repo_models.get_model(map_builder.name, model_id)
    if model_data is None:
        raise FileNotFoundError(f"Model not found in RepositoryModels for {map_builder.name}. Run train mode first.")

    for bbox in bounding_boxes:
        map_builder.build(bbox, ignore_cache=True, model_id=model_id, grid_stride=grid_stride)
    logger.info("Built and cached strength maps for %d regions", len(bounding_boxes))


async def run_train_eval_mode(
    regions: list[str],
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
) -> None:
    """Train model, then evaluate on holdout. No map building - inference on test patches only."""
    model_id = await run_train_mode(regions, dataset_builder, map_builder)
    bounding_boxes = [get_bounding_box(r) for r in regions]
    repo_simple_climb = RepositorySimpleClimb.get_instance()
    all_climbs: list[SimpleClimb] = []
    seen: set[str] = set()
    for bbox in bounding_boxes:
        climbs = await repo_simple_climb.get_all_in_bounding_box_by_ground(bbox, verbose=True)
        for c in climbs:
            if c.simple_climb_id not in seen:
                seen.add(c.simple_climb_id)
                all_climbs.append(c)
    train_climbs, holdout_climbs = dataset_builder.split_climbs(all_climbs)
    holdout_df = _climbs_to_eval_df(holdout_climbs)
    eval_result = map_builder.evaluate_multi_bbox(
        bounding_boxes,
        evaluate_df=holdout_df,
        model_id=model_id,
    )
    print("\n" + "=" * 60)
    print(f"Map evaluation (n_train={len(train_climbs)}, n_holdout={len(holdout_climbs)})")
    print("=" * 60)
    print(f"strength_mae={eval_result.strength_mae:.4f} m/s, strength_rmse={eval_result.strength_rmse:.4f} m/s")
    print("=" * 60 + "\n")


async def run_eval_mode(
    regions: list[str],
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
) -> None:
    """Evaluate on holdout data, print MAE/RMSE to console (requires trained model)."""

    bounding_boxes = [get_bounding_box(r) for r in regions]
    dataset_id = dataset_builder.get_dataset_id(bounding_boxes)
    model_id = map_builder.get_model_id(dataset_id)
    repo_models = RepositoryModels.get_instance()
    model_data = repo_models.get_model(map_builder.name, model_id)
    if model_data is None:
        raise FileNotFoundError(f"Model not found in RepositoryModels for {map_builder.name}. Run train mode first.")

    repo_simple_climb = RepositorySimpleClimb.get_instance()
    all_climbs: list[SimpleClimb] = []
    seen: set[str] = set()
    for bbox in bounding_boxes:
        climbs = await repo_simple_climb.get_all_in_bounding_box_by_ground(bbox, verbose=True)
        for c in climbs:
            if c.simple_climb_id not in seen:
                seen.add(c.simple_climb_id)
                all_climbs.append(c)
    train_climbs, holdout_climbs = dataset_builder.split_climbs(all_climbs)
    holdout_df = _climbs_to_eval_df(holdout_climbs)
    eval_result = map_builder.evaluate_multi_bbox(bounding_boxes, holdout_df, model_id=model_id)
    print("\n" + "=" * 60)
    print(f"Map evaluation (n_train={len(train_climbs)}, n_holdout={len(holdout_climbs)})")
    print("=" * 60)
    print(f"strength_mae={eval_result.strength_mae:.4f} m/s, strength_rmse={eval_result.strength_rmse:.4f} m/s")
    print("=" * 60 + "\n")


async def run_eval_show_mode(
    regions: list[str],
    dataset_builder: DatasetBuilder,
    map_builder: MapBuilderEstimateBase,
    *,
    grid_stride: int = 16,
) -> None:
    """Evaluate MapBuilderEstimateNet on holdout data, print scores and show map chart. Shows first region."""

    bounding_boxes = [get_bounding_box(r) for r in regions]
    dataset_id = dataset_builder.get_dataset_id(bounding_boxes)
    model_id = map_builder.get_model_id(dataset_id)
    repo_models = RepositoryModels.get_instance()
    model_data = repo_models.get_model(map_builder.name, model_id)
    if model_data is None:
        raise FileNotFoundError(f"Model not found in RepositoryModels for {map_builder.name}. Run train mode first.")

    repo_simple_climb = RepositorySimpleClimb.get_instance()
    all_climbs: list[SimpleClimb] = []
    seen: set[str] = set()
    for bbox in bounding_boxes:
        climbs = await repo_simple_climb.get_all_in_bounding_box_by_ground(bbox, verbose=True)
        for c in climbs:
            if c.simple_climb_id not in seen:
                seen.add(c.simple_climb_id)
                all_climbs.append(c)
    train_climbs, holdout_climbs = dataset_builder.split_climbs(all_climbs)
    holdout_df = _climbs_to_eval_df(holdout_climbs)
    eval_result = map_builder.evaluate_multi_bbox(bounding_boxes, holdout_df, model_id=model_id)
    bbox_display = bounding_boxes[0]
    maps = map_builder.build(bbox_display, ignore_cache=True, model_id=model_id, grid_stride=grid_stride)
    vma = maps["strength"]
    print("\n" + "=" * 60)
    print(f"Map evaluation (n_train={len(train_climbs)}, n_holdout={len(holdout_climbs)})")
    print("=" * 60)
    print(f"strength_mae={eval_result.strength_mae:.4f} m/s, strength_rmse={eval_result.strength_rmse:.4f} m/s")
    print("=" * 60 + "\n")

    from paraai.map.show_climb_map import show_map_eval

    save_dir = Path("data", "plots")
    save_path = save_dir / f"eval_{'_'.join(regions)}.png"

    terrain = RepositoryTerrain.get_instance().get_elevation(bbox_display)
    elevation = terrain["elevation"].astype(np.float32)
    elevation = np.nan_to_num(elevation, nan=0.0)
    holdout_in_bbox = holdout_df[
        holdout_df.apply(lambda r: bbox_display.is_in(r["lat"], r["lon"]), axis=1)
    ]
    show_map_eval(
        vma,
        holdout_in_bbox,
        column_name="strength",
        title=f"MapBuilderEstimate: {','.join(regions)}",
        elevation=elevation,
        save_path=save_path,
    )

    log_path = RepositoryTrainLogs.get_instance().get_log_path(map_builder.name, model_id)
    _plot_all_training_results(log_path if log_path.exists() else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN to predict MapBuilderConvolution strength from elevation patches.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["dataset", "train", "estimate", "eval", "train_eval", "eval_show", "show_patch", "show_plots"],
        default="estimate",
        help="dataset: build dataset; train: train model; estimate: produce map (default); eval: evaluate on holdout; train_eval: train then evaluate; eval_show: evaluate, show map and training curves; show_patch: show two patches; show_plots: show all training curves",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="sopot",
        help="Comma-separated regions (e.g. sopot,bassano). Dataset uses climbs from union of all bounding boxes.",
    )
    parser.add_argument(
        "--estimator-type",
        choices=["simple", "time"],
        default="simple",
        help="Map builder: simple (elevation only) or time (elevation + time_of_day, time_of_year)",
    )
    parser.add_argument("--patch-size-m", type=float, default=500.0, help="Patch size in meters")
    parser.add_argument("--image-size", type=int, default=64, help="Patch resolution")
    parser.add_argument(
        "--grid-stride",
        type=int,
        default=16,
        help="Pixel stride for CNN strength map inference (not part of dataset/model cache keys)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fraction for test set")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=300_000,
        help="Max instances (train+test); excess randomly sampled down (default 300000)",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["climb", "centre", "flat_seed"],
        default="climb",
        help="climb: one point per climb; centre: add zero-climb between climbs; flat_seed: remove climbs in flatlands, seed 0-climb from flatlands",
    )
    parser.add_argument("--flatland-radius-m", type=float, default=200.0, help="Radius for flatland planarity (flat_seed mode)")
    parser.add_argument("--flat-seed-planarity-threshold", type=float, default=0.5, help="Planarity > threshold = flatland (flat_seed mode)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    # Parse regions from comma-separated string
    args.regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    if not args.regions:
        parser.error("--regions must specify at least one region")
    return args


def create_dataset_builder(
    dataset_builder_type: Literal["climb", "centre", "flat_seed"],
    *,
    patch_size_m: float = 500.0,
    image_size: int = 64,
    test_frac: float = 0.2,
    split_seed: int = 42,
    dataset_limit: int = 300_000,
    builder_name: str = "MapEstimateDataset",
    flatland_radius_m: float = 200.0,
    flat_seed_planarity_threshold: float = 0.5,
) -> DatasetBuilder:
    """Create dataset builder from type and common params."""
    return DatasetBuilder(
        dataset_builder_type=dataset_builder_type,
        patch_size_m=patch_size_m,
        image_size=image_size,
        test_frac=test_frac,
        split_seed=split_seed,
        dataset_limit=dataset_limit,
        builder_name=builder_name,
        flatland_radius_m=flatland_radius_m,
        flat_seed_planarity_threshold=flat_seed_planarity_threshold,
    )


def create_map_builder(
    map_builder_type: Literal["simple", "time"],
    *,
    epochs: int = 5,
    lr: float = 1e-3,
) -> MapBuilderEstimateSimple | MapBuilderEstimateTime:
    """Create map builder from type and common params."""
    if map_builder_type == "simple":
        return MapBuilderEstimateSimple(epochs=epochs, lr=lr)
    if map_builder_type == "time":
        return MapBuilderEstimateTime(epochs=epochs, lr=lr)
    raise ValueError(f"Unknown map builder type: {map_builder_type}")


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    setup()
    regions = args.regions

    map_builder = create_map_builder(
        args.estimator_type,
        epochs=args.epochs,
        lr=args.lr,
    )
    dataset_builder = create_dataset_builder(
        args.dataset_type,
        patch_size_m=args.patch_size_m,
        image_size=args.image_size,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
        dataset_limit=args.dataset_limit,
        builder_name=map_builder.name,
        flatland_radius_m=args.flatland_radius_m,
        flat_seed_planarity_threshold=args.flat_seed_planarity_threshold,
    )
    if args.mode == "show_patch":
        asyncio.run(run_show_patch_mode(regions, dataset_builder, map_builder))
    elif args.mode == "dataset":
        asyncio.run(run_dataset_mode(regions, dataset_builder, map_builder))
    elif args.mode == "train":
        asyncio.run(run_train_mode(regions, dataset_builder, map_builder))
    elif args.mode == "estimate":
        asyncio.run(run_estimate_mode(regions, dataset_builder, map_builder, grid_stride=args.grid_stride))
    elif args.mode == "eval":
        asyncio.run(run_eval_mode(regions, dataset_builder, map_builder))
    elif args.mode == "train_eval":
        asyncio.run(run_train_eval_mode(regions, dataset_builder, map_builder))
    elif args.mode == "eval_show":
        asyncio.run(run_eval_show_mode(regions, dataset_builder, map_builder, grid_stride=args.grid_stride))
    elif args.mode == "show_plots":
        _plot_all_training_results(None)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
