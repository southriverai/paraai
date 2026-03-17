"""Train a CNN to estimate MapBuilderConvolution(200) strength map from elevation patches.

Modes:
  dataset    - Build (elevation patches, target maps) from region and save to RepositoryDatasets.
  train      - Load dataset from RepositoryDatasets (by builder + region + params), train the CNN, save model.
  estimate   - Load trained model, run inference on region, produce strength map (no evaluation).
  eval       - Evaluate on holdout data, print MAE/RMSE to console (requires trained model).
  train_eval - Train model, then evaluate on holdout (inference on test patches only, no map building).
  eval_show  - Evaluate on holdout data, print scores and show map chart.
  show_patch - Show two random elevation patches from the training set.

Usage examples:

  # 1. Build dataset for region (saves to RepositoryDatasets cache)
  python script/map/train_map_builder.py --mode dataset --region sopot

  # 2. Train model (loads dataset from RepositoryDatasets by builder + region + params)
  python script/map/train_map_builder.py --mode train --region sopot --epochs 10

  # 3. Estimate strength map (default mode, cached by MapBuilderBase)
  python script/map/train_map_builder.py --region sopot

  # 4. Evaluate on holdout data (print scores only)
  python script/map/train_map_builder.py --mode eval --region sopot

  # 5. Train then evaluate (no pre-trained model needed)
  python script/map/train_map_builder.py --mode train_eval --region sopot --builder-type time

  # 6. Evaluate and show map chart
  python script/map/train_map_builder.py --mode eval_show --region sopot

  # 7. Show two random elevation patches from the training set
  python script/map/train_map_builder.py --mode show_patch --region sopot
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import Dataset

from paraai.map.map_builder_estimate_simple import MapBuilderEstimateSimple
from paraai.map.map_builder_estimate_time import MapBuilderEstimateTime
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.setup import setup
from paraai.tool_spacetime import get_bounding_box
from paraai.tools_datasets import split_dataframe

logger = logging.getLogger(__name__)

BATCH_SIZE = 64


class MapDataset(Dataset):
    """Dataset of (input_map, target_map) pairs as tensors."""

    def __init__(
        self,
        input_maps: list[torch.Tensor],
        target_maps: list[torch.Tensor] | None = None,
        example_map: torch.Tensor | None = None,
    ) -> None:
        self.input_maps = input_maps
        if target_maps is not None:
            self.target_maps = target_maps
        elif example_map is not None:
            self.target_maps = [example_map] * len(input_maps)
        else:
            raise ValueError("Provide target_maps or example_map")
        assert len(self.input_maps) == len(self.target_maps)

    def __len__(self) -> int:
        return len(self.input_maps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_maps[idx], self.target_maps[idx]


class MapDatasetTime(Dataset):
    """Dataset for time model: (elevation_patch, time_of_day, time_of_year, target_scalar)."""

    def __init__(
        self,
        input_maps: list[torch.Tensor],
        time_of_day: list[float],
        time_of_year: list[float],
        target_maps: list[torch.Tensor],
    ) -> None:
        self.input_maps = input_maps
        self.time_of_day = time_of_day
        self.time_of_year = time_of_year
        self.target_maps = target_maps
        assert len(self.input_maps) == len(self.target_maps) == len(time_of_day) == len(time_of_year)

    def __len__(self) -> int:
        return len(self.input_maps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_maps[idx],
            torch.tensor(self.time_of_day[idx], dtype=torch.float32),
            torch.tensor(self.time_of_year[idx], dtype=torch.float32),
            self.target_maps[idx],
        )


def run_dataset_mode(
    region: str,
    test_frac: float,
    split_seed: int,
    builder: MapBuilderEstimateSimple | MapBuilderEstimateTime,
) -> None:
    """Build dataset from region and save to RepositoryDatasets cache."""
    bounding_box = get_bounding_box(region)
    climb_df = asyncio.run(RepositorySimpleClimb.get_instance().get_climb_dataframe(bounding_box))
    train_df, test_df = split_dataframe(climb_df, test_frac, split_seed)
    data = builder.get_or_build_dataset(
        bounding_box,
        train_df,
        test_df,
        ignore_cache=True,
    )

    n_train = len(data["input_maps_train"])
    n_test = len(data["input_maps_test"])
    logger.info("Dataset (%s train, %s test patches) built and cached in RepositoryDatasets", n_train, n_test)
    print("\nDataset head (train):")
    print(train_df.head(5))


def run_train_mode(
    region: str,
    test_frac: float,
    split_seed: int,
    lr: float,
    epochs: int,
    builder: MapBuilderEstimateSimple | MapBuilderEstimateTime,
) -> None:
    """Load dataset from RepositoryDatasets (or build from region if missing), train model, save model."""
    bounding_box = get_bounding_box(region)
    climb_df = asyncio.run(RepositorySimpleClimb.get_instance().get_climb_dataframe(bounding_box))
    train_df, test_df = split_dataframe(climb_df, test_frac, split_seed)
    kwargs: dict = {}
    if isinstance(builder, MapBuilderEstimateTime):
        kwargs["test_frac"] = test_frac
        kwargs["split_seed"] = split_seed
    data = builder.get_or_build_dataset(
        bounding_box,
        train_df,
        test_df,
        **kwargs,
    )

    input_maps_train = data["input_maps_train"]
    target_maps_train = data["target_maps_train"]
    input_maps_test = data["input_maps_test"]
    target_maps_test = data["target_maps_test"]
    meta = data["metadata"]

    print("\nDataset head (train):")
    print(train_df.head(5))

    in_c = input_maps_train[0].shape[0]
    out_c = target_maps_train[0].shape[0]
    image_size = meta["image_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    is_time_builder = isinstance(builder, MapBuilderEstimateTime)

    # Preload all data to GPU once (avoids per-batch transfer overhead)
    logger.info("Loading data to %s...", device)
    train_inp = torch.stack(input_maps_train).to(device)
    train_tgt = torch.stack(target_maps_train).to(device)
    test_inp = torch.stack(input_maps_test).to(device)
    test_tgt = torch.stack(target_maps_test).to(device)
    if is_time_builder:
        time_day_train = data["time_of_day_train"]
        time_year_train = data["time_of_year_train"]
        time_day_test = data["time_of_day_test"]
        time_year_test = data["time_of_year_test"]
        train_td = torch.tensor(time_day_train, dtype=torch.float32, device=device)
        train_ty = torch.tensor(time_year_train, dtype=torch.float32, device=device)
        test_td = torch.tensor(time_day_test, dtype=torch.float32, device=device)
        test_ty = torch.tensor(time_year_test, dtype=torch.float32, device=device)

    model = builder._get_model_class()(in_channels=in_c, out_channels=out_c, size=image_size).to(device)
    # print model architecture
    logger.info("Model class: %s", builder._get_model_class())
    logger.info("Model architecture: %s", model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    best_test_loss = float("inf")
    best_state: dict | None = None

    n_train = len(train_inp)
    n_test = len(test_inp)

    t_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = 0.0
        n_b = 0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            batch_inp = train_inp[idx]
            batch_tgt = train_tgt[idx]
            if is_time_builder:
                batch_td = train_td[idx]
                batch_ty = train_ty[idx]
                pred = model(batch_inp, batch_td, batch_ty)
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
                if is_time_builder:
                    batch_td = test_td[i : i + BATCH_SIZE]
                    batch_ty = test_ty[i : i + BATCH_SIZE]
                    pred = model(batch_inp, batch_td, batch_ty)
                else:
                    pred = model(batch_inp)
                    if pred.shape[2:] != batch_tgt.shape[2:]:
                        pred = nn.functional.interpolate(pred, size=batch_tgt.shape[2:], mode="bilinear")
                test_loss += criterion(pred, batch_tgt).item()
                n_b += 1
        test_loss /= n_b if n_b else 1

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.perf_counter() - t_start
        eta_sec = (elapsed / (epoch + 1)) * (epochs - epoch - 1) if epoch < epochs - 1 else 0
        eta_str = f", ETA {eta_sec:.0f}s" if eta_sec > 0 else ""
        logger.info(
            "Epoch %d/%d: train_loss=%.4f test_loss=%.4f (%.1fs)%s",
            epoch + 1,
            epochs,
            train_loss,
            test_loss,
            elapsed,
            eta_str,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    from paraai.repository.repository_models import RepositoryModels

    repo_models = RepositoryModels.get_instance()
    cache_params = {k: v for k, v in builder.get_model_cache_params().items() if k != "image_size"}
    repo_models.save_model(
        model.state_dict(),
        in_c,
        out_c,
        image_size,
        builder.name,
        strength_lo=meta["strength_lo"],
        strength_hi=meta["strength_hi"],
        **cache_params,
    )
    logger.info("Saved model to RepositoryModels")


def run_show_patch_mode(
    region: str,
    test_frac: float,
    split_seed: int,
    builder: MapBuilderEstimateSimple | MapBuilderEstimateTime,
) -> None:
    """Show elevation map with patch locations, and two random patches in separate charts."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    bounding_box = get_bounding_box(region)
    climb_df = asyncio.run(RepositorySimpleClimb.get_instance().get_climb_dataframe(bounding_box))
    train_df, test_df = split_dataframe(climb_df, test_frac, split_seed)
    data = builder.get_or_build_dataset(
        bounding_box,
        train_df,
        test_df,
        test_frac=test_frac,
        split_seed=split_seed,
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

    rng = np.random.default_rng(split_seed)
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


def run_estimate_mode(region: str, builder: MapBuilderEstimateSimple | MapBuilderEstimateTime) -> None:
    """Load trained model from RepositoryModels, run inference on region, cache strength map."""
    from paraai.repository.repository_models import RepositoryModels

    bounding_box = get_bounding_box(region)
    repo_models = RepositoryModels.get_instance()
    model_data = repo_models.get_model(builder.name, **builder.get_model_cache_params())
    if model_data is None:
        raise FileNotFoundError(f"Model not found in RepositoryModels for {builder.name}. Run train mode first.")

    climb_df = asyncio.run(RepositorySimpleClimb.get_instance().get_climb_dataframe(bounding_box))
    builder.build(bounding_box, climb_df, ignore_cache=True)
    logger.info("Built and cached strength map")


def run_train_eval_mode(
    region: str,
    test_frac: float,
    split_seed: int,
    lr: float,
    epochs: int,
    builder: MapBuilderEstimateSimple | MapBuilderEstimateTime,
) -> None:
    """Train model, then evaluate on holdout. No map building - inference on test patches only."""
    run_train_mode(region, test_frac, split_seed, lr, epochs, builder)
    bounding_box = get_bounding_box(region)
    climb_df = asyncio.run(RepositorySimpleClimb.get_instance().get_climb_dataframe(bounding_box))
    train_df, holdout_df = split_dataframe(climb_df, test_frac, split_seed)
    eval_result = builder.evaluate(
        bounding_box=bounding_box,
        evaluate_df=holdout_df,
    )
    print("\n" + "=" * 60)
    print(f"Map evaluation (n_train={len(train_df)}, n_holdout={len(holdout_df)})")
    print("=" * 60)
    print(f"strength_mae={eval_result.strength_mae:.4f} m/s, strength_rmse={eval_result.strength_rmse:.4f} m/s")
    print("=" * 60 + "\n")


def run_eval_mode(
    region: str,
    test_frac: float,
    split_seed: int,
    builder: MapBuilderEstimateSimple | MapBuilderEstimateTime,
) -> None:
    """Evaluate on holdout data, print MAE/RMSE to console (requires trained model)."""
    from paraai.repository.repository_models import RepositoryModels

    bounding_box = get_bounding_box(region)
    repo_models = RepositoryModels.get_instance()
    model_data = repo_models.get_model(builder.name, **builder.get_model_cache_params())
    if model_data is None:
        raise FileNotFoundError(f"Model not found in RepositoryModels for {builder.name}. Run train mode first.")

    climb_df = asyncio.run(RepositorySimpleClimb.get_instance().get_climb_dataframe(bounding_box))
    train_df, holdout_df = split_dataframe(climb_df, test_frac, split_seed)
    maps = builder.build(bounding_box, train_df, ignore_cache=True)
    vma = maps["strength"]

    eval_result = builder.evaluate(vma, holdout_df, column_name="strength")
    print("\n" + "=" * 60)
    print(f"Map evaluation (n_train={len(train_df)}, n_holdout={len(holdout_df)})")
    print("=" * 60)
    print(f"strength_mae={eval_result.strength_mae:.4f} m/s, strength_rmse={eval_result.strength_rmse:.4f} m/s")
    print("=" * 60 + "\n")


def run_eval_show_mode(
    region: str,
    test_frac: float,
    split_seed: int,
    builder: MapBuilderEstimateSimple | MapBuilderEstimateTime,
) -> None:
    """Evaluate MapBuilderEstimateNet on holdout data, print scores and show map chart."""
    from paraai.repository.repository_models import RepositoryModels

    bounding_box = get_bounding_box(region)
    repo_models = RepositoryModels.get_instance()
    model_data = repo_models.get_model(builder.name, **builder.get_model_cache_params())
    if model_data is None:
        raise FileNotFoundError(f"Model not found in RepositoryModels for {builder.name}. Run train mode first.")

    climb_df = asyncio.run(RepositorySimpleClimb.get_instance().get_climb_dataframe(bounding_box))
    train_df, holdout_df = split_dataframe(climb_df, test_frac, split_seed)
    maps = builder.build(bounding_box, train_df, ignore_cache=True)
    vma = maps["strength"]

    eval_result = builder.evaluate(bounding_box, holdout_df)
    print("\n" + "=" * 60)
    print(f"Map evaluation (n_train={len(train_df)}, n_holdout={len(holdout_df)})")
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
        "--builder-type",
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
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def create_builder(
    builder_type: str,
    patch_size_m: float,
    image_size: int,
    grid_stride: int,
):
    """Create map builder from type and common params."""
    if builder_type == "simple":
        return MapBuilderEstimateSimple(
            patch_size_m=patch_size_m,
            image_size=image_size,
            grid_stride=grid_stride,
        )
    if builder_type == "time":
        return MapBuilderEstimateTime(
            patch_size_m=patch_size_m,
            image_size=image_size,
            grid_stride=grid_stride,
        )
    raise ValueError(f"Unknown builder type: {builder_type}")


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    setup()
    region = args.region
    test_frac = args.test_frac
    split_seed = args.split_seed
    builder = create_builder(
        args.builder_type,
        args.patch_size_m,
        args.image_size,
        args.grid_stride,
    )
    if args.mode == "show_patch":
        run_show_patch_mode(region, test_frac, split_seed, builder)
    elif args.mode == "dataset":
        run_dataset_mode(region, test_frac, split_seed, builder)
    elif args.mode == "train":
        run_train_mode(region, test_frac, split_seed, args.lr, args.epochs, builder)
    elif args.mode == "estimate":
        run_estimate_mode(region, builder)
    elif args.mode == "eval":
        run_eval_mode(region, test_frac, split_seed, builder)
    elif args.mode == "train_eval":
        run_train_eval_mode(region, test_frac, split_seed, args.lr, args.epochs, builder)
    elif args.mode == "eval_show":
        run_eval_show_mode(region, test_frac, split_seed, builder)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
