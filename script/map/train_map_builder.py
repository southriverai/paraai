"""Train a CNN to estimate MapBuilderConvolution(200) strength map from elevation patches.

Three modes:
  dataset  - Build (elevation patches, target maps) from region and save to disk.
  train    - Load dataset from disk, train the CNN, save model.
  estimate - Load trained model, run inference on region, produce strength map.

Usage examples:

  # 1. Build dataset for region (requires climb pixels and terrain)
  python script/map/train_map_builder.py dataset --region sopot --dataset-path data/datasets/sopot.pt

  # 2. Train model on saved dataset
  python script/map/train_map_builder.py train --dataset-path data/datasets/sopot.pt --model-path data/models/map_model.pt --epochs 10

  # 3. Estimate strength map for region using trained model
  python script/map/train_map_builder.py estimate --region sopot --model-path data/models/map_model.pt --output data/maps/strength_sopot.tif

  # Or run all-in-one: build dataset on the fly, train, visualize (omit --dataset-path)
  python script/map/train_map_builder.py train --region sopot --epochs 5 --visualize
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
from torch.utils.data import Dataset

from paraai.map.map_builder_estimate_net import MapBuilderEstimateNet, _extract_elevation_patch
from paraai.map.map_estimate_net import MapEstimateNet
from paraai.model.boundingbox import BoundingBox
from paraai.repository.repository_simple_climb_pixel import RepositorySimpleClimbPixel
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.setup import setup
from paraai.tool_spacetime import REGION_BOUNDS

logger = logging.getLogger(__name__)


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


def run_dataset_mode(args: argparse.Namespace) -> None:
    """Build dataset from region and save to disk."""
    bounding_box = _get_bounding_box(args.region)
    climb_df = _load_climb_dataframe(bounding_box)

    builder = MapBuilderEstimateNet(
        kernel_size_m=args.kernel_size_m,
        patch_size_m=args.patch_size_m,
        image_size=args.image_size,
        grid_stride=args.grid_stride,
    )
    data = builder.build_dataset(bounding_box, climb_df)

    # Serialize for disk (bounding_box already in metadata as dict)
    save_data = {
        "input_maps": data["input_maps"],
        "target_maps": data["target_maps"],
        "metadata": data["metadata"],
    }
    path = Path(args.dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, path)
    logger.info("Saved dataset (%s patches) to %s", len(data["input_maps"]), path)


def run_train_mode(args: argparse.Namespace) -> None:
    """Load dataset from disk (or build from region if missing), train model, save model."""
    path = Path(args.dataset_path)
    if path.exists():
        data = torch.load(path, weights_only=False)
    else:
        # Build dataset on the fly from region
        logger.info("Dataset not found at %s, building from region %s", path, args.region)
        bounding_box = _get_bounding_box(args.region)
        climb_df = _load_climb_dataframe(bounding_box)
        builder = MapBuilderEstimateNet(
            kernel_size_m=args.kernel_size_m,
            patch_size_m=args.patch_size_m,
            image_size=args.image_size,
            grid_stride=args.grid_stride,
        )
        data = builder.build_dataset(bounding_box, climb_df)
        data = {
            "input_maps": data["input_maps"],
            "target_maps": data["target_maps"],
            "metadata": data["metadata"],
        }

    input_maps = data["input_maps"]
    target_maps = data["target_maps"]
    meta = data["metadata"]

    in_c = input_maps[0].shape[0]
    out_c = target_maps[0].shape[0]
    image_size = meta["image_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    dataset = MapDataset(input_maps, target_maps=target_maps)
    n_total = len(dataset)
    n_test = max(1, int(n_total * args.test_frac))
    n_train = n_total - n_test
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_test], generator=torch.Generator().manual_seed(args.split_seed)
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MapEstimateNet(in_channels=in_c, out_channels=out_c, size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_test_loss = float("inf")
    best_state: dict | None = None

    from tqdm import tqdm

    pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for _ in pbar:
        model.train()
        train_loss = 0.0
        n_b = 0
        for inp, tgt in train_loader:
            inp_dev, tgt_dev = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred = model(inp_dev)
            if pred.shape[2:] != tgt_dev.shape[2:]:
                pred = nn.functional.interpolate(pred, size=tgt_dev.shape[2:], mode="bilinear")
            loss = criterion(pred, tgt_dev)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_b += 1
        train_loss /= n_b if n_b else 1

        model.eval()
        test_loss = 0.0
        n_b = 0
        with torch.no_grad():
            for inp, tgt in test_loader:
                inp_dev, tgt_dev = inp.to(device), tgt.to(device)
                pred = model(inp_dev)
                if pred.shape[2:] != tgt_dev.shape[2:]:
                    pred = nn.functional.interpolate(pred, size=tgt_dev.shape[2:], mode="bilinear")
                test_loss += criterion(pred, tgt_dev).item()
                n_b += 1
        test_loss /= n_b if n_b else 1

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        pbar.set_postfix(train_loss=f"{train_loss:.4f}", test_loss=f"{test_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_channels": in_c,
            "out_channels": out_c,
            "image_size": image_size,
        },
        model_path,
    )
    logger.info("Saved model to %s", model_path)

    if args.visualize:
        _visualize_after_train(
            model, data, meta, device, list(test_ds.indices), args.pred_stride
        )


def run_estimate_mode(args: argparse.Namespace) -> None:
    """Load trained model, run inference on region, save strength map."""
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train mode first.")

    checkpoint = torch.load(model_path, weights_only=False)
    state_dict = checkpoint["state_dict"]
    in_c = checkpoint["in_channels"]
    out_c = checkpoint["out_channels"]
    image_size = checkpoint["image_size"]

    model = MapEstimateNet(in_channels=in_c, out_channels=out_c, size=image_size)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    bounding_box = _get_bounding_box(args.region)
    patch_size_m = args.patch_size_m
    grid_stride = args.grid_stride

    repo_terrain = RepositoryTerrain.get_instance()
    terrain = repo_terrain.get_elevation(bounding_box)
    elevation = terrain["elevation"]
    transform = terrain["transform"]
    h, w = elevation.shape

    n_rows = (h + grid_stride - 1) // grid_stride
    n_cols = (w + grid_stride - 1) // grid_stride
    pred_sparse = np.zeros((n_rows, n_cols), dtype=np.float32)

    with torch.no_grad():
        for ri, r in enumerate(range(0, h, grid_stride)):
            for ci, c in enumerate(range(0, w, grid_stride)):
                lon, lat = rasterio.transform.xy(transform, r, c)
                patch = _extract_elevation_patch(
                    elevation, transform, float(lat), float(lon), patch_size_m, image_size
                )
                patch_batch = patch.unsqueeze(0).to(device)
                pred = model(patch_batch)
                pred_sparse[ri, ci] = pred[0, 0, image_size // 2, image_size // 2].item()

    pred_arr = pred_sparse
    pred_grid = (
        torch.from_numpy(pred_arr)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    pred_grid = nn.functional.interpolate(
        pred_grid, size=(h, w), mode="bilinear", align_corners=False
    ).squeeze().numpy().astype(np.float32)
    pred_grid = np.clip(pred_grid, 0, 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": w,
        "height": h,
        "count": 1,
        "dtype": pred_grid.dtype,
        "transform": transform,
        "crs": rasterio.CRS.from_epsg(4326),
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred_grid, 1)
    logger.info("Saved strength map to %s", output_path)


def _get_bounding_box(region: str) -> BoundingBox:
    bbox = REGION_BOUNDS.get(region.lower())
    if bbox is None:
        raise ValueError(f"Unknown region '{region}'. Available: {list(REGION_BOUNDS)}")
    return bbox


def _load_climb_dataframe(bounding_box: BoundingBox) -> pd.DataFrame:
    repo = RepositorySimpleClimbPixel.get_instance()
    pixels = repo.get_all_in_bounding_box(
        bounding_box.lat_min, bounding_box.lat_max, bounding_box.lon_min, bounding_box.lon_max
    )
    if len(pixels) < 1:
        raise ValueError("No SimpleClimbPixels in region")
    return pd.DataFrame(
        [(p.lat, p.lon, p.climb_count, p.mean_climb_strength_m_s) for p in pixels],
        columns=["lat", "lon", "count", "strength"],
    )


def _visualize_after_train(
    model: MapEstimateNet,
    data: dict,
    meta: dict,
    device: torch.device,
    test_indices: list[int],
    pred_stride: int,
) -> None:
    """Visualize after training (requires elevation - re-fetch from terrain)."""
    bounding_box = BoundingBox(**meta["bounding_box"])
    repo_terrain = RepositoryTerrain.get_instance()
    terrain = repo_terrain.get_elevation(bounding_box)
    elevation = terrain["elevation"]
    transform = terrain["transform"]
    image_size = meta["image_size"]
    patch_size_m = meta["patch_size_m"]
    strength_lo = meta["strength_lo"]
    strength_hi = meta["strength_hi"]

    # Build true_grid from target_maps (average at each cell - we have one value per patch)
    # For viz we need a full grid. Rebuild from convolution target.
    from paraai.map.map_builder_convolution import MapBuilderConvolution

    climb_df = _load_climb_dataframe(bounding_box)
    conv_builder = MapBuilderConvolution(kernel_size_m=meta["kernel_size_m"])
    conv_maps = conv_builder.build(bounding_box, climb_df)
    strength_arr = conv_maps["strength"].array
    true_grid = np.clip(
        (strength_arr.astype(np.float64) - strength_lo) / (strength_hi - strength_lo + 1e-10),
        0,
        1,
    ).astype(np.float32)

    h, w = elevation.shape
    extent = [bounding_box.lon_min, bounding_box.lon_max, bounding_box.lat_min, bounding_box.lat_max]

    model.eval()
    stride = pred_stride
    n_rows = (h + stride - 1) // stride
    n_cols = (w + stride - 1) // stride
    pred_sparse = np.zeros((n_rows, n_cols), dtype=np.float32)
    with torch.no_grad():
        for ri, r in enumerate(range(0, h, stride)):
            for ci, c in enumerate(range(0, w, stride)):
                lon, lat = rasterio.transform.xy(transform, r, c)
                patch = _extract_elevation_patch(
                    elevation, transform, float(lat), float(lon), patch_size_m, image_size
                )
                pred = model(patch.unsqueeze(0).to(device))
                pred_sparse[ri, ci] = pred[0, 0, image_size // 2, image_size // 2].item()

    pred_grid = (
        nn.functional.interpolate(
            torch.from_numpy(pred_sparse).unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
        .astype(np.float32)
    )
    pred_grid = np.clip(pred_grid, 0, 1)

    input_maps = data["input_maps"]
    target_maps = data["target_maps"]

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 2])
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_elev = fig.add_subplot(gs[0, 1])
    ax_true = fig.add_subplot(gs[1, 0])
    ax_pred = fig.add_subplot(gs[1, 1])

    preds_test = []
    targets_test = []
    for idx in test_indices:
        inp = input_maps[idx].unsqueeze(0).to(device)
        tgt = target_maps[idx]
        pred = model(inp)
        preds_test.append(pred[0, 0, image_size // 2, image_size // 2].item())
        targets_test.append(tgt[0, image_size // 2, image_size // 2].item())

    bins = np.linspace(0, 1, 21)
    if preds_test:
        ax_hist.hist(preds_test, bins=bins, alpha=0.7, color="blue", label="Predicted", density=True)
    if targets_test:
        ax_hist.hist(targets_test, bins=bins, alpha=0.5, color="green", label="Target", density=True)
    ax_hist.set_xlabel("Strength (normalized)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Test set: predicted vs target")
    ax_hist.legend()
    ax_hist.set_xlim(0, 1)

    elev_display = np.clip(
        (elevation - np.nanpercentile(elevation, 2))
        / (np.nanpercentile(elevation, 98) - np.nanpercentile(elevation, 2) + 1e-10),
        0,
        1,
    )
    ax_elev.imshow(elev_display, extent=extent, origin="upper", cmap="terrain")
    ax_elev.set_title("Heightmap")
    ax_elev.set_xlabel("Longitude")
    ax_elev.set_ylabel("Latitude")
    ax_elev.set_aspect("equal")

    ax_true.imshow(true_grid, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1)
    ax_true.set_title("True: MapBuilderConvolution")
    ax_true.set_xlabel("Longitude")
    ax_true.set_ylabel("Latitude")
    ax_true.set_aspect("equal")

    ax_pred.imshow(pred_grid, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1)
    ax_pred.set_title("Predicted strength map")
    ax_pred.set_xlabel("Longitude")
    ax_pred.set_ylabel("Latitude")
    ax_pred.set_aspect("equal")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN to predict MapBuilderConvolution strength from elevation patches.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["dataset", "train", "estimate"],
        help="dataset: build and save dataset; train: load dataset, train, save model; estimate: load model, produce map",
    )
    parser.add_argument("--region", type=str, default="sopot", help="Region (bassano, sopot, bansko, europe)")
    parser.add_argument("--dataset-path", type=str, default="data/datasets/map_dataset.pt", help="Path for dataset file")
    parser.add_argument("--model-path", type=str, default="data/models/map_model.pt", help="Path for model file")
    parser.add_argument("--output", type=str, default="data/maps/strength.tif", help="Output GeoTIFF for estimate mode")
    parser.add_argument("--patch-size-m", type=float, default=500.0, help="Patch size in meters")
    parser.add_argument("--image-size", type=int, default=64, help="Patch resolution")
    parser.add_argument("--grid-stride", type=int, default=16, help="Stride for sampling grid")
    parser.add_argument("--kernel-size-m", type=float, default=200.0, help="MapBuilderConvolution kernel")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fraction for test set")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--pred-stride", type=int, default=16, help="Stride for prediction grid in viz")
    parser.add_argument("--visualize", action="store_true", help="Show plot after train (requires dataset from same region)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.mode == "dataset":
        run_dataset_mode(args)
    elif args.mode == "train":
        run_train_mode(args)
    else:
        run_estimate_mode(args)


if __name__ == "__main__":
    setup()
    main()
