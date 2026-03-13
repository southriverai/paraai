"""Train a CNN to estimate MapBuilderConvolution(200) strength map from elevation patches."""

from __future__ import annotations

import argparse
import copy
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from paraai.map.map_builder_convolution import MapBuilderConvolution
from paraai.repository.repository_simple_climb_pixel import RepositorySimpleClimbPixel
from paraai.repository.repository_terrain import RepositoryTerrain
from paraai.setup import setup
from paraai.tool_spacetime import REGION_BOUNDS

logger = logging.getLogger(__name__)


def evaluate(climbs: list) -> float:
    """Score a list of climbs. Placeholder."""
    logger.debug("evaluate(climbs=%s)", len(climbs))
    _ = climbs
    return 0.0


class MapDataset(Dataset):
    """Dataset of (input_map, target_map) pairs as tensors."""

    def __init__(
        self,
        input_maps: list[torch.Tensor],
        target_maps: list[torch.Tensor] | None = None,
        example_map: torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            input_maps: List of input images (C, H, W).
            target_maps: List of target images. If None, use example_map for all.
            example_map: Single target map used for all inputs when target_maps is None.
        """
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


class MapEstimateNet(nn.Module):
    """CNN that estimates a target map from an input map patch."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, size: int = 64) -> None:
        super().__init__()
        self.size = size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.decoder(h)


def _latlon_to_bbox(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """Bounding box around (lat, lon) with radius_m in meters."""
    radius_km = radius_m / 1000.0
    deg_lat = radius_km / 111.0
    deg_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat


def _extract_elevation_patch(
    elevation: np.ndarray,
    transform: rasterio.Affine,
    lat: float,
    lon: float,
    radius_m: float,
    size: int,
) -> torch.Tensor:
    """Extract elevation patch centered on (lat, lon), resize to (1, size, size), normalize to [0,1]."""
    lon_min, lat_min, lon_max, lat_max = _latlon_to_bbox(lat, lon, radius_m)
    from rasterio.windows import from_bounds

    win = from_bounds(lon_min, lat_min, lon_max, lat_max, transform)
    h, w = elevation.shape
    r0 = int(max(0, win.row_off))
    c0 = int(max(0, win.col_off))
    r1 = int(min(h, win.row_off + win.height))
    c1 = int(min(w, win.col_off + win.width))
    patch = elevation[r0:r1, c0:c1].astype(np.float32)
    valid = ~np.isnan(patch) & (patch > -500)
    if np.any(valid):
        lo, hi = np.percentile(patch[valid], [2, 98])
        patch = np.clip((patch - lo) / (hi - lo + 1e-10), 0, 1)
    else:
        patch = np.zeros_like(patch)
    patch = np.nan_to_num(patch, nan=0.0).astype(np.float32)
    if patch.size == 0:
        return torch.zeros(1, size, size, dtype=torch.float32)
    t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(torch.float32)
    t = nn.functional.interpolate(t, size=(size, size), mode="bilinear").squeeze(0)
    return t


def train_model(
    input_maps: list[torch.Tensor],
    example_map: torch.Tensor,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 8,
    device: str | None = None,
) -> MapEstimateNet:
    """
    Train a CNN to estimate the example map from input maps.

    Args:
        input_maps: List of input map tensors (C, H, W).
        example_map: Target map to estimate (C, H, W).
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        device: Device to train on (cuda/cpu).

    Returns:
        Trained model.
    """
    logger.info("train_model(epochs=%s, batch_size=%s, lr=%s)", epochs, batch_size, lr)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info("Training on %s", device)

    dataset = MapDataset(input_maps, example_map=example_map)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    in_c = input_maps[0].shape[0]
    out_c = example_map.shape[0]
    h, w = example_map.shape[1], example_map.shape[2]
    model = MapEstimateNet(in_channels=in_c, out_channels=out_c, size=max(h, w)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for inp_batch, tgt_batch in loader:
            inp_dev = inp_batch.to(device)
            tgt_dev = tgt_batch.to(device)
            optimizer.zero_grad()
            pred = model(inp_dev)
            if pred.shape != tgt_dev.shape:
                pred = nn.functional.interpolate(pred, size=tgt_dev.shape[2:], mode="bilinear")
            loss = criterion(pred, tgt_dev)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / n_batches if n_batches else 0
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %s/%s loss=%s", epoch + 1, epochs, f"{avg:.6f}")

    logger.info("train_model finished")
    return model


def build_dataset_from_climb_map(
    region: str,
    patch_size_m: float = 500.0,
    image_size: int = 64,
    grid_stride: int = 16,
    kernel_size_m: float = 200.0,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict]:
    """
    Build (elevation_patches, target_maps) from MapBuilderConvolution(kernel_size_m=200) strength map.

    Samples a grid of patches across the region. Target = strength value at patch center, normalized [0,1].
    """
    logger.info(
        "build_dataset_from_climb_map(region=%s, patch_size_m=%s, grid_stride=%s, kernel_size_m=%s)",
        region,
        patch_size_m,
        grid_stride,
        kernel_size_m,
    )
    bounding_box = REGION_BOUNDS.get(region.lower())
    if bounding_box is None:
        raise ValueError(f"Unknown region '{region}'. Available: {list(REGION_BOUNDS)}")

    repo_pixel = RepositorySimpleClimbPixel.get_instance()
    pixels = repo_pixel.get_all_in_bounding_box(
        bounding_box.lat_min, bounding_box.lat_max, bounding_box.lon_min, bounding_box.lon_max
    )
    if len(pixels) < 1:
        raise ValueError(f"No SimpleClimbPixels in region '{region}'")

    df = pd.DataFrame(
        [(p.lat, p.lon, p.climb_count, p.mean_climb_strength_m_s) for p in pixels],
        columns=["lat", "lon", "count", "strength"],
    )
    builder = MapBuilderConvolution(kernel_size_m=kernel_size_m)
    maps = builder.build(bounding_box, df)
    strength_vma = maps["strength"]

    repo_terrain = RepositoryTerrain.get_instance()
    terrain = repo_terrain.get_elevation(bounding_box)
    elevation = terrain["elevation"]
    transform = terrain["transform"]
    h, w = elevation.shape

    # Normalize strength to [0,1] using percentiles
    strength_arr = strength_vma.array
    valid = strength_arr[strength_arr > 0]
    if valid.size > 0:
        p2, p98 = np.percentile(valid, [2, 98])
        strength_lo, strength_hi = float(p2), float(p98)
    else:
        strength_lo, strength_hi = 0.0, 1.0
    if strength_hi <= strength_lo:
        strength_hi = strength_lo + 1e-6

    input_maps: list[torch.Tensor] = []
    target_maps: list[torch.Tensor] = []

    n_patches = 0
    for r in range(0, h, grid_stride):
        for c in range(0, w, grid_stride):
            lon, lat = rasterio.transform.xy(transform, r, c)
            lon_val, lat_val = float(lon), float(lat)
            patch = _extract_elevation_patch(elevation, transform, lat_val, lon_val, patch_size_m, image_size)
            raw_val = strength_vma.get_value(lat_val, lon_val)
            norm_val = float(np.clip((raw_val - strength_lo) / (strength_hi - strength_lo), 0, 1))
            target = torch.full((1, image_size, image_size), norm_val, dtype=torch.float32)
            input_maps.append(patch)
            target_maps.append(target)
            n_patches += 1

    logger.info("Dataset: %s patches from grid (stride=%s)", n_patches, grid_stride)
    logger.info("Strength range: %.3f - %.3f m/s (normalized)", strength_lo, strength_hi)

    # True grid for viz: strength in rasterio order
    true_grid = np.clip(
        (strength_arr.astype(np.float64) - strength_lo) / (strength_hi - strength_lo),
        0,
        1,
    ).astype(np.float32)

    viz_data = {
        "elevation": elevation,
        "transform": transform,
        "lon_min": bounding_box.lon_min,
        "lat_min": bounding_box.lat_min,
        "lon_max": bounding_box.lon_max,
        "lat_max": bounding_box.lat_max,
        "patch_size_m": patch_size_m,
        "image_size": image_size,
        "true_grid": true_grid,
        "strength_lo": strength_lo,
        "strength_hi": strength_hi,
    }
    return input_maps, target_maps, viz_data


def build_model(
    input_maps: list[torch.Tensor],
    target_maps: list[torch.Tensor],
    *,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    test_frac: float = 0.2,
    split_seed: int = 42,
    return_model: str = "lowest_test",
    model_path: Path | str | None = None,
) -> tuple[MapEstimateNet, float, list[int]]:
    """
    Train a CNN to estimate target map from elevation patches.

    Splits data into train/test. Saves best model when test loss improves.

    Args:
        return_model: "latest" | "lowest_train" | "lowest_test" - which model to return.

    Returns:
        (trained_model, evaluation_score, test_indices)
    """
    logger.info(
        "build_model(epochs=%s, batch_size=%s, test_frac=%s, split_seed=%s)",
        epochs,
        batch_size,
        test_frac,
        split_seed,
    )
    if not input_maps:
        raise ValueError("No maps provided")

    dataset = MapDataset(input_maps, target_maps=target_maps)
    n_total = len(dataset)
    n_test = max(1, int(n_total * test_frac))
    n_train = n_total - n_test
    train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(split_seed))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    in_c = input_maps[0].shape[0]
    out_c = target_maps[0].shape[0]
    image_size = input_maps[0].shape[1]
    model = MapEstimateNet(in_channels=in_c, out_channels=out_c, size=image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    logger.info("Train/test split: %s train, %s test", n_train, n_test)
    logger.info("Return model: %s", return_model)

    best_test_loss = float("inf")
    best_test_state: dict | None = None

    pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        for inp_batch, tgt_batch in train_loader:
            inp_dev = inp_batch.to(device)
            tgt_dev = tgt_batch.to(device)
            optimizer.zero_grad()
            pred = model(inp_dev)
            if pred.shape[2:] != tgt_dev.shape[2:]:
                pred = nn.functional.interpolate(pred, size=tgt_dev.shape[2:], mode="bilinear")
            loss = criterion(pred, tgt_dev)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train_batches += 1
        train_loss /= n_train_batches if n_train_batches else 1

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            n_test_batches = 0
            for inp_batch, tgt_batch in test_loader:
                inp_dev = inp_batch.to(device)
                tgt_dev = tgt_batch.to(device)
                pred = model(inp_dev)
                if pred.shape[2:] != tgt_dev.shape[2:]:
                    pred = nn.functional.interpolate(pred, size=tgt_dev.shape[2:], mode="bilinear")
                test_loss += criterion(pred, tgt_dev).item()
                n_test_batches += 1
            test_loss /= n_test_batches if n_test_batches else 1

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_state = copy.deepcopy(model.state_dict())
            if model_path:
                path = Path(model_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_test_state, path)
                logger.info("Saved best model (test_loss=%.6f) to %s", best_test_loss, path)

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", test_loss=f"{test_loss:.4f}")

    if return_model == "lowest_test" and best_test_state is not None:
        model.load_state_dict(best_test_state)
        logger.info("Loaded model with lowest test loss (%.6f)", best_test_loss)

    score = evaluate([])
    logger.info("build_model finished (score=%s)", score)
    test_indices = test_ds.indices
    return model, score, test_indices


def visualize_region_and_prediction(
    model: MapEstimateNet,
    viz_data: dict,
    device: torch.device,
    *,
    input_maps: list[torch.Tensor] | None = None,
    target_maps: list[torch.Tensor] | None = None,
    test_indices: list[int] | None = None,
    pred_stride: int = 16,
) -> None:
    """Show region with MapBuilderConvolution(200) strength map and CNN prediction."""
    logger.info("visualize_region_and_prediction(pred_stride=%s)", pred_stride)
    logger.info("Inference on %s", device)
    elevation = viz_data["elevation"]
    transform = viz_data["transform"]
    lon_min = viz_data["lon_min"]
    lat_min = viz_data["lat_min"]
    lon_max = viz_data["lon_max"]
    lat_max = viz_data["lat_max"]
    patch_size_m = viz_data["patch_size_m"]
    image_size = viz_data["image_size"]
    true_grid = viz_data["true_grid"]

    h, w = elevation.shape
    extent = [lon_min, lon_max, lat_min, lat_max]

    # Build prediction grid (sparse at stride, then interpolate)
    model.eval()
    stride = pred_stride
    n_rows = (h + stride - 1) // stride
    n_cols = (w + stride - 1) // stride
    n_patches = n_rows * n_cols
    logger.info("Building prediction grid: %s patches (stride=%s)", n_patches, stride)
    progress_interval = max(1, n_patches // 20)
    done = 0
    pred_sparse = np.zeros((n_rows, n_cols), dtype=np.float32)
    with torch.no_grad():
        for ri, r in enumerate(range(0, h, stride)):
            for ci, c in enumerate(range(0, w, stride)):
                lon, lat = rasterio.transform.xy(transform, r, c)
                patch = _extract_elevation_patch(elevation, transform, float(lat), float(lon), patch_size_m, image_size)
                patch_batch = patch.unsqueeze(0).to(device)
                pred = model(patch_batch)
                val = pred[0, 0, image_size // 2, image_size // 2].item()
                pred_sparse[ri, ci] = val
                done += 1
                if done % progress_interval == 0:
                    pct = 100 * done / n_patches
                    logger.info("Prediction progress: %s/%s (%.0f%%)", done, n_patches, pct)

    pred_t = torch.from_numpy(pred_sparse).unsqueeze(0).unsqueeze(0)
    pred_grid = nn.functional.interpolate(pred_t, size=(h, w), mode="bilinear", align_corners=False).squeeze().numpy().astype(np.float32)
    pred_grid = np.clip(pred_grid, 0, 1)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 2])
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_elev = fig.add_subplot(gs[0, 1])
    ax_true = fig.add_subplot(gs[1, 0])
    ax_pred = fig.add_subplot(gs[1, 1])

    # Top left: histogram of test set predictions vs targets
    preds_test: list[float] = []
    targets_test: list[float] = []
    if input_maps is not None and target_maps is not None and test_indices is not None:
        model.eval()
        with torch.no_grad():
            for idx in test_indices:
                inp = input_maps[idx].unsqueeze(0).to(device)
                tgt = target_maps[idx]
                pred = model(inp)
                val = pred[0, 0, image_size // 2, image_size // 2].item()
                label = tgt[0, image_size // 2, image_size // 2].item()
                preds_test.append(val)
                targets_test.append(label)
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

    # Row 1 right: heightmap (elevation)
    elev_display = np.clip(
        (elevation - np.nanpercentile(elevation, 2)) / (np.nanpercentile(elevation, 98) - np.nanpercentile(elevation, 2) + 1e-10),
        0,
        1,
    )
    ax_elev.imshow(elev_display, extent=extent, origin="upper", cmap="terrain")
    ax_elev.set_title("Heightmap")
    ax_elev.set_xlabel("Longitude")
    ax_elev.set_ylabel("Latitude")
    ax_elev.set_aspect("equal")

    # Row 2: true (MapBuilderConvolution) and predicted strength maps
    im_true = ax_true.imshow(true_grid, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im_true, ax=ax_true, label="Strength (normalized)")
    ax_true.set_title("True: MapBuilderConvolution(200m)")
    ax_true.set_xlabel("Longitude")
    ax_true.set_ylabel("Latitude")
    ax_true.set_aspect("equal")

    im_pred = ax_pred.imshow(pred_grid, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im_pred, ax=ax_pred, label="Predicted strength (normalized)")
    ax_pred.set_title("Predicted strength map")
    ax_pred.set_xlabel("Longitude")
    ax_pred.set_ylabel("Latitude")
    ax_pred.set_aspect("equal")

    plt.tight_layout()
    logger.info("visualize_region_and_prediction finished, showing plot")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN to predict MapBuilderConvolution(200) strength from elevation patches")
    parser.add_argument(
        "--region",
        type=str,
        default="sopot",
        help="Region (bassano, sopot, bansko, europe)",
    )
    parser.add_argument("--patch-size-m", type=float, default=500.0, help="Patch size in meters")
    parser.add_argument("--image-size", type=int, default=64, help="Patch resolution")
    parser.add_argument("--grid-stride", type=int, default=16, help="Stride for sampling grid (default 16)")
    parser.add_argument("--kernel-size-m", type=float, default=200.0, help="MapBuilderConvolution kernel (default 200)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fraction of data for test set")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for train/test split")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--pred-stride",
        type=int,
        default=16,
        help="Stride for prediction grid. Larger = fewer patches, faster.",
    )
    parser.add_argument(
        "--return-model",
        choices=["latest", "lowest_train", "lowest_test"],
        default="lowest_test",
        help="Which model to return",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="data/models/map_model.pt",
        help="Path to save best model (default: data/models/map_model.pt)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point: parse args, build dataset, train model, visualize."""
    args = parse_args()

    logger.info("main: starting")
    logger.info("parse_args: region=%s, epochs=%s, kernel_size_m=%s", args.region, args.epochs, args.kernel_size_m)

    input_maps, target_maps, viz_data = build_dataset_from_climb_map(
        args.region,
        patch_size_m=args.patch_size_m,
        image_size=args.image_size,
        grid_stride=args.grid_stride,
        kernel_size_m=args.kernel_size_m,
    )

    model, score, test_indices = build_model(
        input_maps,
        target_maps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
        return_model=args.return_model,
        model_path=args.model_path,
    )
    logger.info("Training complete. Score=%s", f"{score:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    visualize_region_and_prediction(
        model,
        viz_data,
        device,
        input_maps=input_maps,
        target_maps=target_maps,
        test_indices=test_indices,
        pred_stride=args.pred_stride,
    )
    logger.info("main: finished")


if __name__ == "__main__":
    setup()
    main()
