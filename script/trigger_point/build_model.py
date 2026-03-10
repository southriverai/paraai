"""Train a CNN to estimate a target map from elevation patches using trigger point labels."""

from __future__ import annotations

import argparse
import copy
import logging
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import Dataset

from paraai.repository.repository_trigger_point import RepositoryTriggerPoint
from paraai.tool_spacetime import haversine_km_tuple
from paraai.tools_terrain import load_terrain

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


def _negative_samples_around_cluster(
    lat_lon_pairs: list[tuple[float, float]],
    negative_example_count: int,
    min_distance_m: float,
    max_distance_m: float,
) -> list[tuple[float, float]]:
    """Sample negative points at least min_distance_m from trigger points, within max_distance_m."""
    logger.debug(
        "_negative_samples_around_cluster(count=%s, min=%sm, max=%sm)",
        negative_example_count,
        min_distance_m,
        max_distance_m,
    )
    out: list[tuple[float, float]] = []
    min_distance_km = min_distance_m / 1000.0
    deg_per_m = 1.0 / 111_000
    while len(out) < negative_example_count:
        trigger_point_lat, trigger_point_lon = random.choice(lat_lon_pairs)
        offset_deg = max_distance_m * deg_per_m
        lat = random.uniform(trigger_point_lat - offset_deg, trigger_point_lat + offset_deg)
        lon_scale = 1.0 / (111_000 * math.cos(math.radians(trigger_point_lat)))
        lon = random.uniform(
            trigger_point_lon - max_distance_m * lon_scale,
            trigger_point_lon + max_distance_m * lon_scale,
        )
        too_close = False
        for lat_lon_pair in lat_lon_pairs:
            if haversine_km_tuple(lat_lon_pair, (lat, lon)) < min_distance_km:
                too_close = True
                break
        if too_close:
            continue

        out.append((lat, lon))

    return out


def get_bounding_box(lat_lon_pairs: list[tuple[float, float]], margin_m: float) -> tuple[float, float, float, float]:
    """Get bounding box around lat_lon_pairs with margin_m in meters."""
    lat_min = min(p[0] for p in lat_lon_pairs)
    lat_max = max(p[0] for p in lat_lon_pairs)
    lon_min = min(p[1] for p in lat_lon_pairs)
    lon_max = max(p[1] for p in lat_lon_pairs)
    center_lat = (lat_min + lat_max) / 2
    deg_per_m_lat = 1.0 / 111_000
    deg_per_m_lon = 1.0 / (111_000 * math.cos(math.radians(center_lat)))
    margin_lat = margin_m * deg_per_m_lat
    margin_lon = margin_m * deg_per_m_lon
    return lat_min - margin_lat, lon_min - margin_lon, lat_max + margin_lat, lon_max + margin_lon


def build_dataset_from_trigger_point_name(
    trigger_point_name: str,
    radius_km: float = 20.0,
    patch_size_m: float = 500.0,
    image_size: int = 64,
    cache_dir: Path | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict]:
    margin_m = patch_size_m / 2.0

    """
    Build (elevation_patches, target_maps) from trigger point name.

    - Positive: all trigger points within radius_km of the named point.
    - Negative: points east and west at center lat of trigger points.
    """
    logger.info(
        "build_dataset_from_trigger_point_name(name=%s, radius_km=%s, patch_size_m=%s)",
        trigger_point_name,
        radius_km,
        patch_size_m,
    )
    repo = RepositoryTriggerPoint.get_instance()
    center = repo.get_by_name(trigger_point_name)
    if center is None:
        raise ValueError(f"No trigger point found with name '{trigger_point_name}'")

    positives_tps = repo.get_all_within_radius(center.lat, center.lon, radius_km * 1000.0)
    positives = [(tp.lat, tp.lon) for tp in positives_tps]
    if not positives:
        raise ValueError(f"No trigger points within {radius_km} km of '{trigger_point_name}'")

    negative_example_count = len(positives) * 10
    min_distance_m = 250
    max_distance_m = 2000
    negatives = _negative_samples_around_cluster(positives, negative_example_count, min_distance_m, max_distance_m)

    # Bbox for full region with padding
    lat_min, lon_min, lat_max, lon_max = get_bounding_box(positives, margin_m=margin_m)

    logger.info(
        "Loading terrain for region (lat=%s-%s, lon=%s-%s)",
        f"{lat_min:.4f}",
        f"{lat_max:.4f}",
        f"{lon_min:.4f}",
        f"{lon_max:.4f}",
    )
    terrain = load_terrain(
        lon_min,
        lat_min,
        lon_max,
        lat_max,
        cache_dir=cache_dir or Path("data", "terrain"),
    )
    elevation = terrain["elevation"]
    transform = terrain["transform"]

    input_maps: list[torch.Tensor] = []
    target_maps: list[torch.Tensor] = []

    logger.debug("Extracting %s positive patches", len(positives))
    for lat, lon in positives:
        patch = _extract_elevation_patch(elevation, transform, lat, lon, patch_size_m, image_size)
        input_maps.append(patch)
        target_maps.append(torch.ones(1, image_size, image_size))

    logger.debug("Extracting %s negative patches", len(negatives))
    for lat, lon in negatives:
        patch = _extract_elevation_patch(elevation, transform, lat, lon, patch_size_m, image_size)
        input_maps.append(patch)
        target_maps.append(torch.zeros(1, image_size, image_size))

    logger.info("Dataset: %s positive, %s negative", len(positives), len(negatives))
    logger.info("build_dataset_from_trigger_point_name finished")

    viz_data = {
        "elevation": elevation,
        "transform": transform,
        "lon_min": lon_min,
        "lat_min": lat_min,
        "lon_max": lon_max,
        "lat_max": lat_max,
        "positives": positives,
        "negatives": negatives,
        "patch_size_m": patch_size_m,
        "image_size": image_size,
    }
    return input_maps, target_maps, viz_data


def build_model(
    input_maps: list[torch.Tensor],
    target_maps: list[torch.Tensor],
    *,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    test_frac: float = 0.2,
    split_seed: int = 42,
    return_model: str = "lowest_test",
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
    best_train_loss = float("inf")
    best_test_state: dict | None = None
    best_train_state: dict | None = None

    for epoch in range(epochs):
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
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "Epoch %s/%s  train_loss=%s  test_loss=%s",
                epoch + 1,
                epochs,
                f"{train_loss:.6f}",
                f"{test_loss:.6f}",
            )

    if return_model == "lowest_test" and best_test_state is not None:
        model.load_state_dict(best_test_state)
        logger.info("Loaded model with lowest test loss (%.6f)", best_test_loss)
    elif return_model == "lowest_train" and best_train_state is not None:
        model.load_state_dict(best_train_state)
        logger.info("Loaded model with lowest train loss (%.6f)", best_train_loss)
    # else "latest" - keep current model

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
    pred_stride: int = 128,
    pred_margin_km: float | None = 10.0,
) -> None:
    """Show region with trigger points and predicted map side by side."""
    logger.info("visualize_region_and_prediction(pred_stride=%s, pred_margin_km=%s)", pred_stride, pred_margin_km)
    logger.info("Inference on %s", device)
    elevation = viz_data["elevation"]
    transform = viz_data["transform"]
    lon_min = viz_data["lon_min"]
    lat_min = viz_data["lat_min"]
    lon_max = viz_data["lon_max"]
    lat_max = viz_data["lat_max"]
    positives = viz_data["positives"]
    negatives = viz_data["negatives"]
    patch_size_m = viz_data["patch_size_m"]
    image_size = viz_data["image_size"]

    # Optionally crop to trigger region to limit patches
    if pred_margin_km is not None and (positives or negatives):
        from rasterio.windows import from_bounds

        pts = positives + negatives
        pt_lat_min = min(p[0] for p in pts)
        pt_lat_max = max(p[0] for p in pts)
        pt_lon_min = min(p[1] for p in pts)
        pt_lon_max = max(p[1] for p in pts)
        center_lat = (pt_lat_min + pt_lat_max) / 2
        deg_per_km = 1.0 / 111.0
        margin_deg_lat = pred_margin_km * deg_per_km
        margin_deg_lon = pred_margin_km * deg_per_km / math.cos(math.radians(center_lat))
        crop_lon_min = max(lon_min, pt_lon_min - margin_deg_lon)
        crop_lon_max = min(lon_max, pt_lon_max + margin_deg_lon)
        crop_lat_min = max(lat_min, pt_lat_min - margin_deg_lat)
        crop_lat_max = min(lat_max, pt_lat_max + margin_deg_lat)
        win = from_bounds(crop_lon_min, crop_lat_min, crop_lon_max, crop_lat_max, transform)
        r0 = int(max(0, win.row_off))
        c0 = int(max(0, win.col_off))
        r1 = int(min(elevation.shape[0], win.row_off + win.height))
        c1 = int(min(elevation.shape[1], win.col_off + win.width))
        elevation = elevation[r0:r1, c0:c1].copy()
        transform = rasterio.windows.transform(win, transform)
        lon_min, lat_min, lon_max, lat_max = crop_lon_min, crop_lat_min, crop_lon_max, crop_lat_max
        logger.info("Cropped to trigger region + %s km: %s x %s pixels", pred_margin_km, elevation.shape[0], elevation.shape[1])

    h, w = elevation.shape
    extent = [lon_min, lon_max, lat_min, lat_max]

    # Build prediction grid (sparse at stride, then interpolate to match true map resolution)
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
                patch = _extract_elevation_patch(elevation, transform, lat, lon, patch_size_m, image_size)
                patch_batch = patch.unsqueeze(0).to(device)
                pred = model(patch_batch)
                val = pred[0, 0, image_size // 2, image_size // 2].item()
                pred_sparse[ri, ci] = val
                done += 1
                if done % progress_interval == 0:
                    pct = 100 * done / n_patches
                    logger.info("Prediction progress: %s/%s (%.0f%%)", done, n_patches, pct)

    # Interpolate to full resolution to match true map
    pred_t = torch.from_numpy(pred_sparse).unsqueeze(0).unsqueeze(0)
    pred_grid = nn.functional.interpolate(pred_t, size=(h, w), mode="bilinear", align_corners=False).squeeze().numpy().astype(np.float32)
    pred_grid = np.clip(pred_grid, 0, 1)

    # Build true map: 1 at trigger points, 0 at negatives, 0.5 elsewhere
    true_grid = np.full((h, w), 0.5, dtype=np.float32)
    marker_radius = 2
    for lat, lon in positives:
        row, col = rasterio.transform.rowcol(transform, [lon], [lat])
        r, c = int(row[0]), int(col[0])
        for dr in range(-marker_radius, marker_radius + 1):
            for dc in range(-marker_radius, marker_radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    true_grid[rr, cc] = 1.0
    for lat, lon in negatives:
        row, col = rasterio.transform.rowcol(transform, [lon], [lat])
        r, c = int(row[0]), int(col[0])
        for dr in range(-marker_radius, marker_radius + 1):
            for dc in range(-marker_radius, marker_radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    true_grid[rr, cc] = 0.0

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 2])
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_elev = fig.add_subplot(gs[0, 1])
    ax_true = fig.add_subplot(gs[1, 0])
    ax_pred = fig.add_subplot(gs[1, 1])

    # Top left: histograms of test set predictions (positive vs negative)
    preds_pos: list[float] = []
    preds_neg: list[float] = []
    if input_maps is not None and target_maps is not None and test_indices is not None:
        model.eval()
        image_size = viz_data["image_size"]
        with torch.no_grad():
            for idx in test_indices:
                inp = input_maps[idx].unsqueeze(0).to(device)
                tgt = target_maps[idx]
                pred = model(inp)
                val = pred[0, 0, image_size // 2, image_size // 2].item()
                label = tgt[0, image_size // 2, image_size // 2].item()
                if label >= 0.5:
                    preds_pos.append(val)
                else:
                    preds_neg.append(val)
    bins = np.linspace(0, 1, 21)
    if preds_pos:
        ax_hist.hist(preds_pos, bins=bins, alpha=0.7, color="green", label="Positive", density=True)
    if preds_neg:
        ax_hist.hist(preds_neg, bins=bins, alpha=0.7, color="red", label="Negative", density=True)
    ax_hist.set_xlabel("Prediction")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Test set predictions")
    ax_hist.legend()
    ax_hist.set_xlim(0, 1)

    # Row 1 right: heightmap (elevation)
    elev_display = np.clip(
        (elevation - np.nanpercentile(elevation, 2)) / (np.nanpercentile(elevation, 98) - np.nanpercentile(elevation, 2) + 1e-10),
        0,
        1,
    )
    ax_elev.imshow(elev_display, extent=extent, origin="upper", cmap="terrain")
    if positives:
        lons_p = [p[1] for p in positives]
        lats_p = [p[0] for p in positives]
        ax_elev.scatter(lons_p, lats_p, c="red", s=30, marker="o", label="Trigger points", zorder=5)
    if negatives:
        lons_n = [p[1] for p in negatives]
        lats_n = [p[0] for p in negatives]
        ax_elev.scatter(lons_n, lats_n, c="blue", s=15, marker="x", label="Negative samples", zorder=4)
    ax_elev.set_title("Heightmap")
    ax_elev.set_xlabel("Longitude")
    ax_elev.set_ylabel("Latitude")
    ax_elev.legend(loc="upper right", fontsize=8)
    ax_elev.set_aspect("equal")

    # Row 2: true and predicted maps with viridis colormap
    im_true = ax_true.imshow(true_grid, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im_true, ax=ax_true)
    ax_true.set_title("True labels")
    ax_true.set_xlabel("Longitude")
    ax_true.set_ylabel("Latitude")
    ax_true.set_aspect("equal")

    im_pred = ax_pred.imshow(pred_grid, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1)
    if positives:
        ax_pred.scatter(lons_p, lats_p, c="red", s=30, marker="o", zorder=5)
    plt.colorbar(im_pred, ax=ax_pred, label="Predicted probability")
    ax_pred.set_title("Predicted trigger map")
    ax_pred.set_xlabel("Longitude")
    ax_pred.set_ylabel("Latitude")
    ax_pred.set_aspect("equal")

    plt.tight_layout()
    logger.info("visualize_region_and_prediction finished, showing plot")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN from trigger point name (elevation + labels)")
    parser.add_argument("--name", type=str, required=True, help="Trigger point name (center of region)")
    parser.add_argument("--radius-km", type=float, default=20.0, help="Gather trigger points within N km")
    parser.add_argument("--patch-size-m", type=float, default=500.0, help="Patch size in meters")
    parser.add_argument("--image-size", type=int, default=64, help="Patch resolution")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fraction of data for test set (default 0.2)")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for train/test split")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default INFO)",
    )
    parser.add_argument(
        "--pred-margin-km",
        type=float,
        default=10.0,
        help="Crop prediction to trigger region + N km (default 10). Set to 0 for full region.",
    )
    parser.add_argument(
        "--pred-stride",
        type=int,
        default=4,
        help="Stride for prediction grid (default 128). Larger = fewer patches, faster.",
    )
    parser.add_argument(
        "--return-model",
        choices=["latest", "lowest_train", "lowest_test"],
        default="lowest_test",
        help="Which model to return: latest, lowest_train, or lowest_test (default).",
    )
    return parser.parse_args()


def _setup_logging(level: str) -> None:
    """Configure logging for the module."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Main entry point: parse args, build dataset, train model, visualize."""
    args = parse_args()
    _setup_logging(args.log_level)

    logger.info("main: starting")
    logger.info("parse_args: name=%s, radius_km=%s, epochs=%s", args.name, args.radius_km, args.epochs)

    logger.info("RepositoryTriggerPoint.initialize_sqlite")
    RepositoryTriggerPoint.initialize_sqlite(Path("data", "database_sqlite"))

    input_maps, target_maps, viz_data = build_dataset_from_trigger_point_name(
        args.name,
        radius_km=args.radius_km,
        patch_size_m=args.patch_size_m,
        image_size=args.image_size,
    )

    model, score, test_indices = build_model(
        input_maps,
        target_maps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
        return_model=args.return_model,
    )
    logger.info("Training complete. Score=%s", f"{score:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    pred_margin = None if args.pred_margin_km <= 0 else args.pred_margin_km
    visualize_region_and_prediction(
        model,
        viz_data,
        device,
        input_maps=input_maps,
        target_maps=target_maps,
        test_indices=test_indices,
        pred_stride=args.pred_stride,
        pred_margin_km=pred_margin,
    )
    logger.info("main: finished")


if __name__ == "__main__":
    main()
