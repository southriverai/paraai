"""Train a CNN to estimate a target map from input map patches."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn
from torch.utils.data import Dataset

# Placeholder types until proper imports
# from paraai.model.simple_climb import SimpleClimb


class MapProvider(Protocol):
    """Provides map images for a given location and radius."""

    def get_image(self, lat: float, lon: float, radius_m: float):
        ...


def evaluate(climbs: list) -> float:
    """Score a list of climbs. Placeholder."""
    _ = climbs
    return 0.0


def extract_patch(
    map_provider: MapProvider, lat: float, lon: float, radius_m: float
) -> list:
    """Extract climbs for a map patch. Placeholder."""
    _ = map_provider, lat, lon, radius_m
    return []


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


def pil_to_tensor(img, size: int = 64) -> torch.Tensor:
    """Convert PIL Image to tensor (C, H, W) in [0, 1], resized to size x size."""
    import numpy as np

    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = arr.transpose(2, 0, 1).astype("float32") / 255.0
    t = torch.from_numpy(arr)
    if t.shape[1] != size or t.shape[2] != size:
        t = nn.functional.interpolate(
            t.unsqueeze(0), size=(size, size), mode="bilinear"
        ).squeeze(0)
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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

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
            print(f"Epoch {epoch + 1}/{epochs} loss={avg:.6f}")

    return model


def build_model(
    list_trigger_points: list[tuple[float, float]],
    list_non: list[tuple[float, float]],
    map_provider: MapProvider,
    patch_size_m: float = 500.0,
    image_size: int = 64,
    epochs: int = 50,
) -> tuple[MapEstimateNet, float]:
    """
    Build a model from trigger points and non-trigger points.

    Uses trigger-point maps as positive targets (1) and non-trigger maps as negatives (0).
    Trains a CNN to estimate a target heatmap from map imagery.

    Returns:
        (trained_model, evaluation_score)
    """
    input_maps: list[torch.Tensor] = []
    target_maps: list[torch.Tensor] = []

    for lat, lon in list_trigger_points:
        img = map_provider.get_image(lat, lon, patch_size_m)
        tensor = pil_to_tensor(img, size=image_size)
        input_maps.append(tensor)
        target_maps.append(torch.ones(1, image_size, image_size))

    for lat, lon in list_non:
        img = map_provider.get_image(lat, lon, patch_size_m)
        tensor = pil_to_tensor(img, size=image_size)
        input_maps.append(tensor)
        target_maps.append(torch.zeros(1, image_size, image_size))

    if not input_maps:
        raise ValueError("No maps provided")

    dataset = MapDataset(input_maps, target_maps=target_maps)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    in_c = input_maps[0].shape[0]
    out_c = target_maps[0].shape[0]
    model = MapEstimateNet(in_channels=in_c, out_channels=out_c, size=image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for inp_batch, tgt_batch in loader:
            inp_dev = inp_batch.to(device)
            tgt_dev = tgt_batch.to(device)
            optimizer.zero_grad()
            pred = model(inp_dev)
            if pred.shape[2:] != tgt_dev.shape[2:]:
                pred = nn.functional.interpolate(pred, size=tgt_dev.shape[2:], mode="bilinear")
            loss = criterion(pred, tgt_dev)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} loss={total_loss / n:.6f}")

    score = evaluate([])
    return model, score


if __name__ == "__main__":
    # Demo: synthetic data
    torch.manual_seed(42)
    n_samples = 32
    size = 64
    input_maps = [torch.rand(3, size, size) for _ in range(n_samples)]
    example_map = torch.rand(1, size, size)

    model = train_model(input_maps, example_map, epochs=30, batch_size=8)
    print("Training complete.")
