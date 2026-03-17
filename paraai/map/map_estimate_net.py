"""CNN that estimates a target map from an input map patch."""

from __future__ import annotations

import torch
from torch import nn


class MapEstimateNetSimple(nn.Module):
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


class MapEstimateNetTime(nn.Module):
    """Estimates a single strength value per patch. Conv encoder on elevation, then dense layer
    with time_of_day and time_of_year concatenated. Output: scalar per patch (no decoder)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, size: int = 64) -> None:
        super().__init__()
        self.size = size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After 2 conv+pool: 64x16x16 for 64x64 input. Global avg pool -> 64.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 + 2, 1)  # 64 features + time_of_day + time_of_year

    def forward(
        self,
        x: torch.Tensor,
        time_of_day_norm: torch.Tensor,
        time_of_year_norm: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(x)
        h = self.pool(h).flatten(1)  # (B, 64)
        time_params = torch.stack([time_of_day_norm, time_of_year_norm], dim=1)  # (B, 2)
        combined = torch.cat([h, time_params], dim=1)  # (B, 66)
        return self.fc(combined)  # (B, 1)
