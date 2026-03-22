"""CNN that estimates a target map from an input map patch."""

from __future__ import annotations

import torch
from torch import nn


class MapEstimateNetSimple(nn.Module):
    """Estimates a single strength value per patch. Conv encoder + dense with time and ground altitude.
    Output: scalar per patch."""

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
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 + 2 + 1, 1)  # 64 + time(2) + ground_alt(1)

    def forward(
        self,
        x: torch.Tensor,
        time_of_day_norm: torch.Tensor,
        time_of_year_norm: torch.Tensor,
        ground_alt_norm: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(x)
        h = self.pool(h).flatten(1)  # (B, 64)
        time_params = torch.stack([time_of_day_norm, time_of_year_norm], dim=1)  # (B, 2)
        combined = torch.cat([h, time_params, ground_alt_norm.unsqueeze(1)], dim=1)  # (B, 67)
        return self.fc(combined)  # (B, 1)


class MapEstimateNetTime(nn.Module):
    """Estimates a single strength value per patch. Conv encoder + dense with time and ground altitude.
    Output: scalar per patch."""

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
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 + 2 + 1, 1)  # 64 + time(2) + ground_alt(1)

    def forward(
        self,
        x: torch.Tensor,
        time_of_day_norm: torch.Tensor,
        time_of_year_norm: torch.Tensor,
        ground_alt_norm: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(x)
        h = self.pool(h).flatten(1)  # (B, 64)
        time_params = torch.stack([time_of_day_norm, time_of_year_norm], dim=1)  # (B, 2)
        combined = torch.cat([h, time_params, ground_alt_norm.unsqueeze(1)], dim=1)  # (B, 67)
        return self.fc(combined)  # (B, 1)
