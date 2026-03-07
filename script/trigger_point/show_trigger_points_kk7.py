"""Show KK7 trigger points from CSV on a map around Sopot."""

from __future__ import annotations

import contextlib
from pathlib import Path

import contextily as cx
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("data", "trigger_points", "hotspots_kk7_all_all_20250301.csv")

# Sopot, Bulgaria area (lat, lon)
SOPOT_LAT = 42.68
SOPOT_LON = 24.75
SOPOT_RADIUS_DEG = 0.15  # ~15 km


def load_trigger_points_near_sopot(path: Path) -> tuple[list[float], list[float]]:
    """Load trigger points from CSV, filter to Sopot area. Returns (lats, lons)."""
    data = pd.read_csv(path)
    lat_col = "latitude degree WGS84"
    lon_col = "longitude degree WGS84"
    if lat_col not in data.columns or lon_col not in data.columns:
        raise ValueError(f"CSV must have columns '{lat_col}' and '{lon_col}'")
    mask = (
        (data[lat_col] >= SOPOT_LAT - SOPOT_RADIUS_DEG)
        & (data[lat_col] <= SOPOT_LAT + SOPOT_RADIUS_DEG)
        & (data[lon_col] >= SOPOT_LON - SOPOT_RADIUS_DEG)
        & (data[lon_col] <= SOPOT_LON + SOPOT_RADIUS_DEG)
    )
    subset = data[mask]
    lats = subset[lat_col].astype(float).tolist()
    lons = subset[lon_col].astype(float).tolist()
    return lats, lons


if __name__ == "__main__":
    print(f"Loading trigger points from {CSV_PATH}...")
    lats, lons = load_trigger_points_near_sopot(CSV_PATH)
    print(f"Found {len(lats)} trigger points near Sopot ({SOPOT_LAT}, {SOPOT_LON})")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(lons, lats, s=8, c="red", alpha=0.7, zorder=2, label="KK7 hotspots")
    ax.scatter([SOPOT_LON], [SOPOT_LAT], s=100, c="blue", marker="*", zorder=3, label="Sopot center")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"KK7 trigger points near Sopot (n={len(lats)})")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlim(SOPOT_LON - SOPOT_RADIUS_DEG, SOPOT_LON + SOPOT_RADIUS_DEG)
    ax.set_ylim(SOPOT_LAT - SOPOT_RADIUS_DEG, SOPOT_LAT + SOPOT_RADIUS_DEG)
    with contextlib.suppress(Exception):
        cx.add_basemap(ax, crs="EPSG:4326", alpha=0.6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
