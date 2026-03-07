"""Load trigger points from hotspots CSV and insert with hash-based IDs."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pandas as pd

from paraai.model.trigger_point import TriggerPoint
from paraai.repository.repository_trigger_point import RepositoryTriggerPoint

CSV_PATH = Path("data", "source", "hotspots_kk7_all_all_20250301.csv")
RADIUS_M = 250.0


def latlon_hash(lat: float, lon: float) -> str:
    """Stable hash-based ID from lat/lon (rounded to 6 decimals)."""
    key = f"{round(lat, 6)}_{round(lon, 6)}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def load_trigger_points_from_csv(path: Path) -> list[TriggerPoint]:
    """Load trigger points from CSV, each with hash-based trigger_point_id from lat/lon."""
    data = pd.read_csv(path)
    lat_col = "latitude degree WGS84"
    lon_col = "longitude degree WGS84"
    if lat_col not in data.columns or lon_col not in data.columns:
        raise ValueError(f"CSV must have columns '{lat_col}' and '{lon_col}'")
    trigger_points = []
    for _, row in data.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        trigger_point_id = f"kk7_{latlon_hash(lat, lon)}"
        trigger_points.append(
            TriggerPoint(
                trigger_point_id=trigger_point_id,
                name=trigger_point_id,
                lat=lat,
                lon=lon,
                radius_m=RADIUS_M,
                climbs=[],
            )
        )
    return trigger_points


async def insert_trigger_points(trigger_points: list[TriggerPoint]) -> None:
    """Load SimpleClimbs within radius of each trigger point and insert."""
    repo_trigger_point = RepositoryTriggerPoint.initialize_sqlite(Path("data", "database_sqlite"))
    for trigger_point in trigger_points:
        repo_trigger_point.insert(trigger_point)


if __name__ == "__main__":
    print(f"Loading trigger points from {CSV_PATH}...")
    trigger_points = load_trigger_points_from_csv(CSV_PATH)
    print(f"Loaded {len(trigger_points)} trigger points")
    asyncio.run(insert_trigger_points(trigger_points))
