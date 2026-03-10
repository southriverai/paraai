"""For each (name, lat, lon) tuple, find the closest trigger point and update its name."""

from __future__ import annotations

import asyncio
from pathlib import Path

from paraai.repository.repository_trigger_point import RepositoryTriggerPoint
from paraai.tool_spacetime import haversine_m


async def name_trigger_points(tuples: list[tuple[str, float, float]]) -> None:
    """For each (name, lat, lon), find the closest trigger point in the repo and set its name."""
    repo = RepositoryTriggerPoint.initialize_sqlite(Path("data", "database_sqlite"))
    all_tps = repo.get_all()
    if not all_tps:
        print("No trigger points in repository")
        return

    for name, lat, lon in tuples:
        closest = min(all_tps, key=lambda tp: haversine_m(tp.lat, tp.lon, lat, lon))
        dist_m = haversine_m(closest.lat, closest.lon, lat, lon)
        closest.name = name
        repo.insert(closest)
        print(f"Updated {closest.trigger_point_id} -> '{name}' (closest at {dist_m:.0f} m)")


if __name__ == "__main__":
    tuples: list[tuple[str, float, float]] = [
        ("sopot_house_a", 42.685910, 24.750476),
        ("sopot_p1_a", 42.695861, 24.768874),
        ("bir_house_a", 32.0598, 76.744566),
    ]
    asyncio.run(name_trigger_points(tuples))
