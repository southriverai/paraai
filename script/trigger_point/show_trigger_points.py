from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from paraai.repository.repository_trigger_point import RepositoryTriggerPoint


async def show_trigger_point(repo_trigger_point: RepositoryTriggerPoint, trigger_point_id: str) -> None:
    trigger_point = repo_trigger_point.get(trigger_point_id)
    if trigger_point is None:
        return
    climbs = trigger_point.climbs
    print(f"{trigger_point.name} ({trigger_point_id}): {len(climbs)} climbs within {trigger_point.radius_m/1000:.1f} km")

    # Scatter plots: time of day vs strength, day of year vs strength, strength vs max altitude
    time_strength: list[tuple[float, float]] = []
    doy_strength: list[tuple[float, float]] = []
    strength_alt_pairs: list[tuple[float, float]] = []
    for c in climbs:
        dt = datetime.fromtimestamp(c.start_timestamp_utc, tz=timezone.utc)
        time_h = dt.hour + dt.minute / 60 + dt.second / 3600
        doy = dt.timetuple().tm_yday - 1  # 0-364
        s = c.climb_strength_m_s()
        if s is not None:
            time_strength.append((time_h, s))
            doy_strength.append((doy, s))
            strength_alt_pairs.append((s, c.end_alt_m))

    fig2, axes = plt.subplots(1, 3, figsize=(14, 5))
    if time_strength:
        times, strengths = zip(*time_strength)
        axes[0].scatter(times, strengths, s=4, alpha=0.5)
    axes[0].set_xlabel("Time of day (h)")
    axes[0].set_ylabel("Climb strength (m/s)")
    axes[0].set_title(f"{trigger_point.name}: time of day vs strength (n={len(time_strength)})")
    axes[0].set_xlim(0, 24)
    axes[0].grid(True, alpha=0.3)

    if doy_strength:
        doys, strengths = zip(*doy_strength)
        axes[1].scatter(doys, strengths, s=4, alpha=0.5)
    axes[1].set_xlabel("Day of year")
    axes[1].set_ylabel("Climb strength (m/s)")
    axes[1].set_title(f"{trigger_point.name}: day of year vs strength (n={len(doy_strength)})")
    axes[1].set_xlim(0, 365)
    axes[1].grid(True, alpha=0.3)

    if strength_alt_pairs:
        strengths, max_alts = zip(*strength_alt_pairs)
        axes[2].scatter(strengths, max_alts, s=4, alpha=0.5)
        axes[2].set_xlabel("Climb strength (m/s)")
        axes[2].set_ylabel("Max climb altitude (m)")
        axes[2].set_title(f"{trigger_point.name}: max altitude vs strength (n={len(strength_alt_pairs)})")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].set_xlabel("Climb strength (m/s)")
        axes[2].set_ylabel("Max climb altitude (m)")
        axes[2].set_title(f"{trigger_point.name}: max altitude vs strength (no data)")
    plt.tight_layout()


if __name__ == "__main__":
    repo_trigger_point = RepositoryTriggerPoint.initialize_sqlite(Path("data", "trigger_points"))
    for trigger_point_id in repo_trigger_point.get_all_ids():
        asyncio.run(show_trigger_point(repo_trigger_point, trigger_point_id))
    plt.show()
