from __future__ import annotations

import contextlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import contextily as cx
import matplotlib.pyplot as plt
from tqdm import tqdm

from paraai.model.simple_climb import SimpleClimb
from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.tool_spacetime import haversine_m, utc_to_solar_hour

CACHE_DIR = Path("data", "area_climbs_cache")


def _cache_path(name: str, lat_deg: float, lng_deg: float, radius_m: float) -> Path:
    safe_name = re.sub(r"[^\w\-]", "_", name.lower()).strip("_") or "area"
    return CACHE_DIR / f"{safe_name}_{lat_deg}_{lng_deg}_{radius_m}.json"


def _load_cache(path: Path) -> list[SimpleClimb] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return [SimpleClimb.model_validate(d) for d in data.get("climbs", [])]
    except (json.JSONDecodeError, KeyError):
        return None


def _save_cache(path: Path, climbs: list[SimpleClimb]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"climbs": [c.model_dump() for c in climbs]}
    path.write_text(json.dumps(data))


def show_area_simple_climbs(name: str, lat_deg: float, lng_deg: float, radius_m: float) -> None:
    """Load SimpleClimbs within radius of (lat, lng), show histograms of strength and time of day."""
    cache_path = _cache_path(name, lat_deg, lng_deg, radius_m)
    in_radius = _load_cache(cache_path)
    if in_radius is None:
        path_db = Path("data", "database_sqlite")
        repo = RepositorySimpleClimb.initialize_sqlite(path_db)
        climbs = repo.get_all()
        print(f"Loaded {len(climbs)} climbs")
        in_radius = []
        for c in tqdm(climbs, desc="Filtering climbs"):
            d = haversine_m(c.start_lat, c.start_lon, lat_deg, lng_deg)
            if d <= radius_m:
                in_radius.append(c)
        _save_cache(cache_path, in_radius)
        print(f"Cached {len(in_radius)} climbs to {cache_path}")
    else:
        print(f"Loaded {len(in_radius)} climbs from cache")
    print(f"{name}: {len(in_radius)} climbs within {radius_m/1000:.1f} km")

    strengths: list[float] = []
    hours: list[float] = []
    months: list[int] = []
    years: list[int] = []
    for c in in_radius:
        s = c.climb_strength_m_s
        if s > 0:
            strengths.append(s)
        dt = datetime.fromtimestamp(c.start_timestamp_utc, tz=timezone.utc)
        hours.append(utc_to_solar_hour(dt, lng_deg))
        months.append(dt.month)
        years.append(dt.year)

    print(f"{name}: {len(in_radius)} climbs within {radius_m/1000:.1f} km, {len(strengths)} with valid strength")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    if strengths:
        ax1.hist(strengths, bins=30, edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Climb strength (m/s)")
        ax1.set_ylabel("Count")
        ax1.set_title("Strength distribution")
        ax1.grid(True, alpha=0.3)

    ax2.hist(hours, bins=24, range=(0, 24), edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Time of day (local solar hour)")
    ax2.set_ylabel("Count")
    ax2.set_title("Time of day distribution")
    ax2.set_xticks(range(0, 25, 2))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    ax2.grid(True, alpha=0.3)

    ax3.hist(months, bins=12, range=(0.5, 12.5), edgecolor="black", alpha=0.7)
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Count")
    ax3.set_title("Climbs per month")
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax3.grid(True, alpha=0.3)

    year_bins = list(range(min(years), max(years) + 2)) if years else [0, 1]
    ax4.hist(years, bins=year_bins, edgecolor="black", alpha=0.7)
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Count")
    ax4.set_title("Climbs per year")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f"{name} (n={len(in_radius)})")
    plt.tight_layout()
    plt.show()

    # Second figure: scatter plot of climb locations with map background
    fig2, ax = plt.subplots(figsize=(8, 6))
    lats = [c.start_lat for c in in_radius]
    lons = [c.start_lon for c in in_radius]
    ax.scatter(lons, lats, s=2, alpha=0.6, zorder=2)
    ax.axhline(lat_deg, color="gray", linestyle="--", alpha=0.5, zorder=1)
    ax.axvline(lng_deg, color="gray", linestyle="--", alpha=0.5, zorder=1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{name}: climb locations (n={len(in_radius)})")
    ax.set_aspect("equal")
    if lons and lats:
        pad_lon = max(0.05, (max(lons) - min(lons)) * 0.1)
        pad_lat = max(0.05, (max(lats) - min(lats)) * 0.1)
        ax.set_xlim(min(lons) - pad_lon, max(lons) + pad_lon)
        ax.set_ylim(min(lats) - pad_lat, max(lats) + pad_lat)
    with contextlib.suppress(Exception):
        cx.add_basemap(ax, crs="EPSG:4326", alpha=0.6)  # fallback if tiles unavailable (e.g. offline)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_area_simple_climbs("Staro planina", 42.71039215412987, 24.7613273643503, 100000.0)
