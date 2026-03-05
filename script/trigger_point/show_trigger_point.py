from __future__ import annotations

import contextlib
import re
from pathlib import Path

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

from paraai.model.simple_climb import SimpleClimb
from paraai.repository.repository_cache import RepositoryCache
from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.tool_spacetime import haversine_m

CACHE_DIR = Path("data", "area_climbs_cache")


def _cache_key(name: str, lat_deg: float, lng_deg: float, radius_m: float) -> str:
    safe_name = re.sub(r"[^\w\-]", "_", name.lower()).strip("_") or "area"
    return f"{safe_name}_{lat_deg}_{lng_deg}_{radius_m}"


def show_trigger_point(name: str, lat_deg: float, lng_deg: float, radius_m: float) -> None:
    """Load SimpleClimbs within radius of (lat, lng), show histograms of strength and time of day."""
    cache = RepositoryCache(CACHE_DIR)
    cache_key = _cache_key(name, lat_deg, lng_deg, radius_m)
    data = cache.get(cache_key)
    in_radius: list[SimpleClimb]
    if data is None:
        path_db = Path("data", "database_sqlite")
        repo = RepositorySimpleClimb.initialize_sqlite(path_db)
        climbs = repo.get_all()
        print(f"Loaded {len(climbs)} climbs")
        in_radius = []
        for c in tqdm(climbs, desc="Filtering climbs"):
            d = haversine_m(c.start_lat, c.start_lon, lat_deg, lng_deg)
            if d <= radius_m:
                in_radius.append(c)
        cache.set(cache_key, {"climbs": [c.model_dump() for c in in_radius]})
        print(f"Cached {len(in_radius)} climbs")
    else:
        in_radius = [SimpleClimb.model_validate(d) for d in data.get("climbs", [])]
        print(f"Loaded {len(in_radius)} climbs from cache")
    print(f"{name}: {len(in_radius)} climbs within {radius_m/1000:.1f} km")

    # Map figure: scatter and KDE as separate subplots with shared background
    lats = [c.start_lat for c in in_radius]
    lons = [c.start_lon for c in in_radius]
    x_min = y_min = x_max = y_max = None
    if lons and lats:
        lons_arr = np.array(lons)
        lats_arr = np.array(lats)
        pad_lon = max(0.05, (lons_arr.max() - lons_arr.min()) * 0.1)
        pad_lat = max(0.05, (lats_arr.max() - lats_arr.min()) * 0.1)
        x_min, x_max = lons_arr.min() - pad_lon, lons_arr.max() + pad_lon
        y_min, y_max = lats_arr.min() - pad_lat, lats_arr.max() + pad_lat

    def _add_shared_background(ax: plt.Axes) -> None:
        if x_min is not None:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            with contextlib.suppress(Exception):
                cx.add_basemap(ax, crs="EPSG:4326", alpha=0.6)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(lat_deg, color="gray", linestyle="--", alpha=0.5, zorder=1)
        ax.axvline(lng_deg, color="gray", linestyle="--", alpha=0.5, zorder=1)

    # Single figure: scatter and KDE as separate subplots with shared background
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _add_shared_background(ax1)
    _add_shared_background(ax2)
    ax1.scatter(lons, lats, s=2, alpha=0.6, zorder=2)
    ax1.set_title(f"{name}: climb locations (n={len(in_radius)})")
    if lons and lats:
        try:
            xx = np.linspace(x_min, x_max, 100)
            yy = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(xx, yy)
            positions = np.vstack([X.ravel(), Y.ravel()])
            values = np.vstack([np.array(lons), np.array(lats)])
            kde = stats.gaussian_kde(values)
            Z = kde(positions).reshape(X.shape)
            Z_norm = Z / Z.max() * 1000 if Z.max() > 0 else Z
            KDE_THRESHOLD = 200
            Z_masked = np.ma.masked_where(Z_norm < KDE_THRESHOLD, Z_norm)
            mesh = ax2.pcolormesh(X, Y, Z_masked, shading="auto", cmap="hot", alpha=0.8, zorder=1)
            plt.colorbar(mesh, ax=ax2, label="Density")
        except Exception:
            pass
    ax2.set_title(f"{name}: climb density (KDE)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_trigger_point("Staro planina", 42.71039215412987, 24.7613273643503, 5000.0)
