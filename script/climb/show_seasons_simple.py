"""
Run the same seasonal analysis as show_seasons.py on a random sample of 100k simple climbs.
"""
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats

from paraai.model.simple_climb import SimpleClimb
from paraai.repository.repository_simple_climb import RepositorySimpleClimb

SAMPLE_SIZE = 100_000

# Europe bounding box (same as show_seasons.py)
EUROPE_LNG_MIN, EUROPE_LAT_MIN = -10, 35
EUROPE_LNG_MAX, EUROPE_LAT_MAX = 40, 71

# Seasons: Spring Jan-May, Summer Jun-Aug, Autumn Sep-Dec
SPRING_MONTHS = {1, 2, 3, 4, 5}
SUMMER_MONTHS = {6, 7, 8}
AUTUMN_MONTHS = {9, 10, 11, 12}


def climb_vertical_speed_m_s(climb: SimpleClimb) -> float | None:
    """Mean vertical speed of climb in m/s, or None if invalid."""
    duration_s = climb.end_timestamp_utc - climb.start_timestamp_utc
    alts = climb.end_alt_m - climb.start_alt_m
    if duration_s <= 0 or alts <= 0:
        return None
    return alts / duration_s


def _in_europe(climb: SimpleClimb) -> bool:
    return (
        EUROPE_LNG_MIN <= climb.start_lon <= EUROPE_LNG_MAX
        and EUROPE_LAT_MIN <= climb.start_lat <= EUROPE_LAT_MAX
    )


async def main() -> None:
    path_dir_database = Path("data", "database_sqlite")
    repo = RepositorySimpleClimb.initialize_sqlite(path_dir_database)

    print(f"Sampling {SAMPLE_SIZE} simple climbs...")
    climbs = await repo.asample(SAMPLE_SIZE)
    climbs = [c for c in climbs if _in_europe(c)]
    print(f"After Europe filter: {len(climbs)} climbs")

    spring_speeds: list[float] = []
    summer_speeds: list[float] = []
    autumn_speeds: list[float] = []

    for c in climbs:
        v = climb_vertical_speed_m_s(c)
        if v is None:
            continue
        month = datetime.fromtimestamp(c.start_timestamp_utc, tz=timezone.utc).month
        if month in SPRING_MONTHS:
            spring_speeds.append(v)
        elif month in SUMMER_MONTHS:
            summer_speeds.append(v)
        elif month in AUTUMN_MONTHS:
            autumn_speeds.append(v)

    # Shared x grid for PDF evaluation
    all_speeds = spring_speeds + summer_speeds + autumn_speeds
    if not all_speeds:
        print("No climbs with valid speeds in Europe.")
        return
    x_min, x_max = np.min(all_speeds), np.max(all_speeds)
    x_grid = np.linspace(x_min, x_max, 200)

    # Print statistics per category
    for name, data in [
        ("Spring (Jan-May)", spring_speeds),
        ("Summer (Jun-Aug)", summer_speeds),
        ("Autumn (Sep-Dec)", autumn_speeds),
    ]:
        arr = np.array(data) if data else np.array([], dtype=float)
        mean = np.mean(arr) if len(arr) else float("nan")
        median = np.median(arr) if len(arr) else float("nan")
        q10, q90 = np.percentile(arr, [10, 90]) if len(arr) else (float("nan"), float("nan"))
        print(f"{name}: mean={mean:.4f} median={median:.4f} Q10={q10:.4f} Q90={q90:.4f} (n={len(arr)})")

    # Day-level scatter: mean vs std per day, by season
    day_speeds: dict[tuple[str, str], list[float]] = defaultdict(list)
    for c in climbs:
        v = climb_vertical_speed_m_s(c)
        if v is None:
            continue
        dt = datetime.fromtimestamp(c.start_timestamp_utc, tz=timezone.utc)
        date_key = dt.strftime("%Y-%m-%d")
        month = dt.month
        if month in SPRING_MONTHS:
            season = "Spring (Jan-May)"
        elif month in SUMMER_MONTHS:
            season = "Summer (Jun-Aug)"
        elif month in AUTUMN_MONTHS:
            season = "Autumn (Sep-Dec)"
        else:
            continue
        day_speeds[(date_key, season)].append(v)

    season_points: dict[str, tuple[list[float], list[float]]] = {
        "Spring (Jan-May)": ([], []),
        "Summer (Jun-Aug)": ([], []),
        "Autumn (Sep-Dec)": ([], []),
    }
    for (_, season), speeds in day_speeds.items():
        if len(speeds) >= 2:
            mean_val = np.mean(speeds)
            std_val = np.std(speeds)
            season_points[season][0].append(mean_val)
            season_points[season][1].append(std_val)

    # Single figure: PDF top, 3 scatter subplots bottom (one per season)
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig)
    ax_pdf = fig.add_subplot(gs[0, :])

    series = [
        (spring_speeds, "Spring (Jan-May)", "green"),
        (summer_speeds, "Summer (Jun-Aug)", "orange"),
        (autumn_speeds, "Autumn (Sep-Dec)", "brown"),
    ]
    for data, label, color in series:
        if len(data) >= 2:
            kde = stats.gaussian_kde(data)
            pdf = kde(x_grid)
            ax_pdf.plot(x_grid, pdf, color=color, label=label, linewidth=2)
            ax_pdf.fill_between(x_grid, pdf, alpha=0.2, color=color)
    ax_pdf.set_xlabel("Vertical speed (m/s)")
    ax_pdf.set_ylabel("Density")
    ax_pdf.set_title(f"Vertical speed by season (n={len(climbs)} simple climbs, Europe)")
    ax_pdf.legend()
    ax_pdf.grid(True, alpha=0.3)

    # Shared axis limits for scatter subplots (5th-95th percentile to exclude outliers)
    all_means = [m for means, _ in season_points.values() for m in means]
    all_stds = [s for _, stds in season_points.values() for s in stds]
    scatter_x_min, scatter_x_max = np.percentile(all_means, [5, 95]) if all_means else (0, 1)
    scatter_y_min, scatter_y_max = np.percentile(all_stds, [5, 95]) if all_stds else (0, 1)

    # 2D grid for KDE evaluation (shared across subplots)
    scatter_xx = np.linspace(scatter_x_min, scatter_x_max, 100)
    scatter_yy = np.linspace(scatter_y_min, scatter_y_max, 100)
    scatter_xx_mesh, scatter_yy_mesh = np.meshgrid(scatter_xx, scatter_yy)
    scatter_pos = np.vstack([scatter_xx_mesh.ravel(), scatter_yy_mesh.ravel()])

    scatter_config = [
        ("Spring (Jan-May)", "green"),
        ("Summer (Jun-Aug)", "orange"),
        ("Autumn (Sep-Dec)", "brown"),
    ]
    for i, (label, color) in enumerate(scatter_config):
        ax = fig.add_subplot(gs[1, i])
        means, stds = season_points[label]
        if len(means) >= 2:
            data_2d = np.vstack([means, stds])
            kde_2d = stats.gaussian_kde(data_2d)
            z = kde_2d(scatter_pos).reshape(scatter_xx_mesh.shape)
            cmaps = {"green": "Greens", "orange": "Oranges", "brown": "YlOrBr"}
            cmap_name = cmaps.get(color, "viridis")
            ax.contourf(scatter_xx_mesh, scatter_yy_mesh, z, levels=15, alpha=0.6, cmap=cmap_name)
            ax.contour(scatter_xx_mesh, scatter_yy_mesh, z, levels=8, colors=color, linewidths=0.8)
            ax.scatter(means, stds, color=color, alpha=0.8, s=20, edgecolors="black")
        ax.set_xlim(scatter_x_min, scatter_x_max)
        ax.set_ylim(scatter_y_min, scatter_y_max)
        ax.set_xlabel("Mean vertical speed (m/s)")
        ax.set_ylabel("Std vertical speed (m/s)")
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
