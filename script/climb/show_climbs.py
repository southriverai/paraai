from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy import stats

from paraai.repository.repository_climb import RepositoryClimb
from paraai.repository.repository_tracklog_body import RepositoryTracklogBody


def climb_vertical_speed_m_s(climb) -> float | None:
    """Mean vertical speed of climb in m/s, or None if invalid."""
    ts = climb.list_timestamp_utc
    alts = climb.list_altitude_m
    if len(ts) < 2 or len(alts) < 2:
        return None
    dt = ts[-1] - ts[0]
    if dt <= 0:
        return None
    return (alts[-1] - alts[0]) / dt


def climb_duration_s(climb) -> float | None:
    """Duration of climb in seconds, or None if invalid."""
    ts = climb.list_timestamp_utc
    if len(ts) < 2:
        return None
    dt = ts[-1] - ts[0]
    return float(dt) if dt > 0 else None


def main():
    path_dir_database = Path("data", "database_sqlite")
    RepositoryTracklogBody.initialize_sqlite(path_dir_database)
    repo_climb = RepositoryClimb.initialize_sqlite(path_dir_database)

    climbs = repo_climb.get_all()

    # Europe bounding box - filter early, no data outside Europe is processed
    europe_lng_min, europe_lat_min = -10, 35
    europe_lng_max, europe_lat_max = 40, 71
    climbs = [
        c
        for c in climbs
        if europe_lng_min <= c.lng <= europe_lng_max and europe_lat_min <= c.lat <= europe_lat_max
    ]
    print(f"Climbs in Europe: {len(climbs)}")

    # Seasons: winter months (Dec, Jan, Feb) go to spring/autumn
    # Spring: Jan, Feb, Mar, Apr, May | Summer: Jun, Jul, Aug | Autumn: Sep, Oct, Nov, Dec
    SPRING_MONTHS = {1, 2, 3, 4, 5}
    SUMMER_MONTHS = {6, 7, 8}
    AUTUMN_MONTHS = {9, 10, 11, 12}

    spring_speeds: list[float] = []
    summer_speeds: list[float] = []
    autumn_speeds: list[float] = []

    for c in climbs:
        v = climb_vertical_speed_m_s(c)
        if v is None:
            continue
        if not c.list_timestamp_utc:
            continue
        month = datetime.fromtimestamp(c.list_timestamp_utc[0], tz=timezone.utc).month
        if month in SPRING_MONTHS:
            spring_speeds.append(v)
        elif month in SUMMER_MONTHS:
            summer_speeds.append(v)
        elif month in AUTUMN_MONTHS:
            autumn_speeds.append(v)

    # Shared x grid for PDF evaluation
    all_speeds = spring_speeds + summer_speeds + autumn_speeds
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
        if v is None or not c.list_timestamp_utc:
            continue
        dt = datetime.fromtimestamp(c.list_timestamp_utc[0], tz=timezone.utc)
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
    ax_pdf.set_title("Vertical speed by season")
    ax_pdf.legend()
    ax_pdf.grid(True, alpha=0.3)

    # Shared axis limits for scatter subplots (5th–95th percentile to exclude outliers)
    all_means = [
        m for means, _ in season_points.values() for m in means
    ]
    all_stds = [
        s for _, stds in season_points.values() for s in stds
    ]
    scatter_x_min, scatter_x_max = (
        np.percentile(all_means, [5, 95]) if all_means else (0, 1)
    )
    scatter_y_min, scatter_y_max = (
        np.percentile(all_stds, [5, 95]) if all_stds else (0, 1)
    )

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

    # Second figure: day with most climbs
    climbs_by_date: dict[str, list] = defaultdict(list)
    for c in climbs:
        if not c.list_timestamp_utc:
            continue
        date_key = datetime.fromtimestamp(
            c.list_timestamp_utc[0], tz=timezone.utc
        ).strftime("%Y-%m-%d")
        climbs_by_date[date_key].append(c)

    busiest_date = max(climbs_by_date.keys(), key=lambda d: len(climbs_by_date[d]))
    busiest_climbs = climbs_by_date[busiest_date]
    busiest_speeds = []
    busiest_times_of_day = []
    for c in busiest_climbs:
        v = climb_vertical_speed_m_s(c)
        if v is None or not c.list_timestamp_utc:
            continue
        dt = datetime.fromtimestamp(c.list_timestamp_utc[0], tz=timezone.utc)
        # Decimal hours since midnight (0–24)
        time_of_day = dt.hour + dt.minute / 60 + dt.second / 3600
        busiest_speeds.append(v)
        busiest_times_of_day.append(time_of_day)

    fig2 = plt.figure(figsize=(10, 8))
    gs2 = GridSpec(2, 1, figure=fig2, height_ratios=[1, 1])

    ax_dist = fig2.add_subplot(gs2[0])
    if len(busiest_speeds) >= 2:
        x_min_b, x_max_b = np.min(busiest_speeds), np.max(busiest_speeds)
        x_grid_b = np.linspace(x_min_b, x_max_b, 200)
        kde = stats.gaussian_kde(busiest_speeds)
        pdf = kde(x_grid_b)
        ax_dist.plot(x_grid_b, pdf, color="steelblue", linewidth=2)
        ax_dist.fill_between(x_grid_b, pdf, alpha=0.3, color="steelblue")
    ax_dist.set_xlabel("Vertical speed (m/s)")
    ax_dist.set_ylabel("Density")
    ax_dist.set_title(f"Climb distribution on busiest day: {busiest_date} ({len(busiest_climbs)} climbs)")
    ax_dist.grid(True, alpha=0.3)

    ax_scatter = fig2.add_subplot(gs2[1])
    if len(busiest_speeds) >= 2:
        # x = strength (vertical speed), y = time of day
        speed_min, speed_max = np.min(busiest_speeds), np.max(busiest_speeds)
        time_min, time_max = np.min(busiest_times_of_day), np.max(busiest_times_of_day)
        xx = np.linspace(speed_min, speed_max, 100)
        yy = np.linspace(time_min, time_max, 100)
        xx_mesh, yy_mesh = np.meshgrid(xx, yy)
        pos = np.vstack([xx_mesh.ravel(), yy_mesh.ravel()])
        data_2d = np.vstack([busiest_speeds, busiest_times_of_day])
        kde_2d = stats.gaussian_kde(data_2d)
        z = kde_2d(pos).reshape(xx_mesh.shape)
        ax_scatter.contourf(xx_mesh, yy_mesh, z, levels=15, alpha=0.6, cmap="Blues")
        ax_scatter.contour(xx_mesh, yy_mesh, z, levels=8, colors="steelblue", linewidths=0.8)
        ax_scatter.scatter(busiest_speeds, busiest_times_of_day, alpha=0.8, s=20, edgecolors="black")
    ax_scatter.yaxis.set_major_formatter(
        FuncFormatter(lambda h, _: f"{int(h):02d}:{int((h % 1) * 60):02d}")
    )
    ax_scatter.set_xlabel("Vertical speed (m/s)")
    ax_scatter.set_ylabel("Time of day (UTC)")
    ax_scatter.set_title(f"Climb strength vs time of day on {busiest_date}")
    ax_scatter.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
