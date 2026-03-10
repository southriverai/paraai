import argparse
import asyncio
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter

from paraai.model.tracklog import TracklogBody, parse_igc_bytes
from paraai.tools_tracklogbody import get_plot_data

METERS_PER_DEG = 111_319.9059


def _trigger_points_in_track_coords(
    tracklog_body: TracklogBody,
    track_xlim: tuple[float, float],
    track_ylim: tuple[float, float],
    pad_m: float = 500,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (east_m, north_m, names) for trigger points in the track region. Empty arrays if unavailable."""
    try:
        from paraai.repository.repository_trigger_point import RepositoryTriggerPoint
    except ImportError:
        return np.array([]), np.array([]), []

    try:
        try:
            repo = RepositoryTriggerPoint.get_instance()
        except ValueError:
            repo = RepositoryTriggerPoint.initialize_sqlite(Path("data", "database_sqlite"))
        trigger_points = repo.get_all()
    except Exception:
        return np.array([]), np.array([]), []

    if not trigger_points:
        return np.array([]), np.array([]), []

    arr = tracklog_body.as_array()
    takeoff_lat, takeoff_lon = float(arr[0, 0]), float(arr[0, 1])
    cos_lat = math.cos(math.radians(takeoff_lat))

    east_m = np.array([(tp.lon - takeoff_lon) * METERS_PER_DEG * cos_lat for tp in trigger_points])
    north_m = np.array([(tp.lat - takeoff_lat) * METERS_PER_DEG for tp in trigger_points])

    # Filter to region: track bounds + padding
    in_region = (
        (east_m >= track_xlim[0] - pad_m)
        & (east_m <= track_xlim[1] + pad_m)
        & (north_m >= track_ylim[0] - pad_m)
        & (north_m <= track_ylim[1] + pad_m)
    )
    east_m = east_m[in_region]
    north_m = north_m[in_region]
    names = [tp.name for tp, keep in zip(trigger_points, in_region) if keep]
    return east_m, north_m, names


def _terrain_data_for_track(
    tracklog_body: TracklogBody,
    track_xlim: tuple[float, float],
    track_ylim: tuple[float, float],
    array_tracklog_si: np.ndarray,
    grid_res: int = 160,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load terrain and return (elev_grid_2d, elev_along_track_1d).
    elev_grid is for the track chart; elev_along_track matches track points for the altitude chart.
    Returns (None, None) on failure.
    """
    try:
        from paraai.tools_terrain import load_terrain
    except ImportError:
        return None, None

    try:
        arr = tracklog_body.as_array()
        takeoff_lat, takeoff_lon = float(arr[0, 0]), float(arr[0, 1])
        cos_lat = math.cos(math.radians(takeoff_lat))

        # Bbox in lat/lon with padding
        pad_m = 500
        north_min_m, north_max_m = track_ylim[0] - pad_m, track_ylim[1] + pad_m
        east_min_m, east_max_m = track_xlim[0] - pad_m, track_xlim[1] + pad_m
        lat_min = takeoff_lat + north_min_m / METERS_PER_DEG
        lat_max = takeoff_lat + north_max_m / METERS_PER_DEG
        lon_min = takeoff_lon + east_min_m / (METERS_PER_DEG * cos_lat)
        lon_max = takeoff_lon + east_max_m / (METERS_PER_DEG * cos_lat)

        terrain = load_terrain(
            lon_min,
            lat_min,
            lon_max,
            lat_max,
            dem_resolution=90,
            cache_dir=Path("data", "terrain"),
        )
        elev = terrain["elevation"]
        transform = terrain["transform"]
        h, w = elev.shape

        # Grid in track coords (higher resolution for first chart)
        xi = np.linspace(track_xlim[0], track_xlim[1], grid_res)
        yi = np.linspace(track_ylim[0], track_ylim[1], grid_res)
        Xi, Yi = np.meshgrid(xi, yi)

        # Convert grid to lat/lon
        lats_grid = takeoff_lat + Yi / METERS_PER_DEG
        lons_grid = takeoff_lon + Xi / (METERS_PER_DEG * cos_lat)

        # Sample elevation at each (lon, lat) for grid
        rows, cols = rasterio.transform.rowcol(transform, lons_grid, lats_grid)
        rows = np.clip(np.asarray(rows), 0, h - 1)
        cols = np.clip(np.asarray(cols), 0, w - 1)
        elev_grid = elev[rows, cols].astype(np.float64)
        elev_grid = np.asarray(elev_grid).reshape(grid_res, grid_res)
        elev_grid[elev_grid < -500] = np.nan  # mask nodata

        # Terrain elevation along track (for altitude chart)
        north_m = array_tracklog_si[:, 0]
        east_m = array_tracklog_si[:, 1]
        lats_track = takeoff_lat + north_m / METERS_PER_DEG
        lons_track = takeoff_lon + east_m / (METERS_PER_DEG * cos_lat)
        rows_t, cols_t = rasterio.transform.rowcol(transform, lons_track, lats_track)
        rows_t = np.clip(np.asarray(rows_t), 0, h - 1)
        cols_t = np.clip(np.asarray(cols_t), 0, w - 1)
        elev_along = elev[rows_t, cols_t].astype(np.float64)
        elev_along[elev_along < -500] = np.nan
    except Exception:
        return None, None
    else:
        return elev_grid, elev_along


async def plot_tracklog(tracklog_body: TracklogBody, short: bool = False, main: bool = False):
    pd = get_plot_data(tracklog_body, smoothing_time_seconds=60.0)
    flight_time_s = pd.flight_time_s
    altitudes = pd.altitudes
    east_m = pd.east_m
    north_m = pd.north_m
    array_tracklog_si = pd.array_tracklog_si
    climb_regions = pd.climb_regions
    progress_regions = pd.progress_regions
    exploration_regions = pd.exploration_regions
    # print tracklog start lat long
    print(f"Tracklog start lat long: {tracklog_body.points_lat_lng_alt_ts[0]}")

    def _fmt_flight_time(s, _):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        return f"{h}h{m:02d}m" if h else f"{m}m"

    if main:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        ax2 = ax3 = ax4 = ax5 = ax6 = None
    elif short:
        fig, axes_dict = plt.subplot_mosaic(
            [["track", "altitude", "heatmap"], ["track", "turn", "heatmap"]],
            figsize=(12, 6),
            gridspec_kw={"width_ratios": [1, 1, 1]},
        )
        ax1 = axes_dict["track"]
        ax2 = axes_dict["altitude"]
        ax3 = axes_dict["turn"]
        ax4 = axes_dict["heatmap"]
        ax5 = ax6 = None
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 6))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # Plot track in meters (SI array)
    # SI array columns: [lat_m, lng_m, alt_m, timestamp]. Plot as (East, North) = (lng_m, lat_m)
    east_m = array_tracklog_si[:, 1]  # lng_m
    north_m = array_tracklog_si[:, 0]  # lat_m

    # Terrain elevation background (from load_terrain)
    elev_grid, elev_along_track = _terrain_data_for_track(tracklog_body, pd.track_xlim, pd.track_ylim, pd.array_tracklog_si)
    if elev_grid is not None:
        extent = [pd.track_xlim[0], pd.track_xlim[1], pd.track_ylim[0], pd.track_ylim[1]]
        terrain_cmap = "gray" if main else "terrain"
        ax1.imshow(
            elev_grid,
            extent=extent,
            origin="lower",
            cmap=terrain_cmap,
            alpha=0.5,
            zorder=0,
        )

    ax1.plot(east_m, north_m, color="steelblue", alpha=0.8, zorder=1)

    # Takeoff is at origin (0, 0) in SI coordinates - plot BEFORE setting limits to ensure it's included
    ax1.scatter(
        0,
        0,
        color="green",
        marker="^",
        s=150,
        label="Takeoff",
        zorder=10,
        edgecolors="black",
        linewidths=2,
    )

    ax1.set_xlim(*pd.track_xlim)
    ax1.set_ylim(*pd.track_ylim)
    ax1.set_aspect("equal", adjustable="box")

    # Landing position in meters (East, North)
    landing_east_m = array_tracklog_si[-1, 1]
    landing_north_m = array_tracklog_si[-1, 0]
    ax1.scatter(
        landing_east_m,
        landing_north_m,
        color="red",
        marker="v",
        s=150,
        label="Landing",
        zorder=10,
        edgecolors="black",
        linewidths=2,
    )
    ax1.set_xlabel("East (m)")
    ax1.set_ylabel("North (m)")
    ax1.set_title("Track (meters)")
    ax1.grid(True)

    # Trigger points (in region only); blue if climb started within 250m
    tp_east, tp_north, tp_names = _trigger_points_in_track_coords(
        tracklog_body, pd.track_xlim, pd.track_ylim
    )
    if len(tp_east) > 0:
        climb_starts_east = np.array([array_tracklog_si[start_idx, 1] for start_idx, _ in climb_regions])
        climb_starts_north = np.array([array_tracklog_si[start_idx, 0] for start_idx, _ in climb_regions])
        near_climb = np.zeros(len(tp_east), dtype=bool)
        for i in range(len(tp_east)):
            dists = np.sqrt((tp_east[i] - climb_starts_east) ** 2 + (tp_north[i] - climb_starts_north) ** 2)
            near_climb[i] = np.any(dists <= 250)

        orange_mask = ~near_climb
        blue_mask = near_climb
        if np.any(orange_mask):
            ax1.scatter(
                tp_east[orange_mask],
                tp_north[orange_mask],
                color="orange",
                marker="*",
                s=80,
                label="Trigger points",
                zorder=8,
                edgecolors="black",
                linewidths=0.5,
            )
        if np.any(blue_mask):
            ax1.scatter(
                tp_east[blue_mask],
                tp_north[blue_mask],
                color="blue",
                marker="*",
                s=80,
                label="Climb started nearby",
                zorder=8,
                edgecolors="black",
                linewidths=0.5,
            )

    if not main:
        # Plot time vs altitude
        if elev_along_track is not None:
            terr_valid = elev_along_track[~np.isnan(elev_along_track)]
            if len(terr_valid) > 0:
                terr_min = np.min(terr_valid)
                y_min = min(altitudes.min(), terr_min)
                elev_fill = np.where(np.isnan(elev_along_track), y_min, elev_along_track)
                ax2.fill_between(
                    flight_time_s,
                    y_min,
                    elev_fill,
                    color="gray",
                    alpha=0.4,
                    zorder=0,
                    label="Terrain",
                )
        ax2.plot(flight_time_s, altitudes, color="steelblue", alpha=0.8)
        ax2.set_xlim(flight_time_s.min(), flight_time_s.max())
        y_min_ax = altitudes.min()
        if elev_along_track is not None:
            terr_valid = elev_along_track[~np.isnan(elev_along_track)]
            if len(terr_valid) > 0:
                terr_min = np.min(terr_valid)
                if terr_min < y_min_ax:
                    y_min_ax = terr_min - 50  # margin below terrain
        ax2.set_ylim(y_min_ax, altitudes.max())
        ax2.xaxis.set_major_formatter(FuncFormatter(_fmt_flight_time))
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        ax2.set_title("Altitude in meters")
        ax2.grid(True)

    # Mark climb regions (yellow)
    for i, (start_idx, end_idx) in enumerate(climb_regions):
        climb_east_m = array_tracklog_si[start_idx : end_idx + 1, 1]
        climb_north_m = array_tracklog_si[start_idx : end_idx + 1, 0]
        ax1.plot(climb_east_m, climb_north_m, color="yellow", linewidth=3, alpha=0.5, zorder=5, label="Climb" if i == 0 else None)
        if ax2 is not None:
            climb_tod = flight_time_s[start_idx : end_idx + 1]
            climb_alts = altitudes[start_idx : end_idx + 1]
            ax2.plot(climb_tod, climb_alts, color="yellow", linewidth=3, alpha=0.5, zorder=5, label="Climb" if i == 0 else None)

    # Mark progress regions (cyan)
    for i, (start_idx, end_idx) in enumerate(progress_regions):
        progress_east_m = array_tracklog_si[start_idx : end_idx + 1, 1]
        progress_north_m = array_tracklog_si[start_idx : end_idx + 1, 0]
        ax1.plot(progress_east_m, progress_north_m, color="cyan", linewidth=3, alpha=0.5, zorder=5, label="Progress" if i == 0 else None)
        if ax2 is not None:
            progress_tod = flight_time_s[start_idx : end_idx + 1]
            progress_alts = altitudes[start_idx : end_idx + 1]
            ax2.plot(progress_tod, progress_alts, color="cyan", linewidth=3, alpha=0.5, zorder=5, label="Progress" if i == 0 else None)

    # Mark exploration regions (red)
    for i, (start_idx, end_idx) in enumerate(exploration_regions):
        exp_east_m = array_tracklog_si[start_idx : end_idx + 1, 1]
        exp_north_m = array_tracklog_si[start_idx : end_idx + 1, 0]
        ax1.plot(exp_east_m, exp_north_m, color="red", linewidth=3, alpha=0.5, zorder=5, label="Exploration" if i == 0 else None)
        if ax2 is not None:
            exp_tod = flight_time_s[start_idx : end_idx + 1]
            exp_alts = altitudes[start_idx : end_idx + 1]
            ax2.plot(exp_tod, exp_alts, color="red", linewidth=3, alpha=0.5, zorder=5, label="Exploration" if i == 0 else None)

    ax1.legend(loc="upper right")
    if ax2 is not None:
        ax2.legend(loc="upper right")

    # Plot turn rate over time
    if ax3 is not None:
        ax3.plot(flight_time_s, pd.array_tracklog_turn, color="steelblue", alpha=0.8)

        for i, (start_idx, end_idx) in enumerate(climb_regions):
            ax3.plot(
                flight_time_s[start_idx : end_idx + 1],
                pd.array_tracklog_turn[start_idx : end_idx + 1],
                color="yellow",
                linewidth=3,
                alpha=0.5,
                zorder=5,
                label="Climb" if i == 0 else None,
            )
        for i, (start_idx, end_idx) in enumerate(progress_regions):
            ax3.plot(
                flight_time_s[start_idx : end_idx + 1],
                pd.array_tracklog_turn[start_idx : end_idx + 1],
                color="cyan",
                linewidth=3,
                alpha=0.5,
                zorder=5,
                label="Progress" if i == 0 else None,
            )
        for i, (start_idx, end_idx) in enumerate(exploration_regions):
            ax3.plot(
                flight_time_s[start_idx : end_idx + 1],
                pd.array_tracklog_turn[start_idx : end_idx + 1],
                color="red",
                linewidth=3,
                alpha=0.5,
                zorder=5,
                label="Exploration" if i == 0 else None,
            )

        ax3.set_xlim(flight_time_s.min(), flight_time_s.max())
        turn_max = pd.array_tracklog_turn.max()
        ax3.set_ylim(0, turn_max * 1.1 if turn_max > 0 else 0.1)
        ax3.xaxis.set_major_formatter(FuncFormatter(_fmt_flight_time))
        ax3.set_xlabel("")
        ax3.set_ylabel("")
        ax3.set_title("Turn rate in radians per second")
        ax3.legend(loc="upper right")
        ax3.grid(True)

    # Plot climb rate vs altitude (heatmap rotated 90°: climb rate on x, altitude on y)
    if ax4 is not None:
        extent = [
            pd.heatmap_y_edges[0],
            pd.heatmap_y_edges[-1],  # x = climb rate
            pd.heatmap_x_edges[0],
            pd.heatmap_x_edges[-1],  # y = altitude
        ]
        ax4.imshow(
            pd.hist_normalized,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            interpolation="bilinear",
            vmax=pd.heatmap_vmax,
        )
        ax4.set_xlabel("Climb Rate (m/s)")
        ax4.set_ylabel("Altitude (m)")
        ax4.set_title("Climb Rate vs Altitude (Thermals only)")
        ax4.axvline(x=0, color="r", linestyle="--", alpha=0.5, linewidth=1)

        for i, (alt_traj, climb_traj) in enumerate(pd.thermal_trajectories):
            ax4.plot(climb_traj, alt_traj, color="blue", linewidth=2, alpha=0.9, zorder=10, label="Actual climbs" if i == 0 else None)

        if pd.mean_climb_alt_centers is not None and pd.mean_climb_smooth is not None:
            ax4.plot(
                pd.mean_climb_smooth,
                pd.mean_climb_alt_centers,
                color="red",
                linestyle="--",
                linewidth=3,
                alpha=0.95,
                zorder=12,
                label="Mean climb",
            )
        ax4.legend(loc="upper right")

    if not short and ax5 is not None:
        # Plot smoothed vertical speed over time colored by vertical speed
        vs = pd.array_tracklog_vertical_speed
        vs_time_points = np.column_stack([flight_time_s, vs]).reshape(-1, 1, 2)
        vs_time_segments = np.concatenate([vs_time_points[:-1], vs_time_points[1:]], axis=1)
        vs_segment_values = vs[:-1] if len(vs) > 0 else np.zeros(len(vs_time_segments))

        lc_vs = LineCollection(vs_time_segments, cmap="viridis", alpha=0.7)
        lc_vs.set_array(vs_segment_values)
        vmin_vs = np.percentile(vs_segment_values, 5) if len(vs_segment_values) > 0 else 0
        vmax_vs = np.percentile(vs_segment_values, 95) if len(vs_segment_values) > 0 else 0
        lc_vs.set_clim(vmin_vs, vmax_vs)
        ax5.add_collection(lc_vs)

        ax5.set_xlim(flight_time_s.min(), flight_time_s.max())
        ax5.set_ylim(vs.min(), vs.max())
        ax5.xaxis.set_major_formatter(FuncFormatter(_fmt_flight_time))
        ax5.set_xlabel("")
        ax5.set_ylabel("Vertical Speed (m/s)")
        ax5.set_title("Smoothed Vertical Speed Over Time")
        ax5.axhline(y=0, color="r", linestyle="--", alpha=0.5, linewidth=1)
        ax5.grid(True)

        # Chart 6: 1m progress rate
        progress_1m = pd.progress_1m
        prog_points = np.column_stack([flight_time_s, progress_1m]).reshape(-1, 1, 2)
        prog_segments = np.concatenate([prog_points[:-1], prog_points[1:]], axis=1)
        prog_vs = pd.array_tracklog_vertical_speed[:-1] if len(pd.array_tracklog_vertical_speed) > 0 else np.zeros(len(prog_segments))

        lc_prog = LineCollection(prog_segments, cmap="viridis", alpha=0.7)
        lc_prog.set_array(prog_vs)
        vmin_p = np.percentile(prog_vs, 5) if len(prog_vs) > 0 else 0
        vmax_p = np.percentile(prog_vs, 95) if len(prog_vs) > 0 else 0
        lc_prog.set_clim(vmin_p, vmax_p)
        ax6.add_collection(lc_prog)
        ax6.set_xlim(flight_time_s.min(), flight_time_s.max())
        y_max = np.max(progress_1m) * 1.1 if np.any(progress_1m > 0) else 100
        ax6.set_ylim(0, max(50, y_max))
        ax6.xaxis.set_major_formatter(FuncFormatter(_fmt_flight_time))
        ax6.set_xlabel("")
        ax6.set_ylabel("1m progress (m)")
        ax6.set_title("Distance from position 60 s ago")
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Plot tracklog from IGC file")
    parser.add_argument("igc_file", type=Path, nargs="?", help="IGC file path")
    parser.add_argument("--short", action="store_true", help="Show only 4 charts (omit vertical speed and 1m progress)")
    parser.add_argument("--main", action="store_true", help="Show only the first chart (track) with grayscale terrain")
    args = parser.parse_args()

    # path_dir_database = Path("data", "database_sqlite")
    # repository_tracklog_body = RepositoryTracklogBody.initialize_sqlite(path_dir_database)

    # RepositoryTracklogHeader.initialize_sqlite(path_dir_database)
    # repository_tracklog_body = RepositoryTracklogBody.initialize_sqlite(path_dir_database)
    # 4b322303-b8bf-415a-85bf-882805a091ab
    # 20fded38-1d69-4a5f-8e1d-0bb643eb2d24
    # d9a3a4ef-b565-4681-acca-a37fbb608614
    # c11a7c89-4ab2-4566-8a4b-d6a814037d91
    # c9c0cb66-f7ca-4eea-b9fd-d81eb59dea38
    # c274557d-22f9-433c-a637-b9d1d6f7ac8b
    # 4b08db8e-4bfb-4220-89c1-cd1a483a7d5f really nice
    # 60da683d-0732-47fe-b84f-f5aeb4c83d78 # nice but corrupted
    # 220eb5e0-d151-48b7-8f42-c378a8b7d3c2  # nice but corrupted
    # 9a1d0ce7-05aa-489b-8c36-74cfabb3fe2c #corrupted at end fix TODO
    # tracklog_body = asyncio.run(repository_tracklog_body.asample(1))[0]
    # print(tracklog_body.tracklog_id)

    path_file_igc = args.igc_file or Path("data", "input", "tracklog", "2025-10-17-XCT-JOO-05.igc")

    tracklog_header, tracklog_body = parse_igc_bytes(path_file_igc.name, path_file_igc.read_bytes())

    asyncio.run(plot_tracklog(tracklog_body, short=args.short, main=args.main))
