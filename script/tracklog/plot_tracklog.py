import argparse
import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter

from paraai.model.tracklog import TracklogBody, parse_igc_bytes
from paraai.tools_tracklogbody import get_plot_data


async def plot_tracklog(tracklog_body: TracklogBody, short: bool = False):
    pd = get_plot_data(tracklog_body, smoothing_time_seconds=60.0)
    flight_time_s = pd.flight_time_s
    altitudes = pd.altitudes
    east_m = pd.east_m
    north_m = pd.north_m
    array_tracklog_si = pd.array_tracklog_si
    climb_regions = pd.climb_regions
    progress_regions = pd.progress_regions
    exploration_regions = pd.exploration_regions

    def _fmt_flight_time(s, _):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        return f"{h}h{m:02d}m" if h else f"{m}m"

    if short:
        fig, axes_dict = plt.subplot_mosaic(
            [["track", "altitude", "heatmap"],
             ["track", "turn", "heatmap"]],
            figsize=(12, 6),
            gridspec_kw={"width_ratios": [1, 1, 1]},
        )
        ax1 = axes_dict["track"]
        ax2 = axes_dict["altitude"]
        ax3 = axes_dict["turn"]
        ax4 = axes_dict["heatmap"]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 6))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # Plot track in meters (SI array)
    # SI array columns: [lat_m, lng_m, alt_m, timestamp]. Plot as (East, North) = (lng_m, lat_m)
    east_m = array_tracklog_si[:, 1]  # lng_m
    north_m = array_tracklog_si[:, 0]  # lat_m
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

    # Plot time vs altitude
    ax2.plot(flight_time_s, altitudes, color="steelblue", alpha=0.8)
    ax2.set_xlim(flight_time_s.min(), flight_time_s.max())
    ax2.set_ylim(altitudes.min(), altitudes.max())
    ax2.xaxis.set_major_formatter(FuncFormatter(_fmt_flight_time))
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_title("Altitude in meters")
    ax2.grid(True)

    # Mark climb regions (yellow)
    for i, (start_idx, end_idx) in enumerate(climb_regions):
        climb_tod = flight_time_s[start_idx : end_idx + 1]
        climb_alts = altitudes[start_idx : end_idx + 1]
        ax2.plot(climb_tod, climb_alts, color="yellow", linewidth=3, alpha=0.5, zorder=5, label="Climb" if i == 0 else None)
        climb_east_m = array_tracklog_si[start_idx : end_idx + 1, 1]
        climb_north_m = array_tracklog_si[start_idx : end_idx + 1, 0]
        ax1.plot(climb_east_m, climb_north_m, color="yellow", linewidth=3, alpha=0.5, zorder=5, label="Climb" if i == 0 else None)

    # Mark progress regions (cyan)
    for i, (start_idx, end_idx) in enumerate(progress_regions):
        progress_tod = flight_time_s[start_idx : end_idx + 1]
        progress_alts = altitudes[start_idx : end_idx + 1]
        ax2.plot(progress_tod, progress_alts, color="cyan", linewidth=3, alpha=0.5, zorder=5, label="Progress" if i == 0 else None)
        progress_east_m = array_tracklog_si[start_idx : end_idx + 1, 1]
        progress_north_m = array_tracklog_si[start_idx : end_idx + 1, 0]
        ax1.plot(progress_east_m, progress_north_m, color="cyan", linewidth=3, alpha=0.5, zorder=5, label="Progress" if i == 0 else None)

    # Mark exploration regions (red)
    for i, (start_idx, end_idx) in enumerate(exploration_regions):
        exp_tod = flight_time_s[start_idx : end_idx + 1]
        exp_alts = altitudes[start_idx : end_idx + 1]
        ax2.plot(exp_tod, exp_alts, color="red", linewidth=3, alpha=0.5, zorder=5, label="Exploration" if i == 0 else None)
        exp_east_m = array_tracklog_si[start_idx : end_idx + 1, 1]
        exp_north_m = array_tracklog_si[start_idx : end_idx + 1, 0]
        ax1.plot(exp_east_m, exp_north_m, color="red", linewidth=3, alpha=0.5, zorder=5, label="Exploration" if i == 0 else None)

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    # Plot turn rate over time
    ax3.plot(flight_time_s, pd.array_tracklog_turn, color="steelblue", alpha=0.8)

    for i, (start_idx, end_idx) in enumerate(climb_regions):
        ax3.plot(
            flight_time_s[start_idx : end_idx + 1],
            pd.array_tracklog_turn[start_idx : end_idx + 1],
            color="yellow", linewidth=3, alpha=0.5, zorder=5, label="Climb" if i == 0 else None,
        )
    for i, (start_idx, end_idx) in enumerate(progress_regions):
        ax3.plot(
            flight_time_s[start_idx : end_idx + 1],
            pd.array_tracklog_turn[start_idx : end_idx + 1],
            color="cyan", linewidth=3, alpha=0.5, zorder=5, label="Progress" if i == 0 else None,
        )
    for i, (start_idx, end_idx) in enumerate(exploration_regions):
        ax3.plot(
            flight_time_s[start_idx : end_idx + 1],
            pd.array_tracklog_turn[start_idx : end_idx + 1],
            color="red", linewidth=3, alpha=0.5, zorder=5, label="Exploration" if i == 0 else None,
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
    extent = [
        pd.heatmap_y_edges[0], pd.heatmap_y_edges[-1],  # x = climb rate
        pd.heatmap_x_edges[0], pd.heatmap_x_edges[-1],  # y = altitude
    ]
    ax4.imshow(
        pd.hist_normalized, origin="lower", aspect="auto", extent=extent,
        cmap="viridis", interpolation="bilinear", vmax=pd.heatmap_vmax,
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

    if not short:
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
        plt.colorbar(lc_vs, ax=ax5)

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
        plt.colorbar(lc_prog, ax=ax6, label="Vertical speed (m/s)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Plot tracklog from IGC file")
    parser.add_argument("igc_file", type=Path, nargs="?", help="IGC file path")
    parser.add_argument("--short", action="store_true", help="Show only 4 charts (omit vertical speed and 1m progress)")
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

    path_file_igc = args.igc_file or Path("C:\\project\\data\\glide-data\\tracklog_kozzion\\2025-10-17-XCT-JOO-05.igc")

    tracklog_header, tracklog_body = parse_igc_bytes(path_file_igc.name, path_file_igc.read_bytes())

    asyncio.run(plot_tracklog(tracklog_body, short=args.short))
