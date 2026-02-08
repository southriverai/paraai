import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from paraai.model.tracklog import TracklogBody


async def plot_tracklog(tracklog_body: TracklogBody):
    array_tracklog_body = tracklog_body.as_array()
    array_tracklog_si = tracklog_body.as_array_si()  # Get SI array (meters)
    array_tracklog_turn = tracklog_body.get_array_turn()
    array_tracklog_vertical_speed = tracklog_body.get_array_vertical_speed(smoothing_time_seconds=60.0)
    takeoff_lat, takeoff_lng, takeoff_alt, takeoff_at_timestamp = array_tracklog_body[0]
    landing_lat, landing_lng, landing_alt, landing_at_timestamp = array_tracklog_body[-1]
    fig, axes = plt.subplots(2, 3, figsize=(24, 8))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # Plot track in meters (SI array) colored by vertical speed (m/s)
    # SI array columns: [lat_m, lng_m, alt_m, timestamp]
    points_si = np.array([array_tracklog_si[:, 0], array_tracklog_si[:, 1]]).T.reshape(-1, 1, 2)
    segments_si = np.concatenate([points_si[:-1], points_si[1:]], axis=1)

    # Use vertical speed values for coloring segments
    # Use the value at the start of each segment
    vertical_speed_segments = array_tracklog_vertical_speed[:-1] if len(array_tracklog_vertical_speed) > 0 else np.zeros(len(segments_si))

    # Create line collection colored by vertical speed
    lc = LineCollection(segments_si, cmap="viridis", alpha=0.7, zorder=1)  # Use same palette as other plots, lower zorder
    lc.set_array(vertical_speed_segments)
    # Set color scale from 5th to 95th percentile to handle outliers
    vmin = np.percentile(vertical_speed_segments, 5) if len(vertical_speed_segments) > 0 else 0
    vmax = np.percentile(vertical_speed_segments, 95) if len(vertical_speed_segments) > 0 else 0
    lc.set_clim(vmin, vmax)
    line = ax1.add_collection(lc)

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

    ax1.set_xlim(array_tracklog_si[:, 0].min(), array_tracklog_si[:, 0].max())
    ax1.set_ylim(array_tracklog_si[:, 1].min(), array_tracklog_si[:, 1].max())

    # Add colorbar
    plt.colorbar(line, ax=ax1)
    # Landing position in meters
    landing_lat_m = array_tracklog_si[-1, 0]
    landing_lng_m = array_tracklog_si[-1, 1]
    ax1.scatter(
        landing_lat_m,
        landing_lng_m,
        color="red",
        marker="v",
        s=150,
        label="Landing",
        zorder=10,
        edgecolors="black",
        linewidths=2,
    )
    ax1.set_xlabel("Latitude (m)")
    ax1.set_ylabel("Longitude (m)")
    ax1.set_title("Track (meters)")
    ax1.grid(True)
    ax1.legend()

    # Plot time vs altitude colored by vertical speed (climb rate)
    time_points = np.array([array_tracklog_body[:, 3], array_tracklog_body[:, 2]]).T.reshape(-1, 1, 2)
    time_segments = np.concatenate([time_points[:-1], time_points[1:]], axis=1)

    # Use vertical speed values for coloring (one per segment)
    time_vertical_speed_values = (
        array_tracklog_vertical_speed[:-1] if len(array_tracklog_vertical_speed) > 0 else np.zeros(len(time_segments))
    )

    lc_time = LineCollection(time_segments, cmap="viridis", alpha=0.7)
    lc_time.set_array(time_vertical_speed_values)
    # Set color scale from 5th to 95th percentile to handle outliers
    vmin_time = np.percentile(time_vertical_speed_values, 5) if len(time_vertical_speed_values) > 0 else 0
    vmax_time = np.percentile(time_vertical_speed_values, 95) if len(time_vertical_speed_values) > 0 else 0
    lc_time.set_clim(vmin_time, vmax_time)
    line_time = ax2.add_collection(lc_time)
    ax2.set_xlim(array_tracklog_body[:, 3].min(), array_tracklog_body[:, 3].max())
    ax2.set_ylim(array_tracklog_body[:, 2].min(), array_tracklog_body[:, 2].max())

    ax2.set_xlabel("Time (timestamp)")
    ax2.set_ylabel("Altitude")
    ax2.set_title("Time vs Altitude (colored by climb rate)")
    ax2.grid(True)

    # Add colorbar
    plt.colorbar(line_time, ax=ax2)

    # Mark climbs that last more than 60 seconds
    MIN_CLIMB_DURATION_SECONDS = 60
    timestamps = array_tracklog_body[:, 3]
    altitudes = array_tracklog_body[:, 2]
    vertical_speeds = array_tracklog_vertical_speed

    # Identify climbs (vertical speed > 0)
    is_climbing = vertical_speeds > 0

    # Find continuous climb regions
    climb_regions = []
    in_climb = False
    climb_start_idx = 0

    for i in range(len(is_climbing)):
        if is_climbing[i] and not in_climb:
            # Start of a climb
            climb_start_idx = i
            in_climb = True
        elif not is_climbing[i] and in_climb:
            # End of a climb
            climb_duration = timestamps[i - 1] - timestamps[climb_start_idx]
            if climb_duration > MIN_CLIMB_DURATION_SECONDS:
                climb_regions.append((climb_start_idx, i - 1))
            in_climb = False

    # Handle case where climb continues to the end
    if in_climb:
        climb_duration = timestamps[-1] - timestamps[climb_start_idx]
        if climb_duration > MIN_CLIMB_DURATION_SECONDS:
            climb_regions.append((climb_start_idx, len(timestamps) - 1))

    # Mark long climbs on the plot
    for start_idx, end_idx in climb_regions:
        # Highlight the climb region on altitude plot (ax2)
        climb_times = timestamps[start_idx : end_idx + 1]
        climb_alts = altitudes[start_idx : end_idx + 1]
        ax2.plot(climb_times, climb_alts, color="yellow", linewidth=3, alpha=0.5, zorder=5)
        # Mark start and end points on altitude plot
        ax2.scatter(
            timestamps[start_idx],
            altitudes[start_idx],
            color="orange",
            marker="o",
            s=100,
            zorder=6,
            edgecolors="black",
            linewidths=1,
        )
        ax2.scatter(
            timestamps[end_idx],
            altitudes[end_idx],
            color="orange",
            marker="s",
            s=100,
            zorder=6,
            edgecolors="black",
            linewidths=1,
        )

        # Highlight the climb region on track plot (ax1) using SI coordinates
        climb_lat_m = array_tracklog_si[start_idx : end_idx + 1, 0]
        climb_lng_m = array_tracklog_si[start_idx : end_idx + 1, 1]
        ax1.plot(climb_lat_m, climb_lng_m, color="yellow", linewidth=3, alpha=0.5, zorder=5)
        # Mark start and end points on track plot
        ax1.scatter(
            array_tracklog_si[start_idx, 0],
            array_tracklog_si[start_idx, 1],
            color="orange",
            marker="o",
            s=100,
            zorder=6,
            edgecolors="black",
            linewidths=1,
        )
        ax1.scatter(
            array_tracklog_si[end_idx, 0],
            array_tracklog_si[end_idx, 1],
            color="orange",
            marker="s",
            s=100,
            zorder=6,
            edgecolors="black",
            linewidths=1,
        )

    # Plot turn scores over time colored by vertical speed (climb rate)
    turn_time_points = np.array([array_tracklog_body[:, 3], array_tracklog_turn]).T.reshape(-1, 1, 2)
    turn_time_segments = np.concatenate([turn_time_points[:-1], turn_time_points[1:]], axis=1)

    # Use vertical speed values for coloring turn plot segments
    turn_vertical_speed_values = (
        array_tracklog_vertical_speed[:-1] if len(array_tracklog_vertical_speed) > 0 else np.zeros(len(turn_time_segments))
    )

    lc_turn = LineCollection(turn_time_segments, cmap="viridis", alpha=0.7)
    lc_turn.set_array(turn_vertical_speed_values)
    # Set color scale from 5th to 95th percentile to handle outliers
    vmin_turn = np.percentile(turn_vertical_speed_values, 5) if len(turn_vertical_speed_values) > 0 else 0
    vmax_turn = np.percentile(turn_vertical_speed_values, 95) if len(turn_vertical_speed_values) > 0 else 0
    lc_turn.set_clim(vmin_turn, vmax_turn)
    line_turn = ax3.add_collection(lc_turn)

    ax3.set_xlim(array_tracklog_body[:, 3].min(), array_tracklog_body[:, 3].max())
    ax3.set_ylim(0, max(0.25, array_tracklog_turn.max()))  # Show up to 0.25 prominently
    ax3.set_xlabel("Time (timestamp)")
    ax3.set_ylabel("Turn Score (0-1)")
    ax3.set_title("Turn Scores Over Time (colored by climb rate)")
    ax3.grid(True)

    # Add colorbar
    plt.colorbar(line_turn, ax=ax3)

    # Plot climb rate vs altitude (scatter)
    altitudes = array_tracklog_body[:, 2]
    timestamps = array_tracklog_body[:, 3]

    # Calculate change in altitude and time
    altitude_diff = np.diff(altitudes)
    time_diff = np.diff(timestamps)

    # Calculate climb rate in m/s (avoid division by zero)
    climb_rate = np.where(time_diff != 0, altitude_diff / time_diff, 0)

    # Smooth climb rates using Gaussian kernel over 5-second interval
    # Calculate average time step to determine sigma in terms of data points
    avg_time_step = np.mean(time_diff[time_diff > 0])  # Average time between points
    window_seconds = 5.0
    # Sigma should be approximately window_seconds / (2 * avg_time_step) for a 5-second window
    # Using a rule of thumb: for Gaussian, ~3*sigma covers most of the window
    sigma_points = window_seconds / (2 * avg_time_step) if avg_time_step > 0 else 1.0

    # Apply Gaussian smoothing
    climb_rate_smoothed = gaussian_filter1d(climb_rate, sigma=sigma_points)

    # Use midpoints of altitudes for x-axis (or use the starting altitude of each segment)
    altitude_midpoints = altitudes[:-1]

    # Build heatmap only from segments that fall inside thermals (long climbs)
    thermal_segment_mask = np.zeros(len(altitude_midpoints), dtype=bool)
    for start_idx, end_idx in climb_regions:
        # Segment i connects point i to i+1; it's in this thermal if start_idx <= i and i < end_idx
        for i in range(start_idx, end_idx):
            if i < len(thermal_segment_mask):
                thermal_segment_mask[i] = True

    altitude_thermal = altitude_midpoints[thermal_segment_mask]
    climb_rate_thermal = climb_rate_smoothed[thermal_segment_mask]

    # Create 2D histogram (heatmap) using smoothed climb rates from thermals only
    altitude_bins = 50
    climb_rate_bins = 50

    if len(altitude_thermal) > 0 and len(climb_rate_thermal) > 0:
        hist, x_edges, y_edges = np.histogram2d(altitude_thermal, climb_rate_thermal, bins=[altitude_bins, climb_rate_bins])
    else:
        # No thermal data: empty histogram with same bin structure as full data
        hist, x_edges, y_edges = np.histogram2d(altitude_midpoints, climb_rate_smoothed, bins=[altitude_bins, climb_rate_bins])
        hist = np.zeros_like(hist)

    # Apply Gaussian convolution kernel to smooth the heatmap
    # Configurable smoothing parameter (sigma for Gaussian kernel)
    smoothing_sigma = 1.5  # Adjust this to control smoothing amount
    hist_smoothed = gaussian_filter(hist, sigma=smoothing_sigma)

    # Normalize per altitude (each row independently) to show distribution of climb rates at each altitude
    # Normalize each row by its maximum value
    row_maxima = hist_smoothed.max(axis=1, keepdims=True)
    row_maxima = np.where(row_maxima > 0, row_maxima, 1)  # Avoid division by zero
    hist_normalized = hist_smoothed / row_maxima

    # Calculate 75th percentile for color scale saturation (on normalized data)
    vmax = np.percentile(hist_normalized, 75) if hist_normalized.size > 0 else 1.0

    # Create heatmap using imshow with color scale saturated at 75th percentile
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    im = ax4.imshow(hist_normalized.T, origin="lower", aspect="auto", extent=extent, cmap="viridis", interpolation="bilinear", vmax=vmax)

    # Add colorbar
    plt.colorbar(im, ax=ax4)

    ax4.set_xlabel("Altitude (m)")
    ax4.set_ylabel("Climb Rate (m/s)")
    ax4.set_title("Climb Rate vs Altitude (Thermals only)")
    ax4.axhline(y=0, color="r", linestyle="--", alpha=0.5, linewidth=1)  # Zero line for reference

    # Draw extracted thermals on top of the heatmap (altitude vs climb rate trajectory per thermal)
    # Smooth each thermal's trajectory at 600 s scale after extraction so edges are not pulled down by non-climb data
    THERMAL_TRAJECTORY_SMOOTHING_SECONDS = 200.0
    MIN_POINTS_FOR_SMOOTHING = 2
    all_alt_thermal = []  # collect (alt, climb) from all smoothed thermals for mean series
    all_climb_thermal = []
    for start_idx, end_idx in climb_regions:
        alt_traj = altitude_midpoints[start_idx:end_idx].astype(np.float64)
        climb_traj = climb_rate_smoothed[start_idx:end_idx].astype(np.float64)
        if len(alt_traj) < MIN_POINTS_FOR_SMOOTHING or len(climb_traj) < MIN_POINTS_FOR_SMOOTHING:
            if len(alt_traj) > 0 and len(climb_traj) > 0:
                ax4.plot(alt_traj, climb_traj, color="blue", linewidth=2, alpha=0.9, zorder=10)
                ax4.scatter(alt_traj[0], climb_traj[0], color="orange", marker="o", s=80, zorder=11, edgecolors="black", linewidths=1)
                ax4.scatter(alt_traj[-1], climb_traj[-1], color="orange", marker="s", s=80, zorder=11, edgecolors="black", linewidths=1)
                all_alt_thermal.extend(alt_traj.tolist())
                all_climb_thermal.extend(climb_traj.tolist())
            continue
        # Time step within this thermal (seconds per segment)
        time_diffs_thermal = np.diff(timestamps[start_idx : end_idx + 1])
        avg_dt = float(np.mean(time_diffs_thermal)) if len(time_diffs_thermal) > 0 else 1.0
        if avg_dt <= 0:
            avg_dt = 1.0
        # Sigma in points so that ~3*sigma corresponds to THERMAL_TRAJECTORY_SMOOTHING_SECONDS
        sigma_points = THERMAL_TRAJECTORY_SMOOTHING_SECONDS / (3.0 * avg_dt)
        sigma_points = max(0.5, min(sigma_points, (end_idx - start_idx) / 2.0))  # avoid over-smoothing short thermals
        # Smooth only within the extracted trajectory (mode="nearest" uses boundary values, no external data)
        alt_traj_smooth = gaussian_filter1d(alt_traj, sigma=sigma_points, mode="nearest")
        climb_traj_smooth = gaussian_filter1d(climb_traj, sigma=sigma_points, mode="nearest")
        ax4.plot(alt_traj_smooth, climb_traj_smooth, color="blue", linewidth=2, alpha=0.9, zorder=10)
        ax4.scatter(alt_traj_smooth[0], climb_traj_smooth[0], color="orange", marker="o", s=80, zorder=11, edgecolors="black", linewidths=1)
        ax4.scatter(
            alt_traj_smooth[-1], climb_traj_smooth[-1], color="orange", marker="s", s=80, zorder=11, edgecolors="black", linewidths=1
        )
        all_alt_thermal.extend(alt_traj_smooth.tolist())
        all_climb_thermal.extend(climb_traj_smooth.tolist())

    # Mean climb rate at every altitude (uniform series, interpolate where no data)
    N_ALT_BINS_MEAN = 128  # uniform series for possible later smoothing
    if len(all_alt_thermal) > 0 and len(all_climb_thermal) > 0:
        all_alt_thermal = np.array(all_alt_thermal)
        all_climb_thermal = np.array(all_climb_thermal)
        alt_min, alt_max = all_alt_thermal.min(), all_alt_thermal.max()
        alt_edges = np.linspace(alt_min, alt_max, N_ALT_BINS_MEAN + 1)
        bin_idx = np.digitize(all_alt_thermal, alt_edges) - 1  # 0 .. N_ALT_BINS_MEAN-1, clip below/above
        bin_idx = np.clip(bin_idx, 0, N_ALT_BINS_MEAN - 1)
        alt_centers = (alt_edges[:-1] + alt_edges[1:]) / 2
        mean_climb = np.full(N_ALT_BINS_MEAN, np.nan)
        for b in range(N_ALT_BINS_MEAN):
            in_b = bin_idx == b
            if np.any(in_b):
                mean_climb[b] = np.mean(all_climb_thermal[in_b])
        # Interpolate missing bins to get a connected series
        valid = np.isfinite(mean_climb)
        if np.any(valid):
            mean_climb_filled = np.interp(
                np.arange(N_ALT_BINS_MEAN),
                np.where(valid)[0],
                mean_climb[valid],
            )
            # Smooth with 100 m wide Gaussian kernel (in altitude space)
            MEAN_CLIMB_SMOOTHING_ALTITUDE_M = 100.0
            bin_width_m = (alt_max - alt_min) / N_ALT_BINS_MEAN
            sigma_bins = MEAN_CLIMB_SMOOTHING_ALTITUDE_M / bin_width_m if bin_width_m > 0 else 1.0
            sigma_bins = max(0.5, sigma_bins)
            mean_climb_smooth = gaussian_filter1d(mean_climb_filled, sigma=sigma_bins, mode="nearest")
            ax4.plot(
                alt_centers,
                mean_climb_smooth,
                color="red",
                linestyle="--",
                linewidth=3,
                alpha=0.95,
                zorder=12,
            )

    # Plot smoothed vertical speed over time colored by vertical speed
    vs_time_points = np.array([timestamps, array_tracklog_vertical_speed]).T.reshape(-1, 1, 2)
    vs_time_segments = np.concatenate([vs_time_points[:-1], vs_time_points[1:]], axis=1)

    # Use vertical speed values for coloring segments
    vs_segment_values = array_tracklog_vertical_speed[:-1] if len(array_tracklog_vertical_speed) > 0 else np.zeros(len(vs_time_segments))

    lc_vs = LineCollection(vs_time_segments, cmap="viridis", alpha=0.7)
    lc_vs.set_array(vs_segment_values)
    # Set color scale from 5th to 95th percentile to handle outliers
    vmin_vs = np.percentile(vs_segment_values, 5) if len(vs_segment_values) > 0 else 0
    vmax_vs = np.percentile(vs_segment_values, 95) if len(vs_segment_values) > 0 else 0
    lc_vs.set_clim(vmin_vs, vmax_vs)
    line_vs = ax5.add_collection(lc_vs)

    ax5.set_xlim(timestamps.min(), timestamps.max())
    ax5.set_ylim(array_tracklog_vertical_speed.min(), array_tracklog_vertical_speed.max())
    ax5.set_xlabel("Time (timestamp)")
    ax5.set_ylabel("Vertical Speed (m/s)")
    ax5.set_title("Smoothed Vertical Speed Over Time")
    ax5.axhline(y=0, color="r", linestyle="--", alpha=0.5, linewidth=1)  # Zero line for reference
    ax5.grid(True)

    # Add colorbar
    plt.colorbar(line_vs, ax=ax5)

    # Hide the unused subplot (6th position)
    ax6.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pathlib import Path

    from paraai.repository.repository_tracklog_body import RepositoryTracklogBody
    from paraai.repository.repository_tracklog_header import RepositoryTracklogHeader

    path_dir_database = Path("data", "database_sqlite")
    repository_tracklog_body = RepositoryTracklogBody.initialize_sqlite(path_dir_database)

    RepositoryTracklogHeader.initialize_sqlite(path_dir_database)
    repository_tracklog_body = RepositoryTracklogBody.initialize_sqlite(path_dir_database)
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
    tracklog_body = asyncio.run(repository_tracklog_body.asample(1))[0]
    print(tracklog_body.tracklog_id)

    asyncio.run(plot_tracklog(tracklog_body.tracklog_id))
