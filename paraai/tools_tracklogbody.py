from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from paraai.model.climb import Climb
from paraai.model.tracklog import TracklogBody

MAX_VERTICAL_SPEED_M_S = 10.0
MIN_CLIMB_DURATION_SECONDS = 60
MIN_PROGRESS_DURATION_SECONDS = 60
MIN_PROGRESS_RATE_M_PER_MIN = 300
NEAR_ZERO_VERTICAL_SPEED_M_S = 0.5
MIN_LANDED_DURATION_SECONDS = 30.0
MAX_EXPLORATION_BEFORE_PROGRESS_SECONDS = 120  # Merge short exploration into progress when between climb and progress


def _kalman_filter_extreme_points(
    points: list[tuple[float, float, float, int]],
) -> tuple[list[tuple[float, float, float, int]], int]:
    """
    Impute altitude for points where vertical speed from estimated state to observation
    exceeds 10 m/s. Lat and lng are kept as-is (more reliable); only altitude is imputed
    using the Kalman filter prediction.
    Returns (filtered_points, num_imputed).
    """
    if len(points) < 2:
        return points, 0

    # State: [alt, v_vertical], P: covariance 2x2
    lat0, lng0, alt0, ts0 = points[0]
    x = np.array([alt0, 0.0], dtype=np.float64)
    P = np.diag([100.0, 25.0])  # initial uncertainty
    H = np.array([[1.0, 0.0]])
    Q = np.diag([0.1, 0.5])  # process noise
    R = np.array([[50.0]])  # measurement noise (m^2)

    result: list[tuple[float, float, float, int]] = [points[0]]
    num_imputed = 0

    for i in range(1, len(points)):
        lat, lng, alt, ts = points[i]
        dt = ts - ts0
        if dt <= 0:
            # Skip duplicate/out-of-order timestamps
            continue

        # Predict
        F_dt = np.array([[1.0, dt], [0.0, 1.0]])
        x_pred = F_dt @ x
        P_pred = F_dt @ P @ F_dt.T + Q * dt

        # Gating: if vertical speed to observation > 10 m/s, impute altitude (keep lat, lng)
        v_obs = (alt - x[0]) / dt
        if abs(v_obs) > MAX_VERTICAL_SPEED_M_S:
            alt_imputed = float(x_pred[0])
            result.append((lat, lng, alt_imputed, ts))
            x = x_pred
            P = P_pred
            num_imputed += 1
        else:
            # Update with observation
            z = np.array([[alt]])
            y = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x = x_pred + (K @ y).ravel()
            P = (np.eye(2) - K @ H) @ P_pred
            result.append((lat, lng, alt, ts))

        ts0 = ts

    return result, num_imputed


def _remove_data_after_landing(
    points: list[tuple[float, float, float, int]],
) -> tuple[list[tuple[float, float, float, int]], int]:
    """
    Remove points after we detect landing: vertical speed near 0 for a sustained period.
    Returns (filtered_points, num_removed).
    """
    if len(points) < 2:
        return points, 0

    # Compute vertical speed for each segment
    n = len(points)
    run_start: int | None = None
    run_duration = 0.0

    for i in range(n - 1):
        _, _, alt_i, ts_i = points[i]
        _, _, alt_next, ts_next = points[i + 1]
        dt = ts_next - ts_i
        if dt <= 0:
            run_start = None
            run_duration = 0.0
            continue

        v = (alt_next - alt_i) / dt
        if abs(v) < NEAR_ZERO_VERTICAL_SPEED_M_S:
            if run_start is None:
                run_start = i
                run_duration = dt
            else:
                run_duration += dt

            if run_duration >= MIN_LANDED_DURATION_SECONDS:
                # Found sustained near-zero: keep points up to and including first "on ground" point
                keep_count = run_start + 2
                return points[:keep_count], n - keep_count
        else:
            run_start = None
            run_duration = 0.0

    return points, 0


class CleanTracklogResult(BaseModel):
    altitudes_imputed: int
    points_after_kalman: int
    points_removed_after_landing: int

    def __str__(self) -> str:
        return f"""CleanTracklogResult:
        altitudes_imputed={self.altitudes_imputed}
        points_after_kalman={self.points_after_kalman}
        points_removed_after_landing={self.points_removed_after_landing}"""

    def __repr__(self) -> str:
        return self.__str__()


def _get_climb_regions(
    tracklog_body: TracklogBody,
    smoothing_time_seconds: float = 60.0,
) -> list[tuple[int, int]]:
    """Internal: detect climb regions. Returns list of (start_idx, end_idx)."""
    if len(tracklog_body.points_lat_lng_alt_ts) < 2:
        return []

    arr = tracklog_body.as_array()
    if arr.ndim != 2:
        return []

    timestamps_rel = arr[:, 3]
    vertical_speeds = tracklog_body.get_array_vertical_speed(smoothing_time_seconds=smoothing_time_seconds)
    is_climbing = vertical_speeds > 0

    climb_regions: list[tuple[int, int]] = []
    in_climb = False
    climb_start_idx = 0

    for i in range(len(is_climbing)):
        if is_climbing[i] and not in_climb:
            climb_start_idx = i
            in_climb = True
        elif not is_climbing[i] and in_climb:
            climb_duration = int(timestamps_rel[i - 1]) - int(timestamps_rel[climb_start_idx])
            if climb_duration > MIN_CLIMB_DURATION_SECONDS:
                climb_regions.append((climb_start_idx, i - 1))
            in_climb = False

    if in_climb:
        climb_duration = int(timestamps_rel[-1]) - int(timestamps_rel[climb_start_idx])
        if climb_duration > MIN_CLIMB_DURATION_SECONDS:
            climb_regions.append((climb_start_idx, len(timestamps_rel) - 1))

    return climb_regions


def _get_progress_regions(
    tracklog_body: TracklogBody,
    climb_regions: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Internal: detect progress regions (punching out climbs). Returns list of (start_idx, end_idx)."""
    if len(tracklog_body.points_lat_lng_alt_ts) < 2:
        return []

    arr = tracklog_body.as_array()
    if arr.ndim != 2:
        return []

    timestamps_rel = arr[:, 3]
    progress_1m = get_array_progress_1m(tracklog_body)
    is_progress = progress_1m > MIN_PROGRESS_RATE_M_PER_MIN

    raw_runs: list[tuple[int, int]] = []
    in_progress = False
    progress_start_idx = 0

    for i in range(len(is_progress)):
        if is_progress[i] and not in_progress:
            progress_start_idx = i
            in_progress = True
        elif not is_progress[i] and in_progress:
            end_idx = i - 1
            duration = int(timestamps_rel[end_idx]) - int(timestamps_rel[progress_start_idx])
            if duration >= MIN_PROGRESS_DURATION_SECONDS:
                raw_runs.append((progress_start_idx, end_idx))
            in_progress = False

    if in_progress:
        end_idx = len(timestamps_rel) - 1
        duration = int(timestamps_rel[end_idx]) - int(timestamps_rel[progress_start_idx])
        if duration >= MIN_PROGRESS_DURATION_SECONDS:
            raw_runs.append((progress_start_idx, end_idx))

    result: list[tuple[int, int]] = []
    for s, e in raw_runs:
        overlapping_climbs = [(cs, ce) for cs, ce in climb_regions if s <= ce and cs <= e]
        overlapping_climbs.sort(key=lambda x: x[0])
        current_start = s
        for cs, ce in overlapping_climbs:
            if current_start < cs:
                seg_end = cs - 1
                duration = int(timestamps_rel[seg_end]) - int(timestamps_rel[current_start])
                if duration >= MIN_PROGRESS_DURATION_SECONDS:
                    result.append((current_start, seg_end))
            current_start = max(current_start, ce + 1)
        if current_start <= e:
            duration = int(timestamps_rel[e]) - int(timestamps_rel[current_start])
            if duration >= MIN_PROGRESS_DURATION_SECONDS:
                result.append((current_start, e))

    return result


def get_regions(
    tracklog_body: TracklogBody,
    smoothing_time_seconds: float = 60.0,
) -> list[tuple[float, float, str]]:
    """
    Detect climb, progress, and exploration regions.
    Returns list of (start_time_s, end_time_s, type_str) where type_str is
    "climb", "progress", or "exploration". Chronologically ordered.
    Times are flight time in seconds (relative to takeoff).
    Short exploration (< 2 min) between climb end and progress start is merged into progress.
    """
    regions = _get_regions_with_indices(tracklog_body, smoothing_time_seconds)
    if not regions:
        return []

    arr = tracklog_body.as_array()
    timestamps = arr[:, 3].astype(float)
    t0 = float(timestamps[0])
    return [(float(timestamps[s]) - t0, float(timestamps[e]) - t0, t) for s, e, t in regions]


def get_array_progress_1m(tracklog_body: TracklogBody) -> np.ndarray:
    """
    Distance in meters from current position to position 60 seconds ago.
    Returns array of same length as tracklog. First ~60 s are 0.
    """
    if len(tracklog_body.points_lat_lng_alt_ts) < 2:
        return np.zeros(len(tracklog_body.points_lat_lng_alt_ts))

    arr_si = tracklog_body.as_array_si()
    east_m = arr_si[:, 1]
    north_m = arr_si[:, 0]
    timestamps = arr_si[:, 3]

    n = len(timestamps)
    progress_1m = np.zeros(n)
    for i in range(n):
        t_target = float(timestamps[i]) - 60
        j = np.searchsorted(timestamps, t_target, side="right") - 1
        if j >= 0:
            de = east_m[i] - east_m[j]
            dn = north_m[i] - north_m[j]
            progress_1m[i] = np.sqrt(de * de + dn * dn)
    return progress_1m


def _get_regions_with_indices(
    tracklog_body: TracklogBody,
    smoothing_time_seconds: float = 60.0,
) -> list[tuple[int, int, str]]:
    """Internal: like get_regions but returns (start_idx, end_idx, type_str) for array slicing."""
    if len(tracklog_body.points_lat_lng_alt_ts) < 2:
        return []

    arr = tracklog_body.as_array()
    if arr.ndim != 2:
        return []

    timestamps = arr[:, 3].astype(float)
    n = len(timestamps)
    climb_regions = _get_climb_regions(tracklog_body, smoothing_time_seconds)
    progress_regions = _get_progress_regions(tracklog_body, climb_regions)

    events: list[tuple[int, int, str]] = []
    for s, e in climb_regions:
        events.append((s, e, "climb"))
    for s, e in progress_regions:
        events.append((s, e, "progress"))
    events.sort(key=lambda x: x[0])

    # Fill exploration in all gaps (everything not climb or progress)
    last_end = -1
    for s, e, _ in list(events):
        if s > last_end + 1:
            events.append((last_end + 1, s - 1, "exploration"))
        last_end = max(last_end, e)
    if last_end < n - 1:
        events.append((last_end + 1, n - 1, "exploration"))

    events.sort(key=lambda x: x[0])

    # Merge short exploration (< 2 min) between climb and progress into progress
    merged: list[tuple[int, int, str]] = []
    i = 0
    while i < len(events):
        s, e, t = events[i]
        if i + 2 < len(events):
            next_s, next_e, next_t = events[i + 1]
            next2_s, next2_e, next2_t = events[i + 2]
            if t == "climb" and next_t == "exploration" and next2_t == "progress":
                exp_duration_s = float(timestamps[next_e]) - float(timestamps[next_s])
                if exp_duration_s < MAX_EXPLORATION_BEFORE_PROGRESS_SECONDS:
                    # Merge exploration into progress; progress starts at exploration start
                    merged.append((s, e, "climb"))
                    merged.append((next_s, next2_e, "progress"))
                    i += 3
                    continue
        merged.append((s, e, t))
        i += 1

    merged.sort(key=lambda x: x[0])
    return merged


def extract_climbs(
    tracklog_body: TracklogBody,
    takeoff_timestamp_utc: int,
    smoothing_time_seconds: float = 60.0,
) -> list[Climb]:
    """
    Detect climbs and return as Climb objects with lat/lng centroid and time/altitude series in UTC.
    """
    if len(tracklog_body.points_lat_lng_alt_ts) < 2:
        return []

    arr = tracklog_body.as_array()
    if arr.ndim != 2:
        return []

    timestamps_rel = arr[:, 3]
    altitudes = arr[:, 2]
    lats = arr[:, 0]
    lngs = arr[:, 1]

    climb_regions = _get_climb_regions(tracklog_body, smoothing_time_seconds)

    climbs: list[Climb] = []
    for climb_index, (start_idx, end_idx) in enumerate(climb_regions):
        lat = float(np.mean(lats[start_idx : end_idx + 1]))
        lng = float(np.mean(lngs[start_idx : end_idx + 1]))
        list_timestamp_utc = [takeoff_timestamp_utc + int(ts) for ts in timestamps_rel[start_idx : end_idx + 1]]
        list_altitude_m = [float(a) for a in altitudes[start_idx : end_idx + 1]]
        climbs.append(
            Climb(
                tracklog_id=tracklog_body.tracklog_id,
                climb_index=climb_index,
                lat=lat,
                lng=lng,
                list_timestamp_utc=list_timestamp_utc,
                list_altitude_m=list_altitude_m,
            )
        )
    return climbs


@dataclass
class PlotData:
    """Precomputed series and data for tracklog plotting."""

    flight_time_s: np.ndarray
    altitudes: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    array_tracklog_si: np.ndarray
    array_tracklog_turn: np.ndarray
    array_tracklog_vertical_speed: np.ndarray
    climb_regions: list[tuple[int, int]]
    progress_regions: list[tuple[int, int]]
    exploration_regions: list[tuple[int, int]]
    progress_1m: np.ndarray
    track_xlim: tuple[float, float]
    track_ylim: tuple[float, float]
    altitude_midpoints: np.ndarray
    climb_rate_smoothed: np.ndarray
    altitude_thermal: np.ndarray
    climb_rate_thermal: np.ndarray
    hist_normalized: np.ndarray
    heatmap_x_edges: np.ndarray
    heatmap_y_edges: np.ndarray
    heatmap_vmax: float
    thermal_trajectories: list[tuple[np.ndarray, np.ndarray]]  # (alt, climb) per thermal
    mean_climb_alt_centers: np.ndarray | None
    mean_climb_smooth: np.ndarray | None


def get_plot_data(
    tracklog_body: TracklogBody,
    smoothing_time_seconds: float = 60.0,
    axes_aspect: float = 2.0,
) -> PlotData:
    """
    Compute all series and derived data needed for tracklog plotting.
    """
    if len(tracklog_body.points_lat_lng_alt_ts) < 2:
        raise ValueError("Tracklog must have at least 2 points for plotting")

    array_tracklog_body = tracklog_body.as_array()
    array_tracklog_si = tracklog_body.as_array_si()
    timestamps = array_tracklog_body[:, 3]
    altitudes = array_tracklog_body[:, 2]
    flight_time_s = timestamps.astype(float) - float(timestamps[0])

    east_m = array_tracklog_si[:, 1]
    north_m = array_tracklog_si[:, 0]
    array_tracklog_turn = tracklog_body.get_array_turn(smoothing_time_seconds=smoothing_time_seconds)
    array_tracklog_vertical_speed = tracklog_body.get_array_vertical_speed(smoothing_time_seconds=smoothing_time_seconds)
    regions = _get_regions_with_indices(tracklog_body, smoothing_time_seconds=smoothing_time_seconds)
    climb_regions = [(s, e) for s, e, t in regions if t == "climb"]
    progress_regions = [(s, e) for s, e, t in regions if t == "progress"]
    exploration_regions = [(s, e) for s, e, t in regions if t == "exploration"]
    progress_1m = get_array_progress_1m(tracklog_body)

    # Track limits (chart 1)
    east_min, east_max = east_m.min(), east_m.max()
    north_min, north_max = north_m.min(), north_m.max()
    east_center = (east_min + east_max) / 2
    north_center = (north_min + north_max) / 2
    base_east_half = max((east_max - east_min) / 2, 50) * 1.15
    base_north_half = max((north_max - north_min) / 2, 50) * 1.15
    if base_east_half / base_north_half >= axes_aspect:
        east_half = base_east_half
        north_half = east_half / axes_aspect
        if north_half < base_north_half:
            north_half = base_north_half
            east_half = north_half * axes_aspect
    else:
        north_half = base_north_half
        east_half = north_half * axes_aspect
        if east_half < base_east_half:
            east_half = base_east_half
            north_half = east_half / axes_aspect
    track_xlim = (east_center - east_half, east_center + east_half)
    track_ylim = (north_center - north_half, north_center + north_half)

    # Heatmap & mean climb: use only climb segments, climb-definition metric (vertical speed), no pre-smoothing
    # (heatmap does its own 2D binning + smoothing; pre-smoothing would leak non-climb data)
    altitude_midpoints = (altitudes[:-1] + altitudes[1:]) / 2.0
    vs_seg = (array_tracklog_vertical_speed[:-1] + array_tracklog_vertical_speed[1:]) / 2.0
    vs_seg = np.maximum(vs_seg, 0.0)

    altitude_thermal: list[float] = []
    climb_rate_thermal: list[float] = []
    for start_idx, end_idx in climb_regions:
        for i in range(start_idx, end_idx):
            if i < len(altitude_midpoints):
                altitude_thermal.append(float(altitude_midpoints[i]))
                climb_rate_thermal.append(float(vs_seg[i]))

    altitude_thermal = np.array(altitude_thermal) if altitude_thermal else np.array([], dtype=float)
    climb_rate_thermal = np.array(climb_rate_thermal) if climb_rate_thermal else np.array([], dtype=float)

    # Heatmap
    altitude_bins = 50
    climb_rate_bins = 50
    if len(altitude_thermal) > 0 and len(climb_rate_thermal) > 0:
        hist, x_edges, y_edges = np.histogram2d(altitude_thermal, climb_rate_thermal, bins=[altitude_bins, climb_rate_bins])
    else:
        hist, x_edges, y_edges = np.histogram2d(altitude_midpoints, vs_seg, bins=[altitude_bins, climb_rate_bins])
        hist = np.zeros_like(hist)
    hist_smoothed = gaussian_filter(hist, sigma=2.5)
    row_maxima = hist_smoothed.max(axis=1, keepdims=True)
    row_maxima = np.where(row_maxima > 0, row_maxima, 1)
    hist_normalized = hist_smoothed / row_maxima
    heatmap_vmax = float(np.percentile(hist_normalized, 75)) if hist_normalized.size > 0 else 1.0

    # Thermal trajectories: use the same vertical speed that defines climbs (guarantees climb_rate >= 0, no sink)
    THERMAL_TRAJECTORY_SMOOTHING_SECONDS = 40.0
    MIN_POINTS_FOR_SMOOTHING = 2
    thermal_trajectories: list[tuple[np.ndarray, np.ndarray]] = []
    all_alt_thermal: list[float] = []
    all_climb_thermal: list[float] = []
    for start_idx, end_idx in climb_regions:
        # Use climb-definition vertical speed: segment i gets mean of vs[i] and vs[i+1] (both > 0 in climb)
        vs_climb = array_tracklog_vertical_speed[start_idx : end_idx + 1].astype(np.float64)
        alts_climb = altitudes[start_idx : end_idx + 1].astype(np.float64)
        # Segment climb rate = average of vertical speed at segment endpoints (always >= 0 in climb)
        climb_rate_seg = (vs_climb[:-1] + vs_climb[1:]) / 2.0
        climb_rate_seg = np.maximum(climb_rate_seg, 0.0)  # floor at 0 (safety)
        alt_mid_climb = (alts_climb[:-1] + alts_climb[1:]) / 2.0

        if len(alt_mid_climb) < MIN_POINTS_FOR_SMOOTHING or len(climb_rate_seg) < MIN_POINTS_FOR_SMOOTHING:
            if len(alt_mid_climb) > 0 and len(climb_rate_seg) > 0:
                thermal_trajectories.append((alt_mid_climb, climb_rate_seg))
                all_alt_thermal.extend(alt_mid_climb.tolist())
                all_climb_thermal.extend(climb_rate_seg.tolist())
            continue

        ts_climb = timestamps[start_idx : end_idx + 1].astype(float)
        time_diff_climb = np.diff(ts_climb)
        avg_dt = float(np.mean(time_diff_climb[time_diff_climb > 0])) if np.any(time_diff_climb > 0) else 1.0
        if avg_dt <= 0:
            avg_dt = 1.0
        sigma_pts = THERMAL_TRAJECTORY_SMOOTHING_SECONDS / (3.0 * avg_dt)
        sigma_pts = max(0.5, min(sigma_pts, (end_idx - start_idx) / 2.0))
        alt_smooth = gaussian_filter1d(alt_mid_climb, sigma=sigma_pts, mode="nearest")
        climb_smooth = gaussian_filter1d(climb_rate_seg, sigma=sigma_pts, mode="nearest")
        climb_smooth = np.maximum(climb_smooth, 0.0)  # ensure no sink after smoothing
        thermal_trajectories.append((alt_smooth, climb_smooth))
        all_alt_thermal.extend(alt_smooth.tolist())
        all_climb_thermal.extend(climb_smooth.tolist())

    # Mean climb series
    mean_climb_alt_centers: np.ndarray | None = None
    mean_climb_smooth: np.ndarray | None = None
    N_ALT_BINS_MEAN = 128
    if len(all_alt_thermal) > 0 and len(all_climb_thermal) > 0:
        all_alt_arr = np.array(all_alt_thermal)
        all_climb_arr = np.array(all_climb_thermal)
        alt_min, alt_max = all_alt_arr.min(), all_alt_arr.max()
        alt_edges = np.linspace(alt_min, alt_max, N_ALT_BINS_MEAN + 1)
        bin_idx = np.digitize(all_alt_arr, alt_edges) - 1
        bin_idx = np.clip(bin_idx, 0, N_ALT_BINS_MEAN - 1)
        alt_centers = (alt_edges[:-1] + alt_edges[1:]) / 2
        mean_climb = np.full(N_ALT_BINS_MEAN, np.nan)
        for b in range(N_ALT_BINS_MEAN):
            in_b = bin_idx == b
            if np.any(in_b):
                mean_climb[b] = np.mean(all_climb_arr[in_b])
        valid = np.isfinite(mean_climb)
        if np.any(valid):
            mean_climb_filled = np.interp(np.arange(N_ALT_BINS_MEAN), np.where(valid)[0], mean_climb[valid])
            MEAN_CLIMB_SMOOTHING_ALTITUDE_M = 100.0
            bin_width_m = (alt_max - alt_min) / N_ALT_BINS_MEAN if N_ALT_BINS_MEAN > 0 else 1.0
            sigma_bins = max(0.5, MEAN_CLIMB_SMOOTHING_ALTITUDE_M / bin_width_m) if bin_width_m > 0 else 1.0
            mean_climb_smooth = gaussian_filter1d(mean_climb_filled, sigma=sigma_bins, mode="nearest")
            mean_climb_alt_centers = alt_centers

    return PlotData(
        flight_time_s=flight_time_s,
        altitudes=altitudes,
        east_m=east_m,
        north_m=north_m,
        array_tracklog_si=array_tracklog_si,
        array_tracklog_turn=array_tracklog_turn,
        array_tracklog_vertical_speed=array_tracklog_vertical_speed,
        climb_regions=climb_regions,
        progress_regions=progress_regions,
        exploration_regions=exploration_regions,
        progress_1m=progress_1m,
        track_xlim=track_xlim,
        track_ylim=track_ylim,
        altitude_midpoints=altitude_midpoints,
        climb_rate_smoothed=vs_seg,
        altitude_thermal=altitude_thermal,
        climb_rate_thermal=climb_rate_thermal,
        hist_normalized=hist_normalized,
        heatmap_x_edges=x_edges,
        heatmap_y_edges=y_edges,
        heatmap_vmax=heatmap_vmax,
        thermal_trajectories=thermal_trajectories,
        mean_climb_alt_centers=mean_climb_alt_centers,
        mean_climb_smooth=mean_climb_smooth,
    )


def get_altitude_min(tracklog_body: TracklogBody) -> float:
    return min(data[2] for data in tracklog_body.points_lat_lng_alt_ts)


def get_altitude_max(tracklog_body: TracklogBody) -> float:
    return max(data[2] for data in tracklog_body.points_lat_lng_alt_ts)


def clean_tracklog(tracklog_body: TracklogBody) -> tuple[TracklogBody, CleanTracklogResult]:
    clean_tracklog_body = tracklog_body.model_copy(deep=True)

    # Step 1: Kalman filter - remove points with vertical speed > 10 m/s (corrupted data)
    # Do this before cropping the end
    (
        clean_tracklog_body.points_lat_lng_alt_ts,
        altitudes_imputed,
    ) = _kalman_filter_extreme_points(clean_tracklog_body.points_lat_lng_alt_ts)
    points_after_kalman = len(clean_tracklog_body.points_lat_lng_alt_ts)

    # Step 2: remove data after landing (vertical speed near 0 for sustained period)
    (
        clean_tracklog_body.points_lat_lng_alt_ts,
        points_removed_after_landing,
    ) = _remove_data_after_landing(clean_tracklog_body.points_lat_lng_alt_ts)

    clean_tracklog_results = CleanTracklogResult(
        altitudes_imputed=altitudes_imputed,
        points_after_kalman=points_after_kalman,
        points_removed_after_landing=points_removed_after_landing,
    )
    return clean_tracklog_body, clean_tracklog_results
