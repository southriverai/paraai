import numpy as np
from pydantic import BaseModel

from paraai.model.climb import Climb
from paraai.model.tracklog import TracklogBody

MAX_VERTICAL_SPEED_M_S = 10.0
MIN_CLIMB_DURATION_SECONDS = 60
NEAR_ZERO_VERTICAL_SPEED_M_S = 0.5
MIN_LANDED_DURATION_SECONDS = 30.0


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


def get_climb_regions(
    tracklog_body: TracklogBody,
    smoothing_time_seconds: float = 60.0,
) -> list[tuple[int, int]]:
    """
    Detect climbs (continuous periods of positive vertical speed > MIN_CLIMB_DURATION_SECONDS).
    Returns list of (start_idx, end_idx) for each climb region.
    """
    if len(tracklog_body.points_lat_lng_alt_ts) < 2:
        return []

    arr = tracklog_body.as_array()
    if arr.ndim != 2:
        return []

    timestamps_rel = arr[:, 3]
    vertical_speeds = tracklog_body.get_array_vertical_speed(
        smoothing_time_seconds=smoothing_time_seconds
    )
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

    climb_regions = get_climb_regions(tracklog_body, smoothing_time_seconds)

    climbs: list[Climb] = []
    for climb_index, (start_idx, end_idx) in enumerate(climb_regions):
        lat = float(np.mean(lats[start_idx : end_idx + 1]))
        lng = float(np.mean(lngs[start_idx : end_idx + 1]))
        list_timestamp_utc = [
            takeoff_timestamp_utc + int(ts) for ts in timestamps_rel[start_idx : end_idx + 1]
        ]
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
