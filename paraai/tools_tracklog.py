from pydantic import BaseModel

import numpy as np

from paraai.model.tracklog import TracklogBody

MAX_VERTICAL_SPEED_M_S = 10.0
NEAR_ZERO_VERTICAL_SPEED_M_S = 0.5
MIN_LANDED_DURATION_SECONDS = 30.0


def _kalman_filter_extreme_points(
    points: list[tuple[float, float, float, int]],
) -> tuple[list[tuple[float, float, float, int]], int]:
    """
    Remove points where vertical speed from estimated state to observation exceeds 10 m/s.
    Uses a simple 1D Kalman filter on altitude. When a point would imply |v| > 10 m/s,
    reject it (don't update state, remove from output).
    Returns (filtered_points, num_removed).
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

    kept: list[tuple[float, float, float, int]] = [points[0]]
    num_removed = 0

    for i in range(1, len(points)):
        lat, lng, alt, ts = points[i]
        dt = ts - ts0
        if dt <= 0:
            num_removed += 1
            continue

        # Gating: reject if vertical speed from last estimate to observation > 10 m/s
        v_obs = (alt - x[0]) / dt
        if abs(v_obs) > MAX_VERTICAL_SPEED_M_S:
            num_removed += 1
            continue

        # Predict
        F_dt = np.array([[1.0, dt], [0.0, 1.0]])
        x_pred = F_dt @ x
        P_pred = F_dt @ P @ F_dt.T + Q * dt

        # Update
        z = np.array([[alt]])
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + (K @ y).ravel()
        P = (np.eye(2) - K @ H) @ P_pred

        kept.append((lat, lng, alt, ts))
        ts0 = ts

    return kept, num_removed


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
    extreme_timepoints_removed: int
    extreme_timepoints_retained: int
    end_timepoints_removed: int
    time_consistency: int

    def __str__(self) -> str:
        return f"""CleanTracklogResult:
        extreme_timepoints_removed={self.extreme_timepoints_removed}
        extreme_timepoints_retained={self.extreme_timepoints_retained}
        end_timepoints_removed={self.end_timepoints_removed}
        time_consistency={self.time_consistency}"""

    def __repr__(self) -> str:
        return self.__str__()


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
        extreme_timepoints_removed,
    ) = _kalman_filter_extreme_points(clean_tracklog_body.points_lat_lng_alt_ts)
    extreme_timepoints_retained = len(clean_tracklog_body.points_lat_lng_alt_ts)

    # Step 2: remove data after landing (vertical speed near 0 for sustained period)
    points_before_crop = len(clean_tracklog_body.points_lat_lng_alt_ts)
    (
        clean_tracklog_body.points_lat_lng_alt_ts,
        end_timepoints_removed,
    ) = _remove_data_after_landing(clean_tracklog_body.points_lat_lng_alt_ts)

    clean_tracklog_results = CleanTracklogResult(
        extreme_timepoints_removed=extreme_timepoints_removed,
        extreme_timepoints_retained=extreme_timepoints_retained,
        end_timepoints_removed=end_timepoints_removed,
        time_consistency=0,
    )
    return clean_tracklog_body, clean_tracklog_results
