from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator

from paraai.model.trigger_point import TriggerPoint
from paraai.repository.repository_trigger_point import RepositoryTriggerPoint

METERS_PER_DEG_LAT = 111_000


def _latlon_to_local_m(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Convert lat/lon to local x (east), y (north) in meters from (lat0, lon0)."""
    x = (lon - lon0) * METERS_PER_DEG_LAT * math.cos(math.radians(lat0))
    y = (lat - lat0) * METERS_PER_DEG_LAT
    return x, y


class LiveThermal:
    """Thermal path computed from trigger point climbs with wind displacement."""

    def __init__(
        self,
        trigger_point: TriggerPoint,
        origin_lat: float,
        origin_lon: float,
        time_of_day_h: float,
        day_of_year: int,
        weather_strength_frac: float,
        wind_north_m_s: float,
        wind_east_m_s: float,
    ):
        self.trigger_point = trigger_point
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.wind_north_m_s = wind_north_m_s
        self.wind_east_m_s = wind_east_m_s

    def compute_path_points(self) -> list[tuple[float, float, float]]:
        """
        Compute path points in meters (x, y, z) relative to origin (origin_lat, origin_lon).
        Path starts at median of climb starts, ends at median of climb ends.
        Wind displacement applied at 5% of wind speed along the path.
        """
        climbs = self.trigger_point.climbs
        if not climbs:
            return []

        # Median of climb start and end positions
        start_lats = np.array([c.start_lat for c in climbs])
        start_lons = np.array([c.start_lon for c in climbs])
        start_alts = np.array([c.start_alt_m for c in climbs])
        end_lats = np.array([c.end_lat for c in climbs])
        end_lons = np.array([c.end_lon for c in climbs])
        end_alts = np.array([c.end_alt_m for c in climbs])

        med_start_lat = float(np.median(start_lats))
        med_start_lon = float(np.median(start_lons))
        med_start_alt = float(np.median(start_alts))
        med_end_lat = float(np.median(end_lats))
        med_end_lon = float(np.median(end_lons))
        med_end_alt = float(np.median(end_alts))

        # Convert to meters from origin
        x_start, y_start = _latlon_to_local_m(med_start_lat, med_start_lon, self.origin_lat, self.origin_lon)
        x_end, y_end = _latlon_to_local_m(med_end_lat, med_end_lon, self.origin_lat, self.origin_lon)

        # Average climb duration for wind displacement
        durations = np.array([c.end_timestamp_utc - c.start_timestamp_utc for c in climbs])
        avg_duration_s = float(np.median(durations))

        # Wind displacement: 5% of wind speed * time along path
        wind_disp_east = 0.05 * self.wind_east_m_s * avg_duration_s
        wind_disp_north = 0.05 * self.wind_north_m_s * avg_duration_s

        # Path points: linear from start to end, with wind displacement increasing from 0 to full
        alt_span = med_end_alt - med_start_alt
        n_points = max(2, int(abs(alt_span) / 25) + 1)
        alts = np.linspace(med_start_alt, med_end_alt, n_points)
        t = (alts - med_start_alt) / (med_end_alt - med_start_alt) if med_end_alt != med_start_alt else np.zeros_like(alts)

        x_path = x_start + t * (x_end - x_start) + t * wind_disp_east
        y_path = y_start + t * (y_end - y_start) + t * wind_disp_north

        return list(zip(x_path.tolist(), y_path.tolist(), alts.tolist()))


def show_live_thermals(
    trigger_points: list[TriggerPoint],
    time_of_day_h: float,
    day_of_year: int,
    weather_strength_frac: float,
    wind_north_m_s: float,
    wind_east_m_s: float,
) -> None:
    """Show all live thermals in a single chart."""
    if not trigger_points:
        return

    origin_lat = trigger_points[0].lat
    origin_lon = trigger_points[0].lon

    # Collect all climb start points for heightmap
    all_xs, all_ys, all_zs = [], [], []
    thermal_data: list[tuple[str, LiveThermal, list, list]] = []  # (name, live_thermal, path_points, circle_data)

    for trigger_point in trigger_points:
        if not trigger_point.climbs:
            continue
        live_thermal = LiveThermal(
            trigger_point,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            time_of_day_h=time_of_day_h,
            day_of_year=day_of_year,
            weather_strength_frac=weather_strength_frac,
            wind_north_m_s=wind_north_m_s,
            wind_east_m_s=wind_east_m_s,
        )
        path_points = live_thermal.compute_path_points()
        if not path_points:
            continue

        lat0, lon0 = origin_lat, origin_lon
        climbs = trigger_point.climbs
        segments: list[tuple[float, float, float, float, float, float]] = []
        for c in climbs:
            x1, y1 = _latlon_to_local_m(c.start_lat, c.start_lon, lat0, lon0)
            x2, y2 = _latlon_to_local_m(c.end_lat, c.end_lon, lat0, lon0)
            segments.append((x1, y1, c.start_alt_m, x2, y2, c.end_alt_m))

        path_x = np.array([p[0] for p in path_points])
        path_y = np.array([p[1] for p in path_points])
        path_z = np.array([p[2] for p in path_points])
        path_z_min, path_z_max = float(path_z.min()), float(path_z.max())
        z_levels = np.arange(path_z_min, path_z_max + 1, 50.0)
        circle_data: list[tuple[float, float, float, float]] = []

        for z in z_levels:
            if path_z.min() <= z <= path_z.max() and path_z[-1] != path_z[0]:
                order = np.argsort(path_z)
                z_sorted, x_sorted, y_sorted = path_z[order], path_x[order], path_y[order]
                cx = float(np.interp(z, z_sorted, x_sorted))
                cy = float(np.interp(z, z_sorted, y_sorted))
            elif path_z.size > 0:
                idx = np.argmin(np.abs(path_z - z))
                cx, cy = float(path_x[idx]), float(path_y[idx])
            else:
                cx, cy = 0.0, 0.0

            points_x, points_y = [], []
            for x1, y1, z1, x2, y2, z2 in segments:
                if z1 <= z <= z2 and z2 > z1:
                    t = (z - z1) / (z2 - z1)
                    points_x.append(x1 + t * (x2 - x1))
                    points_y.append(y1 + t * (y2 - y1))
            if len(points_x) >= 2:
                radii = np.sqrt((np.array(points_x) - cx) ** 2 + (np.array(points_y) - cy) ** 2)
                radius = float(np.percentile(radii, 90))
                circle_data.append((z, cx, cy, max(radius, 20)))
            elif len(points_x) == 1:
                r = np.sqrt((points_x[0] - cx) ** 2 + (points_y[0] - cy) ** 2)
                circle_data.append((z, cx, cy, max(r, 20)))
            elif path_z.min() <= z <= path_z.max():
                circle_data.append((z, cx, cy, 20))

        for c in climbs:
            x, y = _latlon_to_local_m(c.start_lat, c.start_lon, lat0, lon0)
            all_xs.append(x)
            all_ys.append(y)
            all_zs.append(c.start_alt_m)

        thermal_data.append((trigger_point.name, live_thermal, path_points, circle_data))

    if not thermal_data:
        return

    # Heightmap from all climbs
    xs = np.array(all_xs)
    ys = np.array(all_ys)
    zs = np.array(all_zs)
    extent = max(500, np.abs(xs).max() + 200, np.abs(ys).max() + 200)
    grid_res = 30
    xi = np.linspace(-extent, extent, grid_res)
    yi = np.linspace(-extent, extent, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    try:
        rbf = RBFInterpolator(np.column_stack([xs, ys]), zs, kernel="thin_plate_spline")
        Zi = rbf(np.column_stack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        Zi = np.clip(Zi, zs.min() - 50, zs.max() + 50)
    except Exception:
        Zi = np.full_like(Xi, zs.min())

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xi, Yi, Zi, alpha=0.4, cmap="terrain", shade=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(thermal_data)))
    theta = np.linspace(0, 2 * np.pi, 50)

    for i, (name, _live_thermal, path_points, circle_data) in enumerate(thermal_data):
        color = colors[i % len(colors)]
        px, py, pz = zip(*path_points)
        ax.plot(px, py, pz, color=color, linewidth=2, alpha=0.9, label=name)
        for z, cx, cy, r in circle_data:
            circle_x = cx + r * np.cos(theta)
            circle_y = cy + r * np.sin(theta)
            circle_z = np.full_like(theta, z)
            ax.plot(circle_x, circle_y, circle_z, color=color, alpha=0.5, linewidth=1)

    trigger_z = float(Zi[Zi.shape[0] // 2, Zi.shape[1] // 2])
    ax.scatter([0], [0], [trigger_z], c="r", s=100, marker="^", label="Origin")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(f"Live thermals (n={len(thermal_data)})")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    time_of_day_h = 12
    day_of_year = 180
    weather_strength_frac = 0.5
    wind_north_m_s = 5
    wind_east_m_s = 0
    repo_trigger_point = RepositoryTriggerPoint.initialize_sqlite(Path("data", "database_sqlite"))
    ids = repo_trigger_point.get_all_ids()
    trigger_points = [tp for tid in ids if (tp := repo_trigger_point.get(tid)) is not None]

    show_live_thermals(trigger_points, time_of_day_h, day_of_year, weather_strength_frac, wind_north_m_s, wind_east_m_s)
