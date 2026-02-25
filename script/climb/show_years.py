import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from paraai.model.simple_climb import SimpleClimb
from paraai.repository.repository_simple_climb import RepositorySimpleClimb

MIN_POINTS_FOR_SINE_FIT = 5

PATH_CLIMB_SPEEDS = Path("data", "climb_speeds.csv")
COLUMNS = ("timestamp_utc", "vertical_speed_m_s", "top_alt_m")


def climb_vertical_speed_m_s(climb: SimpleClimb) -> float | None:
    """Mean vertical speed of climb in m/s, or None if invalid."""
    duration_s = climb.end_timestamp_utc - climb.start_timestamp_utc
    alts = climb.end_alt_m - climb.start_alt_m
    if duration_s <= 0 or alts <= 0:
        return None
    return alts / duration_s


def _gather_climbs_from_database(path_out: Path) -> list[dict]:
    """Load climb speeds from database, write to file, return records."""
    path_dir_database = Path("data", "database_sqlite")
    repo = RepositorySimpleClimb.initialize_sqlite(path_dir_database)

    print("Scanning keys...")
    keys = list(repo.store.yield_keys())
    total = len(keys)
    print(f"Found {total} simple climbs")

    records: list[dict] = []
    batch_size = repo.BATCH_SIZE
    num_batches = (total + batch_size - 1) // batch_size

    for i in tqdm(range(0, total, batch_size), total=num_batches, desc="Gathering climb speeds"):
        batch = keys[i : i + batch_size]
        climbs = repo.store.mget(batch)
        for c in climbs:
            if c is None:
                continue
            v = climb_vertical_speed_m_s(c)
            if v is not None:
                records.append(
                    {
                        "timestamp_utc": c.start_timestamp_utc,
                        "vertical_speed_m_s": round(v, 6),
                        "top_alt_m": round(max(c.start_alt_m, c.end_alt_m), 2),
                    }
                )

    path_out.parent.mkdir(parents=True, exist_ok=True)
    with path_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} climb speeds to {path_out}")
    return records


def get_climbs(path_file: Path | None = None) -> list[dict]:
    """Return climb data (timestamp_utc, vertical_speed_m_s, top_alt_m). Load from file if it exists, else from database."""
    path = path_file or PATH_CLIMB_SPEEDS
    if path.exists():
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for r in rows:
                r["timestamp_utc"] = int(r["timestamp_utc"])
                r["vertical_speed_m_s"] = float(r["vertical_speed_m_s"])
                r["top_alt_m"] = float(r["top_alt_m"])
            return rows
    return _gather_climbs_from_database(path)


def _two_sine_model_doy(x: np.ndarray, c: float, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    """Sum of two sine waves over 365-day cycle: c + a1*sin(2πx/365) + b1*cos(2πx/365) + a2*sin(4πx/365) + b2*cos(4πx/365)."""
    t = 2 * np.pi * x / 365
    return c + a1 * np.sin(t) + b1 * np.cos(t) + a2 * np.sin(2 * t) + b2 * np.cos(2 * t)


def main() -> None:
    climbs = get_climbs()
    print(f"Loaded {len(climbs)} climb speeds")

    # Group by year, compute mean speed per year
    speeds_by_year: dict[int, list[float]] = defaultdict(list)
    for r in climbs:
        year = datetime.fromtimestamp(r["timestamp_utc"], tz=timezone.utc).year
        speeds_by_year[year].append(r["vertical_speed_m_s"])

    years = sorted(speeds_by_year.keys())
    means = [sum(speeds_by_year[y]) / len(speeds_by_year[y]) for y in years]

    # Group by day of year (0-364, Jan 1 = 0), leap year day 366 -> 364
    speeds_by_doy: dict[int, list[float]] = defaultdict(list)
    for r in climbs:
        dt = datetime.fromtimestamp(r["timestamp_utc"], tz=timezone.utc)
        doy = min(dt.timetuple().tm_yday, 365) - 1  # 0-indexed
        speeds_by_doy[doy].append(r["vertical_speed_m_s"])
    days = list(range(0, 365))
    doy_means = [sum(speeds_by_doy[d]) / len(speeds_by_doy[d]) if speeds_by_doy[d] else np.nan for d in days]
    doy_counts = [len(speeds_by_doy[d]) for d in days]

    # Figure 1: Mean climb speed per year
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    ax1.plot(years, means, color="steelblue", linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean climb speed (m/s)")
    ax1.set_title("Mean climb speed per year")
    ax1.set_xticks([y for y in years if y % 2 == 0])
    ax1.set_ylim(0, None)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Figure 2: Number of climbs per day of year; mean climb speed per day of year
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Number of climbs per day of year
    ax2a.bar(days, doy_counts, color="steelblue", alpha=0.7, width=1)
    ax2a.set_xlabel("Day of year (0-364)")
    ax2a.set_ylabel("Number of climbs")
    ax2a.set_title("Number of climbs per day of year (all years)")
    ax2a.grid(True, alpha=0.3, axis="y")

    # Plot 2: Mean climb speed per day of year with two-sine fit
    valid_doy_vals = [m for m in doy_means if not np.isnan(m)]
    doy_fill = np.nanmean(valid_doy_vals) if valid_doy_vals else 0
    doy_means_valid = np.nan_to_num(np.array(doy_means), nan=doy_fill)
    ax2b.fill_between(days, doy_means_valid, alpha=0.4, color="steelblue")
    ax2b.plot(days, doy_means_valid, color="steelblue", linewidth=0.8, label="Daily mean")
    ax2b.set_xlabel("Day of year (0-364)")
    ax2b.set_ylabel("Mean climb speed (m/s)")
    ax2b.set_title("Mean climb speed per day of year (all years)")
    ax2b.grid(True, alpha=0.3)

    days_arr = np.array(days, dtype=float)
    doy_arr = np.array(doy_means_valid, dtype=float)
    valid_mask = np.array(doy_counts) > 0
    if np.sum(valid_mask) >= MIN_POINTS_FOR_SINE_FIT:
        try:
            popt_doy, _ = curve_fit(
                _two_sine_model_doy,
                days_arr[valid_mask],
                doy_arr[valid_mask],
                p0=[float(np.mean(doy_arr[valid_mask])), 0, 0, 0, 0],
            )
            c_doy, a1_doy, b1_doy, a2_doy, b2_doy = popt_doy
            A1_doy = np.sqrt(a1_doy**2 + b1_doy**2)
            phi1_doy = np.arctan2(b1_doy, a1_doy)
            A2_doy = np.sqrt(a2_doy**2 + b2_doy**2)
            phi2_doy = np.arctan2(b2_doy, a2_doy)
            print("\nTwo-sine fit (day of year): mean = c + A1*sin(2πx/365 + φ1) + A2*sin(4πx/365 + φ2)")
            print("-" * 55)
            print(f"  c (offset)     = {c_doy:.4f} m/s")
            print(f"  A1 (ampl 1)    = {A1_doy:.4f} m/s   φ1 (phase 1) = {phi1_doy:.4f} rad")
            print(f"  A2 (ampl 2)    = {A2_doy:.4f} m/s   φ2 (phase 2) = {phi2_doy:.4f} rad")
            print(f"  (raw: a1={a1_doy:.4f} b1={b1_doy:.4f} a2={a2_doy:.4f} b2={b2_doy:.4f})")
            print("-" * 55)

            x_fit_doy = np.linspace(0, 364, 500)
            y_fit_doy = _two_sine_model_doy(x_fit_doy, *popt_doy)
            ax2b.plot(x_fit_doy, y_fit_doy, color="coral", linewidth=2, label="Two-sine fit")
        except Exception as e:
            print(f"\nTwo-sine fit (day of year) failed: {e}")
    ax2b.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
