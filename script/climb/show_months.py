import asyncio
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from tqdm import tqdm

from paraai.model.simple_climb import SimpleClimb
from paraai.repository.repository_simple_climb import RepositorySimpleClimb

DISTRIBUTION_SAMPLE_SIZE = 100_000
MIN_SPEEDS_FOR_KDE = 2
MIN_POINTS_FOR_SINE_FIT = 5
X_AXIS_MAX_M_S = 4.0

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


async def _sample_speeds_for_distribution() -> dict[int, list[float]]:
    """Sample climbs from DB and return speeds grouped by month."""
    path_dir_database = Path("data", "database_sqlite")
    try:
        repo = RepositorySimpleClimb.get_instance()
    except ValueError:
        repo = RepositorySimpleClimb.initialize_sqlite(path_dir_database)
    climbs = await repo.asample(DISTRIBUTION_SAMPLE_SIZE)
    speeds_by_month: dict[int, list[float]] = defaultdict(list)
    for c in climbs:
        v = climb_vertical_speed_m_s(c)
        if v is not None:
            month = datetime.fromtimestamp(c.start_timestamp_utc, tz=timezone.utc).month
            speeds_by_month[month].append(v)
    return dict(speeds_by_month)


def _two_sine_model(x: np.ndarray, c: float, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    """Sum of two sine waves over 12-month cycle: c + a1*sin(2πx/12) + b1*cos(2πx/12) + a2*sin(4πx/12) + b2*cos(4πx/12)."""
    t = 2 * np.pi * x / 12
    return c + a1 * np.sin(t) + b1 * np.cos(t) + a2 * np.sin(2 * t) + b2 * np.cos(2 * t)


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
    counts = [len(speeds_by_year[y]) for y in years]

    # Group by month, aggregate over all years
    speeds_by_month: dict[int, list[float]] = defaultdict(list)
    for r in climbs:
        month = datetime.fromtimestamp(r["timestamp_utc"], tz=timezone.utc).month
        speeds_by_month[month].append(r["vertical_speed_m_s"])

    months = list(range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_means = [sum(speeds_by_month[m]) / len(speeds_by_month[m]) if speeds_by_month[m] else 0 for m in months]
    month_counts = [len(speeds_by_month[m]) for m in months]

    # Group by day of year (1-365), ignore leap years (day 366 -> 365)
    speeds_by_doy: dict[int, list[float]] = defaultdict(list)
    for r in climbs:
        dt = datetime.fromtimestamp(r["timestamp_utc"], tz=timezone.utc)
        doy = min(dt.timetuple().tm_yday, 365)
        speeds_by_doy[doy].append(r["vertical_speed_m_s"])
    days = list(range(1, 366))
    doy_means = [sum(speeds_by_doy[d]) / len(speeds_by_doy[d]) if speeds_by_doy[d] else np.nan for d in days]
    doy_counts = [len(speeds_by_doy[d]) for d in days]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))

    # Plot 1: Mean climb speed per year
    bars1 = ax1.bar(years, means, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean climb speed (m/s)")
    ax1.set_title("Mean climb speed per year")
    ax1.set_xticks(years)
    for bar, count in zip(bars1, counts):
        ax1.annotate(f"n={count}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    # Plot 2: Mean climb speed per month (aggregated over all years) with two-sine fit
    bars2 = ax2.bar(months, month_means, color="steelblue", edgecolor="white", label="Monthly mean")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Mean climb speed (m/s)")
    ax2.set_title("Mean climb speed per month (all years)")
    ax2.set_xticks(months)
    ax2.set_xticklabels(month_labels)
    for bar, count in zip(bars2, month_counts):
        ax2.annotate(f"n={count}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    # Fit average strength per month with sum of two sine waves
    months_arr = np.array(months, dtype=float)
    means_arr = np.array(month_means, dtype=float)
    try:
        popt, _ = curve_fit(_two_sine_model, months_arr, means_arr, p0=[np.mean(means_arr), 0, 0, 0, 0])
        c, a1, b1, a2, b2 = popt
        A1 = np.sqrt(a1**2 + b1**2)
        phi1 = np.arctan2(b1, a1)
        A2 = np.sqrt(a2**2 + b2**2)
        phi2 = np.arctan2(b2, a2)
        print("\nTwo-sine fit: mean strength per month = c + A1*sin(2πx/12 + φ1) + A2*sin(4πx/12 + φ2)")
        print("-" * 55)
        print(f"  c (offset)     = {c:.4f} m/s")
        print(f"  A1 (ampl 1)    = {A1:.4f} m/s   φ1 (phase 1) = {phi1:.4f} rad")
        print(f"  A2 (ampl 2)    = {A2:.4f} m/s   φ2 (phase 2) = {phi2:.4f} rad")
        print(f"  (raw: a1={a1:.4f} b1={b1:.4f} a2={a2:.4f} b2={b2:.4f})")
        print("-" * 55)

        x_fit = np.linspace(1, 12, 200)
        y_fit = _two_sine_model(x_fit, *popt)
        ax2.plot(x_fit, y_fit, color="coral", linewidth=2, label="Two-sine fit")
    except Exception as e:
        print(f"\nTwo-sine fit failed: {e}")

    ax2.legend()

    # Plot 3: Mean climb speed per day of year (1-365, leap years ignored) with two-sine fit
    valid_doy_vals = [m for m in doy_means if not np.isnan(m)]
    doy_fill = np.nanmean(valid_doy_vals) if valid_doy_vals else 0
    doy_means_valid = np.nan_to_num(np.array(doy_means), nan=doy_fill)
    ax3.fill_between(days, doy_means_valid, alpha=0.4, color="steelblue")
    ax3.plot(days, doy_means_valid, color="steelblue", linewidth=0.8, label="Daily mean")
    ax3.set_xlabel("Day of year (1-365)")
    ax3.set_ylabel("Mean climb speed (m/s)")
    ax3.set_title("Mean climb speed per day of year (all years)")
    ax3.grid(True, alpha=0.3)

    # Fit day-of-year with two sine waves
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

            x_fit_doy = np.linspace(1, 365, 500)
            y_fit_doy = _two_sine_model_doy(x_fit_doy, *popt_doy)
            ax3.plot(x_fit_doy, y_fit_doy, color="coral", linewidth=2, label="Two-sine fit")
        except Exception as e:
            print(f"\nTwo-sine fit (day of year) failed: {e}")
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # Second figure: 3x4 KDE and chi2 fit per month (100k sample from DB), x-axis capped at 4 m/s
    print(f"Sampling {DISTRIBUTION_SAMPLE_SIZE} climbs for per-month distributions...")
    speeds_by_month = asyncio.run(_sample_speeds_for_distribution())

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x_grid = np.linspace(0, X_AXIS_MAX_M_S, 200)

    print("\nMonthly mean and chi-squared fit parameters (df, loc, scale):")
    print("-" * 65)
    for month in range(1, 13):
        speeds = speeds_by_month.get(month, [])
        speeds_arr = np.array(speeds)
        mean_val = np.mean(speeds_arr) if len(speeds_arr) else 0
        if len(speeds_arr) >= MIN_SPEEDS_FOR_KDE and np.all(speeds_arr > 0):
            try:
                df, loc, scale = stats.chi2.fit(speeds_arr, floc=0)
                print(f"{month_labels[month - 1]:3s}: mean={mean_val:.4f} m/s  chi2(df={df:.2f}, loc={loc:.2f}, scale={scale:.4f})  n={len(speeds)}")
            except Exception:
                print(f"{month_labels[month - 1]:3s}: mean={mean_val:.4f} m/s  (fit failed)  n={len(speeds)}")
        else:
            print(f"{month_labels[month - 1]:3s}: mean={mean_val:.4f} m/s  (insufficient data for fit)  n={len(speeds)}")
    print("-" * 65)

    fig2, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes_flat = axes.flatten()

    for i, month in enumerate(range(1, 13)):
        ax = axes_flat[i]
        speeds = speeds_by_month.get(month, [])
        speeds_arr = np.array(speeds)
        if len(speeds) >= MIN_SPEEDS_FOR_KDE:
            kde = stats.gaussian_kde(speeds)
            pdf_kde = kde(x_grid)
            ax.plot(x_grid, pdf_kde, color="steelblue", linewidth=2, label="KDE")
            ax.fill_between(x_grid, pdf_kde, alpha=0.2, color="steelblue")
        if len(speeds) >= MIN_SPEEDS_FOR_KDE and np.all(speeds_arr > 0):
            try:
                df, loc, scale = stats.chi2.fit(speeds_arr, floc=0)
                pdf_chi2 = stats.chi2.pdf(x_grid, df, loc=loc, scale=scale)
                ax.plot(x_grid, pdf_chi2, color="coral", linewidth=1.5, linestyle="--", label="χ² fit")
            except Exception:
                pass
        ax.set_xlabel("Vertical speed (m/s)")
        ax.set_ylabel("Density")
        ax.set_title(f"{month_labels[month - 1]} (n={len(speeds)})")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, X_AXIS_MAX_M_S)

    plt.suptitle(f"Vertical speed distribution per month (sample n={DISTRIBUTION_SAMPLE_SIZE}, x capped at {X_AXIS_MAX_M_S} m/s)", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
