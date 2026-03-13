"""Vertical speed distribution per month (KDE and chi-squared fit)."""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from paraai.model.simple_climb import SimpleClimb
from paraai.repository.repository_simple_climb import RepositorySimpleClimb

DISTRIBUTION_SAMPLE_SIZE = 100_000
MIN_SPEEDS_FOR_KDE = 2
X_AXIS_MAX_M_S = 4.0

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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
        v = c.climb_strength_m_s
        if v > 0:
            month = datetime.fromtimestamp(c.start_timestamp_utc, tz=timezone.utc).month
            speeds_by_month[month].append(v)
    return dict(speeds_by_month)


def main() -> None:
    print(f"Sampling {DISTRIBUTION_SAMPLE_SIZE} climbs for per-month distributions...")
    speeds_by_month = asyncio.run(_sample_speeds_for_distribution())

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
                print(f"{MONTH_LABELS[month - 1]:3s}: mean={mean_val:.4f} m/s  chi2(df={df:.2f}, loc={loc:.2f}, scale={scale:.4f})  n={len(speeds)}")
            except Exception:
                print(f"{MONTH_LABELS[month - 1]:3s}: mean={mean_val:.4f} m/s  (fit failed)  n={len(speeds)}")
        else:
            print(f"{MONTH_LABELS[month - 1]:3s}: mean={mean_val:.4f} m/s  (insufficient data for fit)  n={len(speeds)}")
    print("-" * 65)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
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
        ax.set_title(f"{MONTH_LABELS[month - 1]} (n={len(speeds)})")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, X_AXIS_MAX_M_S)

    plt.suptitle(f"Vertical speed distribution per month (sample n={DISTRIBUTION_SAMPLE_SIZE}, x capped at {X_AXIS_MAX_M_S} m/s)", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
