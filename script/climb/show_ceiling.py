"""Ceiling per flight (highest climb endpoint) by day of year. Caches to JSON, charts mean ceiling per day."""

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from tqdm import tqdm

from paraai.model.simple_climb import SimpleClimb
from paraai.repository.repository_simple_climb import RepositorySimpleClimb

CACHE_PATH = Path("data", "ceiling_cache.json")
PATH_DATABASE = Path("data", "database_sqlite")


def _gather_ceilings() -> list[dict]:
    """Load climbs from DB, group by tracklog_id, compute ceiling per flight."""
    repo = RepositorySimpleClimb.initialize_sqlite(PATH_DATABASE)
    print("Loading SimpleClimbs...")
    climbs = repo.get_all()
    print(f"Loaded {len(climbs)} climbs")

    by_tracklog: dict[str, list[SimpleClimb]] = defaultdict(list)
    for c in climbs:
        by_tracklog[c.tracklog_id].append(c)

    ceilings: list[dict] = []
    for _, flight_climbs in tqdm(by_tracklog.items(), desc="Computing ceilings"):
        ceiling_m = max(c.end_alt_m for c in flight_climbs)
        strengths = [c.climb_strength_m_s() for c in flight_climbs]
        valid_strengths = [s for s in strengths if s is not None]
        avg_climb_m_s = sum(valid_strengths) / len(valid_strengths) if valid_strengths else 0.0
        dt = datetime.fromtimestamp(flight_climbs[0].start_timestamp_utc, tz=timezone.utc)
        doy = min(dt.timetuple().tm_yday, 365) - 1  # 0-364

        ceilings.append({
            "day_of_year": doy,
            "ceiling_m": round(ceiling_m, 2),
            "avg_climb_m_s": round(avg_climb_m_s, 6),
        })

    return ceilings


def _load_cache() -> list[dict] | None:
    """Load ceilings from cache. Returns None if missing/invalid."""
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text())
    except json.JSONDecodeError:
        return None
    if data.get("version") != 1:
        return None
    return data.get("ceilings")


def _gaussian_plus_constant(x: np.ndarray, c: float, A: float, mu: float, sigma: float) -> np.ndarray:
    """Model: ceiling = c + A * exp(-0.5 * ((x - mu) / sigma)^2)."""
    return c + A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _save_cache(ceilings: list[dict]) -> None:
    """Save ceilings to cache."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({"version": 1, "ceilings": ceilings}))


def main() -> None:
    ceilings = _load_cache()
    if ceilings is None:
        ceilings = _gather_ceilings()
        _save_cache(ceilings)
        print(f"Cached {len(ceilings)} ceilings to {CACHE_PATH}")
    else:
        print(f"Loaded {len(ceilings)} ceilings from cache ({CACHE_PATH})")

    # Mean ceiling per month (12 calendar month bins)
    MONTH_ABBREV = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Day-of-year at midpoint of each month (non-leap year)
    MONTH_CENTERS = np.array([15, 44, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    by_month: dict[int, list[float]] = defaultdict(list)
    for c in ceilings:
        doy = c["day_of_year"]
        dt = datetime(2000, 1, 1) + timedelta(days=doy)
        month = dt.month  # 1-12
        by_month[month].append(c["ceiling_m"])
    mean_ceiling_doy = []
    p5_doy = []
    p95_doy = []
    for month in range(1, 13):
        vals = by_month[month]
        if vals:
            arr = np.array(vals)
            mean_ceiling_doy.append(np.mean(arr))
            p5_doy.append(np.percentile(arr, 5))
            p95_doy.append(np.percentile(arr, 95))
        else:
            mean_ceiling_doy.append(np.nan)
            p5_doy.append(np.nan)
            p95_doy.append(np.nan)

    # Mean ceiling vs climb strength (bins of 0.2 m/s from 0 to 3)
    STRENGTH_BIN_WIDTH = 0.2
    n_strength_bins = int(3.0 / STRENGTH_BIN_WIDTH)  # 15 bins
    by_strength_bin: dict[int, list[float]] = defaultdict(list)
    for c in ceilings:
        s = c["avg_climb_m_s"]
        if 0 <= s < 3:
            bin_idx = int(s / STRENGTH_BIN_WIDTH)
            if bin_idx >= n_strength_bins:
                bin_idx = n_strength_bins - 1
            by_strength_bin[bin_idx].append(c["ceiling_m"])
    bin_centers = np.arange(STRENGTH_BIN_WIDTH / 2, 3.0, STRENGTH_BIN_WIDTH)
    mean_ceiling_per_bin = []
    p5_strength = []
    p95_strength = []
    for i in range(n_strength_bins):
        vals = by_strength_bin[i]
        if vals:
            arr = np.array(vals)
            mean_ceiling_per_bin.append(np.mean(arr))
            p5_strength.append(np.percentile(arr, 5))
            p95_strength.append(np.percentile(arr, 95))
        else:
            mean_ceiling_per_bin.append(np.nan)
            p5_strength.append(np.nan)
            p95_strength.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Ceiling per month - line chart
    ax1.fill_between(MONTH_CENTERS, p5_doy, p95_doy, alpha=0.2, color="steelblue", label="5th–95th percentile")
    ax1.plot(MONTH_CENTERS, mean_ceiling_doy, color="steelblue", linewidth=2, marker="o", markersize=4, label="mean")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean ceiling (m AMSL)")
    ax1.set_title("Ceiling per month")
    ax1.set_xticks(MONTH_CENTERS)
    ax1.set_xticklabels(MONTH_ABBREV)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, None)
    ax1.set_xlim(0, 365)
    ax1.legend(loc="upper left")

    # Right: Mean ceiling by climb strength (0.2 m/s bins) - line chart + model fit
    ax2.fill_between(bin_centers, p5_strength, p95_strength, alpha=0.2, color="steelblue", label="5th–95th percentile")
    ax2.plot(bin_centers, mean_ceiling_per_bin, color="steelblue", linewidth=2, marker="o", markersize=4, label="mean")

    pgauss = None
    valid_mask = ~np.isnan(mean_ceiling_per_bin)
    x_smooth = np.linspace(0, 3, 200)
    if np.sum(valid_mask) >= 4:
        x_fit = bin_centers[valid_mask]
        y_fit = np.array(mean_ceiling_per_bin)[valid_mask]
        c0 = float(np.nanmin(mean_ceiling_per_bin))
        A0 = float(np.nanmax(mean_ceiling_per_bin) - c0)

        try:
            mu0 = float(bin_centers[np.nanargmax(mean_ceiling_per_bin)])
            sigma0 = 0.5
            pgauss, _ = curve_fit(
                _gaussian_plus_constant,
                x_fit,
                y_fit,
                p0=[c0, A0, mu0, sigma0],
                bounds=([0, 0, 0, 0.05], [np.inf, np.inf, 5, 2]),
            )
            c_g, A_g, mu_g, sigma_g = pgauss
            y_gauss = _gaussian_plus_constant(x_smooth, *pgauss)
            ax2.plot(x_smooth, y_gauss, color="green", linewidth=2, linestyle=":", label="Model fit")
            print("\nModel fit: ceiling = c + A * exp(-0.5*((x-μ)/σ)²)")
            print(f"  c = {c_g:.2f} m   A = {A_g:.2f}   μ = {mu_g:.2f}   σ = {sigma_g:.3f}")
        except Exception as e:
            print(f"\nModel fit failed: {e}")

    ax2.set_xlabel("Mean climb strength (m/s)")
    ax2.set_ylabel("Mean ceiling (m AMSL)")
    ax2.set_title("Ceiling by climb strength")
    ax2.set_xticks(bin_centers)
    ax2.set_xticklabels([f"{b:.1f}" for b in bin_centers])
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, None)
    ax2.set_xlim(-0.1, 3.1)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    # Second figure: ceiling distribution for 3 middle bins with gamma fit (mean from model)
    STRENGTH_BINS = [(0.6, 0.8), (1.0, 1.2), (1.6, 1.8)]

    def _ceilings_in_range(lo: float, hi: float) -> list[float]:
        return [c["ceiling_m"] for c in ceilings if lo <= c["avg_climb_m_s"] < hi]

    def _model_mean(strength: float) -> float:
        if pgauss is None:
            return 0.0
        return float(_gaussian_plus_constant(np.array([strength]), *pgauss)[0])

    fig2, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

    for ax, (lo, hi) in zip(axes, STRENGTH_BINS):
        vals = _ceilings_in_range(lo, hi)
        bin_center = (lo + hi) / 2
        model_mean = _model_mean(bin_center)
        if vals:
            arr = np.array(vals)
            ax.hist(vals, bins=40, color="steelblue", alpha=0.8, edgecolor="white", density=True)
            if len(vals) >= 2 and np.all(arr > 0) and model_mean > 0:
                try:
                    shape_fit, loc_fit, scale_fit = stats.gamma.fit(arr, floc=0)
                    scale_constrained = model_mean / shape_fit if shape_fit > 0 else scale_fit
                    x_gamma = np.linspace(0.01, max(max(vals), 1000), 200)
                    pdf_gamma = stats.gamma.pdf(x_gamma, shape_fit, loc=0, scale=scale_constrained)
                    ax.plot(x_gamma, pdf_gamma, color="coral", linewidth=2, linestyle="--", label="Gamma fit")
                    print(f"\nGamma fit {lo}-{hi} m/s: shape={shape_fit:.2f}  scale={scale_constrained:.4f}  (mean from model={model_mean:.2f})")
                except Exception:
                    pass
        ax.set_xlabel("Ceiling (m AMSL)")
        ax.set_ylabel("Density")
        ax.set_title(f"{lo}-{hi} m/s (n={len(vals)})")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, None)
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
