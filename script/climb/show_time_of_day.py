"""Mean climb strength by month and hour (08:00-22:00) from SimpleClimb data."""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.tool_spacetime import utc_to_solar_hour

MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
HOUR_START = 6
HOUR_END = 22
MIN_DATAPOINTS = 2000
CACHE_PATH = Path("data", "climb_strength_by_hour_cache.json")


def _sine2d_model(x: np.ndarray, base: float, A_day: float, phi_day: float, A_hour: float, phi_hour: float) -> np.ndarray:
    """2D sine: base + A_day*sin(2π*day/365 + φ_day) + A_hour*sin(2π*hour/24 + φ_hour)."""
    day, hour = x[:, 0], x[:, 1]
    return base + A_day * np.sin(2 * np.pi * day / 365 + phi_day) + A_hour * np.sin(2 * np.pi * hour / 24 + phi_hour)


def _gaussian1d(hour: np.ndarray, base: float, A: float, mu: float, sigma: float) -> np.ndarray:
    """1D Gaussian: base + A * exp(-0.5 * ((hour - mu) / sigma)^2)."""
    return base + A * np.exp(-0.5 * ((hour - mu) / sigma) ** 2)


# Fitted parameters for climb_model (from 2D sine + per-month Gaussian fits)
_SINE_PARAMS = (0.5779, 0.1601, -1.2103, 0.3820, -1.6094)
_GAUSSIAN_PARAMS: dict[int, tuple[float, float, float, float]] = {
    1: (0.0, 39933.0, 12.81, 1.07),
    2: (0.0, 123839.0, 12.96, 1.24),
    3: (0.0, 390375.0, 12.86, 1.49),
    4: (0.0, 568925.0, 12.34, 1.73),
    5: (0.0, 587794.0, 12.21, 1.84),
    6: (0.0, 619674.0, 12.30, 1.88),
    7: (0.0, 708771.0, 12.45, 1.85),
    8: (0.0, 742609.0, 12.49, 1.78),
    9: (0.0, 410823.0, 12.52, 1.58),
    10: (0.0, 140004.0, 12.48, 1.34),
    11: (0.0, 39738.0, 12.65, 1.12),
    12: (0.0, 18550.0, 12.57, 1.07),
}


def climb_model(day_of_year: float, time_of_day: float) -> tuple[float, float]:
    """Given day of year (0-364, Jan 1 = 0) and time_of_day (0-1, 0=midnight, 0.5=solar noon, 1=midnight next day),
    return (climb_strength_m_s, climb_likelihood).

    Strength from 2D sine fit; likelihood from per-month Gaussian with linear interpolation (normalized 0-1).
    """
    hour_24 = time_of_day * 24.0  # convert to 0-24 for internal models
    strength = _sine2d_model(
        np.array([[day_of_year, hour_24]]),
        *_SINE_PARAMS,
    )[0]
    # Linear interpolation between months: month_frac 0 = Jan 1, 12 = end of Dec
    month_frac = day_of_year * 12 / 365
    if month_frac < 1:
        m_lo, m_hi = 1, 2
        t = month_frac
    elif month_frac < 11:
        m_lo = int(month_frac) + 1
        m_hi = m_lo + 1
        t = month_frac - int(month_frac)
    else:
        m_lo, m_hi = 11, 12
        t = month_frac - 11
    default = (0.0, 0.0, 12.55, 2.0)
    b_lo, A_lo, mu_lo, s_lo = _GAUSSIAN_PARAMS.get(m_lo, default)
    b_hi, A_hi, mu_hi, s_hi = _GAUSSIAN_PARAMS.get(m_hi, default)
    base = (1 - t) * b_lo + t * b_hi
    A = (1 - t) * A_lo + t * A_hi
    mu = (1 - t) * mu_lo + t * mu_hi
    sigma = (1 - t) * s_lo + t * s_hi
    likelihood_raw = base + A * np.exp(-0.5 * ((hour_24 - mu) / sigma) ** 2)
    max_likelihood = max(
        _GAUSSIAN_PARAMS[m][0] + _GAUSSIAN_PARAMS[m][1]
        for m in _GAUSSIAN_PARAMS
    ) if _GAUSSIAN_PARAMS else 1.0
    likelihood = likelihood_raw / max_likelihood if max_likelihood > 0 else 0.0
    return (float(strength), float(likelihood))


def _load_cache() -> dict[tuple[int, int], dict] | None:
    """Load aggregated curve data from cache. Returns None if missing/invalid."""
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text())
    except json.JSONDecodeError:
        return None
    if data.get("version") != 1 or data.get("hour_start") != HOUR_START or data.get("hour_end") != HOUR_END:
        return None
    try:
        out: dict[tuple[int, int], dict] = {}
        for k, v in data.get("by_month_hour", {}).items():
            m, h = map(int, k.split("_"))
            out[(m, h)] = {"mean": float(v["mean"]), "count": int(v["count"])}
    except (KeyError, ValueError):
        return None
    else:
        return out


def _save_cache(by_month_hour: dict[tuple[int, int], dict]) -> None:
    """Save aggregated curve data to cache."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"{m}_{h}": {"mean": v["mean"], "count": v["count"]} for (m, h), v in by_month_hour.items()}
    CACHE_PATH.write_text(json.dumps({"version": 1, "hour_start": HOUR_START, "hour_end": HOUR_END, "by_month_hour": flat}))


def main() -> None:
    by_month_hour: dict[tuple[int, int], dict] = {}

    cached = _load_cache()
    if cached:
        print(f"Loaded from cache ({CACHE_PATH})")
        by_month_hour = cached
    else:
        path_db = Path("data", "database_sqlite")
        repo = RepositorySimpleClimb.initialize_sqlite(path_db)
        print("Loading SimpleClimbs...")
        climbs = repo.get_all()
        print(f"Loaded {len(climbs)} climbs")

        raw: dict[tuple[int, int], list[float]] = defaultdict(list)
        for c in tqdm(climbs, desc="Processing"):
            strength = c.climb_strength_m_s()
            if strength is None:
                continue
            dt = datetime.fromtimestamp(c.start_timestamp_utc, tz=timezone.utc)
            solar_h = utc_to_solar_hour(dt, c.start_lon)
            hour_bin = int(solar_h)
            if not (HOUR_START <= hour_bin <= HOUR_END):
                continue
            month = dt.month
            raw[(month, hour_bin)].append(strength)

        for (m, h), vals in raw.items():
            by_month_hour[(m, h)] = {"mean": sum(vals) / len(vals), "count": len(vals)}
        _save_cache(by_month_hour)
        print(f"Cached to {CACHE_PATH}")

    # Print mean climb strength and data count (solar time; only show n>=100)
    print(f"\n=== Mean climb strength (m/s) by month and hour (solar time, n>={MIN_DATAPOINTS}) ===")
    for month in range(1, 13):
        print(f"\n{MONTH_NAMES[month - 1]}:")
        for hour in range(HOUR_START, HOUR_END + 1):
            cell = by_month_hour.get((month, hour), {"mean": 0.0, "count": 0})
            n = cell["count"]
            if n >= MIN_DATAPOINTS:
                print(f"  {hour:02d}:00  {cell['mean']:.4f}  (n={n})")
            else:
                print(f"  {hour:02d}:00  --  (n={n})")

    # 2D sine fit: day_of_year, hour -> strength
    # Use mid-month day as proxy for day_of_year: (month - 0.5) * 365/12
    fit_day, fit_hour, fit_strength = [], [], []
    for month in range(1, 13):
        day_mid = (month - 0.5) * 365 / 12
        for hour in range(HOUR_START, HOUR_END + 1):
            cell = by_month_hour.get((month, hour), {"mean": 0.0, "count": 0})
            if cell["count"] >= MIN_DATAPOINTS:
                fit_day.append(day_mid)
                fit_hour.append(float(hour))
                fit_strength.append(cell["mean"])
    fit_day = np.array(fit_day)
    fit_hour = np.array(fit_hour)
    fit_strength = np.array(fit_strength)
    fit_x = np.column_stack([fit_day, fit_hour])

    def _model_for_fit(x: np.ndarray, base: float, A_day: float, phi_day: float, A_hour: float, phi_hour: float) -> np.ndarray:
        return _sine2d_model(x, base, A_day, phi_day, A_hour, phi_hour)

    fit_params = None
    if len(fit_strength) >= 5:
        try:
            p0 = (0.5, 0.1, 0.0, 0.2, -np.pi / 2)
            bounds = ([0, 0, -np.pi, 0, -2 * np.pi], [2, 1, np.pi, 1, 2 * np.pi])
            popt, _ = curve_fit(_model_for_fit, fit_x, fit_strength, p0=p0, bounds=bounds)
            fit_params = popt
            base_f, A_d, phi_d, A_h, phi_h = popt
            print("\n=== 2D sine fit ===")
            print(f"  base:     {base_f:.4f}")
            print(f"  A_day:   {A_d:.4f}  φ_day:  {phi_d:.4f}")
            print(f"  A_hour:  {A_h:.4f}  φ_hour: {phi_h:.4f}")
        except Exception as e:
            print(f"\n  Fit failed: {e}")

    # 1D Gaussian fit for datapoint counts (per month), each with its own mu
    fit_params_count: dict[int, tuple[float, float, float, float]] = {}
    print("\n=== 1D Gaussian fit (datapoint count) per month ===")
    for month in range(1, 13):
        fit_h, fit_c = [], []
        for h in range(HOUR_START, HOUR_END + 1):
            cell = by_month_hour.get((month, h), {"mean": 0.0, "count": 0})
            if cell["count"] > 0:
                fit_h.append(float(h))
                fit_c.append(float(cell["count"]))
        if len(fit_h) >= 4:
            h_arr = np.array(fit_h)
            c_arr = np.array(fit_c)
            max_c = np.max(c_arr)
            try:
                p0 = (0.0, max_c, 12.5, 3.0)
                bounds = ([0, 0, HOUR_START, 0.5], [max_c * 0.5, max_c * 2, HOUR_END, 12])
                popt, _ = curve_fit(_gaussian1d, h_arr, c_arr, p0=p0, bounds=bounds)
                fit_params_count[month] = tuple(popt)
                base, A, mu, sigma = popt
                print(f"  {MONTH_NAMES[month - 1]:10s}: base={base:8.0f}  A={A:8.0f}  mu={mu:5.2f}  sigma={sigma:5.2f}")
            except Exception as e:
                print(f"  {MONTH_NAMES[month - 1]:10s}: fit failed ({e})")
        else:
            print(f"  {MONTH_NAMES[month - 1]:10s}: insufficient data (n={len(fit_h)})")

    # Single figure with 2 subplots
    hours = np.arange(HOUR_START, HOUR_END + 1)
    hours_fine = np.linspace(HOUR_START, HOUR_END, 100)
    offset = 0.4
    like_scale = 0.2  # max prob ~0.15 when sum=1

    # Precompute sum per month for proper distribution (sum over hours = 1)
    count_sum_by_month: dict[int, float] = {}
    for month in range(1, 13):
        total = sum(
            by_month_hour.get((month, h), {"mean": 0.0, "count": 0})["count"]
            for h in hours
        )
        count_sum_by_month[month] = total if total > 0 else 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for month in range(1, 13):
        base_y = offset * (12 - month)
        base_y_like = like_scale * (12 - month)
        means, counts = [], []
        for h in hours:
            cell = by_month_hour.get((month, h), {"mean": 0.0, "count": 0})
            if cell["count"] >= MIN_DATAPOINTS:
                means.append(cell["mean"])
            else:
                means.append(np.nan)
            counts.append(cell["count"])
        # Top left: mean climb strength + 2D sine fit
        y = np.array(means, dtype=float)
        valid = ~np.isnan(y)
        if np.any(valid):
            y = np.where(valid, y + base_y, np.nan)
            ax1.plot(hours, y, color="C0", alpha=0.8, label="data" if month == 1 else None)
        if fit_params is not None:
            day_mid = (month - 0.5) * 365 / 12
            x_fit = np.column_stack([np.full_like(hours, day_mid), hours.astype(float)])
            y_fit = _sine2d_model(x_fit, *fit_params) + base_y
            ax1.plot(hours, y_fit, color="C1", linestyle="--", alpha=0.6, linewidth=1, label="model" if month == 1 else None)
        # Top right: datapoint count as proper distribution (sum over month = 1)
        counts_arr = np.array(counts)
        count_dist = counts_arr / count_sum_by_month[month]
        ax2.plot(hours, count_dist + base_y_like, color="C0", alpha=0.8, label="data" if month == 1 else None)
        if month in fit_params_count:
            base, A, mu, sigma = fit_params_count[month]
            fit_count = _gaussian1d(hours_fine, base, A, mu, sigma)
            dx = (HOUR_END - HOUR_START) / (len(hours_fine) - 1) if len(hours_fine) > 1 else 1.0
            fit_count_norm = fit_count / (fit_count.sum() * dx)  # density (integral=1)
            ax2.plot(hours_fine, fit_count_norm + base_y_like, color="C1", linestyle="--", alpha=0.6, linewidth=1, label="model" if month == 1 else None)

    # Axis setup
    for ax in (ax1, ax2):
        ax.set_xlim(HOUR_START, HOUR_END)
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45)
        ax.set_xlabel("Hour (local solar)")
        ax.grid(True, alpha=0.3)
    ax1.set_yticks([offset * (12 - m) + 0.35 for m in range(1, 13)])
    ax1.set_yticklabels(MONTH_NAMES)
    ax2.set_yticks([like_scale * (12 - m) + 0.5 * like_scale for m in range(1, 13)])
    ax2.set_yticklabels(MONTH_NAMES)
    ax1.set_title(f"Mean climb strength (n>={MIN_DATAPOINTS})")
    ax2.set_title("Climb distribution")
    ax1.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, wspace=0.3, hspace=0.35)
    out_path = Path("data", "day_correlation_plots", "climb_strength_by_hour.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"\nPlot saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
