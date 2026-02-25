import argparse
import asyncio
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from paraai.model.climb import Climb
from paraai.repository.repository_climb import RepositoryClimb

N_STEPS = 20
MAX_VERTICAL_SPEED_M_S = 10.0  # Cap GPS spikes (matches tools_tracklogbody)
CACHE_PATH = Path("data", "show_inside.json")


def _resample_climb_velocity(climb: Climb) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Rescale altitude 0-1 for one climb, compute segment velocities, resample to N_STEPS bins.
    Uses climb's alt_min/alt_max and mean velocity. Returns (bin_values: velocity/mean per bin, bin_counts), or None.
    """
    alts = np.array(climb.list_altitude_m, dtype=float)
    ts = np.array(climb.list_timestamp_utc, dtype=float)
    ts = ts - ts[0]
    if len(alts) < 2 or len(ts) < 2:
        return None
    alt_min = float(alts.min())
    alt_max = float(alts.max())
    if alt_max <= alt_min:
        return None
    alt_norm = (alts - alt_min) / (alt_max - alt_min)

    dt = np.diff(ts)
    dt = np.where(dt > 0, dt, np.nan)
    vel = np.diff(alts) / dt
    vel = np.where(np.isfinite(vel), vel, np.nan)
    vel = np.clip(vel, 0, MAX_VERTICAL_SPEED_M_S)
    frac_mid = (alt_norm[:-1] + alt_norm[1:]) / 2

    bin_sum_vdt: list[float] = [0.0] * N_STEPS
    bin_sum_dt: list[float] = [0.0] * N_STEPS
    bin_count: list[int] = [0] * N_STEPS
    for f, v, d in zip(frac_mid, vel, dt, strict=True):
        if not np.isfinite(v) or not np.isfinite(d) or v < 0 or d <= 0:
            continue
        idx = min(int(f * N_STEPS), N_STEPS - 1)
        bin_sum_vdt[idx] += float(v * d)
        bin_sum_dt[idx] += float(d)
        bin_count[idx] += 1

    result = np.full(N_STEPS, np.nan)
    for i in range(N_STEPS):
        if bin_sum_dt[i] > 0:
            result[i] = bin_sum_vdt[i] / bin_sum_dt[i]

    total_vdt = sum(bin_sum_vdt)
    total_dt = sum(bin_sum_dt)
    mean_vel_climb = total_vdt / total_dt if total_dt > 0 else np.nan
    if not np.isfinite(mean_vel_climb) or mean_vel_climb <= 0:
        return None

    bin_values = result / mean_vel_climb
    return bin_values, np.array(bin_count, dtype=int)


def _resample_climb_velocity_tracklog(climbs: list[Climb]) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Rescale altitude 0-1 across all climbs in the list (shared tracklog), compute velocities, resample to N_STEPS bins.
    Uses alt_min/alt_max from all climbs and mean velocity across all climbs. Returns (bin_values, bin_counts), or None.
    """
    if not climbs:
        return None
    all_alts = [a for c in climbs for a in c.list_altitude_m]
    if not all_alts:
        return None
    alt_min = min(all_alts)
    alt_max = max(all_alts)
    if alt_max <= alt_min:
        return None

    climb_means: list[float] = []
    all_bin_values: list[np.ndarray] = []
    bin_counts_sum = np.zeros(N_STEPS, dtype=int)

    for climb in climbs:
        alts = np.array(climb.list_altitude_m, dtype=float)
        ts = np.array(climb.list_timestamp_utc, dtype=float)
        ts = ts - ts[0]
        if len(alts) < 2 or len(ts) < 2:
            continue
        alt_norm = (alts - alt_min) / (alt_max - alt_min)

        dt = np.diff(ts)
        dt = np.where(dt > 0, dt, np.nan)
        vel = np.diff(alts) / dt
        vel = np.where(np.isfinite(vel), vel, np.nan)
        vel = np.clip(vel, 0, MAX_VERTICAL_SPEED_M_S)
        frac_mid = (alt_norm[:-1] + alt_norm[1:]) / 2

        bin_sum_vdt: list[float] = [0.0] * N_STEPS
        bin_sum_dt: list[float] = [0.0] * N_STEPS
        bin_count: list[int] = [0] * N_STEPS
        for f, v, d in zip(frac_mid, vel, dt, strict=True):
            if not np.isfinite(v) or not np.isfinite(d) or v < 0 or d <= 0:
                continue
            idx = min(int(f * N_STEPS), N_STEPS - 1)
            bin_sum_vdt[idx] += float(v * d)
            bin_sum_dt[idx] += float(d)
            bin_count[idx] += 1

        result = np.full(N_STEPS, np.nan)
        for i in range(N_STEPS):
            if bin_sum_dt[i] > 0:
                result[i] = bin_sum_vdt[i] / bin_sum_dt[i]

        total_vdt = sum(bin_sum_vdt)
        total_dt = sum(bin_sum_dt)
        mean_vel_climb = total_vdt / total_dt if total_dt > 0 else np.nan
        if not np.isfinite(mean_vel_climb) or mean_vel_climb <= 0:
            continue

        climb_means.append(float(mean_vel_climb))
        bin_counts_sum += np.array(bin_count, dtype=int)
        all_bin_values.append(result)

    if not climb_means or not all_bin_values:
        return None

    mean_vel = float(np.mean(climb_means))
    arr = np.array(all_bin_values)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
        bin_values = np.nanmean(arr, axis=0) / mean_vel
    return bin_values, bin_counts_sum


def _to_json_safe(arr: np.ndarray) -> list:
    return [None if (isinstance(x, (float, np.floating)) and np.isnan(x)) else float(x) for x in arr]


def _from_json_safe(arr: list) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in arr], dtype=float)


def _load_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not CACHE_PATH.exists():
        return None
    with CACHE_PATH.open() as f:
        cache = json.load(f)
    return (
        _from_json_safe(cache["bin_values_climb"]),
        _from_json_safe(cache["bin_counts_climb"]),
        _from_json_safe(cache["bin_values_tracklog"]),
        _from_json_safe(cache["bin_counts_tracklog"]),
    )


def _save_cache(
    bin_values_climb: np.ndarray,
    bin_counts_climb: np.ndarray,
    bin_values_tracklog: np.ndarray,
    bin_counts_tracklog: np.ndarray,
) -> None:
    cache = {
        "bin_values_climb": _to_json_safe(bin_values_climb),
        "bin_counts_climb": _to_json_safe(bin_counts_climb),
        "bin_values_tracklog": _to_json_safe(bin_values_tracklog),
        "bin_counts_tracklog": _to_json_safe(bin_counts_tracklog),
    }
    with CACHE_PATH.open("w") as f:
        json.dump(cache, f, indent=2)


def _compute_data(
    climbs: list[Climb],
    by_tracklog: dict[str, list[Climb]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_bin_values: list[np.ndarray] = []
    bin_counts_sum = np.zeros(N_STEPS, dtype=int)
    for climb in tqdm(climbs, desc="Per-climb analysis"):
        out = _resample_climb_velocity(climb)
        if out is None:
            continue
        bv, bc = out
        all_bin_values.append(bv)
        bin_counts_sum += bc

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
        bin_values_climb = np.nanmean(np.array(all_bin_values), axis=0)

    all_bin_values_tl: list[np.ndarray] = []
    bin_counts_sum_tl = np.zeros(N_STEPS, dtype=int)
    for _, tracklog_climbs in tqdm(by_tracklog.items(), desc="Per-flight analysis"):
        out = _resample_climb_velocity_tracklog(tracklog_climbs)
        if out is None:
            continue
        bv, bc = out
        all_bin_values_tl.append(bv)
        bin_counts_sum_tl += bc

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
        bin_values_tracklog = np.nanmean(np.array(all_bin_values_tl), axis=0)

    return bin_values_climb, bin_counts_sum, bin_values_tracklog, bin_counts_sum_tl


async def main(ignore_cache: bool = False):
    cached = None if ignore_cache else _load_cache()
    if cached is not None:
        bin_values_climb, bin_counts_climb, bin_values_tracklog, bin_counts_tracklog = cached
        print(f"Loaded cache from {CACHE_PATH}")
    else:
        repo = RepositoryClimb.get_instance()
        climbs = await repo.aget_all()

        by_tracklog: dict[str, list[Climb]] = {}
        for climb in tqdm(climbs, desc="Gathering by tracklog"):
            by_tracklog.setdefault(climb.tracklog_id, []).append(climb)

        bin_values_climb, bin_counts_climb, bin_values_tracklog, bin_counts_tracklog = _compute_data(climbs, by_tracklog)
        _save_cache(bin_values_climb, bin_counts_climb, bin_values_tracklog, bin_counts_tracklog)
        print(f"Cached to {CACHE_PATH}")

    total_climb = bin_counts_climb.sum()
    total_tracklog = bin_counts_tracklog.sum()
    bin_counts_climb_pct = (bin_counts_climb / total_climb * 100) if total_climb > 0 else np.zeros(N_STEPS)
    bin_counts_tracklog_pct = (bin_counts_tracklog / total_tracklog * 100) if total_tracklog > 0 else np.zeros(N_STEPS)

    x = np.linspace(0.05, 0.95, N_STEPS)
    width = 0.04
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes[0, 0].bar(x, bin_values_climb, width=width, color="steelblue", alpha=0.8)
    axes[0, 0].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].set_ylabel("Relative velocity")
    axes[0, 0].set_title("Per-climb: velocity")
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    ymin, ymax = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(min(ymin, 0), max(ymax, 1))

    axes[0, 1].bar(x, bin_values_tracklog, width=width, color="coral", alpha=0.8)
    axes[0, 1].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 1].set_ylabel("Relative velocity")
    axes[0, 1].set_title("Per-flight: velocity")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    ymin, ymax = axes[0, 1].get_ylim()
    axes[0, 1].set_ylim(min(ymin, 0), max(ymax, 1))

    axes[1, 0].bar(x, bin_counts_climb_pct, width=width, color="steelblue", alpha=0.8)
    axes[1, 0].set_xlabel("Altitude")
    axes[1, 0].set_ylabel("Time spent (%)")
    axes[1, 0].set_title("Per-climb: time spent at altitude")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].bar(x, bin_counts_tracklog_pct, width=width, color="coral", alpha=0.8)
    axes[1, 1].set_xlabel("Altitude")
    axes[1, 1].set_ylabel("Time spent (%)")
    axes[1, 1].set_title("Per-flight: time spent at altitude")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache and recompute from database")
    args = parser.parse_args()
    path_dir_database = Path("data", "database_sqlite")
    RepositoryClimb.initialize_sqlite(path_dir_database)
    asyncio.run(main(ignore_cache=args.no_cache))
