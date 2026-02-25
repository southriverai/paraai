"""Show progress of climbs."""

import asyncio
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, gaussian_kde
from tqdm import tqdm

from paraai.repository.repository_tracklog_body import RepositoryTracklogBody
from paraai.repository.repository_tracklog_header import RepositoryTracklogHeader
from paraai.tools_tracklogbody import _get_regions_with_indices, clean_tracklog

# Minimum progress length by definition (MIN_PROGRESS_RATE 300 m/min * MIN_PROGRESS_DURATION 60 s)
PROGRESS_LENGTH_CUTOFF_M = 300.0

MONTH_ORDER = (
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


def _parse_month_safe(date_str: str) -> str | None:
    """Return month name (e.g. 'January') if date is valid YYYY-MM-DD, else None."""
    if len(date_str) < 10:
        return None
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.strftime("%B")
    except ValueError:
        return None


def _trim_outliers(vals: list[float] | np.ndarray, p_low: float = 5, p_high: float = 95) -> np.ndarray:
    """Remove values below p_low and above p_high percentiles."""
    arr = np.asarray(vals)
    if len(arr) < 2:
        return arr
    lo, hi = np.percentile(arr, [p_low, p_high])
    return arr[(arr >= lo) & (arr <= hi)]


def _progress_length_m(tracklog_body, start_idx: int, end_idx: int) -> float:
    """Compute straight-line path length (m) for a progress region."""
    arr_si = tracklog_body.as_array_si()
    east_m = arr_si[:, 1]
    north_m = arr_si[:, 0]
    segs = np.sqrt(np.diff(east_m[start_idx : end_idx + 1]) ** 2 + np.diff(north_m[start_idx : end_idx + 1]) ** 2)
    return float(np.sum(segs))


def _climb_strength_m_s(tracklog_body, start_idx: int, end_idx: int) -> float:
    """Climb rate in m/s for a climb region (alt gain / time)."""
    arr = tracklog_body.as_array()
    alts = arr[:, 2]
    ts = arr[:, 3].astype(float)
    dt = float(ts[end_idx]) - float(ts[start_idx])
    if dt <= 0:
        return 0.0
    return float((alts[end_idx] - alts[start_idx]) / dt)


async def main() -> None:
    repo_body = RepositoryTracklogBody.get_instance()
    repo_header = RepositoryTracklogHeader.get_instance()
    tracklogs = await repo_body.asample(2000)

    mean_progress_lengths: list[float] = []
    all_progress_lengths: list[float] = []
    by_month: dict[str, list[float]] = defaultdict(list)  # month -> mean progress lengths per tracklog
    per_tracklog: list[tuple[float, float]] = []  # (mean_climb_m_s, mean_progress_m) for decile analysis

    for tb in tqdm(tracklogs, desc="Gathering"):
        header = repo_header.store.get(tb.tracklog_id)
        if header is None:
            continue
        month = _parse_month_safe(header.date)
        if month is None:
            continue  # skip corrupt dates

        clean_tb, _ = clean_tracklog(tb)
        regions = _get_regions_with_indices(clean_tb)
        climb_regions = [(s, e) for s, e, t in regions if t == "climb"]
        progress_regions = [(s, e) for s, e, t in regions if t == "progress"]
        lengths: list[float] = []
        for s, e in progress_regions:
            length_m = _progress_length_m(clean_tb, s, e)
            lengths.append(length_m)
            all_progress_lengths.append(length_m)
        if lengths and climb_regions:
            mean_len = np.mean(lengths)
            climb_rates = [_climb_strength_m_s(clean_tb, s, e) for s, e in climb_regions]
            mean_climb = np.mean(climb_rates)
            mean_progress_lengths.append(mean_len)
            by_month[month].append(mean_len)
            per_tracklog.append((mean_climb, mean_len))

    mean_progress_lengths = np.array(mean_progress_lengths)
    all_progress_lengths = np.array(all_progress_lengths)

    # Cut 5% and 95% outliers to reduce impact of data corruption
    mean_progress_lengths = _trim_outliers(mean_progress_lengths)
    all_progress_lengths = _trim_outliers(all_progress_lengths)
    by_month = {m: _trim_outliers(vals) for m, vals in by_month.items()}
    by_month = {m: v for m, v in by_month.items() if len(v) > 0}

    # Climb strength bins: 0.1 m/s width, starting at 0.2
    bin_width = 0.1
    bin_start = 0.2
    bin_stats: list[tuple[str, float, float, float, int]] = []
    per_tracklog_arr = np.array(per_tracklog)
    if len(per_tracklog_arr) > 0:
        climbs_full = per_tracklog_arr[:, 0]
        progs_full = per_tracklog_arr[:, 1]
        lo_c, hi_c = np.percentile(climbs_full, [5, 95])
        lo_p, hi_p = np.percentile(progs_full, [5, 95])
        mask = (climbs_full >= lo_c) & (climbs_full <= hi_c) & (progs_full >= lo_p) & (progs_full <= hi_p) & (climbs_full >= bin_start)
        per_tracklog_trimmed = per_tracklog_arr[mask]
        climbs = per_tracklog_trimmed[:, 0]
        progs = per_tracklog_trimmed[:, 1]
        max_climb = float(np.max(climbs))
        edges = np.arange(bin_start, max_climb + bin_width, bin_width)
        bin_indices = np.digitize(climbs, edges) - 1
        for i in range(len(edges) - 1):
            in_bin = bin_indices == i
            if not np.any(in_bin):
                continue
            chunk_progs = progs[in_bin]
            chunk_climbs = climbs[in_bin]
            lo_edge = edges[i]
            hi_edge = edges[i + 1]
            label = f"{lo_edge:.1f}-{hi_edge:.1f}"
            min_climb = float(np.min(chunk_climbs))
            max_climb_bin = float(np.max(chunk_climbs))
            mean_prog = float(np.mean(chunk_progs))
            count = int(np.sum(in_bin))
            bin_stats.append((label, min_climb, max_climb_bin, mean_prog, count))

        print("\n=== Mean progress by climb strength bin (0.1 m/s) ===")
        print("  bin       min_climb(m/s)  max_climb(m/s)  mean_progress(m)")
        for label, min_c, max_c, mean_p, _ in bin_stats:
            print(f"  {label:8}  {min_c:14.3f}  {max_c:14.3f}  {mean_p:15.0f}")

    print("\n=== Mean progress length per tracklog (m) ===")
    if len(mean_progress_lengths) > 0:
        print(f"  mean:  {np.mean(mean_progress_lengths):.0f}")
        print(f"  std:   {np.std(mean_progress_lengths):.0f}")
        print(f"  p10:   {np.percentile(mean_progress_lengths, 10):.0f}")
        print(f"  p50:   {np.percentile(mean_progress_lengths, 50):.0f}")
        print(f"  p90:   {np.percentile(mean_progress_lengths, 90):.0f}")

    print("\n=== Mean progress length per month (m) ===")
    for month in MONTH_ORDER:
        if month not in by_month:
            continue
        vals = by_month.get(month, np.array([]))
        if len(vals) == 0:
            continue
        print(f"  {month}: mean={np.mean(vals):.0f}")

    print("\n=== All progress lengths (m) ===")
    chi2_params = None
    if len(all_progress_lengths) >= 2:
        print(f"  mean:  {np.mean(all_progress_lengths):.0f}")
        print(f"  std:   {np.std(all_progress_lengths):.0f}")
        print(f"  p10:   {np.percentile(all_progress_lengths, 10):.0f}")
        print(f"  p50:   {np.percentile(all_progress_lengths, 50):.0f}")
        print(f"  p90:   {np.percentile(all_progress_lengths, 90):.0f}")
        # Fit chi-squared to excess above truncation (ignore pile-up at 300m cutoff)
        # Data < 300m is impossible by definition; density near 300m is artificially high
        fit_threshold = PROGRESS_LENGTH_CUTOFF_M  # fit on longer progress, slight emphasis on earlier data
        excess_for_fit = all_progress_lengths[all_progress_lengths > fit_threshold] - PROGRESS_LENGTH_CUTOFF_M
        chi2_params = None
        if len(excess_for_fit) >= 10:
            try:
                df_fit, _, scale_fit = chi2.fit(excess_for_fit, floc=0)
                chi2_params = (df_fit, PROGRESS_LENGTH_CUTOFF_M, scale_fit)  # loc for plot = cutoff
            except Exception:
                pass
        if chi2_params is not None:
            df_fit, loc_plot, scale_fit = chi2_params
            print("\n  Chi-squared fit (excess above 300m):")
            print(f"    df (degrees of freedom): {df_fit:.4f}")
            print(f"    scale: {scale_fit:.4f}")

    # Single figure with three charts
    months_with_data = [m for m in MONTH_ORDER if m in by_month]
    has_climb = len(bin_stats) > 0
    has_month = len(months_with_data) > 0
    has_kde = len(all_progress_lengths) >= 2
    if has_climb or has_month or has_kde:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        if has_climb:
            ax_climb = axes[0]
            bin_labels = [s[0] for s in bin_stats]
            bin_means = [s[3] for s in bin_stats]
            ax_climb.bar(bin_labels, bin_means, edgecolor="black", alpha=0.7)
            ax_climb.set_ylabel("Mean progress length (m)")
            ax_climb.set_xlabel("Mean climb strength (m/s)")
            ax_climb.set_title("Mean progress by climb strength")
            ax_climb.tick_params(axis="x", rotation=45)
        else:
            fig.delaxes(axes[0])
        if has_month:
            ax_month = axes[1]
            means = [np.mean(by_month[m]) for m in months_with_data]
            ax_month.bar(months_with_data, means, edgecolor="black", alpha=0.7)
            ax_month.set_ylabel("Mean progress length (m)")
            ax_month.set_title("Mean progress by month")
            ax_month.tick_params(axis="x", rotation=45)
        else:
            fig.delaxes(axes[1])
        if has_kde:
            ax_kde = axes[2]
            kde = gaussian_kde(all_progress_lengths)
            x_kde = np.linspace(all_progress_lengths.min(), all_progress_lengths.max(), 200)
            ax_kde.fill_between(x_kde, kde(x_kde), alpha=0.5)
            ax_kde.plot(x_kde, kde(x_kde), color="black", label="KDE")
            if chi2_params is not None:
                df_fit, _, scale_fit = chi2_params
                # Fit is for excess above 300m; pdf(x) = chi2.pdf(x - cutoff, df, 0, scale)
                mask = x_kde >= PROGRESS_LENGTH_CUTOFF_M
                chi2_pdf = np.zeros_like(x_kde)
                chi2_pdf[mask] = chi2.pdf(x_kde[mask] - PROGRESS_LENGTH_CUTOFF_M, df_fit, loc=0, scale=scale_fit)
                ax_kde.plot(x_kde, chi2_pdf, color="red", linestyle="--", linewidth=2, label="χ² fit")
                ax_kde.legend()
            ax_kde.set_xlabel("Progress length (m)")
            ax_kde.set_ylabel("Density")
            ax_kde.set_title("KDE: All progress lengths")
        else:
            fig.delaxes(axes[2])
        plt.tight_layout()
        out_path = Path("data", "day_correlation_plots", "progress_length_distribution.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        print(f"\nPlot saved to {out_path}")
        plt.show()


if __name__ == "__main__":
    path_db = Path("data", "database_sqlite")
    RepositoryTracklogBody.initialize_sqlite(path_db)
    RepositoryTracklogHeader.initialize_sqlite(path_db)
    asyncio.run(main())
