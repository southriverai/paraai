import asyncio
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from paraai.repository.repository_tracklog_header import RepositoryTracklogHeader

SECONDS_PER_HOUR = 3600
VALID_YEAR_MIN = 2024
VALID_YEAR_MAX = 2026


def _month_bin_edges(timestamps: list[int]) -> list[float]:
    """Return bin edges at month boundaries covering the range of timestamps."""
    if not timestamps:
        return []
    ts_min, ts_max = min(timestamps), max(timestamps)
    dt_min = datetime.fromtimestamp(ts_min, tz=timezone.utc)
    dt_max = datetime.fromtimestamp(ts_max, tz=timezone.utc)
    edges = []
    y, m = dt_min.year, dt_min.month
    while (y, m) <= (dt_max.year, dt_max.month):
        edges.append(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())
        m += 1
        if m > 12:
            m = 1
            y += 1
    edges.append(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())
    return edges


async def amain():
    headers = await repository.get_all()

    total_flights = len(headers)
    flights_over_hour = sum(1 for h in headers if h.duration_seconds > SECONDS_PER_HOUR)

    print(f"Total flights: {total_flights}")
    print(f"Flights over 1 hour: {flights_over_hour}")

    def _safe_takeoff_utc(h):
        try:
            return h.takeoff_timestamp_utc
        except Exception:
            return None

    takeoff_utc_all = [
        ts
        for h in headers
        if (ts := _safe_takeoff_utc(h)) is not None and VALID_YEAR_MIN <= datetime.fromtimestamp(ts, tz=timezone.utc).year <= VALID_YEAR_MAX
    ]

    month_edges_all = _month_bin_edges(takeoff_utc_all) if takeoff_utc_all else []

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    def _year_ticks_and_labels(bin_edges: list[float]) -> tuple[list[float], list[str]]:
        """Return (tick positions, labels) for year boundaries only."""
        if len(bin_edges) < 2:
            return [], []
        years_seen = set()
        tick_ts = []
        labels = []
        for ts in bin_edges:
            y = datetime.fromtimestamp(ts, tz=timezone.utc).year
            if y not in years_seen:
                years_seen.add(y)
                tick_ts.append(ts)
                labels.append(str(y))
        return tick_ts, labels

    def _hist_month_bins_year_labels(ax, data: list[int], title: str, bin_edges: list[float]):
        if not data or len(bin_edges) < 2:
            ax.set_title(title)
            return
        ax.hist(data, bins=bin_edges, alpha=0.8, edgecolor="black")
        ax.set_ylabel("Flights")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)
        tick_ts, labels = _year_ticks_and_labels(bin_edges)
        ax.set_xticks(tick_ts)
        ax.set_xticklabels(labels, rotation=45, ha="right")

    _hist_month_bins_year_labels(
        ax1,
        takeoff_utc_all,
        "Flights per month",
        month_edges_all,
    )

    ax2.hist(takeoff_utc_all, bins=50, alpha=0.8, edgecolor="black")
    ax2.set_xlabel("Takeoff timestamp UTC")
    ax2.set_ylabel("Flights")
    ax2.set_title("Histogram of takeoff timestamp UTC")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    path_dir_database = Path("data", "database_sqlite")
    repository = RepositoryTracklogHeader.initialize_sqlite(path_dir_database)

    asyncio.run(amain())
