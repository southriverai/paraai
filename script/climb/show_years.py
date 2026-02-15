import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

from paraai.model.simple_climb import SimpleClimb
from paraai.repository.repository_simple_climb import RepositorySimpleClimb

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Mean climb speed per year
    bars1 = ax1.bar(years, means, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean climb speed (m/s)")
    ax1.set_title("Mean climb speed per year")
    ax1.set_xticks(years)
    for bar, count in zip(bars1, counts):
        ax1.annotate(f"n={count}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    # Plot 2: Mean climb speed per month (aggregated over all years)
    bars2 = ax2.bar(months, month_means, color="steelblue", edgecolor="white")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Mean climb speed (m/s)")
    ax2.set_title("Mean climb speed per month (all years)")
    ax2.set_xticks(months)
    ax2.set_xticklabels(month_labels)
    for bar, count in zip(bars2, month_counts):
        ax2.annotate(f"n={count}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
