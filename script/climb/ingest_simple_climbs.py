"""
Import thermal/climb data from CSV (plain or .xz compressed).

Expected columns: trigger_lat, trigger_lon, trigger_alt, entry_lat, entry_lon, entry_alt,
  exit_lat, exit_lon, exit_alt, duration_sec, entry_epoch_sec, flight_id.
Entry = climb start (MIN), exit = climb end (MAX). Lat/lon WGS84, altitude AMSL (m).

Example:
  poetry run python script/climb/ingest_simple_climbs.py data/input/thermals.csv --europe --stream --limit 0
"""

import argparse
import csv
import lzma
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from paraai.model.simple_climb import SimpleClimb
from paraai.repository.repository_simple_climb import RepositorySimpleClimb
from paraai.tool_spacetime import EUROPE_BOUNDS


def _header_index(headers: list[str], name: str) -> str | None:
    """Return header string if name matches (case-insensitive), else None."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    for h in headers:
        if h.strip().lower().replace(" ", "_").replace("-", "_") == key:
            return h
    return None


def _in_bounds(lat: float, lon: float, bounds: tuple[float, float, float, float]) -> bool:
    """Check if (lat, lon) is within bounds (lat_min, lat_max, lon_min, lon_max)."""
    lat_min, lat_max, lon_min, lon_max = bounds
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _in_year_range(start_ts: int, year_min: int | None, year_max: int | None) -> bool:
    """Check if timestamp year is within [year_min, year_max] inclusive."""
    if year_min is None and year_max is None:
        return True
    year = datetime.fromtimestamp(start_ts, tz=timezone.utc).year
    return (year_min is None or year >= year_min) and (year_max is None or year <= year_max)


def count_csv_rows(
    path_file: Path,
    limit: int | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> tuple[int, int]:
    """Count data rows and unique tracklog_ids. If bounds given, only count rows within lat/lon.
    If year_min/year_max given, only count rows with start timestamp in that year range.
    Returns (row_count, unique_tracklog_count). Supports .csv and .csv.xz."""
    is_xz = path_file.suffix == ".xz" or ".xz" in path_file.suffixes
    open_fn = lzma.open if is_xz else open
    count = 0
    unique_tracklogs: set[str] = set()
    with open_fn(path_file, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        col_tracklog = _header_index(headers, "flight_id")
        col_start = _header_index(headers, "entry_epoch_sec") if (year_min or year_max) else None
        col_lat = _header_index(headers, "entry_lat") if bounds else None
        col_lon = _header_index(headers, "entry_lon") if bounds else None
        if bounds and (col_lat is None or col_lon is None):
            raise ValueError(f"CSV missing lat/lon columns for bounds filter. Headers: {headers}")
        if (year_min or year_max) and col_start is None:
            raise ValueError(f"CSV missing start timestamp column for year filter. Headers: {headers}")
        if col_tracklog is None:
            raise ValueError(f"CSV missing tracklog_id column. Headers: {headers}")
        pbar = tqdm(total=limit if limit else None, unit="rows", desc="Counting")
        for row in reader:
            if bounds:
                try:
                    lat = float(row[col_lat])
                    lon = float(row[col_lon])
                    if not _in_bounds(lat, lon, bounds):
                        pbar.update(1)
                        if limit and pbar.n >= limit:
                            break
                        continue
                except (ValueError, TypeError, KeyError):
                    pbar.update(1)
                    if limit and pbar.n >= limit:
                        break
                    continue
            if year_min is not None or year_max is not None:
                try:
                    start_ts = int(float(row[col_start]))
                    if not _in_year_range(start_ts, year_min, year_max):
                        pbar.update(1)
                        if limit and pbar.n >= limit:
                            break
                        continue
                except (ValueError, TypeError, KeyError):
                    pbar.update(1)
                    if limit and pbar.n >= limit:
                        break
                    continue
            count += 1
            tid = str(row[col_tracklog]).strip()
            if tid:
                unique_tracklogs.add(tid)
            pbar.update(1)
            if limit and count >= limit:
                break
        pbar.close()
    return count, len(unique_tracklogs)


def load_simple_climbs_from_csv(
    path_file: Path,
    limit: int | None = None,
    debug: bool = False,
    bounds: tuple[float, float, float, float] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[SimpleClimb]:
    """Parse CSV with MIN/MAX thermal format into SimpleClimb list. Supports .csv and .csv.xz."""
    climbs: list[SimpleClimb] = []
    is_xz = path_file.suffix == ".xz" or ".xz" in path_file.suffixes
    open_fn = lzma.open if is_xz else open
    mode = "rt"
    with open_fn(path_file, mode, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        col_ground_lat = _header_index(headers, "trigger_lat")
        col_ground_lon = _header_index(headers, "trigger_lon")
        col_ground_alt = _header_index(headers, "trigger_alt")
        col_start_lat = _header_index(headers, "entry_lat")
        col_start_lon = _header_index(headers, "entry_lon")
        col_start_alt = _header_index(headers, "entry_alt")
        col_end_lat = _header_index(headers, "exit_lat")
        col_end_lon = _header_index(headers, "exit_lon")
        col_end_alt = _header_index(headers, "exit_alt")
        col_start = _header_index(headers, "entry_epoch_sec")
        col_seconds = _header_index(headers, "duration_sec")
        col_tracklog = _header_index(headers, "flight_id")

        required = [
            col_ground_lat,
            col_ground_lon,
            col_ground_alt,
            col_start_lat,
            col_start_lon,
            col_start_alt,
            col_end_lat,
            col_end_lon,
            col_end_alt,
            col_start,
            col_seconds,
            col_tracklog,
        ]
        if any(r is None for r in required):
            raise ValueError(
                f"CSV missing required columns. Headers: {headers}. "
                "Expected: trigger_lat, trigger_lon, trigger_alt, entry_lat, entry_lon, entry_alt, "
                "exit_lat, exit_lon, exit_alt, entry_epoch_sec, duration_sec, flight_id"
            )

        row_iter = iter(reader)
        pbar = tqdm(total=limit if limit else None, unit="rows", desc="Reading CSV")
        for row in row_iter:
            try:
                start_ts = int(float(row[col_start]))
                seconds = int(float(row[col_seconds]))
                end_ts = start_ts + seconds
                tracklog_id = str(row[col_tracklog]).strip()
                if not tracklog_id:
                    continue
                start_lat = float(row[col_start_lat])
                start_lon = float(row[col_start_lon])
                if bounds and not _in_bounds(start_lat, start_lon, bounds):
                    pbar.update(1)
                    if limit and pbar.n >= limit:
                        break
                    continue
                if not _in_year_range(start_ts, year_min, year_max):
                    pbar.update(1)
                    if limit and pbar.n >= limit:
                        break
                    continue
                start_alt = float(row[col_start_alt])
                end_alt = float(row[col_end_alt])
                height_m = end_alt - start_alt
                climb_strength_m_s = height_m / seconds if seconds > 0 else 0.0
                climb = SimpleClimb(
                    simple_climb_id=SimpleClimb.create_id(tracklog_id, start_ts),
                    tracklog_id=tracklog_id,
                    start_lat=start_lat,
                    start_lon=start_lon,
                    start_alt_m=start_alt,
                    start_timestamp_utc=start_ts,
                    end_lat=float(row[col_end_lat]),
                    end_lon=float(row[col_end_lon]),
                    end_alt_m=end_alt,
                    end_timestamp_utc=end_ts,
                    duration_sec=seconds,
                    height_m=height_m,
                    climb_strength_m_s=climb_strength_m_s,
                    ground_lat=float(row[col_ground_lat]),
                    ground_lon=float(row[col_ground_lon]),
                    ground_alt_m=float(row[col_ground_alt]),
                )
                climbs.append(climb)
            except (ValueError, TypeError, KeyError) as e:
                if debug and len(climbs) == 0:
                    print(f"First parse error: {e!r}")
                    print(f"Row: {row}")
            pbar.update(1)
            if limit and pbar.n >= limit:
                break
        pbar.close()
    return climbs


def stream_simple_climbs_from_csv(
    path_file: Path,
    repo: RepositorySimpleClimb,
    limit: int | None = None,
    debug: bool = False,
    batch_size: int = 5000,
    bounds: tuple[float, float, float, float] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> int:
    """Stream CSV rows into repository in batches. Returns total imported count."""
    total_imported = 0
    batch: list[SimpleClimb] = []
    is_xz = path_file.suffix == ".xz" or ".xz" in path_file.suffixes
    open_fn = lzma.open if is_xz else open
    with open_fn(path_file, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        col_ground_lat = _header_index(headers, "trigger_lat")
        col_ground_lon = _header_index(headers, "trigger_lon")
        col_ground_alt = _header_index(headers, "trigger_alt")
        col_start_lat = _header_index(headers, "entry_lat")
        col_start_lon = _header_index(headers, "entry_lon")
        col_start_alt = _header_index(headers, "entry_alt")
        col_end_lat = _header_index(headers, "exit_lat")
        col_end_lon = _header_index(headers, "exit_lon")
        col_end_alt = _header_index(headers, "exit_alt")
        col_start = _header_index(headers, "entry_epoch_sec")
        col_seconds = _header_index(headers, "duration_sec")
        col_tracklog = _header_index(headers, "flight_id")

        required = [
            col_ground_lat,
            col_ground_lon,
            col_ground_alt,
            col_start_lat,
            col_start_lon,
            col_start_alt,
            col_end_lat,
            col_end_lon,
            col_end_alt,
            col_start,
            col_seconds,
            col_tracklog,
        ]
        if any(r is None for r in required):
            raise ValueError(
                f"CSV missing required columns. Headers: {headers}. "
                "Expected: trigger_lat, trigger_lon, trigger_alt, entry_lat, entry_lon, entry_alt, "
                "exit_lat, exit_lon, exit_alt, entry_epoch_sec, duration_sec, flight_id"
            )

        pbar = tqdm(total=limit if limit else None, unit="rows", desc="Streaming CSV")
        for row in reader:
            try:
                start_ts = int(float(row[col_start]))
                seconds = int(float(row[col_seconds]))
                end_ts = start_ts + seconds
                tracklog_id = str(row[col_tracklog]).strip()
                if not tracklog_id:
                    pbar.update(1)
                    continue
                start_lat = float(row[col_start_lat])
                start_lon = float(row[col_start_lon])
                if bounds and not _in_bounds(start_lat, start_lon, bounds):
                    pbar.update(1)
                    continue
                if not _in_year_range(start_ts, year_min, year_max):
                    pbar.update(1)
                    continue
                start_alt = float(row[col_start_alt])
                end_alt = float(row[col_end_alt])
                height_m = end_alt - start_alt
                climb_strength_m_s = height_m / seconds if seconds > 0 else 0.0
                climb = SimpleClimb(
                    simple_climb_id=SimpleClimb.create_id(tracklog_id, start_ts),
                    tracklog_id=tracklog_id,
                    start_lat=start_lat,
                    start_lon=start_lon,
                    start_alt_m=start_alt,
                    start_timestamp_utc=start_ts,
                    end_lat=float(row[col_end_lat]),
                    end_lon=float(row[col_end_lon]),
                    end_alt_m=end_alt,
                    end_timestamp_utc=end_ts,
                    duration_sec=seconds,
                    height_m=height_m,
                    climb_strength_m_s=climb_strength_m_s,
                    ground_lat=float(row[col_ground_lat]),
                    ground_lon=float(row[col_ground_lon]),
                    ground_alt_m=float(row[col_ground_alt]),
                )
                batch.append(climb)
                if len(batch) >= batch_size:
                    repo.insert_many(batch)
                    total_imported += len(batch)
                    batch = []
            except (ValueError, TypeError, KeyError) as e:
                if debug and total_imported == 0 and len(batch) == 0:
                    print(f"First parse error: {e!r}")
                    print(f"Row: {row}")
            pbar.update(1)
            if limit and pbar.n >= limit:
                break
        pbar.close()
        if batch:
            repo.insert_many(batch)
            total_imported += len(batch)
    return total_imported


def main() -> None:
    parser = argparse.ArgumentParser(description="Import thermal/climb CSV into SimpleClimb repository")
    parser.add_argument(
        "input", type=Path, nargs="?", default=Path("data", "input", "thermaldb-2025-02-13.csv.xz"), help="Input CSV or .csv.xz file"
    )
    parser.add_argument("--limit", type=int, default=0, help="Max CSV rows to read (default: 10000, 0=no limit)")
    parser.add_argument("--count", action="store_true", help="Only count rows in CSV, do not import")
    parser.add_argument("--stream", action="store_true", help="Stream rows into database in batches (low memory)")
    parser.add_argument("--europe", action="store_true", help="Filter to Europe (lat 36-72, lon -11-42)")
    parser.add_argument("--lat-min", type=float, help="Min latitude (use with --lat-max, --lon-min, --lon-max)")
    parser.add_argument("--lat-max", type=float, help="Max latitude")
    parser.add_argument("--lon-min", type=float, help="Min longitude")
    parser.add_argument("--lon-max", type=float, help="Max longitude")
    parser.add_argument("--year-min", type=int, help="Min year (inclusive, filter by climb start timestamp)")
    parser.add_argument("--year-max", type=int, help="Max year (inclusive, filter by climb start timestamp)")
    parser.add_argument("--debug", action="store_true", help="Print first parse error on failure")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    if args.year_min is not None and args.year_max is not None and args.year_min > args.year_max:
        parser.error("--year-min must be <= --year-max")

    bounds: tuple[float, float, float, float] | None = None
    if args.europe:
        bounds = EUROPE_BOUNDS
    if all(x is not None for x in (args.lat_min, args.lat_max, args.lon_min, args.lon_max)):
        bounds = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    elif any(x is not None for x in (args.lat_min, args.lat_max, args.lon_min, args.lon_max)):
        parser.error("When using lat/lon bounds, all four --lat-min, --lat-max, --lon-min, --lon-max are required")

    path_file = args.input

    if args.count:
        row_count, unique_tracklog_count = count_csv_rows(
            path_file,
            limit=limit,
            bounds=bounds,
            year_min=args.year_min,
            year_max=args.year_max,
        )
        region = "Europe" if args.europe else "region" if bounds else "total"
        year_range = ""
        if args.year_min is not None or args.year_max is not None:
            lo = args.year_min if args.year_min is not None else "?"
            hi = args.year_max if args.year_max is not None else "?"
            year_range = f" years {lo}-{hi}"
        print(f"Row count ({region}{year_range}): {row_count} (limit={limit or 'none'})")
        print(f"Unique tracklog_id: {unique_tracklog_count}")
        return

    path_dir_database = Path("data", "database_sqlite")
    repo = RepositorySimpleClimb.initialize_sqlite(path_dir_database)
    repo.clear_all()

    if args.stream:
        total = stream_simple_climbs_from_csv(
            path_file,
            repo,
            limit=limit,
            debug=args.debug,
            bounds=bounds,
            year_min=args.year_min,
            year_max=args.year_max,
        )
        print(f"Imported {total} simple climbs from {path_file}")
    else:
        climbs = load_simple_climbs_from_csv(
            path_file,
            limit=limit,
            debug=args.debug,
            bounds=bounds,
            year_min=args.year_min,
            year_max=args.year_max,
        )
        repo.insert_many(climbs)
        print(f"Imported {len(climbs)} simple climbs from {path_file}")


if __name__ == "__main__":
    main()
