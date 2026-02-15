"""
Import thermal/climb data from CSV (plain or .xz compressed).
Format (per email): MIN = entry point, MAX = exit point.
- lat, lon: WGS84
- altitude: AMSL (m)
- START: timestamp of MIN
- START+SECONDS: timestamp of MAX
- FLIGHTSTATS_ID: flight identifier (tracklog_id)

Example with both Europe and year filters (2000-2024, same as show_years.py):
  poetry run python script/climb/collect_simple_climbs.py data/thermals.csv --europe --year-min 2000 --year-max 2024 --stream --limit 0
"""

import argparse
import csv
import lzma
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from paraai.model.simple_climb import SimpleClimb
from paraai.repository.repository_simple_climb import RepositorySimpleClimb

# Column name variants (case-insensitive match)
COLUMN_ALIASES = {
    "start_lat": ["min_lat", "entry_lat", "minlat", "min lat"],
    "start_lon": ["min_lon", "entry_lon", "minlng", "minlon", "min lon", "min lng"],
    "start_alt_m": ["min_alt", "entry_alt", "minalt", "min_alt_m", "min alt"],
    "end_lat": ["max_lat", "exit_lat", "maxlat", "max lat"],
    "end_lon": ["max_lon", "exit_lon", "maxlng", "maxlon", "max lon", "max lng"],
    "end_alt_m": ["max_alt", "exit_alt", "maxalt", "max_alt_m", "max alt"],
    "start_timestamp_utc": ["start", "entry_epoch_sec", "start_timestamp", "start_ts"],
    "seconds": ["duration", "duration_sec", "seconds", "sec", "duration_seconds"],
    "tracklog_id": ["flightstats_id", "flight_id"],
}


def _normalize_header(h: str) -> str:
    return h.strip().lower().replace(" ", "_").replace("-", "_")


# Europe bounds (lat min/max, lon min/max): ~Iberia to Nordkapp, Ireland to Urals
EUROPE_BOUNDS: tuple[float, float, float, float] = (36.0, 72.0, -11.0, 42.0)


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


def _find_column_name(headers: list[str], field: str) -> str | None:
    """Return the actual header string that matches the field, or None."""
    norm_headers = [_normalize_header(h) for h in headers]
    candidates = [field, *COLUMN_ALIASES.get(field, [])]
    for c in candidates:
        cn = _normalize_header(c)
        for i, nh in enumerate(norm_headers):
            if cn == nh or nh.endswith("_" + cn) or cn in nh:
                return headers[i]
    return None


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
        col_tracklog = _find_column_name(headers, "tracklog_id")
        col_start = _find_column_name(headers, "start_timestamp_utc") if (year_min or year_max) else None
        col_lat = _find_column_name(headers, "start_lat") if bounds else None
        col_lon = _find_column_name(headers, "start_lon") if bounds else None
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
        col_start_lat = _find_column_name(headers, "start_lat")
        col_start_lon = _find_column_name(headers, "start_lon")
        col_start_alt = _find_column_name(headers, "start_alt_m")
        col_end_lat = _find_column_name(headers, "end_lat")
        col_end_lon = _find_column_name(headers, "end_lon")
        col_end_alt = _find_column_name(headers, "end_alt_m")
        col_start = _find_column_name(headers, "start_timestamp_utc")
        col_seconds = _find_column_name(headers, "seconds")
        col_tracklog = _find_column_name(headers, "tracklog_id")

        required = [
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
                "Expected: min_lat, min_lon, min_alt, max_lat, max_lon, max_alt, start, seconds, flightstats_id"
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
                climb = SimpleClimb(
                    simple_climb_id=SimpleClimb.create_id(tracklog_id, start_ts),
                    tracklog_id=tracklog_id,
                    start_lat=start_lat,
                    start_lon=start_lon,
                    start_alt_m=float(row[col_start_alt]),
                    start_timestamp_utc=start_ts,
                    end_lat=float(row[col_end_lat]),
                    end_lon=float(row[col_end_lon]),
                    end_alt_m=float(row[col_end_alt]),
                    end_timestamp_utc=end_ts,
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
        col_start_lat = _find_column_name(headers, "start_lat")
        col_start_lon = _find_column_name(headers, "start_lon")
        col_start_alt = _find_column_name(headers, "start_alt_m")
        col_end_lat = _find_column_name(headers, "end_lat")
        col_end_lon = _find_column_name(headers, "end_lon")
        col_end_alt = _find_column_name(headers, "end_alt_m")
        col_start = _find_column_name(headers, "start_timestamp_utc")
        col_seconds = _find_column_name(headers, "seconds")
        col_tracklog = _find_column_name(headers, "tracklog_id")

        required = [col_start_lat, col_start_lon, col_start_alt, col_end_lat, col_end_lon,
                    col_end_alt, col_start, col_seconds, col_tracklog]
        if any(r is None for r in required):
            raise ValueError(
                f"CSV missing required columns. Headers: {headers}. "
                "Expected: min_lat, min_lon, min_alt, max_lat, max_lon, max_alt, start, seconds, flightstats_id"
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
                climb = SimpleClimb(
                    simple_climb_id=SimpleClimb.create_id(tracklog_id, start_ts),
                    tracklog_id=tracklog_id,
                    start_lat=start_lat,
                    start_lon=start_lon,
                    start_alt_m=float(row[col_start_alt]),
                    start_timestamp_utc=start_ts,
                    end_lat=float(row[col_end_lat]),
                    end_lon=float(row[col_end_lon]),
                    end_alt_m=float(row[col_end_alt]),
                    end_timestamp_utc=end_ts,
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


def main():
    parser = argparse.ArgumentParser(description="Import thermal/climb CSV into SimpleClimb repository")
    parser.add_argument("path_file_source", type=Path, help="CSV file (MIN/MAX format)")
    parser.add_argument("--limit", type=int, default=10000, help="Max CSV rows to read (default: 10000, 0=no limit)")
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

    if args.count:
        row_count, unique_tracklog_count = count_csv_rows(
            args.path_file_source,
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
            args.path_file_source,
            repo,
            limit=limit,
            debug=args.debug,
            bounds=bounds,
            year_min=args.year_min,
            year_max=args.year_max,
        )
        print(f"Imported {total} simple climbs from {args.path_file_source}")
    else:
        climbs = load_simple_climbs_from_csv(
            args.path_file_source,
            limit=limit,
            debug=args.debug,
            bounds=bounds,
            year_min=args.year_min,
            year_max=args.year_max,
        )
        repo.insert_many(climbs)
        print(f"Imported {len(climbs)} simple climbs from {args.path_file_source}")


if __name__ == "__main__":
    main()
