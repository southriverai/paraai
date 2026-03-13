"""Add indexes and performance PRAGMAs to srai_store SQLite databases.

Run once after ingestion. Indexes and WAL mode persist. Speeds up:
- ORDER BY on timestamp fields (earliest/latest queries)
- Bounding box queries on lat/lon (simple_climb start+ground, simple_climb_pixel, trigger_point)
- Name lookups (trigger_point)
- General read throughput

Usage:
    poetry run python script/database/optimize_sqlite.py
"""

import sqlite3
from pathlib import Path

# Collections with timestamp fields for ORDER BY / earliest-latest
INDEX_START_TS = {"simple_climb"}
INDEX_LIST_TS_0 = {"climb"}  # list_timestamp_utc[0]

# Collections with lat/lon for bounding box queries (field names -> collections)
INDEX_LAT_LON: dict[tuple[str, str], set[str]] = {
    ("start_lat", "start_lon"): {"simple_climb"},  # get_all_in_bounding_box
    ("ground_lat", "ground_lon"): {"simple_climb"},  # get_all_in_bounding_box_by_ground
    ("lat", "lon"): {"simple_climb_pixel", "trigger_point"},
}


def optimize_db(path_db: Path) -> None:
    """Apply PRAGMAs and indexes to a single .db file."""
    if not path_db.exists():
        return
    size_mb = path_db.stat().st_size / (1024 * 1024)
    collection = path_db.stem
    print(f"Optimizing {path_db.name} ({size_mb:.1f} MB)...")
    with sqlite3.connect(path_db) as conn:
        # WAL mode: better concurrent reads, faster writes
        conn.execute("PRAGMA journal_mode=WAL")
        # Reduce fsync for better throughput (trade durability for speed)
        conn.execute("PRAGMA synchronous=NORMAL")
        # Use more memory for cache (default 2MB; -64MB = 64MB)
        conn.execute("PRAGMA cache_size=-64000")
        # Memory-mapped I/O for large reads
        conn.execute("PRAGMA mmap_size=268435456")  # 256 MB

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='store'")
        if not cursor.fetchone():
            print("  Skipped (no 'store' table)")
            return

        if collection in INDEX_START_TS:
            try:
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_start_timestamp_utc
                    ON store (json_extract(document, '$.start_timestamp_utc'))
                    """
                )
                print("  Created index idx_start_timestamp_utc")
            except sqlite3.OperationalError as e:
                if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                    print("  Index idx_start_timestamp_utc already exists")
                else:
                    raise

        if collection in INDEX_LIST_TS_0:
            try:
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_list_timestamp_utc_0
                    ON store (json_extract(document, '$.list_timestamp_utc[0]'))
                    """
                )
                print("  Created index idx_list_timestamp_utc_0")
            except sqlite3.OperationalError as e:
                if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                    print("  Index idx_list_timestamp_utc_0 already exists")
                else:
                    raise

        for (lat_key, lon_key), collections in INDEX_LAT_LON.items():
            if collection in collections:
                idx_name = f"idx_{lat_key}_{lon_key}".replace(".", "_")
                try:
                    conn.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {idx_name}
                        ON store (json_extract(document, '$.{lat_key}'), json_extract(document, '$.{lon_key}'))
                        """
                    )
                    print(f"  Created index {idx_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                        print(f"  Index {idx_name} already exists")
                    else:
                        raise

        if collection == "trigger_point":
            try:
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_name
                    ON store (json_extract(document, '$.name'))
                    """
                )
                print("  Created index idx_name")
            except sqlite3.OperationalError as e:
                if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                    print("  Index idx_name already exists")
                else:
                    raise

        conn.execute("ANALYZE")
        print("  ANALYZE done")


def main() -> None:
    path_dir = Path("data", "database_sqlite")
    db_files = sorted(path_dir.rglob("*.db"))
    if not db_files:
        if not path_dir.exists():
            print(f"Directory not found: {path_dir}")
        else:
            print(f"No .db files in {path_dir}")
        return
    for path_db in db_files:
        optimize_db(path_db)
    print("Done.")


if __name__ == "__main__":
    main()

# Run:
# poetry run python script/database/optimize_sqlite.py