"""Add indexes and performance PRAGMAs to srai_store SQLite databases.

Run once after ingestion. Indexes and WAL mode persist. Speeds up:
- ORDER BY on timestamp fields (earliest/latest queries)
- Bounding box queries on start_lat/start_lon (simple_climb)
- Key prefix scans (e.g. tracklog_id_*)
- General read throughput

Usage:
    poetry run python script/database/optimize_sqlite.py
"""

import sqlite3
from pathlib import Path

# Collections that have timestamp fields worth indexing (others get PRAGMAs only)
INDEX_START_TS = {"simple_climb"}  # has start_timestamp_utc
INDEX_LIST_TS_0 = {"climb"}  # has list_timestamp_utc[0]
# Collections with lat/lon for bounding box queries
INDEX_START_LAT_LON = {"simple_climb"}  # has start_lat, start_lon


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

        if collection in INDEX_START_LAT_LON:
            try:
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_start_lat_lon
                    ON store (json_extract(document, '$.start_lat'), json_extract(document, '$.start_lon'))
                    """
                )
                print("  Created index idx_start_lat_lon")
            except sqlite3.OperationalError as e:
                if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                    print("  Index idx_start_lat_lon already exists")
                else:
                    raise

        conn.execute("ANALYZE")
        print("  ANALYZE done")


def main() -> None:
    path_dir = Path("data", "database_sqlite") / "tracklogs"
    if not path_dir.exists():
        print(f"Directory not found: {path_dir}")
        return
    for path_db in sorted(path_dir.glob("*.db")):
        optimize_db(path_db)
    print("Done.")


if __name__ == "__main__":
    main()
