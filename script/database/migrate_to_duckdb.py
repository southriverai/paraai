import logging
from pathlib import Path

from pydantic import BaseModel
from srai_store.store_provider_duckdb import StoreProviderDuckdb
from srai_store.store_provider_sqlite import StoreProviderSqlite
from tqdm import tqdm

from paraai.model.simple_climb import SimpleClimb
from paraai.model.simple_climb_pixel import SimpleClimbPixel
from paraai.model.tracklog import TracklogBody, TracklogHeader

logger = logging.getLogger(__name__)


def migrate_store(path_sqlite: Path, path_duckdb: Path, database_name: str, model_class: type[BaseModel]):
    logger.info(f"Migrating {database_name} from SQLite to DuckDB")
    store_provider_sqlite = StoreProviderSqlite(database_name, path_sqlite)
    store_provider_duckdb = StoreProviderDuckdb(database_name, path_duckdb)
    store_sqlite = store_provider_sqlite.get_object_store(database_name, model_class)
    store_duckdb = store_provider_duckdb.get_object_store(database_name, model_class)
    keys = list(store_sqlite.yield_keys())
    if len(keys) == 0:
        raise ValueError(f"SQLite store {database_name!r} has no keys to migrate")
    batch_size = 500
    with tqdm(total=len(keys), desc=f"Migrating {database_name}", unit="row") as pbar:
        for i in range(0, len(keys), batch_size):
            chunk = keys[i : i + batch_size]
            objects = store_sqlite.mget(chunk)
            pairs = [(k, obj) for k, obj in zip(chunk, objects) if obj is not None]
            if pairs:
                store_duckdb.mset(pairs)
            pbar.update(len(chunk))


def migrate_database():
    logger.info("Migrating database to DuckDB")
    path_sqlite = Path("data", "database_sqlite")
    path_duckdb = Path("data", "database_duckdb")
    migrate_store(path_sqlite, path_duckdb, "simple_climb", SimpleClimb)
    migrate_store(path_sqlite, path_duckdb, "simple_climb_pixel", SimpleClimbPixel)
    migrate_store(path_sqlite, path_duckdb, "tracklog_body", TracklogBody)
    migrate_store(path_sqlite, path_duckdb, "tracklog_header", TracklogHeader)


if __name__ == "__main__":
    migrate_database()
