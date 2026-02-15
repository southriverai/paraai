from pathlib import Path
from typing import Optional

from srai_store.store_provider_base import StoreProviderBase
from srai_store.store_provider_sqlite import StoreProviderSqlite

from paraai.model.simple_climb import SimpleClimb


class RepositorySimpleClimb:
    instance: Optional["RepositorySimpleClimb"] = None

    def __init__(self, store_provider: StoreProviderBase):
        self.store_provider = store_provider
        self.store = self.store_provider.get_object_store("simple_climb", SimpleClimb)

    @staticmethod
    def initialize(store_provider: StoreProviderBase) -> "RepositorySimpleClimb":
        if RepositorySimpleClimb.instance is not None:
            raise ValueError("RepositorySimpleClimb already initialized")
        RepositorySimpleClimb.instance = RepositorySimpleClimb(store_provider)
        return RepositorySimpleClimb.instance

    @staticmethod
    def initialize_sqlite(path_dir_database: Path) -> "RepositorySimpleClimb":
        store_provider = StoreProviderSqlite("tracklogs", path_dir_database)
        return RepositorySimpleClimb.initialize(store_provider)

    @staticmethod
    def get_instance() -> "RepositorySimpleClimb":
        if not hasattr(RepositorySimpleClimb, "instance") or RepositorySimpleClimb.instance is None:
            raise ValueError("RepositorySimpleClimb not initialized")
        return RepositorySimpleClimb.instance

    BATCH_SIZE = 500  # SQLite limit on bound parameters (~999)

    def _climb_key(self, tracklog_id: str, start_timestamp_utc: int) -> str:
        return f"{tracklog_id}_{start_timestamp_utc}"

    def insert(self, climb: SimpleClimb) -> None:
        key = self._climb_key(climb.tracklog_id, climb.start_timestamp_utc)
        self.store.set(key, climb)

    def insert_many(self, climbs: list[SimpleClimb]) -> None:
        entries: list[tuple[str, SimpleClimb]] = []
        for climb in climbs:
            if not climb.simple_climb_id:
                climb.simple_climb_id = SimpleClimb.create_id(climb.tracklog_id, climb.start_timestamp_utc)
            key = self._climb_key(climb.tracklog_id, climb.start_timestamp_utc)
            entries.append((key, climb))
        for i in range(0, len(entries), self.BATCH_SIZE):
            batch = entries[i : i + self.BATCH_SIZE]
            self.store.mset(batch)

    def clear_all(self) -> None:
        """Remove all simple climbs from the repository."""
        keys = list(self.store.yield_keys())
        for i in range(0, len(keys), self.BATCH_SIZE):
            batch = keys[i : i + self.BATCH_SIZE]
            self.store.mdelete(batch)

    def delete_climbs_for_tracklog(self, tracklog_id: str) -> None:
        prefix = f"{tracklog_id}_"
        keys_to_delete = [k for k in self.store.yield_keys() if k.startswith(prefix)]
        for i in range(0, len(keys_to_delete), self.BATCH_SIZE):
            batch = keys_to_delete[i : i + self.BATCH_SIZE]
            self.store.mdelete(batch)

    def get_all(self) -> list[SimpleClimb]:
        keys = list(self.store.yield_keys())
        results: list[SimpleClimb] = []
        for i in range(0, len(keys), self.BATCH_SIZE):
            batch = keys[i : i + self.BATCH_SIZE]
            results.extend(c for c in self.store.mget(batch) if c is not None)
        return results

    async def asample(self, count: int) -> list[SimpleClimb]:
        """Randomly sample simple climbs. Returns up to count."""
        return await self.store.asample(count)

    async def get_earliest_and_latest(self) -> tuple[SimpleClimb, SimpleClimb]:
        """Get earliest and latest by start_timestamp_utc without loading all records."""
        earliest_list = self.store.query({}, order_by=[("start_timestamp_utc", True)], limit=1)
        latest_list = self.store.query({}, order_by=[("start_timestamp_utc", False)], limit=1)
        if not earliest_list or not latest_list:
            raise ValueError("No simple climbs found")
        return earliest_list[0], latest_list[0]
