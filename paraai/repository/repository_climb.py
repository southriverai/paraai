from pathlib import Path
from typing import Optional

from srai_store.store_provider_base import StoreProviderBase
from srai_store.store_provider_sqlite import StoreProviderSqlite

from paraai.model.climb import Climb


class RepositoryClimb:
    instance: Optional["RepositoryClimb"] = None

    def __init__(self, store_provider: StoreProviderBase):
        self.store_provider = store_provider
        self.store = self.store_provider.get_object_store("climb", Climb)

    @staticmethod
    def initialize(store_provider: StoreProviderBase) -> "RepositoryClimb":
        if RepositoryClimb.instance is not None:
            raise ValueError("RepositoryClimb already initialized")
        RepositoryClimb.instance = RepositoryClimb(store_provider)
        return RepositoryClimb.instance

    @staticmethod
    def initialize_sqlite(path_dir_database: Path) -> "RepositoryClimb":
        store_provider = StoreProviderSqlite("tracklogs", path_dir_database)
        return RepositoryClimb.initialize(store_provider)

    @staticmethod
    def get_instance() -> "RepositoryClimb":
        if not hasattr(RepositoryClimb, "instance") or RepositoryClimb.instance is None:
            raise ValueError("RepositoryClimb not initialized")
        return RepositoryClimb.instance

    def _climb_key(self, tracklog_id: str, climb_index: int) -> str:
        return f"{tracklog_id}_{climb_index}"

    def insert(self, climb: Climb) -> None:
        key = self._climb_key(climb.tracklog_id, climb.climb_index)
        self.store.set(key, climb)

    def insert_many(self, climbs: list[Climb]) -> None:
        for climb in climbs:
            self.insert(climb)

    def delete_climbs_for_tracklog(self, tracklog_id: str) -> None:
        prefix = f"{tracklog_id}_"
        keys_to_delete = [k for k in self.store.yield_keys() if k.startswith(prefix)]
        if keys_to_delete:
            self.store.mdelete(keys_to_delete)
