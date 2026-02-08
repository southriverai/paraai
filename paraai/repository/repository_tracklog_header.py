from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from srai_store.store_provider_base import StoreProviderBase
from srai_store.store_provider_disk import StoreProviderDisk
from srai_store.store_provider_sqlite import StoreProviderSqlite

from paraai.model.tracklog import TracklogHeader


class TracklogHeaderQueryRequest(BaseModel):
    tracklog_ids: Optional[list[str]] = None
    filenames: Optional[list[str]] = None
    skip: int = 0
    limit: int = 100


class TracklogHeaderQueryResponse(BaseModel):
    tracklogs: list[TracklogHeader] = []
    total: int = 0
    skip: int = 0
    limit: int = 100


class RepositoryTracklogHeader:
    instance: Optional["RepositoryTracklogHeader"] = None

    def __init__(self, store_provider: StoreProviderBase):
        self.store_provider = store_provider
        self.store = self.store_provider.get_object_store("tracklog_header", TracklogHeader)

    @staticmethod
    def initialize(store_provider: StoreProviderBase) -> "RepositoryTracklogHeader":
        if RepositoryTracklogHeader.instance is not None:
            raise ValueError("RepositoryTrackLog already initialized")
        RepositoryTracklogHeader.instance = RepositoryTracklogHeader(store_provider)
        return RepositoryTracklogHeader.instance

    @staticmethod
    def initialize_sqlite(path_dir_database: Path) -> "RepositoryTracklogHeader":
        store_provider = StoreProviderSqlite("tracklogs", path_dir_database)
        return RepositoryTracklogHeader.initialize(store_provider)

    @staticmethod
    def initialize_disk(path_dir_database: Path) -> "RepositoryTracklogHeader":
        store_provider = StoreProviderDisk("tracklogs", path_dir_database)
        return RepositoryTracklogHeader.initialize(store_provider)

    @staticmethod
    def get_instance() -> "RepositoryTracklogHeader":
        if not hasattr(RepositoryTracklogHeader, "instance"):
            raise ValueError("RepositoryTracklogHeader not initialized")
        return RepositoryTracklogHeader.instance

    def insert(self, tracklog_header: TracklogHeader):
        self.store.set(tracklog_header.tracklog_id, tracklog_header)

    def _build_query(self, query: TracklogHeaderQueryRequest) -> dict:
        filter = {}
        if query.tracklog_ids:
            filter["tracklog_ids"] = query.tracklog_ids
        if query.filenames:
            filter["filenames"] = query.filenames
        query_dict = {
            "filter": {
                "filter": filter,
            },
            "skip": query.skip,
            "limit": query.limit,
        }
        return query_dict

    def query(self, query: TracklogHeaderQueryRequest) -> TracklogHeaderQueryResponse:
        return self.store.query(self._build_query(query))

    def query_one(self, query: TracklogHeaderQueryRequest) -> Optional[TracklogHeader]:
        return self.store.get(self._build_query(query))

    async def asample(self, count: int) -> list[TracklogHeader]:
        return await self.store.asample(count)
