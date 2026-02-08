from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from srai_store.store_provider_base import StoreProviderBase
from srai_store.store_provider_disk import StoreProviderDisk
from srai_store.store_provider_sqlite import StoreProviderSqlite

from paraai.model.tracklog import TracklogBody


class TracklogBodyQueryRequest(BaseModel):
    tracklog_ids: list[str]
    skip: int = 0
    limit: int = 100


class TracklogBodyQueryResponse(BaseModel):
    tracklogs: list[TracklogBody] = []
    total: int = 0
    skip: int = 0
    limit: int = 100


class RepositoryTracklogBody:
    instance: Optional["RepositoryTracklogBody"] = None

    def __init__(self, store_provider: StoreProviderBase):
        self.store_provider = store_provider
        self.store = self.store_provider.get_object_store("tracklog_body", TracklogBody)
        self.store_clean = self.store_provider.get_object_store("tracklog_body_clean", TracklogBody)

    @staticmethod
    def initialize(store_provider: StoreProviderBase) -> "RepositoryTracklogBody":
        if RepositoryTracklogBody.instance is not None:
            raise ValueError("RepositoryTrackLog already initialized")
        RepositoryTracklogBody.instance = RepositoryTracklogBody(store_provider)
        return RepositoryTracklogBody.instance

    @staticmethod
    def initialize_sqlite(path_dir_database: Path) -> "RepositoryTracklogBody":
        store_provider = StoreProviderSqlite("tracklogs", path_dir_database)
        return RepositoryTracklogBody.initialize(store_provider)

    @staticmethod
    def initialize_disk(path_dir_database: Path) -> "RepositoryTracklogBody":
        store_provider = StoreProviderDisk("tracklogs", path_dir_database)
        return RepositoryTracklogBody.initialize(store_provider)

    @staticmethod
    def get_instance() -> "RepositoryTracklogBody":
        if not hasattr(RepositoryTracklogBody, "instance"):
            raise ValueError("RepositoryTracklogBody not initialized")
        return RepositoryTracklogBody.instance

    def insert(self, tracklog_body: TracklogBody):
        self.store.set(tracklog_body.tracklog_id, tracklog_body)

    def get_raise(self, tracklog_id: str) -> TracklogBody:
        return self.store.get_raise(tracklog_id)

    def query(self, query: TracklogBodyQueryRequest) -> TracklogBodyQueryResponse:
        query = {
            "filter": {
                "track_log_ids": query.track_log_id,
            },
            "skip": query.skip,
            "limit": query.limit,
        }
        return self.store.query(query)

    async def asample(self, count: int) -> list[TracklogBody]:
        return await self.store.asample(count)
