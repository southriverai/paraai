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
    filename: Optional[str] = None  # single filename for existence check
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

    async def get_all(self) -> list[TracklogHeader]:
        keys = list(self.store.yield_keys())
        batch_size = 500  # SQLite limit on SQL variables
        results: list[TracklogHeader] = []
        for i in range(0, len(keys), batch_size):
            batch = keys[i : i + batch_size]
            results.extend(h for h in self.store.mget(batch) if h is not None)
        return results

    def insert(self, tracklog_header: TracklogHeader):
        self.store.set(tracklog_header.tracklog_id, tracklog_header)

    def _build_store_query(self, request: TracklogHeaderQueryRequest) -> dict:
        """Build MongoDB-style query dict for the underlying store."""
        result: dict = {}
        if request.tracklog_ids:
            result["tracklog_id"] = {"$in": request.tracklog_ids}
        if request.filenames:
            result["file_name"] = {"$in": request.filenames}
        elif request.filename:
            result["file_name"] = {"$eq": request.filename}
        return result

    def query(self, request: TracklogHeaderQueryRequest) -> TracklogHeaderQueryResponse:
        store_query = self._build_store_query(request)
        limit = request.limit if request.limit > 0 else 0
        tracklogs = self.store.query(store_query, None, limit, request.skip)
        return TracklogHeaderQueryResponse(
            tracklogs=tracklogs,
            total=len(tracklogs),
            skip=request.skip,
            limit=request.limit,
        )

    def query_one(self, request: TracklogHeaderQueryRequest) -> Optional[TracklogHeader]:
        response = self.query(request)
        return response.tracklogs[0] if response.tracklogs else None

    async def asample(self, count: int) -> list[TracklogHeader]:
        return await self.store.asample(count)
