import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Optional

from srai_store.store_provider_base import StoreProviderBase
from srai_store.store_provider_sqlite import StoreProviderSqlite
from tqdm import tqdm

from paraai.model.boundingbox import BoundingBox
from paraai.model.simple_climb import SimpleClimb
from paraai.repository.repository_cache import RepositoryCache
from paraai.tool_spacetime import haversine_m

METERS_PER_DEG_LAT = 111_000

logger = logging.getLogger(__name__)


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
        store_provider = StoreProviderSqlite("simple_climb", path_dir_database)
        return RepositorySimpleClimb.initialize(store_provider)

    @staticmethod
    def get_instance() -> "RepositorySimpleClimb":
        if not hasattr(RepositorySimpleClimb, "instance") or RepositorySimpleClimb.instance is None:
            raise ValueError("RepositorySimpleClimb not initialized")
        return RepositorySimpleClimb.instance

    BATCH_SIZE = 500_000  # SQLite limit on bound parameters (~999) # TODO this seems wrong

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

    async def get_all(self, verbose: bool = False) -> list[SimpleClimb]:
        keys = list(self.store.yield_keys())
        results: list[SimpleClimb] = []
        for i in range(0, len(keys), self.BATCH_SIZE):
            if verbose:
                logger.info(f"Querying {i} to {i + self.BATCH_SIZE}")
            batch = keys[i : i + self.BATCH_SIZE]
            results.extend(c for c in self.store.mget(batch) if c is not None)
        return results

    async def get_all_in_bounding_box(
        self,
        lat_deg_min: float,
        lat_deg_max: float,
        lng_deg_min: float,
        lng_deg_max: float,
        *,
        verbose: bool = False,
        ignore_cache: bool = False,
    ) -> list[SimpleClimb]:
        query = {
            "start_lat": {"$gte": lat_deg_min, "$lte": lat_deg_max},
            "start_lon": {"$gte": lng_deg_min, "$lte": lng_deg_max},
        }
        return self._query_all(query, verbose=verbose, ignore_cache=ignore_cache)

    async def get_all_in_bounding_box_by_ground(
        self,
        bounding_box: BoundingBox,
        *,
        verbose: bool = False,
        ignore_cache: bool = False,
    ) -> list[SimpleClimb]:
        lat_deg_min = bounding_box.lat_min
        lat_deg_max = bounding_box.lat_max
        lng_deg_min = bounding_box.lon_min
        lng_deg_max = bounding_box.lon_max
        query = {
            "ground_lat": {"$gte": bounding_box.lat_min, "$lte": bounding_box.lat_max},
            "ground_lon": {"$gte": bounding_box.lon_min, "$lte": bounding_box.lon_max},
        }
        if not ignore_cache:
            try:
                repo_cache = RepositoryCache.get_instance()
                key = self._query_cache_key(query)
                cached = repo_cache.get(key)
                if cached is not None and "climbs" in cached:
                    climbs = [SimpleClimb.model_validate(d) for d in cached["climbs"]]
                    logger.debug("Loaded %d climbs from query cache (bbox_ground)", len(climbs))
                    return climbs
            except Exception as e:
                logger.debug("Query cache miss or error: %s", e)

        keys = list(self.store.yield_keys())
        results: list[SimpleClimb] = []
        batch_size = 1000  # seems to be differnt for different queries
        iterator = (
            tqdm(range(0, len(keys), batch_size), desc="Loading climbs", unit="batch") if verbose else range(0, len(keys), batch_size)
        )
        for i in iterator:
            batch = keys[i : i + batch_size]
            batch_results = self.store.mget(batch)
            filtered = [
                c for c in batch_results if lat_deg_min <= c.ground_lat <= lat_deg_max and lng_deg_min <= c.ground_lon <= lng_deg_max
            ]
            results.extend(filtered)

        if not ignore_cache:
            try:
                repo_cache = RepositoryCache.get_instance()
                key = self._query_cache_key(query)
                repo_cache.set(key, {"climbs": [c.model_dump() for c in results]})
                logger.debug("Cached %d climbs for query (bbox_ground)", len(results))
            except Exception as e:
                logger.debug("Failed to cache query results: %s", e)

        return results

    def _query_cache_key(self, query: dict) -> str:
        """Stable cache key from query dict."""
        qstr = json.dumps(query, sort_keys=True)
        h = hashlib.sha256(qstr.encode()).hexdigest()[:32]
        return f"simple_climb_query_{h}"

    def _query_all(
        self,
        query: dict,
        *,
        verbose: bool = False,
        ignore_cache: bool = False,
    ) -> list[SimpleClimb]:
        """Query all matching climbs. Results are cached in RepositoryCache unless ignore_cache=True."""
        if not ignore_cache:
            try:
                repo_cache = RepositoryCache.get_instance()
                key = self._query_cache_key(query)
                cached = repo_cache.get(key)
                if cached is not None and "climbs" in cached:
                    climbs = [SimpleClimb.model_validate(d) for d in cached["climbs"]]
                    logger.debug("Loaded %d climbs from query cache", len(climbs))
                    return climbs
            except Exception as e:
                logger.debug("Query cache miss or error: %s", e)

        results: list[SimpleClimb] = []
        skip = 0
        while True:
            if verbose:
                logger.info("Querying %s to %s", skip, skip + self.BATCH_SIZE)
            batch = self.store.query(query, None, self.BATCH_SIZE, skip)
            results.extend(batch)
            if len(batch) < self.BATCH_SIZE:
                break
            skip += self.BATCH_SIZE

        if not ignore_cache:
            try:
                repo_cache = RepositoryCache.get_instance()
                key = self._query_cache_key(query)
                repo_cache.set(key, {"climbs": [c.model_dump() for c in results]})
                logger.debug("Cached %d climbs for query", len(results))
            except Exception as e:
                logger.debug("Failed to cache query results: %s", e)

        return results

    async def get_all_in_radius(self, lat_deg: float, lng_deg: float, radius_m: float) -> list[SimpleClimb]:
        deg_lat = radius_m / METERS_PER_DEG_LAT
        deg_lon = radius_m / (METERS_PER_DEG_LAT * math.cos(math.radians(lat_deg)))
        bounding_box = await self.get_all_in_bounding_box(lat_deg - deg_lat, lat_deg + deg_lat, lng_deg - deg_lon, lng_deg + deg_lon)
        results: list[SimpleClimb] = []
        for c in bounding_box:
            d = haversine_m(c.start_lat, c.start_lon, lat_deg, lng_deg)
            if d <= radius_m:
                results.append(c)
        return results

    async def get_all_in_radius_by_ground(self, lat_deg: float, lng_deg: float, radius_m: float) -> list[SimpleClimb]:
        """Query climbs whose ground point (trigger) is within radius."""
        deg_lat = radius_m / METERS_PER_DEG_LAT
        deg_lon = radius_m / (METERS_PER_DEG_LAT * math.cos(math.radians(lat_deg)))
        bbox = await self.get_all_in_bounding_box_by_ground(lat_deg - deg_lat, lat_deg + deg_lat, lng_deg - deg_lon, lng_deg + deg_lon)
        return [c for c in bbox if haversine_m(c.ground_lat, c.ground_lon, lat_deg, lng_deg) <= radius_m]

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
