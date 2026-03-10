"""Repository for trigger points: srai_store SQLite-backed, keyed by trigger_point_id."""

import asyncio
import json
from pathlib import Path
from typing import Optional

from srai_store.store_provider_base import StoreProviderBase
from srai_store.store_provider_sqlite import StoreProviderSqlite

from paraai.model.simple_climb import SimpleClimb
from paraai.model.trigger_point import TriggerPoint
from paraai.tool_spacetime import haversine_m

BATCH_SIZE = 500


class RepositoryTriggerPoint:
    """Store and retrieve trigger points by trigger_point_id. Name is updatable and queryable."""

    instance: Optional["RepositoryTriggerPoint"] = None

    def __init__(self, store_provider: StoreProviderBase) -> None:
        self.store_provider = store_provider
        self.store = self.store_provider.get_object_store("trigger_point", TriggerPoint)

    @staticmethod
    def initialize(store_provider: StoreProviderBase) -> "RepositoryTriggerPoint":
        if RepositoryTriggerPoint.instance is not None:
            raise ValueError("RepositoryTriggerPoint already initialized")
        RepositoryTriggerPoint.instance = RepositoryTriggerPoint(store_provider)
        return RepositoryTriggerPoint.instance

    @staticmethod
    def initialize_sqlite(path_dir_database: Path) -> "RepositoryTriggerPoint":
        store_provider = StoreProviderSqlite("trigger_point", path_dir_database)
        repo = RepositoryTriggerPoint.initialize(store_provider)
        repo._migrate_from_json(path_dir_database)
        return repo

    @staticmethod
    def get_instance() -> "RepositoryTriggerPoint":
        if not hasattr(RepositoryTriggerPoint, "instance") or RepositoryTriggerPoint.instance is None:
            raise ValueError("RepositoryTriggerPoint not initialized")
        return RepositoryTriggerPoint.instance

    def _migrate_from_json(self, path_dir: Path) -> None:
        """Migrate existing JSON files to store if empty."""
        keys = list(self.store.yield_keys())
        if keys:
            return
        for path in Path(path_dir).glob("*.json"):
            try:
                data = json.loads(path.read_text())
                if "trigger_point_id" not in data:
                    data["trigger_point_id"] = data.get("name", path.stem)
                climbs = [SimpleClimb.model_validate(c) for c in data.get("climbs", [])]
                tp = TriggerPoint(
                    trigger_point_id=data["trigger_point_id"],
                    name=data["name"],
                    lat=data["lat"],
                    lon=data["lon"],
                    radius_m=data["radius_m"],
                    climbs=climbs,
                )
                self.insert(tp)
            except (json.JSONDecodeError, KeyError, Exception):
                pass

    def insert(self, trigger_point: TriggerPoint) -> None:
        """Insert or overwrite a single trigger point by trigger_point_id."""
        self.store.set(trigger_point.trigger_point_id, trigger_point)

    async def upsert_many(self, trigger_points: list[TriggerPoint]) -> None:
        """Insert or overwrite trigger points in batches by trigger_point_id."""
        for i in range(0, len(trigger_points), BATCH_SIZE):
            batch = trigger_points[i : i + BATCH_SIZE]
            pairs = [(tp.trigger_point_id, tp) for tp in batch]
            self.store.mset(pairs)
            await asyncio.sleep(0)  # Yield to event loop between batches

    def get(self, trigger_point_id: str) -> Optional[TriggerPoint]:
        """Retrieve a trigger point by ID (primary key). Returns None if not found."""
        return self.store.get(trigger_point_id)

    def get_by_ids(self, trigger_point_ids: list[str]) -> list[TriggerPoint]:
        """Retrieve trigger points by IDs. Returns only found items, order not guaranteed."""
        if not trigger_point_ids:
            return []
        results: list[TriggerPoint] = []
        for i in range(0, len(trigger_point_ids), BATCH_SIZE):
            batch = trigger_point_ids[i : i + BATCH_SIZE]
            results.extend(tp for tp in self.store.mget(batch) if tp is not None)
        return results

    def get_by_name(self, name: str) -> Optional[TriggerPoint]:
        """Query trigger point by name. Returns None if not found."""
        results = self.store.query({"name": name}, limit=1)
        return results[0] if results else None

    def update_name(self, trigger_point_id: str, new_name: str) -> bool:
        """Update the name of a trigger point. Returns False if not found."""
        tp = self.get(trigger_point_id)
        if tp is None:
            return False
        tp.name = new_name
        self.insert(tp)
        return True

    def get_all(self) -> list[TriggerPoint]:
        """Return all stored trigger points."""
        keys = list(self.store.yield_keys())
        results: list[TriggerPoint] = []
        for i in range(0, len(keys), BATCH_SIZE):
            batch = keys[i : i + BATCH_SIZE]
            results.extend(tp for tp in self.store.mget(batch) if tp is not None)
        return results

    def get_all_within_radius(self, lat: float, lon: float, radius_m: float) -> list[TriggerPoint]:
        """Return all trigger points within radius of lat, lon."""
        keys = list(self.store.yield_keys())
        results: list[TriggerPoint] = []
        for i in range(0, len(keys), BATCH_SIZE):
            batch = keys[i : i + BATCH_SIZE]
            results.extend(tp for tp in self.store.mget(batch) if tp is not None and haversine_m(tp.lat, tp.lon, lat, lon) <= radius_m)
        return results

    def get_all_ids(self) -> list[str]:
        """Return all stored trigger point IDs."""
        return list(self.store.yield_keys())

    def get_all_names(self) -> list[str]:
        """Return all stored trigger point names."""
        return [tp.name for tp in self.get_all()]
