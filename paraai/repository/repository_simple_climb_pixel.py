"""Repository for SimpleClimbPixel: srai_store SQLite-backed, keyed by simple_climb_pixel_id."""

import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
from srai_store.store_provider_base import StoreProviderBase
from srai_store.store_provider_sqlite import StoreProviderSqlite

from paraai.model.simple_climb_pixel import SimpleClimbPixel
from paraai.tool_spacetime import haversine_m

if TYPE_CHECKING:
    from paraai.model.boundingbox import BoundingBox

METERS_PER_DEG_LAT = 111_000
BATCH_SIZE = 500


class RepositorySimpleClimbPixel:
    instance: Optional["RepositorySimpleClimbPixel"] = None

    def __init__(self, store_provider: StoreProviderBase) -> None:
        self.store_provider = store_provider
        self.store = self.store_provider.get_object_store("simple_climb_pixel", SimpleClimbPixel)

    @staticmethod
    def initialize(store_provider: StoreProviderBase) -> "RepositorySimpleClimbPixel":
        if RepositorySimpleClimbPixel.instance is not None:
            raise ValueError("RepositorySimpleClimbPixel already initialized")
        RepositorySimpleClimbPixel.instance = RepositorySimpleClimbPixel(store_provider)
        return RepositorySimpleClimbPixel.instance

    @staticmethod
    def initialize_sqlite(path_dir_database: Path) -> "RepositorySimpleClimbPixel":
        store_provider = StoreProviderSqlite("simple_climb_pixel", path_dir_database)
        return RepositorySimpleClimbPixel.initialize(store_provider)

    @staticmethod
    def get_instance() -> "RepositorySimpleClimbPixel":
        if not hasattr(RepositorySimpleClimbPixel, "instance") or RepositorySimpleClimbPixel.instance is None:
            raise ValueError("RepositorySimpleClimbPixel not initialized")
        return RepositorySimpleClimbPixel.instance

    def upsert(self, pixel: SimpleClimbPixel) -> None:
        self.store.set(pixel.simple_climb_pixel_id, pixel)

    def upsert_many(self, pixels: list[SimpleClimbPixel]) -> None:
        for i in range(0, len(pixels), BATCH_SIZE):
            batch = pixels[i : i + BATCH_SIZE]
            pairs = [(p.simple_climb_pixel_id, p) for p in batch]
            self.store.mset(pairs)

    def clear_all(self) -> None:
        keys = list(self.store.yield_keys())
        for i in range(0, len(keys), BATCH_SIZE):
            batch = keys[i : i + BATCH_SIZE]
            self.store.mdelete(batch)

    def get(self, simple_climb_pixel_id: str) -> Optional[SimpleClimbPixel]:
        return self.store.get(simple_climb_pixel_id)

    def get_all(self) -> list[SimpleClimbPixel]:
        keys = list(self.store.yield_keys())
        results: list[SimpleClimbPixel] = []
        for i in range(0, len(keys), BATCH_SIZE):
            batch = keys[i : i + BATCH_SIZE]
            results.extend(p for p in self.store.mget(batch) if p is not None)
        return results

    def get_all_in_bounding_box(
        self, lat_deg_min: float, lat_deg_max: float, lng_deg_min: float, lng_deg_max: float
    ) -> list[SimpleClimbPixel]:
        query = {
            "lat": {"$gte": lat_deg_min, "$lte": lat_deg_max},
            "lon": {"$gte": lng_deg_min, "$lte": lng_deg_max},
        }
        results: list[SimpleClimbPixel] = []
        skip = 0
        while True:
            batch = self.store.query(query, None, BATCH_SIZE, skip)
            results.extend(batch)
            if len(batch) < BATCH_SIZE:
                break
            skip += BATCH_SIZE
        return results

    def get_all_in_radius(self, lat_deg: float, lng_deg: float, radius_m: float) -> list[SimpleClimbPixel]:
        deg_lat = radius_m / METERS_PER_DEG_LAT
        deg_lon = radius_m / (METERS_PER_DEG_LAT * math.cos(math.radians(lat_deg)))
        bbox = self.get_all_in_bounding_box(lat_deg - deg_lat, lat_deg + deg_lat, lng_deg - deg_lon, lng_deg + deg_lon)
        return [p for p in bbox if haversine_m(p.lat, p.lon, lat_deg, lng_deg) <= radius_m]

    def get_climb_dataframe(self, bounding_box: "BoundingBox") -> pd.DataFrame:
        """Load climb pixels in bounding box as DataFrame with lat, lon, count, strength."""
        pixels = self.get_all_in_bounding_box(
            bounding_box.lat_min, bounding_box.lat_max, bounding_box.lon_min, bounding_box.lon_max
        )
        if len(pixels) < 1:
            raise ValueError("No SimpleClimbPixels in region")
        return pd.DataFrame(
            [(p.lat, p.lon, p.climb_count, p.mean_climb_strength_m_s) for p in pixels],
            columns=["lat", "lon", "count", "strength"],
        )
