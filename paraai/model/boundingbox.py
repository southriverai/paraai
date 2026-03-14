from pydantic import BaseModel


class BoundingBox(BaseModel):
    """Bounding box of a region."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def is_in(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max

    def __str__(self) -> str:
        return f"BoundingBox(lat_min={self.lat_min}, lon_min={self.lon_min}, lat_max={self.lat_max}, lon_max={self.lon_max})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundingBox):
            return False
        return (
            self.lat_min == other.lat_min
            and self.lat_max == other.lat_max
            and self.lon_min == other.lon_min
            and self.lon_max == other.lon_max
        )

    def __hash__(self) -> int:
        return hash((self.lat_min, self.lat_max, self.lon_min, self.lon_max))

    def __contains__(self, point: tuple[float, float]) -> bool:
        lat, lon = point
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max


class BoundingBoxRegion(BaseModel):
    """Bounding box of a region."""

    region_name: str
    bounding_box: BoundingBox

    def __str__(self) -> str:
        return f"Rregion_name={self.region_name}, bounding_box={self.bounding_box})"

    @staticmethod
    def get_region_dict() -> dict[str, "BoundingBoxRegion"]:
        region_list: list[BoundingBoxRegion] = []
        region_list.append(
            BoundingBoxRegion(
                region_name="europe",
                bounding_box=BoundingBox(
                    lat_min=36.0,
                    lat_max=72.0,
                    lon_min=-11.0,
                    lon_max=42.0,
                ),
            )
        )
        region_list.append(
            BoundingBoxRegion(
                region_name="bassano",
                bounding_box=BoundingBox(
                    lat_min=45.5,
                    lat_max=46.0,
                    lon_min=11.4,
                    lon_max=12.1,
                ),
            )
        )
        region_list.append(
            BoundingBoxRegion(
                region_name="sopot",
                bounding_box=BoundingBox(
                    lat_min=42.4,
                    lat_max=42.9,
                    lon_min=24.4,
                    lon_max=25.1,
                ),
            )
        )
        region_list.append(
            BoundingBoxRegion(
                region_name="bansko",
                bounding_box=BoundingBox(
                    lat_min=41.5,
                    lat_max=42.2,
                    lon_min=23.2,
                    lon_max=23.8,
                ),
            )
        )
        return {region.region_name: region for region in region_list}


REGION_DICT: dict[str, BoundingBoxRegion] = BoundingBoxRegion.get_region_dict()
