from pydantic import BaseModel


class BoundingBox(BaseModel):
    """Bounding box of a region."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

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
