import hashlib

from pydantic import BaseModel

from paraai.model.simple_climb import SimpleClimb

# Copernicus DEM: 1 arc-second ≈ 30m. 1/3600 degree.
RESOLUTION_DEG = 1 / 3600


def snap_to_dem_pixel_center(lat: float, lon: float) -> tuple[float, float]:
    """Snap (lat, lon) to nearest DEM pixel center (1 arc-second)."""
    lat_center = round(lat / RESOLUTION_DEG) * RESOLUTION_DEG
    lon_center = round(lon / RESOLUTION_DEG) * RESOLUTION_DEG
    return lat_center, lon_center


class SimpleClimbPixel(BaseModel):
    """Simple climb pixel matching DEM heightmap (WGS84, 30m Copernicus)."""
    simple_climb_pixel_id: str
    lat: float
    lon: float
    mean_climb_strength_m_s: float
    climb_count: int

    def add_climb(self, climb: SimpleClimb) -> None:
        self.mean_climb_strength_m_s = (self.mean_climb_strength_m_s * self.climb_count + climb.climb_strength_m_s) / (self.climb_count + 1)
        self.climb_count += 1

    @staticmethod
    def get_simple_climb_pixel_id(lat: float, lon: float) -> tuple[str, float, float]:
        lat_center, lon_center = snap_to_dem_pixel_center(lat, lon)
        pixel_id = hashlib.sha256(f"{lat_center}_{lon_center}".encode()).hexdigest()
        return pixel_id, lat_center, lon_center

    @classmethod
    def from_simple_climb(cls, simple_climb: SimpleClimb) -> "SimpleClimbPixel":
        """Create from SimpleClimb, snapping ground point to nearest DEM pixel center."""
        pixel_id, lat_center, lon_center = cls.get_simple_climb_pixel_id(
            simple_climb.ground_lat, simple_climb.ground_lon
        )
        return cls(
            simple_climb_pixel_id=pixel_id,
            lat=lat_center,
            lon=lon_center,
            mean_climb_strength_m_s=simple_climb.climb_strength_m_s,
            climb_count=1,
        )
