"""Spacetime utilities: UTC/solar time, haversine distance, and related computations."""

import math
from datetime import datetime

import numpy as np

from paraai.model.boundingbox import REGION_DICT, BoundingBox, BoundingBoxRegion

# Europe bounds (lat min, lat max, lon min, lon max): ~Iberia to Nordkapp, Ireland to Urals
EUROPE_BOUNDS: tuple[float, float, float, float] = (36.0, 72.0, -11.0, 42.0)


# Alps bounds (lat min, lat max, lon min, lon max): ~Alps to Pyrenees
ALPS_BOUNDS: tuple[float, float, float, float] = (45.0, 49.0, 5.0, 15.0)

# Bassano del Grappa (Monte Grappa, Semonzo) ~50km bbox
BASSANO_BOUNDS: tuple[float, float, float, float] = (45.5, 46.0, 11.4, 12.1)

# Sopot, Bulgaria ~50km bbox
SOPOT_BOUNDS: tuple[float, float, float, float] = (42.4, 42.9, 24.4, 25.1)

# Bansko, Bulgaria ~50km bbox
BANSKO_BOUNDS: tuple[float, float, float, float] = (41.5, 42.2, 23.2, 23.8)

REGION_BOUNDS: dict[str, BoundingBox] = {
    "europe": BoundingBox(lat_min=EUROPE_BOUNDS[0], lat_max=EUROPE_BOUNDS[1], lon_min=EUROPE_BOUNDS[2], lon_max=EUROPE_BOUNDS[3]),
    "bassano": BoundingBox(lat_min=BASSANO_BOUNDS[0], lat_max=BASSANO_BOUNDS[1], lon_min=BASSANO_BOUNDS[2], lon_max=BASSANO_BOUNDS[3]),
    "sopot": BoundingBox(lat_min=SOPOT_BOUNDS[0], lat_max=SOPOT_BOUNDS[1], lon_min=SOPOT_BOUNDS[2], lon_max=SOPOT_BOUNDS[3]),
    "bansko": BoundingBox(lat_min=BANSKO_BOUNDS[0], lat_max=BANSKO_BOUNDS[1], lon_min=BANSKO_BOUNDS[2], lon_max=BANSKO_BOUNDS[3]),
}


def get_bounding_box(region: str) -> BoundingBox:
    """Return BoundingBox for region. Raises ValueError if unknown."""
    bbox = REGION_BOUNDS.get(region.lower())
    if bbox is None:
        raise ValueError(f"Unknown region '{region}'. Available: {list(REGION_BOUNDS)}")
    return bbox


def get_region_bounding_box(region_name: str) -> BoundingBoxRegion | None:
    return REGION_DICT.get(region_name.lower())


def is_in_region(region_name: str, lat: float, lon: float) -> bool:
    region_bounding_box = get_region_bounding_box(region_name)
    if region_bounding_box is None:
        return False

    return bounding_box.is_in(lat, lon)


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Distance in meters between two lat/lng points (haversine formula)."""
    R = 6_371_000  # Earth radius m
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Distance in km between two lat/lng points (haversine formula)."""
    return haversine_m(lat1, lng1, lat2, lng2) / 1000


def haversine_km_tuple(lat_lon_pair_1: tuple[float, float], lat_lon_pair_2: tuple[float, float]) -> float:
    """Distance in km between two lat/lng points (haversine formula)."""
    return haversine_m(lat_lon_pair_1[0], lat_lon_pair_1[1], lat_lon_pair_2[0], lat_lon_pair_2[1]) / 1000


# 1 arc-second ≈ 30.8 m at equator. Copernicus DEM resolution.
METERS_PER_DEG_LAT = 111_000
RESOLUTION_DEG = 1 / 3600


def dem_pixel_size_m(lat: float) -> tuple[float, float]:
    """Return (width_m, height_m) of a 1 arc-second DEM pixel at latitude.

    Width (lon direction) shrinks with cos(lat); height (lat direction) is constant.
    """
    width_m = METERS_PER_DEG_LAT * RESOLUTION_DEG * math.cos(math.radians(lat))
    height_m = METERS_PER_DEG_LAT * RESOLUTION_DEG
    return width_m, height_m


def build_gaussian_kernel_meters(
    sigma_m: float,
    lat: float,
    lon: float,
    *,
    pixel_size_deg: float = RESOLUTION_DEG,
) -> np.ndarray:
    """Build 2D Gaussian convolution kernel with sigma in meters.

    Accounts for pixel dimensions at (lat, lon): pixels are narrower in the
    longitude direction at higher latitudes.

    Returns:
        Normalized kernel array (2D), odd size, sum=1.
    """
    width_m, height_m = dem_pixel_size_m(lat)
    sigma_x_pixels = sigma_m / width_m
    sigma_y_pixels = sigma_m / height_m
    radius_x = int(math.ceil(3 * sigma_x_pixels))
    radius_y = int(math.ceil(3 * sigma_y_pixels))
    size_x = 2 * radius_x + 1
    size_y = 2 * radius_y + 1
    y = np.arange(size_y, dtype=np.float64) - radius_y
    x = np.arange(size_x, dtype=np.float64) - radius_x
    yy, xx = np.meshgrid(y, x, indexing="ij")
    kernel = np.exp(-(xx**2 / (2 * sigma_x_pixels**2) + yy**2 / (2 * sigma_y_pixels**2)))
    kernel /= kernel.sum()
    return kernel


def utc_to_solar_hour(dt: datetime, longitude_deg: float) -> float:
    """Convert UTC to local solar hour (0-24) using longitude. 15° = 1 hour."""
    utc_hour_frac = dt.hour + dt.minute / 60 + dt.second / 3600
    solar_hour = utc_hour_frac + longitude_deg / 15
    return solar_hour % 24


def utc_to_day_of_year(dt: datetime) -> float:
    """Convert UTC datetime to day of year as float. 0 = Jan 1 00:00, 365.x = Dec 31."""
    tt = dt.timetuple()
    day_num = tt.tm_yday  # 1-366
    frac = (dt.hour + dt.minute / 60 + dt.second / 3600) / 24
    return (day_num - 1) + frac
