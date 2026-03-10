"""Spacetime utilities: UTC/solar time, haversine distance, and related computations."""

import math
from datetime import datetime


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


def utc_to_solar_hour(dt: datetime, longitude_deg: float) -> float:
    """Convert UTC to local solar hour (0-24) using longitude. 15° = 1 hour."""
    utc_hour_frac = dt.hour + dt.minute / 60 + dt.second / 3600
    solar_hour = utc_hour_frac + longitude_deg / 15
    return solar_hour % 24
