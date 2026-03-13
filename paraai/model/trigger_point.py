"""Trigger point model: a named location with associated climbs."""

from pydantic import BaseModel

from paraai.model.simple_climb import SimpleClimb


class TriggerPoint(BaseModel):
    """A named location (lat, lon) with its associated climbs cached for fast access."""

    trigger_point_id: str
    name: str
    lat: float
    lon: float
    altitude_m: float
    radius_m: float
    climbs: list[SimpleClimb]
