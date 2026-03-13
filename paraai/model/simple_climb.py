import uuid

from pydantic import BaseModel


class SimpleClimb(BaseModel):
    """Minimal climb representation: start and end points only."""

    simple_climb_id: str
    tracklog_id: str
    start_lat: float
    start_lon: float
    start_alt_m: float
    start_timestamp_utc: int
    end_lat: float
    end_lon: float
    end_alt_m: float
    end_timestamp_utc: int
    duration_sec: int
    height_m: float
    climb_strength_m_s: float
    ground_lat: float
    ground_lon: float
    ground_alt_m: float

    @staticmethod
    def create_id(tracklog_id: str, start_timestamp_utc: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tracklog_id}_{start_timestamp_utc}"))
