import uuid
from datetime import datetime, timezone

from pydantic import BaseModel

from paraai.tool_spacetime import utc_to_day_of_year, utc_to_solar_hour


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

    @property
    def time_of_day_h(self) -> float:
        """Heliocentric time of day (0-24h) from ground lon, midnight to midnight."""
        dt = datetime.fromtimestamp(self.start_timestamp_utc, tz=timezone.utc)
        return utc_to_solar_hour(dt, self.ground_lon)

    @property
    def time_of_year_d(self) -> float:
        """Day of year (0 = Jan 1, 365.x = Dec 31)."""
        dt = datetime.fromtimestamp(self.start_timestamp_utc, tz=timezone.utc)
        return utc_to_day_of_year(dt)
