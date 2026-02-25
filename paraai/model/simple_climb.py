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

    @staticmethod
    def create_id(tracklog_id: str, start_timestamp_utc: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tracklog_id}_{start_timestamp_utc}"))

    def climb_strength_m_s(self) -> float | None:
        """Mean vertical speed (climb strength) in m/s, or None if invalid."""
        duration_s = self.end_timestamp_utc - self.start_timestamp_utc
        alt_gain = self.end_alt_m - self.start_alt_m
        if duration_s <= 0 or alt_gain <= 0:
            return None
        return alt_gain / duration_s
