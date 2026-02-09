from pydantic import BaseModel


class Climb(BaseModel):
    tracklog_id: str
    climb_index: int
    lat: float
    lng: float
    list_timestamp_utc: list[int]
    list_altitude_m: list[float]
