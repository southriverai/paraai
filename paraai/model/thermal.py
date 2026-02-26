from pydantic import BaseModel


class Thermal(BaseModel):
    distance_m: float
    thermal_strength_m_s: float
