from pydantic import BaseModel
from scipy.interpolate import interp1d


class Thermal1D(BaseModel):
    thermal_id: str
    location_m: float
    width_m: float
    air_speed_value_m_s: list[float]

    def get_air_speed_m_s(self, location_m: float) -> float:
        # check if we are in the reach of the thermal
        if location_m < self.location_m - self.width_m / 2:
            return self.air_speed_value_m_s[0]
        if location_m > self.location_m + self.width_m / 2:
            return self.air_speed_value_m_s[-1]
        return interp1d(self.location_domain_m, self.air_speed_value_m_s)(location_m)
