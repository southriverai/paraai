import math

import numpy as np
from pydantic import BaseModel


class AthmosphereModel(BaseModel):
    def init(self):
        # Day of year thermal strength parameters
        self._THERMAL_STRENGTH_C = 0.9187
        self._THERMAL_STRENGTH_A1 = 0.1619
        self._THERMAL_STRENGTH_PHI1 = -1.2368
        self._THERMAL_STRENGTH_A2 = 0.0375
        self._THERMAL_STRENGTH_PHI2 = -1.3013
        self._THERMAL_STRENGTH_SIZE = 1000

        # Climb profile parameters
        self._CLIMB_PROFILE_DOMAIN = []
        self._CLIMB_PROFILE_VALUE = []

    def day_strength_m_s_mean(self, day_of_year: int) -> float:
        """Return average thermal strength (vertical speed m/s) for day of year 0-364.
        Based on two-sine fit: c + A1*sin(2πx/365 + φ1) + A2*sin(4πx/365 + φ2).
        """
        x = day_of_year % 365
        t1 = 2 * math.pi * x / 365 + self._THERMAL_STRENGTH_PHI1
        t2 = 4 * math.pi * x / 365 + self._THERMAL_STRENGTH_PHI2
        day_strength_m_s = self._THERMAL_STRENGTH_C
        day_strength_m_s += self._THERMAL_STRENGTH_A1 * math.sin(t1)
        day_strength_m_s += self._THERMAL_STRENGTH_A2 * math.sin(t2)
        return day_strength_m_s

    def climb_profile_multiplier(
        self,
        climb_start_altitude_m: float,
        climb_end_altitude_m: float,
        altitude_m: float,
    ) -> float:
        """Give an altitude in meters and a ceiling in meters, return the multiplier for the climb speed.
        The multiplier is a function of the fraction of the total altitude and the ceiling.
        The multiplier is 1 at the ceiling and 0 at the climb start altitude.
        """
        fraction_of_total_altitude = (altitude_m - climb_start_altitude_m) / (climb_end_altitude_m - climb_start_altitude_m)
        return np.clip(fraction_of_total_altitude, 0, 1)

    def sample_day_strength_m_s(self, day_of_year: int) -> float:
        """Sample a day strength (vertical speed m/s) for day of year 0-364. Daystength is chiquare distributed around the mean."""
        day_strength_m_s_mean = self.day_strength_m_s_mean(day_of_year)
        return np.random.chisquare(day_strength_m_s_mean, self._THERMAL_STRENGTH_SIZE)

    def sample_distance_between_thermals_m(self) -> float:
        """Sample a distance between thermals in meters. Distance is chisquare distributed around the mean. The mean is 1000 meters."""
        distance_between_thermals_m_mean = 1000
        return np.random.chisquare(distance_between_thermals_m_mean, 1000)

    def climb_speed_m_s(
        self,
        thermal_strength_m_s: float,
        ceiling_m: float,
        altitude_m: float,
        day_of_year_d: int,
        time_of_day_s: int,
    ) -> float:
        """Give a thermal base strength in m/s comput the multiplier resulting from:
        - altitude in the climbs,
        - the day of the year ranging from 0 to 364 in days after january 1st,
        - the time of the day second from 0 to 86400 in seconds after midnight.
        Return the actual climb speed in m/s."""

        climb_profile_multiplier = self.climb_profile_multiplier(altitude_m, 0, ceiling_m)
        time_strength_m_s = self.sample_time_strength_m_s(time_of_day_s)
        altitude_strength_m_s = self.sample_altitude_strength_m_s(altitude_m)
        thermal_strength_m_s *= climb_profile_multiplier
        return thermal_strength_m_s
