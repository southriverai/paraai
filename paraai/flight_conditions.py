import numpy as np
from pydantic import BaseModel


class FlightConditions(BaseModel):
    take_off_time_s: float
    take_off_altitude_m: float
    landing_time_s: float
    thermal_ceiling_m: float
    thermal_net_climb_min_m_s: float
    thermal_net_climb_mean_m_s: float
    thermal_net_climb_std_m_s: float
    thermal_net_climb_max_m_s: float
    thermal_distance_min_m: float
    thermal_distance_mean_m: float
    thermal_distance_std_m: float
    thermal_distance_max_m: float

    @property
    def distance_max_m(self):
        return self.landing_time_s * 15  # 15 m /s is very fast

    def sample_thermal(self, last_distance_end_m: float) -> Thermal:
        distance_from_last_thermal_m = np.random.normal(self.thermal_distance_mean_m, self.thermal_distance_std_m)
        distance_start_m = np.clip(distance_from_last_thermal_m, self.thermal_distance_min_m, self.thermal_distance_max_m)
        distance_start_m += last_distance_end_m
        net_climb_m_s = np.random.normal(self.thermal_net_climb_mean_m_s, self.thermal_net_climb_std_m_s)
        net_climb_m_s = np.clip(net_climb_m_s, self.thermal_net_climb_min_m_s, self.thermal_net_climb_max_m_s)
        return Thermal(
            distance_start_m=distance_start_m,
            distance_end_m=distance_start_m + 200,
            net_climb_m_s=net_climb_m_s,
        )

    def sample_thermals(self) -> list[Thermal]:
        thermals = [self.sample_thermal(0.0)]
        while thermals[-1].distance_end_m < self.distance_max_m:
            thermals.append(self.sample_thermal(thermals[-1].distance_end_m))
        return thermals

    def series_thermal_sampled(self) -> tuple[list[float], list[float]]:
        """
        Produce a blocky series from newly sampled thermals for plotting.
        Returns:
            Tuple of (distance_series, thermal_strength_series)
        """
        thermals = self.sample_thermals()
        return FlightConditions.series_thermal(thermals, self.distance_max_m)

    @staticmethod
    def series_thermal(
        thermals: list[Thermal],
        distance_max_m: float,
    ) -> tuple[list[float], list[float]]:
        """
        Produce a blocky series derived from thermals for plotting.
        Returns:
            Tuple of (distance_series, thermal_strength_series)
            Both series have the same length and represent thermal strength as constant blocks.
        """
        distance_series = [0.0]
        thermal_strength_series = [0.0]

        for thermal in thermals:
            distance_series.append(thermal.distance_start_m)
            thermal_strength_series.append(0.0)

            distance_series.append(thermal.distance_start_m)
            thermal_strength_series.append(thermal.net_climb_m_s)

            distance_series.append(thermal.distance_end_m)
            thermal_strength_series.append(thermal.net_climb_m_s)

            if thermal.distance_end_m < distance_max_m:
                distance_series.append(thermal.distance_end_m)
                thermal_strength_series.append(0.0)

        if distance_series[-1] < distance_max_m:
            distance_series.append(distance_max_m)
            thermal_strength_series.append(0.0)

        return distance_series, thermal_strength_series


class FlightConditionsDistribution:
    """
    Distribution over flight conditions. sample() returns FlightConditions
    with thermal climb parameters drawn between climb_min_m_s and climb_max_m_s.
    """

    def __init__(
        self,
        thermal_net_climb_mean_min_m_s: float,
        thermal_net_climb_mean_max_m_s: float,
        *,
        take_off_time_s: float = 0.0,
        take_off_altitude_m: float = 1000.0,
        landing_time_s: float = 3600.0 * 6,
        distance_max_m: float = 3600.0 * 6 * 10,
        thermal_ceiling_m: float = 1000.0,
        thermal_distance_min_m: float = 1000.0,
        thermal_distance_mean_m: float = 2000.0,
        thermal_distance_std_m: float = 1000.0,
        thermal_distance_max_m: float = 10000.0,
    ):
        self.thermal_net_climb_mean_min_m_s = thermal_net_climb_mean_min_m_s
        self.thermal_net_climb_mean_max_m_s = thermal_net_climb_mean_max_m_s
        self._take_off_time_s = take_off_time_s
        self._take_off_altitude_m = take_off_altitude_m
        self._landing_time_s = landing_time_s
        self._distance_max_m = distance_max_m
        self._thermal_ceiling_m = thermal_ceiling_m
        self._thermal_distance_min_m = thermal_distance_min_m
        self._thermal_distance_mean_m = thermal_distance_mean_m
        self._thermal_distance_std_m = thermal_distance_std_m
        self._thermal_distance_max_m = thermal_distance_max_m

    def sample(self) -> FlightConditions:
        """Sample a FlightConditions with thermal climb mean in [climb_min_m_s, climb_max_m_s]."""
        mean_m_s = np.random.uniform(self.thermal_net_climb_mean_min_m_s, self.thermal_net_climb_mean_max_m_s)
        # Hardcoded for now
        thermal_net_climb_min_m_s = 0.20  # the frist thermal should be at least 0.20 m/s to be considered a thermal
        thermal_net_climb_max_m_s = 10.0
        return FlightConditions(
            take_off_time_s=self._take_off_time_s,
            take_off_altitude_m=self._take_off_altitude_m,
            landing_time_s=self._landing_time_s,
            distance_max_m=self._distance_max_m,
            thermal_ceiling_m=self._thermal_ceiling_m,
            thermal_net_climb_min_m_s=thermal_net_climb_min_m_s,
            thermal_net_climb_mean_m_s=mean_m_s,
            thermal_net_climb_std_m_s=mean_m_s * 0.5,  # we want 95% of thermals to be larger than 0
            thermal_net_climb_max_m_s=thermal_net_climb_max_m_s,
            thermal_distance_min_m=self._thermal_distance_min_m,
            thermal_distance_mean_m=self._thermal_distance_mean_m,
            thermal_distance_std_m=self._thermal_distance_std_m,
            thermal_distance_max_m=self._thermal_distance_max_m,
        )


class FlightConditionsDistributionChoice:
    """
    Distribution over flight conditions. sample() returns FlightConditions
    with thermal climb parameters drawn between climb_min_m_s and climb_max_m_s.
    """

    def __init__(
        self,
        list_climbs_mean_m_s: list[float],
        *,
        take_off_time_s: float = 0.0,
        take_off_altitude_m: float = 1000.0,
        landing_time_s: float = 3600.0 * 6,
        distance_max_m: float = 3600.0 * 6 * 10,
        thermal_ceiling_m: float = 1000.0,
        thermal_distance_min_m: float = 1000.0,
        thermal_distance_mean_m: float = 2000.0,
        thermal_distance_std_m: float = 1000.0,
        thermal_distance_max_m: float = 10000.0,
    ):
        self.list_climbs_mean_m_s = list_climbs_mean_m_s
        self._take_off_time_s = take_off_time_s
        self._take_off_altitude_m = take_off_altitude_m
        self._landing_time_s = landing_time_s
        self._distance_max_m = distance_max_m
        self._thermal_ceiling_m = thermal_ceiling_m
        self._thermal_distance_min_m = thermal_distance_min_m
        self._thermal_distance_mean_m = thermal_distance_mean_m
        self._thermal_distance_std_m = thermal_distance_std_m
        self._thermal_distance_max_m = thermal_distance_max_m

    def sample(self) -> FlightConditions:
        """Sample a FlightConditions with thermal climb mean in [climb_min_m_s, climb_max_m_s]."""
        mean_m_s = np.random.choice(self.list_climbs_mean_m_s, size=1)[0]
        # Hardcoded for now
        thermal_net_climb_min_m_s = 0.20  # the frist thermal should be at least 0.20 m/s to be considered a thermal
        thermal_net_climb_max_m_s = 10.0
        return FlightConditions(
            take_off_time_s=self._take_off_time_s,
            take_off_altitude_m=self._take_off_altitude_m,
            landing_time_s=self._landing_time_s,
            distance_max_m=self._distance_max_m,
            thermal_ceiling_m=self._thermal_ceiling_m,
            thermal_net_climb_min_m_s=thermal_net_climb_min_m_s,
            thermal_net_climb_mean_m_s=mean_m_s,
            thermal_net_climb_std_m_s=mean_m_s * 0.5,  # we want 95% of thermals to be larger than 0
            thermal_net_climb_max_m_s=thermal_net_climb_max_m_s,
            thermal_distance_min_m=self._thermal_distance_min_m,
            thermal_distance_mean_m=self._thermal_distance_mean_m,
            thermal_distance_std_m=self._thermal_distance_std_m,
            thermal_distance_max_m=self._thermal_distance_max_m,
        )
