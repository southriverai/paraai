import numpy as np
from pydantic import BaseModel


class Termal(BaseModel):
    distance_start_m: float
    distance_end_m: float
    net_climb_m_s: float

    @property
    def distance_center_m(self) -> float:
        return (self.distance_start_m + self.distance_end_m) / 2


class FlightConditions(BaseModel):
    take_off_time_s: float
    take_off_altitude_m: float
    landing_time_s: float
    termal_ceiling_m: float
    termal_net_climb_min_m_s: float
    termal_net_climb_mean_m_s: float
    termal_net_climb_std_m_s: float
    termal_net_climb_max_m_s: float
    termal_distance_min_m: float
    termal_distance_mean_m: float
    termal_distance_std_m: float
    termal_distance_max_m: float

    @property
    def distance_max_m(self):
        return self.landing_time_s * 15  # 15 m /s is very fast

    def sample_termal(self, last_distance_end_m: float) -> Termal:
        distance_from_last_termal_m = np.random.normal(self.termal_distance_mean_m, self.termal_distance_std_m)
        distance_start_m = np.clip(distance_from_last_termal_m, self.termal_distance_min_m, self.termal_distance_max_m)
        distance_start_m += last_distance_end_m
        net_climb_m_s = np.random.normal(self.termal_net_climb_mean_m_s, self.termal_net_climb_std_m_s)
        net_climb_m_s = np.clip(net_climb_m_s, self.termal_net_climb_min_m_s, self.termal_net_climb_max_m_s)
        return Termal(
            distance_start_m=distance_start_m,
            distance_end_m=distance_start_m + 200,
            net_climb_m_s=net_climb_m_s,
        )

    def sample_termals(self) -> list[Termal]:
        termals = [self.sample_termal(0.0)]
        while termals[-1].distance_end_m < self.distance_max_m:
            termals.append(self.sample_termal(termals[-1].distance_end_m))
        return termals

    def series_termal_sampled(self) -> tuple[list[float], list[float]]:
        """
        Produce a blocky series from newly sampled termals for plotting.
        Returns:
            Tuple of (distance_series, thermal_strength_series)
        """
        termals = self.sample_termals()
        return FlightConditions.series_termal(termals, self.distance_max_m)

    @staticmethod
    def series_termal(
        termals: list[Termal],
        distance_max_m: float,
    ) -> tuple[list[float], list[float]]:
        """
        Produce a blocky series derived from termals for plotting.
        Returns:
            Tuple of (distance_series, thermal_strength_series)
            Both series have the same length and represent thermal strength as constant blocks.
        """
        distance_series = [0.0]
        thermal_strength_series = [0.0]

        for termal in termals:
            distance_series.append(termal.distance_start_m)
            thermal_strength_series.append(0.0)

            distance_series.append(termal.distance_start_m)
            thermal_strength_series.append(termal.net_climb_m_s)

            distance_series.append(termal.distance_end_m)
            thermal_strength_series.append(termal.net_climb_m_s)

            if termal.distance_end_m < distance_max_m:
                distance_series.append(termal.distance_end_m)
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
        termal_net_climb_mean_min_m_s: float,
        termal_net_climb_mean_max_m_s: float,
        *,
        take_off_time_s: float = 0.0,
        take_off_altitude_m: float = 1000.0,
        landing_time_s: float = 3600.0 * 6,
        distance_max_m: float = 3600.0 * 6 * 10,
        termal_ceiling_m: float = 1000.0,
        termal_distance_min_m: float = 1000.0,
        termal_distance_mean_m: float = 2000.0,
        termal_distance_std_m: float = 1000.0,
        termal_distance_max_m: float = 10000.0,
    ):
        self.termal_net_climb_mean_min_m_s = termal_net_climb_mean_min_m_s
        self.termal_net_climb_mean_max_m_s = termal_net_climb_mean_max_m_s
        self._take_off_time_s = take_off_time_s
        self._take_off_altitude_m = take_off_altitude_m
        self._landing_time_s = landing_time_s
        self._distance_max_m = distance_max_m
        self._termal_ceiling_m = termal_ceiling_m
        self._termal_distance_min_m = termal_distance_min_m
        self._termal_distance_mean_m = termal_distance_mean_m
        self._termal_distance_std_m = termal_distance_std_m
        self._termal_distance_max_m = termal_distance_max_m

    def sample(self) -> FlightConditions:
        """Sample a FlightConditions with thermal climb mean in [climb_min_m_s, climb_max_m_s]."""
        mean_m_s = np.random.uniform(self.termal_net_climb_mean_min_m_s, self.termal_net_climb_mean_max_m_s)
        # Hardcoded for now
        termal_net_climb_min_m_s = 0.20  # the frist termal should be at least 0.20 m/s to be considered a termal
        termal_net_climb_max_m_s = 10.0
        return FlightConditions(
            take_off_time_s=self._take_off_time_s,
            take_off_altitude_m=self._take_off_altitude_m,
            landing_time_s=self._landing_time_s,
            distance_max_m=self._distance_max_m,
            termal_ceiling_m=self._termal_ceiling_m,
            termal_net_climb_min_m_s=termal_net_climb_min_m_s,
            termal_net_climb_mean_m_s=mean_m_s,
            termal_net_climb_std_m_s=mean_m_s * 0.5,  # we want 95% of termals to be larger than 0
            termal_net_climb_max_m_s=termal_net_climb_max_m_s,
            termal_distance_min_m=self._termal_distance_min_m,
            termal_distance_mean_m=self._termal_distance_mean_m,
            termal_distance_std_m=self._termal_distance_std_m,
            termal_distance_max_m=self._termal_distance_max_m,
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
        termal_ceiling_m: float = 1000.0,
        termal_distance_min_m: float = 1000.0,
        termal_distance_mean_m: float = 2000.0,
        termal_distance_std_m: float = 1000.0,
        termal_distance_max_m: float = 10000.0,
    ):
        self.list_climbs_mean_m_s = list_climbs_mean_m_s
        self._take_off_time_s = take_off_time_s
        self._take_off_altitude_m = take_off_altitude_m
        self._landing_time_s = landing_time_s
        self._distance_max_m = distance_max_m
        self._termal_ceiling_m = termal_ceiling_m
        self._termal_distance_min_m = termal_distance_min_m
        self._termal_distance_mean_m = termal_distance_mean_m
        self._termal_distance_std_m = termal_distance_std_m
        self._termal_distance_max_m = termal_distance_max_m

    def sample(self) -> FlightConditions:
        """Sample a FlightConditions with thermal climb mean in [climb_min_m_s, climb_max_m_s]."""
        mean_m_s = np.random.choice(self.list_climbs_mean_m_s, size=1)[0]
        # Hardcoded for now
        termal_net_climb_min_m_s = 0.20  # the frist termal should be at least 0.20 m/s to be considered a termal
        termal_net_climb_max_m_s = 10.0
        return FlightConditions(
            take_off_time_s=self._take_off_time_s,
            take_off_altitude_m=self._take_off_altitude_m,
            landing_time_s=self._landing_time_s,
            distance_max_m=self._distance_max_m,
            termal_ceiling_m=self._termal_ceiling_m,
            termal_net_climb_min_m_s=termal_net_climb_min_m_s,
            termal_net_climb_mean_m_s=mean_m_s,
            termal_net_climb_std_m_s=mean_m_s * 0.5,  # we want 95% of termals to be larger than 0
            termal_net_climb_max_m_s=termal_net_climb_max_m_s,
            termal_distance_min_m=self._termal_distance_min_m,
            termal_distance_mean_m=self._termal_distance_mean_m,
            termal_distance_std_m=self._termal_distance_std_m,
            termal_distance_max_m=self._termal_distance_max_m,
        )
