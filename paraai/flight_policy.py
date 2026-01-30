import hashlib
from abc import ABC, abstractmethod

import numpy as np

from paraai.model import AircraftModel, FlightState


class FlightPolicyBase(ABC):
    policy_name: str

    def __init__(self, policy_name: str):
        self.policy_name = policy_name

    @abstractmethod
    def use_termal(self, flight_state: FlightState, aircraft_model: AircraftModel) -> bool:
        pass

    def get_hash(self) -> str:
        return hashlib.sha256(self.policy_name.encode()).hexdigest()


class FlightPolicyNeverTermal(FlightPolicyBase):
    def __init__(self):
        super().__init__(policy_name="NeverTermal")

    def use_termal(
        self,
        flight_state: FlightState,
        aircraft_model: AircraftModel,
    ) -> bool:
        return False


class FlightPolicyAlwaysTermal(FlightPolicyBase):
    def __init__(self):
        super().__init__(policy_name="AlwaysTermal")

    def use_termal(
        self,
        flight_state: FlightState,
        aircraft_model: AircraftModel,
    ) -> bool:
        # if we experience climb since the last node we keep termaling
        if len(flight_state.list_altitude_m) < 2:
            return False
        last_climb = flight_state.list_altitude_m[-1] - flight_state.list_altitude_m[-2]
        if last_climb > 0:
            return True
        else:
            return False


class FlightPolicyThreeZones(FlightPolicyBase):
    def __init__(self, progress_quantile: float, lift_zone_quantile: float):
        super().__init__(policy_name=f"ThreeZones {progress_quantile} {lift_zone_quantile}")
        self.progress_quantile = progress_quantile
        self.lift_zone_quantile = lift_zone_quantile
        self.explore_termal_count = 2

    def use_termal(
        self,
        flight_state: FlightState,
        aircraft_model: AircraftModel,
    ) -> bool:
        # Simple three-zone policy: use termal if below starting altitude (survival zone)
        if len(flight_state.list_altitude_m) < 1:
            return False
        altitude = flight_state.list_altitude_m[-1]
        starting_altitude = flight_state.list_altitude_m[0]  # asume we start at ceiling
        progress_altitude = starting_altitude * 0.66
        lift_altitude = starting_altitude * 0.33
        current_climb_m_s = flight_state.current_climb_m_s()
        if current_climb_m_s <= 0:
            # if we are not climbing, we do not use termal
            return False
        termal_climbs = flight_state.termal_climbs()
        if len(termal_climbs) < self.explore_termal_count:
            # if we have not experienced many termals, we do use termal just to try
            return True

        progress_threshold = np.quantile(termal_climbs, self.progress_quantile)
        lift_threshold = np.quantile(termal_climbs, self.lift_zone_quantile)

        # If below starting altitude, use termal for survival
        if altitude > progress_altitude:
            return current_climb_m_s > progress_threshold
        elif altitude > lift_altitude:
            return current_climb_m_s > lift_threshold
        else:
            # we are in the survival zone so we use termal to survive
            return True
