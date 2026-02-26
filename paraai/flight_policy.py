import hashlib
from abc import ABC, abstractmethod

import numpy as np

from paraai.model.model import ActionNode, AircraftModel, FlightState


class FlightPolicyBase(ABC):
    policy_name: str

    def __init__(self, policy_name: str):
        self.policy_name = policy_name

    @abstractmethod
    def get_action(self, flight_state: FlightState, aircraft_model: AircraftModel) -> ActionNode:
        pass

    def get_hash(self) -> str:
        return hashlib.sha256(self.policy_name.encode()).hexdigest()


class FlightPolicyNeverThermal(FlightPolicyBase):
    def __init__(self):
        super().__init__(policy_name="NeverThermal")

    def get_action(
        self,
        flight_state: FlightState,
        aircraft_model: AircraftModel,
    ) -> ActionNode:
        return ActionNode(use_thermal=False)


class FlightPolicyAlwaysThermal(FlightPolicyBase):
    def __init__(self):
        super().__init__(policy_name="AlwaysThermal")

    def get_action(
        self,
        flight_state: FlightState,
        aircraft_model: AircraftModel,
    ) -> ActionNode:
        # if have climb we try to keep thermaling
        if flight_state.has_climb():
            return ActionNode(use_thermal=True)
        else:
            return ActionNode(use_thermal=False)


class FlightPolicyThreeZones(FlightPolicyBase):
    def __init__(self, progress_quantile: float, lift_zone_quantile: float):
        super().__init__(policy_name=f"ThreeZones {progress_quantile} {lift_zone_quantile}")
        self.progress_quantile = progress_quantile
        self.lift_zone_quantile = lift_zone_quantile
        self.explore_thermal_count = 2

    def get_action(
        self,
        flight_state: FlightState,
        aircraft_model: AircraftModel,
    ) -> ActionNode:
        # Simple three-zone policy: use thermal if below starting altitude (survival zone)
        altitude = flight_state.current_altitude_m
        starting_altitude = flight_state.max_altitude_m  # asume we start at ceiling
        progress_altitude = starting_altitude * 0.66
        lift_altitude = starting_altitude * 0.33
        if flight_state.has_climb():
            # if we are not climbing, we do not use thermal
            return ActionNode(use_thermal=False)
        thermal_climbs = flight_state.thermal_climbs()
        if len(thermal_climbs) < self.explore_thermal_count:
            # if we have not experienced many thermals, we do use thermal just to try
            return ActionNode(use_thermal=True)

        progress_threshold = np.quantile(thermal_climbs, self.progress_quantile)
        lift_threshold = np.quantile(thermal_climbs, self.lift_zone_quantile)

        # If below starting altitude, use thermal for survival
        if altitude > progress_altitude:
            use_thermal = flight_state.current_climb_m_s > progress_threshold
        elif altitude > lift_altitude:
            use_thermal = flight_state.current_climb_m_s > lift_threshold
        else:
            use_thermal = True  # we are in the survival zone so we use thermal to survive

        return ActionNode(use_thermal=use_thermal)
