from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field


class FlightAction(BaseModel):
    velocity_m_s: float
    use_termal: bool


# class AircraftModel(BaseModel):
#     velocity_max_m_s: float
#     velocity_min_m_s: float
#     velocity_best_glide_m_s: float
#     list_velocity_m_s: List[float]
#     list_sink_rate_m_s: List[float]

#     def get_sink_rate_m_s(self, velocity_m_s: float) -> float:
#         if velocity_m_s < self.velocity_min_m_s:
#             raise Exception(f"Velocity {velocity_m_s} is below the minimum velocity {self.velocity_min_m_s}")
#         if velocity_m_s > self.velocity_max_m_s:
#             raise Exception(f"Velocity {velocity_m_s} is above the maximum velocity {self.velocity_max_m_s}")
#         return interp1d(self.list_velocity_m_s, self.list_sink_rate_m_s)(velocity_m_s)


class AircraftModel(BaseModel):
    velocity_max_m_s: float
    sink_max_m_s: float


class FlightNode(BaseModel):
    time_s: float
    altitude_m: float
    distance_m: float
    is_landed: bool
    use_thermal: bool


class ActionNode(BaseModel):
    use_thermal: bool


class FlightState2(BaseModel):
    flight_nodes: list[FlightNode]
    action_nodes: list[ActionNode]

    cache: dict[str, Any] = Field(default_factory=dict, exclude=True)  # no serialization


class FlightState(BaseModel):
    list_time_s: list[float]
    list_altitude_m: list[float]
    list_distance_m: list[float]
    list_use_thermal: list[bool]
    status: Literal["flying", "out_of_time", "out_of_altitude"]
    cache: dict[str, Any] = Field(default_factory=dict, exclude=True)  # no serialization

    def current_climb_m_s(self) -> float:
        if len(self.list_altitude_m) < 2:
            return 0
        current_climb_m = self.list_altitude_m[-1] - self.list_altitude_m[-2]

        if self.list_time_s[-2] == self.list_time_s[-1]:
            print(self.list_time_s)
            raise Exception("Time step is 0")
        current_climb_m_s = current_climb_m / (self.list_time_s[-1] - self.list_time_s[-2])
        return current_climb_m_s

    def get_node_climb_m_s(self, node_index: int) -> float:
        if node_index == 0:
            return 0
        if node_index >= len(self.list_time_s):
            return 0
        node_climb_m = self.list_altitude_m[node_index] - self.list_altitude_m[node_index - 1]
        node_climb_m_s = node_climb_m / (self.list_time_s[node_index] - self.list_time_s[node_index - 1])
        self.cache[node_index] = node_climb_m_s
        return node_climb_m_s

    def all_climbs(self) -> list[float]:
        """
        Get all climb rates from all nodes (not just thermal climbs).
        Returns list of climb rates in m/s for each time step.
        """
        climbs = []
        if len(self.list_time_s) < 2:
            return climbs

        for i in range(1, len(self.list_time_s)):
            climb_m = self.list_altitude_m[i] - self.list_altitude_m[i - 1]
            time_diff = self.list_time_s[i] - self.list_time_s[i - 1]
            if time_diff > 0:
                climb_m_s = climb_m / time_diff
                climbs.append(climb_m_s)

        return climbs

    def termal_climbs(self) -> list[float]:
        # merges all consequetive nodes with climb and produces one average number for each termal
        if len(self.list_time_s) < 2:
            return []

        if "termal_climbs" in self.cache:
            node_index = self.cache["termal_climbs"]["node_index"]
            termal_climbs = self.cache["termal_climbs"]["termal_climbs"]
            current_termal_climbs = self.cache["termal_climbs"]["current_termal_climbs"]
        else:
            node_index = 1
            termal_climbs = []
            current_termal_climbs = []
        while node_index < len(self.list_time_s):
            node_climb_m_s = self.get_node_climb_m_s(node_index)
            if node_climb_m_s > 0:
                current_termal_climbs.append(node_climb_m_s)
            elif len(current_termal_climbs) > 0:
                termal_climbs.append(np.mean(current_termal_climbs))
                current_termal_climbs = []
            node_index += 1
        self.cache["termal_climbs"] = {
            "termal_climbs": termal_climbs,
            "current_termal_climbs": current_termal_climbs,
            "node_index": node_index,
        }
        return termal_climbs
