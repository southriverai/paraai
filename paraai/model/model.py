from typing import Any

from pydantic import BaseModel, Field


class FlightAction(BaseModel):
    velocity_m_s: float
    use_thermal: bool


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
    day_of_year_d: float
    time_of_day_s: float
    altitude_m: float
    distance_m: float
    climb_m_s: float


class ActionNode(BaseModel):
    use_thermal: bool


class FlightState(BaseModel):
    flight_nodes: list[FlightNode]
    action_nodes: list[ActionNode]
    cache: dict[str, Any] = Field(default_factory=dict, exclude=True)  # no serialization

    def has_climb(self) -> bool:
        return self.last_climb_m_s > 0

    @property
    def first_altitude_m(self) -> float:
        return self.flight_nodes[0].altitude_m

    @property
    def current_altitude_m(self) -> float:
        return self.flight_nodes[-1].altitude_m

    @property
    def current_distance_m(self) -> float:
        return self.flight_nodes[-1].distance_m

    @property
    def current_time_of_day_s(self) -> float:
        return self.flight_nodes[-1].time_of_day_s

    @property
    def current_climb_m_s(self) -> float:
        return self.flight_nodes[-1].climb_m_s

    @property
    def max_altitude_m(self) -> float:
        return max(node.altitude_m for node in self.flight_nodes)  # TODO cache this for speed up


# class FlightState(BaseModel):
#     list_time_s: list[float]
#     list_altitude_m: list[float]
#     list_distance_m: list[float]
#     list_use_thermal: list[bool]
#     status: Literal["flying", "out_of_time", "out_of_altitude"]
#     cache: dict[str, Any] = Field(default_factory=dict, exclude=True)  # no serialization

#     def current_climb_m_s(self) -> float:
#         if len(self.list_altitude_m) < 2:
#             return 0
#         current_climb_m = self.list_altitude_m[-1] - self.list_altitude_m[-2]

#         if self.list_time_s[-2] == self.list_time_s[-1]:
#             print(self.list_time_s)
#             raise Exception("Time step is 0")
#         current_climb_m_s = current_climb_m / (self.list_time_s[-1] - self.list_time_s[-2])
#         return current_climb_m_s

#     def get_node_climb_m_s(self, node_index: int) -> float:
#         if node_index == 0:
#             return 0
#         if node_index >= len(self.list_time_s):
#             return 0
#         node_climb_m = self.list_altitude_m[node_index] - self.list_altitude_m[node_index - 1]
#         node_climb_m_s = node_climb_m / (self.list_time_s[node_index] - self.list_time_s[node_index - 1])
#         self.cache[node_index] = node_climb_m_s
#         return node_climb_m_s

#     def all_climbs(self) -> list[float]:
#         """
#         Get all climb rates from all nodes (not just thermal climbs).
#         Returns list of climb rates in m/s for each time step.
#         """
#         climbs = []
#         if len(self.list_time_s) < 2:
#             return climbs

#         for i in range(1, len(self.list_time_s)):
#             climb_m = self.list_altitude_m[i] - self.list_altitude_m[i - 1]
#             time_diff = self.list_time_s[i] - self.list_time_s[i - 1]
#             if time_diff > 0:
#                 climb_m_s = climb_m / time_diff
#                 climbs.append(climb_m_s)

#         return climbs

#     def thermal_climbs(self) -> list[float]:
#         # merges all consequetive nodes with climb and produces one average number for each thermal
#         if len(self.list_time_s) < 2:
#             return []

#         if "thermal_climbs" in self.cache:
#             node_index = self.cache["thermal_climbs"]["node_index"]
#             thermal_climbs = self.cache["thermal_climbs"]["thermal_climbs"]
#             current_thermal_climbs = self.cache["thermal_climbs"]["current_thermal_climbs"]
#         else:
#             node_index = 1
#             thermal_climbs = []
#             current_thermal_climbs = []
#         while node_index < len(self.list_time_s):
#             node_climb_m_s = self.get_node_climb_m_s(node_index)
#             if node_climb_m_s > 0:
#                 current_thermal_climbs.append(node_climb_m_s)
#             elif len(current_thermal_climbs) > 0:
#                 thermal_climbs.append(np.mean(current_thermal_climbs))
#                 current_thermal_climbs = []
#             node_index += 1
#         self.cache["thermal_climbs"] = {
#             "thermal_climbs": thermal_climbs,
#             "current_thermal_climbs": current_thermal_climbs,
#             "node_index": node_index,
#         }
#         return thermal_climbs
