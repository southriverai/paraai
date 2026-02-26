import numpy as np
from tqdm import tqdm

from paraai.experiment import ExperimentOutput, ExperimentOutputBatch
from paraai.flight_policy import FlightPolicyBase
from paraai.model.atmosphere_model import AthmosphereModel
from paraai.model.model import ActionNode, AircraftModel, FlightNode, FlightState
from paraai.model.thermal import Thermal


class Simulator3:
    def __init__(
        self,
        athmosphere_model: AthmosphereModel,
        aircraft_model: AircraftModel,
        thermal_time_step_s: float = 60.0,
        track_length_m: int = 200000,
    ):
        self.athmosphere_model = athmosphere_model
        self.aircraft_model = aircraft_model
        self.thermal_time_step_s = thermal_time_step_s
        self.track_length_m = track_length_m

    def simulate_batch(
        self,
        policy: FlightPolicyBase,
        num_simulations: int,
    ) -> ExperimentOutputBatch:
        """Simulate a batch of flights."""
        list_experiment_outputs: list[ExperimentOutput] = []
        for _ in tqdm(range(num_simulations)):
            experiment_output = self.simulate(policy)
            list_experiment_outputs.append(experiment_output)
        return ExperimentOutputBatch(
            list_experiment_outputs=list_experiment_outputs,
        )

    def simulate_flight_for_day(
        self,
        policy: FlightPolicyBase,
        day_of_year_d: int,
        take_off_time_s: int,
    ) -> ExperimentOutput:
        """Simulate a flight."""
        day_strength = self.athmosphere_model.sample_day_strength_m_s(day_of_year_d)
        ceiling_m = self.athmosphere_model.sample_ceiling_m(day_strength)
        thermals = self.athmosphere_model.sample_thermal_track(day_strength, self.track_length_m)
        thermal_index = 0
        # we are varrying the take off altitude uniform randomly in the top half of the usuable athmosphere
        take_off_altitude_m = np.random.uniform(0, ceiling_m / 2)
        climb_speed_m_s = self.athmosphere_model.climb_speed_m_s(
            thermals[thermal_index].thermal_strength_m_s, ceiling_m, take_off_altitude_m, day_of_year_d, take_off_time_s
        )

        # add initial state
        flight_state = FlightState(
            flight_nodes=[
                FlightNode(
                    time_of_day_s=take_off_time_s,
                    altitude_m=0,
                    distance_m=0,
                    climb_m_s=climb_speed_m_s,
                )
            ],
            action_nodes=[],
        )

        while flight_state.flight_nodes[-1].altitude_m > 0:
            # get the action node from the policy
            action_node = policy.get_action(flight_state, self.aircraft_model)

            if action_node.use_thermal:
                flight_state.action_nodes.append(ActionNode(use_thermal=True))
                self.simulate_thermal(flight_state.flight_nodes[-1], thermals[thermal_index])
            else:
                flight_state.action_nodes.append(ActionNode(use_thermal=False))
                thermal_index += 1
                self.simulate_progress(flight_state, thermals[thermal_index])

        return ExperimentOutput(
            flight_state=flight_state,
            thermals=thermals,
            aircraft_model=self.aircraft_model,
        )

    def simulate_progress(
        self,
        flight_node: FlightNode,
        next_thermal: Thermal,
        ceiling_m: float,
    ) -> FlightNode:
        distance_to_thermal_m = next_thermal.distance_m - flight_node.distance_m
        time_to_thermal_s = distance_to_thermal_m / self.aircraft_model.velocity_max_m_s
        altitude_at_thermal_m = flight_node.altitude_m - self.aircraft_model.sink_max_m_s * time_to_thermal_s

        # check if we are hitting the ground before we get to the next thermal
        if altitude_at_thermal_m <= 0:
            # we are hitting the ground before we get to the next thermal
            time_to_land_s = -flight_node.altitude_m / self.aircraft_model.sink_max_m_s
            distance_to_land_m = self.aircraft_model.velocity_max_m_s * time_to_land_s

            return FlightNode(
                day_of_year_d=flight_node.day_of_year_d,
                time_of_day_s=flight_node.time_of_day_s + time_to_land_s,
                altitude_m=0,
                distance_m=flight_node.distance_m + distance_to_land_m,
                climb_m_s=0,
            )
        else:
            # we are not hitting the ground before we get to the next thermal
            time_at_next_thermal_s = flight_node.time_of_day_s + time_to_thermal_s
            climb_in_next_thermal_m_s = self.athmosphere_model.climb_speed_m_s(
                next_thermal.thermal_strength_m_s,
                ceiling_m,
                altitude_at_thermal_m,
                flight_node.day_of_year_d,
                time_at_next_thermal_s,
            )
            return FlightNode(
                time_of_day_s=time_at_next_thermal_s,
                altitude_m=altitude_at_thermal_m,
                distance_m=next_thermal.distance_m,
                climb_m_s=climb_in_next_thermal_m_s,
            )

    def simulate_thermal(
        self,
        flight_node: FlightNode,
        current_thermal: Thermal,
        ceiling_m: float,
    ) -> FlightNode:
        # Check if we've reached thermal ceiling
        node_altitude_m = current_thermal.altitude_m

        # check if we hit the ceiling before we get to the end of the thermal
        if node_altitude_m <= ceiling_m:
            time_to_ceiling_s = (ceiling_m - node_altitude_m) / self.aircraft_model.sink_max_m_s
            return FlightNode(
                time_of_day_s=flight_node.time_of_day_s + time_to_land_s,
                altitude_m=0,
                distance_m=flight_node.distance_m + distance_to_land_m,
                climb_m_s=0,
            )
