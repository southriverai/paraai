from tqdm import tqdm

from paraai.experiment import ExperimentOutput, ExperimentOutputBatch
from paraai.flight_conditions import FlightConditions, FlightConditionsDistribution, Termal
from paraai.flight_policy import FlightPolicyBase
from paraai.model import AircraftModel, FlightState


class SimulatorCrude:
    def __init__(
        self,
        flight_condition_distribution: FlightConditionsDistribution,
        aircraft_model: AircraftModel,
    ):
        self.flight_condition_distribution = flight_condition_distribution
        self.aircraft_model = aircraft_model

    def simulate_batch(
        self,
        policy: FlightPolicyBase,
        num_simulations: int,
    ) -> ExperimentOutputBatch:
        """Simulate a batch of flights."""
        list_experiment_outputs: list[ExperimentOutput] = []
        for _ in tqdm(range(num_simulations)):
            experiment_output = self.simulate_flight(policy)
            list_experiment_outputs.append(experiment_output)
        return ExperimentOutputBatch(
            list_experiment_outputs=list_experiment_outputs,
        )

    def simulate_flight(
        self,
        policy: FlightPolicyBase,
    ) -> ExperimentOutput:
        flight_conditions = self.flight_condition_distribution.sample()
        termals = flight_conditions.sample_termals()
        termal_index = 0

        time_s = flight_conditions.take_off_time_s
        # add initial state
        flight_state = FlightState(
            list_time_s=[time_s],
            list_altitude_m=[flight_conditions.take_off_altitude_m],
            list_distance_m=[0],
            list_use_thermal=[False],
            status="flying",
        )

        while flight_state.status == "flying":
            # get the next termal
            next_termal = termals[termal_index]
            self.simulate_progress_to_termal(flight_conditions, self.aircraft_model, flight_state, next_termal)
            if policy.use_termal(flight_state, self.aircraft_model):
                self.simulate_termal(flight_conditions, self.aircraft_model, flight_state, next_termal)
            termal_index += 1
            # if there are no more termals we land
            if termal_index >= len(termals):
                self.simulate_glide_to_landing(flight_conditions, self.aircraft_model, flight_state)

        return ExperimentOutput(
            flight_state=flight_state,
            termals=termals,
            aircraft_model=self.aircraft_model,
        )

    def simulate_termal(
        self,
        flight_conditions: FlightConditions,
        aircraft_model: AircraftModel,
        flight_state: FlightState,
        termal: Termal,
    ):
        # we are using the termal so in the last state we chose to use one
        flight_state.list_use_thermal[-1] = True
        last_node_time_s = flight_state.list_time_s[-1]
        last_node_altitude_m = flight_state.list_altitude_m[-1]
        last_node_distance_m = flight_state.list_distance_m[-1]
        altitude_to_ceiling_m = flight_conditions.termal_ceiling_m - last_node_altitude_m
        time_to_next_node_s = altitude_to_ceiling_m / termal.net_climb_m_s

        # Check if we've maxed flight time
        if last_node_time_s + time_to_next_node_s >= flight_conditions.landing_time_s:
            time_to_next_node_s = flight_conditions.landing_time_s - last_node_time_s
            flight_state.status = "out_of_time"

        node_time_s = last_node_time_s + time_to_next_node_s
        node_distance_m = last_node_distance_m
        node_altitude_m = last_node_altitude_m + termal.net_climb_m_s * time_to_next_node_s

        flight_state.list_time_s.append(node_time_s)
        flight_state.list_distance_m.append(node_distance_m)
        flight_state.list_altitude_m.append(node_altitude_m)
        flight_state.list_use_thermal.append(False)

    def simulate_progress_to_termal(
        self,
        flight_conditions: FlightConditions,
        aircraft_model: AircraftModel,
        flight_state: FlightState,
        termal: Termal,
    ):
        last_node_time_s = flight_state.list_time_s[-1]
        last_node_altitude_m = flight_state.list_altitude_m[-1]
        last_node_distance_m = flight_state.list_distance_m[-1]

        thermal_distance_m = termal.distance_center_m
        distance_to_termal_m = thermal_distance_m - last_node_distance_m
        time_to_next_node_s = distance_to_termal_m / aircraft_model.velocity_max_m_s

        # Check if we've max flight time
        if last_node_time_s + time_to_next_node_s >= flight_conditions.landing_time_s:
            time_to_next_node_s = flight_conditions.landing_time_s - last_node_time_s
            flight_state.status = "out_of_time"

        node_altitude_m = last_node_altitude_m + aircraft_model.sink_max_m_s * time_to_next_node_s
        # check if we hit the ground
        if node_altitude_m <= 0:
            time_to_next_node_s = -last_node_altitude_m / aircraft_model.sink_max_m_s
            flight_state.status = "out_of_altitude"

        node_time_s = last_node_time_s + time_to_next_node_s
        node_distance_m = last_node_distance_m + aircraft_model.velocity_max_m_s * time_to_next_node_s
        node_altitude_m = last_node_altitude_m + aircraft_model.sink_max_m_s * time_to_next_node_s

        # add the flight to the state
        flight_state.list_time_s.append(node_time_s)
        flight_state.list_distance_m.append(node_distance_m)
        flight_state.list_altitude_m.append(node_altitude_m)
        flight_state.list_use_thermal.append(False)

        # if we have not landed yetthen add one second of lift to the state so the policy can decide to use termal
        if flight_state.status == "flying":
            flight_state.list_time_s.append(node_time_s + 1)
            flight_state.list_distance_m.append(node_distance_m)
            flight_state.list_altitude_m.append(node_altitude_m + termal.net_climb_m_s * 1)
            flight_state.list_use_thermal.append(False)

    def simulate_glide_to_landing(
        self,
        flight_conditions: FlightConditions,
        aircraft_model: AircraftModel,
        flight_state: FlightState,
    ):
        print("Simulating glide to landing")
        last_node_time_s = flight_state.list_time_s[-1]
        last_node_altitude_m = flight_state.list_altitude_m[-1]
        last_node_distance_m = flight_state.list_distance_m[-1]
        time_to_next_node_s = flight_conditions.landing_time_s - last_node_time_s

        node_altitude_m = last_node_altitude_m + aircraft_model.sink_max_m_s * time_to_next_node_s
        if node_altitude_m <= 0:
            # if we land before we run out of time, we land at the ground
            time_to_next_node_s = -last_node_altitude_m / aircraft_model.sink_max_m_s
            flight_state.status = "out_of_altitude"

        node_time_s = last_node_time_s + time_to_next_node_s
        node_distance_m = last_node_distance_m + aircraft_model.velocity_max_m_s * time_to_next_node_s
        node_altitude_m = 0

        flight_state.list_time_s.append(node_time_s)
        flight_state.list_distance_m.append(node_distance_m)
        flight_state.list_altitude_m.append(node_altitude_m)
        flight_state.list_use_thermal.append(False)
        flight_state.status = "out_of_time"
