from tqdm import tqdm

from paraai.experiment import ExperimentOutput, ExperimentOutputBatch
from paraai.flight_conditions import FlightConditions, Termal
from paraai.flight_policy import FlightPolicyBase
from paraai.model import AircraftModel, FlightState


def simulate_termal(
    flight_conditions: FlightConditions,
    aircraft_model: AircraftModel,
    flight_state: FlightState,
    termal: Termal,
    time_step_s: float,
):
    #    print(f"Simulating termal {termal.distance_start_m} to {termal.distance_end_m}")
    last_node_time_s = flight_state.list_time_s[-1]
    last_node_altitude_m = flight_state.list_altitude_m[-1]
    last_node_distance_m = flight_state.list_distance_m[-1]

    node_time_s = last_node_time_s + time_step_s
    node_distance_m = last_node_distance_m
    node_altitude_m = last_node_altitude_m + termal.net_climb_m_s * time_step_s
    is_landed = False

    # Check if we've reached thermal ceiling
    if node_altitude_m >= flight_conditions.termal_ceiling_m:
        # not further climbing
        node_altitude_m = flight_conditions.termal_ceiling_m

    # Check if we've max flight time
    if node_time_s >= flight_conditions.landing_time_s:
        # node_time_s = flight_conditions.landing_time_s
        # Hacky
        is_landed = True

    flight_state.list_time_s.append(node_time_s)
    flight_state.list_distance_m.append(node_distance_m)
    flight_state.list_altitude_m.append(node_altitude_m)
    flight_state.list_use_thermal.append(True)
    flight_state.is_landed = is_landed


def simulate_progress_to_termal(
    flight_conditions: FlightConditions,
    aircraft_model: AircraftModel,
    flight_state: FlightState,
    termal: Termal,
):
    last_node_time_s = flight_state.list_time_s[-1]
    last_node_altitude_m = flight_state.list_altitude_m[-1]
    last_node_distance_m = flight_state.list_distance_m[-1]

    thermal_distance_m = (termal.distance_start_m + termal.distance_end_m) / 2
    distance_to_termal_m = thermal_distance_m - last_node_distance_m
    time_step_s = distance_to_termal_m / aircraft_model.velocity_max_m_s

    # print(f"Simulating progress to termal {distance_to_termal_m} m at {time_step_s} s")

    node_time_s = last_node_time_s + time_step_s
    node_distance_m = last_node_distance_m + aircraft_model.velocity_max_m_s * time_step_s
    node_altitude_m = last_node_altitude_m + aircraft_model.sink_max_m_s * time_step_s
    is_landed = False

    # Check if we've max flight time
    if node_time_s >= flight_conditions.landing_time_s:
        # Hacky
        is_landed = True

    # check if we are at max flight distance
    if node_distance_m >= flight_conditions.distance_max_m:
        # Hacky
        is_landed = True

    # check if we are on ground
    if node_altitude_m <= 0:
        node_altitude_m = 0
        # Hacky
        is_landed = True

    # add the flight to the state
    flight_state.list_time_s.append(node_time_s)
    flight_state.list_distance_m.append(node_distance_m)
    flight_state.list_altitude_m.append(node_altitude_m)
    flight_state.list_use_thermal.append(False)
    flight_state.is_landed = is_landed

    # if we have not landed yetthen add one second of lift to the state so the policy can decide to use termal
    if not is_landed:
        flight_state.list_time_s.append(node_time_s + 1)
        flight_state.list_distance_m.append(node_distance_m)
        flight_state.list_altitude_m.append(node_altitude_m + termal.net_climb_m_s * 1)
        flight_state.list_use_thermal.append(True)
        flight_state.is_landed = False


def simulate_flight(
    flight_conditions: FlightConditions,
    aircraft_model: AircraftModel,
    policy: FlightPolicyBase,
    termal_time_step_s: float = 1.0,
) -> ExperimentOutput:
    termals = flight_conditions.sample_termals()
    termal_index = 0

    time_s = flight_conditions.take_off_time_s
    # add initial state
    flight_state = FlightState(
        list_time_s=[time_s],
        list_altitude_m=[flight_conditions.take_off_altitude_m],
        list_distance_m=[0],
        list_use_thermal=[False],
        is_landed=False,
    )

    # TODO in the futere we might want to make steps along this path rather than just jumping to the next termal to do some best glide stuff
    while not flight_state.is_landed:
        # if there are no more termals we land
        if termal_index >= len(termals):
            flight_state.is_landed = True  # TODO we should glide to out
            break
        # get the next termal
        next_termal = termals[termal_index]
        simulate_progress_to_termal(flight_conditions, aircraft_model, flight_state, next_termal)
        while policy.use_termal(flight_state, aircraft_model):
            simulate_termal(flight_conditions, aircraft_model, flight_state, next_termal, termal_time_step_s)
        termal_index += 1
    return ExperimentOutput(
        flight_state=flight_state,
        termals=termals,
        aircraft_model=aircraft_model,
    )


def simulate_flight_many(
    flight_conditions: FlightConditions,
    aircraft_model: AircraftModel,
    policy: FlightPolicyBase,
    flight_count: int,
    termal_time_step_s: float = 1.0,
) -> ExperimentOutputBatch:
    list_experiment_results: list[ExperimentOutput] = []
    for _ in tqdm(range(flight_count)):
        experiment_result = simulate_flight(flight_conditions, aircraft_model, policy, termal_time_step_s)
        list_experiment_results.append(experiment_result)
    return ExperimentOutputBatch(list_experiment_outputs=list_experiment_results)
