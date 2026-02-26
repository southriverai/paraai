import math

from tqdm import tqdm

from paraai.experiment import ExperimentOutput, ExperimentOutputBatch
from paraai.flight_conditions import FlightConditions, Thermal
from paraai.flight_policy import FlightPolicyBase
from paraai.model import AircraftModel, FlightState


def thermal_strength_m_s(day_of_year: int) -> float:
    """Return average thermal strength (vertical speed m/s) for day of year 0-364.

    Based on two-sine fit: c + A1*sin(2πx/365 + φ1) + A2*sin(4πx/365 + φ2).
    """

    # Two-sine fit from show_months.py (mean climb speed m/s vs day of year)
    _THERMAL_STRENGTH_C = 0.9187
    _THERMAL_STRENGTH_A1, _THERMAL_STRENGTH_PHI1 = 0.1619, -1.2368
    _THERMAL_STRENGTH_A2, _THERMAL_STRENGTH_PHI2 = 0.0375, -1.3013
    x = day_of_year % 365
    t1 = 2 * math.pi * x / 365 + _THERMAL_STRENGTH_PHI1
    t2 = 4 * math.pi * x / 365 + _THERMAL_STRENGTH_PHI2
    return _THERMAL_STRENGTH_C + _THERMAL_STRENGTH_A1 * math.sin(t1) + _THERMAL_STRENGTH_A2 * math.sin(t2)


def simulate_thermal(
    flight_conditions: FlightConditions,
    aircraft_model: AircraftModel,
    flight_state: FlightState,
    thermal: Thermal,
    time_step_s: float,
):
    #    print(f"Simulating thermal {thermal.distance_start_m} to {thermal.distance_end_m}")
    last_node_time_s = flight_state.list_time_s[-1]
    last_node_altitude_m = flight_state.list_altitude_m[-1]
    last_node_distance_m = flight_state.list_distance_m[-1]

    node_time_s = last_node_time_s + time_step_s
    node_distance_m = last_node_distance_m
    node_altitude_m = last_node_altitude_m + thermal.net_climb_m_s * time_step_s
    is_landed = False

    # Check if we've reached thermal ceiling
    node_altitude_m = min(flight_conditions.thermal_ceiling_m, node_altitude_m)

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


def simulate_progress_to_thermal(
    flight_conditions: FlightConditions,
    aircraft_model: AircraftModel,
    flight_state: FlightState,
    thermal: Thermal,
):
    last_node_time_s = flight_state.list_time_s[-1]
    last_node_altitude_m = flight_state.list_altitude_m[-1]
    last_node_distance_m = flight_state.list_distance_m[-1]

    thermal_distance_m = (thermal.distance_start_m + thermal.distance_end_m) / 2
    distance_to_thermal_m = thermal_distance_m - last_node_distance_m
    time_step_s = distance_to_thermal_m / aircraft_model.velocity_max_m_s

    # print(f"Simulating progress to thermal {distance_to_thermal_m} m at {time_step_s} s")

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

    # if we have not landed yetthen add one second of lift to the state so the policy can decide to use thermal
    if not is_landed:
        flight_state.list_time_s.append(node_time_s + 1)
        flight_state.list_distance_m.append(node_distance_m)
        flight_state.list_altitude_m.append(node_altitude_m + thermal.net_climb_m_s * 1)
        flight_state.list_use_thermal.append(True)
        flight_state.is_landed = False


def simulate_flight(
    flight_conditions: FlightConditions,
    aircraft_model: AircraftModel,
    policy: FlightPolicyBase,
    thermal_time_step_s: float = 1.0,
) -> ExperimentOutput:
    thermals = flight_conditions.sample_thermals()
    thermal_index = 0

    time_s = flight_conditions.take_off_time_s
    # add initial state
    flight_state = FlightState(
        list_time_s=[time_s],
        list_altitude_m=[flight_conditions.take_off_altitude_m],
        list_distance_m=[0],
        list_use_thermal=[False],
        is_landed=False,
    )

    # TODO in the futere we might want to make steps along this path rather than just jumping to the next thermal to do some best glide stuff
    while not flight_state.is_landed:
        # if there are no more thermals we land
        if thermal_index >= len(thermals):
            flight_state.is_landed = True  # TODO we should glide to out
            break
        # get the next thermal
        next_thermal = thermals[thermal_index]
        simulate_progress_to_thermal(flight_conditions, aircraft_model, flight_state, next_thermal)
        while policy.use_thermal(flight_state, aircraft_model):
            simulate_thermal(flight_conditions, aircraft_model, flight_state, next_thermal, thermal_time_step_s)
        thermal_index += 1
    return ExperimentOutput(
        flight_state=flight_state,
        thermals=thermals,
        aircraft_model=aircraft_model,
    )


def simulate_flight_many(
    flight_conditions: FlightConditions,
    aircraft_model: AircraftModel,
    policy: FlightPolicyBase,
    flight_count: int,
    thermal_time_step_s: float = 1.0,
) -> ExperimentOutputBatch:
    list_experiment_results: list[ExperimentOutput] = []
    for _ in tqdm(range(flight_count)):
        experiment_result = simulate_flight(flight_conditions, aircraft_model, policy, thermal_time_step_s)
        list_experiment_results.append(experiment_result)
    return ExperimentOutputBatch(list_experiment_outputs=list_experiment_results)
