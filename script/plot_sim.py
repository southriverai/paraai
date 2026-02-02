from pathlib import Path

from paraai.flight_conditions import FlightConditionsDistribution
from paraai.flight_policy import FlightPolicyThreeZones
from paraai.model import AircraftModel
from paraai.simulator_crude import SimulatorCrude
from paraai.tools_plot import plot_flight


def plot_sim(show: bool = True):
    flight_conditions_distribution = FlightConditionsDistribution(
        termal_net_climb_mean_min_m_s=0.5,
        termal_net_climb_mean_max_m_s=0.5,
    )

    aircraft_model = AircraftModel(
        velocity_max_m_s=10,
        sink_max_m_s=-1,
    )
    flight_policy_tz = FlightPolicyThreeZones(0.9, 0.5)

    simulator = SimulatorCrude(
        flight_condition_distribution=flight_conditions_distribution,
        aircraft_model=aircraft_model,
    )
    experiment_result = simulator.simulate_flight(
        flight_policy_tz,
    )

    path_file = Path("data", "images", "sim.png")
    plot_flight(experiment_result, path_file=path_file, show=show)


if __name__ == "__main__":
    plot_sim(show=True)
