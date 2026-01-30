from paraai.flight_conditions import FlightConditionsDistribution
from paraai.model import AircraftModel
from paraai.simulator_crude import SimulatorCrude
from paraai.tools_plot import plot_flight

flight_conditions_distribution = FlightConditionsDistribution(
    termal_net_climb_mean_min_m_s=0.5,
    termal_net_climb_mean_max_m_s=0.5,
)
aircraft_model = AircraftModel(
    velocity_max_m_s=10,
    sink_max_m_s=-1,
)

from paraai.flight_policy import FlightPolicyAlwaysTermal, FlightPolicyNeverTermal, FlightPolicyThreeZones

flight_policy_nt = FlightPolicyNeverTermal()

flight_policy_at = FlightPolicyAlwaysTermal()
# flight_policy_at = FlightPolicyAlwaysTermal()
flight_policy_tz = FlightPolicyThreeZones(0.9, 0.5)
# flight_policy_nn = FlightPolicyNeuralNetwork(
#     model_path=Path("data", "database_model", "neural_policy.pth"),
# )

simulator = SimulatorCrude(
    flight_condition_distribution=flight_conditions_distribution,
    aircraft_model=aircraft_model,
)
experiment_result = simulator.simulate_flight(
    flight_policy_tz,
)

plot_flight(experiment_result)
