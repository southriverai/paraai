from pathlib import Path

from paraai.experiment import ExperimentOutputBatch
from paraai.flight_conditions import FlightConditionsDistribution
from paraai.flight_policy import FlightPolicyAlwaysTermal, FlightPolicyNeverTermal, FlightPolicyThreeZones
from paraai.model import AircraftModel
from paraai.simulator_crude import SimulatorCrude
from paraai.tools_plot import print_statistics

aircraft_model = AircraftModel(
    velocity_max_m_s=10,
    sink_max_m_s=-1,
)

flight_policy_nt = FlightPolicyNeverTermal()
flight_policy_at = FlightPolicyAlwaysTermal()
flight_policies = []
list_threshold_z1 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
list_threshold_z2 = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
experiment_id = "ex_32_16_0"


list_condition_names = ["cond_05", "cond_20", "cond_40"]
list_condition_terms = [0.5, 2.0, 4.0]
for condition_name, condition_term in zip(list_condition_names, list_condition_terms):
    flight_conditions_distribution = FlightConditionsDistribution(
        termal_net_climb_mean_min_m_s=condition_term,
        termal_net_climb_mean_max_m_s=condition_term,
    )
    simulator = SimulatorCrude(
        flight_condition_distribution=flight_conditions_distribution,
        aircraft_model=aircraft_model,
    )
    for threshold_z1 in list_threshold_z1:
        for threshold_z2 in list_threshold_z2:
            flight_policies.append(FlightPolicyThreeZones(threshold_z1, threshold_z2))

    experiment_result_baches: list[ExperimentOutputBatch] = []
    labels = []
    for flight_policy in flight_policies:
        path_file_result = Path("data", "results", condition_name, f"{flight_policy.get_hash()}.json")
        # create directory if not exists
        path_file_result.parent.mkdir(parents=True, exist_ok=True)
        if Path(path_file_result).exists():
            experiment_output_batch = ExperimentOutputBatch.model_validate_json(Path(path_file_result).read_text())
            experiment_result_baches.append(experiment_output_batch)
        else:
            experiment_output_batch = simulator.simulate_batch(
                flight_policy,
                num_simulations=1000,
            )
            experiment_result_baches.append(experiment_output_batch)
            Path(path_file_result).write_text(experiment_output_batch.model_dump_json())
        labels.append(flight_policy.policy_name)
    print(condition_name)
    print_statistics(experiment_result_baches, labels)

# best p50 policy ThreeZones 0.8 0.3
# best p90 policy ThreeZones 0.9 0.6


# flight_policies.append(FlightPolicyThreeZones(0.7, 0.4))
# flight_policies.append(FlightPolicyThreeZones(0.7, 0.3))
# flight_policies.append(FlightPolicyThreeZones(0.7, 0.2))
# flight_policies.append(FlightPolicyThreeZones(0.7, 0.1))


# flight_policies.append(FlightPolicyThreeZones(0.6, 0.6))
# flight_policies.append(FlightPolicyThreeZones(0.6, 0.5))
# flight_policies.append(FlightPolicyThreeZones(0.6, 0.4))  # p95
# flight_policies.append(FlightPolicyThreeZones(0.6, 0.3))
# flight_policies.append(FlightPolicyThreeZones(0.6, 0.2))
# flight_policies.append(FlightPolicyThreeZones(0.6, 0.1))
# flight_policies.append(FlightPolicyThreeZones(0.6, 0.0))

# flight_policies.append(FlightPolicyThreeZones(0.5, 0.4))
# flight_policies.append(FlightPolicyThreeZones(0.5, 0.3))  # p50
# flight_policies.append(FlightPolicyThreeZones(0.5, 0.2))
# flight_policies.append(FlightPolicyThreeZones(0.5, 0.1))
# flight_policies.append(FlightPolicyThreeZones(0.5, 0.0))


# path_file_weights = Path("data", "experiments", experiment_id, "best", "best_model.pth")
# flight_policies.append(
#     FlightPolicyNeuralNetwork(
#         model_path=path_file_weights,
#         hidden_sizes=[32, 16],
#     )
# )
