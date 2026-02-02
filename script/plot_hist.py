from pathlib import Path

from paraai.experiment import ExperimentOutputBatch
from paraai.flight_conditions import FlightConditionsDistribution
from paraai.flight_policy import FlightPolicyAlwaysTermal, FlightPolicyThreeZones
from paraai.flight_policy_neural import FlightPolicyNeuralNetwork
from paraai.model import AircraftModel
from paraai.simulator_crude import SimulatorCrude
from paraai.tools_plot import plot_flight_hists_2


def plot_hist(show: bool = False):
    aircraft_model = AircraftModel(
        velocity_max_m_s=10,
        sink_max_m_s=-1,
    )

    flight_policy_at = FlightPolicyAlwaysTermal()
    flight_policy_tz = FlightPolicyThreeZones(0.9, 0.5)
    flight_policy_nn = FlightPolicyNeuralNetwork(
        policy_name="NeuralNetwork",
        model_path=Path("models", "neural_policy.pth"),
        hidden_sizes=[32, 16],
    )
    flight_policies = [flight_policy_at, flight_policy_tz, flight_policy_nn]

    list_condition_names = ["cond_05"]
    list_condition_terms = [0.5]
    for condition_name, condition_term in zip(list_condition_names, list_condition_terms):
        flight_conditions_distribution = FlightConditionsDistribution(
            termal_net_climb_mean_min_m_s=condition_term,
            termal_net_climb_mean_max_m_s=condition_term,
        )
        simulator = SimulatorCrude(
            flight_condition_distribution=flight_conditions_distribution,
            aircraft_model=aircraft_model,
        )

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
        path_file = Path("data", "images", f"hist_{condition_name}.png")
        plot_flight_hists_2(experiment_result_baches, labels, path_file=path_file, show=show)


if __name__ == "__main__":
    plot_hist(True)
