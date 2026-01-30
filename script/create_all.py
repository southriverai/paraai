from pathlib import Path

import numpy as np
import pandas as pd

from paraai.experiment import ExperimentOutputBatch
from paraai.flight_conditions import FlightConditionsDistribution
from paraai.flight_policy import FlightPolicyAlwaysTermal, FlightPolicyBase, FlightPolicyNeverTermal, FlightPolicyThreeZones
from paraai.model import AircraftModel
from paraai.simulator_crude import SimulatorCrude
from paraai.tools_plot import get_statistics


def creat_table():
    aircraft_model = AircraftModel(
        velocity_max_m_s=10,
        sink_max_m_s=-1,
    )
    list_condition_names = ["cond_05", "cond_20", "cond_40"]
    list_condition_terms = [0.5, 2.0, 4.0]
    flight_policies: list[FlightPolicyBase] = []
    flight_policies.append(FlightPolicyNeverTermal())
    flight_policies.append(FlightPolicyAlwaysTermal())
    flight_policies.append(FlightPolicyThreeZones(1.0, 0.7))
    flight_policies.append(FlightPolicyThreeZones(0.9, 0.5))
    flight_policies.append(FlightPolicyThreeZones(0.8, 0.3))
    flight_policies.append(FlightPolicyThreeZones(0.7, 0.1))
    flight_policies.append(FlightPolicyThreeZones(0.6, 0.0))

    policy_labels = [policy.policy_name for policy in flight_policies]

    # Collect statistics for all policies and conditions
    results_p50 = {}
    results_p90 = {}

    for condition_name, condition_term in zip(list_condition_names, list_condition_terms):
        for flight_policy in flight_policies:
            flight_conditions_distribution = FlightConditionsDistribution(
                termal_net_climb_mean_min_m_s=condition_term,
                termal_net_climb_mean_max_m_s=condition_term,
            )
            simulator = SimulatorCrude(
                flight_condition_distribution=flight_conditions_distribution,
                aircraft_model=aircraft_model,
            )

            path_file_result = Path("data", "results", condition_name, f"{flight_policy.get_hash()}.json")
            # create directory if not exists
            path_file_result.parent.mkdir(parents=True, exist_ok=True)
            if Path(path_file_result).exists():
                experiment_output_batch = ExperimentOutputBatch.model_validate_json(Path(path_file_result).read_text())
            else:
                experiment_output_batch = simulator.simulate_batch(
                    flight_policy,
                    num_simulations=1000,
                )
                Path(path_file_result).write_text(experiment_output_batch.model_dump_json())

            # Get statistics for all policies
            stats = get_statistics(experiment_output_batch)

            # Store results by condition
            if flight_policy.policy_name not in results_p50:
                results_p50[flight_policy.policy_name] = {}
            if flight_policy.policy_name not in results_p90:
                results_p90[flight_policy.policy_name] = {}
            results_p50[flight_policy.policy_name][condition_name] = stats["p50_distance"]
            results_p90[flight_policy.policy_name][condition_name] = stats["p90_distance"]

    # Create DataFrame: policies as rows, conditions as columns
    # Combine p50 and p90 with MultiIndex columns
    data = {}
    for policy_name in policy_labels:
        row_data = {}
        for condition_name in list_condition_names:
            row_data[("p50", condition_name)] = results_p50.get(policy_name, {}).get(condition_name, np.nan)
            row_data[("p90", condition_name)] = results_p90.get(policy_name, {}).get(condition_name, np.nan)
        data[policy_name] = row_data

    results_df = pd.DataFrame(data).T
    results_df.columns = pd.MultiIndex.from_tuples(results_df.columns, names=["metric", "condition"])
    # Ensure proper column order
    results_df = results_df.reindex(columns=pd.MultiIndex.from_product([["p50", "p90"], list_condition_names]))
    results_df.index.name = "Policy"

    print("\nResults DataFrame (policies as rows, conditions as columns):")
    print(results_df)


if __name__ == "__main__":
    creat_table()
