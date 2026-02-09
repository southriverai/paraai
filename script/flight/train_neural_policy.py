"""
Generate training data for neural network policy.
Predicts: If we leave the current thermal, will we find stronger lift?
"""
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from paraai.flight_conditions import FlightConditionsDistributionChoice
from paraai.flight_policy_neural import FlightPolicyNeuralNetwork
from paraai.model import AircraftModel
from paraai.respository import Repository
from paraai.simulator import Simulator
from paraai.simulator_crude import SimulatorCrude


def do_train_rl(
    policy: FlightPolicyNeuralNetwork,
    simulator: Simulator,
    rollout_count: int,
    simulations_per_rollout: int,
    epochs_per_rollout: int,
    learning_rate: float,
    experiment_id: str,
) -> None:
    """Train model."""
    # preserve best policy (most mean distance)
    best_distance = 0
    best_model_path = Path("data", "experiments", experiment_id, "best", "best_model.pth")
    for rollout_idx in range(rollout_count):
        print(f"Rolling out {rollout_idx} of {rollout_count}")
        simulation_results = simulator.simulate_batch(policy, simulations_per_rollout)
        # comput mean distance traveled
        list_distance_m = []
        for experiment_output in simulation_results.list_experiment_outputs:
            list_distance_m.append(experiment_output.flight_state.list_distance_m[-1])
        mean_distance = np.mean(list_distance_m)
        if best_distance < mean_distance:
            best_distance = mean_distance
            policy.save_file(best_model_path)
        print(f"Mean distance: {mean_distance}")
        input_matrix, output_matrix = policy.convert_to_matrixes(
            simulation_results,
        )

        input__tensor = torch.FloatTensor(input_matrix)
        output_tensor = torch.FloatTensor(output_matrix)
        policy.network.train()
        optimizer = torch.optim.Adam(policy.network.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()  # Use MSE for regression (continuous targets), not BCELoss (binary classification)
        list_loss = []
        pbar = tqdm(range(epochs_per_rollout))
        for _ in pbar:
            optimizer.zero_grad()
            output = policy.network(input__tensor)
            loss = criterion(output, output_tensor)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            list_loss.append(loss_value)
            pbar.set_postfix({"loss": loss_value})
        policy.network.eval()
        policy.save()


def train_rl(
    experiment_id: str,
    hidden_sizes,
    load_best_model: bool = True,
) -> None:
    model_path = Path("data", "experiments", experiment_id, "models", "neural_policy.pth")

    policy = FlightPolicyNeuralNetwork(
        policy_name="NeuralNetwork",
        model_path=model_path,
        hidden_sizes=hidden_sizes,
    )
    best_model_path = Path("data", "experiments", experiment_id, "best", "best_model.pth")
    if load_best_model and best_model_path.exists():
        # load best model
        policy.load_path_file(best_model_path)

    aircraft_model = AircraftModel(
        velocity_max_m_s=10,
        sink_max_m_s=-1,
    )
    flight_condition_distribution = FlightConditionsDistributionChoice(
        list_climbs_mean_m_s=[0.5],
    )
    simulator = SimulatorCrude(
        flight_condition_distribution=flight_condition_distribution,
        aircraft_model=aircraft_model,
    )
    do_train_rl(
        policy,
        simulator,
        rollout_count=200,
        simulations_per_rollout=2000,
        epochs_per_rollout=100,
        learning_rate=0.001,
        experiment_id=experiment_id,
    )


if __name__ == "__main__":
    repository = Repository.initialize()

    experiment_id = "ex_32_16_4"
    hidden_sizes = [32, 16]
    # Create experiment directories
    experiment_models_dir = Path("data", "experiments", experiment_id, "models")
    experiment_best_dir = Path("data", "experiments", experiment_id, "best")
    experiment_models_dir.mkdir(parents=True, exist_ok=True)
    experiment_best_dir.mkdir(parents=True, exist_ok=True)

    train_rl(
        experiment_id=experiment_id,
        hidden_sizes=hidden_sizes,
        load_best_model=True,
    )
