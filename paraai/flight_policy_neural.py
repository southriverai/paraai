import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from paraai.experiment import ExperimentOutputBatch
from paraai.flight_policy import FlightPolicyBase
from paraai.model import AircraftModel, FlightState


class ThermalPolicyNetwork(nn.Module):
    """
    Neural network for thermal policy decision making.
    Input features:
    - 10 deciles of past lift (10 values)
    - Current altitude (1 value)
    - Current lift (1 value)
    - Time since takeoff (1 value)
    Total: 13 input features
    Output: Single value (probability of using thermal)
    """

    def __init__(
        self,
        input_size: int = 13,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Use default hidden sizes if not provided
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            # Ensure hidden_size is an integer
            if not isinstance(hidden_size, int):
                raise TypeError(f"hidden_size must be an integer, got {type(hidden_size)}: {hidden_size}")
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer: single value (probability)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # Output probability between 0 and 1

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FlightPolicyNeuralNetwork(FlightPolicyBase):
    """
    Flight policy that uses a neural network to decide whether to use thermal.

    The network predicts: "If we leave the current thermal, will we find stronger lift?"
    - If prediction_use_thermal > prediction_no_thermal: Leave thermal (return False) - stronger lift ahead
    - If prediction_use_thermal <= prediction_no_thermal: Stay in thermal (return True) - current thermal is better

    Features:
    - 10 deciles of past lift (from all climb rates)
    - Current altitude
    - Current lift rate
    - Time since takeoff
    """

    def __init__(
        self,
        policy_name: str,
        model_path: Path,
        hidden_sizes: list[int],
    ):
        """
        Initialize neural network policy.

        Args:
            model_path: Path to saved model weights (if None, uses random initialization)
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
        """
        super().__init__(policy_name=policy_name)

        # Use default hidden sizes if not provided
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        dropout = 0.1
        # Initialize network
        self.network = ThermalPolicyNetwork(
            input_size=14,  # 10 deciles + altitude + current_lift + time + use_thermal
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

        self.model_path = model_path
        if self.model_path.exists():
            # load model
            self.load_path_file(self.model_path)
        else:
            # initialize weights
            self._init_weights()

    def _init_weights(self) -> None:
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose3d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def convert_to_matrixes(
        self,
        experiment_output_batch: ExperimentOutputBatch,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert simulation results to matrixes."""
        list_input_array = []
        list_output_array = []

        for experiment_output in experiment_output_batch.list_experiment_outputs:
            # get_final_time and distance from flight_state

            flight_state = experiment_output.flight_state.model_copy()
            final_distance_m = experiment_output.flight_state.list_distance_m[-1]
            final_time_s = experiment_output.flight_state.list_time_s[-1]

            # remove the last time step
            flight_state.list_time_s.pop()
            flight_state.list_distance_m.pop()
            flight_state.list_altitude_m.pop()
            flight_state.list_use_thermal.pop()
            flight_state.status = "flying"

            # now go backwards in time time and exponentially discount rewards
            # by fraction of max time
            while len(flight_state.list_time_s) > 0:
                time_s = flight_state.list_time_s[-1]
                use_thermal = flight_state.list_use_thermal[-1]

                input_array = self._extract_features(
                    flight_state,
                    experiment_output.aircraft_model,
                    use_thermal=use_thermal,
                )
                output_array = np.array([(final_distance_m / 100000.0)], dtype=np.float32)  # 300 km is max reward
                list_input_array.append(input_array)
                list_output_array.append(output_array)
                # discount reward by fraction of max time

                # move to previous time step
                flight_state.list_time_s.pop()
                flight_state.list_distance_m.pop()
                flight_state.list_altitude_m.pop()
                flight_state.list_use_thermal.pop()

        return np.array(list_input_array), np.array(list_output_array)

    def _extract_features(
        self,
        flight_state: FlightState,
        aircraft_model: AircraftModel,
        use_thermal: bool,
    ) -> np.ndarray:
        """
        Extract features from flight state.

        Returns:
            numpy array of shape (13,) with:
            - 10 deciles of past lift
            - Current altitude (normalized)
            - Current lift rate
            - Time since takeoff (normalized)
        """
        # Get all past climb rates using helper method
        past_climbs = flight_state.all_climbs()

        # Calculate 10 deciles of past lift
        if len(past_climbs) == 0:
            deciles = [0.0] * 10
        else:
            deciles = [np.quantile(past_climbs, q / 10.0) for q in range(1, 11)]  # 10th, 20th, ..., 100th percentile

        # Current altitude (normalized by starting altitude)
        if len(flight_state.list_altitude_m) > 0:
            starting_altitude = flight_state.list_altitude_m[0]
            current_altitude = flight_state.list_altitude_m[-1]
            normalized_altitude = current_altitude / starting_altitude
        else:
            normalized_altitude = 0.0
        # Current lift rate
        current_lift = flight_state.current_climb_m_s()

        # Time since takeoff (normalized by max expected time, e.g., 6 hours)
        if len(flight_state.list_time_s) > 0:
            time_since_takeoff = flight_state.list_time_s[-1] - flight_state.list_time_s[0]
            # Normalize: assume max flight time is 6 hours (21600 seconds)
            max_time_s = 21600.0
            normalized_time = min(time_since_takeoff / max_time_s, 1.0)
        else:
            normalized_time = 0.0

        if use_thermal:
            use_thermal_feature = 1.0
        else:
            use_thermal_feature = 0.0
        # Combine all features
        features = np.array([*deciles, normalized_altitude, current_lift, normalized_time, use_thermal_feature], dtype=np.float32)
        # make it 1 by n array
        return features

    def use_termal(self, flight_state: FlightState, aircraft_model: AircraftModel) -> bool:
        """
        Use neural network to decide whether to use thermal.

        The network predicts: "If we leave, will we find stronger lift?"
        - If prediction_use_thermal > prediction_no_thermal: Leave thermal (return False) - stronger lift ahead
        - If prediction_use_thermal <= prediction_no_thermal: Stay in thermal (return True) - current thermal is better

        Returns:
            True to stay in thermal, False to leave thermal
        """
        # if we are not in current thermal, return Fasle

        if flight_state.current_climb_m_s() <= 0.01:
            return False
        # Extract features
        features_use_thermal = self._extract_features(flight_state, aircraft_model, use_thermal=True)
        features_no_thermal = self._extract_features(flight_state, aircraft_model, use_thermal=False)

        # Convert to tensor and add batch dimension
        features_tensor_use_thermal = torch.FloatTensor(features_use_thermal).unsqueeze(0)
        features_tensor_no_thermal = torch.FloatTensor(features_no_thermal).unsqueeze(0)

        # Forward pass (no gradient computation needed for inference)
        with torch.no_grad():
            output_use_thermal = self.network(features_tensor_use_thermal)
            output_no_thermal = self.network(features_tensor_no_thermal)
            probability_no_thermal = output_no_thermal.item()
            probability_use_thermal = output_use_thermal.item()

        # Decision logic:
        return probability_use_thermal > probability_no_thermal

    def get_hash(self) -> str:
        """Generate hash for uniqueness."""
        hash_string = f"{self.policy_name}"
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epoch_count,
        lr: float,
    ) -> None:
        """Train model."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Training on CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("CUDA not available — training on CPU. " "Install PyTorch with CUDA: https://pytorch.org/get-started/locally/")

        self.network.to(device)
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)
        self.network.train()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()
        list_loss = []
        for _ in tqdm(range(epoch_count)):
            optimizer.zero_grad()
            output = self.network(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            list_loss.append(loss.item())
        self.network.eval()
        self.network.to("cpu")  # back to CPU for inference / save
        # plot loss function
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

        plt.plot(list_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Function")
        plt.show()

    def save(self) -> None:
        self.save_file(self.model_path)

    def save_file(self, path_file: Path) -> None:
        """Save model."""
        print(f"Saving model to {path_file}")
        # make sure partent dir exists
        path_file.parent.mkdir(parents=True, exist_ok=True)
        # save model
        torch.save(self.network.state_dict(), path_file)
        print(f"Model saved to {path_file}")

    def load_path_file(self, path_file: Path) -> None:
        # Load weights if provided
        if path_file.exists():
            print(f"Loading model weights from {path_file}")
            self.network.load_state_dict(torch.load(path_file, map_location="cuda", weights_only=True))
            # Set to evaluation mode
            self.network.eval()
        else:
            print(f"Model weights not found at {path_file}. Using random initialization.")
            self._init_weights()
