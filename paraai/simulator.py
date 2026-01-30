from tqdm import tqdm

from paraai.experiment import ExperimentOutput, ExperimentOutputBatch
from paraai.flight_conditions import FlightConditionsDistribution
from paraai.flight_policy import FlightPolicyBase
from paraai.model import AircraftModel
from paraai.tools_sim import simulate_flight


class Simulator:
    def __init__(
        self,
        flight_condition_distribution: FlightConditionsDistribution,
        aircraft_model: AircraftModel,
        termal_time_step_s: float = 60.0,
    ):
        self.flight_condition_distribution = flight_condition_distribution
        self.aircraft_model = aircraft_model
        self.termal_time_step_s = termal_time_step_s

    def simulate_batch(
        self,
        policy: FlightPolicyBase,
        num_simulations: int,
    ) -> ExperimentOutputBatch:
        """Simulate a batch of flights."""
        list_experiment_outputs: list[ExperimentOutput] = []
        for _ in tqdm(range(num_simulations)):
            experiment_output = self.simulate(policy)
            list_experiment_outputs.append(experiment_output)
        return ExperimentOutputBatch(
            list_experiment_outputs=list_experiment_outputs,
        )

    def simulate(self, policy: FlightPolicyBase) -> ExperimentOutput:
        """Simulate a flight."""
        flight_conditions = self.flight_condition_distribution.sample()
        return simulate_flight(
            flight_conditions,
            self.aircraft_model,
            policy,
            self.termal_time_step_s,
        )
