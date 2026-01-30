from pydantic import BaseModel, Field

from paraai.flight_conditions import FlightConditions, Termal
from paraai.model import AircraftModel, FlightState


class ExperimentInput(BaseModel):
    flight_conditions: FlightConditions
    aircraft_models: list[AircraftModel]
    flight_policies: list[str]
    random_seed: int
    flight_count: int
    termal_time_step_s: float


class ExperimentOutput(BaseModel):
    aircraft_model: AircraftModel
    flight_state: FlightState
    termals: list[Termal]


class ExperimentOutputBatch(BaseModel):
    list_experiment_outputs: list[ExperimentOutput] = Field(default_factory=list)
