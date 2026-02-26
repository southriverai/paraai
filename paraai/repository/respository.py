from pathlib import Path
from typing import Optional

import numpy as np


class Repository:
    instance: Optional["Repository"] = None

    def __init__(self):
        self.path_data = Path("data")
        self.path_data.mkdir(parents=True, exist_ok=True)
        self.path_data_training = Path(self.path_data / "database_training")
        self.path_data_model = Path(self.path_data / "database_model")
        self.path_data_training.mkdir(parents=True, exist_ok=True)
        self.path_data_model.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def initialize() -> None:
        if Repository.instance is not None:
            raise ValueError("Repository already initialized")
        Repository.instance = Repository()

    @staticmethod
    def get_instance() -> "Repository":
        if not hasattr(Repository, "instance"):
            raise ValueError("Repository not initialized")
        return Repository.instance

    def save_training_data(
        self,
        file_name: str,
        X: np.ndarray,
        y: np.ndarray,
    ):
        path_file_training_data = self.path_data_training / file_name
        path_file_training_data.parent.mkdir(parents=True, exist_ok=True)

        """Save training data to file."""
        np.savez(path_file_training_data, X=X, y=y)
        print(f"Saved training data: {X.shape[0]} samples to {self.path_data_training / path_file_training_data}")

    def load_training_data(self, file_name: str) -> tuple[np.ndarray, np.ndarray]:
        path_file_training_data = self.path_data_training / file_name
        """Load training data from file."""
        data = np.load(path_file_training_data)
        return data["X"], data["y"]
