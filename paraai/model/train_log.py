"""Train log model for map builder training runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainLog:
    """Training run log: region, builder name, and per-epoch metrics."""

    region: str
    builder_name: str
    epochs: list[dict]

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "region": self.region,
            "builder": self.builder_name,
            "epochs": self.epochs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrainLog:
        """Deserialize from dict (JSON loaded)."""
        return cls(
            region=data.get("region", ""),
            builder_name=data.get("builder", ""),
            epochs=data.get("epochs", []),
        )
