"""Repository for train logs from map builder training. Uses same cache ID as RepositoryModels."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from paraai.model.train_log import TrainLog

logger = logging.getLogger(__name__)


class RepositoryTrainLogs:
    """Cache for train logs. Keyed by builder name and params (same ID as RepositoryModels)."""

    instance: Optional[RepositoryTrainLogs] = None

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def initialize(cache_dir: Path) -> RepositoryTrainLogs:
        if RepositoryTrainLogs.instance is not None:
            raise ValueError("RepositoryTrainLogs already initialized")
        RepositoryTrainLogs.instance = RepositoryTrainLogs(cache_dir)
        return RepositoryTrainLogs.instance

    @staticmethod
    def get_instance() -> RepositoryTrainLogs:
        if not hasattr(RepositoryTrainLogs, "instance") or RepositoryTrainLogs.instance is None:
            raise ValueError("RepositoryTrainLogs not initialized")
        return RepositoryTrainLogs.instance

    def get_train_log(self, builder_name: str, model_id: str) -> TrainLog | None:
        """Load train log from cache if it exists. Uses same ID as model repository."""
        path = self._path(builder_name, model_id)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("Loaded train log from cache %s", path)
        return TrainLog.from_dict(data)

    def save_train_log(
        self,
        builder_name: str,
        model_id: str,
        train_log: TrainLog,
    ) -> Path:
        """Save train log to cache. Uses same ID as model repository. Returns path."""
        path = self._path(builder_name, model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(train_log.to_dict(), f, indent=2)
        logger.debug("Saved train log to %s", path)
        return path

    def _path(self, builder_name: str, cache_id: str) -> Path:
        safe_builder = str(builder_name).replace(" ", "_").replace(":", "_")
        return self.cache_dir / safe_builder / f"{cache_id}.json"

    def get_log_path(self, builder_name: str, model_id: str) -> Path:
        """Return path to train log for given builder and model_id."""
        return self._path(builder_name, model_id)

    def list_train_logs(self, builder_name: str | None = None) -> list[Path]:
        """List all train log files. If builder_name given, filter to that builder."""
        if builder_name is not None:
            safe_builder = str(builder_name).replace(" ", "_").replace(":", "_")
            dir_path = self.cache_dir / safe_builder
            if not dir_path.exists():
                return []
            return sorted(dir_path.glob("*.json"))
        paths: list[Path] = []
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                paths.extend(sorted(subdir.glob("*.json")))
        return sorted(paths)
