"""Repository for caching JSON-serializable dicts to the filesystem."""

import json
from pathlib import Path


class RepositoryCache:
    """Cache dicts as JSON files in a directory."""

    def __init__(self, path_dir: Path) -> None:
        self.path_dir = Path(path_dir)

    def _path(self, key: str) -> Path:
        key = key.strip("/").replace("/", "_")
        if not key.endswith(".json"):
            key = f"{key}.json"
        return self.path_dir / key

    def get(self, key: str) -> dict | None:
        """Load cached dict by key. Returns None if missing or invalid."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, data: dict) -> None:
        """Save dict to cache."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
