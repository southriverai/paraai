"""String utilities for cache paths and identifiers."""

import hashlib
import json


def validate_safe_name(name: str) -> None:
    """Raise ValueError if name contains spaces or colons (unsafe for cache paths)."""
    if " " in name or ":" in name:
        raise ValueError(
            f"{name} must not contain spaces or colons (got {name!r}). " "Use underscores instead for cache path compatibility."
        )


def dict_to_cache_id(**params: object) -> str:
    """Get cache ID for train log from dict of params."""
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]
