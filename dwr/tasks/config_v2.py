"""Closed, byte-pinned loader for the reviewed v2 task-generation contract."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .run_support import ConfigError, LoadedConfig, sha256_file, stable_json, strict_json_loads


_APPROVED_SHA256 = "cc9fb0a827d27e4f00425ca9364365026bb6c70c75dc8d6dc0eebcc1f6865222"


def _same_typed_value(actual: object, expected: object) -> bool:
    """Compare JSON trees while keeping bool distinct from integer values."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys() and
                all(_same_typed_value(actual[key], expected[key]) for key in expected))
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_value(left, right) for left, right in zip(actual, expected))
    return actual == expected


def load_config_v2(path: str | Path) -> LoadedConfig:
    """Load only the exact registered v2 bytes and typed scientific identity."""
    path = Path(path).resolve()
    registered = Path(__file__).resolve().parents[2] / "configs" / "tasks" / "constraint_v2.json"
    if sha256_file(registered) != _APPROVED_SHA256:
        raise ConfigError("registered v2 configuration bytes differ from the reviewed baseline")
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != _APPROVED_SHA256:
            raise ConfigError("configuration bytes differ from the reviewed raw v2 baseline")
        current = strict_json_loads(raw.decode("utf-8"))
        baseline = strict_json_loads(registered.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise ConfigError("configuration must be readable UTF-8 without a BOM") from exc
    if not isinstance(current, dict) or not isinstance(baseline, dict):
        raise ConfigError("configuration must be an object")
    if not _same_typed_value(current, baseline):
        raise ConfigError("configuration differs from the registered closed v2 scientific contract")
    return LoadedConfig(current, sha256_file(path), path)
