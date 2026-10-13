"""Closed, byte-pinned loader for the reviewed v3 task-generation contract."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .config_v2 import _same_typed_value
from .run_support import ConfigError, LoadedConfig, sha256_file, strict_json_loads


_APPROVED_SHA256 = "e5d7de77fc576b56847e7a55e401cd6c318f4bc35f5aa7f843ef41d1af9ee025"


def load_config_v3(path: str | Path) -> LoadedConfig:
    """Load only the exact reviewed v3 bytes and closed typed contract."""
    path = Path(path).resolve()
    registered = Path(__file__).resolve().parents[2] / "configs" / "tasks" / "constraint_v3.json"
    if sha256_file(registered) != _APPROVED_SHA256:
        raise ConfigError("registered v3 configuration bytes differ from the reviewed baseline")
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != _APPROVED_SHA256:
            raise ConfigError("configuration bytes differ from the reviewed raw v3 baseline")
        current = strict_json_loads(raw.decode("utf-8"))
        baseline = strict_json_loads(registered.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise ConfigError("configuration must be readable UTF-8 without a BOM") from exc
    if not isinstance(current, dict) or not isinstance(baseline, dict):
        raise ConfigError("configuration must be an object")
    if not _same_typed_value(current, baseline):
        raise ConfigError("configuration differs from the registered closed v3 scientific contract")
    return LoadedConfig(current, sha256_file(path), path)
