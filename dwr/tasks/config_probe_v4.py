"""Closed loader for the approved v4 development-proposal configuration."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


APPROVED_CONFIG_SHA256 = "6f86fa7d91a0b469fbd9f657eedfe7de87a6934005054059fe246878ec52f104"
FOLLOWUP_CONFIG_SHA256 = "ccc85523929ce00c15d482df4d32c3d06e639b70ab6df7b74619203cf16a1a32"
_CONFIG_IDENTITIES = {
    APPROVED_CONFIG_SHA256: ("dwr-constraint-development-probe-v4-r2", 900),
    FOLLOWUP_CONFIG_SHA256: ("dwr-constraint-development-probe-cpu-followup-v1", 1800),
}


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate config key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise ValueError(f"nonfinite config constant: {token}")


def _profile_keys(values: list[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(item.split("/")) for item in values)


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    config_sha256: str
    config_version: str
    wall_seconds: int
    root_seed: int
    generator_id: str
    source_profiles: tuple[tuple[str, ...], ...]
    development_profiles: tuple[tuple[str, ...], ...]
    cell_order: tuple[str, ...]
    variables: tuple[str, ...]
    strata: Mapping[str, tuple[int, int]]
    source_quotas: Mapping[tuple[str, ...], Mapping[str, int]]
    development_cell_quota: int
    raw_config: Mapping[str, object]

    def profiles_for(self, partition: str) -> tuple[tuple[str, ...], ...]:
        if partition == "source":
            return self.source_profiles
        if partition == "development":
            return self.development_profiles
        raise ValueError("partition must be source or development")

    def quota_for(self, partition: str, profile: tuple[str, ...], cell: str) -> int:
        if partition == "source":
            return self.source_quotas[profile][cell]
        if partition == "development":
            return self.development_cell_quota
        raise ValueError("partition must be source or development")


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def load_probe_config(path: str | Path) -> ProbeConfig:
    """Load only the exact reviewed config bytes and reject non-JSON extensions."""
    if not isinstance(path, (str, Path)):
        raise TypeError("config path must be a string or pathlib.Path")
    try:
        raw_bytes = Path(path).read_bytes()
        raw_text = raw_bytes.decode("utf-8", errors="strict")
        value = json.loads(raw_text, object_pairs_hook=_pairs_no_duplicates,
                           parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"cannot read strict UTF-8 JSON config: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("probe config root must be an object")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    identity = _CONFIG_IDENTITIES.get(digest)
    if identity is None:
        raise ValueError("probe config bytes differ from an admitted v4 configuration")
    expected_version, wall_seconds = identity
    runtime = value.get("runtime")
    if (value.get("config_version") != expected_version or not isinstance(runtime, dict) or
            type(runtime.get("combined_active_wall_seconds_generation_replay_audit_finalization")) is not int or
            runtime["combined_active_wall_seconds_generation_replay_audit_finalization"] != wall_seconds):
        raise ValueError("probe config identity or registered wall policy mismatch")
    if type(value.get("root_seed")) is not int:
        raise TypeError("root_seed must be an integer, not bool")

    grammar = value["grammar"]
    source_profiles = _profile_keys(value["partitions"]["source_profiles"])
    development_profiles = _profile_keys(value["partitions"]["development_profiles"])
    cell_order = tuple(item["id"] for item in value["strata"])
    source_quotas = {
        profile: MappingProxyType(dict(value["quotas"]["source_accepted_by_profile_cell"]["/".join(profile)]))
        for profile in source_profiles
    }
    return ProbeConfig(
        config_sha256=digest,
        config_version=expected_version,
        wall_seconds=wall_seconds,
        root_seed=value["root_seed"],
        generator_id=value["generator_id"],
        source_profiles=source_profiles,
        development_profiles=development_profiles,
        cell_order=cell_order,
        variables=tuple(f"v{i}" for i in range(grammar["n"])),
        strata=MappingProxyType({
            item["id"]: (item["given_count"], item["constraint_count"])
            for item in value["strata"]
        }),
        source_quotas=MappingProxyType(source_quotas),
        development_cell_quota=value["quotas"]["development_validation_accepted_per_profile_cell"],
        raw_config=_freeze(value),
    )
