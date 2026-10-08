"""Closed contracts for the two registered task-generation streams."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .generator_v2 import generate_candidate
from .generator_v3 import generate_candidate_v3


@dataclass(frozen=True)
class StreamContract:
    generator_id: str
    config_version: str
    seed_derivation: str
    outcome_version: str
    accepted_record_version: str | None
    manifest_version: str
    audit_version: str
    audit_run_manifest_version: str
    audit_gate_version: str | None
    preflight_version: str
    consumed_preflight_version: str
    candidate_fn: Callable
    generation_producer_fields: tuple[str, ...]
    audit_producer_fields: tuple[str, ...]


_V2_GENERATION_FIELDS = (
    "generator_v2_sha256", "dataset_v2_sha256", "replay_v2_sha256", "manifest_v2_sha256")
_V2_AUDIT_FIELDS = (
    "audit_v2_sha256", "audit_cli_v2_sha256")
_V3_GENERATION_FIELDS = (
    "generator_v2_sha256", "generator_v3_sha256", "config_v3_sha256", "dataset_v2_sha256",
    "replay_v2_sha256", "manifest_v2_sha256", "stream_contract_sha256", "cli_v2_sha256",
    "cli_v3_sha256", "canonical_sha256", "oracle_sha256", "schema_sha256", "solver_sha256",
    "run_support_sha256")
_V3_AUDIT_FIELDS = (
    "audit_py_sha256", "audit_v2_sha256", "audit_cli_v2_sha256", "audit_cli_v3_sha256",
    "generator_v2_sha256", "generator_v3_sha256", "dataset_v2_sha256", "replay_v2_sha256",
    "manifest_v2_sha256", "config_v3_sha256", "stream_contract_sha256", "canonical_sha256",
    "oracle_sha256", "schema_sha256", "solver_sha256", "run_support_sha256")


_STREAMS = {
    "dwr-generator-v2": StreamContract(
        "dwr-generator-v2", "constraint-generation-v2", "sha256-prefix64-v2",
        "dwr-task-outcome-v2", None, "dwr-task-run-v2", "dwr-task-v2-audit-v1",
        "dwr-task-v2-audit-run-v1", None, "dwr-task-v2-feasibility-preflight-v1",
        "dwr-task-v2-consumed-preflight-v1", generate_candidate,
        _V2_GENERATION_FIELDS, _V2_AUDIT_FIELDS),
    "dwr-generator-v3": StreamContract(
        "dwr-generator-v3", "constraint-generation-v3", "sha256-prefix64-v3",
        "dwr-task-outcome-v3", "dwr-task-accepted-record-v3", "dwr-task-run-v3",
        "dwr-task-v3-audit-v1", "dwr-task-v3-audit-run-v1",
        "dwr-task-v3-audit-gate-v1", "dwr-task-v3-feasibility-preflight-v1",
        "dwr-task-v3-consumed-preflight-v1", generate_candidate_v3,
        _V3_GENERATION_FIELDS, _V3_AUDIT_FIELDS),
}


def resolve_stream(generator_id: str) -> StreamContract:
    if type(generator_id) is not str or generator_id not in _STREAMS:
        raise ValueError("generator_id must select one of the two registered task streams")
    return _STREAMS[generator_id]


def stream_for_config(config: dict) -> StreamContract:
    if type(config) is not dict:
        raise ValueError("config must be an object")
    version = config.get("config_version")
    for contract in _STREAMS.values():
        if version == contract.config_version:
            return contract
    raise ValueError("configuration version is not registered for a task stream")


def require_matching_config(contract: StreamContract, config: dict) -> None:
    if stream_for_config(config) != contract:
        raise ValueError("config identity does not match selected task stream")
    generation = config.get("generation")
    rng = config.get("rng")
    expected_seed_material = (f"{contract.generator_id}-domain+0x00+root-seed-u64be+0x00+"
                              "partition-ascii+0x00+candidate-index-u64be")
    if (config.get("seed_derivation") != contract.seed_derivation or
        type(generation) is not dict or
        generation.get("seed_derivation") != contract.seed_derivation or
        generation.get("candidate_rng_domain") != contract.generator_id or
        type(rng) is not dict or rng.get("version") != 2 or
        rng.get("seed_material") != expected_seed_material):
        raise ValueError("config RNG identity does not match selected task stream")
