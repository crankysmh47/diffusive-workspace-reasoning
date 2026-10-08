"""Closed, shared validation for v2 generation manifests."""
from __future__ import annotations

import math
import re
from typing import Any

from .generator_v2 import PARTITIONS


def _same_typed(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_typed(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_typed(a, b) for a, b in zip(left, right))
    return left == right


def expected_limits(mode: str, limits: Any) -> dict:
    if mode not in ("feasibility", "full"):
        raise ValueError("unknown v2 manifest mode")
    targets = {"source": sum(limits.source_by_profile.values()),
               "development": sum(limits.development_by_profile.values()),
               "final": sum(limits.final_by_profile.values()) if mode == "full" else 0}
    return {"attempts_per_partition": limits.attempts_per_partition,
            "generation_plus_audit_seconds": limits.wall_seconds,
            "resident_memory_mib": limits.memory_mib, "cpu_workers": 1,
            "quota_targets": targets}


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"manifest {name} must be finite and nonnegative")
    return float(value)


def validate_manifest_v2(manifest: dict, *, config_sha256: str, config: dict,
                          mode: str, limits: Any, expected_producer: dict | None = None,
                          require_complete: bool = False) -> dict:
    """Validate schema, config/source producer binding, limits, terminal states and one run prefix."""
    required = {"manifest_version", "mode", "status", "created_utc", "command", "cwd", "python",
                "platform", "config_path", "config_sha256", "resolved_config", "seed", "seed_derivation",
                "output_dir", "source", "limits", "producer", "generation_elapsed_seconds", "partitions",
                "first_work_started_utc", "generation_cpu_seconds", "finished_utc", "terminal_status_utc",
                "finalization_started_utc", "finalization_elapsed_seconds",
                "artifacts", "error", "exit_code", "peak_resident_bytes", "accounting_end_boundary",
                "accounting_note", "finalization_errors"}
    optional = {"consumed_preflight"}
    if type(manifest) is not dict or not required <= set(manifest) or set(manifest) - required - optional:
        raise ValueError("v2 generation manifest has unknown or missing fields")
    if manifest["manifest_version"] != "dwr-task-run-v2" or manifest["mode"] != mode:
        raise ValueError("v2 generation manifest version or mode mismatch")
    if manifest["config_sha256"] != config_sha256 or not _same_typed(manifest["resolved_config"], config):
        raise ValueError("v2 generation manifest config identity mismatch")
    if (type(manifest["seed"]) is not int or manifest["seed"] != config["seed"] or
        manifest["seed_derivation"] != "sha256-prefix64-v2"):
        raise ValueError("v2 generation seed identity mismatch")
    for name in ("created_utc", "cwd", "python", "platform", "config_path", "output_dir",
                 "first_work_started_utc", "finished_utc", "terminal_status_utc", "finalization_started_utc"):
        if type(manifest[name]) is not str:
            raise ValueError(f"manifest {name} must be a string")
    if type(manifest["command"]) is not list or any(type(x) is not str for x in manifest["command"]):
        raise ValueError("manifest command must be a string list")
    source = manifest["source"]
    if (type(source) is not dict or set(source) != {"commit", "dirty", "dirty_diff_sha256", "checkout_root"} or
        type(source["commit"]) is not str or not source["commit"] or type(source["dirty"]) is not bool or
        type(source["dirty_diff_sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", source["dirty_diff_sha256"]) or
        type(source["checkout_root"]) is not str):
        raise ValueError("manifest source identity is malformed")
    if not _same_typed(manifest["limits"], expected_limits(mode, limits)):
        raise ValueError("manifest registered limits or quotas mismatch")
    producer = manifest["producer"]
    producer_fields = {"generator_v2_sha256", "dataset_v2_sha256", "replay_v2_sha256", "manifest_v2_sha256"}
    if type(producer) is not dict or set(producer) != producer_fields or any(
            type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value) for value in producer.values()):
        raise ValueError("manifest producer identity is malformed")
    if expected_producer is not None and producer != expected_producer:
        raise ValueError("manifest producer source hashes mismatch")
    elapsed = _finite_nonnegative(manifest["generation_elapsed_seconds"], "generation_elapsed_seconds")
    _finite_nonnegative(manifest["generation_cpu_seconds"], "generation_cpu_seconds")
    _finite_nonnegative(manifest["finalization_elapsed_seconds"], "finalization_elapsed_seconds")
    peak = manifest["peak_resident_bytes"]
    if peak is not None and (type(peak) is not int or peak < 0):
        raise ValueError("manifest peak resident bytes must be a nonnegative integer or null")
    if (manifest["accounting_end_boundary"] != "post_first_terminal_manifest_write_resource_check" or
        type(manifest["accounting_note"]) is not str):
        raise ValueError("manifest accounting boundary is invalid")
    if type(manifest["finalization_errors"]) is not list:
        raise ValueError("manifest finalization_errors must be a list")
    for error in manifest["finalization_errors"]:
        if (type(error) is not dict or not {"type", "message", "phase", "partition", "candidate_index"} <= set(error) or
            set(error) - {"type", "message", "phase", "partition", "candidate_index", "operation"} or
            type(error["type"]) is not str or type(error["message"]) is not str or
            error["phase"] != "finalization" or
            ("operation" in error and type(error["operation"]) is not str) or
            (error["partition"] is not None and error["partition"] not in PARTITIONS) or
            (error["candidate_index"] is not None and
             (type(error["candidate_index"]) is not int or error["candidate_index"] < 0))):
            raise ValueError("manifest finalization error is malformed")
    status = manifest["status"]
    if status not in ("complete", "incomplete"):
        raise ValueError("manifest must be terminal, never running")
    if type(manifest["exit_code"]) is not int or manifest["exit_code"] != (0 if status == "complete" else 1):
        raise ValueError("manifest exit code and terminal status disagree")
    if status == "complete":
        if manifest["error"] is not None or manifest["finalization_errors"] or elapsed >= limits.wall_seconds:
            raise ValueError("complete manifest contains an error or exhausted budget")
    else:
        error = manifest["error"]
        if (type(error) is not dict or not {"type", "message", "phase", "partition", "candidate_index"} <= set(error) or
            set(error) - {"type", "message", "phase", "partition", "candidate_index", "operation"} or
            type(error["type"]) is not str or type(error["message"]) is not str or
            error["phase"] not in ("setup", "generation", "finalization") or
            ("operation" in error and type(error["operation"]) is not str) or
            (error["partition"] is not None and error["partition"] not in PARTITIONS) or
            (error["candidate_index"] is not None and
             (type(error["candidate_index"]) is not int or error["candidate_index"] < 0))):
            raise ValueError("incomplete manifest must preserve a typed primary error")
    if require_complete and status != "complete":
        raise ValueError("generation manifest is not complete")

    partitions = manifest["partitions"]
    if type(partitions) is not dict or set(partitions) != set(PARTITIONS):
        raise ValueError("manifest partition set is invalid")
    for part in PARTITIONS:
        state = partitions[part]
        if type(state) is not dict or set(state) != {"status", "attempts", "accepted", "not_attempted"}:
            raise ValueError(f"manifest {part} partition state is not closed")
        if state["status"] not in ("not_attempted", "incomplete", "complete"):
            raise ValueError(f"manifest {part} has invalid terminal partition status")
        if (type(state["attempts"]) is not int or not 0 <= state["attempts"] <= limits.attempts_per_partition or
            type(state["accepted"]) is not int or state["accepted"] < 0 or
            type(state["not_attempted"]) is not bool or
            state["not_attempted"] != (state["status"] == "not_attempted")):
            raise ValueError(f"manifest {part} partition counts or flags are invalid")
        if state["status"] == "not_attempted" and (state["attempts"] != 0 or state["accepted"] != 0):
            raise ValueError(f"manifest {part} not-attempted partition has work counts")
        if state["accepted"] > state["attempts"]:
            raise ValueError(f"manifest {part} accepted count exceeds attempted candidates")
    if mode == "feasibility":
        final = partitions["final"]
        if final != {"status": "not_attempted", "attempts": 0, "accepted": 0, "not_attempted": True}:
            raise ValueError("feasibility manifests must explicitly leave final not attempted")
    active = PARTITIONS if mode == "full" else PARTITIONS[:2]
    unfinished_seen = False
    for part in active:
        part_status = partitions[part]["status"]
        if part_status == "complete":
            if unfinished_seen:
                raise ValueError("partition history is not one global ordered prefix")
        else:
            if unfinished_seen and part_status != "not_attempted":
                raise ValueError("only the first unfinished partition may have attempted work")
            unfinished_seen = True
    if status == "complete":
        if any(partitions[part]["status"] != "complete" for part in active):
            raise ValueError("complete manifest has unfinished registered partitions")
    if "consumed_preflight" in manifest and mode != "full":
        raise ValueError("feasibility manifest cannot consume a full-mode preflight")
    if mode == "full":
        consumed = manifest.get("consumed_preflight")
        consumed_fields = {"linkage_version", "wrapper_path", "wrapper_sha256", "config_sha256", "source",
                           "generation_manifest_sha256", "generation_artifacts", "audit_artifact_path",
                           "audit_artifact_sha256"}
        if type(consumed) is not dict or set(consumed) != consumed_fields:
            raise ValueError("full manifest must retain a closed consumed PASS linkage")
        for key in ("wrapper_sha256", "config_sha256", "generation_manifest_sha256", "audit_artifact_sha256"):
            if type(consumed[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", consumed[key]):
                raise ValueError("consumed preflight hash is malformed")
        if (consumed["linkage_version"] != "dwr-task-v2-consumed-preflight-v1" or
            consumed["config_sha256"] != config_sha256 or consumed["source"] != source or
            consumed["source"].get("dirty") is not False or type(consumed["wrapper_path"]) is not str or
            type(consumed["audit_artifact_path"]) is not str or
            not consumed["wrapper_path"] or not consumed["audit_artifact_path"] or
            type(consumed["generation_artifacts"]) is not dict or
            set(consumed["generation_artifacts"]) != {"outcomes.jsonl", "records.jsonl"}):
            raise ValueError("consumed preflight linkage identity is invalid")
        for identity in consumed["generation_artifacts"].values():
            if (type(identity) is not dict or set(identity) != {"sha256", "bytes"} or
                type(identity["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) or
                type(identity["bytes"]) is not int or identity["bytes"] < 0):
                raise ValueError("consumed preflight artifact identity is malformed")
    artifacts = manifest["artifacts"]
    if status == "complete" and (type(artifacts) is not dict or set(artifacts) != {"outcomes.jsonl", "records.jsonl"}):
        raise ValueError("complete manifest artifact set is invalid")
    if type(artifacts) is not dict or not set(artifacts) <= {"outcomes.jsonl", "records.jsonl"}:
        raise ValueError("manifest artifact set is invalid")
    for identity in artifacts.values():
        if (type(identity) is not dict or set(identity) != {"sha256", "bytes"} or
            type(identity["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) or
            type(identity["bytes"]) is not int or identity["bytes"] < 0):
            raise ValueError("manifest artifact identity is malformed")
    return manifest


def validate_manifest_v3(manifest: dict, *, config_sha256: str, config: dict,
                         mode: str, limits: Any, expected_producer: dict | None = None,
                         require_complete: bool = False) -> dict:
    """Validate the exact v3 terminal extension, then reuse v2 run accounting checks."""
    from copy import deepcopy
    from .stream_contract import resolve_stream, require_matching_config
    contract = resolve_stream("dwr-generator-v3")
    require_matching_config(contract, config)
    extra = {"early_stop", "source_prefix", "source_capacity_bound"}
    if type(manifest) is not dict or not extra <= set(manifest):
        raise ValueError("v3 generation manifest has unknown or missing source-gate fields")
    if (manifest.get("manifest_version") != contract.manifest_version or
        manifest.get("seed_derivation") != contract.seed_derivation or
        manifest.get("config_sha256") != config_sha256):
        raise ValueError("v3 generation manifest identity mismatch")
    producer = manifest.get("producer")
    if (type(producer) is not dict or set(producer) != set(contract.generation_producer_fields) or
        any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in producer.values())):
        raise ValueError("v3 generation producer identity is malformed")
    if expected_producer is not None and producer != expected_producer:
        raise ValueError("v3 generation producer source hashes mismatch")
    prefix = manifest["source_prefix"]
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict:
        raise ValueError("v3 generation artifact map is malformed")
    if prefix is not None:
        prefix_fields = {"context_file", "context_sha256", "replay_file", "replay_sha256"}
        if (type(prefix) is not dict or set(prefix) != prefix_fields or
            prefix["context_file"] != "source_prefix_context.json" or
            prefix["replay_file"] != "source_prefix_replay.json" or
            any(type(prefix[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", prefix[key])
                for key in ("context_sha256", "replay_sha256"))):
            raise ValueError("v3 source prefix linkage is malformed")
        for name, digest in ((prefix["context_file"], prefix["context_sha256"]),
                             (prefix["replay_file"], prefix["replay_sha256"])):
            identity = artifacts.get(name) if type(artifacts) is dict else None
            if type(identity) is not dict or identity.get("sha256") != digest:
                raise ValueError("v3 source prefix linkage differs from terminal artifact identities")
    bound = manifest["source_capacity_bound"]
    if mode == "full":
        if bound != {"status": "NOT_APPLICABLE", "reason": "full_mode_uses_registered_five_cohort_gate"} or prefix is not None or manifest["early_stop"] is not None:
            raise ValueError("full v3 run must leave source capacity bound not applicable")
        allowed_artifacts = {"outcomes.jsonl", "records.jsonl"}
    else:
        bound_fields = {"bound_version", "status", "reason", "source_prefix_replay_sha256",
                        "top_k_cells", "minimum_source_train_mass", "source_train_group_count",
                        "top_cells", "top_k_mass"}
        if type(bound) is not dict or set(bound) != bound_fields or bound.get("bound_version") != "dwr-task-v3-source-capacity-bound-v1":
            raise ValueError("v3 source capacity report has unknown or missing fields")
        allowed = {"NOT_ASSESSED", "PASS_CAPACITY_BOUND", "NO_GO"}
        if bound.get("status") not in allowed:
            raise ValueError("v3 source capacity report status is invalid")
        if (bound["top_k_cells"] != 10 or type(bound["top_k_cells"]) is not int or
            type(bound["minimum_source_train_mass"]) not in (int, float) or
            type(bound["minimum_source_train_mass"]) is bool or bound["minimum_source_train_mass"] != 0.8 or
            type(bound["source_train_group_count"]) is not int or bound["source_train_group_count"] < 0 or
            type(bound["top_cells"]) is not list):
            raise ValueError("v3 source capacity report parameters or counts are invalid")
        previous = None
        top_total = 0
        for item in bound["top_cells"]:
            if (type(item) is not dict or set(item) != {"cell", "count"} or
                type(item["cell"]) is not list or len(item["cell"]) != 5 or
                any(type(value) is not int for value in item["cell"]) or
                type(item["count"]) is not int or item["count"] <= 0):
                raise ValueError("v3 source capacity top cell is malformed")
            key = (-item["count"], tuple(item["cell"]))
            if previous is not None and key < previous:
                raise ValueError("v3 source capacity cells are not sorted")
            previous = key
            top_total += item["count"]
        if len(bound["top_cells"]) > 10 or top_total > bound["source_train_group_count"]:
            raise ValueError("v3 source capacity top-cell counts exceed their denominator")
        mass = bound["top_k_mass"]
        if mass is not None and (type(mass) not in (int, float) or not math.isfinite(mass) or not 0 <= mass <= 1):
            raise ValueError("v3 source capacity mass is invalid")
        if bound["status"] in ("PASS_CAPACITY_BOUND", "NO_GO"):
            if prefix is None or bound["source_prefix_replay_sha256"] != prefix["replay_sha256"]:
                raise ValueError("v3 source capacity report is not bound to a replayed prefix")
            if mass is None or bound["source_train_group_count"] == 0:
                raise ValueError("assessed v3 source capacity bound needs source-train mass")
            if mass != top_total / bound["source_train_group_count"]:
                raise ValueError("v3 source capacity mass differs from top-cell counts")
            if bound["status"] == "PASS_CAPACITY_BOUND" and (mass < 0.8 or bound["reason"] is not None):
                raise ValueError("v3 capacity pass does not meet its registered threshold")
            if bound["status"] == "NO_GO" and (mass >= 0.8 or bound["reason"] != "source_support_capacity_bound"):
                raise ValueError("v3 capacity no-go does not match its registered threshold")
        elif bound["source_prefix_replay_sha256"] is not None:
            raise ValueError("unassessed source bound cannot claim a replay identity")
        elif bound["top_k_mass"] is not None or bound["source_train_group_count"] != 0 or bound["top_cells"]:
            raise ValueError("unassessed source bound cannot report measured source support")
        elif bound["reason"] not in ("source_partition_incomplete", "operational_error"):
            raise ValueError("unassessed source bound needs an incomplete or operational reason")
        early = manifest["early_stop"]
        if bound["status"] == "NO_GO":
            error = manifest.get("error")
            partitions = manifest.get("partitions")
            if (type(early) is not dict or set(early) != {"status", "reason", "source_capacity_bound_file", "source_capacity_bound_sha256"} or
                early.get("status") != "NO_GO" or early.get("reason") != "source_support_capacity_bound" or
                early.get("source_capacity_bound_file") != "source_capacity_bound.json" or
                type(early.get("source_capacity_bound_sha256")) is not str or
                not re.fullmatch(r"[0-9a-f]{64}", early["source_capacity_bound_sha256"]) or
                manifest.get("status") != "incomplete" or type(error) is not dict or
                error.get("type") != "ProposalNoGo" or error.get("phase") != "proposal_gate" or
                error.get("reason") != "source_support_capacity_bound" or
                error.get("message") != "source support capacity bound is below the registered feasibility requirement" or
                manifest.get("finalization_errors") != [] or prefix is None or
                type(partitions) is not dict or
                type(partitions.get("source")) is not dict or partitions["source"].get("status") != "complete" or
                type(partitions.get("development")) is not dict or partitions["development"].get("status") != "not_attempted" or
                type(partitions.get("final")) is not dict or partitions["final"].get("status") != "not_attempted"):
                raise ValueError("v3 capacity no-go terminal linkage is invalid")
            bound_identity = artifacts.get("source_capacity_bound.json")
            if type(bound_identity) is not dict or bound_identity.get("sha256") != early["source_capacity_bound_sha256"]:
                raise ValueError("v3 capacity no-go hash does not bind the report artifact")
        elif early is not None:
            raise ValueError("v3 early_stop is only valid for the registered capacity no-go")
        if manifest.get("status") == "complete" and (prefix is None or bound["status"] != "PASS_CAPACITY_BOUND"):
            raise ValueError("complete feasibility v3 generation needs a replayed capacity pass")
        allowed_artifacts = {"outcomes.jsonl", "records.jsonl", "source_capacity_bound.json",
                             "source_prefix_context.json", "source_prefix_replay.json"}
        if manifest.get("status") == "complete" and not {
                "outcomes.jsonl", "records.jsonl", "source_capacity_bound.json",
                "source_prefix_context.json", "source_prefix_replay.json"} <= set(artifacts):
            raise ValueError("complete feasibility v3 generation is missing closed prefix artifacts")
        if bound.get("status") in ("PASS_CAPACITY_BOUND", "NO_GO") and "source_capacity_bound.json" not in artifacts:
            raise ValueError("assessed source capacity report is missing from artifacts")
    if type(artifacts) is not dict or not set(artifacts) <= allowed_artifacts:
        raise ValueError("v3 generation manifest artifact names are not closed")
    if require_complete and (manifest.get("status") != "complete" or manifest.get("early_stop") is not None):
        raise ValueError("v3 generation manifest is not complete")

    # The shared validator still owns all timestamps, limits, partition prefix, and accounting rules.
    legacy = deepcopy(manifest)
    legacy.pop("early_stop"); legacy.pop("source_prefix"); legacy.pop("source_capacity_bound")
    legacy["manifest_version"] = "dwr-task-run-v2"
    legacy["seed_derivation"] = "sha256-prefix64-v2"
    legacy["producer"] = {name: producer[name] for name in
                          ("generator_v2_sha256", "dataset_v2_sha256", "replay_v2_sha256", "manifest_v2_sha256")}
    legacy_expected = None if expected_producer is None else legacy["producer"]
    legacy_artifacts = legacy.get("artifacts", {})
    if type(legacy_artifacts) is dict:
        legacy["artifacts"] = {name: value for name, value in legacy_artifacts.items()
                               if name in {"outcomes.jsonl", "records.jsonl"}}
    if legacy.get("error") is not None:
        legacy["error"] = {key: value for key, value in legacy["error"].items()
                           if key != "reason"}
        if legacy["error"].get("phase") == "proposal_gate":
            legacy["error"]["phase"] = "generation"
    validate_manifest_v2(legacy, config_sha256=config_sha256, config=config, mode=mode,
                         limits=limits, expected_producer=legacy_expected,
                         require_complete=require_complete)
    return manifest


def validate_manifest(manifest: dict, *, config_sha256: str, config: dict,
                      mode: str, limits: Any, expected_producer: dict | None = None,
                      require_complete: bool = False) -> dict:
    """Closed dispatcher for the two terminal generation-manifest versions."""
    if type(manifest) is not dict:
        raise ValueError("generation manifest must be an object")
    version = manifest.get("manifest_version")
    if version == "dwr-task-run-v2":
        return validate_manifest_v2(manifest, config_sha256=config_sha256, config=config,
                                    mode=mode, limits=limits, expected_producer=expected_producer,
                                    require_complete=require_complete)
    if version == "dwr-task-run-v3":
        return validate_manifest_v3(manifest, config_sha256=config_sha256, config=config,
                                    mode=mode, limits=limits, expected_producer=expected_producer,
                                    require_complete=require_complete)
    raise ValueError("generation manifest version is not registered")
