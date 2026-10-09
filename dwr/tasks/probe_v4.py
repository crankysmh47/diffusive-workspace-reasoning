"""Bounded single-process v4 development proposal, replay and artifact writer."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit_probe_v4 import (AUDIT_VERSION, audit_records, read_jsonl,
                             validate_accepted_record)
from .canonical import content_id, development_split, lineage_id, source_split, structure_signature
from .config_probe_v4 import ProbeConfig, load_probe_config
from .generator_probe_v4 import (ProposalFatalError, ProposalResult,
                                 next_active_cell, propose_candidate)
from .run_support import ResourceGuard, atomic_json, sha256_file, stable_json, strict_json_loads
from .schema import serialize_model_input, serialize_task

MANIFEST_VERSION = "dwr-v4-probe-manifest-v1"
OUTCOME_VERSION = "dwr-v4-probe-outcome-v1"
RECORD_VERSION = "dwr-v4-probe-accepted-record-v1"
ARTIFACTS = tuple(("probe_manifest.initial.json", "outcomes.jsonl", "accepted_records.jsonl",
                   "source_closure.json", "candidate_replay.json", "independent_audit.json",
                   "probe_report.json", "probe_manifest.json"))
_SOURCE_PROFILES = (("eq",), ("neq",), ("add_mod",))
_DEV_PROFILES = (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod"))
_CELLS = ("A", "B", "C", "D")
_OUTCOME_KEYS = ("outcome_version", "partition", "candidate_index", "profile", "stratum", "n", "given_count",
    "constraint_count", "state", "reason", "task_or_null", "witness_or_null", "oracle_solution_or_null",
    "oracle_count_capped_or_null", "solver_stats_or_null", "content_id_or_null", "lineage_id_or_null",
    "structure_signature_or_null", "split_or_null", "private_eligibility_or_null", "denominator_counted",
    "prompt_utf8_bytes_or_null", "prompt_byte_bin_or_null", "first_solution_node_bin_or_null",
    "quota_cell_count_after_or_null")


def _profile_key(profile: tuple[str, ...]) -> str:
    return "/".join(profile)


def _plain_profile(profile: tuple[str, ...]) -> list[str]:
    return list(profile)


def _assignment(solution) -> dict[str, int] | None:
    return None if solution is None else dict(solution.assignments)


def _solver_stats(stats) -> dict[str, int] | None:
    if stats is None:
        return None
    return {"first_solution_nodes": stats.first_solution_nodes,
            "uniqueness_nodes": stats.uniqueness_nodes,
            "uniqueness_solution_count_capped": stats.uniqueness_solution_count_capped}


def _private(value) -> dict | None:
    if value is None:
        return None
    return dict(value)


def _thaw(value):
    if isinstance(value, dict):
        return {key: _thaw(item) for key, item in value.items()}
    if hasattr(value, "items"):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _prompt_bin(size: int) -> int:
    return 0 if size <= 511 else 1 if size <= 639 else 2 if size <= 767 else 3 if size <= 1023 else 4


def _node_bin(nodes: int) -> int:
    return 0 if nodes == 1 else 1 if nodes <= 3 else 2 if nodes <= 7 else 3 if nodes <= 15 else 4 if nodes <= 31 else 5 if nodes <= 63 else 6


class ProbeState:
    """Run-local identity, denominator, and quota state; no quota override surface."""
    def __init__(self, config: ProbeConfig, *, capture_rows: bool = True, _fixture_policy=None):
        self.config = config
        self.capture_rows = capture_rows
        self._fixture_policy = _fixture_policy
        profiles_for = self.profiles_for
        self.seen_content: set[str] = set()
        self.seen_lineage: set[str] = set()
        self.quota_counts = {
            "source": {profile: {cell: 0 for cell in _CELLS}
                       for profile in profiles_for("source")},
            "development": {profile: {cell: 0 for cell in _CELLS}
                            for profile in profiles_for("development")}}
        self.denominators = {"source_train": _empty_counter(),
                             "development_validation": {_profile_key(p): _empty_counter() for p in config.development_profiles}}
        self.development_eligible_signatures = {_profile_key(p): set() for p in config.development_profiles}
        self.last_outcome: dict | None = None
        # Full task rows are kept only by tiny correctness fixtures; production streams to disk.
        self.unique_groups: list[dict] = []
        self.outcomes: list[dict] = []
        self.records: list[dict] = []

    def save_outcome(self, row: dict) -> None:
        self.last_outcome = row
        if self.capture_rows:
            self.outcomes.append(row)

    def remaining(self, partition: str) -> dict:
        return {profile: {cell: self.quota_for(partition, profile, cell) - self.quota_counts[partition][profile][cell]
                          for cell in _CELLS} for profile in self.profiles_for(partition)}

    def profiles_for(self, partition: str):
        if self._fixture_policy is None:
            return self.config.profiles_for(partition)
        return tuple(self._fixture_policy.quotas[partition])

    def quota_for(self, partition: str, profile: tuple[str, ...], cell: str) -> int:
        if self._fixture_policy is not None:
            return self._fixture_policy.quotas[partition][profile].get(cell, 0)
        return self.config.quota_for(partition, profile, cell)

    def next_active(self, partition: str, candidate_index: int):
        remaining = self.remaining(partition)
        open_cells = [(profile, cell) for profile in self.profiles_for(partition)
                      for cell in _CELLS if remaining[profile][cell] > 0]
        if not open_cells:
            raise ValueError("no active profile/cell quota remains")
        if self._fixture_policy is not None:
            return open_cells[candidate_index % len(open_cells)]
        return next_active_cell(self.config, partition, candidate_index, remaining)

    def add_quota(self, partition: str, profile: tuple[str, ...], cell: str) -> bool:
        current = self.quota_counts[partition][profile][cell]
        target = self.quota_for(partition, profile, cell)
        if current >= target:
            return False
        self.quota_counts[partition][profile][cell] = current + 1
        return current + 1 == target

    def partition_closed(self, partition: str) -> bool:
        return all(self.quota_counts[partition][p][c] == self.quota_for(partition, p, c)
                   for p in self.profiles_for(partition) for c in _CELLS)


class _FixturePolicy:
    """Explicit reduced policy for synthetic tests; production entry points cannot pass it."""
    def __init__(self, fixture_id: str, *, source_only: bool = False):
        if not isinstance(fixture_id, str) or not fixture_id or len(fixture_id) > 80:
            raise ValueError("fixture id must be a short nonempty string")
        self.fixture_id = fixture_id
        self.source_only = source_only
        self.max_attempts_per_partition = 48
        if source_only:
            self.quotas = {"source": {("eq",): {"C": 1}}, "development": {}}
        else:
            self.quotas = {"source": {p: {cell: 1 for cell in _CELLS}
                                        for p in (("eq",), ("neq",), ("add_mod",))},
                "development": {p: {cell: 1 for cell in _CELLS}
                    for p in (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod"))}}

    def source_floor_passes(self, projection: dict) -> bool:
        if self.source_only:
            return (projection["accepted_groups"] >= 1 and
                projection["structure_signatures"] >= 1 and
                projection["eligible_groups_by_cell"]["C"] >= 1 and
                projection["four_cell_mass"]["fraction"] >= 0.8)
        return (projection["accepted_groups"] >= 4 and projection["structure_signatures"] >= 1 and
                all(projection["profile_groups"][p] >= 1 for p in ("eq", "neq", "add_mod")) and
                all(projection["eligible_groups_by_cell"][cell] >= 1 for cell in _CELLS) and
                projection["four_cell_mass"]["fraction"] >= 0.8)

    def development_floor_passes(self, metric: dict, denominator: dict, accepted: dict) -> bool:
        return (metric is not None and metric["eligible_signatures"] >= 1 and
                sum(accepted.values()) == 4 and all(accepted[cell] == 1 for cell in _CELLS) and
                denominator["unique_oracle_groups"] > 0 and
                denominator["scientifically_eligible_groups"] / denominator["unique_oracle_groups"] >= 0.25 and
                denominator["eligible_envelope_groups"] / denominator["unique_oracle_groups"] >= 0.25)


def _empty_counter() -> dict[str, int]:
    return {"unique_oracle_groups": 0, "scientifically_eligible_groups": 0,
            "eligible_envelope_groups": 0}


def source_gate_transition(source_status: str) -> bool:
    """Only a fully passed source closure permits a development draw."""
    return source_status == "PASS_SOURCE_CLOSURE"


def project_public_metrics(state: ProbeState) -> dict:
    """Expose permitted denominators only; opaque split keys are never serialized."""
    return {"source_train": dict(state.denominators["source_train"]),
            "development_validation_by_profile": {
                profile: dict(counts) for profile, counts in state.denominators["development_validation"].items()}}


def _denominator_projection(state: ProbeState) -> dict:
    source = state.denominators["source_train"]
    dev = state.denominators["development_validation"]
    return {"source_train": {"unique_oracle_groups": source["unique_oracle_groups"],
                              "sensitive_groups": source["scientifically_eligible_groups"],
                              "eligible_envelope_groups": source["eligible_envelope_groups"]},
            "development_validation_by_profile": {
                profile: {"unique_oracle_groups": counts["unique_oracle_groups"],
                          "scientifically_eligible_groups": counts["scientifically_eligible_groups"],
                          "eligible_envelope_groups": counts["eligible_envelope_groups"]}
                for profile, counts in dev.items()}}


def _new_outcome(partition: str, index: int, profile: tuple[str, ...], cell: str) -> dict:
    given_count, constraint_count = {"A": (2, 8), "B": (3, 6), "C": (3, 7), "D": (3, 8)}[cell]
    return {"outcome_version": OUTCOME_VERSION, "partition": partition, "candidate_index": index,
        "profile": _plain_profile(profile), "stratum": cell, "n": 6, "given_count": given_count,
        "constraint_count": constraint_count, "state": "rejected_proposal", "reason": None,
        "task_or_null": None, "witness_or_null": None, "oracle_solution_or_null": None,
        "oracle_count_capped_or_null": None, "solver_stats_or_null": None, "content_id_or_null": None,
        "lineage_id_or_null": None, "structure_signature_or_null": None, "split_or_null": None,
        "private_eligibility_or_null": None, "denominator_counted": False, "prompt_utf8_bytes_or_null": None,
        "prompt_byte_bin_or_null": None, "first_solution_node_bin_or_null": None,
        "quota_cell_count_after_or_null": None}


def consume_proposal(config: ProbeConfig, state: ProbeState, partition: str, index: int,
                     profile: tuple[str, ...], cell: str, *, _proposal_provider=None) -> tuple[dict, dict | None]:
    """Apply the registered identity, denominator, eligibility, envelope and quota precedence."""
    try:
        producer = propose_candidate if _proposal_provider is None else _proposal_provider
        result = producer(config, partition, index, (profile, cell))
    except ProposalFatalError as exc:
        result = exc.context
        row = _new_outcome(partition, index, profile, cell)
        row.update({"state": "fatal_error", "reason": exc.reason,
            "task_or_null": strict_json_loads(serialize_task(result.task)),
            "witness_or_null": _assignment(result.witness),
            "oracle_count_capped_or_null": result.oracle_count_capped,
            "oracle_solution_or_null": None, "solver_stats_or_null": None,
            "private_eligibility_or_null": None})
        state.save_outcome(row)
        raise
    if not isinstance(result, ProposalResult):
        raise TypeError("proposal producer returned an invalid result")
    row = _new_outcome(partition, index, profile, cell)
    if result.proposal_reason is not None:
        row["reason"] = result.proposal_reason
        if result.witness is not None:
            row["witness_or_null"] = _assignment(result.witness)
        state.save_outcome(row)
        return row, None
    row["task_or_null"] = strict_json_loads(serialize_task(result.task))
    row["witness_or_null"] = _assignment(result.witness)
    row["oracle_count_capped_or_null"] = result.oracle_count_capped
    if result.oracle_count_capped == 2:
        row.update({"state": "oracle_non_unique", "reason": None})
        state.save_outcome(row)
        return row, None
    if result.oracle_count_capped != 1:
        raise ValueError("proposal result has an impossible oracle count")

    task = result.task
    cid = content_id(task)
    lid = lineage_id(task)
    sid = structure_signature(task)
    split = source_split(cid) if partition == "source" else development_split(sid)
    model_input = serialize_model_input(task)
    prompt_bytes = len(model_input.encode("utf-8"))
    prompt_bin = _prompt_bin(prompt_bytes)
    node_bin = _node_bin(result.solver_stats.first_solution_nodes)
    eligibility = _private(result.private_eligibility)
    row.update({"oracle_solution_or_null": _assignment(result.oracle_solution),
        "solver_stats_or_null": _solver_stats(result.solver_stats), "content_id_or_null": cid,
        "lineage_id_or_null": lid, "structure_signature_or_null": sid, "split_or_null": split,
        "private_eligibility_or_null": eligibility, "prompt_utf8_bytes_or_null": prompt_bytes,
        "prompt_byte_bin_or_null": prompt_bin, "first_solution_node_bin_or_null": node_bin})
    if cid in state.seen_content or lid in state.seen_lineage:
        row.update({"state": "duplicate_content", "reason": "duplicate_content",
            "denominator_counted": False})
        state.save_outcome(row)
        return row, None
    state.seen_content.add(cid)
    state.seen_lineage.add(lid)
    eligible = bool(eligibility["eligible"])
    denominator_key = "source_train" if partition == "source" and split == "source_train" else (
        "development_validation" if partition == "development" and split == "development_validation" else None)
    cell_here = (len(task.givens), len(task.constraints), prompt_bin, node_bin) in (
        (2, 8, 4, 2), (3, 6, 4, 2), (3, 7, 4, 2), (3, 8, 4, 2))
    in_envelope = cell_here
    if denominator_key is not None:
        denom = state.denominators[denominator_key]
        if denominator_key == "development_validation":
            denom = denom[_profile_key(profile)]
        denom["unique_oracle_groups"] += 1
        if eligible:
            denom["scientifically_eligible_groups"] += 1
        if eligible and in_envelope:
            denom["eligible_envelope_groups"] += 1
    row.update({"denominator_counted": denominator_key is not None,
        "prompt_utf8_bytes_or_null": prompt_bytes, "prompt_byte_bin_or_null": prompt_bin,
        "first_solution_node_bin_or_null": node_bin})
    if partition == "development" and split == "development_validation" and eligible:
        state.development_eligible_signatures[_profile_key(profile)].add(sid)
    if state.capture_rows:
        state.unique_groups.append({"partition": partition, "split": split, "profile": profile,
            "task": task, "content_id": cid, "lineage_id": lid, "signature": sid,
            "solution": result.oracle_solution, "eligibility": eligibility, "prompt_bytes": prompt_bytes,
            "prompt_bin": prompt_bin, "node_bin": node_bin, "cell": cell if in_envelope else None})
    if not eligible:
        row.update({"state": "scientifically_ineligible", "reason": eligibility["reason"] if partition == "development" else "source_insensitive"})
    elif not in_envelope:
        row.update({"state": "outside_registered_support_envelope", "reason": "outside_registered_support_envelope"})
    elif partition == "development" and split == "development_diagnostic_test":
        row.update({"state": "retained_diagnostic", "reason": "development_diagnostic_split", "denominator_counted": False})
    elif partition == "source" or split == "development_validation":
        closed = state.add_quota(partition, profile, cell)
        row.update({"state": "accepted_for_quota", "reason": None,
                    "quota_cell_count_after_or_null": state.quota_counts[partition][profile][cell]})
        record = {"record_version": RECORD_VERSION, "generator_id": config.generator_id,
            "partition": partition, "candidate_index": index, "profile": _plain_profile(profile),
            "task": strict_json_loads(serialize_task(task)), "content_id": cid, "lineage_id": lid,
            "structure_signature": sid, "split": split, "witness": _assignment(result.witness),
            "solution": _assignment(result.oracle_solution), "solver_stats": _solver_stats(result.solver_stats),
            "model_input": model_input, "model_input_sha256": hashlib.sha256(model_input.encode("utf-8")).hexdigest()}
        if state.capture_rows:
            state.records.append(record)
        return _append(state, row, record)
    else:
        raise ValueError("candidate cannot enter a registered quota")
    state.save_outcome(row)
    return row, None


def _append(state: ProbeState, row: dict, record: dict) -> tuple[dict, dict]:
    state.save_outcome(row)
    return row, record


def _read_jsonl(path: Path, *, check=None) -> list[dict]:
    return read_jsonl(path, check=check)


def _producer_identity(config: ProbeConfig, root: Path) -> list[dict]:
    paths = config.raw_config["producer_hash_paths"]
    items = []
    for relative in paths:
        file = root / relative
        if not file.is_file():
            raise FileNotFoundError(f"registered producer file missing: {relative}")
        items.append({"path": relative, "sha256": sha256_file(file)})
    return items


def _source_identity(root: Path, fixture_id: str | None = None) -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True, encoding="utf-8").stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain=v1", "-z", "-uall"], cwd=root,
                            check=True, capture_output=True).stdout
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root, check=True,
                          capture_output=True).stdout
    if status and fixture_id is None:
        raise ValueError("source checkout is not clean")
    identity = {"commit": commit, "dirty": bool(status), "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
                "checkout_root": str(root)}
    if fixture_id is not None:
        identity["fixture_id"] = fixture_id
    return identity


def _initial_manifest(config: ProbeConfig, config_path: Path, output_dir: Path,
                      command: list[str], root: Path, producer: list[dict], source: dict,
                      fixture_policy=None) -> dict:
    state = ProbeState(config, _fixture_policy=fixture_policy)
    quotas = {part: {_profile_key(p): {c: state.quota_for(part, p, c) for c in _CELLS}
                     for p in state.profiles_for(part)} for part in ("source", "development")}
    partitions = {part: {"status": "not_started", "attempts": 0,
                         "quota_targets": quotas[part],
                         "quota_counts": {p: {c: 0 for c in _CELLS} for p in quotas[part]}}
                  for part in ("source", "development")}
    return {"manifest_version": MANIFEST_VERSION, "probe_id": "dwr-constraint-development-probe-v4",
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": None, "command": command, "cwd": str(Path.cwd()), "source": source,
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version(),
                   "executable": sys.executable},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "config_path": str(config_path.resolve()), "config_sha256": config.config_sha256,
        "resolved_config": _thaw(config.raw_config), "seed": config.root_seed,
        "rng_identity": {"generator_id": config.generator_id, "seed_derivation": "sha256-prefix64-v4",
            "seed_material": config.raw_config["rng"]["material"], "root_seed": config.root_seed},
        "producer": producer, "limits": {"max_attempts_per_partition": 30000 if fixture_policy is None else fixture_policy.max_attempts_per_partition, "workers": 1,
            "active_wall_seconds": 900, "max_peak_rss_bytes": 1073741824}, "output_dir": str(output_dir),
        "partitions": partitions,
        "artifacts": {name: {"path": name, "bytes": None, "sha256": None}
                      for name in ARTIFACTS if name != "probe_manifest.initial.json"},
        "error": None, "exit_code": None, "elapsed_wall_seconds": None, "cpu_seconds": None,
        "peak_resident_bytes": None, "accounting_end_boundary": None}


def _dump_jsonl(stream, row: dict) -> None:
    stream.write(stable_json(row) + "\n")
    stream.flush()


def _artifact_ref(path: Path) -> dict:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _verify_initial_binding(initial: dict, config: ProbeConfig, source: dict,
                            producer: list[dict], config_path: Path, fixture_policy=None) -> None:
    required = {"manifest_version", "probe_id", "status", "created_utc", "finished_utc", "command", "cwd",
        "source", "python", "platform", "config_path", "config_sha256", "resolved_config", "seed",
        "rng_identity", "producer", "limits", "output_dir", "partitions", "artifacts", "error",
        "exit_code", "elapsed_wall_seconds", "cpu_seconds", "peak_resident_bytes", "accounting_end_boundary"}
    if set(initial) != required or initial.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("initial manifest fields or version mismatch")
    source_fields = {"commit", "dirty", "dirty_diff_sha256", "checkout_root"}
    if fixture_policy is not None:
        source_fields.add("fixture_id")
    clean_binding = (source.get("dirty") is False and source.get("dirty_diff_sha256") == hashlib.sha256(b"").hexdigest())
    fixture_binding = (fixture_policy is not None and type(source.get("dirty")) is bool and
                       isinstance(source.get("dirty_diff_sha256"), str) and len(source["dirty_diff_sha256"]) == 64)
    if (not isinstance(source, dict) or set(source) != source_fields or
            not isinstance(source["commit"], str) or len(source["commit"]) != 40 or
            not (clean_binding or fixture_binding) or not isinstance(source["checkout_root"], str) or
            (fixture_policy is not None and source["fixture_id"] != fixture_policy.fixture_id)):
        raise ValueError("source identity shape or clean-checkout binding mismatch")
    producer_paths = list(config.raw_config["producer_hash_paths"])
    if (not isinstance(producer, list) or len(producer) != len(producer_paths) or
            [item.get("path") if isinstance(item, dict) else None for item in producer] != producer_paths or
            any(set(item) != {"path", "sha256"} or not isinstance(item["sha256"], str) or
                len(item["sha256"]) != 64 for item in producer)):
        raise ValueError("producer identity shape or registered order mismatch")
    if (initial["status"] != "running" or initial["finished_utc"] is not None or
            initial["source"] != source or (fixture_policy is None and source.get("dirty") is not False) or
            initial["producer"] != producer or initial["config_sha256"] != config.config_sha256 or
            sha256_file(config_path) != config.config_sha256 or initial["seed"] != config.root_seed or
            initial["resolved_config"] != _thaw(config.raw_config)):
        raise ValueError("initial manifest source, config, producer or status binding mismatch")
    expected_artifacts = set(ARTIFACTS) - {"probe_manifest.initial.json"}
    if set(initial["artifacts"]) != expected_artifacts:
        raise ValueError("initial manifest artifact set mismatch")
    for name, reference in initial["artifacts"].items():
        if set(reference) != {"path", "bytes", "sha256"} or reference["path"] != name or reference["bytes"] is not None or reference["sha256"] is not None:
            raise ValueError("initial manifest artifact declaration is malformed")
    expected_limits = {"max_attempts_per_partition": 30000 if fixture_policy is None else fixture_policy.max_attempts_per_partition, "workers": 1,
                       "active_wall_seconds": 900, "max_peak_rss_bytes": 1073741824}
    if initial["limits"] != expected_limits:
        raise ValueError("initial manifest resource limits mismatch")
    if (not isinstance(initial["python"], dict) or
            set(initial["python"]) != {"implementation", "version", "executable"} or
            any(not isinstance(value, str) or not value for value in initial["python"].values()) or
            not isinstance(initial["platform"], dict) or
            set(initial["platform"]) != {"system", "release", "machine"} or
            any(not isinstance(value, str) or not value for value in initial["platform"].values()) or
            not isinstance(initial["rng_identity"], dict) or
            set(initial["rng_identity"]) != {"generator_id", "seed_derivation", "seed_material", "root_seed"} or
            initial["rng_identity"]["generator_id"] != config.generator_id or
            initial["rng_identity"]["root_seed"] != config.root_seed or
            not isinstance(initial["command"], list) or
            any(not isinstance(value, str) for value in initial["command"])):
        raise ValueError("initial manifest runtime or RNG binding malformed")
    if set(initial["partitions"]) != {"source", "development"}:
        raise ValueError("initial manifest partition map mismatch")
    for part in ("source", "development"):
        item = initial["partitions"][part]
        targets = _profile_maps(config, part, ProbeState(config, _fixture_policy=fixture_policy))[0]
        if (set(item) != {"status", "attempts", "quota_targets", "quota_counts"} or
                item["quota_targets"] != targets or item["quota_counts"] !=
                {profile: {cell: 0 for cell in _CELLS} for profile in targets} or
                type(item["attempts"]) is not int or item["attempts"] != 0 or item["status"] != "not_started"):
            raise ValueError("initial manifest nested partition binding mismatch")


def _prefix_sha256(path: Path, byte_count: int) -> str:
    if type(byte_count) is not int or byte_count < 0 or path.stat().st_size < byte_count:
        raise ValueError("artifact prefix byte count is invalid")
    digest = hashlib.sha256()
    remaining = byte_count
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("artifact ended before its registered prefix")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _verify_source_closure(closure: dict, output: Path, config: ProbeConfig,
                           source: dict, producer: list[dict], fixture_policy=None) -> None:
    fields = {"closure_version", "status", "reason", "config_sha256", "source", "producer", "source_attempts",
        "outcome_prefix_bytes", "outcome_prefix_sha256", "record_prefix_bytes", "record_prefix_sha256",
        "source_quota_targets", "source_quota_counts", "source_quota_closed", "train_projection",
        "source_support", "resource_snapshot"}
    if set(closure) != fields or closure["closure_version"] != "dwr-v4-source-closure-v1":
        raise ValueError("source closure schema or version mismatch")
    if closure["config_sha256"] != config.config_sha256 or closure["source"] != source or closure["producer"] != producer:
        raise ValueError("source closure identity mismatch")
    expected_targets = _profile_maps(config, "source", ProbeState(config, _fixture_policy=fixture_policy))[0]
    counts = closure["source_quota_counts"]
    if (type(closure["source_attempts"]) is not int or closure["source_attempts"] < 0 or
            closure["source_quota_targets"] != expected_targets or not isinstance(counts, dict) or
            set(counts) != set(expected_targets) or any(
                not isinstance(counts[p], dict) or set(counts[p]) != set(_CELLS) or any(
                    type(counts[p][c]) is not int or not 0 <= counts[p][c] <= expected_targets[p][c]
                    for c in _CELLS) for p in expected_targets) or
            type(closure["source_quota_closed"]) is not bool):
        raise ValueError("source closure nested quota shape or target mismatch")
    if closure["status"] not in ("PASS_SOURCE_CLOSURE", "NO_GO_SOURCE_CLOSURE", "NOT_ASSESSED"):
        raise ValueError("source closure status is invalid")
    projection = closure["train_projection"]
    projection_fields = {"accepted_groups", "unique_oracle_groups", "structure_signatures", "profile_groups",
        "solution_value_counts", "given_counts", "sensitive_groups", "sensitivity_fraction",
        "eligible_envelope_groups", "envelope_yield", "groups_by_cell", "eligible_groups_by_cell",
        "four_cell_mass"}
    if projection is not None:
        if (not isinstance(projection, dict) or set(projection) != projection_fields or
                not isinstance(projection["profile_groups"], dict) or
                set(projection["profile_groups"]) != {"eq", "neq", "add_mod"} or
                not isinstance(projection["solution_value_counts"], dict) or
                set(projection["solution_value_counts"]) != {"0", "1", "2"} or
                not isinstance(projection["given_counts"], dict) or
                set(projection["given_counts"]) != {"2", "3"} or
                any(not isinstance(projection[key], dict) or set(projection[key]) != set(_CELLS)
                    for key in ("groups_by_cell", "eligible_groups_by_cell")) or
                not isinstance(projection["four_cell_mass"], dict) or
                set(projection["four_cell_mass"]) != {"numerator", "denominator", "fraction"}):
            raise ValueError("source closure train projection nested schema mismatch")
    support = closure["source_support"]
    if support is not None and (not isinstance(support, dict) or set(support) != {"cells", "cell_min_pass", "mass_pass"} or
            not isinstance(support["cells"], dict) or set(support["cells"]) != set(_CELLS) or
            type(support["cell_min_pass"]) is not bool or type(support["mass_pass"]) is not bool):
        raise ValueError("source closure support nested schema mismatch")
    resource = closure["resource_snapshot"]
    if (not isinstance(resource, dict) or set(resource) != {"wall_seconds", "cpu_seconds", "peak_rss_bytes"} or
            any(value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0)
                for value in resource.values())):
        raise ValueError("source closure resource snapshot malformed")
    if (type(closure["outcome_prefix_bytes"]) is not int or closure["outcome_prefix_bytes"] < 0 or
            type(closure["record_prefix_bytes"]) is not int or closure["record_prefix_bytes"] < 0 or
            not isinstance(closure["outcome_prefix_sha256"], str) or len(closure["outcome_prefix_sha256"]) != 64 or
            not isinstance(closure["record_prefix_sha256"], str) or len(closure["record_prefix_sha256"]) != 64):
        raise ValueError("source closure prefix identity malformed")
    if (_prefix_sha256(output / "outcomes.jsonl", closure["outcome_prefix_bytes"]) != closure["outcome_prefix_sha256"] or
            _prefix_sha256(output / "accepted_records.jsonl", closure["record_prefix_bytes"]) != closure["record_prefix_sha256"]):
        raise ValueError("source closure prefix hash mismatch")


def _verify_replay_document(replay: dict, output: Path, config: ProbeConfig,
                            source: dict, producer: list[dict], fixture_policy=None) -> None:
    fields = {"replay_version", "status", "config_sha256", "source", "producer", "partition_attempts",
        "outcome_prefix_bytes", "outcome_prefix_sha256", "record_prefix_bytes", "record_prefix_sha256",
        "quota_targets", "quota_counts", "quota_closed", "denominator_projection_sha256"}
    if set(replay) != fields or replay["replay_version"] != "dwr-v4-candidate-replay-v1":
        raise ValueError("candidate replay schema or version mismatch")
    if replay["status"] not in ("PASS_REPLAY", "FAIL_REPLAY", "NOT_ASSESSED"):
        raise ValueError("candidate replay status is invalid")
    if replay["config_sha256"] != config.config_sha256 or replay["source"] != source or replay["producer"] != producer:
        raise ValueError("candidate replay identity mismatch")
    expected_targets = {p: _profile_maps(config, p, ProbeState(config, _fixture_policy=fixture_policy))[0]
                        for p in ("source", "development")}
    expected_targets = {"source": expected_targets["source"], "development": expected_targets["development"]}
    if replay["quota_targets"] != expected_targets:
        raise ValueError("candidate replay quota targets differ from pinned config")
    expected_counts_shape = {part: {profile: {cell: 0 for cell in _CELLS}
                                    for profile in expected_targets[part]} for part in expected_targets}
    counts = replay["quota_counts"]
    if (not isinstance(counts, dict) or set(counts) != set(expected_counts_shape) or any(
            not isinstance(counts[part], dict) or set(counts[part]) != set(expected_counts_shape[part]) or any(
                not isinstance(counts[part][profile], dict) or set(counts[part][profile]) != set(_CELLS) or any(
                    type(counts[part][profile][cell]) is not int or
                    not 0 <= counts[part][profile][cell] <= expected_targets[part][profile][cell]
                    for cell in _CELLS) for profile in expected_counts_shape[part]) for part in expected_counts_shape)):
        raise ValueError("candidate replay quota counts nested shape mismatch")
    attempt_limit = 30000 if fixture_policy is None else fixture_policy.max_attempts_per_partition
    if (set(replay["partition_attempts"]) != {"source", "development"} or any(
            type(v) is not int or v < 0 or v > attempt_limit for v in replay["partition_attempts"].values()) or
            set(replay["quota_closed"]) != {"source", "development"} or
            type(replay["quota_closed"]["source"]) is not bool or
            not isinstance(replay["quota_closed"]["development"], dict) or
            set(replay["quota_closed"]["development"]) != set(expected_targets["development"]) or any(
                type(v) is not bool for v in replay["quota_closed"]["development"].values())):
        raise ValueError("candidate replay partition accounting malformed")
    if (_prefix_sha256(output / "outcomes.jsonl", replay["outcome_prefix_bytes"]) != replay["outcome_prefix_sha256"] or
            _prefix_sha256(output / "accepted_records.jsonl", replay["record_prefix_bytes"]) != replay["record_prefix_sha256"]):
        raise ValueError("candidate replay prefix hash mismatch")
    if (not isinstance(replay["denominator_projection_sha256"], str) or
            len(replay["denominator_projection_sha256"]) != 64):
        raise ValueError("candidate replay denominator digest is malformed")


def _verify_audit_document(audit: dict, output: Path, config: ProbeConfig,
                           source: dict, producer: list[dict]) -> None:
    fields = {"audit_version", "status", "config_sha256", "source", "producer", "replay_sha256",
        "source_closure_sha256", "public_projection", "integrity", "resource_snapshot"}
    if set(audit) != fields or audit["audit_version"] != AUDIT_VERSION:
        raise ValueError("independent audit schema or version mismatch")
    if audit["status"] not in ("PASS_AUDIT", "NO_GO_AUDIT", "NOT_ASSESSED"):
        raise ValueError("independent audit status is invalid")
    if audit["config_sha256"] != config.config_sha256 or audit["source"] != source or audit["producer"] != producer:
        raise ValueError("independent audit identity mismatch")
    if audit["replay_sha256"] != sha256_file(output / "candidate_replay.json") or audit["source_closure_sha256"] != sha256_file(output / "source_closure.json"):
        raise ValueError("independent audit artifact linkage mismatch")
    if set(audit["public_projection"]) != {"source_train", "development_validation_by_profile", "four_cohort_proxy"}:
        raise ValueError("independent audit public projection has wrong fields")
    integrity = audit["integrity"]
    if (not isinstance(integrity, dict) or set(integrity) != {"status", "failures"} or
            not isinstance(integrity["failures"], list) or
            (audit["status"] == "PASS_AUDIT" and
             (integrity["status"] != "pass" or integrity["failures"] != [])) or
            (audit["status"] != "PASS_AUDIT" and integrity["status"] not in ("not_assessed", "fail"))):
        raise ValueError("independent audit integrity/status mismatch")


def _ensure_failure_artifacts(output: Path, config: ProbeConfig, source: dict,
                              producer: list[dict], state: ProbeState, manifest: dict,
                              guard: ResourceGuard) -> None:
    """Write typed not-assessed documents for stages that could not complete."""
    outcomes_path, records_path = output / ARTIFACTS[1], output / ARTIFACTS[2]
    for path in (outcomes_path, records_path):
        if not path.exists():
            path.write_text("", encoding="utf-8", newline="\n")
    closure_path = output / ARTIFACTS[3]
    if not closure_path.exists():
        closure = _write_closure(output, config, manifest, state,
                                 "NOT_ASSESSED", "NOT_ASSESSED_INCOMPLETE_STAGE", guard)
        atomic_json(closure_path, closure)
    replay_path = output / ARTIFACTS[4]
    if not replay_path.exists():
        replay = {"replay_version": "dwr-v4-candidate-replay-v1", "status": "NOT_ASSESSED",
            "config_sha256": config.config_sha256, "source": source, "producer": producer,
            "partition_attempts": {part: manifest["partitions"][part]["attempts"]
                                   for part in ("source", "development")},
            "outcome_prefix_bytes": outcomes_path.stat().st_size,
            "outcome_prefix_sha256": sha256_file(outcomes_path),
            "record_prefix_bytes": records_path.stat().st_size,
            "record_prefix_sha256": sha256_file(records_path),
            "quota_targets": {part: _profile_maps(config, part, state)[0]
                              for part in ("source", "development")},
            "quota_counts": {part: _profile_maps(config, part, state)[1]
                             for part in ("source", "development")},
            "quota_closed": {"source": state.partition_closed("source"),
                "development": {_profile_key(p): state.partition_closed("development")
                                 for p in state.profiles_for("development")}},
            "denominator_projection_sha256": hashlib.sha256(b"").hexdigest()}
        atomic_json(replay_path, replay)
    audit_path = output / ARTIFACTS[5]
    if not audit_path.exists():
        audit = {"audit_version": AUDIT_VERSION, "status": "NOT_ASSESSED",
            "config_sha256": config.config_sha256, "source": source, "producer": producer,
            "replay_sha256": sha256_file(replay_path), "source_closure_sha256": sha256_file(closure_path),
            "public_projection": {"source_train": None,
                "development_validation_by_profile": None, "four_cohort_proxy": None},
            "integrity": {"status": "not_assessed", "failures": ["audit_not_completed"]},
            "resource_snapshot": {"wall_seconds": None, "cpu_seconds": None, "peak_rss_bytes": None}}
        atomic_json(audit_path, audit)


def _profile_maps(config: ProbeConfig, partition: str, state: ProbeState) -> tuple[dict, dict]:
    targets = {_profile_key(p): {c: state.quota_for(partition, p, c) for c in _CELLS}
               for p in state.profiles_for(partition)}
    counts = {_profile_key(p): dict(state.quota_counts[partition][p]) for p in state.profiles_for(partition)}
    return targets, counts


def _stage_partition(partition: str, config: ProbeConfig, state: ProbeState,
                     manifest: dict, guard: ResourceGuard, outcomes_stream,
                     records_stream, *, _proposal_provider=None) -> str:
    part_manifest = manifest["partitions"][partition]
    part_manifest["status"] = "running"
    limit = 30000 if state._fixture_policy is None else state._fixture_policy.max_attempts_per_partition
    while not state.partition_closed(partition) and part_manifest["attempts"] < limit:
        guard.check()
        index = part_manifest["attempts"]
        active_profile, cell = state.next_active(partition, index)
        try:
            row, record = consume_proposal(config, state, partition, index, active_profile, cell,
                                           _proposal_provider=_proposal_provider)
        except ProposalFatalError as exc:
            row = state.last_outcome
            _dump_jsonl(outcomes_stream, row)
            part_manifest["attempts"] += 1
            raise ProbeFailure("NOT_ASSESSED_SOLVER_OR_ORACLE_ERROR", "proposal", partition, index, exc.reason) from exc
        _dump_jsonl(outcomes_stream, row)
        if record is not None:
            _dump_jsonl(records_stream, record)
        part_manifest["attempts"] += 1
        part_manifest["quota_counts"] = {_profile_key(p): dict(state.quota_counts[partition][p])
                                          for p in state.profiles_for(partition)}
        guard.check()
    part_manifest["status"] = "complete" if state.partition_closed(partition) else "quota_shortfall"
    return part_manifest["status"]


class ProbeFailure(RuntimeError):
    def __init__(self, code: str, phase: str, partition: str | None = None,
                 candidate_index: int | None = None, reason: str | None = None):
        self.code, self.phase, self.partition = code, phase, partition
        self.candidate_index, self.reason = candidate_index, reason
        super().__init__(code)


def _source_floors(projection: dict, fixture_policy=None) -> tuple[bool, bool]:
    if fixture_policy is not None:
        passed = fixture_policy.source_floor_passes(projection)
        return passed, passed
    accepted_values = projection["solution_value_counts"]
    total_values = sum(accepted_values.values())
    floors = projection["accepted_groups"] >= 100 and projection["structure_signatures"] >= 12
    floors &= all(projection["profile_groups"][p] > 0 for p in ("eq", "neq", "add_mod"))
    floors &= all(accepted_values[str(v)] / total_values >= 0.1 if total_values else False for v in range(3))
    floors &= sum(projection["given_counts"][str(v)] > 0 for v in (2, 3)) >= 2
    floors &= projection["sensitivity_fraction"] >= 0.25 and projection["envelope_yield"] >= 0.25
    support = all(projection["eligible_groups_by_cell"][c] >= 10 for c in _CELLS)
    support &= projection["four_cell_mass"]["fraction"] >= 0.8
    return bool(floors), bool(support)


def _development_floors(state: ProbeState, projection: dict) -> dict[str, str]:
    results = {}
    for profile in state.profiles_for("development"):
        key = _profile_key(profile)
        metric = projection.get("development_validation_by_profile", {}).get(key)
        if not state.partition_closed("development") and not all(
                state.quota_counts["development"][profile][cell] == state.quota_for("development", profile, cell)
                for cell in _CELLS):
            results[key] = "not_assessed"
            continue
        denom = state.denominators["development_validation"][key]
        accepted = sum(state.quota_counts["development"][profile].values())
        if state._fixture_policy is not None:
            results[key] = "pass" if state._fixture_policy.development_floor_passes(
                metric, denom, state.quota_counts["development"][profile]) else "fail"
            continue
        passed = (metric is not None and accepted == 100 and
            metric["eligible_signatures"] >= 12 and
            denom["unique_oracle_groups"] > 0 and
            denom["scientifically_eligible_groups"] / denom["unique_oracle_groups"] >= 0.25 and
            denom["eligible_envelope_groups"] / denom["unique_oracle_groups"] >= 0.25 and
            all(state.quota_counts["development"][profile][cell] == 25 for cell in _CELLS) and
            len(state.development_eligible_signatures[key]) == metric["eligible_signatures"])
        results[key] = "pass" if passed else "fail"
    return results


def replay_prefix(config: ProbeConfig, outcomes: list[dict], expected_records: list[dict],
                  guard: ResourceGuard, *, capture_rows: bool = True, _fixture_policy=None,
                  _proposal_provider=None) -> tuple[ProbeState, str]:
    """Reconstruct candidate order and all stream-level decisions from empty state."""
    state = ProbeState(config, capture_rows=capture_rows, _fixture_policy=_fixture_policy)
    record_by_key = {(r["partition"], r["candidate_index"]): r for r in expected_records}
    if len(record_by_key) != len(expected_records):
        raise ProbeFailure("NOT_ASSESSED_REPLAY_MISMATCH", "replay", reason="duplicate_record_key")
    partitions = [r["partition"] for r in outcomes]
    if partitions != sorted(partitions, key={"source": 0, "development": 1}.get):
        raise ProbeFailure("NOT_ASSESSED_REPLAY_MISMATCH", "replay", reason="partition_order")
    next_index = {"source": 0, "development": 0}
    for expected in outcomes:
        guard.check()
        part = expected["partition"]
        index = expected["candidate_index"]
        if part not in next_index or index != next_index[part]:
            raise ProbeFailure("NOT_ASSESSED_REPLAY_MISMATCH", "replay", part, index, "candidate_index_order")
        next_index[part] += 1
        active, cell = state.next_active(part, index)
        try:
            row, record = consume_proposal(config, state, part, index, active, cell,
                                           _proposal_provider=_proposal_provider)
        except ProposalFatalError:
            # A deterministic fatal proposal is itself replayable evidence;
            # compare its retained outcome row, then let the terminal run stay
            # NOT_ASSESSED because the proposal stage failed closed.
            row, record = state.last_outcome, None
        if row != expected:
            raise ProbeFailure("NOT_ASSESSED_REPLAY_MISMATCH", "replay", part, index, "outcome_mismatch")
        saved = record_by_key.get((part, index))
        if record != saved:
            raise ProbeFailure("NOT_ASSESSED_REPLAY_MISMATCH", "replay", part, index, "accepted_record_mismatch")
    return state, hashlib.sha256(stable_json(_denominator_projection(state)).encode("utf-8")).hexdigest()


def _write_closure(output: Path, config: ProbeConfig, manifest: dict, state: ProbeState,
                   status: str, reason: str | None, guard: ResourceGuard,
                   train_projection: dict | None = None) -> dict:
    outcomes_path, records_path = output / "outcomes.jsonl", output / "accepted_records.jsonl"
    source_attempts = manifest["partitions"]["source"]["attempts"]
    outcome_size = outcomes_path.stat().st_size
    record_size = records_path.stat().st_size
    source_projection = train_projection
    support = None
    if source_projection is not None:
        support = {"cells": source_projection["eligible_groups_by_cell"],
                   "cell_min_pass": all(source_projection["eligible_groups_by_cell"][c] >= 10 for c in _CELLS),
                   "mass_pass": source_projection["four_cell_mass"]["fraction"] >= 0.8}
    return {"closure_version": "dwr-v4-source-closure-v1", "status": status, "reason": reason,
        "config_sha256": config.config_sha256, "source": manifest["source"], "producer": manifest["producer"],
        "source_attempts": source_attempts, "outcome_prefix_bytes": outcome_size,
        "outcome_prefix_sha256": sha256_file(outcomes_path),
        "record_prefix_bytes": record_size, "record_prefix_sha256": sha256_file(records_path),
        "source_quota_targets": manifest["partitions"]["source"]["quota_targets"],
        "source_quota_counts": manifest["partitions"]["source"]["quota_counts"],
        "source_quota_closed": state.partition_closed("source"), "train_projection": source_projection,
        "source_support": support, "resource_snapshot": _resource_snapshot(guard)}


def _resource_snapshot(guard: ResourceGuard) -> dict:
    cpu_started = getattr(guard, "dwr_cpu_started", time.process_time())
    return {"wall_seconds": round(guard.elapsed, 6), "cpu_seconds": round(max(0.0, time.process_time() - cpu_started), 6),
            "peak_rss_bytes": guard.peak_bytes}


def _terminal_report(config: ProbeConfig, source_closure: dict, replay: dict, audit: dict,
                     reason: str, status: str, guard: ResourceGuard, initial_sha: str,
                     closure_sha: str, source_identity: dict,
                     profile_floor_status: dict[str, str] | None = None) -> dict:
    projection = audit.get("public_projection", {})
    source_values = source_closure.get("train_projection")
    source_floor_status = ("pass" if source_closure["status"] == "PASS_SOURCE_CLOSURE" else
                           "fail" if source_values is not None else "not_assessed")
    profile_floor_status = profile_floor_status or {}
    dev_values = projection.get("development_validation_by_profile", {})
    dev_floors = {}
    source_not_attempted = source_closure["status"] != "PASS_SOURCE_CLOSURE"
    for profile in config.development_profiles:
        key = _profile_key(profile)
        metric = dev_values.get(key)
        if source_not_attempted:
            dev_floors[key] = {"status": "not_attempted", "values": None}
        elif metric is None or metric.get("status") == "not_attempted":
            dev_floors[key] = {"status": "not_attempted", "values": None}
        elif metric.get("status") != "complete":
            dev_floors[key] = {"status": "not_assessed", "values": None}
        else:
            dev_floors[key] = {"status": profile_floor_status.get(key, "fail"), "values": metric}
    proxy = projection.get("four_cohort_proxy")
    proxy_status = "not_assessed" if proxy is None else "pass" if proxy.get("pass") else "fail"
    quota_counts = replay.get("quota_counts", {})
    development_attempted = replay.get("partition_attempts", {}).get("development", 0) > 0
    source_by_cell = {cell: sum(quota_counts.get("source", {}).get(_profile_key(p), {}).get(cell, 0)
                                for p in config.source_profiles) for cell in _CELLS}
    source_closed = (None if source_closure["status"] == "NOT_ASSESSED"
                     else source_closure.get("source_quota_closed"))
    dev_closed = (replay.get("quota_closed", {}).get("development", {}) if development_attempted else
                  {_profile_key(profile): None for profile in config.development_profiles})
    return {"report_version": "dwr-v4-probe-report-v1", "status": status, "reason": reason,
        "config_sha256": config.config_sha256, "source": source_identity,
        "initial_manifest_sha256": initial_sha, "source_closure_sha256": closure_sha,
        "replay_sha256": replay.get("artifact_sha256"), "audit_sha256": audit.get("artifact_sha256"),
        "quotas": {"source_by_profile_cell": source_closure.get("source_quota_counts"),
            "source_by_cell": source_by_cell,
            "source_closed": source_closed,
            "development_validation_by_profile_cell": quota_counts.get("development"),
            "development_validation_closed": dev_closed},
        "public_floors": {"source_train": {"status": source_floor_status,
                                                    "values": source_values if source_floor_status in ("pass", "fail") else None},
            "development_validation_by_profile": dev_floors,
            "four_cohort_proxy": {"status": proxy_status,
                                  "values": proxy if proxy_status in ("pass", "fail") else None}},
        "four_cohort_proxy": proxy,
        "resource_snapshot": _resource_snapshot(guard), "g2_status": "not_assessed", "preflight_emitted": False}


def _verify_terminal_report(report: dict, config: ProbeConfig, source: dict,
                            initial_sha: str, closure_sha: str, closure: dict, replay: dict,
                            audit: dict) -> None:
    fields = {"report_version", "status", "reason", "config_sha256", "source",
        "initial_manifest_sha256", "source_closure_sha256", "replay_sha256", "audit_sha256",
        "quotas", "public_floors", "four_cohort_proxy", "resource_snapshot", "g2_status",
        "preflight_emitted"}
    statuses = {"pass", "fail", "not_assessed", "not_attempted"}
    if set(report) != fields or report["report_version"] != "dwr-v4-probe-report-v1":
        raise ValueError("terminal report schema or version mismatch")
    if report["status"] not in ("PASS_DEVELOPMENT_PROPOSAL", "NO_GO_PROPOSAL", "NOT_ASSESSED", "FIXTURE_PASS"):
        raise ValueError("terminal report status is invalid")
    if (report["config_sha256"] != config.config_sha256 or report["source"] != source or
            report["initial_manifest_sha256"] != initial_sha or
            report["source_closure_sha256"] != closure_sha or
            report["replay_sha256"] != replay.get("artifact_sha256") or
            report["audit_sha256"] != audit.get("artifact_sha256")):
        raise ValueError("terminal report identity or artifact linkage mismatch")
    quota = report["quotas"]
    development_attempted = replay.get("partition_attempts", {}).get("development", 0) > 0
    expected_dev_counts = replay["quota_counts"]["development"]
    expected_dev_closed = (replay["quota_closed"]["development"] if development_attempted else
        {_profile_key(profile): None for profile in config.development_profiles})
    expected_source_closed = (None if closure["status"] == "NOT_ASSESSED" else
                              closure["source_quota_closed"])
    if (not isinstance(quota, dict) or set(quota) != {"source_by_profile_cell", "source_by_cell",
            "source_closed", "development_validation_by_profile_cell", "development_validation_closed"} or
            quota["source_by_profile_cell"] != closure["source_quota_counts"] or
            quota["development_validation_by_profile_cell"] != expected_dev_counts or
            quota["source_closed"] != expected_source_closed or
            quota["development_validation_closed"] != expected_dev_closed):
        raise ValueError("terminal report nested quota accounting mismatch")
    floors = report["public_floors"]
    dev_keys = {_profile_key(profile) for profile in config.development_profiles}
    if (not isinstance(floors, dict) or set(floors) != {"source_train", "development_validation_by_profile",
            "four_cohort_proxy"} or set(floors["development_validation_by_profile"]) != dev_keys):
        raise ValueError("terminal report public floor shape mismatch")
    for value in (floors["source_train"], floors["four_cohort_proxy"],
                  *floors["development_validation_by_profile"].values()):
        if (not isinstance(value, dict) or set(value) != {"status", "values"} or
                value["status"] not in statuses):
            raise ValueError("terminal report public floor status malformed")
        if value["status"] in ("not_assessed", "not_attempted") and value["values"] is not None:
            raise ValueError("unassessed terminal report floor carries values")
    source_values = closure["train_projection"]
    source_status = ("pass" if closure["status"] == "PASS_SOURCE_CLOSURE" else
                     "fail" if source_values is not None else "not_assessed")
    if (floors["source_train"]["status"] != source_status or
            floors["source_train"]["values"] != (source_values if source_status in ("pass", "fail") else None)):
        raise ValueError("terminal report source floor differs from source closure")
    dev_projection = audit["public_projection"]["development_validation_by_profile"]
    for key, floor in floors["development_validation_by_profile"].items():
        metric = dev_projection[key]
        if metric["status"] == "complete":
            if floor["status"] not in ("pass", "fail") or floor["values"] != metric:
                raise ValueError("terminal report completed development floor mismatch")
        else:
            expected_status = "not_attempted" if metric["status"] == "not_attempted" else "not_assessed"
            if floor != {"status": expected_status, "values": None}:
                raise ValueError("terminal report incomplete development floor mismatch")
    proxy = audit["public_projection"]["four_cohort_proxy"]
    if (report["four_cohort_proxy"] != proxy or
            floors["four_cohort_proxy"]["values"] != proxy or
            floors["four_cohort_proxy"]["status"] != (
                "not_assessed" if proxy is None else "pass" if proxy["pass"] else "fail")):
        raise ValueError("terminal report proxy projection mismatch")
    resource = report["resource_snapshot"]
    if (not isinstance(resource, dict) or set(resource) != {"wall_seconds", "cpu_seconds", "peak_rss_bytes"} or
            any(value is not None and (type(value) not in (int, float) or
                                       not math.isfinite(value) or value < 0)
                for value in resource.values()) or report["g2_status"] != "not_assessed" or
            report["preflight_emitted"] is not False):
        raise ValueError("terminal report resource or gate status malformed")


def _verify_terminal_manifest(terminal: dict, initial: dict, config: ProbeConfig,
                              output: Path, replay: dict, report: dict) -> None:
    fields = {"manifest_version", "probe_id", "status", "created_utc", "finished_utc", "command", "cwd",
        "source", "python", "platform", "config_path", "config_sha256", "resolved_config", "seed",
        "rng_identity", "producer", "limits", "output_dir", "partitions", "artifacts", "error",
        "exit_code", "elapsed_wall_seconds", "cpu_seconds", "peak_resident_bytes", "accounting_end_boundary"}
    if set(terminal) != fields or terminal.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("terminal manifest fields or version mismatch")
    if (terminal["status"] not in ("complete", "scientific_no_go", "not_assessed") or
            terminal["finished_utc"] is None or terminal["source"] != initial["source"] or
            terminal["producer"] != initial["producer"] or terminal["config_sha256"] != config.config_sha256 or
            terminal["config_path"] != initial["config_path"] or terminal["output_dir"] != str(output.resolve()) or
            terminal["accounting_end_boundary"] != "post_first_terminal_write_resource_check"):
        raise ValueError("terminal manifest source/config/status binding mismatch")
    if (not isinstance(terminal["finished_utc"], str) or type(terminal["exit_code"]) is not int or
            type(terminal["elapsed_wall_seconds"]) not in (int, float) or
            not math.isfinite(terminal["elapsed_wall_seconds"]) or terminal["elapsed_wall_seconds"] < 0 or
            type(terminal["cpu_seconds"]) not in (int, float) or not math.isfinite(terminal["cpu_seconds"]) or
            terminal["cpu_seconds"] < 0 or
            (terminal["peak_resident_bytes"] is not None and
             (type(terminal["peak_resident_bytes"]) is not int or terminal["peak_resident_bytes"] < 0)) or
            terminal["error"] is not None and not isinstance(terminal["error"], dict)):
        raise ValueError("terminal manifest accounting or error payload malformed")
    for key in ("probe_id", "created_utc", "command", "cwd", "python", "platform", "resolved_config",
                "seed", "rng_identity", "limits"):
        if terminal[key] != initial[key]:
            raise ValueError("terminal manifest immutable binding changed")
    expected_names = set(ARTIFACTS[:-1])
    if set(terminal["artifacts"]) != expected_names:
        raise ValueError("terminal manifest artifact set mismatch")
    for name, reference in terminal["artifacts"].items():
        path = output / name
        if (set(reference) != {"path", "bytes", "sha256"} or reference["path"] != name or
                type(reference["bytes"]) is not int or reference["bytes"] != path.stat().st_size or
                reference["sha256"] != sha256_file(path)):
            raise ValueError("terminal manifest artifact hash or size mismatch")
    partitions = terminal["partitions"]
    if set(partitions) != {"source", "development"}:
        raise ValueError("terminal manifest partition set mismatch")
    for part in ("source", "development"):
        value = partitions[part]
        if (set(value) != {"status", "attempts", "quota_targets", "quota_counts"} or
                type(value["attempts"]) is not int or value["attempts"] != replay["partition_attempts"][part] or
                value["quota_targets"] != replay["quota_targets"][part] or
                value["quota_counts"] != replay["quota_counts"][part]):
            raise ValueError("terminal manifest partition accounting differs from replay")
    expected_status = {"PASS_DEVELOPMENT_PROPOSAL": ("complete", 0),
                       "NO_GO_PROPOSAL": ("scientific_no_go", 1),
                       "NOT_ASSESSED": ("not_assessed", 1)}.get(report["status"])
    if expected_status is None or (terminal["status"], terminal["exit_code"]) != expected_status:
        raise ValueError("terminal manifest status disagrees with final report")


def _not_assessed_report(config: ProbeConfig, source: dict, initial_path: Path,
                         closure_path: Path, guard: ResourceGuard, reason: str,
                         state: ProbeState, development_attempts: int = 0) -> dict:
    closure = strict_json_loads(closure_path.read_text(encoding="utf-8")) if closure_path.exists() else None
    source_values = closure.get("train_projection") if closure else None
    source_status = ("pass" if closure and closure.get("status") == "PASS_SOURCE_CLOSURE" else
                     "fail" if source_values is not None else "not_assessed")
    dev_status = "not_assessed" if development_attempts else "not_attempted"
    dev = {_profile_key(p): {"status": dev_status, "values": None}
           for p in config.development_profiles}
    _, source_counts = _profile_maps(config, "source", state)
    _, dev_counts = _profile_maps(config, "development", state)
    return {"report_version": "dwr-v4-probe-report-v1", "status": "NOT_ASSESSED", "reason": reason,
        "config_sha256": config.config_sha256, "source": source,
        "initial_manifest_sha256": sha256_file(initial_path),
        "source_closure_sha256": sha256_file(closure_path) if closure_path.exists() else None,
        "replay_sha256": None, "audit_sha256": None,
        "quotas": {"source_by_profile_cell": source_counts,
        "source_by_cell": {c: sum(source_counts[_profile_key(p)][c] for p in state.profiles_for("source")) for c in _CELLS},
            "source_closed": None if closure is None or closure.get("status") == "NOT_ASSESSED" else closure.get("source_quota_closed"),
            "development_validation_by_profile_cell": dev_counts,
            "development_validation_closed": ({_profile_key(p): all(
                state.quota_counts["development"].get(p, {}).get(cell, 0) == state.quota_for("development", p, cell)
                for cell in _CELLS) for p in config.development_profiles} if development_attempts else
                {_profile_key(p): None for p in config.development_profiles})},
        "public_floors": {"source_train": {"status": source_status,
                "values": source_values if source_status in ("pass", "fail") else None},
            "development_validation_by_profile": dev,
            "four_cohort_proxy": {"status": "not_assessed", "values": None}},
        "four_cohort_proxy": None, "resource_snapshot": _resource_snapshot(guard),
        "g2_status": "not_assessed", "preflight_emitted": False}


def _run_controller(config_path: Path, output_dir: Path, command: list[str] | None = None,
                    *, guard: ResourceGuard | None = None, _fixture_policy=None,
                    _proposal_provider=None) -> int:
    """One guarded invocation; real candidates are only run by the separately reviewed CLI stage."""
    config_path = Path(config_path).resolve()
    config = load_probe_config(config_path)
    output_dir = Path(output_dir).resolve()
    root = Path(__file__).resolve().parents[2]
    if _fixture_policy is None:
        try:
            output_dir.relative_to((root / "data").resolve())
        except ValueError as exc:
            raise ValueError("probe artifacts must be under the ignored data directory") from exc
    else:
        try:
            output_dir.relative_to(root)
        except ValueError:
            pass
        else:
            raise ValueError("fixture artifacts must be outside the repository")
    if output_dir.exists():
        raise FileExistsError("probe output directory already exists; resume is forbidden")
    guard = guard or ResourceGuard(900, 1024)
    guard.dwr_cpu_started = time.process_time()
    guard.check()
    producer = _producer_identity(config, root)
    source = _source_identity(root, None if _fixture_policy is None else _fixture_policy.fixture_id)
    guard.check()
    output_dir.mkdir(parents=True)
    initial = _initial_manifest(config, config_path, output_dir, command or sys.argv[:], root, producer, source,
                                _fixture_policy)
    initial_path = output_dir / ARTIFACTS[0]
    guard.check()
    initial_path.write_text(stable_json(initial) + "\n", encoding="utf-8", newline="\n")
    guard.check()
    _verify_initial_binding(strict_json_loads(initial_path.read_text(encoding="utf-8")),
                            config, source, producer, config_path, _fixture_policy)
    state = ProbeState(config, capture_rows=False, _fixture_policy=_fixture_policy)
    cpu_started = time.process_time()
    source_closure = None
    source_status = "NOT_ASSESSED"
    report_status, reason = "NOT_ASSESSED", "NOT_ASSESSED_IO_OR_SERIALIZATION_ERROR"
    error = None
    outcomes_path, records_path = output_dir / ARTIFACTS[1], output_dir / ARTIFACTS[2]
    try:
        with outcomes_path.open("x", encoding="utf-8", newline="\n") as outcomes_stream, \
             records_path.open("x", encoding="utf-8", newline="\n") as records_stream:
            try:
                source_status_stage = _stage_partition("source", config, state, initial, guard,
                                                       outcomes_stream, records_stream,
                                                       _proposal_provider=_proposal_provider)
                if source_status_stage == "quota_shortfall":
                    source_status = "NO_GO_SOURCE_CLOSURE"
                    reason = "SOURCE_QUOTA_SHORTFALL"
                else:
                    source_rows = _read_jsonl(outcomes_path, check=guard.check)
                    source_records = _read_jsonl(records_path, check=guard.check)
                    replayed, _ = replay_prefix(config, source_rows, source_records, guard,
                        capture_rows=False, _fixture_policy=_fixture_policy,
                        _proposal_provider=_proposal_provider)
                    if replayed.quota_counts != state.quota_counts:
                        raise ProbeFailure("NOT_ASSESSED_REPLAY_MISMATCH", "source_replay", reason="source_quota_mismatch")
                    projection = audit_records(source_records, source_rows, config,
                        check=guard.check,
                        _fixture_targets=None if _fixture_policy is None else _fixture_policy.quotas)["source_train"]
                    floor_pass, support_pass = _source_floors(projection, _fixture_policy)
                    if not floor_pass:
                        source_status, reason = "NO_GO_SOURCE_CLOSURE", "SOURCE_TRAIN_FLOOR_FAILED"
                    elif not support_pass:
                        source_status, reason = "NO_GO_SOURCE_CLOSURE", "SOURCE_SUPPORT_FAILED"
                    else:
                        source_status = "PASS_SOURCE_CLOSURE"
                source_closure = _write_closure(output_dir, config, initial, state,
                    source_status, reason if source_status != "PASS_SOURCE_CLOSURE" else None, guard,
                    projection if source_status_stage == "complete" else None)
                atomic_json(output_dir / ARTIFACTS[3], source_closure)
                _verify_source_closure(strict_json_loads((output_dir / ARTIFACTS[3]).read_text(encoding="utf-8")),
                                       output_dir, config, source, producer, _fixture_policy)
                if source_gate_transition(source_status) and state.profiles_for("development"):
                    _stage_partition("development", config, state, initial, guard,
                                      outcomes_stream, records_stream,
                                      _proposal_provider=_proposal_provider)
                    initial["partitions"]["development"]["status"] = "complete" if state.partition_closed("development") else "quota_shortfall"
                else:
                    initial["partitions"]["development"]["status"] = "not_started"
            except ProbeFailure as exc:
                source_status = source_status if source_status != "NOT_ASSESSED" else "NOT_ASSESSED"
                reason = exc.code
                if exc.partition in initial["partitions"]:
                    initial["partitions"][exc.partition]["status"] = "not_assessed"
                error = {"type": type(exc).__name__, "message": exc.code, "phase": exc.phase,
                    "partition": exc.partition, "candidate_index": exc.candidate_index, "reason": exc.reason}
                if not (output_dir / ARTIFACTS[3]).exists():
                    source_closure = _write_closure(output_dir, config, initial, state, "NOT_ASSESSED", reason, guard)
                    atomic_json(output_dir / ARTIFACTS[3], source_closure)
            except (TimeoutError, MemoryError, RuntimeError) as exc:
                for part in initial["partitions"].values():
                    if part["status"] == "running":
                        part["status"] = "not_assessed"
                reason = "NOT_ASSESSED_RESOURCE_CAP" if isinstance(exc, (TimeoutError, MemoryError)) else "NOT_ASSESSED_RESOURCE_CAP"
                error = {"type": type(exc).__name__, "message": reason, "phase": "generation",
                         "partition": None, "candidate_index": None, "reason": reason}
            except Exception as exc:
                for part in initial["partitions"].values():
                    if part["status"] == "running":
                        part["status"] = "not_assessed"
                reason = "NOT_ASSESSED_IO_OR_SERIALIZATION_ERROR"
                error = {"type": type(exc).__name__, "message": reason, "phase": "generation",
                         "partition": None, "candidate_index": None, "reason": reason}
    except Exception as exc:
        reason = "NOT_ASSESSED_IO_OR_SERIALIZATION_ERROR"
        error = {"type": type(exc).__name__, "message": reason, "phase": "artifact_open",
                 "partition": None, "candidate_index": None, "reason": reason}

    if not (output_dir / ARTIFACTS[3]).exists() and outcomes_path.exists() and records_path.exists():
        source_closure = _write_closure(output_dir, config, initial, state,
                                        "NOT_ASSESSED", reason, guard)
        atomic_json(output_dir / ARTIFACTS[3], source_closure)

    try:
        _verify_initial_binding(strict_json_loads(initial_path.read_text(encoding="utf-8")),
                                config, _source_identity(root, None if _fixture_policy is None else _fixture_policy.fixture_id),
                                _producer_identity(config, root), config_path, _fixture_policy)
        guard.check()
        outcomes = _read_jsonl(outcomes_path, check=guard.check) if outcomes_path.exists() else []
        records = _read_jsonl(records_path, check=guard.check) if records_path.exists() else []
        replayed, denominator_sha = replay_prefix(config, outcomes, records, guard, capture_rows=False,
            _fixture_policy=_fixture_policy, _proposal_provider=_proposal_provider)
        replay_doc = {"replay_version": "dwr-v4-candidate-replay-v1", "status": "PASS_REPLAY",
            "config_sha256": config.config_sha256, "source": initial["source"], "producer": producer,
            "partition_attempts": {p: sum(row["partition"] == p for row in outcomes) for p in ("source", "development")},
            "outcome_prefix_bytes": outcomes_path.stat().st_size if outcomes_path.exists() else 0,
            "outcome_prefix_sha256": sha256_file(outcomes_path) if outcomes_path.exists() else hashlib.sha256(b"").hexdigest(),
            "record_prefix_bytes": records_path.stat().st_size if records_path.exists() else 0,
            "record_prefix_sha256": sha256_file(records_path) if records_path.exists() else hashlib.sha256(b"").hexdigest(),
            "quota_targets": {p: _profile_maps(config, p, replayed)[0] for p in ("source", "development")},
            "quota_counts": {p: _profile_maps(config, p, replayed)[1] for p in ("source", "development")},
            "quota_closed": {"source": replayed.partition_closed("source"),
                "development": {_profile_key(p): all(replayed.quota_counts["development"][p][c] == replayed.quota_for("development", p, c) for c in _CELLS)
                                 for p in replayed.profiles_for("development")}},
            "denominator_projection_sha256": denominator_sha}
        _verify_replay_document(replay_doc, output_dir, config, initial["source"], producer, _fixture_policy)
        atomic_json(output_dir / ARTIFACTS[4], replay_doc)
        public_projection = audit_records(records, outcomes, config, check=guard.check,
            _fixture_targets=None if _fixture_policy is None else _fixture_policy.quotas)
        audit_doc = {"audit_version": AUDIT_VERSION, "status": "PASS_AUDIT", "config_sha256": config.config_sha256,
            "source": initial["source"], "producer": producer,
            "replay_sha256": sha256_file(output_dir / ARTIFACTS[4]),
            "source_closure_sha256": sha256_file(output_dir / ARTIFACTS[3]),
            "public_projection": public_projection,
            "integrity": {"status": "pass", "failures": []}, "resource_snapshot": _resource_snapshot(guard)}
        _verify_audit_document(audit_doc, output_dir, config, initial["source"], producer)
        atomic_json(output_dir / ARTIFACTS[5], audit_doc)
        audit_ref = _artifact_ref(output_dir / ARTIFACTS[5])
        replay_ref = _artifact_ref(output_dir / ARTIFACTS[4])
        if source_status != "PASS_SOURCE_CLOSURE":
            report_status = "NO_GO_PROPOSAL" if source_status == "NO_GO_SOURCE_CLOSURE" else "NOT_ASSESSED"
        elif not replayed.partition_closed("development"):
            reason, report_status = "DEVELOPMENT_VALIDATION_QUOTA_SHORTFALL", "NO_GO_PROPOSAL"
        else:
            profile_status = _development_floors(replayed, public_projection)
            proxy = public_projection.get("four_cohort_proxy")
            if any(value != "pass" for value in profile_status.values()):
                reason, report_status = "DEVELOPMENT_FLOOR_FAILED", "NO_GO_PROPOSAL"
            elif proxy is None or proxy.get("pass") is not True:
                reason, report_status = "FOUR_COHORT_PROXY_FAILED", "NO_GO_PROPOSAL"
            else:
                reason, report_status = None, "PASS_DEVELOPMENT_PROPOSAL"
            if _fixture_policy is not None and report_status == "PASS_DEVELOPMENT_PROPOSAL":
                report_status = "FIXTURE_PASS"
        initial_sha = sha256_file(initial_path)
        closure_sha = sha256_file(output_dir / ARTIFACTS[3])
        report = _terminal_report(config, source_closure,
            {**replay_doc, "artifact_sha256": replay_ref["sha256"]},
            {**audit_doc, "artifact_sha256": audit_ref["sha256"]}, reason,
            report_status, guard, initial_sha, closure_sha, initial["source"],
            profile_status if source_status == "PASS_SOURCE_CLOSURE" and replayed.partition_closed("development") else None)
        _verify_terminal_report(report, config, initial["source"], initial_sha, closure_sha,
            source_closure, {**replay_doc, "artifact_sha256": replay_ref["sha256"]},
            {**audit_doc, "artifact_sha256": audit_ref["sha256"]})
        atomic_json(output_dir / ARTIFACTS[6], report)
    except ProbeFailure as exc:
        reason, report_status = exc.code, "NOT_ASSESSED"
        error = {"type": type(exc).__name__, "message": reason, "phase": "replay_audit_finalization",
                 "partition": None, "candidate_index": None, "reason": reason}
        if not (output_dir / ARTIFACTS[3]).exists():
            source_closure = _write_closure(output_dir, config, initial, state, "NOT_ASSESSED", reason, guard)
            atomic_json(output_dir / ARTIFACTS[3], source_closure)
        atomic_json(output_dir / ARTIFACTS[6], _not_assessed_report(
            config, initial["source"], initial_path, output_dir / ARTIFACTS[3], guard, reason, state,
            initial["partitions"]["development"]["attempts"]))
    except (TimeoutError, MemoryError, RuntimeError) as exc:
        reason, report_status = "NOT_ASSESSED_RESOURCE_CAP", "NOT_ASSESSED"
        error = {"type": type(exc).__name__, "message": reason, "phase": "replay_audit_finalization",
                 "partition": None, "candidate_index": None, "reason": reason}
        if not (output_dir / ARTIFACTS[3]).exists():
            source_closure = _write_closure(output_dir, config, initial, state, "NOT_ASSESSED", reason, guard)
            atomic_json(output_dir / ARTIFACTS[3], source_closure)
        atomic_json(output_dir / ARTIFACTS[6], _not_assessed_report(
            config, initial["source"], initial_path, output_dir / ARTIFACTS[3], guard, reason, state,
            initial["partitions"]["development"]["attempts"]))
    except Exception as exc:
        reason, report_status = "NOT_ASSESSED_AUDIT_MISMATCH", "NOT_ASSESSED"
        error = {"type": type(exc).__name__, "message": reason, "phase": "replay_audit_finalization",
                 "partition": None, "candidate_index": None, "reason": reason}
        if not (output_dir / ARTIFACTS[3]).exists():
            source_closure = _write_closure(output_dir, config, initial, state, "NOT_ASSESSED", reason, guard)
            atomic_json(output_dir / ARTIFACTS[3], source_closure)
        atomic_json(output_dir / ARTIFACTS[6], _not_assessed_report(
            config, initial["source"], initial_path, output_dir / ARTIFACTS[3], guard, reason, state,
            initial["partitions"]["development"]["attempts"]))

    try:
        _ensure_failure_artifacts(output_dir, config, initial["source"], producer, state, initial, guard)
        replay_path, audit_path = output_dir / ARTIFACTS[4], output_dir / ARTIFACTS[5]
        replay_doc = strict_json_loads(replay_path.read_text(encoding="utf-8"))
        audit_doc = strict_json_loads(audit_path.read_text(encoding="utf-8"))
        _verify_replay_document(replay_doc, output_dir, config, initial["source"], producer, _fixture_policy)
        _verify_audit_document(audit_doc, output_dir, config, initial["source"], producer)
        report_path = output_dir / ARTIFACTS[6]
        report = strict_json_loads(report_path.read_text(encoding="utf-8"))
        if report["replay_sha256"] is None:
            report["replay_sha256"] = sha256_file(replay_path)
        if report["audit_sha256"] is None:
            report["audit_sha256"] = sha256_file(audit_path)
        atomic_json(report_path, report)
    except Exception as exc:
        report_status = "NOT_ASSESSED"
        reason = "NOT_ASSESSED_IO_OR_SERIALIZATION_ERROR"
        error = {"type": type(exc).__name__, "message": reason, "phase": "failure_artifact_flush",
                 "partition": None, "candidate_index": None, "reason": reason}

    elapsed = guard.elapsed
    current_cpu = max(0.0, time.process_time() - cpu_started)
    initial.update({"status": "scientific_no_go" if report_status == "NO_GO_PROPOSAL" else
                    "not_assessed" if report_status == "NOT_ASSESSED" else
                    "fixture_complete" if report_status == "FIXTURE_PASS" else "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(), "error": error,
        "exit_code": 0 if report_status in ("PASS_DEVELOPMENT_PROPOSAL", "FIXTURE_PASS") else 1,
        "elapsed_wall_seconds": round(elapsed, 6), "cpu_seconds": round(current_cpu, 6),
        "peak_resident_bytes": guard.peak_bytes,
        "accounting_end_boundary": "post_first_terminal_write_resource_check"})
    terminal_artifacts = {}
    for name in ARTIFACTS[:-1]:
        path = output_dir / name
        if path.exists():
            terminal_artifacts[name] = _artifact_ref(path)
    initial["artifacts"] = terminal_artifacts
    guard.check()
    terminal_path = output_dir / ARTIFACTS[-1]
    atomic_json(terminal_path, initial)
    try:
        guard.check()
    except (TimeoutError, MemoryError, RuntimeError):
        initial["status"] = "not_assessed"
        initial["exit_code"] = 1
        initial["error"] = {"type": "ResourceCap", "message": "NOT_ASSESSED_RESOURCE_CAP",
            "phase": "post_terminal_write", "partition": None, "candidate_index": None,
            "reason": "NOT_ASSESSED_RESOURCE_CAP"}
        # One bounded accounting/status flush; the guard is not reset.
        revoked = strict_json_loads((output_dir / ARTIFACTS[6]).read_text(encoding="utf-8"))
        revoked["status"] = "NOT_ASSESSED"
        revoked["reason"] = "NOT_ASSESSED_RESOURCE_CAP"
        revoked["public_floors"] = {
            "source_train": {"status": "not_assessed", "values": None},
            "development_validation_by_profile": {
                key: {"status": "not_assessed", "values": None}
                for key in revoked["public_floors"]["development_validation_by_profile"]},
            "four_cohort_proxy": {"status": "not_assessed", "values": None}}
        revoked["four_cohort_proxy"] = None
        revoked["g2_status"] = "not_assessed"
        atomic_json(output_dir / ARTIFACTS[6], revoked)
        initial["artifacts"][ARTIFACTS[6]] = _artifact_ref(output_dir / ARTIFACTS[6])
        atomic_json(terminal_path, initial)
    return 0 if report_status in ("PASS_DEVELOPMENT_PROPOSAL", "FIXTURE_PASS") else 1


def run_probe(config_path: Path, output_dir: Path, command: list[str] | None = None,
              *, guard: ResourceGuard | None = None) -> int:
    """Production entry point; pinned config and policy are not caller-overridable."""
    return _run_controller(config_path, output_dir, command, guard=guard)


def run_source_only_fixture(config_path: Path, output_dir: Path, *, fixture_id: str) -> str:
    """Private source-only integration path; this label never implies full probe success."""
    from dataclasses import replace

    policy = _FixturePolicy(fixture_id, source_only=True)
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    root = Path(__file__).resolve().parents[2]

    def fixed_candidate(config, partition, index, active_cell):
        if partition != "source" or index != 0 or active_cell != (("eq",), "C"):
            raise ValueError("fixture proposal stream received an unregistered request")
        # Candidate 2 is a single previously pinned handcrafted stream point from
        # the producer tests. The stream position is local to this fixture.
        fixed = propose_candidate(config, "source", 2, active_cell)
        return replace(fixed, candidate_index=index)

    _run_controller(config_path, output_dir, ["internal-fixture", fixture_id],
                    _fixture_policy=policy, _proposal_provider=fixed_candidate)
    files = {path.name for path in output_dir.iterdir() if path.is_file()}
    if files != set(ARTIFACTS):
        raise ValueError("fixture core did not emit the exact closed artifact set")
    read = lambda name: strict_json_loads((output_dir / name).read_text(encoding="utf-8"))
    source = read("source_closure.json")
    replay = read("candidate_replay.json")
    audit = read("independent_audit.json")
    manifest = read("probe_manifest.json")
    if (source["status"] != "PASS_SOURCE_CLOSURE" or replay["status"] != "PASS_REPLAY" or
            audit["status"] != "PASS_AUDIT" or manifest["source"].get("fixture_id") != fixture_id or
            manifest["partitions"]["development"]["attempts"] != 0):
        raise ValueError("fixture source-close/replay/independent-audit integration failed")
    atomic_json(output_dir / "fixture_identity.json", {
        "fixture_id": fixture_id, "label": "FIXTURE_SOURCE_ONLY_PASS",
        "config_sha256": load_probe_config(config_path).config_sha256,
        "policy": {"source_quota": {"eq/C": 1}, "development_quota": {},
                   "source_minimum_accepted": 1, "source_minimum_signatures": 1,
                   "source_support_cell": "C"},
        "scientific_claim": None,
    })
    return "FIXTURE_SOURCE_ONLY_PASS"


def run_fixture_probe(config_path: Path, output_dir: Path, *, fixture_id: str,
                      _proposal_provider, _guard: ResourceGuard | None = None,
                      _policy: _FixturePolicy | None = None) -> str:
    """Run the full registered profile/cell shape with explicit fixture quotas."""
    if not callable(_proposal_provider):
        raise TypeError("fixture proposal provider must be callable")
    policy = _FixturePolicy(fixture_id) if _policy is None else _policy
    if (not isinstance(policy, _FixturePolicy) or policy.fixture_id != fixture_id or
            policy.source_only):
        raise ValueError("full fixture policy identity or scope is invalid")
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    root = Path(__file__).resolve().parents[2]
    _run_controller(config_path, output_dir, ["internal-fixture", fixture_id],
                    guard=_guard, _fixture_policy=policy, _proposal_provider=_proposal_provider)
    files = {path.name for path in output_dir.iterdir() if path.is_file()}
    if files != set(ARTIFACTS):
        raise ValueError("fixture core did not emit the exact closed artifact set")
    read = lambda name: strict_json_loads((output_dir / name).read_text(encoding="utf-8"))
    report, replay, audit, manifest = (read("probe_report.json"), read("candidate_replay.json"),
                                       read("independent_audit.json"), read("probe_manifest.json"))
    proxy = audit["public_projection"]["four_cohort_proxy"]
    expected_dev_attempts = sum(sum(policy.quotas["development"][profile].values())
                                for profile in policy.quotas["development"])
    if (report["status"] != "FIXTURE_PASS" or replay["status"] != "PASS_REPLAY" or
            audit["status"] != "PASS_AUDIT" or proxy is None or proxy["pass"] is not True or
            manifest["source"].get("fixture_id") != fixture_id or
            manifest["partitions"]["development"]["attempts"] != expected_dev_attempts or
            manifest["status"] != "fixture_complete"):
        raise ValueError("full fixture core did not close all profiles, replay, audit, and proxy")
    atomic_json(output_dir / "fixture_identity.json", {
        "fixture_id": fixture_id, "label": "FIXTURE_PASS",
        "config_sha256": load_probe_config(config_path).config_sha256,
        "policy": {part: {_profile_key(profile): dict(cells)
                           for profile, cells in policy.quotas[part].items()}
                   for part in ("source", "development")},
        "scientific_claim": None,
    })
    return "FIXTURE_PASS"
