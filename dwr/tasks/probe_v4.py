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
from .canonical import content_id, development_split, source_split, structure_signature
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


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def _utc_timestamp(value) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


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
    lid = hashlib.sha256(b"dwr-lineage-v1\0" + cid.encode("ascii")).hexdigest()
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
            any(set(item) != {"path", "sha256"} or not _is_sha256(item["sha256"])
                for item in producer)):
        raise ValueError("producer identity shape or registered order mismatch")
    if (initial["probe_id"] != "dwr-constraint-development-probe-v4" or
            not _utc_timestamp(initial["created_utc"]) or initial["status"] != "running" or
            initial["finished_utc"] is not None or initial["error"] is not None or
            initial["exit_code"] is not None or initial["elapsed_wall_seconds"] is not None or
            initial["cpu_seconds"] is not None or initial["peak_resident_bytes"] is not None or
            initial["accounting_end_boundary"] is not None or
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
    if (not isinstance(initial["limits"], dict) or set(initial["limits"]) != set(expected_limits) or
            any(type(initial["limits"][key]) is not int or initial["limits"][key] != value
                for key, value in expected_limits.items())):
        raise ValueError("initial manifest resource limits mismatch")
    if (not isinstance(initial["python"], dict) or
            set(initial["python"]) != {"implementation", "version", "executable"} or
            any(not isinstance(value, str) or not value for value in initial["python"].values()) or
            not isinstance(initial["platform"], dict) or
            set(initial["platform"]) != {"system", "release", "machine"} or
            any(not isinstance(value, str) or not value for value in initial["platform"].values()) or
            not isinstance(initial["rng_identity"], dict) or
            set(initial["rng_identity"]) != {"generator_id", "seed_derivation", "seed_material", "root_seed"} or
            initial["rng_identity"] != {"generator_id": config.generator_id,
                "seed_derivation": "sha256-prefix64-v4",
                "seed_material": config.raw_config["rng"]["material"],
                "root_seed": config.root_seed} or
            type(initial["seed"]) is not int or type(initial["rng_identity"]["root_seed"]) is not int or
            not isinstance(initial["command"], list) or
            not initial["command"] or any(not isinstance(value, str) for value in initial["command"]) or
            not isinstance(initial["cwd"], str) or not Path(initial["cwd"]).is_absolute() or
            not isinstance(initial["output_dir"], str) or not Path(initial["output_dir"]).is_absolute() or
            initial["config_path"] != str(Path(config_path).resolve())):
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
        if any(type(count) is not int for cells in item["quota_targets"].values() for count in cells.values()):
            raise ValueError("initial manifest target values must be integers")


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


def _nonnegative_integer(value) -> bool:
    return type(value) is int and value >= 0


def _finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_source_projection(projection: dict) -> None:
    fields = {"accepted_groups", "unique_oracle_groups", "structure_signatures", "profile_groups",
        "solution_value_counts", "given_counts", "sensitive_groups", "sensitivity_fraction",
        "eligible_envelope_groups", "envelope_yield", "groups_by_cell", "eligible_groups_by_cell",
        "four_cell_mass"}
    if not isinstance(projection, dict) or set(projection) != fields:
        raise ValueError("source projection fields malformed")
    for key in ("accepted_groups", "unique_oracle_groups", "structure_signatures", "sensitive_groups",
                "eligible_envelope_groups"):
        if not _nonnegative_integer(projection[key]):
            raise ValueError("source projection count malformed")
    if (projection["accepted_groups"] > projection["unique_oracle_groups"] or
            projection["sensitive_groups"] > projection["unique_oracle_groups"] or
            projection["eligible_envelope_groups"] > projection["sensitive_groups"]):
        raise ValueError("source projection count relationship malformed")
    for key, expected in (("profile_groups", {"eq", "neq", "add_mod"}),
                          ("solution_value_counts", {"0", "1", "2"}),
                          ("given_counts", {"2", "3"}),
                          ("groups_by_cell", set(_CELLS)),
                          ("eligible_groups_by_cell", set(_CELLS))):
        value = projection[key]
        if (not isinstance(value, dict) or set(value) != expected or
                any(not _nonnegative_integer(count) for count in value.values())):
            raise ValueError("source projection nested counts malformed")
    mass = projection["four_cell_mass"]
    if (not isinstance(mass, dict) or set(mass) != {"numerator", "denominator", "fraction"} or
            not _nonnegative_integer(mass["numerator"]) or
            not _nonnegative_integer(mass["denominator"]) or
            not _finite_number(mass["fraction"]) or not 0 <= mass["fraction"] <= 1 or
            (mass["denominator"] == 0 and mass["fraction"] != 0)):
        raise ValueError("source projection mass malformed")
    for key in ("sensitivity_fraction", "envelope_yield"):
        if not _finite_number(projection[key]) or not 0 <= projection[key] <= 1:
            raise ValueError("source projection fraction malformed")


def _validate_profile_projection(value: dict) -> None:
    fields = {"status", "eligible_groups", "unique_oracle_groups", "eligible_signatures",
        "scientifically_eligible_groups", "scientific_eligibility_fraction",
        "eligible_envelope_groups", "envelope_yield", "groups_by_cell", "eligible_groups_by_cell"}
    if not isinstance(value, dict) or set(value) != fields or value["status"] not in (
            "complete", "not_attempted", "not_assessed"):
        raise ValueError("development profile projection fields/status malformed")
    metrics = fields - {"status"}
    if value["status"] != "complete":
        if any(value[key] is not None for key in metrics):
            raise ValueError("incomplete development projection carries metrics")
        return
    for key in ("eligible_groups", "unique_oracle_groups", "eligible_signatures",
                "scientifically_eligible_groups", "eligible_envelope_groups"):
        if not _nonnegative_integer(value[key]):
            raise ValueError("development projection count malformed")
    for key in ("scientific_eligibility_fraction", "envelope_yield"):
        if not _finite_number(value[key]) or not 0 <= value[key] <= 1:
            raise ValueError("development projection fraction malformed")
    for key in ("groups_by_cell", "eligible_groups_by_cell"):
        if (not isinstance(value[key], dict) or set(value[key]) != set(_CELLS) or
                any(not _nonnegative_integer(count) for count in value[key].values())):
            raise ValueError("development projection cell map malformed")


def _validate_four_cohort_proxy(proxy: dict | None) -> None:
    names = {"source_train_eligible", "development_validation_eq_neq_primary_eligible",
             "development_validation_eq_add_mod_primary_eligible",
             "development_validation_neq_add_mod_primary_eligible"}
    if proxy is None:
        return
    if not isinstance(proxy, dict) or set(proxy) != {"cohorts", "pass"} or type(proxy["pass"]) is not bool:
        raise ValueError("four-cohort proxy shape malformed")
    cohorts = proxy["cohorts"]
    if not isinstance(cohorts, dict) or set(cohorts) != names:
        raise ValueError("four-cohort proxy cohort map malformed")
    for cohort in cohorts.values():
        if (not isinstance(cohort, dict) or set(cohort) != {"group_count", "groups_by_cell",
                "common_cell_groups", "mass_fraction", "pass"} or
                not _nonnegative_integer(cohort["group_count"]) or
                type(cohort["pass"]) is not bool or not _finite_number(cohort["mass_fraction"]) or
                not 0 <= cohort["mass_fraction"] <= 1 or
                not isinstance(cohort["groups_by_cell"], dict) or
                set(cohort["groups_by_cell"]) != set(_CELLS) or
                any(not _nonnegative_integer(count) for count in cohort["groups_by_cell"].values()) or
                not isinstance(cohort["common_cell_groups"], dict) or
                not set(cohort["common_cell_groups"]).issubset(_CELLS) or
                any(not _nonnegative_integer(count) for count in cohort["common_cell_groups"].values())):
            raise ValueError("four-cohort proxy cohort values malformed")
        expected_pass = bool(cohort["group_count"] and cohort["common_cell_groups"] and
                             cohort["mass_fraction"] >= 0.8)
        if cohort["pass"] is not expected_pass:
            raise ValueError("four-cohort proxy pass flag disagrees with mass")
    if proxy["pass"] is not all(cohort["pass"] for cohort in cohorts.values()):
        raise ValueError("four-cohort aggregate pass disagrees with cohorts")


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
        _validate_source_projection(projection)
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
    if closure["status"] == "PASS_SOURCE_CLOSURE":
        if (closure["reason"] is not None or closure["source_quota_closed"] is not True or
                counts != expected_targets or projection is None or support is None or
                support["cell_min_pass"] is not True or support["mass_pass"] is not True):
            raise ValueError("passing source closure lacks closed quotas, projection, or support")
    elif closure["status"] == "NO_GO_SOURCE_CLOSURE":
        if closure["reason"] not in ("SOURCE_QUOTA_SHORTFALL", "SOURCE_TRAIN_FLOOR_FAILED",
                                      "SOURCE_SUPPORT_FAILED"):
            raise ValueError("source no-go reason is not registered")
        if closure["reason"] == "SOURCE_QUOTA_SHORTFALL" and closure["source_quota_closed"]:
            raise ValueError("source shortfall claims closed quotas")
    elif not isinstance(closure["reason"], str) or not closure["reason"]:
        raise ValueError("unassessed source closure needs a typed reason")


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
    closed = replay["quota_closed"]
    if (not isinstance(replay["partition_attempts"], dict) or
            set(replay["partition_attempts"]) != {"source", "development"} or any(
                type(v) is not int or v < 0 or v > attempt_limit for v in replay["partition_attempts"].values()) or
            not isinstance(closed, dict) or set(closed) != {"source", "development"} or
            (closed["source"] is not None and type(closed["source"]) is not bool) or
            not isinstance(closed["development"], dict) or
            set(closed["development"]) != set(expected_targets["development"]) or any(
                value is not None and type(value) is not bool
                for value in closed["development"].values())):
        raise ValueError("candidate replay partition accounting malformed")
    prefixes = (("outcome_prefix_bytes", "outcome_prefix_sha256", "outcomes.jsonl"),
                ("record_prefix_bytes", "record_prefix_sha256", "accepted_records.jsonl"))
    for byte_key, hash_key, filename in prefixes:
        byte_count, digest = replay[byte_key], replay[hash_key]
        if byte_count is None or digest is None:
            if byte_count is not None or digest is not None:
                raise ValueError("candidate replay prefix byte/hash nullability mismatch")
        elif (type(byte_count) is not int or byte_count < 0 or
              not _is_sha256(digest) or
              _prefix_sha256(output / filename, byte_count) != digest):
            raise ValueError("candidate replay prefix hash mismatch")
    denominator_digest = replay["denominator_projection_sha256"]
    if denominator_digest is not None and not _is_sha256(denominator_digest):
        raise ValueError("candidate replay denominator digest is malformed")
    if replay["status"] == "NOT_ASSESSED" and (
            any(replay[key] is not None for pair in prefixes for key in pair[:2]) or
            denominator_digest is not None or closed["source"] is not None or
            any(value is not None for value in closed["development"].values())):
        raise ValueError("unassessed replay must use null prefix, digest and closure values")
    if replay["status"] == "PASS_REPLAY" and (
            denominator_digest is None or any(replay[key] is None for pair in prefixes for key in pair[:2]) or
            closed["source"] is None or any(value is None for value in closed["development"].values())):
        raise ValueError("passing replay is missing completed closure evidence")


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
    projection = audit["public_projection"]
    if (not isinstance(projection, dict) or
            set(projection) != {"source_train", "development_validation_by_profile", "four_cohort_proxy"}):
        raise ValueError("independent audit public projection has wrong fields")
    if projection["source_train"] is not None:
        _validate_source_projection(projection["source_train"])
    dev = projection["development_validation_by_profile"]
    dev_keys = {_profile_key(profile) for profile in config.development_profiles}
    if not isinstance(dev, dict) or set(dev) != dev_keys:
        raise ValueError("independent audit development projection profile set mismatch")
    for value in dev.values():
        _validate_profile_projection(value)
    _validate_four_cohort_proxy(projection["four_cohort_proxy"])
    integrity = audit["integrity"]
    if (not isinstance(integrity, dict) or set(integrity) != {"status", "failures"} or
            integrity["status"] not in ("pass", "fail", "not_assessed") or
            not isinstance(integrity["failures"], list) or
            any(not isinstance(code, str) or not code for code in integrity["failures"]) or
            integrity["failures"] != sorted(set(integrity["failures"])) or
            (audit["status"] == "PASS_AUDIT" and
             (integrity["status"] != "pass" or integrity["failures"] != [])) or
            (audit["status"] == "NO_GO_AUDIT" and
             (integrity["status"] != "fail" or not integrity["failures"])) or
            (audit["status"] == "NOT_ASSESSED" and integrity["status"] != "not_assessed")):
        raise ValueError("independent audit integrity/status mismatch")
    resource = audit["resource_snapshot"]
    if (not isinstance(resource, dict) or set(resource) != {"wall_seconds", "cpu_seconds", "peak_rss_bytes"} or
            any(value is not None and (not _finite_number(value) or value < 0)
                for value in resource.values())):
        raise ValueError("independent audit resource snapshot malformed")
    if audit["status"] == "PASS_AUDIT" and any(value is None for value in resource.values()):
        raise ValueError("passing independent audit lacks resource accounting")


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
                                 "NOT_ASSESSED", "NOT_ASSESSED_INCOMPLETE_STAGE", guard,
                                 fixture_policy=state._fixture_policy)
        atomic_json(closure_path, closure)
    replay_path = output / ARTIFACTS[4]
    if not replay_path.exists():
        replay = {"replay_version": "dwr-v4-candidate-replay-v1", "status": "NOT_ASSESSED",
            "config_sha256": config.config_sha256, "source": source, "producer": producer,
            "partition_attempts": {part: manifest["partitions"][part]["attempts"]
                                   for part in ("source", "development")},
            "outcome_prefix_bytes": None, "outcome_prefix_sha256": None,
            "record_prefix_bytes": None, "record_prefix_sha256": None,
            "quota_targets": {part: _profile_maps(config, part, state)[0]
                              for part in ("source", "development")},
            "quota_counts": {part: _profile_maps(config, part, state)[1]
                             for part in ("source", "development")},
            "quota_closed": {"source": None,
                "development": {_profile_key(p): None for p in state.profiles_for("development")}},
            "denominator_projection_sha256": None}
        atomic_json(replay_path, replay)
    audit_path = output / ARTIFACTS[5]
    if not audit_path.exists():
        source_closure = strict_json_loads(closure_path.read_text(encoding="utf-8"))
        dev_attempted = manifest["partitions"]["development"]["attempts"] > 0
        dev_status = "not_assessed" if dev_attempted else "not_attempted"
        dev_fields = {"status": dev_status, "eligible_groups": None,
            "unique_oracle_groups": None, "eligible_signatures": None,
            "scientifically_eligible_groups": None, "scientific_eligibility_fraction": None,
            "eligible_envelope_groups": None, "envelope_yield": None,
            "groups_by_cell": None, "eligible_groups_by_cell": None}
        audit = {"audit_version": AUDIT_VERSION, "status": "NOT_ASSESSED",
            "config_sha256": config.config_sha256, "source": source, "producer": producer,
            "replay_sha256": sha256_file(replay_path), "source_closure_sha256": sha256_file(closure_path),
            "public_projection": {"source_train": source_closure["train_projection"],
                "development_validation_by_profile": {
                    _profile_key(profile): dict(dev_fields) for profile in state.profiles_for("development")},
                "four_cohort_proxy": None},
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
                   train_projection: dict | None = None, fixture_policy=None) -> dict:
    outcomes_path, records_path = output / "outcomes.jsonl", output / "accepted_records.jsonl"
    source_attempts = manifest["partitions"]["source"]["attempts"]
    outcome_size = outcomes_path.stat().st_size
    record_size = records_path.stat().st_size
    source_projection = train_projection
    support = None
    if source_projection is not None:
        minimum_cell = 10 if fixture_policy is None else (1 if not fixture_policy.source_only else 0)
        fixture_cell_pass = (source_projection["eligible_groups_by_cell"]["C"] >= 1
                             if fixture_policy is not None and fixture_policy.source_only else
                             all(source_projection["eligible_groups_by_cell"][c] >= minimum_cell
                                 for c in _CELLS))
        support = {"cells": source_projection["eligible_groups_by_cell"],
                   "cell_min_pass": fixture_cell_pass if fixture_policy is not None else
                       all(source_projection["eligible_groups_by_cell"][c] >= 10 for c in _CELLS),
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
    source_by_cell = {cell: sum(quota_counts.get("source", {}).get(_profile_key(p), {}).get(cell, 0)
                                for p in config.source_profiles) for cell in _CELLS}
    source_closed = (None if source_closure["status"] == "NOT_ASSESSED"
                     else source_closure.get("source_quota_closed"))
    dev_closed = (replay.get("quota_closed", {}).get("development",
        {_profile_key(profile): None for profile in config.development_profiles})
        if source_closure["status"] == "PASS_SOURCE_CLOSURE" else
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
                            audit: dict, fixture_policy=None, error: dict | None = None) -> None:
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
    expected_dev_counts = replay["quota_counts"]["development"]
    expected_dev_closed = (replay["quota_closed"]["development"]
        if closure["status"] == "PASS_SOURCE_CLOSURE" else
        {_profile_key(profile): None for profile in config.development_profiles})
    expected_source_closed = (None if closure["status"] == "NOT_ASSESSED" else
                              closure["source_quota_closed"])
    expected_source_by_cell = {cell: sum(closure["source_quota_counts"][profile][cell]
                                         for profile in closure["source_quota_counts"])
                               for cell in _CELLS}
    if (not isinstance(quota, dict) or set(quota) != {"source_by_profile_cell", "source_by_cell",
            "source_closed", "development_validation_by_profile_cell", "development_validation_closed"} or
            quota["source_by_profile_cell"] != closure["source_quota_counts"] or
            quota["source_by_cell"] != expected_source_by_cell or
            quota["development_validation_by_profile_cell"] != expected_dev_counts or
            quota["source_closed"] != expected_source_closed or
            quota["development_validation_closed"] != expected_dev_closed):
        raise ValueError("terminal report nested quota accounting mismatch")
    source_profile_keys = {_profile_key(profile) for profile in config.source_profiles}
    for key, expected in (("source_by_profile_cell", source_profile_keys),):
        value = quota[key]
        if (not isinstance(value, dict) or set(value) != expected or any(
                not isinstance(cells, dict) or set(cells) != set(_CELLS) or any(
                    type(count) is not int or count < 0 for count in cells.values())
                for cells in value.values())):
            raise ValueError("terminal report source quota nested types malformed")
    for key, expected_profiles in (("development_validation_by_profile_cell",
                                    {_profile_key(profile) for profile in config.development_profiles}),):
        value = quota[key]
        if (not isinstance(value, dict) or set(value) != expected_profiles or any(
                not isinstance(cells, dict) or set(cells) != set(_CELLS) or any(
                    type(count) is not int or count < 0 for count in cells.values())
                for cells in value.values())):
            raise ValueError("terminal report development quota nested types malformed")
    for key in ("source_closed",):
        if quota[key] is not None and type(quota[key]) is not bool:
            raise ValueError("terminal report source closure type malformed")
    if (not isinstance(quota["development_validation_closed"], dict) or
            set(quota["development_validation_closed"]) !=
                {_profile_key(profile) for profile in config.development_profiles} or
            any(value is not None and type(value) is not bool
                for value in quota["development_validation_closed"].values())):
        raise ValueError("terminal report development closure types malformed")
    if (not isinstance(quota["source_by_cell"], dict) or set(quota["source_by_cell"]) != set(_CELLS) or
            any(type(value) is not int or value < 0 for value in quota["source_by_cell"].values())):
        raise ValueError("terminal report source cell quota types malformed")
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
    _validate_terminal_acceptance(report, closure, replay, audit, error, fixture_policy)


def _validate_terminal_acceptance(report: dict, closure: dict, replay: dict, audit: dict,
                                  error: dict | None = None, fixture_policy=None) -> None:
    status = report["status"]
    if status in ("PASS_DEVELOPMENT_PROPOSAL", "FIXTURE_PASS"):
        fixture = status == "FIXTURE_PASS"
        if fixture != (fixture_policy is not None) or error is not None or report["reason"] is not None:
            raise ValueError("terminal PASS has fixture or operational identity mismatch")
        if (closure.get("status") != "PASS_SOURCE_CLOSURE" or
                closure.get("source_quota_closed") is not True or
                closure.get("source_quota_counts") != closure.get("source_quota_targets") or
                closure.get("train_projection") is None or
                not isinstance(closure.get("source_support"), dict) or
                closure["source_support"].get("cell_min_pass") is not True or
                closure["source_support"].get("mass_pass") is not True):
            raise ValueError("terminal PASS lacks source closure/support evidence")
        if (replay.get("status") != "PASS_REPLAY" or
                replay.get("quota_closed", {}).get("source") is not True or
                not all(value is True for value in replay.get("quota_closed", {}).get("development", {}).values()) or
                replay.get("quota_counts", {}) != replay.get("quota_targets", {})):
            raise ValueError("terminal PASS lacks complete replay/quota evidence")
        projection = audit.get("public_projection", {})
        dev = projection.get("development_validation_by_profile", {})
        if (audit.get("status") != "PASS_AUDIT" or
                audit.get("integrity", {}).get("status") != "pass" or
                projection.get("source_train") != closure.get("train_projection") or
                set(dev) != {"eq/neq", "eq/add_mod", "neq/add_mod"} or
                any(value.get("status") != "complete" for value in dev.values()) or
                projection.get("four_cohort_proxy") is None or
                projection["four_cohort_proxy"].get("pass") is not True):
            raise ValueError("terminal PASS lacks independent audit/proxy evidence")
        if (report["public_floors"]["source_train"]["status"] != "pass" or
                any(value["status"] != "pass" for value in
                    report["public_floors"]["development_validation_by_profile"].values()) or
                report["public_floors"]["four_cohort_proxy"]["status"] != "pass"):
            raise ValueError("terminal PASS has a failed or unassessed public floor")
        resource = report["resource_snapshot"]
        if (resource["wall_seconds"] is None or resource["wall_seconds"] >= 900 or
                resource["peak_rss_bytes"] is None or resource["peak_rss_bytes"] > 1073741824):
            raise ValueError("terminal PASS exceeds or lacks registered resource accounting")
    elif status == "NO_GO_PROPOSAL":
        allowed = {"SOURCE_QUOTA_SHORTFALL", "SOURCE_TRAIN_FLOOR_FAILED", "SOURCE_SUPPORT_FAILED",
                   "DEVELOPMENT_VALIDATION_QUOTA_SHORTFALL", "DEVELOPMENT_FLOOR_FAILED",
                   "FOUR_COHORT_PROXY_FAILED"}
        if error is not None or report["reason"] not in allowed:
            raise ValueError("terminal scientific no-go has an operational cause or unknown reason")
        reason = report["reason"]
        source_pass = (closure.get("status") == "PASS_SOURCE_CLOSURE" and
                       closure.get("source_quota_closed") is True)
        replay_pass = replay.get("status") == "PASS_REPLAY"
        if reason == "SOURCE_QUOTA_SHORTFALL" and not (
                closure.get("status") == "NO_GO_SOURCE_CLOSURE" and
                closure.get("reason") == reason and closure.get("source_quota_closed") is False):
            raise ValueError("source quota no-go lacks source shortfall evidence")
        if reason in ("SOURCE_TRAIN_FLOOR_FAILED", "SOURCE_SUPPORT_FAILED") and not (
                closure.get("status") == "NO_GO_SOURCE_CLOSURE" and
                closure.get("reason") == reason and closure.get("train_projection") is not None):
            raise ValueError("source floor no-go lacks assessed source evidence")
        if reason == "DEVELOPMENT_VALIDATION_QUOTA_SHORTFALL" and not (
                source_pass and replay_pass and
                not all(replay.get("quota_closed", {}).get("development", {}).values())):
            raise ValueError("development quota no-go lacks replayed shortfall evidence")
        if reason == "DEVELOPMENT_FLOOR_FAILED" and not (
                source_pass and replay_pass and
                all(replay.get("quota_closed", {}).get("development", {}).values()) and
                any(floor["status"] == "fail" for floor in
                    report["public_floors"]["development_validation_by_profile"].values())):
            raise ValueError("development floor no-go lacks a failed assessed profile")
        if reason == "FOUR_COHORT_PROXY_FAILED" and not (
                source_pass and replay_pass and report["four_cohort_proxy"] is not None and
                report["four_cohort_proxy"].get("pass") is False):
            raise ValueError("proxy no-go lacks failed proxy evidence")
    elif status == "NOT_ASSESSED":
        allowed = {"NOT_ASSESSED_RESOURCE_CAP", "NOT_ASSESSED_IO_OR_SERIALIZATION_ERROR",
                   "NOT_ASSESSED_SOLVER_OR_ORACLE_ERROR", "NOT_ASSESSED_REPLAY_MISMATCH",
                   "NOT_ASSESSED_AUDIT_MISMATCH"}
        if (report["reason"] not in allowed or error is None or
                not isinstance(error, dict) or error.get("reason") != report["reason"] or
                any(not isinstance(error.get(key), str) or not error[key]
                    for key in ("type", "message", "phase"))):
            raise ValueError("terminal operational result has an unregistered or mismatched cause")


def _verify_terminal_manifest(terminal: dict, initial: dict, config: ProbeConfig,
                              output: Path, replay: dict, report: dict,
                              fixture_policy=None) -> None:
    fields = {"manifest_version", "probe_id", "status", "created_utc", "finished_utc", "command", "cwd",
        "source", "python", "platform", "config_path", "config_sha256", "resolved_config", "seed",
        "rng_identity", "producer", "limits", "output_dir", "partitions", "artifacts", "error",
        "exit_code", "elapsed_wall_seconds", "cpu_seconds", "peak_resident_bytes", "accounting_end_boundary"}
    if set(terminal) != fields or terminal.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("terminal manifest fields or version mismatch")
    if (terminal["status"] not in ("complete", "scientific_no_go", "not_assessed", "fixture_complete") or
            not _utc_timestamp(terminal["finished_utc"]) or terminal["source"] != initial["source"] or
            terminal["producer"] != initial["producer"] or terminal["config_sha256"] != config.config_sha256 or
            terminal["config_path"] != initial["config_path"] or terminal["output_dir"] != str(output.resolve()) or
            terminal["accounting_end_boundary"] != "post_first_terminal_write_resource_check"):
        raise ValueError("terminal manifest source/config/status binding mismatch")
    if (type(terminal["exit_code"]) is not int or
            type(terminal["elapsed_wall_seconds"]) not in (int, float) or
            not math.isfinite(terminal["elapsed_wall_seconds"]) or terminal["elapsed_wall_seconds"] < 0 or
            type(terminal["cpu_seconds"]) not in (int, float) or not math.isfinite(terminal["cpu_seconds"]) or
            terminal["cpu_seconds"] < 0 or
            (terminal["peak_resident_bytes"] is not None and
             (type(terminal["peak_resident_bytes"]) is not int or terminal["peak_resident_bytes"] < 0)) or
            (terminal["error"] is not None and (not isinstance(terminal["error"], dict) or
                set(terminal["error"]) != {"type", "message", "phase", "partition", "candidate_index", "reason"} or
                any(not isinstance(terminal["error"][key], str) or not terminal["error"][key]
                    for key in ("type", "message", "phase", "reason")) or
                terminal["error"]["partition"] not in (None, "source", "development") or
                (terminal["error"]["candidate_index"] is not None and
                 (type(terminal["error"]["candidate_index"]) is not int or
                  terminal["error"]["candidate_index"] < 0))))) :
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
        expected_targets = replay["quota_targets"][part]
        expected_profiles = set(expected_targets)
        if (not isinstance(value, dict) or set(value) != {"status", "attempts", "quota_targets", "quota_counts"} or
                value["status"] not in ("not_started", "running", "complete", "quota_shortfall", "not_assessed") or
                type(value["attempts"]) is not int or value["attempts"] != replay["partition_attempts"][part] or
                value["quota_targets"] != replay["quota_targets"][part] or
                value["quota_counts"] != replay["quota_counts"][part] or
                not isinstance(value["quota_targets"], dict) or
                set(value["quota_targets"]) != expected_profiles or
                not isinstance(value["quota_counts"], dict) or
                set(value["quota_counts"]) != expected_profiles or
                any(not isinstance(value["quota_targets"][profile], dict) or
                    set(value["quota_targets"][profile]) != set(_CELLS) or
                    not isinstance(value["quota_counts"][profile], dict) or
                    set(value["quota_counts"][profile]) != set(_CELLS) or
                    any(type(value["quota_targets"][profile][cell]) is not int or
                        value["quota_targets"][profile][cell] <= 0 or
                        type(value["quota_counts"][profile][cell]) is not int or
                        not 0 <= value["quota_counts"][profile][cell] <=
                            value["quota_targets"][profile][cell]
                        for cell in _CELLS) for profile in expected_profiles) or
                value["status"] == "running"):
            raise ValueError("terminal manifest partition accounting differs from replay")
        complete = all(value["quota_counts"][profile][cell] == value["quota_targets"][profile][cell]
                       for profile in expected_profiles for cell in _CELLS)
        if ((value["status"] == "not_started" and
             (value["attempts"] != 0 or any(value["quota_counts"][profile][cell] != 0
                                             for profile in expected_profiles for cell in _CELLS))) or
                (value["status"] == "complete" and not complete) or
                (value["status"] == "quota_shortfall" and complete)):
            raise ValueError("terminal manifest partition status disagrees with its quota counts")
    expected_status = {"PASS_DEVELOPMENT_PROPOSAL": ("complete", 0),
                       "NO_GO_PROPOSAL": ("scientific_no_go", 1),
                       "NOT_ASSESSED": ("not_assessed", 1),
                       "FIXTURE_PASS": ("fixture_complete", 0)}.get(report["status"])
    if expected_status is None or (terminal["status"], terminal["exit_code"]) != expected_status:
        raise ValueError("terminal manifest status disagrees with final report")
    if ((report["status"] == "FIXTURE_PASS" and fixture_policy is None) or
            (report["status"] == "PASS_DEVELOPMENT_PROPOSAL" and fixture_policy is not None) or
            (report["status"] == "NOT_ASSESSED" and
             (terminal["error"] is None or terminal["error"]["reason"] != report["reason"])) or
            (report["status"] != "NOT_ASSESSED" and terminal["error"] is not None) or
            (report["status"] in ("PASS_DEVELOPMENT_PROPOSAL", "FIXTURE_PASS") and
             any(partitions[part]["status"] != "complete" for part in ("source", "development"))) or
            (report["status"] == "NOT_ASSESSED" and terminal["exit_code"] != 1)):
        raise ValueError("terminal manifest error and report decision disagree")


def _not_assessed_report(config: ProbeConfig, source: dict, initial_path: Path,
                         closure_path: Path, guard: ResourceGuard, reason: str,
                         state: ProbeState, development_attempts: int = 0) -> dict:
    closure = strict_json_loads(closure_path.read_text(encoding="utf-8")) if closure_path.exists() else None
    audit_path = initial_path.parent / ARTIFACTS[5]
    audit = strict_json_loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    replay_path = initial_path.parent / ARTIFACTS[4]
    replay = strict_json_loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else None
    source_values = closure.get("train_projection") if closure else None
    source_status = ("pass" if closure and closure.get("status") == "PASS_SOURCE_CLOSURE" else
                     "fail" if source_values is not None else "not_assessed")
    dev_projection = (audit.get("public_projection", {}).get("development_validation_by_profile", {})
                     if audit is not None else {})
    derived_floors = _development_floors(state, dev_projection) if dev_projection else {}
    dev = {}
    for profile in config.development_profiles:
        key = _profile_key(profile)
        metric = dev_projection.get(key)
        if metric is not None and metric.get("status") == "complete":
            dev[key] = {"status": derived_floors.get(key, "not_assessed"), "values": metric}
        elif not development_attempts:
            dev[key] = {"status": "not_attempted", "values": None}
        else:
            status = "not_attempted" if metric is not None and metric.get("status") == "not_attempted" else "not_assessed"
            dev[key] = {"status": status, "values": None}
    proxy = audit.get("public_projection", {}).get("four_cohort_proxy") if audit is not None else None
    proxy_status = "not_assessed" if proxy is None else "pass" if proxy.get("pass") else "fail"
    _, source_counts = _profile_maps(config, "source", state)
    _, dev_counts = _profile_maps(config, "development", state)
    return {"report_version": "dwr-v4-probe-report-v1", "status": "NOT_ASSESSED", "reason": reason,
        "config_sha256": config.config_sha256, "source": source,
        "initial_manifest_sha256": sha256_file(initial_path),
        "source_closure_sha256": sha256_file(closure_path) if closure_path.exists() else None,
        "replay_sha256": sha256_file(replay_path) if replay_path.exists() else None,
        "audit_sha256": sha256_file(audit_path) if audit_path.exists() else None,
        "quotas": {"source_by_profile_cell": source_counts,
        "source_by_cell": {c: sum(source_counts[_profile_key(p)][c] for p in state.profiles_for("source")) for c in _CELLS},
            "source_closed": None if closure is None or closure.get("status") == "NOT_ASSESSED" else closure.get("source_quota_closed"),
            "development_validation_by_profile_cell": dev_counts,
            "development_validation_closed": (replay["quota_closed"]["development"]
                if replay is not None and replay.get("status") == "PASS_REPLAY" and
                   closure is not None and closure.get("status") == "PASS_SOURCE_CLOSURE" else
                {_profile_key(p): None for p in config.development_profiles})},
        "public_floors": {"source_train": {"status": source_status,
                "values": source_values if source_status in ("pass", "fail") else None},
            "development_validation_by_profile": dev,
            "four_cohort_proxy": {"status": proxy_status,
                                  "values": proxy if proxy_status in ("pass", "fail") else None}},
        "four_cohort_proxy": proxy, "resource_snapshot": _resource_snapshot(guard),
        "g2_status": "not_assessed", "preflight_emitted": False}


def _run_controller_body(config_path: Path, output_dir: Path, command: list[str] | None = None,
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
                    projection if source_status_stage == "complete" else None, _fixture_policy)
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
                if error is None:
                    error = {"type": type(exc).__name__, "message": exc.reason or exc.code,
                        "phase": exc.phase, "partition": exc.partition,
                        "candidate_index": exc.candidate_index, "reason": exc.code}
                if not (output_dir / ARTIFACTS[3]).exists():
                    source_closure = _write_closure(output_dir, config, initial, state, "NOT_ASSESSED", reason, guard,
                                                    fixture_policy=_fixture_policy)
                    atomic_json(output_dir / ARTIFACTS[3], source_closure)
            except ProposalFatalError as exc:
                for part in initial["partitions"].values():
                    if part["status"] == "running":
                        part["status"] = "not_assessed"
                reason = "NOT_ASSESSED_SOLVER_OR_ORACLE_ERROR"
                context = exc.context
                error = {"type": type(exc).__name__, "message": reason, "phase": "generation",
                    "partition": context.partition, "candidate_index": context.candidate_index,
                    "reason": reason}
            except (TimeoutError, MemoryError, RuntimeError) as exc:
                for part in initial["partitions"].values():
                    if part["status"] == "running":
                        part["status"] = "not_assessed"
                reason = "NOT_ASSESSED_RESOURCE_CAP" if isinstance(exc, (TimeoutError, MemoryError)) else "NOT_ASSESSED_RESOURCE_CAP"
                if error is None:
                    error = {"type": type(exc).__name__, "message": reason, "phase": "generation",
                             "partition": None, "candidate_index": None, "reason": reason}
            except Exception as exc:
                for part in initial["partitions"].values():
                    if part["status"] == "running":
                        part["status"] = "not_assessed"
                reason = "NOT_ASSESSED_IO_OR_SERIALIZATION_ERROR"
                if error is None:
                    error = {"type": type(exc).__name__, "message": reason, "phase": "generation",
                             "partition": None, "candidate_index": None, "reason": reason}
    except Exception as exc:
        reason = "NOT_ASSESSED_IO_OR_SERIALIZATION_ERROR"
        error = {"type": type(exc).__name__, "message": reason, "phase": "artifact_open",
                 "partition": None, "candidate_index": None, "reason": reason}

    if not (output_dir / ARTIFACTS[3]).exists() and outcomes_path.exists() and records_path.exists():
        source_closure = _write_closure(output_dir, config, initial, state,
                                        "NOT_ASSESSED", reason, guard, fixture_policy=_fixture_policy)
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
        if error is not None:
            report_status = "NOT_ASSESSED"
            reason = error["reason"]
        initial_sha = sha256_file(initial_path)
        closure_sha = sha256_file(output_dir / ARTIFACTS[3])
        report = _terminal_report(config, source_closure,
            {**replay_doc, "artifact_sha256": replay_ref["sha256"]},
            {**audit_doc, "artifact_sha256": audit_ref["sha256"]}, reason,
            report_status, guard, initial_sha, closure_sha, initial["source"],
            profile_status if source_status == "PASS_SOURCE_CLOSURE" and replayed.partition_closed("development") else None)
        _verify_terminal_report(report, config, initial["source"], initial_sha, closure_sha,
            source_closure, {**replay_doc, "artifact_sha256": replay_ref["sha256"]},
            {**audit_doc, "artifact_sha256": audit_ref["sha256"]}, _fixture_policy, error)
        atomic_json(output_dir / ARTIFACTS[6], report)
    except ProbeFailure as exc:
        reason, report_status = exc.code, "NOT_ASSESSED"
        if error is None:
            error = {"type": type(exc).__name__, "message": reason, "phase": "replay_audit_finalization",
                     "partition": None, "candidate_index": None, "reason": reason}
        if not (output_dir / ARTIFACTS[3]).exists():
            source_closure = _write_closure(output_dir, config, initial, state, "NOT_ASSESSED", reason, guard,
                                            fixture_policy=_fixture_policy)
            atomic_json(output_dir / ARTIFACTS[3], source_closure)
        atomic_json(output_dir / ARTIFACTS[6], _not_assessed_report(
            config, initial["source"], initial_path, output_dir / ARTIFACTS[3], guard, reason, state,
            initial["partitions"]["development"]["attempts"]))
    except (TimeoutError, MemoryError, RuntimeError) as exc:
        reason, report_status = "NOT_ASSESSED_RESOURCE_CAP", "NOT_ASSESSED"
        if error is None:
            error = {"type": type(exc).__name__, "message": reason, "phase": "replay_audit_finalization",
                     "partition": None, "candidate_index": None, "reason": reason}
        if not (output_dir / ARTIFACTS[3]).exists():
            source_closure = _write_closure(output_dir, config, initial, state, "NOT_ASSESSED", reason, guard,
                                            fixture_policy=_fixture_policy)
            atomic_json(output_dir / ARTIFACTS[3], source_closure)
        atomic_json(output_dir / ARTIFACTS[6], _not_assessed_report(
            config, initial["source"], initial_path, output_dir / ARTIFACTS[3], guard, reason, state,
            initial["partitions"]["development"]["attempts"]))
    except Exception as exc:
        reason, report_status = "NOT_ASSESSED_AUDIT_MISMATCH", "NOT_ASSESSED"
        if error is None:
            error = {"type": type(exc).__name__, "message": reason, "phase": "replay_audit_finalization",
                     "partition": None, "candidate_index": None, "reason": reason}
        if not (output_dir / ARTIFACTS[3]).exists():
            source_closure = _write_closure(output_dir, config, initial, state, "NOT_ASSESSED", reason, guard,
                                            fixture_policy=_fixture_policy)
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
        if report_status == "NOT_ASSESSED":
            report = _not_assessed_report(config, initial["source"], initial_path,
                output_dir / ARTIFACTS[3], guard, reason, state,
                initial["partitions"]["development"]["attempts"])
        report["replay_sha256"] = sha256_file(replay_path)
        report["audit_sha256"] = sha256_file(audit_path)
        atomic_json(report_path, report)
    except Exception as exc:
        report_status = "NOT_ASSESSED"
        reason = "NOT_ASSESSED_IO_OR_SERIALIZATION_ERROR"
        if error is None:
            error = {"type": type(exc).__name__, "message": reason, "phase": "failure_artifact_flush",
                     "partition": None, "candidate_index": None, "reason": reason}

    if error is not None:
        report_status = "NOT_ASSESSED"
        reason = error["reason"]

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
    terminal_replay = strict_json_loads((output_dir / ARTIFACTS[4]).read_text(encoding="utf-8"))
    terminal_report = strict_json_loads((output_dir / ARTIFACTS[6]).read_text(encoding="utf-8"))
    _verify_terminal_manifest(strict_json_loads(terminal_path.read_text(encoding="utf-8")),
        strict_json_loads(initial_path.read_text(encoding="utf-8")), config, output_dir,
        terminal_replay, terminal_report, _fixture_policy)
    try:
        guard.check()
    except (TimeoutError, MemoryError, RuntimeError):
        report_status = "NOT_ASSESSED"
        reason = "NOT_ASSESSED_RESOURCE_CAP"
        error = {"type": "ResourceCap", "message": reason,
            "phase": "post_terminal_write", "partition": None,
            "candidate_index": None, "reason": reason}
        initial["status"] = "not_assessed"
        initial["exit_code"] = 1
        initial["error"] = error
        # One bounded accounting/status flush; the guard is not reset.
        revoked = strict_json_loads((output_dir / ARTIFACTS[6]).read_text(encoding="utf-8"))
        revoked["status"] = "NOT_ASSESSED"
        revoked["reason"] = "NOT_ASSESSED_RESOURCE_CAP"
        revoked["g2_status"] = "not_assessed"
        boundary = _resource_snapshot(guard)
        revoked["resource_snapshot"] = boundary
        atomic_json(output_dir / ARTIFACTS[6], revoked)
        initial["artifacts"][ARTIFACTS[6]] = _artifact_ref(output_dir / ARTIFACTS[6])
        initial["finished_utc"] = datetime.now(timezone.utc).isoformat()
        initial["elapsed_wall_seconds"] = boundary["wall_seconds"]
        initial["cpu_seconds"] = boundary["cpu_seconds"]
        initial["peak_resident_bytes"] = boundary["peak_rss_bytes"]
        atomic_json(terminal_path, initial)
        _verify_terminal_manifest(strict_json_loads(terminal_path.read_text(encoding="utf-8")),
            strict_json_loads(initial_path.read_text(encoding="utf-8")), config, output_dir,
            terminal_replay, revoked, _fixture_policy)
    return 0 if report_status in ("PASS_DEVELOPMENT_PROPOSAL", "FIXTURE_PASS") else 1


def _emergency_failure_flush(config_path: Path, output_dir: Path, command: list[str] | None,
                             guard: ResourceGuard, fixture_policy, started_cpu: float,
                             exc: Exception) -> int:
    """Bounded best-effort terminal bundle for failures escaping the normal finalizer."""
    output = Path(output_dir).resolve()
    root = Path(__file__).resolve().parents[2]
    config = load_probe_config(Path(config_path).resolve())
    if not output.exists():
        return 1
    source = _source_identity(root, None if fixture_policy is None else fixture_policy.fixture_id)
    producer = _producer_identity(config, root)
    initial_path = output / ARTIFACTS[0]
    if initial_path.exists():
        initial = strict_json_loads(initial_path.read_text(encoding="utf-8"))
        source = initial["source"]
        producer = initial["producer"]
    else:
        initial = _initial_manifest(config, Path(config_path).resolve(), output,
            command or sys.argv[:], root, producer, source, fixture_policy)
        initial_path.write_text(stable_json(initial) + "\n", encoding="utf-8", newline="\n")
    state = ProbeState(config, capture_rows=False, _fixture_policy=fixture_policy)
    for part in ("source", "development"):
        for profile_key, cells in initial["partitions"][part]["quota_counts"].items():
            profile = tuple(profile_key.split("/"))
            if profile in state.quota_counts[part]:
                for cell in _CELLS:
                    value = cells.get(cell, 0)
                    if type(value) is int and 0 <= value <= state.quota_for(part, profile, cell):
                        state.quota_counts[part][profile][cell] = value
    for name in (ARTIFACTS[1], ARTIFACTS[2]):
        path = output / name
        if not path.exists():
            path.write_text("", encoding="utf-8", newline="\n")
    if isinstance(exc, ProbeFailure):
        reason = exc.code
        phase, partition, candidate_index = exc.phase, exc.partition, exc.candidate_index
    elif isinstance(exc, ProposalFatalError):
        reason, phase, partition, candidate_index = (
            "NOT_ASSESSED_SOLVER_OR_ORACLE_ERROR", "generation", "development", None)
    elif isinstance(exc, (TimeoutError, MemoryError, RuntimeError)):
        reason, phase, partition, candidate_index = (
            "NOT_ASSESSED_RESOURCE_CAP", "terminal_finalization", None, None)
    else:
        reason, phase, partition, candidate_index = (
            "NOT_ASSESSED_IO_OR_SERIALIZATION_ERROR", "terminal_finalization", None, None)
    error = {"type": type(exc).__name__, "message": reason, "phase": phase,
             "partition": partition, "candidate_index": candidate_index, "reason": reason}
    _ensure_failure_artifacts(output, config, source, producer, state, initial, guard)
    closure_path, replay_path, audit_path, report_path = (output / name for name in ARTIFACTS[3:7])
    closure = strict_json_loads(closure_path.read_text(encoding="utf-8"))
    replay = strict_json_loads(replay_path.read_text(encoding="utf-8"))
    audit = strict_json_loads(audit_path.read_text(encoding="utf-8"))
    if report_path.exists():
        report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    else:
        report = _not_assessed_report(config, source, initial_path, closure_path, guard,
            reason, state, initial["partitions"]["development"]["attempts"])
    report.update({"status": "NOT_ASSESSED", "reason": reason,
        "config_sha256": config.config_sha256, "source": source,
        "initial_manifest_sha256": sha256_file(initial_path),
        "source_closure_sha256": sha256_file(closure_path),
        "replay_sha256": sha256_file(replay_path), "audit_sha256": sha256_file(audit_path),
        "resource_snapshot": _resource_snapshot(guard)})
    atomic_json(report_path, report)
    initial.update({"status": "not_assessed", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "error": error, "exit_code": 1, "elapsed_wall_seconds": round(guard.elapsed, 6),
        "cpu_seconds": round(max(0.0, time.process_time() - started_cpu), 6),
        "peak_resident_bytes": guard.peak_bytes,
        "accounting_end_boundary": "post_first_terminal_write_resource_check"})
    for part in ("source", "development"):
        item = initial["partitions"][part]
        if item["status"] == "running":
            item["status"] = "not_assessed"
        item["quota_counts"] = _profile_maps(config, part, state)[1]
    initial["artifacts"] = {name: _artifact_ref(output / name) for name in ARTIFACTS[:-1]}
    terminal_path = output / ARTIFACTS[-1]
    atomic_json(terminal_path, initial)
    # The resource boundary is sampled immediately after the first durable terminal.
    snapshot = _resource_snapshot(guard)
    report["resource_snapshot"] = snapshot
    atomic_json(report_path, report)
    initial["artifacts"][ARTIFACTS[6]] = _artifact_ref(report_path)
    initial["elapsed_wall_seconds"] = round(snapshot["wall_seconds"], 6)
    initial["cpu_seconds"] = round(snapshot["cpu_seconds"], 6)
    initial["peak_resident_bytes"] = snapshot["peak_rss_bytes"]
    initial["error"] = error
    atomic_json(terminal_path, initial)
    return 1


def _run_controller(config_path: Path, output_dir: Path, command: list[str] | None = None,
                    *, guard: ResourceGuard | None = None, _fixture_policy=None,
                    _proposal_provider=None) -> int:
    output = Path(output_dir).resolve()
    existed_before = output.exists()
    guard = guard or ResourceGuard(900, 1024)
    started_cpu = time.process_time()
    try:
        return _run_controller_body(config_path, output, command, guard=guard,
            _fixture_policy=_fixture_policy, _proposal_provider=_proposal_provider)
    except Exception as exc:
        # Preserve ordinary path/config errors and resume refusal. Only failures
        # after this invocation created its unique output can be terminalized.
        if existed_before or not output.is_dir():
            raise
        try:
            return _emergency_failure_flush(config_path, output, command, guard,
                                            _fixture_policy, started_cpu, exc)
        except Exception:
            return 1


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
