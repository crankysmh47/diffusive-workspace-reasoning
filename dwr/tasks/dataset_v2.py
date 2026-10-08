"""Versioned, bounded orchestration for the prospective v2 task stream."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .canonical import content_id, development_split, lineage_id, source_split, structure_signature
from .config_v2 import load_config_v2
from .generator_v2 import PARTITIONS, PROFILES, CandidateOutcome, generate_candidate
from .manifest_v2 import _same_typed, expected_limits, validate_manifest_v2
from .run_support import (LoadedConfig, ResourceGuard, atomic_json, sha256_file,
                          stable_json, strict_json_loads)
from .schema import serialize_model_input, serialize_task


@dataclass(frozen=True)
class DatasetLimits:
    """Public runs use registered values; tiny values exist only for unit tests."""
    source_by_profile: dict[str, int]
    development_by_profile: dict[tuple[str, ...], int]
    final_by_profile: dict[tuple[str, ...], int]
    attempts_per_partition: int = 30000
    wall_seconds: float = 600.0
    memory_mib: int = 1024


def registered_limits(config: dict, mode: str) -> DatasetLimits:
    if mode not in ("feasibility", "full"):
        raise ValueError("mode must be feasibility or full")
    quotas = config["generation"]["mode_quotas"][mode]
    source = quotas["source_by_profile"]
    dev = quotas.get("development_validation_by_profile", quotas.get("development_by_profile", {}))
    final = quotas.get("final_by_profile", {})
    return DatasetLimits({name: source[name] for name in ("eq", "neq", "add_mod")},
                         {tuple(name.split("/")): value for name, value in dev.items()},
                         {tuple(name.split("/")): value for name, value in final.items()},
                         config["generation"]["max_attempts_per_partition"],
                         config["generation"]["mode_caps"][mode]["active_seconds_generation_plus_audit"],
                         config["generation"]["mode_caps"][mode]["resident_memory_mib"])


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root, check=True,
                              capture_output=True).stdout
        status = subprocess.run(["git", "status", "--porcelain=v1", "-z", "-uall"], cwd=root,
                                check=True, capture_output=True).stdout
        payload = bytearray(diff)
        for entry in status.split(b"\0"):
            if len(entry) < 4:
                continue
            name = entry[3:]
            path = (root / Path(os.fsdecode(name))).resolve()
            if path.is_file() and root.resolve() in path.parents:
                payload.extend(name + b"\0" + path.read_bytes())
        return {"commit": commit, "dirty": bool(status),
                "dirty_diff_sha256": hashlib.sha256(payload).hexdigest(),
                "checkout_root": str(root)}
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot establish source provenance") from exc


def _private(evidence) -> dict | None:
    if evidence is None:
        return None
    if evidence.source is not None:
        return {"oracle_unique": True, "source": {
            "removal_solution_counts": list(evidence.source.removal_solution_counts),
            "sensitive": evidence.source.sensitive}, "mixed": None}
    m = evidence.mixed
    return {"oracle_unique": True, "source": None, "mixed": {
        "kinds": list(m.kinds), "kind_removal_solution_counts": list(m.kind_removal_solution_counts),
        "factor_graph_connected": m.factor_graph_connected,
        "factor_graph_kinds": list(m.factor_graph_kinds),
        "all_present_kinds_in_graph": m.all_present_kinds_in_graph,
        "primary_eligible": m.primary_eligible, "ineligibility_reason": m.ineligibility_reason}}


def _solver_stats(outcome: CandidateOutcome) -> dict | None:
    if outcome.solver_result is None:
        return None
    stats = outcome.solver_result.stats
    return {"first_solution_nodes": stats.first_solution_nodes,
            "uniqueness_nodes": stats.uniqueness_nodes,
            "uniqueness_solution_count_capped": stats.uniqueness_solution_count_capped}


def outcome_row(outcome: CandidateOutcome, terminal_reason: str, *, split: str | None,
                identities: dict | None, denominator_counted: bool,
                expected_generator_id: str = "dwr-generator-v2") -> dict:
    from .stream_contract import resolve_stream
    contract = resolve_stream(expected_generator_id)
    if outcome.generator_id != contract.generator_id:
        raise ValueError("candidate generator identity differs from selected stream")
    row = {"outcome_version": contract.outcome_version, "partition": outcome.partition,
           "candidate_index": outcome.candidate_index, "n": outcome.n,
           "profile": list(outcome.profile), "terminal_reason": terminal_reason,
           "oracle_unique": outcome.oracle_unique, "eligible": outcome.eligible,
           "private_evidence": _private(outcome.private_evidence), "split": split,
           "denominator_counted": denominator_counted, "identities": identities,
           "task": None if outcome.task is None else json.loads(serialize_task(outcome.task)),
           "witness": None if outcome.witness is None else dict(outcome.witness.assignments),
           "solution": None if outcome.solution is None else dict(outcome.solution.assignments),
           "solver_stats": _solver_stats(outcome)}
    if expected_generator_id == "dwr-generator-v3":
        row["generator_id"] = contract.generator_id
    return row


def accepted_record(outcome: CandidateOutcome, split: str, identities: dict, *,
                    expected_generator_id: str = "dwr-generator-v2") -> dict:
    from .stream_contract import resolve_stream
    contract = resolve_stream(expected_generator_id)
    if outcome.generator_id != contract.generator_id:
        raise ValueError("candidate generator identity differs from selected stream")
    task = outcome.task
    model_input = serialize_model_input(task)
    record = {"partition": outcome.partition, "candidate_index": outcome.candidate_index,
            "profile": list(outcome.profile), "task": json.loads(serialize_task(task)),
            **identities, "split": split,
            "witness": dict(outcome.witness.assignments),
            "solution": dict(outcome.solution.assignments), "solver_stats": _solver_stats(outcome),
            "model_input": model_input,
            "model_input_sha256": hashlib.sha256(model_input.encode("utf-8")).hexdigest()}
    if contract.accepted_record_version is not None:
        record = {"record_version": contract.accepted_record_version,
                  "generator_id": contract.generator_id, **record}
    return record


def _quota_counted(mode: str, partition: str, split: str, eligible: bool) -> bool:
    if not eligible:
        return False
    if partition == "source":
        return True
    if partition == "development":
        return mode == "full" or split == "development_validation"
    return mode == "full"


def _quota_tables(limits: DatasetLimits):
    return ({(kind,): 0 for kind in limits.source_by_profile},
            {profile: 0 for profile in limits.development_by_profile},
            {profile: 0 for profile in limits.final_by_profile})


def _full_profiles(partition: str, mode: str, counts: tuple[dict, dict, dict], limits: DatasetLimits) -> frozenset:
    if partition == "source":
        return frozenset((profile,) for profile, target in limits.source_by_profile.items()
                         if counts[0][(profile,)] >= target)
    table = counts[1] if partition == "development" else counts[2]
    quotas = limits.development_by_profile if partition == "development" else limits.final_by_profile
    if mode == "feasibility" and partition == "final":
        return frozenset(PROFILES["final"])
    return frozenset(profile for profile, target in quotas.items() if table[profile] >= target)


def _partition_quota_complete(partition: str, mode: str,
                              counts: tuple[dict, dict, dict], limits: DatasetLimits) -> bool:
    if partition == "source":
        return all(counts[0][(profile,)] >= target for profile, target in limits.source_by_profile.items())
    if partition == "development":
        return all(counts[1][profile] >= target for profile, target in limits.development_by_profile.items())
    if mode == "feasibility":
        return True
    return all(counts[2][profile] >= target for profile, target in limits.final_by_profile.items())


def _split_and_ids(outcome: CandidateOutcome) -> tuple[str, dict]:
    task = outcome.task
    cid = content_id(task)
    signature = structure_signature(task)
    split = source_split(cid) if outcome.partition == "source" else (
        development_split(signature) if outcome.partition == "development" else "final_test")
    return split, {"content_id": cid, "lineage_id": lineage_id(task), "structure_signature": signature,
                   "model_input_sha256": hashlib.sha256(serialize_model_input(task).encode("utf-8")).hexdigest()}


def _failure_context(exc: BaseException, phase: str, partition: str | None,
                     candidate_index: int | None) -> dict:
    return {"type": type(exc).__name__, "message": str(exc), "phase": phase,
            "partition": partition, "candidate_index": candidate_index}


def _write_failure_sidecar(outdir: Path, primary_error: dict,
                           finalization_errors: list[dict], manifest: dict | None = None) -> bool:
    """One exclusive fallback write; never retries or replaces an existing artifact."""
    payload = {"failure_version": "dwr-task-terminal-failure-v1", "status": "incomplete",
               "primary_error": primary_error, "finalization_errors": finalization_errors,
               "manifest_snapshot": manifest}
    path = outdir / "terminal_failure.json"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(stable_json(payload) + "\n")
            stream.flush()
        return True
    except BaseException as sidecar_error:
        print(f"terminal failure sidecar could not be written: {type(sidecar_error).__name__}: {sidecar_error}",
              file=sys.stderr)
        print(f"primary run failure: {stable_json(primary_error)}", file=sys.stderr)
        for error in finalization_errors:
            print(f"finalization failure: {stable_json(error)}", file=sys.stderr)
        return False


def _new_manifest(config: LoadedConfig, outdir: Path, mode: str, argv: list[str],
                  source: dict, limits: DatasetLimits, consumed_preflight: dict | None,
                  created_utc: str, *, generator_id: str = "dwr-generator-v2") -> dict:
    from .stream_contract import require_matching_config, resolve_stream
    contract = resolve_stream(generator_id)
    require_matching_config(contract, config.resolved)
    manifest = {"manifest_version": contract.manifest_version, "mode": mode, "status": "running",
                "created_utc": created_utc, "command": list(argv), "cwd": str(Path.cwd()),
                "python": sys.version, "platform": sys.platform, "config_path": str(config.path),
                "config_sha256": config.sha256, "resolved_config": config.resolved,
                "seed": config.resolved["seed"], "seed_derivation": contract.seed_derivation,
                "output_dir": str(outdir), "source": source,
                "limits": expected_limits(mode, limits), "producer": _producer_identity(generator_id),
                "generation_elapsed_seconds": 0.0, "generation_cpu_seconds": 0.0,
                "partitions": {p: {"status": "not_attempted", "attempts": 0, "accepted": 0,
                                    "not_attempted": True} for p in PARTITIONS},
                "first_work_started_utc": created_utc, "finished_utc": created_utc,
                "terminal_status_utc": created_utc, "artifacts": {}, "error": None,
                "exit_code": 1, "peak_resident_bytes": None,
                "accounting_end_boundary": "post_first_terminal_manifest_write_resource_check",
                "accounting_note": "Elapsed and CPU include the first terminal manifest write and resource check; one bounded accounting flush tail is excluded.",
                "finalization_errors": []}
    if generator_id == "dwr-generator-v3":
        manifest.update(early_stop=None, source_prefix=None,
                        source_capacity_bound=_not_assessed_capacity_bound("source_partition_incomplete"))
        if mode == "full":
            manifest["source_capacity_bound"] = {
                "status": "NOT_APPLICABLE",
                "reason": "full_mode_uses_registered_five_cohort_gate"}
    if mode == "full":
        manifest["consumed_preflight"] = consumed_preflight
    return manifest


def _not_assessed_capacity_bound(reason: str) -> dict:
    return {"bound_version": "dwr-task-v3-source-capacity-bound-v1",
            "status": "NOT_ASSESSED", "reason": reason,
            "source_prefix_replay_sha256": None, "top_k_cells": 10,
            "minimum_source_train_mass": 0.8, "source_train_group_count": 0,
            "top_cells": [], "top_k_mass": None}


_source_closure_fixture_targets: ContextVar[dict | None] = ContextVar(
    "dwr_source_closure_fixture_targets", default=None)


@contextmanager
def _source_closure_fixture(limits: DatasetLimits):
    """Private deterministic-test seam; never changes the registered public capture quotas."""
    targets = {name: limits.source_by_profile[name] for name in ("eq", "neq", "add_mod")}
    if any(type(value) is not int or value < 0 for value in targets.values()) or sum(targets.values()) <= 0:
        raise ValueError("source closure fixture needs nonnegative exact profile quotas with positive total")
    token = _source_closure_fixture_targets.set(targets)
    try:
        yield
    finally:
        _source_closure_fixture_targets.reset(token)


def _preserve_v3_bound_diagnostic(outdir: Path, manifest: dict,
                                  prior_error: dict | None) -> None:
    path = outdir / "source_capacity_bound_provisional.json"
    if path.exists():
        return
    atomic_json(path, {"diagnostic_version": "dwr-task-v3-provisional-bound-v1",
                       "source_capacity_bound": manifest["source_capacity_bound"],
                       "early_stop": manifest["early_stop"], "prior_error": prior_error,
                       "prior_finalization_errors": manifest.get("finalization_errors", [])})


def _set_v3_operational_bound(outdir: Path, manifest: dict,
                              prior_error: dict | None) -> None:
    """Withdraw an assessed decision after operational failure and bind its replacement."""
    bound = manifest.get("source_capacity_bound", {})
    if (bound.get("status") in ("PASS_CAPACITY_BOUND", "NO_GO") or
        manifest.get("early_stop") is not None or prior_error is not None):
        _preserve_v3_bound_diagnostic(outdir, manifest, prior_error)
    manifest["source_capacity_bound"] = _not_assessed_capacity_bound("operational_error")
    manifest["early_stop"] = None
    bound_path = outdir / "source_capacity_bound.json"
    atomic_json(bound_path, manifest["source_capacity_bound"])
    identity = {"sha256": sha256_file(bound_path), "bytes": bound_path.stat().st_size}
    artifacts = manifest.setdefault("artifacts", {})
    artifacts[bound_path.name] = identity
    provisional_path = outdir / "source_capacity_bound_provisional.json"
    if provisional_path.exists():
        artifacts[provisional_path.name] = {
            "sha256": sha256_file(provisional_path), "bytes": provisional_path.stat().st_size}


def _run_dataset(config: LoadedConfig, outdir: Path, mode: str, argv: list[str], *,
                   limits: DatasetLimits | None = None, candidate_fn: Callable = generate_candidate,
                   guard: ResourceGuard | None = None, source: dict | None = None,
                   preflight_report: Path | None = None,
                   preflight_limits: DatasetLimits | None = None,
                   cpu_origin: float | None = None,
                   generator_id: str = "dwr-generator-v2") -> int:
    from .stream_contract import require_matching_config, resolve_stream
    contract = resolve_stream(generator_id)
    require_matching_config(contract, config.resolved)
    if generator_id == "dwr-generator-v3" and mode == "full":
        raise ValueError("v3 full-mode preflight consumption is not available until the independent v3 audit task")
    if cpu_origin is None:
        cpu_origin = time.process_time()
    elif (isinstance(cpu_origin, bool) or not isinstance(cpu_origin, (int, float)) or
          not math.isfinite(cpu_origin) or cpu_origin < 0 or cpu_origin > time.process_time()):
        raise ValueError("cpu_origin must be a finite process_time value from this process")
    if mode not in ("feasibility", "full"):
        raise ValueError("mode must be feasibility or full")
    limits = limits or registered_limits(config.resolved, mode)
    guard = guard or ResourceGuard(limits.wall_seconds, limits.memory_mib)
    guard.reset_origin()  # resource guard precedes provenance capture and setup
    started_utc = datetime.now(timezone.utc).isoformat()
    outdir = Path(outdir).resolve()
    if outdir.exists():
        raise FileExistsError(f"output directory already exists: {outdir}")
    source = source if source is not None else source_identity()
    consumed_preflight = None
    if mode == "full":
        consumed_preflight = validate_preflight(
            preflight_report, config, source,
            limits=preflight_limits or registered_limits(config.resolved, "feasibility"), guard=guard)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=False)
    manifest_path = outdir / "manifest.json"
    manifest = None
    try:
        manifest = _new_manifest(config, outdir, mode, argv, source, limits,
                                 consumed_preflight, started_utc, generator_id=generator_id)
        atomic_json(manifest_path, manifest)
    except BaseException as exc:
        primary_error = _failure_context(exc, "setup", None, None)
        _write_failure_sidecar(outdir, primary_error,
                               [_failure_context(exc, "finalization", None, None)], manifest)
        return 1
    targets = manifest["limits"]["quota_targets"]
    outcomes_path, records_path = outdir / "outcomes.jsonl", outdir / "records.jsonl"
    capacity_no_go = False
    counts = _quota_tables(limits)
    seen: set[str] = set()
    primary_error = None
    finalization_errors: list[dict] = []
    work_start = guard.clock()
    phase = "generation"
    current_partition: str | None = None
    current_index: int | None = None
    try:
        with outcomes_path.open("w", encoding="utf-8", newline="\n") as outcomes, records_path.open("w", encoding="utf-8", newline="\n") as records:
            guard.check()
            for part in PARTITIONS:
                state = manifest["partitions"][part]
                current_partition = part
                current_index = None
                if capacity_no_go:
                    state["status"] = "not_attempted"; state["not_attempted"] = True
                    continue
                if part == "final" and mode == "feasibility":
                    state["status"] = "not_attempted"; state["not_attempted"] = True
                    continue
                state.update(status="running", not_attempted=False)
                for index in range(limits.attempts_per_partition):
                    if _partition_quota_complete(part, mode, counts, limits):
                        break
                    current_index = index
                    guard.check()
                    full = _full_profiles(part, mode, counts, limits)
                    outcome = candidate_fn(config.resolved["seed"], part, index, full_profiles=full)
                    if outcome.partition != part or outcome.candidate_index != index:
                        raise ValueError("candidate producer returned a mismatched partition/index")
                    if outcome.generator_id != generator_id:
                        raise ValueError("candidate producer returned a cross-version outcome")
                    split = None; identities = None; counted = False
                    reason = outcome.rejection_reason or outcome.eligibility_reason
                    if outcome.oracle_unique:
                        split, identities = _split_and_ids(outcome)
                        cid = identities["content_id"]
                        if cid in seen:
                            reason = "duplicate_content"
                        else:
                            seen.add(cid)  # first occurrence enters before denominator and eligibility
                            counted = True
                            if reason is None:
                                reason = "accepted"
                                if _quota_counted(mode, part, split, True):
                                    if part == "source": counts[0][outcome.profile] += 1
                                    elif part == "development": counts[1][outcome.profile] += 1
                                    else: counts[2][outcome.profile] += 1
                                records.write(stable_json(accepted_record(
                                    outcome, split, identities,
                                    expected_generator_id=generator_id)) + "\n")
                                if (mode == "full" or
                                    (part == "source" and split == "source_train") or
                                    (part == "development" and split == "development_validation")):
                                    state["accepted"] += 1
                    row = outcome_row(outcome, reason, split=split, identities=identities,
                                      denominator_counted=counted,
                                      expected_generator_id=generator_id)
                    outcomes.write(stable_json(row) + "\n")
                    state["attempts"] = index + 1
                    outcomes.flush(); records.flush(); guard.check()
                    manifest["generation_elapsed_seconds"] = round(max(0.0, guard.clock() - work_start), 6)
                    atomic_json(manifest_path, manifest)
                if not _partition_quota_complete(part, mode, counts, limits):
                    raise RuntimeError(f"attempt cap reached in {part}: accepted={state['accepted']} quota_target={targets[part]}")
                state["status"] = "complete"
                atomic_json(manifest_path, manifest)
                if generator_id == "dwr-generator-v3" and mode == "feasibility" and part == "source":
                    phase = "source_prefix_closure"
                    current_index = None
                    outcomes.flush(); records.flush(); guard.check()
                    context_path = outdir / "source_prefix_context.json"
                    replay_path = outdir / "source_prefix_replay.json"
                    bound_path = outdir / "source_capacity_bound.json"
                    producer = manifest["producer"]
                    fixture_targets = _source_closure_fixture_targets.get()
                    if fixture_targets is None:
                        context, context_hash = capture_source_prefix_context_v3(
                            outcomes_path, records_path, context_path, config, source, producer, limits, guard)
                    else:
                        context, context_hash = _capture_source_prefix_context_v3(
                            outcomes_path, records_path, context_path, config, source, producer, limits, guard,
                            expected_targets=fixture_targets)
                    from .replay_v2 import replay_source_prefix_v3, _prefix_rows, source_support_capacity_bound
                    replay = replay_source_prefix_v3(outcomes_path, records_path, context, context_hash,
                                                     config, source, producer, limits=limits, guard=guard)
                    atomic_json(replay_path, replay); guard.check()
                    replay_hash = sha256_file(replay_path); guard.check()
                    manifest["source_prefix"] = {"context_file": context_path.name,
                        "context_sha256": context_hash, "replay_file": replay_path.name,
                        "replay_sha256": replay_hash}
                    source_records = _prefix_rows(records_path, context["source_record_prefix_bytes"],
                                                  context["source_record_prefix_sha256"], guard)
                    bound = source_support_capacity_bound(
                        (record for record in source_records if record.get("split") == "source_train"),
                        replay_hash)
                    atomic_json(bound_path, bound); guard.check()
                    bound_hash = sha256_file(bound_path); guard.check()
                    manifest["source_capacity_bound"] = bound
                    if bound["status"] == "NO_GO":
                        manifest["early_stop"] = {"status": "NO_GO",
                            "reason": "source_support_capacity_bound",
                            "source_capacity_bound_file": bound_path.name,
                            "source_capacity_bound_sha256": bound_hash}
                        primary_error = {"type": "ProposalNoGo",
                            "message": "source support capacity bound is below the registered feasibility requirement",
                            "phase": "proposal_gate", "partition": "source",
                            "candidate_index": None, "reason": "source_support_capacity_bound"}
                        capacity_no_go = True
                    phase = "generation"
    except BaseException as exc:
        if (generator_id == "dwr-generator-v3" and mode == "feasibility" and
            current_partition == "source" and phase == "source_prefix_closure"):
            bound_path = outdir / "source_capacity_bound.json"
            if bound_path.exists():
                try:
                    saved_bound = strict_json_loads(bound_path.read_text(encoding="utf-8"))
                    if type(saved_bound) is dict and saved_bound.get("status") in (
                            "PASS_CAPACITY_BOUND", "NO_GO"):
                        manifest["source_capacity_bound"] = saved_bound
                        _preserve_v3_bound_diagnostic(
                            outdir, manifest, _failure_context(exc, phase, current_partition, current_index))
                except BaseException as preserve_exc:
                    finalization_errors.append(_failure_context(
                        preserve_exc, "finalization", current_partition, current_index) |
                        {"operation": "preserve provisional source capacity report"})
            manifest["source_capacity_bound"] = _not_assessed_capacity_bound("operational_error")
            manifest["early_stop"] = None
        primary_error = _failure_context(exc, phase, current_partition, current_index)
    finalization_started = guard.clock()
    manifest["generation_elapsed_seconds"] = round(max(0.0, finalization_started - guard.started), 6)
    manifest["finalization_started_utc"] = datetime.now(timezone.utc).isoformat()
    for state in manifest["partitions"].values():
        if state["status"] == "running": state["status"] = "incomplete"

    def record_finalization_error(exc: BaseException, *, operation: str) -> None:
        nonlocal primary_error
        item = _failure_context(exc, "finalization", current_partition, current_index)
        item["operation"] = operation
        if generator_id == "dwr-generator-v3" and mode == "feasibility":
            prior_error = primary_error if primary_error is not None else manifest.get("error")
            primary_error = item
            manifest["error"] = item
            try:
                _set_v3_operational_bound(outdir, manifest, prior_error)
            except BaseException as invalidate_exc:
                finalization_errors.append(_failure_context(
                    invalidate_exc, "finalization", current_partition, current_index) |
                    {"operation": "invalidate source capacity decision"})
        elif manifest["error"] is None:
            manifest["error"] = item
        else:
            finalization_errors.append(item)
        manifest["status"] = "incomplete"
        manifest["exit_code"] = 1

    artifacts = {}
    manifest["error"] = primary_error
    if generator_id == "dwr-generator-v3" and mode == "feasibility":
        bound_path = outdir / "source_capacity_bound.json"
        if not bound_path.exists():
            try:
                atomic_json(bound_path, manifest["source_capacity_bound"])
                guard.check()
            except BaseException as exc:
                record_finalization_error(exc, operation="write source capacity report")
        elif bound_path.exists():
            try:
                saved_bound = strict_json_loads(bound_path.read_text(encoding="utf-8"))
                if saved_bound != manifest["source_capacity_bound"]:
                    _preserve_v3_bound_diagnostic(outdir, manifest, primary_error)
                    atomic_json(bound_path, manifest["source_capacity_bound"])
                    guard.check()
            except BaseException as exc:
                record_finalization_error(exc, operation="reconcile source capacity report")
        if bound_path.exists():
            for extra_path in (outdir / "source_prefix_context.json",
                               outdir / "source_prefix_replay.json", bound_path,
                               outdir / "source_capacity_bound_provisional.json"):
                try:
                    if extra_path.exists():
                        artifacts[extra_path.name] = {"sha256": sha256_file(extra_path),
                                                      "bytes": extra_path.stat().st_size}
                        guard.check()
                except BaseException as exc:
                    record_finalization_error(exc, operation=f"hash/stat {extra_path.name}")
    for artifact_path in (outcomes_path, records_path):
        try:
            if artifact_path.exists():
                artifacts[artifact_path.name] = {"sha256": sha256_file(artifact_path),
                                                 "bytes": artifact_path.stat().st_size}
        except BaseException as exc:
            record_finalization_error(exc, operation=f"hash/stat {artifact_path.name}")
    manifest["artifacts"] = artifacts
    manifest["status"] = "complete" if manifest["error"] is None and not finalization_errors else "incomplete"
    manifest["exit_code"] = 0 if manifest["status"] == "complete" else 1
    manifest["finalization_errors"] = finalization_errors
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["terminal_status_utc"] = manifest["finished_utc"]
    try:
        guard.check()
    except (MemoryError, RuntimeError, TimeoutError) as exc:
        record_finalization_error(exc, operation="pre-terminal resource check")
    terminal_writes = 0
    terminal_failed = False
    for _ in range(2):  # one terminal write and one bounded accounting flush
        manifest["generation_elapsed_seconds"] = round(max(0.0, guard.clock() - guard.started), 6)
        manifest["generation_cpu_seconds"] = round(max(0.0, time.process_time() - cpu_origin), 6)
        manifest["finalization_elapsed_seconds"] = round(max(0.0, guard.clock() - finalization_started), 6)
        manifest["peak_resident_bytes"] = guard.peak_bytes
        try:
            atomic_json(manifest_path, manifest)
            terminal_writes += 1
            if terminal_writes == 1:
                try:
                    guard.check()
                except (MemoryError, RuntimeError, TimeoutError) as exc:
                    record_finalization_error(exc, operation="post-terminal resource check")
        except BaseException as exc:
            terminal_failed = True
            record_finalization_error(exc, operation="terminal manifest replacement")
        manifest["error"] = primary_error if primary_error is not None else manifest["error"]
        if primary_error is not None and manifest["error"] is None:
            manifest["error"] = primary_error
        manifest["status"] = "incomplete" if manifest["error"] is not None or finalization_errors else "complete"
        manifest["exit_code"] = 0 if manifest["status"] == "complete" else 1
        manifest["finalization_errors"] = finalization_errors
        manifest["generation_elapsed_seconds"] = round(max(0.0, guard.clock() - guard.started), 6)
        manifest["finalization_elapsed_seconds"] = round(max(0.0, guard.clock() - finalization_started), 6)
        manifest["peak_resident_bytes"] = guard.peak_bytes
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["terminal_status_utc"] = manifest["finished_utc"]
    if terminal_failed:
        primary_error = manifest["error"] or _failure_context(
            PermissionError("terminal manifest could not be persisted"), "finalization", current_partition, current_index)
        if manifest["error"] is None:
            manifest["error"] = primary_error
        _write_failure_sidecar(outdir, primary_error, finalization_errors, manifest)
    return 0 if manifest["status"] == "complete" else 1


def _producer_identity(generator_id: str = "dwr-generator-v2") -> dict:
    if generator_id == "dwr-generator-v2":
        return {"generator_v2_sha256": sha256_file(Path(__file__).with_name("generator_v2.py")),
                "dataset_v2_sha256": sha256_file(Path(__file__)),
                "replay_v2_sha256": sha256_file(Path(__file__).with_name("replay_v2.py")),
                "manifest_v2_sha256": sha256_file(Path(__file__).with_name("manifest_v2.py"))}
    if generator_id != "dwr-generator-v3":
        raise ValueError("unknown generation producer identity")
    from .stream_contract import resolve_stream
    resolve_stream(generator_id)
    root = Path(__file__).resolve().parents[2]
    paths = {
        "generator_v2_sha256": Path(__file__).with_name("generator_v2.py"),
        "generator_v3_sha256": Path(__file__).with_name("generator_v3.py"),
        "config_v3_sha256": root / "configs/tasks/constraint_v3.json",
        "dataset_v2_sha256": Path(__file__),
        "replay_v2_sha256": Path(__file__).with_name("replay_v2.py"),
        "manifest_v2_sha256": Path(__file__).with_name("manifest_v2.py"),
        "stream_contract_sha256": Path(__file__).with_name("stream_contract.py"),
        "cli_v2_sha256": Path(__file__).with_name("cli_v2.py"),
        "cli_v3_sha256": Path(__file__).with_name("cli_v3.py"),
        "canonical_sha256": Path(__file__).with_name("canonical.py"),
        "oracle_sha256": Path(__file__).with_name("oracle.py"),
        "schema_sha256": Path(__file__).with_name("schema.py"),
        "solver_sha256": Path(__file__).with_name("solver.py"),
        "run_support_sha256": Path(__file__).with_name("run_support.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_preflight(path: Path | None, config: LoadedConfig, source: dict, *,
                       limits: DatasetLimits | None = None,
                       guard: ResourceGuard | None = None) -> dict:
    """Validate the closed Task 4 feasibility handoff before full-mode setup."""
    from .stream_contract import require_matching_config, resolve_stream
    require_matching_config(resolve_stream("dwr-generator-v2"), config.resolved)
    if path is None:
        raise ValueError("full mode requires --preflight-report")
    limits = limits or registered_limits(config.resolved, "feasibility")
    if guard is not None: guard.check()
    path = Path(path).resolve()
    report = strict_json_loads(path.read_text(encoding="utf-8"))
    if guard is not None: guard.check()
    required = {"report_version", "status", "mode", "config_sha256", "source",
                "generation_dir", "generation_manifest_sha256", "generation_artifacts",
                "audit_artifact_path", "audit_artifact_sha256", "audit_source"}
    if type(report) is not dict or set(report) != required:
        raise ValueError("preflight report has unknown or missing fields")
    if (report["report_version"] != "dwr-task-v2-feasibility-preflight-v1" or
        report["status"] != "PASS_FEASIBILITY" or report["mode"] != "feasibility"):
        raise ValueError("preflight report is not PASS_FEASIBILITY")
    if report["config_sha256"] != config.sha256:
        raise ValueError("preflight configuration identity mismatch")
    for key in ("source", "audit_source"):
        if (type(report[key]) is not dict or set(report[key]) !=
            {"commit", "dirty", "dirty_diff_sha256", "checkout_root"} or
            not _same_typed(report[key], source) or type(source.get("dirty")) is not bool or source["dirty"]):
            raise ValueError("preflight source must match the current clean source identity")
    generation_dir = Path(report["generation_dir"]).resolve()
    if (generation_dir / "terminal_failure.json").exists():
        raise ValueError("generation has a terminal failure sidecar")
    manifest_path = generation_dir / "manifest.json"
    if guard is not None: guard.check()
    wrapper_hash = sha256_file(path)
    if guard is not None: guard.check()
    manifest_hash = sha256_file(manifest_path)
    if guard is not None: guard.check()
    if manifest_hash != report["generation_manifest_sha256"]:
        raise ValueError("preflight generation manifest identity mismatch")
    if guard is not None: guard.check()
    generation_manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    if guard is not None: guard.check()
    validate_manifest_v2(generation_manifest, config_sha256=config.sha256,
                         config=config.resolved, mode="feasibility", limits=limits,
                         expected_producer=_producer_identity(), require_complete=True)
    if generation_manifest.get("source") != source:
        raise ValueError("preflight generation source identity is invalid")
    artifacts = generation_manifest.get("artifacts")
    if (not _same_typed(report["generation_artifacts"], artifacts) or type(artifacts) is not dict or
        set(artifacts) != {"outcomes.jsonl", "records.jsonl"}):
        raise ValueError("preflight evidence artifact identities mismatch")
    for name, identity in artifacts.items():
        file_path = generation_dir / name
        if guard is not None: guard.check()
        artifact_hash = sha256_file(file_path)
        if guard is not None: guard.check()
        artifact_size = file_path.stat().st_size
        if guard is not None: guard.check()
        if (type(identity) is not dict or set(identity) != {"sha256", "bytes"} or
            type(identity["bytes"]) is not int or identity["bytes"] < 0 or
            type(identity["sha256"]) is not str or len(identity["sha256"]) != 64 or
            artifact_hash != identity["sha256"] or artifact_size != identity["bytes"]):
            raise ValueError("preflight evidence artifact is missing or changed")
    audit_path = Path(report["audit_artifact_path"]).resolve()
    if guard is not None: guard.check()
    audit_hash = sha256_file(audit_path)
    if guard is not None: guard.check()
    if audit_path.parent != generation_dir or audit_hash != report["audit_artifact_sha256"]:
        raise ValueError("preflight audit evidence is missing or changed")
    if guard is not None: guard.check()
    audit = strict_json_loads(audit_path.read_text(encoding="utf-8"))
    if guard is not None: guard.check()
    audit_required = {"audit_version", "status", "mode", "g2_status", "config_sha256",
                      "source", "generation_manifest_sha256", "generation_artifacts"}
    if type(audit) is not dict or set(audit) != audit_required:
        raise ValueError("preflight audit artifact has unknown or missing fields")
    if (audit["audit_version"] != "dwr-task-v2-audit-v1" or audit["status"] != "complete" or
        audit["mode"] != "feasibility" or audit["g2_status"] != "PASS_FEASIBILITY" or
        audit["config_sha256"] != config.sha256 or not _same_typed(audit["source"], source) or
        audit["generation_manifest_sha256"] != report["generation_manifest_sha256"] or
        not _same_typed(audit["generation_artifacts"], artifacts)):
        raise ValueError("preflight audit artifact does not bind a passing matching feasibility run")
    return {"linkage_version": "dwr-task-v2-consumed-preflight-v1",
            "wrapper_path": str(path), "wrapper_sha256": wrapper_hash,
            "config_sha256": config.sha256, "source": source,
            "generation_manifest_sha256": report["generation_manifest_sha256"],
            "generation_artifacts": artifacts,
            "audit_artifact_path": str(audit_path),
            "audit_artifact_sha256": report["audit_artifact_sha256"]}


def _closed_file_prefix(path: Path, guard: ResourceGuard) -> tuple[int, str, int, dict[str, int]]:
    digest = hashlib.sha256()
    byte_count = row_count = 0
    profiles = {"eq": 0, "neq": 0, "add_mod": 0}
    with Path(path).open("rb") as stream:
        while True:
            guard.check()
            raw = stream.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                raise ValueError("source artifact ends with an incomplete JSONL row")
            row = strict_json_loads(raw[:-1].decode("utf-8", errors="strict"))
            if type(row) is not dict:
                raise ValueError("source artifact rows must be JSON objects")
            if row.get("partition") != "source":
                raise ValueError("target row appeared before source closure")
            if "terminal_reason" not in row:
                profile = row.get("profile")
                if type(profile) is not list or len(profile) != 1 or profile[0] not in profiles:
                    raise ValueError("source accepted record has an invalid profile")
                profiles[profile[0]] += 1
            digest.update(raw); byte_count += len(raw); row_count += 1
    return byte_count, digest.hexdigest(), row_count, profiles


def _capture_source_prefix_context_v3(outcomes_path: Path, records_path: Path,
                                      context_path: Path, config: LoadedConfig,
                                      source: dict, producer: dict, limits: DatasetLimits,
                                      guard: ResourceGuard, *,
                                      expected_targets: dict[str, int]) -> tuple[dict, str]:
    """Capture a flushed, feasibility-only complete source prefix and bind its hashes."""
    from .stream_contract import require_matching_config, resolve_stream
    contract = resolve_stream("dwr-generator-v3")
    require_matching_config(contract, config.resolved)
    if type(source) is not dict or set(source) != {"commit", "dirty", "dirty_diff_sha256", "checkout_root"}:
        raise ValueError("source identity is malformed")
    if type(producer) is not dict or set(producer) != set(contract.generation_producer_fields):
        raise ValueError("v3 producer fields do not match the closed stream contract")
    if any(type(value) is not str or len(value) != 64 or
           any(char not in "0123456789abcdef" for char in value) for value in producer.values()):
        raise ValueError("v3 producer hashes must be lowercase SHA-256 values")
    guard.check()
    if producer != _producer_identity("dwr-generator-v3"):
        raise ValueError("v3 source prefix producer differs from current generation code")
    guard.check()
    targets = {name: limits.source_by_profile[name] for name in ("eq", "neq", "add_mod")}
    if (type(expected_targets) is not dict or set(expected_targets) != {"eq", "neq", "add_mod"} or
        any(type(value) is not int or value < 0 for value in expected_targets.values()) or
        sum(expected_targets.values()) <= 0 or
        targets != expected_targets):
        raise ValueError("source closure fixture targets need matching nonnegative quotas with positive total")
    out_bytes, out_hash, attempts, out_profiles = _closed_file_prefix(Path(outcomes_path), guard)
    rec_bytes, rec_hash, record_count, counts = _closed_file_prefix(Path(records_path), guard)
    if out_profiles != {"eq": 0, "neq": 0, "add_mod": 0}:
        raise ValueError("source outcomes must not be counted as accepted records")
    if counts != targets:
        raise ValueError("source accepted records do not close every registered profile quota")
    context = {"context_version": "dwr-task-v3-source-prefix-context-v1",
               "generator_id": contract.generator_id, "config_version": contract.config_version,
               "config_sha256": config.sha256, "source": source, "producer": producer,
               "mode": "feasibility", "source_attempts": attempts,
               "source_outcome_prefix_bytes": out_bytes,
               "source_outcome_prefix_sha256": out_hash,
               "source_accepted_record_count": record_count,
               "source_record_prefix_bytes": rec_bytes,
               "source_record_prefix_sha256": rec_hash,
               "source_quota_targets": targets, "source_quota_counts": counts,
               "source_quota_closed": True, "development_attempts": 0, "final_attempts": 0}
    guard.check()
    atomic_json(Path(context_path), context)
    guard.check()
    context_sha256 = sha256_file(Path(context_path))
    guard.check()
    return context, context_sha256


def capture_source_prefix_context_v3(outcomes_path: Path, records_path: Path,
                                     context_path: Path, config: LoadedConfig,
                                     source: dict, producer: dict, limits: DatasetLimits,
                                     guard: ResourceGuard) -> tuple[dict, str]:
    """Capture the public, registered 334/333/333 source prefix only."""
    targets = {"eq": 334, "neq": 333, "add_mod": 333}
    if any(limits.source_by_profile[name] != targets[name] for name in targets):
        raise ValueError("source closure requires the registered complete 1,000-group quotas")
    return _capture_source_prefix_context_v3(
        outcomes_path, records_path, context_path, config, source, producer, limits, guard,
        expected_targets=targets)


def run_dataset_v2(config: LoadedConfig, outdir: Path, mode: str, argv: list[str], *,
                   limits: DatasetLimits | None = None, candidate_fn: Callable = generate_candidate,
                   guard: ResourceGuard | None = None, source: dict | None = None,
                   preflight_report: Path | None = None,
                   preflight_limits: DatasetLimits | None = None,
                   cpu_origin: float | None = None) -> int:
    return _run_dataset(config, outdir, mode, argv, limits=limits, candidate_fn=candidate_fn,
                        guard=guard, source=source, preflight_report=preflight_report,
                        preflight_limits=preflight_limits, cpu_origin=cpu_origin,
                        generator_id="dwr-generator-v2")


def run_dataset_v3(config: LoadedConfig, outdir: Path, mode: str, argv: list[str], *,
                   limits: DatasetLimits | None = None,
                   guard: ResourceGuard | None = None, source: dict | None = None,
                   preflight_report: Path | None = None,
                   cpu_origin: float | None = None) -> int:
    from .generator_v3 import generate_candidate_v3
    return _run_dataset(config, outdir, mode, argv, limits=limits,
                        candidate_fn=generate_candidate_v3, guard=guard, source=source,
                        preflight_report=preflight_report, cpu_origin=cpu_origin,
                        generator_id="dwr-generator-v3")
