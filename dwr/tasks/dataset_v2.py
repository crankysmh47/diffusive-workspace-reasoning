"""Versioned, bounded orchestration for the prospective v2 task stream."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .canonical import content_id, development_split, lineage_id, source_split, structure_signature
from .config_v2 import load_config_v2
from .generator_v2 import PARTITIONS, PROFILES, CandidateOutcome, generate_candidate
from .run_support import LoadedConfig, ResourceGuard, atomic_json, sha256_file, stable_json
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
                identities: dict | None, denominator_counted: bool) -> dict:
    row = {"outcome_version": "dwr-task-outcome-v2", "partition": outcome.partition,
           "candidate_index": outcome.candidate_index, "n": outcome.n,
           "profile": list(outcome.profile), "terminal_reason": terminal_reason,
           "oracle_unique": outcome.oracle_unique, "eligible": outcome.eligible,
           "private_evidence": _private(outcome.private_evidence), "split": split,
           "denominator_counted": denominator_counted, "identities": identities,
           "task": None if outcome.task is None else json.loads(serialize_task(outcome.task)),
           "witness": None if outcome.witness is None else dict(outcome.witness.assignments),
           "solution": None if outcome.solution is None else dict(outcome.solution.assignments),
           "solver_stats": _solver_stats(outcome)}
    return row


def accepted_record(outcome: CandidateOutcome, split: str, identities: dict) -> dict:
    task = outcome.task
    model_input = serialize_model_input(task)
    return {"partition": outcome.partition, "candidate_index": outcome.candidate_index,
            "profile": list(outcome.profile), "task": json.loads(serialize_task(task)),
            **identities, "split": split,
            "witness": dict(outcome.witness.assignments),
            "solution": dict(outcome.solution.assignments), "solver_stats": _solver_stats(outcome),
            "model_input": model_input,
            "model_input_sha256": hashlib.sha256(model_input.encode("utf-8")).hexdigest()}


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


def run_dataset_v2(config: LoadedConfig, outdir: Path, mode: str, argv: list[str], *,
                   limits: DatasetLimits | None = None, candidate_fn: Callable = generate_candidate,
                   guard: ResourceGuard | None = None, source: dict | None = None,
                   preflight_report: Path | None = None) -> int:
    if mode not in ("feasibility", "full"):
        raise ValueError("mode must be feasibility or full")
    limits = limits or registered_limits(config.resolved, mode)
    guard = guard or ResourceGuard(limits.wall_seconds, limits.memory_mib)
    guard.reset_origin()  # resource guard precedes provenance capture and setup
    started_utc = datetime.now(timezone.utc).isoformat()
    outdir = Path(outdir).resolve()
    if outdir.exists():
        raise FileExistsError(f"output directory already exists: {outdir}")
    if mode == "full":
        validate_preflight(preflight_report, config, source or source_identity())
    source = source or source_identity()
    outdir.parent.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=False)
    targets = {"source": sum(limits.source_by_profile.values()),
               "development": sum(limits.development_by_profile.values()),
               "final": sum(limits.final_by_profile.values()) if mode == "full" else 0}
    manifest = {"manifest_version": "dwr-task-run-v2", "mode": mode, "status": "running",
                "created_utc": started_utc, "command": list(argv), "cwd": str(Path.cwd()),
                "python": sys.version, "platform": sys.platform, "config_path": str(config.path),
                "config_sha256": config.sha256, "resolved_config": config.resolved,
                "seed": config.resolved["seed"], "seed_derivation": "sha256-prefix64-v2",
                "output_dir": str(outdir), "source": source,
                "limits": {"attempts_per_partition": limits.attempts_per_partition,
                           "generation_plus_audit_seconds": limits.wall_seconds,
                           "resident_memory_mib": limits.memory_mib, "cpu_workers": 1,
                           "quota_targets": targets},
                "producer": {"generator_v2_sha256": sha256_file(Path(__file__).with_name("generator_v2.py")),
                             "dataset_v2_sha256": sha256_file(Path(__file__)),
                             "replay_v2_sha256": sha256_file(Path(__file__).with_name("replay_v2.py"))},
                "generation_elapsed_seconds": 0.0,
                "partitions": {p: {"status": "not_attempted", "attempts": 0, "accepted": 0,
                                    "not_attempted": True} for p in PARTITIONS}}
    manifest_path = outdir / "manifest.json"
    atomic_json(manifest_path, manifest)
    outcomes_path, records_path = outdir / "outcomes.jsonl", outdir / "records.jsonl"
    counts = _quota_tables(limits)
    seen: set[str] = set()
    primary_error = None
    work_start = guard.clock()
    cpu_start = time.process_time()
    manifest["first_work_started_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        with outcomes_path.open("w", encoding="utf-8", newline="\n") as outcomes, records_path.open("w", encoding="utf-8", newline="\n") as records:
            guard.check()
            for part in PARTITIONS:
                state = manifest["partitions"][part]
                if part == "final" and mode == "feasibility":
                    state["status"] = "not_attempted"; state["not_attempted"] = True
                    continue
                state.update(status="running", not_attempted=False)
                for index in range(limits.attempts_per_partition):
                    if _partition_quota_complete(part, mode, counts, limits):
                        break
                    guard.check()
                    full = _full_profiles(part, mode, counts, limits)
                    outcome = candidate_fn(config.resolved["seed"], part, index, full_profiles=full)
                    if outcome.partition != part or outcome.candidate_index != index:
                        raise ValueError("candidate producer returned a mismatched partition/index")
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
                                records.write(stable_json(accepted_record(outcome, split, identities)) + "\n")
                                if (mode == "full" or
                                    (part == "source" and split == "source_train") or
                                    (part == "development" and split == "development_validation")):
                                    state["accepted"] += 1
                    row = outcome_row(outcome, reason, split=split, identities=identities,
                                      denominator_counted=counted)
                    outcomes.write(stable_json(row) + "\n")
                    state["attempts"] = index + 1
                    outcomes.flush(); records.flush(); guard.check()
                    manifest["generation_elapsed_seconds"] = round(max(0.0, guard.clock() - work_start), 6)
                    atomic_json(manifest_path, manifest)
                if not _partition_quota_complete(part, mode, counts, limits):
                    raise RuntimeError(f"attempt cap reached in {part}: accepted={state['accepted']} quota_target={targets[part]}")
                state["status"] = "complete"
                atomic_json(manifest_path, manifest)
    except BaseException as exc:
        primary_error = {"type": type(exc).__name__, "message": str(exc), "phase": "generation"}
    elapsed = round(max(0.0, guard.clock() - guard.started), 6)
    manifest["generation_elapsed_seconds"] = elapsed
    manifest["generation_cpu_seconds"] = round(time.process_time() - cpu_start, 6)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["terminal_status_utc"] = manifest["finished_utc"]
    manifest["artifacts"] = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                             for p in (outcomes_path, records_path) if p.exists()}
    manifest["error"] = primary_error
    manifest["status"] = "complete" if primary_error is None else "incomplete"
    manifest["exit_code"] = 0 if primary_error is None else 1
    for state in manifest["partitions"].values():
        if state["status"] == "running": state["status"] = "incomplete"
    finalization_errors = []
    def record_finalization_error(exc):
        item = {"type": type(exc).__name__, "message": str(exc), "phase": "finalization"}
        if manifest["error"] is None:
            manifest["error"] = item
        else:
            finalization_errors.append(item)
        manifest["status"] = "incomplete"; manifest["exit_code"] = 1
    try:
        guard.check()
    except (MemoryError, RuntimeError, TimeoutError) as exc:
        record_finalization_error(exc)
    manifest["peak_resident_bytes"] = guard.peak_bytes
    manifest["accounting_end_boundary"] = "post_first_terminal_manifest_write_resource_check"
    manifest["accounting_note"] = "Generation elapsed includes first terminal manifest write and resource check; one bounded terminal metadata flush tail is excluded."
    manifest["finalization_errors"] = finalization_errors
    atomic_json(manifest_path, manifest)
    try:
        guard.check()
    except (MemoryError, RuntimeError, TimeoutError) as exc:
        record_finalization_error(exc)
    manifest["generation_elapsed_seconds"] = round(max(0.0, guard.clock() - guard.started), 6)
    manifest["peak_resident_bytes"] = guard.peak_bytes
    manifest["finalization_errors"] = finalization_errors
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["terminal_status_utc"] = manifest["finished_utc"]
    atomic_json(manifest_path, manifest)  # one bounded accounting flush; its tail is excluded
    return manifest["exit_code"]


def validate_preflight(path: Path | None, config: LoadedConfig, source: dict) -> dict:
    """Validate the closed Task 4 feasibility handoff before full-mode setup."""
    from .run_support import strict_json_loads
    if path is None:
        raise ValueError("full mode requires --preflight-report")
    path = Path(path).resolve()
    report = strict_json_loads(path.read_text(encoding="utf-8"))
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
            report[key] != source or type(source.get("dirty")) is not bool or source["dirty"]):
            raise ValueError("preflight source must match the current clean source identity")
    generation_dir = Path(report["generation_dir"]).resolve()
    manifest_path = generation_dir / "manifest.json"
    if sha256_file(manifest_path) != report["generation_manifest_sha256"]:
        raise ValueError("preflight generation manifest identity mismatch")
    generation_manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    if (generation_manifest.get("mode") != "feasibility" or
        generation_manifest.get("config_sha256") != config.sha256 or
        generation_manifest.get("source") != source or generation_manifest.get("status") != "complete"):
        raise ValueError("preflight generation identity is invalid")
    artifacts = generation_manifest.get("artifacts")
    if (report["generation_artifacts"] != artifacts or type(artifacts) is not dict or
        set(artifacts) != {"outcomes.jsonl", "records.jsonl"}):
        raise ValueError("preflight evidence artifact identities mismatch")
    for name, identity in artifacts.items():
        file_path = generation_dir / name
        if (type(identity) is not dict or set(identity) != {"sha256", "bytes"} or
            type(identity["bytes"]) is not int or identity["bytes"] < 0 or
            type(identity["sha256"]) is not str or len(identity["sha256"]) != 64 or
            sha256_file(file_path) != identity["sha256"] or file_path.stat().st_size != identity["bytes"]):
            raise ValueError("preflight evidence artifact is missing or changed")
    expected_producer = {"generator_v2_sha256": sha256_file(Path(__file__).with_name("generator_v2.py")),
                         "dataset_v2_sha256": sha256_file(Path(__file__)),
                         "replay_v2_sha256": sha256_file(Path(__file__).with_name("replay_v2.py"))}
    if (generation_manifest.get("manifest_version") != "dwr-task-run-v2" or
        generation_manifest.get("exit_code") != 0 or generation_manifest.get("producer") != expected_producer):
        raise ValueError("preflight was generated by a different or incomplete producer")
    audit_path = Path(report["audit_artifact_path"]).resolve()
    if audit_path.parent != generation_dir or sha256_file(audit_path) != report["audit_artifact_sha256"]:
        raise ValueError("preflight audit evidence is missing or changed")
    audit = strict_json_loads(audit_path.read_text(encoding="utf-8"))
    audit_required = {"audit_version", "status", "mode", "g2_status", "config_sha256",
                      "source", "generation_manifest_sha256", "generation_artifacts"}
    if type(audit) is not dict or set(audit) != audit_required:
        raise ValueError("preflight audit artifact has unknown or missing fields")
    if (audit["audit_version"] != "dwr-task-v2-audit-v1" or audit["status"] != "complete" or
        audit["mode"] != "feasibility" or audit["g2_status"] != "PASS_FEASIBILITY" or
        audit["config_sha256"] != config.sha256 or audit["source"] != source or
        audit["generation_manifest_sha256"] != report["generation_manifest_sha256"] or
        audit["generation_artifacts"] != artifacts):
        raise ValueError("preflight audit artifact does not bind a passing matching feasibility run")
    return report
