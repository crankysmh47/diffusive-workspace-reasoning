"""Bounded, deterministic orchestration of the registered candidate streams."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .canonical import content_id, development_split, lineage_id, source_split, structure_signature
from .generator import PARTITIONS, PROFILES, CandidateOutcome, generate_candidate
from .run_support import LoadedConfig, ResourceGuard, atomic_json, new_manifest, sha256_file, stable_json
from .schema import serialize_model_input, serialize_task


@dataclass(frozen=True)
class GenerationLimits:
    accepted: dict[str, int]
    attempts: int
    development_per_profile: int
    wall_seconds: float
    memory_mib: int


def registered_limits(config: dict) -> GenerationLimits:
    generation = config["generation"]
    return GenerationLimits(generation["target_accepted_groups_by_partition"],
                            generation["max_attempts_per_partition"],
                            generation["target_accepted_per_development_profile"],
                            config["audit_gate"]["cap_wall_seconds"],
                            config["audit_gate"]["cap_resident_memory_mib"])


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root, check=True, capture_output=True).stdout
        status = subprocess.run(["git", "status", "--porcelain=v1", "-z", "-uall"], cwd=root, check=True, capture_output=True).stdout
        digest_input = bytearray(diff)
        entries = status.split(b"\0")
        for entry in entries:
            if len(entry) < 4:
                continue
            rel_bytes = entry[3:]
            rel = Path(os.fsdecode(rel_bytes))
            candidate = (root / rel).resolve()
            if candidate.is_file() and root.resolve() in candidate.parents:
                digest_input.extend(rel_bytes + b"\0")
                digest_input.extend(candidate.read_bytes())
        dirty = bool(status)
        digest = hashlib.sha256(digest_input).hexdigest()
        return {"commit": commit, "dirty": dirty, "dirty_diff_sha256": digest,
                "checkout_root": str(root)}
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot establish source provenance from {root}") from exc


def _outcome_record(outcome: CandidateOutcome, repeated: bool) -> dict:
    item = {"partition": outcome.partition, "candidate_index": outcome.candidate_index,
            "n": outcome.n, "profile": list(outcome.profile) if outcome.profile else None,
            "accepted": outcome.accepted and not repeated,
            "rejection_reason": "duplicate_content" if repeated else outcome.rejection_reason}
    if outcome.task is not None:
        item.update({"task": json.loads(serialize_task(outcome.task)),
                     "witness": dict(outcome.witness.assignments) if outcome.witness else None,
                     "solution": dict(outcome.solution.assignments) if outcome.solution else None,
                     "solver_stats": None if outcome.solver_result is None else {"first_solution_nodes": outcome.solver_result.stats.first_solution_nodes,
                                      "uniqueness_nodes": outcome.solver_result.stats.uniqueness_nodes,
                                      "uniqueness_solution_count_capped": outcome.solver_result.stats.uniqueness_solution_count_capped}})
    return item


def _accepted_record(outcome: CandidateOutcome, split: str) -> dict:
    task = outcome.task
    model_input = serialize_model_input(task)
    payload = model_input.encode("utf-8")
    return {"partition": outcome.partition, "candidate_index": outcome.candidate_index,
            "profile": list(outcome.profile), "task": json.loads(serialize_task(task)),
            "content_id": content_id(task), "lineage_id": lineage_id(task),
            "structure_signature": structure_signature(task), "split": split,
            "witness": dict(outcome.witness.assignments), "solution": dict(outcome.solution.assignments),
            "solver_stats": {"first_solution_nodes": outcome.solver_result.stats.first_solution_nodes,
                             "uniqueness_nodes": outcome.solver_result.stats.uniqueness_nodes,
                             "uniqueness_solution_count_capped": outcome.solver_result.stats.uniqueness_solution_count_capped},
            "model_input": model_input, "model_input_sha256": hashlib.sha256(payload).hexdigest()}


def run_dataset(config: LoadedConfig, outdir: Path, argv: list[str], *,
                limits: GenerationLimits | None = None,
                candidate_fn: Callable = generate_candidate,
                guard: ResourceGuard | None = None,
                source: dict | None = None) -> int:
    limits = limits or registered_limits(config.resolved)
    guard = guard or ResourceGuard(limits.wall_seconds, limits.memory_mib)
    guard.reset_origin()
    command_started_utc = datetime.now(timezone.utc).isoformat()
    outdir = Path(outdir).resolve()
    if outdir.exists():
        raise FileExistsError(f"output directory already exists: {outdir}")
    source = source or source_identity()
    outdir.parent.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True)
    manifest = new_manifest(config, argv, outdir, source, wall_cap_seconds=limits.wall_seconds)
    manifest["controller_started_utc"] = command_started_utc
    manifest["setup_elapsed_seconds"] = round(guard.elapsed, 6)
    manifest["limits"].update({"max_attempts_per_partition": limits.attempts,
                               "target_accepted_groups_by_partition": limits.accepted,
                               "target_accepted_per_development_profile": limits.development_per_profile,
                               "wall_seconds_global": limits.wall_seconds,
                               "resident_memory_mib": limits.memory_mib})
    manifest["partitions"] = {part: {"status": "not_attempted", "attempts": 0, "accepted": 0,
                                     "not_attempted": True} for part in PARTITIONS}
    manifest_path = outdir / "manifest.json"
    atomic_json(manifest_path, manifest)  # provenance is durable before the first candidate call
    outcomes_path, records_path = outdir / "outcomes.jsonl", outdir / "records.jsonl"
    seen: set[str] = set()
    start = guard.clock()
    cpu_start = time.process_time()
    manifest["first_work_started_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        with outcomes_path.open("w", encoding="utf-8", newline="\n") as outcomes, records_path.open("w", encoding="utf-8", newline="\n") as records:
            guard.check()
            for part in PARTITIONS:
                state = manifest["partitions"][part]
                state.update(status="running", not_attempted=False)
                counts = {profile: 0 for profile in PROFILES[part]}
                for index in range(limits.attempts):
                    if state["accepted"] >= limits.accepted[part]:
                        break
                    guard.check()
                    full = frozenset(profile for profile, count in counts.items()
                                     if part == "development" and count >= limits.development_per_profile)
                    outcome = candidate_fn(config.resolved["seed"], part, index, full_profiles=full)
                    state["attempts"] = index + 1
                    repeated = bool(outcome.accepted and content_id(outcome.task) in seen)
                    row = _outcome_record(outcome, repeated)
                    outcomes.write(stable_json(row) + "\n")
                    if outcome.accepted and not repeated:
                        seen.add(content_id(outcome.task))
                        counts[outcome.profile] += 1
                        split = source_split(content_id(outcome.task)) if part == "source" else development_split(structure_signature(outcome.task)) if part == "development" else "final_test"
                        records.write(stable_json(_accepted_record(outcome, split)) + "\n")
                        state["accepted"] += 1
                    state["profile_accepted"] = {"/".join(profile): count for profile, count in counts.items()}
                    outcomes.flush(); records.flush()
                    guard.check()
                    atomic_json(manifest_path, manifest)
                if state["accepted"] == limits.accepted[part]:
                    state["status"] = "complete"
                else:
                    raise RuntimeError(f"attempt cap reached in {part}: {state['accepted']} accepted")
                state["profile_accepted"] = {"/".join(profile): count for profile, count in counts.items()}
                atomic_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "incomplete"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc), "phase": "generation"}
        for part, state in manifest["partitions"].items():
            if state["status"] == "running":
                state["status"] = "incomplete"
            if state["status"] == "not_attempted":
                state["attempts"] = 0; state["accepted"] = 0; state["not_attempted"] = True
        manifest["exit_code"] = 1
    else:
        manifest["status"] = "complete"
        manifest["exit_code"] = 0
    manifest["generation_work_elapsed_seconds"] = round(max(0.0, guard.clock() - start), 6)
    manifest["generation_cpu_seconds"] = round(time.process_time() - cpu_start, 6)
    finalization_start = guard.clock()
    manifest["finalization_started_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["terminal_status_utc"] = manifest["finished_utc"]
    manifest["artifacts"] = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                              for p in (outcomes_path, records_path) if p.exists()}
    def record_finalization_error(exc: BaseException) -> None:
        error = {"type": type(exc).__name__, "message": str(exc), "phase": "finalization"}
        if "error" not in manifest:
            manifest["error"] = error
        else:
            manifest.setdefault("finalization_errors", []).append(error)
        manifest["status"] = "incomplete"
        manifest["exit_code"] = 1

    try:
        guard.check()
        if guard.elapsed >= limits.wall_seconds:
            raise TimeoutError("active command wall-time cap reached during finalization")
    except (MemoryError, RuntimeError, TimeoutError) as exc:
        record_finalization_error(exc)
    atomic_json(manifest_path, manifest)
    try:
        guard.check()
        if guard.elapsed >= limits.wall_seconds:
            raise TimeoutError("active command wall-time cap reached while writing terminal manifest")
    except (MemoryError, RuntimeError, TimeoutError) as exc:
        record_finalization_error(exc)
    # The accounting boundary follows the first terminal write and its resource check.
    # One bounded metadata flush persists these measured values; its own tail is excluded.
    accounting_end = guard.clock()
    manifest["generation_elapsed_seconds"] = round(max(0.0, accounting_end - guard.started), 6)
    manifest["finalization_elapsed_seconds"] = round(max(0.0, accounting_end - finalization_start), 6)
    manifest["peak_resident_bytes"] = guard.peak_bytes
    manifest["accounting_end_boundary"] = "post_first_terminal_manifest_write_resource_check"
    manifest["accounting_note"] = (
        "Elapsed time includes the first terminal manifest write and its resource check. "
        "The ultimate metadata flush tail (one bounded flush and the return tail) is not included."
    )
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["terminal_status_utc"] = manifest["finished_utc"]
    atomic_json(manifest_path, manifest)
    return manifest["exit_code"]
