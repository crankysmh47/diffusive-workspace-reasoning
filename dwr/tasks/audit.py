"""Independent, bounded integrity and structural suitability audit for task runs."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform as platform_info
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .canonical import (content_id, development_split, lineage_id, source_split,
                        structure_signature)
from .dataset import source_identity
from .generator import PARTITIONS, PROFILES, CandidateOutcome, generate_candidate
from .oracle import enumerate_assignments, enumerate_solutions
from .run_support import (ResourceGuard, atomic_json, load_config,
                          remaining_global_seconds, sha256_file, stable_json,
                          strict_json_loads)
from .schema import (ConstraintTask, RelationKind, parse_task_json,
                     serialize_model_input, serialize_task)
from .solver import solve

RECORD_KEYS = {"partition", "candidate_index", "profile", "task", "content_id",
               "lineage_id", "structure_signature", "split", "witness", "solution",
               "solver_stats", "model_input", "model_input_sha256"}
OUTCOME_BASE_KEYS = {"partition", "candidate_index", "n", "profile", "accepted", "rejection_reason"}
OUTCOME_TASK_KEYS = {"task", "witness", "solution", "solver_stats"}
SPLITS = ("source_train", "source_validation", "source_test", "development_validation",
          "development_diagnostic_test", "final_test")
COHORTS = ("source_id_test_all_groups", "development_validation_eq_neq_primary_eligible",
           "development_validation_eq_add_mod_primary_eligible",
           "development_validation_neq_add_mod_primary_eligible",
           "final_test_eq_neq_add_mod_primary_eligible")
GENERATION_MANIFEST_KEYS = {
    "manifest_version", "status", "created_utc", "command", "cwd", "source", "python", "platform",
    "config_path", "config_sha256", "resolved_config", "seed", "seed_derivation", "output_dir", "limits",
    "generation_elapsed_seconds", "partitions", "controller_started_utc", "setup_elapsed_seconds",
    "first_work_started_utc", "generation_work_elapsed_seconds", "generation_cpu_seconds",
    "finalization_started_utc", "finalization_elapsed_seconds", "finished_utc", "terminal_status_utc",
    "artifacts", "exit_code", "error", "finalization_errors", "peak_resident_bytes",
    "accounting_end_boundary", "accounting_note"}
GENERATION_REQUIRED_KEYS = {
    "manifest_version", "status", "created_utc", "command", "cwd", "source", "python", "platform",
    "config_path", "config_sha256", "resolved_config", "seed", "seed_derivation", "output_dir", "limits",
    "generation_elapsed_seconds", "partitions", "controller_started_utc", "setup_elapsed_seconds",
    "generation_work_elapsed_seconds", "generation_cpu_seconds", "finalization_started_utc",
    "finalization_elapsed_seconds", "finished_utc", "terminal_status_utc", "artifacts", "exit_code",
    "peak_resident_bytes", "accounting_end_boundary", "accounting_note"}
PARTITION_STATE_KEYS = {"status", "attempts", "accepted", "not_attempted", "profile_accepted"}
ARTIFACT_NAMES = {"outcomes.jsonl", "records.jsonl"}


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_record(task: ConstraintTask, partition: str, candidate_index: int) -> dict:
    """Build the immutable accepted-record shape used by tests and replay consumers."""
    cid = content_id(task)
    signature = structure_signature(task)
    split = source_split(cid) if partition == "source" else development_split(signature) if partition == "development" else "final_test"
    solution = enumerate_solutions(task).solutions[0]
    solver = solve(task)
    model_input = serialize_model_input(task)
    return {"partition": partition, "candidate_index": candidate_index,
            "profile": _kind_profile(task, partition),
            "task": strict_json_loads(serialize_task(task)), "content_id": cid,
            "lineage_id": lineage_id(task), "structure_signature": signature,
            "split": split, "witness": dict(solution.assignments),
            "solution": dict(solution.assignments),
            "solver_stats": {"first_solution_nodes": solver.stats.first_solution_nodes,
                             "uniqueness_nodes": solver.stats.uniqueness_nodes,
                             "uniqueness_solution_count_capped": solver.stats.uniqueness_solution_count_capped},
            "model_input": model_input, "model_input_sha256": _hash_text(model_input)}


def make_outcome_record(outcome: CandidateOutcome, repeated: bool) -> dict:
    item = {"partition": outcome.partition, "candidate_index": outcome.candidate_index,
            "n": outcome.n, "profile": list(outcome.profile) if outcome.profile else None,
            "accepted": outcome.accepted and not repeated,
            "rejection_reason": "duplicate_content" if repeated else outcome.rejection_reason}
    if outcome.task is not None:
        item.update({"task": strict_json_loads(serialize_task(outcome.task)),
                     "witness": dict(outcome.witness.assignments) if outcome.witness else None,
                     "solution": dict(outcome.solution.assignments) if outcome.solution else None,
                     "solver_stats": None if outcome.solver_result is None else {
                         "first_solution_nodes": outcome.solver_result.stats.first_solution_nodes,
                         "uniqueness_nodes": outcome.solver_result.stats.uniqueness_nodes,
                         "uniqueness_solution_count_capped": outcome.solver_result.stats.uniqueness_solution_count_capped}})
    return item


def read_jsonl(path: Path, allowed_keys: set[str], optional_keys: set[str] = frozenset()) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise ValueError(f"{path}:{number}: unterminated JSONL row")
            row = strict_json_loads(line)
            if not isinstance(row, dict) or set(row) - allowed_keys or allowed_keys - set(row) - optional_keys:
                raise ValueError(f"{path}:{number}: wrong record fields")
            rows.append(row)
    return rows


def read_records(path: Path) -> list[dict]:
    return read_jsonl(path, RECORD_KEYS)


def read_outcomes(path: Path) -> list[dict]:
    rows = read_jsonl(path, OUTCOME_BASE_KEYS | OUTCOME_TASK_KEYS, OUTCOME_TASK_KEYS)
    for row in rows:
        if set(row) & OUTCOME_TASK_KEYS and not OUTCOME_TASK_KEYS <= set(row):
            raise ValueError(f"{path}: outcome task metadata must be present as one complete group")
        _validate_outcome_scalars(row)
    return rows


def _validate_outcome_scalars(row: dict) -> None:
    if row["partition"] not in PARTITIONS: raise ValueError("outcome has invalid partition")
    _integer(row["candidate_index"], "outcome.candidate_index")
    if type(row["accepted"]) is not bool: raise ValueError("outcome.accepted must be boolean")
    if row["n"] is not None and (isinstance(row["n"], bool) or not isinstance(row["n"], int) or row["n"] not in (3, 4, 5, 6)):
        raise ValueError("outcome.n must be null or a supported integer")
    profile = row["profile"]
    if profile is not None and (not isinstance(profile, list) or any(not isinstance(x, str) for x in profile)):
        raise ValueError("outcome.profile must be null or a string list")
    if row["rejection_reason"] is not None and not isinstance(row["rejection_reason"], str):
        raise ValueError("outcome.rejection_reason must be null or a string")
    if set(row) & OUTCOME_TASK_KEYS:
        task = parse_task_json(stable_json(row["task"]))
        for name in ("witness", "solution"):
            assignment = row[name]
            if assignment is not None and (not isinstance(assignment, dict) or set(assignment) != set(task.variables) or
                    any(isinstance(v, bool) or not isinstance(v, int) for v in assignment.values())):
                raise ValueError(f"outcome.{name} must map variables to integer values")
        stats = row["solver_stats"]
        if stats is not None:
            keys = {"first_solution_nodes", "uniqueness_nodes", "uniqueness_solution_count_capped"}
            if not isinstance(stats, dict) or set(stats) != keys: raise ValueError("outcome solver statistics have wrong fields")
            for name, value in stats.items(): _integer(value, f"outcome.solver_stats.{name}")


def _kind_profile(task: ConstraintTask, partition: str) -> list[str]:
    kinds = {constraint.kind.value for constraint in task.constraints}
    matches = [profile for profile in PROFILES[partition] if set(profile) == kinds]
    if len(matches) != 1: raise ValueError("task constraint kinds do not match one registered partition profile")
    return list(matches[0])


def _assignment_dict(solution) -> dict:
    return dict(solution.assignments)


def validate_record(record: dict, config: dict) -> dict:
    """Independently reconstruct every task identity, solution and stored statistic."""
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        raise ValueError("accepted record has wrong fields")
    if record["partition"] not in PARTITIONS:
        raise ValueError("invalid partition")
    _integer(record["candidate_index"], "candidate_index")
    if not isinstance(record["task"], dict):
        raise ValueError("task must be an object")
    task = parse_task_json(stable_json(record["task"]))
    if strict_json_loads(serialize_task(task)) != record["task"]:
        raise ValueError("task is not canonically serialized")
    profile = _kind_profile(task, record["partition"])
    if record["profile"] != profile or tuple(profile) not in PROFILES[record["partition"]]:
        raise ValueError("record profile disagrees with task constraints")
    cid, signature = content_id(task), structure_signature(task)
    split = source_split(cid) if record["partition"] == "source" else development_split(signature) if record["partition"] == "development" else "final_test"
    if record["content_id"] != cid or record["lineage_id"] != lineage_id(task) or record["structure_signature"] != signature:
        raise ValueError("task identity mismatch")
    if record["split"] != split:
        raise ValueError("canonical split mismatch")
    if record["witness"] != record["solution"]:
        raise ValueError("witness and solution disagree")
    oracle = enumerate_solutions(task)
    for name in ("witness", "solution"):
        assignment = record[name]
        if (not isinstance(assignment, dict) or set(assignment) != set(task.variables) or
                any(isinstance(value, bool) or not isinstance(value, int) for value in assignment.values())):
            raise ValueError(f"{name} must map each variable to an integer value")
    if len(oracle.solutions) != 1 or record["solution"] != _assignment_dict(oracle.solutions[0]):
        raise ValueError("independently reconstructed unique solution mismatch")
    solved = solve(task)
    expected_stats = {"first_solution_nodes": solved.stats.first_solution_nodes,
                      "uniqueness_nodes": solved.stats.uniqueness_nodes,
                      "uniqueness_solution_count_capped": solved.stats.uniqueness_solution_count_capped}
    if not isinstance(record["solver_stats"], dict) or set(record["solver_stats"]) != set(expected_stats):
        raise ValueError("solver statistics have wrong fields")
    for name, value in record["solver_stats"].items():
        _integer(value, f"solver_stats.{name}")
    if record["solver_stats"] != expected_stats:
        raise ValueError("solver statistics mismatch")
    expected_input = serialize_model_input(task)
    if record["model_input"] != expected_input or record["model_input_sha256"] != _hash_text(expected_input):
        raise ValueError("model-input projection or hash mismatch")
    projected = strict_json_loads(expected_input)
    if any(key in projected for key in ("solution", "witness", "answer", "annotations", "solver_stats")):
        raise ValueError("audit-only field leaked into model input")
    return {"task": task, "partition": record["partition"], "profile": tuple(profile),
            "content_id": cid, "lineage_id": lineage_id(task), "signature": signature,
            "split": split, "solution": oracle.solutions[0], "solver_stats": expected_stats,
            "model_input_bytes": len(expected_input.encode("utf-8"))}


def audit_interactions(task: ConstraintTask, partition: str) -> dict:
    """Measure singleton removal or mixed all-kind necessity and factor connectivity."""
    if partition not in PARTITIONS: raise ValueError("invalid partition for interaction audit")
    base = len(enumerate_solutions(task).solutions)
    if base != 1:
        raise ValueError("interaction audit requires a unique base task")
    kinds = tuple(sorted({constraint.kind for constraint in task.constraints}, key=lambda kind: kind.value))
    if partition == "source":
        counts = [len(enumerate_assignments(task.variables, task.givens,
                    tuple(c for c in task.constraints if c is not removed)).solutions)
                  for removed in task.constraints]
        return {"removal_solution_counts": counts,
                "sensitive": any(count >= 2 for count in counts),
                "removals": [{"constraint": i, "solution_count": count}
                             for i, count in enumerate(counts)]}
    removals = []
    for kind in kinds:
        retained = tuple(c for c in task.constraints if c.kind is not kind)
        count = len(enumerate_assignments(task.variables, task.givens, retained).solutions)
        removals.append({"kind": kind.value, "solution_count": count, "necessary": count >= 2})
    unassigned = set(task.variables) - {given.variable for given in task.givens}
    factors = [(index, constraint) for index, constraint in enumerate(task.constraints)
               if unassigned.intersection(constraint.variables)]
    adjacency: dict[tuple[str, Any], set[tuple[str, Any]]] = defaultdict(set)
    for index, constraint in factors:
        factor = ("factor", index)
        for variable in unassigned.intersection(constraint.variables):
            vnode = ("variable", variable)
            adjacency[factor].add(vnode); adjacency[vnode].add(factor)
    connected = bool(unassigned) and all(("variable", v) in adjacency for v in unassigned)
    if connected:
        reached, stack = set(), [("variable", next(iter(unassigned)))]
        while stack:
            node = stack.pop()
            if node in reached: continue
            reached.add(node); stack.extend(adjacency[node] - reached)
        connected = all(("variable", v) in reached for v in unassigned)
    graph_kinds = {constraint.kind.value for _, constraint in factors}
    return {"removal_solution_counts": [x["solution_count"] for x in removals],
            "removals": removals,
            "factor_graph_connected": connected,
            "factor_graph_kinds": sorted(graph_kinds),
            "all_present_kinds_in_graph": set(k.value for k in kinds) <= graph_kinds,
            "sensitive": all(x["necessary"] for x in removals) and connected and set(k.value for k in kinds) <= graph_kinds}


def replay_outcomes(rows: list[dict], seed: int, full_profiles_by_partition: dict,
                    seen_content_by_partition: dict, development_profile_target: int = 600,
                    check=None, on_row=None) -> list[dict]:
    """Replay a bounded recorded candidate prefix, preserving quota and duplicate semantics."""
    expected_rows = []
    accepted_by_profile: dict[str, Counter] = {p: Counter() for p in PARTITIONS}
    seen: set[str] = set()
    next_index = {p: 0 for p in PARTITIONS}
    partition_order = {p: index for index, p in enumerate(PARTITIONS)}
    last_partition_index = -1
    quota = full_profiles_by_partition
    for row in rows:
        if check is not None: check()
        if not isinstance(row, dict) or set(row) - (OUTCOME_BASE_KEYS | OUTCOME_TASK_KEYS) or not OUTCOME_BASE_KEYS <= set(row):
            raise ValueError("outcome has wrong fields")
        if set(row) & OUTCOME_TASK_KEYS and not OUTCOME_TASK_KEYS <= set(row):
            raise ValueError("outcome task metadata must be complete")
        _validate_outcome_scalars(row)
        part = row.get("partition")
        if part in partition_order and partition_order[part] < last_partition_index:
            raise ValueError("outcomes are not in registered partition order")
        if part in partition_order: last_partition_index = partition_order[part]
        if part not in PARTITIONS or row.get("candidate_index") != next_index[part]:
            raise ValueError("outcome candidate indices must be contiguous from zero")
        next_index[part] += 1
        full = frozenset(tuple(x) for x in quota.get(part, set())
                         if accepted_by_profile[part][tuple(x)] >= development_profile_target) if part == "development" else frozenset()
        outcome = generate_candidate(seed, part, row["candidate_index"], full_profiles=full)
        repeated = bool(outcome.accepted and content_id(outcome.task) in seen)
        expected = make_outcome_record(outcome, repeated)
        if row != expected:
            raise ValueError(f"candidate replay mismatch at {part}:{row['candidate_index']}")
        expected_rows.append(expected)
        if on_row is not None: on_row(expected, outcome, repeated)
        if check is not None: check()
        if outcome.accepted and not repeated:
            seen.add(content_id(outcome.task))
            accepted_by_profile[part][tuple(outcome.profile)] += 1
    return expected_rows


def _cell(record: dict) -> tuple:
    task = record["task"] if isinstance(record["task"], ConstraintTask) else parse_task_json(stable_json(record["task"]))
    n, givens, constraints = len(task.variables), len(task.givens), len(task.constraints)
    size = record["model_input_bytes"]
    prompt_bin = 0 if size <= 511 else 1 if size <= 639 else 2 if size <= 767 else 3 if size <= 1023 else 4
    nodes = record["solver_stats"]["first_solution_nodes"]
    node_bin = 0 if nodes == 1 else 1 if nodes <= 3 else 2 if nodes <= 7 else 3 if nodes <= 15 else 4 if nodes <= 31 else 5 if nodes <= 63 else 6
    return n, givens, constraints, prompt_bin, node_bin


def common_support(cohorts: dict[str, list[dict]], min_per_cell: int = 10,
                   min_fraction: float = 0.8, *,
                   cohort_names: tuple[str, ...] | None = None) -> dict:
    """Compute exact joint support; preserve the registered v1 cohorts by default."""
    names = COHORTS if cohort_names is None else cohort_names
    if (not isinstance(names, tuple) or not names or
            any(type(name) is not str or not name for name in names) or
            len(set(names)) != len(names)):
        raise ValueError("cohort_names must be a nonempty tuple of distinct names")
    selected = {name: list(cohorts.get(name, [])) for name in names}
    for name in names[1:]:
        selected[name] = [row for row in selected[name] if row.get("primary_eligible") is True]
    counts = {name: Counter(_cell(r) for r in selected[name]) for name in names}
    denominators = {name: len(selected[name]) for name in names}
    nonempty = {name: denominators[name] > 0 for name in names}
    common_cells = sorted(set.intersection(*(set(counts[name]) for name in names))) if all(nonempty.values()) else []
    common_cells = [cell for cell in common_cells if all(counts[name][cell] >= min_per_cell for name in names)]
    result = {}
    passed = True
    for name in names:
        n = denominators[name]
        in_common = sum(counts[name][cell] for cell in common_cells)
        fraction = in_common / n if n else 0.0
        ok = n > 0 and fraction >= min_fraction
        passed &= ok
        result[name] = {"count": n, "nonempty": n > 0,
                        "groups_in_common_cells": in_common,
                        "fraction_in_common_cells": fraction, "pass": ok,
                        "cells": {"|".join(map(str, cell)): counts[name][cell] for cell in sorted(counts[name])}}
    return {"pass": passed, "status": "measured_pass" if passed else "measured_fail",
            "common_cells": [list(cell) for cell in common_cells], "cohorts": result,
            "thresholds": {"min_groups_per_cell_per_cohort": min_per_cell,
                           "min_group_fraction_per_cohort_in_common_cells": min_fraction}}


def _floor(failures: list[str], condition: bool, name: str) -> None:
    if not condition: failures.append(name)


def assess_g2(metrics: dict, config: dict) -> dict:
    """Combine independently measured structural and provenance gates."""
    failures = list(metrics.get("floor_failures", []))
    _floor(failures, metrics.get("generation_complete") is True, "generation_not_complete")
    _floor(failures, metrics.get("replay_pass") is True, "replay_failed_or_not_assessed")
    _floor(failures, metrics.get("integrity_pass") is True, "integrity_failed_or_not_assessed")
    _floor(failures, metrics.get("structural_pass") is True, "structural_floors_failed")
    _floor(failures, metrics.get("common_support", {}).get("pass") is True, "common_support_failed")
    return {"status": "measured_pass" if not failures else "measured_fail",
            "pass": not failures, "floor_failures": sorted(set(failures)),
            "claim": "structural G2 suitability only" if not failures else "no-go for model comparison"}


def compute_metrics(records: list[dict], config: dict, check=None, *,
                    validated_infos: list[dict] | None = None,
                    interaction_results: list[dict] | None = None) -> dict:
    """Compute interaction, coverage, variation, strata and common-support floors."""
    gate = config["audit_gate"]
    enriched = []
    if (validated_infos is None) != (interaction_results is None):
        raise ValueError("validated infos and interaction results must be supplied together")
    if validated_infos is not None and (len(validated_infos) != len(records) or len(interaction_results) != len(records)):
        raise ValueError("precomputed audit results do not align with records")
    for index, raw in enumerate(records):
        if check is not None: check()
        info = validate_record(raw, config) if validated_infos is None else validated_infos[index]
        interaction = audit_interactions(info["task"], info["partition"]) if interaction_results is None else interaction_results[index]
        enriched.append({**raw, **info, "interaction": interaction,
                         "primary_eligible": interaction["sensitive"] if info["partition"] != "source" else False})
    validate_corpus_closure(enriched)
    failures: list[str] = []
    partitions = {}
    source_sensitive = [r for r in enriched if r["partition"] == "source" and r["interaction"]["sensitive"]]
    for part in PARTITIONS:
        rows = [r for r in enriched if r["partition"] == part]
        sigs = {r["signature"] for r in rows}
        value_counts = Counter(value for r in rows for _, value in r["solution"].assignments)
        total_values = sum(value_counts.values())
        value_fractions = {str(value): value_counts[value] / total_values if total_values else 0.0 for value in range(3)}
        clue_counts = sorted({len(r["task"].givens) for r in rows})
        profiles = {}
        for profile in PROFILES[part]:
            pr = [r for r in rows if tuple(r["profile"]) == profile]
            eligible = [r for r in pr if r["primary_eligible"]]
            profiles["/".join(profile)] = {"groups": len(pr), "eligible_groups": len(eligible),
                "eligible_fraction": len(eligible) / len(pr) if pr else 0.0,
                "eligible_signatures": len({r["signature"] for r in eligible}),
                "signatures": len({r["signature"] for r in pr}),
                "min_eligible_per_signature": min(Counter(r["signature"] for r in eligible).values(), default=0),
                "eligible_signatures_missing_eligible_example": sorted({r["signature"] for r in pr} - {r["signature"] for r in eligible})}
        split_stats = {}
        for split in SPLITS:
            sr = [r for r in rows if r["split"] == split]
            strata = {}
            fields = {"n": [len(r["task"].variables) for r in sr],
                      "given_count": [len(r["task"].givens) for r in sr],
                      "constraint_count": [len(r["task"].constraints) for r in sr],
                      "model_input_utf8_bytes": [r["model_input_bytes"] for r in sr],
                      "first_solution_nodes": [r["solver_stats"]["first_solution_nodes"] for r in sr]}
            for key, vals in fields.items():
                hist = Counter(vals)
                strata[key] = {"histogram": {str(k): hist[k] for k in sorted(hist)},
                               "min": min(vals) if vals else None, "max": max(vals) if vals else None}
            split_stats[split] = {"groups": len(sr), "signatures": len({r["signature"] for r in sr}), "strata": strata}
        partitions[part] = {"groups": len(rows), "signatures": len(sigs),
                            "solution_value_fractions": value_fractions,
                            "distinct_given_counts": clue_counts, "profiles": profiles,
                            "splits": split_stats,
                            "source_train_primitive_coverage": source_train_primitive_coverage(rows) if part == "source" else None,
                            "source_composition_coverage": source_train_composition_absence(rows) if part == "source" else None,
                            "sensitive_groups": sum(r["interaction"]["sensitive"] for r in rows)}
        _floor(failures, len(rows) >= gate["min_accepted_groups_per_partition"], f"{part}_min_groups")
        _floor(failures, len(sigs) >= gate["min_structure_signatures_per_partition"], f"{part}_min_signatures")
        _floor(failures, len(clue_counts) >= gate["min_distinct_given_counts_per_partition"], f"{part}_given_count_variation")
        _floor(failures, all(value_fractions[str(v)] >= gate["min_solution_value_fraction_each_partition"] for v in range(3)), f"{part}_solution_value_balance")
        for split, minimum in gate["source_id_groups_min"].items() if part == "source" else ():
            _floor(failures, split_stats[f"source_{split}"]["groups"] >= minimum, f"source_{split}_groups")
        _floor(failures, len(profiles) == len(PROFILES[part]) and all(r["profile"] for r in rows), f"{part}_composition")
        if part == "source":
            train = [r for r in rows if r["split"] == "source_train"]
            primitive_coverage = source_train_primitive_coverage(rows)
            train_kinds = set(primitive_coverage["source_train_kinds"])
            _floor(failures, primitive_coverage["pass"], "source_train_primitive_coverage")
            _floor(failures, source_train_composition_absence(rows)["heldout_compositions_absent"], "heldout_composition_absent_from_source_train")
            _floor(failures, len({r["signature"] for r in train}) >= gate["source_train_min_structure_signatures"], "source_train_signatures")
            frac = len(source_sensitive) / len(rows) if rows else 0.0
            _floor(failures, frac >= gate["min_source_constraint_sensitive_fraction"], "source_sensitive_fraction")
            per_signature = Counter(r["signature"] for r in source_sensitive)
            _floor(failures, all(per_signature[sig] >= gate["min_source_sensitive_examples_per_signature"] for sig in sigs), "source_sensitive_per_signature")
        if part in ("development", "final"):
            total_eligible = sum(r["primary_eligible"] for r in rows)
            fraction = total_eligible / len(rows) if rows else 0.0
            _floor(failures, fraction >= gate["min_primary_eligible_fraction_per_mixed_partition"], f"{part}_eligible_fraction")
            for key, p in profiles.items():
                _floor(failures, p["eligible_groups"] >= gate["min_primary_eligible_groups_per_mixed_profile"], f"{part}_{key}_eligible_groups")
                _floor(failures, p["eligible_signatures"] >= gate["min_primary_eligible_structure_signatures_per_mixed_profile"], f"{part}_{key}_eligible_signatures")
                _floor(failures, not p["eligible_signatures_missing_eligible_example"], f"{part}_{key}_per_signature_eligibility")
                _floor(failures, p["min_eligible_per_signature"] >= gate["min_primary_eligible_examples_per_heldout_signature"], f"{part}_{key}_eligible_per_signature_floor")
            if part == "development":
                for split, total_floor in gate["development_target_min_groups"].items():
                    _floor(failures, split_stats[split]["groups"] >= total_floor, f"{split}_groups")
                # Profile quotas are checked within each registered split as specified.
                split_config = config["composition_partition"]["development_target_split"]
                for profile in PROFILES[part]:
                    name = "/".join(profile)
                    for split, floor in (("development_validation", split_config["min_validation_groups_per_profile"]),
                                         ("development_diagnostic_test", split_config["min_diagnostic_test_groups_per_profile"])):
                        count = sum(r["split"] == split and tuple(r["profile"]) == profile for r in rows)
                        _floor(failures, count >= floor, f"{split}_{name}_groups")
    cohorts = {name: [] for name in COHORTS}
    for row in enriched:
        if row["split"] == "source_test": cohorts[COHORTS[0]].append(row)
        if row["split"] == "development_validation" and row["primary_eligible"]:
            cohort = {("eq", "neq"): COHORTS[1], ("eq", "add_mod"): COHORTS[2], ("neq", "add_mod"): COHORTS[3]}.get(tuple(row["profile"]))
            if cohort: cohorts[cohort].append(row)
        if row["split"] == "final_test" and row["primary_eligible"]: cohorts[COHORTS[4]].append(row)
    support = common_support(cohorts, gate["common_support"]["min_groups_per_cell_per_cohort"], gate["common_support"]["min_group_fraction_per_cohort_in_common_cells"])
    _floor(failures, support["pass"], "common_support")
    return {"partitions": partitions, "interactions": [
                {"partition": r["partition"], "content_id": r["content_id"], "signature": r["signature"], **r["interaction"]}
                for r in enriched], "common_support": support,
            "floor_failures": sorted(set(failures)), "structural_pass": not failures}


def _same_typed_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right): return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_same_typed_value(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_typed_value(a, b) for a, b in zip(left, right))
    return left == right


def _registered_limits(config: dict) -> dict:
    return {"wall_seconds_global": config["audit_gate"]["cap_wall_seconds"],
            "resident_memory_mib": config["audit_gate"]["cap_resident_memory_mib"],
            "cpu_workers": config["audit_gate"]["cpu_workers"],
            "max_attempts_per_partition": config["generation"]["max_attempts_per_partition"],
            "target_accepted_groups_by_partition": config["generation"]["target_accepted_groups_by_partition"],
            "target_accepted_per_development_profile": config["generation"]["target_accepted_per_development_profile"]}


def _finite_nonnegative(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _validate_source_identity(source: Any, name: str) -> None:
    if not isinstance(source, dict) or set(source) != {"commit", "dirty", "dirty_diff_sha256", "checkout_root"}:
        raise ValueError(f"{name} source identity has wrong fields")
    if not isinstance(source["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", source["commit"]):
        raise ValueError(f"{name} commit must be a full lowercase Git SHA-1")
    if type(source["dirty"]) is not bool:
        raise ValueError(f"{name}.dirty must be boolean")
    if not isinstance(source["dirty_diff_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", source["dirty_diff_sha256"]):
        raise ValueError(f"{name} dirty diff identity must be SHA-256")
    if not isinstance(source["checkout_root"], str) or not source["checkout_root"]:
        raise ValueError(f"{name} checkout root is required")


def _validate_producer_nested(manifest: dict, config: dict | None, data_dir: Path,
                              expected_limits: dict | None = None) -> None:
    limits = manifest["limits"]
    limit_keys = {"wall_seconds_global", "resident_memory_mib", "cpu_workers",
                  "max_attempts_per_partition", "target_accepted_groups_by_partition",
                  "target_accepted_per_development_profile"}
    if not isinstance(limits, dict) or set(limits) != limit_keys:
        raise ValueError("generation limits have wrong fields")
    for key in ("wall_seconds_global",): _finite_nonnegative(limits[key], f"limits.{key}")
    for key in ("resident_memory_mib", "cpu_workers", "max_attempts_per_partition", "target_accepted_per_development_profile"):
        _integer(limits[key], f"limits.{key}", 1)
    targets = limits["target_accepted_groups_by_partition"]
    if not isinstance(targets, dict) or set(targets) != set(PARTITIONS):
        raise ValueError("partition target limits have wrong fields")
    for part, value in targets.items(): _integer(value, f"limits.target_accepted_groups_by_partition.{part}")
    partitions = manifest["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != set(PARTITIONS):
        raise ValueError("partition manifest has wrong fields")
    for part, state in partitions.items():
        if not isinstance(state, dict) or set(state) - PARTITION_STATE_KEYS:
            raise ValueError(f"partition {part} state has wrong fields")
        required = {"status", "attempts", "accepted", "not_attempted"}
        if not required <= set(state): raise ValueError(f"partition {part} state is missing fields")
        if state["status"] not in ("not_attempted", "running", "incomplete", "complete"):
            raise ValueError(f"partition {part} has invalid status")
        attempts, accepted = _integer(state["attempts"], f"partition.{part}.attempts"), _integer(state["accepted"], f"partition.{part}.accepted")
        if type(state["not_attempted"]) is not bool or state["not_attempted"] != (state["status"] == "not_attempted"):
            raise ValueError(f"partition {part} not_attempted flag disagrees with status")
        if attempts > limits["max_attempts_per_partition"] or accepted > targets[part] or accepted > attempts:
            raise ValueError(f"partition {part} counters exceed registered limits")
        if state["status"] == "not_attempted" and (attempts != 0 or accepted != 0):
            raise ValueError(f"not-attempted partition {part} has nonzero counters")
        if state["status"] == "complete" and accepted != targets[part]:
            raise ValueError(f"complete partition {part} misses accepted target")
        profile_accepted = state.get("profile_accepted")
        if profile_accepted is not None:
            profile_names = {"/".join(p) for p in PROFILES[part]}
            if not isinstance(profile_accepted, dict) or set(profile_accepted) != profile_names:
                raise ValueError(f"partition {part} profile counts have wrong fields")
            counts = [_integer(value, f"partition.{part}.profile_accepted.{name}") for name, value in profile_accepted.items()]
            if sum(counts) != accepted: raise ValueError(f"partition {part} profile counts do not sum to accepted")
            if part == "development" and any(value > limits["target_accepted_per_development_profile"] for value in profile_accepted.values()):
                raise ValueError("development profile count exceeds quota")
        elif accepted:
            raise ValueError(f"partition {part} accepted records lack profile counts")
    if config is not None:
        scientific_limits = expected_limits or _registered_limits(config)
        if not _same_typed_value(limits, scientific_limits):
            raise ValueError("producer limits differ from registered scientific limits")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) - ARTIFACT_NAMES:
        raise ValueError("generation artifacts contain unknown names")
    has_work = any(state["attempts"] > 0 for state in partitions.values())
    for name in ARTIFACT_NAMES:
        path = data_dir / name
        identity = artifacts.get(name)
        if identity is None:
            if path.exists(): raise ValueError(f"unhashed producer artifact exists: {name}")
            if manifest["status"] == "complete" or has_work:
                raise ValueError(f"producer artifact identity missing: {name}")
            continue
        if not isinstance(identity, dict) or set(identity) != {"sha256", "bytes"}:
            raise ValueError(f"artifact {name} identity has wrong fields")
        if (not isinstance(identity["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"])):
            raise ValueError(f"artifact {name} hash must be SHA-256")
        _integer(identity["bytes"], f"artifact.{name}.bytes")
        if not path.is_file() or identity["sha256"] != sha256_file(path) or identity["bytes"] != path.stat().st_size:
            raise ValueError(f"artifact identity mismatch: {name}")
    if "error" in manifest:
        error = manifest["error"]
        if not isinstance(error, dict) or set(error) != {"type", "message", "phase"} or any(not isinstance(error[k], str) for k in error):
            raise ValueError("generation error object has wrong fields or types")
        if error["phase"] not in ("generation", "finalization"): raise ValueError("invalid generation error phase")
    if "finalization_errors" in manifest:
        errors = manifest["finalization_errors"]
        if not isinstance(errors, list) or any(not isinstance(e, dict) or set(e) != {"type", "message", "phase"} or
                e["phase"] != "finalization" or any(not isinstance(e[k], str) for k in e) for e in errors):
            raise ValueError("finalization errors must be well-formed finalization records")


def _verify_manifest(data_dir: Path, config=None, expected_limits: dict | None = None) -> tuple[dict, list[dict], list[dict]]:
    manifest_path = data_dir / "manifest.json"
    manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("manifest_version") != "dwr-task-run-v1":
        raise ValueError("unknown generation manifest")
    if set(manifest) - GENERATION_MANIFEST_KEYS:
        raise ValueError(f"generation manifest has unknown fields: {sorted(set(manifest) - GENERATION_MANIFEST_KEYS)}")
    missing = GENERATION_REQUIRED_KEYS - set(manifest)
    if missing: raise ValueError(f"generation manifest missing fields: {sorted(missing)}")
    if manifest["status"] not in ("complete", "incomplete"):
        raise ValueError("generation manifest must be terminal before audit")
    if type(manifest["exit_code"]) is not int or manifest["exit_code"] != (0 if manifest["status"] == "complete" else 1):
        raise ValueError("generation exit code disagrees with terminal status")
    for key in ("created_utc", "cwd", "python", "platform", "config_path", "output_dir",
                "controller_started_utc", "finalization_started_utc", "finished_utc",
                "terminal_status_utc", "accounting_end_boundary", "accounting_note"):
        if not isinstance(manifest[key], str) or not manifest[key]: raise ValueError(f"manifest.{key} must be a nonempty string")
    if not isinstance(manifest["command"], list) or not manifest["command"] or any(not isinstance(x, str) for x in manifest["command"]):
        raise ValueError("manifest.command must be a nonempty string list")
    if not isinstance(manifest["python"], str) or not isinstance(manifest["platform"], str):
        raise ValueError("manifest runtime identity must be strings")
    _finite_nonnegative(manifest["generation_elapsed_seconds"], "generation_elapsed_seconds")
    _finite_nonnegative(manifest["generation_work_elapsed_seconds"], "generation_work_elapsed_seconds")
    _finite_nonnegative(manifest["generation_cpu_seconds"], "generation_cpu_seconds")
    _finite_nonnegative(manifest["setup_elapsed_seconds"], "setup_elapsed_seconds")
    _finite_nonnegative(manifest["finalization_elapsed_seconds"], "finalization_elapsed_seconds")
    peak = manifest["peak_resident_bytes"]
    if peak is not None: _integer(peak, "peak_resident_bytes")
    if not isinstance(manifest["config_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", manifest["config_sha256"]):
        raise ValueError("config_sha256 must be SHA-256")
    if not isinstance(manifest["seed_derivation"], str): raise ValueError("seed_derivation must be a string")
    _integer(manifest["seed"], "seed")
    if manifest["seed"] >= 2**64: raise ValueError("seed is outside uint64 range")
    _validate_source_identity(manifest["source"], "generation")
    if config is not None:
        if not _same_typed_value(manifest["resolved_config"], config.resolved):
            raise ValueError("generation manifest configuration identity mismatch")
        if manifest["config_sha256"] != config.sha256:
            raise ValueError("generation manifest config hash mismatch")
        if manifest["config_path"] != str(config.path):
            raise ValueError("generation manifest config path mismatch")
        if manifest["output_dir"] != str(data_dir.resolve()):
            raise ValueError("generation manifest output path mismatch")
        if manifest["seed"] != config.resolved["seed"]:
            raise ValueError("generation manifest seed mismatch")
        if manifest["seed_derivation"] != config.resolved["seed_derivation"]:
            raise ValueError("generation manifest seed derivation mismatch")
        expected_limits = expected_limits or _registered_limits(config.resolved)
    if expected_limits is not None:
        if not _same_typed_value(manifest["limits"], expected_limits):
            raise ValueError("generation manifest limits differ from registered/test expected limits")
    _validate_producer_nested(manifest, config.resolved if config is not None else None, data_dir, expected_limits)
    outcomes_path, records_path = data_dir / "outcomes.jsonl", data_dir / "records.jsonl"
    outcomes = read_outcomes(outcomes_path) if outcomes_path.exists() else []
    records = read_records(records_path) if records_path.exists() else []
    return manifest, outcomes, records


def validate_complete_generation(manifest: dict, outcomes: list[dict], records: list[dict], config: dict) -> None:
    """Reject a COMPLETE label that disagrees with the actual partition/artifact counts."""
    if manifest.get("status") != "complete" or manifest.get("exit_code") != 0:
        raise ValueError("generation was not COMPLETE")
    if len(outcomes) != sum(_integer(manifest.get("partitions", {}).get(p, {}).get("attempts"), f"{p}.attempts") for p in PARTITIONS):
        raise ValueError("outcome count disagrees with manifest attempts")
    if len(outcomes) > 3 * config["generation"]["max_attempts_per_partition"]:
        raise ValueError("outcome count exceeds registered attempt budget")
    outcome_counts = Counter(row["partition"] for row in outcomes)
    by_partition = Counter(row["partition"] for row in records)
    for part in PARTITIONS:
        state = manifest["partitions"][part]
        if not isinstance(state, dict): raise ValueError(f"partition {part} state must be an object")
        target = config["generation"]["target_accepted_groups_by_partition"][part]
        if state.get("status") != "complete" or state.get("not_attempted") is not False:
            raise ValueError(f"partition {part} is not complete")
        attempts = _integer(state.get("attempts"), f"{part}.attempts")
        accepted = _integer(state.get("accepted"), f"{part}.accepted")
        if attempts != outcome_counts[part]:
            raise ValueError(f"partition {part} attempt count disagrees with outcomes")
        if attempts > config["generation"]["max_attempts_per_partition"]:
            raise ValueError(f"partition {part} exceeds attempt cap")
        if accepted != target or by_partition[part] != target:
            raise ValueError(f"partition {part} accepted count disagrees with registered target")
    dev_counts = Counter(tuple(row["profile"]) for row in records if row["partition"] == "development")
    expected_profiles = {tuple(p): config["generation"]["target_accepted_per_development_profile"] for p in PROFILES["development"]}
    if dev_counts != expected_profiles:
        raise ValueError("development profile quotas disagree with accepted records")
    validate_profile_count_closure(manifest, records)


def validate_profile_count_closure(manifest: dict, records: list[dict]) -> None:
    """Reconcile any persisted producer profile counts with accepted records."""
    for part in PARTITIONS:
        state = manifest["partitions"][part]
        persisted = state.get("profile_accepted")
        accepted_records = [row for row in records if row["partition"] == part]
        if state["status"] == "not_attempted" and accepted_records:
            raise ValueError(f"not-attempted partition {part} has accepted records")
        if persisted is None:
            # An attempted run may be stopped before the first candidate, before
            # the producer has initialized its all-zero profile counter.
            if accepted_records:
                raise ValueError(f"partition {part} accepted records lack profile counts")
            continue
        actual = Counter(tuple(row["profile"]) for row in accepted_records)
        normalized = {"/".join(profile): actual[profile] for profile in PROFILES[part]}
        if persisted != normalized:
            raise ValueError(f"partition {part} profile counts disagree with records")


def validate_record_set(records: list[dict], config: dict) -> list[dict]:
    """Check per-row identities and corpus-wide content, lineage and signature closure."""
    infos = [validate_record(row, config) for row in records]
    validate_corpus_closure(infos)
    return infos


def validate_corpus_closure(infos: list[dict]) -> None:
    """Enforce unique content/lineage IDs and single-split development signatures."""
    for field in ("content_id", "lineage_id"):
        values = [info[field] for info in infos]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {field} across accepted records")
    split_by_signature: dict[str, set[str]] = defaultdict(set)
    for info in infos:
        if info["partition"] == "development":
            split_by_signature[info["signature"]].add(info["split"])
    if any(len(splits) != 1 for splits in split_by_signature.values()):
        raise ValueError("development structure signature crosses split boundary")


def source_train_primitive_coverage(records: list[dict]) -> dict:
    """Measure source TRAIN primitive exposure separately from corpus coverage."""
    corpus_kinds = {constraint.kind.value for row in records for constraint in row["task"].constraints}
    train_kinds = {constraint.kind.value for row in records if row["split"] == "source_train"
                   for constraint in row["task"].constraints}
    required = {"eq", "neq", "add_mod"}
    return {"corpus_kinds": sorted(corpus_kinds), "source_train_kinds": sorted(train_kinds),
            "pass": required <= train_kinds,
            "missing_source_train_kinds": sorted(required - train_kinds)}


def source_train_composition_absence(records: list[dict]) -> dict:
    """Check instantiated task compositions, excluding universal prompt rules."""
    all_profiles = tuple(profile for profiles in PROFILES.values() for profile in profiles)
    observed = set()
    for row in records:
        if row["split"] != "source_train": continue
        kinds = {constraint.kind.value for constraint in row["task"].constraints}
        matches = [profile for profile in all_profiles if set(profile) == kinds]
        observed.add(tuple(matches[0]) if len(matches) == 1 else tuple(sorted(kinds)))
    targets = set(PROFILES["development"]) | set(PROFILES["final"])
    overlap = sorted("/".join(profile) for profile in observed & targets)
    return {"source_train_compositions": sorted("/".join(profile) for profile in observed),
            "heldout_compositions_absent": not overlap, "overlap": overlap}


def run_audit(config_path: Path, data_dir: Path, argv: list[str], *, guard: ResourceGuard | None = None,
              expected_limits: dict | None = None) -> int:
    """Audit a corpus once, streaming each completed integrity/replay measurement."""
    clock = guard.clock if guard is not None else time.monotonic
    started = clock()
    try:
        cpu_started = time.process_time()
    except Exception:
        cpu_started = None
    config = load_config(config_path)
    data_dir = Path(data_dir).resolve()
    generation_manifest = strict_json_loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    remaining = remaining_global_seconds(generation_manifest, config.resolved["audit_gate"]["cap_wall_seconds"])
    output = data_dir / "audit_manifest.json"
    report_path, results_path = data_dir / "audit_report.json", data_dir / "audit_results.jsonl"
    audit_paths = (output, report_path, results_path, output.with_name(output.name + ".tmp"),
                   report_path.with_name(report_path.name + ".tmp"), results_path.with_name(results_path.name + ".tmp"))
    existing = [path.name for path in audit_paths if path.exists()]
    if existing: raise FileExistsError(f"audit output already exists for this corpus: {existing}")
    audit_source = source_identity()
    guard = guard or ResourceGuard(remaining, config.resolved["audit_gate"]["cap_resident_memory_mib"], clock=clock)
    guard.wall_seconds = min(guard.wall_seconds, remaining)
    guard.memory_bytes = config.resolved["audit_gate"]["cap_resident_memory_mib"] * 1024 * 1024
    guard.started = started
    progress = {"records_seen": 0, "record_integrity_assessed": 0,
                "record_interactions_assessed": 0, "replay_candidates_assessed": 0,
                "accepted_records_reconciled": 0}
    audit_manifest = {"manifest_version": "dwr-task-audit-v1", "status": "running",
                      "created_utc": datetime.now(timezone.utc).isoformat(), "command": argv,
                      "cwd": str(Path.cwd()), "python": sys.version, "platform": platform_info.platform(),
                      "config_sha256": config.sha256, "generation_manifest_sha256": sha256_file(data_dir / "manifest.json"),
                      "resolved_config": config.resolved, "source": audit_source,
                      "generation_source": generation_manifest.get("source"),
                      "generation_status": generation_manifest.get("status"),
                      "generation_elapsed_seconds": generation_manifest.get("generation_elapsed_seconds"),
                      "remaining_global_seconds_at_start": remaining,
                      "limits": _registered_limits(config.resolved),
                      "output_paths": {"report": str(report_path), "results": str(results_path)},
                      "audit_cpu_started_seconds": cpu_started, "progress": dict(progress)}
    # Exclusive creation and the preflight above ensure previous evidence is never replaced.
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(stable_json(audit_manifest) + "\n")
    infos: list[dict] = []
    interaction_results: list[dict] = []
    record_by_key: dict[tuple[str, int], dict] = {}
    metrics = None
    primary_error = None
    finalization_errors = []
    phase = "setup"
    generation_complete = False
    audit_complete = False
    replay_complete = False
    with results_path.open("x", encoding="utf-8", newline="\n") as results:
        def persist_result(value: dict) -> None:
            results.write(stable_json(value) + "\n")
            results.flush()
            audit_manifest["progress"] = dict(progress)
            atomic_json(output, audit_manifest)

        try:
            guard.check()
            phase = "manifest"
            manifest, outcomes, records = _verify_manifest(data_dir, config, expected_limits)
            generation_complete = manifest["status"] == "complete"
            if generation_complete:
                validate_complete_generation(manifest, outcomes, records, config.resolved)
            attempts_by_partition = Counter(row["partition"] for row in outcomes)
            record_counts_by_partition = Counter(row["partition"] for row in records)
            for part, state in manifest["partitions"].items():
                if attempts_by_partition[part] != state["attempts"]:
                    raise ValueError(f"partial outcomes disagree with {part} manifest attempts")
                if record_counts_by_partition[part] != state["accepted"]:
                    raise ValueError(f"partial records disagree with {part} manifest accepted count")
            if not generation_complete:
                validate_profile_count_closure(manifest, records)
            record_by_key = {(row["partition"], row["candidate_index"]): row for row in records}
            if len(record_by_key) != len(records): raise ValueError("duplicate accepted record key")
            phase = "record_integrity_and_interactions"
            for raw in records:
                guard.check()
                progress["records_seen"] += 1
                info = validate_record(raw, config.resolved)
                progress["record_integrity_assessed"] += 1
                interaction = audit_interactions(info["task"], info["partition"])
                progress["record_interactions_assessed"] += 1
                infos.append(info)
                interaction_results.append(interaction)
                persist_result({"kind": "record_interaction", "partition": raw["partition"],
                                "candidate_index": raw["candidate_index"],
                                "content_id": info["content_id"], "integrity": "pass",
                                "interaction": interaction})
                guard.check()
            validate_corpus_closure(infos)
            metrics = compute_metrics(records, config.resolved, check=guard.check,
                                      validated_infos=infos, interaction_results=interaction_results)
            phase = "candidate_replay"
            def on_replay(row: dict, outcome: CandidateOutcome, repeated: bool) -> None:
                key = (row["partition"], row["candidate_index"])
                if row["accepted"]:
                    record = record_by_key.get(key)
                    if record is None: raise ValueError(f"accepted replay outcome has no record: {key}")
                    expected = make_record(outcome.task, outcome.partition, outcome.candidate_index)
                    if record != expected: raise ValueError(f"accepted record differs from replay: {key}")
                    progress["accepted_records_reconciled"] += 1
                elif key in record_by_key:
                    raise ValueError(f"rejected replay outcome has an accepted record: {key}")
                progress["replay_candidates_assessed"] += 1
                persist_result({"kind": "candidate_replay", "partition": row["partition"],
                                "candidate_index": row["candidate_index"], "replay": "pass",
                                "accepted": row["accepted"], "rejection_reason": row["rejection_reason"]})
            replayed = replay_outcomes(outcomes, config.resolved["seed"],
                {p: set(PROFILES[p]) for p in PARTITIONS}, {},
                config.resolved["generation"]["target_accepted_per_development_profile"],
                check=guard.check, on_row=on_replay)
            replay_complete = True
            if progress["accepted_records_reconciled"] != len(records):
                raise ValueError("replay and accepted record counts disagree")
            audit_complete = True
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                primary_error = {"type": type(exc).__name__, "message": "interrupted", "phase": phase}
            else:
                primary_error = {"type": type(exc).__name__, "message": str(exc), "phase": phase}

    structural_complete = metrics is not None
    if metrics is not None:
        metrics["generation_complete"] = generation_complete
        metrics["replay_pass"] = replay_complete
        metrics["integrity_pass"] = progress["record_integrity_assessed"] == len(record_by_key)
        metrics["record_interactions_assessed"] = progress["record_interactions_assessed"]
        metrics["replay_candidates_assessed"] = progress["replay_candidates_assessed"]
        metrics["structural_pass"] = metrics["structural_pass"]
    g2 = assess_g2(metrics, config.resolved) if generation_complete and replay_complete and metrics is not None and primary_error is None else {
        "status": "not_assessed", "pass": False,
        "floor_failures": ["generation_incomplete_or_audit_unfinished"],
        "claim": "not assessed; incomplete generation or audit blocks G2"}
    if primary_error is None:
        terminal_status = "complete" if generation_complete and audit_complete else "incomplete_generation"
    elif primary_error["type"] in ("TimeoutError", "MemoryError", "RuntimeError"):
        terminal_status = "blocked" if remaining <= 0 and primary_error["type"] == "TimeoutError" else "incomplete"
    else:
        terminal_status = "audit_error"
    unassessed = []
    if progress["record_integrity_assessed"] < len(record_by_key):
        unassessed.append("remaining record integrity and interactions")
    if not replay_complete: unassessed.append("remaining candidate replay")
    if not generation_complete or not replay_complete: unassessed.append("G2")
    report = {"status": terminal_status, "generation_status": generation_manifest.get("status"),
              "g2": g2, "progress": progress, "metrics": metrics,
              "unassessed": unassessed,
              "error": primary_error, "finalization_errors": finalization_errors}
    if not generation_complete:
        report["reason"] = "generation was not COMPLETE; partial diagnostics do not establish G2 suitability"

    try:
        guard.check()
    except (MemoryError, RuntimeError, TimeoutError) as exc:
        finalization_errors.append({"type": type(exc).__name__, "message": str(exc), "phase": "finalization"})
        if primary_error is None:
            primary_error = finalization_errors[-1]
            report["error"] = primary_error
            report["status"] = "blocked" if remaining <= 0 and isinstance(exc, TimeoutError) else "incomplete"
            report["g2"] = {"status": "not_assessed", "pass": False,
                            "floor_failures": ["audit_finalization_cap"], "claim": "not assessed"}
    report["finalization_errors"] = finalization_errors
    cpu_measurement_error = None
    try:
        cpu_elapsed = None if cpu_started is None else max(0.0, time.process_time() - cpu_started)
    except Exception as exc:
        cpu_elapsed = None
        cpu_measurement_error = str(exc)
    audit_manifest.update({"status": report["status"], "audit_elapsed_seconds": round(guard.elapsed, 6),
        "combined_generation_and_audit_elapsed_seconds": round(float(generation_manifest.get("generation_elapsed_seconds", 0.0)) + guard.elapsed, 6),
        "audit_cpu_seconds": cpu_elapsed, "audit_cpu_measurement_error": cpu_measurement_error,
        "peak_resident_bytes": guard.peak_bytes,
        "accounting_end_boundary": "post_first_terminal_report_resource_check",
        "accounting_note": "Final bounded report and manifest flush tails are excluded.",
        "progress": progress, "error": primary_error, "finalization_errors": finalization_errors})
    report["resources"] = {"audit_elapsed_seconds": round(guard.elapsed, 6),
        "combined_generation_and_audit_elapsed_seconds": round(float(generation_manifest.get("generation_elapsed_seconds", 0.0)) + guard.elapsed, 6),
        "audit_cpu_seconds": cpu_elapsed, "audit_cpu_measurement_error": cpu_measurement_error,
        "peak_resident_bytes": guard.peak_bytes,
        "accounting_note": audit_manifest["accounting_note"]}
    atomic_json(report_path, report)
    try:
        guard.check()
    except (MemoryError, RuntimeError, TimeoutError) as exc:
        finalization_errors.append({"type": type(exc).__name__, "message": str(exc), "phase": "finalization"})
        if primary_error is None:
            primary_error = finalization_errors[-1]
            report["error"] = primary_error
            report["status"] = "blocked" if remaining <= 0 and isinstance(exc, TimeoutError) else "incomplete"
            report["g2"] = {"status": "not_assessed", "pass": False,
                            "floor_failures": ["audit_finalization_cap"], "claim": "not assessed"}
        report["finalization_errors"] = finalization_errors
        audit_manifest["finalization_errors"] = finalization_errors
    report["finalization_errors"] = finalization_errors
    audit_manifest.update({"status": report["status"], "audit_elapsed_seconds": round(guard.elapsed, 6),
        "combined_generation_and_audit_elapsed_seconds": round(float(generation_manifest.get("generation_elapsed_seconds", 0.0)) + guard.elapsed, 6),
        "audit_cpu_seconds": cpu_elapsed, "audit_cpu_measurement_error": cpu_measurement_error,
        "peak_resident_bytes": guard.peak_bytes, "progress": progress,
        "error": primary_error, "finalization_errors": finalization_errors,
        "exit_code": 0 if report["status"] == "complete" else 1})
    report["resources"].update({"audit_elapsed_seconds": round(guard.elapsed, 6),
        "combined_generation_and_audit_elapsed_seconds": round(float(generation_manifest.get("generation_elapsed_seconds", 0.0)) + guard.elapsed, 6),
        "peak_resident_bytes": guard.peak_bytes})
    atomic_json(report_path, report)
    audit_manifest["artifacts"] = {results_path.name: {"sha256": sha256_file(results_path), "bytes": results_path.stat().st_size},
                                   report_path.name: {"sha256": sha256_file(report_path), "bytes": report_path.stat().st_size}}
    atomic_json(output, audit_manifest)
    return audit_manifest["exit_code"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently audit a generated DWR task corpus")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", "dwr.tasks.audit", "--config", str(args.config), "--data", str(args.data)]
    try:
        return run_audit(args.config, args.data, command)
    except Exception as exc:
        print(f"audit failed before report creation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
