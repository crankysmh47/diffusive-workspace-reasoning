"""Bounded descriptive analysis of the registered retained certificate study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .run_support import ResourceGuard, resident_bytes_windows, stable_json
from .schema import ConstraintTask, RelationKind, parse_task_json


CONFIG_SHA256 = "25de684c415dfddeeee2af538d00e8f1f379452d442c388da36589dffcd993e3"
ORIGINAL_CONFIG_SHA256 = "4f8b76404295828afe7b2079784fca1a2cbf2c782bfe9378f5571383ffe089e6"
SCIENTIFIC_COMMIT = "fdb77b9a8082492b98ad1e7ac473e88564be2be8"
ARTIFACTS = ("manifest.initial.json", "permitted_rows.jsonl", "report.json", "manifest.json")
INPUT_ARTIFACTS = ("manifest.initial.json", "manifest.json", "outcomes.jsonl", "replay.json", "report.json")
SPLITS = {"source": "source_train", "development": "development_validation"}
PROFILES = {"source": ("eq", "neq", "add_mod"),
            "development": ("eq/neq", "eq/add_mod", "neq/add_mod")}
STAGES = ("constructed", "unique", "eligible_groups", "envelope_groups", "usable_groups", "duplicates",
          "nonunique", "ineligible", "envelope_excluded", "construction_failures")
WORK_FIELDS = ("given_position_scans", "witness_pool_scans", "alternative_pool_scans",
               "constraint_pool_scans", "count_vector_scans", "feasible_alternatives",
               "feasible_second_alternatives", "feasible_count_vectors", "rng_calls")


class DiagnosisError(RuntimeError):
    """Safe, typed diagnostic failure; messages must not contain input values."""


def _strict_json(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result

    def constant(_token):
        raise ValueError("nonfinite")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)

    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    registered = root / "configs/tasks/certificate_coverage_diagnosis_v1.json"
    requested = Path(path).resolve(strict=True)
    try:
        expected_bytes = registered.read_bytes()
        actual_bytes = requested.read_bytes()
    except OSError:
        raise DiagnosisError("configuration_unreadable") from None
    if hashlib.sha256(expected_bytes).hexdigest() != CONFIG_SHA256:
        raise DiagnosisError("registered_configuration_identity_mismatch")
    if hashlib.sha256(actual_bytes).hexdigest() != CONFIG_SHA256:
        raise DiagnosisError("configuration_identity_mismatch")
    try:
        expected, actual = _strict_json(expected_bytes.decode("utf-8")), _strict_json(actual_bytes.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError):
        raise DiagnosisError("configuration_invalid") from None
    if not _typed_equal(actual, expected):
        raise DiagnosisError("configuration_contract_mismatch")
    required = {"artifact_names", "candidate_generation", "config_version", "final_access",
                "input_artifact_sha256", "input_config_sha256", "input_run_path",
                "input_scientific_source_commit", "limits", "permitted_splits",
                "registered_attempts", "test_allowance_seconds"}
    if set(actual) != required or actual["artifact_names"] != list(ARTIFACTS):
        raise DiagnosisError("configuration_contract_mismatch")
    return actual


def _typed_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_typed_equal(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _task_features(task: ConstraintTask, values: list[int]) -> dict[str, Any]:
    """Small deterministic EQ/NEQ descriptor; the task object is not retained."""
    if len(task.variables) != 6 or len(values) != 6:
        raise ValueError("registered_task_size_mismatch")
    adjacency = {name: set() for name in task.variables}
    givens = {given.variable: given.value for given in task.givens}
    for constraint in task.constraints:
        if constraint.kind is RelationKind.EQ:
            left, right = constraint.variables
            adjacency[left].add(right)
            adjacency[right].add(left)
    discovered: set[str] = set()
    components: list[tuple[str, ...]] = []
    for start in task.variables:
        if start in discovered:
            continue
        discovered.add(start)
        stack, component = [start], []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in discovered:
                    discovered.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    ordered_components = sorted(components)
    component_index = {name: index for index, members in enumerate(ordered_components) for name in members}
    anchored = {component_index[name] for name in givens}
    inequality_edges = [tuple(sorted((component_index[a], component_index[b])))
                        for rule in task.constraints if rule.kind is RelationKind.NEQ
                        for a, b in [rule.variables]]
    multiplicities = sorted(Counter(values).values(), reverse=True)
    return {
        "component_count": len(ordered_components),
        "component_members": [list(members) for members in ordered_components],
        "component_sizes": [len(members) for members in ordered_components],
        "anchored_components": len(anchored),
        "unanchored_components": len(ordered_components) - len(anchored),
        "raw_inequality_edges": sum(rule.kind is RelationKind.NEQ for rule in task.constraints),
        "distinct_normalized_inequality_edges": len(set(inequality_edges)),
        "witness_value_multiplicities": multiplicities,
        "distinct_given_values": len(set(givens.values())),
        "kind_counts": dict(sorted(Counter(rule.kind.value for rule in task.constraints).items())),
        "given_only_constraints": sum(all(name in givens for name in rule.variables) for rule in task.constraints),
    }


def _counterfactual_arithmetic(*, eligible: int, stalled: int) -> dict[str, Any]:
    if type(eligible) is not int or type(stalled) is not int or eligible < 0 or not 0 <= stalled <= eligible:
        raise ValueError("invalid_count")
    def scenario(extra: int) -> dict[str, Any]:
        total = eligible + extra
        return {"eligible": total, "stalled": stalled,
                "stalled_fraction": stalled / total if total else 0.0,
                "assumption": "all added groups are eligible and nonstalled; existing counts fixed"}
    required = {}
    for extra in (43, 100):
        denominator = eligible + extra
        # Solve (stalled+h)/(eligible+d+h) >= 0.10 for integer h.
        h = max(0, math.ceil((denominator - 10 * stalled) / 9 - 1e-12))
        required[str(extra)] = {"added_stalls_required": h,
                                "minimum_total_stalls": stalled + h,
                                "resulting_fraction": (stalled + h) / (denominator + h)}
    maximum = max(0, 10 * stalled - eligible)
    return {"plus_43_nonstalled": scenario(43), "plus_100_nonstalled": scenario(100),
            "max_added_nonstalled_at_10_percent": maximum,
            "added_stalls_required_for_10_percent": required,
            "threshold": 0.10,
            "inequality": "(stalled + h) / (eligible + d + h) >= 0.10; with h=0, d <= 10*stalled-eligible",
            "assumption": "d and h are eligible groups added to fixed existing counts; d are nonstalled, h are stalled"}


def _empty_counts() -> dict[str, int]:
    return {key: 0 for key in STAGES}


def _analyze_rows(rows: Iterable[dict[str, Any]], *, partition: str,
                  projection_stream=None) -> dict[str, Any]:
    if partition not in SPLITS:
        raise ValueError("invalid_partition")
    totals = {profile: _empty_counts() for profile in PROFILES[partition]}
    seen: set[str] = set()
    lineages: dict[str, str] = {}
    per_signature: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    feature_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    profile_cell: dict[str, Counter[str]] = defaultdict(Counter)
    cell_stages: dict[str, Counter[str]] = defaultdict(Counter)
    construction_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    construction_work: dict[str, Counter[str]] = defaultdict(Counter)
    stall_counts: dict[str, Counter[str]] = defaultdict(Counter)
    exclusion_bins: dict[str, Counter[str]] = defaultdict(Counter)
    permitted = feature_rows = rows_seen = 0
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise DiagnosisError(f"row_invalid:index={index}")
        split = raw.get("split")
        # Do not inspect any private fields before routing by split.
        if split not in (None, SPLITS[partition]):
            continue
        rows_seen += 1
        profile = raw.get("profile")
        cell = raw.get("cell")
        if split is None:
            if (raw.get("task") is not None or raw.get("state") != "rejected_proposal" or
                    raw.get("partition") != partition or type(raw.get("candidate_index")) is not int or
                    raw["candidate_index"] < 0 or
                    profile not in PROFILES[partition] or cell not in ("A", "B", "C", "D") or
                    raw.get("rejection_reason") not in {"empty_witness_pool", "no_feasible_source_alternative",
                                                        "no_feasible_mixed_second_alternative"}):
                raise DiagnosisError(f"row_invalid:index={index}")
            totals[profile]["construction_failures"] += 1
            cell_stages[profile][cell + "/construction_failure"] += 1
            reason = raw["rejection_reason"]
            construction_reasons[profile][reason] += 1
            work = raw.get("work")
            if not isinstance(work, dict) or set(work) != set(WORK_FIELDS) or any(
                    type(work[key]) is not int or work[key] < 0 for key in WORK_FIELDS):
                raise DiagnosisError(f"row_invalid:index={index}")
            for key in WORK_FIELDS:
                construction_work[profile][key] += work[key]
            if projection_stream is not None:
                projection_stream.write(stable_json({"partition": partition, "candidate_index": raw.get("candidate_index"),
                                                     "profile": profile, "cell": cell, "stage": "construction_failure"}) + "\n")
            continue
        if raw.get("partition") != partition or profile not in PROFILES[partition] or cell not in ("A", "B", "C", "D"):
            raise DiagnosisError(f"row_invalid:index={index}")
        permitted += 1
        state = raw.get("state")
        cid = raw.get("content_id")
        if (not isinstance(cid, str) or re.fullmatch(r"[0-9a-f]{64}", cid) is None or
                not isinstance(raw.get("lineage_id"), str) or
                re.fullmatch(r"[0-9a-f]{64}", raw["lineage_id"]) is None or
                not isinstance(raw.get("structure_signature"), str) or
                re.fullmatch(r"[0-9a-f]{64}", raw["structure_signature"]) is None or
                type(raw.get("oracle_unique")) is not bool):
            raise DiagnosisError(f"row_invalid:index={index}")
        prior = lineages.get(raw["lineage_id"])
        if prior is not None and prior != cid:
            raise DiagnosisError(f"row_invalid:index={index}")
        lineages[raw["lineage_id"]] = cid
        counts = totals[profile]
        if type(raw.get("candidate_index")) is not int or raw["candidate_index"] < 0:
            raise DiagnosisError(f"row_invalid:index={index}")
        if type(raw.get("eligible")) is not bool or type(raw.get("envelope")) is not bool or type(raw.get("sensitive")) is not bool:
            raise DiagnosisError(f"row_invalid:index={index}")
        oracle_count = raw.get("oracle_solution_count")
        if (type(oracle_count) is not int or oracle_count < 0 or
                type(raw.get("prompt_bytes")) is not int or raw["prompt_bytes"] < 0 or
                type(raw.get("first_solution_nodes")) is not int or raw["first_solution_nodes"] < 1 or
                type(raw.get("given_count")) is not int or raw["given_count"] < 1):
            raise DiagnosisError(f"row_invalid:index={index}")
        if state not in {"duplicate_content", "oracle_non_unique", "scientifically_ineligible",
                         "outside_registered_support_envelope", "accepted_for_support"}:
            raise DiagnosisError(f"row_invalid:index={index}")
        if state != "duplicate_content" and raw["oracle_unique"] != (oracle_count == 1):
            raise DiagnosisError(f"row_invalid:index={index}")
        if state == "duplicate_content" and (raw.get("duplicate_content") is not True or
                                              raw["oracle_unique"] is not False or oracle_count < 1):
            raise DiagnosisError(f"row_invalid:index={index}")
        if state != "duplicate_content" and raw.get("duplicate_content") is True:
            raise DiagnosisError(f"row_invalid:index={index}")
        expected_givens, expected_constraints = {"A": (2, 8), "B": (3, 6), "C": (3, 7), "D": (3, 8)}[cell]
        if type(raw["given_count"]) is not int or raw["given_count"] != expected_givens:
            raise DiagnosisError(f"row_invalid:index={index}")
        prompt = raw["prompt_bytes"]
        nodes = raw["first_solution_nodes"]
        expected_envelope = oracle_count == 1 and prompt > 1023 and 4 <= nodes <= 7
        if raw["envelope"] != expected_envelope:
            raise DiagnosisError(f"row_invalid:index={index}")
        expected_state = ("duplicate_content" if state == "duplicate_content" else
                          "oracle_non_unique" if raw["oracle_unique"] is not True else
                          "scientifically_ineligible" if raw["eligible"] is not True else
                          "accepted_for_support" if raw["envelope"] is True else
                          "outside_registered_support_envelope")
        if state != expected_state:
            raise DiagnosisError(f"row_invalid:index={index}")
        try:
            task_data = raw["task"]
            parsed_task = parse_task_json(task_data if isinstance(task_data, str) else stable_json(task_data))
            from .schema import serialize_model_input
            values = raw["values"]
            expected_kinds = set(profile.split("/"))
            if (tuple(parsed_task.variables) != tuple(f"v{i}" for i in range(6)) or
                    len(parsed_task.givens) != expected_givens or len(parsed_task.constraints) != expected_constraints or
                    {rule.kind.value for rule in parsed_task.constraints} != expected_kinds or
                    raw["given_count"] != len(parsed_task.givens) or
                    raw["prompt_bytes"] != len(serialize_model_input(parsed_task).encode("utf-8")) or
                    not isinstance(values, list) or len(values) != len(parsed_task.variables) or
                    any(type(value) is not int or value not in (0, 1, 2) for value in values) or
                    any(values[parsed_task.variables.index(given.variable)] != given.value
                        for given in parsed_task.givens)):
                raise ValueError("task_metadata_mismatch")
        except (KeyError, TypeError, ValueError):
            raise DiagnosisError(f"row_invalid:index={index}") from None
        gac_value = raw.get("gac")
        if (not isinstance(gac_value, dict) or type(gac_value.get("stalled")) is not bool or
                type(gac_value.get("solved")) is not bool or type(gac_value.get("contradiction")) is not bool or
                any(type(gac_value.get(key)) is not int or gac_value[key] < 0
                    for key in ("unresolved_unknowns", "rounds", "support_checks"))):
            raise DiagnosisError(f"row_invalid:index={index}")
        counts["constructed"] += 1
        if state == "duplicate_content" or raw.get("duplicate_content") is True:
            counts["duplicates"] += 1
            attempt_stage = "duplicate"
            cell_stages[profile][cell + "/duplicate"] += 1
        elif state == "oracle_non_unique" or raw.get("oracle_unique") is not True:
            counts["nonunique"] += 1
            attempt_stage = "nonunique"
            cell_stages[profile][cell + "/nonunique"] += 1
        elif cid in seen:
            counts["duplicates"] += 1
            attempt_stage = "duplicate"
            cell_stages[profile][cell + "/duplicate"] += 1
        else:
            seen.add(cid)
            counts["unique"] += 1
            cell_stages[profile][cell + "/unique"] += 1
            eligible = raw.get("eligible") is True
            envelope = raw.get("envelope") is True
            if not eligible:
                counts["ineligible"] += 1
                attempt_stage = "ineligible"
                cell_stages[profile][cell + "/ineligible"] += 1
            else:
                counts["eligible_groups"] += 1
                cell_stages[profile][cell + "/eligible"] += 1
                signature = raw.get("structure_signature")
                if not isinstance(signature, str):
                    raise DiagnosisError(f"row_invalid:index={index}")
                per_signature[profile][signature]["eligible"] += 1
                profile_cell[profile]["eligible/" + cell] += 1
                if (raw.get("gac") or {}).get("stalled") is True:
                    stall_counts[profile]["stalled"] += 1
                if envelope:
                    counts["envelope_groups"] += 1
                    cell_stages[profile][cell + "/envelope"] += 1
                    per_signature[profile][signature]["envelope"] += 1
                    profile_cell[profile]["envelope/" + cell] += 1
                    # The source gate's public "usable" cohort also requires
                    # the registered sensitivity flag. Development has no such filter.
                    if partition == "development" or raw["sensitive"]:
                        counts["usable_groups"] += 1
                        per_signature[profile][signature]["usable"] += 1
                        profile_cell[profile]["usable/" + cell] += 1
                    attempt_stage = "envelope"
                else:
                    counts["envelope_excluded"] += 1
                    cell_stages[profile][cell + "/envelope_excluded"] += 1
                    attempt_stage = "eligible_without_envelope"
                    prompt = raw.get("prompt_bytes")
                    nodes = raw.get("first_solution_nodes")
                    if type(prompt) is not int or prompt < 0 or type(nodes) is not int or nodes < 1:
                        raise DiagnosisError(f"row_invalid:index={index}")
                    pb = 0 if prompt <= 511 else 1 if prompt <= 639 else 2 if prompt <= 767 else 3 if prompt <= 1023 else 4
                    nb = 0 if nodes == 1 else 1 if nodes <= 3 else 2 if nodes <= 7 else 3 if nodes <= 15 else 4 if nodes <= 31 else 5 if nodes <= 63 else 6
                    exclusion_bins[profile][f"prompt_bin_{pb}/node_bin_{nb}"] += 1
        if (partition == "development" and profile == "eq/neq"):
            try:
                features = _task_features(parsed_task, values)
                gac = raw.get("gac") or {}
                stage = attempt_stage
                key = stable_json(features)
                feature_counts[profile][stage][key] += 1
                feature_rows += 1
            except (KeyError, TypeError, ValueError):
                raise DiagnosisError(f"row_invalid:index={index}") from None
            if projection_stream is not None:
                projection_stream.write(stable_json({
                    "partition": partition, "candidate_index": raw["candidate_index"],
                    "profile": profile, "cell": cell, "split": split,
                    "content_id": cid, "structure_signature": raw.get("structure_signature"),
                    "stage": attempt_stage, "features": features,
                }) + "\n")
        elif projection_stream is not None:
            projection_stream.write(stable_json({
                "partition": partition, "candidate_index": raw["candidate_index"],
                "profile": profile, "cell": cell, "split": split,
                "content_id": cid, "structure_signature": raw.get("structure_signature"),
                "stage": attempt_stage,
            }) + "\n")
    profile_report = {}
    for profile in PROFILES[partition]:
        c = totals[profile]
        profile_report[profile] = {"constructed": c["constructed"], "unique": c["unique"],
                                   "eligible": c["eligible_groups"], "envelope": c["envelope_groups"],
                                   "usable": c["usable_groups"],
                                   "construction_failures": c["construction_failures"]}
    return {"partition": partition, "rows_seen": rows_seen, "permitted_rows": permitted,
            "feature_rows": feature_rows, "counts": {SPLITS[partition]: totals},
            "profiles": profile_report,
            "profile_cell_counts": {profile: {
                "eligible_groups_by_cell": {cell: profile_cell[profile]["eligible/" + cell] for cell in "ABCD"},
                "eligible_envelope_groups_by_cell": {cell: profile_cell[profile]["envelope/" + cell] for cell in "ABCD"},
                "usable_groups_by_cell": {cell: profile_cell[profile]["usable/" + cell] for cell in "ABCD"},
            } for profile in PROFILES[partition]},
            "stage_counts_by_cell": {profile: dict(sorted(cell_stages[profile].items()))
                                     for profile in PROFILES[partition]},
            "construction_failures": {profile: {
                "reasons": dict(sorted(construction_reasons[profile].items())),
                "work": dict(sorted(construction_work[profile].items())),
            } for profile in PROFILES[partition]},
            "signature_coverage": {profile: {
                "eligible_signatures": len(per_signature[profile]),
                "missing_envelope_signatures": sum(not values["envelope"] for values in per_signature[profile].values()),
                "eligible_groups_on_missing_signatures": sum(values["eligible"] for values in per_signature[profile].values() if not values["envelope"]),
                "eligible_envelope_signatures": sum(values["envelope"] > 0 for values in per_signature[profile].values()),
                "usable_signatures": sum(values["usable"] > 0 for values in per_signature[profile].values()),
                "within_signature_envelope_presence": dict(sorted(Counter(
                    f"eligible_{values['eligible']}/envelope_{values['envelope']}"
                    for values in per_signature[profile].values()).items())),
                "corrected_eligible_envelope_min": min((values["envelope"] for values in per_signature[profile].values()), default=0),
                "eligible_groups_by_cell": {cell: profile_cell[profile]["eligible/" + cell] for cell in "ABCD"},
                "eligible_envelope_groups_by_cell": {cell: profile_cell[profile]["envelope/" + cell] for cell in "ABCD"},
                "eligible_stalled_groups": stall_counts[profile]["stalled"],
                "envelope_exclusion_prompt_node_bins": dict(sorted(exclusion_bins[profile].items())),
            } for profile in PROFILES[partition]},
            "eq_neq_features": {profile: {stage: dict(sorted(counts.items()))
                                          for stage, counts in stages.items()}
                                for profile, stages in feature_counts.items()}}


def _check(guard: ResourceGuard) -> None:
    try:
        guard.check()
    except Exception as exc:
        raise DiagnosisError(f"resource_limit:{type(exc).__name__}") from None


def _hash_file(path: Path, guard: ResourceGuard) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            _check(guard)
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    _check(guard)
    return digest.hexdigest()


def _source_identity(root: Path, guard: ResourceGuard) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True, encoding="utf-8").stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
                                check=True, capture_output=True, text=True, encoding="utf-8").stdout
    except (OSError, subprocess.CalledProcessError):
        raise DiagnosisError("source_identity_unavailable") from None
    _check(guard)
    if status.strip():
        raise DiagnosisError("source_must_be_clean")
    producer_hashes = {}
    package = root / "dwr"
    for path in sorted((package / "tasks").glob("*.py")) + [package / "__init__.py"]:
        _check(guard)
        producer_hashes[path.relative_to(root).as_posix()] = _hash_file(path, guard)
    if platform.python_implementation() != "CPython" or sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise DiagnosisError("runtime_not_registered")
    return {"git_commit": commit, "git_clean": True, "producer_sha256": producer_hashes}


def _atomic_json(path: Path, value: Any, guard: ResourceGuard) -> None:
    _check(guard)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(stable_json(value) + "\n", encoding="utf-8", newline="\n")
    _check(guard)
    temp.replace(path)
    _check(guard)


def _input_identities(root: Path, config: dict[str, Any], guard: ResourceGuard) -> dict[str, str]:
    run = root / config["input_run_path"]
    for part in (run, *run.parents):
        if part.exists() and part.is_symlink():
            raise DiagnosisError("input_path_symlink_refused")
    result = {}
    for name in INPUT_ARTIFACTS:
        path = run / name
        if not path.is_file() or path.is_symlink():
            raise DiagnosisError("input_artifact_missing")
        digest = _hash_file(path, guard)
        if digest != config["input_artifact_sha256"].get(name):
            raise DiagnosisError("input_artifact_hash_mismatch")
        result[name] = digest
    return result


def _verify_input_config(root: Path, config: dict[str, Any], guard: ResourceGuard) -> str:
    path = root / "configs/tasks/certificate_sixvar_v1.json"
    if not path.is_file() or path.is_symlink():
        raise DiagnosisError("input_configuration_missing")
    actual = _hash_file(path, guard)
    if actual != config["input_config_sha256"]:
        raise DiagnosisError("input_configuration_hash_mismatch")
    return actual


def _compare_public_linkage(report: dict[str, Any], source: dict[str, Any], development: dict[str, Any]) -> None:
    def same(observed: Any, expected: Any) -> None:
        if not _typed_equal(observed, expected):
            raise DiagnosisError("public_aggregate_mismatch")

    src, dev = report.get("source"), report.get("development")
    if not isinstance(src, dict) or not isinstance(dev, dict):
        raise DiagnosisError("public_aggregate_mismatch")
    same(src.get("first_seen_oracle_train_groups"), 1272)
    same(src.get("eligible_envelope_train_groups"), 1101)
    same(sum(v["unique"] for v in source["profiles"].values()), 1272)
    same(sum(v["usable"] for v in source["profiles"].values()), 1101)
    same(dev.get("first_seen_oracle_validation_groups"), 751)
    same(dev.get("eligible_groups"), 569)
    same(dev.get("eligible_envelope_groups"), 476)
    same(sum(v["unique"] for v in development["profiles"].values()), 751)
    same(sum(v["eligible"] for v in development["profiles"].values()), 569)
    same(sum(v["envelope"] for v in development["profiles"].values()), 476)
    public_profiles = dev.get("profile_results")
    if not isinstance(public_profiles, dict):
        raise DiagnosisError("public_aggregate_mismatch")
    for profile in PROFILES["development"]:
        public = public_profiles.get(profile)
        analyzed = development["profiles"].get(profile)
        detail = development["signature_coverage"].get(profile)
        if not isinstance(public, dict) or analyzed is None or detail is None:
            raise DiagnosisError("public_aggregate_mismatch")
        for pub_key, local_key in (("first_seen_oracle_unique_groups", "unique"),
                                   ("scientifically_eligible_groups", "eligible"),
                                   ("eligible_envelope_groups", "envelope"),
                                   ("eligible_signatures", "eligible_signatures"),
                                   ("eligible_envelope_signatures", "eligible_envelope_signatures"),
                                   ("eligible_groups_by_cell", "eligible_groups_by_cell"),
                                   ("eligible_envelope_groups_by_cell", "eligible_envelope_groups_by_cell"),
                                   ("stalled_groups", "eligible_stalled_groups")):
            expected = detail[local_key] if local_key in detail else analyzed[local_key]
            same(public.get(pub_key), expected)
    pub_source_profiles = src.get("profile_groups")
    pub_source_cells = src.get("profile_cell_counts")
    if not isinstance(pub_source_profiles, dict) or not isinstance(pub_source_cells, dict):
        raise DiagnosisError("public_aggregate_mismatch")
    for profile in PROFILES["source"]:
        same(pub_source_profiles.get(profile), source["profiles"][profile]["usable"])
        for cell in "ABCD":
            same(pub_source_cells.get(f"{profile}/{cell}"),
                 source["profile_cell_counts"][profile]["usable_groups_by_cell"][cell])
        public_sigs = src.get("profile_signatures")
        if not isinstance(public_sigs, dict):
            raise DiagnosisError("public_aggregate_mismatch")
        same(public_sigs.get(profile), source["signature_coverage"][profile]["usable_signatures"])
    public_support_cells = src.get("support_cell_counts")
    if not isinstance(public_support_cells, dict):
        raise DiagnosisError("public_aggregate_mismatch")
    for cell in "ABCD":
        same(public_support_cells.get(cell), sum(
            source["profile_cell_counts"][profile]["usable_groups_by_cell"][cell]
            for profile in PROFILES["source"]))


def _read_rows(path: Path, partition: str, guard: ResourceGuard, attempts: int,
               source_attempts: int, total_attempts: int,
               output_rows=None) -> dict[str, Any]:
    # Generator so each parsed private task is analyzed and released per line.
    def stream():
        schedule = [(profile, cell) for profile in PROFILES[partition] for cell in "ABCD"]
        if (attempts % len(schedule) or source_attempts < 0 or source_attempts > total_attempts or
                (partition == "source" and attempts != source_attempts) or
                (partition == "development" and attempts != total_attempts - source_attempts)):
            raise DiagnosisError("registered_schedule_invalid")
        line_count = 0
        with path.open("r", encoding="utf-8", newline="") as file:
            for index, line in enumerate(file):
                line_count += 1
                _check(guard)
                try:
                    # Routing reads only top-level integrity/split metadata. The
                    # protected payload is not semantically walked or validated.
                    row = json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    raise DiagnosisError(f"input_row_invalid:index={index}") from None
                if not isinstance(row, dict):
                    raise DiagnosisError(f"input_row_invalid:index={index}")
                row_partition = row.get("partition")
                candidate_index = row.get("candidate_index")
                expected_partition = "source" if index < source_attempts else "development"
                expected_candidate = index if expected_partition == "source" else index - source_attempts
                if (row_partition != expected_partition or type(candidate_index) is not int or
                        candidate_index != expected_candidate):
                    raise DiagnosisError(f"input_row_invalid:index={index}")
                split = row.get("split")
                if split is None:
                    if row_partition != partition:
                        continue
                elif split != SPLITS[partition]:
                    continue
                if split is not None:
                    try:
                        row = _strict_json(line)
                    except (ValueError, json.JSONDecodeError):
                        raise DiagnosisError(f"input_row_invalid:index={index}") from None
                expected_profile, expected_cell = schedule[candidate_index % len(schedule)]
                if (row.get("profile") != expected_profile or
                        row.get("cell") != expected_cell):
                    raise DiagnosisError(f"input_row_invalid:index={index}")
                yield row
        if line_count != total_attempts:
            raise DiagnosisError("input_prefix_count_mismatch")
    # Analyzer's feature aggregate is bounded by small fixed profile/stage bins;
    # projections are written in a second pass after the integrity hash gate.
    return _analyze_rows(stream(), partition=partition, projection_stream=output_rows)


def _write_failure_terminal(output: Path, initial: dict[str, Any], input_hashes: dict[str, str],
                            source: dict[str, Any], exc: Exception) -> None:
    """Downgrade partial output without consulting an expired resource guard."""
    failure = {"status": "NOT_ASSESSED", "g2_status": "not_assessed",
               "preflight_emitted": False,
               "terminal_reason": f"{type(exc).__name__}:diagnosis_incomplete",
               "scientific_status": "NO_GO_DEVELOPMENT_CAPACITY"}
    if not (output / "manifest.initial.json").exists():
        (output / "manifest.initial.json").write_text(stable_json(initial) + "\n",
                                                       encoding="utf-8", newline="\n")
    if not (output / "permitted_rows.jsonl").exists():
        (output / "permitted_rows.jsonl").write_text("", encoding="utf-8", newline="\n")
    (output / "report.json").write_text(stable_json(failure) + "\n", encoding="utf-8", newline="\n")
    partial = {"manifest_version": "dwr-certificate-coverage-diagnosis-v1", **failure,
               "artifacts": {}, "input_artifacts": input_hashes,
               "config_sha256": CONFIG_SHA256, "source": source}
    (output / "manifest.json").write_text(stable_json(partial) + "\n", encoding="utf-8", newline="\n")


def run_diagnosis(config_path: str | Path, output_path: str | Path, *, argv: list[str] | None = None,
                  guard: ResourceGuard | None = None, _root: Path | None = None) -> dict[str, Any]:
    start, cpu_start = time.monotonic(), time.process_time()
    root = (_root or Path(__file__).resolve().parents[2]).resolve()
    if guard is None:
        guard = ResourceGuard(60, 256, memory_reader=resident_bytes_windows)
    config = load_config(config_path)
    _check(guard)
    requested = Path(output_path).absolute()
    output = requested.resolve()
    runs_root = (root / "runs").resolve()
    try:
        output.relative_to(runs_root)
    except ValueError:
        raise DiagnosisError("output_outside_runs") from None
    if not output.name.startswith("certificate-coverage-diagnosis-") or output.exists():
        raise DiagnosisError("output_refused")
    for part in (requested, *requested.parents):
        if part.exists() and part.is_symlink():
            raise DiagnosisError("output_symlink_refused")
    source = _source_identity(root, guard)
    if source["git_commit"] == SCIENTIFIC_COMMIT:
        raise DiagnosisError("diagnosis_source_not_frozen")
    input_hashes = dict(config["input_artifact_sha256"])
    input_run = root / config["input_run_path"]
    initial = {"manifest_version": "dwr-certificate-coverage-diagnosis-initial-v1",
               "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
               "command": list(getattr(sys, "orig_argv", ())) or list(argv or sys.argv),
               "requested_argv": list(argv or sys.argv), "cwd": str(Path.cwd()),
               "output_dir": str(output), "config_sha256": CONFIG_SHA256,
               "config_path": str(Path(config_path).resolve()), "resolved_config": config,
               "input_run_path": config["input_run_path"],
               "input_config_sha256": config["input_config_sha256"],
               "input_artifacts": input_hashes, "source": source,
               "runtime": {"implementation": platform.python_implementation(), "version": list(sys.version_info[:3]),
                           "executable": sys.executable, "python": sys.version,
                           "platform": platform.platform()},
               "workers": 1, "randomness": "none", "ml": "none", "budgets": config["limits"],
               "artifact_names": list(ARTIFACTS)}
    output.mkdir(parents=True, exist_ok=False)
    try:
        report: dict[str, Any] = {"status": "NOT_ASSESSED", "g2_status": "not_assessed",
                                  "preflight_emitted": False, "input_artifacts": input_hashes,
                                  "source_identity": source, "config_sha256": CONFIG_SHA256,
                                  "scientific_status": "NO_GO_DEVELOPMENT_CAPACITY"}
        _atomic_json(output / "manifest.initial.json", initial, guard)
        measured_hashes = _input_identities(root, config, guard)
        input_config_hash = _verify_input_config(root, config, guard)
        if measured_hashes != input_hashes:
            raise DiagnosisError("input_artifact_hash_mismatch")
        try:
            source_manifest = _strict_json((input_run / "manifest.json").read_text(encoding="utf-8"))
            source_report = _strict_json((input_run / "report.json").read_text(encoding="utf-8"))
            initial_manifest = _strict_json((input_run / "manifest.initial.json").read_text(encoding="utf-8"))
            replay_manifest = _strict_json((input_run / "replay.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            raise DiagnosisError("input_metadata_invalid") from None
        if (initial_manifest.get("config_sha256") != ORIGINAL_CONFIG_SHA256 or
                initial_manifest.get("source", {}).get("git_commit") != SCIENTIFIC_COMMIT or
                initial_manifest.get("source", {}).get("git_clean") is not True or
                source_manifest.get("status") != "NO_GO_DEVELOPMENT_CAPACITY" or
                source_report.get("status") != "NO_GO_DEVELOPMENT_CAPACITY" or
                source_report.get("g2_status") != "not_assessed" or
                source_report.get("preflight_emitted") is not False or
                replay_manifest.get("status") != "PASS_REPLAY"):
            raise DiagnosisError("input_study_identity_mismatch")
        expected_manifest_artifacts = {name: digest for name, digest in input_hashes.items()
                                       if name != "manifest.json"}
        if source_manifest.get("artifacts") != expected_manifest_artifacts:
            raise DiagnosisError("input_manifest_linkage_mismatch")
        if source_manifest.get("config_sha256") != ORIGINAL_CONFIG_SHA256:
            raise DiagnosisError("input_study_identity_mismatch")
        with (output / "permitted_rows.jsonl").open("w", encoding="utf-8", newline="\n") as projections:
            source_analysis = _read_rows(input_run / "outcomes.jsonl", "source", guard,
                                         config["registered_attempts"]["source"],
                                         config["registered_attempts"]["source"],
                                         config["registered_attempts"]["source"] + config["registered_attempts"]["development"],
                                         projections)
            _check(guard)
            development_analysis = _read_rows(input_run / "outcomes.jsonl", "development", guard,
                                              config["registered_attempts"]["development"],
                                              config["registered_attempts"]["source"],
                                              config["registered_attempts"]["source"] + config["registered_attempts"]["development"],
                                              projections)
        _compare_public_linkage(source_report, source_analysis, development_analysis)
        report["source"] = source_analysis
        report["development"] = development_analysis
        report["counterfactual"] = _counterfactual_arithmetic(eligible=569, stalled=60)
        report["status"] = "PASS_DIAGNOSIS"
        report["interpretation"] = "descriptive retained-data analysis only; no capacity, difficulty, or G2 pass"
        report["capacity_inference"] = "finite registered prefixes only; no structural ceiling or saturation claim"
        report["counterfactual_scope"] = "conditional arithmetic only; no generated outcomes or mechanism forecast"
        report["elapsed_seconds"] = time.monotonic() - start
        report["peak_rss_bytes"] = guard.peak_bytes
        report["accounting_boundary"] = "after first terminal report and manifest writes"
        report["accounting_excludes"] = "final report hash and bounded report/manifest accounting flush"
        report["input_config_sha256"] = input_config_hash
        _atomic_json(output / "report.json", report, guard)
        hashes = {}
        for name in ARTIFACTS[:-1]:
            hashes[name] = _hash_file(output / name, guard)
        manifest = {"manifest_version": "dwr-certificate-coverage-diagnosis-v1",
                    "status": report["status"], "g2_status": "not_assessed",
                    "scientific_status": "NO_GO_DEVELOPMENT_CAPACITY", "artifacts": hashes,
                    "input_artifacts": input_hashes, "config_sha256": CONFIG_SHA256,
                    "source": source, "preflight_emitted": False}
        _atomic_json(output / "manifest.json", manifest, guard)
        report["elapsed_seconds"] = time.monotonic() - start
        report["cpu_seconds"] = time.process_time() - cpu_start
        report["peak_rss_bytes"] = guard.peak_bytes
        _atomic_json(output / "report.json", report, guard)
        manifest["artifacts"]["report.json"] = _hash_file(output / "report.json", guard)
        _atomic_json(output / "manifest.json", manifest, guard)
        _check(guard)
        return report
    except Exception as exc:
        # Preserve a typed failure without serializing exception text or row values.
        try:
            _write_failure_terminal(output, initial, input_hashes, source, exc)
        except Exception:
            pass
        if isinstance(exc, DiagnosisError):
            raise
        raise DiagnosisError("diagnosis_failed") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        report = run_diagnosis(args.config, args.output, argv=list(sys.orig_argv) if hasattr(sys, "orig_argv") else None)
    except DiagnosisError as exc:
        print(f"NOT_ASSESSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"NOT_ASSESSED: diagnosis failed ({type(exc).__name__})", file=sys.stderr)
        return 2
    print(stable_json(report))
    return 0 if report["status"] == "PASS_DIAGNOSIS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
