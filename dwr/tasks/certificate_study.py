"""Closed, bounded CPU study for the registered six-variable certificates.

The normal CLI accepts only the checked registration. Small injected fixtures
are available to unit tests, but fixture results are never eligible for a
production PASS status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .canonical import content_id, development_split, lineage_id, source_split, structure_signature
from .oracle import enumerate_solutions
from .run_support import ResourceGuard, resident_bytes_windows, stable_json
from .schema import ConstraintTask, RelationKind, Solution, canonical_constraint_key, serialize_task
from .solver import solve
from .audit_probe_v4 import _independent_interaction_projection


ARTIFACTS = ("manifest.initial.json", "outcomes.jsonl", "replay.json", "report.json", "manifest.json")
REGISTERED_CONFIG_SHA256 = "4f8b76404295828afe7b2079784fca1a2cbf2c782bfe9378f5571383ffe089e6"
CELLS = ("A", "B", "C", "D")
CELL_SHAPES = {"A": (2, 8), "B": (3, 6), "C": (3, 7), "D": (3, 8)}
SOURCE_SPLITS = {"source_train", "source_validation", "source_test"}
DEVELOPMENT_SPLITS = {"development_validation", "development_diagnostic_test"}


class ConfigError(ValueError):
    pass


class ResourceFailure(RuntimeError):
    pass


class IntegrityFailure(RuntimeError):
    pass


class CandidateFailure(IntegrityFailure):
    """Fatal candidate failure with the attempt and phase retained in artifacts."""
    def __init__(self, partition: str, candidate_index: int, phase: str, cause_type: str = "StudyError"):
        self.partition = partition
        self.candidate_index = candidate_index
        self.phase = phase
        self.cause_type = cause_type
        super().__init__(f"candidate failed during {phase} ({partition} index {candidate_index})")


def _save_fatal_attempt(stream: Any, partition: str, index: int, phase: str,
                        cause: BaseException, partial: dict[str, Any] | None = None) -> CandidateFailure:
    row = dict(partial or {})
    row.update({"partition": partition, "candidate_index": index, "state": "fatal_error",
                "failure": {"stage": partition, "candidate_index": index, "phase": phase,
                            "error_type": type(cause).__name__, "message": str(cause)}})
    row["construction"] = None
    stream.write(_json_bytes(_to_jsonable(row)))
    stream.flush()
    return CandidateFailure(partition, index, phase, type(cause).__name__)


def _construction_failure_payload(outcome: Any, root_seed: int, partition: str,
                                  index: int) -> dict[str, Any]:
    """Capture only constructor evidence before metric and scientific audits begin."""
    task = getattr(outcome, "task", None)
    return {
        "partition": partition,
        "candidate_index": index,
        "seed": root_seed,
        "profile": "/".join(getattr(outcome, "profile", ())),
        "cell": getattr(outcome, "cell", None),
        "task": serialize_task(task) if task is not None else None,
        "givens": getattr(outcome, "givens", ()),
        "witness": getattr(outcome, "witness", None),
        "alternatives": getattr(outcome, "alternatives", ()),
        "reserved_constraints": getattr(outcome, "reserved_constraints", ()),
        "work": getattr(outcome, "work", None),
        "rejection_reason": getattr(outcome, "rejection_reason", None),
    }


def _require_supported_runtime() -> dict[str, Any]:
    implementation = platform.python_implementation()
    version = tuple(sys.version_info[:2])
    if implementation != "CPython" or version not in ((3, 11), (3, 12)):
        raise IntegrityFailure("registered runtime admission failed")
    return {"executable": sys.executable, "implementation": implementation,
            "version": list(sys.version_info[:3])}


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate config key: {key}")
        result[key] = value
    return result


def _constant(token: str) -> None:
    raise ConfigError(f"non-finite config number: {token}")


def _strict_load(text: str) -> Any:
    try:
        value = json.loads(text, object_pairs_hook=_object_pairs, parse_constant=_constant)
    except ConfigError:
        raise
    except (ValueError, TypeError) as exc:
        raise ConfigError("malformed study config") from exc

    def check_finite(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ConfigError("non-finite config number")
        if isinstance(item, dict):
            for child in item.values():
                check_finite(child)
        elif isinstance(item, list):
            for child in item:
                check_finite(child)
    check_finite(value)
    return value


def _same_typed_tree(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_typed_tree(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_tree(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def registered_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "tasks" / "certificate_sixvar_v1.json"


def load_study_config(path: str | Path) -> "LoadedStudyConfig":
    config_path = Path(path).resolve(strict=True)
    raw_bytes = config_path.read_bytes()
    try:
        actual = _strict_load(raw_bytes.decode("utf-8"))
        expected_bytes = registered_config_path().read_bytes()
        expected = _strict_load(expected_bytes.decode("utf-8"))
    except (UnicodeError, OSError) as exc:
        raise ConfigError("cannot read registered study config") from exc
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ConfigError("study config must be an object")
    if hashlib.sha256(expected_bytes).hexdigest() != REGISTERED_CONFIG_SHA256:
        raise ConfigError("checked-in registration differs from the code-known closed contract")
    if hashlib.sha256(raw_bytes).hexdigest() != REGISTERED_CONFIG_SHA256:
        raise ConfigError("config bytes differ from the registered closed scientific contract")
    if not _same_typed_tree(actual, expected):
        raise ConfigError("config differs from the registered closed scientific contract")
    return LoadedStudyConfig(actual, hashlib.sha256(raw_bytes).hexdigest(), config_path)


@dataclass(frozen=True)
class LoadedStudyConfig:
    resolved: dict[str, Any]
    sha256: str
    path: Path


def public_row(row: dict[str, Any], *, permitted_partition: str) -> dict[str, Any] | None:
    """Project only the designated public split; private construction fields drop here."""
    if not isinstance(row, dict):
        raise IntegrityFailure("outcome row is not an object")
    split = row.get("split")
    permitted = "source_train" if permitted_partition == "source" else "development_validation"
    if split != permitted:
        return None
    allowed = ("partition", "candidate_index", "profile", "cell", "content_id",
               "lineage_id", "structure_signature", "split", "oracle_unique",
               "eligible", "envelope", "sensitive", "values", "given_count",
               "prompt_bytes", "first_solution_nodes", "gac")
    return {key: row[key] for key in allowed if key in row}


def aggregate_permitted_rows(rows: Iterable[dict[str, Any]], *, partition: str,
                             fixture_only: bool = False) -> dict[str, Any]:
    """Accumulate only source_train or development_validation projections."""
    if partition not in ("source", "development"):
        raise ValueError("invalid partition")
    first: dict[str, dict[str, Any]] = {}
    lineages: dict[str, str] = {}
    for raw in rows:
        projected = public_row(raw, permitted_partition=partition)
        if projected is None:
            continue
        split = projected["split"]
        allowed = SOURCE_SPLITS if partition == "source" else DEVELOPMENT_SPLITS
        if split not in allowed:
            raise IntegrityFailure("invalid permitted split")
        cid, lid = projected.get("content_id"), projected.get("lineage_id")
        if not isinstance(cid, str) or not isinstance(lid, str):
            raise IntegrityFailure("missing canonical identities")
        prior = lineages.get(lid)
        if prior is not None and prior != cid:
            raise IntegrityFailure("lineage maps to multiple contents")
        lineages[lid] = cid
        if projected.get("oracle_unique") is not True:
            continue
        first.setdefault(cid, projected)
    unique = list(first.values())
    eligible = [item for item in unique if item.get("eligible") is True]
    envelope = [item for item in eligible if item.get("envelope") is True]
    return {
        "partition": partition,
        "fixture_only": bool(fixture_only),
        "unique_oracle_content_groups": len(unique),
        "eligible_groups": len(eligible),
        "eligible_envelope_groups": len(envelope),
        "profiles": _counts(envelope, "profile"),
        "profile_cells": _counts(envelope, "profile", "cell"),
        "signatures": len({item["structure_signature"] for item in envelope}),
    }


def _counts(rows: Iterable[dict[str, Any]], *keys: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        key = "/".join(str(row.get(name, "")) for name in keys)
        counts[key] += 1
    return dict(sorted(counts.items()))


def common_support_mass(cohorts: dict[str, Iterable[dict[str, Any]]], *,
                        cells: tuple[str, ...] = CELLS, minimum_per_cell: int = 10) -> dict[str, Any]:
    """Use all scientifically eligible groups as denominator before envelope filtering."""
    result: dict[str, Any] = {}
    for cohort, rows_iter in cohorts.items():
        rows = [row for row in rows_iter if row.get("eligible") is True]
        counts = Counter(str(row.get("cell")) for row in rows
                         if row.get("envelope") is True and row.get("cell") in cells)
        denominator = len(rows)
        supported = sum(counts[cell] for cell in cells)
        result[cohort] = {
            "eligible_denominator": denominator,
            "common_cell_counts": {cell: counts[cell] for cell in cells},
            "common_cell_mass": supported / denominator if denominator else 0.0,
            "cell_floor_pass": all(counts[cell] >= minimum_per_cell for cell in cells),
        }
    return result


def _check(guard: Any | None) -> None:
    if guard is None:
        return
    try:
        guard() if callable(guard) else guard.check()
    except ResourceFailure:
        raise
    except Exception as exc:
        raise ResourceFailure(str(exc)) from exc


def _json_bytes(value: Any) -> bytes:
    return (stable_json(value) + "\n").encode("utf-8")


def _atomic_write(path: Path, value: Any, guard: Any | None = None) -> None:
    _check(guard)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(_json_bytes(value))
    temp.replace(path)
    _check(guard)


def _hash(path: Path, *, guard: Any | None = None) -> str:
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


def _artifact_hashes(output: Path, *, include_manifest: bool = False, guard: Any | None = None,
                     into: dict[str, str] | None = None) -> dict[str, str]:
    names = ARTIFACTS if include_manifest else ARTIFACTS[:-1]
    result = into if into is not None else {}
    for name in names:
        path = output / name
        if path.is_file():
            _check(guard)
            result[name] = _hash(path, guard=guard)
            _check(guard)
    return result


def _write_report_failure(output: Path, reason: str) -> None:
    report_path = output / "report.json"
    try:
        report = _strict_load(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    except Exception:
        report = {}
    report["status"] = "NOT_ASSESSED"
    report["terminal_reason"] = reason
    report["g2_status"] = "not_assessed"
    report["preflight_emitted"] = False
    temp = report_path.with_name(report_path.name + ".tmp")
    temp.write_bytes(_json_bytes(report))
    temp.replace(report_path)


def finalize_artifacts(output: str | Path, *, guard: Any | None,
                       verify_existing: bool = False, wall_start: float | None = None,
                       cpu_start: float | None = None,
                       invocation_start: float | None = None) -> dict[str, Any]:
    """Hash bounded terminal artifacts and revoke PASS if the guard expires."""
    directory = Path(output)
    if verify_existing:
        _check(guard)
        manifest_path = directory / "manifest.json"
        manifest = _strict_load(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("artifacts", {})
        if set(expected) != set(ARTIFACTS[:-1]):
            raise ValueError("terminal manifest artifact set is incomplete")
        for name, digest in expected.items():
            _check(guard)
            if _hash(directory / name, guard=guard) != digest:
                raise ValueError(f"artifact hash mismatch: {name}")
            _check(guard)
        return manifest
    report_path = directory / "report.json"
    try:
        report = _strict_load(report_path.read_text(encoding="utf-8"))
    except Exception:
        report = {"status": "NOT_ASSESSED", "g2_status": "not_assessed", "preflight_emitted": False}
    hashes: dict[str, str] = {}
    try:
        _check(guard)
        hashes = _artifact_hashes(directory, guard=guard, into=hashes)
        if set(hashes) != set(ARTIFACTS[:-1]):
            raise IntegrityFailure("terminal artifacts are incomplete")
        initial = _strict_load((directory / "manifest.initial.json").read_text(encoding="utf-8"))
        report = _strict_load(report_path.read_text(encoding="utf-8"))
        # Write the first terminal manifest and pass its guard check before
        # sampling. The sample therefore covers all study work, artifact
        # hashing, and the first terminal writes; only this small accounting
        # flush is excluded from the recorded boundary.
        manifest = _final_manifest(initial, report, hashes)
        (directory / "manifest.json").write_bytes(_json_bytes(manifest))
        _check(guard)
        if wall_start is not None:
            report["elapsed_seconds"] = time.monotonic() - wall_start
        if cpu_start is not None:
            report["cpu_seconds"] = time.process_time() - cpu_start
        if guard is not None:
            report["peak_rss_bytes"] = guard.peak_bytes
        if invocation_start is not None:
            report["whole_wrapper_seconds_before_accounting_flush"] = time.monotonic() - invocation_start
        report["accounting_boundary"] = "after_initial_terminal_writes_and_guard_check_before_small_accounting_flush"
        report["accounting_excludes"] = "final report hash and bounded report/manifest accounting flush"
        _atomic_write(report_path, report, guard)
        hashes["report.json"] = _hash(report_path, guard=guard)
        manifest = _final_manifest(initial, report, hashes)
        (directory / "manifest.json").write_bytes(_json_bytes(manifest))
        _check(guard)
        return manifest
    except ResourceFailure as exc:
        _write_report_failure(directory, str(exc))
        # Reuse hashes already checked under the cap. If failure happened before
        # report.json was hashed, leave it unbound instead of rehashing large
        # prefix artifacts after the guard has expired.
        if "report.json" in hashes and report_path.is_file():
            hashes["report.json"] = _hash(report_path)
        try:
            initial = _strict_load((directory / "manifest.initial.json").read_text(encoding="utf-8"))
        except Exception:
            initial = {}
        try:
            failure_report = _strict_load(report_path.read_text(encoding="utf-8"))
        except Exception:
            failure_report = {}
        failure_report.update({"status": "NOT_ASSESSED", "g2_status": "not_assessed",
                               "preflight_emitted": False, "terminal_reason": str(exc)})
        manifest = _final_manifest(initial, failure_report, hashes)
        (directory / "manifest.json").write_bytes(_json_bytes(manifest))
        raise
    except Exception:
        raise


def _final_manifest(initial: dict[str, Any], report: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    return {"manifest_version": "dwr-certificate-sixvar-run-v1",
            "status": report.get("status", "NOT_ASSESSED"),
            "source": initial.get("source"),
            "config_sha256": initial.get("config_sha256"),
            "root_seed": (initial.get("resolved_config") or {}).get("root_seed"),
            "rng": (initial.get("resolved_config") or {}).get("rng"),
            "resources": {key: report.get(key) for key in
                          ("elapsed_seconds", "cpu_seconds", "peak_rss_bytes",
                           "whole_wrapper_seconds_before_accounting_flush", "accounting_boundary",
                           "accounting_excludes")},
            "accounting": {key: report.get(key) for key in
                           ("source_construction_counts", "development_construction_counts",
                            "source", "development") if key in report},
            "artifacts": hashes, "g2_status": "not_assessed", "preflight_emitted": False,
            **({"terminal_reason": report["terminal_reason"]} if report.get("terminal_reason") else {})}


def _producer_hashes(guard: Any | None = None) -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    files = sorted(package.joinpath("tasks").glob("*.py"))
    files.append(package / "__init__.py")
    hashes = {}
    for path in files:
        _check(guard)
        hashes[str(path.relative_to(package.parent).as_posix())] = _hash(path, guard=guard)
        _check(guard)
    return hashes


def _source_identity(guard: Any | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        _check(guard)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True, encoding="utf-8").stdout.strip()
        _check(guard)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                                cwd=root, check=True, capture_output=True, text=True,
                                encoding="utf-8").stdout
        _check(guard)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IntegrityFailure("cannot identify Git source") from exc
    if status.strip():
        raise IntegrityFailure("production source must be clean and committed")
    return {"git_commit": commit, "git_clean": True, "producer_sha256": _producer_hashes(guard)}


def _solution_values(solution: Solution | None) -> tuple[tuple[str, int], ...] | None:
    return None if solution is None else solution.assignments


def _satisfies(constraint: Any, assignment: dict[str, int]) -> bool:
    values = tuple(assignment[name] for name in constraint.variables)
    if constraint.kind is RelationKind.EQ:
        return values[0] == values[1]
    if constraint.kind is RelationKind.NEQ:
        return values[0] != values[1]
    return (values[0] + values[1]) % 3 == values[2]


def _independent_graph_projection(task: ConstraintTask) -> dict[str, Any]:
    """Recompute mixed unknown-variable connectivity with a separate traversal."""
    unknown = tuple(name for name in task.variables if name not in {g.variable for g in task.givens})
    edges: dict[str, set[str]] = {name: set() for name in unknown}
    participating: set[str] = set()
    for rule in task.constraints:
        scoped = [name for name in rule.variables if name in edges]
        if scoped:
            participating.add(rule.kind.value)
            for left in scoped:
                for right in scoped:
                    if left != right:
                        edges[left].add(right)
    reached: set[str] = set()
    frontier = list(unknown[:1])
    while frontier:
        current = frontier.pop()
        if current in reached:
            continue
        reached.add(current)
        frontier.extend(edges[current] - reached)
    expected_kinds = {rule.kind.value for rule in task.constraints}
    return {"connected": bool(unknown) and reached == set(unknown),
            "participating_kinds": sorted(participating),
            "all_kinds_participate": expected_kinds <= participating}


def _audit_local_gac(task: ConstraintTask, actual: Any, *,
                     check_budget: Callable[[], None] | None = None,
                     work_counters: dict[str, int] | None = None) -> dict[str, Any]:
    """Independent support/fixed-point audit; does not call production propagation helpers."""
    from itertools import product
    position = {name: i for i, name in enumerate(task.variables)}
    domains = [set((0, 1, 2)) for _ in task.variables]
    for fixed in task.givens:
        domains[position[fixed.variable]] = {fixed.value}
    support_checks = 0
    rounds = 0
    while True:
        _check(check_budget)
        rounds += 1
        before = [set(domain) for domain in domains]
        next_domains = [set(domain) for domain in before]
        for rule in task.constraints:
            slots = [position[name] for name in rule.variables]
            supported = [set() for _ in slots]
            for values in product(*(sorted(before[slot]) for slot in slots)):
                _check(check_budget)
                support_checks += 1
                if work_counters is not None:
                    work_counters["independent_gac_support_checks"] = work_counters.get("independent_gac_support_checks", 0) + 1
                if rule.kind is RelationKind.EQ:
                    holds = values[0] == values[1]
                elif rule.kind is RelationKind.NEQ:
                    holds = values[0] != values[1]
                else:
                    holds = (values[0] + values[1]) % 3 == values[2]
                if holds:
                    for local_index, value in enumerate(values):
                        supported[local_index].add(value)
            for slot, allowed in zip(slots, supported):
                next_domains[slot].intersection_update(allowed)
        domains = next_domains
        if domains == before:
            break
    empty = any(not domain for domain in domains)
    unresolved = sum(len(domain) > 1 for domain in domains)
    solved = not empty and unresolved == 0
    stalled = not empty and not solved and unresolved > 0
    expected = {"domains": tuple(frozenset(domain) for domain in domains), "solved": solved,
                "stalled": stalled, "contradiction": empty, "unresolved_unknowns": unresolved,
                "rounds": rounds, "support_checks": support_checks}
    for field, value in expected.items():
        if getattr(actual, field, None) != value:
            raise IntegrityFailure("independent GAC support audit disagrees with propagation result")
    # Every oracle solution must remain represented in the independently audited domains.
    oracle = enumerate_solutions(task)
    if work_counters is not None:
        work_counters["oracle_assignments_examined"] = work_counters.get("oracle_assignments_examined", 0) + oracle.assignments_examined
    for solution in oracle.solutions:
        assignment = dict(solution.assignments)
        if any(assignment[name] not in domains[position[name]] for name in task.variables):
            raise IntegrityFailure("GAC removed an oracle-supported value")
    return expected


def _candidate_dict(outcome: Any, partition: str, index: int) -> dict[str, Any]:
    task = getattr(outcome, "task", None)
    row: dict[str, Any] = {
        "partition": partition, "candidate_index": index,
        "profile": getattr(outcome, "profile", None), "cell": getattr(outcome, "cell", None),
        "rejection_reason": getattr(outcome, "rejection_reason", None),
        "task": serialize_task(task) if task is not None else None,
        "construction": outcome,
    }
    return row


def _audit_task(row: dict[str, Any], *, partition: str,
                check_budget: Callable[[], None] | None = None,
                work_counters: dict[str, int] | None = None) -> dict[str, Any]:
    task = row["task_object"]
    witness = row["witness"]
    if not isinstance(task, ConstraintTask) or not isinstance(witness, Solution):
        raise IntegrityFailure("constructed task is missing its certificate witness")
    scalar_witness = dict(witness.assignments)
    if any(not _satisfies(constraint, scalar_witness) for constraint in task.constraints):
        raise IntegrityFailure("witness violates constructed constraint")
    profile = str(row.get("profile", ""))
    cell = row.get("cell")
    if cell not in CELL_SHAPES:
        raise IntegrityFailure("constructed task has an unknown support cell")
    expected_givens, expected_constraints = CELL_SHAPES[cell]
    expected_kinds = set(profile.split("/"))
    if (len(task.variables) != 6 or len(task.givens) != expected_givens or
            len(task.constraints) != expected_constraints or
            {constraint.kind.value for constraint in task.constraints} != expected_kinds):
        raise IntegrityFailure("constructed task changed its registered shape or profile")
    if partition == "source" and profile == "neq":
        fixed = {given.variable for given in task.givens}
        if any(set(constraint.variables) <= fixed for constraint in task.constraints):
            raise IntegrityFailure("pure source NEQ contains an all-given constraint")
    _check(check_budget)
    oracle = enumerate_solutions(task)
    if work_counters is not None:
        work_counters["oracle_assignments_examined"] = work_counters.get("oracle_assignments_examined", 0) + oracle.assignments_examined
    _check(check_budget)
    scalar_count = _scalar_solution_count(task, check_budget=check_budget, work_counters=work_counters)
    if scalar_count != len(oracle.solutions):
        raise IntegrityFailure("scalar/oracle solution count disagreement")
    solved = solve(task)
    if work_counters is not None:
        work_counters["solver_nodes"] = work_counters.get("solver_nodes", 0) + solved.stats.first_solution_nodes + solved.stats.uniqueness_nodes
    _check(check_budget)
    oracle_first = oracle.solutions[0] if oracle.solutions else None
    if solved.solution != oracle_first:
        raise IntegrityFailure("reference solver and independent oracle disagree")
    given_values = {given.variable: given.value for given in task.givens}
    for solution in (*oracle.solutions, *((solved.solution,) if solved.solution is not None else ())):
        assignment = dict(solution.assignments)
        if any(assignment.get(name) != value for name, value in given_values.items()):
            raise IntegrityFailure("oracle or solver solution violates a given")
        if any(not _satisfies(constraint, assignment) for constraint in task.constraints):
            raise IntegrityFailure("oracle or solver returned a nonsatisfying solution")
    if witness not in oracle.solutions:
        raise IntegrityFailure("constructor witness is not a satisfying oracle solution")
    if scalar_count == 1 and (len(oracle.solutions) != 1 or solved.solution != witness or oracle.solutions[0] != witness):
        raise IntegrityFailure("unique scalar, oracle, solver, and constructor witness identities disagree")
    cert = _independent_interaction_projection(task, partition)
    _check(check_budget)
    if partition == "source":
        independent_removals = []
        for index in range(len(task.constraints)):
            retained = task.constraints[:index] + task.constraints[index + 1:]
            independent_removals.append(_scalar_solution_count(task, constraints=retained,
                check_budget=check_budget, work_counters=work_counters))
        reported_removals = cert.get("removal_solution_counts")
        if independent_removals != reported_removals:
            raise IntegrityFailure("independent per-constraint deletion counts disagree")
        independently_eligible = any(count >= 2 for count in independent_removals)
        cert["independent_removal_solution_counts"] = independent_removals
    else:
        kinds = sorted({constraint.kind.value for constraint in task.constraints})
        independent_removals = []
        for kind in kinds:
            retained = tuple(c for c in task.constraints if c.kind.value != kind)
            independent_removals.append(_scalar_solution_count(task, constraints=retained,
                check_budget=check_budget, work_counters=work_counters))
        if independent_removals != cert.get("kind_removal_solution_counts"):
            raise IntegrityFailure("independent per-kind deletion counts disagree")
        graph = _independent_graph_projection(task)
        if (graph["connected"] != cert.get("factor_graph_connected") or
                graph["participating_kinds"] != cert.get("factor_graph_kinds") or
                graph["all_kinds_participate"] != cert.get("all_present_kinds_in_graph")):
            raise IntegrityFailure("independent factor graph audit disagrees")
        independently_eligible = (all(count >= 2 for count in independent_removals) and
                                  graph["connected"] and graph["all_kinds_participate"])
        cert["independent_kind_removal_solution_counts"] = independent_removals
        cert["independent_factor_graph"] = graph
    if cert.get("eligible") is not independently_eligible:
        raise IntegrityFailure("certificate eligibility disagrees with independent counts and graph")
    producer_eligible = row.get("eligible")
    if producer_eligible is not None and cert["eligible"] != producer_eligible:
        raise IntegrityFailure("producer and independent interaction eligibility disagree")
    # The audit also independently validates certificate deletion semantics.
    reserved = row.get("reserved_constraints", ())
    alternatives = dict(row.get("alternatives", ()))
    expected_kinds = set(profile.split("/"))
    if set(alternatives) != expected_kinds or len(reserved) != len(expected_kinds):
        raise IntegrityFailure("certificate metadata does not cover the registered profile")
    if {item.kind.value for item in reserved} != expected_kinds:
        raise IntegrityFailure("protected constraints do not cover the registered profile")
    for protected in reserved:
        kind = protected.kind.value
        alternative_value = alternatives.get(kind)
        if alternative_value is None:
            raise IntegrityFailure("protected constraint has no corresponding alternative")
        alternative = dict(alternative_value.assignments)
        if alternative_value == witness:
            raise IntegrityFailure("certificate alternative equals witness")
        if any(alternative.get(given.variable) != given.value for given in task.givens):
            raise IntegrityFailure("certificate alternative changes a given")
        if protected not in task.constraints:
            raise IntegrityFailure("protected constraint is absent from task")
        if _satisfies(protected, alternative):
            raise IntegrityFailure("protected constraint does not reject its alternative")
        remaining = (c for c in task.constraints if c.kind.value != kind) if "/" in profile else (
            c for c in task.constraints if c != protected)
        if not all(_satisfies(c, alternative) for c in remaining):
            raise IntegrityFailure("certificate alternative fails a retained constraint")
    return {"oracle_unique": scalar_count == 1,
            "eligible": scalar_count == 1 and bool(cert["eligible"]),
            "sensitive": scalar_count == 1 and partition == "source" and
                         any(count >= 2 for count in cert.get("removal_solution_counts", ())),
            "oracle_solution_count": scalar_count,
            "interaction": cert}


def _scalar_solution_count(task: ConstraintTask, *, constraints: Iterable[Any] | None = None,
                           check_budget: Callable[[], None] | None = None,
                           work_counters: dict[str, int] | None = None) -> int:
    from itertools import product
    fixed = {given.variable: given.value for given in task.givens}
    unknown = tuple(name for name in task.variables if name not in fixed)
    selected = tuple(task.constraints if constraints is None else constraints)
    count = 0
    for values in product((0, 1, 2), repeat=len(unknown)):
        _check(check_budget)
        if work_counters is not None:
            work_counters["independent_scalar_assignments"] = work_counters.get("independent_scalar_assignments", 0) + 1
        assignment = fixed | dict(zip(unknown, values))
        if all(_satisfies(item, assignment) for item in selected):
            count += 1
    return count


def _count_solver_work(work_counters: dict[str, int] | None, result: Any) -> None:
    if work_counters is not None:
        total = result.stats.first_solution_nodes + result.stats.uniqueness_nodes
        work_counters["solver_nodes"] = work_counters.get("solver_nodes", 0) + total


def _task_metrics(task: ConstraintTask, *, check_budget: Callable[[], None] | None = None,
                  work_counters: dict[str, int] | None = None) -> dict[str, Any]:
    from .schema import serialize_model_input
    _check(check_budget)
    solved = solve(task)
    _count_solver_work(work_counters, solved)
    _check(check_budget)
    cid = content_id(task)
    sig = structure_signature(task)
    lid = lineage_id(task)
    _check(check_budget)
    return {"content_id": cid, "lineage_id": lid,
            "structure_signature": sig,
            "source_split": source_split(cid),
            "development_split": development_split(sig),
            "prompt_bytes": len(serialize_model_input(task).encode("utf-8")),
            "first_solution_nodes": solved.stats.first_solution_nodes}


def _outcome_private(outcome: Any, *, check_budget: Callable[[], None] | None = None,
                     work_counters: dict[str, int] | None = None) -> dict[str, Any]:
    task = getattr(outcome, "task", None)
    raw_profile = getattr(outcome, "profile", ())
    profile = "/".join(raw_profile) if isinstance(raw_profile, tuple) else str(raw_profile)
    base = {"partition": getattr(outcome, "partition", "source" if "/" not in profile else "development"),
            "profile": profile, "cell": getattr(outcome, "cell", None),
            "candidate_index": getattr(outcome, "candidate_index", None),
            "rejection_reason": getattr(outcome, "rejection_reason", None),
            "givens": getattr(outcome, "givens", ()), "witness": getattr(outcome, "witness", None),
            "alternatives": getattr(outcome, "alternatives", ()),
            "reserved_constraints": getattr(outcome, "reserved_constraints", ()),
            "work": getattr(outcome, "work", None)}
    if task is None:
        return {**base, "task": None, "oracle_unique": False, "eligible": False,
                "state": "rejected_proposal",
                "rejection_reason": base["rejection_reason"] or "construction_failed",
                "split": None, "content_id": None, "lineage_id": None,
                "structure_signature": None, "envelope": False, "gac": None}
    metric = _task_metrics(task, check_budget=check_budget, work_counters=work_counters)
    eligible = getattr(outcome, "eligible", None)
    partition = "source" if "/" not in profile else "development"
    private = {**base, "partition": partition, "profile": profile,
               "task": serialize_task(task),
               "content_id": metric["content_id"], "lineage_id": metric["lineage_id"],
               "structure_signature": metric["structure_signature"],
               "split": metric["source_split"] if partition == "source" else metric["development_split"],
               "oracle_unique": False, "eligible": eligible,
               "envelope": False,
               "prompt_bytes": metric["prompt_bytes"], "first_solution_nodes": metric["first_solution_nodes"],
               "values": [value for _, value in getattr(outcome, "witness").assignments],
               "given_count": len(task.givens),
               "sensitive": bool(getattr(outcome, "sensitive", False)),
               "gac": None,
               }
    return private


def _prompt_bin(size: int) -> int:
    return 0 if size <= 511 else 1 if size <= 639 else 2 if size <= 767 else 3 if size <= 1023 else 4


def _node_bin(nodes: int) -> int:
    return 0 if nodes == 1 else 1 if nodes <= 3 else 2 if nodes <= 7 else 3 if nodes <= 15 else 4 if nodes <= 31 else 5 if nodes <= 63 else 6


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, ConstraintTask):
        return json.loads(serialize_task(value))
    if isinstance(value, Solution):
        return {"assignments": [[key, val] for key, val in value.assignments]}
    if hasattr(value, "__dataclass_fields__"):
        return {name: _to_jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, RelationKind):
        return value.value
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _source_schedule(partition: str) -> tuple[tuple[str, str], ...]:
    profiles = (("eq",), ("neq",), ("add_mod",)) if partition == "source" else (
        ("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod"))
    return tuple(("/".join(profile), cell) for profile in profiles for cell in CELLS)


def _build_registered_row(outcome: Any, partition: str, index: int, *, guard: Any,
                          seen_content: set[str], lineage_map: dict[str, tuple[str, str]],
                          work_counters: dict[str, int] | None = None) -> dict[str, Any]:
    """Single row-building pipeline for initial persistence and disk replay."""
    from .local_propagation import propagate
    raw = _outcome_private(outcome, check_budget=lambda: _check(guard), work_counters=work_counters)
    raw["partition"] = partition
    raw["candidate_index"] = index
    if raw["task"] is not None:
        task = outcome.task
        raw["task_object"] = task
        audit = _audit_task(raw, partition=partition, check_budget=lambda: _check(guard),
                            work_counters=work_counters)
        raw.update({key: audit[key] for key in ("oracle_unique", "eligible", "sensitive", "oracle_solution_count")})
        raw["private_interaction"] = audit["interaction"]
        # The independent projection enumerates one complete domain for each
        # removed source constraint or each removed mixed kind.
        deletion_calls = len(task.constraints) if partition == "source" else len(set(raw["profile"].split("/")))
        if work_counters is not None:
            work_counters["oracle_assignments_examined"] = work_counters.get("oracle_assignments_examined", 0) + (3 ** len(task.variables)) * deletion_calls
        if not audit["oracle_unique"]:
            raw["rejection_reason"] = "oracle_non_unique"
        raw["envelope"] = (audit["oracle_unique"] and _prompt_bin(raw["prompt_bytes"]) == 4 and
                           _node_bin(raw["first_solution_nodes"]) == 2)
        if not audit["oracle_unique"]:
            raw["state"] = "oracle_non_unique"
        elif not audit["eligible"]:
            raw["state"] = "scientifically_ineligible"
            raw["rejection_reason"] = "scientifically_ineligible"
        elif not raw["envelope"]:
            raw["state"] = "outside_registered_support_envelope"
            raw["rejection_reason"] = "outside_registered_support_envelope"
        else:
            raw["state"] = "accepted_for_support"
            raw["rejection_reason"] = None
        propagation = propagate(task, check_budget=lambda: _check(guard))
        if audit["oracle_unique"] and propagation.contradiction:
            raise IntegrityFailure("oracle-unique task contradicts GAC")
        allowed_split = "source_train" if partition == "source" else "development_validation"
        if work_counters is not None and raw.get("split") == allowed_split:
            work_counters["production_gac_support_checks"] = work_counters.get("production_gac_support_checks", 0) + propagation.support_checks
        independent_gac = None
        if audit["oracle_unique"] and raw.get("split") == allowed_split:
            independent_gac = _audit_local_gac(task, propagation, check_budget=lambda: _check(guard),
                                               work_counters=work_counters)
        raw["gac"] = {"domains": [sorted(domain) for domain in propagation.domains],
                      "solved": propagation.solved, "stalled": propagation.stalled,
                      "contradiction": propagation.contradiction,
                      "unresolved_unknowns": propagation.unresolved_unknowns,
                      "rounds": propagation.rounds, "support_checks": propagation.support_checks,
                      "independent_audit": independent_gac is not None}
        raw.pop("task_object", None)
        identities = _task_metrics(task, check_budget=lambda: _check(guard), work_counters=work_counters)
        expected_split = identities["source_split"] if partition == "source" else identities["development_split"]
        if (raw.get("content_id"), raw.get("lineage_id"), raw.get("structure_signature"), raw.get("split")) != (
                identities["content_id"], identities["lineage_id"], identities["structure_signature"], expected_split):
            raise IntegrityFailure("producer identity differs from canonical identity")
        identity = (identities["content_id"], identities["structure_signature"])
        prior = lineage_map.get(identities["lineage_id"])
        if prior is not None and prior != identity:
            raise IntegrityFailure("lineage closure violation")
        lineage_map[identities["lineage_id"]] = identity
        cid = identities["content_id"]
        if cid in seen_content:
            raw["oracle_unique"] = False
            raw["duplicate_content"] = True
            raw["state"] = "duplicate_content"
            raw["rejection_reason"] = "duplicate_content"
        else:
            seen_content.add(cid)
            if raw.get("split") not in ("source_train", "development_validation"):
                raw["state"] = "retained_diagnostic"
    raw["audit_work"] = dict(sorted((work_counters or {}).items()))
    raw["construction"] = None
    return raw


def _verify_persisted_prefix(stream: Any, start: int, end: int, partition: str, expected_count: int, *,
                             root_seed: int, fixture_rows: list[dict[str, Any]] | None,
                             seen_content: set[str], lineage_map: dict[str, tuple[str, str]],
                             guard: Any) -> dict[str, Any]:
    from .certificate_generator import construct
    _check(guard)
    stream.flush()
    stream.seek(start)
    retained: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    work: Counter[str] = Counter()
    splitless_audit_work: Counter[str] = Counter()
    construction_permitted: Counter[str] = Counter()
    construction_splitless: Counter[str] = Counter()
    allowed_split = "source_train" if partition == "source" else "development_validation"
    digest = hashlib.sha256()
    attempts_rederived = 0
    for index in range(expected_count):
        _check(guard)
        line = stream.readline()
        if not line or not line.endswith(b"\n") or stream.tell() > end:
            raise CandidateFailure(partition, index, "persisted_prefix_row_count", "MalformedPrefix") from None
        try:
            persisted = _strict_load(line.decode("utf-8").rstrip("\n"))
            if persisted.get("partition") != partition or persisted.get("candidate_index") != index:
                raise IntegrityFailure("persisted prefix order or identity mismatch")
            if fixture_rows is None:
                audit_work: dict[str, int] = {}
                outcome = construct(root_seed, partition, index, check_budget=lambda: _check(guard))
                rederived = _build_registered_row(outcome, partition, index, guard=guard,
                    seen_content=seen_content, lineage_map=lineage_map, work_counters=audit_work)
                if _json_bytes(_to_jsonable(rederived)) != line:
                    raise IntegrityFailure("persisted outcome differs from independent construction replay")
                if rederived.get("task") is None:
                    splitless_audit_work.update(audit_work)
                elif rederived.get("split") == allowed_split:
                    work.update(audit_work)
                if rederived.get("task") is None:
                    construction_splitless.update(_work_dict(outcome.work))
                elif rederived.get("split") == allowed_split:
                    construction_permitted.update(_work_dict(outcome.work))
                checked = rederived
                attempts_rederived += 1
            else:
                expected = dict(fixture_rows[index])
                expected.setdefault("partition", partition)
                expected.setdefault("candidate_index", index)
                expected.setdefault("audit_work", {})
                if expected.get("task") is not None:
                    cid = expected.get("content_id")
                    lid = str(expected.get("lineage_id"))
                    identity = (str(cid), str(expected.get("structure_signature")))
                    prior = lineage_map.get(lid)
                    if prior is not None and prior != identity:
                        raise IntegrityFailure("fixture lineage closure violation")
                    lineage_map[lid] = identity
                    if isinstance(cid, str) and cid in seen_content:
                        expected.update({"oracle_unique": False, "duplicate_content": True,
                                         "state": "duplicate_content", "rejection_reason": "duplicate_content"})
                    elif isinstance(cid, str):
                        seen_content.add(cid)
                expected["construction"] = None
                if _json_bytes(_to_jsonable(expected)) != line:
                    raise IntegrityFailure("persisted fixture outcome differs from supplied fixture")
                checked = persisted
        except CandidateFailure:
            raise
        except Exception as exc:
            raise CandidateFailure(partition, index, "persisted_prefix_replay", type(exc).__name__) from None
        digest.update(line)
        if checked.get("task") is None or checked.get("split") == allowed_split:
            counts[str(checked.get("rejection_reason") or checked.get("state") or "unclassified")] += 1
        if checked.get("task") is None and fixture_rows is not None:
            construction_splitless.update(_work_dict(checked.get("work")))
        if checked.get("task") is None or checked.get("split") == allowed_split:
            if checked.get("split") == allowed_split and checked.get("oracle_unique") is True:
                retained.append({key: checked[key] for key in (
                    "content_id", "lineage_id", "structure_signature", "split", "profile", "cell",
                    "eligible", "envelope", "oracle_unique", "sensitive", "values", "given_count",
                    "prompt_bytes", "first_solution_nodes", "gac") if key in checked})
    if stream.tell() != end:
        raise CandidateFailure(partition, expected_count, "persisted_prefix_extra_rows", "MalformedPrefix") from None
    _check(guard)
    return {"permitted_rows": retained, "counts": dict(sorted(counts.items())),
            "actual_permitted_work": dict(sorted(work.items())),
            "splitless_audit_work": dict(sorted(splitless_audit_work.items())),
            "permitted_construction_work": dict(sorted(construction_permitted.items())),
            "splitless_construction_failure_work": dict(sorted(construction_splitless.items())),
            "attempts_rederived": attempts_rederived, "metadata_sha256": digest.hexdigest()}


def _work_dict(value: Any) -> dict[str, int]:
    if hasattr(value, "__dataclass_fields__"):
        value = {name: getattr(value, name) for name in value.__dataclass_fields__}
    if not isinstance(value, dict):
        return {}
    return {str(key): number for key, number in value.items() if type(number) is int and number >= 0}


def _run_prefix(partition: str, attempts: int, output: Path, stream: Any,
                guard: Any, *, root_seed: int = 20261011,
                fixture_rows: list[dict[str, Any]] | None = None,
                seen_content: set[str] | None = None,
                lineage_map: dict[str, tuple[str, str]] | None = None,
                replay_seen_content: set[str] | None = None,
                replay_lineage_map: dict[str, tuple[str, str]] | None = None) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    from .certificate_generator import construct
    seen_content = seen_content if seen_content is not None else set()
    lineage_map = lineage_map if lineage_map is not None else {}
    replay_seen_content = replay_seen_content if replay_seen_content is not None else set()
    replay_lineage_map = replay_lineage_map if replay_lineage_map is not None else {}
    production_work: Counter[str] = Counter()
    splitless_audit_work: Counter[str] = Counter()
    permitted_constructor_work: Counter[str] = Counter()
    splitless_constructor_work: Counter[str] = Counter()
    start_offset = stream.tell()
    attempts_processed = 0
    for index in range(attempts):
        try:
            _check(guard)
        except Exception as exc:
            raise _save_fatal_attempt(stream, partition, index, "resource_guard_before_candidate", exc)
        if fixture_rows is not None and index >= len(fixture_rows):
            break
        raw: dict[str, Any] | None = None
        if fixture_rows is not None:
            raw = dict(fixture_rows[index])
            raw.setdefault("partition", partition)
            raw.setdefault("candidate_index", index)
            raw.setdefault("audit_work", {})
            if raw.get("task") is not None:
                cid = raw.get("content_id")
                lid = str(raw.get("lineage_id"))
                identity = (str(cid), str(raw.get("structure_signature")))
                prior = lineage_map.get(lid)
                if prior is not None and prior != identity:
                    raise _save_fatal_attempt(stream, partition, index, "lineage_closure",
                                              IntegrityFailure("lineage closure violation"), raw)
                lineage_map[lid] = identity
                if isinstance(cid, str) and cid in seen_content:
                    raw.update({"oracle_unique": False, "duplicate_content": True,
                                "state": "duplicate_content", "rejection_reason": "duplicate_content"})
                elif isinstance(cid, str):
                    seen_content.add(cid)
            raw["construction"] = None
        else:
            work_counters: dict[str, int] = {}
            try:
                outcome = construct(root_seed, partition, index, check_budget=lambda: _check(guard))
                raw = _construction_failure_payload(outcome, root_seed, partition, index)
                expected_profile, expected_cell = _source_schedule(partition)[index % len(_source_schedule(partition))]
                if ("/".join(outcome.profile), outcome.cell) != (expected_profile, expected_cell):
                    raise IntegrityFailure("constructor changed the fixed profile/cell schedule")
                raw = _build_registered_row(outcome, partition, index, guard=guard,
                    seen_content=seen_content, lineage_map=lineage_map, work_counters=work_counters)
            except Exception as exc:
                raise _save_fatal_attempt(stream, partition, index, "candidate_build", exc, raw)
        payload = _json_bytes(_to_jsonable(raw))
        stream.write(payload)
        stream.flush()
        attempts_processed += 1
        if fixture_rows is None:
            allowed_split = "source_train" if partition == "source" else "development_validation"
            if raw.get("task") is None:
                splitless_audit_work.update(work_counters)
            elif raw.get("split") == allowed_split:
                production_work.update(work_counters)
            constructor_work = _work_dict(raw.get("work"))
            if raw.get("task") is None:
                splitless_constructor_work.update(constructor_work)
            elif raw.get("split") == allowed_split:
                permitted_constructor_work.update(constructor_work)
        try:
            _check(guard)
        except Exception as exc:
            raise CandidateFailure(partition, index, "resource_guard_after_candidate", type(exc).__name__) from None
    end_offset = stream.tell()
    verification = _verify_persisted_prefix(
        stream, start_offset, end_offset, partition, attempts_processed, root_seed=root_seed,
        fixture_rows=fixture_rows[:attempts_processed] if fixture_rows is not None else None,
        seen_content=replay_seen_content, lineage_map=replay_lineage_map, guard=guard)
    allowed = "source_train" if partition == "source" else "development_validation"
    all_actual_work: Counter[str] = Counter(production_work)
    all_actual_work.update(verification["actual_permitted_work"])
    permitted_constructor_work.update(verification["permitted_construction_work"])
    splitless_constructor_work.update(verification["splitless_construction_failure_work"])
    permitted_rows = verification["permitted_rows"]
    aggregates = permitted_rows
    counts = verification["counts"]
    return aggregates, dict(sorted(counts.items())), {
        "status": "PASS_REPLAY" if fixture_rows is None else "FIXTURE_PREFIX_CHECKED",
        "attempts_rederived": verification["attempts_rederived"],
        "attempts_processed": attempts_processed, "stage_start_offset": start_offset,
        "stage_end_offset": end_offset, "rederived_metadata_sha256": verification["metadata_sha256"],
        "permitted_construction_work": dict(sorted(permitted_constructor_work.items())),
        "splitless_construction_failure_work": dict(sorted(splitless_constructor_work.items())),
        "actual_permitted_work": dict(sorted(all_actual_work.items())),
        "splitless_audit_work": dict(sorted((Counter(splitless_audit_work) +
                                              Counter(verification["splitless_audit_work"])).items())),
        "independent_audit_rows": attempts_processed if fixture_rows is None else 0}


def _source_gate(rows: list[dict[str, Any]], construction_counts: dict[str, int], config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    rows = _first_seen_rows(rows)
    result = aggregate_permitted_rows(rows, partition="source")
    train = [row for row in rows if row.get("split") == "source_train"]
    sensitive = [row for row in train if row.get("sensitive") is True]
    env = [row for row in sensitive if row.get("eligible") is True and row.get("envelope") is True]
    floors = config["source_floors"]
    sigs = {row["structure_signature"] for row in env}
    cell_counts = Counter((row["profile"], row["cell"]) for row in env)
    all_cell_counts = Counter(row["cell"] for row in env)
    profile_sig_counts = {profile: len({r["structure_signature"] for r in env if r["profile"] == profile})
                          for profile in config["source_profiles"]}
    attempted = config["attempts"]["source"]
    value_counts = Counter(value for row in env for value in row.get("values", ()))
    value_total = sum(value_counts.values())
    per_signature_sensitive = defaultdict(int)
    for row in env:
        if row.get("sensitive") is True:
            per_signature_sensitive[row["structure_signature"]] += 1
    accepted_signatures = {row["structure_signature"] for row in env}
    sensitive_fraction = len(sensitive) / len(train) if train else 0.0
    envelope_fraction = len(env) / len(train) if train else 0.0
    checks = {
        "train_groups": len(env) >= floors["train_groups"],
        "train_signatures": len(sigs) >= floors["train_signatures"],
        "primitives": len({k for row in env for k in row.get("profile", "").split("/")}) >= floors["primitives"],
        "cell_min": all(all_cell_counts[cell] >= config["support"]["cell_min"] for cell in CELLS),
        "profile_cell_capacity": all(cell_counts[p, c] >= floors["eligible_envelope_train_per_profile_cell"] for p in config["source_profiles"] for c in CELLS),
        "profile_signatures": all(profile_sig_counts[p] >= floors["accepted_signatures_per_profile"] for p in config["source_profiles"]),
        "raw_yield": len(env) / attempted >= floors["eligible_envelope_train_per_all_raw_attempts"] if attempted else False,
        "value_mass": value_total > 0 and all(value_counts[value] / value_total >= floors["value_fraction_each"] for value in (0, 1, 2)),
        "given_counts": all(any(row.get("given_count") == count for row in env) for count in floors["given_counts"]),
        "sensitivity": sensitive_fraction >= floors["sensitivity"],
        "eligible_envelope_fraction": envelope_fraction >= floors["eligible_envelope_fraction"],
        "sensitive_signatures": all(per_signature_sensitive[sig] >= floors["sensitive_per_accepted_signature"] for sig in accepted_signatures),
    }
    # All-eligible denominator for common support; the support cells remain fixed.
    eligible_support = [r for r in sensitive if r.get("envelope") is True and
                        _prompt_bin(r.get("prompt_bytes", 0)) == 4 and
                        _node_bin(r.get("first_solution_nodes", 0)) == 2]
    support_mass = len(eligible_support) / len(sensitive) if sensitive else 0.0
    checks["common_support_mass"] = support_mass >= config["support"]["cohort_mass_min"]
    result.update({"floors": checks, "eligible_envelope_train_groups_per_raw_attempt": len(env) / attempted if attempted else 0.0,
                   "common_support_mass": support_mass, "profile_signatures": profile_sig_counts,
                   "first_seen_oracle_train_groups": len(train), "sensitive_train_groups": len(sensitive),
                   "eligible_envelope_train_groups": len(env),
                   "sensitivity_fraction": sensitive_fraction,
                   "eligible_envelope_fraction": envelope_fraction,
                   "profile_groups": {p: sum(row["profile"] == p for row in env) for p in config["source_profiles"]},
                   "solution_value_counts": {str(value): value_counts[value] for value in (0, 1, 2)},
                   "solution_value_fractions": {str(value): value_counts[value] / value_total if value_total else 0.0
                                                for value in (0, 1, 2)},
                   "given_counts": {str(count): sum(row.get("given_count") == count for row in env)
                                    for count in floors["given_counts"]},
                   "sensitive_accepted_signatures": {sig: per_signature_sensitive[sig]
                                                      for sig in sorted(accepted_signatures)},
                   "support_cell_counts": {cell: all_cell_counts[cell] for cell in CELLS},
                   "profile_cell_counts": {f"{p}/{c}": cell_counts[p, c] for p in config["source_profiles"] for c in CELLS},
                   "gac_descriptive": _gac_profile_summary(train, config["source_profiles"]),
                   "gac_denominator_set": "first-seen oracle-unique source_train groups; descriptive only"})
    return all(checks.values()), result


def difficulty_counts(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Use scientifically eligible validation groups as the registered GAC denominator."""
    eligible = [row for row in _first_seen_rows(rows) if row.get("oracle_unique") is True and row.get("eligible") is True]
    stalled = [row for row in eligible if (row.get("gac") or {}).get("stalled") is True]
    groups = {row["content_id"] for row in stalled}
    return {"eligible_denominator": len({row["content_id"] for row in eligible}),
            "stalled_groups": len(groups),
            "stalled_fraction": len(groups) / len({row["content_id"] for row in eligible}) if eligible else 0.0,
            "profiles_with_stall": sorted({row["profile"] for row in stalled})}


def _gac_profile_summary(rows: Iterable[dict[str, Any]], profiles: Iterable[str]) -> dict[str, Any]:
    """Describe GAC only for the explicitly supplied permitted unique cohorts."""
    unique = _first_seen_rows(rows)
    result: dict[str, Any] = {}
    for profile in profiles:
        members = [row for row in unique if row.get("profile") == profile]
        measured = [row["gac"] for row in members if isinstance(row.get("gac"), dict)]
        def values(key: str) -> list[int]:
            return [entry[key] for entry in measured if type(entry.get(key)) is int]
        unresolved = values("unresolved_unknowns")
        rounds = values("rounds")
        support = values("support_checks")
        stalled_groups = sum(entry.get("stalled") is True for entry in measured)
        result[profile] = {
            "oracle_unique_group_denominator": len(members),
            "gac_measured_groups": len(measured),
            "solved_groups": sum(entry.get("solved") is True for entry in measured),
            "stalled_groups": stalled_groups,
            "zero_stalls": stalled_groups == 0,
            "unresolved_unknowns": {"sum": sum(unresolved), "min": min(unresolved, default=None),
                                    "max": max(unresolved, default=None),
                                    "mean": sum(unresolved) / len(unresolved) if unresolved else None},
            "rounds": {"sum": sum(rounds), "min": min(rounds, default=None),
                       "max": max(rounds, default=None),
                       "mean": sum(rounds) / len(rounds) if rounds else None},
            "local_support_work": {"support_checks_sum": sum(support),
                                   "support_checks_mean": sum(support) / len(support) if support else None},
        }
    return result


def _development_gate(rows: list[dict[str, Any]], config: dict[str, Any], *,
                       source_rows: list[dict[str, Any]] | None = None) -> tuple[bool, str, dict[str, Any]]:
    rows = _first_seen_rows(rows)
    eligible = [row for row in rows if row.get("oracle_unique") is True and row.get("eligible") is True]
    env = [row for row in eligible if row.get("envelope") is True]
    floors = config["development_floors"]
    profiles = config["development_profiles"]
    env_counts = Counter(r["profile"] for r in env)
    profile_unique_counts = {p: sum(r.get("profile") == p for r in rows) for p in profiles}
    profile_eligible_counts = {p: sum(r.get("profile") == p and r.get("eligible") is True for r in rows)
                               for p in profiles}
    profile_envelope_counts = {p: sum(r.get("profile") == p and r.get("eligible") is True and
                                      r.get("envelope") is True for r in rows) for p in profiles}
    profile_eligibility_fractions = {p: profile_eligible_counts[p] / profile_unique_counts[p]
                                     if profile_unique_counts[p] else 0.0 for p in profiles}
    profile_envelope_fractions = {p: profile_envelope_counts[p] / profile_unique_counts[p]
                                  if profile_unique_counts[p] else 0.0 for p in profiles}
    eligible_signatures = {p: {r["structure_signature"] for r in eligible if r["profile"] == p} for p in profiles}
    envelope_signatures = {p: {r["structure_signature"] for r in env if r["profile"] == p} for p in profiles}
    checks = {
        "envelope_groups": all(env_counts[p] >= floors["eligible_envelope_groups_per_profile"] for p in profiles),
        "signatures": all(len(eligible_signatures[p]) >= floors["eligible_signatures_per_profile"] for p in profiles),
        "represented_signatures": all(
            sum(r["structure_signature"] == sig for r in env if r["profile"] == p) >=
            floors["eligible_per_represented_eligible_signature"]
            for p in profiles for sig in eligible_signatures[p]),
        "scientific_eligibility": len(eligible) / len(rows) >= floors["scientific_eligibility_fraction"] if rows else False,
        "envelope_fraction": len(env) / len(rows) >= floors["eligible_envelope_fraction"] if rows else False,
        "profile_scientific_eligibility": all(profile_unique_counts[p] > 0 and
            profile_eligibility_fractions[p] >= floors["scientific_eligibility_fraction"] for p in profiles),
        "profile_envelope_fraction": all(profile_unique_counts[p] > 0 and
            profile_envelope_fractions[p] >= floors["eligible_envelope_fraction"] for p in profiles),
    }
    difficulty = difficulty_counts(rows)
    stalled = [r for r in eligible if (r.get("gac") or {}).get("stalled") is True]
    stalled_profiles = set(difficulty["profiles_with_stall"])
    diff = config["difficulty"]
    difficulty_pass = (difficulty["stalled_groups"] >= diff["development_stalled_groups_min"] and
                       difficulty["stalled_fraction"] >= diff["development_stalled_fraction_min"] and
                       len(stalled_profiles) >= diff["development_profiles_with_stall_min"] if eligible else False)
    support_rows = {"source_train": [r for r in (source_rows or [])
                                      if r.get("oracle_unique") is True and r.get("eligible") is True]}
    support_rows.update({f"development_validation_{profile}": [r for r in eligible if r["profile"] == profile]
                    for profile in profiles}
    )
    support = common_support_mass(support_rows, minimum_per_cell=config["support"]["cell_min"])
    checks["common_support_cell_floor"] = all(item["cell_floor_pass"] for item in support.values())
    checks["common_support_mass"] = all(item["common_cell_mass"] >= config["support"]["cohort_mass_min"] for item in support.values())
    checks["local_difficulty"] = difficulty_pass
    status = "PASS_CAPACITY_DIFFICULTY" if all(checks.values()) else "NO_GO_LOCAL_DIFFICULTY" if not difficulty_pass else "NO_GO_DEVELOPMENT_CAPACITY"
    per_profile = {}
    for profile in profiles:
        profile_rows = [r for r in rows if r["profile"] == profile]
        profile_eligible = [r for r in profile_rows if r.get("eligible") is True]
        profile_env = [r for r in profile_eligible if r.get("envelope") is True]
        profile_stalled = [r for r in profile_eligible if (r.get("gac") or {}).get("stalled") is True]
        unique_groups = len(profile_rows)
        eligible_ids = {r["content_id"] for r in profile_eligible}
        stalled_ids = {r["content_id"] for r in profile_stalled}
        per_signature_env = Counter(r["structure_signature"] for r in profile_env)
        per_profile[profile] = {
            "first_seen_oracle_unique_groups": unique_groups,
            "scientifically_eligible_groups": len(eligible_ids),
            "scientific_eligibility_fraction": profile_eligibility_fractions[profile],
            "eligible_envelope_groups": len(profile_env),
            "eligible_envelope_yield": profile_envelope_fractions[profile],
            "eligible_signatures": len(eligible_signatures[profile]),
            "eligible_envelope_signatures": len(envelope_signatures[profile]),
            "eligible_per_signature_min": min(per_signature_env.values(), default=0),
            "stalled_groups": len(stalled_ids),
            "eligible_groups_by_cell": {cell: sum(r.get("cell") == cell for r in profile_eligible) for cell in CELLS},
            "eligible_envelope_groups_by_cell": {cell: sum(r.get("cell") == cell for r in profile_env) for cell in CELLS},
        }
    result = {"floors": checks, "eligible_groups": len(eligible), "eligible_envelope_groups": len(env),
              "stalled_groups": len({r["content_id"] for r in stalled}),
              "stalled_fraction": difficulty["stalled_fraction"],
              "difficulty_denominator": difficulty["eligible_denominator"],
              "stalled_profiles": sorted(stalled_profiles), "common_support": support,
              "profile_envelope_groups": {p: env_counts[p] for p in profiles},
              "profile_signature_counts": {p: len(eligible_signatures[p]) for p in profiles},
              "profile_envelope_signature_counts": {p: len(envelope_signatures[p]) for p in profiles},
              "profile_results": per_profile,
              "gac_denominator_set": "first-seen oracle-unique development_validation groups by registered profile",
              "gac_by_profile": _gac_profile_summary(rows, profiles),
              "first_seen_oracle_validation_groups": len(rows)}
    return all(checks.values()), status, result


def _first_seen_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    first: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("oracle_unique") is True and isinstance(row.get("content_id"), str):
            first.setdefault(row["content_id"], row)
    return list(first.values())


def _make_initial_manifest(config: LoadedStudyConfig, output: Path, argv: list[str], source: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {"manifest_version": "dwr-certificate-sixvar-initial-v1", "status": "running",
            "created_utc": now, "command": list(getattr(sys, "orig_argv", ())) or [sys.executable, *argv],
            "requested_argv": argv, "executable": sys.executable,
            "implementation": platform.python_implementation(), "python_version": list(sys.version_info[:3]),
            "cwd": str(Path.cwd()),
            "output_dir": str(output.resolve()), "config_path": str(config.path),
            "config_sha256": config.sha256, "resolved_config": config.resolved,
            "source": source, "python": sys.version, "platform": platform.platform(),
            "precision": "standard-library CPU; no ML tensors", "workers": 1,
            "seed_derivation": config.resolved["rng"], "independent_streams": ["source", "development"],
            "budgets": config.resolved["limits"], "artifact_names": list(ARTIFACTS)}


def run_study(config_path: str | Path, output_path: str | Path, *, argv: list[str] | None = None,
              guard: Any | None = None, fixture_rows: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Execute the sole registered study. fixture_rows is test-only and cannot PASS."""
    invocation_started = time.monotonic()
    config = load_study_config(config_path)
    requested_output = Path(output_path).absolute()
    output = requested_output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output path: {output}")
    if fixture_rows is None:
        run_root = Path(__file__).resolve().parents[2] / "runs"
        try:
            output.relative_to(run_root.resolve())
        except ValueError as exc:
            raise ValueError("production output must be under runs/") from exc
        if not output.name.startswith("certificate-sixvar-"):
            raise ValueError("production output directory must use certificate-sixvar-* name")
        for parent in (requested_output, *requested_output.parents):
            if parent.exists() and parent.is_symlink():
                raise ValueError("output path contains a symlink")
        if guard is None:
            limits = config.resolved["limits"]
            guard = ResourceGuard(limits["active_wall_seconds"],
                                  limits["max_rss_bytes"] // (1024 * 1024),
                                  memory_reader=resident_bytes_windows)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    cpu_start = time.process_time()
    report: dict[str, Any] = {"status": "NOT_ASSESSED", "g2_status": "not_assessed",
                              "preflight_emitted": False, "fixture_only": fixture_rows is not None,
                              "config_sha256": config.sha256}
    try:
        source = _source_identity(guard)
        initial = _make_initial_manifest(config, output, argv or sys.argv, source)
        _atomic_write(output / "manifest.initial.json", initial, guard)
        runtime = _require_supported_runtime()
        report["runtime"] = runtime
        report["source_identity"] = source
        report["root_seed"] = config.resolved["root_seed"]
        report["rng"] = config.resolved["rng"]
        report["accounting_boundary"] = "prefix processing, independent replay/audit, and terminal artifact hashing"
        seen_content: set[str] = set()
        lineage_map: dict[str, tuple[str, str]] = {}
        replay_seen_content: set[str] = set()
        replay_lineage_map: dict[str, tuple[str, str]] = {}
        with (output / "outcomes.jsonl").open("w+b") as stream:
            source_rows, source_counts, source_replay = _run_prefix("source", config.resolved["attempts"]["source"], output,
                                                     stream, guard, root_seed=config.resolved["root_seed"],
                                                     fixture_rows=(fixture_rows.get("source", []) if fixture_rows is not None else None),
                                                     seen_content=seen_content, lineage_map=lineage_map,
                                                     replay_seen_content=replay_seen_content,
                                                     replay_lineage_map=replay_lineage_map)
            _check(guard)
            _check(guard)
            source_complete = source_replay.get("attempts_processed") == config.resolved["attempts"]["source"]
            if fixture_rows is not None and not source_complete:
                report["source"] = {"status": "not_assessed_fixture_prefix_incomplete",
                                     "attempts_processed": source_replay.get("attempts_processed"),
                                     "registered_attempts": config.resolved["attempts"]["source"]}
                report["source_construction_counts"] = source_counts
                report["status"] = "NOT_ASSESSED_FIXTURE_ONLY"
                report["development"] = {"status": "not_attempted", "attempts_processed": 0,
                                          "reason": "source fixture prefix incomplete"}
                source_pass = False
                source_report = report["source"]
            else:
                source_pass, source_report = _source_gate(source_rows, source_counts, config.resolved)
                report["source"] = source_report
                report["source_construction_counts"] = source_counts
            replay_stages = {"source": source_replay}
            if not source_pass and report["status"] != "NOT_ASSESSED_FIXTURE_ONLY":
                report["status"] = "NO_GO_SOURCE_CAPACITY"
                report["development"] = {"status": "not_attempted", "attempts_processed": 0,
                                          "reason": "source capacity gate failed"}
            elif source_pass:
                dev_rows, dev_counts, dev_replay = _run_prefix("development", config.resolved["attempts"]["development"],
                                                   output, stream, guard, root_seed=config.resolved["root_seed"],
                                                   fixture_rows=(fixture_rows.get("development", []) if fixture_rows is not None else None),
                                                   seen_content=seen_content, lineage_map=lineage_map,
                                                   replay_seen_content=replay_seen_content,
                                                   replay_lineage_map=replay_lineage_map)
                replay_stages["development"] = dev_replay
                _check(guard)
                dev_pass, dev_status, dev_report = _development_gate(dev_rows, config.resolved,
                                                                     source_rows=source_rows)
                _check(guard)
                report["development"] = dev_report
                report["development_construction_counts"] = dev_counts
                report["status"] = dev_status if dev_pass or dev_status else "NO_GO_DEVELOPMENT_CAPACITY"
                dev_complete = dev_replay.get("attempts_processed") == config.resolved["attempts"]["development"]
                if fixture_rows is not None:
                    report["status"] = "NOT_ASSESSED_FIXTURE_ONLY"
                    if not dev_complete:
                        report["development"]["status"] = "not_assessed_fixture_prefix_incomplete"
            if fixture_rows is not None:
                report["status"] = "NOT_ASSESSED_FIXTURE_ONLY"
        report["elapsed_seconds"] = time.monotonic() - start
        report["cpu_seconds"] = time.process_time() - cpu_start
        report["peak_rss_bytes"] = (getattr(guard, "peak_bytes", None) if guard is not None else None)
        if report["peak_rss_bytes"] is None:
            report["peak_rss_bytes"] = resident_bytes_windows()
        report["whole_wrapper_seconds_at_report"] = time.monotonic() - invocation_started
        report["g2_status"] = "not_assessed"
        report["preflight_emitted"] = False
        _atomic_write(output / "replay.json", {"status": "PASS_REPLAY" if fixture_rows is None else "FIXTURE_REPLAY",
                                                "stages": replay_stages,
                                                "attempts_rederived": sum(stage["attempts_rederived"] for stage in replay_stages.values())}, guard)
        _atomic_write(output / "report.json", report, guard)
        finalize_artifacts(output, guard=guard, wall_start=start, cpu_start=cpu_start,
                           invocation_start=invocation_started)
        return _strict_load((output / "report.json").read_text(encoding="utf-8"))
    except Exception as exc:
        safe_context = (f"candidate failure during {exc.phase} ({exc.partition} index {exc.candidate_index})"
                        if isinstance(exc, CandidateFailure) else f"study failure ({type(exc).__name__})")
        report.update({"status": "NOT_ASSESSED", "terminal_reason": safe_context,
                       "g2_status": "not_assessed", "preflight_emitted": False,
                       "elapsed_seconds": time.monotonic() - start,
                       "cpu_seconds": time.process_time() - cpu_start,
                       "whole_wrapper_seconds_before_accounting_flush": time.monotonic() - invocation_started,
                       "accounting_boundary": "failure_report_before_partial_manifest_flush",
                       "accounting_excludes": "partial report hash and bounded manifest flush"})
        if isinstance(exc, CandidateFailure):
            report["failure"] = {"stage": exc.partition, "candidate_index": exc.candidate_index,
                                 "phase": exc.phase, "error_type": exc.cause_type}
        if guard is not None:
            report["peak_rss_bytes"] = getattr(guard, "peak_bytes", None)
        try:
            _atomic_write(output / "report.json", report)
            if not (output / "outcomes.jsonl").exists():
                (output / "outcomes.jsonl").write_bytes(b"")
            if not (output / "replay.json").exists():
                _atomic_write(output / "replay.json", {"status": "NOT_ASSESSED", "reason": "partial run"})
            # Do not rehash the potentially large outcome prefix after a cap or
            # I/O failure. Reuse an existing bounded manifest and refresh only
            # the small report hash; otherwise record an explicitly partial one.
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = _strict_load(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    manifest = {}
                hashes = manifest.get("artifacts", {})
                if "report.json" in hashes:
                    hashes["report.json"] = _hash(output / "report.json")
                manifest.update({"status": "NOT_ASSESSED", "artifacts": hashes,
                                 "g2_status": "not_assessed", "preflight_emitted": False,
                                 "terminal_reason": report["terminal_reason"]})
            else:
                manifest = {"manifest_version": "dwr-certificate-sixvar-run-v1", "status": "NOT_ASSESSED",
                            "source": None, "config_sha256": config.sha256,
                            "root_seed": config.resolved["root_seed"], "rng": config.resolved["rng"],
                            "resources": {"elapsed_seconds": report["elapsed_seconds"],
                                          "cpu_seconds": report["cpu_seconds"],
                                          "peak_rss_bytes": report.get("peak_rss_bytes")},
                            "accounting": {}, "artifacts": {}, "g2_status": "not_assessed",
                            "preflight_emitted": False, "terminal_reason": report["terminal_reason"]}
            (manifest_path).write_bytes(_json_bytes(manifest))
        except Exception:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_study_config(args.config)
    except Exception as exc:
        print(f"NOT_ASSESSED: configuration admission failed ({type(exc).__name__})", file=sys.stderr)
        return 2
    limits = config.resolved["limits"]
    guard = ResourceGuard(limits["active_wall_seconds"], limits["max_rss_bytes"] // (1024 * 1024),
                          memory_reader=resident_bytes_windows)
    try:
        report = run_study(args.config, args.output, argv=sys.orig_argv, guard=guard)
    except Exception as exc:
        print(f"NOT_ASSESSED: study execution failed ({type(exc).__name__})", file=sys.stderr)
        return 2
    print(stable_json(report))
    return 0 if report["status"] == "PASS_CAPACITY_DIFFICULTY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
