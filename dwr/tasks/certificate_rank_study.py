"""Bounded, versioned qualification and full G2 runner for rank certificates."""
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import certificate_study as old
from . import certificate_rank_generator as rank_generator
from . import certificate_rank_audit as audit
from . import certificate_generator as legacy_generator
from . import certificate_generator_v2 as v2_generator
from .canonical import content_id, development_split, lineage_id, source_split, structure_signature
from .local_propagation import propagate
from .oracle import enumerate_solutions
from .run_support import ResourceGuard, resident_bytes_windows, stable_json
from .schema import ConstraintTask, Solution, serialize_model_input, serialize_task
from .solver import solve
from .verifier import verify_answer

ARTIFACTS = ("manifest.initial.json", "outcomes.jsonl", "records.jsonl", "replay.json", "report.json", "manifest.json")
CONFIG_HASHES = {
    "qualification": "8fa592fe2bedc012a1a10fb5ecd74bbd871df2f0cedcf5026562f5b518c2d601",
    "full": "c01056ece9e4d63981301b2cff2c224df952499a54d8b746ac1d73b2df15be6f",
}
FIVE_CONFIG_HASHES = {
    "qualification": "d66ad33da9111287223e7e94ab849e8b710692db32f71f94410f305ce1cf1002",
    "full": "3aff6fcaba6143a26ad6d8a86cc83f9782ff63d689912f5fb04ffc4e9cc81afa",
}


class ConfigError(ValueError):
    pass


class StudyError(RuntimeError):
    pass


def _strict_load(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ConfigError("duplicate config key")
            result[key] = value
        return result
    def constant(_):
        raise ConfigError("non-finite config number")
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except ConfigError:
        raise
    except (ValueError, TypeError) as exc:
        raise ConfigError("malformed config") from exc
    def finite(item):
        if type(item) is float and not math.isfinite(item):
            raise ConfigError("non-finite config number")
        if isinstance(item, dict):
            for child in item.values(): finite(child)
        elif isinstance(item, list):
            for child in item: finite(child)
    finite(value)
    return value


def _same_typed(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected): return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_same_typed(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_typed(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _registered_path(mode: str, *, five_stratum: bool = False) -> Path:
    stem = "certificate_five_stratum" if five_stratum else "certificate_rank"
    return Path(__file__).resolve().parents[2] / "configs" / "tasks" / f"{stem}_{mode}_v1.json"


def load_config(path: str | Path) -> tuple[dict, str, Path]:
    p = Path(path).resolve(strict=True)
    raw = p.read_bytes()
    try:
        value = _strict_load(raw.decode("utf-8"))
        # Mode is admitted only after a strict parse; it selects one closed baseline.
        mode = value.get("mode") if type(value) is dict else None
        five = value.get("config_version") == "dwr-five-stratum-rank-study-v1"
        registry = FIVE_CONFIG_HASHES if five else CONFIG_HASHES
        expected_hash = registry.get(mode)
        if expected_hash is None: raise ConfigError("unknown registered mode")
        expected_raw = _registered_path(mode, five_stratum=five).read_bytes()
        expected = _strict_load(expected_raw.decode("utf-8"))
    except (UnicodeError, OSError) as exc:
        raise ConfigError("cannot read registered config") from exc
    if hashlib.sha256(expected_raw).hexdigest() != expected_hash or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ConfigError("config bytes differ from registered contract")
    if not _same_typed(value, expected):
        raise ConfigError("config differs from closed typed contract")
    return value, expected_hash, p


def _check(guard):
    if guard is not None: guard.check()


def _json_bytes(value):
    return (stable_json(old._to_jsonable(value)) + "\n").encode("utf-8")


def _a_projection(row):
    """The A-only ledger is an explicit model-input-free public projection."""
    fields = ("partition", "candidate_index", "profile", "cell", "branch", "split", "content_id",
              "lineage_id", "structure_signature", "oracle_unique", "eligible", "sensitive", "envelope",
              "values", "given_count", "prompt_bytes", "first_solution_nodes", "model_input_sha256",
              "gac", "core", "config_sha256")
    projection = {key: row[key] for key in fields if key in row}
    if "fine_stratum" in row:
        projection["fine_stratum"] = row["fine_stratum"]
    return projection


def _is_A(row, config=None):
    """True only for a first-seen, unique, eligible support-envelope row."""
    if config is not None and config.get("support", {}).get("scheme") == "five-fixed-strata-v1":
        if not audit.row_schema_valid(row, config):
            return False
    if not all(row.get(key) is True for key in
               ("first_seen_content", "oracle_unique", "eligible", "envelope")):
        return False
    if config is not None and config.get("support", {}).get("scheme") == "five-fixed-strata-v1":
        return (row.get("fine_stratum") in {"A2", "A3", "B2", "C2", "D2"} and
                row.get("fine_stratum") == audit.support_stratum(row, config))
    return True


def _exact_true_floor_map(actual, expected_keys):
    return (type(actual) is dict and bool(expected_keys) and set(actual) == set(expected_keys) and
            all(type(value) is bool and value is True for value in actual.values()))


def _count_A_for_quota(row, profile, counts, config=None):
    if _is_A(row, config):
        counts[profile] += 1
        return True
    return False


def _replay_matches(saved_rows, rebuilt_rows, raw_bytes, record_bytes, config_hash, config=None):
    if len(saved_rows) != len(rebuilt_rows): return False
    if config is not None and config.get("support", {}).get("scheme") == "five-fixed-strata-v1":
        if any(not audit.row_schema_valid(row, config)
               for row in (*saved_rows, *rebuilt_rows)):
            return False
    full_rows = [{**row, "config_sha256": config_hash} for row in rebuilt_rows]
    if any(_json_bytes(expected) != _json_bytes(actual)
           for actual, expected in zip(saved_rows, full_rows)):
        return False
    expected_raw = b"".join(_json_bytes(row) for row in saved_rows)
    if raw_bytes != expected_raw: return False
    projections = []
    for row in full_rows:
        if _is_A(row, config):
            payload = _json_bytes(row)
            projection = _a_projection(row)
            projection["outcome_sha256"] = hashlib.sha256(payload).hexdigest()
            projections.append(_json_bytes(projection))
    return record_bytes == b"".join(projections)


def _revoke_terminal(output, config_hash, report, exc):
    """Write partial evidence and revoke terminal success without rescanning ledgers."""
    for name in ("outcomes.jsonl", "records.jsonl"):
        if not (output / name).exists(): (output / name).write_bytes(b"")
    if not (output / "manifest.initial.json").exists():
        _write(output / "manifest.initial.json", {"status": "NOT_ASSESSED", "config_sha256": config_hash}, None)
    _write(output / "replay.json", {"status": "NOT_ASSESSED", "reason": type(exc).__name__}, None)
    report["status"] = "NOT_ASSESSED"
    report["g2_status"] = "not_assessed"
    report["terminal_reason"] = type(exc).__name__
    report["artifacts"] = {}
    report["artifact_hashes_valid"] = False
    _write(output / "report.json", report, None)
    _write(output / "manifest.json", {"status": "NOT_ASSESSED", "g2_status": "not_assessed",
                                      "config_sha256": config_hash,
                                      "artifacts": {}, "artifact_hashes_valid": False,
                                      "terminal_reason": type(exc).__name__}, None)


def _g2_status(status, production, replay_ok=True):
    """Expose G2 only when the complete production audit and replay are trustworthy."""
    if not production or not replay_ok:
        return "not_assessed"
    if status in ("PASS_G2_RANK_V1", "PASS_G2_FIVE_STRATUM_V1"):
        return "pass"
    if status == "NO_GO_FULL_AUDIT":
        return "fail"
    return "not_assessed"


def _audit_ordered_solutions(task, scalar_solutions, oracle_solutions, solved, witness, guard, counters):
    """Independent ordered scalar/oracle/solver and production-verifier agreement."""
    scalar = tuple(scalar_solutions)
    oracle = tuple(oracle_solutions)
    _check(guard)
    if scalar != oracle:
        raise StudyError("ordered scalar solutions differ from exhaustive oracle")
    expected_first = oracle[0] if oracle else None
    if solved.solution != expected_first or solved.stats.uniqueness_solution_count_capped != min(2, len(scalar)):
        raise StudyError("solver first solution or capped uniqueness count differs")
    for solution in oracle:
        _check(guard)
        result = verify_answer(task, stable_json({"assignment": dict(solution.assignments)}))
        counters["verifier_calls"] = counters.get("verifier_calls", 0) + 1
        if not result.valid: raise StudyError("production verifier rejected an oracle solution")
    if witness is not None:
        _check(guard)
        result = verify_answer(task, stable_json({"assignment": dict(witness.assignments)}))
        counters["verifier_calls"] = counters.get("verifier_calls", 0) + 1
        if not result.valid: raise StudyError("production verifier rejected constructor witness")
    return {"solution_count": len(scalar), "solver_first_matches": True,
            "solver_uniqueness_capped": solved.stats.uniqueness_solution_count_capped}


def _scalar_solutions(task, guard, counters):
    from itertools import product
    fixed = {given.variable: given.value for given in task.givens}
    unknown = tuple(name for name in task.variables if name not in fixed)
    solutions = []
    for values in product((0, 1, 2), repeat=len(unknown)):
        _check(guard)
        counters["scalar_assignments"] = counters.get("scalar_assignments", 0) + 1
        assignment = fixed | dict(zip(unknown, values))
        if all(old._satisfies(rule, assignment) for rule in task.constraints):
            solutions.append(Solution(tuple((name, assignment[name]) for name in task.variables)))
    return tuple(solutions)


def _write(path: Path, value, guard):
    wall_begin, cpu_begin = time.monotonic(), time.process_time()
    _check(guard)
    payload = _json_bytes(value)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _check(guard)
        delays = (0.01, 0.02, 0.04)
        for attempt in range(4):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if os.name != "nt" or getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 3:
                    raise
                time.sleep(delays[attempt])
        _check(guard)
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass
        if guard is not None:
            guard.artifact_write_wall_seconds = getattr(guard, "artifact_write_wall_seconds", 0.0) + (time.monotonic() - wall_begin)
            guard.artifact_write_cpu_seconds = getattr(guard, "artifact_write_cpu_seconds", 0.0) + (time.process_time() - cpu_begin)


def _sha(path: Path, guard):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            _check(guard)
            b = f.read(1024 * 1024)
            if not b: break
            h.update(b)
    return h.hexdigest()


def _source_identity(guard):
    root = Path(__file__).resolve().parents[2]
    _check(guard)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True,
                            text=True, encoding="utf-8").stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, check=True,
                            capture_output=True, text=True, encoding="utf-8").stdout
    _check(guard)
    if status.strip(): raise StudyError("production source must be clean and committed")
    paths = sorted((root / "dwr" / "tasks").glob("*.py")) + [root / "dwr" / "__init__.py"]
    blobs = {}
    for path in paths:
        rel = path.relative_to(root).as_posix()
        disk = _sha(path, guard)
        disk_bytes = path.read_bytes()
        _check(guard)
        committed_bytes = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=root, check=True,
                                         capture_output=True).stdout
        _check(guard)
        if hashlib.sha256(disk_bytes).digest() != hashlib.sha256(committed_bytes).digest():
            raise StudyError("runtime producer bytes differ from committed Git blob bytes")
        blobs[rel] = disk
    return {"git_commit": commit, "git_clean": True, "producer_sha256": blobs}


def _safe_path(path: Path):
    # Resolve without following an existing reparse point anywhere in the requested chain.
    cur = path.absolute()
    for part in (cur, *cur.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if part.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("output path has a symlink or reparse-point ancestor")


def _private_outcome(outcome, partition, index):
    task = outcome.task
    profile = "/".join(outcome.profile)
    return {"partition": partition, "candidate_index": index, "profile": profile, "cell": outcome.cell,
            "branch": getattr(outcome, "branch", "ordinary"),
            "task": serialize_task(task) if task is not None else None,
            "witness": getattr(outcome, "witness", None), "givens": outcome.givens,
            "alternatives": getattr(outcome, "alternatives", ()),
            "reserved_constraints": getattr(outcome, "reserved_constraints", ()),
            "rejection_reason": outcome.rejection_reason, "work": outcome.work}


def _validate_outcome(outcome, partition, index, config):
    """Admit only the frozen constructor type and exact scheduled identity."""
    profile, cell = _schedule(partition, index)
    if (type(getattr(outcome, "root_seed", None)) is not int or
            outcome.root_seed != config["root_seed"] or
            type(getattr(outcome, "candidate_index", None)) is not int or outcome.candidate_index != index or
            getattr(outcome, "partition", None) != partition or
            getattr(outcome, "profile", None) != tuple(profile.split("/")) or
            getattr(outcome, "cell", None) != cell):
        raise StudyError("constructor outcome identity does not match its scheduled attempt")
    if partition == "source":
        if type(outcome) is not legacy_generator.CertificateOutcome:
            raise StudyError("source constructor returned an unregistered outcome type")
    elif partition == "development":
        expected_branch = v2_generator._add_branch(index)
        if expected_branch == "no_unary":
            if type(outcome) is not rank_generator.RankOutcome or outcome.generator_id != rank_generator.GENERATOR_ID or outcome.branch != "no_unary":
                raise StudyError("rank branch returned an unregistered outcome type or identity")
        elif type(outcome) is not legacy_generator.CertificateOutcome or getattr(outcome, "branch", None) not in (None, "ordinary"):
            raise StudyError("ordinary development branch returned an unregistered outcome type")
    else:
        if (type(outcome) is not rank_generator.RankOutcome or
                getattr(outcome, "generator_id", None) != rank_generator.GENERATOR_ID or
                getattr(outcome, "branch", None) != "ordinary"):
            raise StudyError("final constructor returned an unregistered outcome type or identity")


def _build_row(outcome, partition, index, config, seen, lineage, guard, *, fixture=False):
    from .certificate_fixture_v2 import _assert_no_unary_gac, _audit_certificate, _core
    raw = _private_outcome(outcome, partition, index)
    row = dict(raw)
    row.update({"split": None, "content_id": None, "lineage_id": None, "structure_signature": None,
                "oracle_unique": False, "eligible": False, "sensitive": False, "envelope": False,
                "gac": None, "first_seen_content": False, "state": "construction_failed", "values": [], "given_count": 0})
    five_mode = config.get("support", {}).get("scheme") == "five-fixed-strata-v1"
    task = outcome.task
    if task is None: return row
    row["given_count"] = len(task.givens)
    row["values"] = [v for _, v in outcome.witness.assignments]
    row["given_count"] = len(task.givens)
    row["task"] = serialize_task(task)
    counters = {}
    row["model_input_sha256"] = hashlib.sha256(serialize_model_input(task).encode("utf-8")).hexdigest()
    metrics = old._task_metrics(task, check_budget=lambda: _check(guard), work_counters=counters)
    row.update({"content_id": metrics["content_id"], "lineage_id": metrics["lineage_id"],
                "structure_signature": metrics["structure_signature"], "prompt_bytes": metrics["prompt_bytes"],
                "first_solution_nodes": metrics["first_solution_nodes"]})
    row["split"] = (metrics["source_split"] if partition == "source" else
                    "final_test" if partition == "final" else metrics["development_split"])
    # Qualification may choose only from source training and development validation.
    # The full registration computes fixed structural gates after the prior stages pass.
    if config.get("mode") == "full":
        protected = False
    else:
        choice_split = ("source_train" if partition == "source" else
                        "development_validation" if partition == "development" else "final_test")
        protected = row["split"] != choice_split
    identity = (row["content_id"], row["structure_signature"])
    prior = lineage.get(row["lineage_id"])
    if prior is not None and prior != identity: raise StudyError("lineage closure violation")
    lineage[row["lineage_id"]] = identity
    duplicate = row["content_id"] in seen
    row["first_seen_content"] = not duplicate
    row["duplicate_content"] = duplicate
    if not duplicate: seen.add(row["content_id"])
    sem = old._audit_task({"task_object": task, "witness": outcome.witness, "profile": row["profile"],
                           "cell": row["cell"], "reserved_constraints": outcome.reserved_constraints,
                           "alternatives": outcome.alternatives, "eligible": None}, partition=partition,
                          check_budget=lambda: _check(guard), work_counters=counters)
    scalar = _scalar_solutions(task, guard, counters)
    oracle = enumerate_solutions(task)
    ordered = _audit_ordered_solutions(task, scalar, oracle.solutions, solve(task), outcome.witness,
                                       guard, counters)
    row["ordered_solution_audit"] = ordered
    actual = propagate(task, check_budget=lambda: _check(guard))
    projection = old._audit_local_gac(task, actual, check_budget=lambda: _check(guard), work_counters=counters)
    if protected:
        # Validate private semantics while suppressing every choice statistic and payload.
        if row["branch"] == "no_unary":
            _assert_no_unary_gac(task, lambda: _check(guard))
            _audit_certificate(task, outcome.witness, outcome.alternatives, outcome.reserved_constraints,
                               tuple(outcome.profile))
        row["state"] = "protected_integrity_only"
        row["integrity_checked"] = True
        row["verification_work"] = counters
        return row
    row.update({k: sem[k] for k in ("oracle_unique", "eligible", "sensitive", "oracle_solution_count")})
    row["eligible"] = bool(sem["eligible"])
    row["sensitive"] = bool(sem["sensitive"])
    if five_mode:
        row["fine_stratum"] = audit.support_stratum(row, config)
        row["envelope"] = bool(row["oracle_unique"] and row["fine_stratum"] is not None)
    else:
        row["envelope"] = bool(row["oracle_unique"] and old._prompt_bin(row["prompt_bytes"]) == 4 and old._node_bin(row["first_solution_nodes"]) == 2)
    row["gac"] = {"stalled": actual.stalled, "solved": actual.solved, "contradiction": actual.contradiction,
                   "independent_audit": projection is not None}
    row["state"] = "first_seen" if not duplicate else "duplicate_content"
    row["private_interaction"] = sem["interaction"]
    if row["branch"] == "no_unary":
        _assert_no_unary_gac(task, lambda: _check(guard))
        _audit_certificate(task, outcome.witness, outcome.alternatives, outcome.reserved_constraints,
                           tuple(outcome.profile))
        row["no_unary_gac_checked"] = True
        if (partition == "development" and row["split"] == "development_validation" and
                row["first_seen_content"] is True and row["oracle_unique"] is True and
                row["eligible"] is True):
            row["core"] = _core(task, lambda: _check(guard))
    row["task"] = serialize_task(task)
    row["verification_work"] = counters
    return row


def _validate_row_annotation(row, config):
    if config.get("support", {}).get("scheme") == "five-fixed-strata-v1" and not audit.row_schema_valid(row, config):
        raise StudyError("five-stratum row annotation or state schema is inconsistent")
    return True


def _schedule(partition, index):
    if partition == "source": profiles = ("eq", "neq", "add_mod")
    elif partition == "development": profiles = ("eq/neq", "eq/add_mod", "neq/add_mod")
    else: profiles = ("eq/neq/add_mod",)
    cells = ("A", "B", "C", "D")
    return profiles[(index // 4) % len(profiles)], cells[index % 4]


def _construct(root, partition, index, guard):
    if partition != "final":
        return rank_generator.construct(root, partition, index, check_budget=lambda: _check(guard))
    return rank_generator.construct(root, partition, index, check_budget=lambda: _check(guard))


def _verify_historical_producers(historical_source, current_source, guard):
    """Prove a five-mode qualification producer map against its committed Git blobs."""
    def valid_commit(value):
        return (type(value) is str and len(value) == 40 and
                all(ch in "0123456789abcdefABCDEF" for ch in value))
    if (type(historical_source) is not dict or historical_source.get("git_clean") is not True or
            not valid_commit(historical_source.get("git_commit"))):
        raise StudyError("historical qualification source identity is invalid")
    producers = historical_source.get("producer_sha256")
    current = current_source.get("producer_sha256") if type(current_source) is dict else None
    if (type(current_source) is not dict or current_source.get("git_clean") is not True or
            not valid_commit(current_source.get("git_commit")) or
            type(producers) is not dict or not producers or
            not _same_typed(producers, current) or
            any(type(name) is not str or not name or type(digest) is not str or len(digest) != 64 or
                name.startswith(("/", "\\")) or ".." in name.split("/") or "\\" in name or ":" in name or
                any(ch not in "0123456789abcdef" for ch in digest)
                for name, digest in producers.items())):
        raise StudyError("historical producer map is empty or differs from current runtime")
    root = Path(__file__).resolve().parents[2]
    commit = historical_source["git_commit"]
    for name, expected_hash in producers.items():
        _check(guard)
        try:
            result = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=root, check=True,
                                    capture_output=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise StudyError("historical producer Git blob is unavailable") from exc
        _check(guard)
        if hashlib.sha256(result.stdout).hexdigest() != expected_hash:
            raise StudyError("historical producer blob does not match its registered hash")
    return True


def _five_qualification_projection_bytes(outcome_lines, config, config_hash):
    """Derive the exact public A ledger from complete metadata-only raw rows."""
    expected_counts = {"source": config["attempts"]["source"],
                       "development": config["attempts"]["development"],
                       "final": config["attempts"]["final"]}
    expected_sequence = [(part, index) for part in ("source", "development", "final")
                         for index in range(expected_counts[part])]
    if len(outcome_lines) != len(expected_sequence):
        raise StudyError("qualification raw outcome count differs from registration")
    records = []
    protected_splits = {"source_validation", "source_test", "development_diagnostic_test"}
    for index, (line, expected_identity) in enumerate(zip(outcome_lines, expected_sequence)):
        row = _strict_load(line.decode("utf-8"))
        if (type(row) is not dict or (row.get("partition"), row.get("candidate_index")) != expected_identity or
                row.get("config_sha256") != config_hash):
            raise StudyError("qualification raw outcome identity or config binding is invalid")
        if _json_bytes(row) != line:
            raise StudyError("qualification raw outcome is not in canonical ledger form")
        split = row.get("split")
        if split in protected_splits:
            # Route before inspecting the fine annotation or A predicate.
            if row.get("state") != "protected_integrity_only" or not audit.row_schema_valid(row, config):
                raise StudyError("qualification protected row schema is invalid")
            continue
        if not audit.row_schema_valid(row, config):
            raise StudyError("qualification raw outcome row schema or fine annotation is invalid")
        if split is None:
            continue
        if split not in ("source_train", "development_validation"):
            raise StudyError("qualification outcome has an unregistered choice split")
        if _is_A(row, config):
            payload = line if line.endswith(b"\n") else line + b"\n"
            projection = _a_projection(row)
            projection["outcome_sha256"] = hashlib.sha256(payload).hexdigest()
            if not isinstance(row.get("model_input_sha256"), str):
                raise StudyError("accepted qualification row lacks its model-input hash")
            records.append(_json_bytes(projection))
    if not records:
        raise StudyError("passing qualification cannot have an empty accepted-record ledger")
    return b"".join(records)


def _verify_exact_qualification_records(outcome_lines, record_bytes, config, config_hash):
    expected = _five_qualification_projection_bytes(outcome_lines, config, config_hash)
    if record_bytes != expected:
        raise StudyError("qualification records are not the exact ordered projection of accepted raw outcomes")
    return True


def _load_qualification_receipt(qout, link, source, guard):
    """Bind full five-mode execution to the complete exact qualification ledger."""
    manifest_path, initial_path = qout / "manifest.json", qout / "manifest.initial.json"
    replay_path, report_path = qout / "replay.json", qout / "report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    qconfig_path = _registered_path("qualification", five_stratum=True)
    qconfig, qhash, _ = load_config(qconfig_path)
    if (qhash != link.get("config_sha256") or manifest.get("config_sha256") != qhash or
            initial.get("config_sha256") != qhash or
            not _same_typed(initial.get("resolved_config"), qconfig)):
        raise StudyError("qualification config identity differs from the registered five-stratum tree")
    if (manifest.get("status") != "PASS_QUALIFICATION" or report.get("status") != "PASS_QUALIFICATION" or
            report.get("fixture_only") is not False or replay.get("status") != "PASS_REPLAY" or
            replay.get("exact_indices") is not True):
        raise StudyError("new qualification is not an exact production PASS with replay")
    expected_counts = {"source": 3600, "development": 4800, "final": 0}
    if (not _same_typed(report.get("raw_attempts"), expected_counts) or
            type(replay.get("attempts_rederived")) is not int or replay["attempts_rederived"] != 8400):
        raise StudyError("new qualification attempt and replay counts differ from registration")
    expected_source = set(audit.source_gate([], 0, qconfig)[1]["floors"])
    expected_development = set(audit.development_gate([], [], qconfig)[1]["floors"])
    for gate_name, expected_floors in (("source_gate", expected_source),
                                       ("development_gate", expected_development)):
        gate = report.get(gate_name)
        floors = gate.get("floors") if type(gate) is dict else None
        if not _exact_true_floor_map(floors, expected_floors):
            raise StudyError("new qualification has an empty or nonpassing floor map")
    if initial.get("source") != manifest.get("source"):
        raise StudyError("new qualification source identities differ between manifests")
    _verify_historical_producers(manifest.get("source"), source, guard)
    if (Path(initial.get("output_dir", "")).resolve() != qout.resolve() or
            Path(initial.get("config_path", "")).resolve(strict=True) != qconfig_path.resolve(strict=True) or
            not _argv_binds(initial.get("requested_argv"), qconfig_path, qout)):
        raise StudyError("new qualification command and paths are not the exact registration")
    hashes = manifest.get("artifacts")
    if type(hashes) is not dict or set(hashes) != set(ARTIFACTS[:-1]):
        raise StudyError("new qualification artifact binding is incomplete")
    for name, digest in hashes.items():
        _check(guard)
        if _sha(qout / name, guard) != digest:
            raise StudyError("new qualification artifact hash mismatch")
    outcome_lines = (qout / "outcomes.jsonl").read_bytes().splitlines(keepends=True)
    _verify_exact_qualification_records(outcome_lines, (qout / "records.jsonl").read_bytes(), qconfig, qhash)
    return {"manifest_sha256": _sha(manifest_path, guard), "artifact_sha256": hashes,
            "source_producer_sha256": source["producer_sha256"], "replay_status": replay["status"]}


def _verify_qualification_link(config, source, guard):
    link = config["required_qualification"]
    five_mode = config.get("support", {}).get("scheme") == "five-fixed-strata-v1"
    registry = FIVE_CONFIG_HASHES if five_mode else CONFIG_HASHES
    if link.get("config_sha256") != registry["qualification"]:
        raise StudyError("qualification config binding changed")
    root = Path(__file__).resolve().parents[2]
    path_key = "run_path" if five_mode else "output_path"
    qout = Path(link[path_key])
    if not qout.is_absolute(): qout = root / qout
    expected_qout = root / "runs" / ("certificate-five-stratum-qualification-20261014-v1" if five_mode else "certificate-rank-qualification-20261012-v1")
    if qout.resolve() != expected_qout.resolve():
        raise StudyError("qualification output binding changed")
    if five_mode:
        receipt = _load_qualification_receipt(qout, link, source, guard)
        receipt["status"] = "PASS_QUALIFICATION"
        return receipt
    manifest_path = qout / "manifest.json"
    initial_path = qout / "manifest.initial.json"
    replay_path = qout / "replay.json"
    report_path = qout / "report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "PASS_QUALIFICATION" or report.get("status") != "PASS_QUALIFICATION" or
            manifest.get("config_sha256") != link["config_sha256"] or replay.get("status") != "PASS_REPLAY" or
            replay.get("exact_indices") is not True):
        raise StudyError("qualification run is not a complete exact PASS")
    if report.get("fixture_only") is not False:
        raise StudyError("fixture output cannot satisfy qualification linkage")
    expected_counts = {"source": 3600, "development": 4800, "final": 0}
    if (not _same_typed(report.get("raw_attempts"), expected_counts) or
            type(replay.get("attempts_rederived")) is not int or replay["attempts_rederived"] != 8400):
        raise StudyError("qualification attempt/replay counts differ from registration")
    qualification_config = load_config(_registered_path("qualification"))[0]
    expected_source_floors = set(audit.source_gate([], 0, qualification_config)[1]["floors"])
    expected_development_floors = set(audit.development_gate([], [], qualification_config)[1]["floors"])
    for gate_name, expected_keys in (("source_gate", expected_source_floors),
                                     ("development_gate", expected_development_floors)):
        floors = report.get(gate_name, {}).get("floors")
        if not _exact_true_floor_map(floors, expected_keys):
            raise StudyError("qualification report does not retain every exact passing floor")
    if (initial.get("config_sha256") != link["config_sha256"] or
            not _same_typed(initial.get("resolved_config"), load_config(_registered_path("qualification"))[0]) or
            initial.get("source") != manifest.get("source") or
            Path(initial.get("output_dir", "")).resolve() != qout.resolve() or
            Path(initial.get("config_path", "")).resolve(strict=True) != _registered_path("qualification").resolve(strict=True)):
        raise StudyError("qualification initial manifest does not bind exact config and source")
    if not manifest.get("source", {}).get("git_clean"):
        raise StudyError("qualification source was not clean")
    requested = initial.get("requested_argv")
    if type(requested) is not list or not _argv_binds(requested, _registered_path("qualification"), qout):
        raise StudyError("qualification command does not bind registered config/output paths")
    if manifest.get("source", {}).get("producer_sha256") != source.get("producer_sha256"):
        raise StudyError("qualification producer identities differ from full runtime")
    hashes = manifest.get("artifacts")
    if type(hashes) is not dict or set(hashes) != set(ARTIFACTS[:-1]):
        raise StudyError("qualification artifact binding is incomplete")
    for name, digest in hashes.items():
        _check(guard)
        if _sha(qout / name, guard) != digest: raise StudyError("qualification artifact hash mismatch")
    return {"manifest_sha256": _sha(manifest_path, guard), "artifact_sha256": hashes,
            "source_producer_sha256": source["producer_sha256"], "replay_status": replay["status"]}


def _argv_binds(argv, config_path, output_path):
    values = {}
    i = 0
    while i < len(argv):
        item = argv[i]
        if item in ("--config", "--output") and i + 1 < len(argv):
            values[item] = argv[i + 1]; i += 2; continue
        if isinstance(item, str) and item.startswith("--config="):
            values["--config"] = item.split("=", 1)[1]
        elif isinstance(item, str) and item.startswith("--output="):
            values["--output"] = item.split("=", 1)[1]
        i += 1
    try:
        return (Path(values["--config"]).resolve(strict=True) == Path(config_path).resolve(strict=True) and
                Path(values["--output"]).resolve() == Path(output_path).resolve())
    except (KeyError, OSError, RuntimeError):
        return False


def _initial(config, cfg_hash, cfg_path, output, argv, source):
    return {"manifest_version": "dwr-certificate-rank-study-initial-v1", "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(), "command": list(sys.orig_argv),
            "requested_argv": argv, "config_path": str(cfg_path), "config_sha256": cfg_hash,
            "resolved_config": config, "source": source, "output_dir": str(output),
            "python": sys.version, "implementation": platform.python_implementation(),
            "platform": platform.platform(), "independent_rng_streams": ["source-v1", "ordinary-v2", "rank-v1"],
            "precision": "standard-library CPU", "budgets": config["limits"], "artifact_names": list(ARTIFACTS)}


def run_study(config_path, output_path, *, argv=None, _guard=None, _constructor=None):
    """Run the exact qualification or full registration; private injections are fixture-only."""
    started = time.monotonic()
    cpu_started = time.process_time()
    # Guard exists before any registration, source, or hash reads.
    guard = _guard
    if guard is None:
        guard = ResourceGuard(3600, 1024, memory_reader=resident_bytes_windows)
    _check(guard)
    config, cfg_hash, cfg_path = load_config(config_path)
    guard.wall_seconds = config["limits"]["active_wall_seconds"]
    output_requested = Path(output_path).absolute()
    _safe_path(output_requested)
    output = output_requested.resolve()
    if output.exists(): raise FileExistsError("output already exists")
    production = _constructor is None and _guard is None
    runs = Path(__file__).resolve().parents[2] / "runs"
    if production:
        try: output.relative_to(runs.resolve())
        except ValueError as exc: raise ValueError("production output must be under runs/") from exc
        if config.get("support", {}).get("scheme") == "five-fixed-strata-v1":
            registered = Path(config["output_path"])
            if not registered.is_absolute(): registered = runs.parent / registered
        elif config["mode"] == "qualification":
            registered = runs / "certificate-rank-qualification-20261012-v1"
        else:
            registered = runs / "certificate-rank-full-20261013-v1"
        if output != registered.resolve():
            if config.get("support", {}).get("scheme") == "five-fixed-strata-v1":
                raise ValueError(f"{config['mode']} output path differs from registered path")
            if config["mode"] == "qualification":
                raise ValueError("qualification output path differs from registered path")
            raise ValueError("full output path differs from registered path")
    output.mkdir(parents=True)
    initial = None
    report = {"status": "NOT_ASSESSED", "g2_status": "not_assessed", "preflight_emitted": False,
              "fixture_only": not production, "config_sha256": cfg_hash}
    stream = None
    try:
        source = _source_identity(guard) if production else {"git_commit": None, "fixture_only": True}
        initial = _initial(config, cfg_hash, cfg_path, output, argv or sys.argv, source)
        if config["mode"] == "full" and production:
            link_receipt = _verify_qualification_link(config, source, guard)
            report["qualification_link"] = link_receipt
            initial["qualification_link"] = link_receipt
        if config["mode"] == "full":
            initial["final_structural_access"] = "permitted only after qualification linkage and source/development stage PASS; no model evaluation"
        _write(output / "manifest.initial.json", initial, guard)
        rows, raw_counts, replay_rows = [], Counter(), []
        seen, lineage = set(), {}
        report["status"] = "RUNNING"
        partitions = ("source", "development") if config["mode"] == "qualification" else ("source", "development", "final")
        for part in partitions:
            if part == "development" and config["mode"] == "qualification":
                passed, gate = audit.source_gate(rows, raw_counts["source"], config)
                report["source_gate"] = gate
                if not passed:
                    report["status"] = "NO_GO_SOURCE"
                    break
            if part == "final" and config["mode"] == "full":
                # Final records are admitted only after the exact registered qualification link and preceding stages.
                if not report.get("qualification_link"):
                    raise StudyError("registered qualification did not pass exact linkage")
                event = {"event": "final_structural_access", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                         "scope": "final_test structural qualification only; no model evaluation or construction choices",
                         "config_sha256": cfg_hash,
                         "qualification_manifest_sha256": report["qualification_link"]["manifest_sha256"],
                         "source_stage_sha256": hashlib.sha256(_json_bytes(report["source_gate"])).hexdigest(),
                         "development_stage_sha256": hashlib.sha256(_json_bytes(report["development_gate"])).hexdigest(),
                         "frozen_contract": {"final_quota": config["full_floors"]["final_quota"],
                                             "raw_limit": config["attempts"]["final"],
                                             "support": config["support"], "floors": config["full_floors"]}}
                initial.setdefault("events", []).append(event)
                report["final_structural_access"] = event
                _write(output / "manifest.initial.json", initial, guard)
            limit = config["attempts"][part]
            quota = (config["full_floors"]["source_quota"] if part == "source" else
                     config["full_floors"]["final_quota"] if part == "final" else
                     config["full_floors"]["development_quota_per_profile"])
            by_profile = Counter()
            partition_rows = []
            with (output / "outcomes.jsonl").open("ab") as outcomes, (output / "records.jsonl").open("ab") as records:
                for index in range(limit):
                    _check(guard)
                    outcome = None
                    phase = "construction"
                    try:
                        outcome = (_constructor(config, part, index) if _constructor else _construct(config["root_seed"], part, index, guard))
                        phase = "row_assembly"
                        if _constructor is None:
                            _validate_outcome(outcome, part, index, config)
                        row = _build_row(outcome, part, index, config, seen, lineage, guard, fixture=not production)
                        _validate_row_annotation(row, config)
                        row["config_sha256"] = cfg_hash
                        payload = _json_bytes(row)
                        outcomes.write(payload); outcomes.flush()
                        # A-only projections link to their exact full outcome line.
                        if _is_A(row, config):
                            record = _a_projection(row)
                            record["outcome_sha256"] = hashlib.sha256(payload).hexdigest()
                            records.write(_json_bytes(record)); records.flush()
                        partition_rows.append(row); rows.append(row); raw_counts[part] += 1
                        profile, _ = _schedule(part, index)
                        _count_A_for_quota(row, profile, by_profile, config)
                    except Exception as exc:
                        fatal = {"partition": part, "candidate_index": index, "phase": phase,
                                 "error_type": type(exc).__name__, "state": "fatal_error"}
                        if outcome is not None:
                            try: fatal["raw_outcome"] = _private_outcome(outcome, part, index)
                            except Exception: pass
                        outcomes.write(_json_bytes(fatal)); outcomes.flush(); raw_counts[part] += 1
                        report.update({"status": "NOT_ASSESSED", "failure": {"index": index, "phase": phase, "error_type": type(exc).__name__}})
                        raise
                    if config["mode"] == "full":
                        done = (by_profile[profile] >= quota if part == "development" else
                                sum(by_profile.values()) >= quota)
                        # Fixed schedule continues until every development pair has quota.
                        if part == "development": done = all(by_profile[p] >= quota for p in config["development_profiles"])
                        if done:
                            break
            if config["mode"] == "qualification":
                if part == "development":
                    ok, gate = audit.development_gate(partition_rows, [r for r in rows if r["partition"] == "source"], config)
                    report["development_gate"] = gate
                    report["status"] = "PASS_QUALIFICATION" if ok and production else "NOT_ASSESSED_FIXTURE_ONLY" if ok else "NO_GO_DEVELOPMENT"
            else:
                if part == "source":
                    ok, gate = audit.full_stage_gate(partition_rows, "source", config, raw_counts[part])
                elif part == "development":
                    ok, gate = audit.full_stage_gate(partition_rows, "development", config, raw_counts[part], source_rows=rows)
                else:
                    ok, gate = audit.full_stage_gate(partition_rows, "final", config, raw_counts[part])
                report[f"{part}_gate"] = gate
                if not ok:
                    report["status"] = "NO_GO_FULL_STAGE"
                    break
        # Reconstruct the persisted records from their exact indices and compare canonical bytes.
        replay_rows = []
        rseen, rlineage = set(), {}
        for saved in rows:
            part, index = saved["partition"], saved["candidate_index"]
            outcome = (_constructor(config, part, index) if _constructor else _construct(config["root_seed"], part, index, guard))
            if _constructor is None: _validate_outcome(outcome, part, index, config)
            rebuilt = _build_row(outcome, part, index, config, rseen, rlineage, guard, fixture=not production)
            _validate_row_annotation(rebuilt, config)
            replay_rows.append(rebuilt)
        # Rebuild and compare the exact raw and A-projection ledgers from the same indices.
        replay_ok = _replay_matches(rows, replay_rows,
                                    (output / "outcomes.jsonl").read_bytes(),
                                    (output / "records.jsonl").read_bytes(), cfg_hash, config)
        _write(output / "replay.json", {"status": "PASS_REPLAY" if replay_ok else "FAIL_REPLAY",
                                         "attempts_rederived": len(replay_rows), "exact_indices": True}, guard)
        if not replay_ok: report["status"] = "NOT_ASSESSED_REPLAY_MISMATCH"
        report.update({"raw_attempts": {p: raw_counts[p] for p in ("source", "development", "final")},
                       "g2_status": "not_assessed", "preflight_emitted": False})
        if not production: report["status"] = "NOT_ASSESSED_FIXTURE_ONLY"
        if config["mode"] == "full" and report.get("status") == "RUNNING":
            full_pass, full_report = audit.full_gate(rows, dict(raw_counts), config)
            report["full_audit"] = full_report
            pass_status = ("PASS_G2_FIVE_STRATUM_V1" if config.get("support", {}).get("scheme") == "five-fixed-strata-v1"
                           else "PASS_G2_RANK_V1")
            report["status"] = pass_status if full_pass and replay_ok and production else "NO_GO_FULL_AUDIT"
        report["g2_status"] = _g2_status(report["status"], production, replay_ok)
        hashes = {name: _sha(output / name, guard) for name in ARTIFACTS[:-2] if (output / name).exists()}
        report.update({"elapsed_seconds": time.monotonic() - started,
                       "cpu_seconds": time.process_time() - cpu_started,
                       "peak_rss_bytes": guard.peak_bytes,
                       "artifact_write_wall_seconds": getattr(guard, "artifact_write_wall_seconds", 0.0),
                       "artifact_write_cpu_seconds": getattr(guard, "artifact_write_cpu_seconds", 0.0),
                       "accounting_boundary": "through replay and pre-report ledger hashing; terminal report and manifest costs are in manifest resource receipt"})
        _write(output / "report.json", report, guard)
        hashes["report.json"] = _sha(output / "report.json", guard)
        manifest = {"manifest_version": "dwr-certificate-rank-study-v1", "status": report["status"],
                    "g2_status": report["g2_status"],
                    "config_sha256": cfg_hash, "source": source, "artifacts": hashes,
                    "resources": {"elapsed_seconds": time.monotonic() - started,
                                  "cpu_seconds": time.process_time() - cpu_started,
                                  "peak_rss_bytes": guard.peak_bytes,
                                  "artifact_write_wall_seconds": getattr(guard, "artifact_write_wall_seconds", 0.0),
                                  "artifact_write_cpu_seconds": getattr(guard, "artifact_write_cpu_seconds", 0.0),
                                  "accounting_boundary": "through report hash; final manifest write is guard-checked"}}
        _write(output / "manifest.json", manifest, guard)
        return report
    except Exception as exc:
        report.update({"status": "NOT_ASSESSED", "g2_status": "not_assessed",
                       "terminal_reason": type(exc).__name__,
                       "elapsed_seconds": time.monotonic() - started,
                       "cpu_seconds": time.process_time() - cpu_started,
                       "peak_rss_bytes": guard.peak_bytes,
                       "artifact_write_wall_seconds": getattr(guard, "artifact_write_wall_seconds", 0.0),
                       "artifact_write_cpu_seconds": getattr(guard, "artifact_write_cpu_seconds", 0.0)})
        # Best-effort partial evidence, with no rehash of large ledgers after a cap.
        # A late cap or terminal write failure revokes any just-written PASS;
        # stale hashes are discarded without rescanning large ledgers.
        try: _revoke_terminal(output, cfg_hash, report, exc)
        except Exception: pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        # Guard before config/source/hash reads, including admission failures.
        report = run_study(args.config, args.output, argv=sys.argv)
        print(stable_json(report))
        return 0 if report["status"] in ("PASS_QUALIFICATION", "PASS_G2_RANK_V1", "PASS_G2_FIVE_STRATUM_V1") else 1
    except Exception as exc:
        print(f"NOT_ASSESSED: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
