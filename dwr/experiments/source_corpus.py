"""Deterministic export of the registered, accepted source task population."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from dwr.tasks import canonical, certificate_rank_study as producer, oracle, schema, solver
from dwr.tasks.run_support import stable_json, strict_json_loads
from dwr.tasks.verifier import verify_answer


_ARTIFACT_NAMES = (
    "manifest.initial.json",
    "outcomes.jsonl",
    "records.jsonl",
    "replay.json",
    "report.json",
    "manifest.json",
)
_SPLIT_PARTITION = {
    "source_train": "source",
    "source_validation": "source",
    "source_test": "source",
    "development_validation": "development",
    "development_diagnostic_test": "development",
    "final_test": "final",
}
_PROFILE_KINDS = {"eq": {"eq"}, "neq": {"neq"}, "add_mod": {"add_mod"}}
_FINE_STRATA = {"A2", "A3", "B2", "C2", "D2"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_WINDOWS_INVALID_LEAF = re.compile(r'[\x00-\x1f<>:"/\\|?*]')
_WINDOWS_RESERVED_DEVICES = {
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
    *(f"COM{suffix}" for suffix in ("¹", "²", "³")),
    *(f"LPT{suffix}" for suffix in ("¹", "²", "³")),
}


class SourceExportError(ValueError):
    """Input artifacts or a source row violate the frozen export contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceExportError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _validate_windows_output_leaf(leaf: str) -> None:
    """Reject Windows-invalid names while allowing drive-qualified parent paths."""
    if (not leaf or leaf in (".", "..") or leaf.endswith((".", " ")) or
            _WINDOWS_INVALID_LEAF.search(leaf) is not None or len(leaf.encode("utf-16-le")) > 510):
        raise SourceExportError("unsafe Windows output filename")
    device_stem = leaf.split(".", 1)[0].rstrip(" .").upper()
    if device_stem in _WINDOWS_RESERVED_DEVICES:
        raise SourceExportError("unsafe Windows output filename")


def _typed_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int equality shortcut."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (left.keys() == right.keys() and
                all(_typed_equal(left[key], right[key]) for key in left))
    if type(left) is list:
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _reject_reparse_chain(path: Path, *, must_exist: bool) -> Path:
    if ".." in path.parts:
        raise SourceExportError("unsafe path contains parent traversal")
    absolute = Path(os.path.abspath(path))
    parts = list(reversed((absolute, *absolute.parents)))
    for item in parts:
        if not item.exists() and not item.is_symlink():
            if must_exist:
                raise SourceExportError(f"path does not exist: {item}")
            continue
        try:
            mode = item.lstat().st_mode
            attributes = getattr(item.lstat(), "st_file_attributes", 0)
        except OSError as exc:
            raise SourceExportError(f"cannot inspect path: {item}") from exc
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if stat.S_ISLNK(mode) or (reparse and attributes & reparse):
            raise SourceExportError("symbolic links and reparse points are not allowed")
    return absolute.resolve(strict=must_exist)


def _check(check: Callable[[], Any]) -> None:
    check()


def _hash_file(path: Path, check: Callable[[], Any]) -> str:
    _check(check)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                _check(check)
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise SourceExportError(f"cannot read artifact: {path.name}") from exc
    _check(check)
    return digest.hexdigest()


def _strict_object(data: bytes, name: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8", errors="strict")
        value = strict_json_loads(text)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise SourceExportError(f"invalid strict JSON in {name}") from exc
    _require(type(value) is dict, f"{name} must contain a JSON object")
    return value


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SourceExportError(f"cannot read metadata artifact: {name}") from exc
    return _strict_object(data, name)


def _route(row: dict[str, Any], *, context: str) -> tuple[str, int, str | None]:
    _require("split" in row, f"{context} is missing split routing metadata")
    partition = row.get("partition")
    candidate_index = row.get("candidate_index")
    split = row.get("split")
    _require(type(partition) is str and partition in {"source", "development", "final"},
             f"{context} has an unknown partition")
    _require(type(candidate_index) is int and candidate_index >= 0,
             f"{context} has an invalid candidate_index")
    if split is None:
        _require(row.get("state") == "construction_failed",
                 f"{context} has null split outside construction_failed")
        return partition, candidate_index, None
    _require(type(split) is str and split in _SPLIT_PARTITION,
             f"{context} has an unknown split")
    _require(_SPLIT_PARTITION[split] == partition,
             f"{context} partition and split disagree")
    _require(row.get("state") != "construction_failed",
             f"{context} construction_failed row must have null split")
    return partition, candidate_index, split


def _iter_jsonl(path: Path, name: str, check: Callable[[], Any]):
    try:
        with path.open("rb") as stream:
            line_number = 0
            for raw_line in stream:
                _check(check)
                line_number += 1
                _require(raw_line.endswith(b"\n"), f"{name}:{line_number} lacks its LF terminator")
                try:
                    value = strict_json_loads(raw_line[:-1].decode("utf-8", errors="strict"))
                except (UnicodeError, ValueError, TypeError) as exc:
                    raise SourceExportError(f"invalid strict JSON at {name}:{line_number}") from exc
                _require(type(value) is dict, f"{name}:{line_number} must be an object")
                yield raw_line, value
                _check(check)
    except OSError as exc:
        raise SourceExportError(f"cannot read JSONL artifact: {name}") from exc


def _validate_metadata(
    input_dir: Path,
    expected_artifacts: dict[str, str],
    config_bytes: bytes,
    expected_config_sha256: str,
    expected_source_commit: str,
    check: Callable[[], Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    observed_hashes = {}
    for name in _ARTIFACT_NAMES:
        observed_hashes[name] = _hash_file(input_dir / name, check)
        _require(observed_hashes[name] == expected_artifacts[name],
                 f"artifact hash mismatch: {name}")

    _require(_sha256(config_bytes) == expected_config_sha256,
             "generation config bytes differ from the registered hash")
    config = _strict_object(config_bytes, "generation config")
    initial = _read_json(input_dir / "manifest.initial.json", "manifest.initial.json")
    terminal = _read_json(input_dir / "manifest.json", "manifest.json")
    report = _read_json(input_dir / "report.json", "report.json")
    replay = _read_json(input_dir / "replay.json", "replay.json")

    for name, value in (("initial", initial), ("terminal", terminal), ("report", report)):
        _require(value.get("config_sha256") == expected_config_sha256,
                 f"{name} config hash mismatch")
    _require(_typed_equal(initial.get("resolved_config"), config),
             "initial resolved config differs from original typed config")
    _require(report.get("fixture_only") is False,
             "fixture-only reports cannot authorize source export")
    _require(report.get("status") == "PASS_G2_FIVE_STRATUM_V1" and
             report.get("g2_status") == "pass",
             "report does not record the registered G2 pass")
    _require(terminal.get("status") == "PASS_G2_FIVE_STRATUM_V1" and
             terminal.get("g2_status") == "pass",
             "terminal manifest does not record the registered G2 pass")
    _require(replay.get("status") == "PASS_REPLAY" and replay.get("exact_indices") is True,
             "replay does not record exact successful replay")

    for name, manifest in (("initial", initial), ("terminal", terminal)):
        source = manifest.get("source")
        _require(type(source) is dict and source.get("git_commit") == expected_source_commit and
                 source.get("git_clean") is True,
                 f"{name} manifest source commit is not the expected clean producer")
    artifact_map = terminal.get("artifacts")
    _require(type(artifact_map) is dict, "terminal artifact hash map is missing")
    for name, digest in artifact_map.items():
        _require(name in expected_artifacts and _valid_hash(digest) and
                 digest == expected_artifacts[name],
                 f"terminal artifact hash mismatch: {name}")
    return config, observed_hashes


def _validate_expected(
    expected_artifacts: object,
    expected_config_sha256: object,
    expected_source_commit: object,
    expected_counts: object,
    check: object,
) -> None:
    _require(type(expected_artifacts) is dict and set(expected_artifacts) == set(_ARTIFACT_NAMES),
             "expected_artifacts must name exactly the six registered artifacts")
    _require(all(_valid_hash(value) for value in expected_artifacts.values()),
             "expected artifact hashes must be lowercase SHA-256 strings")
    _require(_valid_hash(expected_config_sha256), "expected config hash must be lowercase SHA-256")
    _require(isinstance(expected_source_commit, str) and
             _COMMIT.fullmatch(expected_source_commit) is not None,
             "expected source commit must be a lowercase 40-character commit")
    _require(type(expected_counts) is dict and set(expected_counts) ==
             {"source_train", "source_validation"} and
             all(type(value) is int and value > 0 for value in expected_counts.values()),
             "expected_counts must contain positive source_train and source_validation counts")
    _require(callable(check), "check must be callable")


def _validate_projection(row: dict[str, Any], route: tuple[str, int, str | None]) -> tuple[str, int]:
    partition, candidate_index, split = route
    key = (partition, candidate_index)
    _require(partition == "source" and split in ("source_train", "source_validation"),
             "only source_train and source_validation projections may be exported")
    _require(type(row.get("profile")) is str and row["profile"] in _PROFILE_KINDS,
             "source projection has an unsupported primitive profile")
    _require(row.get("oracle_unique") is True and row.get("eligible") is True and
             row.get("sensitive") is True and row.get("envelope") is True,
             "source projection is not an accepted unique eligible envelope")
    _require(type(row.get("fine_stratum")) is str and row["fine_stratum"] in _FINE_STRATA,
             "source projection has an unknown fine stratum")
    for field in ("content_id", "lineage_id", "structure_signature", "model_input_sha256",
                  "outcome_sha256"):
        _require(_valid_hash(row.get(field)), f"source projection has invalid {field}")
    _require(type(row.get("values")) is list and
             all(type(value) is int and value in (0, 1, 2) for value in row["values"]),
             "source projection has invalid public value annotations")
    _require(type(row.get("given_count")) is int and not isinstance(row.get("given_count"), bool) and
             row["given_count"] > 0, "source projection has invalid given_count")
    _require(type(row.get("prompt_bytes")) is int and not isinstance(row.get("prompt_bytes"), bool) and
             row["prompt_bytes"] > 0, "source projection has invalid prompt_bytes")
    _require(type(row.get("first_solution_nodes")) is int and
             not isinstance(row.get("first_solution_nodes"), bool) and row["first_solution_nodes"] > 0,
             "source projection has invalid solver node annotation")
    return key


def _load_allowed_projections(input_dir: Path, check: Callable[[], Any], expected_counts: dict[str, int]):
    allowed: dict[tuple[str, int], dict[str, Any]] = {}
    seen_keys: set[tuple[str, int]] = set()
    seen_content: set[str] = set()
    counts = Counter()
    for raw_line, row in _iter_jsonl(input_dir / "records.jsonl", "records.jsonl", check):
        route = _route(row, context="records.jsonl")
        key = (route[0], route[1])
        _require(key not in seen_keys, "duplicate (partition, candidate_index) in accepted ledger")
        seen_keys.add(key)
        _require(route[2] is not None, "accepted projection ledger cannot contain null split rows")
        content_id = row.get("content_id")
        _require(_valid_hash(content_id), "accepted projection has invalid content_id")
        _require(content_id not in seen_content, "duplicate content_id in accepted projection ledger")
        seen_content.add(content_id)
        if route[2] not in ("source_train", "source_validation"):
            continue
        key = _validate_projection(row, route)
        _require(raw_line == producer._json_bytes(row),
                 "accepted projection line is not the producer's canonical serialization")
        allowed[key] = row
        counts[route[2]] += 1
    _require(dict(counts) == expected_counts, "accepted source projection counts differ from registration")
    return allowed


def _convert_allowed_row(row: dict[str, Any], raw: dict[str, Any], check: Callable[[], Any]) -> dict[str, Any]:
    _require(raw.get("state") == "first_seen" and raw.get("first_seen_content") is True,
             "allowed source task is not marked first_seen_content")
    task_text = raw.get("task")
    _require(type(task_text) is str, "allowed source task is not a serialized task string")
    try:
        task = schema.parse_task_json(task_text)
        model_input = schema.serialize_model_input(task)
        canonical_task = schema.serialize_task(task)
    except (TypeError, ValueError) as exc:
        raise SourceExportError("allowed source task violates strict task schema") from exc

    _require(_sha256(model_input.encode("utf-8")) == row["model_input_sha256"],
             "source task model-input hash mismatch")
    task_kinds = {constraint.kind.value for constraint in task.constraints}
    _require(task_kinds == _PROFILE_KINDS[row["profile"]],
             "source task primitive kinds differ from its profile")
    _check(check)
    solved = solver.solve(task)
    _check(check)
    _require(solved.solution is not None and solved.stats.uniqueness_solution_count_capped == 1,
             "reference solver did not find exactly one source solution")
    _check(check)
    exhaustive = oracle.enumerate_solutions(task)
    _check(check)
    _require(len(exhaustive.solutions) == 1,
             "independent exhaustive oracle did not find exactly one source solution")
    solution = exhaustive.solutions[0]
    _require(solved.solution.assignments == solution.assignments,
             "reference solver and independent oracle solutions differ")
    answer_text = stable_json({"assignment": dict(solution.assignments)})
    _require(verify_answer(task, answer_text).valid,
             "strict verifier rejected independent source solution")

    _check(check)
    identities = {
        "content_id": canonical.content_id(task),
        "lineage_id": canonical.lineage_id(task),
        "structure_signature": canonical.structure_signature(task),
    }
    _check(check)
    for name, computed in identities.items():
        _require(computed == row[name], f"source task {name} mismatch")
    return {
        "schema_version": "dwr-source-example-v1",
        "example_id": row["content_id"],
        "split": row["split"],
        "lineage_id": row["lineage_id"],
        "structure_signature": row["structure_signature"],
        "fine_stratum": row["fine_stratum"],
        "profile": row["profile"],
        "model_input_sha256": row["model_input_sha256"],
        "task": canonical_task,
        "solution": {"assignment": dict(solution.assignments)},
        "origin": {
            "partition": raw["partition"],
            "candidate_index": raw["candidate_index"],
            "outcome_sha256": row["outcome_sha256"],
        },
    }


def export_source_records(
    input_dir,
    output_path,
    *,
    expected_artifacts,
    generation_config_path,
    expected_config_sha256,
    expected_source_commit,
    expected_counts,
    check,
):
    """Convert only the registered source train/validation projections.

    All artifact bytes are hash-bound before JSON parsing. Routing metadata is
    checked before task, witness, or value semantics; output is exclusive and
    remains in place if a later validation fails.
    """
    _validate_expected(expected_artifacts, expected_config_sha256, expected_source_commit,
                       expected_counts, check)
    input_dir = _reject_reparse_chain(Path(input_dir), must_exist=True)
    _require(input_dir.is_dir(), "input_dir must be a directory")
    config_path = _reject_reparse_chain(Path(generation_config_path), must_exist=True)
    _require(config_path.is_file(), "generation_config_path must be a file")
    output_requested = Path(output_path)
    output_parent = _reject_reparse_chain(output_requested.parent, must_exist=True)
    _require(output_parent.is_dir(), "output parent must be a directory")
    _validate_windows_output_leaf(output_requested.name)
    output_path = output_parent / output_requested.name
    _require(not output_path.exists() and not output_path.is_symlink(),
             "output_path already exists")
    _require(not output_path.is_relative_to(input_dir),
             "output_path must be outside the input artifact directory")
    _require(output_path != config_path, "output_path overlaps the generation config")
    for name in _ARTIFACT_NAMES:
        _reject_reparse_chain(input_dir / name, must_exist=True)

    _check(check)
    try:
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise SourceExportError("cannot read generation config") from exc
    _check(check)
    _require(_sha256(config_bytes) == expected_config_sha256,
             "generation config bytes differ from the registered hash")
    _check(check)

    config, observed_hashes = _validate_metadata(
        input_dir, expected_artifacts, config_bytes, expected_config_sha256,
        expected_source_commit, check,
    )
    source_profiles = config.get("source_profiles")
    _require(type(source_profiles) is list and
             all(type(profile) is str for profile in source_profiles) and
             len(source_profiles) == len(set(source_profiles)) and
             set(source_profiles) == set(_PROFILE_KINDS),
             "generation config does not bind the registered source primitive profiles")

    try:
        with output_path.open("xb"):
            pass
    except OSError as exc:
        raise SourceExportError("cannot create exclusive source output") from exc

    allowed = _load_allowed_projections(input_dir, check, expected_counts)
    outputs: dict[tuple[str, int], dict[str, Any]] = {}
    seen_raw: set[tuple[str, int]] = set()
    split_counts = Counter()
    skipped_raw_rows = 0
    null_split_rows = 0

    for raw_line, raw in _iter_jsonl(input_dir / "outcomes.jsonl", "outcomes.jsonl", check):
        route = _route(raw, context="outcomes.jsonl")
        key = (route[0], route[1])
        _require(key not in seen_raw, "duplicate (partition, candidate_index) in raw outcomes")
        seen_raw.add(key)
        if route[2] is None:
            null_split_rows += 1
            skipped_raw_rows += 1
            continue
        if key not in allowed:
            skipped_raw_rows += 1
            continue

        projection = allowed[key]
        _require(route[2] == projection["split"],
                 "raw outcome split differs from its accepted projection")
        _require(_sha256(raw_line) == projection["outcome_sha256"],
                 "raw outcome line hash differs from accepted projection")
        raw_projection = producer._a_projection(raw)
        raw_projection["outcome_sha256"] = _sha256(raw_line)
        _require(_typed_equal(raw_projection, projection),
                 "raw outcome does not reproduce the producer's accepted projection")
        outputs[key] = _convert_allowed_row(projection, raw, check)
        split_counts[route[2]] += 1

    _require(set(outputs) == set(allowed), "accepted source projection has missing raw outcome rows")
    _require(dict(split_counts) == expected_counts,
             "raw accepted source outcome counts differ from registration")

    try:
        output_stat = output_path.lstat()
        output_reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        output_attributes = getattr(output_stat, "st_file_attributes", 0)
        _require(stat.S_ISREG(output_stat.st_mode) and not output_path.is_symlink() and
                 not (output_reparse and output_attributes & output_reparse),
                 "output_path is no longer the exclusive regular output file")
        with output_path.open("wb") as stream:
            for key in allowed:
                _check(check)
                stream.write((stable_json(outputs[key]) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise SourceExportError("cannot write exclusive source output") from exc
    output_hash = _hash_file(output_path, check)
    final_hashes = {}
    for name in _ARTIFACT_NAMES:
        final_hashes[name] = _hash_file(input_dir / name, check)
        _require(final_hashes[name] == observed_hashes[name],
                 f"artifact changed before export completion: {name}")
    fine_counts = Counter(row["fine_stratum"] for row in outputs.values())
    return {
        "exported_counts": dict(expected_counts),
        "fine_stratum_counts": dict(sorted(fine_counts.items())),
        "exported_records": len(outputs),
        "allowed_task_payloads_parsed": len(outputs),
        "protected_task_payloads_semantically_parsed": 0,
        "skipped_raw_rows": skipped_raw_rows,
        "construction_failed_null_split_rows": null_split_rows,
        "input_sha256": final_hashes,
        "generation_config_sha256": expected_config_sha256,
        "output_sha256": output_hash,
    }
