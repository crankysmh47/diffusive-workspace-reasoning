"""Strict, standard-library reader for a completed retained-source export."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from dwr.experiments.records import verify_artifacts
from dwr.tasks import canonical, schema
from dwr.tasks.run_support import stable_json, strict_json_loads
from dwr.tasks.verifier import verify_answer


_SPLITS = {"source_train", "source_validation"}
_PROFILES = {"eq": {"eq"}, "neq": {"neq"}, "add_mod": {"add_mod"}}
_STRATA = {"A2", "A3", "B2", "C2", "D2"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {
    "schema_version", "example_id", "split", "lineage_id", "structure_signature", "fine_stratum",
    "profile", "model_input_sha256", "task", "solution", "origin",
}
_SUMMARY_FIELDS = {
    "schema_version", "operation", "exported_counts", "fine_stratum_counts", "exported_records",
    "allowed_task_payloads_parsed", "protected_task_payloads_semantically_parsed", "skipped_raw_rows",
    "construction_failed_null_split_rows", "input_sha256", "generation_config_sha256", "output_sha256",
    "access", "costs",
}


class SourceReaderError(ValueError):
    """A source export artifact violates its byte-bound consumer schema."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise SourceReaderError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_hash(value: Any) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def _reject_links(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for item in reversed((absolute, *absolute.parents)):
        if not item.exists() and not item.is_symlink():
            continue
        try:
            info = item.lstat()
        except OSError as exc:
            raise SourceReaderError(f"cannot inspect export path: {item}") from exc
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
            raise SourceReaderError("source export paths cannot contain symlinks or reparse points")


def _read_regular(path: Path, label: str) -> bytes:
    _reject_links(path)
    _need(path.is_file(), f"{label} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SourceReaderError(f"cannot read {label}") from exc


def _hash_regular(path: Path, label: str) -> str:
    _reject_links(path)
    _need(path.is_file(), f"{label} must be a regular file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SourceReaderError(f"cannot hash {label}") from exc
    return digest.hexdigest()


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise SourceReaderError(f"{label} must be strict UTF-8 JSON") from exc
    _need(type(value) is dict, f"{label} must be a JSON object")
    return value


def _validate_terminal_artifacts(terminal: dict[str, Any], summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = terminal.get("artifacts")
    _need(type(artifacts) is list and len(artifacts) == 2,
          "completed source export must declare exactly its output and summary")
    mapping: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        _need(type(artifact) is dict and set(artifact) == {"path", "size_bytes", "sha256"},
              "invalid terminal artifact entry")
        path = artifact.get("path")
        _need(type(path) is str and path in {"source_examples.jsonl", "summary.json"},
              "terminal declares an unexpected source-export artifact")
        _need(path not in mapping, "terminal declares a duplicate source-export artifact")
        _need(type(artifact.get("size_bytes")) is int and artifact["size_bytes"] >= 0 and
              _valid_hash(artifact.get("sha256")), "terminal has an invalid artifact identity")
        mapping[path] = artifact
    _need(set(mapping) == {"source_examples.jsonl", "summary.json"},
          "terminal source-export artifact set is incomplete")
    _need(summary["output_sha256"] == mapping["source_examples.jsonl"]["sha256"],
          "summary output hash differs from terminal artifact identity")
    return mapping


def _validate_summary(summary: dict[str, Any], terminal: dict[str, Any], initial: dict[str, Any]) -> dict[str, int]:
    _need(set(summary) == _SUMMARY_FIELDS, "summary has unknown or missing fields")
    _need(summary["schema_version"] == "dwr-source-export-summary-v1" and
          summary["operation"] == "source-only-retained-corpus-conversion",
          "unsupported source-export summary")
    config_record = initial.get("config")
    resolved = config_record.get("resolved") if type(config_record) is dict else None
    _need(type(resolved) is dict and resolved.get("config_version") == "dwr-source-export-v1" and
          resolved.get("operation") == summary["operation"],
          "initial config does not bind this source-export operation and version")
    _need(_valid_hash(summary["generation_config_sha256"]), "summary has an invalid generation config hash")
    _need(type(summary["output_sha256"]) is str and _HASH.fullmatch(summary["output_sha256"]),
          "summary has an invalid output hash")
    counts = summary["exported_counts"]
    _need(type(counts) is dict and set(counts) == _SPLITS and
          all(type(count) is int and count >= 0 for count in counts.values()),
          "summary has invalid exported split counts")
    registered_counts = resolved.get("expected_counts")
    _need(type(registered_counts) is dict and set(registered_counts) == _SPLITS and
          all(type(count) is int and count > 0 for count in registered_counts.values()) and
          counts == registered_counts,
          "summary split counts differ from the initial registered config")
    _need(type(summary["exported_records"]) is int and not isinstance(summary["exported_records"], bool) and
          summary["exported_records"] == sum(counts.values()), "summary record count differs from split counts")
    _need(type(terminal.get("counts")) is dict and
          terminal["counts"] == {**counts, "exported_records": summary["exported_records"]},
          "terminal counts differ from the source-export summary")
    for key in ("allowed_task_payloads_parsed", "protected_task_payloads_semantically_parsed",
                "skipped_raw_rows", "construction_failed_null_split_rows"):
        _need(type(summary[key]) is int and summary[key] >= 0, f"summary {key} is invalid")
    _need(summary["allowed_task_payloads_parsed"] == summary["exported_records"] and
          summary["protected_task_payloads_semantically_parsed"] == 0,
          "summary source access totals are inconsistent")
    _need(type(summary["fine_stratum_counts"]) is dict and
          all(type(name) is str and name in _STRATA and type(count) is int and count >= 0
              for name, count in summary["fine_stratum_counts"].items()) and
          sum(summary["fine_stratum_counts"].values()) == summary["exported_records"],
          "summary fine-stratum counts are inconsistent")
    input_hashes = summary["input_sha256"]
    _need(type(input_hashes) is dict and set(input_hashes) == {
        "manifest.initial.json", "manifest.json", "outcomes.jsonl", "records.jsonl", "replay.json", "report.json"} and
        all(_valid_hash(value) for value in input_hashes.values()), "summary input identity map is invalid")
    _need(input_hashes == resolved.get("artifact_sha256") and
          summary["generation_config_sha256"] == resolved.get("generation_config_sha256"),
          "summary source identities differ from the initial registered config")
    access = summary["access"]
    _need(type(access) is dict and set(access) == {"permitted_payload_splits", "forbidden_payload_splits",
          "full_file_opaque_hashes", "raw_json_parsing", "protected_task_payloads_semantically_parsed"} and
          access.get("permitted_payload_splits") == ["source_train", "source_validation"] and
          access.get("forbidden_payload_splits") == ["source_test", "development_validation",
             "development_diagnostic_test", "final_test"] and
          access.get("full_file_opaque_hashes") is True and
          access.get("raw_json_parsing") == "full outcomes JSONL strict parse materializes row fields; route first, then skip task, witness and values semantics outside permitted source splits" and
          type(access.get("protected_task_payloads_semantically_parsed")) is int and
          access.get("protected_task_payloads_semantically_parsed") == 0,
          "summary source access accounting does not match the registered boundary")
    costs = summary["costs"]
    _need(type(costs) is dict and set(costs) == {"total_wall_seconds", "total_cpu_seconds", "peak_rss_bytes",
         "artifact_write_wall_seconds", "artifact_write_cpu_seconds", "flops", "gpu"},
         "summary cost fields are incomplete")
    for key in ("total_wall_seconds", "total_cpu_seconds"):
        _need(type(costs[key]) in (int, float) and not isinstance(costs[key], bool) and costs[key] >= 0,
              f"summary {key} is invalid")
    _need(costs["peak_rss_bytes"] is None or
          (type(costs["peak_rss_bytes"]) is int and costs["peak_rss_bytes"] >= 0),
          "summary peak RSS is invalid")
    for key in ("artifact_write_wall_seconds", "artifact_write_cpu_seconds", "flops", "gpu"):
        _need(costs[key] is None, f"summary unavailable cost {key} must be null")
    return counts


def _validate_metadata(row: dict[str, Any], seen_ids: set[str], seen_origins: set[tuple[str, int]]) -> tuple[str, tuple[str, int]]:
    _need(set(row) == _FIELDS, "source example has unknown or missing fields")
    _need(row.get("schema_version") == "dwr-source-example-v1", "unsupported source example schema")
    split = row.get("split")
    _need(type(split) is str and split in _SPLITS, "source example has a forbidden or unknown split")
    example_id = row.get("example_id")
    _need(_valid_hash(example_id) and _valid_hash(row.get("lineage_id")) and
          _valid_hash(row.get("structure_signature")) and _valid_hash(row.get("model_input_sha256")),
          "source example has an invalid public identity")
    _need(example_id not in seen_ids, "duplicate source example id in export")
    seen_ids.add(example_id)
    _need(type(row.get("fine_stratum")) is str and row["fine_stratum"] in _STRATA,
          "source example has an unknown fine stratum")
    _need(type(row.get("profile")) is str and row["profile"] in _PROFILES,
          "source example has an unknown primitive profile")
    origin = row.get("origin")
    _need(type(origin) is dict and set(origin) == {"partition", "candidate_index", "outcome_sha256"},
          "source example origin has unknown or missing fields")
    _need(origin.get("partition") == "source" and type(origin.get("candidate_index")) is int and
          origin["candidate_index"] >= 0 and _valid_hash(origin.get("outcome_sha256")),
          "source example has invalid source origin metadata")
    origin_key = (origin["partition"], origin["candidate_index"])
    _need(origin_key not in seen_origins, "duplicate source origin in export")
    seen_origins.add(origin_key)
    return split, origin_key


def _validate_requested_semantics(row: dict[str, Any]) -> None:
    task_text = row.get("task")
    _need(type(task_text) is str, "requested source task must be a serialized task string")
    try:
        task = schema.parse_task_json(task_text)
        canonical_task = schema.serialize_task(task)
        model_input = schema.serialize_model_input(task)
    except (TypeError, ValueError) as exc:
        raise SourceReaderError("requested source task violates strict task schema") from exc
    _need(task_text == canonical_task, "source task serialization is not canonical")
    _need(_sha(model_input.encode("utf-8")) == row["model_input_sha256"],
          "source task model-input identity mismatch")
    kinds = {constraint.kind.value for constraint in task.constraints}
    _need(kinds == _PROFILES[row["profile"]], "source task primitive profile differs from metadata")
    identities = {
        "example_id": canonical.content_id(task),
        "lineage_id": canonical.lineage_id(task),
        "structure_signature": canonical.structure_signature(task),
    }
    for key, observed in identities.items():
        _need(row[key] == observed, f"source task {key} mismatch")

    solution = row.get("solution")
    _need(type(solution) is dict and set(solution) == {"assignment"} and
          type(solution["assignment"]) is dict, "source solution must be exactly an assignment object")
    assignment = solution["assignment"]
    _need(set(assignment) == set(task.variables) and
          all(type(value) is int for value in assignment.values()),
          "source solution assignment has invalid variables or values")
    answer_text = stable_json({"assignment": assignment})
    _need(verify_answer(task, answer_text).valid, "strict verifier rejected the source solution")
    _need(stable_json(solution) == stable_json({"assignment": assignment}),
          "source solution serialization is not canonical")


def read_source_export(run_dir: str | os.PathLike[str], *, expected_manifest_sha256: str,
                       split: str) -> tuple[dict[str, Any], ...]:
    """Validate all routing/integrity and requested task semantics, returning only one split."""
    _need(type(expected_manifest_sha256) is str and _HASH.fullmatch(expected_manifest_sha256),
          "expected_manifest_sha256 must be lowercase SHA-256")
    _need(type(split) is str and split in _SPLITS, "split must be source_train or source_validation")
    root = Path(os.path.abspath(os.fspath(run_dir)))
    _reject_links(root)
    _need(root.is_dir(), "source-export run directory is missing")
    terminal_path = root / "manifest.json"
    terminal_raw = _read_regular(terminal_path, "terminal manifest")
    _need(_sha(terminal_raw) == expected_manifest_sha256,
          "terminal manifest bytes differ from the caller-bound identity")
    terminal = _json_object(terminal_raw, "terminal manifest")
    _need(terminal.get("status") == "completed", "source export is not completed")
    validated_terminal = verify_artifacts(root)
    _need(validated_terminal == terminal, "terminal changed during source-export validation")
    initial = _json_object(_read_regular(root / "manifest.initial.json", "initial manifest"), "initial manifest")
    summary_raw = _read_regular(root / "summary.json", "source-export summary")
    summary = _json_object(summary_raw, "source-export summary")
    artifact_map = _validate_terminal_artifacts(terminal, summary)
    counts = _validate_summary(summary, terminal, initial)
    _need(len(summary_raw) == artifact_map["summary.json"]["size_bytes"] and
          _sha(summary_raw) == artifact_map["summary.json"]["sha256"],
          "summary bytes differ from terminal artifact identity")

    output_path = root / "source_examples.jsonl"
    _reject_links(output_path)
    _need(output_path.is_file(), "source example output is missing")
    digest = hashlib.sha256()
    seen_ids: set[str] = set()
    seen_origins: set[tuple[str, int]] = set()
    observed_counts = {name: 0 for name in _SPLITS}
    observed_fine_counts: dict[str, int] = {}
    requested: list[dict[str, Any]] = []
    try:
        with output_path.open("rb") as stream:
            line_number = 0
            while True:
                raw_line = stream.readline()
                if not raw_line:
                    break
                line_number += 1
                digest.update(raw_line)
                _need(raw_line.endswith(b"\n"), f"source example line {line_number} lacks LF terminator")
                try:
                    row = strict_json_loads(raw_line[:-1].decode("utf-8", errors="strict"))
                except (UnicodeError, ValueError, TypeError) as exc:
                    raise SourceReaderError(f"source example line {line_number} is not strict JSON") from exc
                _need(type(row) is dict, f"source example line {line_number} must be an object")
                route = row.get("split")
                _need(type(route) is str and route in _SPLITS,
                      f"source example line {line_number} has an invalid split route")
                _need(raw_line == (stable_json(row) + "\n").encode("utf-8"),
                      f"source example line {line_number} is not canonical JSONL")
                row_split, _ = _validate_metadata(row, seen_ids, seen_origins)
                observed_counts[row_split] += 1
                observed_fine_counts[row["fine_stratum"]] = observed_fine_counts.get(row["fine_stratum"], 0) + 1
                if row_split == split:
                    _validate_requested_semantics(row)
                    requested.append(row)
    except OSError as exc:
        raise SourceReaderError("cannot stream source example output") from exc

    _need(observed_counts == counts, "source example routing counts differ from summary and terminal")
    _need(observed_fine_counts == summary["fine_stratum_counts"],
          "source example fine-stratum counts differ from summary")
    _need(digest.hexdigest() == summary["output_sha256"] == artifact_map["source_examples.jsonl"]["sha256"],
          "source example bytes differ from summary or terminal identity")
    _need(_hash_regular(output_path, "source example output") == digest.hexdigest(),
          "source example bytes changed while being read")
    final_terminal_raw = _read_regular(terminal_path, "terminal manifest")
    _need(_sha(final_terminal_raw) == expected_manifest_sha256 and final_terminal_raw == terminal_raw,
          "terminal manifest changed while reading source examples")
    _need(verify_artifacts(root) == terminal, "source-export artifacts changed while being read")
    return tuple(requested)
