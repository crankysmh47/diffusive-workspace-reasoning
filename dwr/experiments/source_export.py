"""Closed, byte-bound wrapper for the registered retained-source export."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

from dwr.experiments.records import RunRecord
from dwr.experiments.source_corpus import export_source_records
from dwr.tasks.run_support import ResourceGuard, stable_json, strict_json_loads


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_PACKAGE_ROOT = _REPO_ROOT
_RUNTIME_SOURCE_PATHS = (
    "dwr/experiments/source_export.py",
    "dwr/experiments/source_corpus.py",
    "dwr/experiments/records.py",
    "dwr/tasks/run_support.py",
    "dwr/tasks/certificate_rank_study.py",
    "dwr/tasks/schema.py",
    "dwr/tasks/canonical.py",
    "dwr/tasks/solver.py",
    "dwr/tasks/oracle.py",
    "dwr/tasks/verifier.py",
)
_CONFIG_RELATIVE = "configs/tasks/source_export_v1.json"
_PRODUCTION_CONFIG_SHA256 = "662395596baa37f432a2eff7003e99c22697172cf6dff584a39e96e734ecd7f8"
_REGISTERED_PATHS = {
    "input_dir": "runs/certificate-five-stratum-full-20261014-v1",
    "generation_config_path": "configs/tasks/certificate_five_stratum_full_v1.json",
    "structural_result_path": "configs/splits/certificate-five-stratum-study-result.json",
    "output_dir": "runs/source-export-20261015-v1",
}
_SIX_ARTIFACTS = {
    "manifest.initial.json", "manifest.json", "outcomes.jsonl", "records.jsonl", "replay.json", "report.json",
}
_CONFIG_KEYS = {
    "artifact_sha256", "config_version", "expected_counts", "forbidden_payload_splits",
    "generation_config_path", "generation_config_sha256", "input_dir", "limits", "model_execution",
    "operation", "output_dir", "output_filename", "permitted_payload_splits", "producer_commit",
    "root_seed_unused", "structural_result_path", "structural_result_sha256",
}
_FORBIDDEN = ["source_test", "development_validation", "development_diagnostic_test", "final_test"]
_PERMITTED = ["source_train", "source_validation"]
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SUMMARY_KEYS = {
    "schema_version", "operation", "exported_counts", "fine_stratum_counts", "exported_records",
    "allowed_task_payloads_parsed", "protected_task_payloads_semantically_parsed", "skipped_raw_rows",
    "construction_failed_null_split_rows", "input_sha256", "generation_config_sha256", "output_sha256",
    "access", "costs",
}


class SourceExportRunError(RuntimeError):
    """A closed source-export invocation failed and its evidence was preserved."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_typed_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc
    _need(type(value) is dict, f"{label} must contain a JSON object")
    return value


def _linked(path: Path, *, must_exist: bool) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for item in reversed((absolute, *absolute.parents)):
        if not item.exists() and not item.is_symlink():
            if must_exist:
                raise ValueError(f"registered path is missing: {item}")
            continue
        try:
            info = item.lstat()
        except OSError as exc:
            raise ValueError(f"cannot inspect registered path: {item}") from exc
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(info.st_mode) or (getattr(info, "st_file_attributes", 0) & reparse):
            raise ValueError("registered paths cannot contain symlinks or reparse points")


def _repo_path(root: Path, relative: Any, *, must_exist: bool) -> Path:
    _need(type(relative) is str and bool(relative) and "\\" not in relative,
          "registered paths must be nonempty root-relative POSIX paths")
    pure = PurePosixPath(relative)
    _need(not pure.is_absolute() and not re.match(r"^[A-Za-z]:", relative),
          "registered path must be relative")
    _need(all(part not in ("", ".", "..") for part in relative.split("/")),
          "registered path contains traversal or alias components")
    candidate = root.joinpath(*pure.parts)
    _linked(candidate, must_exist=must_exist)
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=must_exist)
    _need(resolved == resolved_root or resolved_root in resolved.parents,
          "registered path escapes repository root")
    if must_exist:
        _need(resolved.is_file() or resolved.is_dir(), "registered path has an unsupported file type")
    return resolved


def _read_regular(path: Path, label: str) -> bytes:
    _linked(path, must_exist=True)
    _need(path.is_file(), f"{label} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}") from exc


def _load_config(root: Path, requested: str | os.PathLike[str], check) -> tuple[dict[str, Any], bytes, Path]:
    check()
    expected_path = _repo_path(root, _CONFIG_RELATIVE, must_exist=True)
    requested_path = Path(requested)
    if not requested_path.is_absolute():
        _need(requested_path.as_posix() == _CONFIG_RELATIVE,
              "only the registered source-export config path is accepted")
        actual_path = _repo_path(root, requested_path.as_posix(), must_exist=True)
    else:
        _linked(requested_path, must_exist=True)
        actual_path = requested_path.resolve(strict=True)
    _need(actual_path == expected_path, "config path is outside the registered source-export config")
    config_raw = _read_regular(actual_path, "source-export config")
    _need(_sha(config_raw) == _PRODUCTION_CONFIG_SHA256,
          "source-export config bytes differ from the closed baseline hash")
    check()
    baseline_raw = _read_regular(expected_path, "committed source-export baseline")
    _need(_sha(baseline_raw) == _PRODUCTION_CONFIG_SHA256,
          "committed source-export baseline bytes differ from the reviewed hash")
    baseline = _strict_json(baseline_raw, "committed source-export baseline")
    config = _strict_json(config_raw, "source-export config")
    _need(set(config) == _CONFIG_KEYS and set(baseline) == _CONFIG_KEYS,
          "source-export config has unknown or missing fields")
    _need(_typed_equal(config, baseline), "source-export config differs from its byte-bound baseline")
    _validate_config(config)
    return config, config_raw, actual_path


def _validate_config(config: dict[str, Any]) -> None:
    _need(config["config_version"] == "dwr-source-export-v1" and
          config["operation"] == "source-only-retained-corpus-conversion",
          "unsupported source-export operation or config version")
    _need(config["model_execution"] is False, "source export cannot enable model execution")
    _need(config["permitted_payload_splits"] == _PERMITTED and
          config["forbidden_payload_splits"] == _FORBIDDEN,
          "source-export split permissions differ from registration")
    _need(type(config["expected_counts"]) is dict and set(config["expected_counts"]) == set(_PERMITTED) and
          all(type(value) is int and value > 0 for value in config["expected_counts"].values()),
          "source-export counts must be positive registered source counts")
    _need(type(config["artifact_sha256"]) is dict and set(config["artifact_sha256"]) == _SIX_ARTIFACTS and
          all(type(value) is str and _HASH.fullmatch(value) for value in config["artifact_sha256"].values()),
          "source-export must bind exactly the six retained artifacts")
    _need(type(config["generation_config_sha256"]) is str and _HASH.fullmatch(config["generation_config_sha256"]),
          "invalid generation config identity")
    _need(type(config["structural_result_sha256"]) is str and _HASH.fullmatch(config["structural_result_sha256"]),
          "invalid structural result identity")
    _need(type(config["producer_commit"]) is str and _COMMIT.fullmatch(config["producer_commit"]),
          "invalid retained producer commit")
    _need(type(config["root_seed_unused"]) is int and config["root_seed_unused"] >= 0,
          "invalid declared unused seed")
    limits = config["limits"]
    _need(type(limits) is dict and set(limits) == {
        "attempts", "cpu_workers", "external_watchdog_seconds", "memory_mib", "wall_seconds"},
        "source-export limits have unknown or missing fields")
    _need(_typed_equal(limits, {"attempts": 1, "cpu_workers": 1, "external_watchdog_seconds": 305,
                                "memory_mib": 512, "wall_seconds": 300}),
          "source-export limits differ from the registered budget")
    _need(config["output_filename"] == "source_examples.jsonl", "source-export output filename is fixed")
    for key, expected in _REGISTERED_PATHS.items():
        _need(config.get(key) == expected, f"{key} differs from the root-relative allowlist")


def _verify_proof(root: Path, config: dict[str, Any], check) -> Path:
    proof_path = _repo_path(root, config["structural_result_path"], must_exist=True)
    check()
    raw = _read_regular(proof_path, "public structural proof")
    _need(_sha(raw) == config["structural_result_sha256"],
          "public structural proof bytes differ from the registered hash")
    proof = _strict_json(raw, "public structural proof")
    full = proof.get("full")
    _need(proof.get("status") == "PASS_G2_FIVE_STRATUM_V1" and type(full) is dict,
          "public structural proof does not record the registered G2 pass")
    _need(full.get("config_sha256") == config["generation_config_sha256"] and
          full.get("artifact_sha256") == config["artifact_sha256"] and
          proof.get("source_commit") == config["producer_commit"],
          "public structural proof identities differ from the source-export recipe")
    checks = full.get("controller_checks")
    _need(type(checks) is dict and checks.get("all_stages_pass") is True and
          all(value is True for value in checks.values()),
          "public structural proof controller checks are not all passing")
    return proof_path


def _verify_runtime_code(source_root: Path, check) -> None:
    for relative in _RUNTIME_SOURCE_PATHS:
        check()
        running_path = _RUNTIME_PACKAGE_ROOT.joinpath(*PurePosixPath(relative).parts)
        committed_path = source_root.joinpath(*PurePosixPath(relative).parts)
        running = _read_regular(running_path, f"runtime package file {relative}")
        committed = _read_regular(committed_path, f"source package file {relative}")
        _need(running == committed, f"runtime package bytes differ from registered source: {relative}")


def _load_output_paths(root: Path, config: dict[str, Any]) -> tuple[Path, Path, Path]:
    input_dir = _repo_path(root, config["input_dir"], must_exist=True)
    _need(input_dir.is_dir(), "registered source input path must be a directory")
    generation = _repo_path(root, config["generation_config_path"], must_exist=True)
    _need(generation.is_file(), "registered generation config must be a regular file")
    output_dir = _repo_path(root, config["output_dir"], must_exist=False)
    _need(not output_dir.exists(), "registered source-export output directory already exists")
    _need(output_dir.parent.is_dir(), "registered source-export output parent is missing")
    _need(output_dir.parent == (root / "runs").resolve(strict=True),
          "registered output must be a unique direct child of runs")
    return input_dir, generation, output_dir


def _reject_tracked_payloads(root: Path, input_dir: Path, check) -> None:
    """Keep RunRecord's source inventory from opening retained payload bytes pre-manifest."""
    relative = input_dir.relative_to(root).as_posix()
    check()
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--", relative], cwd=root,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("cannot inspect the source index before manifest creation") from exc
    check()
    tracked = {os.fsdecode(item).replace("\\", "/") for item in result.stdout.split(b"\0") if item}
    protected_paths = {f"{relative}/{name}" for name in _SIX_ARTIFACTS}
    _need(not (tracked & protected_paths),
          "retained source artifacts must be untracked and ignored before initial manifest creation")


def _summary_from_export(result: dict[str, Any], *, config: dict[str, Any], wall: float,
                         cpu: float, peak_rss: int | None) -> dict[str, Any]:
    value = {
        "schema_version": "dwr-source-export-summary-v1",
        "operation": config["operation"],
        **result,
        "access": {
            "permitted_payload_splits": list(_PERMITTED),
            "forbidden_payload_splits": list(_FORBIDDEN),
            "full_file_opaque_hashes": True,
            "raw_json_parsing": "full outcomes JSONL strict parse materializes row fields; route first, then skip task, witness and values semantics outside permitted source splits",
            "protected_task_payloads_semantically_parsed": 0,
        },
        "costs": {
            "total_wall_seconds": wall,
            "total_cpu_seconds": cpu,
            "peak_rss_bytes": peak_rss,
            "artifact_write_wall_seconds": None,
            "artifact_write_cpu_seconds": None,
            "flops": None,
            "gpu": None,
        },
    }
    _need(set(value) == _SUMMARY_KEYS, "internal source-export summary schema mismatch")
    return value


def _write_exclusive(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _partial_artifacts(run_dir: Path) -> list[str]:
    result = []
    for name in ("source_examples.jsonl", "summary.json"):
        candidate = run_dir / name
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and not (
                getattr(info, "st_file_attributes", 0) & reparse):
            result.append(name)
    return result


def _preserve_failure(record: RunRecord | None, run_dir: Path | None, error: BaseException,
                      status: str) -> None:
    if record is None or run_dir is None or not run_dir.exists() or (run_dir / "manifest.json").exists():
        return
    message = f"{type(error).__name__}: {error}"[:2000] or type(error).__name__
    try:
        record.append_event("run_interrupted" if status == "interrupted" else "run_failed",
                            {"error_type": type(error).__name__, "error": str(error)[:2000]})
    except Exception:
        pass
    try:
        record.finish(status, counts={}, artifacts=_partial_artifacts(run_dir), error=message)
    except Exception as finalize_error:
        if hasattr(error, "add_note"):
            error.add_note(f"terminal evidence finalization also failed: {type(finalize_error).__name__}: {finalize_error}")


def run_export(config_path: str | os.PathLike[str], *, command: list[str]) -> dict[str, Any]:
    """Run the one registered conversion; no caller-supplied scientific overrides exist."""
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    guard = ResourceGuard(300, 512)
    guard.check()
    record: RunRecord | None = None
    output_dir: Path | None = None
    try:
        _need(type(command) is list and bool(command) and
              all(type(part) is str and bool(part) for part in command),
              "command must be a nonempty list of strings")
        root = Path(os.path.abspath(os.fspath(_REPO_ROOT)))
        _linked(root, must_exist=True)
        root = root.resolve(strict=True)
        config, config_raw, resolved_config_path = _load_config(root, config_path, guard.check)
        input_dir, generation_path, output_dir = _load_output_paths(root, config)
        _reject_tracked_payloads(root, input_dir, guard.check)
        proof_path = _verify_proof(root, config, guard.check)
        _verify_runtime_code(root, guard.check)
        guard.check()

        artifact_map = config["artifact_sha256"]
        identities = {
            "retained_source_corpus": {
                "name": "five-stratum-retained-source-population",
                "revision": config["producer_commit"],
                "sha256": _sha(stable_json(artifact_map).encode("utf-8")),
            },
            "structural_proof": {
                "name": "five-stratum-structural-result",
                "revision": "PASS_G2_FIVE_STRATUM_V1",
                "sha256": config["structural_result_sha256"],
            },
            "generation_config": {
                "name": "certificate-five-stratum-generation-config",
                "revision": "registered-v1",
                "sha256": config["generation_config_sha256"],
            },
            "source_export_config": {
                "name": "source-export-config",
                "revision": config["config_version"],
                "sha256": _sha(config_raw),
            },
        }
        settings = {
            "scientific_status": "deterministic_export_not_hypothesis_test",
            "operation": config["operation"],
            "permitted_payload_splits": list(_PERMITTED),
            "forbidden_payload_splits": list(_FORBIDDEN),
            "access_policy": "opaque_hash_all_six; strict JSON parsing materializes outcomes row fields; route first and semantically consume task payloads only for source_train and source_validation",
            "protected_task_semantics_consumed": 0,
            "root_seed_policy": "declared_but_unused_deterministic_conversion",
            "cpu_workers": 1,
            "dtype": "not_applicable_stdlib_cpu",
            "cache": "not_applicable",
            "gpu": "unavailable_not_probed",
            "model_execution": False,
        }
        budgets = {"wall_seconds": 300, "memory_mib": 512, "attempts": 1}
        record = RunRecord.create(
            output_dir, command=command, config_path=resolved_config_path,
            expected_config_sha256=_PRODUCTION_CONFIG_SHA256, identities=identities,
            rng_streams={"root_seed_unused": config["root_seed_unused"]}, settings=settings,
            budgets=budgets, confirmatory=True, source_root=root,
        )
        guard.check()
        record.append_event("source_access_policy", {
            "permitted_payload_splits": list(_PERMITTED),
            "forbidden_payload_splits": list(_FORBIDDEN),
            "full_file_opaque_hashes": True,
            "raw_json_parsing": "full outcomes JSONL strict parse materializes row fields; route first, then skip task, witness and values semantics outside permitted source splits",
            "protected_task_payloads_semantically_parsed": 0,
        })
        guard.check()
        result = export_source_records(
            input_dir, output_dir / config["output_filename"],
            expected_artifacts=artifact_map,
            generation_config_path=generation_path,
            expected_config_sha256=config["generation_config_sha256"],
            expected_source_commit=config["producer_commit"],
            expected_counts=config["expected_counts"],
            check=guard.check,
        )
        guard.check()
        wall = max(0.0, time.perf_counter() - wall_started)
        cpu = max(0.0, time.process_time() - cpu_started)
        summary = _summary_from_export(result, config=config, wall=wall, cpu=cpu,
                                       peak_rss=guard.peak_bytes)
        summary_raw = (stable_json(summary) + "\n").encode("utf-8")
        _write_exclusive(output_dir / "summary.json", summary_raw)
        guard.check()
        terminal_counts = {**result["exported_counts"], "exported_records": result["exported_records"]}
        record.append_event("export_completed", {
            "exported_counts": result["exported_counts"],
            "exported_records": result["exported_records"],
            "source_output_sha256": result["output_sha256"],
        })
        record.finish("completed", counts=terminal_counts,
                      artifacts=[config["output_filename"], "summary.json"])
        return summary
    except BaseException as error:
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit, TimeoutError, MemoryError)) else "failed"
        _preserve_failure(record, output_dir, error, status)
        raise


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the closed retained-source export")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", "dwr.experiments.source_export", "--config", args.config]
    summary = run_export(args.config, command=command)
    print(stable_json({
        "status": "completed",
        "exported_counts": summary["exported_counts"],
        "exported_records": summary["exported_records"],
        "output_sha256": summary["output_sha256"],
        "output_path": str((_REPO_ROOT / _REGISTERED_PATHS["output_dir"] / "source_examples.jsonl").resolve()),
        "summary_path": str((_REPO_ROOT / _REGISTERED_PATHS["output_dir"] / "summary.json").resolve()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
