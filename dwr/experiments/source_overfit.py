"""Pinned worker for the recoverable three-example source overfit."""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import copy
import random
import stat
import sys
import threading
import time
from typing import Any, Callable, Mapping

from dwr.experiments.records import RunRecord, verify_artifacts
from dwr.tasks import schema
from dwr.tasks.run_support import stable_json


CONFIG_PATH = "configs/experiments/source_overfit_recoverable_v2.json"
CONFIG_SHA256 = "e6c3de7e59e58c494b89052086c0840dcd3421d17396fd5a55cac40611049594"
_HEARTBEAT_WRITE_LOCK = threading.Lock()
COHORT_SHA256 = "b7319af014508a13dea46c19169ff36800c5db17c6d22d7c28c6e72d44b46932"
FEASIBILITY_MODULE_SHA256 = "5c8c8f36f198f08ab906be86e88c2dbac04471f7e6f5585b93d1fb402a3c5139"
REFERENCE_MODULE_SHA256 = "729431dd17b9e7b4c612d314f0f9dea1a266945a04f68818ce194ec181f17cad"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_DATA_IDENTITY_FIELDS = (
    "source_export_manifest_sha256", "source_export_data_sha256",
    "source_token_audit_manifest_sha256", "token_lengths_sha256", "cohort_sha256",
    "feasibility_manifest_sha256", "feasibility_outcome_sha256",
    "reference_manifest_sha256", "reference_summary_sha256",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _verified_reference_proofs(reference_dir: Path, dependency: Mapping[str, Any]) -> tuple[dict, bytes, dict]:
    terminal = verify_artifacts(reference_dir)
    manifest_raw = _read(reference_dir / "manifest.json", "frozen-reference terminal manifest")
    if _sha(manifest_raw) != dependency["reference_manifest_sha256"]:
        raise ValueError("frozen-reference terminal manifest differs from its registered identity")
    if terminal.get("status") != "completed":
        raise ValueError("pinned frozen reference record is not completed")
    summary_raw = _read(reference_dir / "summary.json", "frozen-reference summary")
    if _sha(summary_raw) != dependency["reference_summary_sha256"]:
        raise ValueError("frozen-reference summary differs from its registered identity")
    return terminal, summary_raw, _json_object(summary_raw, "frozen-reference summary")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _reject_links(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for item in reversed((absolute, *absolute.parents)):
        if not item.exists() and not item.is_symlink():
            continue
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
            raise ValueError("source-overfit paths cannot contain links or reparse points")


def _read(path: Path, label: str) -> bytes:
    _reject_links(path)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.read_bytes()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_unique_object,
                           parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_registration(root: Path, config_path: str | os.PathLike[str]) -> dict[str, Any]:
    expected_lexical = root.joinpath(*PurePosixPath(CONFIG_PATH).parts)
    _reject_links(expected_lexical)
    expected = expected_lexical.resolve(strict=True)
    requested = Path(os.path.abspath(os.fspath(config_path)))
    _reject_links(requested)
    if requested.resolve(strict=True) != expected:
        raise ValueError("only the registered recoverable source-overfit config is accepted")
    config_raw = _read(expected_lexical, "source-overfit config")
    if _sha(config_raw) != CONFIG_SHA256:
        raise ValueError("source-overfit config bytes differ from the closed registration")
    config = _json_object(config_raw, "source-overfit config")
    if config.get("config_version") != "dwr-source-overfit-recoverable-v2":
        raise ValueError("source-overfit config version is invalid")
    if config.get("data", {}).get("source_split") != "source_train" or \
            config["data"].get("reader_expected_records") != 2436 or config["data"].get("selected_examples") != 3:
        raise ValueError("registered source-training scope is invalid")
    dependencies = config["dependencies"]
    cohort_path = root.joinpath(*PurePosixPath(dependencies["cohort_path"]).parts)
    cohort_raw = _read(cohort_path, "source-overfit cohort")
    if _sha(cohort_raw) != COHORT_SHA256 or dependencies["cohort_sha256"] != COHORT_SHA256:
        raise ValueError("source-overfit cohort bytes differ from the closed registration")
    cohort = _json_object(cohort_raw, "source-overfit cohort")
    examples = cohort.get("examples")
    if (cohort.get("schema_version") != "dwr-source-overfit-cohort-v1" or type(examples) is not list or
            len(examples) != 3 or len({x.get("example_id") for x in examples}) != 3):
        raise ValueError("source-overfit cohort must contain three unique examples")
    if tuple(x.get("profile") for x in examples) != ("eq", "neq", "add_mod") or \
            any(x.get("split") != "source_train" for x in examples):
        raise ValueError("source-overfit cohort profile order or split is invalid")
    from dwr.experiments import gpu_feasibility, frozen_reference
    feasibility_path = root.joinpath(*PurePosixPath(dependencies["feasibility_config_path"]).parts)
    feasibility_raw = _read(feasibility_path, "feasibility config")
    helper_path = Path(gpu_feasibility.__file__).resolve(strict=True)
    if (_sha(feasibility_raw) != dependencies["feasibility_config_sha256"] or
            _sha(_read(helper_path, "feasibility helper")) != FEASIBILITY_MODULE_SHA256 or
            dependencies["feasibility_module_sha256"] != FEASIBILITY_MODULE_SHA256 or
            _sha(_read(Path(frozen_reference.__file__).resolve(strict=True), "frozen reference")) != REFERENCE_MODULE_SHA256 or
            dependencies["reference_module_sha256"] != REFERENCE_MODULE_SHA256):
        raise ValueError("feasibility config bytes differ from the registration")
    feasibility = gpu_feasibility.validate_config_bytes(feasibility_raw)
    reference_path = Path(frozen_reference.__file__).resolve(strict=True)
    if _sha(_read(reference_path, "frozen-reference module")) != dependencies["reference_module_sha256"]:
        raise ValueError("frozen-reference module bytes differ from the registered pin")
    return {"config": config, "config_raw": config_raw, "config_path": expected,
            "config_sha256": CONFIG_SHA256, "cohort": cohort, "cohort_raw": cohort_raw,
            "cohort_sha256": COHORT_SHA256, "selected": tuple(examples),
            "feasibility_config": feasibility}


def _canonical_sha(value: object) -> str:
    return _sha(stable_json(value).encode("utf-8"))


def _kernel_identities(config_raw: bytes, config: Mapping[str, Any], *, source_commit: str,
                       weight_sha256: str, tokenizer_inventory_sha256: str) -> dict[str, str]:
    if not isinstance(source_commit, str) or not source_commit or not source_commit.isascii():
        raise ValueError("source commit must be nonempty ASCII")
    for name, value in (("weight_sha256", weight_sha256),
                        ("tokenizer_inventory_sha256", tokenizer_inventory_sha256)):
        if not isinstance(value, str) or _HASH.fullmatch(value) is None:
            raise ValueError(f"{name} must be lowercase SHA-256")
    data = config["data"]
    dependencies = config["dependencies"]
    data_identity = {
        "source_export_manifest_sha256": data["source_export_manifest_sha256"],
        "source_export_data_sha256": data["source_export_data_sha256"],
        "source_token_audit_manifest_sha256": data["source_token_audit_manifest_sha256"],
        "token_lengths_sha256": data["token_lengths_sha256"],
        "cohort_sha256": dependencies["cohort_sha256"],
        "feasibility_manifest_sha256": dependencies["feasibility_manifest_sha256"],
        "feasibility_outcome_sha256": dependencies["feasibility_outcome_sha256"],
        "reference_manifest_sha256": dependencies["reference_manifest_sha256"],
        "reference_summary_sha256": dependencies["reference_summary_sha256"],
    }
    if tuple(data_identity) != _DATA_IDENTITY_FIELDS:
        raise ValueError("data identity field order differs from the registered recipe")
    schedule = config["training"]["schedule"]
    identities = {
        "source": _sha(source_commit.encode("ascii")),
        "config": _sha(config_raw),
        "data": _canonical_sha(data_identity),
        "model": weight_sha256,
        "tokenizer": tokenizer_inventory_sha256,
        "schedule": _canonical_sha(schedule),
    }
    if any(_HASH.fullmatch(value) is None for value in identities.values()):
        raise ValueError("resolved kernel identities are malformed")
    return identities


def _validate_release_source(source: Mapping[str, Any], registered_base: str) -> dict[str, Any]:
    """Accept the clean release HEAD returned by the ancestry-checking source helper."""
    if type(source) is not dict or source.get("clean") is not True:
        raise ValueError("source must be a clean validated release")
    commit = source.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        raise ValueError("source release commit is malformed")
    if not isinstance(registered_base, str) or re.fullmatch(r"[0-9a-f]{40,64}", registered_base) is None:
        raise ValueError("registered source ancestry base is malformed")
    if type(source.get("files")) is not list:
        raise ValueError("source physical-file receipt is missing")
    return dict(source)


def _join_token_audit(reader_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
                      token_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...], *,
                      expected_count: int, max_answer_tokens: int) -> dict[str, dict[str, Any]]:
    """Validate exact metadata membership while keeping prompt lengths out of reader rows."""
    tokens: dict[str, dict[str, Any]] = {}
    for row in token_rows:
        if type(row) is not dict:
            raise ValueError("token audit rows must be objects")
        eid = row.get("example_id")
        if not isinstance(eid, str) or _HASH.fullmatch(eid) is None or eid in tokens:
            raise ValueError("token audit IDs must be unique SHA-256 values")
        if (type(row.get("prompt_tokens")) is not int or row["prompt_tokens"] < 1 or
                type(row.get("answer_tokens_including_eos")) is not int or
                row["answer_tokens_including_eos"] < 1 or
                row["answer_tokens_including_eos"] > max_answer_tokens or
                row.get("total_tokens") != row["prompt_tokens"] + row["answer_tokens_including_eos"] or
                row.get("within_caps") is not True):
            raise ValueError("token audit lengths do not satisfy registered sequence caps")
        tokens[eid] = row
    if len(reader_rows) != expected_count or len(tokens) != expected_count:
        raise ValueError("reader and token audit membership counts differ from registration")
    reader_ids = [row.get("example_id") for row in reader_rows]
    if len(set(reader_ids)) != len(reader_ids) or set(reader_ids) != set(tokens):
        raise ValueError("reader and token audit do not bind exact source membership")
    for row in reader_rows:
        token = tokens[row["example_id"]]
        if any(row.get(field) != token.get(field) for field in ("split", "profile", "fine_stratum")):
            raise ValueError("token audit metadata differs from source reader")
    return tokens


def _resolved_host_config(feasibility: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(feasibility)
    seeds = config["seeds"]
    resolved_seeds = resolved["seeds"]
    for registered, inherited in (("workspace_initialization", "workspace_initialization"),
                                  ("reader_initialization", "reader_initialization"),
                                  ("python_global", "python_global"), ("numpy_global", "numpy_global"),
                                  ("torch_cpu_global", "torch_cpu_global"),
                                  ("torch_cuda_global", "torch_cuda_global")):
        resolved_seeds[inherited] = seeds[registered]
    return resolved


def _install_global_rngs(seeds: Mapping[str, int]) -> None:
    import numpy as np
    import torch
    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"])
    torch.manual_seed(seeds["torch_cpu"])
    torch.cuda.manual_seed_all(seeds["torch_cuda"])


def _kernel_resume_pointer(pointer: Mapping[str, Any]) -> dict[str, str]:
    if type(pointer) is not dict or not {"jobroot", "cursor_path", "cursor_sha256"}.issubset(pointer):
        raise ValueError("verified resume pointer has an unexpected schema")
    result = {key: pointer[key] for key in ("jobroot", "cursor_path", "cursor_sha256")}
    if any(not isinstance(result[key], str) or not result[key] for key in result):
        raise ValueError("verified resume pointer fields must be nonempty strings")
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _write_exclusive_json(path: Path, value: object) -> str:
    raw = _canonical_json(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return _sha(raw)


def _parameter_counts(host: Any) -> dict[str, int]:
    original = sum(parameter.numel() for parameter in host.backbone.parameters())
    new = sum(parameter.numel() for module in (host.workspace, host.readers)
              for parameter in module.parameters())
    return {"original_parameter_elements": original, "new_parameter_elements": new,
            "total_parameter_elements": original + new}


def _check_physical_job_caps(jobroot: Path, config: Mapping[str, Any], next_kind: str) -> None:
    started_updates = completed_updates = started_predictions = completed_predictions = decoder_calls = 0
    from dwr.experiments.records import _validate_events
    for segment in jobroot.glob("segment-[0-9][0-9][0-9]"):
        path = segment / "events.jsonl"
        if not path.is_file():
            continue
        raw = path.read_bytes(); _validate_events(raw)
        for line in raw.splitlines():
            event = json.loads(line.decode("utf-8")); kind = event["kind"]
            if kind == "update_started": started_updates += 1
            elif kind == "update": completed_updates += 1
            elif kind == "prediction_started": started_predictions += 1
            elif kind == "prediction_completed":
                completed_predictions += 1
                decoder_calls += int(event["payload"].get("decoder_calls", 0))
    physical_updates = max(started_updates, completed_updates)
    prediction_count = started_predictions + int(next_kind == "prediction_started")
    physical_updates += int(next_kind == "update_started")
    outstanding = max(0, prediction_count - completed_predictions)
    worst_decoder_calls = decoder_calls + outstanding * config["decoding"]["max_new_tokens_including_eos"]
    if (physical_updates > config["limits"]["maximum_physical_updates"] or
            prediction_count > config["limits"]["maximum_predictions"] or
            worst_decoder_calls > config["limits"]["maximum_decoder_calls"]):
        raise RuntimeError("whole-job physical update/prediction/decoder cap would be exceeded")


def _record_identities(identities: Mapping[str, str], *, config_version: str,
                       model_repo: str, model_revision: str, schedule_type: str) -> dict[str, dict[str, str]]:
    labels = {
        "source": ("dwr-source-commit", "HEAD"),
        "config": ("source-overfit-registration", config_version),
        "data": ("source-overfit-data-and-proofs", "v1"),
        "model": (model_repo, model_revision),
        "tokenizer": ("pinned-local-original-tokenizer", model_revision),
        "schedule": ("optimizer-schedule", schedule_type),
    }
    if set(identities) != set(labels):
        raise ValueError("kernel identities must contain the exact six registered names")
    return {key: {"name": name, "revision": revision, "sha256": identities[key]}
            for key, (name, revision) in labels.items()}


def _select_training_rows(rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
                          selected: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    if not isinstance(rows, (tuple, list)) or not isinstance(selected, (tuple, list)) or len(selected) != 3:
        raise ValueError("source rows and three-row registration are required")
    expected = {entry["example_id"]: entry for entry in selected}
    if len(expected) != 3:
        raise ValueError("registered source IDs must be unique")
    found: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for row in rows:
        if type(row) is not dict or row.get("split") != "source_train":
            raise ValueError("reader returned a non-source-train row")
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or _HASH.fullmatch(example_id) is None or example_id in seen:
            raise ValueError("reader returned a malformed or duplicate source ID")
        seen.add(example_id)
        if example_id not in expected:
            continue
        registered = expected[example_id]
        for field in ("profile", "fine_stratum"):
            if row.get(field) != registered[field]:
                raise ValueError(f"source row {field} differs from registered metadata")
        found[example_id] = row
    if set(found) != set(expected):
        raise ValueError("source reader did not return every selected ID exactly once")
    return tuple(found[entry["example_id"]] for entry in selected)


def _source_examples(rows: tuple[dict[str, Any], ...]) -> tuple[Any, ...]:
    from dwr.tasks.schema import Solution
    from dwr.tasks.run_support import stable_json
    from dwr.tasks.verifier import verify_answer
    from dwr.training.batches import SourceExample

    output = []
    for row in rows:
        task = schema.parse_task_json(row["task"])
        assignment = row["solution"].get("assignment") if type(row.get("solution")) is dict else None
        if type(assignment) is not dict or set(assignment) != set(task.variables) or \
                any(type(value) is not int for value in assignment.values()):
            raise ValueError("source solution assignment is malformed")
        answer = stable_json({"assignment": assignment})
        if not verify_answer(task, answer).valid:
            raise ValueError("source solution failed the strict verifier")
        solution = Solution(tuple((name, assignment[name]) for name in task.variables))
        output.append(SourceExample(row["example_id"], "source_train", task, solution))
    if tuple(row["profile"] for row in rows) != ("eq", "neq", "add_mod"):
        raise ValueError("selected source rows are not ordered EQ, NEQ, ADD_MOD")
    return tuple(output)


def run_source_overfit(config_path: str | os.PathLike[str], *, command: list[str],
                       segment_index: int, resume_descriptor: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one supervisor-authorized immutable source-training segment."""
    if type(command) is not list or not command or any(type(x) is not str or not x for x in command):
        raise TypeError("command must be a nonempty list of strings")
    if type(segment_index) is not int or segment_index not in (0, 1, 2):
        raise ValueError("segment_index must be 0, 1 or 2")
    root = _repo_root()
    registration = _load_registration(root, config_path)
    from dwr.experiments.source_overfit_job import verify_worker_authorization
    authorization = verify_worker_authorization(root, registration, segment_index, resume_descriptor)
    try:
        return _execute_registered(root, registration, authorization, command=command)
    finally:
        authorization["worker_lock"].close()


def _execute_registered(root: Path, registration: Mapping[str, Any], authorization: Mapping[str, Any],
                        *, command: list[str], _test_seams: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run a segment; `_test_seams` is private and never exposed by the CLI."""
    from dwr.experiments import gpu_feasibility, frozen_reference
    from dwr.training.overfit import (NumericalDivergence, run_overfit,
                                      verify_evaluation_receipt)
    from dwr.training.recovery import RecoveryStore
    from dwr.training.batches import TokenizerIdentity

    seam = dict(_test_seams or {})
    config = registration["config"]
    root = Path(root).resolve(strict=True)
    jobroot = Path(authorization["jobroot"]).resolve(strict=True)
    segment_index = authorization["segment_index"]
    segment = jobroot / f"segment-{segment_index:03d}"
    if segment.exists():
        raise ValueError("segment directory already exists; segments are immutable")
    started = time.monotonic()
    source = seam["source"]() if "source" in seam else gpu_feasibility.validate_clean_source(
        root, config["source_base_commit"])
    if _test_seams is None:
        source = _validate_release_source(source, config["source_base_commit"])
    elif source.get("clean") is not True:
        raise ValueError("test source identity must be clean")
    if _test_seams is None and source != authorization.get("physical_source_receipt"):
        raise ValueError("worker source files differ from the supervisor-bound physical receipt")
    helper_config = registration["feasibility_config"]
    feasibility = seam["feasibility"]() if "feasibility" in seam else frozen_reference._validate_feasibility(
        root, config, gpu_feasibility)
    if feasibility.get("outcome", {}).get("status") != "PASS_FEASIBILITY":
        raise ValueError("registered frozen feasibility receipt did not pass")
    dependency = config["dependencies"]
    if "reference" in seam:
        reference_terminal, reference_summary_raw, reference_summary = seam["reference"]()
    else:
        reference_dir = root.joinpath(*PurePosixPath(dependency["reference_run_dir"]).parts)
        reference_terminal, reference_summary_raw, reference_summary = _verified_reference_proofs(
            reference_dir, dependency)
    if reference_summary.get("status") != "PASS_FROZEN_REFERENCE_OPERATIONAL_V1":
        raise ValueError("pinned reference summary is not the registered operational PASS")
    if (reference_summary.get("attempt_count") != 45 or
            reference_summary.get("quality", {}).get("denominator") != 15 or
            reference_summary.get("repetitions") != {"0": {"examples": 15, "valid_count": 0},
                "1": {"examples": 15, "valid_count": 0}, "2": {"examples": 15, "valid_count": 0}}):
        raise ValueError("pinned frozen-reference summary counts differ from the registered 45/15 result")
    data_receipt = seam["data_preconditions"]() if "data_preconditions" in seam else gpu_feasibility._data_preconditions(
        root, helper_config)
    inventory = seam["artifact_inventory"]() if "artifact_inventory" in seam else gpu_feasibility._artifact_inventory(
        root, helper_config)
    if inventory.get("weight_sha256") != helper_config["model"]["weight_sha256"]:
        raise ValueError("local model weight identity differs from the pinned artifact inventory")

    model = helper_config["model"]
    tokenizer_identity = TokenizerIdentity(model["repo_id"], model["revision"])
    identities = _kernel_identities(registration["config_raw"], config,
        source_commit=source["commit"], weight_sha256=inventory["weight_sha256"],
        tokenizer_inventory_sha256=inventory["inventory_sha256"])
    record_identities = _record_identities(identities, config_version=config["config_version"],
        model_repo=model["repo_id"], model_revision=model["revision"],
        schedule_type=config["training"]["schedule"]["type"])
    seeds = config["seeds"]
    rng_streams = {key: value for key, value in seeds.items()}
    budgets = {"wall_seconds": config["limits"]["wall_seconds"],
               "memory_mib": config["limits"]["resident_memory_mib"],
               "attempts": config["limits"]["attempts"]}
    settings = {"command": command, "config_sha256": registration["config_sha256"],
        "source_commit": source["commit"], "physical_source_files": source.get("files", []),
        "environment": seam["environment"] if "environment" in seam else _environment_receipt(),
        "model_inventory": inventory, "data_preconditions": data_receipt,
        "reference_manifest_sha256": dependency["reference_manifest_sha256"],
        "reference_summary_sha256": _sha(reference_summary_raw),
        "segment_index": segment_index, "worker_pid": os.getpid(),
        "worker_creation_time": authorization.get("worker_creation_time"),
        "supervisor_pid": authorization.get("supervisor_pid"),
        "supervisor_creation_time": authorization.get("supervisor_creation_time"),
        "tokenizer_identity": {"name": tokenizer_identity.name, "revision": tokenizer_identity.revision}}
    record = RunRecord.create(segment, command=command, config_path=registration["config_path"],
        expected_config_sha256=registration["config_sha256"], identities=record_identities,
        rng_streams=rng_streams, settings=settings, budgets=budgets, confirmatory=False, source_root=root)
    record.append_event("source_access_authorized", {"source_split": "source_train",
        "task_semantics_accessed": False, "segment_index": segment_index})
    try:
        lifecycle = {"stage": "work"}
        heartbeat_stop = threading.Event()
        heartbeat_errors = []
        heartbeat_thread = None
        if _test_seams is None and authorization.get("worker_heartbeat_path"):
            def heartbeat_loop():
                while not heartbeat_stop.wait(10.0):
                    try:
                        _write_worker_heartbeat(authorization, interval=0, stage=lifecycle["stage"])
                    except Exception as error:
                        heartbeat_errors.append(error)
                        return
            heartbeat_thread = threading.Thread(target=heartbeat_loop,
                name="source-overfit-heartbeat", daemon=True)
            heartbeat_thread.start()
        def check():
            if "check" in seam:
                return seam["check"]()
            if heartbeat_errors:
                raise RuntimeError("worker heartbeat writer failed") from heartbeat_errors[0]
            _write_worker_heartbeat(authorization, interval=5 if lifecycle["stage"] == "seal" else
                                    config["recovery"]["heartbeat_seconds"], stage=lifecycle["stage"])
            if lifecycle["stage"] == "work":
                return _check_worker_budget(authorization, started, config)
            return _check_worker_seal_budget(authorization, config)
        if "host" not in seam:
            environment = gpu_feasibility._check_environment(helper_config)
            hardware = gpu_feasibility._hardware_metadata(helper_config)
            numerical_policy = gpu_feasibility._set_numerics(helper_config)
            record.append_event("runtime_hardware_preflight", {"environment": environment,
                "hardware": hardware, "numerical_policy": numerical_policy})
        cohort = registration["selected"]
        if "rows" in seam:
            all_rows = tuple(seam["rows"])
            if "token_rows" in seam:
                token_by_id = _join_token_audit(all_rows, tuple(seam["token_rows"]),
                    expected_count=len(all_rows),
                    max_answer_tokens=config["decoding"]["max_new_tokens_including_eos"])
                for row in all_rows:
                    row["_registered_prompt_tokens"] = token_by_id[row["example_id"]]["prompt_tokens"]
        else:
            audit_path = root.joinpath(*PurePosixPath(config["data"]["source_token_audit_dir"]).parts,
                                       "token_lengths.jsonl")
            audit_raw = _read(audit_path, "token-length audit")
            if _sha(audit_raw) != config["data"]["token_lengths_sha256"]:
                raise ValueError("token-length audit bytes differ from the registered pin")
            token_rows = [_json_object(line.encode("utf-8"), "token-length row")
                          for line in audit_raw.decode("utf-8", errors="strict").splitlines() if line]
            source_token_rows = [row for row in token_rows if row.get("split") == "source_train"]
            reader = __import__("dwr.experiments.source_reader", fromlist=["read_source_export"]).read_source_export
            source_rows = reader(root.joinpath(*PurePosixPath(config["data"]["source_export_dir"]).parts),
                expected_manifest_sha256=config["data"]["source_export_manifest_sha256"], split="source_train")
            token_by_id = _join_token_audit(source_rows, source_token_rows,
                expected_count=config["data"]["reader_expected_records"],
                max_answer_tokens=config["decoding"]["max_new_tokens_including_eos"])
            selected_rows = _select_training_rows(source_rows, cohort)
            for row in selected_rows:
                token_row = token_by_id[row["example_id"]]
                row["_registered_prompt_tokens"] = token_row["prompt_tokens"]
            all_rows = selected_rows
        rows = _select_training_rows(all_rows, cohort)
        examples = _source_examples(rows)
        metadata = {row["example_id"]: {key: row[key] for key in
                    ("profile", "lineage_id", "structure_signature", "fine_stratum")} for row in rows}
        if "tokenizer" in seam:
            tokenizer = seam["tokenizer"]
        else:
            from transformers import AutoTokenizer
            tokenizer_dir = root.joinpath(*PurePosixPath(model["local_dir"]).parts)
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True,
                trust_remote_code=False)
        if getattr(tokenizer, "eos_token_id", None) != config["decoding"]["eos_token_id"]:
            raise ValueError("loaded tokenizer EOS differs from registration")
        framed_rows = []
        from dwr.tasks import serialize_model_input
        for row in rows:
            prompt = serialize_model_input(schema.parse_task_json(row["task"])) + "\n"
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            expected_prompt_tokens = row.get("_registered_prompt_tokens", row.get("prompt_tokens"))
            if not token_ids or len(token_ids) != expected_prompt_tokens:
                raise ValueError("selected source prompt token count differs from the frozen cohort")
            if len(token_ids) > config["decoding"]["max_prompt_tokens"]:
                raise ValueError("source prompt exceeds the fixed token cap")
            framed_rows.append({"example_id": row["example_id"], "prompt_sha256": _sha(prompt.encode("utf-8")),
                                "prompt_tokens": len(token_ids)})
        record.append_event("selected_source_rows", {"rows": framed_rows,
            "selected_ids": [row["example_id"] for row in rows], "reader_count": len(all_rows)})
        if "host" in seam:
            host = seam["host"]
        else:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            resolved_host_config = _resolved_host_config(helper_config, config)
            _install_global_rngs({"python": seeds["python_global"], "numpy": seeds["numpy_global"],
                "torch_cpu": seeds["torch_cpu_global"], "torch_cuda": seeds["torch_cuda_global"]})
            host = gpu_feasibility._load_production_host(root, resolved_host_config)
        original_hashes = seam["original_hashes"] if "original_hashes" in seam else gpu_feasibility.hash_original_parameters(host)
        parameter_counts = _parameter_counts(host)
        initial_hash_receipt = {"schema_version": 1, "source_commit": source["commit"],
            "config_sha256": registration["config_sha256"], "model_sha256": identities["model"],
            "original_parameter_hashes": original_hashes, "parameter_counts": parameter_counts}
        initial_hash_path = jobroot / f"initial-frozen-parameter-audit-{segment_index:03d}.json"
        _write_exclusive_json(initial_hash_path, initial_hash_receipt)
        initial_hash_ref = {"path": initial_hash_path.relative_to(jobroot).as_posix(),
                            "sha256": _sha(_read(initial_hash_path, "initial frozen-parameter audit"))}
        actual_config = seam.get("training_config", config)
        if _test_seams is not None:
            actual_config = seam.get("training_config", config)
        def kernel_event(kind, payload):
            if kind == "evaluation" and payload.get("step") == actual_config["training"]["optimizer_steps"]:
                check()
                after = seam["original_hashes_after"] if "original_hashes_after" in seam else gpu_feasibility.hash_original_parameters(host)
                if after != original_hashes:
                    raise ValueError("frozen original-model parameters changed before final evaluation")
                costs = (seam.get("runtime_costs", {"process_peak_working_set_mib": _process_resident_mib(),
                    "peak_allocated_mib": None, "peak_reserved_mib": None}) if "host" in seam else
                    _measured_runtime_costs(host, config))
                audit = {"schema_version": 1, "source_commit": source["commit"],
                    "config_sha256": registration["config_sha256"], "model_sha256": identities["model"],
                    "initial_state_sha256": payload.get("initial_state_sha256"),
                    "initial_hash_receipt": initial_hash_ref,
                    "final_checkpoint_sha256": payload.get("checkpoint_sha256"),
                    "original_parameters_unchanged": True, "original_parameter_hashes_after": after,
                    "parameter_counts": parameter_counts,
                    "runtime_costs": costs, "evaluation_state_preservation": payload.get("after")}
                raw_audit = _canonical_json(audit)
                audit_path = segment / "controller-audit.json"
                with audit_path.open("xb") as stream:
                    stream.write(raw_audit + b"\n"); stream.flush(); os.fsync(stream.fileno())
                audit_ref = {"path": audit_path.relative_to(jobroot).as_posix(), "sha256": _sha(raw_audit + b"\n")}
                record.append_event("controller_audit_receipt", audit_ref)
                seam["controller_audit_ref"] = audit_ref
                seam["controller_audit_costs"] = costs
            if "physical_budget_check" in seam:
                seam["physical_budget_check"](kind, payload)
            _record_kernel_event(record, seam, kind, payload)

        if _test_seams is None:
            seam["physical_budget_check"] = lambda kind, payload: _check_physical_job_caps(jobroot, config, kind)

        kernel_result = run_overfit(host, tokenizer, tokenizer_identity, examples, config=actual_config,
            identities=identities, metadata=metadata, output_dir=segment, check=check,
            on_event=kernel_event,
            resume=_kernel_resume_pointer(authorization["resume_descriptor"]) if authorization.get("resume_descriptor") else None)
        lifecycle["stage"] = "seal"
        _write_worker_heartbeat(authorization, interval=0, stage="seal")
        check()
        if kernel_result.get("net_optimizer_steps") != actual_config["training"]["optimizer_steps"]:
            raise ValueError("kernel returned a result with an unexpected net-step count")
        if "original_hashes_after" in seam:
            after_hashes = seam["original_hashes_after"]
        else:
            after_hashes = gpu_feasibility.hash_original_parameters(host)
        if original_hashes != after_hashes:
            raise ValueError("frozen original-model parameters changed during source overfit")
        runtime_costs = seam.get("controller_audit_costs") if "host" not in seam else {
            "process_peak_working_set_mib": None, "peak_allocated_mib": None, "peak_reserved_mib": None}
        cursor = RecoveryStore.verify_descriptor(kernel_result["recovery_descriptor"], identities)
        if cursor.get("phase") != "complete" or cursor.get("progress", {}).get("optimizer_step") != actual_config["training"]["optimizer_steps"]:
            raise ValueError("kernel cursor is not complete at the registered final step")
        facts = cursor.get("facts")
        if type(facts) is not dict or type(facts.get("final_result")) is not dict:
            raise ValueError("complete cursor lacks the final result and receipt references")
        stored = facts["final_result"]
        initial_ref = facts.get("initial_evaluation_receipt")
        final_ref = stored.get("final_evaluation_receipt")
        initial_verified = verify_evaluation_receipt(jobroot, initial_ref, expected_step=0,
            expected_example_ids=[ex.example_id for ex in examples],
            expected_checkpoint_sha256=facts.get("initial_state_sha256"))
        final_verified = verify_evaluation_receipt(jobroot, final_ref,
            expected_step=actual_config["training"]["optimizer_steps"],
            expected_example_ids=[ex.example_id for ex in examples],
            expected_checkpoint_sha256=cursor["checkpoint_sha256"])
        if stored.get("initial_evaluation_receipt") != initial_ref or \
                stored.get("initial_state_sha256") != facts.get("initial_state_sha256") or \
                stored.get("predictions") != initial_verified["rows"] + final_verified["rows"]:
            raise ValueError("complete cursor result does not bind its verified evaluation receipts")
        ratio = final_verified["ce"] / initial_verified["ce"]
        if not math.isclose(stored.get("initial_ce", math.nan), initial_verified["ce"], rel_tol=1e-12, abs_tol=1e-12) or \
                not math.isclose(stored.get("final_ce", math.nan), final_verified["ce"], rel_tol=1e-12, abs_tol=1e-12) or \
                not math.isclose(stored.get("final_to_initial_ce_ratio", math.nan), ratio, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("complete cursor metrics do not recompute from verified receipts")
        final_valid = final_verified["valid"]
        expected_pass = final_valid == 3 and ratio <= actual_config["evaluation"]["gate_max_final_to_initial_CE_ratio"]
        expected_status = "PASS_TINY_SOURCE_OVERFIT" if expected_pass else "NO_GO_TINY_SOURCE_OVERFIT"
        if stored.get("status") != expected_status or kernel_result.get("status") != expected_status:
            raise ValueError("kernel terminal result does not match independently recomputed source-overfit gate")
        outcome_path = segment / "outcome.json"
        outcome_bytes = json.dumps({"schema_version": "dwr-source-overfit-worker-outcome-v1",
            "status": expected_status, "config_sha256": registration["config_sha256"],
            "source_commit": source["commit"], "example_ids": [ex.example_id for ex in examples],
            "initial_ce": initial_verified["ce"], "final_ce": final_verified["ce"],
            "final_to_initial_ce_ratio": ratio, "valid_numerator": final_valid,
            "valid_denominator": 3, "final_checkpoint_sha256": cursor["checkpoint_sha256"],
            "initial_state_sha256": facts["initial_state_sha256"],
            "initial_evaluation_receipt": initial_ref, "final_evaluation_receipt": final_ref,
            "net_optimizer_steps": kernel_result["net_optimizer_steps"],
            "physical_optimizer_updates": kernel_result["physical_optimizer_updates"]},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        with outcome_path.open("xb") as stream:
            stream.write(outcome_bytes); stream.flush(); os.fsync(stream.fileno())
        check()
        artifacts = _segment_artifacts(segment)
        counts = {"physical_optimizer_updates": kernel_result["physical_optimizer_updates"],
                  "net_optimizer_steps": kernel_result["net_optimizer_steps"],
                  "examples_seen": kernel_result["examples_seen"],
                  "supervised_tokens_seen": kernel_result["supervised_tokens_seen"],
                  "valid_numerator": kernel_result["valid_numerator"], "valid_denominator": 3,
                  "pass": int(expected_status == "PASS_TINY_SOURCE_OVERFIT")}
        if not seam.get("controller_audit_ref"):
            raise ValueError("final kernel evaluation did not durably publish controller audit evidence")
        record.append_event("controller_audit", {"status": expected_status, "original_parameters_unchanged": True,
            "initial_receipt_verified": True, "final_receipt_verified": True,
            "audit_receipt": seam["controller_audit_ref"], "initial_hash_receipt": initial_hash_ref,
            "recovery_phase": cursor["phase"], "artifacts": artifacts})
        kernel_result["costs"].update(runtime_costs)
        record.append_event("measured_costs", runtime_costs)
        record.finish("completed", counts=counts, artifacts=artifacts,
                      budget_check=lambda stage: _check_worker_seal_budget(authorization, config))
        return {**kernel_result, "record_dir": str(segment), "record_status": "completed",
                "status": expected_status, "segment_index": segment_index}
    except NumericalDivergence as error:
        lifecycle["stage"] = "seal"; _write_worker_heartbeat(authorization, interval=0, stage="seal")
        partial = dict(error.partial)
        _finish_failure(record, segment, "failed", "NUMERICAL_DIVERGENCE", partial,
                        budget_check=lambda stage: _check_worker_seal_budget(authorization, config))
        raise
    except KeyboardInterrupt:
        lifecycle["stage"] = "seal"; _write_worker_heartbeat(authorization, interval=0, stage="seal")
        _finish_failure(record, segment, "interrupted", "worker received KeyboardInterrupt",
                        {"segment_index": segment_index, "last_good_phase": "controller_or_kernel"},
                        budget_check=lambda stage: _check_worker_seal_budget(authorization, config))
        raise
    except Exception as error:
        lifecycle["stage"] = "seal"; _write_worker_heartbeat(authorization, interval=0, stage="seal")
        _finish_failure(record, segment, "failed", f"{type(error).__name__}: {error}",
                        {"segment_index": segment_index, "last_good_phase": "controller_or_kernel"},
                        budget_check=lambda stage: _check_worker_seal_budget(authorization, config))
        raise
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2.0)


def _environment_receipt() -> dict[str, Any]:
    import importlib.metadata
    import platform
    import sys
    packages = sorted(({"name": dist.metadata.get("Name", dist.name), "version": dist.version}
                       for dist in importlib.metadata.distributions()),
                      key=lambda item: (item["name"].lower(), item["version"]))
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "implementation": sys.implementation.name, "packages": packages}


def _check_worker_budget(authorization: Mapping[str, Any], started: float, config: Mapping[str, Any]) -> None:
    deadline = authorization.get("work_deadline_monotonic")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("source-overfit work cutoff reached")
    deadline_epoch = authorization.get("work_deadline_utc_epoch")
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise TimeoutError("source-overfit registered work cutoff reached")
    if time.monotonic() - started >= config["limits"]["work_seconds"]:
        raise TimeoutError("source-overfit segment work cutoff reached")
    resident = _process_resident_mib()
    if resident is not None and resident > config["limits"]["resident_memory_mib"]:
        raise MemoryError("source-overfit process resident memory cap exceeded")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized() and torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / (1024**2)
        reserved = torch.cuda.max_memory_reserved() / (1024**2)
        if allocated > config["limits"]["maximum_peak_allocated_mib"] or reserved > config["limits"]["maximum_peak_reserved_mib"]:
            raise MemoryError("source-overfit CUDA peak memory cap exceeded")
    _write_worker_heartbeat(authorization, interval=config["recovery"]["heartbeat_seconds"])


def _check_worker_seal_budget(authorization: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    deadline_epoch = authorization.get("seal_deadline_utc_epoch")
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise TimeoutError("source-overfit total seal cutoff reached")
    resident = _process_resident_mib()
    if resident > config["limits"]["resident_memory_mib"]:
        raise MemoryError("source-overfit process peak working-set cap exceeded")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized() and torch.cuda.is_available():
        if (torch.cuda.max_memory_allocated() / 1024**2 > config["limits"]["maximum_peak_allocated_mib"] or
                torch.cuda.max_memory_reserved() / 1024**2 > config["limits"]["maximum_peak_reserved_mib"]):
            raise MemoryError("source-overfit CUDA peak memory cap exceeded during sealing")
    heartbeat_path = authorization.get("worker_heartbeat_path")
    if heartbeat_path:
        _write_worker_heartbeat(authorization, interval=30)


def _write_worker_heartbeat(authorization: Mapping[str, Any], interval: int, *, stage: str | None = None) -> None:
    heartbeat_path = authorization.get("worker_heartbeat_path")
    if not heartbeat_path:
        return
    with _HEARTBEAT_WRITE_LOCK:
        marker = authorization.get("_last_heartbeat_monotonic", 0.0)
        now = time.monotonic()
        if now - marker >= interval:
            if stage is not None:
                try:
                    authorization["worker_stage"] = stage
                except TypeError:
                    pass
            _atomic_write_json(Path(heartbeat_path), {"pid": os.getpid(),
                "creation_time": authorization.get("worker_creation_time"), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "utc_epoch": time.time(), "monotonic": now,
                "descriptor_nonce": authorization.get("worker_receipt", {}).get("descriptor_nonce"),
                "stage": authorization.get("worker_stage", "work")})
            try:
                authorization["_last_heartbeat_monotonic"] = now
            except TypeError:
                pass


def _segment_artifacts(segment: Path) -> list[str]:
    return sorted(path.relative_to(segment).as_posix() for path in segment.rglob("*")
                  if path.is_file() and path.name not in {"manifest.initial.json", "manifest.json", "events.jsonl"})


def _record_kernel_event(record: RunRecord, seams: Mapping[str, Any], kind: str, payload: Mapping[str, Any]) -> None:
    record.append_event(kind, payload)
    observer = seams.get("observe_kernel_event")
    if observer is not None:
        observer(kind, payload)


def _atomic_write_json(path: Path, value: object) -> None:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _process_resident_mib() -> float | None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters(); counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed; memory cap is unverifiable")
        return counters.PeakWorkingSetSize / (1024**2)
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024**2) if sys.platform == "darwin" else value / 1024
    except (ImportError, OSError) as exc:
        raise OSError("process high-water memory measurement is required") from exc


def _measured_runtime_costs(host: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    torch = sys.modules.get("torch")
    allocated = reserved = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated() / (1024**2)
        reserved = torch.cuda.max_memory_reserved() / (1024**2)
    resident = _process_resident_mib()
    if resident > config["limits"]["resident_memory_mib"]:
        raise MemoryError("source-overfit final resident-memory cap exceeded")
    if allocated is not None and (allocated > config["limits"]["maximum_peak_allocated_mib"] or
                                  reserved > config["limits"]["maximum_peak_reserved_mib"]):
        raise MemoryError("source-overfit final CUDA peak-memory cap exceeded")
    return {"process_peak_working_set_mib": resident, "peak_allocated_mib": allocated,
            "peak_reserved_mib": reserved}


def _finish_failure(record: RunRecord, segment: Path, status: str, error: str,
                    details: Mapping[str, Any], budget_check: Callable[[str], None] | None = None) -> None:
    if (segment / "manifest.json").exists():
        return
    for name in ("worker-failure.json",):
        path = segment / name
        try:
            raw = (json.dumps({"error": error, "details": details}, sort_keys=True,
                              separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            with path.open("xb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        except FileExistsError:
            pass
    try:
        record.append_event("worker_failure", {"error": error, "details": dict(details)})
        record.finish(status, counts={"failed": 1}, artifacts=_segment_artifacts(segment), error=error,
                       budget_check=budget_check)
    except Exception:
        # Preserve the initial manifest, event stream and all partial artifacts if sealing fails.
        pass


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--descriptor-sha256", required=True)
    args = parser.parse_args(argv)
    from dwr.experiments.source_overfit_job import read_worker_descriptor
    descriptor = read_worker_descriptor(args.descriptor, args.descriptor_sha256)
    run_source_overfit(args.config, command=[sys.executable, "-m", __name__, *sys.argv[1:]],
                       segment_index=descriptor["segment_index"], resume_descriptor=descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
