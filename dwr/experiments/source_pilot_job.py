"""Detached supervisor for the registered source-validation pilot."""
from __future__ import annotations

import json
import math
import hashlib
import os
import stat
from pathlib import Path
from datetime import datetime, timezone
from typing import Mapping
import platform
import sys
import subprocess
import uuid
import time
import argparse
import re
from pathlib import PurePosixPath

from dwr.experiments.records import RunRecord, verify_artifacts

_CONFIG_PATH = "configs/experiments/source_validation_pilot_v1.json"
_MATRIX_LOCK_PATH = "runs/source-validation-pilot-20261017-v1.lock"
_ARM_ORDER = ("K0", "K1", "K4")

from dwr.tasks.run_support import strict_json_loads


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _validated_elapsed(last_utc_epoch: float, now_utc_epoch: float, prior_elapsed: float) -> float:
    values = (last_utc_epoch, now_utc_epoch, prior_elapsed)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("clock receipt values must be finite")
    if now_utc_epoch < last_utc_epoch:
        raise ValueError("clock rollback detected against durable elapsed receipt")
    return max(float(prior_elapsed), float(prior_elapsed) + now_utc_epoch - last_utc_epoch)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_links(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for item in reversed((absolute, *absolute.parents)):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
            raise ValueError("pilot paths cannot contain symlinks or reparse points")


def _read_descriptor_file(path: str | os.PathLike[str], expected_sha256: str) -> tuple[Path, dict]:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in expected_sha256):
        raise ValueError("descriptor SHA256 must be lowercase hexadecimal")
    lexical = Path(os.path.abspath(os.fspath(path)))
    _reject_links(lexical)
    resolved = lexical.resolve(strict=True)
    raw = resolved.read_bytes()
    if _sha(raw) != expected_sha256:
        raise ValueError("worker descriptor bytes differ from issued digest")
    try:
        value = strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("worker descriptor is not strict UTF-8 JSON") from exc
    if type(value) is not dict or raw != _canonical(value) + b"\n":
        raise ValueError("worker descriptor is not canonical stable JSON")
    return resolved, value


def read_worker_authorization(root: str | os.PathLike[str], descriptor_path: str | os.PathLike[str],
                              descriptor_sha256: str) -> dict:
    """Verify the immutable worker issue before any task, model, or CUDA access."""
    descriptor_file, descriptor = _read_descriptor_file(descriptor_path, descriptor_sha256)
    keys = {"schema_version", "matrix_root", "arm_root", "segment_dir", "arm_id", "arm_index",
        "segment_index", "nonce", "config_sha256", "cohort_sha256", "source_commit",
        "physical_source_receipt", "physical_source_receipt_sha256", "runtime_receipt",
        "runtime_receipt_sha256", "matrix_initial_manifest_sha256", "arm_initial_manifest_sha256",
        "matrix_event_prefix", "arm_event_prefix", "segment_initial_manifest_sha256",
        "segment_event_prefix", "matrix_clock", "arm_clock", "deadlines", "supervisor",
        "launcher", "resume_descriptor"}
    if set(descriptor) != keys or descriptor.get("schema_version") != \
            "dwr-source-validation-pilot-worker-descriptor-v1":
        raise ValueError("worker descriptor has unknown schema or fields")
    expected_root = Path(__file__).resolve().parents[2]
    source_root = Path(root).resolve(strict=True)
    if source_root != expected_root:
        raise ValueError("worker root differs from the source checkout")
    from dwr.experiments.source_pilot_registration import load_pilot_registration, pilot_identities
    from dwr.experiments import gpu_feasibility
    config_path = source_root / "configs" / "experiments" / "source_validation_pilot_v1.json"
    registration = load_pilot_registration(source_root, config_path)
    config = registration["config"]
    arm_id, arm_index, segment_index = descriptor["arm_id"], descriptor["arm_index"], descriptor["segment_index"]
    arms = config["arms"]
    if type(arm_index) is not int or not (0 <= arm_index < len(arms)) or arms[arm_index]["arm_id"] != arm_id:
        raise ValueError("worker descriptor arm order differs from registration")
    if type(segment_index) is not int or not (0 <= segment_index < config["limits"]["maximum_workers_per_arm"]):
        raise ValueError("worker descriptor segment index exceeds registered recovery allowance")
    if not isinstance(descriptor["nonce"], str) or len(descriptor["nonce"]) != 32 or any(
            char not in "0123456789abcdef" for char in descriptor["nonce"]):
        raise ValueError("worker descriptor nonce is malformed")
    for name, expected in (("config_sha256", registration["config_sha256"]),
                           ("cohort_sha256", registration["cohort_sha256"])):
        if descriptor[name] != expected:
            raise ValueError(f"worker descriptor {name} differs from registration")
    physical = gpu_feasibility.validate_clean_source(source_root, config["source"]["ancestry_minimum"])
    runtime = _runtime_receipt()
    if (descriptor["physical_source_receipt"] != physical or
            descriptor["physical_source_receipt_sha256"] != _sha(_canonical(physical))):
        raise ValueError("worker descriptor physical source receipt differs from this exact source")
    if (descriptor["runtime_receipt"] != runtime or
            descriptor["runtime_receipt_sha256"] != _sha(_canonical(runtime))):
        raise ValueError("worker descriptor runtime receipt differs from this Python environment")
    if descriptor["source_commit"] != physical["commit"]:
        raise ValueError("worker descriptor source commit differs from physical source")
    matrix_root = (source_root / config["output_dir"]).resolve(strict=True)
    arm_root = (matrix_root / "arms" / arm_id).resolve(strict=True)
    segment_dir = (arm_root / f"segment-{segment_index:03d}").resolve(strict=True)
    def require_path(key: str, expected: Path) -> None:
        actual = Path(descriptor[key]).resolve(strict=True)
        _reject_links(actual)
        if actual != expected:
            raise ValueError(f"worker descriptor {key} differs from its registered location")
    require_path("matrix_root", matrix_root)
    require_path("arm_root", arm_root)
    require_path("segment_dir", segment_dir)
    _reject_links(descriptor_file)
    if descriptor_file.parent != segment_dir or descriptor_file.name != "worker-descriptor.json":
        raise ValueError("worker descriptor is outside its issued segment")
    identities = pilot_identities(registration, source_commit=physical["commit"], arm_id=arm_id)
    matrix_initial = _verify_initial_binding(matrix_root, descriptor["matrix_initial_manifest_sha256"],
                            expected_kind="matrix", config_sha256=registration["config_sha256"],
                            cohort_sha256=registration["cohort_sha256"], source_commit=physical["commit"],
                            runtime_sha256=descriptor["runtime_receipt_sha256"],
                            physical_receipt=physical, runtime_receipt=runtime)
    arm_initial = _verify_initial_binding(arm_root, descriptor["arm_initial_manifest_sha256"],
                            expected_kind="arm", arm_id=arm_id, config_sha256=registration["config_sha256"],
                            cohort_sha256=registration["cohort_sha256"], source_commit=physical["commit"],
                            runtime_sha256=descriptor["runtime_receipt_sha256"],
                            physical_receipt=physical, runtime_receipt=runtime)
    segment_initial = _verify_initial_binding(segment_dir, descriptor["segment_initial_manifest_sha256"],
                            expected_kind="segment", arm_id=arm_id, segment_index=segment_index,
                            nonce=descriptor["nonce"], config_sha256=registration["config_sha256"],
                            cohort_sha256=registration["cohort_sha256"], source_commit=physical["commit"],
                            runtime_sha256=descriptor["runtime_receipt_sha256"],
                            physical_receipt=physical, runtime_receipt=runtime)
    expected_matrix_identities = _record_identities(registration, physical["commit"],
        descriptor["runtime_receipt_sha256"], "K0")
    expected_arm_identities = _record_identities(registration, physical["commit"],
        descriptor["runtime_receipt_sha256"], arm_id)
    for value, expected in ((matrix_initial, expected_matrix_identities),
                            (arm_initial, expected_arm_identities),
                            (segment_initial, expected_arm_identities)):
        if value.get("identities") != expected or value.get("rng_streams") != config["seeds"]:
            raise ValueError("original pilot RunRecord checkpoint identities or RNG streams differ from registration")
    matrix_prefix = _verify_event_prefix(matrix_root / "events.jsonl", descriptor["matrix_event_prefix"])
    arm_prefix = _verify_event_prefix(arm_root / "events.jsonl", descriptor["arm_event_prefix"])
    segment_prefix = _verify_event_prefix(segment_dir / "events.jsonl", descriptor["segment_event_prefix"],
                                          require_empty=True)
    clocks = {"matrix": _validate_clock(descriptor["matrix_clock"], matrix_root),
              "arm": _validate_clock(descriptor["arm_clock"], arm_root)}
    if matrix_initial["started_utc"] != clocks["matrix"]["started_utc"] or \
            arm_initial["started_utc"] != clocks["arm"]["started_utc"]:
        raise ValueError("RunRecord original UTC anchors differ from persisted clock anchors")
    deadlines = _derive_deadlines(config["limits"], clocks)
    if descriptor["deadlines"] != deadlines["epochs"]:
        raise ValueError("worker descriptor original deadline epochs differ from registered anchors")
    matrix_deadlines = {key: deadlines["epochs"][key] for key in
        ("matrix_deadline_epoch", "matrix_watchdog_epoch")}
    if matrix_initial.get("settings", {}).get("original_deadlines") != matrix_deadlines or \
            arm_initial.get("settings", {}).get("original_deadlines") != deadlines["epochs"] or \
            segment_initial.get("settings", {}).get("original_deadlines") != deadlines["epochs"]:
        raise ValueError("RunRecord original deadlines differ from registered anchor derivation")
    expected_launcher = _validate_process_ref(descriptor["launcher"], "launcher")
    supervisor = _validate_process_ref(descriptor["supervisor"], "supervisor")
    arm_rows = _event_rows(arm_root / "events.jsonl")
    launch_rows = [row for row in arm_rows if row["kind"] == "worker_launch_command_issued" and
        row["payload"].get("segment_index") == segment_index]
    if len(launch_rows) != 1 or launch_rows[0]["payload"].get("descriptor_sha256") != descriptor_sha256:
        raise ValueError("actual worker launch command is not durably bound to this descriptor")
    command = launch_rows[0]["payload"].get("command")
    if type(command) is not list or not command or any(not isinstance(item, str) or not item for item in command) or \
            launch_rows[0]["payload"].get("command_sha256") != _sha(_canonical(command)):
        raise ValueError("actual worker launch command receipt is malformed")
    expected_command = [sys.executable, "-m", "dwr.experiments.source_pilot", "worker",
        "--authorization", str(descriptor_file), "--sha256", descriptor_sha256]
    if command != expected_command:
        raise ValueError("production worker argv differs from the closed authorization-only CLI")
    resume = _verify_opaque_resume(descriptor["resume_descriptor"], arm_root, arm_id, physical["commit"])
    return {"schema_version": "dwr-source-pilot-worker-authorization-v1", "root": str(source_root),
        "descriptor_path": str(descriptor_file), "descriptor_sha256": descriptor_sha256,
        "descriptor": descriptor, "registration": registration,
        "matrix_root": str(matrix_root),
        "arm_root": str(arm_root), "segment_dir": str(segment_dir), "arm_id": arm_id,
        "arm_index": arm_index, "arm": arms[arm_index], "segment_index": segment_index,
        "nonce": descriptor["nonce"], "identities": identities,
        "bindings": {"source": {"commit": physical["commit"], "physical_receipt": physical,
            "receipt_sha256": descriptor["physical_source_receipt_sha256"]},
            "runtime": {"receipt": runtime, "receipt_sha256": descriptor["runtime_receipt_sha256"]},
            "matrix": {"initial_manifest_sha256": descriptor["matrix_initial_manifest_sha256"],
                "config_sha256": registration["config_sha256"], "cohort_sha256": registration["cohort_sha256"]},
            "arm": {"initial_manifest_sha256": descriptor["arm_initial_manifest_sha256"],
                "arm_id": arm_id, "identities": identities}},
        "clocks": clocks, "deadlines": deadlines,
        "expected_launcher": expected_launcher,
        "resume_descriptor": resume}


def _runtime_receipt() -> dict:
    import importlib.metadata
    packages = sorted(({"name": dist.metadata.get("Name", dist.name), "version": dist.version}
                       for dist in importlib.metadata.distributions()),
                      key=lambda item: (item["name"].lower(), item["version"]))
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "implementation": sys.implementation.name, "packages": packages}


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _registration(root: Path):
    from dwr.experiments.source_pilot_registration import load_pilot_registration
    return load_pilot_registration(root, root / _CONFIG_PATH)


def _record_identities(registration: Mapping[str, object], source_commit: str,
                       runtime_sha256: str, arm_id: str | None = None) -> dict:
    from dwr.experiments.source_pilot_registration import pilot_identities
    ident = pilot_identities(registration, source_commit=source_commit, arm_id=arm_id or "K0")
    model = registration["model_pins"]
    return {
        "source_commit": {"name": "source-commit", "revision": source_commit,
            "sha256": _sha(source_commit.encode("ascii"))},
        "pilot_config": {"name": "source-validation-pilot-config",
            "revision": registration["config"]["config_version"], "sha256": registration["config_sha256"]},
        "cohort": {"name": "source-validation-pilot-cohort", "revision": "v1",
            "sha256": registration["cohort_sha256"]},
        "model": {"name": model["repo_id"], "revision": model["revision"], "sha256": ident["model"]},
        "tokenizer": {"name": model["repo_id"], "revision": model["revision"], "sha256": ident["tokenizer"]},
        "runtime": {"name": "source-validation-pilot-runtime", "revision": "registered-v1",
            "sha256": runtime_sha256}}


def _record_settings(registration: Mapping[str, object], physical: Mapping[str, object],
                     runtime: Mapping[str, object], *, record_kind: str,
                     original_deadlines: Mapping[str, object], arm_id: str | None = None,
                     arm_index: int | None = None, segment_index: int | None = None,
                     nonce: str | None = None) -> dict:
    settings = {"record_kind": record_kind, "study_type": registration["config"]["study_type"],
        "config_sha256": registration["config_sha256"], "cohort_sha256": registration["cohort_sha256"],
        "source_commit": physical["commit"], "physical_source_receipt": dict(physical),
        "physical_source_receipt_sha256": _sha(_canonical(physical)), "runtime_receipt": dict(runtime),
        "runtime_sha256": _sha(_canonical(runtime)), "original_deadlines": dict(original_deadlines)}
    if record_kind in {"arm", "segment"}:
        settings.update({"arm_id": arm_id, "arm_index": arm_index})
    if record_kind == "segment":
        settings.update({"segment_index": segment_index, "nonce": nonce})
    return settings


def _new_record(output_dir: Path, registration: Mapping[str, object], physical: Mapping[str, object],
                runtime: Mapping[str, object], *, record_kind: str,
                original_deadlines: Mapping[str, object], arm_id: str | None = None,
                arm_index: int | None = None, segment_index: int | None = None,
                nonce: str | None = None, config_path: Path | None = None) -> RunRecord:
    config = registration["config"]
    arm_identity = arm_id or "K0"
    if record_kind == "matrix":
        wall = config["limits"]["total_wall_seconds"]
    elif record_kind == "arm":
        wall = config["limits"]["arm_wall_seconds"]
    else:
        wall = min(config["limits"]["arm_wall_seconds"],
                   max(1, int(original_deadlines.get("effective_arm_deadline_epoch", time.time()+5400)-time.time())))
    budgets = {"wall_seconds": wall, "memory_mib": config["limits"]["maximum_peak_allocated_mib"],
               "attempts": config["limits"]["maximum_workers_per_arm"]}
    command = list(sys.argv)
    if not command:
        raise ValueError("pilot RunRecord requires the actual creating process command")
    return RunRecord.create(output_dir, command=command,
        config_path=config_path or registration["config_path"],
        expected_config_sha256=registration["config_sha256"],
        identities=_record_identities(registration, physical["commit"], _sha(_canonical(runtime)), arm_identity),
        rng_streams=config["seeds"], settings=_record_settings(registration, physical, runtime,
            record_kind=record_kind, original_deadlines=original_deadlines, arm_id=arm_id,
            arm_index=arm_index, segment_index=segment_index, nonce=nonce),
        budgets=budgets, confirmatory=False, source_root=Path(registration["config_path"]).resolve().parents[2])


def _clock_from_record(record_root: Path) -> dict:
    initial = strict_json_loads((record_root / "manifest.initial.json").read_text(encoding="utf-8"))
    started = initial["started_utc"]
    epoch = _parse_utc(started).timestamp()
    now = __import__("time").time()
    clock = {"started_utc": started, "started_epoch": epoch,
        "elapsed_high_water_seconds": max(0.0, now - epoch), "last_utc_epoch": now}
    _exclusive_json(record_root / "elapsed-budget.json", {"schema_version": "dwr-source-pilot-elapsed-v1", **clock})
    return clock


def _clock_snapshot(record_root: Path) -> dict:
    path = record_root / "elapsed-budget.json"
    raw = path.read_bytes()
    value = strict_json_loads(raw.decode("utf-8"))
    if type(value) is not dict or raw != _canonical(value) + b"\n" or \
            value.get("schema_version") != "dwr-source-pilot-elapsed-v1":
        raise ValueError("durable pilot clock receipt is malformed")
    return {key: value[key] for key in ("started_utc", "started_epoch", "elapsed_high_water_seconds", "last_utc_epoch")}


def _event_prefix(record_root: Path) -> dict:
    path = record_root / "events.jsonl"
    raw = path.read_bytes()
    from dwr.experiments.records import _validate_events
    _validate_events(raw)
    return {"byte_length": len(raw), "sha256": _sha(raw)}


def _issue_worker_descriptor(root: Path, registration: Mapping[str, object], physical: Mapping[str, object],
                             runtime: Mapping[str, object], matrix_root: Path, arm_root: Path,
                             segment_root: Path, *, arm_id: str, arm_index: int, segment_index: int,
                             matrix_initial_sha256: str, arm_initial_sha256: str,
                             resume_descriptor: Mapping[str, object] | None) -> tuple[Path, str, dict]:
    config = registration["config"]
    nonce = uuid.uuid4().hex
    segment_record = _new_record(segment_root, registration, physical, runtime,
        record_kind="segment", original_deadlines=_derive_deadlines(config["limits"],
            {"matrix": _clock_snapshot(matrix_root), "arm": _clock_snapshot(arm_root)})["epochs"],
        arm_id=arm_id, arm_index=arm_index, segment_index=segment_index, nonce=nonce)
    segment_initial_raw = (segment_root / "manifest.initial.json").read_bytes()
    segment_events = _event_prefix(segment_root)
    if segment_events != {"byte_length": 0, "sha256": _sha(b"")}:
        raise ValueError("new segment RunRecord event stream must be empty before issue")
    matrix_clock, arm_clock = _clock_snapshot(matrix_root), _clock_snapshot(arm_root)
    deadlines = _derive_deadlines(config["limits"], {"matrix": matrix_clock, "arm": arm_clock})
    supervisor_identity = _process_identity(os.getpid())
    process_ref = {"pid": supervisor_identity["pid"], "creation_time": supervisor_identity["creation_time"],
                   "image": supervisor_identity["image"]}
    resume = None if resume_descriptor is None else dict(resume_descriptor)
    descriptor = {"schema_version": "dwr-source-validation-pilot-worker-descriptor-v1",
        "matrix_root": str(matrix_root.resolve(strict=True)), "arm_root": str(arm_root.resolve(strict=True)),
        "segment_dir": str(segment_root.resolve(strict=True)), "arm_id": arm_id, "arm_index": arm_index,
        "segment_index": segment_index, "nonce": nonce, "config_sha256": registration["config_sha256"],
        "cohort_sha256": registration["cohort_sha256"], "source_commit": physical["commit"],
        "physical_source_receipt": dict(physical), "physical_source_receipt_sha256": _sha(_canonical(physical)),
        "runtime_receipt": dict(runtime), "runtime_receipt_sha256": _sha(_canonical(runtime)),
        "matrix_initial_manifest_sha256": matrix_initial_sha256,
        "arm_initial_manifest_sha256": arm_initial_sha256,
        "matrix_event_prefix": _event_prefix(matrix_root), "arm_event_prefix": _event_prefix(arm_root),
        "segment_initial_manifest_sha256": _sha(segment_initial_raw), "segment_event_prefix": segment_events,
        "matrix_clock": matrix_clock, "arm_clock": arm_clock, "deadlines": deadlines["epochs"],
        "supervisor": process_ref, "launcher": process_ref, "resume_descriptor": resume}
    descriptor_path = segment_root / "worker-descriptor.json"
    digest = _exclusive_json(descriptor_path, descriptor)
    return descriptor_path, digest, descriptor


def _validate_clock(value: object, anchor_root: Path) -> dict:
    expected = {"started_utc", "started_epoch", "elapsed_high_water_seconds", "last_utc_epoch"}
    if type(value) is not dict or set(value) != expected:
        raise ValueError("clock anchor has unknown or missing fields")
    from datetime import datetime
    try:
        parsed = datetime.fromisoformat(value["started_utc"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("clock anchor timestamp is malformed") from exc
    if parsed.tzinfo != timezone.utc or abs(parsed.timestamp() - value["started_epoch"]) > 0.01:
        raise ValueError("clock anchor UTC and epoch disagree")
    if any(type(value[key]) not in (int, float) or not math.isfinite(value[key])
           for key in ("started_epoch", "elapsed_high_water_seconds", "last_utc_epoch")):
        raise ValueError("clock anchor values must be finite numbers")
    if value["elapsed_high_water_seconds"] < 0 or value["last_utc_epoch"] < value["started_epoch"]:
        raise ValueError("clock anchor high-water values are invalid")
    path = anchor_root / "elapsed-budget.json"
    _reject_links(path)
    try:
        raw = path.read_bytes()
        receipt = strict_json_loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("durable clock high-water receipt is unavailable") from exc
    if type(receipt) is not dict or raw != _canonical(receipt) + b"\n":
        raise ValueError("durable clock high-water receipt is not canonical")
    if receipt != {"schema_version": "dwr-source-pilot-elapsed-v1", **value}:
        raise ValueError("descriptor clock differs from durable elapsed high-water receipt")
    _validated_elapsed(value["last_utc_epoch"], __import__("time").time(), value["elapsed_high_water_seconds"])
    return dict(value)


def _validate_issued_clock(value: object, anchor_root: Path) -> dict:
    """Validate a historical issue snapshot against the monotone live receipt.

    Fresh worker authorization uses _validate_clock and therefore still requires
    an exact snapshot. This variant is only for replaying issued descriptors
    after work has advanced the durable elapsed-budget receipt.
    """
    expected = {"started_utc", "started_epoch", "elapsed_high_water_seconds", "last_utc_epoch"}
    if type(value) is not dict or set(value) != expected:
        raise ValueError("clock anchor has unknown or missing fields")
    from datetime import datetime
    try:
        parsed = datetime.fromisoformat(value["started_utc"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("clock anchor timestamp is malformed") from exc
    if parsed.tzinfo != timezone.utc or abs(parsed.timestamp() - value["started_epoch"]) > 0.01:
        raise ValueError("clock anchor UTC and epoch disagree")
    if any(type(value[key]) not in (int, float) or not math.isfinite(value[key])
           for key in ("started_epoch", "elapsed_high_water_seconds", "last_utc_epoch")):
        raise ValueError("clock anchor values must be finite numbers")
    if value["elapsed_high_water_seconds"] < 0 or value["last_utc_epoch"] < value["started_epoch"]:
        raise ValueError("clock anchor high-water values are invalid")
    path = anchor_root / "elapsed-budget.json"
    _reject_links(path)
    try:
        raw = path.read_bytes()
        receipt = strict_json_loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("durable clock high-water receipt is unavailable") from exc
    if type(receipt) is not dict or raw != _canonical(receipt) + b"\n":
        raise ValueError("durable clock high-water receipt is not canonical")
    if receipt.get("schema_version") != "dwr-source-pilot-elapsed-v1" or \
            set(receipt) != {"schema_version", *expected}:
        raise ValueError("durable clock high-water receipt has unknown or missing fields")
    current = {key: receipt[key] for key in expected}
    if current["started_utc"] != value["started_utc"] or current["started_epoch"] != value["started_epoch"]:
        raise ValueError("durable clock no longer has the original anchor")
    if any(type(current[key]) not in (int, float) or not math.isfinite(current[key])
           for key in ("started_epoch", "elapsed_high_water_seconds", "last_utc_epoch")):
        raise ValueError("durable clock high-water values must be finite numbers")
    if current["elapsed_high_water_seconds"] < value["elapsed_high_water_seconds"] or \
            current["last_utc_epoch"] < value["last_utc_epoch"]:
        raise ValueError("durable clock high-water receipt rolled back")
    _validated_elapsed(current["last_utc_epoch"], __import__("time").time(),
                       current["elapsed_high_water_seconds"])
    return dict(value)


def _derive_deadlines(limits: Mapping[str, object], clocks: Mapping[str, dict]) -> dict:
    matrix_start = clocks["matrix"]["started_epoch"]
    arm_start = clocks["arm"]["started_epoch"]
    matrix_deadline = matrix_start + limits["total_wall_seconds"]
    matrix_watchdog = matrix_start + limits["matrix_external_watchdog_seconds"]
    arm_deadline = min(arm_start + limits["arm_wall_seconds"], matrix_deadline)
    work_deadline = arm_deadline - limits["arm_finalization_reserve_seconds"]
    arm_watchdog = min(arm_start + limits["arm_external_watchdog_seconds"], matrix_watchdog)
    epochs = {"matrix_deadline_epoch": matrix_deadline, "matrix_watchdog_epoch": matrix_watchdog,
        "effective_arm_deadline_epoch": arm_deadline, "arm_work_deadline_epoch": work_deadline,
        "arm_watchdog_epoch": arm_watchdog}
    now_epoch = __import__("time").time()
    now_monotonic = __import__("time").monotonic()
    matrix_elapsed = _validated_elapsed(clocks["matrix"]["last_utc_epoch"], now_epoch,
                                        clocks["matrix"]["elapsed_high_water_seconds"])
    arm_elapsed = _validated_elapsed(clocks["arm"]["last_utc_epoch"], now_epoch,
                                     clocks["arm"]["elapsed_high_water_seconds"])
    mono = {"matrix_monotonic_deadline": now_monotonic + max(0.0, limits["total_wall_seconds"] - matrix_elapsed),
        "matrix_watchdog_monotonic_deadline": now_monotonic + max(0.0, limits["matrix_external_watchdog_seconds"] - matrix_elapsed),
        "arm_monotonic_deadline": now_monotonic + max(0.0, arm_deadline - now_epoch),
        "work_monotonic_deadline": now_monotonic + max(0.0, work_deadline - now_epoch),
        "seal_monotonic_deadline": now_monotonic + max(0.0, arm_deadline - now_epoch),
        "arm_watchdog_monotonic_deadline": now_monotonic + max(0.0, arm_watchdog - now_epoch)}
    return {"epochs": epochs, **mono}


def _validate_process_ref(value: object, label: str) -> dict:
    if type(value) is not dict or set(value) != {"pid", "creation_time", "image"}:
        raise ValueError(f"{label} identity has unknown or missing fields")
    if type(value["pid"]) is not int or value["pid"] <= 0 or not all(
            isinstance(value[name], str) and value[name] for name in ("creation_time", "image")):
        raise ValueError(f"{label} identity is malformed")
    return dict(value)


def _verify_initial_binding(record_root: Path, expected_sha256: str, *, expected_kind: str,
                            config_sha256: str, cohort_sha256: str, source_commit: str,
                            runtime_sha256: str, physical_receipt: Mapping[str, object],
                            runtime_receipt: Mapping[str, object], arm_id: str | None = None,
                            segment_index: int | None = None, nonce: str | None = None) -> dict:
    from dwr.experiments.records import _validate_events
    initial_path = record_root / "manifest.initial.json"
    events_path = record_root / "events.jsonl"
    terminal_path = record_root / "manifest.json"
    for path in (initial_path, events_path):
        _reject_links(path)
    if terminal_path.exists():
        raise ValueError("worker cannot authorize a terminal matrix or arm record")
    try:
        raw = initial_path.read_bytes()
        initial = strict_json_loads(raw.decode("utf-8"))
        event_raw = events_path.read_bytes()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("original pilot RunRecord is incomplete or malformed") from exc
    if _sha(raw) != expected_sha256 or raw != _canonical(initial) + b"\n":
        raise ValueError("original pilot RunRecord initial bytes/hash changed")
    _validate_events(event_raw)
    settings = initial.get("settings")
    expected_settings = {"record_kind", "study_type", "config_sha256", "cohort_sha256", "source_commit",
        "physical_source_receipt", "physical_source_receipt_sha256", "runtime_receipt", "runtime_sha256",
        "original_deadlines"}
    if expected_kind in {"arm", "segment"}:
        expected_settings.update({"arm_id", "arm_index"})
    if expected_kind == "segment":
        expected_settings.update({"segment_index", "nonce"})
    if (type(settings) is not dict or settings.get("record_kind") != expected_kind or
            set(settings) != expected_settings or settings.get("study_type") != _registration(_root())["config"]["study_type"] or
            settings.get("config_sha256") != config_sha256 or
            settings.get("cohort_sha256") != cohort_sha256 or
            settings.get("source_commit") != source_commit or
            settings.get("runtime_sha256") != runtime_sha256):
        raise ValueError("original pilot RunRecord binding differs from worker descriptor")
    if settings.get("physical_source_receipt") != physical_receipt or settings.get("runtime_receipt") != runtime_receipt or \
            settings.get("physical_source_receipt_sha256") != _sha(_canonical(physical_receipt)):
        raise ValueError("original pilot RunRecord source/runtime receipts differ from worker descriptor")
    if expected_kind == "arm" and settings.get("arm_id") != arm_id:
        raise ValueError("original arm RunRecord names another arm")
    if expected_kind == "segment" and (settings.get("arm_id") != arm_id or
            settings.get("segment_index") != segment_index or settings.get("nonce") != nonce):
        raise ValueError("original segment RunRecord differs from issued arm, segment or nonce")
    if initial.get("config", {}).get("sha256") != config_sha256:
        raise ValueError("original pilot RunRecord config hash changed")
    if initial.get("source", {}).get("commit") != source_commit:
        raise ValueError("original pilot RunRecord source commit changed")
    return initial


def _verify_event_prefix(path: Path, reference: object, *, require_empty: bool = False) -> dict:
    from dwr.experiments.records import _validate_events
    if type(reference) is not dict or set(reference) != {"byte_length", "sha256"}:
        raise ValueError("event prefix reference has unknown or missing fields")
    length, digest = reference["byte_length"], reference["sha256"]
    if type(length) is not int or length < 0 or not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest):
        raise ValueError("event prefix reference is malformed")
    _reject_links(path)
    raw = path.read_bytes()
    if length > len(raw):
        raise ValueError("event stream is shorter than its issued prefix")
    prefix = raw[:length]
    if _sha(prefix) != digest or (prefix and not prefix.endswith(b"\n")):
        raise ValueError("event stream issued prefix bytes or boundary changed")
    if require_empty and (length != 0 or raw != b""):
        raise ValueError("segment event stream changed before worker authorization")
    _validate_events(raw)
    return {"byte_length": length, "sha256": digest, "current_byte_length": len(raw),
            "current_sha256": _sha(raw)}


def _verify_opaque_resume(value: object, arm_root: Path, arm_id: str, source_commit: str):
    if value is None:
        return None
    expected = {"path", "sha256", "source_commit", "arm_id"}
    if type(value) is not dict or set(value) != expected:
        raise ValueError("resume reference has unknown or missing fields")
    if value["source_commit"] != source_commit or value["arm_id"] != arm_id:
        raise ValueError("resume reference source or arm differs from authorization")
    if not isinstance(value["path"], str) or "\\" in value["path"]:
        raise ValueError("resume reference path must use forward slashes")
    candidate = Path(value["path"])
    if candidate.is_absolute():
        path = candidate
    else:
        if ".." in candidate.parts:
            raise ValueError("resume reference path contains traversal")
        path = arm_root.joinpath(*candidate.parts)
    _reject_links(path)
    resolved = path.resolve(strict=True)
    if arm_root.resolve(strict=True) not in resolved.parents:
        raise ValueError("resume reference escapes its arm root")
    raw = resolved.read_bytes()
    try:
        opaque = strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("resume cursor is not strict UTF-8 JSON") from exc
    if raw != _canonical(opaque) + b"\n":
        raise ValueError("resume cursor JSON is not canonical stable JSON")
    if not isinstance(value["sha256"], str) or len(value["sha256"]) != 64 or any(
            char not in "0123456789abcdef" for char in value["sha256"]):
        raise ValueError("resume reference SHA256 is malformed")
    if _sha(raw) != value["sha256"]:
        raise ValueError("resume reference SHA256 mismatch")
    return {**value, "path": str(resolved), "resolved_path": str(resolved)}


def open_segment_record(authorization: Mapping[str, object]):
    """Reconstitute only the issued, unsealed segment writer at its original anchor."""
    from dwr.experiments.records import RunRecord, _validate_events
    root = Path(str(authorization["segment_dir"])).resolve(strict=True)
    initial_path, events_path = root / "manifest.initial.json", root / "events.jsonl"
    for path in (initial_path, events_path, root / "manifest.json"):
        _reject_links(path)
    if (root / "manifest.json").exists():
        raise ValueError("cannot reopen a terminal worker segment")
    raw = initial_path.read_bytes()
    if _sha(raw) != authorization["descriptor"]["segment_initial_manifest_sha256"]:
        raise ValueError("segment initial manifest SHA differs from authorization")
    initial = strict_json_loads(raw.decode("utf-8"))
    if raw != _canonical(initial) + b"\n":
        raise ValueError("segment initial manifest is not canonical")
    settings = initial.get("settings")
    expected = {"record_kind": "segment", "arm_id": authorization["arm_id"],
        "segment_index": authorization["segment_index"], "nonce": authorization["nonce"],
        "config_sha256": authorization["registration"]["config_sha256"],
        "cohort_sha256": authorization["registration"]["cohort_sha256"],
        "source_commit": authorization["bindings"]["source"]["commit"],
        "runtime_sha256": authorization["bindings"]["runtime"]["receipt_sha256"]}
    if type(settings) is not dict or any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("segment initial immutable binding differs from worker authorization")
    event_raw = events_path.read_bytes()
    _validate_events(event_raw)
    ref = authorization["descriptor"]["segment_event_prefix"]
    if len(event_raw) < ref["byte_length"] or \
            _sha(event_raw[:ref["byte_length"]]) != ref["sha256"]:
        raise ValueError("segment event stream no longer begins with its issued prefix")
    if event_raw and not event_raw.endswith(b"\n"):
        raise ValueError("segment event stream has an incomplete final row")
    start = initial["started_utc"]
    epoch = _parse_utc(start).timestamp()
    elapsed = max(0.0, __import__("time").time() - epoch)
    return RunRecord(root, start, __import__("time").monotonic() - elapsed, _sha(raw))


def _event_rows(path: Path) -> list[dict]:
    from dwr.experiments.records import _validate_events
    raw = path.read_bytes()
    _validate_events(raw)
    rows = []
    for line in raw.splitlines():
        if not line:
            continue
        row = strict_json_loads(line.decode("utf-8"))
        segment_text = path.parent.name.removeprefix("segment-")
        segment_index = int(segment_text) if segment_text.isdigit() else -1
        rows.append({"segment_index": segment_index, "kind": row["kind"], "payload": row["payload"]})
    return rows


def _event_paths(authorization: Mapping[str, object]) -> list[Path]:
    arm_root = Path(str(authorization["arm_root"]))
    current = authorization["segment_index"]
    paths = []
    for index in range(current + 1):
        path = arm_root / f"segment-{index:03d}" / "events.jsonl"
        if not path.is_file():
            if index == current:
                raise ValueError("current worker segment has no RunRecord event stream")
            continue
        paths.append(path)
    return paths


def _physical_ledger(authorization: Mapping[str, object], *, proposed: tuple[str, Mapping[str, object]] | None = None) -> dict:
    events = []
    for path in _event_paths(authorization):
        events.extend(_event_rows(path))
    if proposed is not None:
        events.append({"segment_index": authorization["segment_index"],
                       "kind": proposed[0], "payload": dict(proposed[1])})
    arm_id = authorization["arm_id"]
    config = authorization.get("config", authorization.get("registration", {}).get("config"))
    if not isinstance(config, Mapping):
        raise ValueError("worker authorization lacks registered physical-work config")
    limits = config["limits"]
    starts = [row for row in events if row["kind"] == "update_started"]
    updates = [row for row in events if row["kind"] == "update_completed"]
    eval_starts = [row for row in events if row["kind"] == "evaluation_case_started"]
    ce_starts = [row for row in eval_starts if row["payload"].get("include_ce") is True]
    ce_done = [row for row in events if row["kind"] == "teacher_forced_completed"]
    predictions = [row for row in events if row["kind"] == "prediction_completed"]
    if len(starts) > limits["maximum_physical_updates_per_arm"]:
        raise ValueError("registered physical optimizer-update cap exceeded")
    pred_cap = limits["per_arm_physical_predictions"][arm_id]
    ce_cap = limits["per_arm_teacher_forced_ce_answer_forwards"][arm_id]
    if len(eval_starts) > pred_cap:
        raise ValueError("registered physical evaluation-prediction cap exceeded")
    if len(ce_starts) > ce_cap:
        raise ValueError("registered teacher-forced CE forward cap exceeded")
    completed_by_pair: dict[tuple[object, object], int] = {}
    for row in predictions:
        payload = row["payload"]
        pair = (row["segment_index"], payload.get("stage"), payload.get("example_id"))
        completed_by_pair[pair] = completed_by_pair.get(pair, 0) + 1
        calls = payload.get("greedy_answer_forwards", payload.get("greedy_decoder_calls"))
        if type(calls) is not int or not (0 <= calls <= limits["decoder_calls_per_prediction_bound"]):
            raise ValueError("prediction event has an invalid greedy decoder-call count")
    started_attempts = set()
    for row in eval_starts:
        payload = row["payload"]
        if type(payload.get("include_ce")) is not bool:
            raise ValueError("evaluation start event does not declare CE accounting")
        attempt = (row["segment_index"], payload.get("stage"), payload.get("example_id"), payload.get("attempt"))
        if attempt in started_attempts:
            raise ValueError("duplicate started prediction attempt in physical event history")
        started_attempts.add(attempt)
    started_by_pair: dict[tuple[object, object], int] = {}
    for segment_index, stage, example_id, _attempt in started_attempts:
        pair = (segment_index, stage, example_id)
        started_by_pair[pair] = started_by_pair.get(pair, 0) + 1
    for pair, count in completed_by_pair.items():
        if count > started_by_pair.get(pair, 0):
            raise ValueError("completed prediction has no matching started physical attempt")
    if len(ce_done) > len(ce_starts):
        raise ValueError("completed CE forwards exceed their durable started reservations")
    known_calls = sum(row["payload"].get("greedy_answer_forwards",
        row["payload"].get("greedy_decoder_calls", 0)) for row in predictions)
    unknown_predictions = sum(count - completed_by_pair.get(pair, 0)
                              for pair, count in started_by_pair.items())
    conservative_calls = known_calls + unknown_predictions * limits["decoder_calls_per_prediction_bound"]
    greedy_cap = limits["per_arm_greedy_decoder_calls"][arm_id]
    if conservative_calls > greedy_cap:
        raise ValueError("registered greedy decoder-forward cap exceeded")
    bank_starts = [row for row in events if row["kind"] == "bank_started"]
    bank_completes = [row for row in events if row["kind"] == "bank_completed"]
    arm_steps = next(item["steps"] for item in config["arms"] if item["arm_id"] == arm_id)
    bank_ids = sum(len(row["payload"].get("example_ids", [])) for row in bank_starts)
    result = {"physical_updates_started": len(starts), "physical_updates_completed": len(updates),
        "updates_in_flight_or_unknown": max(0, len(starts) - len(updates)),
        "prediction_attempts_started": len(started_attempts),
        "predictions_completed": sum(completed_by_pair.values()), "predictions_in_flight_or_unknown": unknown_predictions,
        "greedy_decoder_calls_known": known_calls, "greedy_decoder_calls_conservative": conservative_calls,
        "teacher_forced_ce_started": len(ce_starts), "teacher_forced_ce_completed": len(ce_done),
        "teacher_forced_ce_in_flight_or_unknown": max(0, len(ce_starts) - len(ce_done)),
        "training_answer_forwards_started": len(starts),
        "training_prompt_encodes_started": len(starts),
        "training_workspace_updates_started": len(starts) * arm_steps,
        "bank_preparation_prompt_encodes_started": bank_ids,
        "bank_preparation_workspace_updates_started": bank_ids * arm_steps,
        "bank_preparations_completed": len(bank_completes),
        "prompt_encodes_reported": sum(int(row["payload"].get("prompt_encodes", 0)) for row in predictions),
        "workspace_updates_reported": sum(int(row["payload"].get("workspace_updates", 0)) for row in predictions)}
    recoveries = max(0, len(_event_paths(authorization)) - 1)
    current_segment = Path(str(authorization["arm_root"])) / f"segment-{authorization['segment_index']:03d}"
    active_reservation = proposed is not None or not (current_segment / "manifest.json").exists()
    allowed_unknown = recoveries + int(active_reservation)
    if result["updates_in_flight_or_unknown"] > allowed_unknown or \
            result["predictions_in_flight_or_unknown"] > allowed_unknown or \
            result["teacher_forced_ce_in_flight_or_unknown"] > allowed_unknown or \
            len(bank_starts) - len(bank_completes) > allowed_unknown:
        raise ValueError("incomplete physical work exceeds the registered per-recovery reservation allowance")
    return result


def check_worker_budget(authorization: Mapping[str, object], stage: str) -> None:
    """Enforce immutable wall and physical-compute limits; memory is controller-owned."""
    import time
    _touch_clocks(authorization)
    deadlines = authorization["deadlines"]
    seal = stage in {"seal", "finalization", "audit"}
    cutoff = deadlines["seal_monotonic_deadline"] if seal else deadlines["work_monotonic_deadline"]
    if time.monotonic() >= cutoff or time.monotonic() >= deadlines["matrix_monotonic_deadline"]:
        raise TimeoutError("registered pilot work or seal deadline reached")
    if time.monotonic() >= deadlines["matrix_watchdog_monotonic_deadline"]:
        raise TimeoutError("registered matrix outer watchdog deadline reached")
    _physical_ledger(authorization)


def guard_worker_event(authorization: Mapping[str, object], record, kind: str,
                       payload: Mapping[str, object]) -> None:
    """Reserve work then append the same event with Python sequences in JSON-native form."""
    normalized = _json_sequence_copy(payload)
    seal_only = kind == "arm_complete" or (kind == "recovery_committed" and
        normalized.get("phase") == "complete")
    check_worker_budget(authorization, "seal" if seal_only else "model_work")
    if kind in {"update_started", "evaluation_case_started"}:
        _physical_ledger(authorization, proposed=(kind, normalized))
    record.append_event(kind, normalized)


def _json_sequence_copy(value: object):
    """Copy nested mappings/sequences, mapping tuples to JSON arrays without coercing values."""
    if isinstance(value, Mapping):
        return {key: _json_sequence_copy(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sequence_copy(child) for child in value]
    return value


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("UTC anchor must be text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("UTC anchor is malformed") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError("UTC anchor must be timezone-aware UTC")
    return parsed


def _atomic_json(path: Path, value: Mapping[str, object]) -> str:
    from dwr.experiments.source_overfit_job import _atomic_json as atomic_json
    return atomic_json(path, dict(value))


def _advance_clock(root: Path) -> dict:
    path = root / "elapsed-budget.json"
    _reject_links(path)
    raw = path.read_bytes()
    value = strict_json_loads(raw.decode("utf-8"))
    if type(value) is not dict or raw != _canonical(value) + b"\n" or \
            value.get("schema_version") != "dwr-source-pilot-elapsed-v1":
        raise ValueError("durable pilot clock receipt is malformed")
    now = __import__("time").time()
    elapsed = _validated_elapsed(value["last_utc_epoch"], now, value["elapsed_high_water_seconds"])
    value["elapsed_high_water_seconds"] = elapsed
    value["last_utc_epoch"] = now
    _atomic_json(path, value)
    return value


def _touch_clocks(authorization: Mapping[str, object]) -> None:
    _advance_clock(Path(str(authorization["matrix_root"])))
    _advance_clock(Path(str(authorization["arm_root"])))


def _exclusive_json(path: Path, value: Mapping[str, object]) -> str:
    raw = _canonical(dict(value)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha(raw)


def _process_identity(pid: int) -> dict:
    from dwr.experiments.source_overfit_job import (
        _process_creation_time, _process_image, _process_parent_pid,
    )
    creation = _process_creation_time(pid)
    parent_pid = _process_parent_pid(pid)
    return {"pid": pid, "creation_time": creation, "image": _process_image(pid),
            "parent_pid": parent_pid, "parent_creation_time": _process_creation_time(parent_pid),
            "parent_image": _process_image(parent_pid)}


def _worker_lock_path(arm_root: Path) -> Path:
    return arm_root / "worker.lock"


def _matrix_lock_path(root: Path) -> Path:
    return root / _MATRIX_LOCK_PATH


def _status_path(root: Path) -> Path:
    return root / "status.json"


def _read_status(root: Path) -> dict:
    path = _status_path(root)
    _reject_links(path)
    raw = path.read_bytes()
    value = strict_json_loads(raw.decode("utf-8"))
    if type(value) is not dict or raw != _canonical(value) + b"\n":
        raise ValueError("pilot status is malformed or noncanonical")
    allowed = {"schema_version", "matrix_root", "state", "supervisor", "worker", "worker_launcher", "watchdog",
        "config_sha256", "source_commit", "matrix_started_utc", "completed_arms", "current_arm",
        "matrix_deadline_epoch", "matrix_watchdog_epoch", "arm_started_utc", "error", "decision",
        "watchdog_failure_sha256"}
    if value.get("schema_version") != "dwr-source-validation-pilot-status-v1" or \
            value.get("matrix_root") != str(root.resolve(strict=True)) or set(value) - allowed:
        raise ValueError("pilot status schema or matrix binding differs")
    if value.get("state") not in {"starting", "running", "recovering", "completed", "scientific_no_go", "operational_failure", "failed"}:
        raise ValueError("pilot status lifecycle state is unknown")
    completed = value.get("completed_arms", [])
    if type(completed) is not list or completed != list(_ARM_ORDER[:len(completed)]):
        raise ValueError("pilot status arm progression is not an ordered registered prefix")
    for key in ("supervisor", "worker", "worker_launcher", "watchdog"):
        reference = value.get(key)
        if reference is not None and (type(reference) is not dict or set(reference) != {"pid", "creation_time", "image"} or
                type(reference["pid"]) is not int or reference["pid"] <= 0 or
                not all(isinstance(reference[name], str) and reference[name] for name in ("creation_time", "image"))):
            raise ValueError(f"pilot status {key} process reference is malformed")
    if value.get("current_arm") is not None and value["current_arm"] not in _ARM_ORDER:
        raise ValueError("pilot status current arm is unregistered")
    return value


def _write_status(root: Path, **changes) -> dict:
    current = {}
    if _status_path(root).exists():
        current = _read_status(root)
    current.update(changes)
    current.setdefault("schema_version", "dwr-source-validation-pilot-status-v1")
    current["matrix_root"] = str(root.resolve())
    _atomic_json(_status_path(root), current)
    return current


def _pid_live(reference: Mapping[str, object] | None) -> bool:
    if not isinstance(reference, Mapping):
        return False
    try:
        from dwr.experiments.source_overfit_job import _pid_matches, _process_image
        return bool(_pid_matches(reference.get("pid"), reference.get("creation_time"))) and \
            _process_image(reference["pid"]) == reference.get("image")
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _reference(pid: int) -> dict:
    ident = _process_identity(pid)
    return {key: ident[key] for key in ("pid", "creation_time", "image")}


_LAUNCH_RECEIPT = "worker-launch-receipt.json"
_LAUNCH_EVENT = "worker_launch_process_published"


def _launch_receipt_path(segment: Path) -> Path:
    return Path(segment) / _LAUNCH_RECEIPT


def _verified_launch_receipt(authorization: Mapping[str, object], *, live: bool = False) -> dict:
    """Join the post-Popen receipt to its issued command and exact OS parent chain."""
    segment = Path(str(authorization["segment_dir"])).resolve(strict=True)
    receipt_path = _launch_receipt_path(segment)
    receipt = _read_json_path(receipt_path)
    expected_keys = {"schema_version", "descriptor_sha256", "nonce", "command", "command_sha256",
        "issuer", "process", "process_parent"}
    if type(receipt) is not dict or set(receipt) != expected_keys or \
            receipt.get("schema_version") != "dwr-source-pilot-launch-receipt-v1":
        raise ValueError("worker launch process receipt has unknown or missing fields")
    descriptor = authorization["descriptor"]
    if receipt["descriptor_sha256"] != authorization["descriptor_sha256"] or \
            receipt["nonce"] != authorization["nonce"] or receipt["issuer"] != descriptor["launcher"] or \
            receipt["issuer"] != authorization["expected_launcher"]:
        raise ValueError("worker launch receipt differs from the issued descriptor owner")
    arm_root = Path(str(authorization["arm_root"])).resolve(strict=True)
    command = receipt["command"]
    command_rows = [row for row in _event_rows(arm_root / "events.jsonl")
        if row["kind"] == "worker_launch_command_issued" and
        row["payload"].get("segment_index") == authorization["segment_index"]]
    if len(command_rows) != 1 or command_rows[0]["payload"].get("descriptor_sha256") != authorization["descriptor_sha256"]:
        raise ValueError("worker launch command is not uniquely bound to the issued descriptor")
    issued_command = command_rows[0]["payload"].get("command")
    if type(command) is not list or command != issued_command or \
            command_rows[0]["payload"].get("command_sha256") != _sha(_canonical(issued_command)) or \
            receipt["command_sha256"] != _sha(_canonical(command)):
        raise ValueError("worker launch receipt differs from the exact issued argv")
    for name in ("issuer", "process", "process_parent"):
        value = receipt[name]
        if type(value) is not dict or set(value) != {"pid", "creation_time", "image"} or \
                type(value.get("pid")) is not int or value["pid"] <= 0 or \
                not all(isinstance(value.get(key), str) and value[key] for key in ("creation_time", "image")):
            raise ValueError("worker launch receipt contains a malformed process identity")
    issuer, process_ref, process_parent = receipt["issuer"], receipt["process"], receipt["process_parent"]
    if process_parent != issuer:
        raise ValueError("actual Popen launcher is not a direct child of the issuing supervisor")
    segment_index = authorization["segment_index"]
    rows = [row for row in _event_rows(arm_root / "events.jsonl")
        if row["kind"] == _LAUNCH_EVENT and row["payload"].get("segment_index") == segment_index]
    expected_ref = {"path": segment.relative_to(arm_root).as_posix() + "/" + _LAUNCH_RECEIPT,
        "sha256": _sha(receipt_path.read_bytes())}
    if len(rows) != 1 or rows[0]["payload"] != {"segment_index": segment_index,
            "descriptor_sha256": authorization["descriptor_sha256"], "nonce": authorization["nonce"],
            "launch_receipt": expected_ref}:
        raise ValueError("worker launch receipt lacks its unique durable arm event")
    if live:
        from dwr.experiments.source_overfit_job import _pid_matches, _process_creation_time, _process_image, _process_parent_pid
        if not _pid_matches(process_ref["pid"], process_ref["creation_time"]) or \
                _process_image(process_ref["pid"]) != process_ref["image"] or \
                _process_parent_pid(process_ref["pid"]) != issuer["pid"] or \
                _process_creation_time(issuer["pid"]) != issuer["creation_time"] or \
                _process_image(issuer["pid"]) != issuer["image"]:
            raise ValueError("live Popen launcher no longer matches its issued process chain")
    return receipt


def _worker_launcher_for_parent(authorization: Mapping[str, object], parent: Mapping[str, object], *,
                                live: bool, worker: Mapping[str, object] | None = None) -> dict:
    launch = _verified_launch_receipt(authorization, live=live)
    issuer, process_ref = launch["issuer"], launch["process"]
    if parent == issuer:
        if worker is not None and {key: worker.get(key) for key in ("pid", "creation_time", "image")} != process_ref:
            raise ValueError("direct worker is not the exact process launched by the issuing supervisor")
        return dict(issuer)
    if parent != process_ref:
        raise ValueError("actual worker parent is neither the issuer nor its exact Popen launcher")
    if launch["process_parent"] != issuer:
        raise ValueError("actual worker intermediary is not bound directly to the issuer")
    return dict(process_ref)


def start(config_path: str | os.PathLike[str] = _CONFIG_PATH) -> dict:
    """Idempotently attach to or detach the sole registered matrix supervisor."""
    root = _root()
    registration = _registration(root)
    supplied = Path(config_path)
    if not supplied.is_absolute():
        supplied = root / supplied
    if supplied.resolve(strict=True) != Path(registration["config_path"]).resolve(strict=True):
        raise ValueError("start accepts only the registered fixed pilot configuration")
    matrix_root = root / PurePosixPath(registration["config"]["output_dir"])
    lock_path = _matrix_lock_path(root)
    from dwr.experiments.source_overfit_job import _ByteLock
    try:
        lock = _ByteLock(lock_path, 0, wait=False)
    except RuntimeError:
        if matrix_root.exists():
            if not _status_path(matrix_root).is_file():
                return {"attached": True, "state": "starting", "matrix_root": str(matrix_root)}
            return {"attached": True, **_read_status(matrix_root)}
        raise
    try:
        if matrix_root.exists():
            status = _read_status(matrix_root)
            supervisor = status.get("supervisor")
            if _pid_live(supervisor):
                lock.close()
                for _ in range(40):
                    time.sleep(0.05)
                    try:
                        lock = _ByteLock(lock_path, 0, wait=False)
                    except RuntimeError:
                        return {"attached": True, **_read_status(matrix_root)}
                    if not _pid_live(supervisor):
                        break
                    lock.close()
                else:
                    raise ValueError("live supervisor does not hold the registered matrix lock")
            if status.get("state") in {"completed", "failed", "operational_failure", "scientific_no_go"}:
                return {"attached": True, **status}
            # If a worker survived supervisor loss, a new supervisor is launched below
            # and attaches to its exact process receipt before considering any new worker.
        else:
            matrix_root.mkdir(parents=True, exist_ok=False)
            _write_status(matrix_root, state="starting", supervisor=None, worker=None,
                config_sha256=registration["config_sha256"], completed_arms=[], current_arm=None)
        logs = root / "runs"
        logs.mkdir(exist_ok=True)
        stdout_path, stderr_path = logs / "source-pilot-supervisor.stdout.log", logs / "source-pilot-supervisor.stderr.log"
        process = subprocess.Popen([sys.executable, "-m", "dwr.experiments.source_pilot_job",
            "_supervise", "--config", str(supplied.resolve())], cwd=root, stdin=subprocess.DEVNULL,
            stdout=stdout_path.open("ab"), stderr=stderr_path.open("ab"),
            creationflags=(getattr(subprocess, "DETACHED_PROCESS", 0) |
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)))
        supervisor = _reference(process.pid)
        _write_status(matrix_root, state="starting", supervisor=supervisor)
        return {"attached": False, "supervisor": supervisor,
            "stdout": str(stdout_path), "stderr": str(stderr_path), "config_sha256": registration["config_sha256"]}
    finally:
        lock.close()


def _verify_record_if_terminal(path: Path) -> dict | None:
    from dwr.experiments.records import verify_artifacts
    terminal = path / "manifest.json"
    if not terminal.is_file():
        return None
    return verify_artifacts(path)


def _integrity_of_record(path: Path) -> dict:
    terminal = _verify_record_if_terminal(path)
    if terminal is not None:
        return {"status": terminal["status"], "terminal_verified": True}
    if not (path / "manifest.initial.json").is_file():
        return {"status": "not_created", "terminal_verified": False}
    from dwr.experiments.records import _validate_events, _validate_initial
    initial_raw = (path / "manifest.initial.json").read_bytes()
    initial = strict_json_loads(initial_raw.decode("utf-8"))
    if initial_raw != _canonical(initial) + b"\n":
        raise ValueError("unsealed pilot RunRecord initial manifest is not canonical")
    _validate_initial(initial)
    events_raw = (path / "events.jsonl").read_bytes()
    count = _validate_events(events_raw)
    elapsed = None
    if (path / "elapsed-budget.json").is_file():
        elapsed = _clock_snapshot(path)
        _validated_elapsed(elapsed["last_utc_epoch"], time.time(), elapsed["elapsed_high_water_seconds"])
    return {"status": "running", "terminal_verified": False, "initial_manifest_sha256": _sha(initial_raw),
        "event_count": count, "events_sha256": _sha(events_raw), "elapsed_high_water": elapsed}


def status(config_path: str | os.PathLike[str] = _CONFIG_PATH) -> dict:
    """Read-only status with integrity checks for every terminal artifact already present."""
    root = _root()
    registration = _registration(root)
    supplied = Path(config_path)
    if not supplied.is_absolute():
        supplied = root / supplied
    if supplied.resolve(strict=True) != Path(registration["config_path"]).resolve(strict=True):
        raise ValueError("status accepts only the registered fixed pilot configuration")
    matrix_root = root / PurePosixPath(registration["config"]["output_dir"])
    if not matrix_root.exists():
        return {"state": "not_started", "matrix_root": str(matrix_root)}
    if not _status_path(matrix_root).is_file():
        return {"state": "initializing", "matrix_root": str(matrix_root)}
    result = _read_status(matrix_root)
    result["supervisor_alive"] = _pid_live(result.get("supervisor"))
    result["worker_alive"] = _pid_live(result.get("worker"))
    result["worker_launcher_alive"] = _pid_live(result.get("worker_launcher"))
    # Status does not parse protected examples or ask the controller to audit semantics.
    terminal_records = []
    for candidate in (matrix_root, *(matrix_root / "arms" / arm for arm in _ARM_ORDER)):
        if candidate.is_dir():
            integrity = _integrity_of_record(candidate)
            terminal_records.append({"path": str(candidate), "integrity": integrity})
        if candidate.parent == matrix_root / "arms" and candidate.is_dir():
            if integrity.get("terminal_verified") and integrity.get("status") == "completed":
                physical_audit = _read_json_path(candidate / "physical-audit.json")
                _verify_worker_log_receipts(candidate, physical_audit)
            for segment in _segments(candidate):
                terminal_records.append({"path": str(segment), "integrity": _integrity_of_record(segment)})
    result["verified_terminal_records"] = terminal_records
    result["arm_progress"] = {arm: _artifact_progress(matrix_root / "arms" / arm)
        for arm in _ARM_ORDER if (matrix_root / "arms" / arm).is_dir()}
    return result


def _artifact_progress(arm_root: Path) -> dict:
    """Project committed progress and physical event counts without semantic reads."""
    progress = {"optimizer_step": 0, "phase": None}
    physical = {"evaluation_cases_started": 0, "teacher_forced_completions": 0,
        "predictions_completed": 0, "updates_started": 0, "updates_completed": 0,
        "bank_preparations_started": 0, "bank_preparations_completed": 0}
    stage_counters: dict[str, dict] = {}
    stream_refs = []
    for segment in _segments(arm_root):
        path = segment / "events.jsonl"
        if not path.is_file() and not (segment / "manifest.initial.json").is_file():
            continue
        raw = path.read_bytes()
        from dwr.experiments.records import _validate_events
        _validate_events(raw)
        stream_refs.append({"path": path.relative_to(arm_root).as_posix(),
            "byte_length": len(raw), "sha256": _sha(raw)})
        for row in _event_rows(path):
            kind, payload = row["kind"], row["payload"]
            if kind == "evaluation_case_started":
                physical["evaluation_cases_started"] += 1
            elif kind == "teacher_forced_completed":
                physical["teacher_forced_completions"] += 1
            elif kind == "prediction_completed":
                physical["predictions_completed"] += 1
            elif kind == "update_started":
                physical["updates_started"] += 1
            elif kind == "update_completed":
                physical["updates_completed"] += 1
            elif kind == "bank_started":
                physical["bank_preparations_started"] += 1
            elif kind == "bank_completed":
                physical["bank_preparations_completed"] += 1
            elif kind == "stage_committed":
                stage = payload.get("stage")
                if isinstance(stage, str):
                    slot = stage_counters.setdefault(stage, {"commits": 0, "counts": {}})
                    slot["commits"] += 1
                    counts = payload.get("counts")
                    if type(counts) is dict:
                        for key, value in counts.items():
                            if type(value) is int and value >= 0:
                                slot["counts"][key] = slot["counts"].get(key, 0) + value
            elif kind == "recovery_committed":
                committed = payload.get("progress")
                if isinstance(committed, Mapping) and type(committed.get("optimizer_step")) is int:
                    if committed["optimizer_step"] >= progress["optimizer_step"]:
                        progress = {"optimizer_step": committed["optimizer_step"],
                            "phase": payload.get("phase")}
    cursor = None
    cursor_path = arm_root / "recovery.latest.json"
    if cursor_path.is_file() and not cursor_path.is_symlink():
        raw = cursor_path.read_bytes()
        value = strict_json_loads(raw.decode("utf-8"))
        if type(value) is not dict or raw != _canonical(value) + b"\n":
            raise ValueError("read-only status cursor is not canonical JSON")
        cursor = {"path": cursor_path.relative_to(arm_root).as_posix(), "sha256": _sha(raw)}
    deadlines = None
    initial_path = arm_root / "manifest.initial.json"
    if initial_path.is_file():
        initial = _read_json_path(initial_path)
        deadlines = initial.get("settings", {}).get("original_deadlines")
    return {"progress": progress, "stage_counters": stage_counters,
        "physical_event_counters": physical, "event_streams": stream_refs,
        "cursor": cursor, "effective_original_deadlines": deadlines}


def _create_matrix_record(root: Path, registration: Mapping[str, object], physical: Mapping[str, object],
                          runtime: Mapping[str, object], matrix_root: Path) -> RunRecord:
    config = registration["config"]
    record = _new_record(matrix_root, registration, physical, runtime, record_kind="matrix",
        original_deadlines={})
    clock = _clock_from_record(matrix_root)
    deadlines = _derive_deadlines(config["limits"], {"matrix": clock,
        "arm": {"started_epoch": clock["started_epoch"], "last_utc_epoch": clock["last_utc_epoch"],
                "elapsed_high_water_seconds": clock["elapsed_high_water_seconds"]}})["epochs"]
    _set_initial_deadlines(record, {key: deadlines[key] for key in ("matrix_deadline_epoch", "matrix_watchdog_epoch")})
    return record, clock, deadlines


def _create_arm_record(registration: Mapping[str, object], physical: Mapping[str, object],
                       runtime: Mapping[str, object], arm_root: Path, arm_id: str, arm_index: int,
                       deadlines: Mapping[str, object]) -> tuple[RunRecord, dict]:
    record = _new_record(arm_root, registration, physical, runtime, record_kind="arm",
        original_deadlines={}, arm_id=arm_id, arm_index=arm_index)
    clock = _clock_from_record(arm_root)
    exact_deadlines = _arm_deadlines(registration["config"]["limits"],
        _clock_snapshot(Path(record.output_dir).parents[1]), clock)
    _set_initial_deadlines(record, exact_deadlines)
    return record, clock


def _set_initial_deadlines(record: RunRecord, deadlines: Mapping[str, object]) -> None:
    """Finalize anchor-derived settings before the initial record is issued to a worker."""
    path = record.output_dir / "manifest.initial.json"
    initial = strict_json_loads(path.read_text(encoding="utf-8"))
    settings = initial.get("settings")
    if type(settings) is not dict or settings.get("original_deadlines") != {}:
        raise ValueError("only a fresh unpublished pilot RunRecord can receive original deadlines")
    settings["original_deadlines"] = dict(deadlines)
    raw = _canonical(initial) + b"\n"
    with path.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    record._initial_sha256 = _sha(raw)


def _arm_deadlines(limits: Mapping[str, object], matrix_clock: Mapping[str, object],
                   arm_clock: Mapping[str, object]) -> dict:
    matrix_deadline = matrix_clock["started_epoch"] + limits["total_wall_seconds"]
    matrix_watchdog = matrix_clock["started_epoch"] + limits["matrix_external_watchdog_seconds"]
    effective = min(arm_clock["started_epoch"] + limits["arm_wall_seconds"], matrix_deadline)
    return {"matrix_deadline_epoch": matrix_deadline,
        "matrix_watchdog_epoch": matrix_watchdog,
        "effective_arm_deadline_epoch": effective,
        "arm_work_deadline_epoch": effective - limits["arm_finalization_reserve_seconds"],
        "arm_watchdog_epoch": min(arm_clock["started_epoch"] + limits["arm_external_watchdog_seconds"], matrix_watchdog)}


def _segments(arm_root: Path) -> list[Path]:
    return sorted((path for path in arm_root.glob("segment-[0-9][0-9][0-9]") if path.is_dir()),
        key=lambda path: path.name)


def _arm_physical_audit(authorization: Mapping[str, object], arm_audit: Mapping[str, object]) -> dict:
    ledger = _physical_ledger(authorization)
    pairing = arm_audit.get("pairing")
    progress = arm_audit.get("progress")
    if not isinstance(pairing, list) or not isinstance(progress, Mapping):
        raise ValueError("completed arm audit lacks committed update pairing")
    committed = [row for row in pairing if type(row) is dict and set(row) ==
        {"optimizer_step", "example_id", "batch_sha256", "noise_sha256"}]
    if len(committed) != len(pairing) or len(committed) != progress.get("optimizer_step"):
        raise ValueError("logical committed update pairing differs from progress")
    if ledger["physical_updates_completed"] < len(committed):
        raise ValueError("physical update ledger has fewer completed updates than committed logical steps")
    if ledger["physical_updates_started"] > 514 or ledger["updates_in_flight_or_unknown"] != 0:
        raise ValueError("physical update ledger is incomplete or exceeds registered cap")
    expected_steps = authorization.get("expected_steps", 512)
    if type(expected_steps) is not int or expected_steps <= 0 or len(committed) != expected_steps:
        raise ValueError("terminal arm does not contain the registered committed paired update count")
    return ledger


def _supervise(config_path: str) -> None:
    return _supervise_core(config_path)


def _supervise_core(config_path: str, *, factories: Mapping[str, object] | None = None) -> None:
    """Own matrix records and dispatch the three arms in immutable registered order."""
    if factories and callable(factories.get("record_identities")):
        from unittest.mock import patch
        with patch("dwr.experiments.source_pilot_job._record_identities",
                   side_effect=factories["record_identities"]):
            nested = {key: value for key, value in factories.items() if key != "record_identities"}
            return _supervise_core(config_path, factories=nested)
    from dwr.experiments.source_overfit_job import _ByteLock
    root = _root()
    registration = factories.get("registration") if factories and factories.get("registration") is not None else _registration(root)
    matrix_root = root / PurePosixPath(registration["config"]["output_dir"])
    matrix_lock = _ByteLock(_matrix_lock_path(root), 0, wait=True)
    try:
        physical_factory = factories.get("physical") if factories else None
        physical = physical_factory() if callable(physical_factory) else (
            physical_factory if isinstance(physical_factory, Mapping) else
            __import__("dwr.experiments.gpu_feasibility", fromlist=["validate_clean_source"]).validate_clean_source(
                root, registration["config"]["source"]["ancestry_minimum"]))
        runtime_factory = factories.get("runtime") if factories else None
        runtime = runtime_factory() if callable(runtime_factory) else (
            runtime_factory if isinstance(runtime_factory, Mapping) else _runtime_receipt())
        if not (matrix_root / "manifest.initial.json").exists():
            # `start` persists a small attachment marker before launching this process.
            # Remove only that exact unissued marker so RunRecord can create its root.
            if set(path.name for path in matrix_root.iterdir()) == {"status.json"}:
                (matrix_root / "status.json").unlink()
                matrix_root.rmdir()
            matrix_record, matrix_clock, matrix_deadlines = _create_matrix_record(root, registration, physical, runtime, matrix_root)
            matrix_record.append_event("matrix_supervisor_started", {"supervisor": _reference(os.getpid()),
                "config_sha256": registration["config_sha256"], "source_commit": physical["commit"]})
            _write_status(matrix_root, state="running", supervisor=_reference(os.getpid()), worker=None,
                config_sha256=registration["config_sha256"], source_commit=physical["commit"],
                matrix_started_utc=matrix_clock["started_utc"], completed_arms=[], current_arm=None,
                matrix_deadline_epoch=matrix_deadlines["matrix_deadline_epoch"],
                matrix_watchdog_epoch=matrix_deadlines["matrix_watchdog_epoch"])
            watchdog_factory = factories.get("watchdog_factory") if factories else None
            (watchdog_factory or _launch_watchdog)(matrix_root)
        else:
            matrix_record, matrix_clock, matrix_deadlines = _reopen_matrix(root, registration, physical, runtime, matrix_root)
            matrix_record.append_event("matrix_supervisor_attached", {
                "supervisor": _reference(os.getpid()), "matrix_started_utc": matrix_clock["started_utc"]})
            _write_status(matrix_root, state="recovering", supervisor=_reference(os.getpid()))
            if not _pid_live(_read_status(matrix_root).get("watchdog")):
                watchdog_factory = factories.get("watchdog_factory") if factories else None
                (watchdog_factory or _launch_watchdog)(matrix_root)
        audits = {}
        if factories and callable(factories.get("read_validation")):
            read_validation = factories["read_validation"]
        else:
            from dwr.experiments.source_pilot_registration import read_pilot_examples
            read_validation = lambda root, registration, split: read_pilot_examples(root, registration, split=split)
        if factories and callable(factories.get("audit")):
            audit_completed_arm = factories["audit"]
        else:
            from dwr.experiments.source_pilot import audit_completed_arm
        if factories and callable(factories.get("matrix_decision")):
            matrix_decision = factories["matrix_decision"]
        else:
            from dwr.experiments.source_pilot import matrix_decision
        validation_examples = metadata = None
        for arm_index, arm in enumerate(registration["config"]["arms"]):
            arm_id = arm["arm_id"]
            arm_root = matrix_root / "arms" / arm_id
            if (arm_root / "manifest.json").exists():
                terminal = _verify_record_if_terminal(arm_root)
                if terminal["status"] != "completed":
                    raise ValueError("previous arm has a noncompleted terminal record; no retries are allowed")
                if not _arm_audit_path(arm_root).is_file():
                    raise ValueError("completed arm lacks its independently audited immutable result")
                audits[arm_id] = _read_json_path(_arm_audit_path(arm_root))
                continue
            (matrix_root / "arms").mkdir(parents=True, exist_ok=True)
            if not (arm_root / "manifest.initial.json").exists():
                _check_interarm_overhead(matrix_root, registration, matrix_record)
                arm_record, arm_clock = _create_arm_record(registration, physical, runtime, arm_root,
                    arm_id, arm_index, {})
                deadlines = _arm_deadlines(registration["config"]["limits"], matrix_clock, arm_clock)
                arm_record.append_event("arm_supervisor_started", {"arm_id": arm_id,
                    "supervisor": _reference(os.getpid()), "original_deadlines": deadlines})
            else:
                arm_clock = _clock_snapshot(arm_root)
                deadlines = _arm_deadlines(registration["config"]["limits"], matrix_clock, arm_clock)
                arm_record, arm_clock = _reopen_arm(registration, physical, runtime, arm_root, arm_id, arm_index, deadlines)
            _write_status(matrix_root, state="running", supervisor=_reference(os.getpid()),
                current_arm=arm_id, arm_started_utc=arm_clock["started_utc"])
            command_factory = factories.get("worker_command") if factories else None
            worker_command = None if not callable(command_factory) else \
                (lambda descriptor_path, descriptor_sha, arm_id=arm_id:
                    command_factory(arm_id, descriptor_path, descriptor_sha))
            outcome = _run_arm(registration, physical, runtime, matrix_root, arm_root,
                arm_record, arm_clock, matrix_clock, arm_index, worker_command=worker_command)
            if outcome.get("status") != "completed":
                raise RuntimeError(f"{arm_id} worker did not complete its registered segment")
            if validation_examples is None:
                # Require a durable worker before-access event before the supervisor reads
                # the same fixed semantic validation cohort for independent arm auditing.
                if not any(_has_authorized_validation_access(path) for path in _segments(arm_root)):
                    raise ValueError("semantic validation access has no durable before-access record")
                _check_arm_seal(matrix_root, arm_root, "before_semantic_arm_audit")
                ordered_ids = [row["example_id"] for row in registration["cohort"]["selected"]]
                matrix_record.append_event("source_examples_access_authorized", {
                    "split": "source_validation", "supervisor": _reference(os.getpid()),
                    "config_sha256": registration["config_sha256"],
                    "cohort_sha256": registration["cohort_sha256"],
                    "ordered_ids_sha256": _sha(_canonical(ordered_ids)),
                    "count": len(ordered_ids), "purpose": "independent_arm_audit"})
                validation_examples, metadata = read_validation(root, registration, "source_validation")
            arm_audit = audit_completed_arm(root, registration, arm_root, arm_id=arm_id,
                source_commit=physical["commit"], validation_examples=tuple(validation_examples), metadata=metadata)
            physical_audit = _reconcile_arm_physical(registration, arm_root, arm_id, arm_audit)
            audit_path = _arm_audit_path(arm_root)
            _exclusive_json(audit_path, arm_audit)
            physical_path = arm_root / "physical-audit.json"
            _exclusive_json(physical_path, physical_audit)
            arm_record.append_event("arm_audited", {"audit_sha256": _sha(audit_path.read_bytes()),
                "physical_audit_sha256": _sha(physical_path.read_bytes())})
            terminal = arm_record.finish("completed", counts={"committed_steps": arm_audit["progress"]["optimizer_step"]},
                artifacts=[audit_path.relative_to(arm_root).as_posix(), physical_path.relative_to(arm_root).as_posix()],
                budget_check=lambda stage: _check_arm_seal(matrix_root, arm_root, stage))
            audits[arm_id] = arm_audit
            _write_status(matrix_root, completed_arms=list(audits), current_arm=None)
        result = matrix_decision(registration["config"], audits)
        _check_interarm_overhead(matrix_root, registration, matrix_record)
        result_path = matrix_root / "matrix-decision.json"
        _exclusive_json(result_path, result)
        matrix_record.append_event("matrix_decision_recorded", {"decision_sha256": _sha(result_path.read_bytes()),
            "status": result["status"]})
        scientific_no_go = result["status"].startswith("NO_GO")
        matrix_record.finish("completed", counts={"arms_completed": 3},
            artifacts=["matrix-decision.json"], budget_check=lambda stage: _check_matrix_seal(matrix_root, stage))
        _write_status(matrix_root, state="scientific_no_go" if scientific_no_go else "completed",
            current_arm=None, completed_arms=list(audits), decision=result)
    except BaseException as exc:
        try:
            _write_status(matrix_root, state="operational_failure", supervisor=_reference(os.getpid()),
                error=f"{type(exc).__name__}: {exc}")
        finally:
            raise
    finally:
        matrix_lock.close()


def _arm_audit_path(arm_root: Path) -> Path:
    return arm_root / "arm-audit.json"


def _read_json_path(path: Path) -> dict:
    raw = path.read_bytes()
    value = strict_json_loads(raw.decode("utf-8"))
    if type(value) is not dict or raw != _canonical(value) + b"\n":
        raise ValueError(f"{path.name} is not canonical stable JSON")
    return value


def _authorization_from_issued_receipt(segment: Path, receipt: Mapping[str, object],
                                       registration: Mapping[str, object]) -> dict:
    descriptor_path = segment / "worker-descriptor.json"
    descriptor_sha = receipt.get("descriptor_sha256")
    descriptor_file, descriptor = _read_descriptor_file(descriptor_path, descriptor_sha)
    if descriptor_file.parent != segment.resolve(strict=True):
        raise ValueError("surviving worker descriptor escaped its issued segment")
    if receipt.get("nonce") != descriptor.get("nonce") or receipt.get("arm_id") != descriptor.get("arm_id") or \
            receipt.get("segment_index") != descriptor.get("segment_index"):
        raise ValueError("surviving worker receipt does not match immutable issued descriptor")
    matrix_root = Path(descriptor["matrix_root"]).resolve(strict=True)
    arm_root = Path(descriptor["arm_root"]).resolve(strict=True)
    if _sha((matrix_root / "manifest.initial.json").read_bytes()) != descriptor["matrix_initial_manifest_sha256"] or \
            _sha((arm_root / "manifest.initial.json").read_bytes()) != descriptor["arm_initial_manifest_sha256"] or \
            _sha((segment / "manifest.initial.json").read_bytes()) != descriptor["segment_initial_manifest_sha256"]:
        raise ValueError("surviving worker original RunRecord initial bytes changed")
    _verify_event_prefix(matrix_root / "events.jsonl", descriptor["matrix_event_prefix"])
    _verify_event_prefix(arm_root / "events.jsonl", descriptor["arm_event_prefix"])
    _verify_event_prefix(segment / "events.jsonl", descriptor["segment_event_prefix"])
    issue_rows = [row for row in _event_rows(arm_root / "events.jsonl") if row["kind"] == "worker_descriptor_issued" and
        row["payload"].get("segment_index") == descriptor["segment_index"]]
    if len(issue_rows) != 1 or issue_rows[0]["payload"].get("descriptor_sha256") != descriptor_sha or \
            issue_rows[0]["payload"].get("nonce") != descriptor["nonce"]:
        raise ValueError("surviving worker descriptor is not the uniquely issued arm segment")
    launch_rows = [row for row in _event_rows(arm_root / "events.jsonl") if
        row["kind"] == "worker_launch_command_issued" and
        row["payload"].get("segment_index") == descriptor["segment_index"]]
    if len(launch_rows) != 1 or launch_rows[0]["payload"].get("descriptor_sha256") != descriptor_sha:
        raise ValueError("worker launch command is not uniquely bound to its issued descriptor")
    launch_command = launch_rows[0]["payload"].get("command")
    if type(launch_command) is not list or not launch_command or \
            any(not isinstance(item, str) or not item for item in launch_command) or \
            launch_rows[0]["payload"].get("command_sha256") != _sha(_canonical(launch_command)):
        raise ValueError("issued worker launch command receipt is malformed")
    receipt_event = [row for row in _event_rows(segment / "events.jsonl") if row["kind"] == "worker_receipt_published"]
    if len(receipt_event) != 1 or receipt_event[0]["payload"].get("descriptor_sha256") != descriptor_sha or \
            receipt_event[0]["payload"].get("nonce") != descriptor["nonce"]:
        raise ValueError("surviving worker process receipt was not durably published in its segment")
    clocks = {"matrix": _validate_issued_clock(descriptor["matrix_clock"], matrix_root),
        "arm": _validate_issued_clock(descriptor["arm_clock"], arm_root)}
    deadlines = _derive_deadlines(registration["config"]["limits"], clocks)
    if descriptor["deadlines"] != deadlines["epochs"]:
        raise ValueError("issued worker original deadline epochs changed")
    authorization = {"descriptor": descriptor, "descriptor_sha256": descriptor_sha,
        "descriptor_path": str(descriptor_file), "nonce": descriptor["nonce"],
        "matrix_root": descriptor["matrix_root"], "arm_root": descriptor["arm_root"],
        "segment_dir": str(segment.resolve(strict=True)), "arm_id": descriptor["arm_id"],
        "segment_index": descriptor["segment_index"],
        "bindings": {"source": {"commit": descriptor["source_commit"]},
            "runtime": {"receipt_sha256": descriptor["runtime_receipt_sha256"]}},
        "registration": {"config_sha256": descriptor["config_sha256"],
            "cohort_sha256": descriptor["cohort_sha256"], "config": registration["config"]},
        "expected_launcher": descriptor["launcher"]}
    _verified_launch_receipt(authorization)
    return authorization


_LOSS_EVENT = "worker_loss_finalized"
_LOSS_KEYS = {"descriptor_sha256", "worker_receipt_sha256", "worker_identity",
    "launcher_identity", "source_commit", "runtime_receipt_sha256", "worker_lock_released",
    "postloss_observation"}


def verify_worker_loss_record(arm_root: Path, segment: Path,
                              registration: Mapping[str, object], *, source_commit: str) -> dict | None:
    """Verify historical loss-only provenance without depending on PID reuse state."""
    segment = Path(segment).resolve(strict=True)
    arm_root = Path(arm_root).resolve(strict=True)
    events = _event_rows(segment / "events.jsonl")
    matches = [row for row in events if row["kind"] == _LOSS_EVENT]
    if not matches:
        return None
    terminal = _verify_record_if_terminal(segment)
    if terminal is None or terminal.get("status") != "interrupted":
        raise ValueError("worker loss marker must belong to a terminal interrupted segment")
    if len(matches) != 1:
        raise ValueError("worker loss marker is duplicated")
    payload = matches[0]["payload"]
    if type(payload) is not dict or set(payload) != _LOSS_KEYS or \
            payload.get("worker_lock_released") is not True or \
            payload.get("postloss_observation") != {"available": False,
                "reason": "process_exited_before_observation"}:
        raise ValueError("worker loss marker has unknown fields or claims unavailable observations")
    receipt_path = segment / "worker-receipt.json"
    receipt = _read_json_path(receipt_path)
    auth = _authorization_from_issued_receipt(segment, receipt, registration)
    descriptor = auth["descriptor"]
    expected_launcher = _worker_launcher_for_parent(auth, receipt.get("parent", {}), live=False,
        worker=receipt.get("worker", {}))
    if receipt.get("launcher") != expected_launcher or receipt.get("parent") != expected_launcher or \
            receipt.get("launcher", {}).get("pid") == receipt.get("worker", {}).get("pid"):
        raise ValueError("worker loss receipt parent/launcher chain differs from its issued launch receipt")
    if payload["descriptor_sha256"] != auth["descriptor_sha256"] or \
            payload["worker_receipt_sha256"] != _sha(receipt_path.read_bytes()) or \
            payload["worker_identity"] != {key: receipt["worker"][key]
                for key in ("pid", "creation_time", "image")} or \
            payload["launcher_identity"] != receipt["launcher"] or \
            payload["source_commit"] != source_commit or \
            payload["runtime_receipt_sha256"] != auth["bindings"]["runtime"]["receipt_sha256"]:
        raise ValueError("worker loss marker does not join its immutable issued receipt")
    if receipt.get("source_commit") != source_commit or \
            receipt.get("runtime_sha256") != auth["bindings"]["runtime"]["receipt_sha256"]:
        raise ValueError("worker loss receipt source/runtime binding differs")
    if _known_worker_fault(segment):
        raise ValueError("worker loss segment contains a known or unclassified terminal fault")
    return dict(payload)


def _finalize_worker_loss(arm_root: Path, segment: Path, receipt: Mapping[str, object],
                          registration: Mapping[str, object], source_commit: str) -> dict:
    """Seal an unsealed segment only after its worker and original launcher are gone."""
    from dwr.experiments.source_overfit_job import _ByteLock, _pid_matches
    receipt_path = segment / "worker-receipt.json"
    authorization = _authorization_from_issued_receipt(segment, receipt, registration)
    _check_hard_loss_process_state(authorization, receipt, pid_matches=_pid_matches)
    if receipt.get("source_commit") != source_commit or _known_worker_fault(segment):
        raise ValueError("worker loss source binding or fault scan forbids recovery")
    try:
        lock = _ByteLock(_worker_lock_path(arm_root), 1, wait=False)
    except RuntimeError as exc:
        raise ValueError("worker OS ownership lock remains held after process loss") from exc
    lock.close()
    loss = {"descriptor_sha256": authorization["descriptor_sha256"],
        "worker_receipt_sha256": _sha(receipt_path.read_bytes()),
        "worker_identity": {key: receipt["worker"][key] for key in ("pid", "creation_time", "image")},
        "launcher_identity": dict(receipt["launcher"]), "source_commit": source_commit,
        "runtime_receipt_sha256": authorization["bindings"]["runtime"]["receipt_sha256"],
        "worker_lock_released": True,
        "postloss_observation": {"available": False, "reason": "process_exited_before_observation"}}
    record = open_segment_record(authorization)
    record.append_event(_LOSS_EVENT, loss)
    from dwr.experiments.source_pilot import _segment_artifacts
    record.finish("interrupted", counts={}, artifacts=_segment_artifacts(segment),
        error="supervisor-observed hard worker loss; post-loss process observation unavailable")
    return loss


def _check_hard_loss_process_state(authorization: Mapping[str, object], receipt: Mapping[str, object], *,
                                   pid_matches=None) -> None:
    """Require the worker gone; a live original issuing supervisor is a valid recovery owner."""
    if pid_matches is None:
        from dwr.experiments.source_overfit_job import _pid_matches as pid_matches
    worker = receipt["worker"]
    if pid_matches(worker["pid"], worker["creation_time"]):
        raise ValueError("hard-loss recovery requires the original worker process to be dead")
    launcher = receipt["launcher"]
    if launcher != authorization["descriptor"]["supervisor"] and \
            pid_matches(launcher["pid"], launcher["creation_time"]):
        raise ValueError("hard-loss recovery requires a distinct worker-owned launcher to be dead")


def _has_authorized_validation_access(segment: Path) -> bool:
    events = _event_rows(segment / "events.jsonl")
    return any(row["kind"] == "source_examples_access_authorized" and
        row["payload"].get("split") == "source_validation" for row in events)


def _reconcile_arm_physical(registration: Mapping[str, object], arm_root: Path, arm_id: str,
                            audit: Mapping[str, object]) -> dict:
    segments = _segments(arm_root)
    if not segments or [path.name for path in segments] != [f"segment-{index:03d}" for index in range(len(segments))]:
        raise ValueError("arm physical ledger has a missing or noncontiguous segment")
    authorization = {"arm_root": str(arm_root), "segment_index": len(segments) - 1,
        "arm_id": arm_id, "config": registration["config"]}
    ledger = _physical_ledger(authorization)
    pairing = audit.get("pairing")
    progress = audit.get("progress")
    if not isinstance(pairing, list) or not isinstance(progress, Mapping):
        raise ValueError("arm audit lacks committed pairing or progress")
    expected_steps = registration["config"]["training"]["optimizer_steps"]
    if type(expected_steps) is not int or expected_steps <= 0 or len(pairing) != expected_steps or \
            progress.get("optimizer_step") != expected_steps:
        raise ValueError("arm must commit exactly the registered logical optimizer steps")
    committed = _event_committed_pairing(segments, expected_steps)
    if committed != pairing:
        raise ValueError("controller pairing differs from the full durable direct-event stream")
    update_cap = registration["config"]["limits"]["maximum_physical_updates_per_arm"]
    if ledger["physical_updates_completed"] < expected_steps or ledger["physical_updates_started"] > update_cap:
        raise ValueError("full physical update ledger is incomplete or exceeds the registered cap")
    limits = registration["config"]["limits"]
    if ledger["prediction_attempts_started"] > limits["per_arm_physical_predictions"][arm_id] or \
            ledger["greedy_decoder_calls_conservative"] > limits["per_arm_greedy_decoder_calls"][arm_id] or \
            ledger["teacher_forced_ce_started"] > limits["per_arm_teacher_forced_ce_answer_forwards"][arm_id]:
        raise ValueError("physical evaluation ledger exceeds its registered arm cap")
    refs = audit.get("evidence")
    if not isinstance(refs, Mapping):
        raise ValueError("arm audit lacks evidence references for physical reconciliation")
    expected_manifests, expected_events = [], []
    for segment in segments:
        terminal = (segment / "manifest.json").read_bytes()
        event_raw = (segment / "events.jsonl").read_bytes()
        expected_manifests.append({"path": segment.relative_to(arm_root).as_posix() + "/manifest.json", "sha256": _sha(terminal)})
        expected_events.append({"path": segment.relative_to(arm_root).as_posix() + "/events.jsonl", "sha256": _sha(event_raw)})
    if refs.get("segment_manifests") != expected_manifests or refs.get("event_streams") != expected_events:
        raise ValueError("arm semantic audit and job physical ledger refer to different segment evidence")
    ownership = [_verify_segment_ownership(segment, arm_root, registration, arm_id) for segment in segments]
    worker_logs = []
    for segment in segments:
        for path in _worker_log_paths(arm_root, segment):
            if not path.is_file() or path.is_symlink():
                raise ValueError("completed worker segment is missing its regular stdout/stderr log")
            worker_logs.append({"path": path.relative_to(arm_root).as_posix(), "sha256": _sha(path.read_bytes())})
    matrix_root = arm_root.parents[1]
    matrix_clock = _advance_clock(matrix_root)
    arm_clock = _advance_clock(arm_root)
    limits = registration["config"]["limits"]
    if matrix_clock["elapsed_high_water_seconds"] > limits["total_wall_seconds"] or \
            arm_clock["elapsed_high_water_seconds"] > limits["arm_wall_seconds"]:
        raise TimeoutError("original matrix or arm wall budget exceeded before physical audit seal")
    return {"schema_version": "dwr-source-pilot-physical-audit-v1", "arm_id": arm_id,
        "segment_manifests": expected_manifests, "event_streams": expected_events,
        "ledger": ledger, "committed_optimizer_steps": len(pairing),
        "ownership_receipts": ownership, "worker_logs": worker_logs,
        "matrix_clock_high_water": matrix_clock,
        "arm_clock_high_water": arm_clock,
        "original_deadlines": strict_json_loads((arm_root / "manifest.initial.json").read_text(encoding="utf-8"))["settings"]["original_deadlines"]}


def _worker_log_paths(arm_root: Path, segment_dir: Path) -> tuple[Path, Path]:
    """Resolve the stable arm-owned stdout/stderr pair for one worker segment."""
    arm = Path(arm_root).resolve(strict=True)
    segment = Path(segment_dir).resolve(strict=True)
    if segment.parent != arm or not re.fullmatch(r"segment-[0-9]{3}", segment.name):
        raise ValueError("worker log owner must be a direct registered arm segment")
    directory = arm / "worker-logs"
    _reject_links(directory)
    return directory / f"{segment.name}.stdout.log", directory / f"{segment.name}.stderr.log"


def _verify_worker_log_receipts(arm_root: Path, physical_audit: Mapping[str, object]) -> None:
    """Verify post-exit worker log bytes against the sealed physical audit."""
    if not isinstance(physical_audit, Mapping) or physical_audit.get("schema_version") != "dwr-source-pilot-physical-audit-v1":
        raise ValueError("physical audit is malformed while checking worker log receipts")
    arm = Path(arm_root).resolve(strict=True)
    segments = _segments(arm)
    expected = []
    for segment in segments:
        for path in _worker_log_paths(arm, segment):
            _reject_links(path)
            if not path.is_file():
                raise ValueError("sealed worker log receipt names a missing regular log")
            expected.append({"path": path.relative_to(arm).as_posix(), "sha256": _sha(path.read_bytes())})
    if physical_audit.get("worker_logs") != expected:
        raise ValueError("worker log receipt identity differs from current final bytes")


def _verify_segment_ownership(segment: Path, arm_root: Path, registration: Mapping[str, object], arm_id: str) -> dict:
    descriptor_path = segment / "worker-descriptor.json"
    receipt_path = segment / "worker-receipt.json"
    receipt_event_rows = [row for row in _event_rows(segment / "events.jsonl")
        if row["kind"] == "worker_receipt_published"]
    if len(receipt_event_rows) != 1:
        raise ValueError("segment lacks exactly one durable worker ownership event")
    receipt = _read_json_path(receipt_path)
    digest = receipt_event_rows[0]["payload"].get("descriptor_sha256")
    descriptor_file, descriptor = _read_descriptor_file(descriptor_path, digest)
    issue_events = [row for row in _event_rows(arm_root / "events.jsonl")
        if row["kind"] == "worker_descriptor_issued" and
        row["payload"].get("segment_index") == descriptor.get("segment_index")]
    if len(issue_events) != 1 or issue_events[0]["payload"].get("descriptor_sha256") != digest or \
            issue_events[0]["payload"].get("nonce") != descriptor.get("nonce"):
        raise ValueError("worker ownership descriptor differs from its unique arm issue event")
    receipt_payload = receipt_event_rows[0]["payload"]
    ref = receipt_payload.get("receipt")
    if set(receipt_payload) != {"receipt", "descriptor_sha256", "nonce"} or \
            receipt_payload.get("descriptor_sha256") != digest or receipt_payload.get("nonce") != descriptor.get("nonce") or \
            type(ref) is not dict or set(ref) != {"path", "sha256"} or \
            ref.get("path") != (segment.relative_to(arm_root).as_posix() + "/worker-receipt.json") or \
            ref.get("sha256") != _sha(receipt_path.read_bytes()):
        raise ValueError("worker receipt event does not bind the exact immutable receipt bytes")
    expected = {"schema_version": "dwr-source-pilot-worker-receipt-v1", "descriptor_sha256": digest,
        "nonce": descriptor["nonce"], "matrix_root": descriptor["matrix_root"],
        "arm_root": descriptor["arm_root"], "segment_dir": descriptor["segment_dir"],
        "arm_id": arm_id, "segment_index": descriptor["segment_index"],
        "source_commit": descriptor["source_commit"], "config_sha256": registration["config_sha256"],
        "runtime_sha256": descriptor["runtime_receipt_sha256"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("actual worker ownership receipt differs from the issued registration")
    for name in ("worker", "parent", "launcher"):
        value = receipt.get(name)
        expected_keys = {"pid", "creation_time", "image", "python_executable", "base_executable"} if name == "worker" else {"pid", "creation_time", "image"}
        if type(value) is not dict or set(value) != expected_keys or type(value.get("pid")) is not int or \
                not isinstance(value.get("creation_time"), str) or not isinstance(value.get("image"), str):
            raise ValueError("actual worker ownership receipt has malformed OS process identity")
    authorization = _authorization_from_issued_receipt(segment, receipt, registration)
    expected_launcher = _worker_launcher_for_parent(authorization, receipt["parent"], live=False,
        worker=receipt["worker"])
    if receipt["launcher"] != expected_launcher or receipt["parent"] != expected_launcher or \
            receipt["worker"]["pid"] == receipt["launcher"]["pid"]:
        raise ValueError("worker launcher/parent chain differs from the actual issued process")
    return {"segment_index": descriptor["segment_index"], "descriptor_sha256": digest,
        "receipt_sha256": _sha(receipt_path.read_bytes()), "worker": receipt["worker"],
        "parent": receipt["parent"], "launcher": receipt["launcher"]}


def _event_committed_pairing(segments: list[Path], expected_steps: int = 512) -> list[dict]:
    pairs = {}
    for segment in segments:
        pending, started = {}, {}
        for row in _event_rows(segment / "events.jsonl"):
            kind, payload = row["kind"], row["payload"]
            if kind == "update_started":
                physical_index = payload.get("physical_index")
                if type(physical_index) is not int or physical_index <= 0 or physical_index in started:
                    raise ValueError("physical update start has an invalid or duplicate physical index")
                started[physical_index] = payload
            if kind == "update_completed":
                step = payload.get("net_step")
                physical_index = payload.get("physical_index")
                start = started.pop(physical_index, None)
                if type(step) is not int or step <= 0 or physical_index is None or start is None:
                    raise ValueError("physical update completion has no matching started update")
                for field in ("example_id", "batch_sha256", "noise_sha256"):
                    if payload.get(field) != start.get(field):
                        raise ValueError("physical update completion differs from its started reservation")
                if step in pending:
                    raise ValueError("segment contains duplicate completion evidence for one logical step")
                pending[step] = payload
            elif kind == "recovery_committed":
                progress = payload.get("progress")
                step = progress.get("optimizer_step") if isinstance(progress, Mapping) else None
                if type(step) is not int or step <= 0:
                    continue
                if step in pairs:
                    if step in pending:
                        raise ValueError("duplicate committed physical update evidence")
                    continue
                update = pending.pop(step, None)
                if update is None:
                    raise ValueError("committed step lacks a durable update-completed event")
                pairs[step] = {"optimizer_step": step, **{key: update[key] for key in
                    ("example_id", "batch_sha256", "noise_sha256")}}
    if type(expected_steps) is not int or expected_steps <= 0 or sorted(pairs) != list(range(1, expected_steps + 1)):
        raise ValueError("physical event stream does not prove every unique registered committed update")
    return [pairs[index] for index in range(1, expected_steps + 1)]


def _check_interarm_overhead(matrix_root: Path, registration: Mapping[str, object], record: RunRecord) -> None:
    cap = registration["config"]["limits"]["maximum_interarm_overhead_seconds"]
    initial = strict_json_loads((matrix_root / "manifest.initial.json").read_text(encoding="utf-8"))
    matrix_start = _parse_utc(initial["started_utc"]).timestamp()
    elapsed = max(0.0, time.time() - matrix_start)
    arm_seconds = 0.0
    for arm in registration["config"]["arms"]:
        terminal = _verify_record_if_terminal(matrix_root / "arms" / arm["arm_id"])
        if terminal is not None and terminal["status"] == "completed":
            arm_seconds += float(terminal["elapsed_wall_seconds"])
    overhead = max(0.0, elapsed - arm_seconds)
    if overhead > cap:
        raise TimeoutError("registered matrix startup/interarm/final overhead cap exhausted")
    record.append_event("matrix_overhead_checked", {"elapsed_matrix_seconds": elapsed,
        "completed_arm_wall_seconds": arm_seconds, "overhead_seconds": overhead, "cap_seconds": cap})


def _run_arm(registration: Mapping[str, object], physical: Mapping[str, object], runtime: Mapping[str, object],
             matrix_root: Path, arm_root: Path, arm_record: RunRecord, arm_clock: Mapping[str, object],
             matrix_clock: Mapping[str, object], arm_index: int, *, worker_command=None) -> dict:
    arm = registration["config"]["arms"][arm_index]
    arm_id = arm["arm_id"]
    deadlines = _arm_deadlines(registration["config"]["limits"], matrix_clock, arm_clock)
    segments = _segments(arm_root)
    if segments:
        latest = segments[-1]
        terminal = _verify_record_if_terminal(latest)
        if terminal and terminal["status"] == "completed":
            return {"status": "completed", "segment_index": len(segments) - 1}
        if terminal and terminal["status"] == "failed":
            raise ValueError("failed worker segment is immutable and not eligible for retry")
        if terminal and terminal["status"] == "interrupted":
            if _known_worker_fault(latest, allow_keyboard_interrupt=True):
                raise RuntimeError("interrupted worker segment contains a known fault; recovery is forbidden")
            if len(segments) >= registration["config"]["limits"]["maximum_workers_per_arm"]:
                raise RuntimeError("registered worker recovery allowance is exhausted")
            arm_record.append_event("worker_interruption_recovered", {"segment_index": len(segments)-1,
                "receipt_sha256": _sha((latest / "worker-receipt.json").read_bytes()),
                "fault_scan": "no known fault marker", "sealed_status": "interrupted"})
        receipt_path = latest / "worker-receipt.json"
        if receipt_path.exists():
            if _known_worker_fault(latest,
                    allow_keyboard_interrupt=terminal is not None and terminal.get("status") == "interrupted"):
                raise RuntimeError("worker loss has known numerical, OOM, CUDA or configuration fault; retries are disabled")
            receipt = _read_json_path(receipt_path)
            descriptor = _read_descriptor_file(latest / "worker-descriptor.json",
                receipt.get("descriptor_sha256"))[1]
            issuer = descriptor["launcher"]
            live_worker = _pid_live(receipt.get("worker"))
            live_launcher = receipt.get("launcher") != issuer and _pid_live(receipt.get("launcher"))
            if live_worker or live_launcher:
                # A restarted supervisor must attach to this exact process and never issue
                # another worker while either the worker or its bound intermediary remains live.
                authorization = _authorization_from_issued_receipt(latest, receipt, registration)
                if live_worker:
                    _verify_live_worker_receipt(receipt, authorization)
                code = _wait_existing_worker(matrix_root, arm_root, latest, receipt)
                terminal = _verify_record_if_terminal(latest)
                if terminal and terminal["status"] == "completed":
                    return {"status": "completed", "segment_index": len(segments) - 1}
                if _known_worker_fault(latest):
                    raise RuntimeError("attached worker exited with a known fault; retry is forbidden")
            if not terminal:
                if _known_worker_fault(latest):
                    raise RuntimeError("worker loss has known numerical, OOM, CUDA or configuration fault; retries are disabled")
                _finalize_worker_loss(arm_root, latest, receipt, registration, physical["commit"])
                terminal = _verify_record_if_terminal(latest)
                if terminal is None or terminal.get("status") != "interrupted":
                    raise ValueError("hard worker loss did not produce its loss-only interrupted record")
                if len(segments) >= registration["config"]["limits"]["maximum_workers_per_arm"]:
                    raise RuntimeError("registered worker recovery count is exhausted")
                arm_record.append_event("worker_interruption_recovered", {"segment_index": len(segments)-1,
                    "receipt_sha256": _sha(receipt_path.read_bytes()), "fault_scan": "no known fault marker"})
            elif terminal["status"] == "interrupted":
                if _known_worker_fault(latest, allow_keyboard_interrupt=True):
                    raise RuntimeError("attached worker exited with a known fault; retry is forbidden")
                if len(segments) >= registration["config"]["limits"]["maximum_workers_per_arm"]:
                    raise RuntimeError("attached worker exhausted the registered recovery allowance")
                arm_record.append_event("worker_interruption_recovered", {"segment_index": len(segments)-1,
                    "exit_code": 1, "receipt_sha256": _sha(receipt_path.read_bytes()),
                    "fault_scan": "no known fault marker"})
        elif not terminal:
            raise ValueError("issued worker segment lacks its actual process receipt")
    for segment_index in range(len(_segments(arm_root)), registration["config"]["limits"]["maximum_workers_per_arm"]):
        segment_dir = arm_root / f"segment-{segment_index:03d}"
        if segment_dir.exists():
            terminal = _verify_record_if_terminal(segment_dir)
            if terminal and terminal["status"] == "completed":
                return {"status": "completed", "segment_index": segment_index}
            if terminal and terminal["status"] == "failed":
                raise ValueError("failed terminal worker segment forbids recovery")
        resume = None
        cursor = arm_root / "recovery.latest.json"
        if cursor.is_file():
            resume = {"path": cursor.relative_to(arm_root).as_posix(), "sha256": _sha(cursor.read_bytes()),
                "source_commit": physical["commit"], "arm_id": arm_id}
        descriptor_path, descriptor_sha, descriptor = _issue_worker_descriptor(_root(), registration,
            physical, runtime, matrix_root, arm_root, segment_dir, arm_id=arm_id,
            arm_index=arm_index, segment_index=segment_index,
            matrix_initial_sha256=_sha((matrix_root / "manifest.initial.json").read_bytes()),
            arm_initial_sha256=_sha((arm_root / "manifest.initial.json").read_bytes()),
            resume_descriptor=resume)
        arm_record.append_event("worker_descriptor_issued", {"segment_index": segment_index,
            "descriptor_sha256": descriptor_sha, "nonce": descriptor["nonce"],
            "descriptor_path": descriptor_path.relative_to(matrix_root).as_posix()})
        command = ([sys.executable, "-m", "dwr.experiments.source_pilot", "worker",
            "--authorization", str(descriptor_path), "--sha256", descriptor_sha]
            if worker_command is None else list(worker_command(descriptor_path, descriptor_sha)))
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("private worker command factory returned an invalid process command")
        arm_record.append_event("worker_launch_command_issued", {"segment_index": segment_index,
            "descriptor_sha256": descriptor_sha, "command": command,
            "command_sha256": _sha(_canonical(command))})
        proc, stdout_path, stderr_path = _launch_worker(descriptor_path, descriptor_sha, matrix_root, arm_root,
            segment_dir, command=command, arm_record=arm_record)
        code = _wait_worker(matrix_root, arm_root, segment_dir, proc)
        terminal = _verify_record_if_terminal(segment_dir)
        if terminal is None:
            if _known_worker_fault(segment_dir, stderr_path, allow_keyboard_interrupt=True):
                return {"status": "known_fault", "exit_code": code}
            receipt_path = segment_dir / "worker-receipt.json"
            if not receipt_path.is_file():
                raise ValueError("worker exited without a terminal segment or durable process receipt")
            receipt = _read_json_path(receipt_path)
            _finalize_worker_loss(arm_root, segment_dir, receipt, registration, physical["commit"])
            terminal = _verify_record_if_terminal(segment_dir)
            if terminal is None or terminal.get("status") != "interrupted":
                raise ValueError("live-supervisor hard loss did not produce its loss-only interrupted record")
            if segment_index + 1 >= registration["config"]["limits"]["maximum_workers_per_arm"]:
                raise RuntimeError("worker interruption recovery allowance is exhausted")
            arm_record.append_event("worker_interruption_recovered", {"segment_index": segment_index,
                "exit_code": code, "receipt_sha256": _sha(receipt_path.read_bytes()),
                "fault_scan": "no known fault marker", "sealed_status": "interrupted"})
            continue
        if terminal["status"] == "completed":
            return {"status": "completed", "segment_index": segment_index}
        if terminal["status"] == "interrupted":
            if _known_worker_fault(segment_dir, stderr_path, allow_keyboard_interrupt=True):
                raise RuntimeError("interrupted worker segment contains a known fault; recovery is forbidden")
            if segment_index + 1 >= registration["config"]["limits"]["maximum_workers_per_arm"]:
                raise RuntimeError("worker interruption recovery allowance is exhausted")
            arm_record.append_event("worker_interruption_recovered", {"segment_index": segment_index,
                "exit_code": code, "fault_scan": "no known fault marker", "sealed_status": "interrupted"})
            continue
        raise RuntimeError("failed worker segment retains its outcome and is never retried")
    raise RuntimeError("arm did not complete within the registered worker count")


def _launch_worker(descriptor_path: Path, descriptor_sha: str, matrix_root: Path, arm_root: Path,
                   segment_dir: Path, *, command=None, worker_command=None, arm_record=None):
    stdout_path, stderr_path = _worker_log_paths(arm_root, segment_dir)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if command is None:
        command = ([sys.executable, "-m", "dwr.experiments.source_pilot", "worker",
            "--authorization", str(descriptor_path), "--sha256", descriptor_sha]
            if worker_command is None else list(worker_command(descriptor_path, descriptor_sha)))
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError("private worker command factory returned an invalid process command")
    with stdout_path.open("ab") as stdout_stream, stderr_path.open("ab") as stderr_stream:
        proc = subprocess.Popen(command, cwd=_root(), stdin=subprocess.DEVNULL,
            stdout=stdout_stream, stderr=stderr_stream,
            creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) |
                getattr(subprocess, "CREATE_NO_WINDOW", 0)))
    descriptor_file = Path(descriptor_path).resolve(strict=True)
    descriptor = _read_json_path(descriptor_file)
    if descriptor_file != (Path(segment_dir).resolve(strict=True) / "worker-descriptor.json") or \
            _sha(_canonical(descriptor) + b"\n") != descriptor_sha:
        raise ValueError("launched worker descriptor bytes differ from the issued process receipt")
    launcher_identity = _process_identity(proc.pid)
    process_ref = {key: launcher_identity[key] for key in ("pid", "creation_time", "image")}
    issuer = descriptor["launcher"]
    process_parent = {"pid": launcher_identity["parent_pid"],
        "creation_time": launcher_identity["parent_creation_time"], "image": launcher_identity["parent_image"]}
    if process_parent != issuer:
        raise ValueError("actual Popen launcher is not a direct child of the issuing supervisor")
    launch_receipt = {"schema_version": "dwr-source-pilot-launch-receipt-v1",
        "descriptor_sha256": descriptor_sha, "nonce": descriptor["nonce"],
        "command": list(command), "command_sha256": _sha(_canonical(list(command))),
        "issuer": issuer, "process": process_ref, "process_parent": process_parent}
    launch_digest = _exclusive_json(_launch_receipt_path(Path(segment_dir)), launch_receipt)
    if arm_record is not None:
        arm_record.append_event(_LAUNCH_EVENT, {"segment_index": descriptor["segment_index"],
            "descriptor_sha256": descriptor_sha, "nonce": descriptor["nonce"],
            "launch_receipt": {"path": Path(segment_dir).resolve(strict=True).relative_to(
                Path(arm_root).resolve(strict=True)).as_posix() + "/" + _LAUNCH_RECEIPT,
                "sha256": launch_digest}})
    _write_status(matrix_root, worker=None, worker_launcher=process_ref, current_arm=arm_root.name)
    return proc, stdout_path, stderr_path


def _wait_worker(matrix_root: Path, arm_root: Path, segment_dir: Path, process) -> int:
    limits = _registration(_root())["config"]["limits"]
    matrix_initial = strict_json_loads((matrix_root / "manifest.initial.json").read_text(encoding="utf-8"))
    arm_initial = strict_json_loads((arm_root / "manifest.initial.json").read_text(encoding="utf-8"))
    matrix_start = _parse_utc(matrix_initial["started_utc"]).timestamp()
    arm_start = _parse_utc(arm_initial["started_utc"]).timestamp()
    deadline = min(matrix_start + limits["matrix_external_watchdog_seconds"],
        arm_start + limits["arm_external_watchdog_seconds"])
    while process.poll() is None:
        receipt_path = Path(segment_dir) / "worker-receipt.json"
        if receipt_path.is_file():
            receipt = _read_json_path(receipt_path)
            _write_status(matrix_root, worker={key: receipt["worker"][key]
                for key in ("pid", "creation_time", "image")},
                worker_launcher=dict(receipt["launcher"]))
        if time.time() >= deadline:
            _terminate_worker_by_receipt(segment_dir)
            raise TimeoutError("original arm/matrix watchdog deadline expired")
        time.sleep(0.25)
    _write_status(matrix_root, worker=None, worker_launcher=None)
    return int(process.returncode)


def _wait_existing_worker(matrix_root: Path, arm_root: Path, segment_dir: Path, receipt: Mapping[str, object]) -> int:
    from dwr.experiments.source_overfit_job import _pid_matches
    limits = _registration(_root())["config"]["limits"]
    matrix_initial = strict_json_loads((matrix_root / "manifest.initial.json").read_text(encoding="utf-8"))
    arm_initial = strict_json_loads((arm_root / "manifest.initial.json").read_text(encoding="utf-8"))
    deadline = min(_parse_utc(matrix_initial["started_utc"]).timestamp() + limits["matrix_external_watchdog_seconds"],
        _parse_utc(arm_initial["started_utc"]).timestamp() + limits["arm_external_watchdog_seconds"])
    worker = receipt["worker"]
    launcher = receipt["launcher"]
    descriptor = _read_descriptor_file(segment_dir / "worker-descriptor.json", receipt["descriptor_sha256"])[1]
    issuer = descriptor["launcher"]
    while _pid_matches(worker["pid"], worker["creation_time"]) or \
            (launcher != issuer and _pid_live(launcher)):
        if time.time() >= deadline:
            _terminate_worker_by_receipt(segment_dir)
            raise TimeoutError("original watchdog deadline expired while attaching to worker")
        worker_live = _pid_matches(worker["pid"], worker["creation_time"])
        _write_status(matrix_root,
            worker={key: worker[key] for key in ("pid", "creation_time", "image")} if worker_live else None,
            worker_launcher=launcher if launcher != issuer and _pid_live(launcher) else None)
        time.sleep(0.25)
    _write_status(matrix_root, worker=None, worker_launcher=None)
    return 1


def _terminate_worker_by_receipt(segment_dir: Path) -> None:
    receipt_path = segment_dir / "worker-receipt.json"
    if not receipt_path.is_file():
        return
    receipt = _read_json_path(receipt_path)
    from dwr.experiments.source_overfit_job import _pid_matches, _process_image, _terminate_owned_process
    registration = _registration(_root())
    authorization = _authorization_from_issued_receipt(segment_dir, receipt, registration)
    issuer = authorization["expected_launcher"]
    worker = receipt["worker"]
    if _pid_matches(worker["pid"], worker["creation_time"]) and _process_image(worker["pid"]) == worker["image"]:
        _terminate_owned_process(worker["pid"], worker["creation_time"])
    launcher = receipt["launcher"]
    if launcher != issuer and _pid_matches(launcher["pid"], launcher["creation_time"]) and \
            _process_image(launcher["pid"]) == launcher["image"]:
        _terminate_owned_process(launcher["pid"], launcher["creation_time"])


def _launch_watchdog(matrix_root: Path) -> None:
    log_path = matrix_root / "watchdog.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen([sys.executable, "-m", "dwr.experiments.source_pilot_watchdog",
            "--matrix-root", str(matrix_root.resolve())], cwd=_root(), stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, creationflags=(getattr(subprocess, "DETACHED_PROCESS", 0) |
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)))
    _write_status(matrix_root, watchdog=_reference(process.pid))


def _known_worker_fault(segment: Path, stderr_path: Path | None = None, *, allow_keyboard_interrupt: bool = False) -> bool:
    forbidden = ("out of memory", "cuda out of memory", "cuda error", "cuda failure", "diverged",
        "numerical", "floatingpoint", "overflow", "underflow", "configuration mismatch",
        "config differs", "deadline", "budget exhausted", "timed out", "timeout")
    paths = [stderr_path] if stderr_path is not None else [_worker_log_paths(segment.parent, segment)[1]]
    text = ""
    for path in paths:
        try:
            text += path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            pass
    if any(token in text for token in forbidden) or re.search(r"\b(?:nan|infinite|nonfinite)\b", text):
        return True
    exception_line = re.search(r"(?m)^[A-Za-z_][A-Za-z_0-9]*(?:Error|Exception|Failure|Interrupt)(?::|\s*$)", text)
    if "traceback (most recent call last)" in text or exception_line:
        final_exception = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
        if not (allow_keyboard_interrupt and re.fullmatch(r"keyboardinterrupt(?::.*)?", final_exception)):
            return True
    for path in (segment / "events.jsonl",):
        if path.exists():
            return any(row["kind"] in {"known_fault", "numerical_failure", "cuda_failure", "oom_failure"}
                for row in _event_rows(path))
    return False


def _reopen_record(record_root: Path, original_deadlines: Mapping[str, object]) -> RunRecord:
    from dwr.experiments.records import _validate_events
    initial_path, events_path = record_root / "manifest.initial.json", record_root / "events.jsonl"
    raw = initial_path.read_bytes()
    initial = strict_json_loads(raw.decode("utf-8"))
    if raw != _canonical(initial) + b"\n" or initial.get("settings", {}).get("original_deadlines") != dict(original_deadlines):
        raise ValueError("original RunRecord anchor/deadline settings changed")
    if (record_root / "manifest.json").exists():
        raise ValueError("cannot reopen a terminal RunRecord")
    _validate_events(events_path.read_bytes())
    start = initial["started_utc"]
    elapsed = max(0, time.time() - _parse_utc(start).timestamp())
    return RunRecord(record_root, start, time.monotonic() - elapsed, _sha(raw))


def _reopen_matrix(root: Path, registration: Mapping[str, object], physical: Mapping[str, object],
                   runtime: Mapping[str, object], matrix_root: Path):
    clock = _clock_snapshot(matrix_root)
    deadlines = _derive_deadlines(registration["config"]["limits"], {"matrix": clock,
        "arm": {**clock}})["epochs"]
    initial = strict_json_loads((matrix_root / "manifest.initial.json").read_text(encoding="utf-8"))
    _verify_initial_binding(matrix_root, _sha((matrix_root / "manifest.initial.json").read_bytes()),
        expected_kind="matrix", config_sha256=registration["config_sha256"],
        cohort_sha256=registration["cohort_sha256"], source_commit=physical["commit"],
        runtime_sha256=_sha(_canonical(runtime)), physical_receipt=physical, runtime_receipt=runtime)
    expected = {key: deadlines[key] for key in ("matrix_deadline_epoch", "matrix_watchdog_epoch")}
    if initial["settings"].get("original_deadlines") != expected:
        raise ValueError("matrix original deadlines differ from anchor derivation")
    return _reopen_record(matrix_root, expected), clock, deadlines


def _reopen_arm(registration: Mapping[str, object], physical: Mapping[str, object], runtime: Mapping[str, object],
                arm_root: Path, arm_id: str, arm_index: int, deadlines: Mapping[str, object]):
    clock = _clock_snapshot(arm_root)
    expected = _arm_deadlines(registration["config"]["limits"],
        _clock_snapshot(arm_root.parents[1]), clock)
    if dict(deadlines) != expected:
        raise ValueError("arm original deadlines differ from matrix and arm anchors")
    _verify_initial_binding(arm_root, _sha((arm_root / "manifest.initial.json").read_bytes()),
        expected_kind="arm", arm_id=arm_id, config_sha256=registration["config_sha256"],
        cohort_sha256=registration["cohort_sha256"], source_commit=physical["commit"],
        runtime_sha256=_sha(_canonical(runtime)), physical_receipt=physical, runtime_receipt=runtime)
    return _reopen_record(arm_root, expected), clock


def _check_arm_seal(matrix_root: Path, arm_root: Path, stage: str) -> None:
    limits = _registration(_root())["config"]["limits"]
    clocks = {"matrix": _clock_snapshot(matrix_root), "arm": _clock_snapshot(arm_root)}
    deadlines = _derive_deadlines(limits, clocks)
    _touch_clocks({"matrix_root": str(matrix_root), "arm_root": str(arm_root)})
    if time.time() >= deadlines["epochs"]["effective_arm_deadline_epoch"] or \
            time.time() >= deadlines["epochs"]["matrix_deadline_epoch"]:
        raise TimeoutError(f"registered arm/matrix deadline reached during {stage}")


def _check_matrix_seal(matrix_root: Path, stage: str) -> None:
    limits = _registration(_root())["config"]["limits"]
    clock = _clock_snapshot(matrix_root)
    _advance_clock(matrix_root)
    if _validated_elapsed(clock["last_utc_epoch"], time.time(), clock["elapsed_high_water_seconds"]) >= limits["total_wall_seconds"]:
        raise TimeoutError(f"registered matrix total wall deadline reached during {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "status"):
        item = sub.add_parser(name)
        item.add_argument("--config", default=_CONFIG_PATH)
    supervisor = sub.add_parser("_supervise")
    supervisor.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "start":
        result = start(args.config)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    if args.command == "status":
        print(json.dumps(status(args.config), sort_keys=True, ensure_ascii=False))
        return 0
    _supervise(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def publish_worker_receipt(authorization: Mapping[str, object]) -> dict:
    """Acquire the distinct arm lock and bind this actual process before model work."""
    from dwr.experiments.source_overfit_job import _ByteLock, _process_image
    segment_dir = Path(str(authorization["segment_dir"])).resolve(strict=True)
    expected = authorization["expected_launcher"]
    if not isinstance(expected, dict):
        raise ValueError("authorization lacks expected launcher identity")
    deadline = time.monotonic() + 15
    launch_path = _launch_receipt_path(segment_dir)
    arm_events = Path(str(authorization["arm_root"])) / "events.jsonl"
    while time.monotonic() < deadline:
        if launch_path.is_file() and arm_events.is_file():
            rows = _event_rows(arm_events)
            if any(row["kind"] == _LAUNCH_EVENT and
                    row["payload"].get("segment_index") == authorization["segment_index"] for row in rows):
                break
        time.sleep(0.025)
    else:
        raise TimeoutError("supervisor did not durably publish the actual Popen launcher receipt and arm event")
    _verified_launch_receipt(authorization)
    current = _process_identity(os.getpid())
    parent = {"pid": current["parent_pid"], "creation_time": current["parent_creation_time"],
              "image": current["parent_image"]}
    actual_launcher = _worker_launcher_for_parent(authorization, parent, live=True, worker=current)
    if _process_image(actual_launcher["pid"]) != actual_launcher["image"]:
        raise ValueError("actual worker launcher image changed")
    arm_root = Path(str(authorization["arm_root"])).resolve(strict=True)
    worker_lock = _ByteLock(_worker_lock_path(arm_root), 1, wait=False)
    worker = {"pid": current["pid"], "creation_time": current["creation_time"],
              "image": current["image"], "python_executable": os.path.realpath(os.sys.executable),
              "base_executable": os.path.realpath(getattr(os.sys, "_base_executable", os.sys.executable))}
    launcher = actual_launcher
    receipt = {"schema_version": "dwr-source-pilot-worker-receipt-v1",
        "descriptor_sha256": authorization["descriptor_sha256"], "nonce": authorization["nonce"],
        "matrix_root": str(authorization["matrix_root"]), "arm_root": str(arm_root),
        "segment_dir": str(segment_dir), "arm_id": authorization["arm_id"],
        "segment_index": authorization["segment_index"],
        "source_commit": authorization["bindings"]["source"]["commit"],
        "config_sha256": authorization["registration"]["config_sha256"],
        "runtime_sha256": authorization["bindings"]["runtime"]["receipt_sha256"],
        "worker": worker, "parent": parent, "launcher": launcher, "issued_utc": _utc()}
    receipt_path = segment_dir / "worker-receipt.json"
    try:
        _exclusive_json(receipt_path, receipt)
    except Exception:
        worker_lock.close()
        raise
    return {"receipt_path": str(receipt_path), "receipt": receipt, "worker_lock": worker_lock}


def _verify_live_worker_receipt(receipt: Mapping[str, object], authorization: Mapping[str, object]) -> bool:
    expected_keys = {"schema_version", "descriptor_sha256", "nonce", "matrix_root", "arm_root",
        "segment_dir", "arm_id", "segment_index", "source_commit", "config_sha256",
        "runtime_sha256", "worker", "parent", "launcher", "issued_utc"}
    if type(receipt) is not dict or set(receipt) != expected_keys:
        raise ValueError("worker receipt has unknown or missing fields")
    direct = ("descriptor_sha256", "nonce", "matrix_root", "arm_root", "segment_dir", "arm_id", "segment_index")
    for name in direct:
        if receipt[name] != authorization[name]:
            raise ValueError(f"worker receipt {name} differs from issued authorization")
    if receipt["source_commit"] != authorization["bindings"]["source"]["commit"]:
        raise ValueError("worker receipt source commit differs from authorization")
    if receipt["config_sha256"] != authorization["registration"]["config_sha256"]:
        raise ValueError("worker receipt config identity differs from authorization")
    runtime_sha = authorization["bindings"]["runtime"]["receipt_sha256"]
    if receipt["runtime_sha256"] != runtime_sha:
        raise ValueError("worker receipt runtime identity differs from authorization")
    worker, parent, launcher = receipt["worker"], receipt["parent"], receipt["launcher"]
    if type(worker) is not dict or set(worker) != {"pid", "creation_time", "image", "python_executable", "base_executable"}:
        raise ValueError("worker receipt has malformed actual worker identity")
    if type(parent) is not dict or set(parent) != {"pid", "creation_time", "image"}:
        raise ValueError("worker receipt has malformed parent identity")
    if type(launcher) is not dict or set(launcher) != {"pid", "creation_time", "image"}:
        raise ValueError("worker receipt has malformed launcher identity")
    expected_launcher = _worker_launcher_for_parent(authorization, parent, live=False, worker=worker)
    if launcher != expected_launcher:
        raise ValueError("worker receipt actual launcher is not bound to the issued process chain")
    from dwr.experiments.source_overfit_job import (
        _process_creation_time, _process_image, _process_parent_pid, _pid_matches,
    )
    pid, creation = worker.get("pid"), worker.get("creation_time")
    if type(pid) is not int or not _pid_matches(pid, creation):
        raise ValueError("worker process is no longer live with its issued creation identity")
    actual_image = _process_image(pid)
    actual_parent_pid = _process_parent_pid(pid)
    if actual_image != worker.get("image"):
        raise ValueError("worker process image differs from receipt")
    if parent != launcher:
        raise ValueError("worker parent differs from its actual launcher")
    if actual_parent_pid != parent.get("pid"):
        raise ValueError("worker actual parent PID differs from its immutable receipt")
    if _pid_matches(parent["pid"], parent["creation_time"]):
        if _process_creation_time(actual_parent_pid) != parent.get("creation_time") or \
                _process_image(actual_parent_pid) != parent.get("image"):
            raise ValueError("worker actual live parent identity differs from receipt")
    # A surviving worker may be reattached after its original launcher exits.
    # In that case the immutable receipt binds the historical parent chain;
    # the worker's own PID/creation/image still must match live OS state above.
    return True
