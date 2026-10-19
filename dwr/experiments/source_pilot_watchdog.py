"""Independent deadline watchdog for the registered source-validation pilot matrix."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from dwr.tasks.run_support import strict_json_loads


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict:
    raw = path.read_bytes()
    value = strict_json_loads(raw.decode("utf-8"))
    if type(value) is not dict or raw != _canonical(value) + b"\n":
        raise ValueError(f"watchdog input is malformed: {path.name}")
    return value


def bounded_artifact_hashes(root: Path, budget_seconds: float = 55.0) -> dict:
    """Hash regular files within a strict failure-only wall-clock allowance."""
    deadline = time.monotonic() + budget_seconds
    entries, complete = [], True
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if time.monotonic() >= deadline:
            complete = False
            break
        if not path.is_file() or path.is_symlink():
            continue
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as stream:
                while True:
                    if time.monotonic() >= deadline:
                        complete = False
                        break
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    digest.update(block)
        except OSError as error:
            entries.append({"path": path.relative_to(root).as_posix(), "error": type(error).__name__})
            complete = False
            continue
        entries.append({"path": path.relative_to(root).as_posix(), "size_bytes": size,
            "sha256": digest.hexdigest()})
        if not complete:
            break
    return {"complete": complete, "files": entries, "elapsed_seconds": budget_seconds - max(0, deadline - time.monotonic())}


def _same_process(reference: object) -> bool:
    if type(reference) is not dict or set(reference) != {"pid", "creation_time", "image"}:
        return False
    from dwr.experiments.source_overfit_job import _pid_matches, _process_image
    try:
        return _pid_matches(reference["pid"], reference["creation_time"]) and \
            _process_image(reference["pid"]) == reference["image"]
    except (OSError, ValueError):
        return False


def _terminate(reference: object) -> bool:
    if not _same_process(reference):
        return False
    from dwr.experiments.source_overfit_job import _terminate_owned_process
    try:
        _terminate_owned_process(reference["pid"], reference["creation_time"])
        return True
    except (OSError, ValueError):
        return False


def _has_live_owned_process(worker: object, launcher: object, supervisor: object) -> bool:
    """Whether any recorded owned process still matches its PID and creation time."""
    return any(_same_process(reference) for reference in (worker, launcher, supervisor))


def _anchored_deadline(initial: dict, key: str) -> float | None:
    settings = initial.get("settings", {})
    deadlines = settings.get("original_deadlines", {}) if type(settings) is dict else {}
    if type(deadlines) is not dict or key not in deadlines:
        return None
    value = deadlines[key]
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid immutable watchdog deadline: {key}")
    return float(value)


def _deadline(matrix_root: Path, config: dict) -> tuple[float, float | None, str | None]:
    matrix_initial = _read(matrix_root / "manifest.initial.json")
    matrix_start = __import__("datetime").datetime.fromisoformat(matrix_initial["started_utc"].replace("Z", "+00:00")).timestamp()
    matrix_outer = matrix_start + config["limits"]["matrix_external_watchdog_seconds"]
    matrix_anchor = _anchored_deadline(matrix_initial, "matrix_watchdog_epoch")
    if matrix_anchor is not None:
        matrix_outer = min(matrix_outer, matrix_anchor)
    arm_id = None
    arm_outer = None
    for arm in config["arms"]:
        candidate = matrix_root / "arms" / arm["arm_id"]
        initial_path = candidate / "manifest.initial.json"
        if not initial_path.is_file() or (candidate / "manifest.json").is_file():
            continue
        arm_initial = _read(initial_path)
        arm_start = __import__("datetime").datetime.fromisoformat(arm_initial["started_utc"].replace("Z", "+00:00")).timestamp()
        arm_id = arm["arm_id"]
        arm_outer = min(arm_start + config["limits"]["arm_external_watchdog_seconds"], matrix_outer)
        arm_anchor = _anchored_deadline(arm_initial, "arm_watchdog_epoch")
        if arm_anchor is not None:
            arm_outer = min(arm_outer, arm_anchor)
        break
    return matrix_outer, arm_outer, arm_id


def _owned_references(matrix_root: Path, status: dict | None) -> tuple[dict | None, dict | None, dict | None]:
    """Find actual worker and bound launcher identities, with status as a fallback."""
    worker, launcher = None, None
    arm_root = None
    for arm in (matrix_root / "arms" / name for name in ("K0", "K1", "K4")):
        if arm.is_dir() and not (arm / "manifest.json").is_file():
            arm_root = arm
            break
    if arm_root is not None:
        segments = sorted(arm_root.glob("segment-[0-9][0-9][0-9]"), key=lambda p: p.name)
        for segment in reversed(segments):
            path = segment / "worker-receipt.json"
            if path.is_file():
                try:
                    receipt = _read(path)
                    from dwr.experiments import source_pilot_job as job
                    registration = job._registration(job._root())
                    ownership = job._verify_segment_ownership(segment, arm_root, registration, arm_root.name)
                    worker = {key: ownership["worker"][key] for key in ("pid", "creation_time", "image")}
                    if ownership["launcher"] != receipt.get("parent") or \
                            ownership["launcher"] != job._read_descriptor_file(
                                segment / "worker-descriptor.json", receipt["descriptor_sha256"])[1]["launcher"]:
                        launcher = {key: ownership["launcher"][key] for key in ("pid", "creation_time", "image")}
                    break
                except (OSError, ValueError, KeyError, TypeError):
                    worker, launcher = None, None
    supervisor = None
    events = matrix_root / "events.jsonl"
    if events.is_file():
        try:
            rows = events.read_bytes().splitlines()
            for line in reversed(rows):
                row = strict_json_loads(line.decode("utf-8"))
                if row.get("kind") in {"matrix_supervisor_started", "matrix_supervisor_attached"}:
                    supervisor = row.get("payload", {}).get("supervisor")
                    if type(supervisor) is dict and set(supervisor) == {"pid", "creation_time", "image"}:
                        break
                    supervisor = None
        except (OSError, UnicodeDecodeError, ValueError, AttributeError):
            supervisor = None
    if status:
        worker = worker or status.get("worker")
        launcher = launcher or status.get("worker_launcher")
        supervisor = supervisor or status.get("supervisor")
    return worker, launcher, supervisor


def watch(matrix_root: str | os.PathLike[str]) -> None:
    root = Path(matrix_root).resolve(strict=True)
    config_path = Path(__file__).resolve().parents[2] / "configs/experiments/source_validation_pilot_v1.json"
    config = strict_json_loads(config_path.read_text(encoding="utf-8"))
    if type(config) is not dict or type(config.get("limits")) is not dict:
        raise ValueError("registered watchdog configuration is malformed")
    while True:
        if (root / "manifest.json").is_file():
            return
        status = None
        try:
            status = _read(root / "status.json")
        except (OSError, ValueError, KeyError):
            status = None
        try:
            matrix_outer, arm_outer, arm_id = _deadline(root, config)
        except (OSError, ValueError, KeyError):
            time.sleep(0.5)
            continue
        now = time.time()
        expired_role = "matrix" if now >= matrix_outer else ("arm" if arm_outer is not None and now >= arm_outer else None)
        if expired_role is None:
            time.sleep(0.5)
            continue
        worker, launcher, supervisor = _owned_references(root, status)
        # A previous watchdog may have sealed an operational failure. Do not
        # replace its evidence on a later stale invocation once all owned
        # processes have exited. A still-live owned process remains subject to
        # this independent deadline enforcement.
        if status and status.get("state") == "operational_failure" and not \
                _has_live_owned_process(worker, launcher, supervisor):
            return
        worker_stopped = _terminate(worker)
        launcher_stopped = False
        if launcher is not None and launcher != supervisor:
            launcher_stopped = _terminate(launcher)
        supervisor_stopped = _terminate(supervisor)
        receipt = {"schema_version": "dwr-source-pilot-watchdog-failure-v1",
            "expired_scope": expired_role, "arm_id": arm_id, "observed_utc_epoch": now,
            "matrix_deadline_epoch": matrix_outer, "arm_deadline_epoch": arm_outer,
            "worker_identity": worker, "worker_launcher_identity": launcher,
            "supervisor_identity": supervisor,
            "worker_terminated": worker_stopped, "worker_launcher_terminated": launcher_stopped,
            "supervisor_terminated": supervisor_stopped,
            "hash_budget_seconds": 55.0, **bounded_artifact_hashes(root, 55.0)}
        path = root / "watchdog-failure.json"
        raw = _canonical(receipt) + b"\n"
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, path)
        except OSError:
            temporary.unlink(missing_ok=True)
        try:
            from dwr.experiments.source_pilot_job import _write_status
            _write_status(root, state="operational_failure", worker=None,
                error=f"independent {expired_role} watchdog deadline expired",
                watchdog_failure_sha256=_sha(raw))
        except (OSError, ValueError):
            pass
        return


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", required=True)
    args = parser.parse_args(argv)
    watch(args.matrix_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
