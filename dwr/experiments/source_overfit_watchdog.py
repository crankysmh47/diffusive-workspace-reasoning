"""Independent bounded outer watchdog for the registered source-overfit supervisor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

MAX_HASH_ENTRIES = 20000


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def bounded_artifact_hashes(jobroot: Path, seconds: float = 55.0) -> dict:
    deadline = time.monotonic()+max(0.0, min(float(seconds), 55.0))
    files = []
    root = jobroot.resolve()
    pending = [root]
    visited = 0
    completed = True
    while pending and time.monotonic() < deadline and visited < MAX_HASH_ENTRIES:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if time.monotonic() >= deadline or visited >= MAX_HASH_ENTRIES:
                        completed = False
                        break
                    visited += 1
                    path = Path(entry.path)
                    relative = path.relative_to(root).as_posix()
                    try:
                        info = entry.stat(follow_symlinks=False)
                        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                        if (stat.S_ISLNK(info.st_mode) or
                                getattr(info, "st_file_attributes", 0) & reparse):
                            files.append({"path": relative, "error": "link_or_reparse_point"})
                            continue
                        resolved = path.resolve()
                        try:
                            resolved.relative_to(root)
                        except ValueError:
                            files.append({"path": relative, "error": "path_escape"})
                            continue
                        if stat.S_ISDIR(info.st_mode):
                            pending.append(path)
                            continue
                        if not stat.S_ISREG(info.st_mode):
                            continue
                        digest = hashlib.sha256(); size = 0
                        with path.open("rb") as stream:
                            while time.monotonic() < deadline:
                                chunk = stream.read(1024*1024)
                                if not chunk: break
                                digest.update(chunk); size += len(chunk)
                            else:
                                files.append({"path": relative, "size_bytes": size,
                                    "partial_sha256": digest.hexdigest(), "complete": False})
                                completed = False
                                break
                        files.append({"path": relative, "size_bytes": size,
                            "sha256": digest.hexdigest(), "complete": True})
                    except OSError as error:
                        files.append({"path": relative, "error": type(error).__name__})
                if time.monotonic() >= deadline:
                    completed = False
                    break
        except OSError as error:
            files.append({"path": directory.relative_to(root).as_posix(), "error": type(error).__name__})
    if pending or visited >= MAX_HASH_ENTRIES or time.monotonic() >= deadline:
        completed = False
    files.sort(key=lambda item: item["path"])
    return {"files": files, "hash_scan_completed": completed,
            "hash_budget_seconds": max(0.0, min(float(seconds), 55.0))}


def _run(jobroot: Path, supervisor_pid: int, supervisor_creation: str, deadline_epoch: float) -> int:
    from dwr.experiments import source_overfit_job as job
    wall_anchor = time.time()
    deadline_mono = time.monotonic() + max(0.0, deadline_epoch-wall_anchor)
    last_wall, last_mono = wall_anchor, time.monotonic()
    rollback = False
    while time.monotonic() < deadline_mono:
        now_wall, now_mono = time.time(), time.monotonic()
        if _clock_rollback(last_wall, now_wall, last_mono, now_mono):
            rollback = True
            break
        last_wall, last_mono = now_wall, now_mono
        if _valid_terminal_and_no_live_workers(job, jobroot):
            return 0
        time.sleep(min(1.0, max(0.01, deadline_mono-time.monotonic())))
    # A backward wall-clock step is charged as an immediate deadline; forward
    # wall-clock jumps cannot extend the monotonic deadline anchored at launch.
    enforcement_mono = time.monotonic()
    owners = []
    discovery_errors = []
    for identity_path in sorted(jobroot.glob("worker-[0-9][0-9][0-9].identity.json")):
        try:
            identity = job._read_json(identity_path); segment_index = identity["segment_index"]
            launcher = job._read_json(jobroot/f"worker-{segment_index:03d}.launcher-receipt.json")
            descriptor = job._read_json(jobroot/f"worker-descriptor-{segment_index:03d}.json")
            worker_alive = job._pid_matches(identity.get("pid"), identity.get("creation_time"))
            launcher_alive = job._pid_matches(launcher.get("pid"), launcher.get("creation_time"))
            if ((worker_alive or launcher_alive) and
                    identity.get("descriptor_nonce") == descriptor.get("nonce") == launcher.get("descriptor_nonce") and
                    identity.get("config_sha256") == job.CONFIG_SHA256 and
                    (not worker_alive or identity.get("image") == job._process_image(identity["pid"])) and
                    (not launcher_alive or launcher.get("image") == job._process_image(launcher["pid"])) and
                    launcher.get("schema_version") == 1 and
                    type(launcher.get("pid")) is int and launcher.get("pid") > 0 and
                    isinstance(launcher.get("creation_time"), str) and
                    identity.get("parent_pid") in {launcher.get("pid"), supervisor_pid} and
                    identity.get("parent_creation_time") == (launcher.get("creation_time")
                        if identity.get("parent_pid") == launcher.get("pid") else supervisor_creation) and
                    identity.get("source_commit") == descriptor.get("source_commit") and
                    descriptor.get("supervisor_pid") == supervisor_pid and
                    descriptor.get("supervisor_creation_time") == supervisor_creation and
                    launcher.get("supervisor_pid") == descriptor.get("supervisor_pid") and
                    launcher.get("supervisor_creation_time") == descriptor.get("supervisor_creation_time")):
                owners.append((identity, launcher))
            elif worker_alive or launcher_alive:
                discovery_errors.append({"path": identity_path.name, "type": "ValueError",
                    "message": "live worker tree could not be bound to the registered descriptor/launcher"})
        except Exception as error:
            discovery_errors.append({"path": identity_path.name, "type": type(error).__name__,
                "message": str(error)})
            continue
    termination_errors = list(discovery_errors)
    for identity, launcher in owners:
        try:
            if job._pid_matches(identity.get("pid"), identity.get("creation_time")):
                job._terminate_worker_tree(identity, launcher["pid"], launcher["creation_time"])
            elif job._pid_matches(launcher.get("pid"), launcher.get("creation_time")):
                job._terminate_owned_process(launcher["pid"], launcher["creation_time"])
        except Exception as error:
            termination_errors.append({"pid": identity.get("pid"), "type": type(error).__name__,
                "message": str(error)})
    try:
        job._terminate_owned_process(supervisor_pid, supervisor_creation)
    except Exception as error:
        termination_errors.append({"pid": supervisor_pid, "type": type(error).__name__, "message": str(error)})
    for identity, launcher in owners:
        if (job._pid_matches(identity.get("pid"), identity.get("creation_time")) or
                job._pid_matches(launcher.get("pid"), launcher.get("creation_time"))):
            termination_errors.append({"pid": identity.get("pid"), "type": "TimeoutError",
                "message": "verified worker tree remained live after outer watchdog termination"})
    if job._pid_matches(supervisor_pid, supervisor_creation):
        termination_errors.append({"pid": supervisor_pid, "type": "TimeoutError",
            "message": "verified supervisor remained live after outer watchdog termination"})
    hash_budget = max(0.0, min(55.0, enforcement_mono+60.0-time.monotonic()-1.0))
    hashes = bounded_artifact_hashes(jobroot, hash_budget)
    receipt = {"schema_version": 1, "deadline_epoch": deadline_epoch, "supervisor_pid": supervisor_pid,
        "supervisor_creation_time": supervisor_creation,
        "workers": [{"pid": identity.get("pid"), "creation_time": identity.get("creation_time")}
                    for identity, _ in owners], "termination_errors": termination_errors,
        "clock_rollback_detected": rollback, **hashes,
        "recorded_utc_epoch": time.time()}
    target = jobroot/"outer-watchdog-receipt.json"
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(_canonical(receipt)+b"\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, target)
    except OSError:
        return 2
    return 3 if termination_errors else 0


def _valid_terminal_and_no_live_workers(job, jobroot: Path) -> bool:
    try:
        job._verify_terminal_job(jobroot)
    except Exception:
        return False
    for identity_path in jobroot.glob("worker-[0-9][0-9][0-9].identity.json"):
        try:
            identity = job._read_json(identity_path)
            if job._pid_matches(identity.get("pid"), identity.get("creation_time")):
                return False
            launcher = job._read_json(jobroot/f"worker-{identity['segment_index']:03d}.launcher-receipt.json")
            if job._pid_matches(launcher.get("pid"), launcher.get("creation_time")):
                return False
        except Exception:
            return False
    return True


def _clock_rollback(previous_wall: float, current_wall: float,
                    previous_monotonic: float, current_monotonic: float) -> bool:
    return current_wall < previous_wall or current_monotonic < previous_monotonic


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobroot", required=True)
    parser.add_argument("--supervisor-pid", required=True, type=int)
    parser.add_argument("--supervisor-creation", required=True)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    args = parser.parse_args(argv)
    return _run(Path(args.jobroot), args.supervisor_pid, args.supervisor_creation, args.deadline_epoch)


if __name__ == "__main__":
    raise SystemExit(main())
