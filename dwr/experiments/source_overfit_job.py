"""Small detached supervisor for the one registered recoverable source-overfit job."""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time
import uuid

from dwr.experiments.records import RunRecord, verify_artifacts

CONFIG_PATH = "configs/experiments/source_overfit_recoverable_v2.json"
CONFIG_SHA256 = "e6c3de7e59e58c494b89052086c0840dcd3421d17396fd5a55cac40611049594"
JOB_DIR = "runs/source-overfit-20261017-recoverable-v2"
LOCK_PATH = "runs/source-overfit-20261017-recoverable-v2.lock"
_KEY = b"dwr-source-overfit-job-v2"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _atomic_json(path: Path, value: object) -> str:
    raw = _canonical(value) + b"\n"
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    for attempt in range(6):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            # Windows readers can briefly hold a status file without delete
            # sharing while another process polls the lifecycle cache. Keep
            # the completed temp receipt and retry this transient handoff.
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(0.02 * (attempt + 1))
    return _sha(raw)


def _launch_outer_watchdog(jobroot: Path, started_utc: str) -> None:
    deadline = _parse_utc(started_utc).timestamp()+5460
    with open(os.devnull, "ab") as watchdog_log:
        subprocess.Popen([sys.executable, "-m", "dwr.experiments.source_overfit_watchdog",
            "--jobroot", str(jobroot.resolve()), "--supervisor-pid", str(os.getpid()),
            "--supervisor-creation", _process_creation_time(os.getpid()),
            "--deadline-epoch", str(deadline)], cwd=_root(), stdin=subprocess.DEVNULL,
            stdout=watchdog_log, stderr=watchdog_log,
            creationflags=(getattr(subprocess, "DETACHED_PROCESS", 0) |
                           getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) |
                           getattr(subprocess, "CREATE_NO_WINDOW", 0)))


def _record_seal_failure_receipt(jobroot: Path, started_utc: str, elapsed: float,
                                stage: str, error: Exception, source_commit: str) -> None:
    if elapsed > 5460:
        return
    from dwr.experiments.source_overfit_watchdog import bounded_artifact_hashes
    _write_watchdog_receipt(jobroot, {"schema_version": 1, "started_utc": started_utc,
        "elapsed_seconds": elapsed, "stage": stage, "error_type": type(error).__name__,
        "error": str(error), "config_sha256": CONFIG_SHA256, "source_commit": source_commit,
        **bounded_artifact_hashes(jobroot, 55.0)})


class _ByteLock:
    def __init__(self, path: Path, byte: int, *, wait: bool = False):
        self.path, self.byte, self.stream = path, byte, None
        _reject_links(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() < byte + 1:
            self.stream.write(b"\0" * (byte + 1 - self.stream.tell()))
            self.stream.flush()
        self.stream.seek(byte)
        try:
            if os.name == "nt":
                import msvcrt
                mode = msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK
                msvcrt.locking(self.stream.fileno(), mode, 1)
            else:
                import fcntl
                flags = fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB)
                fcntl.lockf(self.stream.fileno(), flags, 1, byte, os.SEEK_SET)
        except (OSError, BlockingIOError):
            self.stream.close(); self.stream = None
            raise RuntimeError(f"registered job lock byte {byte} is already owned")

    def close(self):
        if self.stream is None:
            return
        if os.name == "nt":
            import msvcrt
            self.stream.seek(self.byte); msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.lockf(self.stream.fileno(), fcntl.LOCK_UN, 1, self.byte, os.SEEK_SET)
        self.stream.close(); self.stream = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def _process_creation_time(pid: int) -> str:
    if type(pid) is not int or pid <= 0:
        raise ValueError("PID must be positive")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        process = kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            raise OSError(ctypes.get_last_error(), "OpenProcess failed")
        creation = wintypes.FILETIME(); exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME(); user = wintypes.FILETIME()
        try:
            kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME)]
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            if not kernel32.GetProcessTimes(process, ctypes.byref(creation), ctypes.byref(exit_time),
                                           ctypes.byref(kernel), ctypes.byref(user)):
                raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
            return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
        finally:
            kernel32.CloseHandle(process)
    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    return stat_text[stat_text.rfind(")") + 2:].split()[19]


def _process_parent_pid(pid: int) -> int:
    if os.name != "nt":
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        return int(stat_text[stat_text.rfind(")") + 2:].split()[1])
    import ctypes
    from ctypes import wintypes
    class ProcessBasicInformation(ctypes.Structure):
        _fields_ = [("Reserved1", ctypes.c_void_p), ("PebBaseAddress", ctypes.c_void_p),
            ("Reserved2", ctypes.c_void_p * 2), ("UniqueProcessId", ctypes.c_size_t),
            ("InheritedFromUniqueProcessId", ctypes.c_size_t)]
    ntdll = ctypes.WinDLL("ntdll")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), "OpenProcess for parent identity failed")
    try:
        info = ProcessBasicInformation()
        status = ntdll.NtQueryInformationProcess(handle, 0, ctypes.byref(info), ctypes.sizeof(info), None)
        if status != 0:
            raise OSError(f"NtQueryInformationProcess failed: {status}")
        return int(info.InheritedFromUniqueProcessId)
    finally:
        kernel32.CloseHandle(handle)


def _process_image(pid: int) -> str:
    if os.name != "nt":
        return os.path.realpath(f"/proc/{pid}/exe")
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), "OpenProcess for image identity failed")
    try:
        size = wintypes.DWORD(32768)
        buf = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            raise OSError(ctypes.get_last_error(), "QueryFullProcessImageNameW failed")
        return os.path.normcase(os.path.realpath(buf.value))
    finally:
        kernel32.CloseHandle(handle)


def _read_json(path: Path) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if type(value) is not dict or _canonical(value) + b"\n" != raw:
        raise ValueError(f"{path.name} is not canonical stable JSON")
    return value


def _config(config_path: str | os.PathLike[str]) -> Path:
    root = _root()
    path = Path(os.path.abspath(os.fspath(config_path)))
    expected = Path(os.path.abspath(os.fspath(root.joinpath(*PurePosixPath(CONFIG_PATH).parts))))
    _reject_links(path)
    if path != expected:
        raise ValueError("only the registered recoverable source-overfit config is accepted")
    if _sha(path.read_bytes()) != CONFIG_SHA256:
        raise ValueError("registered config bytes differ from the pinned SHA-256")
    return path


def _reject_links(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for item in reversed((absolute, *absolute.parents)):
        if not item.exists() and not item.is_symlink():
            continue
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
            raise ValueError("source-overfit job paths cannot contain links or reparse points")


def _lock_path() -> Path:
    return _root().joinpath(*PurePosixPath(LOCK_PATH).parts)


def start(config_path: str | os.PathLike[str]) -> dict:
    config = _config(config_path)
    root = _root(); runs = root / "runs"; runs.mkdir(exist_ok=True)
    jobroot = root.joinpath(*PurePosixPath(JOB_DIR).parts)
    try:
        lock = _ByteLock(_lock_path(), 0, wait=False)
    except RuntimeError:
        if jobroot.exists():
            return {"attached": True, **_status_data(jobroot)}
        raise
    try:
        if jobroot.exists():
            if (jobroot/"manifest.json").is_file():
                return {"attached": True, "state": "terminal", **_verify_terminal_job(jobroot)}
            try:
                status = _status_data(jobroot)
            except Exception:
                status = {}
            if status.get("state") in {"running", "recovering", "starting"}:
                try:
                    current_creation = _process_creation_time(status["supervisor_pid"])
                except (OSError, ValueError):
                    current_creation = None
                if current_creation == status.get("supervisor_creation_time"):
                    # A live process without the OS lock is an ownership inconsistency.
                    raise ValueError("supervisor PID is live but does not own the registered lock")
        stdout_path = runs / "source-overfit-supervisor.stdout.log"
        stderr_path = runs / "source-overfit-supervisor.stderr.log"
        process = subprocess.Popen([sys.executable, "-m", "dwr.experiments.source_overfit_job",
            "_supervise", "--config", str(config)], cwd=root, stdin=subprocess.DEVNULL,
            stdout=stdout_path.open("ab"), stderr=stderr_path.open("ab"),
            creationflags=(getattr(subprocess, "DETACHED_PROCESS", 0) |
                           getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) |
                           getattr(subprocess, "CREATE_NO_WINDOW", 0)))
        return {"attached": False, "supervisor_pid": process.pid,
                "supervisor_creation_time": _process_creation_time(process.pid),
                "stdout": str(stdout_path), "stderr": str(stderr_path),
                "start_detaches": True, "config_sha256": CONFIG_SHA256}
    finally:
        lock.close()


def status(config_path: str | os.PathLike[str]) -> dict:
    _config(config_path)
    root = _root().joinpath(*PurePosixPath(JOB_DIR).parts)
    if not root.exists():
        return {"state": "not_started", "jobroot": str(root)}
    result = _status_data(root)
    if (root / "manifest.json").is_file():
        result["verified_terminal"] = _verify_terminal_job(root)
    for role in ("supervisor", "worker"):
        pid = result.get(f"{role}_pid")
        creation = result.get(f"{role}_creation_time")
        try:
            alive = bool(pid and creation and _process_creation_time(pid) == creation)
        except (OSError, ValueError):
            alive = False
        result[f"{role}_alive"] = alive
    return result


def _status_data(jobroot: Path) -> dict:
    _reject_links(jobroot)
    status_path = jobroot / "status.json"
    if not status_path.is_file():
        return {"state": "corrupt_or_initializing", "jobroot": str(jobroot)}
    status_data = _read_json(status_path)
    if status_data.get("jobroot") != str(jobroot.resolve()):
        raise ValueError("status jobroot identity mismatch")
    return status_data


def _create_job_record(config: Path, jobroot: Path, supervisor_pid: int) -> RunRecord:
    from dwr.experiments.source_overfit import _load_registration, _record_identities, _kernel_identities
    registration = _load_registration(_root(), config)
    # Preflight identities are resolved by the controller before task access; the supervisor
    # records source/config/runtime binding here and worker records exact model/data identities.
    source = __import__("dwr.experiments.gpu_feasibility", fromlist=["validate_clean_source"]).validate_clean_source(
        _root(), registration["config"]["source_base_commit"])
    config_obj = registration["config"]
    model = registration["feasibility_config"]["model"]
    identities = _kernel_identities(registration["config_raw"], config_obj,
        source_commit=source["commit"], weight_sha256=model["weight_sha256"],
        tokenizer_inventory_sha256=model["artifact_inventory_sha256"])
    runtime = __import__("dwr.experiments.source_overfit", fromlist=["_environment_receipt"])._environment_receipt()
    record = RunRecord.create(jobroot, command=[sys.executable, "-m", "dwr.experiments.source_overfit_job",
        "start", "--config", CONFIG_PATH], config_path=config,
        expected_config_sha256=CONFIG_SHA256,
        identities=_record_identities(identities, config_version=config_obj["config_version"],
            model_repo=model["repo_id"], model_revision=model["revision"],
            schedule_type=config_obj["training"]["schedule"]["type"]),
        rng_streams=config_obj["seeds"], settings={"jobroot": str(jobroot), "supervisor_pid": supervisor_pid,
            "supervisor_creation_time": _process_creation_time(supervisor_pid),
            "source_commit": source["commit"], "physical_source_files": source["files"],
            "environment_receipt": runtime, "environment_receipt_sha256": _sha(_canonical(runtime)),
            "start_detaches": True}, budgets={"wall_seconds": 5400, "memory_mib": 12288, "attempts": 3},
        confirmatory=False, source_root=_root())
    initial_raw = (jobroot / "manifest.initial.json").read_bytes()
    binding = {"schema_version": 1, "initial_manifest_sha256": _sha(initial_raw),
        "started_utc": record._started_utc, "source_commit": source["commit"],
        "physical_source_files": source["files"], "environment_receipt": runtime,
        "environment_receipt_sha256": _sha(_canonical(runtime)), "config_sha256": CONFIG_SHA256,
        "budgets": {"work_seconds": 5220, "seal_seconds": 5400, "watchdog_seconds": 5460}}
    binding_path = jobroot / "job-binding.json"
    with binding_path.open("xb") as stream:
        stream.write(_canonical(binding) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    record.append_event("job_binding_created", {"sha256": _sha(_canonical(binding) + b"\n"),
        "initial_manifest_sha256": binding["initial_manifest_sha256"], "started_utc": record._started_utc})
    _atomic_json(jobroot / "elapsed-budget.json", {"schema_version": 1, "started_utc": record._started_utc,
        "last_observed_utc_epoch": _parse_utc(record._started_utc).timestamp(), "elapsed_seconds": 0.0})
    return record


def _open_job_record(jobroot: Path, config: Path) -> RunRecord:
    from dwr.experiments import records
    from dwr.experiments import gpu_feasibility
    from dwr.experiments.source_overfit import _load_registration, _kernel_identities, _record_identities
    initial, raw = records._parse_json_file(jobroot / "manifest.initial.json", "job initial manifest")
    records._validate_initial(initial)
    if (jobroot / "manifest.json").exists() or initial["status"] != "running":
        raise ValueError("only an unsealed original job record can be resumed")
    if initial["config"]["sha256"] != CONFIG_SHA256:
        raise ValueError("existing job record config identity differs")
    binding_raw = (jobroot / "job-binding.json").read_bytes()
    binding = _read_json(jobroot / "job-binding.json")
    if binding.get("initial_manifest_sha256") != _sha(raw) or binding.get("started_utc") != initial["started_utc"]:
        raise ValueError("original job initial manifest or immutable UTC anchor changed")
    if binding.get("config_sha256") != CONFIG_SHA256 or binding.get("budgets") != {
            "work_seconds": 5220, "seal_seconds": 5400, "watchdog_seconds": 5460}:
        raise ValueError("immutable job binding differs from registration")
    registration = _load_registration(_root(), config)
    config_obj = registration["config"]
    source = gpu_feasibility.validate_clean_source(_root(), config_obj["source_base_commit"])
    model = registration["feasibility_config"]["model"]
    identities = _kernel_identities(registration["config_raw"], config_obj,
        source_commit=source["commit"], weight_sha256=model["weight_sha256"],
        tokenizer_inventory_sha256=model["artifact_inventory_sha256"])
    expected = _record_identities(identities, config_version=config_obj["config_version"],
        model_repo=model["repo_id"], model_revision=model["revision"],
        schedule_type=config_obj["training"]["schedule"]["type"])
    if initial["identities"] != expected or source["commit"] != binding.get("source_commit") or \
            source["files"] != binding.get("physical_source_files"):
        raise ValueError("resumed job initial identities differ from the current exact source/model pins")
    if initial["budgets"] != {"wall_seconds": 5400, "memory_mib": 12288, "attempts": 3}:
        raise ValueError("resumed job budget differs from the original closed registration")
    events = records._read_regular(jobroot / "events.jsonl", "job event stream")
    records._validate_events(events)
    parsed_events = [json.loads(line.decode("utf-8")) for line in events.splitlines()]
    binding_digest = _sha(_canonical(binding) + b"\n")
    if not any(event["kind"] == "job_binding_created" and event["payload"].get("sha256") == binding_digest
               for event in parsed_events):
        raise ValueError("original immutable job binding event is missing or changed")
    settings = initial.get("settings", {})
    if (settings.get("source_commit") != binding["source_commit"] or
            settings.get("physical_source_files") != binding["physical_source_files"] or
            settings.get("environment_receipt") != binding["environment_receipt"] or
            settings.get("environment_receipt_sha256") != binding["environment_receipt_sha256"]):
        raise ValueError("original job source/runtime settings changed")
    runtime = __import__("dwr.experiments.source_overfit", fromlist=["_environment_receipt"])._environment_receipt()
    if (_sha(_canonical(runtime)) != binding.get("environment_receipt_sha256") or
            runtime != binding.get("environment_receipt")):
        raise ValueError("resumed job runtime differs from the immutable original environment")
    initial_epoch = _parse_utc(initial["started_utc"]).timestamp()
    elapsed = _read_json(jobroot / "elapsed-budget.json")
    if elapsed.get("started_utc") != initial["started_utc"]:
        raise ValueError("durable elapsed receipt does not bind the original UTC anchor")
    now_epoch = time.time()
    elapsed["elapsed_seconds"] = _validated_elapsed(elapsed["last_observed_utc_epoch"], now_epoch,
                                                        elapsed["elapsed_seconds"])
    elapsed["last_observed_utc_epoch"] = now_epoch
    _atomic_json(jobroot / "elapsed-budget.json", elapsed)
    original_epoch = initial_epoch
    return RunRecord(jobroot, initial["started_utc"], time.monotonic() - (time.time()-original_epoch), _sha(raw))


def _verify_terminal_job(jobroot: Path) -> dict:
    from dwr.experiments.records import _validate_events
    terminal = verify_artifacts(jobroot)
    initial_raw = (jobroot / "manifest.initial.json").read_bytes()
    binding = _read_json(jobroot / "job-binding.json")
    if binding.get("initial_manifest_sha256") != _sha(initial_raw):
        raise ValueError("terminal job initial manifest differs from its immutable binding")
    events_raw = (jobroot / "events.jsonl").read_bytes(); _validate_events(events_raw)
    if terminal.get("status") not in {"completed", "failed", "interrupted"}:
        raise ValueError("terminal job manifest has an invalid status")
    return {"status": terminal["status"], "manifest_sha256": _sha((jobroot / "manifest.json").read_bytes()),
            "events_sha256": _sha(events_raw), "initial_manifest_sha256": _sha(initial_raw)}


def _write_integrity_error_receipt(jobroot: Path, error: Exception) -> None:
    path = jobroot / "integrity-error-receipt.json"
    if path.exists():
        return
    value = {"schema_version": 1, "utc": _utc(), "error_type": type(error).__name__,
        "error": str(error), "preserved_files": sorted(path.name for path in jobroot.iterdir() if path.is_file())}
    try:
        with path.open("xb") as stream:
            stream.write(_canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    except FileExistsError:
        pass


def _supervise(config_path: str | os.PathLike[str], *, _test_factory: dict | None = None) -> int:
    config = Path(config_path).resolve() if _test_factory else _config(config_path)
    root = _root(); jobroot = Path(_test_factory["jobroot"]) if _test_factory else root.joinpath(*PurePosixPath(JOB_DIR).parts)
    lock_path = Path(_test_factory["lock_path"]) if _test_factory else _lock_path()
    with _ByteLock(lock_path, 0, wait=True):
        resumed = jobroot.exists()
        watchdog_started = False
        if resumed and not _test_factory and (jobroot/"job-binding.json").is_file():
            early_binding = _read_json(jobroot/"job-binding.json")
            _launch_outer_watchdog(jobroot, early_binding["started_utc"])
            watchdog_started = True
        discovered_worker = _discover_live_worker(jobroot, lock_path) if resumed else None
        previous = {}
        if resumed:
            try:
                previous = _status_data(jobroot)
            except Exception:
                # Status is a cache; the immutable binding, records and worker receipts are authoritative.
                previous = {}
        if resumed:
            try:
                record = (_test_factory["open_record"](jobroot, config) if _test_factory else
                          _open_job_record(jobroot, config))
            except Exception as error:
                try:
                    _cleanup_worker_from_status(jobroot, previous)
                except Exception:
                    pass
                _write_integrity_error_receipt(jobroot, error)
                raise
        else:
            record = (_test_factory["create_record"](config, jobroot, os.getpid()) if _test_factory else
                      _create_job_record(config, jobroot, os.getpid()))
        status_path = jobroot / "status.json"
        orphan_worker = discovered_worker
        if orphan_worker is None and resumed and previous.get("worker_pid") and previous.get("worker_creation_time"):
            orphan_worker = {"pid": previous["worker_pid"], "creation_time": previous["worker_creation_time"],
                "segment_index": previous.get("segment_index"),
                "nonce": previous.get("worker_nonce"),
                "launcher_pid": previous.get("worker_launcher_pid"),
                "launcher_creation_time": previous.get("worker_launcher_creation_time"),
                "supervisor_pid": previous.get("supervisor_pid"),
                "supervisor_creation_time": previous.get("supervisor_creation_time")}
        binding = _read_json(jobroot / "job-binding.json")
        started_utc = binding["started_utc"]
        elapsed_receipt = _read_json(jobroot / "elapsed-budget.json")
        started = time.monotonic()
        elapsed_before = float(elapsed_receipt["elapsed_seconds"])
        deadline = started + max(0.0, 5460 - elapsed_before)
        seal_deadline = started + max(0.0, 5400 - elapsed_before)
        work_deadline = started + max(0.0, 5220 - elapsed_before)
        if not _test_factory and not watchdog_started:
            _launch_outer_watchdog(jobroot, started_utc)
        state = {**previous, "schema_version": 1, "state": "starting", "jobroot": str(jobroot.resolve()),
            "config_sha256": CONFIG_SHA256, "supervisor_pid": os.getpid(),
            "supervisor_creation_time": _process_creation_time(os.getpid()), "worker_pid": None,
            "worker_creation_time": None, "segment_index": None, "committed_step": 0,
            "heartbeat_utc": _utc(), "started_utc": started_utc, "elapsed_seconds": elapsed_before,
            "outcome": None, "stdout": str(root / "runs" / "source-overfit-supervisor.stdout.log"),
            "stderr": str(root / "runs" / "source-overfit-supervisor.stderr.log")}
        if orphan_worker:
            state["_orphan_worker"] = orphan_worker
            state.update({"worker_pid": orphan_worker["pid"],
                "worker_creation_time": orphan_worker["creation_time"],
                "worker_launcher_pid": orphan_worker["launcher_pid"],
                "worker_launcher_creation_time": orphan_worker["launcher_creation_time"],
                "worker_nonce": orphan_worker["nonce"],
                "segment_index": orphan_worker["segment_index"]})
        _atomic_json(status_path, state)
        record.append_event("supervisor_resumed" if resumed else "supervisor_started", {key: state[key] for key in
            ("supervisor_pid", "supervisor_creation_time", "started_utc")})
        try:
            result = _run_workers(config, jobroot, record, state, deadline, work_deadline=work_deadline,
                                  seal_deadline=seal_deadline,
                                  _test_factory=_test_factory)
        except Exception as error:
            try:
                _cleanup_worker_from_status(jobroot, state)
            except Exception as cleanup_error:
                record.append_event("worker_cleanup_failed", {"type": type(cleanup_error).__name__,
                    "message": str(cleanup_error)})
            result = {"outcome": f"failed_supervisor: {type(error).__name__}: {error}",
                      "segments": int(state.get("segment_index") is not None),
                      "recoveries": int(state.get("recoveries", 0)),
                      "worker_processes": int(state.get("worker_pid") is not None),
                      "elapsed_seconds": _persist_elapsed(jobroot)}
            record.append_event("supervisor_failure", {"type": type(error).__name__, "message": str(error)})
        state.update(result); state["state"] = "terminal"; state["ended_utc"] = _utc()
        state["elapsed_seconds"] = _persist_elapsed(jobroot)
        _atomic_json(status_path, state)
        successful = result["outcome"] in {"PASS_TINY_SOURCE_OVERFIT", "NO_GO_TINY_SOURCE_OVERFIT"}
        if _test_factory:
            accounting_config = _test_factory["registration"]["config"]
        else:
            from dwr.experiments.source_overfit import _load_registration
            accounting_config = _load_registration(root, config)["config"]
        accounting = _aggregate_physical_work(jobroot, accounting_config)
        _atomic_json(jobroot / "physical-work-ledger.json", accounting)
        record.append_event("whole_job_physical_accounting", accounting)
        seal_stage = {"value": "before_artifact_hashing"}
        def seal_budget(stage: str) -> None:
            seal_stage["value"] = stage
            _persist_elapsed(jobroot)
            if time.monotonic() >= seal_deadline:
                raise TimeoutError(f"registered total seal deadline reached at {stage}")
        try:
            record.finish("completed" if successful else "failed",
                counts={"segments": int(result["segments"]), "recoveries": int(result["recoveries"]),
                        "elapsed_milliseconds": int(result["elapsed_seconds"] * 1000),
                        "pass": int(result["outcome"] == "PASS_TINY_SOURCE_OVERFIT"),
                        "observed_physical_updates": accounting["observed"]["observed_physical_updates"],
                        "started_physical_updates": accounting["observed"]["started_physical_updates"],
                        "inflight_physical_updates": accounting["observed"]["unknown_inflight_worker_updates"],
                        "prediction_attempts": accounting["observed"]["prediction_attempts"],
                        "decoder_calls_conservative": accounting["observed"]["conservative_decoder_calls"],
                        "official_evaluation_denominator": 3},
                artifacts=_job_artifacts(jobroot),
                error=None if successful else result["outcome"], budget_check=seal_budget)
        except Exception as error:
            elapsed = _persist_elapsed(jobroot)
            _record_seal_failure_receipt(jobroot, started_utc, elapsed, seal_stage["value"],
                                         error, binding["source_commit"])
            raise
        return 0


def _run_workers(config: Path, jobroot: Path, record: RunRecord, state: dict, deadline: float,
                 *, work_deadline: float, seal_deadline: float, _test_factory: dict | None = None) -> dict:
    from dwr.experiments.source_overfit import _load_registration, _kernel_identities, _validate_release_source
    from dwr.training.recovery import RecoveryStore
    registration = _test_factory["registration"] if _test_factory else _load_registration(_root(), config)
    worker_lock_path = Path(_test_factory["lock_path"]) if _test_factory else _lock_path()
    c = registration["config"]
    model = registration["feasibility_config"]["model"]
    binding = _read_json(jobroot / "job-binding.json")
    source_commit = binding["source_commit"]
    source_receipt = (_test_factory["source_receipt"] if _test_factory else
        _validate_release_source(__import__("dwr.experiments.gpu_feasibility", fromlist=["validate_clean_source"])
        .validate_clean_source(_root(), c["source_base_commit"]), c["source_base_commit"]))
    if source_receipt["commit"] != source_commit or source_receipt["files"] != binding["physical_source_files"]:
        raise ValueError("current physical source differs from immutable first-launch source identity")
    identities = _kernel_identities(registration["config_raw"], c, source_commit=source_commit,
        weight_sha256=model["weight_sha256"], tokenizer_inventory_sha256=model["artifact_inventory_sha256"])
    orphan = state.pop("_orphan_worker", None)
    orphan_segment = None
    if orphan is not None:
        orphan_segment = int(orphan.get("segment_index")) if type(orphan.get("segment_index")) is int else None
        if orphan_segment is not None and _pid_matches(orphan["pid"], orphan["creation_time"]):
            descriptor_path = jobroot / f"worker-descriptor-{orphan_segment:03d}.json"
            descriptor = _read_json(descriptor_path)
            identity_path = jobroot / f"worker-{orphan_segment:03d}.identity.json"
            worker_identity = _read_json(identity_path)
            if (descriptor.get("config_sha256") != CONFIG_SHA256 or
                    descriptor.get("supervisor_pid") != orphan["supervisor_pid"] or
                    descriptor.get("supervisor_creation_time") != orphan["supervisor_creation_time"] or
                    descriptor.get("segment_index") != orphan_segment or
                    descriptor.get("nonce") != orphan.get("nonce") or
                    worker_identity.get("descriptor_nonce") != orphan.get("nonce") or
                    worker_identity.get("pid") != orphan["pid"] or
                    worker_identity.get("creation_time") != orphan["creation_time"] or
                    worker_identity.get("config_sha256") != CONFIG_SHA256 or
                    worker_identity.get("source_commit") != source_commit or
                    worker_identity.get("image") != _process_image(orphan["pid"])):
                raise ValueError("orphan worker descriptor does not match the prior status owner")
            try:
                duplicate_lock = _ByteLock(worker_lock_path, 1, wait=False)
            except RuntimeError:
                duplicate_lock = None
            else:
                duplicate_lock.close()
                time.sleep(0.2)
                if _pid_matches(orphan["pid"], orphan["creation_time"]):
                    raise ValueError("status names a live worker PID that does not own the worker lock")
            record.append_event("orphan_worker_attached", {"pid": orphan["pid"],
                "creation_time": orphan["creation_time"], "segment_index": orphan_segment,
                "descriptor_sha256": _sha(_canonical(descriptor)+b"\n")})
            state.update({"state": "running", "worker_pid": orphan["pid"],
                "worker_creation_time": orphan["creation_time"], "segment_index": orphan_segment})
            _atomic_json(jobroot / "status.json", state)
            outcome = _wait_orphan_worker(jobroot, record, state, orphan, deadline, work_deadline, seal_deadline,
                                          _test_factory=_test_factory)
            if outcome is not None:
                return {"outcome": outcome, "segments": orphan_segment+1,
                    "recoveries": orphan_segment, "worker_processes": 0,
                    "elapsed_seconds": float(state.get("elapsed_seconds", 0.0))}
    prior_cursor = _load_cursor(jobroot, identities) if (jobroot / "recovery.latest.json").exists() else None
    fault = _known_worker_fault(jobroot)
    if fault is not None:
        return {"outcome": fault, "segments": int(state.get("segment_index") is not None),
            "recoveries": int(state.get("recoveries", 0)), "worker_processes": int(state.get("worker_pid") is not None),
            "elapsed_seconds": _persist_elapsed(jobroot)}
    if prior_cursor is not None and prior_cursor["phase"] == "complete":
        completed_segment = jobroot / prior_cursor["segment"]
        outcome = _audit_completed_outcome(config, jobroot, prior_cursor, registration=registration,
                                           source_receipt=source_receipt)
        record.append_event("job_completed_from_verified_complete_cursor", {"segment": prior_cursor["segment"],
            "segment_terminal_present": (completed_segment / "manifest.json").is_file(),
            "cursor_sha256": _cursor_pointer(jobroot)["cursor_sha256"]})
        return {"outcome": outcome, "segments": int(prior_cursor["segment"][-3:]) + 1,
                "recoveries": int(prior_cursor["segment"][-3:]), "worker_processes": 0,
                "elapsed_seconds": _persist_elapsed(jobroot)}
    if prior_cursor is None:
        if any(jobroot.glob("segment-[0-9][0-9][0-9]")):
            return {"outcome": "failed_no_verified_checkpoint", "segments": 0, "recoveries": 0,
                    "worker_processes": 0, "elapsed_seconds": 0.0}
        segment_index, recoveries = 0, 0
    else:
        segment_index = int(prior_cursor["segment"][-3:]) + 1
        recoveries = segment_index
    outcome = "total_budget_exhausted" if time.monotonic() >= work_deadline or time.monotonic() >= deadline else "failed"
    started = time.monotonic(); elapsed_before = float(state.get("elapsed_seconds", 0.0)); worker_processes = 0
    while segment_index < 3 and time.monotonic() < work_deadline and time.monotonic() < deadline:
        segment = jobroot / f"segment-{segment_index:03d}"
        from dwr.experiments import gpu_feasibility
        physical_source = source_receipt if _test_factory else gpu_feasibility.validate_clean_source(
            _root(), c["source_base_commit"])
        from dwr.experiments.source_overfit import _environment_receipt
        environment = _test_factory["environment_receipt"] if _test_factory else _environment_receipt()
        if environment != binding.get("environment_receipt") or \
                _sha(_canonical(environment)) != binding.get("environment_receipt_sha256"):
            raise ValueError("new worker runtime differs from the immutable first-launch binding")
        descriptor = {"schema_version": 1, "jobroot": str(jobroot.resolve()),
            "segment_index": segment_index, "config_sha256": CONFIG_SHA256,
            "source_commit": source_commit, "supervisor_pid": os.getpid(),
            "supervisor_creation_time": _process_creation_time(os.getpid()), "nonce": uuid.uuid4().hex,
            "resume_descriptor": _cursor_pointer(jobroot) if segment_index else None,
            "physical_source_receipt": physical_source,
            "physical_source_receipt_sha256": _sha(_canonical(physical_source)),
            "environment_receipt": environment,
            "environment_receipt_sha256": _sha(_canonical(environment)),
            "created_utc": _utc(), "work_deadline_epoch": time.time() + max(0.0, work_deadline-time.monotonic()),
            "seal_deadline_epoch": _parse_utc(binding["started_utc"]).timestamp() + 5400}
        if segment_index and descriptor["resume_descriptor"] is None:
            state.update({"state": "terminal", "outcome": "failed_no_verified_checkpoint"})
            break
        ledger = _reserve_physical_work(jobroot, c, segment_index)
        record.append_event("physical_work_reserved", {"segment_index": segment_index,
            "ledger_sha256": ledger["ledger_sha256"], "reserved": ledger["reserved"]})
        descriptor_raw = _canonical(descriptor) + b"\n"
        descriptor_path = jobroot / f"worker-descriptor-{segment_index:03d}.json"
        with descriptor_path.open("xb") as stream:
            stream.write(descriptor_raw); stream.flush(); os.fsync(stream.fileno())
        descriptor_sha = _sha(descriptor_raw)
        stdout_path = jobroot / f"worker-{segment_index:03d}.stdout.log"
        stderr_path = jobroot / f"worker-{segment_index:03d}.stderr.log"
        worker_command = (_test_factory["worker_command"](descriptor_path, descriptor_sha) if _test_factory else
            [sys.executable, "-m", "dwr.experiments.source_overfit", "--config", str(config),
             "--descriptor", str(descriptor_path), "--descriptor-sha256", descriptor_sha])
        with stdout_path.open("ab") as stdout_stream, stderr_path.open("ab") as stderr_stream:
            child = subprocess.Popen(worker_command, cwd=_root(), stdin=subprocess.DEVNULL,
                stdout=stdout_stream, stderr=stderr_stream,
                creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) |
                               getattr(subprocess, "CREATE_NO_WINDOW", 0)))
        worker_processes += 1
        launcher_creation = _process_creation_time(child.pid)
        launcher_receipt = {"schema_version": 1, "pid": child.pid, "creation_time": launcher_creation,
            "image": _process_image(child.pid), "descriptor_nonce": descriptor["nonce"],
            "supervisor_pid": os.getpid(), "supervisor_creation_time": _process_creation_time(os.getpid())}
        _atomic_json(jobroot/f"worker-{segment_index:03d}.launcher-receipt.json", launcher_receipt)
        identity_path = jobroot / f"worker-{segment_index:03d}.identity.json"
        state.update({"state": "worker_starting", "worker_pid": None, "worker_creation_time": None,
            "worker_launcher_pid": child.pid, "worker_launcher_creation_time": launcher_creation,
            "worker_identity_path": str(identity_path), "worker_nonce": descriptor["nonce"],
            "segment_index": segment_index, "heartbeat_utc": _utc(), "worker_stdout": str(stdout_path),
            "worker_stderr": str(stderr_path)})
        _atomic_json(jobroot / "status.json", state)
        handshake_deadline = min(work_deadline, time.monotonic() + 15)
        while not identity_path.is_file() and time.monotonic() < handshake_deadline:
            if child.poll() is not None:
                break
            time.sleep(0.05)
        if not identity_path.is_file():
            if _pid_matches(child.pid, launcher_creation):
                _terminate_owned_process(child.pid, launcher_creation)
            child.wait(timeout=10)
            raise RuntimeError("worker did not publish its actual-process identity handshake")
        worker_receipt = _read_json(identity_path)
        worker_pid, worker_creation = worker_receipt.get("pid"), worker_receipt.get("creation_time")
        if (worker_receipt.get("descriptor_nonce") != descriptor["nonce"] or
                worker_receipt.get("config_sha256") != CONFIG_SHA256 or
                worker_receipt.get("source_commit") != source_commit or
                worker_receipt.get("segment_index") != segment_index or
                type(worker_pid) is not int or not _pid_matches(worker_pid, worker_creation) or
                _process_image(worker_pid) != worker_receipt.get("image") or
                Path(worker_receipt["image"]).name.lower() not in {"python.exe", "python", "python3", "python3.exe"}):
            raise ValueError("worker actual-process identity handshake is invalid")
        parent_pid = worker_receipt.get("parent_pid")
        if parent_pid not in {child.pid, os.getpid()}:
            raise ValueError("worker actual-process parent is not the launched process tree")
        if parent_pid == child.pid and worker_receipt.get("parent_creation_time") != launcher_creation:
            raise ValueError("worker launcher parent creation identity changed")
        try:
            lock_probe = _ByteLock(worker_lock_path, 1, wait=False)
        except RuntimeError:
            pass
        else:
            lock_probe.close()
            raise ValueError("worker identity handshake arrived without actual worker lock ownership")
        record.append_event("worker_started", {"segment_index": segment_index, "pid": worker_pid,
            "creation_time": worker_creation, "launcher_pid": child.pid, "launcher_creation_time": launcher_creation,
            "descriptor_sha256": descriptor_sha, "identity_sha256": _sha(identity_path.read_bytes()),
            "stdout": str(stdout_path), "stderr": str(stderr_path)})
        if _test_factory and callable(_test_factory.get("after_identity")):
            _test_factory["after_identity"](segment_index, worker_receipt)
        state.update({"state": "running", "worker_pid": worker_pid, "worker_creation_time": worker_creation,
            "worker_launcher_pid": child.pid, "worker_launcher_creation_time": launcher_creation,
            "worker_identity_path": str(identity_path), "worker_nonce": descriptor["nonce"]})
        _atomic_json(jobroot / "status.json", state)
        while _pid_matches(worker_pid, worker_creation) or child.poll() is None:
            cutoff = _worker_cutoff(time.monotonic(), "seal", work_deadline, seal_deadline, deadline)
            if cutoff == "outer_watchdog":
                _terminate_worker_tree(worker_receipt, child.pid, launcher_creation)
                break
            if time.monotonic() >= work_deadline:
                try: stage = _read_json(jobroot/f"worker-{segment_index:03d}.heartbeat.json").get("stage", "work")
                except Exception: stage = "work"
                cutoff = _worker_cutoff(time.monotonic(), stage, work_deadline, seal_deadline, deadline)
                if cutoff:
                    record.append_event("registered_seal_cutoff" if stage == "seal" else "registered_work_cutoff",
                        {"segment_index": segment_index, "pid": worker_pid,
                         "creation_time": worker_creation, "utc": _utc()})
                    _terminate_worker_tree(worker_receipt, child.pid, launcher_creation)
                    break
            heartbeat_path = jobroot / f"worker-{segment_index:03d}.heartbeat.json"
            if heartbeat_path.exists():
                try:
                    heartbeat = _read_json(heartbeat_path)
                    heartbeat_age = time.time() - heartbeat["utc_epoch"]
                except Exception:
                    heartbeat_age = float("inf")
                if heartbeat_age > 180:
                    record.append_event("stale_worker_heartbeat", {"segment_index": segment_index,
                        "pid": worker_pid, "creation_time": worker_creation, "age_seconds": heartbeat_age})
                    _terminate_worker_tree(worker_receipt, child.pid, launcher_creation)
                    break
            elif time.time() - _parse_utc(descriptor["created_utc"]).timestamp() > 180:
                record.append_event("missing_worker_heartbeat", {"segment_index": segment_index,
                    "pid": worker_pid, "creation_time": worker_creation})
                _terminate_worker_tree(worker_receipt, child.pid, launcher_creation)
                break
            cursor = _load_cursor(jobroot, identities)
            if cursor:
                state["committed_step"] = cursor["progress"]["optimizer_step"]
            state["heartbeat_utc"] = _utc(); state["elapsed_seconds"] = _persist_elapsed(jobroot)
            _atomic_json(jobroot / "status.json", state)
            time.sleep(min(5.0, max(0.2, deadline-time.monotonic())))
        if child.poll() is None:
            child.wait(timeout=10)
        terminal = segment / "manifest.json"
        record.append_event("worker_exited", {"segment_index": segment_index, "return_code": child.returncode,
            "terminal_present": terminal.is_file(), "utc": _utc()})
        if terminal.is_file():
            verified = verify_artifacts(segment)
            if verified["status"] == "completed":
                fault = _known_worker_fault(jobroot)
                if fault is not None:
                    outcome = fault
                    break
                outcome_path = segment / "outcome.json"
                outcome_raw = outcome_path.read_bytes()
                worker_outcome = json.loads(outcome_raw.decode("utf-8"))
                if outcome_raw != _canonical(worker_outcome) + b"\n":
                    outcome = "corrupt_worker_outcome"; break
                cursor = _load_cursor(jobroot, identities)
                if (cursor and cursor["phase"] == "complete" and
                        worker_outcome.get("config_sha256") == registration["config_sha256"]):
                    from dwr.training.overfit import verify_evaluation_receipt
                    final_step = cursor["progress"]["optimizer_step"]
                    ids = [item["example_id"] for item in registration["selected"]]
                    if worker_outcome.get("example_ids") != ids:
                        outcome = "failed_independent_terminal_audit"; break
                    initial = verify_evaluation_receipt(jobroot,
                        worker_outcome.get("initial_evaluation_receipt"), expected_step=0,
                        expected_example_ids=ids, expected_checkpoint_sha256=worker_outcome.get("initial_state_sha256"))
                    final = verify_evaluation_receipt(jobroot,
                        worker_outcome.get("final_evaluation_receipt"), expected_step=final_step,
                        expected_example_ids=ids, expected_checkpoint_sha256=cursor["checkpoint_sha256"])
                    ratio = final["ce"] / initial["ce"]
                    if not math.isfinite(ratio) or not math.isfinite(worker_outcome.get("final_to_initial_ce_ratio", float("nan"))):
                        outcome = "failed_independent_terminal_audit"; break
                    threshold = c["evaluation"]["gate_max_final_to_initial_CE_ratio"]
                    expected = "PASS_TINY_SOURCE_OVERFIT" if final["valid"] == 3 and ratio <= threshold else "NO_GO_TINY_SOURCE_OVERFIT"
                    stored = cursor["facts"].get("final_result", {})
                    if (worker_outcome.get("status") != expected or worker_outcome.get("valid_numerator") != final["valid"] or
                            worker_outcome.get("valid_denominator") != 3 or
                            not math.isclose(worker_outcome.get("initial_ce", initial["ce"]), initial["ce"], rel_tol=1e-12, abs_tol=1e-12) or
                            not math.isclose(worker_outcome.get("final_ce", final["ce"]), final["ce"], rel_tol=1e-12, abs_tol=1e-12) or
                            not math.isclose(worker_outcome.get("final_to_initial_ce_ratio", float("nan")), ratio, rel_tol=1e-12, abs_tol=1e-12) or
                            worker_outcome.get("initial_evaluation_receipt") != cursor["facts"].get("initial_evaluation_receipt") or
                            worker_outcome.get("final_evaluation_receipt") != stored.get("final_evaluation_receipt") or
                            worker_outcome.get("net_optimizer_steps") != final_step or stored.get("status") != expected):
                        outcome = "failed_independent_terminal_audit"; break
                    outcome = _audit_completed_outcome(config, jobroot, cursor, registration=registration,
                                                       source_receipt=source_receipt)
                    if outcome != expected:
                        outcome = "failed_independent_terminal_audit"
                    break
                outcome = "failed_completed_record_without_complete_cursor"; break
            outcome = verified["error"] or "worker_failed"; break
        cursor = _load_cursor(jobroot, identities)
        fault = _known_worker_fault(jobroot)
        if fault is not None:
            outcome = fault
            break
        if cursor is not None and cursor["phase"] == "complete":
            outcome = _audit_completed_outcome(config, jobroot, cursor, registration=registration,
                                               source_receipt=source_receipt)
            record.append_event("job_completed_from_verified_complete_cursor", {"segment": cursor["segment"],
                "segment_terminal_present": False, "cursor_sha256": _cursor_pointer(jobroot)["cursor_sha256"]})
            return {"outcome": outcome, "segments": int(cursor["segment"][-3:]) + 1,
                    "recoveries": int(cursor["segment"][-3:]), "worker_processes": worker_processes,
                    "elapsed_seconds": _persist_elapsed(jobroot)}
        recoverable = cursor is not None and recoveries < 2 and worker_processes < 3 and time.monotonic() < work_deadline
        if not recoverable:
            outcome = ("total_budget_exhausted" if time.monotonic() >= work_deadline or
                       time.monotonic() >= deadline else "interrupted_without_verified_cursor")
            break
        recoveries += 1; segment_index += 1
        state.update({"state": "recovering", "worker_pid": None, "worker_creation_time": None,
            "recoveries": recoveries, "committed_step": cursor["progress"]["optimizer_step"]})
        _atomic_json(jobroot / "status.json", state)
    return {"outcome": outcome, "segments": min(segment_index+1, 3), "recoveries": recoveries,
            "worker_processes": worker_processes, "elapsed_seconds": elapsed_before+time.monotonic()-started}


def _pid_matches(pid: int, creation_time: str) -> bool:
    try:
        if _process_creation_time(pid) != creation_time:
            return False
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            process = kernel32.OpenProcess(0x1000, False, pid)
            if not process:
                return False
            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(process)
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        return stat_text[stat_text.rfind(")") + 2:].split()[0] != "Z"
    except (OSError, ValueError):
        return False


def _worker_cutoff(now: float, stage: str, work_deadline: float,
                   seal_deadline: float, watchdog_deadline: float) -> str | None:
    if now >= watchdog_deadline:
        return "outer_watchdog"
    if now >= seal_deadline and stage == "seal":
        return "seal_cutoff"
    if now >= work_deadline and stage != "seal":
        return "work_cutoff"
    return None


def _discover_live_worker(jobroot: Path, lock_path: Path) -> dict | None:
    """Reconcile durable launch/worker receipts and byte-1 ownership before status is read."""
    live = []
    for identity_path in sorted(jobroot.glob("worker-[0-9][0-9][0-9].identity.json")):
        try:
            identity = _read_json(identity_path)
            segment_index = identity["segment_index"]
            descriptor = _read_json(jobroot/f"worker-{segment_index:03d}.descriptor.json")
        except FileNotFoundError:
            # Correct descriptor name is worker-descriptor-NNN.json.
            try:
                identity = _read_json(identity_path)
                segment_index = identity["segment_index"]
                descriptor = _read_json(jobroot/f"worker-descriptor-{segment_index:03d}.json")
            except Exception:
                continue
        except Exception:
            continue
        if not _pid_matches(identity.get("pid"), identity.get("creation_time")):
            continue
        if (identity.get("config_sha256") != CONFIG_SHA256 or
                identity.get("descriptor_nonce") != descriptor.get("nonce") or
                identity.get("source_commit") != descriptor.get("source_commit") or
                identity.get("segment_index") != segment_index or
                _process_image(identity["pid"]) != identity.get("image")):
            raise ValueError("live actual worker identity receipt is inconsistent")
        launcher_path = jobroot/f"worker-{segment_index:03d}.launcher-receipt.json"
        if not launcher_path.is_file():
            raise RuntimeError("worker lock owner has no durable launcher receipt; refusing duplicate launch")
        launcher = _read_json(launcher_path)
        if (launcher.get("descriptor_nonce") != descriptor.get("nonce") or
                launcher.get("supervisor_pid") != descriptor.get("supervisor_pid") or
                launcher.get("supervisor_creation_time") != descriptor.get("supervisor_creation_time") or
                identity.get("parent_pid") not in {launcher.get("pid"), descriptor.get("supervisor_pid")}):
            raise ValueError("live worker and launcher receipts do not share the issued nonce/owner")
        expected_parent_creation = (launcher.get("creation_time") if identity["parent_pid"] == launcher.get("pid")
                                    else descriptor.get("supervisor_creation_time"))
        if identity.get("parent_creation_time") != expected_parent_creation:
            raise ValueError("live worker parent creation receipt differs from its launcher/supervisor identity")
        if _pid_matches(launcher.get("pid"), launcher.get("creation_time")) and \
                _process_image(launcher["pid"]) != launcher.get("image"):
            raise ValueError("live worker launcher image differs from its durable receipt")
        live.append((identity, descriptor, launcher))
    try:
        probe = _ByteLock(lock_path, 1, wait=False)
    except RuntimeError:
        locked = True
    else:
        probe.close(); locked = False
    if locked and not live:
        raise RuntimeError("worker byte lock is owned but no live verified identity receipt exists")
    if not locked and live:
        raise RuntimeError("live worker identity exists without byte-1 OS-lock ownership")
    if len(live) > 1:
        raise RuntimeError("multiple live registered worker identities found")
    if not live:
        return None
    identity, descriptor, launcher = live[0]
    return {"pid": identity["pid"], "creation_time": identity["creation_time"],
        "segment_index": identity["segment_index"], "nonce": descriptor["nonce"],
        "launcher_pid": launcher["pid"], "launcher_creation_time": launcher["creation_time"],
        "supervisor_pid": descriptor["supervisor_pid"],
        "supervisor_creation_time": descriptor["supervisor_creation_time"]}


def _wait_orphan_worker(jobroot: Path, record: RunRecord, state: dict, orphan: dict,
                        deadline: float, work_deadline: float, seal_deadline: float, *,
                        _test_factory: dict | None = None) -> str | None:
    pid, creation = orphan["pid"], orphan["creation_time"]
    heartbeat_path = jobroot / f"worker-{orphan['segment_index']:03d}.heartbeat.json"
    last_event = time.monotonic()
    while _pid_matches(pid, creation):
        heartbeat = {}
        if heartbeat_path.is_file():
            try: heartbeat = _read_json(heartbeat_path)
            except Exception: heartbeat = {}
        if _worker_cutoff(time.monotonic(), heartbeat.get("stage", "work"),
                          work_deadline, seal_deadline, deadline):
            _terminate_worker_tree(_read_json(jobroot / f"worker-{orphan['segment_index']:03d}.identity.json"),
                orphan["launcher_pid"], orphan["launcher_creation_time"])
            record.append_event("orphan_worker_cutoff_terminated", {"pid": pid, "creation_time": creation,
                "segment_index": orphan["segment_index"], "utc": _utc()})
            break
        stale = False
        if heartbeat_path.is_file():
            try:
                heartbeat = _read_json(heartbeat_path)
                stale = time.time()-heartbeat["utc_epoch"] > 180
            except Exception:
                stale = True
        else:
            descriptor = _read_json(jobroot/f"worker-descriptor-{orphan['segment_index']:03d}.json")
            stale = time.time()-_parse_utc(descriptor["created_utc"]).timestamp() > 180
        if stale:
            _terminate_worker_tree(_read_json(jobroot / f"worker-{orphan['segment_index']:03d}.identity.json"),
                orphan["launcher_pid"], orphan["launcher_creation_time"])
            record.append_event("orphan_worker_stale_terminated", {"pid": pid,
                "creation_time": creation, "segment_index": orphan["segment_index"], "utc": _utc()})
            break
        if time.monotonic()-last_event >= 5:
            cursor_path = jobroot / "recovery.latest.json"
            if cursor_path.is_file():
                try:
                    cursor_raw = cursor_path.read_bytes()
                    cursor = json.loads(cursor_raw.decode("utf-8"))
                    state["committed_step"] = cursor.get("progress", {}).get("optimizer_step", 0)
                except Exception:
                    pass
            state["heartbeat_utc"] = _utc(); state["elapsed_seconds"] = _persist_elapsed(jobroot)
            _atomic_json(jobroot / "status.json", state)
            last_event = time.monotonic()
        time.sleep(1)
    record.append_event("orphan_worker_exited", {"pid": pid, "creation_time": creation,
        "segment_index": orphan["segment_index"], "terminal_present":
        (jobroot / f"segment-{orphan['segment_index']:03d}" / "manifest.json").is_file(), "utc": _utc()})
    segment = jobroot / f"segment-{orphan['segment_index']:03d}"
    terminal_path = segment / "manifest.json"
    if terminal_path.is_file():
        terminal = verify_artifacts(segment)
        if terminal["status"] != "completed":
            return terminal.get("error") or "orphan_worker_failed"
        cursor = jobroot / "recovery.latest.json"
        if cursor.is_file():
            try:
                data = json.loads(cursor.read_text(encoding="utf-8"))
                from dwr.experiments.source_overfit import _kernel_identities, _load_registration
                registration = (_test_factory["registration"] if _test_factory else
                                _load_registration(_root(), _root() / CONFIG_PATH))
                model = registration["feasibility_config"]["model"]
                identities = _kernel_identities(registration["config_raw"], registration["config"],
                    source_commit=_read_json(jobroot / "job-binding.json")["source_commit"],
                    weight_sha256=model["weight_sha256"],
                    tokenizer_inventory_sha256=model["artifact_inventory_sha256"])
                verified = _load_cursor(jobroot, identities)
                if verified and verified["phase"] == "complete":
                    fault = _known_worker_fault(jobroot)
                    if fault is not None:
                        return fault
                    audit_config = _test_factory["config_path"] if _test_factory else _root() / CONFIG_PATH
                    audit_source = _test_factory["source_receipt"] if _test_factory else None
                    return _audit_completed_outcome(audit_config, jobroot, verified,
                        registration=registration, source_receipt=audit_source)
            except Exception as error:
                record.append_event("orphan_result_audit_failed", {"type": type(error).__name__,
                    "message": str(error), "segment_index": orphan["segment_index"]})
                return f"failed_orphan_result_audit: {type(error).__name__}: {error}"
        return "failed_completed_orphan_without_complete_cursor"
    return None


def _terminate_owned_process(pid: int, creation_time: str) -> None:
    if not _pid_matches(pid, creation_time):
        return
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        process = kernel32.OpenProcess(0x0001 | 0x00100000 | 0x1000, False, pid)
        if not process:
            error = ctypes.get_last_error()
            if not _pid_matches(pid, creation_time):
                return
            raise OSError(error, "cannot open owned orphan worker")
        try:
            if _process_creation_time(pid) != creation_time:
                return
            if not kernel32.TerminateProcess(process, 1):
                error = ctypes.get_last_error()
                if kernel32.WaitForSingleObject(process, 0) == 0 or not _pid_matches(pid, creation_time):
                    return
                raise OSError(error, "cannot terminate owned orphan worker")
            if kernel32.WaitForSingleObject(process, 10000) != 0:
                raise TimeoutError("owned process did not exit after TerminateProcess")
        finally:
            kernel32.CloseHandle(process)
    else:
        import signal
        if _pid_matches(pid, creation_time):
            os.kill(pid, signal.SIGTERM)


def _terminate_worker_tree(receipt: dict, launcher_pid: int, launcher_creation: str) -> None:
    pid, creation = receipt.get("pid"), receipt.get("creation_time")
    if type(pid) is not int or not isinstance(creation, str) or not _pid_matches(pid, creation):
        raise ValueError("refusing to terminate worker without matching actual PID creation identity")
    if _process_image(pid) != receipt.get("image"):
        raise ValueError("refusing to terminate worker whose executable image changed")
    _terminate_owned_process(pid, creation)
    if _pid_matches(launcher_pid, launcher_creation):
        _terminate_owned_process(launcher_pid, launcher_creation)
    until = time.monotonic() + 10
    while (_pid_matches(pid, creation) or _pid_matches(launcher_pid, launcher_creation)) and time.monotonic() < until:
        time.sleep(0.05)
    if _pid_matches(pid, creation) or _pid_matches(launcher_pid, launcher_creation):
        raise TimeoutError("owned worker process tree did not exit after verified termination")


def _cleanup_worker_from_status(jobroot: Path, state: dict) -> None:
    launcher_pid = state.get("worker_launcher_pid")
    launcher_creation = state.get("worker_launcher_creation_time")
    if type(launcher_pid) is not int or not isinstance(launcher_creation, str):
        return
    receipt_path = jobroot / f"worker-{int(state.get('segment_index', 0)):03d}.identity.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        if (receipt.get("descriptor_nonce") != state.get("worker_nonce") or
                receipt.get("config_sha256") != CONFIG_SHA256 or
                receipt.get("pid") != state.get("worker_pid") or
                receipt.get("creation_time") != state.get("worker_creation_time")):
            raise ValueError("worker cleanup identity receipt differs from durable status")
        if _pid_matches(receipt.get("pid"), receipt.get("creation_time")):
            _terminate_worker_tree(receipt, launcher_pid, launcher_creation)
    elif _pid_matches(launcher_pid, launcher_creation):
        _terminate_owned_process(launcher_pid, launcher_creation)
        until = time.monotonic() + 10
        while _pid_matches(launcher_pid, launcher_creation) and time.monotonic() < until:
            time.sleep(0.05)
        if _pid_matches(launcher_pid, launcher_creation):
            raise TimeoutError("unhandshaken launched process did not exit after termination")


def _audit_completed_outcome(config: Path, jobroot: Path, cursor: dict, *,
                             registration: dict | None = None, source_receipt: dict | None = None) -> str:
    from dwr.training.overfit import verify_evaluation_receipt
    from dwr.experiments.source_overfit import _load_registration
    from dwr.experiments.records import _validate_events
    registration = registration or _load_registration(_root(), config)
    ids = [item["example_id"] for item in registration["selected"]]
    c = registration["config"]
    expected_steps = c["training"]["optimizer_steps"]
    probe_step = c["resume"]["checkpoint_step"]
    if cursor.get("phase") != "complete" or cursor.get("progress", {}).get("optimizer_step") != expected_steps:
        raise ValueError("complete cursor does not assert the registered net update count")
    source = source_receipt or __import__("dwr.experiments.gpu_feasibility", fromlist=["validate_clean_source"]).validate_clean_source(
        _root(), c["source_base_commit"])
    binding = _read_json(jobroot / "job-binding.json")
    if source.get("commit") != binding.get("source_commit") or source.get("files") != binding.get("physical_source_files"):
        raise ValueError("completed cursor source commit differs from immutable first-launch source")
    model = registration["feasibility_config"]["model"]
    segment = jobroot / cursor["segment"]
    events_raw = (segment / "events.jsonl").read_bytes()
    _validate_events(events_raw)
    events = [json.loads(line.decode("utf-8")) for line in events_raw.splitlines()]
    audit_refs = [event["payload"] for event in events if event["kind"] == "controller_audit_receipt"]
    if len(audit_refs) != 1:
        raise ValueError("complete cursor lacks exactly one durable controller audit event")
    audit_ref = audit_refs[0]
    audit_rel = audit_ref.get("path")
    if _safe_job_file_sha(jobroot, audit_rel) != audit_ref.get("sha256"):
        raise ValueError("controller audit receipt path or digest is invalid")
    audit_path = jobroot / audit_rel
    audit_raw = audit_path.read_bytes()
    audit = json.loads(audit_raw.decode("utf-8"))
    if audit_raw != _canonical(audit) + b"\n":
        raise ValueError("controller audit receipt is not canonical stable JSON")
    facts = cursor["facts"]
    initial_hash_ref = audit.get("initial_hash_receipt")
    if _safe_job_file_sha(jobroot, initial_hash_ref.get("path")) != initial_hash_ref.get("sha256"):
        raise ValueError("pre-kernel original-parameter hash receipt failed verification")
    initial_hash = _read_json(jobroot / initial_hash_ref["path"])
    if (audit.get("source_commit") != binding["source_commit"] or
            audit.get("config_sha256") != registration["config_sha256"] or audit.get("model_sha256") != model["weight_sha256"] or
            audit.get("initial_state_sha256") != facts.get("initial_state_sha256") or
            audit.get("final_checkpoint_sha256") != cursor.get("checkpoint_sha256") or
            audit.get("original_parameters_unchanged") is not True or
            audit.get("original_parameter_hashes_after") != initial_hash.get("original_parameter_hashes") or
            audit.get("runtime_costs", {}).get("process_peak_working_set_mib") is None or
            audit["runtime_costs"]["process_peak_working_set_mib"] > c["limits"]["resident_memory_mib"] or
            (audit["runtime_costs"].get("peak_allocated_mib") is not None and
             audit["runtime_costs"]["peak_allocated_mib"] > c["limits"]["maximum_peak_allocated_mib"]) or
            (audit["runtime_costs"].get("peak_reserved_mib") is not None and
             audit["runtime_costs"]["peak_reserved_mib"] > c["limits"]["maximum_peak_reserved_mib"])):
        raise ValueError("controller audit does not bind source/config/model/checkpoint and unchanged originals")
    if (initial_hash.get("source_commit") != binding["source_commit"] or
            initial_hash.get("config_sha256") != registration["config_sha256"] or
            initial_hash.get("model_sha256") != model["weight_sha256"] or
            initial_hash.get("original_parameter_hashes") != audit.get("original_parameter_hashes_after")):
        raise ValueError("pre-kernel original-parameter receipt identity mismatch")
    stored = cursor["facts"].get("final_result")
    if not isinstance(stored, dict):
        raise ValueError("complete cursor has no bound final result")
    initial = verify_evaluation_receipt(jobroot, cursor["facts"].get("initial_evaluation_receipt"),
        expected_step=0, expected_example_ids=ids,
        expected_checkpoint_sha256=cursor["facts"].get("initial_state_sha256"))
    final = verify_evaluation_receipt(jobroot, stored.get("final_evaluation_receipt"),
        expected_step=expected_steps,
        expected_example_ids=ids,
        expected_checkpoint_sha256=cursor["checkpoint_sha256"])
    ratio = final["ce"] / initial["ce"]
    probe = facts.get("resume_probe")
    if (not isinstance(probe, dict) or probe.get("step") != probe_step or
            probe.get("checkpoint_sha256") != _safe_job_file_sha(jobroot, probe.get("checkpoint_path")) or
            probe.get("sha256") != _safe_job_file_sha(jobroot, probe.get("path"))):
        raise ValueError("complete cursor resume-probe evidence or checkpoint hash is invalid")
    probe_result = probe.get("result", {})
    tolerance = c["resume"]
    resume_receipt = _verified_resume_event(jobroot, cursor, tolerance)
    if (probe_result.get("first_replayed_ids") != probe_result.get("second_replayed_ids") or
            not isinstance(probe_result.get("first_replayed_ids"), list) or
            len(probe_result["first_replayed_ids"]) != c["training"]["batch_size"] or
            resume_receipt.get("verdict") is not True or
            resume_receipt.get("source_sha256") != cursor["identities"]["source"] or
            resume_receipt.get("config_sha256") != cursor["identities"]["config"] or
            resume_receipt.get("checkpoint_sha256") != probe.get("checkpoint_sha256") or
            resume_receipt.get("probe_state_sha256") != probe.get("sha256") or
            any(resume_receipt.get(key) != probe_result.get(key) for key in (
                "first_replayed_ids", "second_replayed_ids", "max_parameter_abs_error",
                "max_parameter_relative_error", "max_optimizer_abs_error", "max_optimizer_relative_error",
                "max_loss_abs_error", "max_loss_relative_error", "max_scheduler_abs_error",
                "max_rng_abs_error", "max_sampler_abs_error")) or
            probe_result.get("max_scheduler_abs_error") != 0 or
            probe_result.get("max_rng_abs_error") != 0 or probe_result.get("max_sampler_abs_error") != 0):
        raise ValueError("complete cursor resume probe exceeds registered replay tolerance")
    expected = "PASS_TINY_SOURCE_OVERFIT" if final["valid"] == 3 and ratio <= c["evaluation"]["gate_max_final_to_initial_CE_ratio"] else "NO_GO_TINY_SOURCE_OVERFIT"
    if stored.get("status") != expected or stored.get("valid_numerator") != final["valid"] or \
            stored.get("valid_denominator") != 3 or stored.get("net_optimizer_steps") != expected_steps or \
            not math.isclose(stored.get("final_to_initial_ce_ratio", float("nan")), ratio, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("complete cursor fails independent final evaluation audit")
    return expected


def _verified_resume_event(jobroot: Path, cursor: dict, tolerance: dict) -> dict:
    from dwr.experiments.records import _validate_events
    facts = cursor.get("facts")
    probe = facts.get("resume_probe") if isinstance(facts, dict) else None
    if not isinstance(probe, dict):
        raise ValueError("complete cursor has no verified resume probe reference")
    probe_path, checkpoint_path = probe.get("path"), probe.get("checkpoint_path")
    probe_digest = probe.get("sha256")
    checkpoint_digest = probe.get("checkpoint_sha256")
    if (_safe_job_file_sha(jobroot, probe_path) != probe_digest or
            _safe_job_file_sha(jobroot, checkpoint_path) != checkpoint_digest):
        raise ValueError("resume probe evidence digest differs from the complete cursor")
    probe_segment = PurePosixPath(probe_path).parts[0]
    checkpoint_segment = PurePosixPath(checkpoint_path).parts[0]
    if (not re.fullmatch(r"segment-[0-9]{3}", probe_segment) or
            checkpoint_segment != probe_segment):
        raise ValueError("resume probe and checkpoint must share a contained producing segment")
    raw = (jobroot / probe_segment / "events.jsonl").read_bytes()
    _validate_events(raw)
    parsed = [json.loads(line.decode("utf-8")) for line in raw.splitlines()]
    matches = [event["payload"] for event in parsed if event["kind"] == "resume_verified"]
    if not matches or any(item != matches[0] for item in matches[1:]):
        raise ValueError("complete cursor requires consistent durable resume_verified evidence")
    receipt = matches[0]
    identities = cursor.get("identities")
    metrics = probe.get("result")
    expected_bindings = {
        "source_sha256": identities.get("source") if isinstance(identities, dict) else None,
        "config_sha256": identities.get("config") if isinstance(identities, dict) else None,
        "probe_state_sha256": probe_digest,
        "checkpoint_sha256": checkpoint_digest,
    }
    if any(receipt.get(key) != value for key, value in expected_bindings.items()) or not isinstance(metrics, dict):
        raise ValueError("resume_verified event does not bind the cursor source/config/probe evidence")
    metric_keys = ("max_parameter_abs_error", "max_parameter_relative_error", "max_optimizer_abs_error",
        "max_optimizer_relative_error", "max_loss_abs_error", "max_loss_relative_error",
        "max_scheduler_abs_error", "max_rng_abs_error", "max_sampler_abs_error")
    if any(receipt.get(key) != metrics.get(key) for key in metric_keys):
        raise ValueError("resume_verified metrics differ from the stored probe result")
    if (receipt.get("verdict") is not True or receipt.get("atol") != tolerance["float_atol"] or
            receipt.get("rtol") != tolerance["float_rtol"] or
            any(type(receipt.get(key)) not in (int, float) or not math.isfinite(receipt[key]) or receipt[key] < 0
                for key in ("max_parameter_abs_error", "max_parameter_relative_error",
                    "max_optimizer_abs_error", "max_optimizer_relative_error", "max_loss_abs_error",
                    "max_loss_relative_error", "max_scheduler_abs_error", "max_rng_abs_error",
                    "max_sampler_abs_error"))):
        raise ValueError("resume_verified event tolerance/verdict differs from registration")
    return receipt


def _safe_job_file_sha(jobroot: Path, relative: str) -> str:
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute() or
            PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts or
            "\\" in relative or Path(relative).drive):
        raise ValueError("job evidence reference path must be relative and contained")
    path = jobroot.joinpath(*PurePosixPath(relative).parts)
    try:
        path.resolve().relative_to(jobroot.resolve())
    except ValueError as error:
        raise ValueError("job evidence reference path escapes the job root") from error
    _reject_links(path)
    if not path.is_file():
        raise ValueError("job evidence reference is missing")
    return _sha(path.read_bytes())


def _known_worker_fault(jobroot: Path) -> str | None:
    for segment in sorted(jobroot.glob("segment-[0-9][0-9][0-9]")):
        if (segment / "worker-failure.json").is_file():
            return "known_worker_fault_no_recovery"
        terminal = segment / "manifest.json"
        if terminal.is_file():
            outcome = verify_artifacts(segment)
            if outcome.get("status") != "completed":
                return "known_terminal_worker_failure_no_recovery"
        events_path = segment / "events.jsonl"
        if events_path.is_file():
            raw = events_path.read_bytes()
            from dwr.experiments.records import _validate_events
            _validate_events(raw)
            events = [json.loads(line.decode("utf-8")) for line in raw.splitlines()]
            for event in events:
                if event["kind"] == "worker_failure":
                    return "known_worker_failure_event_no_recovery"
                if event["kind"] in {"gradient_invariant_failure", "numerical_divergence"}:
                    return "known_kernel_fault_no_recovery"
    return None


def _within_registered_tolerance(actual, expected, *, atol: float, rtol: float) -> bool:
    """Apply the registered elementwise additive replay rule without requiring NumPy/Torch."""
    try:
        if len(actual) != len(expected):
            return False
        for value, reference in zip(actual, expected):
            value, reference = float(value), float(reference)
            if not math.isfinite(value) or not math.isfinite(reference):
                return False
            if abs(value-reference) > atol + rtol*abs(reference):
                return False
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def _aggregate_physical_work(jobroot: Path, config: dict) -> dict:
    from dwr.experiments.records import _validate_events
    completed_updates = started_updates = prediction_started = prediction_completed = decoder_calls = 0
    for segment in sorted(jobroot.glob("segment-[0-9][0-9][0-9]")):
        path = segment / "events.jsonl"
        if not path.is_file():
            continue
        raw = path.read_bytes(); _validate_events(raw)
        for line in raw.splitlines():
            event = json.loads(line.decode("utf-8")); kind = event["kind"]; payload = event["payload"]
            if kind == "update_started": started_updates += 1
            elif kind == "update": completed_updates += 1
            elif kind == "prediction_started": prediction_started += 1
            elif kind == "prediction_completed":
                prediction_completed += 1
                decoder_calls += payload.get("decoder_calls", 0)
    job_path = jobroot / "events.jsonl"
    started_workers = exited_workers = 0
    if job_path.is_file():
        raw = job_path.read_bytes(); _validate_events(raw)
        for line in raw.splitlines():
            event = json.loads(line.decode("utf-8")); kind = event["kind"]
            if kind == "worker_started": started_workers += 1
            elif kind in {"worker_exited", "orphan_worker_exited"}: exited_workers += 1
    unresolved_updates = max(0, started_updates-completed_updates)
    live_workers = max(0, started_workers-exited_workers)
    unknown_worker_updates = max(unresolved_updates, live_workers)
    unknown_predictions = max(0, prediction_started-prediction_completed)
    totals = {"observed_physical_updates": completed_updates,
        "started_physical_updates": started_updates,
        "unknown_inflight_worker_updates": unknown_worker_updates,
        "conservative_physical_updates": completed_updates+unknown_worker_updates,
        "prediction_attempts": prediction_started,
        "completed_predictions": prediction_completed,
        "unknown_inflight_predictions": unknown_predictions,
        "conservative_decoder_calls": decoder_calls+unknown_predictions*128,
        "started_workers": started_workers, "exited_workers": exited_workers}
    limits = config["limits"]
    return {"schema_version": 1, "observed": totals, "limits": {
        "physical_updates": limits["maximum_physical_updates"],
        "predictions": limits["maximum_predictions"], "decoder_calls": limits["maximum_decoder_calls"],
        "workers": limits["maximum_worker_processes"]}}


def _reserve_physical_work(jobroot: Path, config: dict, segment_index: int) -> dict:
    totals = _aggregate_physical_work(jobroot, config)["observed"]
    started_workers = totals["started_workers"]
    reserved = {"physical_updates": 1, "predictions": 6, "decoder_calls": 6 * 128, "workers": 1}
    limits = config["limits"]
    if (totals["conservative_physical_updates"] + reserved["physical_updates"] > limits["maximum_physical_updates"] or
            totals["prediction_attempts"] + reserved["predictions"] > limits["maximum_predictions"] or
            totals["conservative_decoder_calls"] + reserved["decoder_calls"] > limits["maximum_decoder_calls"] or
            started_workers + reserved["workers"] > limits["maximum_worker_processes"] or
            segment_index > limits["maximum_operational_recoveries"]):
        raise RuntimeError("whole-job physical work or recovery cap would be exceeded")
    ledger = {"schema_version": 1, "segment_index": segment_index, "observed": totals,
        "reserved": reserved, "limits": {"physical_updates": limits["maximum_physical_updates"],
            "predictions": limits["maximum_predictions"], "decoder_calls": limits["maximum_decoder_calls"],
            "workers": limits["maximum_worker_processes"]}}
    raw = _canonical(ledger) + b"\n"
    _atomic_json(jobroot / "physical-work-ledger.json", ledger)
    return {"ledger_sha256": _sha(raw), "reserved": reserved, "observed": totals}


def _load_cursor(jobroot: Path, identities: dict) -> dict | None:
    path = jobroot / "recovery.latest.json"
    if not path.exists(): return None
    from dwr.training.recovery import RecoveryStore
    descriptor = {"jobroot": str(jobroot.resolve()), "cursor_path": str(path.resolve()),
                  "cursor_sha256": _sha(path.read_bytes())}
    return RecoveryStore.verify_descriptor(descriptor, identities)


def _cursor_pointer(jobroot: Path) -> dict[str, str]:
    path = jobroot / "recovery.latest.json"
    raw = path.read_bytes()
    pointer = {"jobroot": str(jobroot.resolve()), "cursor_path": str(path.resolve()),
               "cursor_sha256": _sha(raw)}
    return pointer


def _job_artifacts(jobroot: Path) -> list[str]:
    return sorted(path.relative_to(jobroot).as_posix() for path in jobroot.rglob("*")
                  if path.is_file() and path.name not in {"manifest.initial.json", "manifest.json", "events.jsonl",
                                                          "elapsed-budget.json"})


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> _dt.datetime:
    if not isinstance(value, str):
        raise ValueError("UTC anchor must be a string")
    try:
        parsed = _dt.datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise ValueError("UTC anchor is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != _dt.timedelta(0):
        raise ValueError("UTC anchor must be timezone-aware UTC")
    return parsed


def _validated_elapsed(last_utc_epoch: float, now_utc_epoch: float, prior_elapsed: float) -> float:
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in (last_utc_epoch, now_utc_epoch, prior_elapsed)):
        raise ValueError("clock receipt values must be finite")
    if now_utc_epoch < last_utc_epoch:
        raise ValueError("clock rollback detected against durable elapsed receipt")
    return max(float(prior_elapsed), float(prior_elapsed) + now_utc_epoch - last_utc_epoch)


def _persist_elapsed(jobroot: Path) -> float:
    path = jobroot / "elapsed-budget.json"
    receipt = _read_json(path)
    now = time.time()
    receipt["elapsed_seconds"] = _validated_elapsed(receipt["last_observed_utc_epoch"], now,
                                                    receipt["elapsed_seconds"])
    receipt["last_observed_utc_epoch"] = now
    _atomic_json(path, receipt)
    return receipt["elapsed_seconds"]


def _write_watchdog_receipt(jobroot: Path, value: dict) -> None:
    path = jobroot / "watchdog-receipt.json"
    raw = _canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def read_worker_descriptor(path: str | os.PathLike[str], expected_sha256: str) -> dict:
    _reject_links(Path(path))
    descriptor_path = Path(path).resolve(strict=True)
    raw = descriptor_path.read_bytes()
    if _sha(raw) != expected_sha256:
        raise ValueError("worker descriptor bytes differ from supervisor-issued digest")
    return _read_json(descriptor_path)


def verify_worker_authorization(root: Path, registration: dict, segment_index: int,
                                descriptor: dict | None) -> dict:
    if type(descriptor) is not dict or descriptor.get("schema_version") != 1:
        raise ValueError("worker requires a supervisor-issued descriptor")
    jobroot = Path(descriptor.get("jobroot", "")).resolve(strict=True)
    expected_jobroot = root.joinpath(*PurePosixPath(JOB_DIR).parts).resolve(strict=True)
    if (jobroot != expected_jobroot or descriptor.get("segment_index") != segment_index or
            descriptor.get("config_sha256") != CONFIG_SHA256):
        raise ValueError("worker descriptor does not bind this registered job/config/source/segment")
    from dwr.experiments import gpu_feasibility
    from dwr.experiments.source_overfit import _environment_receipt
    physical_source = gpu_feasibility.validate_clean_source(root, registration["config"]["source_base_commit"])
    if (descriptor.get("source_commit") != physical_source.get("commit") or
            descriptor.get("physical_source_receipt") != physical_source):
        raise ValueError("worker descriptor differs from the exact current release source")
    environment = _environment_receipt()
    if (descriptor.get("physical_source_receipt_sha256") != _sha(_canonical(physical_source)) or
            descriptor.get("environment_receipt") != environment or
            descriptor.get("environment_receipt_sha256") != _sha(_canonical(environment))):
        raise ValueError("worker physical source or environment receipt differs from supervisor descriptor")
    if descriptor.get("resume_descriptor") is not None:
        from dwr.experiments.source_overfit import _kernel_identities, _kernel_resume_pointer
        model = registration["feasibility_config"]["model"]
        identities = _kernel_identities(registration["config_raw"], registration["config"],
            source_commit=physical_source["commit"], weight_sha256=model["weight_sha256"],
            tokenizer_inventory_sha256=model["artifact_inventory_sha256"])
        from dwr.training.recovery import RecoveryStore
        pointer = _kernel_resume_pointer(descriptor["resume_descriptor"])
        RecoveryStore.verify_descriptor(pointer, identities)
    descriptor_path = jobroot / f"worker-descriptor-{segment_index:03d}.json"
    if _read_json(descriptor_path) != descriptor:
        raise ValueError("worker descriptor is not the immutable supervisor-issued segment descriptor")
    pid = os.getpid(); creation = _process_creation_time(pid)
    if descriptor.get("supervisor_pid") <= 0 or not descriptor.get("supervisor_creation_time"):
        raise ValueError("worker descriptor lacks supervisor process identity")
    if _process_creation_time(descriptor["supervisor_pid"]) != descriptor["supervisor_creation_time"]:
        raise ValueError("supervisor PID creation identity changed")
    lock = _ByteLock(_lock_path(), 1, wait=False)
    parent_pid = _process_parent_pid(pid)
    status = _status_data(jobroot)
    wait_until = time.monotonic() + 10
    while parent_pid != descriptor["supervisor_pid"] and parent_pid != status.get("worker_launcher_pid") and time.monotonic() < wait_until:
        time.sleep(0.05)
        status = _status_data(jobroot)
    if status.get("supervisor_pid") != descriptor["supervisor_pid"] or \
            status.get("supervisor_creation_time") != descriptor["supervisor_creation_time"]:
        lock.close()
        raise ValueError("worker descriptor supervisor differs from current job owner")
    launcher_pid = status.get("worker_launcher_pid")
    if parent_pid not in {descriptor["supervisor_pid"], launcher_pid}:
        lock.close()
        raise ValueError("actual worker parent is not the supervisor-issued launcher")
    try:
        parent_creation = _process_creation_time(parent_pid)
    except (OSError, ValueError):
        lock.close()
        raise ValueError("actual worker parent identity is unavailable")
    image = _process_image(pid)
    receipt = {"schema_version": 1, "pid": pid, "creation_time": creation,
        "parent_pid": parent_pid, "parent_creation_time": parent_creation, "image": image,
        "python_executable": sys.executable, "base_executable": getattr(sys, "_base_executable", sys.executable),
        "descriptor_nonce": descriptor.get("nonce"), "config_sha256": CONFIG_SHA256,
        "source_commit": physical_source["commit"], "segment_index": segment_index}
    receipt_path = jobroot / f"worker-{segment_index:03d}.identity.json"
    try:
        with receipt_path.open("xb") as stream:
            stream.write(_canonical(receipt) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    except Exception:
        lock.close()
        raise
    return {"jobroot": str(jobroot), "segment_index": segment_index,
            "resume_descriptor": descriptor.get("resume_descriptor"),
            "physical_source_receipt": descriptor["physical_source_receipt"],
            "environment_receipt": descriptor["environment_receipt"],
            "supervisor_pid": descriptor["supervisor_pid"],
            "supervisor_creation_time": descriptor["supervisor_creation_time"],
            "worker_pid": pid, "worker_creation_time": creation, "worker_lock": lock,
            "worker_receipt_path": str(receipt_path), "worker_receipt": receipt,
            "work_deadline_monotonic": None,
            "work_deadline_utc_epoch": descriptor.get("work_deadline_epoch"),
            "seal_deadline_utc_epoch": descriptor.get("seal_deadline_epoch"),
            "worker_heartbeat_path": str(jobroot / f"worker-{segment_index:03d}.heartbeat.json")}


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("start", "status"):
        child = sub.add_parser(name); child.add_argument("--config", required=True)
    hidden = sub.add_parser("_supervise"); hidden.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.action == "start": result = start(args.config)
    elif args.action == "status": result = status(args.config)
    else: return _supervise(args.config)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
