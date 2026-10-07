"""Strict configuration, provenance and resource helpers for task runs."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any


class ConfigError(ValueError):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ConfigError(f"non-finite JSON constant: {value}")


def strict_json_loads(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except ConfigError:
        raise
    except (ValueError, TypeError) as exc:
        raise ConfigError("malformed JSON") from exc


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedConfig:
    resolved: dict
    sha256: str
    path: Path


def load_config(path: str | Path) -> LoadedConfig:
    path = Path(path).resolve()
    current = strict_json_loads(path.read_text(encoding="utf-8"))
    registered = Path(__file__).resolve().parents[2] / "configs" / "tasks" / "constraint_v1.json"
    approved_sha256 = "26e43360a005fc41fd00981d8ff8471ab9a4bbc90494f7ea312958c0e428e8f3"
    if sha256_file(registered) != approved_sha256:
        raise ConfigError("registered configuration bytes differ from the reviewed baseline")
    baseline = strict_json_loads(registered.read_text(encoding="utf-8"))
    if not isinstance(current, dict) or not isinstance(baseline, dict):
        raise ConfigError("configuration must be an object")
    seed = current.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ConfigError("seed must be an unsigned 64-bit integer")
    comparable = dict(current)
    comparable["seed"] = baseline["seed"]
    if stable_json(comparable) != stable_json(baseline):
        raise ConfigError("configuration differs from the registered closed scientific contract")
    return LoadedConfig(current, sha256_file(path), path)


def resident_bytes_windows() -> int:
    """Current process working set, with explicit Windows API types."""
    if os.name != "nt":
        try:
            with open("/proc/self/statm", "r", encoding="ascii") as stream:
                resident_pages = int(stream.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError, AttributeError) as exc:
            raise RuntimeError("cannot measure resident memory") from exc
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
    kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
    fn = psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    current_process = kernel32.GetCurrentProcess
    current_process.argtypes = []
    current_process.restype = wintypes.HANDLE
    process = current_process()
    if not fn(process, ctypes.byref(counters), counters.cb):
        raise RuntimeError("GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


class ResourceGuard:
    def __init__(self, wall_seconds: float, memory_mib: int, *, clock: Callable[[], float] = time.monotonic,
                 memory_reader: Callable[[], int] = resident_bytes_windows):
        self.wall_seconds = wall_seconds
        self.memory_bytes = memory_mib * 1024 * 1024
        self.clock = clock
        self.memory_reader = memory_reader
        self.started = clock()
        self.peak_bytes: int | None = None

    @property
    def elapsed(self) -> float:
        return max(0.0, self.clock() - self.started)

    def check(self) -> None:
        elapsed = self.elapsed
        try:
            resident = self.memory_reader()
        except Exception as exc:
            raise RuntimeError("resource measurement failed closed") from exc
        if isinstance(resident, bool) or not isinstance(resident, int) or resident < 0:
            raise RuntimeError("resource measurement returned an invalid byte count")
        self.peak_bytes = resident if self.peak_bytes is None else max(self.peak_bytes, resident)
        if resident > self.memory_bytes:
            raise MemoryError("resident memory cap exceeded")
        if elapsed >= self.wall_seconds:
            raise TimeoutError("active command wall-time cap reached")


def remaining_global_seconds(manifest: dict, global_cap_seconds: float = 600) -> float:
    """Budget inherited by audit/replay from the generator manifest."""
    elapsed = manifest.get("generation_elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("manifest has invalid generation elapsed time")
    if isinstance(global_cap_seconds, bool) or not isinstance(global_cap_seconds, (int, float)) or not math.isfinite(global_cap_seconds) or global_cap_seconds < 0:
        raise ValueError("global cap must be a finite nonnegative number")
    return max(0.0, float(global_cap_seconds) - float(elapsed))


def new_manifest(config: LoadedConfig, argv: list[str], outdir: Path, source: dict,
                 *, wall_cap_seconds: float = 600) -> dict:
    return {"manifest_version": "dwr-task-run-v1", "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(), "command": argv,
            "cwd": str(Path.cwd()), "source": source, "python": sys.version,
            "platform": platform.platform(), "config_path": str(config.path),
            "config_sha256": config.sha256, "resolved_config": config.resolved,
            "seed": config.resolved["seed"], "seed_derivation": config.resolved["seed_derivation"],
            "output_dir": str(outdir.resolve()), "limits": {"wall_seconds_global": wall_cap_seconds,
            "resident_memory_mib": config.resolved["audit_gate"]["cap_resident_memory_mib"],
            "cpu_workers": config.resolved["audit_gate"]["cpu_workers"]},
            "generation_elapsed_seconds": 0.0, "partitions": {}}


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(stable_json(value) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)
