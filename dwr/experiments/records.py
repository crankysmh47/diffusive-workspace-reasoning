"""Write immutable-start and terminal records for local experiment runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from dwr.tasks.run_support import stable_json, strict_json_loads


_SCHEMA = "dwr-run-record-v1"
_INITIAL_KEYS = {
    "schema_version", "status", "started_utc", "command", "paths", "config",
    "identities", "rng_streams", "settings", "budgets", "confirmatory",
    "environment", "source",
}
_TERMINAL_KEYS = {
    "schema_version", "status", "started_utc", "ended_utc", "elapsed_wall_seconds",
    "initial_manifest_sha256", "events_sha256", "event_count", "counts", "artifacts", "error",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_utc_timestamp(value: Any, label: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    _require(parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed),
             f"{label} must use an explicit UTC offset")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_copy(value: Any, label: str, *, object_required: bool = False) -> Any:
    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            _require(math.isfinite(item), f"{label} contains a non-finite number")
            return item
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, Mapping):
            result = {}
            for key, child in item.items():
                _require(isinstance(key, str), f"{label} keys must be strings")
                result[key] = normalize(child)
            return result
        raise ValueError(f"{label} must contain only JSON values")

    plain = normalize(value)
    if object_required:
        _require(isinstance(plain, dict) and bool(plain), f"{label} must be a nonempty JSON object")
    try:
        return strict_json_loads(stable_json(plain))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid JSON metadata") from exc


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    attrs = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attrs & reparse)


def _reject_linked_ancestors(path: Path, *, include_missing_leaf: bool = True) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    chain = list(reversed((absolute, *absolute.parents)))
    for item in chain:
        if not include_missing_leaf and item == absolute:
            continue
        if _is_reparse_or_symlink(item):
            raise ValueError(f"symlink or reparse path is not allowed: {item}")


def _read_regular(path: Path, label: str) -> bytes:
    _reject_linked_ancestors(path)
    _require(path.is_file(), f"{label} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}") from exc


def _git(root: Path, *args: str) -> bytes:
    try:
        proc = subprocess.run(["git", *args], cwd=root, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Git command failed: {args[0]}") from exc
    return proc.stdout


def _git_identity(source_root: Path, *, confirmatory: bool) -> dict[str, Any]:
    root = source_root.resolve(strict=True)
    top = Path(os.fsdecode(_git(root, "rev-parse", "--show-toplevel")).strip()).resolve(strict=True)
    _require(root == top, "source_root must be the Git top-level directory")
    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    status_bytes = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    clean = not status_bytes
    if confirmatory:
        _require(clean, "confirmatory runs require a clean Git worktree")

    tracked_raw = _git(root, "ls-files", "-z", "--cached")
    tracked = {os.fsdecode(item) for item in tracked_raw.split(b"\0") if item}
    listed = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    names = sorted({os.fsdecode(item) for item in listed.split(b"\0") if item}, key=lambda x: x.encode("utf-8", "surrogateescape"))
    files: list[dict[str, Any]] = []
    for name in names:
        rel = PurePosixPath(name.replace("\\", "/"))
        _require(not rel.is_absolute() and ".." not in rel.parts, "Git returned an unsafe source path")
        path = root.joinpath(*rel.parts)
        _reject_linked_ancestors(path)
        if not path.exists():
            _require(name in tracked, f"untracked source path disappeared: {name}")
            files.append({"path": name.replace("\\", "/"), "status": "missing", "sha256": None})
            if confirmatory:
                raise ValueError(f"confirmatory source file is missing: {name}")
            continue
        _require(path.is_file(), f"source path is not a regular file: {name}")
        raw = path.read_bytes()
        files.append({"path": name.replace("\\", "/"), "status": "present", "sha256": _digest(raw)})
        if confirmatory and name in tracked:
            blob = _git(root, "cat-file", "blob", f"HEAD:{name}")
            _require(raw == blob, f"confirmatory worktree bytes differ from HEAD: {name}")

    git_dir_raw = _git(root, "rev-parse", "--absolute-git-dir")
    return {
        "root": str(root),
        "commit": head,
        "clean": clean,
        "files": files,
        "head_diff_sha256": _digest(_git(root, "diff", "--binary", "HEAD")),
        "status_sha256": _digest(status_bytes),
        "git_dir": str(Path(os.fsdecode(git_dir_raw).strip()).resolve(strict=True)),
    }


def _validate_identity_map(value: Any) -> dict[str, Any]:
    data = _json_copy(value, "identities", object_required=True)
    for key, identity in data.items():
        _require(bool(key.strip()), "identity keys must be nonempty")
        _require(isinstance(identity, dict) and set(identity) == {"name", "revision", "sha256"},
                 "each identity must contain exactly name, revision and sha256")
        _require(isinstance(identity["name"], str) and bool(identity["name"].strip()), "identity name must be nonempty")
        _require(isinstance(identity["revision"], str) and bool(identity["revision"].strip()), "identity revision must be nonempty")
        _require(isinstance(identity["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is not None,
                 "identity sha256 must be lowercase hexadecimal")
    return data


def _validate_rng(value: Any) -> dict[str, int]:
    data = _json_copy(value, "rng_streams", object_required=True)
    _require(all(isinstance(key, str) and bool(key.strip()) for key in data), "RNG stream names must be nonempty strings")
    vals = list(data.values())
    _require(all(isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**63 for seed in vals),
             "RNG seeds must be distinct integers in [0, 2**63)")
    _require(len(set(vals)) == len(vals), "RNG stream seeds must be distinct")
    return data


def _validate_budgets(value: Any) -> dict[str, Any]:
    data = _json_copy(value, "budgets", object_required=True)
    _require(set(data) == {"wall_seconds", "memory_mib", "attempts"}, "budgets must contain exactly wall_seconds, memory_mib and attempts")
    wall = data["wall_seconds"]
    _require(isinstance(wall, (int, float)) and not isinstance(wall, bool) and math.isfinite(wall) and wall > 0,
             "wall_seconds must be a positive finite number")
    for key in ("memory_mib", "attempts"):
        _require(isinstance(data[key], int) and not isinstance(data[key], bool) and data[key] > 0,
                 f"{key} must be a positive integer")
    return data


def _validate_counts(value: Any) -> dict[str, int]:
    data = _json_copy(value, "counts", object_required=False)
    _require(isinstance(data, dict), "counts must be an object")
    for key, count in data.items():
        _require(bool(key.strip()) and isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                 "counts require nonempty names and nonnegative integers")
    return data


def _write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_exclusive(path: Path, value: Any) -> bytes:
    data = (stable_json(value) + "\n").encode("utf-8")
    _write_exclusive(path, data)
    return data


def _safe_output_path(output_dir: Path, relative: Any, reserved: set[str]) -> Path:
    _require(isinstance(relative, str) and bool(relative), "artifact paths must be nonempty relative strings")
    _require("\\" not in relative, "artifact paths must use forward slashes")
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute() and not re.match(r"^[A-Za-z]:", relative), "artifact path must be relative")
    _require(all(part not in ("", ".", "..") for part in relative.split("/")), "artifact path contains traversal or alias components")
    basename = pure.name
    reserved_names = set(reserved)
    if os.name == "nt":
        basename = basename.rstrip(" .").casefold()
        reserved_names = {name.rstrip(" .").casefold() for name in reserved}
    _require(len(pure.parts) != 1 or basename not in reserved_names, "record files cannot be declared as artifacts")
    candidate = output_dir.joinpath(*pure.parts)
    _reject_linked_ancestors(candidate)
    base = output_dir.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    _require(resolved != base and base in resolved.parents, "artifact path escapes the run directory")
    _require(resolved.is_file(), "artifact must be an existing regular file")
    return resolved


def _parse_json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, label)
    try:
        value = strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value, raw


def _validate_initial(value: Any) -> None:
    _require(set(value) == _INITIAL_KEYS, "initial manifest has unknown or missing fields")
    _require(value["schema_version"] == _SCHEMA and value["status"] == "running", "invalid initial manifest version or status")
    _validate_utc_timestamp(value["started_utc"], "initial timestamp")
    _require(isinstance(value["command"], list) and bool(value["command"]) and all(isinstance(x, str) for x in value["command"]), "invalid initial command")
    _require(isinstance(value["config"], dict) and set(value["config"]) == {"sha256", "resolved"}, "invalid initial config")
    _require(isinstance(value["config"]["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", value["config"]["sha256"]) is not None
             and isinstance(value["config"]["resolved"], dict), "invalid initial config identity")
    _require(isinstance(value["source"], dict) and set(value["source"]) == {"root", "commit", "clean", "files", "head_diff_sha256", "status_sha256"}, "invalid source inventory")
    source = value["source"]
    _require(isinstance(source["root"], str) and bool(source["root"]) and isinstance(source["commit"], str)
             and bool(source["commit"]) and isinstance(source["clean"], bool) and isinstance(source["files"], list), "invalid source inventory values")
    _require(not value["confirmatory"] or source["clean"], "confirmatory initial manifest cannot contain a dirty source state")
    for key in ("head_diff_sha256", "status_sha256"):
        _require(isinstance(source[key], str) and re.fullmatch(r"[0-9a-f]{64}", source[key]) is not None, "invalid source digest")
    inventory_names: set[str] = set()
    for entry in source["files"]:
        _require(isinstance(entry, dict) and set(entry) == {"path", "status", "sha256"}, "invalid source inventory entry")
        name = entry["path"]
        _require(isinstance(name, str) and bool(name) and "\\" not in name and not PurePosixPath(name).is_absolute()
                 and ".." not in PurePosixPath(name).parts, "invalid source inventory path")
        _require(name not in inventory_names, "duplicate source inventory path")
        inventory_names.add(name)
        if entry["status"] == "present":
            _require(isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None,
                     "invalid source file hash")
        else:
            _require(entry["status"] == "missing" and entry["sha256"] is None, "invalid missing source entry")
    _require(isinstance(value["paths"], dict) and set(value["paths"]) == {"output_dir", "config"}
             and all(isinstance(path, str) and bool(path) for path in value["paths"].values()), "invalid initial paths")
    _validate_identity_map(value["identities"])
    _validate_rng(value["rng_streams"])
    _json_copy(value["settings"], "settings", object_required=True)
    _validate_budgets(value["budgets"])
    _require(isinstance(value["confirmatory"], bool), "invalid confirmatory flag")
    environment = value["environment"]
    _require(isinstance(environment, dict) and set(environment) == {"python_executable", "python_version", "platform", "packages"}
             and all(isinstance(environment[key], str) and bool(environment[key]) for key in ("python_executable", "python_version", "platform"))
             and isinstance(environment["packages"], list), "invalid environment")
    for package in environment["packages"]:
        _require(isinstance(package, dict) and set(package) == {"name", "version"}
                 and all(isinstance(package[key], str) and bool(package[key]) for key in ("name", "version")), "invalid package inventory entry")


def _validate_events(raw: bytes) -> int:
    count = 0
    for expected, line in enumerate(raw.splitlines(), start=1):
        _require(bool(line), "event stream contains an empty line")
        try:
            event = strict_json_loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("event stream contains malformed JSON") from exc
        _require(isinstance(event, dict) and set(event) == {"sequence", "utc", "kind", "payload"}, "event has unknown or missing fields")
        _require(event["sequence"] == expected and isinstance(event["sequence"], int) and not isinstance(event["sequence"], bool), "event sequence is not consecutive")
        _require(isinstance(event["kind"], str) and bool(event["kind"].strip()), "event kind must be nonempty")
        _require(isinstance(event["payload"], dict), "event payload must be an object")
        _validate_utc_timestamp(event["utc"], "event timestamp")
        count += 1
    if raw:
        _require(raw.endswith(b"\n"), "event stream must end with LF")
    return count


class RunRecord:
    """Single-process writer for one unique run directory."""

    def __init__(self, output_dir: Path, started_utc: str, monotonic_start: float, initial_sha256: str):
        self.output_dir = output_dir
        self._started_utc = started_utc
        self._monotonic_start = monotonic_start
        self._initial_sha256 = initial_sha256
        self._finished = False
        self._sealed = False

    @classmethod
    def create(cls, output_dir: str | Path, *, command: list[str], config_path: str | Path,
               expected_config_sha256: str, identities: Mapping[str, Any], rng_streams: Mapping[str, Any],
               settings: Mapping[str, Any], budgets: Mapping[str, Any], confirmatory: bool = False,
               source_root: str | Path | None = None) -> "RunRecord":
        requested = Path(output_dir)
        _require(re.fullmatch(r"[A-Za-z0-9_-]{1,100}", requested.name) is not None, "run directory name must be 1..100 ASCII letters, digits, underscores or hyphens")
        absolute = Path(os.path.abspath(os.fspath(requested)))
        _reject_linked_ancestors(absolute, include_missing_leaf=False)
        _require(absolute.parent.is_dir(), "run directory parent must already exist")
        _require(not absolute.exists(), "run directory already exists")
        _require(isinstance(command, list) and bool(command) and all(isinstance(item, str) and bool(item) for item in command), "command must be a nonempty list of strings")
        _require(isinstance(confirmatory, bool), "confirmatory must be a boolean")
        _require(isinstance(expected_config_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_config_sha256) is not None,
                 "expected config SHA256 must be lowercase hexadecimal")
        ident = _validate_identity_map(identities)
        rng = _validate_rng(rng_streams)
        resolved_settings = _json_copy(settings, "settings", object_required=True)
        resolved_budgets = _validate_budgets(budgets)

        config_lexical = Path(os.path.abspath(os.fspath(config_path)))
        config_raw = _read_regular(config_lexical, "config")
        actual_config_hash = _digest(config_raw)
        _require(actual_config_hash == expected_config_sha256, "config bytes do not match expected SHA256")
        try:
            config = strict_json_loads(config_raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("config is not strict UTF-8 JSON") from exc
        _require(isinstance(config, dict), "config must be a JSON object")
        config = _json_copy(config, "config", object_required=False)

        root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[2]
        _reject_linked_ancestors(Path(os.path.abspath(os.fspath(root))))
        root = root.resolve(strict=True)
        source = _git_identity(root, confirmatory=confirmatory)
        git_dir = Path(source.pop("git_dir"))
        try:
            absolute.relative_to(git_dir)
        except ValueError:
            pass
        else:
            raise ValueError("run output cannot be inside the source Git metadata directory")
        _require(not any(part.lower() == ".git" for part in absolute.parts), "run output cannot be inside a .git path")

        packages = sorted(({"name": dist.metadata.get("Name", dist.name), "version": dist.version}
                           for dist in importlib.metadata.distributions()), key=lambda item: (item["name"].lower(), item["version"]))
        started = _utc_now()
        initial = {
            "schema_version": _SCHEMA,
            "status": "running",
            "started_utc": started,
            "command": list(command),
            "paths": {"output_dir": str(absolute), "config": str(config_lexical.resolve(strict=True))},
            "config": {"sha256": actual_config_hash, "resolved": config},
            "identities": ident,
            "rng_streams": rng,
            "settings": resolved_settings,
            "budgets": resolved_budgets,
            "confirmatory": confirmatory,
            "environment": {"python_executable": sys.executable, "python_version": sys.version,
                            "platform": platform.platform(), "packages": packages},
            "source": source,
        }
        _validate_initial(initial)
        absolute.mkdir()
        try:
            initial_raw = _write_json_exclusive(absolute / "manifest.initial.json", initial)
            _write_exclusive(absolute / "events.jsonl", b"")
        except Exception:
            # Preserve any evidence successfully created; never erase a partial record.
            raise
        return cls(absolute, started, time.monotonic(), _digest(initial_raw))

    def append_event(self, kind: str, payload: Mapping[str, Any]) -> None:
        terminal_path = self.output_dir / "manifest.json"
        if self._sealed or self._finished or terminal_path.exists():
            self._sealed = True
            raise ValueError("cannot append events after terminal record")
        _require(isinstance(kind, str) and bool(kind.strip()), "event kind must be nonempty")
        body = _json_copy(payload, "event payload", object_required=False)
        _require(isinstance(body, dict), "event payload must be an object")
        event_path = self.output_dir / "events.jsonl"
        raw = _read_regular(event_path, "event stream")
        sequence = _validate_events(raw) + 1
        event = {"sequence": sequence, "utc": _utc_now(), "kind": kind, "payload": body}
        line = (stable_json(event) + "\n").encode("utf-8")
        with event_path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def finish(self, status: str, *, counts: Mapping[str, Any], artifacts: list[str], error: str | None = None) -> dict[str, Any]:
        terminal_path = self.output_dir / "manifest.json"
        if self._sealed or self._finished or terminal_path.exists():
            self._sealed = True
            raise ValueError("run already has a terminal record")
        _require(status in {"completed", "failed", "interrupted"}, "invalid terminal status")
        resolved_counts = _validate_counts(counts)
        _require(isinstance(artifacts, list), "artifacts must be a list")
        _require((error is None or (isinstance(error, str) and bool(error.strip()))), "error must be null or a nonempty string")
        _require((status == "completed" and error is None) or (status != "completed" and error is not None),
                 "failed and interrupted outcomes require an error; completed rejects one")
        reserved = {"manifest.initial.json", "manifest.json", "events.jsonl"}
        seen: set[str] = set()
        resolved_artifacts: list[dict[str, Any]] = []
        for relative in artifacts:
            path = _safe_output_path(self.output_dir, relative, reserved)
            normalized = path.relative_to(self.output_dir).as_posix()
            artifact_key = os.path.normcase(os.path.normpath(str(path)))
            _require(artifact_key not in seen, "artifact paths must be unique")
            seen.add(artifact_key)
            raw = _read_regular(path, "artifact")
            resolved_artifacts.append({"path": normalized, "size_bytes": len(raw), "sha256": _digest(raw)})
        initial, initial_raw = _parse_json_file(self.output_dir / "manifest.initial.json", "initial manifest")
        _validate_initial(initial)
        _require(_digest(initial_raw) == self._initial_sha256, "initial manifest changed after run creation")
        _require(initial["started_utc"] == self._started_utc, "initial start time changed after run creation")
        events_raw = _read_regular(self.output_dir / "events.jsonl", "event stream")
        event_count = _validate_events(events_raw)
        terminal = {
            "schema_version": _SCHEMA,
            "status": status,
            "started_utc": self._started_utc,
            "ended_utc": _utc_now(),
            "elapsed_wall_seconds": max(0.0, time.monotonic() - self._monotonic_start),
            "initial_manifest_sha256": _digest(initial_raw),
            "events_sha256": _digest(events_raw),
            "event_count": event_count,
            "counts": resolved_counts,
            "artifacts": resolved_artifacts,
            "error": error,
        }
        _require(set(terminal) == _TERMINAL_KEYS, "internal terminal schema mismatch")
        # Seal before the exclusive write: even a flush/fsync failure may leave a terminal file.
        self._sealed = True
        _write_json_exclusive(terminal_path, terminal)
        self._finished = True
        return terminal


def verify_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Validate run record links and every artifact declared by its terminal record."""
    root = Path(os.path.abspath(os.fspath(run_dir)))
    _reject_linked_ancestors(root)
    _require(root.is_dir(), "run directory is missing")
    terminal, terminal_raw = _parse_json_file(root / "manifest.json", "terminal manifest")
    _require(set(terminal) == _TERMINAL_KEYS, "terminal manifest has unknown or missing fields")
    _require(terminal["schema_version"] == _SCHEMA, "unknown terminal schema version")
    _require(terminal["status"] in {"completed", "failed", "interrupted"}, "unknown terminal status")
    _require(isinstance(terminal["elapsed_wall_seconds"], (int, float)) and not isinstance(terminal["elapsed_wall_seconds"], bool)
             and math.isfinite(terminal["elapsed_wall_seconds"]) and terminal["elapsed_wall_seconds"] >= 0, "invalid elapsed wall time")
    for field in ("initial_manifest_sha256", "events_sha256"):
        _require(isinstance(terminal[field], str) and re.fullmatch(r"[0-9a-f]{64}", terminal[field]) is not None, f"invalid {field}")
    _require(isinstance(terminal["event_count"], int) and not isinstance(terminal["event_count"], bool) and terminal["event_count"] >= 0, "invalid event count")
    _validate_counts(terminal["counts"])
    _require(isinstance(terminal["artifacts"], list), "terminal artifacts must be a list")
    error = terminal["error"]
    _require((terminal["status"] == "completed" and error is None)
             or (terminal["status"] != "completed" and isinstance(error, str) and bool(error.strip())), "terminal error/status mismatch")
    _validate_utc_timestamp(terminal["started_utc"], "terminal start timestamp")
    _validate_utc_timestamp(terminal["ended_utc"], "terminal end timestamp")

    initial_path = root / "manifest.initial.json"
    initial, initial_raw = _parse_json_file(initial_path, "initial manifest")
    _validate_initial(initial)
    _require(terminal["started_utc"] == initial["started_utc"], "terminal start time does not match initial manifest")
    _require(_digest(initial_raw) == terminal["initial_manifest_sha256"], "initial manifest hash mismatch")
    events_raw = _read_regular(root / "events.jsonl", "event stream")
    _require(_digest(events_raw) == terminal["events_sha256"], "event stream hash mismatch")
    _require(_validate_events(events_raw) == terminal["event_count"], "event count mismatch")

    seen: set[str] = set()
    for artifact in terminal["artifacts"]:
        _require(isinstance(artifact, dict) and set(artifact) == {"path", "size_bytes", "sha256"}, "invalid artifact record")
        relative = artifact["path"]
        path = _safe_output_path(root, relative, {"manifest.initial.json", "manifest.json", "events.jsonl"})
        normalized = path.relative_to(root.resolve(strict=True)).as_posix()
        artifact_key = os.path.normcase(os.path.normpath(str(path)))
        _require(artifact_key not in seen, "duplicate artifact path")
        seen.add(artifact_key)
        _require(isinstance(artifact["size_bytes"], int) and not isinstance(artifact["size_bytes"], bool) and artifact["size_bytes"] >= 0, "invalid artifact size")
        _require(isinstance(artifact["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is not None, "invalid artifact hash")
        raw = _read_regular(path, "artifact")
        _require(len(raw) == artifact["size_bytes"] and _digest(raw) == artifact["sha256"], f"artifact identity mismatch: {relative}")
    return terminal
