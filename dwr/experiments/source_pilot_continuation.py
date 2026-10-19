"""Validation helpers for the bounded V5 operational recovery continuation."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path, PurePosixPath
from typing import Mapping


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"continuation {label} must be lowercase SHA256")
    return value


def validate_launch_window(registration: Mapping[str, object], *, now_epoch: float | None = None) -> None:
    """Gate creation of a new matrix; attaching/resuming an existing matrix is unaffected."""
    continuation = registration.get("continuation")
    if not isinstance(continuation, Mapping):
        return
    deadline = continuation.get("launch_not_after_epoch")
    original = continuation.get("original_absolute_deadline_epoch")
    now = time.time() if now_epoch is None else now_epoch
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (deadline, original, now)):
        raise ValueError("continuation launch and original deadline epochs must be finite")
    if deadline >= original:
        raise ValueError("continuation launch window must end before the original absolute deadline")
    if now > deadline:
        raise ValueError("continuation matrix creation is past its registered launch cutoff")


def load_parent_imports(root: str | Path, registration: Mapping[str, object]) -> tuple[dict, dict[str, str]]:
    """Load and verify the already-completed K0/K1 artifacts named by V5 registration."""
    continuation = registration.get("continuation")
    if not isinstance(continuation, Mapping):
        return {}, {}
    required = {"parent_matrix", "parent_source_commit", "parent_config_sha256", "imports",
                "original_absolute_deadline_epoch", "launch_not_after_epoch"}
    if set(continuation) != required:
        raise ValueError("continuation manifest fields differ from the registered schema")
    _strict_digest(continuation["parent_config_sha256"], "parent config identity")
    parent_commit = continuation["parent_source_commit"]
    if not isinstance(parent_commit, str) or len(parent_commit) not in (40, 64) or any(
            c not in "0123456789abcdef" for c in parent_commit):
        raise ValueError("continuation parent source commit is malformed")
    imports = continuation["imports"]
    if not isinstance(imports, Mapping) or set(imports) != {"K0", "K1"}:
        raise ValueError("continuation must import exactly K0 and K1")
    expected_fields = {"arm_audit_sha256", "manifest_sha256", "physical_audit_sha256", "source_identity"}
    for arm_id in ("K0", "K1"):
        expected = imports[arm_id]
        if not isinstance(expected, Mapping) or set(expected) != expected_fields:
            raise ValueError(f"continuation {arm_id} import receipt has unknown or missing fields")
        for name in expected_fields:
            _strict_digest(expected[name], f"{arm_id} {name}")
        if expected["source_identity"] != _sha(parent_commit.encode("ascii")):
            raise ValueError(f"continuation {arm_id} source identity does not match the pinned parent commit")
    relative_parent = PurePosixPath(str(continuation["parent_matrix"]))
    if relative_parent.is_absolute() or ".." in relative_parent.parts:
        raise ValueError("continuation parent matrix path must be a safe relative path")
    parent = Path(root).resolve(strict=True) / relative_parent
    if not parent.is_dir():
        raise ValueError("continuation parent matrix is missing")
    config_path = parent / "manifest.initial.json"
    config_initial = json.loads(config_path.read_text(encoding="utf-8"))
    if config_initial.get("source", {}).get("commit") != parent_commit:
        raise ValueError("continuation parent source commit differs from its matrix record")
    if config_initial.get("config", {}).get("sha256") != continuation["parent_config_sha256"]:
        raise ValueError("continuation parent config identity differs from its matrix record")

    from dwr.experiments.records import verify_artifacts
    audits: dict[str, dict] = {}
    source_identities: dict[str, str] = {}
    for arm_id in ("K0", "K1"):
        expected = imports[arm_id]
        arm_root = parent / "arms" / arm_id
        terminal = verify_artifacts(arm_root)
        if terminal.get("status") != "completed":
            raise ValueError(f"continuation parent {arm_id} record is not completed")
        actual_manifest_sha = _sha((arm_root / "manifest.json").read_bytes())
        if actual_manifest_sha != expected["manifest_sha256"]:
            raise ValueError(f"continuation parent {arm_id} manifest digest mismatch")
        audit_raw = (arm_root / "arm-audit.json").read_bytes()
        if _sha(audit_raw) != expected["arm_audit_sha256"]:
            raise ValueError(f"continuation parent {arm_id} audit digest mismatch")
        physical_raw = (arm_root / "physical-audit.json").read_bytes()
        if _sha(physical_raw) != expected["physical_audit_sha256"]:
            raise ValueError(f"continuation parent {arm_id} physical audit digest mismatch")
        audit = json.loads(audit_raw.decode("utf-8"))
        if audit.get("arm_id") != arm_id or audit.get("identities", {}).get("source") != expected["source_identity"]:
            raise ValueError(f"continuation parent {arm_id} source identity differs from its audit")
        audits[arm_id] = audit
        source_identities[arm_id] = expected["source_identity"]
    return audits, source_identities
