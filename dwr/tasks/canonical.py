"""Exact task identities and deterministic split assignment."""

from __future__ import annotations

import hashlib
import itertools
import json

from .schema import ConstraintTask, RelationKind


def _payload(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _renamings(task: ConstraintTask):
    for ordering in itertools.permutations(task.variables):
        yield {old: f"v{index}" for index, old in enumerate(ordering)}


def canonical_content(task: ConstraintTask) -> str:
    """Return the lexicographically least exact task JSON over all variable names."""
    best = None
    for names in _renamings(task):
        constraints = []
        for constraint in task.constraints:
            operands = tuple(names[name] for name in constraint.variables)
            if constraint.kind in (RelationKind.EQ, RelationKind.NEQ):
                operands = tuple(sorted(operands))
            else:
                operands = (*sorted(operands[:2]), operands[2])
            constraints.append({"kind": constraint.kind.value, "variables": list(operands)})
        constraints.sort(key=lambda item: (item["kind"], item["variables"]))
        givens = sorted(({"variable": names[g.variable], "value": g.value} for g in task.givens), key=lambda item: item["variable"])
        payload = _payload({"schema_version": task.schema_version, "modulus": task.modulus, "variables": [f"v{i}" for i in range(len(task.variables))], "constraints": constraints, "givens": givens})
        if best is None or payload < best:
            best = payload
    assert best is not None
    return best


def canonical_structure(task: ConstraintTask) -> str:
    """Return the least structure payload, retaining anchors but omitting values."""
    best = None
    for names in _renamings(task):
        constraints = []
        for constraint in task.constraints:
            operands = tuple(names[name] for name in constraint.variables)
            if constraint.kind in (RelationKind.EQ, RelationKind.NEQ):
                operands = tuple(sorted(operands))
            else:
                operands = (*sorted(operands[:2]), operands[2])
            constraints.append({"kind": constraint.kind.value, "variables": list(operands)})
        constraints.sort(key=lambda item: (item["kind"], item["variables"]))
        anchors = sorted(names[g.variable] for g in task.givens)
        payload = _payload({"anchors": anchors, "constraints": constraints, "modulus": task.modulus, "variables": [f"v{i}" for i in range(len(task.variables))], "version": "constraint-structure-v1"})
        if best is None or payload < best:
            best = payload
    assert best is not None
    return best


def _digest(domain: bytes, payload: str) -> str:
    return hashlib.sha256(domain + b"\0" + payload.encode("utf-8")).hexdigest()


def content_id(task: ConstraintTask) -> str:
    return _digest(b"dwr-content-v1", canonical_content(task))


def structure_signature(task: ConstraintTask) -> str:
    return _digest(b"dwr-structure-v1", canonical_structure(task))


def lineage_id(task: ConstraintTask) -> str:
    return hashlib.sha256(b"dwr-lineage-v1\0" + content_id(task).encode("ascii")).hexdigest()


def _bucket(domain: bytes, identity: str) -> int:
    return int.from_bytes(hashlib.sha256(domain + b"\0" + identity.encode("ascii")).digest()[:8], "big") % 100


def source_split(identity: str) -> str:
    bucket = _bucket(b"dwr-id-split-v1", identity)
    return "source_train" if bucket < 80 else "source_validation" if bucket < 90 else "source_test"


def development_split(identity: str) -> str:
    return "development_validation" if _bucket(b"dwr-dev-split-v1", identity) < 80 else "development_diagnostic_test"
