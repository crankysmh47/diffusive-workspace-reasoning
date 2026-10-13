"""Exact task identities and deterministic split assignment."""

from __future__ import annotations

import hashlib
import itertools
import json

from .schema import ConstraintTask, RelationKind


def _payload(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _ranked_permutations(task: ConstraintTask):
    """Yield normalized tuple ranks and their mapped variable names.

    For this fixed schema, tuple order matches the corresponding serialized
    JSON order: names are v0..v5, values are single digits, and each task's
    constraint/given/anchor list lengths stay fixed throughout renaming.
    Content's invariant leading fields are followed by constraints then
    givens because JSON keys are sorted. Constraint records compare by kind
    then operand arrays. Given records are emitted in mapped-variable order,
    and each record compares by value then mapped variable because its JSON
    keys sort that way. Structure compares sorted anchors before constraints.
    Fixed list lengths and these restricted strings/integers make tuple
    lexicographic comparison identical to comparing each exact payload.
    """
    variables = task.variables
    positions = {name: index for index, name in enumerate(variables)}
    compiled = tuple(
        (constraint.kind.value, tuple(positions[name] for name in constraint.variables))
        for constraint in task.constraints
    )
    givens = tuple((given.value, positions[given.variable]) for given in task.givens)
    for ordering in itertools.permutations(range(len(variables))):
        mapped = tuple(f"v{index}" for index in range(len(variables)))
        # ordering gives old indices in new-name order; invert it for mapping.
        names = [""] * len(variables)
        for new_index, old_index in enumerate(ordering):
            names[old_index] = mapped[new_index]
        normalized = []
        for kind, operands in compiled:
            renamed = tuple(names[index] for index in operands)
            if kind in (RelationKind.EQ.value, RelationKind.NEQ.value):
                renamed = tuple(sorted(renamed))
            else:
                renamed = (*sorted(renamed[:2]), renamed[2])
            normalized.append((kind, renamed))
        constraints = tuple(sorted(normalized))
        ranked_givens = tuple(sorted(((value, names[index]) for value, index in givens),
                                     key=lambda item: item[1]))
        anchors = tuple(sorted(names[index] for _, index in givens))
        yield names, constraints, ranked_givens, anchors


def canonical_content(task: ConstraintTask) -> str:
    """Return the lexicographically least exact task JSON over all variable names."""
    best = min((constraints, givens) for _, constraints, givens, _ in _ranked_permutations(task))
    constraints, givens = best
    return _payload({
        "schema_version": task.schema_version,
        "modulus": task.modulus,
        "variables": [f"v{i}" for i in range(len(task.variables))],
        "constraints": [{"kind": kind, "variables": list(operands)}
                        for kind, operands in constraints],
        "givens": [{"variable": variable, "value": value}
                   for value, variable in givens],
    })


def canonical_structure(task: ConstraintTask) -> str:
    """Return the least structure payload, retaining anchors but omitting values."""
    best = min((anchors, constraints)
               for _, constraints, _, anchors in _ranked_permutations(task))
    anchors, constraints = best
    return _payload({
        "anchors": list(anchors),
        "constraints": [{"kind": kind, "variables": list(operands)}
                        for kind, operands in constraints],
        "modulus": task.modulus,
        "variables": [f"v{i}" for i in range(len(task.variables))],
        "version": "constraint-structure-v1",
    })


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
