"""Strict, immutable records and canonical task/model-input serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RelationKind(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    ADD_MOD = "add_mod"


class VerificationReason(str, Enum):
    VALID = "valid"
    MALFORMED_JSON = "malformed_json"
    NON_FINITE_CONSTANT = "non_finite_constant"
    DUPLICATE_JSON_KEY = "duplicate_json_key"
    WRONG_CONTAINER = "wrong_container"
    UNKNOWN_KEY = "unknown_key"
    MISSING_ASSIGNMENT = "missing_assignment"
    UNKNOWN_VARIABLE = "unknown_variable"
    MISSING_VARIABLE = "missing_variable"
    NON_INTEGER_VALUE = "non_integer_value"
    VALUE_OUT_OF_DOMAIN = "value_out_of_domain"
    GIVEN_MISMATCH = "given_mismatch"
    CONSTRAINT_VIOLATION = "constraint_violation"


class TaskFormatError(ValueError):
    """A task record or JSON representation violates the frozen task contract."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _require_tuple(value: object, code: str = "wrong_container") -> tuple:
    if not isinstance(value, tuple):
        raise TaskFormatError(code)
    return value


def _valid_variable_name(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _normalized_operands(kind: RelationKind, variables: tuple[str, ...]) -> tuple[str, ...]:
    if kind in (RelationKind.EQ, RelationKind.NEQ):
        return tuple(sorted(variables))
    return (*sorted(variables[:2]), variables[2])


def canonical_constraint_key(constraint: Constraint) -> tuple[str, tuple[str, ...]]:
    return constraint.kind.value, _normalized_operands(constraint.kind, constraint.variables)


@dataclass(frozen=True, slots=True)
class Constraint:
    kind: RelationKind
    variables: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RelationKind):
            raise TaskFormatError("invalid_kind")
        variables = _require_tuple(self.variables)
        expected = 3 if self.kind is RelationKind.ADD_MOD else 2
        if len(variables) != expected:
            raise TaskFormatError("invalid_arity")
        if any(not _valid_variable_name(variable) for variable in variables):
            raise TaskFormatError("unknown_variable")
        if len(set(variables)) != len(variables):
            raise TaskFormatError("repeated_operand")


@dataclass(frozen=True, slots=True)
class GivenAssignment:
    variable: str
    value: int

    def __post_init__(self) -> None:
        if not _valid_variable_name(self.variable):
            raise TaskFormatError("unknown_variable")
        if self.value is _OUT_OF_RANGE_INTEGER:
            raise TaskFormatError("value_out_of_domain")
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TaskFormatError("non_integer_value")
        if self.value not in (0, 1, 2):
            raise TaskFormatError("value_out_of_domain")


@dataclass(frozen=True, slots=True)
class ConstraintTask:
    schema_version: str
    modulus: int
    variables: tuple[str, ...]
    constraints: tuple[Constraint, ...]
    givens: tuple[GivenAssignment, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "constraint-task-v1":
            raise TaskFormatError("unsupported_schema_version")
        if isinstance(self.modulus, bool) or not isinstance(self.modulus, int) or self.modulus != 3:
            raise TaskFormatError("unsupported_modulus")
        variables = _require_tuple(self.variables)
        if not 3 <= len(variables) <= 6:
            raise TaskFormatError("unsupported_size")
        if variables != tuple(f"v{i}" for i in range(len(variables))):
            raise TaskFormatError("invalid_variable_sequence")
        constraints = _require_tuple(self.constraints)
        if not all(isinstance(item, Constraint) for item in constraints):
            raise TaskFormatError("wrong_container")
        if not 2 <= len(constraints) <= 8:
            raise TaskFormatError("invalid_constraint_count")
        for constraint in constraints:
            if any(variable not in variables for variable in constraint.variables):
                raise TaskFormatError("unknown_variable")
        keys = [canonical_constraint_key(item) for item in constraints]
        if len(set(keys)) != len(keys):
            raise TaskFormatError("duplicate_constraint")
        givens = _require_tuple(self.givens)
        if not all(isinstance(item, GivenAssignment) for item in givens):
            raise TaskFormatError("wrong_container")
        given_variables = [given.variable for given in givens]
        if any(variable not in variables for variable in given_variables):
            raise TaskFormatError("unknown_variable")
        if len(set(given_variables)) != len(given_variables):
            raise TaskFormatError("duplicate_given")
        if not 1 <= len(givens) <= len(variables) - 2:
            raise TaskFormatError("invalid_given_count")


@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool
    reason: VerificationReason

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool) or not isinstance(self.reason, VerificationReason):
            raise TaskFormatError("wrong_container")


@dataclass(frozen=True, slots=True)
class Solution:
    assignments: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        assignments = _require_tuple(self.assignments)
        if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in assignments):
            raise TaskFormatError("wrong_container")
        if not 3 <= len(assignments) <= 6:
            raise TaskFormatError("unsupported_size")
        if any(not _valid_variable_name(name) for name, _ in assignments):
            raise TaskFormatError("unknown_variable")
        if tuple(name for name, _ in assignments) != tuple(f"v{i}" for i in range(len(assignments))):
            raise TaskFormatError("invalid_variable_sequence")
        if any(isinstance(value, bool) or not isinstance(value, int) for _, value in assignments):
            raise TaskFormatError("non_integer_value")
        if any(value not in (0, 1, 2) for _, value in assignments):
            raise TaskFormatError("value_out_of_domain")


@dataclass(frozen=True, slots=True)
class OracleResult:
    solutions: tuple[Solution, ...]
    assignments_examined: int

    def __post_init__(self) -> None:
        solutions = _require_tuple(self.solutions)
        if any(not isinstance(solution, Solution) for solution in solutions):
            raise TaskFormatError("wrong_container")
        if isinstance(self.assignments_examined, bool) or not isinstance(self.assignments_examined, int):
            raise TaskFormatError("non_integer_value")
        if self.assignments_examined < 1:
            raise ValueError("assignments_examined must be positive")


def _json_object(pairs: list[tuple[str, Any]], state: dict[str, bool]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            state["duplicate"] = True
        obj[key] = value
    return obj


def _non_finite(_: str, state: dict[str, bool]) -> object:
    state["non_finite"] = True
    return _NON_FINITE


_NON_FINITE = object()
_OUT_OF_RANGE_INTEGER = object()


def _parse_integer(token: str) -> int | object:
    digits = token[1:] if token.startswith("-") else token
    if len(digits) > 18:
        return _OUT_OF_RANGE_INTEGER
    return int(token)


def _parse_json(text: str) -> Any:
    if not isinstance(text, str):
        raise TaskFormatError("wrong_container")
    state = {"duplicate": False, "non_finite": False}
    try:
        value = json.loads(
            text,
            object_pairs_hook=lambda pairs: _json_object(pairs, state),
            parse_constant=lambda token: _non_finite(token, state),
            parse_int=_parse_integer,
        )
    except (json.JSONDecodeError, UnicodeError, RecursionError, ValueError):
        raise TaskFormatError("malformed_json") from None
    if state["non_finite"]:
        raise TaskFormatError("non_finite_constant")
    if state["duplicate"]:
        raise TaskFormatError("duplicate_json_key")
    return value


def _expect_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskFormatError("wrong_container")
    return value


def _exact_fields(value: dict[str, Any], required: tuple[str, ...]) -> None:
    if any(key not in required for key in value):
        raise TaskFormatError("unknown_key")
    if any(key not in value for key in required):
        raise TaskFormatError("missing_field")


def parse_task_json(text: str) -> ConstraintTask:
    obj = _expect_object(_parse_json(text))
    fields = ("schema_version", "modulus", "variables", "constraints", "givens")
    _exact_fields(obj, fields)
    if not isinstance(obj["schema_version"], str):
        raise TaskFormatError("unsupported_schema_version")
    if not isinstance(obj["modulus"], int) or isinstance(obj["modulus"], bool) or obj["modulus"] != 3:
        raise TaskFormatError("unsupported_modulus")
    if not isinstance(obj["variables"], list):
        raise TaskFormatError("wrong_container")
    if not isinstance(obj["constraints"], list) or not isinstance(obj["givens"], list):
        raise TaskFormatError("wrong_container")
    variables = tuple(obj["variables"])
    constraints: list[Constraint] = []
    for raw_constraint in obj["constraints"]:
        item = _expect_object(raw_constraint)
        _exact_fields(item, ("kind", "variables"))
        if not isinstance(item["kind"], str):
            raise TaskFormatError("invalid_kind")
        try:
            kind = RelationKind(item["kind"])
        except ValueError:
            raise TaskFormatError("invalid_kind") from None
        if not isinstance(item["variables"], list):
            raise TaskFormatError("wrong_container")
        constraints.append(Constraint(kind, tuple(item["variables"])))
    givens: list[GivenAssignment] = []
    for raw_given in obj["givens"]:
        item = _expect_object(raw_given)
        _exact_fields(item, ("variable", "value"))
        givens.append(GivenAssignment(item["variable"], item["value"]))
    return ConstraintTask(obj["schema_version"], obj["modulus"], variables, tuple(constraints), tuple(givens))


def _task_object(task: ConstraintTask) -> dict[str, Any]:
    if not isinstance(task, ConstraintTask):
        raise TaskFormatError("wrong_container")
    constraints = []
    for constraint in sorted(task.constraints, key=canonical_constraint_key):
        constraints.append({"kind": constraint.kind.value, "variables": list(_normalized_operands(constraint.kind, constraint.variables))})
    givens = [{"variable": given.variable, "value": given.value} for given in sorted(task.givens, key=lambda item: item.variable)]
    return {
        "schema_version": task.schema_version,
        "modulus": task.modulus,
        "variables": list(task.variables),
        "constraints": constraints,
        "givens": givens,
    }


def _json_bytes_object(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def serialize_task(task: ConstraintTask) -> str:
    return _json_bytes_object(_task_object(task))


def serialize_model_input(task: ConstraintTask) -> str:
    properties = {variable: {"enum": [0, 1, 2], "type": "integer"} for variable in task.variables}
    model_input = {
        "answer_schema": {
            "additionalProperties": False,
            "properties": {"assignment": {"additionalProperties": False, "properties": properties, "required": list(task.variables), "type": "object"}},
            "required": ["assignment"],
            "type": "object",
        },
        "instruction": "Return one complete satisfying assignment as JSON.",
        "rules": {
            "add_mod": "(value(a)+value(b)) % 3 == value(c)",
            "domain": [0, 1, 2],
            "eq": "value(a) == value(b)",
            "neq": "value(a) != value(b)",
        },
        "task": _task_object(task),
    }
    return _json_bytes_object(model_input)
