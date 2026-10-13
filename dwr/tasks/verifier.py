"""Strict answer parsing and constraint verification."""

from __future__ import annotations

from .schema import (
    ConstraintTask,
    TaskFormatError,
    VerificationReason,
    VerificationResult,
    _OUT_OF_RANGE_INTEGER,
    _parse_json,
    canonical_constraint_key,
)


def _result(reason: VerificationReason) -> VerificationResult:
    return VerificationResult(reason is VerificationReason.VALID, reason)


def verify_answer(task: ConstraintTask, answer_text: str) -> VerificationResult:
    """Return the first contract-defined failure for an answer to ``task``."""
    if not isinstance(task, ConstraintTask):
        raise TypeError("task must be a ConstraintTask")
    try:
        answer = _parse_json(answer_text)
    except TaskFormatError as error:
        return _result(VerificationReason(error.code))

    if not isinstance(answer, dict):
        return _result(VerificationReason.WRONG_CONTAINER)
    if any(key not in ("assignment",) for key in answer):
        return _result(VerificationReason.UNKNOWN_KEY)
    if "assignment" not in answer:
        return _result(VerificationReason.MISSING_ASSIGNMENT)

    assignments = answer["assignment"]
    if not isinstance(assignments, dict):
        return _result(VerificationReason.WRONG_CONTAINER)

    if any(variable not in task.variables for variable in assignments):
        return _result(VerificationReason.UNKNOWN_VARIABLE)
    if any(variable not in assignments for variable in task.variables):
        return _result(VerificationReason.MISSING_VARIABLE)
    if any(
        isinstance(assignments[variable], bool)
        or (not isinstance(assignments[variable], int) and assignments[variable] is not _OUT_OF_RANGE_INTEGER)
        for variable in task.variables
    ):
        return _result(VerificationReason.NON_INTEGER_VALUE)
    if any(assignments[variable] not in (0, 1, 2) for variable in task.variables):
        return _result(VerificationReason.VALUE_OUT_OF_DOMAIN)

    for given in task.givens:
        if assignments[given.variable] != given.value:
            return _result(VerificationReason.GIVEN_MISMATCH)

    for constraint in sorted(task.constraints, key=canonical_constraint_key):
        values = tuple(assignments[variable] for variable in constraint.variables)
        if constraint.kind.value == "eq":
            satisfied = values[0] == values[1]
        elif constraint.kind.value == "neq":
            satisfied = values[0] != values[1]
        else:
            satisfied = (values[0] + values[1]) % 3 == values[2]
        if not satisfied:
            return _result(VerificationReason.CONSTRAINT_VIOLATION)

    return _result(VerificationReason.VALID)
