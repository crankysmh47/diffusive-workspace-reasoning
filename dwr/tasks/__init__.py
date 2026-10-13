"""Finite-domain task records, strict verification, and exhaustive oracle."""

from .oracle import enumerate_assignments, enumerate_solutions
from .schema import (
    Constraint,
    ConstraintTask,
    GivenAssignment,
    OracleResult,
    RelationKind,
    Solution,
    TaskFormatError,
    VerificationReason,
    VerificationResult,
    parse_task_json,
    serialize_model_input,
    serialize_task,
)
from .verifier import verify_answer

__all__ = [
    "Constraint",
    "ConstraintTask",
    "GivenAssignment",
    "OracleResult",
    "RelationKind",
    "Solution",
    "TaskFormatError",
    "VerificationReason",
    "VerificationResult",
    "enumerate_assignments",
    "enumerate_solutions",
    "parse_task_json",
    "serialize_model_input",
    "serialize_task",
    "verify_answer",
]
