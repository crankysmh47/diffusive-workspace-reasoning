"""Independent exhaustive assignment oracle for the finite task grammar."""

from __future__ import annotations

from itertools import product

from .schema import (
    Constraint,
    ConstraintTask,
    GivenAssignment,
    OracleResult,
    RelationKind,
    Solution,
    TaskFormatError,
    canonical_constraint_key,
    serialize_task,
)


def _check_internal_inputs(
    variables: tuple[str, ...],
    givens: tuple[GivenAssignment, ...],
    constraints: tuple[Constraint, ...],
) -> None:
    if not isinstance(variables, tuple) or not 3 <= len(variables) <= 6:
        raise TaskFormatError("unsupported_size")
    if variables != tuple(f"v{i}" for i in range(len(variables))):
        raise TaskFormatError("invalid_variable_sequence")
    if not isinstance(givens, tuple) or not isinstance(constraints, tuple):
        raise TaskFormatError("wrong_container")
    if any(not isinstance(given, GivenAssignment) for given in givens):
        raise TaskFormatError("wrong_container")
    if any(given.variable not in variables for given in givens):
        raise TaskFormatError("unknown_variable")
    given_variables = tuple(given.variable for given in givens)
    if len(set(given_variables)) != len(given_variables):
        raise TaskFormatError("duplicate_given")
    if any(not isinstance(constraint, Constraint) for constraint in constraints):
        raise TaskFormatError("wrong_container")
    for constraint in constraints:
        if any(variable not in variables for variable in constraint.variables):
            raise TaskFormatError("unknown_variable")


def enumerate_assignments(
    variables: tuple[str, ...],
    givens: tuple[GivenAssignment, ...],
    constraints: tuple[Constraint, ...],
) -> OracleResult:
    """Exhaustively evaluate primitive formulas for internal counterfactuals.

    This intentionally permits zero constraints and any number of givens from
    zero through all variables; the external task limits do not apply here.
    """
    _check_internal_inputs(variables, givens, constraints)
    solutions: list[Solution] = []
    for values in product(range(3), repeat=len(variables)):
        assignment = dict(zip(variables, values))
        if any(assignment[given.variable] != given.value for given in givens):
            continue
        accepted = True
        for constraint in constraints:
            relation_values = tuple(assignment[name] for name in constraint.variables)
            if constraint.kind is RelationKind.EQ:
                if relation_values[0] != relation_values[1]:
                    accepted = False
                    break
            elif constraint.kind is RelationKind.NEQ:
                if relation_values[0] == relation_values[1]:
                    accepted = False
                    break
            elif (relation_values[0] + relation_values[1]) % 3 != relation_values[2]:
                accepted = False
                break
        if accepted:
            solutions.append(Solution(tuple((name, assignment[name]) for name in variables)))
    return OracleResult(tuple(solutions), 3 ** len(variables))


def enumerate_solutions(
    task: ConstraintTask,
    *,
    excluded_kinds: tuple[RelationKind, ...] = (),
    excluded_constraint_indices: tuple[int, ...] = (),
) -> OracleResult:
    """Enumerate all solutions, optionally omitting canonical constraints."""
    if not isinstance(task, ConstraintTask):
        raise TaskFormatError("wrong_container")
    # Serializing validates the full external task shape before applying any
    # counterfactual exclusions.
    serialize_task(task)
    if not isinstance(excluded_kinds, tuple) or not isinstance(excluded_constraint_indices, tuple):
        raise TaskFormatError("wrong_container")
    if any(not isinstance(kind, RelationKind) for kind in excluded_kinds):
        raise TaskFormatError("invalid_kind")
    if len(set(excluded_kinds)) != len(excluded_kinds):
        raise TaskFormatError("duplicate_constraint")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in excluded_constraint_indices):
        raise TaskFormatError("non_integer_value")
    if len(set(excluded_constraint_indices)) != len(excluded_constraint_indices):
        raise TaskFormatError("duplicate_constraint")
    constraints = tuple(sorted(task.constraints, key=canonical_constraint_key))
    if any(index < 0 or index >= len(constraints) for index in excluded_constraint_indices):
        raise TaskFormatError("invalid_constraint_count")
    indices = set(excluded_constraint_indices)
    kinds = set(excluded_kinds)
    retained = tuple(
        constraint
        for index, constraint in enumerate(constraints)
        if index not in indices and constraint.kind not in kinds
    )
    return enumerate_assignments(task.variables, task.givens, retained)
