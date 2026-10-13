"""Deterministic depth-first reference solver for finite-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import (
    Constraint,
    ConstraintTask,
    RelationKind,
    Solution,
    TaskFormatError,
    canonical_constraint_key,
    serialize_task,
)


@dataclass(frozen=True, slots=True)
class SolverStats:
    first_solution_nodes: int
    uniqueness_nodes: int
    uniqueness_solution_count_capped: int

    def __post_init__(self) -> None:
        counts = (self.first_solution_nodes, self.uniqueness_nodes)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (*counts, self.uniqueness_solution_count_capped)):
            raise TypeError("solver statistics must be integers")
        if any(value < 0 for value in counts):
            raise ValueError("node counts must be nonnegative")
        if self.uniqueness_solution_count_capped not in (0, 1, 2):
            raise ValueError("uniqueness count must be capped at two")


@dataclass(frozen=True, slots=True)
class SolveResult:
    solution: Solution | None
    stats: SolverStats

    def __post_init__(self) -> None:
        if self.solution is not None and not isinstance(self.solution, Solution):
            raise TypeError("solution must be a Solution or None")
        if not isinstance(self.stats, SolverStats):
            raise TypeError("stats must be SolverStats")


def _satisfied_when_grounded(constraint: Constraint, assignment: dict[str, int]) -> bool:
    """Return true for a partial assignment unless a fully grounded rule fails."""
    if any(variable not in assignment for variable in constraint.variables):
        return True
    values = tuple(assignment[variable] for variable in constraint.variables)
    if constraint.kind is RelationKind.EQ:
        return values[0] == values[1]
    if constraint.kind is RelationKind.NEQ:
        return values[0] != values[1]
    return (values[0] + values[1]) % 3 == values[2]


def _consistent(constraints: tuple[Constraint, ...], assignment: dict[str, int]) -> bool:
    return all(_satisfied_when_grounded(constraint, assignment) for constraint in constraints)


def solve(task: ConstraintTask) -> SolveResult:
    """Return the lexicographically first satisfying assignment and search stats.

    Both passes assign the first unassigned variable in task order and try values
    in ascending order. Partial checks reject only fully grounded false rules.
    """
    if not isinstance(task, ConstraintTask):
        raise TaskFormatError("wrong_container")
    serialize_task(task)  # Validate the external task contract before searching.
    constraints = tuple(sorted(task.constraints, key=canonical_constraint_key))
    assignment = {given.variable: given.value for given in task.givens}

    # A grounded contradiction has no consistent root node in either pass.
    if not _consistent(constraints, assignment):
        return SolveResult(None, SolverStats(0, 0, 0))

    def search(solution_limit: int) -> tuple[tuple[Solution, ...], int]:
        found: list[Solution] = []
        nodes = 0

        def visit() -> bool:
            nonlocal nodes
            nodes += 1
            if len(assignment) == len(task.variables):
                found.append(Solution(tuple((name, assignment[name]) for name in task.variables)))
                return len(found) >= solution_limit

            variable = next(name for name in task.variables if name not in assignment)
            for value in range(3):
                assignment[variable] = value
                stop = False
                if _consistent(constraints, assignment):
                    stop = visit()
                del assignment[variable]
                if stop:
                    return True
            return False

        visit()
        return tuple(found), nodes

    first_solutions, first_nodes = search(1)
    uniqueness_solutions, uniqueness_nodes = search(2)
    solution = first_solutions[0] if first_solutions else None
    return SolveResult(
        solution,
        SolverStats(first_nodes, uniqueness_nodes, len(uniqueness_solutions)),
    )
