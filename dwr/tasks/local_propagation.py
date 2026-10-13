"""Synchronous generalized arc consistency for finite constraint tasks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

from .schema import (
    Constraint,
    ConstraintTask,
    RelationKind,
    TaskFormatError,
    canonical_constraint_key,
    serialize_task,
)


@dataclass(frozen=True, slots=True)
class PropagationResult:
    domains: tuple[frozenset[int], ...]
    solved: bool
    stalled: bool
    contradiction: bool
    unresolved_unknowns: int
    rounds: int
    support_checks: int

    def __post_init__(self) -> None:
        if (not isinstance(self.domains, tuple) or not 3 <= len(self.domains) <= 6 or any(
            not isinstance(domain, frozenset) or
            any(type(value) is not int or value not in (0, 1, 2) for value in domain)
            for domain in self.domains
        )):
            raise TaskFormatError("wrong_container")
        if any(type(value) is not bool for value in
               (self.solved, self.stalled, self.contradiction)):
            raise TaskFormatError("wrong_container")
        counts = (self.unresolved_unknowns, self.rounds, self.support_checks)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in counts):
            raise TaskFormatError("non_integer_value")
        if self.rounds < 1:
            raise TaskFormatError("invalid_propagation_rounds")
        has_empty_domain = any(not domain for domain in self.domains)
        has_nonsingleton_domain = any(len(domain) > 1 for domain in self.domains)
        expected_state = (
            not has_empty_domain and not has_nonsingleton_domain,
            not has_empty_domain and has_nonsingleton_domain,
            has_empty_domain,
        )
        actual_state = (self.solved, self.stalled, self.contradiction)
        if actual_state != expected_state or sum(actual_state) != 1:
            raise TaskFormatError("invalid_propagation_state")
        if self.unresolved_unknowns != sum(len(domain) > 1 for domain in self.domains):
            raise TaskFormatError("invalid_unresolved_count")


def _satisfies(constraint: Constraint, values: tuple[int, ...]) -> bool:
    if constraint.kind is RelationKind.EQ:
        return values[0] == values[1]
    if constraint.kind is RelationKind.NEQ:
        return values[0] != values[1]
    return (values[0] + values[1]) % 3 == values[2]


def propagate(
    task: ConstraintTask,
    *,
    check_budget: Callable[[], None] | None = None,
) -> PropagationResult:
    """Propagate local supports to a synchronous fixed point.

    Every sweep reads one immutable snapshot and intersects all constraints'
    supports together. ``check_budget`` is called in finite support and
    revision loops so bounded callers can abort work promptly.
    """
    if not isinstance(task, ConstraintTask):
        raise TaskFormatError("wrong_container")
    serialize_task(task)
    if check_budget is not None and not callable(check_budget):
        raise TypeError("check_budget must be callable or None")

    positions = {name: index for index, name in enumerate(task.variables)}
    domains: tuple[frozenset[int], ...] = tuple(
        frozenset({0, 1, 2}) for _ in task.variables
    )
    seeded = [set(domain) for domain in domains]
    for given in task.givens:
        seeded[positions[given.variable]] = {given.value}
    domains = tuple(frozenset(domain) for domain in seeded)
    constraints = tuple(sorted(task.constraints, key=canonical_constraint_key))
    rounds = 0
    support_checks = 0

    while True:
        rounds += 1
        snapshot = domains
        supported: list[tuple[tuple[int, ...], tuple[frozenset[int], ...]]] = []

        for constraint in constraints:
            if check_budget is not None:
                check_budget()
            indices = tuple(positions[name] for name in constraint.variables)
            per_operand = [set() for _ in indices]
            for values in product(*(sorted(snapshot[index]) for index in indices)):
                if check_budget is not None:
                    check_budget()
                support_checks += 1
                if _satisfies(constraint, values):
                    for slot, value in enumerate(values):
                        per_operand[slot].add(value)
            supported.append((indices, tuple(frozenset(values) for values in per_operand)))

        revised = [set(domain) for domain in snapshot]
        for indices, per_operand in supported:
            for index, values in zip(indices, per_operand):
                if check_budget is not None:
                    check_budget()
                revised[index].intersection_update(values)

        domains = tuple(frozenset(domain) for domain in revised)
        if domains == snapshot:
            break

    contradiction = any(not domain for domain in domains)
    unresolved = sum(len(domain) > 1 for domain in domains)
    solved = not contradiction and unresolved == 0 and all(
        _satisfies(constraint, tuple(next(iter(domains[positions[name]]))
                                     for name in constraint.variables))
        for constraint in constraints
    )
    stalled = not contradiction and not solved and unresolved > 0
    return PropagationResult(
        domains, solved, stalled, contradiction, unresolved, rounds, support_checks,
    )
