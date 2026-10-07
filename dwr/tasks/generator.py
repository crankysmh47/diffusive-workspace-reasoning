"""Pure deterministic candidate generation for the v1 constraint grammar."""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass

from .canonical import canonical_content
from .oracle import enumerate_solutions
from .schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, Solution
from .solver import SolveResult, solve

PARTITIONS = ("source", "development", "final")
PROFILES = {
    "source": (("eq",), ("neq",), ("add_mod",)),
    "development": (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod")),
    "final": (("eq", "neq", "add_mod"),),
}
KIND_ORDER = (RelationKind.EQ, RelationKind.NEQ, RelationKind.ADD_MOD)


@dataclass(frozen=True, slots=True)
class CandidateHeader:
    n: int
    profile: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n not in (3, 4, 5, 6):
            raise ValueError("n must be an integer from 3 through 6")
        if not isinstance(self.profile, tuple) or self.profile not in tuple(profile for profiles in PROFILES.values() for profile in profiles):
            raise ValueError("profile must be a registered immutable relation-kind tuple")


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    partition: str
    candidate_index: int
    n: int | None
    profile: tuple[str, ...] | None
    task: ConstraintTask | None
    witness: Solution | None
    solution: Solution | None
    solver_result: SolveResult | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if self.partition not in PARTITIONS:
            raise ValueError("invalid partition")
        if isinstance(self.candidate_index, bool) or not isinstance(self.candidate_index, int) or not 0 <= self.candidate_index < 2**64:
            raise ValueError("candidate_index must be an unsigned 64-bit integer")
        if self.n is not None and (isinstance(self.n, bool) or not isinstance(self.n, int) or self.n not in (3, 4, 5, 6)):
            raise ValueError("invalid n")
        if self.profile is not None:
            CandidateHeader(self.n, self.profile)
            if self.profile not in PROFILES[self.partition]:
                raise ValueError("profile is not registered for this partition")
        if self.task is not None and not isinstance(self.task, ConstraintTask):
            raise TypeError("task must be ConstraintTask or None")
        if self.witness is not None and not isinstance(self.witness, Solution):
            raise TypeError("witness must be Solution or None")
        if self.solution is not None and not isinstance(self.solution, Solution):
            raise TypeError("solution must be Solution or None")
        if self.solver_result is not None and not isinstance(self.solver_result, SolveResult):
            raise TypeError("solver_result must be SolveResult or None")
        if self.rejection_reason is not None and not isinstance(self.rejection_reason, str):
            raise TypeError("rejection_reason must be a string or None")
        if self.rejection_reason not in (None, "profile_quota_full", "witness_invalid", "oracle_non_unique"):
            raise ValueError("unknown rejection reason")
        if self.rejection_reason is None and any(value is None for value in (self.n, self.profile, self.task, self.witness, self.solution, self.solver_result)):
            raise ValueError("accepted outcomes require task, witness, solution, and solver result")

    @property
    def accepted(self) -> bool:
        return self.rejection_reason is None


def _validate(root_seed: int, partition: str, candidate_index: int) -> None:
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or not 0 <= root_seed < 2**64:
        raise ValueError("root_seed must be an unsigned 64-bit integer")
    if partition not in PARTITIONS:
        raise ValueError("partition must be source, development, or final")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or candidate_index < 0 or candidate_index >= 2**64:
        raise ValueError("candidate_index must be an unsigned 64-bit integer")


def _rng(root_seed: int, partition: str, candidate_index: int) -> random.Random:
    material = (b"dwr-generator-v1\0" + root_seed.to_bytes(8, "big") + b"\0" + partition.encode("ascii") + b"\0" + candidate_index.to_bytes(8, "big"))
    stream_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = random.Random()
    rng.seed(stream_seed, version=2)
    return rng


def sample_candidate_header(root_seed: int, partition: str, candidate_index: int) -> CandidateHeader:
    """Sample n and the registered profile; this precedes profile quota checks."""
    _validate(root_seed, partition, candidate_index)
    rng = _rng(root_seed, partition, candidate_index)
    n = rng.randint(3, 6)
    profile = rng.choice(PROFILES[partition])
    return CandidateHeader(n, profile)


def _eligible(n: int, active: tuple[str, ...]) -> dict[RelationKind, tuple[tuple[str, ...], ...]]:
    names = tuple(f"v{i}" for i in range(n))
    pairs = tuple(itertools.combinations(names, 2))
    result = {}
    for kind in KIND_ORDER:
        if kind.value not in active:
            continue
        result[kind] = pairs if kind is not RelationKind.ADD_MOD else tuple((*pair, target) for pair in pairs for target in names if target not in pair)
    return result


def _count_vectors(active: tuple[str, ...], capacities: tuple[int, ...], total: int):
    active_indices = tuple(i for i, kind in enumerate(KIND_ORDER) if kind.value in active)
    for vector in itertools.product(*(range(1, min(capacities[i], total) + 1) for i in active_indices)):
        full = [0, 0, 0]
        for i, count in zip(active_indices, vector):
            full[i] = count
        if sum(full) == total:
            yield tuple(full)


def generate_candidate(root_seed: int, partition: str, candidate_index: int, *, full_profiles: frozenset[tuple[str, ...]] = frozenset()) -> CandidateOutcome:
    """Generate one deterministic candidate; no module-global RNG or I/O is used.

    ``full_profiles`` lets dataset orchestration apply quotas immediately after
    n/profile draws. A full development profile returns before any more RNG use.
    """
    _validate(root_seed, partition, candidate_index)
    rng = _rng(root_seed, partition, candidate_index)
    n = rng.randint(3, 6)
    profile = rng.choice(PROFILES[partition])

    def rejected(reason: str, task: ConstraintTask | None = None, witness: Solution | None = None):
        return CandidateOutcome(partition, candidate_index, n, profile, task, witness, None, None, reason)

    if partition == "development" and profile in full_profiles:
        return rejected("profile_quota_full")

    eligible = _eligible(n, profile)
    capacities = tuple(len(eligible.get(kind, ())) for kind in KIND_ORDER)
    feasible_totals = [total for total in range(2, 9) if tuple(_count_vectors(profile, capacities, total))]
    total = rng.choice(feasible_totals)
    vectors = tuple(_count_vectors(profile, capacities, total))
    counts = rng.choice(vectors)
    chosen: list[Constraint] = []
    for index, kind in enumerate(KIND_ORDER):
        if counts[index]:
            chosen.extend(Constraint(kind, operands) for operands in rng.sample(eligible[kind], counts[index]))
    chosen.sort(key=lambda constraint: (constraint.kind.value, constraint.variables))

    names = tuple(f"v{i}" for i in range(n))
    witness_values = tuple(rng.randrange(3) for _ in names)
    witness = Solution(tuple(zip(names, witness_values)))
    assignment = dict(witness.assignments)
    for constraint in chosen:
        values = tuple(assignment[name] for name in constraint.variables)
        satisfied = values[0] == values[1] if constraint.kind is RelationKind.EQ else values[0] != values[1] if constraint.kind is RelationKind.NEQ else (values[0] + values[1]) % 3 == values[2]
        if not satisfied:
            return rejected("witness_invalid", witness=witness)

    given_count = rng.randint(1, n - 2)
    given_variables = sorted(rng.sample(names, given_count))
    givens = tuple(GivenAssignment(name, assignment[name]) for name in given_variables)
    task = ConstraintTask("constraint-task-v1", 3, names, tuple(chosen), givens)
    oracle = enumerate_solutions(task)
    if len(oracle.solutions) != 1:
        return rejected("oracle_non_unique", task, witness)
    solver_result = solve(task)
    if solver_result.solution != oracle.solutions[0]:
        raise RuntimeError("solver and independent oracle disagree on accepted candidate")
    return CandidateOutcome(partition, candidate_index, n, profile, task, witness, oracle.solutions[0], solver_result, None)


def profile_quota_outcome(root_seed: int, partition: str, candidate_index: int, full_profiles: frozenset[tuple[str, ...]]) -> CandidateHeader | CandidateOutcome:
    """Quota-check helper: return header if open, otherwise a zero-extra-draw result."""
    header = sample_candidate_header(root_seed, partition, candidate_index)
    if partition == "development" and header.profile in full_profiles:
        return CandidateOutcome(partition, candidate_index, header.n, header.profile, None, None, None, None, "profile_quota_full")
    return header
