"""Pure, deterministic witness-first candidate generation for task v2."""
from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass

from .oracle import enumerate_assignments, enumerate_solutions
from .schema import (
    Constraint, ConstraintTask, GivenAssignment, RelationKind, Solution,
    canonical_constraint_key,
)
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
        if type(self.profile) is not tuple or any(type(kind) is not str for kind in self.profile) or self.profile not in tuple(
                profile for profiles in PROFILES.values() for profile in profiles):
            raise ValueError("profile must be a registered immutable relation-kind tuple")


@dataclass(frozen=True, slots=True)
class SourceEligibilityEvidence:
    removal_solution_counts: tuple[int, ...]
    sensitive: bool

    def __post_init__(self) -> None:
        if type(self.removal_solution_counts) is not tuple:
            raise TypeError("source removal counts must be an immutable tuple")
        if not 2 <= len(self.removal_solution_counts) <= 8:
            raise ValueError("source removal counts must cover 2 through 8 constraints")
        if any(type(count) is not int or not 0 <= count <= 729 for count in self.removal_solution_counts):
            raise TypeError("source removal counts must be nonnegative bounded integers")
        if type(self.sensitive) is not bool:
            raise TypeError("source sensitivity must be a bool")
        if self.sensitive != any(count >= 2 for count in self.removal_solution_counts):
            raise ValueError("source sensitivity disagrees with removal counts")


@dataclass(frozen=True, slots=True)
class MixedEligibilityEvidence:
    kinds: tuple[str, ...]
    kind_removal_solution_counts: tuple[int, ...]
    factor_graph_connected: bool
    factor_graph_kinds: tuple[str, ...]
    all_present_kinds_in_graph: bool
    primary_eligible: bool
    ineligibility_reason: str | None

    def __post_init__(self) -> None:
        if type(self.kinds) is not tuple or any(type(kind) is not str for kind in self.kinds):
            raise TypeError("mixed kinds must be an immutable tuple of strings")
        if (not 2 <= len(self.kinds) <= 3 or self.kinds != tuple(sorted(set(self.kinds))) or
                any(kind not in {item.value for item in KIND_ORDER} for kind in self.kinds)):
            raise ValueError("mixed kinds must be sorted, unique registered relation kinds")
        if type(self.kind_removal_solution_counts) is not tuple:
            raise TypeError("mixed removal counts must be an immutable tuple")
        if len(self.kind_removal_solution_counts) != len(self.kinds):
            raise ValueError("mixed removal counts must align with kinds")
        if any(type(count) is not int or not 0 <= count <= 729
               for count in self.kind_removal_solution_counts):
            raise TypeError("mixed removal counts must be nonnegative bounded integers")
        if type(self.factor_graph_connected) is not bool:
            raise TypeError("factor graph connectivity must be a bool")
        if type(self.factor_graph_kinds) is not tuple or any(type(kind) is not str for kind in self.factor_graph_kinds):
            raise TypeError("factor graph kinds must be an immutable tuple of strings")
        if (self.factor_graph_kinds != tuple(sorted(set(self.factor_graph_kinds))) or
                not set(self.factor_graph_kinds) <= set(self.kinds)):
            raise ValueError("factor graph kinds must be a sorted subset of present kinds")
        if any(kind not in {item.value for item in KIND_ORDER} for kind in self.factor_graph_kinds):
            raise ValueError("factor graph kinds must be registered relation kinds")
        if type(self.all_present_kinds_in_graph) is not bool:
            raise TypeError("all-present-kinds flag must be a bool")
        expected_all_in_graph = set(self.kinds) <= set(self.factor_graph_kinds)
        if self.all_present_kinds_in_graph != expected_all_in_graph:
            raise ValueError("all-present-kinds flag disagrees with factor graph kinds")
        if self.factor_graph_connected and not self.factor_graph_kinds:
            raise ValueError("connected factor graph must contain at least one relation kind")
        if type(self.primary_eligible) is not bool:
            raise TypeError("primary eligibility must be a bool")
        necessary = all(count >= 2 for count in self.kind_removal_solution_counts)
        expected_eligible = necessary and self.factor_graph_connected and expected_all_in_graph
        if self.primary_eligible != expected_eligible:
            raise ValueError("primary eligibility disagrees with private evidence")
        expected_reason = (None if expected_eligible else "mixed_kind_not_necessary" if not necessary
                           else "mixed_unknown_factor_disconnected")
        if self.ineligibility_reason is not None and type(self.ineligibility_reason) is not str:
            raise TypeError("mixed ineligibility reason must be a string or None")
        if self.ineligibility_reason != expected_reason:
            raise ValueError("mixed ineligibility reason disagrees with private evidence")


@dataclass(frozen=True, slots=True)
class CandidatePrivateEvidence:
    oracle_unique: bool
    source: SourceEligibilityEvidence | None
    mixed: MixedEligibilityEvidence | None

    def __post_init__(self) -> None:
        if type(self.oracle_unique) is not bool:
            raise TypeError("oracle_unique must be a bool")
        if not self.oracle_unique:
            raise ValueError("private eligibility evidence requires an oracle-unique task")
        if self.source is not None and type(self.source) is not SourceEligibilityEvidence:
            raise TypeError("source evidence has the wrong type")
        if self.mixed is not None and type(self.mixed) is not MixedEligibilityEvidence:
            raise TypeError("mixed evidence has the wrong type")
        if (self.source is None) == (self.mixed is None):
            raise ValueError("private evidence must contain exactly one eligibility record")


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    partition: str
    candidate_index: int
    n: int
    profile: tuple[str, ...]
    task: ConstraintTask | None
    witness: Solution | None
    solution: Solution | None
    solver_result: SolveResult | None
    rejection_reason: str | None
    private_evidence: CandidatePrivateEvidence | None

    def __post_init__(self) -> None:
        if type(self.partition) is not str:
            raise TypeError("partition must be a string")
        _validate(self.partition, self.candidate_index, 0)
        if type(self.candidate_index) is not int:
            raise TypeError("candidate_index must be an integer")
        if type(self.n) is not int:
            raise TypeError("n must be an integer")
        if type(self.profile) is not tuple:
            raise TypeError("profile must be an immutable tuple")
        CandidateHeader(self.n, self.profile)
        if self.profile not in PROFILES[self.partition]:
            raise ValueError("profile is not registered for candidate partition")
        if self.task is not None and type(self.task) is not ConstraintTask:
            raise TypeError("task must be ConstraintTask or None")
        if self.witness is not None and type(self.witness) is not Solution:
            raise TypeError("witness must be Solution or None")
        if self.solution is not None and type(self.solution) is not Solution:
            raise TypeError("solution must be Solution or None")
        if self.solver_result is not None and type(self.solver_result) is not SolveResult:
            raise TypeError("solver_result must be SolveResult or None")
        if self.private_evidence is not None and type(self.private_evidence) is not CandidatePrivateEvidence:
            raise TypeError("private_evidence must be CandidatePrivateEvidence or None")
        if self.rejection_reason is not None and type(self.rejection_reason) is not str:
            raise TypeError("rejection_reason must be a string or None")
        if self.rejection_reason not in (None, "profile_quota_full", "no_witness_feasible_constraint_set", "oracle_non_unique"):
            raise ValueError("unknown v2 candidate rejection reason")
        task_payload = (self.task, self.witness, self.solution, self.solver_result, self.private_evidence)
        if self.rejection_reason == "profile_quota_full":
            if any(value is not None for value in task_payload):
                raise ValueError("quota outcomes cannot contain candidate work")
        elif self.rejection_reason == "no_witness_feasible_constraint_set":
            if self.witness is None or any(value is not None for value in
                    (self.task, self.solution, self.solver_result, self.private_evidence)):
                raise ValueError("no-feasible-vector outcomes require only the sampled witness")
        elif self.rejection_reason == "oracle_non_unique":
            if self.task is None or self.witness is None or any(value is not None for value in
                    (self.solution, self.solver_result, self.private_evidence)):
                raise ValueError("non-unique outcomes require task and witness without solution evidence")
        else:
            if any(value is None for value in task_payload):
                raise ValueError("oracle-unique outcomes require task, witness, solution, solver, and private evidence")
        if self.task is not None:
            self._validate_task_header()
        if self.witness is not None:
            self._validate_solution_shape(self.witness, "witness")
        if self.task is not None and self.solution is not None:
            self._validate_solution_shape(self.solution, "solution")
        if self.rejection_reason is None and self.witness != self.solution:
            raise ValueError("unique task witness and solution must match")
        if self.rejection_reason is None and self.solver_result.solution != self.solution:
            raise ValueError("solver result must agree with the unique solution")
        if self.private_evidence is not None:
            if not self.private_evidence.oracle_unique:
                raise ValueError("private evidence is reserved for oracle-unique tasks")
            if self.partition == "source":
                if self.private_evidence.source is None or self.private_evidence.mixed is not None:
                    raise ValueError("source candidate has inconsistent private evidence")
                if self.task is not None and len(self.private_evidence.source.removal_solution_counts) != len(self.task.constraints):
                    raise ValueError("source evidence must contain one result per constraint")
            elif self.private_evidence.mixed is None or self.private_evidence.source is not None:
                raise ValueError("mixed candidate has inconsistent private evidence")
            elif self.private_evidence.mixed.kinds != tuple(sorted(self.profile)):
                raise ValueError("mixed evidence kinds disagree with candidate profile")

    def _validate_task_header(self) -> None:
        expected_kinds = tuple(kind.value for kind in KIND_ORDER
                               if any(constraint.kind is kind for constraint in self.task.constraints))
        if len(self.task.variables) != self.n or expected_kinds != self.profile:
            raise ValueError("candidate header disagrees with its task")

    def _validate_solution_shape(self, value: Solution, name: str) -> None:
        expected_names = tuple(f"v{index}" for index in range(self.n))
        actual_names = tuple(variable for variable, _ in value.assignments)
        if actual_names != expected_names or self.task is not None and expected_names != self.task.variables:
            raise ValueError(f"{name} variables disagree with candidate task")

    @property
    def oracle_unique(self) -> bool:
        return self.private_evidence is not None and self.private_evidence.oracle_unique

    @property
    def eligible(self) -> bool:
        if self.private_evidence is None:
            return False
        if self.partition == "source":
            assert self.private_evidence.source is not None
            return self.private_evidence.source.sensitive
        assert self.private_evidence.mixed is not None
        return self.private_evidence.mixed.primary_eligible

    @property
    def eligibility_reason(self) -> str | None:
        """Private candidate-level reason; stream duplicate reasons belong to orchestration."""
        if self.private_evidence is None or self.eligible:
            return None
        if self.partition == "source":
            return "source_insensitive"
        assert self.private_evidence.mixed is not None
        return self.private_evidence.mixed.ineligibility_reason


def _validate(partition: str, candidate_index: int, root_seed: int) -> None:
    if partition not in PARTITIONS:
        raise ValueError("partition must be source, development, or final")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or not 0 <= candidate_index < 2**64:
        raise ValueError("candidate_index must be an unsigned 64-bit integer")
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or not 0 <= root_seed < 2**64:
        raise ValueError("root_seed must be an unsigned 64-bit integer")


def _rng(root_seed: int, partition: str, candidate_index: int) -> random.Random:
    _validate(partition, candidate_index, root_seed)
    material = (b"dwr-generator-v2\0" + root_seed.to_bytes(8, "big") + b"\0" +
                partition.encode("ascii") + b"\0" + candidate_index.to_bytes(8, "big"))
    stream_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = random.Random()
    rng.seed(stream_seed, version=2)
    return rng


def sample_candidate_header(root_seed: int, partition: str, candidate_index: int) -> CandidateHeader:
    """Draw n first, then assign the round-robin profile without another RNG call."""
    _validate(partition, candidate_index, root_seed)
    rng = _rng(root_seed, partition, candidate_index)
    n = rng.randint(3, 6)
    profiles = PROFILES[partition]
    return CandidateHeader(n, profiles[candidate_index % len(profiles)])


def _normalized_tuples(n: int, kind: RelationKind) -> tuple[tuple[str, ...], ...]:
    names = tuple(f"v{i}" for i in range(n))
    pairs = tuple(itertools.combinations(names, 2))
    if kind in (RelationKind.EQ, RelationKind.NEQ):
        return pairs
    return tuple((*pair, target) for pair in pairs for target in names if target not in pair)


def _witness_satisfied_tuples(n: int, kind: RelationKind, witness: dict[str, int]) -> tuple[tuple[str, ...], ...]:
    result = []
    for operands in _normalized_tuples(n, kind):
        values = tuple(witness[name] for name in operands)
        satisfied = (values[0] == values[1] if kind is RelationKind.EQ else
                     values[0] != values[1] if kind is RelationKind.NEQ else
                     (values[0] + values[1]) % 3 == values[2])
        if satisfied:
            result.append(operands)
    return tuple(result)


def _count_vectors(active: tuple[str, ...], capacities: tuple[int, ...], total: int):
    active_indices = tuple(i for i, kind in enumerate(KIND_ORDER) if kind.value in active)
    ranges = (range(1, min(capacities[index], total) + 1) for index in active_indices)
    for values in itertools.product(*ranges):
        vector = [0] * len(KIND_ORDER)
        for index, count in zip(active_indices, values):
            vector[index] = count
        if sum(vector) == total:
            yield tuple(vector)


def _source_sensitive(task: ConstraintTask) -> SourceEligibilityEvidence:
    counts = tuple(len(enumerate_assignments(
        task.variables, task.givens,
        tuple(constraint for index, constraint in enumerate(sorted(task.constraints, key=canonical_constraint_key))
              if index != removed)).solutions)
        for removed in range(len(task.constraints)))
    return SourceEligibilityEvidence(counts, any(count >= 2 for count in counts))


def _mixed_primary_eligibility(task: ConstraintTask) -> MixedEligibilityEvidence:
    kinds = tuple(sorted({constraint.kind.value for constraint in task.constraints}))
    kind_counts = tuple(len(enumerate_assignments(
        task.variables, task.givens,
        tuple(constraint for constraint in task.constraints if constraint.kind.value != kind)).solutions)
        for kind in kinds)
    unassigned = set(task.variables) - {given.variable for given in task.givens}
    factors = [(index, constraint) for index, constraint in enumerate(task.constraints)
               if unassigned.intersection(constraint.variables)]
    adjacency: dict[tuple[str, object], set[tuple[str, object]]] = {}
    for index, constraint in factors:
        factor = ("factor", index)
        adjacency.setdefault(factor, set())
        for variable in unassigned.intersection(constraint.variables):
            vnode = ("variable", variable)
            adjacency.setdefault(vnode, set()).add(factor)
            adjacency[factor].add(vnode)
    connected = bool(unassigned) and all(("variable", variable) in adjacency for variable in unassigned)
    if connected:
        reached: set[tuple[str, object]] = set()
        stack = [("variable", next(iter(unassigned)))]
        while stack:
            node = stack.pop()
            if node in reached:
                continue
            reached.add(node)
            stack.extend(adjacency[node] - reached)
        connected = all(("variable", variable) in reached for variable in unassigned)
    graph_kinds = tuple(sorted({constraint.kind.value for _, constraint in factors}))
    all_in_graph = set(kinds) <= set(graph_kinds)
    necessary = all(count >= 2 for count in kind_counts)
    eligible = necessary and connected and all_in_graph
    reason = None if eligible else "mixed_kind_not_necessary" if not necessary else "mixed_unknown_factor_disconnected"
    return MixedEligibilityEvidence(kinds, kind_counts, connected, graph_kinds, all_in_graph, eligible, reason)


def generate_candidate(root_seed: int, partition: str, candidate_index: int, *,
                       full_profiles: frozenset[tuple[str, ...]] = frozenset()) -> CandidateOutcome:
    """Generate one stateless candidate; corpus deduplication belongs to orchestration."""
    _validate(partition, candidate_index, root_seed)
    if not isinstance(full_profiles, frozenset) or any(not isinstance(item, tuple) for item in full_profiles):
        raise TypeError("full_profiles must be a frozenset of profile tuples")
    rng = _rng(root_seed, partition, candidate_index)
    n = rng.randint(3, 6)
    profile = PROFILES[partition][candidate_index % len(PROFILES[partition])]

    def rejected(reason: str | None, witness: Solution | None, task: ConstraintTask | None = None,
                 solution: Solution | None = None, solver_result: SolveResult | None = None,
                 evidence: CandidatePrivateEvidence | None = None) -> CandidateOutcome:
        return CandidateOutcome(partition, candidate_index, n, profile, task, witness,
                                solution, solver_result, reason, evidence)

    if profile in full_profiles:
        # The candidate index and n/profile header are consumed; no witness draw occurs.
        return rejected("profile_quota_full", None)

    names = tuple(f"v{i}" for i in range(n))
    witness = Solution(tuple((name, rng.randrange(3)) for name in names))
    witness_values = dict(witness.assignments)
    pools = tuple(_witness_satisfied_tuples(n, kind, witness_values) for kind in KIND_ORDER)
    capacities = tuple(len(pool) if kind.value in profile else 0 for kind, pool in zip(KIND_ORDER, pools))
    feasible = tuple((total, tuple(_count_vectors(profile, capacities, total)))
                     for total in range(2, 9))
    feasible = tuple((total, vectors) for total, vectors in feasible if vectors)
    if not feasible:
        return rejected("no_witness_feasible_constraint_set", witness)
    total, vectors = rng.choice(feasible)
    counts = rng.choice(vectors)
    constraints: list[Constraint] = []
    for index, kind in enumerate(KIND_ORDER):
        if counts[index]:
            constraints.extend(Constraint(kind, operands)
                               for operands in rng.sample(pools[index], counts[index]))
    constraints.sort(key=canonical_constraint_key)
    given_count = rng.randint(1, n - 2)
    given_variables = tuple(sorted(rng.sample(names, given_count)))
    givens = tuple(GivenAssignment(variable, witness_values[variable]) for variable in given_variables)
    task = ConstraintTask("constraint-task-v1", 3, names, tuple(constraints), givens)
    oracle = enumerate_solutions(task)
    if not oracle.solutions:
        raise RuntimeError("witness-first generation produced a task that rejects its witness")
    if len(oracle.solutions) != 1:
        return rejected("oracle_non_unique", witness, task)
    solver_result = solve(task)
    if solver_result.solution != oracle.solutions[0]:
        raise RuntimeError("solver and independent oracle disagree on accepted candidate")
    source_evidence = _source_sensitive(task) if partition == "source" else None
    mixed_evidence = _mixed_primary_eligibility(task) if partition != "source" else None
    evidence = CandidatePrivateEvidence(True, source_evidence, mixed_evidence)
    return rejected(None, witness, task, oracle.solutions[0], solver_result, evidence)
