"""Deterministic six-variable certificate construction without an oracle."""
from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Callable

from .schema import (
    Constraint,
    ConstraintTask,
    GivenAssignment,
    RelationKind,
    Solution,
    canonical_constraint_key,
)


CELLS = ("A", "B", "C", "D")
CELL_SETTINGS = {"A": (2, 8), "B": (3, 6), "C": (3, 7), "D": (3, 8)}
PROFILES = {
    "source": (("eq",), ("neq",), ("add_mod",)),
    "development": (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod")),
}
KIND_ORDER = (RelationKind.EQ, RelationKind.NEQ, RelationKind.ADD_MOD)
KIND_BY_NAME = {kind.value: kind for kind in KIND_ORDER}
_SEED_TAG = b"dwr-certificate-sixvar-v1\0"
_CHECK_INTERVAL = 64


@dataclass(frozen=True, slots=True)
class WorkCounters:
    """Catalogue-entry counts, not counts of primitive predicate operations.

    ``given_position_scans`` counts the six locations offered to ``sample``, not
    its internal work. ``witness_pool_scans`` counts assignment candidates, not
    individual EQ relation tests. ``alternative_pool_scans`` counts the initial
    complete catalog and additional mixed second-catalog visits. ``constraint_pool_scans``
    counts normalized tuple candidates considered per alternative, including
    source-NEQ anchor-only tuples skipped by its filter. ``count_vector_scans``
    counts bounded positive vectors considered; ``feasible_count_vectors`` sums
    vectors across every feasible second alternative. ``rng_calls`` counts Python
    Random method calls, not the underlying random bits consumed.
    """

    given_position_scans: int = 0
    witness_pool_scans: int = 0
    alternative_pool_scans: int = 0
    constraint_pool_scans: int = 0
    count_vector_scans: int = 0
    feasible_alternatives: int = 0
    feasible_second_alternatives: int = 0
    feasible_count_vectors: int = 0
    rng_calls: int = 0

    def __post_init__(self) -> None:
        for value in (self.given_position_scans, self.witness_pool_scans,
                      self.alternative_pool_scans, self.constraint_pool_scans,
                      self.count_vector_scans,
                      self.feasible_alternatives, self.feasible_second_alternatives,
                      self.feasible_count_vectors, self.rng_calls):
            if type(value) is not int or value < 0:
                raise ValueError("work counters must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class CertificateOutcome:
    root_seed: int
    partition: str
    candidate_index: int
    profile: tuple[str, ...]
    cell: str
    givens: tuple[GivenAssignment, ...]
    witness: Solution | None
    alternatives: tuple[tuple[str, Solution], ...]
    reserved_constraints: tuple[Constraint, ...]
    work: WorkCounters
    task: ConstraintTask | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        _validate(self.root_seed, self.partition, self.candidate_index, None)
        if type(self.profile) is not tuple or any(type(kind) is not str for kind in self.profile):
            raise TypeError("profile must be an immutable tuple of strings")
        schedule = _profile_cell_schedule(self.partition)
        if (self.profile, self.cell) != schedule[self.candidate_index % len(schedule)]:
            raise ValueError("profile and cell must match the fixed candidate schedule")
        if type(self.givens) is not tuple or any(type(item) is not GivenAssignment for item in self.givens):
            raise TypeError("givens must be an immutable tuple of assignments")
        if type(self.alternatives) is not tuple or any(
                type(item) is not tuple or len(item) != 2 or type(item[0]) is not str or
                type(item[1]) is not Solution for item in self.alternatives):
            raise TypeError("alternatives must be immutable kind and solution pairs")
        if type(self.reserved_constraints) is not tuple or any(
                type(item) is not Constraint for item in self.reserved_constraints):
            raise TypeError("reserved constraints must be an immutable tuple")
        if type(self.work) is not WorkCounters:
            raise TypeError("work must be WorkCounters")
        if self.task is not None and type(self.task) is not ConstraintTask:
            raise TypeError("task must be ConstraintTask or None")
        if self.witness is not None and type(self.witness) is not Solution:
            raise TypeError("witness must be Solution or None")
        if self.rejection_reason is not None and type(self.rejection_reason) is not str:
            raise TypeError("rejection_reason must be a string or None")
        if self.rejection_reason not in (None, "empty_witness_pool", "no_feasible_source_alternative",
                                         "no_feasible_mixed_second_alternative"):
            raise ValueError("rejection_reason is not registered for certificate construction")

        if self.witness is not None:
            _require_six_solution(self.witness, "witness")
            self._validate_witness_givens()
        for _, alternative in self.alternatives:
            _require_six_solution(alternative, "alternative")

        if self.task is not None:
            if self.rejection_reason is not None:
                raise ValueError("constructed tasks cannot carry a rejection reason")
            self._validate_successful_task()
            return

        if self.rejection_reason is None:
            raise ValueError("a missing task requires a construction rejection reason")
        self._validate_rejected_outcome()

    def _validate_successful_task(self) -> None:
        assert self.task is not None and self.witness is not None
        _, expected_constraints = CELL_SETTINGS[self.cell]
        names = tuple(f"v{i}" for i in range(6))
        given_map = {given.variable: given.value for given in self.givens}
        witness_map = dict(self.witness.assignments)
        if (self.task.variables != names or len(self.task.constraints) != expected_constraints or
                self.task.givens != self.givens):
            raise ValueError("task shape, support cell, or givens disagree with the outcome")
        if self.task.constraints != tuple(sorted(self.task.constraints, key=canonical_constraint_key)):
            raise ValueError("task constraints must use canonical tuple order")

        kinds = tuple(kind.value for kind in (RelationKind.EQ, RelationKind.NEQ, RelationKind.ADD_MOD)
                      if kind.value in self.profile)
        task_kinds = {constraint.kind.value for constraint in self.task.constraints}
        if task_kinds != set(kinds):
            raise ValueError("task relation kinds must exactly match its profile")
        if any(not _satisfies(constraint.kind, constraint.variables, witness_map)
               for constraint in self.task.constraints):
            raise ValueError("task constraints must preserve the witness")

        alternative_kinds = tuple(kind for kind, _ in self.alternatives)
        if alternative_kinds != self.profile:
            raise ValueError("successful alternatives must follow the exact profile order")
        alternative_maps = []
        for _, alternative in self.alternatives:
            values = dict(alternative.assignments)
            if alternative == self.witness:
                raise ValueError("alternatives must differ from the witness")
            if any(values[name] != value for name, value in given_map.items()):
                raise ValueError("alternatives must preserve givens")
            alternative_maps.append(values)
        if len(set(tuple(sorted(values.items())) for values in alternative_maps)) != len(alternative_maps):
            raise ValueError("candidate alternatives must be distinct")

        if (len(self.reserved_constraints) != len(kinds) or
                tuple(item.kind.value for item in self.reserved_constraints) != kinds):
            raise ValueError("successful construction requires one reserved constraint per profile kind")
        if any(reserved not in self.task.constraints for reserved in self.reserved_constraints):
            raise ValueError("every reserved constraint must be present in the task")

        if self.partition == "source":
            alternative_values = alternative_maps[0]
            reserved = self.reserved_constraints[0]
            if not _satisfies(reserved.kind, reserved.variables, witness_map):
                raise ValueError("source reserve must satisfy the witness")
            if _satisfies(reserved.kind, reserved.variables, alternative_values):
                raise ValueError("source reserve must reject its alternative")
            if any(constraint != reserved and not _satisfies(
                    constraint.kind, constraint.variables, alternative_values)
                   for constraint in self.task.constraints):
                raise ValueError("source filler constraints must preserve the certified alternative")
        else:
            for index, kind in enumerate(kinds):
                own_values = alternative_maps[index]
                other_values = alternative_maps[1 - index]
                if any(constraint.kind.value == kind and not _satisfies(
                        constraint.kind, constraint.variables, other_values)
                       for constraint in self.task.constraints):
                    raise ValueError("mixed constraints must hold under the other kind's alternative")
                reserved = self.reserved_constraints[index]
                if (not _satisfies(reserved.kind, reserved.variables, witness_map) or
                        _satisfies(reserved.kind, reserved.variables, own_values)):
                    raise ValueError("mixed reserve must hold at the witness and reject its own alternative")

    def _validate_witness_givens(self) -> None:
        assert self.witness is not None
        expected_givens, _ = CELL_SETTINGS[self.cell]
        names = tuple(f"v{i}" for i in range(6))
        given_map = {given.variable: given.value for given in self.givens}
        if (len(given_map) != len(self.givens) or len(self.givens) != expected_givens or
                tuple(given.variable for given in self.givens) != tuple(sorted(given_map)) or
                any(name not in names for name in given_map)):
            raise ValueError("givens must be unique, in-range, sorted, and match the support cell")
        witness_map = dict(self.witness.assignments)
        if any(witness_map[name] != value for name, value in given_map.items()):
            raise ValueError("givens must agree with the witness")

    def _validate_rejected_outcome(self) -> None:
        if self.rejection_reason == "empty_witness_pool":
            if (self.witness is not None or self.givens or self.alternatives or self.reserved_constraints or
                    self.task is not None or self.work.rng_calls != 1 or
                    self.work.given_position_scans != 6 or self.work.witness_pool_scans != 3**6 or
                    self.work.alternative_pool_scans or self.work.constraint_pool_scans or
                    self.work.count_vector_scans or self.work.feasible_alternatives or
                    self.work.feasible_second_alternatives or self.work.feasible_count_vectors):
                raise ValueError("empty witness-pool outcomes must stop before downstream construction")
            filtered_stage = ((self.profile == ("eq",) and self.cell in ("B", "C")) or
                              self.profile == ("neq",))
            if self.partition != "source" or not filtered_stage:
                raise ValueError("empty witness pools are only possible in filtered source witness stages")
            return

        if self.witness is None:
            raise ValueError("feasible-alternative rejection requires a sampled witness")
        if self.rejection_reason == "no_feasible_source_alternative":
            if (self.partition != "source" or self.alternatives or self.reserved_constraints or
                    self.work.feasible_alternatives != 0 or self.work.feasible_second_alternatives or
                    self.work.feasible_count_vectors):
                raise ValueError("source-pool rejection has incompatible construction metadata")
            return

        if (self.partition != "development" or len(self.alternatives) != 1 or
                self.alternatives[0][0] != self.profile[0] or self.reserved_constraints or
                self.work.feasible_second_alternatives != 0):
            raise ValueError("mixed-pool rejection has incompatible construction metadata")
        first = self.alternatives[0][1]
        if first == self.witness:
            raise ValueError("first mixed alternative must differ from the witness")
        witness_map = dict(self.witness.assignments)
        first_map = dict(first.assignments)
        given_map = {given.variable: given.value for given in self.givens}
        if any(first_map[name] != value for name, value in given_map.items()):
            raise ValueError("first mixed alternative must preserve givens")


def _require_six_solution(solution: Solution, label: str) -> None:
    names = tuple(name for name, _ in solution.assignments)
    if names != tuple(f"v{i}" for i in range(6)):
        raise ValueError(f"{label} must assign exactly v0 through v5 in order")


class _Work:
    __slots__ = ("given_position_scans", "witness_pool_scans", "alternative_pool_scans",
                 "constraint_pool_scans", "count_vector_scans", "feasible_alternatives", "feasible_second_alternatives",
                 "feasible_count_vectors", "rng_calls", "_scans", "_check_budget")

    def __init__(self, check_budget: Callable[[], None] | None) -> None:
        self.given_position_scans = 0
        self.witness_pool_scans = 0
        self.alternative_pool_scans = 0
        self.constraint_pool_scans = 0
        self.count_vector_scans = 0
        self.feasible_alternatives = 0
        self.feasible_second_alternatives = 0
        self.feasible_count_vectors = 0
        self.rng_calls = 0
        self._scans = 0
        self._check_budget = check_budget

    def scan(self, field: str) -> None:
        setattr(self, field, getattr(self, field) + 1)
        self._scans += 1
        if self._check_budget is not None and self._scans % _CHECK_INTERVAL == 0:
            self._check_budget()

    def boundary(self) -> None:
        if self._check_budget is not None:
            self._check_budget()

    def freeze(self) -> WorkCounters:
        return WorkCounters(
            self.given_position_scans, self.witness_pool_scans,
            self.alternative_pool_scans, self.constraint_pool_scans,
            self.count_vector_scans,
            self.feasible_alternatives, self.feasible_second_alternatives,
            self.feasible_count_vectors, self.rng_calls,
        )


def _validate(root_seed: int, partition: str, candidate_index: int,
              check_budget: Callable[[], None] | None) -> None:
    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise TypeError("root_seed must be an unsigned 64-bit integer")
    if not 0 <= root_seed < 2**64:
        raise ValueError("root_seed must be an unsigned 64-bit integer")
    if type(partition) is not str or partition not in PROFILES:
        raise ValueError("partition must be source or development")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
        raise TypeError("candidate_index must be an unsigned 64-bit integer")
    if not 0 <= candidate_index < 2**64:
        raise ValueError("candidate_index must be an unsigned 64-bit integer")
    if check_budget is not None and not callable(check_budget):
        raise TypeError("check_budget must be callable or None")


def _rng(root_seed: int, partition: str, candidate_index: int) -> random.Random:
    material = (_SEED_TAG + root_seed.to_bytes(8, "big") + b"\0" + partition.encode("ascii") +
                b"\0" + candidate_index.to_bytes(8, "big"))
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = random.Random()
    rng.seed(seed, version=2)
    return rng


def _normalized_tuples(kind: RelationKind) -> tuple[tuple[str, ...], ...]:
    names = tuple(f"v{i}" for i in range(6))
    pairs = tuple(itertools.combinations(names, 2))
    if kind in (RelationKind.EQ, RelationKind.NEQ):
        return pairs
    return tuple((*pair, target) for pair in pairs for target in names if target not in pair)


def _profile_cell_schedule(partition: str) -> tuple[tuple[tuple[str, ...], str], ...]:
    if type(partition) is not str or partition not in PROFILES:
        raise ValueError("partition must be source or development")
    return tuple((profile, cell) for profile in PROFILES[partition] for cell in CELLS)


def _satisfies(kind: RelationKind, operands: tuple[str, ...], values: dict[str, int]) -> bool:
    selected = tuple(values[name] for name in operands)
    if kind is RelationKind.EQ:
        return selected[0] == selected[1]
    if kind is RelationKind.NEQ:
        return selected[0] != selected[1]
    return (selected[0] + selected[1]) % 3 == selected[2]


def _solution(values: tuple[int, ...]) -> Solution:
    return Solution(tuple((f"v{i}", value) for i, value in enumerate(values)))


def _unknown_alternatives(given_map: dict[str, int], witness: Solution,
                          work: _Work) -> tuple[Solution, ...]:
    names = tuple(f"v{i}" for i in range(6))
    unknown = tuple(name for name in names if name not in given_map)
    witness_values = dict(witness.assignments)
    result: list[Solution] = []
    for unknown_values in itertools.product(range(3), repeat=len(unknown)):
        work.scan("alternative_pool_scans")
        values = dict(given_map)
        values.update(zip(unknown, unknown_values))
        if values != witness_values:
            result.append(Solution(tuple((name, values[name]) for name in names)))
    work.boundary()
    return tuple(result)


def _choice(rng: random.Random, values: tuple, work: _Work):
    work.rng_calls += 1
    return rng.choice(values)


def _sample(rng: random.Random, values: tuple, count: int, work: _Work) -> tuple:
    work.rng_calls += 1
    return tuple(rng.sample(values, count))


def _witness_values(profile: tuple[str, ...], cell: str, given_positions: tuple[str, ...],
                    given_count: int, constraint_count: int, rng: random.Random,
                    work: _Work) -> tuple[int, ...] | None:
    if profile == ("eq",) and cell in ("A", "D"):
        constants = tuple((value,) * 6 for value in range(3))
        for _ in constants:
            work.scan("witness_pool_scans")
        work.boundary()
        return _choice(rng, constants, work)
    if profile == ("eq",):
        feasible = []
        eq_pairs = _normalized_tuples(RelationKind.EQ)
        for candidate in itertools.product(range(3), repeat=6):
            work.scan("witness_pool_scans")
            values = {f"v{i}": value for i, value in enumerate(candidate)}
            given_values = {values[name] for name in given_positions}
            if (set(candidate) <= given_values and
                    sum(values[left] == values[right] for left, right in eq_pairs) >= constraint_count):
                feasible.append(candidate)
        work.boundary()
        return _choice(rng, tuple(feasible), work) if feasible else None
    if profile == ("neq",):
        feasible = []
        for candidate in itertools.product(range(3), repeat=6):
            work.scan("witness_pool_scans")
            if (len(set(candidate)) == 3 and
                    len({candidate[int(name[1:])] for name in given_positions}) >= 2):
                feasible.append(candidate)
        work.boundary()
        return _choice(rng, tuple(feasible), work) if feasible else None
    work.rng_calls += 6
    return tuple(rng.randrange(3) for _ in range(6))


def _source_feasible_alternatives(profile: tuple[str, ...], constraint_count: int,
                                  witness: Solution, alternatives: tuple[Solution, ...],
                                  given_variables: frozenset[str], work: _Work
                                  ) -> tuple[tuple[Solution, tuple[tuple[str, ...], ...],
                                                             tuple[tuple[str, ...], ...]], ...]:
    kind = KIND_BY_NAME[profile[0]]
    pool = _normalized_tuples(kind)
    witness_values = dict(witness.assignments)
    feasible = []
    for alternative in alternatives:
        rejecting = []
        neutral = []
        alternative_values = dict(alternative.assignments)
        for operands in pool:
            work.scan("constraint_pool_scans")
            if kind is RelationKind.NEQ and set(operands) <= given_variables:
                continue
            under_witness = _satisfies(kind, operands, witness_values)
            under_alternative = _satisfies(kind, operands, alternative_values)
            if under_witness and not under_alternative:
                rejecting.append(operands)
            elif under_witness and under_alternative:
                neutral.append(operands)
        if rejecting and len(neutral) >= constraint_count - 1:
            feasible.append((alternative, tuple(rejecting), tuple(neutral)))
    work.boundary()
    work.feasible_alternatives = len(feasible)
    return tuple(feasible)


def _positive_count_vectors(capacities: tuple[int, ...], total: int,
                            work: _Work) -> tuple[tuple[int, ...], ...]:
    result = []
    for counts in itertools.product(*(range(1, min(capacity, total) + 1) for capacity in capacities)):
        work.scan("count_vector_scans")
        if sum(counts) == total:
            result.append(counts)
    return tuple(result)


def _mixed_feasible_second_alternatives(profile: tuple[str, ...], constraint_count: int,
                                        witness: Solution, first: Solution,
                                        alternatives: tuple[Solution, ...], work: _Work):
    kinds = tuple(KIND_BY_NAME[name] for name in profile)
    witness_values = dict(witness.assignments)
    first_values = dict(first.assignments)
    feasible = []
    for second in alternatives:
        if second == first:
            continue
        work.scan("alternative_pool_scans")
        second_values = dict(second.assignments)
        pools = []
        rejecting = []
        for kind_index, kind in enumerate(kinds):
            allowed = []
            rejects_own = []
            own_values = first_values if kind_index == 0 else second_values
            other_values = second_values if kind_index == 0 else first_values
            for operands in _normalized_tuples(kind):
                work.scan("constraint_pool_scans")
                if (_satisfies(kind, operands, witness_values) and
                        _satisfies(kind, operands, other_values)):
                    allowed.append(operands)
                    if not _satisfies(kind, operands, own_values):
                        rejects_own.append(operands)
            pools.append(tuple(allowed))
            rejecting.append(tuple(rejects_own))
        if any(not items for items in rejecting):
            continue
        vectors = _positive_count_vectors(tuple(map(len, pools)), constraint_count, work)
        if vectors:
            feasible.append((second, tuple(pools), tuple(rejecting), vectors))
    work.boundary()
    work.feasible_second_alternatives = len(feasible)
    work.feasible_count_vectors = sum(len(item[3]) for item in feasible)
    return tuple(feasible)


def _outcome(root_seed: int, partition: str, candidate_index: int, profile: tuple[str, ...],
             cell: str, givens: tuple[GivenAssignment, ...], witness: Solution | None,
             alternatives: tuple[tuple[str, Solution], ...], reserved: tuple[Constraint, ...],
             work: _Work, task: ConstraintTask | None, reason: str | None) -> CertificateOutcome:
    work.boundary()
    return CertificateOutcome(root_seed, partition, candidate_index, profile, cell, givens,
                              witness, alternatives, reserved, work.freeze(), task, reason)


def construct(root_seed: int, partition: str, candidate_index: int, *,
              check_budget: Callable[[], None] | None = None) -> CertificateOutcome:
    """Construct one registered source/development candidate with one RNG stream."""
    _validate(root_seed, partition, candidate_index, check_budget)
    rng = _rng(root_seed, partition, candidate_index)
    work = _Work(check_budget)
    schedule = _profile_cell_schedule(partition)
    profile, cell = schedule[candidate_index % len(schedule)]
    given_count, constraint_count = CELL_SETTINGS[cell]
    names = tuple(f"v{i}" for i in range(6))
    work.rng_calls += 1
    given_positions = tuple(sorted(rng.sample(names, given_count)))
    work.given_position_scans = len(names)
    work.boundary()

    witness_values = _witness_values(profile, cell, given_positions, given_count,
                                     constraint_count, rng, work)
    if witness_values is None:
        givens = ()
        return _outcome(root_seed, partition, candidate_index, profile, cell, givens,
                        None, (), (), work, None, "empty_witness_pool")
    witness = _solution(witness_values)
    values = dict(witness.assignments)
    givens = tuple(GivenAssignment(name, values[name]) for name in given_positions)
    given_map = {given.variable: given.value for given in givens}
    all_alternatives = _unknown_alternatives(given_map, witness, work)

    if partition == "source":
        feasible = _source_feasible_alternatives(profile, constraint_count, witness,
                                                 all_alternatives, frozenset(given_map), work)
        if not feasible:
            return _outcome(root_seed, partition, candidate_index, profile, cell, givens,
                            witness, (), (), work, None, "no_feasible_source_alternative")
        alternative, rejecting, neutral = _choice(rng, feasible, work)
        protected_operands = _choice(rng, rejecting, work)
        fillers = _sample(rng, neutral, constraint_count - 1, work)
        constraints = tuple(sorted(
            (Constraint(KIND_BY_NAME[profile[0]], operands)
             for operands in (protected_operands, *fillers)), key=canonical_constraint_key))
        task = ConstraintTask("constraint-task-v1", 3, names, constraints, givens)
        return _outcome(root_seed, partition, candidate_index, profile, cell, givens,
                        witness, ((profile[0], alternative),),
                        (Constraint(KIND_BY_NAME[profile[0]], protected_operands),),
                        work, task, None)

    first_kind = profile[0]
    first = _choice(rng, all_alternatives, work)
    second_feasible = _mixed_feasible_second_alternatives(profile, constraint_count,
                                                          witness, first, all_alternatives, work)
    if not second_feasible:
        return _outcome(root_seed, partition, candidate_index, profile, cell, givens,
                        witness, ((first_kind, first),), (), work, None,
                        "no_feasible_mixed_second_alternative")
    second, allowed_pools, rejecting_pools, count_vectors = _choice(rng, second_feasible, work)
    reserved_operands = tuple(_choice(rng, pool, work) for pool in rejecting_pools)
    counts = _choice(rng, count_vectors, work)
    constraints = []
    reserved = []
    for kind, allowed, reject, count, protected in zip(
            (KIND_BY_NAME[name] for name in profile), allowed_pools,
            rejecting_pools, counts, reserved_operands):
        del reject
        protected_constraint = Constraint(kind, protected)
        reserved.append(protected_constraint)
        remaining = tuple(item for item in allowed if item != protected)
        selected = _sample(rng, remaining, count - 1, work)
        constraints.extend((protected_constraint, *(Constraint(kind, operands) for operands in selected)))
    task = ConstraintTask("constraint-task-v1", 3, names,
                         tuple(sorted(constraints, key=canonical_constraint_key)), givens)
    return _outcome(root_seed, partition, candidate_index, profile, cell, givens, witness,
                    ((profile[0], first), (profile[1], second)), tuple(reserved), work, task, None)
