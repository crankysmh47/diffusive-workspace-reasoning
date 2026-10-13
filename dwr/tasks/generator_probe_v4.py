"""Stateless candidate proposal for the reviewed v4 development probe."""
from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Mapping

from .config_probe_v4 import ProbeConfig
from .generator_v2 import (
    KIND_ORDER,
    _count_vectors,
    _mixed_primary_eligibility,
    _normalized_tuples,
    _source_sensitive,
    _witness_satisfied_tuples,
)
from .oracle import enumerate_solutions
from .schema import (
    Constraint,
    ConstraintTask,
    GivenAssignment,
    RelationKind,
    Solution,
    canonical_constraint_key,
)
from .solver import SolverStats, solve


GENERATOR_ID = "dwr-generator-v4-probe"
PARTITIONS = ("source", "development")
_PROPOSAL_REASONS = {
    "no_source_eq_witness",
    "no_neq_witness",
    "no_feasible_constraint_vector",
    "no_feasible_neq_constraint_total",
}


def _solution_satisfies_task(task: ConstraintTask, solution: Solution) -> bool:
    if tuple(name for name, _ in solution.assignments) != task.variables:
        return False
    values = dict(solution.assignments)
    if any(values[given.variable] != given.value for given in task.givens):
        return False
    for constraint in task.constraints:
        operands = tuple(values[name] for name in constraint.variables)
        if constraint.kind is RelationKind.EQ and operands[0] != operands[1]:
            return False
        if constraint.kind is RelationKind.NEQ and operands[0] == operands[1]:
            return False
        if (constraint.kind is RelationKind.ADD_MOD and
                (operands[0] + operands[1]) % 3 != operands[2]):
            return False
    return True


@dataclass(frozen=True, slots=True)
class ProposalResult:
    partition: str
    candidate_index: int
    profile: tuple[str, ...]
    stratum: str
    n: int
    g: int
    c: int
    proposal_reason_or_null: str | None
    task_or_null: ConstraintTask | None
    witness_or_null: Solution | None
    oracle_count_capped_or_null: int | None
    oracle_solution_or_null: Solution | None
    solver_stats_or_null: SolverStats | None
    private_eligibility_or_null: Mapping[str, object] | None

    def __post_init__(self) -> None:
        if self.partition not in PARTITIONS:
            raise ValueError("proposal partition must be source or development")
        if type(self.candidate_index) is not int or not 0 <= self.candidate_index < 2**64:
            raise ValueError("candidate_index must be an unsigned 64-bit integer")
        if type(self.profile) is not tuple or not self.profile:
            raise TypeError("profile must be a nonempty immutable tuple")
        registered_profiles = {
            "source": (("eq",), ("neq",), ("add_mod",)),
            "development": (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod")),
        }
        if self.profile not in registered_profiles[self.partition]:
            raise ValueError("profile is not registered for the proposal partition")
        if type(self.stratum) is not str or self.stratum not in ("A", "B", "C", "D"):
            raise ValueError("stratum must be A, B, C, or D")
        for name, value in (("n", self.n), ("g", self.g), ("c", self.c)):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
        if self.n != 6 or (self.g, self.c) != {
                "A": (2, 8), "B": (3, 6), "C": (3, 7), "D": (3, 8)}[self.stratum]:
            raise ValueError("proposal header disagrees with its registered stratum")
        if self.task_or_null is not None and type(self.task_or_null) is not ConstraintTask:
            raise TypeError("task_or_null must be ConstraintTask or None")
        if self.witness_or_null is not None and type(self.witness_or_null) is not Solution:
            raise TypeError("witness_or_null must be Solution or None")
        if self.oracle_count_capped_or_null is not None and (
                type(self.oracle_count_capped_or_null) is not int or
                self.oracle_count_capped_or_null not in (0, 1, 2)):
            raise TypeError("oracle_count_capped_or_null must be an integer from zero through two")
        if self.oracle_solution_or_null is not None and type(self.oracle_solution_or_null) is not Solution:
            raise TypeError("oracle_solution_or_null must be Solution or None")
        if self.solver_stats_or_null is not None and type(self.solver_stats_or_null) is not SolverStats:
            raise TypeError("solver_stats_or_null must be SolverStats or None")
        if self.private_eligibility_or_null is not None and not isinstance(
                self.private_eligibility_or_null, Mapping):
            raise TypeError("private_eligibility_or_null must be a mapping or None")
        if self.proposal_reason_or_null is not None and self.proposal_reason_or_null not in _PROPOSAL_REASONS:
            raise ValueError("unregistered proposal rejection reason")
        if self.proposal_reason_or_null is not None:
            if (self.oracle_count_capped_or_null is not None or self.oracle_solution_or_null is not None or
                    self.solver_stats_or_null is not None or self.private_eligibility_or_null is not None):
                raise ValueError("proposal rejection cannot carry oracle, solver, or eligibility fields")
            if self.proposal_reason_or_null in ("no_source_eq_witness", "no_neq_witness"):
                if self.task_or_null is not None or self.witness_or_null is not None:
                    raise ValueError("empty witness populations cannot carry task or witness")
            elif self.task_or_null is not None or self.witness_or_null is None:
                raise ValueError("empty constraint pools carry only their sampled witness")
        elif (self.task_or_null is None or self.witness_or_null is None or
              self.oracle_count_capped_or_null not in (1, 2)):
            raise ValueError("constructed proposal must carry task, witness and capped oracle count")
        if self.proposal_reason_or_null is None:
            task = self.task_or_null
            witness = self.witness_or_null
            expected_variables = tuple(f"v{i}" for i in range(self.n))
            if task.variables != expected_variables or len(task.variables) != self.n:
                raise ValueError("task variables do not match the proposal size")
            if len(task.givens) != self.g or len(task.constraints) != self.c:
                raise ValueError("task given or constraint count does not match the proposal header")
            present_kinds = {constraint.kind.value for constraint in task.constraints}
            actual_profile = tuple(kind.value for kind in KIND_ORDER if kind.value in present_kinds)
            if actual_profile != self.profile:
                raise ValueError("task relation kinds do not match the proposal profile")
            if not _solution_satisfies_task(task, witness):
                raise ValueError("private witness does not satisfy its task and givens")
            if self.oracle_count_capped_or_null == 1:
                if (self.oracle_solution_or_null is None or
                    tuple(name for name, _ in self.oracle_solution_or_null.assignments) != expected_variables or
                    self.oracle_solution_or_null != witness or
                    not _solution_satisfies_task(task, self.oracle_solution_or_null)):
                    raise ValueError("unique oracle solution must match the satisfying private witness")
        if self.oracle_count_capped_or_null == 2 and any(
                value is not None for value in (self.oracle_solution_or_null, self.solver_stats_or_null,
                                                self.private_eligibility_or_null)):
            raise ValueError("nonunique proposal cannot carry solution, solver or eligibility")
        if self.oracle_count_capped_or_null == 1 and any(
                value is None for value in (self.oracle_solution_or_null, self.solver_stats_or_null,
                                            self.private_eligibility_or_null)):
            raise ValueError("unique proposal requires solution, solver and private eligibility")
        if self.oracle_count_capped_or_null == 1:
            expected_kind = "source" if self.partition == "source" else "mixed"
            eligibility = self.private_eligibility_or_null
            if eligibility.get("kind") != expected_kind or type(eligibility.get("eligible")) is not bool:
                raise ValueError("private eligibility kind or flag disagrees with partition")
            if expected_kind == "source":
                counts = eligibility.get("removal_solution_counts")
                if (set(eligibility) != {"kind", "eligible", "removal_solution_counts"} or
                        type(counts) is not list or len(counts) != len(self.task_or_null.constraints) or
                        any(type(value) is not int or not 0 <= value <= 729 for value in counts) or
                        eligibility["eligible"] != any(value >= 2 for value in counts)):
                    raise ValueError("source eligibility payload is malformed or inconsistent")
            else:
                expected = {"kind", "eligible", "kinds", "kind_removal_solution_counts",
                            "factor_graph_connected", "factor_graph_kinds",
                            "all_present_kinds_in_graph", "reason"}
                counts = eligibility.get("kind_removal_solution_counts")
                if (set(eligibility) != expected or type(eligibility.get("kinds")) is not list or
                        type(counts) is not list or len(counts) != len(eligibility["kinds"]) or
                        any(type(value) is not int or not 0 <= value <= 729 for value in counts) or
                        type(eligibility.get("factor_graph_connected")) is not bool or
                        type(eligibility.get("factor_graph_kinds")) is not list or
                        type(eligibility.get("all_present_kinds_in_graph")) is not bool or
                        (eligibility.get("reason") is not None and
                         type(eligibility.get("reason")) is not str)):
                    raise ValueError("mixed eligibility payload is malformed")
                kinds = eligibility["kinds"]
                graph_kinds = eligibility["factor_graph_kinds"]
                if (any(type(kind) is not str for kind in (*kinds, *graph_kinds)) or
                        not kinds or kinds != sorted(set(kinds)) or graph_kinds != sorted(set(graph_kinds)) or
                        set(kinds) != {constraint.kind.value for constraint in self.task_or_null.constraints} or
                        not set(graph_kinds) <= set(kinds) or
                        eligibility["all_present_kinds_in_graph"] != (
                            {constraint.kind.value for constraint in self.task_or_null.constraints}
                            <= set(graph_kinds))):
                    raise ValueError("mixed eligibility kind projections are inconsistent")
                eligible = (all(value >= 2 for value in counts) and
                            eligibility["factor_graph_connected"] and
                            eligibility["all_present_kinds_in_graph"])
                reason = (None if eligible else "mixed_kind_not_necessary"
                          if not all(value >= 2 for value in counts)
                          else "mixed_unknown_factor_disconnected")
                if eligibility["eligible"] != eligible or eligibility["reason"] != reason:
                    raise ValueError("mixed eligibility outcome disagrees with its evidence")

    @property
    def proposal_reason(self) -> str | None:
        return self.proposal_reason_or_null

    @property
    def task(self) -> ConstraintTask | None:
        return self.task_or_null

    @property
    def witness(self) -> Solution | None:
        return self.witness_or_null

    @property
    def oracle_count_capped(self) -> int | None:
        return self.oracle_count_capped_or_null

    @property
    def oracle_solution(self) -> Solution | None:
        return self.oracle_solution_or_null

    @property
    def solver_stats(self) -> SolverStats | None:
        return self.solver_stats_or_null

    @property
    def private_eligibility(self) -> Mapping[str, object] | None:
        return self.private_eligibility_or_null


class ProposalFatalError(RuntimeError):
    """Fatal generator/oracle contract violation with its retained proposal context."""

    def __init__(self, reason: str, context: ProposalFatalContext):
        if reason not in ("oracle_witness_disagreement", "solver_oracle_disagreement"):
            raise ValueError("unregistered fatal proposal reason")
        self.reason = reason
        self.context = context
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ProposalFatalContext:
    partition: str
    candidate_index: int
    profile: tuple[str, ...]
    stratum: str
    n: int
    g: int
    c: int
    task: ConstraintTask
    witness: Solution
    oracle_count_capped: int


def _check_index(partition: str, candidate_index: int) -> None:
    if type(partition) is not str or partition not in PARTITIONS:
        raise ValueError("partition must be source or development")
    if type(candidate_index) is not int or not 0 <= candidate_index < 2**64:
        raise ValueError("candidate_index must be an unsigned 64-bit integer")


def candidate_stream(config: ProbeConfig, partition: str, candidate_index: int) -> random.Random:
    """Return a fresh v4 candidate-local stream using the pinned domain bytes."""
    if type(config) is not ProbeConfig:
        raise TypeError("config must be a ProbeConfig")
    _check_index(partition, candidate_index)
    root_seed = config.root_seed
    if type(root_seed) is not int or not 0 <= root_seed < 2**64:
        raise ValueError("configured root seed must be an unsigned 64-bit integer")
    material = (b"dwr-generator-v4-probe\0" + root_seed.to_bytes(8, "big") + b"\0" +
                partition.encode("ascii") + b"\0" + candidate_index.to_bytes(8, "big"))
    stream_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = random.Random()
    rng.seed(stream_seed, version=2)
    return rng


def _remaining_profiles(config: ProbeConfig, partition: str) -> tuple[tuple[str, ...], ...]:
    return config.profiles_for(partition)


def next_active_cell(
    config: ProbeConfig,
    partition: str,
    candidate_index: int,
    remaining_quota: Mapping[tuple[str, ...], Mapping[str, int]],
) -> tuple[tuple[str, ...], str]:
    """Select an open profile/cell in registered order without creating a stream."""
    if type(config) is not ProbeConfig:
        raise TypeError("config must be a ProbeConfig")
    _check_index(partition, candidate_index)
    if not isinstance(remaining_quota, Mapping):
        raise TypeError("remaining_quota must be a profile-to-cell mapping")
    profiles = _remaining_profiles(config, partition)
    if set(remaining_quota) != set(profiles):
        raise ValueError("remaining_quota profile keys do not match the partition")
    active: list[tuple[tuple[str, ...], str]] = []
    for profile in profiles:
        cells = remaining_quota[profile]
        if not isinstance(cells, Mapping) or set(cells) != set(config.cell_order):
            raise ValueError("remaining_quota cell keys do not match the registered strata")
        for cell in config.cell_order:
            value = cells[cell]
            if type(value) is not int or not 0 <= value <= config.quota_for(partition, profile, cell):
                raise ValueError("remaining quota counts must be valid nonnegative integers")
            if value > 0:
                active.append((profile, cell))
    if not active:
        raise ValueError("no active profile/cell quota remains")
    return active[candidate_index % len(active)]


def _as_solution(config: ProbeConfig, values: tuple[int, ...]) -> Solution:
    return Solution(tuple(zip(config.variables, values)))


def _private_eligibility(partition: str, task: ConstraintTask) -> dict[str, object]:
    if partition == "source":
        evidence = _source_sensitive(task)
        return {
            "kind": "source",
            "eligible": evidence.sensitive,
            "removal_solution_counts": list(evidence.removal_solution_counts),
        }
    evidence = _mixed_primary_eligibility(task)
    return {
        "kind": "mixed",
        "eligible": evidence.primary_eligible,
        "kinds": list(evidence.kinds),
        "kind_removal_solution_counts": list(evidence.kind_removal_solution_counts),
        "factor_graph_connected": evidence.factor_graph_connected,
        "factor_graph_kinds": list(evidence.factor_graph_kinds),
        "all_present_kinds_in_graph": evidence.all_present_kinds_in_graph,
        "reason": evidence.ineligibility_reason,
    }


def _result(
    partition: str,
    candidate_index: int,
    profile: tuple[str, ...],
    stratum: str,
    g: int,
    c: int,
    *,
    reason: str | None = None,
    task: ConstraintTask | None = None,
    witness: Solution | None = None,
    count: int | None = None,
    solution: Solution | None = None,
    stats: SolverStats | None = None,
    eligibility: Mapping[str, object] | None = None,
) -> ProposalResult:
    return ProposalResult(partition, candidate_index, profile, stratum, 6, g, c,
                          reason, task, witness, count, solution, stats, eligibility)


def _validate_cell(
    config: ProbeConfig,
    partition: str,
    active_cell: object,
) -> tuple[tuple[str, ...], str, int, int]:
    if type(active_cell) is not tuple or len(active_cell) != 2:
        raise TypeError("active_cell must be a (profile, stratum) tuple")
    profile, stratum = active_cell
    if type(profile) is not tuple or profile not in config.profiles_for(partition):
        raise ValueError("active_cell profile is not registered for the partition")
    if type(stratum) is not str or stratum not in config.cell_order:
        raise ValueError("active_cell stratum is not registered")
    given_count, constraint_count = config.strata[stratum]
    return profile, stratum, given_count, constraint_count


def _source_eq_witnesses(
    config: ProbeConfig,
    given_variables: tuple[str, ...],
    constraint_count: int,
) -> tuple[Solution, ...]:
    result = []
    given_names = tuple(given_variables)
    for values in itertools.product((0, 1, 2), repeat=6):
        allowed = {values[config.variables.index(name)] for name in given_names}
        if set(values) <= allowed:
            witness = dict(zip(config.variables, values))
            if len(_witness_satisfied_tuples(6, RelationKind.EQ, witness)) >= constraint_count:
                result.append(_as_solution(config, values))
    return tuple(result)


def _source_neq_witnesses(
    config: ProbeConfig,
    given_variables: tuple[str, ...],
) -> tuple[Solution, ...]:
    given_indices = tuple(config.variables.index(name) for name in given_variables)
    result = []
    for values in itertools.product((0, 1, 2), repeat=6):
        if len(set(values)) == 3 and len({values[index] for index in given_indices}) >= 2:
            result.append(_as_solution(config, values))
    return tuple(result)


def _proposal_reject(
    partition: str,
    candidate_index: int,
    profile: tuple[str, ...],
    stratum: str,
    g: int,
    c: int,
    reason: str,
    witness: Solution | None = None,
) -> ProposalResult:
    return _result(partition, candidate_index, profile, stratum, g, c,
                   reason=reason, witness=witness)


def propose_candidate(
    config: ProbeConfig,
    partition: str,
    candidate_index: int,
    active_cell: tuple[tuple[str, ...], str],
) -> ProposalResult:
    """Propose one fixed-v4 task; all stream-level decisions belong to the caller."""
    if type(config) is not ProbeConfig:
        raise TypeError("config must be a ProbeConfig")
    _check_index(partition, candidate_index)
    profile, stratum, given_count, constraint_count = _validate_cell(config, partition, active_cell)
    rng = candidate_stream(config, partition, candidate_index)
    given_variables = tuple(sorted(rng.sample(config.variables, given_count)))

    witness: Solution
    if partition == "source" and profile == ("eq",):
        witnesses = _source_eq_witnesses(config, given_variables, constraint_count)
        if not witnesses:
            return _proposal_reject(partition, candidate_index, profile, stratum,
                                    given_count, constraint_count, "no_source_eq_witness")
        witness = rng.choice(witnesses)
        witness_values = dict(witness.assignments)
        pool = _witness_satisfied_tuples(6, RelationKind.EQ, witness_values)
        count_vectors = tuple(_count_vectors(profile, (len(pool), 0, 0), constraint_count))
        # Even the unique source-EQ vector consumes a uniform choice from the stream.
        counts = rng.choice(count_vectors)
        constraints = [Constraint(RelationKind.EQ, operands)
                       for operands in rng.sample(pool, counts[0])]
    elif partition == "source" and profile == ("neq",):
        witnesses = _source_neq_witnesses(config, given_variables)
        if not witnesses:
            return _proposal_reject(partition, candidate_index, profile, stratum,
                                    given_count, constraint_count, "no_neq_witness")
        witness = rng.choice(witnesses)
        witness_values = dict(witness.assignments)
        unknown = set(config.variables) - set(given_variables)
        pool = tuple(operands for operands in _witness_satisfied_tuples(
            6, RelationKind.NEQ, witness_values) if unknown.intersection(operands))
        if len(pool) < constraint_count:
            return _proposal_reject(partition, candidate_index, profile, stratum,
                                    given_count, constraint_count,
                                    "no_feasible_neq_constraint_total", witness)
        constraints = [Constraint(RelationKind.NEQ, operands)
                       for operands in rng.sample(pool, constraint_count)]
    else:
        values = tuple(rng.randrange(3) for _ in config.variables)
        witness = _as_solution(config, values)
        witness_values = dict(witness.assignments)
        pools = tuple(_witness_satisfied_tuples(6, kind, witness_values) for kind in KIND_ORDER)
        capacities = tuple(len(pool) if kind.value in profile else 0
                           for kind, pool in zip(KIND_ORDER, pools))
        vectors = tuple(_count_vectors(profile, capacities, constraint_count))
        if not vectors:
            return _proposal_reject(partition, candidate_index, profile, stratum,
                                    given_count, constraint_count,
                                    "no_feasible_constraint_vector", witness)
        counts = rng.choice(vectors)
        constraints = []
        for index, kind in enumerate(KIND_ORDER):
            if counts[index]:
                constraints.extend(Constraint(kind, operands)
                                   for operands in rng.sample(pools[index], counts[index]))

    constraints.sort(key=canonical_constraint_key)
    witness_values = dict(witness.assignments)
    givens = tuple(GivenAssignment(name, witness_values[name]) for name in given_variables)
    task = ConstraintTask("constraint-task-v1", 3, config.variables,
                          tuple(constraints), givens)
    oracle = enumerate_solutions(task)
    oracle_count = min(2, len(oracle.solutions))
    if oracle_count == 0:
        context = ProposalFatalContext(partition, candidate_index, profile, stratum, 6,
                                       given_count, constraint_count, task, witness, 0)
        raise ProposalFatalError("oracle_witness_disagreement", context)
    if oracle_count == 2:
        return _result(partition, candidate_index, profile, stratum,
                       given_count, constraint_count, task=task, witness=witness, count=2)

    oracle_solution = oracle.solutions[0]
    solved = solve(task)
    if solved.solution != oracle_solution:
        context = ProposalFatalContext(partition, candidate_index, profile, stratum, 6,
                                       given_count, constraint_count, task, witness, 1)
        raise ProposalFatalError("solver_oracle_disagreement", context)
    eligibility = _private_eligibility(partition, task)
    return _result(partition, candidate_index, profile, stratum, given_count,
                   constraint_count, task=task, witness=witness, count=1,
                   solution=oracle_solution, stats=solved.stats, eligibility=eligibility)
