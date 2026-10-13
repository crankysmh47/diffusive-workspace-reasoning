"""Versioned, algebraically certified pair construction for six-variable F3 tasks.

No solver, verifier, split, GAC, or signature logic is used here.  The final
partition is intentionally unavailable until the separately reviewed route lands.
"""
from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Callable

from . import certificate_generator as legacy
from . import certificate_generator_v2 as v2
from .f3 import nullspace as _nullspace, rank
from .schema import (Constraint, ConstraintTask, GivenAssignment, RelationKind,
                     Solution, canonical_constraint_key)

GENERATOR_ID = "dwr-certificate-rank-sixvar-v1"
_SEED_TAG = b"dwr-certificate-rank-sixvar-v1\0"
_KINDS = {"eq": RelationKind.EQ, "neq": RelationKind.NEQ, "add_mod": RelationKind.ADD_MOD}
_NAMES = tuple(f"v{i}" for i in range(6))


@dataclass(frozen=True, slots=True)
class RankWork:
    catalog_visits: int = 0
    rank_calls: int = 0
    row_operations: int = 0
    graph_operations: int = 0
    rng_calls: int = 0

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (
                self.catalog_visits, self.rank_calls, self.row_operations,
                self.graph_operations, self.rng_calls)):
            raise ValueError("work counters must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class RankOutcome:
    root_seed: int
    partition: str
    candidate_index: int
    profile: tuple[str, ...]
    cell: str
    givens: tuple[GivenAssignment, ...]
    witness: Solution | None
    alternatives: tuple[tuple[str, Solution], ...]
    reserved_constraints: tuple[Constraint, ...]
    work: RankWork
    task: ConstraintTask | None
    rejection_reason: str | None
    generator_id: str = GENERATOR_ID
    branch: str = "no_unary"

    def __post_init__(self):
        _identity(self.root_seed, self.partition, self.candidate_index, None)
        if (type(self.generator_id) is not str or type(self.branch) is not str or
                self.generator_id != GENERATOR_ID or self.branch != "no_unary"):
            raise ValueError("rank outcomes require their registered identity and branch")
        if (self.partition != "development" or type(self.profile) is not tuple or
                any(type(kind) is not str for kind in self.profile) or
                self.profile not in (("eq", "add_mod"), ("neq", "add_mod"))):
            raise ValueError("rank outcomes are limited to development pair profiles")
        if type(self.work) is not RankWork or type(self.givens) is not tuple:
            raise TypeError("outcome collections and work must be immutable registered types")
        schedule = legacy._profile_cell_schedule("development")
        if (self.profile, self.cell) != schedule[self.candidate_index % len(schedule)]:
            raise ValueError("profile and cell must match the fixed development schedule")
        if v2._add_branch(self.candidate_index) != "no_unary":
            raise ValueError("algebraic outcomes must occupy the frozen no_unary half-schedule")
        if any(type(item) is not GivenAssignment for item in self.givens):
            raise TypeError("givens must contain immutable GivenAssignment records")
        if type(self.alternatives) is not tuple or any(
                type(item) is not tuple or len(item) != 2 or type(item[0]) is not str or
                type(item[1]) is not Solution for item in self.alternatives):
            raise TypeError("alternatives must be immutable kind and solution pairs")
        if type(self.reserved_constraints) is not tuple or any(
                type(item) is not Constraint for item in self.reserved_constraints):
            raise TypeError("reserved constraints must be an immutable tuple")
        if self.task is not None and type(self.task) is not ConstraintTask:
            raise TypeError("task must be ConstraintTask or None")
        if self.witness is not None and type(self.witness) is not Solution:
            raise TypeError("witness must be Solution or None")
        if self.witness is None or tuple(name for name, _ in self.witness.assignments) != _NAMES:
            raise ValueError("every algebraic outcome must preserve its complete drawn witness")
        witness_map = dict(self.witness.assignments)
        if any(type(value) is not int or value not in (0, 1, 2)
               for _, value in self.witness.assignments):
            raise ValueError("witness values must be F3 integers")
        given_map = {g.variable: g.value for g in self.givens}
        if (len(given_map) != len(self.givens) or
                len(self.givens) != legacy.CELL_SETTINGS[self.cell][0] or
                tuple(g.variable for g in self.givens) != tuple(sorted(given_map)) or
                any(witness_map.get(name) != value for name, value in given_map.items())):
            raise ValueError("givens must be unique, sorted, cell-sized, and agree with witness")
        failures = {"empty_eq_pool", "rank_completion_failed", "missing_fiber_exclusion",
                    "insufficient_filler", "disconnected_graph"}
        if self.task is None and self.rejection_reason not in failures:
            raise ValueError("missing task requires a registered failure tag")
        if ((self.rejection_reason == "empty_eq_pool" and self.profile[0] != "eq") or
                (self.rejection_reason == "missing_fiber_exclusion" and self.profile[0] != "neq")):
            raise ValueError("failure tag is incompatible with the pair profile")
        if self.task is not None and self.rejection_reason is not None:
            raise ValueError("successful task cannot carry a failure tag")
        if self.task is None and (self.alternatives or self.reserved_constraints):
            raise ValueError("failed outcomes cannot carry partial alternatives or reserves")
        if self.task is not None:
            _, count = legacy.CELL_SETTINGS[self.cell]
            if (self.task.variables != _NAMES or self.task.givens != self.givens or
                    len(self.task.constraints) != count or
                    tuple(sorted(self.task.constraints, key=canonical_constraint_key)) != self.task.constraints):
                raise ValueError("task shape, givens, or constraint order disagree with its outcome")
            if tuple(kind for kind, _ in self.alternatives) != self.profile:
                raise ValueError("successful alternatives must follow profile order")
            if tuple(c.kind.value for c in self.reserved_constraints) != self.profile:
                raise ValueError("successful outcomes require one ordered reserve per kind")
            if self.witness is None or tuple(name for name, _ in self.witness.assignments) != _NAMES:
                raise ValueError("successful outcome requires a complete six-variable witness")
            if {c.kind.value for c in self.task.constraints} != set(self.profile) or any(
                    not _holds(c.kind.value, c.variables, witness_map) for c in self.task.constraints):
                raise ValueError("task kinds and witness must match the registered profile")
            alt_maps = {}
            for kind, solution in self.alternatives:
                if tuple(name for name, _ in solution.assignments) != _NAMES:
                    raise ValueError("alternatives must be complete six-variable assignments")
                alt = dict(solution.assignments)
                if alt == witness_map or any(alt[name] != value for name, value in given_map.items()):
                    raise ValueError("alternatives must differ from the witness and preserve givens")
                alt_maps[kind] = alt
            for kind, reserve in zip(self.profile, self.reserved_constraints):
                if reserve not in self.task.constraints or _holds(kind, reserve.variables, alt_maps[kind]):
                    raise ValueError("each reserve must be present and reject its own alternative")
                other = alt_maps[self.profile[1] if kind == self.profile[0] else self.profile[0]]
                if any(c.kind.value == kind and not _holds(kind, c.variables, other)
                       for c in self.task.constraints):
                    raise ValueError("all constraints of a kind must preserve the other-kind alternative")


class _Work:
    def __init__(self, callback):
        self.callback = callback
        self.catalog_visits = self.rank_calls = self.row_operations = 0
        self.graph_operations = self.rng_calls = 0
        self._rank_counts = {}

    def tick(self):
        if self.callback is not None:
            self.callback()

    def catalog(self):
        self.catalog_visits += 1
        self.tick()

    def rank(self, rows, width):
        before = self._rank_counts.get("row_operations", 0)
        result = rank(rows, width, check_budget=self.callback, counters=self._rank_counts)
        self.rank_calls = self._rank_counts.get("rank_calls", 0)
        self.row_operations += self._rank_counts.get("row_operations", 0) - before
        return result

    def nullspace(self, rows, width):
        before = self._rank_counts.get("row_operations", 0)
        result = _nullspace(rows, width, check_budget=self.callback, counters=self._rank_counts)
        self.rank_calls = self._rank_counts.get("rank_calls", 0)
        self.row_operations += self._rank_counts.get("row_operations", 0) - before
        return result

    def freeze(self):
        return RankWork(self.catalog_visits, self.rank_calls, self.row_operations,
                        self.graph_operations, self.rng_calls)


def _identity(root_seed, partition, index, check_budget):
    if type(root_seed) is not int or not 0 <= root_seed < 2**64:
        raise ValueError("root_seed must be an unsigned 64-bit integer")
    if type(index) is not int or not 0 <= index < 2**64:
        raise ValueError("candidate_index must be an unsigned 64-bit integer")
    if type(partition) is not str or partition not in ("source", "development", "final"):
        raise ValueError("partition must be source, development, or final")
    if check_budget is not None and not callable(check_budget):
        raise TypeError("check_budget must be callable or None")


def _rng(root_seed, partition, index):
    material = (_SEED_TAG + root_seed.to_bytes(8, "big") + b"\0" + partition.encode("ascii") +
                b"\0" + index.to_bytes(8, "big"))
    rng = random.Random()
    rng.seed(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"), version=2)
    return rng


def _pick(rng, pool, work):
    if not pool:
        return None
    work.rng_calls += 1
    return rng.choice(pool)


def _sample(rng, pool, count, work):
    if count < 0 or len(pool) < count:
        return None
    if count == 0:
        return ()
    work.rng_calls += 1
    return tuple(rng.sample(pool, count))


def _full_catalog(given_map, witness, unknown, kinds, *, no_unary, work):
    """Return allowed witness-true legal rows, with given roles substituted."""
    U = tuple(unknown)
    pos = {name: i for i, name in enumerate(U)}
    values = dict(witness.assignments)
    out = {kind: [] for kind in kinds}
    tuples = {}
    for kind in kinds:
        if kind in ("eq", "neq"):
            tuples[kind] = itertools.combinations(_NAMES, 2)
        else:
            tuples[kind] = ((*pair, target) for pair in itertools.combinations(_NAMES, 2)
                            for target in _NAMES if target not in pair)
        for operands in tuples[kind]:
            work.catalog()
            if not _holds(kind, operands, values):
                continue
            scope = tuple(name for name in operands if name in pos)
            if no_unary and len(scope) == 1:
                continue
            if kind == "neq":
                if values[operands[0]] == values[operands[1]]:
                    continue
                row = [0] * len(U)
                for name, sign in ((operands[0], 1), (operands[1], -1)):
                    if name in pos:
                        row[pos[name]] += sign
                out[kind].append((operands, tuple(x % 3 for x in row)))
                continue
            coeff = [0] * len(U)
            full = (((operands[0], 1), (operands[1], -1)) if kind == "eq" else
                    ((operands[0], 1), (operands[1], 1), (operands[2], -1)))
            for name, sign in full:
                if name in pos:
                    coeff[pos[name]] += sign
            out[kind].append((operands, tuple(x % 3 for x in coeff)))
    return out


def _constraint(kind, operands):
    return Constraint(_KINDS[kind], tuple(operands))


def _holds(kind, operands, values):
    x = tuple(values[name] for name in operands)
    return (x[0] == x[1] if kind == "eq" else
            x[0] != x[1] if kind == "neq" else (x[0] + x[1]) % 3 == x[2])


def _connected(selected, unknown, work):
    graph = {name: set() for name in unknown}
    for kind, operands in selected:
        scope = [name for name in operands if name in graph]
        for a in scope:
            for b in scope:
                work.graph_operations += 1
                work.tick()
                if a != b:
                    graph[a].add(b)
    seen = set()
    pending = [unknown[0]] if unknown else []
    while pending:
        node = pending.pop()
        work.graph_operations += 1
        work.tick()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(graph[node] - seen)
    return len(seen) == len(unknown)


def _fiber_exclusion_pools(catalog, unknown, values, vector, work):
    """Partition true unknown-pair NEQs by the nonzero fiber point they exclude."""
    pools = {1: [], 2: []}
    for operands, row in catalog:
        work.catalog()
        a, b = operands
        if a not in unknown or b not in unknown:
            continue
        i, j = unknown.index(a), unknown.index(b)
        slope = (vector[i] - vector[j]) % 3
        if not slope:
            continue
        delta = (values[a] - values[b]) % 3
        t = (-delta * (1 if slope == 1 else 2)) % 3
        if t in pools:
            pools[t].append((operands, row))
    return pools


def _construct_pair(root_seed, partition, index, profile, cell, given_positions,
                    witness_values, rng, *, constraint_count=None, check_budget=None,
                    branch="no_unary", _work=None):
    """Production-connected helper; accepts a canned witness for correctness checks."""
    _identity(root_seed, partition, index, check_budget)
    if type(profile) is not tuple or profile not in (("eq", "add_mod"), ("neq", "add_mod")):
        raise ValueError("pair profile must be EQ/ADD or NEQ/ADD")
    if type(cell) is not str or cell not in legacy.CELL_SETTINGS:
        raise ValueError("unknown support cell")
    given_count, target_count = legacy.CELL_SETTINGS[cell]
    if constraint_count is not None:
        if type(constraint_count) is not int or constraint_count != target_count:
            raise ValueError("constraint_count must match the frozen support cell")
    if type(given_positions) is not tuple or len(given_positions) != given_count or \
            len(set(given_positions)) != given_count or any(x not in _NAMES for x in given_positions):
        raise ValueError("given positions must be unique valid names matching the cell")
    if type(witness_values) is not tuple or len(witness_values) != 6 or \
            any(type(x) is not int or x not in (0, 1, 2) for x in witness_values):
        raise ValueError("witness must contain six F3 integers")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be a local random.Random instance")
    work = _work if _work is not None else _Work(check_budget)
    values = dict(zip(_NAMES, witness_values))
    givens = tuple(GivenAssignment(name, values[name]) for name in sorted(given_positions))
    given_map = dict((g.variable, g.value) for g in givens)
    unknown = tuple(name for name in _NAMES if name not in given_map)
    m = len(unknown)
    witness = Solution(tuple(zip(_NAMES, witness_values)))
    catalogs = _full_catalog(given_map, witness, unknown, profile, no_unary=True, work=work)

    if profile[0] == "eq":
        eq_pool = tuple(item for item in catalogs["eq"] if any(item[1]))
        first = _pick(rng, eq_pool, work)
        if first is None:
            reason = "empty_eq_pool"
            return RankOutcome(root_seed, partition, index, profile, cell, givens, witness, (), (), work.freeze(), None, reason, branch=branch)
        selected = [("eq", first[0])]
        rows = [first[1]]
        add_pool = catalogs["add_mod"]
        eligible = []
        for item in add_pool:
            work.catalog()
            if work.rank(rows + [item[1]], m) > len(rows):
                eligible.append(item)
        work.tick()
        second = _pick(rng, eligible, work)
        if second is None:
            reason = "rank_completion_failed"
            return RankOutcome(root_seed, partition, index, profile, cell, givens, witness, (), (), work.freeze(), None, reason, branch=branch)
        selected.append(("add_mod", second[0])); rows.append(second[1])
        while len(rows) < m:
            choices = []
            for kind in ("eq", "add_mod"):
                for item in catalogs[kind]:
                    work.catalog()
                    if (kind, item[0]) in selected:
                        continue
                    if work.rank(rows + [item[1]], m) > len(rows):
                        choices.append((kind, item))
            work.tick()
            choice = _pick(rng, choices, work)
            if choice is None:
                break
            kind, item = choice
            selected.append((kind, item[0])); rows.append(item[1])
        if len(rows) != m:
            reason = "rank_completion_failed"
            return RankOutcome(root_seed, partition, index, profile, cell, givens, witness, (), (), work.freeze(), None, reason, branch=branch)
        add_rows = [row for (kind, _), row in zip(selected, rows) if kind == "add_mod"]
        kernel = work.nullspace(add_rows, m)
        if not kernel:
            raise AssertionError("mixed full-rank basis must have a nontrivial ADD kernel")
        coefficient_vectors = []
        for candidate in itertools.product(range(3), repeat=len(kernel)):
            work.catalog()
            if any(candidate):
                coefficient_vectors.append(candidate)
        coefficient = _pick(rng, coefficient_vectors, work)
        vector = tuple(sum(coefficient[j] * kernel[j][i] for j in range(len(kernel))) % 3
                       for i in range(m))
        alternative_vector = tuple((a + b) % 3 for a, b in zip(tuple(values[n] for n in unknown), vector))
        alternatives = {"eq": _lift(alternative_vector, unknown, given_map),
                        "add_mod": _lift(tuple((values[n] + 1) % 3 for n in unknown), unknown, given_map)}
        return _finish(root_seed, partition, index, profile, cell, givens, witness, unknown,
                       selected, catalogs, alternatives, target_count, rng, work, check_budget, branch)

    # NEQ/ADD: exactly m-1 ADD rows, then opposing fiber exclusions.
    selected = []
    rows = []
    while len(rows) < m - 1:
        eligible = []
        for item in catalogs["add_mod"]:
            work.catalog()
            if ("add_mod", item[0]) not in selected and work.rank(rows + [item[1]], m) > len(rows):
                eligible.append(item)
        work.tick()
        item = _pick(rng, eligible, work)
        if item is None:
            break
        selected.append(("add_mod", item[0])); rows.append(item[1])
    if len(rows) != m - 1:
        reason = "rank_completion_failed"
        return RankOutcome(root_seed, partition, index, profile, cell, givens, witness, (), (), work.freeze(), None, reason, branch=branch)
    vector = work.nullspace(rows, m)[0]
    first_nonzero = next(value for value in vector if value)
    vector = tuple((value if first_nonzero == 1 else 2 * value) % 3 for value in vector)
    fiber_values = tuple(values[n] for n in unknown)
    exclusion_pools = _fiber_exclusion_pools(catalogs["neq"], unknown, values, vector, work)
    first, second = _pick(rng, exclusion_pools[1], work), _pick(rng, exclusion_pools[2], work)
    if first is None or second is None:
        reason = "missing_fiber_exclusion"
        return RankOutcome(root_seed, partition, index, profile, cell, givens, witness, (), (), work.freeze(), None, reason, branch=branch)
    selected.extend((("neq", first[0]), ("neq", second[0])))
    alternatives = {"neq": _lift(tuple((x + y) % 3 for x, y in zip(fiber_values, vector)), unknown, given_map),
                    "add_mod": _lift(tuple((values[n] + 1) % 3 for n in unknown), unknown, given_map)}
    return _finish(root_seed, partition, index, profile, cell, givens, witness, unknown,
                   selected, catalogs, alternatives, target_count, rng, work, check_budget, branch,
                   neq_vector=vector)


def _lift(unknown_values, unknown, given_map):
    vals = dict(zip(unknown, unknown_values)); vals.update(given_map)
    return Solution(tuple((n, vals[n]) for n in _NAMES))


def _linear_rank_for_kind(selected, catalogs, kind, work):
    lookup = {operands: row for operands, row in catalogs[kind]}
    rows = [lookup[operands] for selected_kind, operands in selected if selected_kind == kind]
    return work.rank(rows, len(next(iter(lookup.values()))) if lookup else 0)


def _finish(root_seed, partition, index, profile, cell, givens, witness, unknown,
            selected, catalogs, alternatives, target_count, rng, work, check_budget,
            branch, neq_vector=None):
    witness_map = dict(witness.assignments)
    reserved = []
    for kind in profile:
        alt = dict(alternatives[kind].assignments)
        own = next((operands for selected_kind, operands in selected
                    if selected_kind == kind and not _holds(kind, operands, alt)), None)
        if own is None:
            return RankOutcome(root_seed, partition, index, profile, cell, givens, witness, (), (), work.freeze(), None,
                               "rank_completion_failed", branch=branch)
        reserved.append(_constraint(kind, own))
    chosen = set(selected)
    fillers = []
    for kind in profile:
        for operands, row in catalogs[kind]:
            work.catalog()
            if (kind, operands) in chosen or not _holds(kind, operands, witness_map):
                continue
            if kind == "add_mod":
                if profile[0] == "eq":
                    v = tuple((dict(alternatives["eq"].assignments)[n] - witness_map[n]) % 3 for n in unknown)
                else:
                    v = neq_vector
                if sum(a*b for a, b in zip(row, v)) % 3:
                    continue
            fillers.append((kind, operands))
    needed = target_count - len(selected)
    sampled = _sample(rng, tuple(fillers), needed, work)
    if sampled is None:
        return RankOutcome(root_seed, partition, index, profile, cell, givens, witness, (), (), work.freeze(), None,
                           "insufficient_filler", branch=branch)
    selected.extend(sampled)
    if not _connected(selected, unknown, work):
        return RankOutcome(root_seed, partition, index, profile, cell, givens, witness, (), (), work.freeze(), None,
                           "disconnected_graph", branch=branch)
    constraints = tuple(sorted((_constraint(k, op) for k, op in selected), key=canonical_constraint_key))
    task = ConstraintTask("constraint-task-v1", 3, _NAMES, constraints, givens)
    ordered_alts = tuple((kind, alternatives[kind]) for kind in profile)
    return RankOutcome(root_seed, partition, index, profile, cell, givens, witness, ordered_alts,
                       tuple(reserved), work.freeze(), task, None, branch=branch)


def construct(root_seed, partition, candidate_index, *, check_budget=None):
    """Construct one candidate; unchanged branches delegate to the v1 producer."""
    _identity(root_seed, partition, candidate_index, check_budget)
    if partition == "final":
        raise NotImplementedError("final rank-certified construction is reserved for Task 2")
    schedule = legacy._profile_cell_schedule(partition)
    profile, cell = schedule[candidate_index % len(schedule)]
    if partition == "source":
        return legacy.construct(root_seed, partition, candidate_index, check_budget=check_budget)
    branch = v2._add_branch(candidate_index)
    if branch != "no_unary":
        return v2.construct(root_seed, partition, candidate_index, check_budget=check_budget)
    rng = _rng(root_seed, partition, candidate_index)
    work = _Work(check_budget)
    given_count, _ = legacy.CELL_SETTINGS[cell]
    work.rng_calls += 1
    given_positions = tuple(sorted(rng.sample(_NAMES, given_count)))
    work.tick()
    witness_values = []
    for _ in range(6):
        work.rng_calls += 1
        witness_values.append(rng.randrange(3))
        work.tick()
    return _construct_pair(root_seed, partition, candidate_index, profile, cell,
                           given_positions, tuple(witness_values), rng, constraint_count=legacy.CELL_SETTINGS[cell][1],
                           check_budget=check_budget,
                           branch=branch, _work=work)
