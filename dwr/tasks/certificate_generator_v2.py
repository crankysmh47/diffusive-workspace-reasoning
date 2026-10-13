"""Versioned six-variable certificate construction for the v2 fixture."""
from __future__ import annotations

import hashlib
import itertools
import random
from typing import Callable

from .certificate_generator import (
    CELL_SETTINGS,
    KIND_BY_NAME,
    PROFILES,
    CertificateOutcome,
    _Work,
    _choice,
    _normalized_tuples,
    _outcome,
    _positive_count_vectors,
    _profile_cell_schedule,
    _sample,
    _solution,
    _unknown_alternatives,
    _validate,
    _witness_values as _v1_witness_values,
    _satisfies,
)
from .schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, canonical_constraint_key


_SEED_TAG = b"dwr-certificate-sixvar-v2\0"


def _rng(root_seed: int, partition: str, candidate_index: int) -> random.Random:
    material = (_SEED_TAG + root_seed.to_bytes(8, "big") + b"\0" + partition.encode("ascii") +
                b"\0" + candidate_index.to_bytes(8, "big"))
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = random.Random()
    rng.seed(seed, version=2)
    return rng


def _add_branch(candidate_index: int) -> str | None:
    """Return the fixed ordinary/no-unary branch, or None for non-ADD profiles."""
    profile, _ = _profile_cell_schedule("development")[candidate_index % 12]
    if "add_mod" not in profile:
        return None
    return "ordinary" if (candidate_index // 12) % 2 == 0 else "no_unary"


def _eqneq_witnesses(given_positions: tuple[str, ...], work: _Work) -> tuple[tuple[int, ...], ...]:
    result = []
    for values in itertools.product(range(3), repeat=6):
        work.scan("witness_pool_scans")
        if len(set(values)) == 3 and len({values[int(name[1:])] for name in given_positions}) >= 2:
            result.append(values)
    work.boundary()
    return tuple(result)


def _eqneq_first_alternatives(witness, alternatives, work):
    witness_values = dict(witness.assignments)
    eq_pairs = _normalized_tuples(RelationKind.EQ)
    result = []
    for alternative in alternatives:
        values = dict(alternative.assignments)
        rejects_witness_pair = False
        for pair in eq_pairs:
            work.scan("constraint_pool_scans")
            if (_satisfies(RelationKind.EQ, pair, witness_values) and
                    not _satisfies(RelationKind.EQ, pair, values)):
                rejects_witness_pair = True
        if rejects_witness_pair:
            result.append(alternative)
    work.boundary()
    return tuple(result)


def _allowed_no_unary_tuple(operands, given_variables):
    unknown_operands = sum(name not in given_variables for name in operands)
    return unknown_operands != 1


def _mixed_feasible_second_alternatives(profile, constraint_count, witness, first,
                                        alternatives, work, *, givens, no_unary):
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
                if no_unary and not _allowed_no_unary_tuple(operands, givens):
                    continue
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


def construct(root_seed: int, partition: str, candidate_index: int, *,
              check_budget: Callable[[], None] | None = None) -> CertificateOutcome:
    """Construct exactly one source/development candidate using its v2 RNG stream."""
    _validate(root_seed, partition, candidate_index, check_budget)
    if partition == "source":
        from .certificate_generator import construct as construct_v1
        return construct_v1(root_seed, partition, candidate_index, check_budget=check_budget)

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

    if profile == ("eq", "neq"):
        witness_pool = _eqneq_witnesses(given_positions, work)
        if not witness_pool:
            return _outcome(root_seed, partition, candidate_index, profile, cell, (), None,
                            (), (), work, None, "empty_witness_pool")
        work.rng_calls += 1
        witness_values = rng.choice(witness_pool)
    else:
        witness_values = _v1_witness_values(profile, cell, given_positions, given_count,
                                            constraint_count, rng, work)
        if witness_values is None:
            return _outcome(root_seed, partition, candidate_index, profile, cell, (), None,
                            (), (), work, None, "empty_witness_pool")

    witness = _solution(witness_values)
    values = dict(witness.assignments)
    givens = tuple(GivenAssignment(name, values[name]) for name in given_positions)
    given_map = {given.variable: given.value for given in givens}
    all_alternatives = _unknown_alternatives(given_map, witness, work)
    first_pool = (_eqneq_first_alternatives(witness, all_alternatives, work)
                  if profile == ("eq", "neq") else all_alternatives)
    first = _choice(rng, first_pool, work)
    branch = _add_branch(candidate_index)
    second_feasible = _mixed_feasible_second_alternatives(
        profile, constraint_count, witness, first, all_alternatives, work,
        givens=frozenset(given_map), no_unary=(branch == "no_unary"))
    if not second_feasible:
        return _outcome(root_seed, partition, candidate_index, profile, cell, givens,
                        witness, ((profile[0], first),), (), work, None,
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
