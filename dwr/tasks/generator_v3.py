"""Pure deterministic v3 candidates with active-profile and source-neq proposals."""
from __future__ import annotations

import hashlib
import itertools
import random

from .generator_v2 import (
    CandidateHeader, CandidateOutcome, PROFILES, _finish_candidate,
    _propose_from_v2_witness, _validate, _normalized_tuples,
)
from .schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, Solution, canonical_constraint_key


GENERATOR_ID = "dwr-generator-v3"


def _validate_full_profiles(partition: str, full_profiles: frozenset[tuple[str, ...]]) -> None:
    if type(full_profiles) is not frozenset:
        raise TypeError("full_profiles must be a frozenset of profile tuples")
    if any(type(profile) is not tuple or profile not in PROFILES[partition]
           for profile in full_profiles):
        raise ValueError("full_profiles must contain only registered partition profiles")


def active_profile(partition: str, candidate_index: int,
                   full_profiles: frozenset[tuple[str, ...]]) -> tuple[str, ...]:
    """Select by candidate index among profiles still below quota, before n is drawn."""
    if partition not in PROFILES:
        raise ValueError("partition must be source, development, or final")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or not 0 <= candidate_index < 2**64:
        raise ValueError("candidate_index must be an unsigned 64-bit integer")
    _validate_full_profiles(partition, full_profiles)
    active = tuple(profile for profile in PROFILES[partition] if profile not in full_profiles)
    if not active:
        raise ValueError("no active profiles remain for candidate generation")
    return active[candidate_index % len(active)]


def _stream_seed(root_seed: int, partition: str, candidate_index: int) -> int:
    _validate(partition, candidate_index, root_seed)
    material = (b"dwr-generator-v3\0" + root_seed.to_bytes(8, "big") + b"\0" +
                partition.encode("ascii") + b"\0" + candidate_index.to_bytes(8, "big"))
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _rng(root_seed: int, partition: str, candidate_index: int) -> random.Random:
    rng = random.Random()
    rng.seed(_stream_seed(root_seed, partition, candidate_index), version=2)
    return rng


def sample_candidate_header_v3(root_seed: int, partition: str, candidate_index: int, *,
                               full_profiles: frozenset[tuple[str, ...]] = frozenset()) -> CandidateHeader:
    """Choose the active profile without RNG, then draw n from its candidate stream."""
    _validate(partition, candidate_index, root_seed)
    profile = active_profile(partition, candidate_index, full_profiles)
    rng = _rng(root_seed, partition, candidate_index)
    n = rng.randint(4, 6) if partition == "source" and profile == ("neq",) else rng.randint(3, 6)
    return CandidateHeader(n, profile)


def _neq_candidate(rng: random.Random, partition: str, candidate_index: int,
                   n: int, profile: tuple[str, ...]) -> CandidateOutcome:
    names = tuple(f"v{i}" for i in range(n))
    given_count = rng.randint(2, n - 2)
    given_variables = tuple(sorted(rng.sample(names, given_count)))
    given_set = set(given_variables)
    witnesses = tuple(
        Solution(tuple(zip(names, values)))
        for values in itertools.product((0, 1, 2), repeat=n)
        if len(set(values)) == 3 and len({values[names.index(name)] for name in given_variables}) >= 2
    )
    if not witnesses:
        return CandidateOutcome(partition, candidate_index, n, profile, None, None,
                                None, None, "no_neq_witness", None, GENERATOR_ID)
    witness = rng.choice(witnesses)
    values = dict(witness.assignments)
    unknown = set(names) - given_set
    pool = tuple(
        operands for operands in _normalized_tuples(n, RelationKind.NEQ)
        if values[operands[0]] != values[operands[1]] and unknown.intersection(operands)
    )
    low = 2 * (n - given_count)
    high = min(8, len(pool))
    if low > high:
        return CandidateOutcome(partition, candidate_index, n, profile, None, witness,
                                None, None, "no_feasible_neq_constraint_total", None, GENERATOR_ID)
    total = rng.randint(low, high)
    constraints = tuple(sorted(
        (Constraint(RelationKind.NEQ, operands) for operands in rng.sample(pool, total)),
        key=canonical_constraint_key))
    givens = tuple(GivenAssignment(name, values[name]) for name in given_variables)
    task = ConstraintTask("constraint-task-v1", 3, names, constraints, givens)
    return _finish_candidate(partition, candidate_index, n, profile, witness, task,
                             generator_id=GENERATOR_ID)


def generate_candidate_v3(root_seed: int, partition: str, candidate_index: int, *,
                          full_profiles: frozenset[tuple[str, ...]] = frozenset()) -> CandidateOutcome:
    """Generate one stateless v3 candidate; corpus deduplication remains orchestration-owned."""
    _validate(partition, candidate_index, root_seed)
    profile = active_profile(partition, candidate_index, full_profiles)
    rng = _rng(root_seed, partition, candidate_index)
    n = rng.randint(4, 6) if partition == "source" and profile == ("neq",) else rng.randint(3, 6)
    if partition == "source" and profile == ("neq",):
        return _neq_candidate(rng, partition, candidate_index, n, profile)
    names = tuple(f"v{i}" for i in range(n))
    witness = Solution(tuple((name, rng.randrange(3)) for name in names))
    return _propose_from_v2_witness(rng, partition, candidate_index, n, profile,
                                    witness, generator_id=GENERATOR_ID)
