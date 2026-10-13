import hashlib
import itertools
import random
import unittest

from dwr.tasks.generator import (
    CandidateHeader, _count_vectors, generate_candidate, profile_quota_outcome, sample_candidate_header,
)
from dwr.tasks.oracle import enumerate_solutions
from dwr.tasks.schema import RelationKind, serialize_task
from dwr.tasks.solver import solve
from dwr.tasks.verifier import verify_answer


def independent_rng(root, partition, index):
    material = (b"dwr-generator-v1\0" + root.to_bytes(8, "big") + b"\0" +
                partition.encode("ascii") + b"\0" + index.to_bytes(8, "big"))
    digest = hashlib.sha256(material).digest()
    stream_seed = int.from_bytes(digest[:8], "big")
    rng = random.Random()
    rng.seed(stream_seed, version=2)
    return rng, digest.hex(), stream_seed


class CandidateGeneratorTests(unittest.TestCase):
    def test_seed_material_and_header_match_independent_fixture(self):
        rng, digest, stream_seed = independent_rng(20261007, "source", 3)
        self.assertEqual(digest, "da206af82243d9d0f09cc20e5b31e56968033d3f681c86ccd6605b99fea50fa4")
        self.assertEqual(stream_seed, 15717680313482336720)
        expected = CandidateHeader(rng.randint(3, 6), rng.choice((("eq",), ("neq",), ("add_mod",))))
        self.assertEqual(expected, CandidateHeader(6, ("neq",)))
        self.assertEqual(sample_candidate_header(20261007, "source", 3), expected)

    def test_candidate_index_partition_streams_and_profile_quota_exit(self):
        a = sample_candidate_header(20261007, "source", 3)
        b = sample_candidate_header(20261007, "source", 4)
        c = sample_candidate_header(20261007, "development", 3)
        self.assertEqual((a.n, a.profile), (6, ("neq",)))
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        header = sample_candidate_header(7, "development", 5)
        quota = profile_quota_outcome(7, "development", 5, frozenset((header.profile,)))
        self.assertEqual(quota.rejection_reason, "profile_quota_full")
        self.assertEqual((quota.n, quota.profile), (header.n, header.profile))
        self.assertIsNone(quota.task)

    def test_independently_traced_accepted_candidate(self):
        # Trace candidate 27 directly from its independently derived RNG stream.
        rng, _, _ = independent_rng(20261007, "source", 27)
        n = rng.randint(3, 6)
        profile = rng.choice((("eq",), ("neq",), ("add_mod",)))
        names = tuple(f"v{i}" for i in range(n))
        self.assertEqual((n, profile), (4, ("eq",)))
        tuples = tuple(itertools.combinations(names, 2))
        total = rng.choice((2, 3, 4, 5, 6))
        vectors = tuple((count, 0, 0) for count in range(1, len(tuples) + 1) if count == total)
        counts = rng.choice(vectors)
        chosen = rng.sample(tuples, counts[0])
        self.assertEqual((total, counts, chosen), (2, (2, 0, 0), [("v0", "v1"), ("v0", "v2")]))
        witness = tuple(rng.randrange(3) for _ in names)
        clue_count = rng.randint(1, n - 2)
        given_vars = sorted(rng.sample(names, clue_count))
        self.assertEqual((witness, given_vars), ((0, 0, 0, 1), ["v2", "v3"]))

        outcome = generate_candidate(20261007, "source", 27)
        self.assertTrue(outcome.accepted)
        self.assertEqual(serialize_task(outcome.task), '{"constraints":[{"kind":"eq","variables":["v0","v1"]},{"kind":"eq","variables":["v0","v2"]}],"givens":[{"value":0,"variable":"v2"},{"value":1,"variable":"v3"}],"modulus":3,"schema_version":"constraint-task-v1","variables":["v0","v1","v2","v3"]}')
        self.assertEqual(outcome.witness.assignments, outcome.solution.assignments)
        self.assertEqual(outcome.solution, enumerate_solutions(outcome.task).solutions[0])
        self.assertEqual(outcome.solver_result.solution, outcome.solution)
        self.assertEqual(outcome.solver_result.stats.uniqueness_solution_count_capped, 1)
        answer = '{"assignment":{"v0":0,"v1":0,"v2":0,"v3":1}}'
        self.assertTrue(verify_answer(outcome.task, answer).valid)

    def test_rejection_reason_partition_and_no_module_random_state_change(self):
        state = random.getstate()
        witness_bad = generate_candidate(20261007, "source", 3)
        ambiguous = generate_candidate(20261007, "source", 7)
        self.assertEqual(witness_bad.rejection_reason, "witness_invalid")
        self.assertEqual(ambiguous.rejection_reason, "oracle_non_unique")
        self.assertEqual(random.getstate(), state)
        self.assertIsNone(witness_bad.solver_result)
        self.assertIsNone(ambiguous.solver_result)

    def test_invalid_indices_and_partitions_fail_closed(self):
        for args in ((True, "source", 0), (-1, "source", 0), (2**64, "source", 0),
                     (1, "other", 0), (1, "source", -1), (1, "source", True), (1, "source", 2**64)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                generate_candidate(*args)

    def test_count_vector_order_and_immutable_public_records(self):
        self.assertEqual(tuple(_count_vectors(("eq", "neq"), (3, 3, 0), 4)),
                         ((1, 3, 0), (2, 2, 0), (3, 1, 0)))
        with self.assertRaises(ValueError):
            CandidateHeader(3, ["eq"])


if __name__ == "__main__":
    unittest.main()
