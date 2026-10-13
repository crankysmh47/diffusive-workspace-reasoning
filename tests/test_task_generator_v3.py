import hashlib
import itertools
import json
import random
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from dwr.tasks.generator_v2 import (
    CandidateOutcome, CandidateHeader, PROFILES, generate_candidate,
)
from dwr.tasks.oracle import enumerate_solutions
from dwr.tasks.schema import (Constraint, ConstraintTask, GivenAssignment, RelationKind,
                              Solution, serialize_model_input, serialize_task)
from dwr.tasks import generator_v3 as v3


class GeneratorV3Tests(unittest.TestCase):
    def test_v2_outcome_defaults_to_v2_and_rejects_v3_only_reason(self):
        source = generate_candidate(20261007, "source", 5)
        self.assertEqual(source.generator_id, "dwr-generator-v2")
        with self.assertRaises(ValueError):
            replace(source, rejection_reason="no_neq_witness")

    def test_stream_seed_matches_independent_known_answer_vectors(self):
        expected = {
            ("source", 0): ("953a4844612e7e0a9dc3d8b3d1e3046b2d72614d37aaa7f20bcb2e06f04c5bba", 10752986518825893386),
            ("development", 17): ("f2805809153b37fb997b1e7e00fef5adae5a6f61e9bf2fe7a6eaab611364e33b", 17474063350231676923),
            ("final", 9): ("7c7986d4016141b78e5ec24531c05068f59d3a4947d428e12a8d401d94151f0e", 8969348377999393207),
        }
        for (partition, index), (digest_hex, seed) in expected.items():
            material = (b"dwr-generator-v3\0" + (20261007).to_bytes(8, "big") + b"\0" +
                        partition.encode("ascii") + b"\0" + index.to_bytes(8, "big"))
            digest = hashlib.sha256(material).digest()
            self.assertEqual(digest.hex(), digest_hex)
            self.assertEqual(int.from_bytes(digest[:8], "big"), seed)
            self.assertEqual(v3._stream_seed(20261007, partition, index), seed)

    def test_active_profile_is_selected_before_n_and_without_rng_draws(self):
        self.assertEqual(v3.active_profile("development", 4, frozenset({("eq", "neq")})),
                         ("eq", "add_mod"))
        tracker = _CountingRng()
        with patch.object(v3, "_rng", return_value=tracker):
            header = v3.sample_candidate_header_v3(20261007, "development", 4)
        self.assertEqual(header.profile, ("eq", "add_mod"))
        self.assertEqual(tracker.calls, [("randint", 3, 6)])
        self.assertEqual(v3.active_profile("source", 7, frozenset({("eq",)})), ("add_mod",))
        tracker = _CountingRng()
        with patch.object(v3, "_rng", return_value=tracker):
            header = v3.sample_candidate_header_v3(20261007, "source", 0,
                                                  full_profiles=frozenset({("eq",)}))
        self.assertEqual((header.n, header.profile), (4, ("neq",)))
        self.assertEqual(tracker.calls, [("randint", 4, 6)])

    def test_empty_active_profiles_fail_before_rng_construction(self):
        with patch.object(v3, "_rng", side_effect=AssertionError("RNG constructed")):
            with self.assertRaises(ValueError):
                v3.generate_candidate_v3(20261007, "source", 0,
                                         full_profiles=frozenset(PROFILES["source"]))
            with self.assertRaises(ValueError):
                v3.generate_candidate_v3(20261007, "source", 0,
                                         full_profiles=frozenset({("unregistered",)}))
        with self.assertRaises(TypeError):
            v3.generate_candidate_v3(20261007, "source", 0, full_profiles=[("eq",)])

    def test_v3_candidate_fixtures_match_exact_replay_and_private_projection(self):
        expected = {
            ("source", 0): (5, ("eq",), "oracle_non_unique",
                (("v0", 0), ("v1", 1), ("v2", 0), ("v3", 0), ("v4", 1)),
                '{"constraints":[{"kind":"eq","variables":["v0","v2"]},{"kind":"eq","variables":["v0","v3"]}],"givens":[{"value":0,"variable":"v0"},{"value":0,"variable":"v2"},{"value":0,"variable":"v3"}],"modulus":3,"schema_version":"constraint-task-v1","variables":["v0","v1","v2","v3","v4"]}'),
            ("source", 1): (4, ("neq",), None,
                (("v0", 2), ("v1", 0), ("v2", 1), ("v3", 2)),
                '{"constraints":[{"kind":"neq","variables":["v0","v1"]},{"kind":"neq","variables":["v0","v2"]},{"kind":"neq","variables":["v1","v3"]},{"kind":"neq","variables":["v2","v3"]}],"givens":[{"value":0,"variable":"v1"},{"value":1,"variable":"v2"}],"modulus":3,"schema_version":"constraint-task-v1","variables":["v0","v1","v2","v3"]}'),
            ("source", 8): (4, ("add_mod",), None,
                (("v0", 2), ("v1", 0), ("v2", 0), ("v3", 1)),
                '{"constraints":[{"kind":"add_mod","variables":["v0","v3","v1"]},{"kind":"add_mod","variables":["v0","v3","v2"]}],"givens":[{"value":0,"variable":"v2"},{"value":1,"variable":"v3"}],"modulus":3,"schema_version":"constraint-task-v1","variables":["v0","v1","v2","v3"]}'),
            ("development", 17): (4, ("neq", "add_mod"), None,
                (("v0", 2), ("v1", 2), ("v2", 2), ("v3", 0)),
                '{"constraints":[{"kind":"add_mod","variables":["v0","v3","v1"]},{"kind":"add_mod","variables":["v0","v3","v2"]},{"kind":"add_mod","variables":["v1","v3","v0"]},{"kind":"add_mod","variables":["v1","v3","v2"]},{"kind":"add_mod","variables":["v2","v3","v0"]},{"kind":"add_mod","variables":["v2","v3","v1"]},{"kind":"neq","variables":["v1","v3"]},{"kind":"neq","variables":["v2","v3"]}],"givens":[{"value":2,"variable":"v2"}],"modulus":3,"schema_version":"constraint-task-v1","variables":["v0","v1","v2","v3"]}'),
            ("final", 9): (3, ("eq", "neq", "add_mod"), None,
                (("v0", 2), ("v1", 2), ("v2", 1)),
                '{"constraints":[{"kind":"add_mod","variables":["v0","v1","v2"]},{"kind":"eq","variables":["v0","v1"]},{"kind":"neq","variables":["v0","v2"]},{"kind":"neq","variables":["v1","v2"]}],"givens":[{"value":2,"variable":"v0"}],"modulus":3,"schema_version":"constraint-task-v1","variables":["v0","v1","v2"]}'),
        }
        expected_private = {
            ("source", 0): None,
            ("source", 1): {"oracle_unique": True,
                             "source": {"removal_solution_counts": (2, 2, 2, 2), "sensitive": True},
                             "mixed": None},
            ("source", 8): {"oracle_unique": True,
                             "source": {"removal_solution_counts": (3, 3), "sensitive": True},
                             "mixed": None},
            ("development", 17): {"oracle_unique": True, "source": None,
                "mixed": {"kinds": ("add_mod", "neq"), "kind_removal_solution_counts": (12, 1),
                          "factor_graph_connected": True, "factor_graph_kinds": ("add_mod", "neq"),
                          "all_present_kinds_in_graph": True, "primary_eligible": False,
                          "ineligibility_reason": "mixed_kind_not_necessary"}},
            ("final", 9): {"oracle_unique": True, "source": None,
                "mixed": {"kinds": ("add_mod", "eq", "neq"),
                          "kind_removal_solution_counts": (2, 2, 1),
                          "factor_graph_connected": True, "factor_graph_kinds": ("add_mod", "eq", "neq"),
                          "all_present_kinds_in_graph": True, "primary_eligible": False,
                          "ineligibility_reason": "mixed_kind_not_necessary"}},
        }
        fixtures = tuple(expected)
        for partition, index in fixtures:
            with self.subTest(partition=partition, index=index):
                outcome = v3.generate_candidate_v3(20261007, partition, index)
                self.assertEqual(outcome, v3.generate_candidate_v3(20261007, partition, index))
                self.assertEqual(outcome.generator_id, "dwr-generator-v3")
                n, profile, reason, witness, task_json = expected[(partition, index)]
                self.assertEqual((outcome.n, outcome.profile, outcome.rejection_reason), (n, profile, reason))
                self.assertEqual(outcome.witness.assignments, witness)
                self.assertEqual(serialize_task(outcome.task), task_json)
                actual_private = asdict(outcome.private_evidence) if outcome.private_evidence is not None else None
                self.assertEqual(actual_private, expected_private[(partition, index)])
                if outcome.oracle_unique:
                    self.assertEqual(len(enumerate_solutions(outcome.task).solutions), 1)
                    self.assertEqual(outcome.witness, outcome.solution)
                    self.assertNotIn(repr(outcome.witness.assignments), serialize_model_input(outcome.task))

    def test_source_neq_proposal_obeys_draw_trace_and_necessary_pool_floor(self):
        seen = 0
        for index in range(1, 250):
            outcome = v3.generate_candidate_v3(20261007, "source", index)
            if outcome.profile != ("neq",):
                continue
            seen += 1
            self.assertGreaterEqual(outcome.n, 4)
            if outcome.task is None:
                self.assertIn(outcome.rejection_reason,
                              ("no_neq_witness", "no_feasible_neq_constraint_total"))
                continue
            given_names = {given.variable for given in outcome.task.givens}
            unknown = set(outcome.task.variables) - given_names
            witness_values = dict(outcome.witness.assignments)
            self.assertEqual(set(witness_values.values()), {0, 1, 2})
            self.assertGreaterEqual(len({given.value for given in outcome.task.givens}), 2)
            touching = [c for c in outcome.task.constraints if unknown.intersection(c.variables)]
            self.assertGreaterEqual(len(touching), 2 * len(unknown))
            if outcome.oracle_unique:
                self.assertEqual(len(enumerate_solutions(outcome.task).solutions), 1)
        self.assertGreater(seen, 0)

    def test_v3_only_reasons_require_the_registered_payloads_and_identity(self):
        v2 = generate_candidate(20261007, "source", 1)
        none = dict(task=None, witness=None, solution=None, solver_result=None,
                    private_evidence=None, rejection_reason="no_neq_witness")
        accepted = replace(v2, **none, generator_id="dwr-generator-v3")
        self.assertEqual(accepted.rejection_reason, "no_neq_witness")
        with self.assertRaises(ValueError):
            replace(v2, **none)
        witness_only = dict(task=None, witness=v2.witness, solution=None, solver_result=None,
                            private_evidence=None, rejection_reason="no_feasible_neq_constraint_total",
                            generator_id="dwr-generator-v3")
        self.assertEqual(replace(v2, **witness_only).witness, v2.witness)
        with self.assertRaises(ValueError):
            replace(v2, **dict(witness_only, witness=None))
        with self.assertRaises(ValueError):
            replace(v2, **dict(none, witness=v2.witness, generator_id="dwr-generator-v3"))
        with self.assertRaises(ValueError):
            replace(v2, rejection_reason="profile_quota_full", generator_id="dwr-generator-v3")
        with self.assertRaises(ValueError):
            replace(v2, generator_id="dwr-generator-v99")

    def test_source_neq_draw_order_matches_an_independent_rng_trace(self):
        partition, index, root_seed = "source", 1, 20261007
        material = (b"dwr-generator-v3\0" + root_seed.to_bytes(8, "big") + b"\0source\0" +
                    index.to_bytes(8, "big"))
        rng = random.Random()
        rng.seed(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"), version=2)
        n = rng.randint(4, 6)
        given_count = rng.randint(2, n - 2)
        names = tuple(f"v{i}" for i in range(n))
        given_variables = tuple(sorted(rng.sample(names, given_count)))
        witnesses = tuple(values for values in itertools.product((0, 1, 2), repeat=n)
                          if len(set(values)) == 3 and
                          len({values[names.index(name)] for name in given_variables}) >= 2)
        witness_values = rng.choice(witnesses)
        unknown = set(names) - set(given_variables)
        pool = tuple((left, right) for left, right in itertools.combinations(names, 2)
                     if witness_values[names.index(left)] != witness_values[names.index(right)]
                     and unknown.intersection((left, right)))
        total = rng.randint(2 * (n - given_count), min(8, len(pool)))
        chosen = tuple(sorted(rng.sample(pool, total)))
        expected_task = ConstraintTask(
            "constraint-task-v1", 3, names,
            tuple(Constraint(RelationKind.NEQ, edge) for edge in chosen),
            tuple(GivenAssignment(name, witness_values[names.index(name)]) for name in given_variables))
        outcome = v3.generate_candidate_v3(root_seed, partition, index)
        self.assertEqual(outcome.n, n)
        self.assertEqual(outcome.witness.assignments, tuple(zip(names, witness_values)))
        self.assertEqual(serialize_task(outcome.task), serialize_task(expected_task))


class _CountingRng:
    def __init__(self):
        self.calls = []

    def randint(self, left, right):
        self.calls.append(("randint", left, right))
        return left


if __name__ == "__main__":
    unittest.main()
