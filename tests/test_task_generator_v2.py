import hashlib
import itertools
import json
import random
import unittest
from dataclasses import replace

from dwr.tasks.audit import audit_interactions
from dwr.tasks.generator_v2 import (
    CandidateHeader, CandidatePrivateEvidence, MixedEligibilityEvidence,
    SourceEligibilityEvidence, _count_vectors, _mixed_primary_eligibility, _source_sensitive,
    generate_candidate, sample_candidate_header,
)
from dwr.tasks.generator import generate_candidate as generate_candidate_v1
from dwr.tasks.oracle import enumerate_solutions
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, Solution, serialize_model_input, serialize_task
from dwr.tasks.solver import solve


def independent_rng(root, partition, index):
    material = (b"dwr-generator-v2\0" + root.to_bytes(8, "big") + b"\0" +
                partition.encode("ascii") + b"\0" + index.to_bytes(8, "big"))
    digest = hashlib.sha256(material).digest()
    rng = random.Random()
    rng.seed(int.from_bytes(digest[:8], "big"), version=2)
    return rng, digest


class CandidateGeneratorV2Tests(unittest.TestCase):
    def test_seed_domain_and_profile_schedule_match_independent_trace(self):
        rng, digest = independent_rng(20261007, "source", 3)
        self.assertEqual(digest.hex(), "f08e8312c399fe46436ecd2e69af6687552cadc385e9ec20a1447149e1ec094f")
        n = rng.randint(3, 6)
        expected = CandidateHeader(n, (("eq",), ("neq",), ("add_mod",))[3 % 3])
        self.assertEqual((expected.n, expected.profile), (3, ("eq",)))
        self.assertEqual(sample_candidate_header(20261007, "source", 3), expected)
        self.assertEqual(sample_candidate_header(20261007, "development", 5).profile,
                         (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod"))[5 % 3])

    def test_profile_quota_is_after_n_and_profile_but_before_witness_draws(self):
        header = sample_candidate_header(9, "development", 2)
        quota = generate_candidate(9, "development", 2, full_profiles=frozenset((header.profile,)))
        self.assertEqual((quota.n, quota.profile), (header.n, header.profile))
        self.assertEqual(quota.rejection_reason, "profile_quota_full")
        self.assertIsNone(quota.witness)
        self.assertIsNone(quota.task)

    def test_witness_first_draw_order_and_candidate_replay(self):
        outcome = generate_candidate(20261007, "source", 5)
        self.assertEqual(outcome, generate_candidate(20261007, "source", 5))
        self.assertEqual(outcome.n, 5)
        self.assertEqual(outcome.profile, ("add_mod",))
        rng, _ = independent_rng(20261007, "source", 5)
        rng.randint(3, 6)
        names = tuple(f"v{i}" for i in range(outcome.n))
        witness = tuple(rng.randrange(3) for _ in names)
        self.assertEqual(outcome.witness.assignments, tuple(zip(names, witness)))
        self.assertTrue(outcome.private_evidence.oracle_unique)
        witness_map = dict(zip(names, witness))
        add_pool = []
        for left, right in itertools.combinations(names, 2):
            for target in names:
                if target != left and target != right and (witness_map[left] + witness_map[right]) % 3 == witness_map[target]:
                    add_pool.append((left, right, target))
        feasible_totals = tuple(range(2, min(8, len(add_pool)) + 1))
        selected_total = rng.choice(feasible_totals)
        vectors = tuple((0, 0, count) for count in feasible_totals if count == selected_total)
        selected_vector = rng.choice(vectors)
        expected_constraints = tuple(sorted(rng.sample(add_pool, selected_vector[2])))
        given_count = rng.randint(1, outcome.n - 2)
        expected_given_variables = tuple(sorted(rng.sample(names, given_count)))
        self.assertEqual(tuple(item.variables for item in outcome.task.constraints), expected_constraints)
        self.assertEqual(tuple(item.variable for item in outcome.task.givens), expected_given_variables)

    def test_no_witness_feasible_count_vector_is_a_rejection(self):
        self.assertEqual(tuple(_count_vectors(("eq", "neq"), (0, 0, 0), 2)), ())
        self.assertEqual(tuple(_count_vectors(("eq", "neq"), (1, 1, 0), 3)), ())
        # This independently pinned stream draws a witness with no feasible EQ tuple.
        outcome = generate_candidate(20261007, "source", 0)
        self.assertEqual(outcome.rejection_reason, "no_witness_feasible_constraint_set")
        self.assertIsNotNone(outcome.witness)

    def test_n3_equality_triangle_is_unique_but_source_insensitive(self):
        task = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
                              (Constraint(RelationKind.EQ, ("v0", "v1")),
                               Constraint(RelationKind.EQ, ("v0", "v2")),
                               Constraint(RelationKind.EQ, ("v1", "v2"))),
                              (GivenAssignment("v0", 1),))
        evidence = _source_sensitive(task)
        self.assertFalse(evidence.sensitive)
        self.assertTrue(all(count < 2 for count in evidence.removal_solution_counts))
        self.assertEqual(len(enumerate_solutions(task).solutions), 1)

    def test_private_predicates_match_independent_audit_for_stream_fixtures(self):
        fixtures = {"source": (2, 5, 20, 23), "development": (1, 22), "final": (0, 1)}
        for partition, indices in fixtures.items():
            for index in indices:
                outcome = generate_candidate(42, partition, index)
                self.assertTrue(outcome.oracle_unique)
                if partition == "source":
                    private = _source_sensitive(outcome.task)
                    independent = audit_interactions(outcome.task, partition)
                    self.assertEqual(private.sensitive, independent["sensitive"])
                    self.assertEqual(list(private.removal_solution_counts), independent["removal_solution_counts"])
                else:
                    private = _mixed_primary_eligibility(outcome.task)
                    independent = audit_interactions(outcome.task, partition)
                    self.assertEqual(private.primary_eligible, independent["sensitive"])
                    self.assertEqual(private.kind_removal_solution_counts,
                                     tuple(item["solution_count"] for item in independent["removals"]))
                    self.assertEqual(private.factor_graph_connected, independent["factor_graph_connected"])
                    self.assertEqual(private.factor_graph_kinds, tuple(independent["factor_graph_kinds"]))

    def test_missing_present_kind_in_unknown_factor_graph_cannot_be_necessary(self):
        outcome = generate_candidate(100, "development", 106)
        self.assertTrue(outcome.oracle_unique)
        evidence = _mixed_primary_eligibility(outcome.task)
        self.assertFalse(evidence.all_present_kinds_in_graph)
        missing = set(outcome.profile) - set(evidence.factor_graph_kinds)
        counts = dict(zip(evidence.kinds, evidence.kind_removal_solution_counts))
        for kind in missing:
            self.assertLess(counts[kind], 2)

    def test_mixed_reason_precedence_and_connectivity_only_rejections(self):
        for partition, index, expected in (
                ("development", 73, "mixed_kind_not_necessary"),
                ("final", 6, "mixed_kind_not_necessary"),
                ("development", 25, "mixed_unknown_factor_disconnected"),
                ("final", 9, "mixed_unknown_factor_disconnected")):
            outcome = generate_candidate(31, partition, index)
            self.assertTrue(outcome.oracle_unique)
            evidence = outcome.private_evidence.mixed
            fails_necessity = any(count < 2 for count in evidence.kind_removal_solution_counts)
            fails_connectivity = (not evidence.factor_graph_connected or
                                  not evidence.all_present_kinds_in_graph)
            actual = ("mixed_kind_not_necessary" if fails_necessity else
                      "mixed_unknown_factor_disconnected" if fails_connectivity else None)
            self.assertEqual(actual, expected)
            self.assertEqual(outcome.eligibility_reason, expected)

    def test_solver_oracle_agreement_and_private_metadata_do_not_change_prompt_bytes(self):
        outcome = next(generate_candidate(123, "source", i) for i in range(1000)
                       if generate_candidate(123, "source", i).task is not None)
        self.assertEqual(solve(outcome.task).solution, enumerate_solutions(outcome.task).solutions[0])
        prompt = serialize_model_input(outcome.task)
        self.assertEqual(prompt, serialize_model_input(outcome.task))
        self.assertNotIn(outcome.witness.assignments.__repr__(), prompt)
        self.assertEqual(set(json.loads(prompt)), {"answer_schema", "instruction", "rules", "task"})

    def test_prompt_projection_is_differentially_independent_of_private_evidence(self):
        outcome = generate_candidate(20261007, "source", 5)
        source = outcome.private_evidence.source
        counts = list(source.removal_solution_counts)
        counts[0] += 1 if counts[0] < 729 else -1
        changed = replace(outcome, private_evidence=CandidatePrivateEvidence(
            True, SourceEligibilityEvidence(tuple(counts), source.sensitive), None))
        self.assertNotEqual(outcome.private_evidence, changed.private_evidence)
        self.assertEqual(serialize_model_input(outcome.task), serialize_model_input(changed.task))

    def test_v1_candidate_27_serialization_golden_remains_unchanged(self):
        outcome = generate_candidate_v1(20261007, "source", 27)
        self.assertEqual(outcome.rejection_reason, None)
        self.assertEqual(serialize_task(outcome.task),
                         '{"constraints":[{"kind":"eq","variables":["v0","v1"]},{"kind":"eq","variables":["v0","v2"]}],"givens":[{"value":0,"variable":"v2"},{"value":1,"variable":"v3"}],"modulus":3,"schema_version":"constraint-task-v1","variables":["v0","v1","v2","v3"]}')

    def test_private_evidence_rejects_mutable_counts_and_nonboolean_flags(self):
        outcome = generate_candidate(20261007, "source", 5)
        with self.assertRaises((TypeError, ValueError)):
            replace(outcome, private_evidence=CandidatePrivateEvidence(
                "yes", SourceEligibilityEvidence([1], False), None))
        with self.assertRaises((TypeError, ValueError)):
            SourceEligibilityEvidence((1,), "no")
        with self.assertRaises((TypeError, ValueError)):
            CandidatePrivateEvidence("yes", outcome.private_evidence.source, None)
        with self.assertRaises((TypeError, ValueError)):
            CandidatePrivateEvidence(False, outcome.private_evidence.source, None)

    def test_private_evidence_rejects_inconsistent_counts_kinds_and_reasons(self):
        with self.assertRaises((TypeError, ValueError)):
            SourceEligibilityEvidence((True, 1), True)
        with self.assertRaises((TypeError, ValueError)):
            SourceEligibilityEvidence((-1, 1), False)
        with self.assertRaises((TypeError, ValueError)):
            MixedEligibilityEvidence(("eq", "eq"), (2, 2), True, ("eq",), True, True, None)
        with self.assertRaises((TypeError, ValueError)):
            MixedEligibilityEvidence(("eq", "neq"), (1, 2), True, ("eq", "neq"), True,
                                     True, None)
        with self.assertRaises((TypeError, ValueError)):
            MixedEligibilityEvidence(("eq", "neq"), (2, 2), True, ("eq",), False,
                                     True, None)
        with self.assertRaises((TypeError, ValueError)):
            MixedEligibilityEvidence(("eq", "neq"), [2, 2], True, ("eq", "neq"), True,
                                     True, None)
        with self.assertRaises((TypeError, ValueError)):
            MixedEligibilityEvidence(("eq", "neq"), (2, 2), True, ["eq", "neq"], True,
                                     True, None)
        for field in ("factor_graph_connected", "all_present_kinds_in_graph", "primary_eligible"):
            values = {"kinds": ("eq", "neq"), "kind_removal_solution_counts": (2, 2),
                      "factor_graph_connected": True, "factor_graph_kinds": ("eq", "neq"),
                      "all_present_kinds_in_graph": True, "primary_eligible": True,
                      "ineligibility_reason": None}
            values[field] = "yes"
            with self.subTest(field=field), self.assertRaises((TypeError, ValueError)):
                MixedEligibilityEvidence(**values)
        with self.assertRaises((TypeError, ValueError)):
            MixedEligibilityEvidence(("eq", "neq"), (1, 2), True, ("eq", "neq"), True,
                                     False, "unknown_reason")

    def test_outcome_rejects_header_witness_solution_and_status_mismatches(self):
        unique = generate_candidate(20261007, "source", 5)
        with self.assertRaises((TypeError, ValueError)):
            replace(unique, n=3)
        with self.assertRaises((TypeError, ValueError)):
            replace(unique, witness=Solution((("v0", 0), ("v1", 0), ("v2", 0))))
        with self.assertRaises((TypeError, ValueError)):
            replace(unique, profile=("eq",))
        nonunique = generate_candidate(20261007, "source", 1)
        self.assertEqual(nonunique.rejection_reason, "oracle_non_unique")
        with self.assertRaises((TypeError, ValueError)):
            replace(nonunique, solution=unique.solution)
        with self.assertRaises((TypeError, ValueError)):
            replace(unique, rejection_reason="profile_quota_full")
        with self.assertRaises((TypeError, ValueError)):
            replace(unique, rejection_reason="no_witness_feasible_constraint_set")

    def test_unique_ineligible_candidates_remain_nonterminal_and_consistent(self):
        outcome = generate_candidate(42, "source", 5)
        self.assertTrue(outcome.oracle_unique)
        self.assertFalse(outcome.eligible)
        self.assertIs(type(outcome.eligible), bool)
        self.assertEqual(outcome.eligibility_reason, "source_insensitive")
        self.assertIsNone(outcome.rejection_reason)

    def test_module_random_state_and_invalid_bool_indices(self):
        state = random.getstate()
        generate_candidate(7, "final", 0)
        self.assertEqual(random.getstate(), state)
        for args in ((True, "source", 0), (1, "source", True), (1, "nope", 0), (1, "source", 2**64)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                generate_candidate(*args)


if __name__ == "__main__":
    unittest.main()
