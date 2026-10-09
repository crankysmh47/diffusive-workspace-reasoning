"""Stateless v4 proposal checks using fixed tiny candidate fixtures only."""
from __future__ import annotations

import hashlib
import itertools
import random
import unittest
from pathlib import Path

from dwr.tasks.config_probe_v4 import load_probe_config
from dwr.tasks.generator_probe_v4 import (
    ProposalFatalError,
    ProposalResult,
    candidate_stream,
    next_active_cell,
    propose_candidate,
)
from dwr.tasks.generator_v2 import CandidateOutcome, _count_vectors, _normalized_tuples
from dwr.tasks.schema import (
    Constraint, ConstraintTask, GivenAssignment, RelationKind, Solution,
    canonical_constraint_key,
)
from dwr.tasks.solver import SolverStats


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "tasks" / "constraint_probe_v4.json"
FIXTURE_ID = "fixed-v4-proposals:source-0..2,development-0..2;no-search"


def _remaining(config, *, closed=()):
    profiles = config.source_profiles
    return {profile: {cell: (0 if (profile, cell) in closed else config.source_quotas[profile][cell])
                      for cell in config.cell_order} for profile in profiles}


class ProbeGeneratorV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_probe_config(CONFIG_PATH)

    def test_candidate_stream_uses_independent_domain_seed_material(self) -> None:
        root_seed, partition, index = self.config.root_seed, "source", 17
        material = (b"dwr-generator-v4-probe\0" + root_seed.to_bytes(8, "big") + b"\0" +
                    partition.encode("ascii") + b"\0" + index.to_bytes(8, "big"))
        expected = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))
        expected.seed(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"), version=2)
        actual = candidate_stream(self.config, partition, index)
        self.assertEqual([actual.randrange(1000) for _ in range(8)],
                         [expected.randrange(1000) for _ in range(8)])
        for bad in (True, -1, 2**64, 1.5):
            with self.subTest(candidate_index=bad), self.assertRaises((TypeError, ValueError)):
                candidate_stream(self.config, partition, bad)

    def test_cell_schedule_preserves_profile_cell_order_and_rejects_bad_state(self) -> None:
        remaining = _remaining(self.config, closed=(("eq", "A"), ("neq", "A")))
        expected_active = [(profile, cell) for profile in self.config.source_profiles
                           for cell in self.config.cell_order if remaining[profile][cell] > 0]
        for index in range(6):
            self.assertEqual(next_active_cell(self.config, "source", index, remaining),
                             expected_active[index % len(expected_active)])
        for invalid in ({}, {"eq": {"A": True}}, remaining | {"other": {"A": 1}}):
            with self.subTest(remaining=invalid), self.assertRaises((TypeError, ValueError)):
                next_active_cell(self.config, "source", 0, invalid)
        with self.assertRaises(ValueError):
            next_active_cell(self.config, "source", 0, _remaining(self.config, closed=tuple(
                (profile, cell) for profile in self.config.source_profiles for cell in self.config.cell_order)))

        development = {profile: {cell: self.config.development_cell_quota
                                 for cell in self.config.cell_order}
                       for profile in self.config.development_profiles}
        self.assertEqual(next_active_cell(self.config, "development", 4, development),
                         (self.config.development_profiles[1], "A"))

    def test_v1_tuple_and_count_vector_orders_are_the_pinned_finite_orders(self) -> None:
        variables = tuple(f"v{i}" for i in range(6))
        self.assertEqual(_normalized_tuples(6, RelationKind.EQ), tuple(itertools.combinations(variables, 2)))
        expected_add = tuple((*pair, target) for pair in itertools.combinations(variables, 2)
                             for target in variables if target not in pair)
        self.assertEqual(_normalized_tuples(6, RelationKind.ADD_MOD), expected_add)
        self.assertEqual(tuple(_count_vectors(("eq", "neq"), (3, 4, 0), 5)),
                         ((1, 4, 0), (2, 3, 0), (3, 2, 0)))

    def test_fixed_proposals_return_typed_payloads_and_real_oracle_solver_agreement(self) -> None:
        self.assertEqual(FIXTURE_ID, "fixed-v4-proposals:source-0..2,development-0..2;no-search")
        cases = (
            ("source", 0, ("eq",), "A"),
            ("source", 1, ("neq",), "B"),
            ("source", 2, ("add_mod",), "C"),
            ("development", 0, ("eq", "neq"), "A"),
            ("development", 1, ("eq", "add_mod"), "B"),
            ("development", 2, ("neq", "add_mod"), "D"),
        )
        results = [propose_candidate(self.config, partition, index, (profile, cell))
                   for partition, index, profile, cell in cases]
        self.assertTrue(all(type(result) is ProposalResult for result in results))
        self.assertEqual(tuple(ProposalResult.__dataclass_fields__), (
            "partition", "candidate_index", "profile", "stratum", "n", "g", "c",
            "proposal_reason_or_null", "task_or_null", "witness_or_null",
            "oracle_count_capped_or_null", "oracle_solution_or_null", "solver_stats_or_null",
            "private_eligibility_or_null"))
        self.assertTrue(any(result.oracle_count_capped == 1 for result in results),
                        "fixed fixture set must exercise the real oracle and solver path")
        for result in results:
            if result.oracle_count_capped == 1:
                self.assertIsNotNone(result.solver_stats)
                self.assertEqual(result.oracle_solution, result.witness)
                self.assertEqual(len(result.task.constraints), result.c)
                self.assertEqual(tuple(sorted(result.task.constraints, key=canonical_constraint_key)),
                                 result.task.constraints)
                self.assertEqual(result.private_eligibility["kind"],
                                 "source" if result.partition == "source" else "mixed")
                if result.partition == "source":
                    self.assertEqual(set(result.private_eligibility),
                                     {"kind", "eligible", "removal_solution_counts"})
                    self.assertEqual(len(result.private_eligibility["removal_solution_counts"]), result.c)
                else:
                    self.assertEqual(set(result.private_eligibility),
                                     {"kind", "eligible", "kinds", "kind_removal_solution_counts",
                                      "factor_graph_connected", "factor_graph_kinds",
                                      "all_present_kinds_in_graph", "reason"})
            elif result.oracle_count_capped == 2:
                self.assertIsNone(result.solver_stats)
                self.assertIsNone(result.oracle_solution)
        self.assertTrue(all(not isinstance(result, CandidateOutcome) and not hasattr(result, "accepted")
                            for result in results),
                        "v4 proposal results must remain separate from historical outcome dispatch")

    def test_source_witness_populations_and_exact_draw_order_on_fixed_candidates(self) -> None:
        source_eq = propose_candidate(self.config, "source", 0, (("eq",), "A"))
        source_neq = propose_candidate(self.config, "source", 1, (("neq",), "B"))
        expected_eq = (("v0", "v2"), ("v0", "v5"), ("v1", "v2"), ("v1", "v3"),
                       ("v2", "v3"), ("v3", "v4"), ("v3", "v5"), ("v4", "v5"))
        expected_neq = (("v0", "v2"), ("v0", "v5"), ("v1", "v2"), ("v1", "v3"),
                        ("v1", "v5"), ("v2", "v5"))
        self.assertEqual(source_eq.witness.assignments,
                         tuple((f"v{i}", 2) for i in range(6)))
        self.assertEqual(tuple(item.variables for item in source_eq.task.constraints), expected_eq)
        self.assertEqual(source_neq.witness.assignments,
                         (("v0", 0), ("v1", 0), ("v2", 1), ("v3", 2), ("v4", 0), ("v5", 2)))
        self.assertEqual(tuple(item.variables for item in source_neq.task.constraints), expected_neq)

        # Independent finite-population checks: EQ uses only values present at givens
        # and has enough satisfied pairs; NEQ uses all values and two anchor values.
        eq_values = tuple(value for _, value in source_eq.witness.assignments)
        eq_given_values = {source_eq.task.givens[0].value, source_eq.task.givens[1].value}
        eq_pool_size = sum(eq_values[left] == eq_values[right]
                           for left, right in itertools.combinations(range(6), 2))
        self.assertLessEqual(set(eq_values), eq_given_values)
        self.assertGreaterEqual(eq_pool_size, source_eq.c)
        neq_values = tuple(value for _, value in source_neq.witness.assignments)
        neq_given_values = {given.value for given in source_neq.task.givens}
        self.assertEqual(set(neq_values), {0, 1, 2})
        self.assertGreaterEqual(len(neq_given_values), 2)
        unknowns = set(self.config.variables) - {given.variable for given in source_neq.task.givens}
        self.assertTrue(all(unknowns.intersection(constraint.variables)
                            for constraint in source_neq.task.constraints))

    def test_fixed_mixed_tuple_selection_pins_count_vector_and_sample_order(self) -> None:
        result = propose_candidate(self.config, "development", 0, (("eq", "neq"), "A"))
        expected = (
            ("eq", ("v0", "v1")), ("eq", ("v0", "v4")), ("eq", ("v4", "v5")),
            ("neq", ("v0", "v2")), ("neq", ("v0", "v3")), ("neq", ("v1", "v3")),
            ("neq", ("v2", "v4")), ("neq", ("v3", "v4")),
        )
        self.assertEqual(tuple((item.kind.value, item.variables) for item in result.task.constraints), expected)
        self.assertEqual(result.oracle_count_capped, 2)
        self.assertIsNone(result.solver_stats)

    def test_source_eq_and_other_profile_draw_order_is_deterministic(self) -> None:
        # Exact fixed candidate draws are pinned independently by the expected stream outputs.
        cases = (("source", 0, ("eq",), "A"),
                 ("source", 2, ("add_mod",), "C"),
                 ("development", 0, ("eq", "neq"), "A"))
        first = [propose_candidate(self.config, *case[:2], (case[2], case[3])) for case in cases]
        second = [propose_candidate(self.config, *case[:2], (case[2], case[3])) for case in cases]
        self.assertEqual(first, second)
        for result in first:
            self.assertEqual(result.n, 6)
            self.assertEqual((result.g, result.c), self.config.strata[result.stratum])
            if result.witness is not None:
                self.assertEqual(tuple(name for name, _ in result.witness.assignments),
                                 self.config.variables)

    def test_empty_and_wrong_active_cell_fails_before_proposal(self) -> None:
        with self.assertRaises(ValueError):
            propose_candidate(self.config, "final", 0, (("eq", "neq", "add_mod"), "A"))
        with self.assertRaises((TypeError, ValueError)):
            propose_candidate(self.config, "source", 0, (("eq",), "missing"))

    def test_empty_witness_and_constraint_populations_have_typed_proposal_reasons(self) -> None:
        from unittest.mock import patch
        from dwr.tasks import generator_probe_v4 as module

        class FixedRng:
            def sample(self, population, count):
                return list(population[:count])

            def randrange(self, stop):
                return 0

            def choice(self, population):
                raise AssertionError("an empty feasible population must not consume choice")

        with patch.object(module, "candidate_stream", return_value=FixedRng()), \
             patch.object(module, "_source_eq_witnesses", return_value=()):
            no_eq = propose_candidate(self.config, "source", 0, (("eq",), "A"))
        with patch.object(module, "candidate_stream", return_value=FixedRng()), \
             patch.object(module, "_source_neq_witnesses", return_value=()):
            no_neq = propose_candidate(self.config, "source", 1, (("neq",), "B"))
        witness = module._as_solution(self.config, (0, 1, 2, 0, 0, 0))
        class ChosenRng(FixedRng):
            def choice(self, population):
                return witness

        with patch.object(module, "candidate_stream", return_value=ChosenRng()), \
             patch.object(module, "_source_neq_witnesses", return_value=(witness,)), \
             patch.object(module, "_witness_satisfied_tuples", return_value=()):
            no_neq_pool = propose_candidate(self.config, "source", 1, (("neq",), "B"))
        with patch.object(module, "candidate_stream", return_value=FixedRng()):
            no_vector = propose_candidate(self.config, "development", 0, (("eq", "neq"), "A"))
        self.assertEqual(no_eq.proposal_reason, "no_source_eq_witness")
        self.assertIsNone(no_eq.witness)
        self.assertEqual(no_neq.proposal_reason, "no_neq_witness")
        self.assertIsNone(no_neq.witness)
        self.assertEqual(no_neq_pool.proposal_reason, "no_feasible_neq_constraint_total")
        self.assertEqual(no_neq_pool.witness, witness)
        self.assertEqual(no_vector.proposal_reason, "no_feasible_constraint_vector")
        self.assertIsNotNone(no_vector.witness)
        self.assertIsNone(no_vector.task)
        self.assertIsNone(no_vector.oracle_count_capped)

    def test_zero_oracle_and_solver_mismatch_are_typed_fatal_errors(self) -> None:
        # Fault injection is confined to the two registered fatal boundaries.
        from unittest.mock import patch
        from dwr.tasks import generator_probe_v4 as module

        candidate = (self.config, "source", 2, (("add_mod",), "C"))
        with patch.object(module, "enumerate_solutions", return_value=type("O", (), {"solutions": ()})()):
            with self.assertRaises(ProposalFatalError) as caught:
                propose_candidate(*candidate)
            self.assertEqual(caught.exception.reason, "oracle_witness_disagreement")
            self.assertIsNotNone(caught.exception.context.task)
            self.assertEqual(caught.exception.context.oracle_count_capped, 0)

        class OracleOne:
            solutions = (self.configured_solution(),)

        with patch.object(module, "enumerate_solutions", return_value=OracleOne()), \
             patch.object(module, "solve", return_value=type("S", (), {"solution": None, "stats": None})()):
            with self.assertRaises(ProposalFatalError) as caught:
                propose_candidate(*candidate)
            self.assertEqual(caught.exception.reason, "solver_oracle_disagreement")

    def test_result_constructor_rejects_task_shape_profile_and_invalid_witness_fixture(self) -> None:
        task = ConstraintTask(
            "constraint-task-v1", 3, ("v0", "v1", "v2"),
            (Constraint(RelationKind.NEQ, ("v0", "v1")),
             Constraint(RelationKind.NEQ, ("v1", "v2"))),
            (GivenAssignment("v0", 0),),
        )
        invalid_witness = Solution((("v0", 0), ("v1", 0), ("v2", 0)))
        with self.assertRaises((TypeError, ValueError)):
            ProposalResult("source", 41, ("eq",), "A", 6, 2, 8, None,
                           task, invalid_witness, 2, None, None, None)

    def test_result_constructor_rejects_empty_mixed_present_kind_projection(self) -> None:
        task = _pure_eq_task()
        solution = Solution(tuple((f"v{i}", 0) for i in range(6)))
        eligibility = {
            "kind": "mixed", "eligible": True, "kinds": [],
            "kind_removal_solution_counts": [], "factor_graph_connected": True,
            "factor_graph_kinds": [], "all_present_kinds_in_graph": True, "reason": None,
        }
        with self.assertRaises((TypeError, ValueError)):
            ProposalResult("development", 9, ("eq", "neq"), "A", 6, 2, 8, None,
                           task, solution, 1, solution, SolverStats(1, 1, 1), eligibility)

    def test_result_constructor_rejects_wrong_witness_length_and_given_value(self) -> None:
        task = _pure_eq_task()
        solution = Solution(tuple((f"v{i}", 0) for i in range(6)))
        eligibility = {"kind": "source", "eligible": True,
                       "removal_solution_counts": [1, 1, 1, 1, 3, 1, 1, 1]}
        base = ("source", 12, ("eq",), "A", 6, 2, 8, None, task)
        with self.subTest(case="short witness"), self.assertRaises((TypeError, ValueError)):
            ProposalResult(*base, Solution((("v0", 0), ("v1", 0), ("v2", 0))),
                           1, solution, SolverStats(1, 1, 1), eligibility)
        wrong_given = Solution((("v0", 1), ("v1", 0), ("v2", 0),
                                ("v3", 0), ("v4", 0), ("v5", 0)))
        with self.subTest(case="wrong given value"), self.assertRaises((TypeError, ValueError)):
            ProposalResult(*base, wrong_given, 1, solution,
                           SolverStats(1, 1, 1), eligibility)
        wrong_oracle_solution = Solution((("v0", 0), ("v1", 0), ("v2", 0),
                                          ("v3", 0), ("v4", 0), ("v5", 1)))
        with self.subTest(case="unique oracle differs from witness"), self.assertRaises((TypeError, ValueError)):
            ProposalResult(*base, solution, 1, wrong_oracle_solution,
                           SolverStats(1, 1, 1), eligibility)

    def test_result_constructor_rejects_mixed_projection_not_matching_task_kinds(self) -> None:
        task = _mixed_unique_task()
        witness = Solution((("v0", 0), ("v1", 1), ("v2", 2),
                            ("v3", 2), ("v4", 2), ("v5", 2)))
        eligibility = {
            "kind": "mixed", "eligible": False, "kinds": ["eq"],
            "kind_removal_solution_counts": [1], "factor_graph_connected": True,
            "factor_graph_kinds": ["eq"], "all_present_kinds_in_graph": True,
            "reason": "mixed_kind_not_necessary",
        }
        with self.assertRaises((TypeError, ValueError)):
            ProposalResult("development", 13, ("eq", "neq"), "D", 6, 3, 8,
                           None, task, witness, 1, witness, SolverStats(1, 1, 1), eligibility)

    @staticmethod
    def configured_solution():
        from dwr.tasks.schema import Solution
        return Solution(tuple((f"v{i}", 0) for i in range(6)))


def _pure_eq_task() -> ConstraintTask:
    pairs = (("v0", "v1"), ("v0", "v2"), ("v0", "v3"), ("v0", "v4"),
             ("v0", "v5"), ("v1", "v2"), ("v1", "v3"), ("v1", "v4"))
    return ConstraintTask("constraint-task-v1", 3, tuple(f"v{i}" for i in range(6)),
                          tuple(Constraint(RelationKind.EQ, pair) for pair in pairs),
                          (GivenAssignment("v0", 0), GivenAssignment("v1", 0)))


def _mixed_unique_task() -> ConstraintTask:
    constraints = (
        Constraint(RelationKind.NEQ, ("v0", "v3")),
        Constraint(RelationKind.NEQ, ("v1", "v3")),
        Constraint(RelationKind.NEQ, ("v0", "v4")),
        Constraint(RelationKind.NEQ, ("v1", "v4")),
        Constraint(RelationKind.NEQ, ("v0", "v5")),
        Constraint(RelationKind.NEQ, ("v1", "v5")),
        Constraint(RelationKind.EQ, ("v2", "v3")),
        Constraint(RelationKind.EQ, ("v3", "v4")),
    )
    return ConstraintTask("constraint-task-v1", 3, tuple(f"v{i}" for i in range(6)),
                          constraints, (GivenAssignment("v0", 0), GivenAssignment("v1", 1),
                                        GivenAssignment("v2", 2)))


if __name__ == "__main__":
    unittest.main()
