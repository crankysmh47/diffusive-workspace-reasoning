import unittest
import hashlib
import inspect
import json
import random
from dataclasses import replace
from unittest.mock import patch

from dwr.tasks import certificate_generator as legacy
from dwr.tasks import certificate_generator_v2 as v2
from dwr.tasks.certificate_rank_generator import (
    RankOutcome, RankWork, _construct_pair, _rng as rank_rng, construct,
    _select_final_certificates,
)
from dwr.tasks.certificate_rank_generator import (
    _Work, _connected, _fiber_exclusion_pools, _finish, _full_catalog, _lift,
    _construct_final_path,
)
from dwr.tasks.oracle import enumerate_assignments
from dwr.tasks.local_propagation import propagate
from dwr.tasks.verifier import verify_answer
from dwr.tasks.f3 import rank
from dwr.tasks.schema import Constraint, GivenAssignment, RelationKind, Solution


def _scalar_holds(constraint, values):
    selected = tuple(values[name] for name in constraint.variables)
    if constraint.kind.value == "eq":
        return selected[0] == selected[1]
    if constraint.kind.value == "neq":
        return selected[0] != selected[1]
    return (selected[0] + selected[1]) % 3 == selected[2]


def _assert_final_fixture_semantics(testcase, result, witness_values, alt_values, given_names):
    task = result["task"]
    testcase.assertIsNotNone(task)
    witness = Solution(tuple((f"v{i}", value) for i, value in enumerate(witness_values)))
    alternatives = {kind: Solution(tuple((f"v{i}", value) for i, value in enumerate(values)))
                    for kind, values in alt_values.items()}
    witness_map = dict(witness.assignments)
    testcase.assertTrue(all(_scalar_holds(c, witness_map) for c in task.constraints))
    testcase.assertEqual(enumerate_assignments(task.variables, task.givens,
                                               task.constraints).solutions, (witness,))
    testcase.assertTrue(verify_answer(
        task, json.dumps({"assignment": witness_map})).valid)
    testcase.assertEqual(len(result["reserved_constraints"]), 3)
    for reserve, kind in zip(result["reserved_constraints"], ("eq", "neq", "add_mod")):
        alt_map = dict(alternatives[kind].assignments)
        testcase.assertFalse(_scalar_holds(reserve, alt_map))
        testcase.assertIn(reserve, task.constraints)
        for constraint in task.constraints:
            if constraint.kind.value != kind:
                testcase.assertTrue(_scalar_holds(constraint, alt_map))
        without_kind = tuple(c for c in task.constraints if c.kind.value != kind)
        remaining = enumerate_assignments(task.variables, task.givens, without_kind).solutions
        testcase.assertIn(alternatives[kind], remaining)
        testcase.assertGreaterEqual(len(remaining), 2)

    unknown = {name for name in task.variables if name not in set(given_names)}
    neighbors = {name: set() for name in unknown}
    for constraint in task.constraints:
        scope = [name for name in constraint.variables if name in unknown]
        for name in scope:
            neighbors[name].update(other for other in scope if other != name)
    reached, pending = set(), [next(iter(unknown))]
    while pending:
        name = pending.pop()
        if name not in reached:
            reached.add(name)
            pending.extend(neighbors[name] - reached)
    testcase.assertEqual(reached, unknown)


def _assert_pair_outcome_semantics(testcase, result):
    task = result.task
    witness = dict(result.witness.assignments)
    testcase.assertTrue(all(_scalar_holds(c, witness) for c in task.constraints))
    solutions = enumerate_assignments(task.variables, task.givens, task.constraints).solutions
    testcase.assertEqual(solutions, (result.witness,))
    unknown = tuple(name for name in task.variables
                    if name not in {given.variable for given in task.givens})
    work = _Work(None)
    testcase.assertTrue(_connected(tuple((c.kind.value, c.variables) for c in task.constraints),
                                   unknown, work))
    propagated = propagate(task)
    testcase.assertTrue(propagated.stalled)
    testcase.assertTrue(all(len(propagated.domains[task.variables.index(name)]) == 3
                            for name in unknown))
    alternatives = dict(result.alternatives)
    for kind, alternative in result.alternatives:
        alt = dict(alternative.assignments)
        testcase.assertNotEqual(alt, witness)
        remaining = tuple(c for c in task.constraints if c.kind.value != kind)
        testcase.assertIn(alternative,
                          enumerate_assignments(task.variables, task.givens, remaining).solutions)
        reserve = next(c for c in result.reserved_constraints if c.kind.value == kind)
        testcase.assertFalse(_scalar_holds(reserve, alt))
        other_alt = dict(alternatives[result.profile[1] if kind == result.profile[0]
                                      else result.profile[0]].assignments)
        testcase.assertTrue(all(c.kind.value != kind or _scalar_holds(c, other_alt)
                                for c in task.constraints))


class RankConstructorTests(unittest.TestCase):
    def test_final_route_is_reserved_without_drawing_a_candidate(self):
        # Final population access is gated. This only checks the frozen topology.
        from dwr.tasks import certificate_rank_generator as module
        self.assertEqual(module._FINAL_PROFILE_CELL_SCHEDULE,
                         tuple(("eq", "neq", "add_mod", cell) for cell in "ABCD"))
        source = inspect.getsource(module.construct)
        self.assertLess(source.index('if partition == "final"'),
                        source.index('schedule = legacy._profile_cell_schedule(partition)'))
        self.assertIn("_construct_final(root_seed, candidate_index", source)

    def test_final_outer_selection_draws_e_a_once_then_shuffles_full_n_catalog(self):
        e, a, n1, n2 = ((1,), (2,), (3,), (4,))

        class ScriptedRng:
            def __init__(self):
                self.choice_pools = []
                self.shuffle_calls = 0
            def choice(self, pool):
                self.choice_pools.append(tuple(pool))
                return pool[0]
            def shuffle(self, pool):
                self.shuffle_calls += 1
                pool.reverse()

        rng = ScriptedRng()
        work = _Work(None)
        selected = _select_final_certificates((e, a, n1, n2), (e, a, n1), rng, work)
        self.assertEqual(selected, (e, a, (n2, n1)))
        self.assertEqual(rng.choice_pools, [(e, a, n1), (a, n1, n2)])
        self.assertEqual(rng.shuffle_calls, 1)
        self.assertEqual(work.rng_calls, 3)

    def test_final_catalog_records_first_failed_n_path_then_success(self):
        result = _construct_final_path(17, "B", ("v3", "v4", "v5"),
                                       (0, 0, 0, 0, 1, 2),
                                       (2, 0, 2, 0, 1, 2), (0, 0, 1, 0, 1, 2),
                                       ((2, 1, 0, 0, 1, 2), (1, 1, 0, 0, 1, 2)),
                                       random.Random(7))
        self.assertIsNotNone(result["task"])
        self.assertEqual(result["work"].final_path_visits, 2)
        self.assertEqual(result["work"].final_path_failures,
                         (("empty_own_reject_pool", 1),))
        self.assertLessEqual(result["work"].final_path_visits, 24)

    def test_exhausted_explicit_n_catalog_returns_typed_failure_without_retry(self):
        witness_values = (0, 0, 0, 0, 1, 2)
        givens = tuple(GivenAssignment(name, value) for name, value in
                       (("v3", 0), ("v4", 1), ("v5", 2)))
        result = _construct_final_path(17, "B", ("v3", "v4", "v5"), witness_values,
                                       (2, 0, 2, 0, 1, 2), (0, 0, 1, 0, 1, 2),
                                       ((2, 1, 0, 0, 1, 2),), random.Random(7))
        self.assertEqual(result["reason"], "no_feasible_final_certificate")
        self.assertIsNone(result["task"])
        self.assertEqual(result["selected"], ())
        self.assertEqual(result["reserved_constraints"], ())
        self.assertEqual(result["work"].final_path_visits, 1)
        self.assertEqual(result["work"].final_path_failures,
                         (("empty_own_reject_pool", 1),))
        self.assertEqual(result["work"].rank_calls, 0)
        self.assertEqual(result["work"].rng_calls, 0)
        failure = RankOutcome(1, "final", 17, ("eq", "neq", "add_mod"), "B",
                              givens, Solution(tuple((f"v{i}", value)
                                                    for i, value in enumerate(witness_values))),
                              (), (), result["work"], None, result["reason"],
                              branch="ordinary")
        self.assertEqual(failure.rejection_reason, "no_feasible_final_certificate")

    def test_hand_final_g3_path_preserves_three_certificates_and_padding(self):
        witness = (0, 0, 0, 0, 1, 2)
        givens = ("v3", "v4", "v5")
        eq_alt = (2, 0, 2, 0, 1, 2)
        add_alt = (0, 0, 1, 0, 1, 2)
        neq_alt = (1, 1, 0, 0, 1, 2)
        for index, cell, count in ((17, "B", 6), (18, "C", 7), (19, "D", 8)):
            with self.subTest(cell=cell):
                result = _construct_final_path(index, cell, givens, witness, eq_alt, add_alt,
                                               (neq_alt,), random.Random(7))
                task = result["task"]
                self.assertIsNotNone(task)
                self.assertEqual(len(task.constraints), count)
                self.assertEqual(set(c.kind.value for c in task.constraints), {"eq", "neq", "add_mod"})
                self.assertTrue(any(sum(name not in {"v3", "v4", "v5"}
                                        for name in c.variables) == 1
                                    for c in task.constraints))
                self.assertEqual(result["work"].final_path_visits, 1)
                _assert_final_fixture_semantics(
                    self, result, witness,
                    {"eq": eq_alt, "neq": neq_alt, "add_mod": add_alt}, givens)
                alternatives = tuple((kind, Solution(tuple((f"v{i}", value)
                                                           for i, value in enumerate(alt))))
                                     for kind, alt in (("eq", eq_alt), ("neq", neq_alt),
                                                       ("add_mod", add_alt)))
                outcome = RankOutcome(1, "final", index, ("eq", "neq", "add_mod"), cell,
                                      task.givens, Solution(tuple((f"v{i}", value)
                                                                  for i, value in enumerate(witness))),
                                      alternatives, result["reserved_constraints"], result["work"],
                                      task, None, branch="ordinary")
                self.assertEqual(outcome.task, task)
                with self.assertRaises(ValueError):
                    replace(outcome, branch="no_unary")
                with self.assertRaises(ValueError):
                    replace(outcome, work=RankWork(final_path_visits=2))

    def test_hand_final_g2_path_supports_c8_and_unknown_given_neq_slopes(self):
        witness = (2, 2, 0, 1, 0, 1)
        eq_alt = (1, 2, 2, 0, 0, 1)
        add_alt = (2, 2, 1, 1, 0, 1)
        neq_alt = (0, 0, 0, 1, 0, 1)
        result = _construct_final_path(16, "A", ("v4", "v5"), witness,
                                       eq_alt, add_alt, (neq_alt,), random.Random(9))
        self.assertEqual(len(result["task"].constraints), 8)
        self.assertEqual(result["work"].final_path_visits, 1)
        _assert_final_fixture_semantics(
            self, result, witness,
            {"eq": eq_alt, "neq": neq_alt, "add_mod": add_alt}, ("v4", "v5"))

    def test_final_path_failure_is_finite_and_callback_interrupts(self):
        calls = 0
        def stop():
            nonlocal calls
            calls += 1
            if calls == 25:
                raise RuntimeError("budget")
        with self.assertRaisesRegex(RuntimeError, "budget"):
            _construct_final_path(17, "B", ("v3", "v4", "v5"), (0, 0, 0, 0, 1, 2),
                                  (2, 0, 2, 0, 1, 2), (0, 0, 1, 0, 1, 2),
                                  ((1, 1, 0, 0, 1, 2),), random.Random(7),
                                  check_budget=stop)

    def test_final_failure_schema_keeps_drawn_private_state_and_reason_counts(self):
        witness = Solution(tuple((f"v{i}", value) for i, value in
                                 enumerate((0, 0, 0, 0, 1, 2))))
        givens = tuple(GivenAssignment(name, value) for name, value in
                       (("v3", 0), ("v4", 1), ("v5", 2)))
        work = RankWork(final_path_visits=1,
                        final_path_failures=(("empty_own_reject_pool", 1),))
        failure = RankOutcome(1, "final", 17, ("eq", "neq", "add_mod"), "B",
                              givens, witness, (), (), work, None,
                              "no_feasible_final_certificate", branch="ordinary")
        self.assertEqual(failure.witness, witness)
        self.assertEqual(failure.givens, givens)
        self.assertEqual(failure.work.final_path_failures,
                         (("empty_own_reject_pool", 1),))
        with self.assertRaises(ValueError):
            replace(failure, rejection_reason="empty_own_reject_pool")
        with self.assertRaises(ValueError):
            replace(failure, work=RankWork(final_path_visits=25))
        with self.assertRaises(ValueError):
            replace(failure, rejection_reason="rank_completion_failed")
        with self.assertRaises(ValueError):
            replace(failure, work=RankWork(final_path_visits=1))

    def test_final_aggregate_failure_tag_is_not_valid_for_development_pair(self):
        good_pair = _construct_pair(1, "development", 16, ("eq", "add_mod"), "A",
                                    ("v4", "v5"), (0, 0, 0, 0, 0, 1), random.Random(11))
        with self.assertRaises(ValueError):
            replace(good_pair, task=None, alternatives=(), reserved_constraints=(),
                    rejection_reason="no_feasible_final_certificate")

    def test_invalid_config_identity_rejected(self):
        for args in ((-1, "source", 0), (1, "other", 0), (1, "development", True)):
            with self.subTest(args=args):
                with self.assertRaises((TypeError, ValueError)):
                    construct(*args)

    def test_algebraic_helper_preserves_pair_certificates(self):
        examples = (
            (("eq", "add_mod"), (0, 0, 0, 0, 0, 1)),
            (("neq", "add_mod"), (0, 1, 2, 0, 0, 1)),
        )
        for profile, values in examples:
            with self.subTest(profile=profile):
                index = 16 if profile[0] == "eq" else 20
                result = _construct_pair(1, "development", index, profile, "A",
                                         ("v4", "v5"), values, random.Random(11))
                self.assertIsNone(result.rejection_reason)
                self.assertEqual(result.branch, "no_unary")
                task = result.task
                self.assertEqual(len(task.constraints), 8)
                self.assertEqual(tuple(sorted(task.constraints, key=lambda c: (c.kind.value, c.variables))),
                                 task.constraints)
                witness = dict(result.witness.assignments)
                self.assertTrue(all(_scalar_holds(c, witness) for c in task.constraints))
                self.assertEqual(len(enumerate_assignments(task.variables, task.givens,
                                                           task.constraints).solutions), 1)
                self.assertTrue(all(len([n for n in c.variables if n not in {g.variable for g in task.givens}]) != 1
                                    for c in task.constraints))
                for kind, alternative in result.alternatives:
                    remaining = tuple(c for c in task.constraints if c.kind.value != kind)
                    self.assertIn(alternative, enumerate_assignments(task.variables, task.givens, remaining).solutions)
                alternatives = dict(result.alternatives)
                for kind in profile:
                    other = dict(alternatives[profile[1] if kind == profile[0] else profile[0]].assignments)
                    self.assertTrue(all(c.kind.value != kind or _scalar_holds(c, other)
                                        for c in task.constraints))
                propagated = propagate(task)
                self.assertTrue(propagated.stalled)
                self.assertTrue(all(len(d) == 3 for d in propagated.domains
                                    if d and len(d) > 1))
                self.assertEqual(tuple(x.kind.value for x in result.reserved_constraints), profile)
                if profile[0] == "eq":
                    unary = Constraint(RelationKind.EQ, ("v0", "v4"))
                    altered = list(task.constraints)
                    eq_index = next(i for i, item in enumerate(altered) if item.kind is RelationKind.EQ)
                    altered[eq_index] = unary
                    with self.assertRaises(ValueError):
                        replace(result, task=type(task)(task.schema_version, task.modulus,
                                                        task.variables,
                                                        tuple(sorted(altered, key=lambda c: (c.kind.value, c.variables))),
                                                        task.givens))

    def test_helper_rejects_invalid_shapes_and_given_counts(self):
        with self.assertRaises(ValueError):
            _construct_pair(1, "development", 16, ("eq", "add_mod"), "A",
                            ("v4",), (0, 0, 0, 0, 0, 1), random.Random(1))
        with self.assertRaises(ValueError):
            _construct_pair(1, "development", 16, ("eq", "add_mod"), "A",
                            ("v4", "v4"), (0, 0, 0, 0, 0, 1), random.Random(1))
        with self.assertRaises(ValueError):
            _construct_pair(1, "development", 16, ("eq", "add_mod"), "A",
                            ("v4", "v5"), (0, 0, 0, 0, 0, 1), random.Random(1),
                            constraint_count=7)

    def test_source_and_ordinary_development_delegate_unchanged(self):
        expected_source = legacy.construct(20261011, "source", 0)
        with patch.object(legacy, "construct", wraps=legacy.construct) as delegated:
            self.assertEqual(construct(20261011, "source", 0), expected_source)
            delegated.assert_called_once_with(20261011, "source", 0, check_budget=None)
        expected = v2.construct(20261011, "development", 0)
        with patch.object(v2, "construct", wraps=v2.construct) as delegated:
            self.assertEqual(construct(20261011, "development", 0), expected)
            delegated.assert_called_once_with(20261011, "development", 0, check_budget=None)
        expected = v2.construct(20261011, "development", 4)
        with patch.object(v2, "construct", wraps=v2.construct) as delegated:
            self.assertEqual(construct(20261011, "development", 4), expected)
            delegated.assert_called_once_with(20261011, "development", 4, check_budget=None)

    def test_production_rank_branch_is_versioned_and_counts_all_draws(self):
        result = construct(20261014, "development", 16)
        self.assertEqual((result.generator_id, result.branch),
                         ("dwr-certificate-rank-sixvar-v1", "no_unary"))
        self.assertIsNone(result.rejection_reason)
        self.assertEqual(result.work.rng_calls, 13)
        self.assertGreater(result.work.catalog_visits, 0)
        self.assertGreater(result.work.rank_calls, 0)
        self.assertGreater(result.work.row_operations, 0)

    def test_rank_path_callback_can_interrupt_and_is_observable(self):
        calls = 0
        def budget():
            nonlocal calls
            calls += 1
            if calls == 8:
                raise RuntimeError("budget")
        with self.assertRaisesRegex(RuntimeError, "budget"):
            _construct_pair(1, "development", 16, ("eq", "add_mod"), "A",
                            ("v4", "v5"), (0, 0, 0, 0, 0, 1), random.Random(11),
                            check_budget=budget)
        self.assertEqual(calls, 8)

    def test_new_rng_changes_only_v1_tag_and_retains_payload_encoding(self):
        root, partition, index = 123, "development", 16
        material = (b"dwr-certificate-rank-sixvar-v1\0" + root.to_bytes(8, "big") + b"\0" +
                    partition.encode("ascii") + b"\0" + index.to_bytes(8, "big"))
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        expected = random.Random()
        expected.seed(seed, version=2)
        actual = rank_rng(root, partition, index)
        self.assertEqual([actual.randrange(3) for _ in range(12)],
                         [expected.randrange(3) for _ in range(12)])

    def test_outcome_schema_rejects_unknown_failure(self):
        result = _construct_pair(1, "development", 16, ("eq", "add_mod"), "A",
                                 ("v4", "v5"), (0, 0, 0, 0, 0, 1), random.Random(11))
        with self.assertRaises(ValueError):
            replace(result, task=None, rejection_reason="legacy_reason")

    def test_failure_outcome_validates_witness_givens_reason_and_branch(self):
        failure = _construct_pair(1, "development", 17, ("eq", "add_mod"), "B",
                                  ("v3", "v4", "v5"), (0, 1, 2, 0, 1, 2), random.Random(11))
        self.assertEqual(failure.rejection_reason, "empty_eq_pool")
        with self.assertRaises(ValueError):
            replace(failure, witness=None)
        with self.assertRaises(ValueError):
            replace(failure, givens=())
        with self.assertRaises(ValueError):
            replace(failure, rejection_reason="unregistered")
        with self.assertRaises(ValueError):
            replace(failure, branch="ordinary")

    def test_catalog_preserves_distinct_given_addend_and_target_roles(self):
        from dwr.tasks.schema import Solution
        witness = Solution(tuple((f"v{i}", value) for i, value in enumerate((0, 1, 2, 1, 0, 2))))
        catalogs = _full_catalog({"v3": 1, "v4": 0, "v5": 2}, witness,
                                ("v0", "v1", "v2"), ("add_mod",), no_unary=True,
                                work=_Work(None))["add_mod"]
        rows = dict((operands, row) for operands, row in catalogs)
        self.assertEqual(rows[("v0", "v3", "v1")], (1, 2, 0))  # given addend
        self.assertEqual(rows[("v0", "v1", "v3")], (1, 1, 0))  # given target
        self.assertEqual(rows[("v1", "v2", "v0")], (2, 1, 1))  # unknown-only

    def test_constant_kernel_cannot_supply_two_unknown_neq_exclusions(self):
        from dwr.tasks.schema import Solution
        values = dict(Solution(tuple((f"v{i}", x) for i, x in enumerate((0, 1, 2, 0, 1, 2)))).assignments)
        work = _Work(None)
        neq_pool = tuple(((f"v{i}", f"v{j}"), (0, 0, 0))
                         for i, j in ((0, 1), (0, 2), (1, 2)))
        pools = _fiber_exclusion_pools(neq_pool, ("v0", "v1", "v2"), values,
                                       (1, 1, 1), work)
        self.assertEqual(pools, {1: [], 2: []})
        self.assertGreater(work.catalog_visits, 0)

    def test_nonconstant_kernel_has_two_hand_checked_neq_fiber_exclusions(self):
        values = dict(Solution(tuple((f"v{i}", x) for i, x in enumerate((0, 1, 2, 0, 1, 2)))).assignments)
        work = _Work(None)
        neq_pool = tuple(((f"v{i}", f"v{j}"), (0, 0, 0))
                         for i, j in ((0, 1), (0, 2), (1, 2)))
        pools = _fiber_exclusion_pools(neq_pool, ("v0", "v1", "v2"), values,
                                       (1, 0, 0), work)
        self.assertEqual(tuple(map(len, (pools[1], pools[2]))), (1, 1))

    def test_hand_g3_pair_certificates_cover_all_add_counts(self):
        witness = Solution(tuple((f"v{i}", x) for i, x in enumerate((0, 0, 0, 0, 1, 2))))
        givens = tuple(GivenAssignment(name, value) for name, value in (("v3", 0), ("v4", 1), ("v5", 2)))
        unknown = ("v0", "v1", "v2")
        given_map = {g.variable: g.value for g in givens}
        selected = [("eq", ("v0", "v1")), ("add_mod", ("v0", "v1", "v2")),
                    ("add_mod", ("v0", "v2", "v1"))]
        basis = tuple(selected)
        eq_alternative = _lift((0, 1, 1), unknown, given_map)
        add_alternative = _lift((1, 1, 1), unknown, given_map)

        class FirstSample(random.Random):
            def sample(self, population, k):
                return list(population[:k])

        for cell, index in (("B", 17), ("C", 18), ("D", 19)):
            with self.subTest(profile="eq_add", cell=cell):
                work = _Work(None)
                catalogs = _full_catalog(given_map, witness, unknown, ("eq", "add_mod"),
                                         no_unary=True, work=work)
                result = _finish(1, "development", index, ("eq", "add_mod"), cell,
                                 givens, witness, unknown, selected, catalogs,
                                 {"eq": eq_alternative, "add_mod": add_alternative},
                                 {"B": 6, "C": 7, "D": 8}[cell], FirstSample(1),
                                 work, None, "no_unary")
                self.assertIsNone(result.rejection_reason)
                task = result.task
                _assert_pair_outcome_semantics(self, result)
                # This witness-true EQ filler rejects the EQ-removal alternative;
                # that is allowed because all EQ constraints are removed together.
                self.assertTrue(any(c.kind.value == "eq" and c not in result.reserved_constraints and
                                    not _scalar_holds(c, dict(eq_alternative.assignments))
                                    for c in task.constraints))
                self.assertTrue(all(c.kind.value != "add_mod" or
                                    _scalar_holds(c, dict(eq_alternative.assignments))
                                    for c in task.constraints))

        neq_selected = [("add_mod", ("v1", "v4", "v2")),
                        ("add_mod", ("v1", "v2", "v3")),
                        ("neq", ("v0", "v1")), ("neq", ("v0", "v2"))]
        neq_witness = Solution(tuple((f"v{i}", x) for i, x in enumerate((0, 1, 2, 0, 1, 2))))
        neq_given_map = {"v3": 0, "v4": 1, "v5": 2}
        neq_givens = tuple(GivenAssignment(n, v) for n, v in neq_given_map.items())
        neq_unknown = ("v0", "v1", "v2")
        neq_alternative = _lift((1, 1, 2), neq_unknown, neq_given_map)
        neq_add_alternative = _lift((1, 2, 0), neq_unknown, neq_given_map)
        for cell, index in (("B", 21), ("C", 22), ("D", 23)):
            with self.subTest(profile="neq_add", cell=cell):
                work = _Work(None)
                catalogs = _full_catalog(neq_given_map, neq_witness, neq_unknown,
                                         ("neq", "add_mod"), no_unary=True, work=work)
                result = _finish(1, "development", index, ("neq", "add_mod"), cell,
                                 neq_givens, neq_witness, neq_unknown, neq_selected, catalogs,
                                 {"neq": neq_alternative, "add_mod": neq_add_alternative},
                                 {"B": 6, "C": 7, "D": 8}[cell], FirstSample(1),
                                 work, None, "no_unary", neq_vector=(1, 0, 0))
                self.assertIsNone(result.rejection_reason)
                self.assertEqual(len(result.task.constraints), {"B": 6, "C": 7, "D": 8}[cell])
                _assert_pair_outcome_semantics(self, result)

    def test_hand_g3_eq_add_padding_may_reject_own_alternative(self):
        witness = Solution(tuple((f"v{i}", x) for i, x in enumerate((0, 0, 0, 0, 1, 2))))
        givens = tuple(GivenAssignment(name, value) for name, value in (("v3", 0), ("v4", 1), ("v5", 2)))
        unknown = ("v0", "v1", "v2")
        given_map = {g.variable: g.value for g in givens}
        work = _Work(None)
        catalogs = _full_catalog(given_map, witness, unknown, ("eq", "add_mod"),
                                 no_unary=True, work=work)
        selected = [("eq", ("v0", "v1")), ("add_mod", ("v0", "v1", "v2")),
                    ("add_mod", ("v0", "v2", "v1"))]
        alternatives = {"eq": _lift((0, 1, 1), unknown, given_map),
                        "add_mod": _lift((1, 1, 1), unknown, given_map)}
        class FirstSample(random.Random):
            def sample(self, population, k):
                return list(population[:k])
        result = _finish(1, "development", 17, ("eq", "add_mod"), "B", givens,
                         witness, unknown, selected, catalogs, alternatives, 6,
                         FirstSample(1), work, None, "no_unary")
        self.assertIsNone(result.rejection_reason)
        self.assertTrue(any(c.kind is RelationKind.EQ and c not in result.reserved_constraints and
                            not _scalar_holds(c, dict(alternatives["eq"].assignments))
                            for c in result.task.constraints))
        self.assertTrue(all(c.kind is not RelationKind.ADD_MOD or
                            _scalar_holds(c, dict(alternatives["eq"].assignments))
                            for c in result.task.constraints))

    def test_success_outcome_rejects_unary_add_filler_only_by_no_unary_guard(self):
        witness = Solution(tuple((f"v{i}", x) for i, x in enumerate((0, 0, 0, 0, 1, 2))))
        givens = tuple(GivenAssignment(name, value) for name, value in (("v3", 0), ("v4", 1), ("v5", 2)))
        unknown = ("v0", "v1", "v2")
        given_map = {g.variable: g.value for g in givens}
        work = _Work(None)
        catalogs = _full_catalog(given_map, witness, unknown, ("eq", "add_mod"),
                                 no_unary=True, work=work)
        selected = [("eq", ("v0", "v1")), ("add_mod", ("v0", "v1", "v2")),
                    ("add_mod", ("v0", "v2", "v1"))]
        basis = tuple(selected)
        alternatives = {"eq": _lift((0, 1, 1), unknown, given_map),
                        "add_mod": _lift((1, 1, 1), unknown, given_map)}
        class FirstSample(random.Random):
            def sample(self, population, k):
                return list(population[:k])
        result = _finish(1, "development", 17, ("eq", "add_mod"), "B", givens,
                         witness, unknown, selected, catalogs, alternatives, 6,
                         FirstSample(1), work, None, "no_unary")
        self.assertIsNone(result.rejection_reason)
        self.assertTrue(_scalar_holds(Constraint(RelationKind.ADD_MOD, ("v4", "v5", "v0")),
                                      dict(alternatives["eq"].assignments)))
        self.assertFalse(_scalar_holds(Constraint(RelationKind.ADD_MOD, ("v4", "v5", "v0")),
                                       dict(alternatives["add_mod"].assignments)))
        reserved_add = next(c for c in result.reserved_constraints if c.kind is RelationKind.ADD_MOD)
        filler_index = next(i for i, c in enumerate(result.task.constraints)
                            if c.kind is RelationKind.ADD_MOD and c != reserved_add and c not in
                            tuple(Constraint(RelationKind.ADD_MOD, operands)
                                  for kind, operands in basis if kind == "add_mod"))
        altered = list(result.task.constraints)
        altered[filler_index] = Constraint(RelationKind.ADD_MOD, ("v4", "v5", "v0"))
        invalid_task = type(result.task)(result.task.schema_version, result.task.modulus,
                                         result.task.variables,
                                         tuple(sorted(altered, key=lambda c: (c.kind.value, c.variables))),
                                         result.task.givens)
        with self.assertRaisesRegex(ValueError, "no-unary"):
            replace(result, task=invalid_task)

    def test_missing_own_rejecting_basis_row_fails_without_partial_output(self):
        witness = Solution(tuple((f"v{i}", x) for i, x in enumerate((0, 0, 0, 0, 1, 2))))
        givens = tuple(GivenAssignment(name, value) for name, value in (("v3", 0), ("v4", 1), ("v5", 2)))
        unknown = ("v0", "v1", "v2")
        given_map = {g.variable: g.value for g in givens}
        work = _Work(None)
        catalogs = _full_catalog(given_map, witness, unknown, ("eq", "add_mod"),
                                 no_unary=True, work=work)
        selected = [("eq", ("v0", "v1")), ("add_mod", ("v0", "v1", "v2")),
                    ("add_mod", ("v0", "v2", "v1"))]
        alternatives = {"eq": _lift((0, 0, 0), unknown, given_map),
                        "add_mod": _lift((1, 1, 1), unknown, given_map)}
        result = _finish(1, "development", 17, ("eq", "add_mod"), "B", givens,
                         witness, unknown, selected, catalogs, alternatives, 6,
                         random.Random(1), work, None, "no_unary")
        self.assertEqual(result.rejection_reason, "rank_completion_failed")
        self.assertIsNone(result.task)
        self.assertEqual(result.alternatives, ())
        self.assertEqual(result.reserved_constraints, ())

    def test_rank_choice_failure_returns_empty_outcome_without_retry(self):
        witness = Solution(tuple((f"v{i}", x) for i, x in enumerate((0, 0, 0, 0, 1, 2))))
        catalog = {"eq": (( ("v0", "v1"), (1, 2, 0)),),
                   "add_mod": ((("v0", "v1", "v2"), (0, 0, 0)),)}
        with patch("dwr.tasks.certificate_rank_generator._full_catalog", return_value=catalog):
            result = _construct_pair(1, "development", 17, ("eq", "add_mod"), "B",
                                     ("v3", "v4", "v5"), (0, 0, 0, 0, 1, 2),
                                     random.Random(11))
        self.assertEqual(result.rejection_reason, "rank_completion_failed")
        self.assertIsNone(result.task)
        self.assertEqual(result.alternatives, ())
        self.assertEqual(result.reserved_constraints, ())

    def test_missing_fiber_exclusion_returns_empty_outcome_without_retry(self):
        witness_values = (0, 1, 2, 0, 1, 2)
        empty_pools = {1: [], 2: []}
        with patch("dwr.tasks.certificate_rank_generator._fiber_exclusion_pools",
                   return_value=empty_pools):
            result = _construct_pair(1, "development", 21, ("neq", "add_mod"), "B",
                                     ("v3", "v4", "v5"), witness_values,
                                     random.Random(11))
        self.assertEqual(result.rejection_reason, "missing_fiber_exclusion")
        self.assertIsNone(result.task)
        self.assertEqual(result.alternatives, ())
        self.assertEqual(result.reserved_constraints, ())

    def test_insufficient_filler_returns_empty_outcome_without_retry(self):
        witness = Solution(tuple((f"v{i}", x) for i, x in enumerate((0, 0, 0, 0, 1, 2))))
        givens = tuple(GivenAssignment(name, value) for name, value in
                       (("v3", 0), ("v4", 1), ("v5", 2)))
        unknown = ("v0", "v1", "v2")
        given_map = {g.variable: g.value for g in givens}
        selected = [("eq", ("v0", "v1")), ("add_mod", ("v0", "v1", "v2")),
                    ("add_mod", ("v0", "v2", "v1"))]
        alternatives = {"eq": _lift((0, 1, 1), unknown, given_map),
                        "add_mod": _lift((1, 1, 1), unknown, given_map)}
        work = _Work(None)
        catalogs = _full_catalog(given_map, witness, unknown, ("eq", "add_mod"),
                                 no_unary=True, work=work)
        result = _finish(1, "development", 17, ("eq", "add_mod"), "B", givens,
                         witness, unknown, selected, catalogs, alternatives, 99,
                         random.Random(1), work, None, "no_unary")
        self.assertEqual(result.rejection_reason, "insufficient_filler")
        self.assertIsNone(result.task)
        self.assertEqual(result.alternatives, ())
        self.assertEqual(result.reserved_constraints, ())

    def test_full_rank_rows_do_not_imply_connected_unknown_graph(self):
        # Two disconnected components can jointly have full rank.
        rows = ((1, 2, 0, 0), (1, 1, 0, 0),
                (0, 0, 1, 2), (0, 0, 1, 1))
        self.assertEqual(rank(rows, 4), 4)
        work = _Work(None)
        self.assertFalse(_connected((("eq", ("v0", "v1")), ("eq", ("v2", "v3"))),
                                    ("v0", "v1", "v2", "v3"), work))
        self.assertGreater(work.graph_operations, 0)

    def test_disconnected_full_rank_basis_terminates_with_registered_failure(self):
        witness = Solution(tuple((f"v{i}", x) for i, x in enumerate((0, 0, 0, 0, 0, 1))))
        givens = (GivenAssignment("v4", 0), GivenAssignment("v5", 1))
        unknown = ("v0", "v1", "v2", "v3")
        given_map = {g.variable: g.value for g in givens}
        selected = [("eq", ("v0", "v1")), ("add_mod", ("v0", "v1", "v4")),
                    ("eq", ("v2", "v3")), ("add_mod", ("v2", "v3", "v4"))]
        alternatives = {
            "eq": _lift((1, 2, 1, 2), unknown, given_map),
            "add_mod": _lift((1, 1, 1, 1), unknown, given_map),
        }
        work = _Work(None)
        catalogs = _full_catalog(given_map, witness, unknown, ("eq", "add_mod"),
                                 no_unary=True, work=work)
        result = _finish(1, "development", 16, ("eq", "add_mod"), "A", givens,
                         witness, unknown, selected, catalogs, alternatives, 4,
                         random.Random(1), work, None, "no_unary")
        self.assertEqual(result.rejection_reason, "disconnected_graph")


if __name__ == "__main__":
    unittest.main()
