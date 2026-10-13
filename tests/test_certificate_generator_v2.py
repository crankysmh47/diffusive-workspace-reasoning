from __future__ import annotations

import inspect
import unittest

from dwr.tasks import certificate_generator as v1
from dwr.tasks import certificate_generator_v2 as v2
from dwr.tasks.schema import RelationKind


class CertificateGeneratorV2Tests(unittest.TestCase):
    def test_source_is_exact_v1_delegation(self):
        for index in (0, 1, 7):
            self.assertEqual(v2.construct(20261012, "source", index),
                             v1.construct(20261012, "source", index))

    def test_fixed_no_unary_add_candidates_never_emit_unary_initial_tuple(self):
        for index in (16, 20):
            result = v2.construct(20261012, "development", index)
            self.assertIsNone(result.rejection_reason)
            unknown = {name for name in result.task.variables} - {g.variable for g in result.givens}
            for constraint in result.task.constraints:
                self.assertNotEqual(sum(name in unknown for name in constraint.variables), 1)

    def test_fixed_eqneq_indices_apply_exact_witness_and_first_alternative_filters(self):
        for index in (0, 1, 2, 3):
            result = v2.construct(20261012, "development", index)
            witness = dict(result.witness.assignments)
            given_values = {given.variable: given.value for given in result.givens}
            self.assertEqual(set(witness.values()), {0, 1, 2})
            self.assertGreaterEqual(len(set(given_values.values())), 2)
            first = result.alternatives[0][1]
            alternative = dict(first.assignments)
            self.assertTrue(all(alternative[name] == value for name, value in given_values.items()))
            self.assertTrue(any(witness[left] == witness[right] and
                                alternative[left] != alternative[right]
                                for left, right in v1._normalized_tuples(RelationKind.EQ)))

    def test_budget_callback_aborts_and_guard_does_not_change_candidate(self):
        plain = v2.construct(20261012, "development", 4)
        guarded = v2.construct(20261012, "development", 4, check_budget=lambda: None)
        self.assertEqual(plain, guarded)

        class BudgetExpired(RuntimeError):
            pass

        calls = 0
        def expire():
            nonlocal calls
            calls += 1
            raise BudgetExpired

        with self.assertRaises(BudgetExpired):
            v2.construct(20261012, "development", 4, check_budget=expire)
        self.assertEqual(calls, 1)

    def test_development_seed_tag_and_repeatability(self):
        self.assertEqual(v2._SEED_TAG, b"dwr-certificate-sixvar-v2\0")
        self.assertEqual(v2.construct(20261012, "development", 4),
                         v2.construct(20261012, "development", 4))
        self.assertNotEqual(v2._rng(20261012, "development", 4).getrandbits(128),
                            v2._rng(20261012, "development", 5).getrandbits(128))

    def test_add_branch_schedule_is_exactly_alternating_by_twelve(self):
        for index in range(48):
            profile, _ = v1._profile_cell_schedule("development")[index % 12]
            if "add_mod" not in profile:
                self.assertIsNone(v2._add_branch(index))
            else:
                expected = "ordinary" if (index // 12) % 2 == 0 else "no_unary"
                self.assertEqual(v2._add_branch(index), expected)

    def test_initial_unknown_role_filter_allows_zero_or_two_or_three(self):
        given = frozenset(("v0", "v1", "v2"))
        self.assertTrue(v2._allowed_no_unary_tuple(("v0", "v1"), given))
        self.assertFalse(v2._allowed_no_unary_tuple(("v0", "v1", "v3"), given))
        self.assertTrue(v2._allowed_no_unary_tuple(("v3", "v4", "v5"), given))
        self.assertTrue(v2._allowed_no_unary_tuple(("v0", "v1", "v2"), given))
        operands = ("v0", "v1", "v2")
        for role in range(3):
            with self.subTest(two_unknown_given_role=role):
                self.assertTrue(v2._allowed_no_unary_tuple(
                    operands, frozenset((operands[role],))))
            with self.subTest(one_unknown_role=role):
                fixed = frozenset(operands[position] for position in range(3) if position != role)
                self.assertFalse(v2._allowed_no_unary_tuple(operands, fixed))

    def test_mixed_pool_counts_restricted_tuple_visits(self):
        from dwr.tasks.certificate_generator import _Work, _solution
        witness = _solution((0, 1, 2, 0, 1, 2))
        first = _solution((0, 1, 2, 1, 0, 2))
        alternatives = (first, _solution((0, 1, 2, 2, 1, 0)))
        givens = frozenset(("v0", "v1", "v2"))
        work = _Work(None)
        v2._mixed_feasible_second_alternatives(
            ("eq", "add_mod"), 2, witness, first, alternatives, work,
            givens=givens, no_unary=True)
        # Every normalized tuple is visited for each alternative and each kind;
        # filtering does not hide the work performed.
        expected = (len(v1._normalized_tuples(RelationKind.EQ)) +
                    len(v1._normalized_tuples(RelationKind.ADD_MOD)))
        self.assertEqual(work.constraint_pool_scans, expected)

    def test_eqneq_witness_filter_catalog_is_uniformly_filtered(self):
        from dwr.tasks.certificate_generator import _Work
        work = _Work(None)
        candidates = v2._eqneq_witnesses(("v0", "v1"), work)
        self.assertTrue(candidates)
        self.assertTrue(all(len(set(values)) == 3 for values in candidates))
        self.assertTrue(all(len({values[int(name[1:])] for name in ("v0", "v1")}) >= 2
                            for values in candidates))
        self.assertEqual(len(candidates), 6 * (3**4 - 2**4))
        self.assertEqual(candidates, tuple(sorted(candidates)))
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertEqual(work.witness_pool_scans, 3**6)

    def test_eq_first_alternative_filter_requires_rejecting_witness_true_pair(self):
        from dwr.tasks.certificate_generator import _Work, _solution
        witness = _solution((0, 0, 1, 1, 2, 2))
        givens = frozenset(("v0", "v2"))
        preserving = _solution((0, 0, 1, 1, 0, 0))
        rejecting = _solution((0, 2, 1, 0, 1, 2))
        alternatives = (preserving, rejecting)
        work = _Work(None)
        filtered = v2._eqneq_first_alternatives(witness, alternatives, work)
        self.assertEqual(filtered, (rejecting,))
        given_values = {name: value for name, value in witness.assignments if name in givens}
        self.assertTrue(all(dict(item.assignments)[name] == value
                            for item in filtered for name, value in given_values.items()))
        self.assertEqual(work.constraint_pool_scans,
                         len(alternatives) * len(v1._normalized_tuples(RelationKind.EQ)))

    def test_eqneq_witness_catalog_checks_abort_mid_scan(self):
        class BudgetExpired(RuntimeError):
            pass

        calls = 0
        def expire_mid_scan():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise BudgetExpired

        with self.assertRaises(BudgetExpired):
            v2.construct(20261012, "development", 0, check_budget=expire_mid_scan)
        self.assertEqual(calls, 3)

    def test_constructor_has_no_oracle_or_solver_dependencies(self):
        source = inspect.getsource(v2)
        self.assertNotIn("oracle", source.lower())
        self.assertNotIn("solver", source.lower())
        self.assertNotIn("gac", source.lower())
        self.assertNotIn("canonicaliz", source.lower())


if __name__ == "__main__":
    unittest.main()
