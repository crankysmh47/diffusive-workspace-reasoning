from __future__ import annotations

import itertools
import ast
from dataclasses import replace
import inspect
import random
import unittest
from dataclasses import FrozenInstanceError

from dwr.tasks.certificate_generator import (
    CELLS,
    PROFILES,
    _profile_cell_schedule,
    construct,
)
from dwr.tasks.schema import Constraint, ConstraintTask, RelationKind, Solution


def scalar_satisfies(kind: RelationKind, operands: tuple[str, ...], values: dict[str, int]) -> bool:
    selected = tuple(values[name] for name in operands)
    if kind is RelationKind.EQ:
        return selected[0] == selected[1]
    if kind is RelationKind.NEQ:
        return selected[0] != selected[1]
    return (selected[0] + selected[1]) % 3 == selected[2]


def scalar_solutions(task, constraints=None):
    fixed = {given.variable: given.value for given in task.givens}
    unknown = tuple(name for name in task.variables if name not in fixed)
    result = []
    for values in itertools.product(range(3), repeat=len(unknown)):
        assignment = fixed | dict(zip(unknown, values))
        checked = task.constraints if constraints is None else constraints
        if all(scalar_satisfies(item.kind, item.variables, assignment) for item in checked):
            result.append(Solution(tuple((name, assignment[name]) for name in task.variables)))
    return tuple(result)


def sol(values):
    return Solution(tuple((f"v{i}", value) for i, value in enumerate(values)))


def independent_tuples(kind):
    names = tuple(f"v{i}" for i in range(6))
    pairs = tuple(itertools.combinations(names, 2))
    if kind in (RelationKind.EQ, RelationKind.NEQ):
        return pairs
    return tuple((*pair, target) for pair in pairs for target in names if target not in pair)


def independent_source_pools(kind, witness, alternative, givens):
    rejecting, neutral = [], []
    for operands in independent_tuples(kind):
        if kind is RelationKind.NEQ and set(operands) <= givens:
            continue
        under_w = scalar_satisfies(kind, operands, witness)
        under_a = scalar_satisfies(kind, operands, alternative)
        if under_w and not under_a:
            rejecting.append(operands)
        elif under_w and under_a:
            neutral.append(operands)
    return tuple(rejecting), tuple(neutral)


def independent_mixed_pools(kinds, witness, first, second):
    allowed_pools, rejecting_pools = [], []
    alternatives = (first, second)
    for index, kind in enumerate(kinds):
        own, other = alternatives[index], alternatives[1 - index]
        allowed, rejecting = [], []
        for operands in independent_tuples(kind):
            if scalar_satisfies(kind, operands, witness) and scalar_satisfies(kind, operands, other):
                allowed.append(operands)
                if not scalar_satisfies(kind, operands, own):
                    rejecting.append(operands)
        allowed_pools.append(tuple(allowed))
        rejecting_pools.append(tuple(rejecting))
    return tuple(allowed_pools), tuple(rejecting_pools)


def independent_count_vectors(capacities, total):
    return tuple(counts for counts in itertools.product(
        *(range(1, min(capacity, total) + 1) for capacity in capacities)) if sum(counts) == total)


class CertificateGeneratorTests(unittest.TestCase):
    def test_profile_and_cell_schedule_is_profile_major(self):
        expected = {
            "source": tuple((profile, cell) for profile in PROFILES["source"] for cell in CELLS),
            "development": tuple((profile, cell) for profile in PROFILES["development"] for cell in CELLS),
        }
        for partition, schedule in expected.items():
            with self.subTest(partition=partition):
                self.assertEqual(_profile_cell_schedule(partition), schedule)

    def test_fixed_candidate_metadata_pins_rng_derivation_and_draw_order(self):
        result = construct(20261011, "source", 0)
        self.assertEqual((result.profile, result.cell), (("eq",), "A"))
        self.assertEqual(tuple((g.variable, g.value) for g in result.givens), (("v2", 1), ("v3", 1)))
        self.assertEqual(tuple(value for _, value in result.witness.assignments), (1, 1, 1, 1, 1, 1))
        self.assertEqual(result.work.rng_calls, 5)
        self.assertEqual(result.reserved_constraints[0].variables, ("v1", "v4"))

        mixed = construct(20261011, "development", 4)
        self.assertEqual(mixed.work.rng_calls, 14)
        self.assertEqual(mixed.work.feasible_second_alternatives, 76)
        self.assertEqual(mixed.work.feasible_count_vectors, 178)
        self.assertEqual(tuple(c.kind.value for c in mixed.reserved_constraints), ("eq", "add_mod"))

    def test_candidate_rng_is_local_and_reproducible(self):
        random.seed(917)
        before = random.getstate()
        first = construct(20261011, "source", 7)
        after = random.getstate()
        second = construct(20261011, "source", 7)
        self.assertEqual(before, after)
        self.assertEqual(first, second)

    def test_successful_task_has_scalar_certificate_and_fixed_givens(self):
        # This fixed candidate is a fixture, not a population search or a training draw.
        result = construct(20261011, "development", 4)
        self.assertIsNone(result.rejection_reason)
        self.assertIsNotNone(result.task)
        self.assertIsNotNone(result.witness)
        task = result.task
        witness = dict(result.witness.assignments)
        givens = {given.variable: given.value for given in task.givens}
        self.assertEqual(givens, {given.variable: given.value for given in result.givens})
        self.assertIn(result.witness, scalar_solutions(task))
        self.assertEqual(tuple(sorted(task.constraints, key=lambda c: (c.kind.value, c.variables))), task.constraints)
        self.assertEqual(tuple(sorted(task.givens, key=lambda g: g.variable)), task.givens)
        self.assertTrue(all(scalar_satisfies(c.kind, c.variables, witness) for c in task.constraints))
        alternatives = dict(result.alternatives)
        for kind, alternative in result.alternatives:
            alt = dict(alternative.assignments)
            self.assertEqual({name: alt[name] for name in givens}, givens)
            self.assertNotEqual(alternative, result.witness)
        for protected in result.reserved_constraints:
            self.assertIn(protected, task.constraints)
            self.assertTrue(scalar_satisfies(protected.kind, protected.variables, witness))
        for kind, alternative in alternatives.items():
            alt = dict(alternative.assignments)
            self.assertTrue(any(
                c.kind.value == kind and scalar_satisfies(c.kind, c.variables, witness) and
                not scalar_satisfies(c.kind, c.variables, alt) for c in task.constraints
            ))
            remaining = tuple(c for c in task.constraints if c.kind.value != kind)
            self.assertIn(alternative, scalar_solutions(task, remaining))

    def test_source_reserved_constraint_preserves_its_alternative(self):
        result = construct(20261011, "source", 0)
        self.assertIsNone(result.rejection_reason)
        self.assertEqual(len(result.reserved_constraints), 1)
        task = result.task
        protected = result.reserved_constraints[0]
        self.assertIn(protected, task.constraints)
        self.assertIn(result.witness, scalar_solutions(task))
        alternative = result.alternatives[0][1]
        remaining = tuple(c for c in task.constraints if c != protected)
        self.assertIn(alternative, scalar_solutions(task, remaining))

    def test_registered_profiles_have_feasible_fixed_pool_examples(self):
        from dwr.tasks.certificate_generator import (
            _mixed_feasible_second_alternatives,
            _source_feasible_alternatives,
            _Work,
        )

        witness = sol((0, 1, 0, 0, 0, 1))
        givens = frozenset(("v0", "v1", "v5"))
        given_values = {name: dict(witness.assignments)[name] for name in givens}
        source_examples = {
            ("eq",): sol((0, 1, 0, 0, 1, 1)),
            ("neq",): sol((0, 1, 0, 0, 1, 1)),
            ("add_mod",): sol((0, 1, 0, 0, 1, 1)),
        }
        for profile, alternative in source_examples.items():
            with self.subTest(profile=profile):
                alt_values = dict(alternative.assignments)
                self.assertNotEqual(alternative, witness)
                self.assertEqual({name: alt_values[name] for name in givens}, given_values)
                feasible = _source_feasible_alternatives(
                    profile, 2, witness, (alternative,), givens, _Work(None))
                self.assertEqual(len(feasible), 1)
                _, rejecting, neutral = feasible[0]
                kind = RelationKind(profile[0])
                expected_rejecting, expected_neutral = independent_source_pools(
                    kind, dict(witness.assignments), alt_values, givens)
                self.assertEqual(rejecting, expected_rejecting)
                self.assertEqual(neutral, expected_neutral)

        mixed_examples = {
            ("eq", "neq"): (sol((0, 1, 0, 0, 1, 1)), sol((0, 1, 0, 1, 0, 1))),
            ("eq", "add_mod"): (sol((0, 1, 0, 0, 1, 1)), sol((0, 1, 0, 1, 0, 1))),
            ("neq", "add_mod"): (sol((0, 1, 0, 0, 1, 1)), sol((0, 1, 0, 1, 0, 1))),
        }
        for profile, (first, second) in mixed_examples.items():
            with self.subTest(profile=profile):
                for alternative in (first, second):
                    alt_values = dict(alternative.assignments)
                    self.assertNotEqual(alternative, witness)
                    self.assertEqual({name: alt_values[name] for name in givens}, given_values)
                feasible = _mixed_feasible_second_alternatives(
                    profile, 2, witness, first, (second,), _Work(None))
                self.assertEqual(len(feasible), 1)
                returned_second, allowed, rejecting, vectors = feasible[0]
                self.assertEqual(returned_second, second)
                expected_allowed, expected_rejecting = independent_mixed_pools(
                    tuple(RelationKind(name) for name in profile), dict(witness.assignments),
                    dict(first.assignments), dict(second.assignments))
                self.assertEqual(allowed, expected_allowed)
                self.assertEqual(rejecting, expected_rejecting)
                self.assertTrue(all(items for items in rejecting))
                self.assertEqual(vectors, independent_count_vectors(tuple(map(len, allowed)), 2))

    def test_empty_source_pool_is_reported_without_refill(self):
        from dwr.tasks.certificate_generator import _source_feasible_alternatives, _Work

        witness = sol((0, 1, 0, 1, 0, 1))
        self.assertEqual(
            _source_feasible_alternatives(("eq",), 2, witness, (witness,), frozenset(("v0", "v1")), _Work(None)), ()
        )

    def test_fixed_empty_candidate_pools_end_the_attempt(self):
        source = construct(20261011, "source", 1)
        self.assertEqual(source.rejection_reason, "no_feasible_source_alternative")
        self.assertIsNone(source.task)
        self.assertEqual(source.work.feasible_alternatives, 0)
        mixed = construct(20261011, "development", 1)
        self.assertEqual(mixed.rejection_reason, "no_feasible_mixed_second_alternative")
        self.assertIsNone(mixed.task)
        self.assertEqual(mixed.work.feasible_second_alternatives, 0)

    def test_public_outcome_rejects_malformed_identity_and_solution_shape(self):
        result = construct(20261011, "source", 0)
        with self.assertRaises((TypeError, ValueError)):
            replace(result, root_seed=-1)
        with self.assertRaises((TypeError, ValueError)):
            replace(result, candidate_index=True)
        with self.assertRaises((TypeError, ValueError)):
            replace(result, partition="final")
        with self.assertRaises(ValueError):
            replace(result, witness=Solution((("v0", 1), ("v1", 1), ("v2", 1))))

    def test_public_outcome_rejects_task_profile_and_witness_disagreement(self):
        result = construct(20261011, "development", 4)
        task = result.task
        wrong_profile_constraints = tuple(
            Constraint(RelationKind.NEQ, (f"v{i}", f"v{j}"))
            for i, j in ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4))
        )
        wrong_profile_task = ConstraintTask(task.schema_version, task.modulus, task.variables,
                                            wrong_profile_constraints, task.givens)
        with self.assertRaises(ValueError):
            replace(result, task=wrong_profile_task)
        false_constraint = Constraint(RelationKind.ADD_MOD, ("v0", "v1", "v2"))
        changed = list(task.constraints)
        changed.remove(next(c for c in changed if c.kind is RelationKind.ADD_MOD))
        changed.append(false_constraint)
        invalid_witness_task = ConstraintTask(task.schema_version, task.modulus, task.variables,
                                              tuple(sorted(changed, key=lambda c: (c.kind.value, c.variables))),
                                              task.givens)
        with self.assertRaises(ValueError):
            replace(result, task=invalid_witness_task)

    def test_public_outcome_rejects_alternative_order_and_reserve_mismatch(self):
        result = construct(20261011, "development", 4)
        with self.assertRaises(ValueError):
            replace(result, alternatives=tuple(reversed(result.alternatives)))
        with self.assertRaises(ValueError):
            replace(result, alternatives=(("eq", result.witness), result.alternatives[1]))
        with self.assertRaises(ValueError):
            replace(result, alternatives=(result.alternatives[0], ("add_mod", result.alternatives[0][1])))
        with self.assertRaises(ValueError):
            replace(result, reserved_constraints=(result.reserved_constraints[1],) * 2)
        source = construct(20261011, "source", 0)
        absent = Constraint(RelationKind.EQ, ("v0", "v5"))
        self.assertNotIn(absent, source.task.constraints)
        with self.assertRaises(ValueError):
            replace(source, reserved_constraints=(absent,))

    def test_public_outcome_rejects_stage_incompatible_rejections(self):
        source = construct(20261011, "source", 1)
        with self.assertRaises(ValueError):
            replace(source, rejection_reason="no_feasible_mixed_second_alternative")
        with self.assertRaises(ValueError):
            replace(source, witness=None, givens=())
        with self.assertRaises(ValueError):
            replace(source, alternatives=(("eq", source.witness),))
        with self.assertRaises(ValueError):
            replace(source, rejection_reason="empty_witness_pool", witness=None, givens=())
        mixed = construct(20261011, "development", 1)
        with self.assertRaises(ValueError):
            replace(mixed, rejection_reason="no_feasible_source_alternative")
        with self.assertRaises(ValueError):
            replace(mixed, alternatives=())
        wrong_kind = (mixed.profile[1], mixed.alternatives[0][1])
        with self.assertRaises(ValueError):
            replace(mixed, alternatives=(wrong_kind,))

    def test_pool_enumeration_checks_budget_and_counts_work(self):
        calls = []
        result = construct(20261011, "source", 0, check_budget=lambda: calls.append(1))
        self.assertGreater(result.work.constraint_pool_scans, 0)
        self.assertGreater(len(calls), 0)

    def test_budget_callback_exception_aborts_pool_enumeration(self):
        class BudgetExpired(RuntimeError):
            pass

        calls = 0

        def check_budget():
            nonlocal calls
            calls += 1
            if calls == 4:
                raise BudgetExpired

        with self.assertRaises(BudgetExpired):
            construct(20261011, "source", 0, check_budget=check_budget)
        self.assertEqual(calls, 4)

    def test_budget_callback_does_not_change_candidate(self):
        plain = construct(20261011, "development", 4)
        guarded = construct(20261011, "development", 4, check_budget=lambda: None)
        self.assertEqual(plain, guarded)

    def test_outcome_is_immutable(self):
        result = construct(20261011, "source", 0)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            result.cell = "Z"

    def test_rejects_invalid_indices_seeds_partitions_and_final(self):
        for args in ((-1, "source", 0), (2**64, "source", 0), (True, "source", 0),
                     (0, "source", -1), (0, "source", 2**64), (0, "source", True),
                     (0, "final", 0), (0, "unknown", 0)):
            with self.subTest(args=args):
                with self.assertRaises((TypeError, ValueError)):
                    construct(args[0], args[1], args[2])
        with self.assertRaises(TypeError):
            construct(0, "source", 0, check_budget=True)
        with self.assertRaises(TypeError):
            construct("1", "source", 0)
        with self.assertRaises(TypeError):
            construct(1, "source", "0")

    def test_constructor_source_has_no_solver_or_split_dependencies(self):
        source = inspect.getsource(inspect.getmodule(construct))
        tree = ast.parse(source)
        forbidden = {"oracle", "solver", "split", "verifier"}
        imported = set()
        referenced = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.update(node.module.split(".")[0] for _ in node.names)
            elif isinstance(node, ast.Name):
                referenced.add(node.id.lower())
        self.assertFalse(imported & forbidden)
        self.assertFalse({name for name in referenced if any(word in name for word in forbidden)})


if __name__ == "__main__":
    unittest.main()
