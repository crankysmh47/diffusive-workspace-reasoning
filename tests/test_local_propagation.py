from __future__ import annotations

import itertools
import unittest

from dwr.tasks.local_propagation import PropagationResult, propagate
from dwr.tasks.oracle import enumerate_solutions
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, TaskFormatError


def task(n, constraints, givens):
    return ConstraintTask(
        "constraint-task-v1", 3, tuple(f"v{i}" for i in range(n)),
        tuple(constraints), tuple(givens),
    )


def eq(left, right):
    return Constraint(RelationKind.EQ, (left, right))


def neq(left, right):
    return Constraint(RelationKind.NEQ, (left, right))


def add(left, right, result):
    return Constraint(RelationKind.ADD_MOD, (left, right, result))


def scalar_holds(constraint, assignment):
    values = tuple(assignment[name] for name in constraint.variables)
    if constraint.kind is RelationKind.EQ:
        return values[0] == values[1]
    if constraint.kind is RelationKind.NEQ:
        return values[0] != values[1]
    return (values[0] + values[1]) % 3 == values[2]


def reference_fixed_point(problem):
    """Independent, deliberately simple snapshot GAC used as a tiny oracle."""
    domains = [{0, 1, 2} for _ in problem.variables]
    position = {name: i for i, name in enumerate(problem.variables)}
    for given in problem.givens:
        domains[position[given.variable]] = {given.value}
    while True:
        snapshot = [set(domain) for domain in domains]
        supported_by_constraint = []
        for constraint in problem.constraints:
            indices = [position[name] for name in constraint.variables]
            supported = [set() for _ in indices]
            for values in itertools.product(*(snapshot[i] for i in indices)):
                local = dict(zip(constraint.variables, values))
                if scalar_holds(constraint, local):
                    for slot, value in enumerate(values):
                        supported[slot].add(value)
            supported_by_constraint.append((indices, supported))
        revised = [set(domain) for domain in snapshot]
        for indices, supported in supported_by_constraint:
            for index, values in zip(indices, supported):
                revised[index].intersection_update(values)
        if revised == snapshot:
            return tuple(frozenset(domain) for domain in revised)
        domains = revised


class LocalPropagationTests(unittest.TestCase):
    def test_anchored_equality_chain_solves_every_variable(self):
        problem = task(6, [eq(f"v{i}", f"v{i + 1}") for i in range(5)],
                       [GivenAssignment("v0", 2)])
        result = propagate(problem)
        self.assertIsInstance(result, PropagationResult)
        self.assertEqual(result.domains, (frozenset({2}),) * 6)
        self.assertTrue(result.solved)
        self.assertFalse(result.stalled)
        self.assertFalse(result.contradiction)
        self.assertEqual(result.unresolved_unknowns, 0)
        self.assertEqual(result.rounds, 6)
        self.assertGreater(result.support_checks, 0)

    def test_snapshot_sweeps_are_constraint_order_invariant(self):
        constraints = [
            add("v1", "v2", "v0"),
            eq("v2", "v0"),
            neq("v3", "v4"),
        ]
        givens = [GivenAssignment("v0", 0)]
        forward = propagate(task(5, constraints, givens))
        reverse = propagate(task(5, list(reversed(constraints)), givens))
        self.assertEqual(forward.domains, reverse.domains)
        self.assertEqual(forward.solved, reverse.solved)
        self.assertEqual(forward.stalled, reverse.stalled)
        self.assertEqual(forward.contradiction, reverse.contradiction)

    def test_add_constraint_prunes_values_supported_by_other_rule_only(self):
        problem = task(3, [add("v1", "v2", "v0"), eq("v2", "v0")],
                       [GivenAssignment("v0", 0)])
        result = propagate(problem)
        self.assertEqual(result.domains, (frozenset({0}),) * 3)
        self.assertTrue(result.solved)

    def test_disconnected_variables_stall_without_claiming_uniqueness(self):
        problem = task(6, [eq("v0", "v1"), neq("v2", "v3")],
                       [GivenAssignment("v0", 1)])
        result = propagate(problem)
        self.assertFalse(result.solved)
        self.assertTrue(result.stalled)
        self.assertFalse(result.contradiction)
        self.assertEqual(result.unresolved_unknowns, 4)
        self.assertEqual(len(enumerate_solutions(problem).solutions), 54)

    def test_unique_task_can_stall_at_gac_fixed_point(self):
        problem = task(3, [eq("v1", "v2"), add("v1", "v2", "v0")],
                       [GivenAssignment("v0", 2)])
        oracle = enumerate_solutions(problem).solutions
        self.assertEqual(tuple(solution.assignments for solution in oracle),
                         ((("v0", 2), ("v1", 1), ("v2", 1)),))
        result = propagate(problem)
        self.assertEqual(result.domains,
                         (frozenset({2}), frozenset({0, 1, 2}), frozenset({0, 1, 2})))
        self.assertFalse(result.solved)
        self.assertTrue(result.stalled)
        self.assertFalse(result.contradiction)
        self.assertEqual(result.unresolved_unknowns, 2)

    def test_inconsistent_givens_and_equality_reach_empty_domain(self):
        problem = task(4, [eq("v0", "v1"), eq("v2", "v3")], [
            GivenAssignment("v0", 0), GivenAssignment("v1", 1),
        ])
        result = propagate(problem)
        self.assertTrue(result.contradiction)
        self.assertFalse(result.solved)
        self.assertFalse(result.stalled)
        self.assertTrue(any(not domain for domain in result.domains))

    def test_oracle_solutions_are_preserved_in_final_domains(self):
        fixtures = [
            task(3, [add("v0", "v1", "v2"), neq("v0", "v2")],
                 [GivenAssignment("v0", 1)]),
            task(4, [eq("v0", "v1"), add("v1", "v2", "v3")],
                 [GivenAssignment("v0", 2)]),
            task(4, [neq("v0", "v1"), neq("v1", "v2"), neq("v2", "v3")],
                 [GivenAssignment("v0", 0)]),
        ]
        for problem in fixtures:
            with self.subTest(problem=problem):
                result = propagate(problem)
                for solution in enumerate_solutions(problem).solutions:
                    values = dict(solution.assignments)
                    self.assertTrue(all(values[name] in result.domains[i]
                                        for i, name in enumerate(problem.variables)))

    def test_matches_independent_support_fixed_point_on_tiny_fixtures(self):
        fixtures = [
            task(3, [eq("v0", "v1"), neq("v1", "v2")], [GivenAssignment("v0", 0)]),
            task(3, [add("v0", "v1", "v2"), eq("v0", "v2")], [GivenAssignment("v1", 2)]),
            task(4, [neq("v0", "v1"), add("v1", "v2", "v3")], [GivenAssignment("v3", 1)]),
            task(4, [eq("v0", "v1"), eq("v1", "v2")],
                 [GivenAssignment("v0", 0), GivenAssignment("v2", 1)]),
        ]
        for problem in fixtures:
            with self.subTest(problem=problem):
                self.assertEqual(propagate(problem).domains, reference_fixed_point(problem))

    def test_strict_input_and_budget_callback(self):
        with self.assertRaises(TaskFormatError):
            propagate(object())
        calls = []

        def guard():
            calls.append(None)

        problem = task(3, [eq("v0", "v1"), eq("v1", "v2")],
                       [GivenAssignment("v0", 1)])
        propagate(problem, check_budget=guard)
        self.assertGreater(len(calls), 0)
        with self.assertRaises(TypeError):
            propagate(problem, check_budget=True)

        class BudgetStop(Exception):
            pass

        marker = BudgetStop("stop now")

        def stop():
            raise marker

        with self.assertRaises(BudgetStop) as caught:
            propagate(problem, check_budget=stop)
        self.assertIs(caught.exception, marker)

    def test_propagation_result_rejects_noninteger_domains_and_inconsistent_states(self):
        with self.assertRaises(TaskFormatError):
            PropagationResult((frozenset({1.0}), frozenset({0}), frozenset({0})),
                              True, False, False, 0, 1, 1)

        singleton = (frozenset({0}),) * 3
        empty = (frozenset(), frozenset({0}), frozenset({0}))
        nonsingleton = (frozenset({0, 1}), frozenset({0}), frozenset({0}))
        invalid_results = [
            (singleton, False, False, False, 0, 1, 1),
            (empty, False, True, False, 0, 1, 1),
            (nonsingleton, True, False, False, 1, 1, 1),
            (singleton, False, True, False, 0, 1, 1),
            (empty, False, False, False, 0, 1, 1),
            ((frozenset({0}), frozenset({1})), True, False, False, 0, 1, 1),
            ((frozenset({0, 1}),) * 3, False, True, False, 2, 1, 1),
            (singleton, True, False, False, 0, 0, 1),
        ]
        for arguments in invalid_results:
            with self.subTest(arguments=arguments), self.assertRaises(TaskFormatError):
                PropagationResult(*arguments)


if __name__ == "__main__":
    unittest.main()
