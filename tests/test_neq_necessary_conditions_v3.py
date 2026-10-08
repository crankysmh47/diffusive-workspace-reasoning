import itertools
import unittest

from dwr.tasks.generator_v3 import generate_candidate_v3
from dwr.tasks.oracle import enumerate_solutions
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, Solution


def task(edges, assignments, given_names):
    names = tuple(f"v{i}" for i in range(len(assignments)))
    constraints = tuple(Constraint(RelationKind.NEQ, (names[a], names[b])) for a, b in edges)
    givens = tuple(GivenAssignment(names[i], assignments[i]) for i in given_names)
    return ConstraintTask("constraint-task-v1", 3, names, constraints, givens)


def independent_solutions(candidate):
    """Tiny inline neq oracle independent of production solver/generator predicates."""
    names = candidate.variables
    edges = tuple(tuple(names.index(name) for name in row.variables)
                  for row in candidate.constraints)
    given_values = {names.index(given.variable): given.value for given in candidate.givens}
    return tuple(values for values in itertools.product((0, 1, 2), repeat=len(names))
                 if all(values[i] == value for i, value in given_values.items())
                 and all(values[left] != values[right] for left, right in edges))


def has_unanchored_bichromatic_component(candidate, witness):
    """Return the independent component obstruction described in task_spec_v3.md."""
    names = candidate.variables
    values = dict(witness.assignments)
    givens = {given.variable for given in candidate.givens}
    edges = [tuple(edge.variables) for edge in candidate.constraints]
    for first, second in ((0, 1), (0, 2), (1, 2)):
        vertices = {name for name in names if values[name] in (first, second)}
        adjacency = {name: set() for name in vertices}
        for left, right in edges:
            if left in vertices and right in vertices and not ({left, right} <= givens):
                adjacency[left].add(right)
                adjacency[right].add(left)
        pending = set(vertices)
        while pending:
            start = pending.pop()
            component = {start}
            stack = [start]
            while stack:
                here = stack.pop()
                for neighbor in adjacency[here] & pending:
                    pending.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
            if component - givens and not component & givens:
                return True
    return False


class NeqNecessaryConditionTests(unittest.TestCase):
    def test_valid_witness_contexts_always_have_capacity_for_the_registered_floor(self):
        contexts = 0
        for n in range(4, 7):
            names = tuple(range(n))
            pairs = tuple(itertools.combinations(names, 2))
            for given_count in range(2, n - 1):
                for given_names in itertools.combinations(names, given_count):
                    unknown = set(names) - set(given_names)
                    for values in itertools.product((0, 1, 2), repeat=n):
                        if len(set(values)) != 3 or len({values[index] for index in given_names}) < 2:
                            continue
                        capacity = sum(values[left] != values[right] and
                                       bool({left, right} & unknown) for left, right in pairs)
                        self.assertGreaterEqual(capacity, 2 * len(unknown))
                        self.assertLessEqual(2 * len(unknown), 8)
                        contexts += 1
        self.assertGreater(contexts, 0)

    def test_one_color_givens_leave_two_unanchored_colors_swappable(self):
        candidate = task(((0, 2), (1, 3), (2, 3)), (0, 0, 1, 2), (0, 1))
        solutions = independent_solutions(candidate)
        self.assertGreaterEqual(len(solutions), 2)
        self.assertIn((0, 0, 2, 1), solutions)

    def test_two_color_witness_allows_unknown_recoloring_to_third_value(self):
        candidate = task(((0, 2), (1, 3)), (0, 1, 1, 0), (0, 1))
        solutions = independent_solutions(candidate)
        self.assertIn((0, 1, 2, 0), solutions)
        self.assertIn((0, 1, 1, 0), solutions)

    def test_unanchored_bichromatic_component_is_a_nonunique_counterexample(self):
        candidate = task(((0, 2), (1, 3)), (0, 2, 1, 0), (0, 1))
        self.assertGreaterEqual(len(independent_solutions(candidate)), 2)
        witness = Solution(tuple(zip(candidate.variables, (0, 2, 1, 0))))
        self.assertTrue(has_unanchored_bichromatic_component(candidate, witness))

    def test_necessary_conditions_are_not_sufficient_for_uniqueness(self):
        # This six-variable graph has all three colors, two fixed distinct values,
        # anchored bichromatic components, and exactly 2u edges, yet admits two completions.
        candidate = task(((1, 3), (4, 5), (0, 3), (0, 1), (0, 2), (3, 4), (1, 5), (2, 4)),
                         (0, 2, 1, 1, 2, 0), (2, 5))
        solutions = independent_solutions(candidate)
        self.assertGreaterEqual(len(solutions), 2)

    def test_generated_oracle_unique_source_neq_tasks_satisfy_all_registered_necessities(self):
        unique_count = 0
        for index in range(1, 400):
            outcome = generate_candidate_v3(20261007, "source", index)
            if outcome.profile != ("neq",) or not outcome.oracle_unique:
                continue
            unique_count += 1
            given_names = {given.variable for given in outcome.task.givens}
            unknown_count = outcome.n - len(given_names)
            witness_values = {value for _, value in outcome.witness.assignments}
            self.assertEqual(witness_values, {0, 1, 2})
            self.assertGreaterEqual(len({given.value for given in outcome.task.givens}), 2)
            self.assertFalse(has_unanchored_bichromatic_component(outcome.task, outcome.witness))
            touching_count = sum(bool(set(constraint.variables) - given_names)
                                 for constraint in outcome.task.constraints)
            # For each color pair, every unknown-containing component has an anchor;
            # a connected component with U unknown and G>=1 givens needs U+G-1>=U
            # edges. Summing the three pair bounds counts each unknown twice and each
            # unknown-touching neq edge once, so at least 2u distinct edges are necessary.
            self.assertGreaterEqual(touching_count, 2 * unknown_count)
            self.assertEqual(len(independent_solutions(outcome.task)), 1)
            self.assertEqual(len(enumerate_solutions(outcome.task).solutions), 1)
        self.assertGreater(unique_count, 0)


if __name__ == "__main__":
    unittest.main()
