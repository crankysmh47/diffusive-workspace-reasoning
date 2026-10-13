import itertools
import unittest

from dwr.tasks.oracle import enumerate_solutions
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, Solution
from dwr.tasks.solver import SolveResult, SolverStats, solve
from dwr.tasks.verifier import verify_answer


def c(kind, *variables):
    return Constraint(kind, tuple(variables))


def task(n, constraints, givens):
    return ConstraintTask(
        "constraint-task-v1", 3, tuple(f"v{i}" for i in range(n)),
        tuple(constraints), tuple(givens),
    )


class SolverTests(unittest.TestCase):
    def test_hand_computable_first_and_uniqueness_node_counts(self):
        problem = task(3, (
            c(RelationKind.EQ, "v0", "v1"),
            c(RelationKind.NEQ, "v1", "v2"),
        ), (GivenAssignment("v0", 1),))

        result = solve(problem)

        self.assertEqual(result, SolveResult(
            solution=Solution((("v0", 1), ("v1", 1), ("v2", 0))),
            stats=SolverStats(3, 4, 2),
        ))

    def test_contradictory_grounded_relation_fails_before_a_dfs_root(self):
        problem = task(6, (
            c(RelationKind.EQ, "v0", "v1"),
            c(RelationKind.NEQ, "v0", "v1"),
        ), tuple(GivenAssignment(f"v{i}", 0) for i in range(4)))

        self.assertEqual(solve(problem), SolveResult(None, SolverStats(0, 0, 0)))

    def test_exhaustive_n3_constraint_pairs_match_independent_oracle(self):
        variables = ("v0", "v1", "v2")
        candidates = [
            *(c(kind, *pair) for kind in (RelationKind.EQ, RelationKind.NEQ)
              for pair in itertools.combinations(variables, 2)),
            *(c(RelationKind.ADD_MOD, *pair, result)
              for pair in itertools.combinations(variables, 2)
              for result in variables if result not in pair),
        ]
        cases = 0
        seen_counts = set()
        for constraints in itertools.combinations(candidates, 2):
            for variable in variables:
                for value in range(3):
                    problem = task(3, constraints, (GivenAssignment(variable, value),))
                    oracle = enumerate_solutions(problem).solutions
                    result = solve(problem)
                    self.assertEqual(result.stats.uniqueness_solution_count_capped, min(2, len(oracle)))
                    self.assertEqual(result.solution, oracle[0] if oracle else None)
                    self.assertGreaterEqual(result.stats.first_solution_nodes, 1)
                    self.assertGreaterEqual(result.stats.uniqueness_nodes, result.stats.first_solution_nodes)
                    if result.solution is not None:
                        answer = '{"assignment":{' + ",".join(
                            f'"{name}":{value}' for name, value in result.solution.assignments
                        ) + "}}"
                        self.assertTrue(verify_answer(problem, answer).valid)
                    seen_counts.add(min(2, len(oracle)))
                    cases += 1
        self.assertEqual(cases, 324)
        self.assertEqual(seen_counts, {0, 1, 2})

    def test_constraint_and_given_order_do_not_change_result_or_mutate_task(self):
        constraints = (
            c(RelationKind.NEQ, "v1", "v2"),
            c(RelationKind.EQ, "v0", "v1"),
        )
        givens = (GivenAssignment("v0", 1),)
        original = task(3, constraints, givens)
        before = (original.constraints, original.givens)
        baseline = solve(original)
        reordered = task(3, tuple(reversed(constraints)), tuple(reversed(givens)))
        operand_reordered = task(3, (
            c(RelationKind.EQ, "v1", "v0"),
            c(RelationKind.NEQ, "v2", "v1"),
        ), givens)

        self.assertEqual(solve(reordered), baseline)
        self.assertEqual(solve(operand_reordered), baseline)
        self.assertEqual((original.constraints, original.givens), before)
        self.assertEqual(solve(original), baseline)

    def test_result_records_reject_invalid_mutable_or_out_of_range_fields(self):
        for values in ((True, 0, 0), (1, -1, 0), (1, 0, 3), (1.0, 0, 0)):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                SolverStats(*values)
        with self.assertRaises(TypeError):
            SolveResult([], SolverStats(0, 0, 0))
        with self.assertRaises(TypeError):
            SolveResult(None, {})

    def test_transfer_interaction_fixtures_preserve_unique_answers(self):
        positive = task(5, (
            c(RelationKind.EQ, "v1", "v2"),
            c(RelationKind.ADD_MOD, "v2", "v3", "v4"),
            c(RelationKind.NEQ, "v1", "v0"),
            c(RelationKind.NEQ, "v3", "v0"),
        ), (GivenAssignment("v0", 0), GivenAssignment("v4", 1)))
        negative = task(6, (
            c(RelationKind.EQ, "v0", "v5"),
            c(RelationKind.NEQ, "v1", "v0"),
            c(RelationKind.NEQ, "v3", "v0"),
            c(RelationKind.NEQ, "v1", "v3"),
            c(RelationKind.NEQ, "v2", "v0"),
            c(RelationKind.ADD_MOD, "v2", "v3", "v4"),
        ), (GivenAssignment("v0", 0), GivenAssignment("v4", 1), GivenAssignment("v5", 0)))

        for problem in (positive, negative):
            with self.subTest(n=len(problem.variables)):
                expected = enumerate_solutions(problem).solutions
                self.assertEqual(len(expected), 1)
                self.assertEqual(solve(problem).solution, expected[0])


if __name__ == "__main__":
    unittest.main()
