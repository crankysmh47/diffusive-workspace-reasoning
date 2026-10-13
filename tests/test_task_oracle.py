import itertools
import unittest

from dwr.tasks.oracle import enumerate_assignments, enumerate_solutions
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, TaskFormatError
from dwr.tasks.verifier import verify_answer


VARS = ("v0", "v1", "v2")


def c(kind, *variables):
    return Constraint(kind, tuple(variables))


def t(constraints, givens=()):
    return ConstraintTask("constraint-task-v1", 3, VARS, tuple(constraints), tuple(givens))


class OracleTests(unittest.TestCase):
    def test_relation_truth_tables_match_inline_math(self):
        relations = [
            (c(RelationKind.EQ, "v0", "v1"), lambda a: a[0] == a[1]),
            (c(RelationKind.NEQ, "v0", "v1"), lambda a: a[0] != a[1]),
            (c(RelationKind.ADD_MOD, "v0", "v1", "v2"), lambda a: (a[0] + a[1]) % 3 == a[2]),
        ]
        for constraint, formula in relations:
            with self.subTest(kind=constraint.kind):
                result = enumerate_assignments(VARS, (), (constraint,))
                expected = [dict(zip(VARS, values)) for values in itertools.product(range(3), repeat=3) if formula(values)]
                self.assertEqual([dict(solution.assignments) for solution in result.solutions], expected)
                self.assertEqual(result.assignments_examined, 27)

    def test_zero_one_and_multiple_solutions_and_hand_fixture(self):
        unique_eq = t((c(RelationKind.EQ, "v0", "v1"), c(RelationKind.EQ, "v1", "v2")), (GivenAssignment("v0", 2),))
        unique = enumerate_solutions(unique_eq)
        self.assertEqual([dict(s.assignments) for s in unique.solutions], [{"v0": 2, "v1": 2, "v2": 2}])
        contradiction = t((c(RelationKind.EQ, "v0", "v1"), c(RelationKind.NEQ, "v0", "v1")), (GivenAssignment("v0", 1),))
        self.assertEqual(enumerate_solutions(contradiction).solutions, ())
        ambiguous = t((c(RelationKind.EQ, "v0", "v1"), c(RelationKind.NEQ, "v0", "v2")), (GivenAssignment("v0", 1),))
        result = enumerate_solutions(ambiguous)
        self.assertEqual(len(result.solutions), 2)
        modular = t((c(RelationKind.ADD_MOD, "v0", "v1", "v2"), c(RelationKind.NEQ, "v0", "v1")), (GivenAssignment("v0", 1),))
        self.assertEqual([dict(s.assignments) for s in enumerate_solutions(modular).solutions], [
            {"v0": 1, "v1": 0, "v2": 1}, {"v0": 1, "v1": 2, "v2": 0},
        ])

    def test_oracle_and_verifier_agree_for_every_assignment(self):
        fixtures = [
            t((c(RelationKind.EQ, "v0", "v1"), c(RelationKind.NEQ, "v1", "v2")), (GivenAssignment("v0", 1),)),
            t((c(RelationKind.ADD_MOD, "v0", "v1", "v2"), c(RelationKind.NEQ, "v0", "v1")), (GivenAssignment("v0", 1),)),
        ]
        for task in fixtures:
            oracle = {solution.assignments for solution in enumerate_solutions(task).solutions}
            for values in itertools.product(range(3), repeat=3):
                pairs = tuple(zip(VARS, values))
                answer = "{" + '"assignment":{' + ",".join(f'"{name}":{value}' for name, value in pairs) + "}}"
                self.assertEqual(verify_answer(task, answer).valid, pairs in oracle)

    def test_exclusions_use_canonical_constraint_indices(self):
        canonical = [c(RelationKind.EQ, "v0", "v1"), c(RelationKind.NEQ, "v0", "v2")]
        shuffled = t(tuple(reversed(canonical)), (GivenAssignment("v0", 1),))
        all_constraints = enumerate_solutions(shuffled)
        exclude_first = enumerate_solutions(shuffled, excluded_constraint_indices=(0,))
        self.assertEqual(len(all_constraints.solutions), 2)
        self.assertEqual(len(exclude_first.solutions), 6)
        no_constraints = enumerate_solutions(shuffled, excluded_constraint_indices=(0, 1))
        self.assertEqual(len(no_constraints.solutions), 9)
        self.assertEqual(no_constraints.assignments_examined, 27)

    def test_internal_counterfactual_allows_all_givens_and_no_constraints(self):
        result = enumerate_assignments(VARS, tuple(GivenAssignment(v, i) for i, v in enumerate(VARS)), ())
        self.assertEqual([dict(s.assignments) for s in result.solutions], [{"v0": 0, "v1": 1, "v2": 2}])
        self.assertEqual(result.assignments_examined, 27)

    def test_invalid_exclusions_are_rejected(self):
        task = t((c(RelationKind.EQ, "v0", "v1"), c(RelationKind.NEQ, "v0", "v2")), (GivenAssignment("v0", 1),))
        for kwargs in (
            {"excluded_kinds": (RelationKind.EQ, RelationKind.EQ)},
            {"excluded_kinds": ("eq",)},
            {"excluded_constraint_indices": (True,)},
            {"excluded_constraint_indices": (2,)},
            {"excluded_constraint_indices": (0, 0)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(TaskFormatError):
                enumerate_solutions(task, **kwargs)


if __name__ == "__main__":
    unittest.main()
