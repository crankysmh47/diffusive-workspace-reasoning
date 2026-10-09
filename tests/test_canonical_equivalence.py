import itertools
import unittest
from unittest.mock import patch

from dwr.tasks import canonical
from tests.canonical_reference import (
    canonical_content as legacy_content,
    canonical_structure as legacy_structure,
    content_id as legacy_content_id,
    structure_signature as legacy_structure_signature,
    lineage_id as legacy_lineage_id,
    source_split as legacy_source_split,
    development_split as legacy_development_split,
)
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind


def make_task(n, relations, givens):
    variables = tuple(f"v{i}" for i in range(n))
    constraints = tuple(Constraint(kind, tuple(variables[i] for i in operands))
                        for kind, operands in relations)
    return ConstraintTask("constraint-task-v1", 3, variables, constraints,
                          tuple(GivenAssignment(variables[i], value) for i, value in givens))


class CanonicalEquivalenceTests(unittest.TestCase):
    def assert_equivalent(self, task):
        content = legacy_content(task)
        structure = legacy_structure(task)
        self.assertEqual(canonical.canonical_content(task), content)
        self.assertEqual(canonical.canonical_structure(task), structure)
        self.assertEqual(canonical.content_id(task), legacy_content_id(task))
        self.assertEqual(canonical.structure_signature(task), legacy_structure_signature(task))
        self.assertEqual(canonical.lineage_id(task), legacy_lineage_id(task))
        self.assertEqual(canonical.source_split(canonical.content_id(task)),
                         legacy_source_split(legacy_content_id(task)))
        self.assertEqual(canonical.development_split(canonical.content_id(task)),
                         legacy_development_split(legacy_content_id(task)))

    def test_all_three_variable_relation_subsets_and_single_givens(self):
        relations = []
        for kind in (RelationKind.EQ, RelationKind.NEQ):
            relations.extend((kind, pair) for pair in itertools.combinations(range(3), 2))
        relations.extend((RelationKind.ADD_MOD, operands) for operands in
                         ((0, 1, 2), (0, 2, 1), (1, 2, 0)))
        for size in range(2, 9):
            for subset in itertools.combinations(relations, size):
                for variable in range(3):
                    for value in range(3):
                        with self.subTest(size=size, subset=subset, variable=variable, value=value):
                            self.assert_equivalent(make_task(3, subset, ((variable, value),)))

    def test_supported_four_and_five_variable_tasks(self):
        for n in (4, 5):
            pool = []
            pool.extend((RelationKind.EQ, pair) for pair in itertools.combinations(range(n), 2))
            pool.extend((RelationKind.NEQ, pair) for pair in itertools.combinations(range(n), 2))
            pool.extend((RelationKind.ADD_MOD, (a, b, out))
                        for out in range(n) for a, b in itertools.combinations(
                            (i for i in range(n) if i != out), 2))
            selected = (pool[0], pool[3], pool[-1])
            self.assert_equivalent(make_task(n, selected, ((0, 2), (n - 1, 1))))

    def test_all_twelve_fixed_six_variable_source_fixtures(self):
        from tests.test_probe_driver_v4 import _fixture_task
        for profile in (("eq",), ("neq",), ("add_mod",)):
            for cell in "ABCD":
                with self.subTest(profile=profile, cell=cell):
                    self.assert_equivalent(_fixture_task("source", profile, cell))

    def test_orderings_operand_symmetry_and_add_output_are_preserved(self):
        base = make_task(4, ((RelationKind.EQ, (0, 1)),
                             (RelationKind.NEQ, (1, 2)),
                             (RelationKind.ADD_MOD, (0, 2, 3))),
                         ((0, 1), (3, 2)))
        reordered = make_task(4, ((RelationKind.ADD_MOD, (2, 0, 3)),
                                  (RelationKind.NEQ, (2, 1)),
                                  (RelationKind.EQ, (1, 0))), ((3, 2), (0, 1)))
        self.assert_equivalent(base)
        self.assertEqual(canonical.canonical_content(base), canonical.canonical_content(reordered))
        self.assertEqual(canonical.canonical_structure(base), canonical.canonical_structure(reordered))
        changed_output = make_task(4, ((RelationKind.EQ, (0, 1)),
                                       (RelationKind.NEQ, (1, 2)),
                                       (RelationKind.ADD_MOD, (0, 3, 2))),
                                  ((0, 1), (3, 2)))
        self.assertNotEqual(canonical.canonical_content(base), canonical.canonical_content(changed_output))

    def test_serializes_only_winning_rank_once_per_public_call(self):
        task = make_task(4, ((RelationKind.EQ, (0, 1)), (RelationKind.NEQ, (1, 2))), ((0, 0),))
        for function in (canonical.canonical_content, canonical.canonical_structure):
            with patch.object(canonical, "_payload", wraps=canonical._payload) as payload:
                function(task)
                self.assertEqual(payload.call_count, 1)


if __name__ == "__main__":
    unittest.main()

