import hashlib
import itertools
import unittest

from dwr.tasks.canonical import (
    canonical_content, canonical_structure, content_id, structure_signature,
    lineage_id, source_split, development_split,
)
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind


def task(constraints=None, givens=None):
    return ConstraintTask(
        "constraint-task-v1", 3, ("v0", "v1", "v2"),
        tuple(constraints or (Constraint(RelationKind.EQ, ("v0", "v1")),
                             Constraint(RelationKind.NEQ, ("v1", "v2")))),
        tuple(givens or (GivenAssignment("v0", 1),)),
    )


class CanonicalTests(unittest.TestCase):
    def test_hand_encoding_hashes_and_domains(self):
        example = task()
        expected = '{"constraints":[{"kind":"eq","variables":["v0","v1"]},{"kind":"neq","variables":["v0","v2"]}],"givens":[{"value":1,"variable":"v1"}],"modulus":3,"schema_version":"constraint-task-v1","variables":["v0","v1","v2"]}'
        self.assertEqual(canonical_content(example), expected)
        self.assertEqual(content_id(example), hashlib.sha256(b"dwr-content-v1\0" + expected.encode()).hexdigest())
        structure = '{"anchors":["v0"],"constraints":[{"kind":"eq","variables":["v0","v1"]},{"kind":"neq","variables":["v1","v2"]}],"modulus":3,"variables":["v0","v1","v2"],"version":"constraint-structure-v1"}'
        self.assertEqual(canonical_structure(example), structure)
        self.assertEqual(structure_signature(example), hashlib.sha256(b"dwr-structure-v1\0" + structure.encode()).hexdigest())
        self.assertEqual(lineage_id(example), hashlib.sha256(b"dwr-lineage-v1\0" + content_id(example).encode("ascii")).hexdigest())

    def test_renamings_order_and_operand_symmetry(self):
        base = task((Constraint(RelationKind.ADD_MOD, ("v0", "v1", "v2")), Constraint(RelationKind.NEQ, ("v0", "v1"))), (GivenAssignment("v2", 2),))
        expected = canonical_content(base)
        for perm in itertools.permutations(range(3)):
            names = {f"v{i}": f"v{perm[i]}" for i in range(3)}
            constraints = []
            for c in reversed(base.constraints):
                operands = tuple(names[v] for v in c.variables)
                if c.kind is RelationKind.ADD_MOD:
                    operands = (operands[1], operands[0], operands[2])
                constraints.append(Constraint(c.kind, operands))
            renamed = ConstraintTask("constraint-task-v1", 3, tuple(f"v{i}" for i in range(3)), tuple(constraints), (GivenAssignment(names["v2"], 2),))
            self.assertEqual(canonical_content(renamed), expected)
            self.assertEqual(canonical_structure(renamed), canonical_structure(base))
        changed_result = task((Constraint(RelationKind.ADD_MOD, ("v0", "v2", "v1")), Constraint(RelationKind.NEQ, ("v0", "v1"))), (GivenAssignment("v2", 2),))
        self.assertNotEqual(canonical_content(base), canonical_content(changed_result))

    def test_modular_value_relabeling_is_not_a_canonical_symmetry(self):
        modular = task((Constraint(RelationKind.ADD_MOD, ("v0", "v1", "v2")), Constraint(RelationKind.NEQ, ("v0", "v2"))), (GivenAssignment("v0", 0),))
        relabeled = task((Constraint(RelationKind.ADD_MOD, ("v0", "v1", "v2")), Constraint(RelationKind.NEQ, ("v0", "v2"))), (GivenAssignment("v0", 1),))
        self.assertNotEqual(canonical_content(modular), canonical_content(relabeled))

    def test_structure_omits_clue_values_but_keeps_anchor_set(self):
        a = task(givens=(GivenAssignment("v0", 1),))
        b = task(givens=(GivenAssignment("v0", 2),))
        c = task(givens=(GivenAssignment("v1", 2),))
        self.assertEqual(canonical_structure(a), canonical_structure(b))
        self.assertNotEqual(content_id(a), content_id(b))
        self.assertNotEqual(canonical_structure(a), canonical_structure(c))

    def test_split_hashes_use_contract_domains(self):
        value = "abc"
        bucket = int.from_bytes(hashlib.sha256(b"dwr-id-split-v1\0abc").digest()[:8], "big") % 100
        self.assertEqual(source_split(value), "source_train" if bucket < 80 else "source_validation" if bucket < 90 else "source_test")
        dev = int.from_bytes(hashlib.sha256(b"dwr-dev-split-v1\0abc").digest()[:8], "big") % 100
        self.assertEqual(development_split(value), "development_validation" if dev < 80 else "development_diagnostic_test")


if __name__ == "__main__":
    unittest.main()
