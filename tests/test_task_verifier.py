import itertools
import json
import sys
import unittest

from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, VerificationReason
from dwr.tasks.verifier import verify_answer


def task(constraints, givens=()):
    if not givens:
        givens = (GivenAssignment("v0", 0),)
    return ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"), tuple(constraints), tuple(givens))


def answer(**values):
    return json.dumps({"assignment": values}, separators=(",", ":"))


class VerifierTests(unittest.TestCase):
    def test_truth_tables_for_each_primitive(self):
        samples = list(itertools.product(range(3), repeat=3))
        eq = task((Constraint(RelationKind.EQ, ("v0", "v1")), Constraint(RelationKind.NEQ, ("v0", "v2"))))
        neq = task((Constraint(RelationKind.NEQ, ("v0", "v1")), Constraint(RelationKind.NEQ, ("v0", "v2"))))
        add = task((Constraint(RelationKind.ADD_MOD, ("v0", "v1", "v2")), Constraint(RelationKind.NEQ, ("v0", "v2"))))
        for values in samples:
            mapping = dict(zip(("v0", "v1", "v2"), values))
            self.assertEqual(verify_answer(eq, answer(**mapping)).valid, values[0] == 0 and values[0] == values[1] and values[0] != values[2])
            self.assertEqual(verify_answer(neq, answer(**mapping)).valid, values[0] == 0 and values[0] != values[1] and values[0] != values[2])
            self.assertEqual(verify_answer(add, answer(**mapping)).valid, values[0] == 0 and (values[0] + values[1]) % 3 == values[2] and values[0] != values[2])

    def test_reason_codes_and_precedence(self):
        base = task((Constraint(RelationKind.EQ, ("v0", "v1")), Constraint(RelationKind.NEQ, ("v1", "v2"))), (GivenAssignment("v0", 1),))
        examples = [
            ("{", "malformed_json"),
            ('{"x":NaN}', "non_finite_constant"),
            ('{"x":1,"x":2}', "duplicate_json_key"),
            ("[]", "wrong_container"),
            ('{"other":0}', "unknown_key"),
            ("{}", "missing_assignment"),
            ('{"assignment":[]}', "wrong_container"),
            ('{"assignment":{"v9":0,"v0":1,"v1":1,"v2":0}}', "unknown_variable"),
            ('{"assignment":{"v0":1}}', "missing_variable"),
            ('{"assignment":{"v0":1,"v1":"1","v2":0}}', "non_integer_value"),
            ('{"assignment":{"v0":1,"v1":4,"v2":0}}', "value_out_of_domain"),
            (answer(v0=0, v1=0, v2=1), "given_mismatch"),
            (answer(v0=1, v1=1, v2=1), "constraint_violation"),
            (answer(v0=1, v1=1, v2=0), "valid"),
        ]
        for text, expected in examples:
            with self.subTest(expected=expected, text=text):
                result = verify_answer(base, text)
                self.assertEqual(result.reason, VerificationReason(expected))
                self.assertEqual(result.valid, expected == "valid")

    def test_global_error_precedence_is_independent_of_key_order(self):
        cases = [
            ('{"a":1,"a":2,', "malformed_json"),
            ('{"a":1,"a":2,"b":NaN}', "non_finite_constant"),
            ('{"b":NaN,"a":1,"a":2}', "non_finite_constant"),
            ('{"assignment":{"v9":0,"v0":1,"v1":1}}', "unknown_variable"),
            ('{"assignment":{"v0":1,"v1":1,"v2":1,"v9":0}}', "unknown_variable"),
            ('{"assignment":{"v0":3,"v1":"1","v2":0}}', "non_integer_value"),
            ('{"assignment":{"v0":"1","v1":3,"v2":0}}', "non_integer_value"),
        ]
        empty = task((Constraint(RelationKind.EQ, ("v0", "v1")), Constraint(RelationKind.NEQ, ("v1", "v2"))))
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(verify_answer(empty, text).reason, VerificationReason(expected))

    def test_duplicate_nonfinite_and_huge_integer_precedence(self):
        empty = task((Constraint(RelationKind.EQ, ("v0", "v1")), Constraint(RelationKind.NEQ, ("v1", "v2"))))
        for text in ('{"x":NaN,"x":0}', '{"x":0,"x":NaN}'):
            with self.subTest(text=text):
                self.assertEqual(verify_answer(empty, text).reason, VerificationReason.NON_FINITE_CONSTANT)
        self.assertEqual(verify_answer(empty, '{"x":NaN,"x":0,').reason, VerificationReason.MALFORMED_JSON)
        digit_limit = sys.get_int_max_str_digits()
        if digit_limit:
            huge = "9" * (digit_limit + 1)
            text = '{"assignment":{"v0":' + huge + ',"v1":0,"v2":0}}'
            self.assertEqual(verify_answer(empty, text).reason, VerificationReason.VALUE_OUT_OF_DOMAIN)


if __name__ == "__main__":
    unittest.main()
