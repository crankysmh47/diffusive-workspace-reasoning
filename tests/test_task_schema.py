import json
import unittest

from dwr.tasks.schema import (
    Constraint,
    ConstraintTask,
    GivenAssignment,
    RelationKind,
    OracleResult,
    Solution,
    TaskFormatError,
    VerificationReason,
    VerificationResult,
    parse_task_json,
    serialize_model_input,
    serialize_task,
)


BASE = {
    "schema_version": "constraint-task-v1",
    "modulus": 3,
    "variables": ["v0", "v1", "v2"],
    "constraints": [
        {"kind": "neq", "variables": ["v1", "v2"]},
        {"kind": "add_mod", "variables": ["v1", "v0", "v2"]},
    ],
    "givens": [{"variable": "v0", "value": 1}],
}


def raw(data):
    return json.dumps(data, separators=(",", ":"))


class TaskSchemaTests(unittest.TestCase):
    def test_parse_and_canonical_round_trip(self):
        task = parse_task_json(raw(BASE))
        canonical = serialize_task(task)
        self.assertEqual(canonical, '{"constraints":[{"kind":"add_mod","variables":["v0","v1","v2"]},{"kind":"neq","variables":["v1","v2"]}],"givens":[{"value":1,"variable":"v0"}],"modulus":3,"schema_version":"constraint-task-v1","variables":["v0","v1","v2"]}')
        self.assertEqual(serialize_task(parse_task_json(canonical)), canonical)
        with self.assertRaises((AttributeError, TypeError)):
            task.variables[0] = "changed"

    def test_operand_symmetries_normalize_duplicates_but_keep_result_position(self):
        data = dict(BASE)
        data["constraints"] = [
            {"kind": "add_mod", "variables": ["v0", "v1", "v2"]},
            {"kind": "add_mod", "variables": ["v1", "v0", "v2"]},
        ]
        with self.assertRaises(TaskFormatError) as caught:
            parse_task_json(raw(data))
        self.assertEqual(caught.exception.code, "duplicate_constraint")
        data["constraints"] = [
            {"kind": "add_mod", "variables": ["v0", "v1", "v2"]},
            {"kind": "add_mod", "variables": ["v0", "v2", "v1"]},
        ]
        task = parse_task_json(raw(data))
        self.assertEqual(len(task.constraints), 2)
        self.assertEqual(serialize_task(task).count('"kind":"add_mod"'), 2)

    def test_strict_json_errors_and_field_validation(self):
        cases = [
            ('{"x":1,"x":2}', "duplicate_json_key"),
            (raw({**BASE, "surprise": 4}), "unknown_key"),
            (raw({k: v for k, v in BASE.items() if k != "givens"}), "missing_field"),
            (raw({**BASE, "modulus": True}), "unsupported_modulus"),
            (raw({**BASE, "schema_version": "constraint-task-v2"}), "unsupported_schema_version"),
            (raw({**BASE, "variables": ["v0", "v1"]}), "unsupported_size"),
            (raw({**BASE, "variables": ["v0", "v1", "v3"]}), "invalid_variable_sequence"),
            (raw({**BASE, "constraints": [{"kind": "xor", "variables": ["v0", "v1"]}, BASE["constraints"][0]]}), "invalid_kind"),
            (raw({**BASE, "constraints": [{"kind": "eq", "variables": ["v0", "v3"]}, BASE["constraints"][0]]}), "unknown_variable"),
            (raw({**BASE, "constraints": [{"kind": "eq", "variables": ["v0"]}, BASE["constraints"][0]]}), "invalid_arity"),
            (raw({**BASE, "constraints": [{"kind": "eq", "variables": ["v0", "v0"]}, BASE["constraints"][0]]}), "repeated_operand"),
            (raw({**BASE, "constraints": [BASE["constraints"][0]]}), "invalid_constraint_count"),
            (raw({**BASE, "constraints": [{"kind": "add_mod", "variables": ["v0", "v1", "v2"]}, {"kind": "add_mod", "variables": ["v1", "v0", "v2"]}]}), "duplicate_constraint"),
            (raw({**BASE, "givens": []}), "invalid_given_count"),
            (raw({**BASE, "givens": [{"variable": "v0", "value": True}]}), "non_integer_value"),
            (raw({**BASE, "givens": [{"variable": "v0", "value": 3}]}), "value_out_of_domain"),
            (raw({**BASE, "givens": [{"variable": "v0", "value": 0}, {"variable": "v1", "value": 1}]}), "invalid_given_count"),
            (raw({**BASE, "givens": [{"variable": "v0", "value": 0}, {"variable": "v0", "value": 1}]}), "duplicate_given"),
            (raw({**BASE, "givens": [{"variable": "v3", "value": 1}]}), "unknown_variable"),
            (raw({**BASE, "variables": "bad"}), "wrong_container"),
            ("{", "malformed_json"),
            ('{"n":NaN}', "non_finite_constant"),
        ]
        for text, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(TaskFormatError) as caught:
                    parse_task_json(text)
                self.assertEqual(caught.exception.code, code)

    def test_large_json_integer_is_classified_as_out_of_domain(self):
        import sys

        digit_limit = sys.get_int_max_str_digits()
        if digit_limit:
            huge = "9" * (digit_limit + 1)
            text = raw(BASE).replace('"value":1', '"value":' + huge)
            with self.assertRaises(TaskFormatError) as caught:
                parse_task_json(text)
            self.assertEqual(caught.exception.code, "value_out_of_domain")

    def test_direct_records_reject_mutable_or_invalid_nested_values(self):
        with self.assertRaises((TypeError, ValueError)):
            Constraint(RelationKind.EQ, ["v0", "v1"])
        with self.assertRaises((TypeError, ValueError)):
            GivenAssignment("v0", True)
        with self.assertRaises((TypeError, ValueError)):
            ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"), (), (GivenAssignment("v0", 1),))
        task = ConstraintTask(
            "constraint-task-v1", 3, ("v0", "v1", "v2"),
            (Constraint(RelationKind.EQ, ("v0", "v1")), Constraint(RelationKind.NEQ, ("v1", "v2"))),
            (GivenAssignment("v0", 1),),
        )
        self.assertIsInstance(task.variables, tuple)

    def test_model_input_contains_fixed_rules_and_no_answer_annotation(self):
        model_input = serialize_model_input(parse_task_json(raw(BASE)))
        parsed = json.loads(model_input)
        self.assertEqual(parsed["instruction"], "Return one complete satisfying assignment as JSON.")
        self.assertEqual(parsed["rules"]["domain"], [0, 1, 2])
        self.assertEqual(parsed["answer_schema"]["properties"]["assignment"]["required"], ["v0", "v1", "v2"])
        self.assertEqual(list(parsed), sorted(parsed))
        self.assertNotIn("solution", model_input)

    def test_public_result_records_reject_mutable_or_noncanonical_values(self):
        solution = Solution((("v0", 0), ("v1", 1), ("v2", 2)))
        self.assertIsInstance(solution.assignments, tuple)
        with self.assertRaises((TypeError, ValueError)):
            Solution((("v1", 0), ("v0", 1), ("v2", 2)))
        with self.assertRaises((TypeError, ValueError)):
            OracleResult([solution], 27)
        with self.assertRaises((TypeError, ValueError)):
            VerificationResult(True, "valid")
        self.assertEqual(VerificationResult(True, VerificationReason.VALID).reason, VerificationReason.VALID)


if __name__ == "__main__":
    unittest.main()
