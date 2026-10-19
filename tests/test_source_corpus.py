import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dwr.tasks import canonical, certificate_rank_study, oracle, schema, solver
from dwr.tasks.verifier import verify_answer

from dwr.experiments.source_corpus import SourceExportError, export_source_records


def _json_line(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                        separators=(",", ":")) + "\n").encode("utf-8")


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _task(profile):
    if profile == "eq":
        constraints = (
            schema.Constraint(schema.RelationKind.EQ, ("v0", "v1")),
            schema.Constraint(schema.RelationKind.EQ, ("v1", "v2")),
        )
        givens = (schema.GivenAssignment("v0", 1),)
    else:
        constraints = (
            schema.Constraint(schema.RelationKind.ADD_MOD, ("v0", "v1", "v2")),
            schema.Constraint(schema.RelationKind.ADD_MOD, ("v1", "v2", "v0")),
        )
        givens = (schema.GivenAssignment("v0", 0),)
    return schema.ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"), constraints, givens)


def _raw_row(task, split, partition, candidate_index, profile):
    solved = solver.solve(task)
    projection = {
        "partition": partition,
        "candidate_index": candidate_index,
        "profile": profile,
        "cell": "A",
        "branch": "ordinary",
        "split": split,
        "content_id": canonical.content_id(task),
        "lineage_id": canonical.lineage_id(task),
        "structure_signature": canonical.structure_signature(task),
        "oracle_unique": True,
        "eligible": True,
        "sensitive": True,
        "envelope": True,
        "values": [0, 1, 2],
        "given_count": len(task.givens),
        "prompt_bytes": len(schema.serialize_model_input(task).encode("utf-8")),
        "first_solution_nodes": solved.stats.first_solution_nodes,
        "model_input_sha256": _hash(schema.serialize_model_input(task).encode("utf-8")),
        "gac": {},
        "core": {},
        "config_sha256": "",
        "fine_stratum": "A2",
        "first_seen_content": True,
        "duplicate_content": False,
        "state": "first_seen",
        "task": schema.serialize_task(task),
        "witness": {"assignment": dict(solved.solution.assignments)} if solved.solution else {"poison": True},
    }
    return projection


class SourceCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.output_path = self.root / "source.jsonl"
        self.config_path = self.root / "generation.json"
        self.config = {"fixture": "source-corpus-test", "version": 1,
                       "source_profiles": ["eq", "neq", "add_mod"]}
        self.config_bytes = json.dumps(self.config, ensure_ascii=False, indent=2).encode("utf-8")
        self.config_path.write_bytes(self.config_bytes)
        self.expected_config_sha = _hash(self.config_bytes)
        self.source_commit = "a" * 40
        self.expected_counts = {"source_train": 1, "source_validation": 1}
        self.tasks = (_task("eq"), _task("add_mod"))
        self.rows = [
            _raw_row(self.tasks[0], "source_train", "source", 0, "eq"),
            _raw_row(self.tasks[1], "source_validation", "source", 1, "add_mod"),
        ]
        for row in self.rows:
            row["config_sha256"] = self.expected_config_sha
        self._write_inputs()

    def tearDown(self):
        self.temp.cleanup()

    def _write_inputs(self, *, records=None, outcomes=None, initial=None, terminal=None,
                      report=None, replay=None):
        records = self.rows if records is None else records
        outcomes = self.rows if outcomes is None else outcomes
        raw_lines = [_json_line(row) for row in outcomes]
        outcome_by_key = {
            (row.get("partition"), row.get("candidate_index")): line
            for row, line in zip(outcomes, raw_lines)
        }
        accepted_records = []
        for row in records:
            raw_line = outcome_by_key.get((row.get("partition"), row.get("candidate_index")), b"")
            projection = certificate_rank_study._a_projection(row)
            projection["outcome_sha256"] = _hash(raw_line)
            accepted_records.append(_json_line(projection))

        initial = initial or {
            "config_sha256": self.expected_config_sha,
            "resolved_config": self.config,
            "source": {"git_commit": self.source_commit, "git_clean": True},
            "status": "running",
        }
        report = report or {
            "config_sha256": self.expected_config_sha,
            "fixture_only": False,
            "g2_status": "pass",
            "status": "PASS_G2_FIVE_STRATUM_V1",
        }
        replay = replay or {"exact_indices": True, "status": "PASS_REPLAY"}
        entries = {
            "manifest.initial.json": _json_line(initial),
            "outcomes.jsonl": b"".join(raw_lines),
            "records.jsonl": b"".join(accepted_records),
            "replay.json": _json_line(replay),
            "report.json": _json_line(report),
        }
        if terminal is None:
            terminal = {
                "artifacts": {name: _hash(data) for name, data in entries.items()},
                "config_sha256": self.expected_config_sha,
                "g2_status": "pass",
                "source": {"git_commit": self.source_commit, "git_clean": True},
                "status": "PASS_G2_FIVE_STRATUM_V1",
            }
        entries["manifest.json"] = _json_line(terminal)
        for name, data in entries.items():
            (self.input_dir / name).write_bytes(data)
        self.expected_artifacts = {name: _hash(data) for name, data in entries.items()}

    def _export(self, *, output_path=None, check=None):
        return export_source_records(
            self.input_dir,
            output_path or self.output_path,
            expected_artifacts=self.expected_artifacts,
            generation_config_path=self.config_path,
            expected_config_sha256=self.expected_config_sha,
            expected_source_commit=self.source_commit,
            expected_counts=self.expected_counts,
            check=check or (lambda: None),
        )

    def _replace_artifact(self, name, data):
        (self.input_dir / name).write_bytes(data)
        terminal_path = self.input_dir / "manifest.json"
        terminal = json.loads(terminal_path.read_text("utf-8"))
        if name != "manifest.json":
            terminal["artifacts"][name] = _hash(data)
            terminal_path.write_bytes(_json_line(terminal))
        self.expected_artifacts = {
            artifact.name: _hash(artifact.read_bytes())
            for artifact in self.input_dir.iterdir()
        }

    def test_exports_only_allowed_source_rows_in_projection_order_with_independent_gold(self):
        # Raw ledger order is incidental. Output order follows accepted projection order.
        self._write_inputs(outcomes=list(reversed(self.rows)))
        result = self._export()
        exported = [json.loads(line) for line in self.output_path.read_text("utf-8").splitlines()]
        self.assertEqual([row["example_id"] for row in exported],
                         [canonical.content_id(task) for task in self.tasks])
        self.assertEqual([row["split"] for row in exported], ["source_train", "source_validation"])
        self.assertEqual(result["exported_counts"], self.expected_counts)
        for task, row in zip(self.tasks, exported):
            solved = solver.solve(task)
            independently = oracle.enumerate_solutions(task)
            self.assertEqual(solved.stats.uniqueness_solution_count_capped, 1)
            self.assertEqual(len(independently.solutions), 1)
            self.assertEqual(dict(solved.solution.assignments), dict(independently.solutions[0].assignments))
            self.assertTrue(verify_answer(task, json.dumps(row["solution"], sort_keys=True)).valid)
            self.assertEqual(row["task"], schema.serialize_task(task))

    def test_forbidden_rows_and_null_split_construction_failure_never_parse_private_payloads(self):
        forbidden = {
            "partition": "final", "candidate_index": 40, "split": "final_test",
            "task": "{broken", "witness": {"poison": True}, "values": ["poison"],
        }
        source_test = {
            "partition": "source", "candidate_index": 39, "split": "source_test",
            "task": "{broken", "witness": {"poison": True}, "values": ["poison"],
        }
        development = {
            "partition": "development", "candidate_index": 41,
            "split": "development_validation", "task": "{broken",
            "witness": {"poison": True}, "values": ["poison"],
        }
        failed = {
            "partition": "source", "candidate_index": 42, "split": None,
            "state": "construction_failed", "task": "{broken",
            "witness": {"poison": True}, "values": ["poison"],
        }
        self._write_inputs(outcomes=[*self.rows, source_test, development, forbidden, failed])
        parse_task_impl = schema.parse_task_json

        def reject_poisoned_task(text):
            if text == "{broken":
                raise AssertionError("forbidden task parsed")
            return parse_task_impl(text)

        with patch("dwr.tasks.schema.parse_task_json", side_effect=reject_poisoned_task) as parse_task, \
                patch("dwr.experiments.source_corpus.solver.solve", wraps=solver.solve) as solve_task, \
                patch("dwr.experiments.source_corpus.oracle.enumerate_solutions",
                      wraps=oracle.enumerate_solutions) as enumerate_task:
            result = self._export()
        self.assertEqual(parse_task.call_count, 2)
        self.assertEqual(solve_task.call_count, 2)
        self.assertEqual(enumerate_task.call_count, 2)
        self.assertEqual(result["skipped_raw_rows"], 4)

    def test_rejects_null_split_for_any_non_construction_failure_state(self):
        bad = {"partition": "source", "candidate_index": 7, "split": None,
               "state": "first_seen", "task": "bad"}
        self._write_inputs(outcomes=[*self.rows, bad])
        with self.assertRaises(ValueError):
            self._export()

    def test_rejects_nonunique_source_task_even_when_outer_hashes_are_rebound(self):
        task = schema.ConstraintTask(
            "constraint-task-v1", 3, ("v0", "v1", "v2", "v3"),
            (schema.Constraint(schema.RelationKind.EQ, ("v0", "v1")),
             schema.Constraint(schema.RelationKind.EQ, ("v1", "v2")),
             schema.Constraint(schema.RelationKind.EQ, ("v2", "v0"))),
            (schema.GivenAssignment("v0", 1),),
        )
        row = _raw_row(task, "source_train", "source", 0, "eq")
        row["task"] = schema.serialize_task(task)
        row["witness"] = {"poison": True}
        row["model_input_sha256"] = _hash(schema.serialize_model_input(task).encode("utf-8"))
        row["content_id"] = canonical.content_id(task)
        row["lineage_id"] = canonical.lineage_id(task)
        row["structure_signature"] = canonical.structure_signature(task)
        row2 = self.rows[1]
        self.expected_counts = {"source_train": 1, "source_validation": 1}
        self._write_inputs(records=[row, row2], outcomes=[row, row2])
        with self.assertRaisesRegex(SourceExportError,
                                    "reference solver did not find exactly one source solution"):
            self._export()

    def test_rejects_unsafe_windows_output_leaf_names_before_creation(self):
        for leaf in ("alternate.jsonl:export", "CON", "nul.jsonl", "COM1.txt",
                     "LPT9", "report.jsonl.", "report.jsonl "):
            with self.subTest(leaf=leaf):
                output = self.root / leaf
                with self.assertRaisesRegex(SourceExportError, "unsafe Windows output filename"):
                    self._export(output_path=output)
                self.assertFalse(output.exists())
        self.assertFalse((self.root / "alternate.jsonl").exists())

    def test_rejects_duplicate_raw_routing_key_and_leaves_partial_output(self):
        self._write_inputs(outcomes=[*self.rows, self.rows[0]])
        with self.assertRaises(ValueError):
            self._export()
        self.assertTrue(self.output_path.exists())

    def test_rejects_hash_status_and_config_binding_mismatches(self):
        broken = dict(self.expected_artifacts)
        broken["outcomes.jsonl"] = "0" * 64
        with self.assertRaises(ValueError):
            export_source_records(
                self.input_dir, self.output_path, expected_artifacts=broken,
                generation_config_path=self.config_path,
                expected_config_sha256=self.expected_config_sha,
                expected_source_commit=self.source_commit,
                expected_counts=self.expected_counts, check=lambda: None,
            )
        self.output_path = self.root / "second.jsonl"
        self._write_inputs(report={"config_sha256": self.expected_config_sha,
                                  "fixture_only": False, "g2_status": "fail",
                                  "status": "PASS_G2_FIVE_STRATUM_V1"})
        with self.assertRaises(ValueError):
            self._export()

    def test_rejects_typed_config_or_source_commit_mismatch(self):
        initial = json.loads((self.input_dir / "manifest.initial.json").read_text("utf-8"))
        initial["resolved_config"]["version"] = True
        self._replace_artifact("manifest.initial.json", _json_line(initial))
        with self.assertRaises(ValueError):
            self._export()
        self.output_path = self.root / "typed-config-second.jsonl"
        self._write_inputs()
        terminal = json.loads((self.input_dir / "manifest.json").read_text("utf-8"))
        terminal["source"]["git_commit"] = "b" * 40
        self._replace_artifact("manifest.json", _json_line(terminal))
        with self.assertRaises(ValueError):
            self._export()

    def test_rejects_raw_projection_and_outcome_line_hash_mismatches(self):
        changed = [dict(row) for row in self.rows]
        changed[0]["values"] = [2, 2, 2]
        self._write_inputs(records=self.rows, outcomes=changed)
        with self.assertRaises(ValueError):
            self._export()
        self.output_path = self.root / "raw-hash-second.jsonl"
        self._write_inputs()
        record_lines = (self.input_dir / "records.jsonl").read_bytes().splitlines()
        record = json.loads(record_lines[0])
        record["outcome_sha256"] = "f" * 64
        record_lines[0] = _json_line(record).rstrip(b"\n")
        self._replace_artifact("records.jsonl", b"\n".join(record_lines) + b"\n")
        with self.assertRaises(ValueError):
            self._export()

    def test_rejects_model_input_identity_and_profile_mismatches(self):
        changed = [dict(row) for row in self.rows]
        changed[0]["model_input_sha256"] = "e" * 64
        self._write_inputs(records=changed, outcomes=changed)
        with self.assertRaises(ValueError):
            self._export()
        self.output_path = self.root / "profile-second.jsonl"
        changed[0]["model_input_sha256"] = self.rows[0]["model_input_sha256"]
        changed[0]["profile"] = "neq"
        self._write_inputs(records=changed, outcomes=changed)
        with self.assertRaises(ValueError):
            self._export()

    def test_rejects_canonical_identity_mismatches(self):
        for field in ("content_id", "lineage_id", "structure_signature"):
            with self.subTest(field=field):
                self.output_path = self.root / f"{field}.jsonl"
                changed = [dict(row) for row in self.rows]
                changed[0][field] = "d" * 64
                self._write_inputs(records=changed, outcomes=changed)
                with self.assertRaises(ValueError):
                    self._export()

    def test_rejects_duplicate_accepted_projection_keys_and_content_ids(self):
        self._write_inputs(records=[self.rows[0], self.rows[0], self.rows[1]])
        with self.assertRaises(ValueError):
            self._export()
        self.output_path = self.root / "duplicate-id.jsonl"
        changed = [dict(row) for row in self.rows]
        changed[1]["content_id"] = changed[0]["content_id"]
        self._write_inputs(records=changed, outcomes=changed)
        with self.assertRaises(ValueError):
            self._export()

    def test_rejects_missing_allowed_row_and_unsatisfiable_source_gold(self):
        self._write_inputs(outcomes=[self.rows[0]])
        with self.assertRaises(ValueError):
            self._export()
        self.output_path = self.root / "unsat.jsonl"
        impossible = schema.ConstraintTask(
            "constraint-task-v1", 3, ("v0", "v1", "v2", "v3"),
            (schema.Constraint(schema.RelationKind.EQ, ("v0", "v1")),
             schema.Constraint(schema.RelationKind.EQ, ("v1", "v3"))),
            (schema.GivenAssignment("v0", 0), schema.GivenAssignment("v3", 1)),
        )
        row = _raw_row(impossible, "source_train", "source", 0, "eq")
        self._write_inputs(records=[row, self.rows[1]], outcomes=[row, self.rows[1]])
        with self.assertRaises(ValueError):
            self._export()

    def test_rejects_duplicate_json_keys_and_unknown_routing(self):
        bad_jsonl = b'{"partition":"source","partition":"source","candidate_index":0,"split":"source_train"}\n'
        self._replace_artifact("outcomes.jsonl", bad_jsonl)
        with self.assertRaises(ValueError):
            self._export()
        self.output_path = self.root / "route.jsonl"
        bad_route = {"partition": "source", "candidate_index": 5, "split": "not-a-split"}
        self._write_inputs(outcomes=[*self.rows, bad_route])
        with self.assertRaises(ValueError):
            self._export()

    def test_rejects_existing_or_input_overlapping_output(self):
        self.output_path.write_text("occupied", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._export()
        self.output_path = self.input_dir / "export.jsonl"
        with self.assertRaises(ValueError):
            self._export()
        self.output_path = self.input_dir / ".." / "traversal.jsonl"
        with self.assertRaises(ValueError):
            self._export()

    def test_detects_inputs_mutated_during_conversion(self):
        calls = 0

        def change_after_start():
            nonlocal calls
            calls += 1
            if calls == 30:
                with (self.input_dir / "outcomes.jsonl").open("ab") as stream:
                    stream.write(b"{}\n")

        with self.assertRaises(ValueError):
            self._export(check=change_after_start)


if __name__ == "__main__":
    unittest.main()
