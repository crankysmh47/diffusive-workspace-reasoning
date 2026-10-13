"""Contract checks for rank-study admission, routing, and failure preservation."""
from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dwr.tasks import certificate_rank_study as study
from dwr.tasks import certificate_generator as legacy
from dwr.tasks import certificate_rank_generator as rank_generator
from dwr.tasks.schema import Constraint, ConstraintTask, RelationKind
from dwr.tasks.oracle import enumerate_solutions
from dwr.tasks.solver import solve


class StrictConfigTests(unittest.TestCase):
    def test_duplicate_keys_and_nonfinite_values_are_rejected(self):
        for text in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(text=text), self.assertRaises(study.ConfigError):
                study._strict_load(text)

    def test_boolean_cannot_match_registered_integer(self):
        self.assertFalse(study._same_typed(True, 1))
        self.assertFalse(study._same_typed({"x": True}, {"x": 1}))

    def test_unknown_keys_do_not_match_closed_tree(self):
        self.assertFalse(study._same_typed({"mode": "qualification", "surprise": 1},
                                           {"mode": "qualification"}))

    def test_qualification_floor_map_must_be_exact_nonempty_and_strict_true(self):
        expected = {"source", "dev"}
        self.assertTrue(study._exact_true_floor_map({"source": True, "dev": True}, expected))
        for invalid in ({}, {"source": True}, {"source": True, "dev": True, "extra": True},
                        {"source": True, "dev": 1}, {"source": True, "dev": False}):
            with self.subTest(invalid=invalid):
                self.assertFalse(study._exact_true_floor_map(invalid, expected))
        self.assertFalse(study._exact_true_floor_map({"source": True}, set()))

    def test_registered_config_bytes_are_exact(self):
        for mode, digest in study.CONFIG_HASHES.items():
            with self.subTest(mode=mode):
                cfg, actual, path = study.load_config(study._registered_path(mode))
                self.assertEqual(actual, digest)
                self.assertEqual(cfg["mode"], mode)
                self.assertEqual(path, study._registered_path(mode).resolve())


class FixtureRunTests(unittest.TestCase):
    def test_g2_terminal_state_tracks_only_assessed_full_audits(self):
        for status, production, expected, replay_ok in (
                ("PASS_G2_RANK_V1", True, "pass", True),
                ("NO_GO_FULL_AUDIT", True, "fail", True),
                ("PASS_G2_RANK_V1", False, "not_assessed", True),
                ("NOT_ASSESSED_FIXTURE_ONLY", False, "not_assessed", True),
                ("NO_GO_FULL_STAGE", True, "not_assessed", True),
                ("NOT_ASSESSED", True, "not_assessed", True),
                ("PASS_G2_RANK_V1", True, "not_assessed", False)):
            with self.subTest(status=status, production=production):
                self.assertEqual(study._g2_status(status, production, replay_ok), expected)

    def test_ordered_scalar_oracle_solver_and_verifier_agreement(self):
        names = tuple(f"v{i}" for i in range(6))
        pairs = (("v0", "v1"), ("v0", "v2"), ("v0", "v3"), ("v0", "v4"),
                 ("v1", "v2"), ("v1", "v3"), ("v1", "v4"), ("v2", "v3"))
        constraints = tuple(Constraint(RelationKind.EQ, pair) for pair in pairs)
        task = ConstraintTask("constraint-task-v1", 3, names, constraints,
                              (legacy.GivenAssignment("v0", 0), legacy.GivenAssignment("v1", 0)))
        scalar = study._scalar_solutions(task, None, {})
        oracle = enumerate_solutions(task).solutions
        self.assertGreater(len(oracle), 1)
        solved = solve(task)
        self.assertTrue(study._audit_ordered_solutions(task, scalar, oracle, solved, None, None, {})["solver_first_matches"])
        with self.assertRaisesRegex(study.StudyError, "ordered scalar"):
            study._audit_ordered_solutions(task, scalar, tuple(reversed(oracle)), solved, None, None, {})
        with patch.object(study, "verify_answer", return_value=SimpleNamespace(valid=False)):
            with self.assertRaisesRegex(study.StudyError, "verifier rejected"):
                study._audit_ordered_solutions(task, scalar, oracle, solved, None, None, {})

    def test_row_assembly_audits_canned_source_and_protected_pair(self):
        source = legacy.construct(20261011, "source", 0)
        source_row = study._build_row(source, "source", 0, {}, set(), {}, None)
        self.assertTrue(source_row["ordered_solution_audit"]["solver_first_matches"])
        self.assertTrue(source_row["oracle_unique"])

        import random
        pair = rank_generator._construct_pair(1, "development", 16, ("eq", "add_mod"), "A",
                                              ("v4", "v5"), (0, 0, 0, 0, 0, 1), random.Random(11))
        original_metrics = study.old._task_metrics
        def protected_metrics(task, **kwargs):
            values = original_metrics(task, **kwargs)
            return {**values, "development_split": "development_diagnostic_test"}
        with patch.object(study.old, "_task_metrics", side_effect=protected_metrics), \
             patch("dwr.tasks.certificate_fixture_v2._core", side_effect=AssertionError("protected core")):
            protected = study._build_row(pair, "development", 16, {}, set(), {}, None)
        self.assertEqual(protected["state"], "protected_integrity_only")
        self.assertTrue(protected["integrity_checked"])
        self.assertNotIn("core", protected)

    def test_core_population_is_eligible_unique_hard_validation_E_before_envelope(self):
        import random
        outcome = rank_generator._construct_pair(
            1, "development", 16, ("eq", "add_mod"), "A", ("v4", "v5"),
            (0, 0, 0, 0, 0, 1), random.Random(11))
        original_metrics = study.old._task_metrics
        eligible = {"oracle_unique": True, "eligible": True, "sensitive": True,
                    "oracle_solution_count": 1, "interaction": {"fixture": True}}

        def out_of_envelope_validation(task, **kwargs):
            values = original_metrics(task, **kwargs)
            return {**values, "development_split": "development_validation",
                    "prompt_bytes": 100, "first_solution_nodes": 1}

        with patch.object(study.old, "_task_metrics", side_effect=out_of_envelope_validation), \
             patch.object(study.old, "_audit_task", return_value=eligible), \
             patch("dwr.tasks.certificate_fixture_v2._core", return_value={"signature": "canned"}) as core:
            seen, lineage = set(), {}
            unique = study._build_row(outcome, "development", 16, {}, seen, lineage, None)
            duplicate = study._build_row(outcome, "development", 16, {}, seen, lineage, None)
            self.assertFalse(unique["envelope"])
            self.assertIn("core", unique)
            self.assertNotIn("core", duplicate)
            self.assertEqual(core.call_count, 1)

        def protected_validation(task, **kwargs):
            values = original_metrics(task, **kwargs)
            return {**values, "development_split": "development_diagnostic_test"}

        with patch.object(study.old, "_task_metrics", side_effect=protected_validation), \
             patch("dwr.tasks.certificate_fixture_v2._core", side_effect=AssertionError("protected core")):
            protected = study._build_row(outcome, "development", 16, {}, set(), {}, None)
        self.assertEqual(protected["state"], "protected_integrity_only")
        self.assertNotIn("core", protected)

    def test_nonunique_duplicate_content_is_retained_but_not_counted_twice(self):
        names = tuple(f"v{i}" for i in range(6))
        witness = legacy._solution((0, 0, 0, 0, 0, 0))
        alternative = legacy._solution((0, 0, 1, 1, 1, 1))
        pairs = (("v0", "v1"), ("v0", "v2"), ("v0", "v3"), ("v0", "v4"),
                 ("v1", "v2"), ("v1", "v3"), ("v1", "v4"), ("v2", "v3"))
        constraints = tuple(sorted((Constraint(RelationKind.EQ, pair) for pair in pairs),
                                  key=lambda c: (c.kind.value, tuple(sorted(c.variables)))))
        task = ConstraintTask("constraint-task-v1", 3, names, constraints,
                              (legacy.GivenAssignment("v0", 0), legacy.GivenAssignment("v1", 0)))
        # The canned six-variable task leaves v5 free. Its certificate audit is
        # stubbed because this test isolates the runner's duplicate accounting.
        fake = SimpleNamespace(profile=("eq",), cell="A", task=task, witness=witness,
                               givens=task.givens, alternatives=(("eq", alternative),),
                               reserved_constraints=(next(c for c in constraints if c.variables == ("v0", "v2")),), rejection_reason=None,
                               work=legacy.WorkCounters(), branch="ordinary")
        seen, lineage = set(), {}
        nonunique = {"oracle_unique": False, "eligible": False, "sensitive": False,
                     "oracle_solution_count": 3, "interaction": {"fixture": True}}
        with patch.object(study.old, "_audit_task", return_value=nonunique):
            one = study._build_row(fake, "source", 0, {}, seen, lineage, None)
            two = study._build_row(fake, "source", 0, {}, seen, lineage, None)
        self.assertFalse(one["oracle_unique"])
        self.assertFalse(one["eligible"])
        self.assertTrue(one["first_seen_content"])
        self.assertFalse(two["first_seen_content"])
        self.assertFalse(two["oracle_unique"])

    def test_unique_eligible_duplicate_is_only_one_A_quota_row(self):
        outcome = legacy.construct(20261014, "source", 0)
        original_metrics = study.old._task_metrics

        def envelope_metrics(task, **kwargs):
            values = original_metrics(task, **kwargs)
            return {**values, "prompt_bytes": 1200, "first_solution_nodes": 5}

        eligible = {"oracle_unique": True, "eligible": True, "sensitive": True,
                    "oracle_solution_count": 1, "interaction": {"fixture": True}}
        seen, lineage = set(), {}
        with patch.object(study.old, "_task_metrics", side_effect=envelope_metrics), \
             patch.object(study.old, "_audit_task", return_value=eligible):
            one = study._build_row(outcome, "source", 0, {}, seen, lineage, None)
            two = study._build_row(outcome, "source", 0, {}, seen, lineage, None)
        self.assertTrue(study._is_A(one))
        self.assertFalse(study._is_A(two))
        quotas = Counter()
        study._count_A_for_quota(one, "eq", quotas)
        study._count_A_for_quota(two, "eq", quotas)
        self.assertEqual(quotas["eq"], 1)

    def test_full_production_rejects_reusable_arbitrary_output_path(self):
        output = study._registered_path("full").parents[2] / "runs" / "certificate-rank-repeat"
        with self.assertRaisesRegex(ValueError, "full output path"):
            study.run_study(study._registered_path("full"), output)

    def test_private_constructor_failure_cannot_create_production_pass(self):
        config_path = study._registered_path("qualification")
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "study-fixture"
            def fail(config, partition, index):
                raise RuntimeError("canned constructor failure")
            guard = type("Guard", (), {"wall_seconds": 1800, "peak_bytes": 10,
                                        "reset_origin": lambda self: None,
                                        "check": lambda self: None})()
            with self.assertRaises(RuntimeError):
                study.run_study(config_path, out, _guard=guard, _constructor=fail)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(report["g2_status"], "not_assessed")
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["g2_status"], report["g2_status"])
            self.assertTrue(report["fixture_only"])
            for name in study.ARTIFACTS:
                self.assertTrue((out / name).exists(), name)
            raw = json.loads((out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(raw["state"], "fatal_error")
            self.assertEqual(raw["candidate_index"], 0)
            self.assertEqual(raw["phase"], "construction")

    def test_row_assembly_failure_preserves_private_raw_certificate_payload(self):
        config_path = study._registered_path("qualification")
        outcome = legacy.construct(20261014, "source", 0)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "assembly-failure"
            guard = type("Guard", (), {"wall_seconds": 1800, "peak_bytes": 10,
                                        "reset_origin": lambda self: None,
                                        "check": lambda self: None})()
            with patch.object(study, "_construct", return_value=outcome), \
                 patch.object(study, "_build_row", side_effect=RuntimeError("private detail")):
                with self.assertRaisesRegex(RuntimeError, "private detail"):
                    study.run_study(config_path, out, _guard=guard)
            raw = json.loads((out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(raw["phase"], "row_assembly")
            self.assertEqual(raw["error_type"], "RuntimeError")
            self.assertNotIn("error", raw)
            self.assertIn("task", raw["raw_outcome"])
            self.assertIn("witness", raw["raw_outcome"])
            self.assertIn("work", raw["raw_outcome"])

    def test_private_guard_alone_marks_a_run_fixture_only(self):
        config_path = study._registered_path("qualification")
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "guard-fixture"
            guard = type("Guard", (), {"wall_seconds": 1800, "peak_bytes": 10,
                                        "check": lambda self: None})()
            with patch.object(study, "_construct", side_effect=RuntimeError("fixture stop")):
                with self.assertRaises(RuntimeError):
                    study.run_study(config_path, out, _guard=guard)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertIs(report["fixture_only"], True)
            self.assertNotEqual(report["status"], "PASS_QUALIFICATION")

    def test_late_cap_revocation_clears_terminal_pass_and_stale_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            report = {"status": "PASS_G2_RANK_V1", "g2_status": "pass",
                      "artifacts": {"records.jsonl": "stale"},
                      "fixture_only": False, "cpu_seconds": 4.2}
            study._write(output / "manifest.json", {"status": "PASS_G2_RANK_V1", "g2_status": "pass",
                                                       "artifacts": {"x": "stale"}}, None)
            study._revoke_terminal(output, study.CONFIG_HASHES["qualification"], report, TimeoutError())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            saved_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["g2_status"], "not_assessed")
            self.assertEqual(manifest["artifacts"], {})
            self.assertFalse(manifest["artifact_hashes_valid"])
            self.assertEqual(saved_report["status"], "NOT_ASSESSED")
            self.assertEqual(saved_report["g2_status"], "not_assessed")
            self.assertEqual(saved_report["artifacts"], {})
            self.assertEqual(saved_report["cpu_seconds"], 4.2)

    def test_atomic_artifact_write_checks_guard_before_and_after(self):
        class CountingGuard:
            calls = 0
            def check(self): self.calls += 1
        with tempfile.TemporaryDirectory() as temp:
            guard = CountingGuard()
            target = Path(temp) / "record.json"
            study._write(target, {"ok": True}, guard)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"ok": True})
            self.assertEqual(guard.calls, 3)  # pre-write, pre-replace, post-replace

    def test_branch_schedule_is_fixed_and_alternates_only_mixed_add(self):
        self.assertEqual(study._schedule("source", 0), ("eq", "A"))
        self.assertEqual(study._schedule("source", 4), ("neq", "A"))
        self.assertEqual(study._schedule("development", 0), ("eq/neq", "A"))
        self.assertEqual(study._schedule("development", 12), ("eq/neq", "A"))
        self.assertEqual((0 // 12) % 2, 0)
        self.assertEqual((12 // 12) % 2, 1)

    def test_a_projection_keeps_identity_and_input_hash_but_drops_private_payload(self):
        row = {"partition": "development", "split": "development_validation",
               "content_id": "c", "model_input_sha256": "a" * 64,
               "eligible": True, "envelope": True, "witness": {"secret": 1},
               "task": "private task", "alternatives": ["private"],
               "private_interaction": {"secret": 2}}
        projected = study._a_projection(row)
        self.assertEqual(projected["model_input_sha256"], "a" * 64)
        self.assertEqual(projected["content_id"], "c")
        self.assertNotIn("task", projected)
        self.assertNotIn("witness", projected)
        self.assertNotIn("alternatives", projected)
        self.assertNotIn("private_interaction", projected)

    def test_replay_rejects_tampered_raw_and_a_projection_bytes(self):
        digest = "b" * 64
        saved = {"partition": "source", "candidate_index": 0, "content_id": "c",
                 "oracle_unique": True,
                 "first_seen_content": True, "eligible": True, "envelope": True,
                 "model_input_sha256": "a" * 64, "config_sha256": digest}
        rebuilt = {k: v for k, v in saved.items() if k != "config_sha256"}
        raw = study._json_bytes(saved)
        rec = study._a_projection(saved)
        rec["outcome_sha256"] = hashlib.sha256(raw).hexdigest()
        records = study._json_bytes(rec)
        self.assertTrue(study._replay_matches([saved], [rebuilt], raw, records, digest))
        self.assertFalse(study._replay_matches([saved], [rebuilt], raw + b"{}\n", records, digest))
        self.assertFalse(study._replay_matches([saved], [rebuilt], raw, records + b"{}\n", digest))

    def test_null_typed_constructor_outcome_is_retained_as_raw_failure(self):
        profile, cell = legacy._profile_cell_schedule("source")[1]
        work = legacy.WorkCounters(given_position_scans=6, witness_pool_scans=3 ** 6, rng_calls=1)
        outcome = legacy.CertificateOutcome(20261014, "source", 1, profile, cell, (), None, (), (),
                                             work, None, "empty_witness_pool")
        row = study._build_row(outcome, "source", 1, {}, set(), {}, None)
        self.assertIsNone(row["task"])
        self.assertEqual(row["rejection_reason"], "empty_witness_pool")
        self.assertEqual(row["state"], "construction_failed")
        self.assertIsNone(row["content_id"])

    def test_windows_reparse_point_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "missing" / "out"
            with (patch.object(Path, "exists", autospec=True,
                               side_effect=lambda self: self.name == "missing"),
                  patch.object(Path, "is_symlink", autospec=True, return_value=False),
                  patch.object(Path, "lstat", autospec=True,
                               return_value=type("Stat", (), {"st_file_attributes": 0x400})())):
                with self.assertRaises(ValueError):
                    study._safe_path(path)


if __name__ == "__main__":
    unittest.main()
