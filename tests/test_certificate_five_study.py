"""Canned runner-contract checks for the separately registered five-stratum study."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dwr.tasks import certificate_rank_study as study
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind


ROOT = Path(__file__).resolve().parents[1]
QUAL = ROOT / "configs/tasks/certificate_five_stratum_qualification_v1.json"
FULL = ROOT / "configs/tasks/certificate_five_stratum_full_v1.json"


class FiveConfigAdmissionTests(unittest.TestCase):
    def test_new_configs_are_distinct_and_legacy_registry_is_untouched(self):
        self.assertEqual(study.CONFIG_HASHES["qualification"], "8fa592fe2bedc012a1a10fb5ecd74bbd871df2f0cedcf5026562f5b518c2d601")
        self.assertEqual(study.CONFIG_HASHES["full"], "c01056ece9e4d63981301b2cff2c224df952499a54d8b746ac1d73b2df15be6f")
        for path, expected in ((QUAL, "d66ad33da9111287223e7e94ab849e8b710692db32f71f94410f305ce1cf1002"),
                               (FULL, "3aff6fcaba6143a26ad6d8a86cc83f9782ff63d689912f5fb04ffc4e9cc81afa")):
            config, digest, resolved = study.load_config(path)
            self.assertEqual(digest, expected)
            self.assertEqual(resolved, path.resolve())
            self.assertEqual(config["support"]["scheme"], "five-fixed-strata-v1")

    def test_new_full_links_run_path_and_rejects_legacy_qualification(self):
        config, _, _ = study.load_config(FULL)
        self.assertEqual(config["required_qualification"]["run_path"],
                         "runs/certificate-five-stratum-qualification-20261014-v1")
        source = {"producer_sha256": {}}
        with patch.object(study, "_load_qualification_receipt", return_value={"status": "PASS_QUALIFICATION"}) as receipt:
            self.assertEqual(study._verify_qualification_link(config, source, None)["status"], "PASS_QUALIFICATION")
            receipt.assert_called_once()
        legacy_link = {**config, "required_qualification": {
            "config_sha256": study.CONFIG_HASHES["qualification"],
            "run_path": "runs/certificate-rank-qualification-20261012-v1"}}
        with self.assertRaisesRegex(study.StudyError, "config binding"):
            study._verify_qualification_link(legacy_link, source, None)

    def test_receipt_rejects_missing_historical_source_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            qout = Path(temp) / "qualification"
            qout.mkdir()
            qconfig, qhash, _ = study.load_config(QUAL)
            source = {"git_clean": True, "producer_sha256": {"dwr/tasks/x.py": "a" * 64}}
            expected_counts = {"source": 3600, "development": 4800, "final": 0}
            true_floors = lambda keys: {key: True for key in keys}
            source_floors = true_floors(study.audit.source_gate([], 0, qconfig)[1]["floors"])
            development_floors = true_floors(study.audit.development_gate([], [], qconfig)[1]["floors"])
            initial = {"config_sha256": qhash, "resolved_config": qconfig, "source": source}
            report = {"status": "PASS_QUALIFICATION", "fixture_only": False,
                      "raw_attempts": expected_counts,
                      "source_gate": {"floors": source_floors},
                      "development_gate": {"floors": development_floors}}
            replay = {"status": "PASS_REPLAY", "exact_indices": True, "attempts_rederived": 8400}
            manifest = {"status": "PASS_QUALIFICATION", "config_sha256": qhash, "source": source}
            for name, value in (("manifest.initial.json", initial), ("report.json", report),
                                ("replay.json", replay), ("manifest.json", manifest)):
                (qout / name).write_bytes(study._json_bytes(value))
            with self.assertRaisesRegex(study.StudyError, "source identity"):
                study._load_qualification_receipt(qout, {"config_sha256": qhash}, source, None)

    def test_complete_receipt_admits_protected_first_seen_then_real_full_runner_stops_at_source(self):
        qconfig, qhash, _ = study.load_config(QUAL)
        full_config, full_hash, _ = study.load_config(FULL)
        source_blob = b"canned producer bytes"
        producer_map = {"dwr/tasks/canned_producer.py": hashlib.sha256(source_blob).hexdigest()}
        historical_source = {"git_commit": "a" * 40, "git_clean": True,
                             "producer_sha256": producer_map}
        current_source = {"git_commit": "b" * 40, "git_clean": True,
                          "producer_sha256": producer_map}
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            qconfig_path = repo / "configs/tasks/certificate_five_stratum_qualification_v1.json"
            full_config_path = repo / "configs/tasks/certificate_five_stratum_full_v1.json"
            qconfig_path.parent.mkdir(parents=True)
            qconfig_path.write_bytes(QUAL.read_bytes())
            full_config_path.write_bytes(FULL.read_bytes())
            run_root = repo / "runs"
            qout = run_root / "certificate-five-stratum-qualification-20261014-v1"
            qout.mkdir(parents=True)

            outcome = _hand_outcome()
            protected_metrics = {"content_id": "protected-content", "lineage_id": "protected-lineage",
                                 "structure_signature": "protected-signature", "prompt_bytes": 1200,
                                 "first_solution_nodes": 9, "source_split": "source_test",
                                 "development_split": "development_diagnostic_test"}
            semantic = {"oracle_unique": True, "eligible": True, "sensitive": True,
                        "oracle_solution_count": 1, "interaction": {}}
            semantic_patches = [
                patch.object(study.old, "_task_metrics", return_value=protected_metrics),
                patch.object(study, "_scalar_solutions", return_value=()),
                patch.object(study.old, "_audit_task", return_value=semantic),
                patch.object(study, "enumerate_solutions", return_value=SimpleNamespace(solutions=())),
                patch.object(study, "solve", return_value=SimpleNamespace(
                    solution=None, stats=SimpleNamespace(uniqueness_solution_count_capped=0))),
                patch.object(study, "propagate", return_value=SimpleNamespace(
                    stalled=False, solved=False, contradiction=False)),
                patch.object(study.old, "_audit_local_gac", return_value=True),
            ]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for active in semantic_patches:
                    stack.enter_context(active)
                seen, lineage = set(), {}
                protected_first = study._build_row(outcome, "source", 1, qconfig, seen, lineage, None)
                protected_duplicate = study._build_row(outcome, "source", 2, qconfig, seen, lineage, None)

            rows = []
            for part, start, stop in (("source", 0, 3600), ("development", 0, 4800)):
                for index in range(start, stop):
                    if part == "source" and index == 1:
                        row = _public_row(protected_first)
                    elif part == "source" and index == 2:
                        row = _public_row(protected_duplicate)
                    else:
                        profile, cell = study._schedule(part, index)
                        row = _receipt_row(part, index, profile, cell, qconfig,
                                           accepted=(part == "source" and index == 0))
                    row["config_sha256"] = qhash
                    rows.append(row)
            outcome_bytes = b"".join(study._json_bytes(row) for row in rows)
            outcome_lines = outcome_bytes.splitlines(keepends=True)
            record_bytes = study._five_qualification_projection_bytes(outcome_lines, qconfig, qhash)
            (qout / "outcomes.jsonl").write_bytes(outcome_bytes)
            (qout / "records.jsonl").write_bytes(record_bytes)
            replay = {"status": "PASS_REPLAY", "exact_indices": True, "attempts_rederived": 8400}
            (qout / "replay.json").write_bytes(study._json_bytes(replay))
            argv = ["python", "-m", "dwr.tasks.certificate_rank_study", "--config",
                    str(qconfig_path), "--output", str(qout)]
            initial = {"config_sha256": qhash, "resolved_config": qconfig,
                       "source": historical_source, "output_dir": str(qout),
                       "config_path": str(qconfig_path), "requested_argv": argv}
            (qout / "manifest.initial.json").write_bytes(study._json_bytes(initial))
            source_floors = study.audit.source_gate([], 0, qconfig)[1]["floors"]
            development_floors = study.audit.development_gate([], [], qconfig)[1]["floors"]
            report = {"status": "PASS_QUALIFICATION", "fixture_only": False,
                      "raw_attempts": {"source": 3600, "development": 4800, "final": 0},
                      "source_gate": {"floors": {key: True for key in source_floors}},
                      "development_gate": {"floors": {key: True for key in development_floors}}}
            (qout / "report.json").write_bytes(study._json_bytes(report))
            artifact_hashes = {name: hashlib.sha256((qout / name).read_bytes()).hexdigest()
                               for name in study.ARTIFACTS[:-1]}
            manifest = {"status": "PASS_QUALIFICATION", "config_sha256": qhash,
                        "source": historical_source, "artifacts": artifact_hashes}
            (qout / "manifest.json").write_bytes(study._json_bytes(manifest))

            full_config["output_path"] = str(run_root / "certificate-five-stratum-full-20261014-v1")
            full_config["attempts"] = {"source": 1, "development": 1, "final": 1}
            full_output = Path(full_config["output_path"])
            def load_temp_config(path):
                if Path(path).resolve() == qconfig_path.resolve():
                    return qconfig, qhash, qconfig_path
                if Path(path).resolve() == full_config_path.resolve():
                    return full_config, full_hash, full_config_path
                raise AssertionError(f"unexpected config path: {path}")
            with patch.object(study, "__file__", str(repo / "dwr/tasks/certificate_rank_study.py")), \
                 patch.object(study, "load_config", side_effect=load_temp_config), \
                 patch.object(study, "ResourceGuard", return_value=_StopGuard()), \
                 patch.object(study, "_source_identity", return_value=current_source), \
                 patch.object(study, "_verify_historical_producers", return_value=True) as proof, \
                 patch.object(study, "_construct", return_value=object()) as constructor, \
                 patch.object(study, "_validate_outcome"), \
                 patch.object(study, "_build_row", return_value=_failed_row(0, full_config)), \
                 patch.object(study.audit, "full_stage_gate", return_value=(False, {"floors": {"source": False}})):
                result = study.run_study(full_config_path, full_output)
            proof.assert_called_once()
            self.assertEqual(result["qualification_link"]["status"], "PASS_QUALIFICATION")
            self.assertEqual(result["status"], "NO_GO_FULL_STAGE")
            self.assertEqual(result["source_gate"]["floors"]["source"], False)
            self.assertNotIn("final_structural_access", result)
            self.assertEqual(constructor.call_count, 2)  # source attempt and exact replay only
            saved_manifest = json.loads((full_output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["status"], "NO_GO_FULL_STAGE")

    def test_divergent_duplicate_source_policy_fields_are_not_admitted(self):
        original = QUAL.read_bytes()
        changed = original.replace(b'"legacy_node2_primitive_cell_min": 10',
                                   b'"legacy_node2_primitive_cell_min": 11')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "changed.json"
            path.write_bytes(changed)
            with self.assertRaises(study.ConfigError):
                study.load_config(path)

    def test_historical_producers_require_commit_clean_map_and_raw_git_blobs(self):
        blob = b"frozen task producer\n"
        digest = hashlib.sha256(blob).hexdigest()
        current = {"git_commit": "b" * 40, "git_clean": True,
                   "producer_sha256": {"dwr/tasks/frozen.py": digest}}
        with patch.object(study.subprocess, "run", return_value=SimpleNamespace(stdout=blob)) as run, \
             patch.object(study, "_check"):
            self.assertTrue(study._verify_historical_producers(current, current, None))
        run.assert_called_once()
        for invalid in ({**current, "git_commit": None}, {**current, "git_commit": "x" * 40},
                        {**current, "git_clean": 1},
                        {**current, "producer_sha256": {}}):
            with self.subTest(invalid=invalid), patch.object(study, "_check"):
                with self.assertRaises(study.StudyError):
                    study._verify_historical_producers(invalid, current, None)
        with patch.object(study.subprocess, "run", return_value=SimpleNamespace(stdout=b"changed")), \
             patch.object(study, "_check"):
            with self.assertRaisesRegex(study.StudyError, "historical producer"):
                study._verify_historical_producers(current, current, None)

    def test_historical_documentation_commit_difference_is_allowed(self):
        blob = b"same producer bytes"
        digest = hashlib.sha256(blob).hexdigest()
        historical = {"git_commit": "a" * 40, "git_clean": True,
                      "producer_sha256": {"dwr/tasks/frozen.py": digest}}
        current = {"git_commit": "b" * 40, "git_clean": True,
                   "producer_sha256": {"dwr/tasks/frozen.py": digest}}
        with patch.object(study.subprocess, "run", return_value=SimpleNamespace(stdout=blob)), \
             patch.object(study, "_check"):
            self.assertTrue(study._verify_historical_producers(historical, current, None))


class FiveRowPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = study.load_config(QUAL)[0]

    def test_fine_annotation_is_added_after_protected_route_and_is_input_free(self):
        outcome = _hand_outcome()
        metrics = {"content_id": "c", "lineage_id": "l", "structure_signature": "s",
                   "prompt_bytes": 1200, "first_solution_nodes": 9, "source_split": "source_test",
                   "development_split": "development_diagnostic_test"}
        with patch.object(study.old, "_task_metrics", return_value=metrics), \
             patch.object(study, "_scalar_solutions", return_value=()), \
             patch.object(study.old, "_audit_task", return_value={"oracle_unique": True, "eligible": True,
                                                                  "sensitive": True, "interaction": {}}), \
             patch.object(study, "enumerate_solutions", return_value=SimpleNamespace(solutions=())), \
             patch.object(study, "solve", return_value=SimpleNamespace(solution=None, stats=SimpleNamespace(uniqueness_solution_count_capped=0))), \
             patch.object(study, "propagate", return_value=SimpleNamespace(stalled=False, solved=False, contradiction=False)), \
             patch.object(study.old, "_audit_local_gac", return_value=True):
            row = study._build_row(outcome, "source", 0, self.config, set(), {}, None)
        self.assertEqual(row["state"], "protected_integrity_only")
        self.assertNotIn("fine_stratum", row)
        self.assertEqual(study.serialize_model_input(outcome.task), outcome.model_input)
        projection = study._a_projection({**row, "fine_stratum": "A3", "witness": {"private": 1}})
        self.assertEqual(projection["fine_stratum"], "A3")
        self.assertNotIn("witness", projection)
        self.assertNotIn("fine_stratum", json.loads(outcome.model_input))

    def test_actual_protected_first_seen_and_duplicate_rows_pass_five_schema(self):
        outcome = _hand_outcome()
        metrics = {"content_id": "protected-content", "lineage_id": "protected-lineage",
                   "structure_signature": "protected-signature", "prompt_bytes": 1200,
                   "first_solution_nodes": 9, "source_split": "source_test",
                   "development_split": "development_diagnostic_test"}
        sem = {"oracle_unique": True, "eligible": True, "sensitive": True, "oracle_solution_count": 1,
               "interaction": {}}
        patches = [
            patch.object(study.old, "_task_metrics", return_value=metrics),
            patch.object(study, "_scalar_solutions", return_value=()),
            patch.object(study.old, "_audit_task", return_value=sem),
            patch.object(study, "enumerate_solutions", return_value=SimpleNamespace(solutions=())),
            patch.object(study, "solve", return_value=SimpleNamespace(
                solution=None, stats=SimpleNamespace(uniqueness_solution_count_capped=0))),
            patch.object(study, "propagate", return_value=SimpleNamespace(
                stalled=False, solved=False, contradiction=False)),
            patch.object(study.old, "_audit_local_gac", return_value=True),
        ]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for active in patches:
                stack.enter_context(active)
            seen, lineage = set(), {}
            first = study._build_row(outcome, "source", 0, self.config, seen, lineage, None)
            second = study._build_row(outcome, "source", 1, self.config, seen, lineage, None)
        self.assertEqual((first["first_seen_content"], first["duplicate_content"]), (True, False))
        self.assertEqual((second["first_seen_content"], second["duplicate_content"]), (False, True))
        self.assertTrue(study._validate_row_annotation(first, self.config))
        self.assertTrue(study._validate_row_annotation(second, self.config))
        self.assertTrue(study.audit.row_schema_valid(first, self.config))
        self.assertTrue(study.audit.row_schema_valid(second, self.config))

    def test_a3_envelope_uses_real_metadata_and_duplicate_is_not_a_quota(self):
        outcome = _hand_outcome()
        metrics = {"content_id": "c", "lineage_id": "l", "structure_signature": "s",
                   "prompt_bytes": 1200, "first_solution_nodes": 9, "source_split": "source_train",
                   "development_split": "development_validation"}
        sem = {"oracle_unique": True, "eligible": True, "sensitive": True,
               "oracle_solution_count": 1, "interaction": {}}
        with patch.object(study.old, "_task_metrics", return_value=metrics), \
             patch.object(study.old, "_audit_task", return_value=sem), \
             patch.object(study, "_scalar_solutions", return_value=()), \
             patch.object(study, "enumerate_solutions", return_value=SimpleNamespace(solutions=())), \
             patch.object(study, "solve", return_value=SimpleNamespace(solution=None, stats=SimpleNamespace(uniqueness_solution_count_capped=0))), \
             patch.object(study, "propagate", return_value=SimpleNamespace(stalled=False, solved=False, contradiction=False)), \
             patch.object(study.old, "_audit_local_gac", return_value=True):
            seen, lineage = set(), {}
            first = study._build_row(outcome, "source", 0, self.config, seen, lineage, None)
            second = study._build_row(outcome, "source", 0, self.config, seen, lineage, None)
        self.assertEqual(first["fine_stratum"], "A3")
        self.assertEqual(first["first_solution_nodes"], 9)
        self.assertTrue(study._is_A(first, self.config))
        self.assertFalse(study._is_A(second, self.config))
        quotas = Counter()
        study._count_A_for_quota(first, "eq", quotas, self.config)
        study._count_A_for_quota(second, "eq", quotas, self.config)
        self.assertEqual(quotas["eq"], 1)

    def test_legacy_row_schema_and_predicates_remain_unchanged(self):
        row = {"first_seen_content": True, "oracle_unique": True, "eligible": True, "envelope": True,
               "cell": "A", "first_solution_nodes": 5, "prompt_bytes": 1200}
        self.assertTrue(study._is_A(row))
        self.assertNotIn("fine_stratum", study._a_projection(row))

    def test_new_row_annotation_validator_rejects_missing_and_mismatched_labels(self):
        base = {"state": "first_seen", "split": "source_train", "partition": "source",
                "candidate_index": 0, "profile": "eq", "cell": "A", "prompt_bytes": 1200,
                "first_solution_nodes": 9, "fine_stratum": "A3", "oracle_unique": True,
                "eligible": True, "sensitive": True, "envelope": True,
                "first_seen_content": True, "given_count": 2, "values": [0, 1],
                "content_id": "c", "lineage_id": "l", "structure_signature": "s",
                "branch": "ordinary", "model_input_sha256": "b" * 64, "gac": {}}
        self.assertTrue(study._validate_row_annotation(base, self.config))
        for invalid in ({**base, "fine_stratum": "A2"}, {k: v for k, v in base.items() if k != "fine_stratum"}):
            with self.subTest(invalid=invalid), self.assertRaises(study.StudyError):
                study._validate_row_annotation(invalid, self.config)
            self.assertFalse(study.audit.row_schema_valid(invalid, self.config))

    def test_out_of_support_null_and_protected_failure_rows_have_explicit_valid_shapes(self):
        outside = {"state": "first_seen", "split": "source_train", "partition": "source",
                   "candidate_index": 0, "profile": "eq", "cell": "A", "prompt_bytes": 900,
                   "first_solution_nodes": 20, "fine_stratum": None, "oracle_unique": True,
                   "eligible": True, "sensitive": True, "envelope": False,
                   "first_seen_content": True, "given_count": 2, "values": [0, 1],
                   "content_id": "c", "lineage_id": "l", "structure_signature": "s",
                   "branch": "ordinary", "model_input_sha256": "b" * 64, "gac": {}}
        protected = {"state": "protected_integrity_only", "split": "source_test", "partition": "source",
                     "candidate_index": 1, "profile": "eq", "cell": "A",
                     "oracle_unique": False, "eligible": False, "sensitive": False, "envelope": False,
                     "first_seen_content": True, "duplicate_content": False,
                     "given_count": 2, "values": [0, 1],
                     "content_id": "c", "lineage_id": "l", "structure_signature": "s",
                     "branch": "ordinary"}
        failed = {"state": "construction_failed", "split": None, "partition": "source",
                  "candidate_index": 2, "profile": "eq",
                  "oracle_unique": False, "eligible": False, "sensitive": False, "envelope": False,
                  "first_seen_content": False, "given_count": 0, "values": []}
        for row in (outside, protected, failed):
            self.assertTrue(study._validate_row_annotation(row, self.config))
            self.assertTrue(study.audit.row_schema_valid(row, self.config))

    def test_identically_malformed_saved_and_rebuilt_annotations_fail_replay(self):
        cfg, digest, _ = study.load_config(QUAL)
        bad = {"partition": "source", "candidate_index": 0, "profile": "eq", "cell": "A",
               "prompt_bytes": 1200, "first_solution_nodes": 9, "fine_stratum": "A2",
               "first_seen_content": True, "oracle_unique": True, "eligible": True, "envelope": True}
        saved = {**bad, "config_sha256": digest}
        self.assertFalse(study._replay_matches([saved], [bad], study._json_bytes(saved), b"", digest, cfg))


class FiveRunnerFixtureTests(unittest.TestCase):
    def test_late_terminal_manifest_write_failure_revokes_persisted_pass(self):
        cfg, digest, cfg_path = study.load_config(QUAL)
        cfg["attempts"] = {"source": 1, "development": 1, "final": 0}
        run_root = ROOT / "runs"
        with tempfile.TemporaryDirectory(dir=run_root) as temp:
            output = Path(temp) / "qualification"
            cfg["output_path"] = str(output)
            def build(outcome, part, index, config, seen, lineage, guard, fixture=False):
                row = _normal_row(index, "A3", True,
                                  "source_train" if part == "source" else "development_validation", config)
                row["partition"] = part
                row["candidate_index"] = index
                row["profile"], row["cell"] = study._schedule(part, index)
                return row
            original_write = study._write
            failed = False
            def fail_after_manifest(path, value, guard):
                nonlocal failed
                original_write(path, value, guard)
                if path.name == "manifest.json" and not failed:
                    failed = True
                    raise TimeoutError("late terminal guard injection")
            with patch.object(study, "load_config", return_value=(cfg, digest, cfg_path)), \
                 patch.object(study, "ResourceGuard", return_value=_StopGuard()), \
                 patch.object(study, "_source_identity", return_value={"git_commit": "a" * 40, "git_clean": True}), \
                 patch.object(study, "_construct", return_value=object()), \
                 patch.object(study, "_validate_outcome"), \
                 patch.object(study, "_build_row", side_effect=build), \
                 patch.object(study.audit, "source_gate", return_value=(True, {"floors": {"source": True}})), \
                 patch.object(study.audit, "development_gate", return_value=(True, {"floors": {"development": True}})), \
                 patch.object(study, "_write", side_effect=fail_after_manifest):
                with self.assertRaisesRegex(TimeoutError, "late terminal"):
                    study.run_study(QUAL, output)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(failed)
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["status"], "NOT_ASSESSED")
            self.assertEqual(report["g2_status"], "not_assessed")
            self.assertEqual(manifest["artifacts"], {})

    def test_real_qualification_runner_terminal_manifest_on_source_gate_failure(self):
        cfg, digest, cfg_path = study.load_config(QUAL)
        cfg["attempts"] = {"source": 1, "development": 1, "final": 0}
        row = _failed_row(0, cfg)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "qualification"
            with patch.object(study, "load_config", return_value=(cfg, digest, cfg_path)), \
                 patch.object(study.audit, "source_gate", return_value=(False, {"floors": {"source": False}})), \
                 patch.object(study, "_build_row", return_value=row):
                report = study.run_study(QUAL, output, _guard=_StopGuard(), _constructor=lambda *args: object())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED_FIXTURE_ONLY")
            self.assertEqual(manifest["status"], "NOT_ASSESSED_FIXTURE_ONLY")
            self.assertTrue((output / "report.json").is_file())
            self.assertTrue((output / "manifest.initial.json").is_file())
            self.assertEqual(report["raw_attempts"]["source"], 1)

    def test_real_full_fixture_stops_at_source_failure_before_final_draw(self):
        cfg, digest, cfg_path = study.load_config(FULL)
        cfg["attempts"] = {"source": 1, "development": 1, "final": 1}
        row = _failed_row(0, cfg)
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "full"
            def constructor(config, part, index):
                calls.append(part)
                return object()
            with patch.object(study, "load_config", return_value=(cfg, digest, cfg_path)), \
                 patch.object(study.audit, "full_stage_gate", return_value=(False, {"floors": {"source": False}})), \
                 patch.object(study, "_build_row", return_value=row):
                report = study.run_study(FULL, output, _guard=_StopGuard(), _constructor=constructor)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(calls, ["source", "source"])
            self.assertEqual(report["status"], "NOT_ASSESSED_FIXTURE_ONLY")
            self.assertEqual(manifest["status"], "NOT_ASSESSED_FIXTURE_ONLY")
            self.assertNotIn("final_structural_access", report)
            self.assertTrue((output / "replay.json").is_file())

    def test_canned_fixture_runner_writes_terminal_artifacts_but_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "fixture"
            guard = _StopGuard()
            def constructor(*args):
                raise TimeoutError("canned bounded stop")
            with self.assertRaises(TimeoutError):
                study.run_study(QUAL, output, _guard=guard, _constructor=constructor)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(report["fixture_only"])
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["status"], "NOT_ASSESSED")
            self.assertTrue(all((output / name).exists() for name in study.ARTIFACTS))

    def test_new_terminal_status_is_distinct_and_revocable(self):
        self.assertEqual(study._g2_status("PASS_G2_FIVE_STRATUM_V1", True, True), "pass")
        with tempfile.TemporaryDirectory() as temp:
            report = {"status": "PASS_G2_FIVE_STRATUM_V1", "g2_status": "pass", "artifacts": {"stale": "x"}}
            study._revoke_terminal(Path(temp), study.FIVE_CONFIG_HASHES["full"], report, TimeoutError())
            saved = json.loads((Path(temp) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "NOT_ASSESSED")
            self.assertEqual(saved["g2_status"], "not_assessed")

    def test_cli_accepts_the_registered_new_full_pass_status(self):
        with patch.object(study, "run_study", return_value={
                "status": "PASS_G2_FIVE_STRATUM_V1", "g2_status": "pass"}):
            with patch("builtins.print"):
                self.assertEqual(study.main(["--config", str(FULL), "--output", "unused"]), 0)

    def test_constructor_delegation_is_unchanged(self):
        with patch.object(study.rank_generator, "construct", return_value="canned") as construct:
            self.assertEqual(study._construct(20261016, "source", 7, None), "canned")
        construct.assert_called_once()
        self.assertEqual(construct.call_args.args[:3], (20261016, "source", 7))

    def test_new_full_rejects_wrong_and_existing_output_paths(self):
        wrong = ROOT / "runs/certificate-five-stratum-wrong-test"
        with self.assertRaisesRegex(ValueError, "registered path"):
            study.run_study(FULL, wrong)
        self.assertFalse(wrong.exists())
        with tempfile.TemporaryDirectory() as temp:
            existing = Path(temp) / "already-there"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                study.run_study(FULL, existing)
            self.assertTrue(existing.is_dir())

    def test_new_replay_binds_fine_annotation_and_late_terminal_cap(self):
        cfg, digest, _ = study.load_config(QUAL)
        row = _normal_row(0, "A3", True, "source_train", cfg)
        saved = {**row, "config_sha256": digest}
        raw = study._json_bytes(saved)
        projection = study._a_projection(saved)
        projection["outcome_sha256"] = hashlib.sha256(raw).hexdigest()
        records = study._json_bytes(projection)
        self.assertTrue(study._replay_matches([saved], [row], raw, records, digest, cfg))
        corrupt = {**row, "fine_stratum": "A2"}
        self.assertFalse(study._replay_matches([saved], [corrupt], raw, records, digest, cfg))

    def test_exact_projection_requires_all_and_only_accepted_rows_in_order(self):
        cfg = study.load_config(QUAL)[0]
        cfg["attempts"] = {"source": 5, "development": 0, "final": 0}
        digest = study.FIVE_CONFIG_HASHES["qualification"]
        rows = [
            _normal_row(0, "A3", True, "source_train", cfg),
            _normal_row(1, None, True, "source_train", cfg, supported=False),
            _protected_row(2, cfg),
            _normal_row(3, "A3", False, "source_train", cfg),
            _failed_row(4, cfg),
        ]
        lines = [study._json_bytes({**row, "config_sha256": digest}) for row in rows]
        expected = study._five_qualification_projection_bytes(lines, cfg, digest)
        self.assertEqual(len(expected.splitlines()), 1)
        self.assertTrue(study._verify_exact_qualification_records(lines, expected, cfg, digest))
        accepted_projection = study._json_bytes({**study._a_projection({**rows[0], "config_sha256": digest}),
                                                 "outcome_sha256": hashlib.sha256(lines[0]).hexdigest()})
        for corrupted in (b"", expected + accepted_projection, accepted_projection + accepted_projection,
                          expected.replace(b'"model_input_sha256":"' + b"b" * 64,
                                           b'"model_input_sha256":"' + b"c" * 64)):
            with self.subTest(corrupted=corrupted[:40]), self.assertRaises(study.StudyError):
                study._verify_exact_qualification_records(lines, corrupted, cfg, digest)
        for excluded_index in (2, 3):
            excluded = rows[excluded_index]
            projection = {**study._a_projection({**excluded, "config_sha256": digest}),
                          "outcome_sha256": hashlib.sha256(lines[excluded_index]).hexdigest()}
            with self.subTest(excluded=excluded_index), self.assertRaises(study.StudyError):
                study._verify_exact_qualification_records(lines, expected + study._json_bytes(projection), cfg, digest)
        mismatched = {**rows[0], "fine_stratum": "A2", "config_sha256": digest}
        cfg["attempts"] = {"source": 1, "development": 0, "final": 0}
        with self.assertRaisesRegex(study.StudyError, "schema or fine annotation"):
            study._five_qualification_projection_bytes([study._json_bytes(mismatched)], cfg, digest)

    def test_five_full_audit_rejects_a_consistently_mislabeled_row(self):
        cfg = study.load_config(FULL)[0]
        bad = _normal_row(0, "A2", True, "source_train", cfg)
        ok, report = study.audit.full_gate([bad], {"source": 1, "development": 0, "final": 0}, cfg)
        self.assertFalse(ok)
        self.assertFalse(report["floors"]["row_schema"])


def _hand_outcome():
    names = tuple(f"v{i}" for i in range(6))
    pairs = (("v0", "v1"), ("v0", "v2"), ("v0", "v3"), ("v0", "v4"),
             ("v1", "v2"), ("v1", "v3"), ("v1", "v4"), ("v2", "v3"))
    constraints = tuple(Constraint(RelationKind.EQ, pair) for pair in pairs)
    task = ConstraintTask("constraint-task-v1", 3, names, constraints,
                          (GivenAssignment("v0", 0), GivenAssignment("v1", 0)))
    return SimpleNamespace(task=task, profile=("eq",), cell="A", branch="ordinary",
                           witness=SimpleNamespace(assignments=tuple((n, 0) for n in names)),
                           givens=(), alternatives=(), reserved_constraints=(), rejection_reason=None,
                           work={}, model_input=study.serialize_model_input(task))


def _normal_row(index, fine, eligible, split, config, *, supported=True):
    node_count, prompt_bytes = (9, 1200) if supported else (20, 900)
    return {"state": "first_seen", "partition": "source", "candidate_index": index,
            "profile": "eq", "cell": "A", "branch": "ordinary", "split": split,
            "content_id": f"c{index}", "lineage_id": f"l{index}",
            "structure_signature": f"s{index}", "oracle_unique": True,
            "eligible": eligible, "sensitive": True, "envelope": supported,
            "first_seen_content": True, "values": [0, 1], "given_count": 2,
            "prompt_bytes": prompt_bytes, "first_solution_nodes": node_count,
            "model_input_sha256": "b" * 64, "gac": {}, "fine_stratum": fine}


def _receipt_row(partition, index, profile, cell, config, *, accepted=False):
    split = "source_train" if partition == "source" else "development_validation"
    prompt_bytes, nodes = (1200, 9) if accepted else (900, 1)
    row = {"state": "first_seen", "partition": partition, "candidate_index": index,
           "profile": profile, "cell": cell, "branch": "ordinary", "split": split,
           "content_id": f"receipt-{partition}-{index}", "lineage_id": f"receipt-lineage-{partition}-{index}",
           "structure_signature": f"receipt-signature-{partition}-{index}",
           "oracle_unique": True, "eligible": True, "sensitive": True,
           "envelope": accepted, "first_seen_content": True, "duplicate_content": False,
           "values": [0, 1], "given_count": 2, "prompt_bytes": prompt_bytes,
           "first_solution_nodes": nodes, "model_input_sha256": "c" * 64, "gac": {}}
    row["fine_stratum"] = study.audit.support_stratum(row, config)
    return row


def _public_row(row):
    fields = ("state", "partition", "candidate_index", "profile", "cell", "branch", "split",
              "content_id", "lineage_id", "structure_signature", "oracle_unique", "eligible",
              "sensitive", "envelope", "first_seen_content", "duplicate_content", "values",
              "given_count", "prompt_bytes", "first_solution_nodes", "model_input_sha256", "gac")
    return {key: row[key] for key in fields if key in row}


def _protected_row(index, config):
    return {"state": "protected_integrity_only", "partition": "source", "candidate_index": index,
            "profile": "eq", "cell": "A", "branch": "ordinary", "split": "source_test",
            "content_id": f"c{index}", "lineage_id": f"l{index}",
            "structure_signature": f"s{index}", "oracle_unique": False,
            "eligible": False, "sensitive": False, "envelope": False,
            "first_seen_content": True, "duplicate_content": False,
            "values": [0, 1], "given_count": 2}


def _failed_row(index, config):
    profile, cell = study._schedule("source", index)
    return {"state": "construction_failed", "partition": "source", "candidate_index": index,
            "profile": profile, "cell": cell, "split": None, "oracle_unique": False,
            "eligible": False, "sensitive": False, "envelope": False,
            "first_seen_content": False}


class _StopGuard:
    wall_seconds = 1800
    peak_bytes = 1
    def check(self): pass


if __name__ == "__main__":
    unittest.main()
