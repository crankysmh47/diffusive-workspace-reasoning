"""Closed source-export runner tests using small real producer fixtures."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dwr.experiments import source_export
from dwr.experiments.records import verify_artifacts
from dwr.experiments.source_corpus import export_source_records

from test_source_corpus import _json_line, _raw_row, _task


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=10).stdout


class SourceExportRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        _git(self.root, "init", "-q")
        _git(self.root, "config", "user.name", "Source Export Fixture")
        _git(self.root, "config", "user.email", "fixture@example.invalid")
        _git(self.root, "config", "core.autocrlf", "false")
        self._copy_runtime()
        (self.root / ".gitignore").write_text("runs/\n", encoding="utf-8", newline="\n")
        (self.root / "runs").mkdir()
        self.input_dir = self.root / "runs" / "source-fixture"
        self.input_dir.mkdir()
        self.generation_config = self.root / "configs" / "tasks" / "generation-fixture.json"
        self.generation_config.parent.mkdir(parents=True, exist_ok=True)
        self.generation = {"fixture": "runner", "source_profiles": ["eq", "neq", "add_mod"]}
        self.generation_bytes = json.dumps(self.generation, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        self.generation_config.write_bytes(self.generation_bytes)
        self.proof_path = self.root / "configs" / "splits" / "proof-fixture.json"
        self.proof_path.parent.mkdir(parents=True, exist_ok=True)
        self.proof_path.write_bytes(b"pending proof\n")
        self.output_dir = "runs/source-export-fixture"
        self.config = {
            "artifact_sha256": {},
            "config_version": "dwr-source-export-v1",
            "expected_counts": {"source_train": 1, "source_validation": 1},
            "forbidden_payload_splits": ["source_test", "development_validation", "development_diagnostic_test", "final_test"],
            "generation_config_path": "configs/tasks/generation-fixture.json",
            "generation_config_sha256": _sha(self.generation_bytes),
            "input_dir": "runs/source-fixture",
            "limits": {"attempts": 1, "cpu_workers": 1, "external_watchdog_seconds": 305,
                       "memory_mib": 512, "wall_seconds": 300},
            "model_execution": False,
            "operation": "source-only-retained-corpus-conversion",
            "output_dir": self.output_dir,
            "output_filename": "source_examples.jsonl",
            "permitted_payload_splits": ["source_train", "source_validation"],
            "producer_commit": "a" * 40,
            "root_seed_unused": 20261019,
            "structural_result_path": "configs/splits/proof-fixture.json",
            "structural_result_sha256": _sha(self.proof_path.read_bytes()),
        }
        self.config_path = self.root / "configs" / "tasks" / "source_export_v1.json"
        self._write_config()
        self._write_inputs()
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-qm", "fixture source export implementation")

    def tearDown(self):
        self.temp.cleanup()

    def _copy_runtime(self):
        source_root = Path(source_export._RUNTIME_PACKAGE_ROOT)
        for rel in source_export._RUNTIME_SOURCE_PATHS:
            source = source_root / rel
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def _write_config(self):
        self.config_bytes = json.dumps(self.config, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        self.config_path.write_bytes(self.config_bytes)

    def _write_inputs(self, *, outcomes=None, records=None):
        commit = "a" * 40
        rows = [
            _raw_row(_task("eq"), "source_train", "source", 0, "eq"),
            _raw_row(_task("add_mod"), "source_validation", "source", 1, "add_mod"),
        ]
        for row in rows:
            row["config_sha256"] = self.config["generation_config_sha256"]
        if outcomes is None:
            outcomes = [*rows, {"partition": "source", "candidate_index": 99,
                "split": "source_test", "state": "first_seen", "task": "poison protected task",
                "witness": {"poison": True}, "values": {"poison": True}}]
        lines = [_json_line(row) for row in outcomes]
        by_key = {(row["partition"], row["candidate_index"]): line for row, line in zip(outcomes, lines)}
        projections = []
        from dwr.tasks import certificate_rank_study
        for row in rows if records is None else records:
            projection = certificate_rank_study._a_projection(row)
            projection["outcome_sha256"] = _sha(by_key[(row["partition"], row["candidate_index"])])
            projections.append(_json_line(projection))
        artifacts = {
            "manifest.initial.json": _json_line({"config_sha256": self.config["generation_config_sha256"],
                "resolved_config": self.generation, "source": {"git_commit": commit, "git_clean": True}, "status": "running"}),
            "outcomes.jsonl": b"".join(lines),
            "records.jsonl": b"".join(projections),
            "replay.json": _json_line({"exact_indices": True, "status": "PASS_REPLAY"}),
            "report.json": _json_line({"config_sha256": self.config["generation_config_sha256"], "fixture_only": False,
                "g2_status": "pass", "status": "PASS_G2_FIVE_STRATUM_V1"}),
        }
        for name, raw in artifacts.items():
            (self.input_dir / name).write_bytes(raw)
        artifact_hashes = {name: _sha(raw) for name, raw in artifacts.items()}
        terminal = {"artifacts": artifact_hashes, "config_sha256": self.config["generation_config_sha256"],
            "g2_status": "pass", "source": {"git_commit": commit, "git_clean": True},
            "status": "PASS_G2_FIVE_STRATUM_V1"}
        terminal_raw = _json_line(terminal)
        (self.input_dir / "manifest.json").write_bytes(terminal_raw)
        self.config["artifact_sha256"] = {**artifact_hashes, "manifest.json": _sha(terminal_raw)}
        proof = {"full": {"artifact_sha256": self.config["artifact_sha256"],
                 "config_sha256": self.config["generation_config_sha256"],
                 "controller_checks": {"all_stages_pass": True}},
                 "source_commit": self.config["producer_commit"],
                 "status": "PASS_G2_FIVE_STRATUM_V1"}
        self.proof_path.write_bytes(_json_line(proof))
        self.config["structural_result_sha256"] = _sha(self.proof_path.read_bytes())
        self._write_config()

    def _run(self, *, semantic_hook=None, expected_config_hash=None):
        with patch.object(source_export, "_REPO_ROOT", self.root), \
             patch.object(source_export, "_PRODUCTION_CONFIG_SHA256", expected_config_hash or _sha(self.config_bytes)), \
             patch.object(source_export, "_REGISTERED_PATHS", {
                 "input_dir": "runs/source-fixture",
                 "generation_config_path": "configs/tasks/generation-fixture.json",
                 "structural_result_path": "configs/splits/proof-fixture.json",
                 "output_dir": self.output_dir,
             }):
            if semantic_hook is None:
                return source_export.run_export("configs/tasks/source_export_v1.json", command=["python", "fixture"])
            with patch("dwr.experiments.source_corpus.schema.parse_task_json", side_effect=semantic_hook):
                return source_export.run_export("configs/tasks/source_export_v1.json", command=["python", "fixture"])

    def test_closed_runner_creates_initial_and_access_event_before_converter_semantics(self):
        called = []
        from dwr.tasks import schema
        original = schema.parse_task_json

        def observe(task_text):
            run_dir = self.root / self.output_dir
            initial = json.loads((run_dir / "manifest.initial.json").read_text("utf-8"))
            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual(initial["status"], "running")
            self.assertEqual(events[0]["kind"], "source_access_policy")
            called.append(task_text)
            return original(task_text)

        result = self._run(semantic_hook=observe)
        run_dir = self.root / self.output_dir
        terminal = verify_artifacts(run_dir)
        self.assertEqual(terminal["status"], "completed")
        self.assertEqual(result["exported_counts"], {"source_train": 1, "source_validation": 1})
        self.assertEqual(len(called), 2)
        self.assertEqual(terminal["artifacts"], [
            {"path": "source_examples.jsonl", "size_bytes": (run_dir / "source_examples.jsonl").stat().st_size,
             "sha256": _sha((run_dir / "source_examples.jsonl").read_bytes())},
            {"path": "summary.json", "size_bytes": (run_dir / "summary.json").stat().st_size,
             "sha256": _sha((run_dir / "summary.json").read_bytes())},
        ])
        initial = json.loads((run_dir / "manifest.initial.json").read_text("utf-8"))
        self.assertTrue(initial["confirmatory"])
        self.assertEqual(initial["settings"]["scientific_status"], "deterministic_export_not_hypothesis_test")
        self.assertNotIn("torch", __import__("sys").modules)

    def test_closed_loader_rejects_config_hash_unknown_keys_and_path_changes(self):
        self.config["unexpected"] = True
        self._write_config()
        with self.assertRaises(ValueError):
            self._run()
        self.assertFalse((self.root / self.output_dir).exists())

        self.config.pop("unexpected")
        self.config["input_dir"] = "runs/../source-fixture"
        self._write_config()
        with self.assertRaises(ValueError):
            self._run()
        self.assertFalse((self.root / self.output_dir).exists())

    def test_rejects_existing_unique_output_and_bad_baseline_hash(self):
        occupied = self.root / self.output_dir
        occupied.mkdir()
        (occupied / "old.json").write_text("preserve", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._run()
        self.assertEqual((occupied / "old.json").read_text("utf-8"), "preserve")
        (occupied / "old.json").unlink()
        occupied.rmdir()

        with self.assertRaises(ValueError):
            self._run(expected_config_hash="0" * 64)
        self.assertFalse(occupied.exists())

    def test_strong_source_policy_rejects_dirty_fixture_before_run_directory(self):
        with (self.root / "dwr" / "experiments" / "source_export.py").open("ab") as stream:
            stream.write(b"# dirty\n")
        with self.assertRaises(ValueError):
            self._run()
        self.assertFalse((self.root / self.output_dir).exists())

    def test_rejects_tracked_payload_paths_before_runrecord_inventory(self):
        _git(self.root, "add", "-f", "runs/source-fixture/outcomes.jsonl")
        _git(self.root, "commit", "-qm", "invalid tracked payload fixture")
        with patch.object(source_export.RunRecord, "create", side_effect=AssertionError("manifest was too early")):
            with self.assertRaisesRegex(ValueError, "must be untracked and ignored"):
                self._run()
        self.assertFalse((self.root / self.output_dir).exists())

    def test_converter_failure_is_terminal_and_preserves_partial_export(self):
        def fail_after_output(*args, **kwargs):
            output_path = Path(args[1])
            output_path.write_bytes(b"partial evidence\n")
            raise RuntimeError("fixture converter failure")

        with patch("dwr.experiments.source_export.export_source_records", side_effect=fail_after_output):
            with self.assertRaisesRegex(RuntimeError, "fixture converter failure"):
                self._run()
        run_dir = self.root / self.output_dir
        terminal = verify_artifacts(run_dir)
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["error"], "RuntimeError: fixture converter failure")
        self.assertTrue((run_dir / "source_examples.jsonl").exists())

    def test_resource_timeout_is_terminal_as_interrupted(self):
        def timeout_after_partial(*args, **kwargs):
            Path(args[1]).write_bytes(b"interrupted partial\n")
            raise TimeoutError("registered wall cap")

        with patch("dwr.experiments.source_export.export_source_records", side_effect=timeout_after_partial):
            with self.assertRaisesRegex(TimeoutError, "registered wall cap"):
                self._run()
        run_dir = self.root / self.output_dir
        terminal = verify_artifacts(run_dir)
        self.assertEqual(terminal["status"], "interrupted")
        self.assertEqual(terminal["error"], "TimeoutError: registered wall cap")
        self.assertTrue((run_dir / "source_examples.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
