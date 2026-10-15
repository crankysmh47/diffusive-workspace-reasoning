"""Contract tests for byte-bound experiment run records."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dwr.experiments.records import RunRecord, verify_artifacts


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=10).stdout


class RunRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        git(self.root, "init", "-q")
        git(self.root, "config", "user.name", "Run Record Fixture")
        git(self.root, "config", "user.email", "fixture@example.invalid")
        git(self.root, "config", "core.autocrlf", "false")
        (self.root / "source.py").write_bytes(b"source = 1\n")
        git(self.root, "add", "source.py")
        git(self.root, "commit", "-qm", "fixture")
        self.work = self.root / "work"
        self.work.mkdir()
        self.config = self.root / "config.json"
        self.config.write_bytes(b'{"mode":"fixture","seed":7}\n')
        git(self.root, "add", "config.json")
        git(self.root, "commit", "-qm", "config fixture")

    def tearDown(self):
        self.temp.cleanup()

    def metadata(self):
        return dict(command=["python", "run.py"], config_path=self.config,
                    expected_config_sha256=hashlib.sha256(self.config.read_bytes()).hexdigest(),
                    identities={"task": {"name": "fixture", "revision": "v1", "sha256": "a" * 64}},
                    rng_streams={"python": 7, "numpy": 8},
                    settings={"precision": "float32", "model": None},
                    budgets={"wall_seconds": 10.5, "memory_mib": 128, "attempts": 1},
                    source_root=self.root)

    def create(self, name="run", **overrides):
        args = self.metadata()
        args.update(overrides)
        return RunRecord.create(self.work / name, **args)

    def finish(self, record, artifacts=()):
        return record.finish("completed", counts={"examples": 1}, artifacts=list(artifacts))

    def test_initial_record_precedes_work_and_terminal_verifies(self):
        record = self.create()
        initial = self.work / "run" / "manifest.initial.json"
        terminal = self.work / "run" / "manifest.json"
        self.assertTrue(initial.is_file())
        self.assertFalse(terminal.exists())
        data = json.loads(initial.read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["config"]["resolved"], {"mode": "fixture", "seed": 7})
        artifact = self.work / "run" / "result.json"
        artifact.write_text('{"ok":true}\n', encoding="utf-8", newline="\n")
        record.append_event("batch", {"count": 1})
        result = self.finish(record, ["result.json"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(verify_artifacts(self.work / "run"), result)
        with self.assertRaises(ValueError):
            record.finish("completed", counts={"examples": 1}, artifacts=["result.json"])

    def test_config_hash_and_strict_json_fail_before_output_creation(self):
        bad = self.work / "bad"
        args = self.metadata()
        args["expected_config_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            RunRecord.create(bad, **args)
        self.assertFalse(bad.exists())
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b"[]"):
            self.config.write_bytes(raw)
            args = self.metadata()
            with self.assertRaises(ValueError):
                RunRecord.create(self.work / ("bad" + str(len(raw))), **args)

    def test_metadata_types_and_unknown_kwargs_are_rejected_before_output(self):
        for field, value in (("confirmatory", 1), ("rng_streams", {"a": True}),
                             ("budgets", {"wall_seconds": float("inf"), "memory_mib": 1, "attempts": 1})):
            args = self.metadata()
            args[field] = value
            path = self.work / field
            with self.assertRaises((TypeError, ValueError)):
                RunRecord.create(path, **args)
            self.assertFalse(path.exists())
        with self.assertRaises(TypeError):
            RunRecord.create(self.work / "unknown", **self.metadata(), extra=True)

    def test_caller_metadata_is_detached(self):
        settings = {"precision": {"name": "float32"}}
        record = self.create(settings=settings)
        settings["precision"]["name"] = "mutated"
        doc = json.loads((self.work / "run" / "manifest.initial.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["settings"]["precision"]["name"], "float32")

    def test_output_directory_is_exclusive_and_name_is_constrained(self):
        path = self.work / "existing"
        path.mkdir()
        with self.assertRaises((FileExistsError, ValueError)):
            self.create("existing")
        with self.assertRaises(ValueError):
            self.create("bad name")
        with self.assertRaises(ValueError):
            RunRecord.create(self.root / ".git" / "run", **self.metadata())

    def test_exploratory_inventory_binds_dirty_and_untracked_bytes(self):
        (self.root / "source.py").write_bytes(b"source = 2\n")
        (self.root / "new.py").write_bytes(b"new = True\n")
        doc = json.loads((self.create().output_dir / "manifest.initial.json").read_text(encoding="utf-8"))
        self.assertFalse(doc["source"]["clean"])
        files = {entry["path"]: entry for entry in doc["source"]["files"]}
        self.assertIn("new.py", files)
        self.assertEqual(files["new.py"]["sha256"], hashlib.sha256(b"new = True\n").hexdigest())
        self.assertEqual(files["source.py"]["sha256"], hashlib.sha256(b"source = 2\n").hexdigest())

    def test_confirmatory_requires_clean_status_and_exact_head_bytes(self):
        args = self.metadata()
        args["confirmatory"] = True
        (self.root / "source.py").write_bytes(b"source = 2\n")
        with self.assertRaises(ValueError):
            RunRecord.create(self.work / "dirty", **args)
        git(self.root, "checkout", "--", "source.py")
        record = RunRecord.create(self.work / "clean", **args)
        self.assertTrue(json.loads((self.work / "clean" / "manifest.initial.json").read_text())["source"]["clean"])
        record.finish("failed", counts={}, artifacts=[], error="fixture failure")

    def test_confirmatory_detects_crlf_worktree_even_when_git_status_is_clean(self):
        (self.root / "line.txt").write_bytes(b"one\n")
        git(self.root, "add", "line.txt")
        git(self.root, "commit", "-qm", "line ending fixture")
        git(self.root, "update-index", "--assume-unchanged", "line.txt")
        (self.root / "line.txt").write_bytes(b"one\r\n")
        self.assertEqual(git(self.root, "status", "--porcelain"), b"")
        args = self.metadata()
        args["confirmatory"] = True
        with self.assertRaises(ValueError):
            RunRecord.create(self.work / "crlf", **args)

    def test_event_sequence_and_terminal_is_immutable(self):
        record = self.create()
        record.append_event("start", {})
        record.append_event("progress", {"step": 1})
        events = (self.work / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["sequence"] for line in events], [1, 2])
        self.finish(record)
        with self.assertRaises(ValueError):
            record.append_event("late", {})
        with self.assertRaises(ValueError):
            record.finish("failed", counts={}, artifacts=[], error="again")

    def test_failure_and_interruption_require_error_and_preserve_records(self):
        record = self.create()
        with self.assertRaises(ValueError):
            record.finish("failed", counts={}, artifacts=[])
        (self.work / "run" / "log.txt").write_text("failure evidence\n", encoding="utf-8")
        terminal = record.finish("interrupted", counts={"steps": 2}, artifacts=["log.txt"], error="stopped")
        self.assertEqual(terminal["status"], "interrupted")
        self.assertTrue((self.work / "run" / "manifest.initial.json").exists())
        self.assertTrue((self.work / "run" / "events.jsonl").exists())
        self.assertEqual(verify_artifacts(self.work / "run")["error"], "stopped")

    def test_artifact_paths_duplicates_missing_and_reserved_names_fail(self):
        cases = (["../outside"], ["/absolute"], ["manifest.json"], ["manifest.initial.json"],
                 ["Manifest.JSON"], ["events.jsonl"], ["same.txt", "same.txt"], ["missing.txt"])
        for index, artifacts in enumerate(cases):
            record = self.create("badartifact" + str(index))
            with self.assertRaises((ValueError, FileNotFoundError)):
                record.finish("completed", counts={}, artifacts=artifacts)
            self.assertFalse((record.output_dir / "manifest.json").exists())

    def test_tampered_or_absent_artifacts_are_detected(self):
        record = self.create()
        out = record.output_dir
        (out / "artifact.bin").write_bytes(b"original")
        self.finish(record, ["artifact.bin"])
        (out / "artifact.bin").write_bytes(b"changed!")
        with self.assertRaises(ValueError):
            verify_artifacts(out)

    def test_tampered_initial_and_events_are_detected(self):
        record = self.create()
        out = record.output_dir
        record.append_event("one", {})
        self.finish(record)
        for name in ("manifest.initial.json", "events.jsonl"):
            raw = (out / name).read_bytes()
            (out / name).write_bytes(raw + b" ")
            with self.assertRaises(ValueError):
                verify_artifacts(out)
            (out / name).write_bytes(raw)

    def test_malformed_event_state_is_not_rewritten_or_accepted(self):
        record = self.create()
        events = record.output_dir / "events.jsonl"
        events.write_bytes(b'{"sequence":2,"utc":"bad","kind":"x","payload":{}}\n')
        original = events.read_bytes()
        with self.assertRaises(ValueError):
            record.append_event("next", {})
        self.assertEqual(events.read_bytes(), original)
        with self.assertRaises(ValueError):
            record.finish("completed", counts={}, artifacts=[])
        self.assertFalse((record.output_dir / "manifest.json").exists())

    def test_finish_rejects_valid_shape_edits_to_initial_evidence(self):
        record = self.create()
        initial_path = record.output_dir / "manifest.initial.json"
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
        initial["command"] = ["python", "different.py"]
        initial_path.write_text(json.dumps(initial), encoding="utf-8")
        changed = initial_path.read_bytes()
        with self.assertRaises(ValueError):
            self.finish(record)
        self.assertEqual(initial_path.read_bytes(), changed)
        self.assertFalse((record.output_dir / "manifest.json").exists())

    def test_terminal_fsync_failure_seals_writer_and_preserves_partial_terminal(self):
        record = self.create()
        record.append_event("before_finish", {})
        events = record.output_dir / "events.jsonl"
        original_events = events.read_bytes()
        with mock.patch("dwr.experiments.records.os.fsync", side_effect=OSError("simulated fsync failure")):
            with self.assertRaisesRegex(OSError, "simulated fsync failure"):
                self.finish(record)
        terminal = record.output_dir / "manifest.json"
        self.assertTrue(terminal.exists())
        terminal_bytes = terminal.read_bytes()
        with self.assertRaises(ValueError):
            record.append_event("must_not_append", {})
        with self.assertRaises(ValueError):
            self.finish(record)
        self.assertEqual(events.read_bytes(), original_events)
        self.assertEqual(terminal.read_bytes(), terminal_bytes)

    def test_verifier_requires_terminal_start_to_match_initial_start(self):
        record = self.create()
        self.finish(record)
        terminal_path = record.output_dir / "manifest.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["started_utc"] = "2000-01-01T00:00:00+00:00"
        terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_artifacts(record.output_dir)

    def test_verifier_rejects_unknown_initial_and_terminal_fields(self):
        record = self.create()
        self.finish(record)
        initial_path = record.output_dir / "manifest.initial.json"
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
        initial["unregistered"] = True
        initial_path.write_text(json.dumps(initial), encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_artifacts(record.output_dir)

        initial.pop("unregistered")
        initial_path.write_text(json.dumps(initial), encoding="utf-8")
        terminal_path = record.output_dir / "manifest.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["unregistered"] = True
        terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_artifacts(record.output_dir)

    def test_no_symlink_or_reparse_parent_paths(self):
        link = self.root / "link"
        try:
            link.symlink_to(self.work, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlinks unavailable")
        with self.assertRaises(ValueError):
            RunRecord.create(link / "run", **self.metadata())
        record = self.create()
        outside = self.root / "external.txt"
        outside.write_text("x", encoding="utf-8")
        (record.output_dir / "alias.txt").symlink_to(outside)
        with self.assertRaises(ValueError):
            record.finish("completed", counts={}, artifacts=["alias.txt"])


if __name__ == "__main__":
    unittest.main()
