"""Strict export reader tests over genuine run-record artifacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from dwr.experiments.records import RunRecord
from dwr.experiments.source_reader import SourceReaderError, read_source_export
from dwr.tasks import canonical, schema, solver
from dwr.tasks.run_support import stable_json

from test_source_corpus import _task


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=10).stdout


def _line(value) -> bytes:
    return (stable_json(value) + "\n").encode("utf-8")


class SourceReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        _git(self.root, "init", "-q")
        _git(self.root, "config", "user.name", "Source Reader Fixture")
        _git(self.root, "config", "user.email", "fixture@example.invalid")
        (self.root / "runs").mkdir()
        self.config_path = self.root / "config.json"
        self.config = {"config_version": "dwr-source-export-v1",
                       "operation": "source-only-retained-corpus-conversion",
                       "expected_counts": {"source_train": 1, "source_validation": 1},
                       "generation_config_sha256": "b" * 64,
                       "artifact_sha256": {name: hashlib.sha256(name.encode()).hexdigest() for name in (
                           "manifest.initial.json", "manifest.json", "outcomes.jsonl", "records.jsonl", "replay.json", "report.json")}}
        self.config_bytes = _line(self.config)
        self.config_path.write_bytes(self.config_bytes)
        self.run_number = 0
        _git(self.root, "add", "config.json")
        _git(self.root, "commit", "-qm", "reader fixture")

    def tearDown(self):
        self.temp.cleanup()

    def _row(self, split, index, *, task=None, solution=None, example_id=None, profile=None):
        if task is None:
            task = _task("eq" if index == 0 else "add_mod")
        if not hasattr(task, "variables"):
            task = _task("eq" if index == 0 else "add_mod")
        profile = profile or ("eq" if index == 0 else "add_mod")
        solved = solver.solve(task)
        solution = {"assignment": dict(solved.solution.assignments)} if solution is None else solution
        return {
            "schema_version": "dwr-source-example-v1",
            "example_id": example_id or canonical.content_id(task),
            "split": split,
            "lineage_id": canonical.lineage_id(task),
            "structure_signature": canonical.structure_signature(task),
            "fine_stratum": "A2",
            "profile": profile,
            "model_input_sha256": _sha(schema.serialize_model_input(task).encode("utf-8")),
            "task": schema.serialize_task(task),
            "solution": solution,
            "origin": {"partition": "source", "candidate_index": index,
                       "outcome_sha256": hashlib.sha256(f"outcome-{index}".encode()).hexdigest()},
        }

    def _export(self, rows=None, *, summary_overrides=None):
        rows = [self._row("source_train", 0), self._row("source_validation", 1)] if rows is None else rows
        self.run_number += 1
        run_dir = self.root / "runs" / f"reader-fixture-{self.run_number}"
        counts = {split: sum(row["split"] == split for row in rows)
                  for split in ("source_train", "source_validation")}
        run_config = {**self.config, "expected_counts": counts}
        config_path = self.root / f"config-{self.run_number}.json"
        config_bytes = _line(run_config)
        config_path.write_bytes(config_bytes)
        record = RunRecord.create(run_dir, command=["python", "reader-fixture"], config_path=config_path,
            expected_config_sha256=_sha(config_bytes),
            identities={"source": {"name": "fixture", "revision": "v1", "sha256": "a" * 64}},
            rng_streams={"unused": 1}, settings={"operation": "source-only-retained-corpus-conversion"},
            budgets={"wall_seconds": 10, "memory_mib": 128, "attempts": 1}, source_root=self.root)
        output_raw = b"".join(_line(row) for row in rows)
        output_hash = _sha(output_raw)
        (run_dir / "source_examples.jsonl").write_bytes(output_raw)
        summary = {
            "schema_version": "dwr-source-export-summary-v1",
            "operation": "source-only-retained-corpus-conversion",
            "exported_counts": counts,
            "fine_stratum_counts": {"A2": len(rows)},
            "exported_records": len(rows),
            "allowed_task_payloads_parsed": len(rows),
            "protected_task_payloads_semantically_parsed": 0,
            "skipped_raw_rows": 0,
            "construction_failed_null_split_rows": 0,
            "input_sha256": {name: hashlib.sha256(name.encode()).hexdigest() for name in (
                "manifest.initial.json", "manifest.json", "outcomes.jsonl", "records.jsonl", "replay.json", "report.json")},
            "generation_config_sha256": "b" * 64,
            "output_sha256": output_hash,
            "access": {"permitted_payload_splits": ["source_train", "source_validation"],
                "forbidden_payload_splits": ["source_test", "development_validation", "development_diagnostic_test", "final_test"],
                "full_file_opaque_hashes": True,
                "raw_json_parsing": "full outcomes JSONL strict parse materializes row fields; route first, then skip task, witness and values semantics outside permitted source splits",
                "protected_task_payloads_semantically_parsed": 0},
            "costs": {"total_wall_seconds": 1.0, "total_cpu_seconds": 0.5, "peak_rss_bytes": 1024,
                "artifact_write_wall_seconds": None, "artifact_write_cpu_seconds": None, "flops": None, "gpu": None},
        }
        if summary_overrides:
            summary.update(summary_overrides)
        (run_dir / "summary.json").write_bytes(_line(summary))
        record.finish("completed", counts={**counts, "exported_records": len(rows)},
                      artifacts=["source_examples.jsonl", "summary.json"])
        terminal_raw = (run_dir / "manifest.json").read_bytes()
        return run_dir, _sha(terminal_raw)

    def test_returns_only_requested_split_in_export_order(self):
        train = self._row("source_train", 0)
        validation = self._row("source_validation", 1)
        run_dir, manifest_hash = self._export([train, validation])
        result = read_source_export(run_dir, expected_manifest_sha256=manifest_hash, split="source_train")
        self.assertEqual(result, (train,))

    def test_nonrequested_task_and_solution_are_not_semantically_parsed(self):
        train = self._row("source_train", 0)
        validation = self._row("source_validation", 1)
        validation["task"] = "poison task"
        validation["solution"] = ["poison gold"]
        run_dir, manifest_hash = self._export([train, validation])
        result = read_source_export(run_dir, expected_manifest_sha256=manifest_hash, split="source_train")
        self.assertEqual(result, (train,))

    def test_rejects_duplicate_example_ids_independently_of_origin_uniqueness(self):
        first = self._row("source_train", 0)
        second = self._row("source_validation", 1, example_id=first["example_id"])
        run_dir, manifest_hash = self._export([first, second])
        with self.assertRaisesRegex(SourceReaderError, "duplicate source example id"):
            read_source_export(run_dir, expected_manifest_sha256=manifest_hash, split="source_train")

    def test_unknown_split_and_wrong_caller_manifest_hash_fail(self):
        run_dir, manifest_hash = self._export()
        with self.assertRaises(SourceReaderError):
            read_source_export(run_dir, expected_manifest_sha256=manifest_hash, split="source_test")
        with self.assertRaises(SourceReaderError):
            read_source_export(run_dir, expected_manifest_sha256="0" * 64, split="source_train")

    def test_terminal_artifact_tampering_and_summary_counts_fail(self):
        run_dir, manifest_hash = self._export()
        with (run_dir / "source_examples.jsonl").open("ab") as stream:
            stream.write(b"{}\n")
        with self.assertRaises(ValueError):
            read_source_export(run_dir, expected_manifest_sha256=manifest_hash, split="source_train")

        run_dir, manifest_hash = self._export(summary_overrides={"exported_counts": {
            "source_train": 2, "source_validation": 1}})
        with self.assertRaises(SourceReaderError):
            read_source_export(run_dir, expected_manifest_sha256=manifest_hash, split="source_train")

        access = {"permitted_payload_splits": ["source_train", "source_validation"],
            "forbidden_payload_splits": ["source_test", "development_validation", "development_diagnostic_test", "final_test"],
            "full_file_opaque_hashes": True,
            "raw_json_parsing": "full outcomes JSONL strict parse materializes row fields; route first, then skip task, witness and values semantics outside permitted source splits",
            "protected_task_payloads_semantically_parsed": False}
        run_dir, manifest_hash = self._export(summary_overrides={"access": access})
        with self.assertRaises(SourceReaderError):
            read_source_export(run_dir, expected_manifest_sha256=manifest_hash, split="source_train")

    def test_duplicate_origin_identity_task_profile_and_solution_fail(self):
        rows = [self._row("source_train", 0),
                self._row("source_validation", 0, example_id="c" * 64)]
        run_dir, manifest_hash = self._export(rows)
        with self.assertRaisesRegex(SourceReaderError, "duplicate source origin"):
            read_source_export(run_dir, expected_manifest_sha256=manifest_hash, split="source_train")

        bad = self._row("source_train", 0, example_id="c" * 64)
        run_dir, manifest_hash = self._export([bad, self._row("source_validation", 1)])
        with self.assertRaisesRegex(SourceReaderError, "example_id mismatch"):
            read_source_export(run_dir, expected_manifest_sha256=manifest_hash, split="source_train")

        bad = self._row("source_train", 0, profile="neq")
        run_dir, manifest_hash = self._export([bad, self._row("source_validation", 1)])
        with self.assertRaisesRegex(SourceReaderError, "primitive profile"):
            read_source_export(run_dir, expected_manifest_sha256=manifest_hash, split="source_train")

        bad_solution = {"assignment": {"v0": 2, "v1": 1, "v2": 1}}
        bad = self._row("source_train", 0, solution=bad_solution)
        run_dir, manifest_hash = self._export([bad, self._row("source_validation", 1)])
        with self.assertRaisesRegex(SourceReaderError, "verifier rejected"):
            read_source_export(run_dir, expected_manifest_sha256=manifest_hash, split="source_train")


if __name__ == "__main__":
    unittest.main()
