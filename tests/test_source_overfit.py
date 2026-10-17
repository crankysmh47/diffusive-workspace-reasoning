"""Closed controller checks on tiny native source-training fixtures."""
from __future__ import annotations

import hashlib
import json
import ctypes
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import time

from dwr.experiments import source_overfit

try:
    import numpy as _numpy
except ModuleNotFoundError as _numpy_error:
    if _numpy_error.name != "numpy": raise
    _numpy = None
try:
    import torch as _torch
except ModuleNotFoundError as _torch_error:
    if _torch_error.name != "torch":
        raise
    _torch = None
_ML_READY = _torch is not None and _numpy is not None


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class SourceOverfitRegistrationTests(unittest.TestCase):
    def test_production_source_identity_accepts_clean_descendant_of_registered_base(self):
        source = {"commit": "b" * 40, "clean": True, "files": [{"path": "dwr/a.py", "sha256": "a" * 64}]}
        self.assertEqual(source_overfit._validate_release_source(source, "a" * 40), source)
        with self.assertRaisesRegex(ValueError, "clean|source"):
            source_overfit._validate_release_source({**source, "clean": False}, "registered-base")

    def test_reference_manifest_pin_rejects_valid_but_unregistered_terminal(self):
        from dwr.experiments.records import RunRecord, verify_artifacts
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s); config_path = root/"config.json"
            config_raw = b"{}\n"; config_path.write_bytes(config_raw)
            run = root/"reference-001"
            identities = {name: {"name": name, "revision": "fixture", "sha256": char*64}
                for name, char in zip(("source", "config", "data", "model", "tokenizer", "schedule"), "abcdef")}
            record = RunRecord.create(run, command=["fixture-reference"], config_path=config_path,
                expected_config_sha256=hashlib.sha256(config_raw).hexdigest(), identities=identities,
                rng_streams={"python": 11, "numpy": 12}, settings={"fixture": True},
                budgets={"wall_seconds": 10, "memory_mib": 100, "attempts": 1})
            summary = {"status": "PASS_FROZEN_REFERENCE_OPERATIONAL_V1", "attempt_count": 45,
                "quality": {"denominator": 15}, "repetitions": {"0": {"examples": 15, "valid_count": 0},
                "1": {"examples": 15, "valid_count": 0}, "2": {"examples": 15, "valid_count": 0}}}
            summary_raw = _canonical(summary)+b"\n"
            (run/"summary.json").write_bytes(summary_raw)
            record.finish("completed", counts={}, artifacts=["summary.json"])
            self.assertEqual(verify_artifacts(run)["status"], "completed")
            expected_summary_sha = hashlib.sha256(summary_raw).hexdigest()
            expected_manifest_sha = hashlib.sha256((run/"manifest.json").read_bytes()).hexdigest()
            _, loaded_raw, loaded = source_overfit._verified_reference_proofs(run,
                {"reference_manifest_sha256": expected_manifest_sha,
                 "reference_summary_sha256": expected_summary_sha})
            self.assertEqual(loaded_raw, summary_raw)
            self.assertEqual(loaded, summary)
            with self.assertRaisesRegex(ValueError, "terminal manifest differs"):
                source_overfit._verified_reference_proofs(run,
                    {"reference_manifest_sha256": "f"*64,
                     "reference_summary_sha256": expected_summary_sha})

    def test_token_audit_joins_closed_reader_rows_by_id(self):
        reader_rows = [{"example_id": "a" * 64, "split": "source_train", "profile": "eq",
                        "fine_stratum": "A2"}]
        token_rows = [{"example_id": "a" * 64, "split": "source_train", "profile": "eq",
                       "fine_stratum": "A2", "prompt_tokens": 7, "answer_tokens_including_eos": 3,
                       "total_tokens": 10, "within_caps": True}]
        joined = source_overfit._join_token_audit(reader_rows, token_rows, expected_count=1,
                                                   max_answer_tokens=4)
        self.assertEqual(joined["a" * 64]["prompt_tokens"], 7)
        self.assertNotIn("prompt_tokens", reader_rows[0])

    def test_host_policy_uses_registered_initialization_seeds(self):
        host_config = source_overfit._resolved_host_config(
            {"seeds": {"workspace_initialization": 20261026, "reader_initialization": 20261027}},
            {"seeds": {"workspace_initialization": 20261205, "reader_initialization": 20261206,
                       "python_global": 20261201, "numpy_global": 20261202,
                       "torch_cpu_global": 20261203, "torch_cuda_global": 20261204}})
        self.assertEqual((host_config["seeds"]["workspace_initialization"], host_config["seeds"]["reader_initialization"]),
                         (20261205, 20261206))

    @unittest.skipUnless(_ML_READY, "registered RNG contract requires optional NumPy/Torch")
    def test_registered_global_rngs_are_installed(self):
        seeds = {"python": 20261201, "numpy": 20261202, "torch_cpu": 20261203, "torch_cuda": 20261204}
        with patch("random.seed") as py_seed, patch("numpy.random.seed") as np_seed, \
                patch("torch.manual_seed") as cpu_seed, patch("torch.cuda.manual_seed_all") as cuda_seed:
            source_overfit._install_global_rngs(seeds)
        py_seed.assert_called_once_with(seeds["python"])
        np_seed.assert_called_once_with(seeds["numpy"])
        cpu_seed.assert_called_once_with(seeds["torch_cpu"])
        cuda_seed.assert_called_once_with(seeds["torch_cuda"])

    @unittest.skipUnless(_ML_READY, "memory-cap fixture requires optional Torch")
    def test_windows_memory_cap_uses_high_water_and_fails_closed(self):
        if __import__("os").name != "nt":
            self.skipTest("Windows PROCESS_MEMORY_COUNTERS contract")
        psapi = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        kernel32 = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        def dll(name, **_):
            return psapi if name == "psapi" else kernel32
        def fill_counters(_process, pointer, _size):
            counters = pointer._obj
            counters.WorkingSetSize = 1 * 1024**2
            counters.PeakWorkingSetSize = 3 * 1024**2
            return 1
        psapi.GetProcessMemoryInfo.side_effect = fill_counters
        with patch.object(ctypes, "WinDLL", side_effect=dll):
            self.assertEqual(source_overfit._process_resident_mib(), 3.0)
        psapi.GetProcessMemoryInfo.side_effect = lambda *_: 0
        with patch.object(ctypes, "WinDLL", side_effect=dll):
            with self.assertRaisesRegex(OSError, "memory cap is unverifiable"):
                source_overfit._process_resident_mib()
        torch = __import__("torch")
        cuda_initialized_before = torch.cuda.is_initialized()
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(source_overfit, "_process_resident_mib", side_effect=OSError("unavailable")):
            with self.assertRaisesRegex(OSError, "unavailable"):
                source_overfit._measured_runtime_costs(object(), {"limits": {}})
        self.assertEqual(torch.cuda.is_initialized(), cuda_initialized_before,
            "CPU memory fixture must not initialize CUDA")

    def test_resume_pointer_is_exact_kernel_descriptor(self):
        pointer = {"jobroot": "job", "cursor_path": "job/recovery.latest.json", "cursor_sha256": "a" * 64}
        self.assertEqual(source_overfit._kernel_resume_pointer(pointer), pointer)
        self.assertEqual(source_overfit._kernel_resume_pointer({**pointer, "phase": "training"}), pointer)

    def test_kernel_identities_match_registered_canonical_recipes(self):
        config = {
            "config_version": "fixture-registration",
            "data": {
                "source_export_manifest_sha256": "1" * 64,
                "source_export_data_sha256": "2" * 64,
                "source_token_audit_manifest_sha256": "3" * 64,
                "token_lengths_sha256": "4" * 64,
            },
            "dependencies": {
                "cohort_sha256": "5" * 64,
                "feasibility_manifest_sha256": "6" * 64,
                "feasibility_outcome_sha256": "7" * 64,
                "reference_manifest_sha256": "8" * 64,
                "reference_summary_sha256": "9" * 64,
            },
            "training": {"schedule": {"type": "LambdaLR", "warmup_steps": 8}},
        }
        raw = b'{"fixture":"registered bytes"}\n'
        identities = source_overfit._kernel_identities(
            raw, config, source_commit="abc123", weight_sha256="a" * 64,
            tokenizer_inventory_sha256="b" * 64)
        data_fields = {
            "source_export_manifest_sha256": "1" * 64,
            "source_export_data_sha256": "2" * 64,
            "source_token_audit_manifest_sha256": "3" * 64,
            "token_lengths_sha256": "4" * 64,
            "cohort_sha256": "5" * 64,
            "feasibility_manifest_sha256": "6" * 64,
            "feasibility_outcome_sha256": "7" * 64,
            "reference_manifest_sha256": "8" * 64,
            "reference_summary_sha256": "9" * 64,
        }
        self.assertEqual(set(identities), {"source", "config", "data", "model", "tokenizer", "schedule"})
        self.assertEqual(identities["source"], hashlib.sha256(b"abc123").hexdigest())
        self.assertEqual(identities["config"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(identities["data"], hashlib.sha256(_canonical(data_fields)).hexdigest())
        self.assertEqual(identities["model"], "a" * 64)
        self.assertEqual(identities["tokenizer"], "b" * 64)
        self.assertEqual(identities["schedule"], hashlib.sha256(_canonical(config["training"]["schedule"])).hexdigest())

    def test_source_selection_uses_registered_order_without_refill(self):
        selected = [
            {"example_id": "a" * 64, "split": "source_train", "profile": "eq", "fine_stratum": "A2"},
            {"example_id": "b" * 64, "split": "source_train", "profile": "neq", "fine_stratum": "B2"},
            {"example_id": "c" * 64, "split": "source_train", "profile": "add_mod", "fine_stratum": "C2"},
        ]
        rows = [dict(item) for item in selected[::-1]]
        rows.append({"example_id": "d" * 64, "split": "source_train", "profile": "eq", "fine_stratum": "D2"})
        ordered = source_overfit._select_training_rows(rows, selected)
        self.assertEqual([row["example_id"] for row in ordered], ["a" * 64, "b" * 64, "c" * 64])
        missing = [row for row in rows if row["example_id"] != "c" * 64]
        with self.assertRaisesRegex(ValueError, "every|selected|exactly"):
            source_overfit._select_training_rows(missing, selected)

    def test_production_entry_rejects_other_config_and_mutable_flags_before_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong = Path(tmp) / "config.json"
            wrong.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                source_overfit.run_source_overfit(wrong, command=["test"], segment_index=0)
        with self.assertRaises(TypeError):
            source_overfit.run_source_overfit(
                source_overfit.CONFIG_PATH, command=["test"], segment_index=0, learning_rate=1.0)


class SourceOverfitControllerTests(unittest.TestCase):
    @unittest.skipIf(_torch is None, "optional ML stack unavailable")
    @unittest.skipUnless(_ML_READY, "native controller fixture requires optional NumPy/Torch")
    def test_controller_runs_real_kernel_and_seals_real_run_record(self):
        from test_source_overfit_learning import _fixtures, RecordingTokenizer, IDS
        from dwr.experiments.records import verify_artifacts
        from dwr.tasks.schema import serialize_task
        from dwr.training.batches import TokenizerIdentity

        host, examples, config, metadata = _fixtures(390)
        config["training"]["schedule"]["type"] = "LambdaLR"
        model = {"repo_id": "fixture/model", "revision": "fixture-revision",
                 "weight_sha256": "a" * 64, "artifact_inventory_sha256": "b" * 64}
        config["data"] = {"source_export_manifest_sha256": "1" * 64,
            "source_export_data_sha256": "2" * 64, "source_token_audit_manifest_sha256": "3" * 64,
            "token_lengths_sha256": "4" * 64, "source_split": "source_train",
            "reader_expected_records": 3, "selected_examples": 3}
        config["dependencies"] = {key: value for key, value in {
            "cohort_sha256": "5" * 64, "feasibility_manifest_sha256": "6" * 64,
            "feasibility_outcome_sha256": "7" * 64, "reference_manifest_sha256": "8" * 64,
            "reference_summary_sha256": "9" * 64}.items()}
        config["config_version"] = "fixture-registration"
        config["model"] = model
        config["source_base_commit"] = "fixture-source"
        config["limits"] = {"wall_seconds": 120, "resident_memory_mib": 2048, "attempts": 1,
                            "work_seconds": 110}
        config["recovery"] = {"heartbeat_seconds": 30}
        config["evaluation"]["gate_max_final_to_initial_CE_ratio"] = 1e9
        raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        selected = []
        rows = []
        token_rows = []
        for example, kind in zip(examples, ("eq", "neq", "add_mod")):
            entry = {"example_id": example.example_id, "split": "source_train", "profile": kind,
                     "fine_stratum": metadata[example.example_id]["fine_stratum"]}
            selected.append(entry)
            from dwr.tasks import serialize_model_input
            prompt_count = len(RecordingTokenizer().encode(serialize_model_input(example.task) + "\n",
                                                            add_special_tokens=False))
            assignment = dict(example.solution.assignments)
            rows.append({**entry, "task": serialize_task(example.task),
                "solution": {"assignment": assignment},
                "lineage_id": metadata[example.example_id]["lineage_id"],
                "structure_signature": metadata[example.example_id]["structure_signature"]})
            token_rows.append({**entry, "prompt_tokens": prompt_count, "answer_tokens_including_eos": 2,
                "total_tokens": prompt_count + 2, "within_caps": True})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); config_path = base / "fixture.json"; config_path.write_bytes(raw)
            jobroot = base / "job"; jobroot.mkdir()
            registration = {"config": config, "config_raw": raw, "config_path": config_path,
                "config_sha256": hashlib.sha256(raw).hexdigest(), "selected": tuple(selected),
                "feasibility_config": {"fixture": True, "model": model}}
            authorization = {"jobroot": str(jobroot), "segment_index": 0,
                "worker_pid": 123, "worker_creation_time": "fixture", "supervisor_pid": 456,
                "supervisor_creation_time": "fixture", "resume_descriptor": None}
            seams = {
                "source": lambda: {"commit": "fixture-source", "clean": True, "files": []},
                "feasibility": lambda: {"outcome": {"status": "PASS_FEASIBILITY"}},
                "reference": lambda: ({"status": "completed"}, b'{"status":"PASS_FROZEN_REFERENCE_OPERATIONAL_V1"}',
                    {"status": "PASS_FROZEN_REFERENCE_OPERATIONAL_V1", "attempt_count": 45,
                     "quality": {"denominator": 15}, "repetitions": {
                        "0": {"examples": 15, "valid_count": 0},
                        "1": {"examples": 15, "valid_count": 0},
                        "2": {"examples": 15, "valid_count": 0}}}),
                "data_preconditions": lambda: {"task_semantics_read": False},
                "artifact_inventory": lambda: {"weight_sha256": model["weight_sha256"],
                    "inventory_sha256": model["artifact_inventory_sha256"]},
                "rows": rows, "token_rows": token_rows, "tokenizer": RecordingTokenizer(), "host": host,
                "original_hashes": {"original": "stable"},
                "original_hashes_after": {"original": "stable"},
                "training_config": config, "environment": {"fixture": True},
                "check": lambda: None}
            result = source_overfit._execute_registered(Path(__file__).resolve().parents[1],
                registration, authorization, command=["fixture-worker"], _test_seams=seams)
            terminal = verify_artifacts(Path(result["record_dir"]))
            self.assertEqual(terminal["status"], "completed")
            self.assertIn(result["status"], {"PASS_TINY_SOURCE_OVERFIT", "NO_GO_TINY_SOURCE_OVERFIT"})
            self.assertEqual(result["net_optimizer_steps"], 9)
            self.assertTrue((Path(result["record_dir"]) / "updates.jsonl").is_file())
            self.assertEqual([row["example_id"] for row in result["predictions"][:3]], list(IDS))

            failed_jobroot = base / "job-failed"; failed_jobroot.mkdir()
            failed_authorization = {**authorization, "jobroot": str(failed_jobroot)}
            failed_host, _, _, _ = _fixtures(391)
            saw_update = False
            def observe(kind, _payload):
                nonlocal saw_update
                if kind == "update": saw_update = True
            def fail_after_update():
                if saw_update: raise RuntimeError("fixture process-budget interruption")
            failed_seams = {**seams, "host": failed_host, "original_hashes": {"original": "stable"},
                "original_hashes_after": {"original": "stable"}, "observe_kernel_event": observe,
                "check": fail_after_update}
            with self.assertRaisesRegex(RuntimeError, "fixture process-budget interruption"):
                source_overfit._execute_registered(Path(__file__).resolve().parents[1], registration,
                    failed_authorization, command=["fixture-worker"], _test_seams=failed_seams)
            failed_segment = failed_jobroot / "segment-000"
            failed_terminal = verify_artifacts(failed_segment)
            self.assertEqual(failed_terminal["status"], "failed")
            updates = failed_segment / "updates.jsonl"
            self.assertTrue(updates.is_file())
            self.assertIn('"role":"main"', updates.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
