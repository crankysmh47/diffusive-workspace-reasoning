import unittest
import tempfile
import json
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

from dwr.tasks.config_v3 import load_config_v3
from dwr.tasks.dataset_v2 import DatasetLimits, run_dataset_v3, _source_closure_fixture, _split_and_ids
from dwr.tasks.generator_v3 import generate_candidate_v3
from dwr.tasks.run_support import ResourceGuard, remaining_global_seconds

ROOT = Path(__file__).resolve().parents[1]


class BudgetV3Tests(unittest.TestCase):
    def test_generation_elapsed_is_removed_once_from_combined_budget(self):
        remaining = remaining_global_seconds({"generation_elapsed_seconds": 247.25}, 600.0)
        self.assertEqual(remaining, 352.75)
        guard = ResourceGuard(600.0, 1024, clock=lambda: 5.0, memory_reader=lambda: 0)
        guard.started = 0.0
        guard.wall_seconds = min(guard.wall_seconds, remaining)
        self.assertEqual(guard.wall_seconds, 352.75)
        self.assertEqual(guard.elapsed, 5.0)
        # The generation time is not deducted again during a prefix replay/second check.
        guard.wall_seconds = min(guard.wall_seconds, remaining)
        self.assertEqual(guard.wall_seconds, 352.75)

    def test_terminal_budget_check_fails_after_guard_limit(self):
        clock_values = iter((0.0, 0.0, 1.0))
        guard = ResourceGuard(0.5, 1024, clock=lambda: next(clock_values), memory_reader=lambda: 0)
        guard.started = 0.0
        guard.check()
        with self.assertRaises(TimeoutError):
            guard.check()

    def test_cli_terminal_resource_fault_downgrades_report_gate_and_preflight(self):
        from dwr.tasks.audit_cli_v3 import main as audit_main
        from dwr.tasks import audit_cli_v2
        from dwr.tasks.stream_contract import _STREAMS
        config = load_config_v3(ROOT / "configs/tasks/constraint_v3.json")
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        limits = DatasetLimits({"eq": 0, "neq": 1, "add_mod": 0},
            {("eq", "neq"): 0, ("eq", "add_mod"): 0, ("neq", "add_mod"): 0}, {}, 4, 30, 1024)
        selected = None
        for index in range(100):
            candidate = generate_candidate_v3(20261007, "source", index)
            if candidate.oracle_unique and candidate.eligible and _split_and_ids(candidate)[0] != "source_train":
                selected = replace(candidate, candidate_index=0)
                break
        self.assertIsNotNone(selected)
        fixture_candidate = lambda seed, partition, index, *, full_profiles=frozenset(): selected
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "run"
            with _source_closure_fixture(limits), patch.dict(_STREAMS, {"dwr-generator-v3": replace(
                    _STREAMS["dwr-generator-v3"], candidate_fn=fixture_candidate)}), \
                 patch("dwr.tasks.generator_v3.generate_candidate_v3", side_effect=fixture_candidate):
                run_dataset_v3(config, data, "feasibility", ["fixture"], limits=limits, source=source)
            now = [0.0]
            rss = [0]
            charged_operations = [0]
            terminal_rss = [0]
            guard_factory = lambda wall, memory: ResourceGuard(
                wall, memory, clock=lambda: now[0], memory_reader=lambda: rss[0])
            original_atomic_json = audit_cli_v2.atomic_json
            original_sha256_file = audit_cli_v2.sha256_file
            advanced = [False]
            def charged_write(path, value):
                original_atomic_json(path, value)
                charged_operations[0] += 1
                rss[0] += 1024
                now[0] += 0.01
                if (Path(path).name == "audit_manifest.json" and value.get("status") == "complete"
                        and not advanced[0]):
                    advanced[0] = True
                    now[0] = 10000.0
                    terminal_rss[0] = rss[0]
            def charged_hash(path):
                digest = original_sha256_file(path)
                charged_operations[0] += 1
                rss[0] += 1024
                now[0] += 0.01
                return digest
            with patch.dict(_STREAMS, {"dwr-generator-v3": replace(
                    _STREAMS["dwr-generator-v3"], candidate_fn=fixture_candidate)}), \
                 patch("dwr.tasks.dataset_v2.registered_limits", return_value=limits), \
                 patch("dwr.tasks.audit_cli_v2.registered_limits", return_value=limits), \
                 patch("dwr.tasks.audit_cli_v2.source_identity", return_value=source), \
                 patch("dwr.tasks.audit_cli_v2.ResourceGuard", side_effect=guard_factory), \
                 patch("dwr.tasks.audit_cli_v2.atomic_json", side_effect=charged_write), \
                 patch("dwr.tasks.audit_cli_v2.sha256_file", side_effect=charged_hash):
                code = audit_main(["--config", str(ROOT / "configs/tasks/constraint_v3.json"),
                                   "--mode", "feasibility", "--data", str(data)])
            report = json.loads((data / "audit_report.json").read_text(encoding="utf-8"))
            gate = json.loads((data / "audit_gate.json").read_text(encoding="utf-8"))
            manifest = json.loads((data / "audit_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertTrue(advanced[0], "the terminal manifest must be charged before the final guard check")
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["g2_status"], "not_assessed")
            self.assertEqual(gate["status"], "incomplete")
            self.assertEqual(manifest["error"]["type"], "TimeoutError")
            self.assertIn("TimeoutError", [item["type"] for item in manifest["finalization_errors"]])
            self.assertGreater(manifest["audit_elapsed_seconds"], 9999.0)
            self.assertEqual(manifest["peak_resident_bytes"], terminal_rss[0])
            self.assertFalse((data / "preflight_report.json").exists())
