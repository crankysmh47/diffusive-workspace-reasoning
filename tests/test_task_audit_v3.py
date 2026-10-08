import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from dwr.tasks.audit_v2 import run_audit_v3
from dwr.tasks.config_v3 import load_config_v3
from dwr.tasks.dataset_v2 import DatasetLimits, run_dataset_v3, _source_closure_fixture
from dwr.tasks.run_support import ResourceGuard

ROOT = Path(__file__).resolve().parents[1]


class AuditV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config_v3(ROOT / "configs/tasks/constraint_v3.json")

    def test_closed_source_no_go_is_not_reported_as_measured_target_failure(self):
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        limits = DatasetLimits({"eq": 0, "neq": 1, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 4, 30, 1024)
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "run"
            from unittest.mock import patch
            from dwr.tasks.generator_v3 import generate_candidate_v3
            from dwr.tasks.dataset_v2 import _split_and_ids
            from dwr.tasks.stream_contract import _STREAMS
            selected = None
            for index in range(100):
                outcome = generate_candidate_v3(20261007, "source", index)
                if outcome.oracle_unique and outcome.eligible and _split_and_ids(outcome)[0] != "source_train":
                    selected = replace(outcome, candidate_index=0)
                    break
            self.assertIsNotNone(selected, "fixture search must find a valid held-out source split")
            fixture_candidate = lambda seed, partition, index, *, full_profiles=frozenset(): selected
            with _source_closure_fixture(limits), patch.dict(_STREAMS, {
                    "dwr-generator-v3": replace(_STREAMS["dwr-generator-v3"], candidate_fn=fixture_candidate)}), \
                 patch("dwr.tasks.generator_v3.generate_candidate_v3", side_effect=fixture_candidate):
                run_dataset_v3(self.config, data, "feasibility", ["fixture"],
                               limits=limits, source=source)
            with patch("dwr.tasks.dataset_v2.registered_limits", return_value=limits), \
                 patch.dict(_STREAMS, {"dwr-generator-v3": replace(
                     _STREAMS["dwr-generator-v3"], candidate_fn=fixture_candidate)}):
                report = run_audit_v3(data, self.config, source, "feasibility",
                                      guard=ResourceGuard(30, 1024, memory_reader=lambda: 0))
            from dwr.tasks.audit_cli_v3 import main as audit_main
            with patch("dwr.tasks.dataset_v2.registered_limits", return_value=limits), \
                 patch("dwr.tasks.audit_cli_v2.registered_limits", return_value=limits), \
                 patch("dwr.tasks.audit_cli_v2.source_identity", return_value=source), \
                 patch.dict(_STREAMS, {"dwr-generator-v3": replace(
                     _STREAMS["dwr-generator-v3"], candidate_fn=fixture_candidate)}):
                self.assertEqual(audit_main(["--config", str(ROOT / "configs/tasks/constraint_v3.json"),
                                             "--mode", "feasibility", "--data", str(data)]), 0)
            persisted = json.loads((data / "audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "complete")
            self.assertEqual(persisted["g2_status"], "not_assessed")
            self.assertFalse((data / "preflight_report.json").exists())
        self.assertEqual(report["audit_version"], "dwr-task-v3-audit-v1")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["g2_status"], "not_assessed", report)
        self.assertEqual(report["proposal_status"], "NO_GO")
        self.assertEqual(report["proposal_reason"], "source_support_capacity_bound")
        self.assertEqual(report["capacity_bound"]["status"], "NO_GO")
        self.assertIsNone(report["four_cohort_common_support_proxy"])
        self.assertEqual(report["final_assessment"], "not_assessed")

    def test_wrong_stream_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises((ValueError, FileNotFoundError)):
                run_audit_v3(Path(temp), self.config, {}, "feasibility")
