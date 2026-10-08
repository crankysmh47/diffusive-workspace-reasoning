import tempfile
import unittest
import contextlib
import io
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from dwr.tasks.config_v3 import load_config_v3
from dwr.tasks.cli_v3 import main as cli_v3_main
from dwr.tasks.dataset_v2 import DatasetLimits, accepted_record, outcome_row, run_dataset_v3, _split_and_ids
from dwr.tasks.generator_v2 import generate_candidate
from dwr.tasks.generator_v3 import generate_candidate_v3
from dwr.tasks.generator_v2 import CandidatePrivateEvidence, SourceEligibilityEvidence
from dwr.tasks.replay_v2 import replay_files_v3


ROOT = Path(__file__).resolve().parents[1]


class DatasetV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config_v3(ROOT / "configs/tasks/constraint_v3.json")

    def test_v3_outcome_schema_is_explicit_and_rejects_a_v2_candidate(self):
        outcome = generate_candidate_v3(20261007, "source", 0)
        row = outcome_row(outcome, outcome.rejection_reason or outcome.eligibility_reason or "accepted",
                          split=None, identities=None, denominator_counted=False,
                          expected_generator_id="dwr-generator-v3")
        self.assertEqual(row["outcome_version"], "dwr-task-outcome-v3")
        legacy = generate_candidate(20261007, "source", 0)
        with self.assertRaisesRegex(ValueError, "generator identity"):
            outcome_row(legacy, legacy.rejection_reason or legacy.eligibility_reason or "accepted",
                        split=None, identities=None, denominator_counted=False,
                        expected_generator_id="dwr-generator-v3")

    def test_v3_accepted_record_adds_only_registered_identity_fields(self):
        outcome = generate_candidate_v3(20261007, "source", 1)
        self.assertTrue(outcome.oracle_unique and outcome.eligible)
        split, identities = _split_and_ids(outcome)
        record = accepted_record(outcome, split, identities, expected_generator_id="dwr-generator-v3")
        from dwr.tasks.audit import RECORD_KEYS
        self.assertEqual(record["record_version"], "dwr-task-accepted-record-v3")
        self.assertEqual(record["generator_id"], "dwr-generator-v3")
        self.assertEqual(set(record) - {"record_version", "generator_id"}, RECORD_KEYS)

    def test_v3_full_mode_fails_closed_before_creating_output(self):
        limits = DatasetLimits({"eq": 0, "neq": 0, "add_mod": 0}, {}, {}, 1, 30, 1024)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "full"
            with self.assertRaisesRegex(ValueError, "preflight consumption is not available"):
                run_dataset_v3(self.config, out, "full", ["test"], limits=limits)
            self.assertFalse(out.exists())

    def test_v3_cli_does_not_accept_v2_config(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_v3_main(["--config", str(ROOT / "configs/tasks/constraint_v2.json"),
                                    "--mode", "feasibility", "--out", str(out)])
            self.assertEqual(code, 1)
            self.assertIn("v3 generation failed", stderr.getvalue())
            self.assertFalse(out.exists())

    def test_small_incomplete_source_fixture_logs_v3_and_replays_as_partial(self):
        limits = DatasetLimits({"eq": 2, "neq": 0, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 1, 30, 1024)
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            code = run_dataset_v3(self.config, out, "feasibility", ["fixture"],
                                  limits=limits, source=source)
            self.assertEqual(code, 1)
            manifest = __import__("json").loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_version"], "dwr-task-run-v3")
            self.assertEqual(manifest["producer"].keys(), {
                "generator_v2_sha256", "generator_v3_sha256", "config_v3_sha256",
                "dataset_v2_sha256", "replay_v2_sha256", "manifest_v2_sha256",
                "stream_contract_sha256", "cli_v2_sha256", "cli_v3_sha256",
                "canonical_sha256", "oracle_sha256", "schema_sha256", "solver_sha256",
                "run_support_sha256"})
            self.assertIsNone(manifest["early_stop"])
            self.assertEqual(manifest["source_capacity_bound"]["status"], "NOT_ASSESSED")
            replay = replay_files_v3(out / "outcomes.jsonl", out / "records.jsonl",
                                     self.config, "feasibility", manifest,
                                     limits=limits, allow_partial=True)
            self.assertEqual(replay["status"], "partial")

    def test_first_ineligible_unique_owns_content_and_later_copy_is_duplicate(self):
        base = generate_candidate_v3(20261007, "source", 1)
        self.assertTrue(base.oracle_unique and base.eligible)
        ineligible_evidence = CandidatePrivateEvidence(
            True, SourceEligibilityEvidence(tuple(1 for _ in
                base.private_evidence.source.removal_solution_counts), False), None)
        limits = DatasetLimits({"eq": 2, "neq": 0, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 3, 30, 1024)
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        def injected(seed, partition, index, *, full_profiles=frozenset()):
            if partition == "source" and index in (1, 2):
                return replace(base, candidate_index=index,
                               private_evidence=ineligible_evidence if index == 1 else base.private_evidence)
            return generate_candidate_v3(seed, partition, index, full_profiles=full_profiles)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            with patch("dwr.tasks.generator_v3.generate_candidate_v3", side_effect=injected):
                code = run_dataset_v3(self.config, out, "feasibility", ["fixture"],
                                      limits=limits, source=source)
            self.assertEqual(code, 1)
            import json
            rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[1]["terminal_reason"], "source_insensitive")
            self.assertTrue(rows[1]["denominator_counted"])
            self.assertEqual(rows[2]["terminal_reason"], "duplicate_content")
            self.assertFalse(rows[2]["denominator_counted"])
            self.assertEqual((out / "records.jsonl").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
