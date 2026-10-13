import unittest
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

from dwr.tasks.config_v2 import load_config_v2
from dwr.tasks.config_v3 import load_config_v3
from dwr.tasks.dataset_v2 import DatasetLimits, outcome_row, validate_preflight
from dwr.tasks.generator_v2 import generate_candidate
from dwr.tasks.generator_v3 import generate_candidate_v3
from dwr.tasks.replay_v2 import replay_outcomes_v3, source_support_capacity_bound
from dwr.tasks.replay_v2 import _prefix_rows
from dwr.tasks.run_support import ResourceGuard


ROOT = Path(__file__).resolve().parents[1]


class ReplayV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config_v3(ROOT / "configs/tasks/constraint_v3.json")
        cls.limits = DatasetLimits({"eq": 1, "neq": 0, "add_mod": 0}, {}, {}, 1, 30, 1024)

    def test_v3_replay_rejects_v2_config_before_reading_rows(self):
        legacy = load_config_v2(ROOT / "configs/tasks/constraint_v2.json")
        with self.assertRaisesRegex(ValueError, "config identity"):
            replay_outcomes_v3([], legacy, limits=self.limits)

    def test_v3_replay_rejects_v2_outcome_identity_under_v3_config(self):
        outcome = generate_candidate(20261007, "source", 0)
        row = outcome_row(outcome, outcome.rejection_reason or outcome.eligibility_reason or "accepted",
                          split=None, identities=None, denominator_counted=False)
        with self.assertRaisesRegex(ValueError, "deterministic replay"):
            replay_outcomes_v3([row], self.config, limits=self.limits)

    def test_v2_replay_rejects_a_v3_outcome_row(self):
        legacy = load_config_v2(ROOT / "configs/tasks/constraint_v2.json")
        limits = DatasetLimits({"eq": 1, "neq": 0, "add_mod": 0}, {}, {}, 1, 30, 1024)
        outcome = generate_candidate_v3(20261007, "source", 0)
        row = outcome_row(outcome, outcome.rejection_reason or outcome.eligibility_reason or "accepted",
                          split=None, identities=None, denominator_counted=False,
                          expected_generator_id="dwr-generator-v3")
        from dwr.tasks.replay_v2 import replay_outcomes_v2
        with self.assertRaises(ValueError):
            replay_outcomes_v2([row], legacy, limits=limits)

    def test_v2_preflight_validator_rejects_v3_config_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "v2-pass.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config identity"):
                validate_preflight(path, self.config, {"commit": "x"})

    def test_capacity_bound_counts_only_source_train_and_tie_breaks_cells(self):
        records = [{"partition": "source", "split": "source_train", "id": str(i), "model_input": "x"}
                   for i in range(13)]
        cells = {str(i): (i, 1, 2, 3, 4) for i in range(13)}
        # Inject the registered cell extractor at this unit boundary to isolate ordering/mass math.
        with patch("dwr.tasks.audit._cell", side_effect=lambda record: cells[record["id"]]):
            report = source_support_capacity_bound(records, "a" * 64)
        self.assertEqual(report["status"], "NO_GO")
        self.assertEqual(report["top_k_mass"], 10 / 13)
        self.assertEqual(report["top_cells"][0]["cell"], [0, 1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "source_train records only"):
            source_support_capacity_bound([{"partition": "source", "split": "source_validation"}], "a" * 64)

    def test_capacity_bound_pins_exact_and_above_threshold_and_empty_train(self):
        def evaluate(counts):
            records, cells = [], {}
            i = 0
            for cell_index, count in enumerate(counts):
                cell = (cell_index, 1, 2, 3, 4)
                for _ in range(count):
                    record = {"partition": "source", "split": "source_train",
                              "id": str(i), "model_input": "x"}
                    records.append(record); cells[str(i)] = cell; i += 1
            with patch("dwr.tasks.audit._cell", side_effect=lambda record: cells[record["id"]]):
                return source_support_capacity_bound(records, "b" * 64)
        at_gate = evaluate([4] * 10 + [3, 3, 2, 2])
        above_gate = evaluate([4] * 10 + [2, 2, 2, 2])
        self.assertEqual(at_gate["top_k_mass"], 0.8)
        self.assertEqual(at_gate["status"], "PASS_CAPACITY_BOUND")
        self.assertGreater(above_gate["top_k_mass"], 0.8)
        self.assertEqual(above_gate["status"], "PASS_CAPACITY_BOUND")
        empty = evaluate([])
        self.assertEqual(empty["status"], "NO_GO")
        self.assertEqual(empty["source_train_group_count"], 0)
        self.assertEqual(empty["source_prefix_replay_sha256"], "b" * 64)
        self.assertEqual(empty["top_k_mass"], 0.0)

    def test_prefix_reader_stops_at_recorded_row_boundary_after_append(self):
        prefix = b'{"partition":"source","i":0}\n{"partition":"source","i":1}\n'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "outcomes.jsonl"
            path.write_bytes(prefix + b'{"partition":"development","i":2}\n')
            rows = list(_prefix_rows(path, len(prefix), hashlib.sha256(prefix).hexdigest(),
                                     ResourceGuard(30, 1024, memory_reader=lambda: 0)))
        self.assertEqual([row["i"] for row in rows], [0, 1])

    def test_prefix_reader_rejects_tampered_context_hash(self):
        payload = b'{"partition":"source"}\n'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "outcomes.jsonl"
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                list(_prefix_rows(path, len(payload), "0" * 64,
                                  ResourceGuard(30, 1024, memory_reader=lambda: 0)))


if __name__ == "__main__":
    unittest.main()
