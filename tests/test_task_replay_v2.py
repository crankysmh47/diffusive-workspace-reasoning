import json
import tempfile
import unittest
from pathlib import Path

from dwr.tasks.config_v2 import load_config_v2
from dwr.tasks.dataset_v2 import DatasetLimits, run_dataset_v2
from dwr.tasks.replay_v2 import replay_files_v2, replay_outcomes_v2
from dwr.tasks.generator_v2 import generate_candidate
from dwr.tasks.audit import validate_record
from dwr.tasks.run_support import ResourceGuard


ROOT = Path(__file__).resolve().parents[1]


class ReplayV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config_v2(ROOT / "configs/tasks/constraint_v2.json")
        cls.target = next(i for i in range(0, 300, 3)
                          if (lambda o: o.oracle_unique and o.eligible)(
                              generate_candidate(20261007, "source", i)))
        cls.limits = DatasetLimits({"eq": 1, "neq": 0, "add_mod": 0},
                                  {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                   ("neq", "add_mod"): 0}, {}, cls.target + 1, 30, 1024)

    def make_run(self, directory):
        code = run_dataset_v2(self.config, directory, "feasibility", ["test"], limits=self.limits,
                              source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64})
        self.assertEqual(code, 0)
        return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    def test_stream_replay_reconstructs_records_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            manifest = self.make_run(out)
            result = replay_files_v2(out / "outcomes.jsonl", out / "records.jsonl", self.config,
                                     "feasibility", manifest, limits=self.limits)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["accepted_records"], 1)
            record = json.loads((out / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
            validate_record(record, self.config.resolved)
            manifest["partitions"]["source"]["accepted"] += 1
            with self.assertRaises(ValueError):
                replay_files_v2(out / "outcomes.jsonl", out / "records.jsonl", self.config,
                                "feasibility", manifest, limits=self.limits)

    def test_extra_fields_and_tampered_candidate_order_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            self.make_run(out)
            rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["unexpected"] = True
            with self.assertRaises(ValueError):
                replay_outcomes_v2(rows, self.config, "feasibility", limits=self.limits)
            rows[0].pop("unexpected")
            rows[0]["candidate_index"] = True
            with self.assertRaises(ValueError):
                replay_outcomes_v2(rows, self.config, "feasibility", limits=self.limits)

    def test_extra_rows_after_exact_quota_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            self.make_run(out)
            rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
            rows.append(dict(rows[-1]))
            with self.assertRaises(ValueError):
                replay_outcomes_v2(rows, self.config, "feasibility", limits=self.limits)

    def test_manifest_bound_partial_prefix_replays_but_missing_row_fails(self):
        partial_limits = DatasetLimits({"eq": 1, "neq": 0, "add_mod": 0},
                                       {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                        ("neq", "add_mod"): 0}, {}, 1, 30, 1024)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            code = run_dataset_v2(self.config, out, "feasibility", ["partial"], limits=partial_limits,
                                  source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64})
            self.assertEqual(code, 1)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            result = replay_files_v2(out / "outcomes.jsonl", out / "records.jsonl", self.config,
                                     "feasibility", manifest, limits=partial_limits, allow_partial=True)
            self.assertEqual(result["status"], "partial")
            original_outcomes = (out / "outcomes.jsonl").read_text(encoding="utf-8")
            (out / "outcomes.jsonl").write_text("", encoding="utf-8")
            manifest["artifacts"]["outcomes.jsonl"] = {"sha256": __import__("hashlib").sha256(b"").hexdigest(), "bytes": 0}
            with self.assertRaises(ValueError):
                replay_files_v2(out / "outcomes.jsonl", out / "records.jsonl", self.config,
                                "feasibility", manifest, limits=partial_limits, allow_partial=True)
            (out / "outcomes.jsonl").write_text(original_outcomes, encoding="utf-8", newline="\n")
            manifest["artifacts"]["outcomes.jsonl"] = {
                "sha256": __import__("hashlib").sha256(original_outcomes.encode("utf-8")).hexdigest(),
                "bytes": len(original_outcomes.encode("utf-8"))}
            manifest["partitions"]["source"]["attempts"] = 0
            with self.assertRaises(ValueError):
                replay_files_v2(out / "outcomes.jsonl", out / "records.jsonl", self.config,
                                "feasibility", manifest, limits=partial_limits, allow_partial=True)

    def test_rejection_only_replay_stops_at_inherited_time_cap(self):
        partial_limits = DatasetLimits({"eq": 1, "neq": 0, "add_mod": 0},
                                       {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                        ("neq", "add_mod"): 0}, {}, 1, 30, 1024)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            run_dataset_v2(self.config, out, "feasibility", ["partial"], limits=partial_limits,
                           source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64})
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            class StopAfterProposed(ResourceGuard):
                checks = 0
                def check(self):
                    self.checks += 1
                    if self.checks >= 9:
                        raise TimeoutError("synthetic inherited audit cap")
                    super().check()
            guard = StopAfterProposed(30, 1024)
            calls = []
            def candidate(*args, **kwargs):
                calls.append(args)
                return generate_candidate(*args, **kwargs)
            with self.assertRaises(TimeoutError):
                replay_files_v2(out / "outcomes.jsonl", out / "records.jsonl", self.config,
                                "feasibility", manifest, limits=partial_limits, guard=guard,
                                allow_partial=True, candidate_fn=candidate)
            self.assertEqual(len(calls), 1)  # proposal was a rejection, and its post-check consumed the cap


if __name__ == "__main__":
    unittest.main()
