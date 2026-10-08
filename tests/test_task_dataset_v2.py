import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from dwr.tasks.config_v2 import load_config_v2
from dwr.tasks.dataset_v2 import DatasetLimits, run_dataset_v2
from dwr.tasks.generator_v2 import CandidatePrivateEvidence, SourceEligibilityEvidence, generate_candidate
from dwr.tasks.run_support import ResourceGuard, atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]


class DatasetV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config_v2(ROOT / "configs/tasks/constraint_v2.json")
        cls.accepted_index = next(i for i in range(0, 300, 3)
                                  if (lambda o: o.oracle_unique and o.eligible)(
                                      generate_candidate(20261007, "source", i)))

    def limits(self, *, source=1, attempts=100):
        return DatasetLimits({"eq": source, "neq": 0, "add_mod": 0},
                             {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                              ("neq", "add_mod"): 0}, {}, attempts, 30, 1024)

    def test_feasibility_runs_only_source_and_development_and_replays_stream(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            code = run_dataset_v2(self.config, out, "feasibility", ["test"],
                                  limits=self.limits(attempts=self.accepted_index + 1),
                                  source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64})
            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "feasibility")
            self.assertEqual(manifest["partitions"]["final"]["status"], "not_attempted")
            self.assertEqual(manifest["partitions"]["source"]["attempts"], self.accepted_index + 1)
            self.assertNotIn("profile_accepted", manifest["partitions"]["source"])
            self.assertNotIn("source_validation", manifest["partitions"])
            self.assertEqual(manifest["artifacts"]["outcomes.jsonl"]["sha256"],
                             __import__("hashlib").sha256((out / "outcomes.jsonl").read_bytes()).hexdigest())

    def test_duplicate_after_first_ineligible_unique_is_terminal_duplicate(self):
        base = generate_candidate(20261007, "source", self.accepted_index)
        evidence = CandidatePrivateEvidence(True,
            SourceEligibilityEvidence(tuple(1 for _ in base.private_evidence.source.removal_solution_counts), False), None)
        def candidates(seed, part, index, *, full_profiles=frozenset()):
            if part == "source" and index in (0, self.accepted_index):
                return replace(base, candidate_index=index, private_evidence=evidence if index == 0 else base.private_evidence)
            return generate_candidate(seed, part, index, full_profiles=full_profiles)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            code = run_dataset_v2(self.config, out, "feasibility", ["test"],
                                  limits=self.limits(attempts=self.accepted_index + 1),
                                  candidate_fn=candidates,
                                  source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64})
            self.assertEqual(code, 1)  # no quota is filled; partial stream remains auditable
            rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["terminal_reason"], "source_insensitive")
            self.assertTrue(rows[0]["denominator_counted"])
            self.assertEqual(rows[self.accepted_index]["terminal_reason"], "duplicate_content")
            self.assertFalse(rows[self.accepted_index]["denominator_counted"])
            self.assertEqual((out / "records.jsonl").read_text(encoding="utf-8"), "")

    def test_duplicate_after_accepted_unique_does_not_fill_second_quota(self):
        base = generate_candidate(20261007, "source", self.accepted_index)
        def candidates(seed, part, index, *, full_profiles=frozenset()):
            if part == "source" and index in (0, 3):
                return replace(base, candidate_index=index)
            return generate_candidate(seed, part, index, full_profiles=full_profiles)
        limits = self.limits(source=2, attempts=4)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            code = run_dataset_v2(self.config, out, "feasibility", ["test"], limits=limits,
                                  candidate_fn=candidates,
                                  source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64})
            self.assertEqual(code, 1)
            rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["terminal_reason"], "accepted")
            self.assertEqual(rows[3]["terminal_reason"], "duplicate_content")
            self.assertEqual(len((out / "records.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    def test_attempt_cap_preserves_partial_hashes_and_marks_later_partitions_unattempted(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            code = run_dataset_v2(self.config, out, "feasibility", ["test"], limits=self.limits(attempts=1),
                                  source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64})
            self.assertEqual(code, 1)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "incomplete")
            self.assertEqual(manifest["partitions"]["source"]["status"], "incomplete")
            self.assertTrue(manifest["partitions"]["development"]["not_attempted"])
            self.assertTrue(manifest["partitions"]["final"]["not_attempted"])
            self.assertIn("outcomes.jsonl", manifest["artifacts"])

    def test_resource_guard_stops_before_candidate_work(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            calls = []
            def candidates(*args, **kwargs):
                calls.append(args)
                return generate_candidate(*args, **kwargs)
            guard = ResourceGuard(0, 1024)
            code = run_dataset_v2(self.config, out, "feasibility", ["test"], limits=self.limits(),
                                  candidate_fn=candidates, guard=guard,
                                  source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64})
            self.assertEqual(code, 1)
            self.assertEqual(calls, [])
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["error"]["type"], "TimeoutError")
            self.assertTrue(manifest["partitions"]["source"]["not_attempted"])

    def test_directory_collision_and_full_mode_without_preflight_do_not_mutate(self):
        with tempfile.TemporaryDirectory() as temp:
            existing = Path(temp) / "already-there"
            existing.mkdir()
            marker = existing / "marker.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_dataset_v2(self.config, existing, "feasibility", ["test"], limits=self.limits())
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            with self.assertRaises(ValueError):
                run_dataset_v2(self.config, Path(temp) / "full", "full", ["test"], limits=self.limits())
            self.assertFalse((Path(temp) / "full").exists())

    def test_synthetic_matching_pass_fixture_dispatches_full_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / "feasibility"; data.mkdir()
            source = {"commit": "a" * 40, "dirty": False, "dirty_diff_sha256": "0" * 64,
                      "checkout_root": str(root)}
            for filename in ("outcomes.jsonl", "records.jsonl"):
                (data / filename).write_text("", encoding="utf-8")
            artifacts = {name: {"sha256": sha256_file(data / name),
                                "bytes": (data / name).stat().st_size}
                         for name in ("outcomes.jsonl", "records.jsonl")}
            generation = {"manifest_version": "dwr-task-run-v2", "mode": "feasibility",
                          "config_sha256": self.config.sha256, "source": source,
                          "status": "complete", "exit_code": 0, "artifacts": artifacts,
                          "producer": {"generator_v2_sha256": sha256_file(ROOT / "dwr/tasks/generator_v2.py"),
                                       "dataset_v2_sha256": sha256_file(ROOT / "dwr/tasks/dataset_v2.py"),
                                       "replay_v2_sha256": sha256_file(ROOT / "dwr/tasks/replay_v2.py")}}
            atomic_json(data / "manifest.json", generation)
            manifest_sha = sha256_file(data / "manifest.json")
            audit = {"audit_version": "dwr-task-v2-audit-v1", "status": "complete",
                     "mode": "feasibility", "g2_status": "PASS_FEASIBILITY",
                     "config_sha256": self.config.sha256, "source": source,
                     "generation_manifest_sha256": manifest_sha,
                     "generation_artifacts": artifacts}
            atomic_json(data / "audit_report.json", audit)
            report = {"report_version": "dwr-task-v2-feasibility-preflight-v1",
                      "status": "PASS_FEASIBILITY", "mode": "feasibility",
                      "config_sha256": self.config.sha256, "source": source,
                      "generation_dir": str(data), "generation_manifest_sha256": manifest_sha,
                      "generation_artifacts": artifacts,
                      "audit_artifact_path": str(data / "audit_report.json"),
                      "audit_artifact_sha256": sha256_file(data / "audit_report.json"),
                      "audit_source": source}
            preflight = root / "preflight.json"; atomic_json(preflight, report)
            empty_limits = DatasetLimits({"eq": 0, "neq": 0, "add_mod": 0}, {},
                                         {("eq", "neq", "add_mod"): 0}, 1, 30, 1024)
            code = run_dataset_v2(self.config, root / "full", "full", ["synthetic-test"],
                                  limits=empty_limits, source=source, preflight_report=preflight)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((root / "full/manifest.json").read_text(encoding="utf-8"))["mode"], "full")

    def test_tampered_audit_evidence_blocks_full_dispatch_before_directory_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / "feasibility"; data.mkdir()
            source = {"commit": "a" * 40, "dirty": False, "dirty_diff_sha256": "0" * 64,
                      "checkout_root": str(root)}
            for filename in ("outcomes.jsonl", "records.jsonl"):
                (data / filename).write_text("", encoding="utf-8")
            artifacts = {name: {"sha256": sha256_file(data / name), "bytes": 0}
                         for name in ("outcomes.jsonl", "records.jsonl")}
            atomic_json(data / "manifest.json", {"manifest_version": "dwr-task-run-v2", "mode": "feasibility",
                        "config_sha256": self.config.sha256, "source": source, "status": "complete",
                        "exit_code": 0, "artifacts": artifacts,
                        "producer": {"generator_v2_sha256": sha256_file(ROOT / "dwr/tasks/generator_v2.py"),
                                     "dataset_v2_sha256": sha256_file(ROOT / "dwr/tasks/dataset_v2.py"),
                                     "replay_v2_sha256": sha256_file(ROOT / "dwr/tasks/replay_v2.py")}})
            manifest_sha = sha256_file(data / "manifest.json")
            atomic_json(data / "audit_report.json", {"tampered": True})
            report = {"report_version": "dwr-task-v2-feasibility-preflight-v1",
                      "status": "PASS_FEASIBILITY", "mode": "feasibility",
                      "config_sha256": self.config.sha256, "source": source,
                      "generation_dir": str(data), "generation_manifest_sha256": manifest_sha,
                      "generation_artifacts": artifacts,
                      "audit_artifact_path": str(data / "audit_report.json"),
                      "audit_artifact_sha256": sha256_file(data / "audit_report.json"),
                      "audit_source": source}
            preflight = root / "preflight.json"; atomic_json(preflight, report)
            with self.assertRaises(ValueError):
                run_dataset_v2(self.config, root / "full", "full", ["synthetic-test"],
                               limits=self.limits(), source=source, preflight_report=preflight)
            self.assertFalse((root / "full").exists())


if __name__ == "__main__":
    unittest.main()
