import json
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from dwr.tasks.config_v2 import load_config_v2
from dwr.tasks.dataset_v2 import DatasetLimits, _split_and_ids, run_dataset_v2, validate_preflight
from dwr.tasks.generator_v2 import CandidatePrivateEvidence, SourceEligibilityEvidence, generate_candidate
from dwr.tasks.replay_v2 import replay_files_v2
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
                                  source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64, "checkout_root": str(ROOT)})
            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "feasibility")
            self.assertEqual(manifest["partitions"]["final"]["status"], "not_attempted")
            self.assertEqual(manifest["partitions"]["source"]["attempts"], self.accepted_index + 1)
            self.assertNotIn("profile_accepted", manifest["partitions"]["source"])
            self.assertNotIn("source_validation", manifest["partitions"])
            self.assertEqual(manifest["artifacts"]["outcomes.jsonl"]["sha256"],
                             __import__("hashlib").sha256((out / "outcomes.jsonl").read_bytes()).hexdigest())

    def test_development_validation_quota_excludes_opaque_diagnostic_acceptance(self):
        diagnostic_index, validation_index = 0, 1
        diagnostic = generate_candidate(20261007, "development", 113)
        validation = generate_candidate(20261007, "development", 29)
        self.assertEqual(_split_and_ids(diagnostic)[0], "development_diagnostic_test")
        self.assertEqual(_split_and_ids(validation)[0], "development_validation")
        limits = DatasetLimits({"eq": 0, "neq": 0, "add_mod": 0},
                              {("neq", "add_mod"): 1}, {}, validation_index + 1, 30, 1024)
        def candidates(seed, part, index, *, full_profiles=frozenset()):
            if part == "development" and index == diagnostic_index:
                return replace(diagnostic, candidate_index=index)
            if part == "development" and index == validation_index:
                return replace(validation, candidate_index=index)
            return generate_candidate(seed, part, index, full_profiles=full_profiles)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            code = run_dataset_v2(self.config, out, "feasibility", ["dev-quota"], limits=limits,
                                  candidate_fn=candidates,
                                  source={"commit": "test", "dirty": True,
                                          "dirty_diff_sha256": "0" * 64, "checkout_root": str(ROOT)})
            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
            diagnostic_row = next(row for row in rows if row["partition"] == "development")
            self.assertEqual(diagnostic_row["split"], "development_diagnostic_test")
            self.assertEqual(diagnostic_row["terminal_reason"], "accepted")
            self.assertNotIn("diagnostic", manifest["partitions"]["development"])
            self.assertEqual(manifest["partitions"]["development"]["accepted"], 1)
            self.assertEqual(manifest["partitions"]["development"]["attempts"], validation_index + 1)

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
                                  source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64, "checkout_root": str(ROOT)})
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
                                  source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64, "checkout_root": str(ROOT)})
            self.assertEqual(code, 1)
            rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["terminal_reason"], "accepted")
            self.assertEqual(rows[3]["terminal_reason"], "duplicate_content")
            self.assertEqual(len((out / "records.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    def test_seen_content_set_is_shared_across_partitions(self):
        development_candidate = generate_candidate(20261007, "development", 29)
        limits = DatasetLimits({"eq": 1, "neq": 0, "add_mod": 0},
                              {development_candidate.profile: 1}, {}, self.accepted_index + 1, 30, 1024)
        def candidates(seed, part, index, *, full_profiles=frozenset()):
            return generate_candidate(seed, part, index, full_profiles=full_profiles)
        from dwr.tasks.dataset_v2 import _split_and_ids as real_split_and_ids
        def shared_for_pair(outcome):
            split, identities = real_split_and_ids(outcome)
            if ((outcome.partition == "source" and outcome.candidate_index == self.accepted_index) or
                (outcome.partition == "development" and outcome.eligible)):
                identities["content_id"] = "synthetic-cross-partition-duplicate"
            return split, identities
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            with patch("dwr.tasks.dataset_v2._split_and_ids", side_effect=shared_for_pair):
                code = run_dataset_v2(self.config, out, "feasibility", ["cross-partition-duplicate"],
                                      limits=limits, candidate_fn=candidates,
                                      source={"commit": "test", "dirty": True,
                                              "dirty_diff_sha256": "0" * 64, "checkout_root": str(ROOT)})
            self.assertEqual(code, 1)
            rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
            duplicate = next(row for row in rows if row["partition"] == "development" and
                             row["candidate_index"] == 29)
            self.assertEqual(duplicate["terminal_reason"], "duplicate_content")
            self.assertFalse(duplicate["denominator_counted"])

    def test_attempt_cap_preserves_partial_hashes_and_marks_later_partitions_unattempted(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            code = run_dataset_v2(self.config, out, "feasibility", ["test"], limits=self.limits(attempts=1),
                                  source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64, "checkout_root": str(ROOT)})
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
                                  source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64, "checkout_root": str(ROOT)})
            self.assertEqual(code, 1)
            self.assertEqual(calls, [])
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["error"]["type"], "TimeoutError")
            self.assertTrue(manifest["partitions"]["source"]["not_attempted"])

    def test_returned_candidate_is_logged_before_postproposal_cap_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            now = [0.0]
            calls = []
            def candidates(seed, part, index, *, full_profiles=frozenset()):
                calls.append((part, index))
                outcome = generate_candidate(seed, part, index, full_profiles=full_profiles)
                now[0] = 31.0  # proposal returned after the unit 30-second wall cap
                return outcome
            guard = ResourceGuard(30, 1024, clock=lambda: now[0], memory_reader=lambda: 0)
            code = run_dataset_v2(
                self.config, out, "feasibility", ["proposal-crossed-cap"],
                limits=self.limits(source=1, attempts=3), candidate_fn=candidates, guard=guard,
                source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64,
                        "checkout_root": str(ROOT)})
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(code, 1)
            self.assertEqual(calls, [("source", 0)])
            self.assertEqual(manifest["status"], "incomplete")
            self.assertEqual(manifest["error"]["type"], "TimeoutError")
            self.assertIn("wall-time cap reached", manifest["error"]["message"])
            self.assertEqual(manifest["error"]["phase"], "generation")
            self.assertEqual(manifest["error"]["partition"], "source")
            self.assertEqual(manifest["error"]["candidate_index"], 0)
            self.assertEqual(manifest["partitions"]["source"]["attempts"], 1)
            self.assertEqual(manifest["partitions"]["source"]["status"], "incomplete")
            self.assertTrue(manifest["partitions"]["development"]["not_attempted"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_index"], 0)
            self.assertEqual(rows[0]["terminal_reason"], "no_witness_feasible_constraint_set")
            self.assertGreaterEqual(manifest["generation_elapsed_seconds"], 31.0)

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
            root = Path(temp); data = root / "feasibility"
            source = {"commit": "a" * 40, "dirty": False, "dirty_diff_sha256": "0" * 64,
                      "checkout_root": str(root)}
            feasibility_limits = DatasetLimits({"eq": 0, "neq": 0, "add_mod": 0}, {}, {}, 1, 30, 1024)
            self.assertEqual(run_dataset_v2(self.config, data, "feasibility", ["synthetic-feasibility"],
                                            limits=feasibility_limits, source=source), 0)
            generation = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
            artifacts = generation["artifacts"]
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
            consumed_wrapper_hash = sha256_file(preflight)
            empty_limits = DatasetLimits({"eq": 0, "neq": 0, "add_mod": 0}, {},
                                         {("eq", "neq", "add_mod"): 0}, 1, 30, 1024)
            code = run_dataset_v2(self.config, root / "full", "full", ["synthetic-test"],
                                  limits=empty_limits, source=source, preflight_report=preflight,
                                  preflight_limits=feasibility_limits)
            self.assertEqual(code, 0)
            full_manifest = json.loads((root / "full/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(full_manifest["mode"], "full")
            consumed = full_manifest["consumed_preflight"]
            self.assertEqual(consumed["wrapper_sha256"], consumed_wrapper_hash)
            self.assertEqual(consumed["generation_manifest_sha256"], manifest_sha)
            self.assertEqual(consumed["audit_artifact_sha256"], report["audit_artifact_sha256"])
            self.assertEqual(consumed["generation_artifacts"], artifacts)
            preflight.write_text("replacement after dispatch", encoding="utf-8")
            self.assertNotEqual(consumed["wrapper_sha256"], sha256_file(preflight))
            self.assertEqual(json.loads((root / "full/manifest.json").read_text(encoding="utf-8"))
                             ["consumed_preflight"]["wrapper_sha256"], consumed_wrapper_hash)

    def test_tampered_audit_evidence_blocks_full_dispatch_before_directory_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / "feasibility"
            source = {"commit": "a" * 40, "dirty": False, "dirty_diff_sha256": "0" * 64,
                      "checkout_root": str(root)}
            feasibility_limits = DatasetLimits({"eq": 0, "neq": 0, "add_mod": 0}, {}, {}, 1, 30, 1024)
            self.assertEqual(run_dataset_v2(self.config, data, "feasibility", ["synthetic-feasibility"],
                                            limits=feasibility_limits, source=source), 0)
            artifacts = json.loads((data / "manifest.json").read_text(encoding="utf-8"))["artifacts"]
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
                               limits=self.limits(), source=source, preflight_report=preflight,
                               preflight_limits=feasibility_limits)
            self.assertFalse((root / "full").exists())

    def test_initial_manifest_write_failure_leaves_terminal_failure_record(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            with patch("dwr.tasks.dataset_v2.atomic_json", side_effect=PermissionError("INITIAL manifest failure")):
                code = run_dataset_v2(self.config, out, "feasibility", ["test"], limits=self.limits(),
                                      source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64, "checkout_root": str(ROOT)})
            self.assertEqual(code, 1)
            failure = json.loads((out / "terminal_failure.json").read_text(encoding="utf-8"))
            self.assertIn("INITIAL manifest failure", failure["primary_error"]["message"])
            self.assertEqual(failure["primary_error"]["phase"], "setup")

    def test_primary_candidate_error_survives_artifact_hash_and_terminal_write_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            real_hash = sha256_file
            calls = {"manifest": 0}
            def fail_hash(path):
                if Path(path).name == "outcomes.jsonl":
                    raise PermissionError("LATER hash failure")
                return real_hash(path)
            def fail_terminal(path, value):
                calls["manifest"] += 1
                if calls["manifest"] == 1:
                    from dwr.tasks.run_support import atomic_json as real_atomic_json
                    return real_atomic_json(path, value)
                if calls["manifest"] >= 2:
                    raise PermissionError("LATER manifest failure")
            def candidate(*args, **kwargs):
                raise RuntimeError("PRIMARY candidate failure")
            with patch("dwr.tasks.dataset_v2.sha256_file", side_effect=fail_hash), \
                 patch("dwr.tasks.dataset_v2.atomic_json", side_effect=fail_terminal):
                code = run_dataset_v2(self.config, out, "feasibility", ["test"], limits=self.limits(),
                                      candidate_fn=candidate,
                                      source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64, "checkout_root": str(ROOT)})
            self.assertEqual(code, 1)
            failure = json.loads((out / "terminal_failure.json").read_text(encoding="utf-8"))
            self.assertIn("PRIMARY candidate failure", failure["primary_error"]["message"])
            self.assertTrue(failure["finalization_errors"])

    def test_resource_cap_primary_survives_later_artifact_hash_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            real_hash = sha256_file
            def fail_outcome_hash(path):
                if Path(path).name == "outcomes.jsonl":
                    raise PermissionError("LATER outcome hash failure")
                return real_hash(path)
            with patch("dwr.tasks.dataset_v2.sha256_file", side_effect=fail_outcome_hash):
                code = run_dataset_v2(self.config, out, "feasibility", ["test"], limits=self.limits(),
                                      guard=ResourceGuard(0, 1024),
                                      source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64,
                                              "checkout_root": str(ROOT)})
            self.assertEqual(code, 1)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("wall-time cap reached", manifest["error"]["message"])
            self.assertIn("outcome hash failure", manifest["finalization_errors"][0]["message"])

    def test_partial_source_then_development_is_not_a_global_prefix(self):
        from dwr.tasks.dataset_v2 import outcome_row
        from dwr.tasks.generator_v2 import generate_candidate
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            limits = DatasetLimits({"eq": 1, "neq": 0, "add_mod": 0},
                                  {("eq", "neq"): 1}, {}, 1, 30, 1024)
            code = run_dataset_v2(self.config, out, "feasibility", ["prefix"], limits=limits,
                                  source={"commit": "test", "dirty": True, "dirty_diff_sha256": "0" * 64, "checkout_root": str(ROOT)})
            self.assertEqual(code, 1)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            dev = generate_candidate(20261007, "development", 0)
            dev_row = outcome_row(dev, dev.rejection_reason or dev.eligibility_reason,
                                  split=None, identities=None, denominator_counted=False)
            with (out / "outcomes.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(dev_row, sort_keys=True, separators=(",", ":")) + "\n")
            manifest["partitions"]["development"].update(status="incomplete", attempts=1,
                                                           accepted=0, not_attempted=False)
            outcome_path = out / "outcomes.jsonl"
            manifest["artifacts"]["outcomes.jsonl"] = {"sha256": hashlib.sha256(outcome_path.read_bytes()).hexdigest(),
                                                         "bytes": outcome_path.stat().st_size}
            with self.assertRaises(ValueError):
                replay_files_v2(outcome_path, out / "records.jsonl", self.config,
                                "feasibility", manifest, limits=limits, allow_partial=True)
            del manifest["partitions"]["source"]["status"]
            with self.assertRaises(ValueError):
                replay_files_v2(outcome_path, out / "records.jsonl", self.config,
                                "feasibility", manifest, limits=limits, allow_partial=True)

    def test_preflight_rejects_minimal_manifest_bool_exit_and_missing_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / "feasibility"; data.mkdir()
            source = {"commit": "a" * 40, "dirty": False, "dirty_diff_sha256": "0" * 64,
                      "checkout_root": str(root)}
            for name in ("outcomes.jsonl", "records.jsonl"):
                (data / name).write_text("", encoding="utf-8")
            artifacts = {name: {"sha256": sha256_file(data / name), "bytes": 0}
                         for name in ("outcomes.jsonl", "records.jsonl")}
            minimal = {"manifest_version": "dwr-task-run-v2", "mode": "feasibility",
                       "config_sha256": self.config.sha256, "source": source,
                       "status": "complete", "exit_code": False, "artifacts": artifacts,
                       "producer": {"generator_v2_sha256": sha256_file(ROOT / "dwr/tasks/generator_v2.py"),
                                    "dataset_v2_sha256": sha256_file(ROOT / "dwr/tasks/dataset_v2.py"),
                                    "replay_v2_sha256": sha256_file(ROOT / "dwr/tasks/replay_v2.py")}}
            atomic_json(data / "manifest.json", minimal)
            manifest_sha = sha256_file(data / "manifest.json")
            audit = {"audit_version": "dwr-task-v2-audit-v1", "status": "complete", "mode": "feasibility",
                     "g2_status": "PASS_FEASIBILITY", "config_sha256": self.config.sha256,
                     "source": source, "generation_manifest_sha256": manifest_sha,
                     "generation_artifacts": artifacts}
            atomic_json(data / "audit_report.json", audit)
            report = {"report_version": "dwr-task-v2-feasibility-preflight-v1", "status": "PASS_FEASIBILITY",
                      "mode": "feasibility", "config_sha256": self.config.sha256, "source": source,
                      "generation_dir": str(data), "generation_manifest_sha256": manifest_sha,
                      "generation_artifacts": artifacts, "audit_artifact_path": str(data / "audit_report.json"),
                      "audit_artifact_sha256": sha256_file(data / "audit_report.json"), "audit_source": source}
            preflight = root / "preflight.json"; atomic_json(preflight, report)
            with self.assertRaises(ValueError):
                validate_preflight(preflight, self.config, source)

    def test_preflight_rejects_bool_exit_altered_quota_and_inconsistent_partition_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / "feasibility"
            source = {"commit": "a" * 40, "dirty": False, "dirty_diff_sha256": "0" * 64,
                      "checkout_root": str(root)}
            limits = DatasetLimits({"eq": 0, "neq": 0, "add_mod": 0}, {}, {}, 1, 30, 1024)
            self.assertEqual(run_dataset_v2(self.config, data, "feasibility", ["valid-fixture"],
                                            limits=limits, source=source), 0)
            manifest_path = data / "manifest.json"
            audit_path = data / "audit_report.json"
            wrapper_path = root / "preflight.json"

            def write_linkage(manifest):
                atomic_json(manifest_path, manifest)
                manifest_sha = sha256_file(manifest_path)
                artifacts = manifest["artifacts"]
                audit = {"audit_version": "dwr-task-v2-audit-v1", "status": "complete",
                         "mode": "feasibility", "g2_status": "PASS_FEASIBILITY",
                         "config_sha256": self.config.sha256, "source": source,
                         "generation_manifest_sha256": manifest_sha,
                         "generation_artifacts": artifacts}
                atomic_json(audit_path, audit)
                report = {"report_version": "dwr-task-v2-feasibility-preflight-v1",
                          "status": "PASS_FEASIBILITY", "mode": "feasibility",
                          "config_sha256": self.config.sha256, "source": source,
                          "generation_dir": str(data), "generation_manifest_sha256": manifest_sha,
                          "generation_artifacts": artifacts, "audit_artifact_path": str(audit_path),
                          "audit_artifact_sha256": sha256_file(audit_path), "audit_source": source}
                atomic_json(wrapper_path, report)

            baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
            for mutate in (
                lambda m: m.update(exit_code=False),
                lambda m: m["limits"]["quota_targets"].update(source=1),
                lambda m: m["partitions"]["development"].update(status="incomplete"),
            ):
                candidate = json.loads(json.dumps(baseline))
                mutate(candidate)
                write_linkage(candidate)
                with self.assertRaises(ValueError):
                    validate_preflight(wrapper_path, self.config, source, limits=limits)


if __name__ == "__main__":
    unittest.main()
