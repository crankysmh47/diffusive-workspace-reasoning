import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dwr.tasks.audit import _verify_manifest, run_audit
from dwr.tasks.dataset import GenerationLimits, run_dataset
from dwr.tasks.generator import CandidateOutcome
from dwr.tasks.run_support import ResourceGuard, atomic_json, load_config, sha256_file
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind
from dwr.tasks.solver import solve


def triangle_outcome():
    task = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
        (Constraint(RelationKind.EQ, ("v0", "v1")),
         Constraint(RelationKind.EQ, ("v0", "v2")),
         Constraint(RelationKind.EQ, ("v1", "v2"))),
        (GivenAssignment("v0", 0),))
    solved = solve(task)
    return CandidateOutcome("source", 0, 3, ("eq",), task, solved.solution,
                            solved.solution, solved, None)


def expected_limits(limits):
    return {"wall_seconds_global": limits.wall_seconds,
            "resident_memory_mib": limits.memory_mib,
            "cpu_workers": 1,
            "max_attempts_per_partition": limits.attempts,
            "target_accepted_groups_by_partition": limits.accepted,
            "target_accepted_per_development_profile": limits.development_per_profile}


def make_partial_dataset(data):
    config = load_config("configs/tasks/constraint_v1.json")
    limits = GenerationLimits({"source": 2, "development": 0, "final": 0}, 1, 1, 60, 1024)
    source = {"commit": "a" * 40, "dirty": False,
              "dirty_diff_sha256": "0" * 64, "checkout_root": str(Path.cwd())}
    status = run_dataset(config, data, ["fixture generator"], limits=limits,
                         candidate_fn=lambda *args, **kwargs: triangle_outcome(),
                         guard=ResourceGuard(60, 1024, clock=lambda: 0, memory_reader=lambda: 1),
                         source=source)
    return config, limits, status


class AuditReviewRegressionTests(unittest.TestCase):
    def test_partial_dataset_is_validated_measured_replayed_and_kept_out_of_g2(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            config, limits, gen_status = make_partial_dataset(data)
            self.assertEqual(gen_status, 1)
            audit_source = {"commit": "fixture-audit", "dirty": False,
                            "dirty_diff_sha256": "1" * 64, "checkout_root": str(Path.cwd())}
            with patch("dwr.tasks.audit.source_identity", return_value=audit_source), \
                 patch("dwr.tasks.audit.generate_candidate", side_effect=lambda *args, **kwargs: triangle_outcome()):
                status = run_audit(Path("configs/tasks/constraint_v1.json"), data,
                    ["fixture audit"], expected_limits=expected_limits(limits))
            self.assertEqual(status, 1)
            report = json.loads((data / "audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "incomplete_generation", report)
            self.assertEqual(report["g2"]["status"], "not_assessed")
            self.assertEqual(report["progress"]["record_interactions_assessed"], 1)
            self.assertEqual(report["progress"]["replay_candidates_assessed"], 1)
            results = [json.loads(line) for line in (data / "audit_results.jsonl").read_text(encoding="utf-8").splitlines()]
            measured = next(row for row in results if row["kind"] == "record_interaction")
            self.assertFalse(measured["interaction"]["sensitive"])
            audit_manifest = json.loads((data / "audit_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(audit_manifest["source"], audit_source)
            self.assertEqual(audit_manifest["generation_source"]["commit"], "a" * 40)
            self.assertIn("platform", audit_manifest)
            self.assertIn("audit_cpu_seconds", audit_manifest)

    def test_interruption_after_one_record_keeps_measurement_and_primary_error(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            config, limits, _ = make_partial_dataset(data)
            ticks = iter((0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
            guard = ResourceGuard(0.5, 1024, clock=lambda: next(ticks, 1.0), memory_reader=lambda: 1)
            status = run_audit(Path("configs/tasks/constraint_v1.json"), data,
                ["fixture interrupted audit"], guard=guard, expected_limits=expected_limits(limits))
            self.assertEqual(status, 1)
            report = json.loads((data / "audit_report.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (data / "audit_results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["progress"]["record_interactions_assessed"], 1)
            self.assertEqual(report["g2"]["status"], "not_assessed")
            self.assertEqual(sum(row["kind"] == "record_interaction" for row in rows), 1)
            self.assertEqual(report["error"]["type"], "TimeoutError")
            audit_manifest = json.loads((data / "audit_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(audit_manifest["progress"]["record_interactions_assessed"], 1)

    def test_finalization_cap_does_not_replace_substantive_error(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            _, limits, _ = make_partial_dataset(data)
            ticks = iter((0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0))
            guard = ResourceGuard(0.5, 1024, clock=lambda: next(ticks, 1.0), memory_reader=lambda: 1)
            with patch("dwr.tasks.audit.validate_record", side_effect=ValueError("substantive integrity failure")):
                status = run_audit(Path("configs/tasks/constraint_v1.json"), data,
                    ["fixture primary plus finalization error"], guard=guard,
                    expected_limits=expected_limits(limits))
            self.assertEqual(status, 1)
            report = json.loads((data / "audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["error"]["message"], "substantive integrity failure")
            self.assertTrue(any(item["phase"] == "finalization" for item in report["finalization_errors"]))

    def test_existing_audit_artifacts_are_refused_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            _, limits, _ = make_partial_dataset(data)
            sentinel = data / "audit_results.jsonl"
            sentinel.write_text("prior evidence\n", encoding="utf-8")
            before = sentinel.read_bytes()
            with self.assertRaises(FileExistsError):
                run_audit(Path("configs/tasks/constraint_v1.json"), data, ["second audit"],
                          expected_limits=expected_limits(limits))
            self.assertEqual(sentinel.read_bytes(), before)
            self.assertFalse((data / "audit_manifest.json").exists())

    def test_wrong_partial_solution_is_rejected_even_when_artifact_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            _, limits, _ = make_partial_dataset(data)
            record_path = data / "records.jsonl"
            row = json.loads(record_path.read_text(encoding="utf-8"))
            row["solution"] = {"v0": 2, "v1": 2, "v2": 2}
            record_path.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            manifest_path = data / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["records.jsonl"] = {"sha256": sha256_file(record_path), "bytes": record_path.stat().st_size}
            atomic_json(manifest_path, manifest)
            status = run_audit(Path("configs/tasks/constraint_v1.json"), data, ["tampered partial audit"],
                               expected_limits=expected_limits(limits))
            self.assertEqual(status, 1)
            report = json.loads((data / "audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "audit_error")
            self.assertIn("solution", report["error"]["message"])

    def test_nested_real_producer_manifest_mutations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            config, limits, _ = make_partial_dataset(data)
            original = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
            mutations = (
                lambda m: m.update(seed_derivation="invented"),
                lambda m: m["limits"].update(wall_seconds_global=999999),
                lambda m: m["source"].update(unknown=True),
                lambda m: m["partitions"]["source"].update(unknown=True),
                lambda m: m["artifacts"]["records.jsonl"].update(unknown=True),
                lambda m: m["artifacts"].update({"unexpected-artifact": {"sha256": "0" * 64, "bytes": 0}}),
            )
            for mutate in mutations:
                changed = json.loads(json.dumps(original))
                mutate(changed)
                atomic_json(data / "manifest.json", changed)
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    _verify_manifest(data, config, expected_limits(limits))
            atomic_json(data / "manifest.json", original)


if __name__ == "__main__":
    unittest.main()
