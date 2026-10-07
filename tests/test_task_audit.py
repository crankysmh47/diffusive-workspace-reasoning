import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dwr.tasks.audit import (audit_interactions, common_support, read_records,
                             replay_outcomes, run_audit, source_train_primitive_coverage,
                             source_train_composition_absence, validate_corpus_closure,
                             validate_record)
from dwr.tasks.canonical import content_id
from dwr.tasks.run_support import load_config, sha256_file, stable_json
from dwr.tasks.run_support import ResourceGuard
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind


def task_for_negative():
    # The source nontriviality counterexample: every removed edge leaves the
    # equality graph connected, so uniqueness does not change.
    return ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
                          tuple(Constraint(RelationKind.EQ, pair) for pair in
                                (("v0", "v1"), ("v0", "v2"), ("v1", "v2"))),
                          (GivenAssignment("v0", 0),))


class AuditTests(unittest.TestCase):
    def test_source_sensitivity_preserves_triangle_no_go(self):
        result = audit_interactions(task_for_negative(), "source")
        self.assertFalse(result["sensitive"])
        self.assertEqual(result["removal_solution_counts"], [1, 1, 1])

    def test_record_reconstructs_identity_and_rejects_tampered_solution(self):
        config = load_config("configs/tasks/constraint_v1.json")
        task = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
                              (Constraint(RelationKind.EQ, ("v0", "v1")),
                               Constraint(RelationKind.EQ, ("v1", "v2"))),
                              (GivenAssignment("v0", 1),))
        from dwr.tasks.audit import make_record
        record = make_record(task, "source", 0)
        validate_record(record, config.resolved)
        record["solution"]["v2"] = 0
        with self.assertRaises(ValueError):
            validate_record(record, config.resolved)

    def test_record_preserves_registered_mixed_profile_order(self):
        config = load_config("configs/tasks/constraint_v1.json")
        from dwr.tasks.audit import make_record
        task = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
            (Constraint(RelationKind.EQ, ("v0", "v1")),
             Constraint(RelationKind.ADD_MOD, ("v0", "v1", "v2"))),
            (GivenAssignment("v0", 0),))
        record = make_record(task, "development", 0)
        self.assertEqual(record["profile"], ["eq", "add_mod"])
        self.assertEqual(validate_record(record, config.resolved)["profile"], ("eq", "add_mod"))

    def test_record_rejects_identity_split_hash_and_annotation_tampering(self):
        config = load_config("configs/tasks/constraint_v1.json")
        task = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
                              (Constraint(RelationKind.EQ, ("v0", "v1")),
                               Constraint(RelationKind.EQ, ("v1", "v2"))),
                              (GivenAssignment("v0", 1),))
        from dwr.tasks.audit import make_record
        original = make_record(task, "source", 0)
        for key, value in (("content_id", "0" * 64), ("lineage_id", "1" * 64),
                           ("structure_signature", "2" * 64), ("split", "source_test"),
                           ("model_input_sha256", "3" * 64), ("model_input", original["model_input"] + " ")):
            changed = dict(original, **{key: value})
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_record(changed, config.resolved)
        changed = dict(original, answer_annotations={"solution": original["solution"]})
        with self.assertRaises(ValueError):
            validate_record(changed, config.resolved)

    def test_mixed_all_kind_removal_and_factor_graph_positive_negative(self):
        positive = ConstraintTask("constraint-task-v1", 3,
            ("v0", "v1", "v2", "v3", "v4"),
            (Constraint(RelationKind.EQ, ("v1", "v2")),
             Constraint(RelationKind.ADD_MOD, ("v2", "v3", "v4")),
             Constraint(RelationKind.NEQ, ("v1", "v0")),
             Constraint(RelationKind.NEQ, ("v3", "v0"))),
            (GivenAssignment("v0", 0), GivenAssignment("v4", 1)))
        negative = ConstraintTask("constraint-task-v1", 3,
            tuple(f"v{i}" for i in range(6)),
            (Constraint(RelationKind.EQ, ("v0", "v5")),
             Constraint(RelationKind.NEQ, ("v1", "v0")),
             Constraint(RelationKind.NEQ, ("v3", "v0")),
             Constraint(RelationKind.NEQ, ("v1", "v3")),
             Constraint(RelationKind.NEQ, ("v2", "v0")),
             Constraint(RelationKind.ADD_MOD, ("v2", "v3", "v4"))),
            (GivenAssignment("v0", 0), GivenAssignment("v4", 1), GivenAssignment("v5", 0)))
        self.assertEqual(len(__import__("dwr.tasks.oracle", fromlist=["enumerate_solutions"]).enumerate_solutions(positive).solutions), 1)
        self.assertTrue(audit_interactions(positive, "development")["sensitive"])
        negative_result = audit_interactions(negative, "final")
        self.assertFalse(negative_result["sensitive"])
        self.assertEqual(next(item["solution_count"] for item in negative_result["removals"]
                              if item.get("kind") == "eq"), 1)
        self.assertFalse(negative_result["all_present_kinds_in_graph"])

    def test_reader_rejects_unknown_record_fields_and_hash_mismatch(self):
        row = {"unexpected": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(stable_json(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_records(path)

    def test_corpus_closure_rejects_split_crossing_signature_and_duplicate_lineage(self):
        rows = [dict(content_id="a", lineage_id="la", partition="development", signature="s",
                     split="development_validation"),
                dict(content_id="b", lineage_id="lb", partition="development", signature="s",
                     split="development_diagnostic_test")]
        with self.assertRaisesRegex(ValueError, "signature crosses split"):
            validate_corpus_closure(rows)
        rows[1]["split"] = "development_validation"
        rows[1]["lineage_id"] = "la"
        with self.assertRaisesRegex(ValueError, "duplicate lineage_id"):
            validate_corpus_closure(rows)

    def test_source_train_primitive_coverage_is_not_corpus_coverage(self):
        eq = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
            (Constraint(RelationKind.EQ, ("v0", "v1")), Constraint(RelationKind.EQ, ("v1", "v2"))),
            (GivenAssignment("v0", 0),))
        neq = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
            (Constraint(RelationKind.NEQ, ("v0", "v1")), Constraint(RelationKind.NEQ, ("v1", "v2"))),
            (GivenAssignment("v0", 0),))
        result = source_train_primitive_coverage([
            {"task": eq, "split": "source_train"},
            {"task": neq, "split": "source_validation"}])
        self.assertEqual(result["corpus_kinds"], ["eq", "neq"])
        self.assertEqual(result["source_train_kinds"], ["eq"])
        self.assertFalse(result["pass"])
        singletons = source_train_composition_absence([
            {"task": eq, "split": "source_train"},
            {"task": neq, "split": "source_train"}])
        self.assertTrue(singletons["heldout_compositions_absent"])
        mixed = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
            (Constraint(RelationKind.EQ, ("v0", "v1")),
             Constraint(RelationKind.NEQ, ("v1", "v2"))), (GivenAssignment("v0", 0),))
        self.assertFalse(source_train_composition_absence([
            {"task": mixed, "split": "source_train"}])["heldout_compositions_absent"])

    def test_replay_matches_frozen_candidate_and_detects_tampering(self):
        from dwr.tasks.audit import make_outcome_record
        from dwr.tasks.generator import generate_candidate
        rows = []
        seen = set()
        for index in range(28):
            outcome = generate_candidate(20261007, "source", index)
            repeated = outcome.accepted and content_id(outcome.task) in seen
            rows.append(make_outcome_record(outcome, repeated))
            if outcome.accepted and not repeated:
                seen.add(content_id(outcome.task))
        self.assertEqual(replay_outcomes(rows, 20261007, {"source": {"eq"}},
                                         {"source": set()}), rows)
        tampered = [*rows]
        tampered[-1] = dict(tampered[-1], candidate_index=28)
        with self.assertRaises(ValueError):
            replay_outcomes(tampered, 20261007, {"source": {"eq"}}, {"source": set()})

    def test_common_support_empty_cohort_fails_explicitly(self):
        cohorts = {name: [] for name in ("source_id_test_all_groups",
                   "development_validation_eq_neq_primary_eligible",
                   "development_validation_eq_add_mod_primary_eligible",
                   "development_validation_neq_add_mod_primary_eligible",
                   "final_test_eq_neq_add_mod_primary_eligible")}
        result = common_support(cohorts)
        self.assertFalse(result["pass"])
        self.assertEqual(result["status"], "measured_fail")
        self.assertTrue(all(not item["nonempty"] for item in result["cohorts"].values()))

    def test_common_support_drops_ineligible_mixed_groups_before_denominator(self):
        task = task_for_negative()
        from dwr.tasks.audit import make_record
        row = make_record(task, "source", 0)
        row.update({"model_input_bytes": len(row["model_input"].encode()), "primary_eligible": True})
        cohorts = {name: [dict(row) for _ in range(10)] for name in
                   ("source_id_test_all_groups", "development_validation_eq_neq_primary_eligible",
                    "development_validation_eq_add_mod_primary_eligible",
                    "development_validation_neq_add_mod_primary_eligible",
                    "final_test_eq_neq_add_mod_primary_eligible")}
        cohorts["development_validation_eq_neq_primary_eligible"].append(dict(row, primary_eligible=False))
        result = common_support(cohorts)
        self.assertTrue(result["pass"])
        self.assertEqual(result["cohorts"]["development_validation_eq_neq_primary_eligible"]["count"], 10)

    def test_prospective_g2_fixture_passes_and_no_go_fixture_fails(self):
        from dwr.tasks.audit import assess_g2
        config = load_config("configs/tasks/constraint_v1.json").resolved
        positive = {"floor_failures": [], "common_support": {"pass": True},
                    "generation_complete": True, "replay_pass": True,
                    "integrity_pass": True, "structural_pass": True}
        self.assertTrue(assess_g2(positive, config)["pass"])
        no_go = dict(positive, structural_pass=False)
        self.assertFalse(assess_g2(no_go, config)["pass"])

    def test_zero_remaining_budget_writes_blocked_report_without_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "manifest.json").write_text(json.dumps({"generation_elapsed_seconds": 600,
                "status": "incomplete"}), encoding="utf-8")
            guard = ResourceGuard(600, 1024, clock=lambda: 1.0, memory_reader=lambda: 1)
            result = run_audit(Path("configs/tasks/constraint_v1.json"), data, ["fixture"], guard=guard)
            self.assertEqual(result, 1)
            report = json.loads((data / "audit_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((data / "audit_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["g2"]["status"], "not_assessed")
            self.assertNotIn("replay_pass", manifest)
            self.assertFalse((data / "audit_results.jsonl").exists())

    def test_incomplete_generation_run_stays_not_assessed_and_strict_manifest_rejects_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            data.mkdir()
            (data / "outcomes.jsonl").write_text("", encoding="utf-8")
            (data / "records.jsonl").write_text("", encoding="utf-8")
            config = load_config("configs/tasks/constraint_v1.json")
            manifest = {"manifest_version": "dwr-task-run-v1", "status": "incomplete",
                "exit_code": 1, "generation_elapsed_seconds": 1.0, "config_sha256": config.sha256,
                "resolved_config": config.resolved, "seed": config.resolved["seed"],
                "seed_derivation": config.resolved["seed_derivation"],
                "source": {"commit": "fixture", "dirty": False, "dirty_diff_sha256": "0" * 64},
                "partitions": {}, "artifacts": {}}
            for name in ("outcomes.jsonl", "records.jsonl"):
                path = data / name
                manifest["artifacts"][name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            (data / "manifest.json").write_text(stable_json(manifest) + "\n", encoding="utf-8")
            result = run_audit(Path("configs/tasks/constraint_v1.json"), data, ["fixture"])
            self.assertEqual(result, 1)
            report = json.loads((data / "audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "incomplete_generation", report)
            self.assertEqual(report["g2"]["status"], "not_assessed")
            self.assertFalse(json.loads((data / "audit_manifest.json").read_text(encoding="utf-8")).get("replay_pass", False))

            manifest["unexpected"] = True
            (data / "manifest.json").write_text(stable_json(manifest) + "\n", encoding="utf-8")
            result = run_audit(Path("configs/tasks/constraint_v1.json"), data, ["fixture"])
            self.assertEqual(result, 1)
            report = json.loads((data / "audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "audit_error")
            self.assertEqual(report["g2"]["status"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
