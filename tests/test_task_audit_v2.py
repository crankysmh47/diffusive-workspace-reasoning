import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from dwr.tasks import audit_v2
from dwr.tasks.audit import audit_interactions, make_record
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind


def mixed_task():
    return ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2", "v3", "v4"),
        (Constraint(RelationKind.EQ, ("v1", "v2")),
         Constraint(RelationKind.ADD_MOD, ("v2", "v3", "v4")),
         Constraint(RelationKind.NEQ, ("v1", "v0")),
         Constraint(RelationKind.NEQ, ("v3", "v0"))),
        (GivenAssignment("v0", 0), GivenAssignment("v4", 1)))


def sensitive_source_task():
    return ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
        (Constraint(RelationKind.EQ, ("v0", "v1")),
         Constraint(RelationKind.EQ, ("v1", "v2"))),
        (GivenAssignment("v0", 0),))


class AuditV2Tests(unittest.TestCase):
    def test_independent_mixed_annotation_check_rejects_corrupt_private_fields(self):
        task = mixed_task()
        interaction = audit_interactions(task, "final")
        self.assertTrue(interaction["sensitive"])
        valid = {"partition": "final", "private_evidence": {
            "oracle_unique": True, "source": None,
            "mixed": {"kinds": ["add_mod", "eq", "neq"],
                      "kind_removal_solution_counts": interaction["removal_solution_counts"],
                      "factor_graph_connected": interaction["factor_graph_connected"],
                      "factor_graph_kinds": interaction["factor_graph_kinds"],
                      "all_present_kinds_in_graph": interaction["all_present_kinds_in_graph"],
                      "primary_eligible": interaction["sensitive"],
                      "ineligibility_reason": None}}}
        audit_v2._private_matches(valid, interaction)
        for field, changed in (("factor_graph_connected", False),
                               ("factor_graph_connected", 1),
                               ("kind_removal_solution_counts", [1, 1, 1]),
                               ("primary_eligible", False)):
            corrupt = {**valid, "private_evidence": {**valid["private_evidence"],
                "mixed": {**valid["private_evidence"]["mixed"], field: changed}}}
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_v2._private_matches(corrupt, interaction)

    def test_common_support_four_cohort_exact_cell_floor_and_default_names(self):
        task = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
            tuple(Constraint(RelationKind.EQ, pair) for pair in
                  (("v0", "v1"), ("v0", "v2"), ("v1", "v2"))),
            (GivenAssignment("v0", 0),))
        source = make_record(task, "source", 0)
        source.update({"model_input_bytes": len(source["model_input"].encode()), "primary_eligible": True})
        names = audit_v2.FEASIBILITY_COHORTS
        cohorts = {name: [dict(source) for _ in range(10)] for name in names}
        proxy = audit_v2.v1.common_support(cohorts, min_per_cell=10, min_fraction=0.8,
                                           cohort_names=names)
        self.assertTrue(proxy["pass"])
        self.assertEqual(tuple(proxy["cohorts"]), names)
        with self.assertRaises(ValueError):
            audit_v2.v1.common_support(cohorts, cohort_names=("same", "same"))
        # Four of five examples lie in the shared cell: exactly the registered 80% mass floor.
        cohorts = {}
        for name in names:
            cohorts[name] = [dict(source) for _ in range(4)] + [dict(source, model_input_bytes=2048)]
        boundary = audit_v2.v1.common_support(cohorts, min_per_cell=2, min_fraction=0.8,
                                              cohort_names=names)
        self.assertTrue(boundary["pass"])
        self.assertEqual(boundary["cohorts"][names[0]]["fraction_in_common_cells"], 0.8)

    def test_feasibility_aggregation_ignores_heldout_metric_rows(self):
        from dwr.tasks.config_v2 import load_config_v2
        config = load_config_v2("configs/tasks/constraint_v2.json").resolved
        base = audit_v2._feasibility_metrics([], config)
        opaque = [{"partition": "source", "split": "source_test", "task": object(),
                   "solution": object(), "interaction": {"sensitive": True},
                   "profile": ["eq"], "signature": "opaque"}]
        self.assertEqual(audit_v2._feasibility_metrics(opaque, config), base)

    def test_feasibility_raw_mixed_eligible_fraction_uses_preeligibility_denominator(self):
        from dwr.tasks.config_v2 import load_config_v2
        config = load_config_v2("configs/tasks/constraint_v2.json").resolved
        base = make_record(mixed_task(), "final", 0)
        rows = []
        for index in range(4):
            interaction = audit_interactions(mixed_task(), "final")
            interaction["sensitive"] = index == 0
            rows.append({**base, "partition": "development", "candidate_index": index,
                         "split": "development_validation", "profile": ["eq", "neq"],
                         "model_input_bytes": len(base["model_input"].encode("utf-8")),
                         "interaction": interaction, "primary_eligible": index == 0,
                         "signature": f"signature-{index}"})
        result = audit_v2._feasibility_metrics(rows, config)
        dev = result["development_validation"]
        self.assertEqual(dev["groups"], 4)
        self.assertEqual(dev["eligible_groups"], 1)
        self.assertEqual(dev["eligible_fraction"], 0.25)

    def test_source_sensitivity_fraction_uses_raw_unique_denominator(self):
        from dwr.tasks.config_v2 import load_config_v2
        config = load_config_v2("configs/tasks/constraint_v2.json").resolved
        rows = []
        for index in range(4):
            # This aggregation-only fixture places one sensitive and three insensitive
            # unique rows into the registered source-train denominator.
            if index == 0:
                task = sensitive_source_task()
            else:
                task = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
                    tuple(Constraint(RelationKind.EQ, pair) for pair in
                          (("v0", "v1"), ("v0", "v2"), ("v1", "v2"))),
                    (GivenAssignment("v0", 0),))
            record = make_record(task, "source", index)
            info = audit_v2.v1.validate_record(record, config)
            interaction = audit_interactions(task, "source")
            rows.append({**record, **info, "split": "source_train", "interaction": interaction,
                         "primary_eligible": False})
        failures = []
        result = audit_v2._source_summary(rows, config, failures, feasibility=True)
        self.assertEqual(result["source_groups_in_denominator"], 4)
        self.assertEqual(result["sensitive_groups"], 1)
        self.assertEqual(result["sensitivity_fraction"], 0.25)

    def test_source_common_support_cohorts_exclude_rejected_unique_rows(self):
        from dwr.tasks.config_v2 import load_config_v2
        config = load_config_v2("configs/tasks/constraint_v2.json").resolved
        task = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
            tuple(Constraint(RelationKind.EQ, pair) for pair in
                  (("v0", "v1"), ("v0", "v2"), ("v1", "v2"))),
            (GivenAssignment("v0", 0),))
        record = make_record(task, "source", 0)
        info = audit_v2.v1.validate_record(record, config)
        interaction = audit_interactions(task, "source")
        self.assertFalse(interaction["sensitive"])
        row = {**record, **info, "interaction": interaction, "primary_eligible": False}
        feasibility = audit_v2._feasibility_metrics([dict(row, split="source_train")], config)
        full = audit_v2._full_metrics([dict(row, split="source_test")], config)
        self.assertEqual(feasibility["four_cohort_common_support_proxy"]["cohorts"]["source_train"]["count"], 0)
        self.assertEqual(full["five_cohort_common_support"]["cohorts"][audit_v2.v1.COHORTS[0]]["count"], 0)
        self.assertEqual(feasibility["source_train"]["source_groups_in_denominator"], 1)

    def test_full_source_does_not_apply_feasibility_only_train_value_or_clue_floors(self):
        from dwr.tasks.config_v2 import load_config_v2
        config = load_config_v2("configs/tasks/constraint_v2.json").resolved
        rows = []
        for split, value, given_count, count in (
                ("source_train", 0, 1, 100), ("source_validation", 1, 2, 20),
                ("source_test", 2, 2, 20)):
            variables = ("v0", "v1", "v2", "v3")
            task = ConstraintTask("constraint-task-v1", 3, variables,
                tuple(Constraint(RelationKind.EQ, (variables[i], variables[i + 1]))
                      for i in range(given_count - 1, 3)),
                tuple(GivenAssignment(variables[i], value) for i in range(given_count)))
            record = make_record(task, "source", 0)
            info = audit_v2.v1.validate_record(record, config)
            interaction = audit_interactions(task, "source")
            self.assertTrue(interaction["sensitive"])
            rows.extend({**record, **info, "split": split, "interaction": interaction,
                         "primary_eligible": False} for _ in range(count))
        full_failures = audit_v2._full_metrics(rows, config)["floor_failures"]
        self.assertNotIn("source_train_solution_value_balance", full_failures)
        self.assertNotIn("source_train_given_count_variation", full_failures)
        feasibility_failures = []
        audit_v2._source_summary(rows, config, feasibility_failures, feasibility=True)
        self.assertIn("source_train_solution_value_balance", feasibility_failures)
        self.assertIn("source_train_given_count_variation", feasibility_failures)

    def test_full_metrics_uses_raw_unique_denominators_and_reports_rejected_only_signatures(self):
        # This fixture exercises the pure aggregation interface; it does not certify G2.
        task = mixed_task()
        base = make_record(task, "final", 0)
        info = audit_v2.v1.validate_record(base, {
            "audit_gate": {"min_accepted_groups_per_partition": 100,
                "min_structure_signatures_per_partition": 12,
                "min_distinct_given_counts_per_partition": 2,
                "min_solution_value_fraction_each_partition": 0.1,
                "source_train_min_structure_signatures": 12,
                "min_source_constraint_sensitive_fraction": 0.25,
                "min_source_sensitive_examples_per_signature": 1,
                "min_primary_eligible_groups_per_mixed_profile": 100,
                "min_primary_eligible_structure_signatures_per_mixed_profile": 12,
                "min_primary_eligible_examples_per_heldout_signature": 1,
                "min_primary_eligible_fraction_per_mixed_partition": 0.25,
                "development_target_min_groups": {"development_validation": 80,
                                                    "development_diagnostic_test": 20},
                "common_support": {"min_groups_per_cell_per_cohort": 10,
                                   "min_group_fraction_per_cohort_in_common_cells": 0.8},
                "source_id_groups_min": {"train": 100, "validation": 20, "test": 20}},
            "composition_partition": {"development_target_split": {
                "min_validation_groups_per_profile": 20,
                "min_diagnostic_test_groups_per_profile": 5}}})
        interaction = audit_interactions(task, "final")
        row = {**base, **info, "interaction": interaction,
               "primary_eligible": interaction["sensitive"]}
        config = {"audit_gate": {"min_accepted_groups_per_partition": 100,
                "min_structure_signatures_per_partition": 12,
                "min_distinct_given_counts_per_partition": 2,
                "min_solution_value_fraction_each_partition": 0.1,
                "source_train_min_structure_signatures": 12,
                "min_source_constraint_sensitive_fraction": 0.25,
                "min_source_sensitive_examples_per_signature": 1,
                "min_primary_eligible_groups_per_mixed_profile": 100,
                "min_primary_eligible_structure_signatures_per_mixed_profile": 12,
                "min_primary_eligible_examples_per_heldout_signature": 1,
                "min_primary_eligible_fraction_per_mixed_partition": 0.25,
                "development_target_min_groups": {"development_validation": 80,
                                                    "development_diagnostic_test": 20},
                "common_support": {"min_groups_per_cell_per_cohort": 10,
                                   "min_group_fraction_per_cohort_in_common_cells": 0.8},
                "source_id_groups_min": {"train": 100, "validation": 20, "test": 20}},
            "composition_partition": {"development_target_split": {
                "min_validation_groups_per_profile": 20,
                "min_diagnostic_test_groups_per_profile": 5}}}
        result = audit_v2._full_metrics([row], config)
        profile = "/".join(base["profile"])
        self.assertEqual(result["partitions"]["final"]["profiles"][profile]["unique_pre_eligibility"], 1)
        self.assertEqual(result["partitions"]["final"]["profiles"][profile]["eligible"], 1)
        self.assertFalse(result["structural_pass"])
        self.assertIn("final_min_groups", result["floor_failures"])

    def test_full_support_uses_registered_five_cohort_names(self):
        from dwr.tasks.config_v2 import load_config_v2
        config = load_config_v2("configs/tasks/constraint_v2.json").resolved
        source_task = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
            tuple(Constraint(RelationKind.EQ, pair) for pair in
                  (("v0", "v1"), ("v0", "v2"), ("v1", "v2"))),
            (GivenAssignment("v0", 0),))
        source_record = make_record(source_task, "source", 0)
        source_info = audit_v2.v1.validate_record(source_record, config)
        source_interaction = audit_interactions(source_task, "source")
        source_row = {**source_record, **source_info, "interaction": source_interaction,
                      "primary_eligible": False}
        final_task = mixed_task()
        final_record = make_record(final_task, "final", 0)
        final_info = audit_v2.v1.validate_record(final_record, config)
        final_interaction = audit_interactions(final_task, "final")
        final_row = {**final_record, **final_info, "interaction": final_interaction,
                     "primary_eligible": final_interaction["sensitive"]}
        dev_rows = [dict(final_row, partition="development", split="development_validation",
                         profile=list(profile), primary_eligible=True)
                    for profile in audit_v2.v1.PROFILES["development"]]
        result = audit_v2._full_metrics([source_row, *dev_rows, final_row], config)
        self.assertEqual(tuple(result["five_cohort_common_support"]["cohorts"]), audit_v2.v1.COHORTS)

    def test_cli_refuses_preexisting_audit_artifacts(self):
        from dwr.tasks.audit_cli_v2 import main
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            marker = data / "audit_report.json"
            marker.write_text("preserve\n", encoding="utf-8")
            with redirect_stdout(StringIO()):
                rc = main(["--config", "missing.json", "--mode", "feasibility", "--data", str(data)])
            self.assertEqual(rc, 1)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve\n")
            self.assertFalse((data / "audit_manifest.json").exists())

    def test_final_manifest_write_failure_invalidates_pass_gate_and_exit(self):
        from dwr.tasks import audit_cli_v2
        from dwr.tasks.run_support import ResourceGuard, atomic_json, stable_json
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            report = {"status": "complete", "g2_status": "PASS_FEASIBILITY"}
            source = {"commit": "a" * 40, "dirty": False,
                      "dirty_diff_sha256": "b" * 64, "checkout_root": directory}
            gate = {"audit_version": audit_v2.AUDIT_VERSION, "status": "complete",
                    "mode": "feasibility", "g2_status": "PASS_FEASIBILITY",
                    "config_sha256": "c" * 64, "source": source,
                    "generation_manifest_sha256": "d" * 64, "generation_artifacts": {}}
            manifest = {"status": "complete"}
            (data / "audit_manifest.json").write_text("{}\n", encoding="utf-8")
            (data / "audit_report.json").write_text(stable_json(report) + "\n", encoding="utf-8")
            (data / "audit_gate.json").write_text(stable_json(gate) + "\n", encoding="utf-8")
            (data / "preflight_report.json").write_text("{}\n", encoding="utf-8")
            failed_once = {"value": False}

            def fail_initial_terminal_write(path, value):
                if Path(path).name == "audit_manifest.json" and not failed_once["value"]:
                    failed_once["value"] = True
                    raise OSError("synthetic terminal manifest write failure")
                return atomic_json(path, value)

            guard = ResourceGuard(60, 1024, memory_reader=lambda: 1)
            with patch.object(audit_cli_v2, "atomic_json", side_effect=fail_initial_terminal_write):
                error = audit_cli_v2._finalize_manifest(data, manifest, report, gate, guard,
                    {"generation_elapsed_seconds": 1.0, "generation_cpu_seconds": 1.0,
                     "peak_resident_bytes": None}, time.process_time(), None, [], charge_guard=True)
            self.assertIsNotNone(error)
            self.assertEqual(error["phase"], "audit_finalization")
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(gate["g2_status"], "not_assessed")
            self.assertFalse((data / "preflight_report.json").exists())
            self.assertTrue((data / "preflight_report.invalidated.json").exists())
            self.assertEqual(audit_cli_v2._exit_code(report), 1)

    def test_setup_refusal_records_command_environment_and_original_sidecar(self):
        import json
        from contextlib import redirect_stdout
        from io import StringIO
        from dwr.tasks.audit_cli_v2 import main
        from dwr.tasks.run_support import strict_json_loads
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            original = {"status": "incomplete", "primary_error": {"message": "generation failed"}}
            sidecar = data / "terminal_failure.json"
            sidecar.write_text(json.dumps(original), encoding="utf-8")
            with redirect_stdout(StringIO()):
                rc = main(["--config", "configs/tasks/constraint_v2.json", "--mode", "feasibility", "--data", str(data)])
            self.assertEqual(rc, 1)
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8")), original)
            report = strict_json_loads((data / "audit_report.json").read_text(encoding="utf-8"))
            manifest = strict_json_loads((data / "audit_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "not_assessed")
            self.assertIn("generation has a terminal_failure.json", report["error"]["message"])
            self.assertEqual(report["generation_terminal_failure"]["content"], original)
            self.assertIn("dwr.tasks.audit_cli_v2", manifest["command"])
            self.assertIn("python", manifest)
            self.assertEqual(manifest["status"], "incomplete")

    def test_setup_config_error_is_durably_recorded_once(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from dwr.tasks.audit_cli_v2 import main
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            with redirect_stdout(StringIO()):
                rc = main(["--config", str(data / "missing.json"), "--mode", "feasibility", "--data", str(data)])
            self.assertEqual(rc, 1)
            self.assertTrue((data / "audit_manifest.json").exists())
            self.assertTrue((data / "audit_report.json").exists())
            captured = (data / "audit_manifest.json").read_text(encoding="utf-8")
            self.assertIn("ConfigError", captured)
            with redirect_stdout(StringIO()):
                retry = main(["--config", "configs/tasks/constraint_v2.json", "--mode", "feasibility", "--data", str(data)])
            self.assertEqual(retry, 1)
            self.assertEqual((data / "audit_manifest.json").read_text(encoding="utf-8"), captured)

    def test_detailed_report_write_crossing_budget_invalidates_gate(self):
        from dwr.tasks import audit_cli_v2
        from dwr.tasks.run_support import ResourceGuard
        now = [0.0]
        guard = ResourceGuard(0.5, 1024, clock=lambda: now[0], memory_reader=lambda: 1)
        report = {"status": "complete", "g2_status": "PASS_FEASIBILITY", "resource": {}}
        gate = {"audit_version": audit_v2.AUDIT_VERSION, "status": "complete",
                "mode": "feasibility", "g2_status": "PASS_FEASIBILITY",
                "config_sha256": "c" * 64, "source": {},
                "generation_manifest_sha256": "d" * 64, "generation_artifacts": {}}
        manifest = {"status": "complete"}
        writes = []
        def writer(path, value):
            writes.append(Path(path).name)
            if Path(path).name == "audit_report.json":
                now[0] += 1.0
        with tempfile.TemporaryDirectory() as directory, patch.object(audit_cli_v2, "atomic_json", side_effect=writer):
            error = audit_cli_v2._finalize_manifest(Path(directory), manifest, report, gate, guard,
                {"generation_elapsed_seconds": 599.5, "generation_cpu_seconds": 0.0,
                 "peak_resident_bytes": 1}, time.process_time(), None, [], charge_guard=True)
        self.assertIsNotNone(error)
        self.assertIn("audit_report.json", writes)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(gate["g2_status"], "not_assessed")
        self.assertEqual(audit_cli_v2._exit_code(report), 1)

    def test_persisted_late_failures_replace_false_pass_with_bounded_summary(self):
        from dwr.tasks import audit_cli_v2
        from dwr.tasks.run_support import ResourceGuard, atomic_json, sha256_file, strict_json_loads

        def exercise(*, delayed_report=False, fail_terminal_tail=False):
            now = [0.0]
            guard = ResourceGuard(0.5 if delayed_report else 60.0, 1024,
                                  clock=lambda: now[0], memory_reader=lambda: 1)
            report = {"status": "complete", "g2_status": "PASS_FEASIBILITY",
                      "mode": "feasibility", "metrics": {"fixture": "retained"},
                      "per_example_outcomes": [{"candidate": 3}],
                      "resource": {"audit_elapsed_seconds": 0.0}}
            gate = {"audit_version": audit_v2.AUDIT_VERSION, "status": "complete",
                    "mode": "feasibility", "g2_status": "PASS_FEASIBILITY",
                    "config_sha256": "c" * 64, "source": {},
                    "generation_manifest_sha256": "d" * 64, "generation_artifacts": {}}
            manifest = {"status": "complete"}
            with tempfile.TemporaryDirectory() as directory:
                data = Path(directory)
                wrapper = data / "preflight_report.json"
                wrapper.write_text("{}\n", encoding="utf-8")
                calls = {"manifest": 0}

                def writer(path, value):
                    path = Path(path)
                    if path.name == "audit_manifest.json":
                        calls["manifest"] += 1
                        if fail_terminal_tail and calls["manifest"] == 2:
                            raise OSError("injected terminal metadata flush failure")
                    atomic_json(path, value)
                    if delayed_report and path.name == "audit_report.json":
                        now[0] += 1.0

                with patch.object(audit_cli_v2, "atomic_json", side_effect=writer):
                    error = audit_cli_v2._finalize_manifest(data, manifest, report, gate, guard,
                        {"generation_elapsed_seconds": 599.5 if delayed_report else 0.0,
                         "generation_cpu_seconds": 0.0, "peak_resident_bytes": 1},
                        time.process_time(), None, [], charge_guard=True)

                self.assertIsNotNone(error)
                summary = strict_json_loads((data / "audit_report.json").read_text(encoding="utf-8"))
                detail_path = data / "audit_report.provisional.json"
                detail = strict_json_loads(detail_path.read_text(encoding="utf-8"))
                saved_gate = strict_json_loads((data / "audit_gate.json").read_text(encoding="utf-8"))
                saved_manifest = strict_json_loads((data / "audit_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["status"], "incomplete")
                self.assertEqual(summary["g2_status"], "not_assessed")
                self.assertTrue(summary["error"])
                self.assertEqual(summary["retained_detail_report"]["path"], detail_path.name)
                self.assertEqual(summary["retained_detail_report"]["sha256"], sha256_file(detail_path))
                self.assertEqual(detail["per_example_outcomes"], [{"candidate": 3}])
                self.assertEqual(detail["metrics"], {"fixture": "retained"})
                self.assertEqual(saved_gate["status"], "incomplete")
                self.assertEqual(saved_gate["g2_status"], "not_assessed")
                self.assertEqual(saved_manifest["status"], "incomplete")
                self.assertEqual(saved_manifest["error"], error)
                if delayed_report:
                    self.assertGreaterEqual(saved_manifest["combined_generation_audit_elapsed_seconds"], 600.5)
                if fail_terminal_tail:
                    self.assertTrue(saved_manifest["finalization_errors"])
                self.assertFalse(wrapper.exists())
                self.assertTrue((data / "preflight_report.invalidated.json").exists())
                self.assertEqual(audit_cli_v2._exit_code(summary), 1)

        exercise(delayed_report=True)
        exercise(fail_terminal_tail=True)

    def test_success_report_marks_manifest_as_authoritative_final_accounting(self):
        from dwr.tasks import audit_cli_v2
        from dwr.tasks.run_support import ResourceGuard, strict_json_loads
        report = {"status": "complete", "g2_status": "PASS_FEASIBILITY",
                  "resource": {"audit_elapsed_seconds": 0.0}}
        gate = {"audit_version": audit_v2.AUDIT_VERSION, "status": "complete",
                "mode": "feasibility", "g2_status": "PASS_FEASIBILITY",
                "config_sha256": "c" * 64, "source": {},
                "generation_manifest_sha256": "d" * 64, "generation_artifacts": {}}
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            manifest = {"status": "complete"}
            error = audit_cli_v2._finalize_manifest(data, manifest, report, gate,
                ResourceGuard(60, 1024, memory_reader=lambda: 1),
                {"generation_elapsed_seconds": 1.0, "generation_cpu_seconds": 1.0,
                 "peak_resident_bytes": 1}, time.process_time(), None, [], charge_guard=True)
            self.assertIsNone(error)
            saved_report = strict_json_loads((data / "audit_report.json").read_text(encoding="utf-8"))
            saved_manifest = strict_json_loads((data / "audit_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_report["resource"]["accounting_snapshot_boundary"],
                         "before_detailed_report_serialization")
        self.assertEqual(saved_report["resource"]["authoritative_final_accounting"], "audit_manifest.json")
        self.assertFalse(saved_report["resource"]["accounting_is_final"])
        self.assertIn("authoritative charged snapshot", saved_manifest["accounting_note"])

    def test_invalidation_failure_keeps_memory_and_sidecar_fail_closed(self):
        from dwr.tasks import audit_cli_v2
        from dwr.tasks.run_support import atomic_json, strict_json_loads
        report = {"status": "complete", "g2_status": "PASS_FEASIBILITY",
                  "mode": "feasibility", "per_example_outcomes": [{"candidate": 8}]}
        gate = {"status": "complete", "g2_status": "PASS_FEASIBILITY"}
        manifest = {"status": "complete"}
        primary = {"type": "TimeoutError", "message": "primary budget failure", "phase": "audit_finalization"}
        finalization_errors = []
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            atomic_json(data / "audit_report.json", report)

            def fail_detail_rename(path, target):
                if Path(path).name == "audit_report.json":
                    raise OSError("injected detail retention failure")
                return original_replace(path, target)

            original_replace = Path.replace
            with patch.object(Path, "replace", fail_detail_rename):
                audit_cli_v2._mark_incomplete(data, report, gate, manifest, primary, finalization_errors)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["g2_status"], "not_assessed")
            self.assertEqual(audit_cli_v2._exit_code(report), 1)
            sidecar = strict_json_loads((data / "terminal_failure.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["primary_error"], primary)
            self.assertIn("injected detail retention failure",
                          sidecar["finalization_errors"][0]["message"])

    def test_complete_no_go_analysis_exits_nonzero(self):
        from dwr.tasks.audit_cli_v2 import _exit_code
        self.assertEqual(_exit_code({"status": "complete", "g2_status": "NO_GO"}), 1)


if __name__ == "__main__":
    unittest.main()
