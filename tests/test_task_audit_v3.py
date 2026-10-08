import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stderr
from io import StringIO

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
            audit_manifest = json.loads((data / "audit_manifest.json").read_text(encoding="utf-8"))
            generation_manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "complete")
            self.assertEqual(persisted["g2_status"], "not_assessed")
            self.assertFalse((data / "preflight_report.json").exists())
            self.assertAlmostEqual(audit_manifest["remaining_global_seconds_at_start"],
                limits.wall_seconds - generation_manifest["generation_elapsed_seconds"], places=5)
            self.assertEqual(audit_manifest["accounting_end_boundary"],
                "post_first_terminal_manifest_write_resource_check")
            self.assertAlmostEqual(audit_manifest["combined_generation_audit_cpu_seconds"],
                generation_manifest["generation_cpu_seconds"] + audit_manifest["audit_cpu_seconds"], places=5)
            self.assertEqual(audit_manifest["combined_peak_resident_bytes"], max(
                generation_manifest["peak_resident_bytes"], audit_manifest["peak_resident_bytes"]))
        self.assertEqual(report["audit_version"], "dwr-task-v3-audit-v1")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["g2_status"], "not_assessed", report)
        self.assertEqual(report["proposal_status"], "NO_GO")
        self.assertEqual(report["proposal_reason"], "source_support_capacity_bound")
        self.assertEqual(report["capacity_bound"]["status"], "NO_GO")
        self.assertIsNone(report["four_cohort_common_support_proxy"])
        self.assertEqual(report["final_assessment"], "not_assessed")
        self.assertNotIn("accepted_records", report["replay"])
        self.assertEqual(report["accepted_record_count"], 0)
        self.assertEqual(report["independent_unique_candidate_count"], 0)

    def test_wrong_stream_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises((ValueError, FileNotFoundError)):
                run_audit_v3(Path(temp), self.config, {}, "feasibility")

    def test_independent_capacity_histogram_uses_shared_cell_boundaries(self):
        from dwr.tasks import audit
        from dwr.tasks.audit_v2 import _audit_source_capacity
        from dwr.tasks.dataset_v2 import accepted_record, _split_and_ids
        from dwr.tasks.generator_v3 import generate_candidate_v3

        outcome = generate_candidate_v3(20261007, "source", 1)
        self.assertTrue(outcome.oracle_unique and outcome.eligible)
        split, identities = _split_and_ids(outcome)
        record = accepted_record(outcome, "source_train", identities,
                                 expected_generator_id="dwr-generator-v3")
        sizes = (511, 512, 639, 640, 767, 768, 1023, 1024)
        nodes = (1, 2, 4, 8, 16, 32, 64)
        prompt_bins = {511: 0, 512: 1, 639: 1, 640: 2, 767: 2, 768: 3, 1023: 3, 1024: 4}
        node_bins = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6}
        records = []
        expected_cells = []
        for size in sizes:
            for node_count in nodes:
                item = dict(record, model_input="x" * size,
                            solver_stats=dict(record["solver_stats"], first_solution_nodes=node_count))
                item["model_input_bytes"] = len(item["model_input"].encode("utf-8"))
                records.append(item)
                cell = audit._cell(item)
                self.assertEqual(cell[3:], (prompt_bins[size], node_bins[node_count]))
                expected_cells.append(cell)
        observed = []
        actual_cell = audit._cell
        def capture(prepared):
            observed.append(prepared["model_input_bytes"])
            return actual_cell(prepared)
        with patch.object(audit, "_cell", side_effect=capture):
            result = _audit_source_capacity(records, "a" * 64)
        self.assertEqual(observed, [len(item["model_input"].encode("utf-8")) for item in records])
        self.assertEqual(result["source_train_group_count"], len(records))
        from collections import Counter
        expected_counts = Counter(expected_cells)
        ordered = sorted(expected_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        self.assertEqual(result["top_cells"], [
            {"cell": list(cell), "count": count} for cell, count in ordered
        ])

    def test_incomplete_full_report_keeps_final_assessment_not_assessed(self):
        import json
        from dwr.tasks.dataset_v2 import DatasetLimits, run_dataset_v3
        from dwr.tasks.run_support import ResourceGuard

        limits = DatasetLimits({"eq": 1, "neq": 0, "add_mod": 0}, {}, {}, 0, 30, 1024)
        source = {"commit": "fixture", "dirty": False, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        identity = {"sha256": "a" * 64, "bytes": 0}
        consumed = {"linkage_version": "dwr-task-v3-consumed-preflight-v1",
                    "wrapper_path": "fixture/preflight_report.json", "wrapper_sha256": "b" * 64,
                    "config_sha256": self.config.sha256, "source": dict(source),
                    "generation_manifest_sha256": "c" * 64,
                    "generation_artifacts": {name: dict(identity) for name in (
                        "outcomes.jsonl", "records.jsonl", "source_capacity_bound.json",
                        "source_prefix_context.json", "source_prefix_replay.json")},
                    "audit_artifact_path": "fixture/audit_gate.json", "audit_artifact_sha256": "d" * 64}
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "full"
            with patch("dwr.tasks.dataset_v2.validate_preflight_v3", return_value=consumed):
                exit_code = run_dataset_v3(self.config, data, "full", ["fixture"], limits=limits,
                                           source=source, preflight_report=Path("fixture"))
            self.assertEqual(exit_code, 1)
            from dwr.tasks.manifest_v2 import expected_limits
            saved = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["limits"], expected_limits("full", limits))
            with patch("dwr.tasks.dataset_v2.registered_limits", return_value=limits):
                report = run_audit_v3(data, self.config, source, "full",
                                      guard=ResourceGuard(30, 1024, memory_reader=lambda: 0))
        self.assertFalse(report["generation_complete"])
        self.assertEqual(report["unassessed"], ["unattempted generation targets", "G2"])
        self.assertEqual(report["final_assessment"], "not_assessed")

    def test_prefix_replay_operational_failure_writes_durable_not_assessed_report(self):
        from dwr.tasks.audit_cli_v3 import main as audit_main
        from dwr.tasks.stream_contract import _STREAMS
        from dwr.tasks.generator_v3 import generate_candidate_v3
        from dwr.tasks.dataset_v2 import _split_and_ids
        source_candidate = next(replace(generate_candidate_v3(20261007, "source", index), candidate_index=0)
            for index in range(100) if (lambda c: c.oracle_unique and c.eligible and
            _split_and_ids(c)[0] != "source_train")(generate_candidate_v3(20261007, "source", index)))
        fixture_candidate = lambda seed, partition, index, *, full_profiles=frozenset(): source_candidate
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        limits = DatasetLimits({"eq": 0, "neq": 1, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 4, 30, 1024)
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "run"
            with _source_closure_fixture(limits), patch.dict(_STREAMS, {"dwr-generator-v3": replace(
                    _STREAMS["dwr-generator-v3"], candidate_fn=fixture_candidate)}), \
                 patch("dwr.tasks.generator_v3.generate_candidate_v3", side_effect=fixture_candidate):
                run_dataset_v3(self.config, data, "feasibility", ["fixture"], limits=limits, source=source)
            def fail_prefix(*args, **kwargs):
                initial = json.loads((data / "audit_report.json").read_text(encoding="utf-8"))
                self.assertEqual(initial["status"], "incomplete")
                self.assertEqual(initial["error"]["type"], "AuditNotStarted")
                raise ValueError("prefix replay fixture fault")
            with patch.dict(_STREAMS, {"dwr-generator-v3": replace(
                    _STREAMS["dwr-generator-v3"], candidate_fn=fixture_candidate)}), \
                 patch("dwr.tasks.dataset_v2.registered_limits", return_value=limits), \
                 patch("dwr.tasks.audit_cli_v2.registered_limits", return_value=limits), \
                 patch("dwr.tasks.audit_cli_v2.source_identity", return_value=source), \
                 patch("dwr.tasks.replay_v2.replay_source_prefix_v3", side_effect=fail_prefix), \
                 redirect_stderr(StringIO()):
                code = audit_main(["--config", str(ROOT / "configs/tasks/constraint_v3.json"),
                                   "--mode", "feasibility", "--data", str(data)])
            report = json.loads((data / "audit_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((data / "audit_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["g2_status"], "not_assessed")
            self.assertEqual(report["proposal_status"], "NOT_ASSESSED")
            self.assertEqual(report["error"]["type"], "ValueError")
            self.assertEqual(report["error"]["message"], "prefix replay fixture fault")
            self.assertEqual(manifest["error"], report["error"])
            self.assertFalse((data / "preflight_report.json").exists())

    def test_rejected_targets_cannot_supply_feasibility_support_cells(self):
        from dwr.tasks import audit_v2
        from dwr.tasks.audit_v2 import _feasibility_support_proxy, _counts
        from dwr.tasks.audit import make_record
        from tests.test_task_audit_v2 import mixed_task
        record = make_record(mixed_task(), "final", 0)
        record["model_input_bytes"] = 400
        record.update({"interaction": {"sensitive": True}, "primary_eligible": True,
                       "signature": "fixture"})
        rows = []
        source = {**record, "partition": "source", "split": "source_train", "profile": ["eq"]}
        rows.extend([{**source, "candidate_index": i} for i in range(100)])
        for profile in (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod")):
            for index in range(50):
                rows.append({**record, "partition": "development", "split": "development_validation",
                             "profile": list(profile), "candidate_index": index,
                             "model_input_bytes": 400})
            for index in range(50, 100):
                rows.append({**record, "partition": "development", "split": "development_validation",
                             "profile": list(profile), "candidate_index": index,
                             "model_input_bytes": 2000})
            for index in range(100, 400):
                rows.append({**record, "partition": "development", "split": "development_validation",
                             "profile": list(profile), "candidate_index": index,
                             "model_input_bytes": 400, "primary_eligible": False})
        names = ("source_train", "development_validation_eq_neq",
                 "development_validation_eq_add_mod", "development_validation_neq_add_mod")
        real_common_support = audit_v2.v1.common_support
        observed = []
        def capture(cohorts, *args, **kwargs):
            observed.append({name: len(items) for name, items in cohorts.items()})
            return real_common_support(cohorts, *args, **kwargs)
        with patch.object(audit_v2.v1, "common_support", side_effect=capture):
            unfiltered = _feasibility_support_proxy(rows, eligible_targets=False)
            filtered = _feasibility_support_proxy(rows, eligible_targets=True)
        self.assertEqual([observed[0][name] for name in names[1:]], [400, 400, 400])
        self.assertEqual([observed[1][name] for name in names[1:]], [100, 100, 100])
        self.assertFalse(unfiltered["pass"], "shared support currently also filters rejected targets")
        self.assertFalse(filtered["pass"])
        self.assertEqual(tuple(filtered["cohorts"]), names)
        counts = _counts(rows)
        for profile in ("eq/neq", "eq/add_mod", "neq/add_mod"):
            cell = counts[f"development|development_validation|{profile}"]
            self.assertEqual(cell["unique_pre_eligibility"], 400)
            self.assertEqual(cell["eligible"], 100)
        for name in names[1:]:
            self.assertEqual(filtered["cohorts"][name]["count"], 100)
        source_bound = audit_v2._audit_source_capacity(rows, "a" * 64)
        self.assertEqual(source_bound["status"], "PASS_CAPACITY_BOUND")
        self.assertFalse(filtered["pass"], "a capacity proof does not upgrade measured four-cohort support")

    def test_source_capacity_bound_uses_strict_less_than_point_eight(self):
        from dwr.tasks.audit_v2 import _audit_source_capacity
        from dwr.tasks.audit import make_record
        from tests.test_task_audit_v2 import mixed_task
        base = make_record(mixed_task(), "final", 0)
        # Ten dominant cells contribute 80/100 exactly; ten smaller cells fill the tail.
        records = []
        sizes = (400, 600, 700, 800, 1100)
        node_values = (1, 2, 4, 8, 16, 32, 64)
        combinations = [(size, node) for size in sizes for node in node_values]
        for index, (size, nodes) in enumerate(combinations[:20]):
            count = 8 if index < 10 else 2
            for _ in range(count):
                records.append({**base, "partition": "source", "split": "source_train",
                    "model_input": "x" * size,
                    "solver_stats": {**base["solver_stats"], "first_solution_nodes": nodes}})
        at_boundary = _audit_source_capacity(records, "b" * 64)
        self.assertEqual(at_boundary["top_k_mass"], 0.8)
        self.assertEqual(at_boundary["status"], "PASS_CAPACITY_BOUND")
        below = [dict(row) for row in records]
        # Move one dominant count to a fresh cell, increasing the uncaptured tail.
        below.append({**base, "partition": "source", "split": "source_train",
            "model_input": "z" * 900, "solver_stats": {**base["solver_stats"], "first_solution_nodes": 64}})
        below[0] = {**below[0], "model_input": "z" * 900,
                    "solver_stats": {**below[0]["solver_stats"], "first_solution_nodes": 64}}
        no_go = _audit_source_capacity(below, "c" * 64)
        self.assertLess(no_go["top_k_mass"], 0.8)
        self.assertEqual(no_go["status"], "NO_GO")
        above = [dict(row) for row in records]
        above.extend([{**base, "partition": "source", "split": "source_train",
            "model_input": "x" * 400, "solver_stats": {**base["solver_stats"], "first_solution_nodes": 1}}
            for _ in range(10)])
        enough = _audit_source_capacity(above, "d" * 64)
        self.assertGreater(enough["top_k_mass"], 0.8)
        self.assertEqual(enough["status"], "PASS_CAPACITY_BOUND")

    def test_full_five_cohort_support_pass_and_measured_no_go_keep_source_test(self):
        from dwr.tasks import audit_v2
        from dwr.tasks.audit import audit_interactions, make_record
        from dwr.tasks.generator_v3 import generate_candidate_v3
        from tests.test_task_audit_v2 import mixed_task
        config = json.loads(json.dumps(self.config.resolved))
        gate = config["audit_gate"]
        for key in ("min_accepted_groups_per_partition", "min_structure_signatures_per_partition",
                    "min_distinct_given_counts_per_partition", "source_train_min_structure_signatures",
                    "min_source_sensitive_examples_per_signature", "min_primary_eligible_groups_per_mixed_profile",
                    "min_primary_eligible_structure_signatures_per_mixed_profile",
                    "min_primary_eligible_examples_per_heldout_signature",
                    "min_primary_eligible_fraction_per_mixed_partition", "min_source_constraint_sensitive_fraction",
                    "min_solution_value_fraction_each_partition"):
            gate[key] = 0
        gate["source_id_groups_min"] = {"train": 0, "validation": 0, "test": 0}
        gate["development_target_min_groups"] = {"development_validation": 0,
                                                   "development_diagnostic_test": 0}
        config["composition_partition"]["development_target_split"]["min_validation_groups_per_profile"] = 0
        config["composition_partition"]["development_target_split"]["min_diagnostic_test_groups_per_profile"] = 0
        source_candidates = {}
        for index in range(1000):
            candidate = generate_candidate_v3(20261007, "source", index)
            if not (candidate.oracle_unique and candidate.eligible):
                continue
            profile = tuple(candidate.profile)
            if profile in (("eq",), ("neq",), ("add_mod",)) and profile not in source_candidates:
                if audit_interactions(candidate.task, "source")["sensitive"]:
                    source_candidates[profile] = candidate.task
            if len(source_candidates) == 3:
                break
        self.assertEqual(set(source_candidates), {("eq",), ("neq",), ("add_mod",)})
        rows = []
        source_index = 0
        for profile, task in source_candidates.items():
            base = make_record(task, "source", source_index)
            info = audit_v2.v1.validate_record(base, config)
            interaction = audit_interactions(task, "source")
            rows.append({**base, **info, "split": "source_train", "profile": list(profile),
                         "interaction": interaction, "primary_eligible": False})
            source_index += 1
        target_task = mixed_task()
        base = make_record(target_task, "final", 100)
        info = audit_v2.v1.validate_record(base, config)
        for index in range(10):
            interaction = audit_interactions(target_task, "source")
            rows.append({**base, **info, "partition": "source", "candidate_index": source_index + index,
                         "split": "source_test", "profile": ["neq"], "interaction": interaction,
                         "primary_eligible": False})
        for profile in audit_v2.v1.PROFILES["development"]:
            for index in range(10):
                interaction = audit_interactions(target_task, "final")
                rows.append({**base, **info, "partition": "development", "candidate_index": index,
                    "split": "development_validation", "profile": list(profile), "interaction": interaction,
                    "primary_eligible": True})
        for index in range(10):
            interaction = audit_interactions(target_task, "final")
            rows.append({**base, **info, "candidate_index": index, "split": "final_test",
                         "interaction": interaction, "primary_eligible": True})
        passed = audit_v2._full_metrics(rows, config)
        self.assertTrue(passed["five_cohort_common_support"]["pass"], passed)
        self.assertEqual(tuple(passed["five_cohort_common_support"]["cohorts"]), audit_v2.v1.COHORTS)
        self.assertEqual(passed["five_cohort_common_support"]["cohorts"][audit_v2.v1.COHORTS[0]]["count"], 10)
        shifted = [dict(row, model_input_bytes=2048) if row["partition"] == "final" else row for row in rows]
        failed = audit_v2._full_metrics(shifted, config)
        self.assertFalse(failed["five_cohort_common_support"]["pass"])
        self.assertIn("common_support", failed["floor_failures"])

    def test_feasibility_projection_hides_diagnostic_accepted_totals_and_rows(self):
        from dwr.tasks.audit_v2 import _feasibility_visible_projection
        visible_rows = [
            {"partition": "source", "split": "source_train", "candidate_index": 0},
            {"partition": "development", "split": "development_validation", "candidate_index": 1},
        ]
        visible_records = [("source", 0), ("development", 1)]
        replay_a = {"status": "complete", "outcomes_sha256": "a" * 64,
                    "records_sha256": "b" * 64, "accepted_records": 100,
                    "diagnostic_task_histogram": {"private": 100}}
        replay_b = {**replay_a, "accepted_records": 900,
                    "diagnostic_task_histogram": {"private": 900}}
        records_a = [{"partition": part, "candidate_index": index} for part, index in visible_records] + [
            {"partition": "development", "candidate_index": 2, "solver_stats": {"secret": 100}}]
        records_b = [{"partition": part, "candidate_index": index} for part, index in visible_records] + [
            {"partition": "development", "candidate_index": 2, "solver_stats": {"secret": 900}}]
        rows_a = visible_rows + [{"partition": "development", "split": "development_diagnostic_test",
                                 "candidate_index": 2, "primary_eligible": True}]
        rows_b = visible_rows + [{"partition": "development", "split": "development_diagnostic_test",
                                 "candidate_index": 2, "primary_eligible": True}]
        projected_a = _feasibility_visible_projection(replay_a, records_a, rows_a, "feasibility")
        projected_b = _feasibility_visible_projection(replay_b, records_b, rows_b, "feasibility")
        self.assertEqual(projected_a, projected_b)
        replay_projection, visible, accepted = projected_a
        self.assertEqual(set(replay_projection), {"status", "outcomes_sha256", "records_sha256"})
        self.assertEqual([(r["partition"], r["split"]) for r in visible],
                         [("source", "source_train"), ("development", "development_validation")])
        self.assertEqual(len(accepted), 2)
        self.assertNotIn("accepted_records", replay_projection)

    def test_independent_audit_rejects_v3_schema_and_private_evidence_substitutions(self):
        from dwr.tasks.audit_v2 import _validate_v3_artifacts
        from dwr.tasks.dataset_v2 import outcome_row, accepted_record, _split_and_ids
        from dwr.tasks.generator_v3 import generate_candidate_v3
        candidate = generate_candidate_v3(20261007, "source", 1)
        self.assertTrue(candidate.oracle_unique and candidate.eligible)
        split, identities = _split_and_ids(candidate)
        outcome = outcome_row(candidate, "accepted", split=split, identities=identities,
            denominator_counted=True, expected_generator_id="dwr-generator-v3")
        record = accepted_record(candidate, split, identities, expected_generator_id="dwr-generator-v3")
        rows = _validate_v3_artifacts([outcome], [record], self.config.resolved)
        self.assertEqual(rows[0]["split"], "source_train")
        bad_schema = {**outcome, "unregistered": True}
        bad_private = {**outcome, "private_evidence": {**outcome["private_evidence"],
            "source": {"eligible": True}}}
        bad_record_version = {**record, "record_version": "dwr-task-accepted-record-v2"}
        for bad_outcomes, bad_records in (([bad_schema], [record]),
                                          ([bad_private], [record]),
                                          ([outcome], [bad_record_version])):
            with self.subTest(outcome=bad_outcomes[0].get("unregistered"),
                              record_version=bad_records[0].get("record_version")), self.assertRaises(ValueError):
                _validate_v3_artifacts(bad_outcomes, bad_records, self.config.resolved)
