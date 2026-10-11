from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from contextlib import redirect_stderr
from unittest.mock import patch

from dwr.tasks.certificate_study import (
    CandidateFailure,
    ConfigError,
    IntegrityFailure,
    ResourceFailure,
    aggregate_permitted_rows,
    common_support_mass,
    difficulty_counts,
    load_study_config,
    public_row,
    finalize_artifacts,
    run_study,
    _audit_task,
    _scalar_solution_count,
    _outcome_private,
    _run_prefix,
    _gac_profile_summary,
    _development_gate,
    _save_fatal_attempt,
    _require_supported_runtime,
    _audit_local_gac,
    _hash,
    _independent_interaction_projection,
)
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, Solution
from dwr.tasks.schema import OracleResult
from dwr.tasks.solver import SolveResult, SolverStats


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tasks" / "certificate_sixvar_v1.json"


def row(content: str, split: str, *, eligible: bool = True, envelope: bool = True,
        profile: str = "eq", cell: str = "A", cohort: str = "source_train",
        signature: str | None = None) -> dict:
    return {
        "content_id": content,
        "lineage_id": "lineage-" + content,
        "structure_signature": signature or "sig-" + content,
        "split": split,
        "profile": profile,
        "cell": cell,
        "oracle_unique": True,
        "eligible": eligible,
        "envelope": envelope,
        "cohort": cohort,
        "sensitive": True,
        "values": [0, 1, 2, 0, 1, 2],
        "given_count": 2,
        "task": {"private": "must not publish"},
        "witness": {"poison": True},
        "alternatives": [{"poison": True}],
        "source_validation_metrics": {"poison": 10**9},
    }


class CertificateStudyTests(unittest.TestCase):
    def test_config_is_closed_and_rejects_unknown_or_wrong_typed_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            original = json.loads(CONFIG.read_text(encoding="utf-8"))
            path.write_bytes(CONFIG.read_bytes())
            self.assertEqual(load_study_config(path).resolved, original)
            for mutate in (
                lambda value: value.update(extra=True),
                lambda value: value.update(root_seed=20261011.0),
            ):
                invalid = copy.deepcopy(original)
                mutate(invalid)
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_study_config(path)
            invalid = copy.deepcopy(original)
            invalid["root_seed"] = True
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_study_config(path)
            path.write_text('{"root_seed":1,"root_seed":2}', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_study_config(path)
            path.write_text('{"root_seed":NaN}', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_study_config(path)

    def test_opaque_rows_never_enter_public_projection(self):
        opaque = row("opaque", "source_validation", cohort="source_validation")
        projected = public_row(opaque, permitted_partition="source")
        self.assertIsNone(projected)
        allowed = public_row(row("train", "source_train"), permitted_partition="source")
        self.assertNotIn("task", allowed)
        self.assertNotIn("witness", allowed)
        self.assertNotIn("alternatives", allowed)
        self.assertNotIn("source_validation_metrics", allowed)

    def test_source_aggregation_deduplicates_before_eligibility_and_envelope(self):
        rows = [
            row("duplicate-first-ineligible", "source_train", eligible=False, envelope=False),
            row("duplicate-first-ineligible", "source_train", eligible=True, envelope=True),
            row("opaque", "source_validation", eligible=True, envelope=True,
                cohort="source_validation"),
        ]
        aggregate = aggregate_permitted_rows(rows, partition="source", fixture_only=True)
        self.assertEqual(aggregate["unique_oracle_content_groups"], 1)
        self.assertEqual(aggregate["eligible_groups"], 0)
        self.assertEqual(aggregate["eligible_envelope_groups"], 0)
        encoded = json.dumps(aggregate, sort_keys=True)
        self.assertNotIn("opaque", encoded)

    def test_content_and_lineage_deduplication_maps_span_both_prefixes(self):
        identity = row("same-content", "source_train")
        seen: set[str] = set()
        lineage: dict[str, tuple[str, str]] = {}
        source_stream = io.BytesIO()
        replay_seen: set[str] = set()
        replay_lineage: dict[str, tuple[str, str]] = {}
        _run_prefix("source", 1, Path("."), source_stream, None, fixture_rows=[identity],
                    seen_content=seen, lineage_map=lineage,
                    replay_seen_content=replay_seen, replay_lineage_map=replay_lineage)
        development_row = row("same-content", "development_validation", cohort="development_validation")
        development_stream = io.BytesIO()
        _run_prefix("development", 1, Path("."), development_stream, None,
                    fixture_rows=[development_row], seen_content=seen, lineage_map=lineage,
                    replay_seen_content=replay_seen, replay_lineage_map=replay_lineage)
        saved = json.loads(development_stream.getvalue().decode("utf-8"))
        self.assertTrue(saved["duplicate_content"])
        self.assertEqual(saved["state"], "duplicate_content")

    def test_common_mass_denominator_is_all_eligible_groups_before_envelope(self):
        cohorts = {}
        for name in ("source_train", "development_validation", "source_validation", "source_test"):
            cohorts[name] = [
                row(f"{name}-in", "source_train", cohort=name, envelope=True, cell="A"),
                row(f"{name}-out", "source_train", cohort=name, envelope=False, cell="Z"),
            ]
        result = common_support_mass(cohorts, cells=("A",), minimum_per_cell=1)
        self.assertEqual(result["source_train"]["eligible_denominator"], 2)
        self.assertEqual(result["source_train"]["common_cell_mass"], 0.5)

    def test_resource_failure_revokes_previously_written_pass_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name, content in (("manifest.initial.json", "{}\n"),
                                  ("outcomes.jsonl", "{}\n"),
                                  ("replay.json", "{}\n"),
                                  ("report.json", '{"status":"PASS_CAPACITY_DIFFICULTY"}\n')):
                (output / name).write_text(content, encoding="utf-8")
            guard = type("Guard", (), {"check": lambda self: (_ for _ in ()).throw(
                TimeoutError("cap"))})()
            with self.assertRaises(ResourceFailure):
                finalize_artifacts(output, guard=guard)
            self.assertTrue((output / "outcomes.jsonl").exists())
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertNotEqual(report["status"], "PASS_CAPACITY_DIFFICULTY")

    def test_terminal_manifest_binds_artifacts_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("manifest.initial.json", "outcomes.jsonl", "replay.json", "report.json"):
                (output / name).write_text("{}\n", encoding="utf-8")
            manifest = finalize_artifacts(output, guard=None)
            self.assertEqual(set(manifest["artifacts"]), {
                "manifest.initial.json", "outcomes.jsonl", "replay.json", "report.json"})
            (output / "outcomes.jsonl").write_text('{"mutated":true}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                finalize_artifacts(output, guard=None, verify_existing=True)

    def test_output_collision_keeps_existing_directory_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            marker = output / "user-data.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_study(CONFIG, output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertEqual(list(output.iterdir()), [marker])

    def test_failed_source_gate_does_not_open_development(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            fake_identity = {"git_commit": "fixture", "git_clean": True, "producer_sha256": {}}
            empty_prefix = ([], {"construction_failure": 3600},
                            {"attempts_rederived": 0, "attempts_processed": 3600,
                             "rederived_metadata_sha256": "fixture"})
            with patch("dwr.tasks.certificate_study._source_identity", return_value=fake_identity), \
                    patch("dwr.tasks.certificate_study._run_prefix", return_value=empty_prefix) as prefix:
                report = run_study(CONFIG, output, fixture_rows={})
            self.assertEqual(report["status"], "NOT_ASSESSED_FIXTURE_ONLY")
            self.assertEqual(report["development"], {"status": "not_attempted", "attempts_processed": 0,
                                                       "reason": "source capacity gate failed"})
            self.assertEqual(prefix.call_count, 1)
            self.assertEqual(prefix.call_args.args[0], "source")

    def test_short_fixture_prefix_is_not_a_scientific_no_go_and_skips_development(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            fake_identity = {"git_commit": "fixture", "git_clean": True, "producer_sha256": {}}
            short_prefix = ([], {}, {"attempts_rederived": 0, "attempts_processed": 1,
                                     "rederived_metadata_sha256": "fixture"})
            with patch("dwr.tasks.certificate_study._source_identity", return_value=fake_identity), \
                    patch("dwr.tasks.certificate_study._run_prefix", return_value=short_prefix) as prefix:
                report = run_study(CONFIG, output, fixture_rows={})
            self.assertEqual(report["status"], "NOT_ASSESSED_FIXTURE_ONLY")
            self.assertEqual(report["source"]["status"], "not_assessed_fixture_prefix_incomplete")
            self.assertEqual(report["development"]["attempts_processed"], 0)
            self.assertEqual(prefix.call_count, 1)
            self.assertEqual(prefix.call_args.args[0], "source")

    def test_gac_stall_fraction_uses_all_eligible_unique_validation_groups(self):
        rows = [row("eligible-stalled", "development_validation", cohort="development_validation")]
        rows[0]["gac"] = {"stalled": True}
        rows.append(row("eligible-solved", "development_validation", cohort="development_validation"))
        rows[1]["gac"] = {"stalled": False}
        rows.extend(row(f"ineligible-{index}", "development_validation", eligible=False,
                        cohort="development_validation") for index in range(20))
        counts = difficulty_counts(rows)
        self.assertEqual(counts["eligible_denominator"], 2)
        self.assertEqual(counts["stalled_groups"], 1)
        self.assertEqual(counts["stalled_fraction"], 0.5)

    def test_development_profile_floors_do_not_pool_across_profiles(self):
        cfg = load_study_config(CONFIG).resolved
        cfg["development_floors"] = dict(cfg["development_floors"])
        cfg["development_floors"].update({"eligible_envelope_groups_per_profile": 0,
            "eligible_signatures_per_profile": 0, "eligible_per_represented_eligible_signature": 0,
            "scientific_eligibility_fraction": 0.25, "eligible_envelope_fraction": 0.25})
        cfg["support"] = dict(cfg["support"])
        cfg["support"].update({"cell_min": 0, "cohort_mass_min": 0})
        rows = []
        profiles = cfg["development_profiles"]
        for profile_index, profile in enumerate(profiles):
            for i in range(8):
                r = row(f"{profile}-{i}", "development_validation", cohort="development_validation",
                        profile=profile,
                        eligible=(profile_index != 1 or i < 1),
                        envelope=(profile_index != 1), signature=f"sig-{profile}-{i}")
                r["gac"] = {"stalled": False}
                rows.append(r)
        _, _, result = _development_gate(rows, cfg)
        # The pooled fractions pass, while the thin middle profile violates
        # both registered per-profile floors.
        self.assertTrue(result["floors"]["scientific_eligibility"])
        self.assertTrue(result["floors"]["envelope_fraction"])
        self.assertFalse(result["floors"]["profile_scientific_eligibility"])
        self.assertFalse(result["floors"]["profile_envelope_fraction"])
        self.assertEqual(result["profile_results"][profiles[1]]["scientific_eligibility_fraction"], 0.125)

    def test_development_profile_floors_accept_exact_quarter_and_reject_empty_profile(self):
        cfg = load_study_config(CONFIG).resolved
        cfg["development_floors"] = dict(cfg["development_floors"])
        cfg["development_floors"].update({"eligible_envelope_groups_per_profile": 0,
            "eligible_signatures_per_profile": 0, "eligible_per_represented_eligible_signature": 0,
            "scientific_eligibility_fraction": 0.25, "eligible_envelope_fraction": 0.25})
        cfg["support"] = dict(cfg["support"])
        cfg["support"].update({"cell_min": 0, "cohort_mass_min": 0})
        profiles = cfg["development_profiles"]

        def make_rows(empty_profile=None, envelope_thin=False):
            result = []
            for profile in profiles:
                if profile == empty_profile:
                    continue
                for i in range(8):
                    eligible = True if envelope_thin else (profile != profiles[0] or i < 2)
                    envelope = (profile != profiles[0] or i < 2) if envelope_thin else eligible
                    item = row(f"{profile}-{empty_profile}-{envelope_thin}-{i}",
                               "development_validation", cohort="development_validation",
                               profile=profile, eligible=eligible, envelope=envelope,
                               signature=f"{profile}-{i}")
                    item["gac"] = {"stalled": False}
                    result.append(item)
            return result

        _, _, exact_scientific = _development_gate(make_rows(), cfg)
        self.assertTrue(exact_scientific["floors"]["profile_scientific_eligibility"])
        self.assertTrue(exact_scientific["floors"]["profile_envelope_fraction"])
        _, _, exact_envelope = _development_gate(make_rows(envelope_thin=True), cfg)
        self.assertTrue(exact_envelope["floors"]["profile_scientific_eligibility"])
        self.assertTrue(exact_envelope["floors"]["profile_envelope_fraction"])
        _, _, empty = _development_gate(make_rows(empty_profile=profiles[0]), cfg)
        self.assertFalse(empty["floors"]["profile_scientific_eligibility"])
        self.assertFalse(empty["floors"]["profile_envelope_fraction"])

    def test_public_candidate_failure_does_not_reveal_cause_text(self):
        failure = _save_fatal_attempt(io.BytesIO(), "source", 7, "opaque_audit",
                                      RuntimeError("PROTECTED_METRIC=999"))
        self.assertNotIn("PROTECTED_METRIC", str(failure))
        self.assertEqual(failure.phase, "opaque_audit")

    def test_public_report_and_cli_errors_do_not_reveal_exception_text(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            fake_identity = {"git_commit": "fixture", "git_clean": True, "producer_sha256": {}}
            with patch("dwr.tasks.certificate_study._source_identity", return_value=fake_identity), \
                 patch("dwr.tasks.certificate_study._run_prefix",
                       side_effect=RuntimeError("OPAQUE_METRIC=123")):
                with self.assertRaisesRegex(RuntimeError, "OPAQUE_METRIC"):
                    run_study(CONFIG, output, fixture_rows={})
            report = (output / "report.json").read_text(encoding="utf-8")
            manifest = (output / "manifest.json").read_text(encoding="utf-8")
            self.assertNotIn("OPAQUE_METRIC", report)
            self.assertNotIn("OPAQUE_METRIC", manifest)
        stderr = io.StringIO()
        with patch("dwr.tasks.certificate_study.run_study",
                   side_effect=RuntimeError("OPAQUE_METRIC=123")), redirect_stderr(stderr):
            from dwr.tasks.certificate_study import main
            result = main(["--config", str(CONFIG), "--output", "unused"])
        self.assertEqual(result, 2)
        self.assertNotIn("OPAQUE_METRIC", stderr.getvalue())

    def test_initial_manifest_pins_executable_runtime_and_actual_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            fake_identity = {"git_commit": "fixture", "git_clean": True, "producer_sha256": {}}
            empty_prefix = ([], {"construction_failure": 3600},
                            {"attempts_rederived": 0, "attempts_processed": 0,
                             "rederived_metadata_sha256": "fixture"})
            with patch("dwr.tasks.certificate_study._source_identity", return_value=fake_identity), \
                    patch("dwr.tasks.certificate_study._run_prefix", return_value=empty_prefix):
                run_study(CONFIG, output, argv=["--config", str(CONFIG), "--output", str(output)],
                          fixture_rows={})
            manifest = json.loads((output / "manifest.initial.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["executable"], sys.executable)
            self.assertEqual(manifest["implementation"], "CPython")
            self.assertEqual(manifest["python_version"], list(sys.version_info[:3]))
            self.assertEqual(manifest["command"], list(sys.orig_argv))

    def test_candidate_guard_failure_after_flush_does_not_duplicate_outcome(self):
        class FlushStream(io.BytesIO):
            flushed = False
            def flush(self):
                self.flushed = True
                return super().flush()
        class Guard:
            def __init__(self, stream): self.stream = stream
            def check(self):
                if self.stream.flushed:
                    raise TimeoutError("cap")
        stream = FlushStream()
        with self.assertRaises(IntegrityFailure):
            _run_prefix("source", 1, Path("."), stream, Guard(stream))
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)

    def test_candidate_audit_failure_retains_private_construction_evidence_once(self):
        stream = io.BytesIO()
        with patch("dwr.tasks.certificate_study._build_registered_row",
                   side_effect=RuntimeError("OPAQUE_PRIVATE_CAUSE")):
            with self.assertRaises(CandidateFailure) as raised:
                _run_prefix("source", 1, Path("."), stream, None)
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        fatal = json.loads(lines[0])
        self.assertEqual(fatal["state"], "fatal_error")
        self.assertIsInstance(fatal["task"], str)
        self.assertIn("givens", fatal)
        self.assertIn("witness", fatal)
        self.assertIn("alternatives", fatal)
        self.assertIn("reserved_constraints", fatal)
        self.assertIn("work", fatal)
        self.assertIn("profile", fatal)
        self.assertIn("cell", fatal)
        self.assertEqual(fatal["seed"], 20261011)
        self.assertEqual(fatal["candidate_index"], 0)
        self.assertEqual(fatal["failure"]["error_type"], "RuntimeError")
        self.assertNotIn("OPAQUE_PRIVATE_CAUSE", str(raised.exception))
        self.assertIn("OPAQUE_PRIVATE_CAUSE", fatal["failure"]["message"])

    def test_runtime_admission_is_closed_to_registered_cpython_minors(self):
        with patch("dwr.tasks.certificate_study.platform.python_implementation", return_value="PyPy"), \
             patch("dwr.tasks.certificate_study.sys.version_info", (3, 11, 0, "final", 0)):
            with self.assertRaises(IntegrityFailure):
                _require_supported_runtime()

    def test_gac_profile_report_keeps_zero_stalls_and_work_denominator_explicit(self):
        rows = [row("solved", "development_validation", cohort="development_validation", profile="eq/neq"),
                row("stalled", "development_validation", cohort="development_validation", profile="eq/neq"),
                row("opaque", "development_diagnostic_test", cohort="development_diagnostic_test",
                    profile="eq/neq")]
        rows[0]["gac"] = {"solved": True, "stalled": False, "unresolved_unknowns": 0,
                           "rounds": 1, "support_checks": 9}
        rows[1]["gac"] = {"solved": False, "stalled": True, "unresolved_unknowns": 2,
                           "rounds": 3, "support_checks": 27}
        rows[2]["gac"] = {"solved": True, "stalled": True, "unresolved_unknowns": 99,
                           "rounds": 99, "support_checks": 999}
        summary = _gac_profile_summary(rows[:2], ("eq/neq", "eq/add_mod"))
        self.assertEqual(summary["eq/neq"]["oracle_unique_group_denominator"], 2)
        self.assertEqual(summary["eq/neq"]["stalled_groups"], 1)
        self.assertFalse(summary["eq/neq"]["zero_stalls"])
        self.assertEqual(summary["eq/neq"]["local_support_work"]["support_checks_sum"], 36)
        self.assertEqual(summary["eq/add_mod"]["stalled_groups"], 0)
        self.assertTrue(summary["eq/add_mod"]["zero_stalls"])

    def test_scalar_oracle_disagreement_is_fatal(self):
        from dwr.tasks.certificate_generator import construct
        from dwr.tasks.schema import parse_task_json
        outcome = construct(20261011, "source", 0)
        task = outcome.task
        private = _outcome_private(outcome)
        private["task_object"] = task
        self.assertEqual(_scalar_solution_count(task), 1)
        with patch("dwr.tasks.certificate_study._scalar_solution_count", return_value=2):
            with self.assertRaisesRegex(IntegrityFailure, "scalar/oracle"):
                _audit_task(private, partition="source")

    def test_jointly_wrong_oracle_and_solver_identity_is_fatal(self):
        from dwr.tasks.certificate_generator import construct
        outcome = construct(20261011, "source", 0)
        task = outcome.task
        record = _outcome_private(outcome)
        record["task_object"] = task
        jointly_wrong = Solution(record["witness"].assignments[:-1] +
                                 (("v5", (record["witness"].assignments[-1][1] + 1) % 3),))
        with patch("dwr.tasks.certificate_study.enumerate_solutions",
                   return_value=OracleResult((jointly_wrong,), assignments_examined=1)) as oracle, \
             patch("dwr.tasks.certificate_study.solve",
                   return_value=SolveResult(jointly_wrong, SolverStats(1, 1, 1))) as solver:
            with self.assertRaises(IntegrityFailure):
                _audit_task(record, partition="source")
        oracle.assert_called_once_with(task)
        solver.assert_called_once_with(task)

    def test_registered_constructor_outcome_integrates_with_independent_audit(self):
        from dwr.tasks.certificate_generator import construct
        from dwr.tasks.canonical import development_split, structure_signature
        from dwr.tasks.schema import parse_task_json

        outcome = construct(20261011, "source", 0)
        private = _outcome_private(outcome)
        task = parse_task_json(private["task"])
        private["task_object"] = task
        audit = _audit_task(private, partition="source")
        self.assertTrue(audit["oracle_unique"])
        self.assertTrue(audit["eligible"])
        development = _outcome_private(construct(20261011, "development", 0))
        dev_task = parse_task_json(development["task"])
        self.assertEqual(development["split"], development_split(structure_signature(dev_task)))

    def test_single_candidate_prefix_streams_private_row_and_replays_exactly(self):
        stream = io.BytesIO()
        permitted_rows, reasons, replay = _run_prefix("source", 1, Path("."), stream, None)
        saved = json.loads(stream.getvalue().decode("utf-8"))
        self.assertEqual(saved["candidate_index"], 0)
        self.assertIn("witness", saved)
        self.assertIn("private_interaction", saved)
        self.assertEqual(replay["attempts_rederived"], 1)
        self.assertTrue(replay["rederived_metadata_sha256"])
        self.assertIsInstance(permitted_rows, list)
        self.assertIsInstance(reasons, dict)

    def test_persisted_replay_metadata_mutation_is_fatal_without_duplicate_row(self):
        from dwr.tasks import certificate_generator

        class MutatingStream(io.BytesIO):
            def write(self, data):
                if b'"candidate_index":0' in data:
                    row = json.loads(data.decode("utf-8"))
                    row["work"]["rng_calls"] += 1
                    data = (json.dumps(row, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                return super().write(data)

        with tempfile.TemporaryDirectory() as directory:
            stream = MutatingStream()
            with self.assertRaises(CandidateFailure) as raised:
                _run_prefix("source", 1, Path(directory), stream, None)
            saved = stream.getvalue().splitlines()
            self.assertEqual(len(saved), 1)
            self.assertEqual(raised.exception.phase, "persisted_prefix_replay")
            self.assertEqual(json.loads(saved[0])["candidate_index"], 0)

    def test_independent_certificate_eligibility_flag_is_checked(self):
        from dwr.tasks.certificate_generator import construct
        outcome = construct(20261011, "source", 0)
        private = _outcome_private(outcome)
        private["task_object"] = outcome.task
        projection = _independent_interaction_projection(outcome.task, "source")
        projection["eligible"] = not projection["eligible"]
        with patch("dwr.tasks.certificate_study._independent_interaction_projection",
                   return_value=projection):
            with self.assertRaisesRegex(IntegrityFailure, "eligibility"):
                _audit_task(private, partition="source")

    def test_independent_deletion_counts_and_mixed_graph_facts_are_checked(self):
        from dwr.tasks.certificate_generator import construct
        source = construct(20261011, "source", 0)
        source_row = _outcome_private(source)
        source_row["task_object"] = source.task
        original_projection = _independent_interaction_projection
        bad_source = original_projection(source.task, "source")
        bad_source["removal_solution_counts"] = list(bad_source["removal_solution_counts"])
        bad_source["removal_solution_counts"][0] += 1
        with patch("dwr.tasks.certificate_study._independent_interaction_projection",
                   return_value=bad_source):
            with self.assertRaisesRegex(IntegrityFailure, "deletion counts"):
                _audit_task(source_row, partition="source")

        mixed = construct(20261011, "development", 0)
        mixed_row = _outcome_private(mixed)
        mixed_row["task_object"] = mixed.task
        bad_mixed = original_projection(mixed.task, "development")
        bad_mixed["factor_graph_kinds"] = []
        with patch("dwr.tasks.certificate_study._independent_interaction_projection",
                   return_value=bad_mixed):
            with self.assertRaisesRegex(IntegrityFailure, "factor graph"):
                _audit_task(mixed_row, partition="development")

    def test_wrong_but_well_formed_gac_metadata_is_fatal(self):
        from dwr.tasks.certificate_generator import construct
        from dwr.tasks.local_propagation import propagate
        outcome = construct(20261011, "source", 0)
        actual = propagate(outcome.task)
        wrong = replace(actual, support_checks=actual.support_checks + 1)
        counters = {}
        with self.assertRaisesRegex(IntegrityFailure, "GAC support audit"):
            _audit_local_gac(outcome.task, wrong, work_counters=counters)
        self.assertGreater(counters.get("independent_gac_support_checks", 0), 0)

    def test_actual_permitted_work_counts_include_both_construction_and_replay_audits(self):
        stream = io.BytesIO()
        with patch("dwr.tasks.certificate_study.source_split", return_value="source_train"):
            _, _, replay = _run_prefix("source", 1, Path("."), stream, None)
        persisted = json.loads(stream.getvalue().decode("utf-8"))
        work = replay["actual_permitted_work"]
        self.assertGreater(work["oracle_assignments_examined"], 0)
        self.assertGreater(work["solver_nodes"], 0)
        self.assertGreater(work["independent_scalar_assignments"], 0)
        self.assertEqual(work["production_gac_support_checks"],
                         2 * persisted["gac"]["support_checks"])
        self.assertGreater(work["independent_gac_support_checks"], 0)

    def test_terminal_postwrite_resource_failure_revokes_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name, content in (("manifest.initial.json", "{}\n"),
                                  ("outcomes.jsonl", "{}\n"),
                                  ("replay.json", "{}\n"),
                                  ("report.json", '{"status":"PASS_CAPACITY_DIFFICULTY"}\n')):
                (output / name).write_text(content, encoding="utf-8")
            class Guard:
                expired = False
                peak_bytes = 123
                def check(self):
                    if (output / "manifest.json").exists():
                        self.expired = True
                        raise TimeoutError("post-write cap")
            guard = Guard()
            hashed = []
            def guarded_hash(path, *, guard=None):
                path = Path(path)
                hashed.append(path.name)
                if guard is not None and guard.expired and path.name == "outcomes.jsonl":
                    raise AssertionError("large prefix was rehashed after cap expiry")
                return _hash(path, guard=guard)
            with patch("dwr.tasks.certificate_study._hash", side_effect=guarded_hash):
                with self.assertRaises(ResourceFailure):
                    finalize_artifacts(output, guard=guard)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotEqual(report["status"], "PASS_CAPACITY_DIFFICULTY")
            self.assertIn("report.json", manifest["artifacts"])
            self.assertEqual(hashed.count("outcomes.jsonl"), 1)
            self.assertEqual(hashed[-1], "report.json")

    def test_terminal_accounting_flush_returns_saved_report_without_rehashing_large_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name, content in (("manifest.initial.json", "{}\n"),
                                  ("outcomes.jsonl", "{}\n"),
                                  ("replay.json", "{}\n"),
                                  ("report.json", '{"status":"PASS_CAPACITY_DIFFICULTY"}\n')):
                (output / name).write_text(content, encoding="utf-8")
            class Guard:
                post_terminal_check = False
                peak_bytes = 456
                def check(self):
                    if (output / "manifest.json").exists():
                        self.post_terminal_check = True
            guard = Guard()
            original_hash = _hash
            hashed = []
            def guarded_hash(path, *, guard=None):
                path = Path(path)
                if guard is not None and guard.post_terminal_check and path.name == "outcomes.jsonl":
                    raise AssertionError("large artifact rehashed after terminal guard check")
                hashed.append(path.name)
                return original_hash(path, guard=guard)
            with patch("dwr.tasks.certificate_study._hash", side_effect=guarded_hash):
                manifest = finalize_artifacts(output, guard=guard, wall_start=time.monotonic() - 1,
                                              cpu_start=time.process_time() - .1,
                                              invocation_start=time.monotonic() - 2)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["resources"]["whole_wrapper_seconds_before_accounting_flush"],
                             report["whole_wrapper_seconds_before_accounting_flush"])
            self.assertEqual(report["accounting_boundary"],
                             "after_initial_terminal_writes_and_guard_check_before_small_accounting_flush")
            self.assertEqual(hashed.count("outcomes.jsonl"), 1)


if __name__ == "__main__":
    unittest.main()
