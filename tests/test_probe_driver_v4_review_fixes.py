"""Focused regressions for the independent Task 3 review findings."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_probe_driver_v4 import (
    CONFIG, _fixed_semantic_fixture_provider, _fixture_task,
)
from dwr.tasks.audit_probe_v4 import (_public_projection, _verify_source_terminal_links,
    audit_records, validate_accepted_record)
from dwr.tasks.canonical import content_id, lineage_id, structure_signature
from dwr.tasks.config_probe_v4 import load_probe_config
from dwr.tasks.generator_probe_v4 import ProposalResult
from dwr.tasks.generator_probe_v4 import ProposalFatalContext, ProposalFatalError
from dwr.tasks.oracle import enumerate_solutions
from dwr.tasks.probe_v4 import (
    ARTIFACTS, ProbeState, _FixturePolicy, _ensure_failure_artifacts,
    _initial_manifest, _profile_maps, _terminal_report, _verify_terminal_report,
    consume_proposal, run_fixture_probe, _run_controller,
)
from dwr.tasks.run_support import ResourceGuard
from dwr.tasks.schema import parse_task_json, serialize_model_input, serialize_task
from dwr.tasks.solver import solve


class OneShotGuard(ResourceGuard):
    def __init__(self):
        super().__init__(900, 1024)
        self.trip = False
        self.elapsed_at_trip = None
        self.peak_at_trip = None
        self.cpu_at_trip = None

    def check(self):
        if self.trip:
            self.trip = False
            import time
            self.elapsed_at_trip = self.elapsed
            self.peak_at_trip = self.peak_bytes
            self.cpu_at_trip = time.process_time() - getattr(self, "dwr_cpu_started", time.process_time())
            raise RuntimeError("resource measurement failed closed")
        super().check()


class PersistentFailureGuard(ResourceGuard):
    def __init__(self):
        super().__init__(900, 1024)
        self.fail = False

    def check(self):
        if self.fail:
            raise RuntimeError("persistent resource measurement failure")
        super().check()


class ProbeDriverV4ReviewFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_probe_config(CONFIG)

    def test_operational_error_after_final_dev_candidate_cannot_become_fixture_pass(self):
        guard = OneShotGuard()
        tripped = False
        def provider(config, partition, index, active):
            nonlocal tripped
            result = _fixed_semantic_fixture_provider(config, partition, index, active)
            if partition == "development" and index == 11 and not tripped:
                tripped = True
                guard.trip = True
            return result
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "late-resource-error"
            code = _run_controller(CONFIG, output, ["internal-fixture", "late-error"],
                guard=guard, _fixture_policy=_FixturePolicy("late-error"),
                _proposal_provider=provider)
            report = json.loads((output / "probe_report.json").read_text(encoding="utf-8"))
            terminal = json.loads((output / "probe_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(report["reason"], "NOT_ASSESSED_RESOURCE_CAP")
            self.assertEqual(terminal["status"], "not_assessed")
            self.assertEqual(terminal["error"]["reason"], "NOT_ASSESSED_RESOURCE_CAP")

    def test_fatal_after_source_closure_retains_attempted_candidate_and_cause(self):
        def provider(config, partition, index, active):
            result = _fixed_semantic_fixture_provider(config, partition, index, active)
            if partition == "development" and index == 0:
                context = ProposalFatalContext(partition, index, active, "A", 6,
                    len(result.task.givens), len(result.task.constraints), result.task,
                    result.witness, 1)
                raise ProposalFatalError("solver_oracle_disagreement", context)
            return result
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "fatal-after-source"
            code = _run_controller(CONFIG, output, ["internal-fixture", "fatal-after-source"],
                _fixture_policy=_FixturePolicy("fatal-after-source"), _proposal_provider=provider)
            report = json.loads((output / ARTIFACTS[6]).read_text(encoding="utf-8"))
            terminal = json.loads((output / ARTIFACTS[-1]).read_text(encoding="utf-8"))
            outcomes = [json.loads(line) for line in (output / ARTIFACTS[1]).read_text(encoding="utf-8").splitlines()]
            fatal = next(row for row in outcomes if row["state"] == "fatal_error")
            self.assertEqual(code, 1)
            self.assertEqual((report["status"], report["reason"]),
                             ("NOT_ASSESSED", "NOT_ASSESSED_SOLVER_OR_ORACLE_ERROR"))
            self.assertEqual((fatal["partition"], fatal["candidate_index"]), ("development", 0))
            self.assertEqual(terminal["error"]["candidate_index"], 0)
            self.assertEqual(terminal["error"]["partition"], "development")

    def test_io_after_final_dev_row_is_sticky_and_terminalized(self):
        from dwr.tasks import probe_v4
        real_dump = probe_v4._dump_jsonl
        tripped = False
        def fail_after_durable_final_dev_row(stream, row):
            nonlocal tripped
            real_dump(stream, row)
            if row["partition"] == "development" and row["candidate_index"] == 11 and not tripped:
                tripped = True
                raise OSError("injected post-row write failure")
        with tempfile.TemporaryDirectory() as temp, patch.object(probe_v4, "_dump_jsonl",
                                                                    side_effect=fail_after_durable_final_dev_row):
            output = Path(temp) / "io-after-final-dev"
            code = _run_controller(CONFIG, output, ["internal-fixture", "io-after-final-dev"],
                _fixture_policy=_FixturePolicy("io-after-final-dev"),
                _proposal_provider=_fixed_semantic_fixture_provider)
            initial = json.loads((output / ARTIFACTS[0]).read_text(encoding="utf-8"))
            report = json.loads((output / ARTIFACTS[6]).read_text(encoding="utf-8"))
            replay = json.loads((output / ARTIFACTS[4]).read_text(encoding="utf-8"))
            audit = json.loads((output / ARTIFACTS[5]).read_text(encoding="utf-8"))
            terminal = json.loads((output / ARTIFACTS[-1]).read_text(encoding="utf-8"))
            outcomes = [json.loads(line) for line in (output / ARTIFACTS[1]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(code, 1)
            self.assertEqual((report["status"], report["reason"]),
                ("NOT_ASSESSED", "NOT_ASSESSED_IO_OR_SERIALIZATION_ERROR"))
            self.assertEqual(outcomes[-1]["candidate_index"], 11)
            self.assertEqual(terminal["error"]["reason"], "NOT_ASSESSED_IO_OR_SERIALIZATION_ERROR")
            self.assertEqual(terminal["error"]["type"], "OSError")
            self.assertEqual(terminal["error"]["phase"], "generation")
            self.assertEqual(terminal["error"]["partition"], "development")
            self.assertEqual(terminal["error"]["candidate_index"], 11)
            self.assertIn("injected post-row write failure", terminal["error"]["message"])
            for part in ("source", "development"):
                self.assertEqual(terminal["partitions"][part]["attempts"], replay["partition_attempts"][part])
                self.assertEqual(terminal["partitions"][part]["quota_counts"], replay["quota_counts"][part])
            from dwr.tasks.probe_v4 import (_verify_audit_document, _verify_replay_document,
                _verify_source_closure, _verify_terminal_manifest, _verify_terminal_report)
            from dwr.tasks.run_support import sha256_file
            closure = json.loads((output / ARTIFACTS[3]).read_text(encoding="utf-8"))
            policy = _FixturePolicy("io-after-final-dev")
            _verify_source_closure(closure, output, self.config, initial["source"], initial["producer"], policy)
            _verify_replay_document(replay, output, self.config, initial["source"], initial["producer"], policy)
            _verify_audit_document(audit, output, self.config, initial["source"], initial["producer"], policy)
            _verify_terminal_report(report, self.config, initial["source"], sha256_file(output / ARTIFACTS[0]),
                sha256_file(output / ARTIFACTS[3]), closure,
                {**replay, "artifact_sha256": sha256_file(output / ARTIFACTS[4])},
                {**audit, "artifact_sha256": sha256_file(output / ARTIFACTS[5])}, policy, terminal["error"])
            _verify_terminal_manifest(terminal, initial, self.config, output, replay, report, policy)

    def test_failure_placeholder_does_not_read_or_semantically_audit_corpus(self):
        from dwr.tasks import probe_v4
        from dwr.tasks.probe_v4 import _producer_identity, _source_identity
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bounded-flush"
            output.mkdir()
            (output / ARTIFACTS[1]).write_text("", encoding="utf-8")
            (output / ARTIFACTS[2]).write_text("", encoding="utf-8")
            source = _source_identity(Path(__file__).parents[1], "bounded-flush-test")
            producer = _producer_identity(self.config, Path(__file__).parents[1])
            state = ProbeState(self.config, _fixture_policy=_FixturePolicy("bounded-flush-test"))
            state.attempted_dev_profiles.add(("eq", "neq"))
            manifest = _initial_manifest(self.config, CONFIG, output, ["internal-fixture"],
                Path(__file__).parents[1], producer, source, state._fixture_policy)
            exhausted_guard = PersistentFailureGuard()
            exhausted_guard.fail = True
            with patch.object(probe_v4, "_read_jsonl", wraps=probe_v4._read_jsonl) as read_rows, \
                 patch.object(probe_v4, "audit_records", wraps=probe_v4.audit_records) as audit_rows:
                probe_v4._ensure_failure_artifacts(output, self.config, source, producer, state,
                    manifest, exhausted_guard)
                read_rows.assert_not_called()
                audit_rows.assert_not_called()
            audit = json.loads((output / ARTIFACTS[5]).read_text(encoding="utf-8"))
            dev = audit["public_projection"]["development_validation_by_profile"]
            self.assertEqual(dev["eq/neq"]["status"], "not_assessed")
            self.assertEqual(dev["eq/add_mod"]["status"], "not_attempted")

    def test_four_cohort_proxy_denominator_includes_eligible_out_of_envelope_groups(self):
        grouped = {("source",): [], ("development", "eq", "neq"): [],
                   ("development", "eq", "add_mod"): [],
                   ("development", "neq", "add_mod"): []}
        accepted = {key: [] for key in grouped}
        task = parse_task_json(serialize_task(_fixture_task("source", ("eq",), "A")))
        solution = enumerate_solutions(task).solutions[0]
        signature = structure_signature(task)
        for cell in "ABCD":
            for i in range(25):
                info = {"cell": cell, "interaction": {"eligible": True}, "signature": signature,
                        "profile": ("eq",), "solution": solution, "task": task}
                accepted[("source",)].append(info)
                grouped[("source",)].append(info)
                for profile in (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod")):
                    accepted[("development", *profile)].append(info)
                    grouped[("development", *profile)].append(info)
        for profile in (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod")):
            grouped[("development", *profile)].extend(
                {"cell": None, "interaction": {"eligible": True}, "signature": signature}
                for _ in range(50))
        projection = _public_projection(grouped, accepted)
        for profile in ("eq/neq", "eq/add_mod", "neq/add_mod"):
            self.assertEqual(projection["development_validation_by_profile"][profile]["eligible_groups"], 150)
        proxy = projection["four_cohort_proxy"]
        self.assertEqual(proxy["cohorts"]["source_train_eligible"]["group_count"], 100)
        self.assertEqual(proxy["cohorts"]["development_validation_eq_neq_primary_eligible"]["group_count"], 150)
        self.assertFalse(proxy["pass"])

    def test_profile_attempt_status_is_per_profile_even_when_rows_have_no_metrics(self):
        from dwr.tasks.audit_probe_v4 import _public_projection
        groups = {("development", "eq", "neq"): [],
                  ("development", "eq", "add_mod"): [],
                  ("development", "neq", "add_mod"): []}
        attempted = {("eq", "neq")}
        projection = _public_projection(groups, {}, attempted_dev_profiles=attempted)
        statuses = projection["development_validation_by_profile"]
        self.assertEqual(statuses["eq/neq"]["status"], "not_assessed")
        self.assertEqual(statuses["eq/add_mod"]["status"], "not_attempted")
        self.assertEqual(statuses["neq/add_mod"]["status"], "not_attempted")

    def test_source_prefix_bytes_and_hash_bind_exact_reconstructed_source_rows(self):
        from dwr.tasks.audit_probe_v4 import _verify_reconstructed_source_prefix
        from dwr.tasks.run_support import stable_json
        outcomes = [{"partition": "source", "candidate_index": 0},
                    {"partition": "development", "candidate_index": 0}]
        records = [{"partition": "source", "candidate_index": 0},
                   {"partition": "development", "candidate_index": 0}]
        outcome_prefix = (stable_json(outcomes[0]) + "\n").encode("utf-8")
        record_prefix = (stable_json(records[0]) + "\n").encode("utf-8")
        closure = {"source_attempts": 1, "outcome_prefix_bytes": len(outcome_prefix),
            "outcome_prefix_sha256": hashlib.sha256(outcome_prefix).hexdigest(),
            "record_prefix_bytes": len(record_prefix),
            "record_prefix_sha256": hashlib.sha256(record_prefix).hexdigest()}
        _verify_reconstructed_source_prefix(closure, outcomes, records)
        altered = dict(closure, outcome_prefix_bytes=closure["outcome_prefix_bytes"] + 1)
        with self.assertRaises(ValueError):
            _verify_reconstructed_source_prefix(altered, outcomes, records)
        altered = dict(closure, record_prefix_sha256="0" * 64)
        with self.assertRaises(ValueError):
            _verify_reconstructed_source_prefix(altered, outcomes, records)

    def test_terminal_report_pass_requires_all_source_dev_and_proxy_gates(self):
        from dwr.tasks.probe_v4 import ProbeState
        state = ProbeState(self.config)
        source = {"commit": "a" * 40, "dirty": False,
                  "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(), "checkout_root": "fixture"}
        closure = {"status": "NO_GO_SOURCE_CLOSURE", "source_quota_closed": False,
                   "source_quota_counts": _profile_maps(self.config, "source", state)[1],
                   "train_projection": None}
        targets = {p: _profile_maps(self.config, p, state)[0] for p in ("source", "development")}
        counts = {p: _profile_maps(self.config, p, state)[1] for p in ("source", "development")}
        replay = {"quota_counts": counts, "quota_targets": targets,
            "partition_attempts": {"source": 0, "development": 0},
            "quota_closed": {"source": False,
                "development": {"eq/neq": False, "eq/add_mod": False, "neq/add_mod": False}},
            "artifact_sha256": "c" * 64}
        metrics = {"status": "not_attempted", "eligible_groups": None,
            "unique_oracle_groups": None, "eligible_signatures": None,
            "scientifically_eligible_groups": None, "scientific_eligibility_fraction": None,
            "eligible_envelope_groups": None, "envelope_yield": None,
            "groups_by_cell": None, "eligible_groups_by_cell": None}
        audit = {"public_projection": {"source_train": None,
            "development_validation_by_profile": {key: dict(metrics) for key in
                ("eq/neq", "eq/add_mod", "neq/add_mod")}, "four_cohort_proxy": None},
            "artifact_sha256": "d" * 64}
        report = _terminal_report(self.config, closure, replay, audit, None,
            "PASS_DEVELOPMENT_PROPOSAL", ResourceGuard(900, 1024), "e" * 64,
            "f" * 64, source)
        with self.assertRaises(ValueError):
            _verify_terminal_report(report, self.config, source, "e" * 64, "f" * 64,
                                    closure, replay, audit)

    def test_synthetic_production_shape_pass_is_accepted_without_sampling(self):
        """Closed-document validator contract only; these synthetic values are not probe evidence."""
        from dwr.tasks.probe_v4 import _verify_terminal_report
        state = ProbeState(self.config)
        source_targets, source_counts = _profile_maps(self.config, "source", state)
        dev_targets, dev_counts = _profile_maps(self.config, "development", state)
        source_counts = copy.deepcopy(source_targets)
        dev_counts = copy.deepcopy(dev_targets)
        source_projection = {
            "accepted_groups": 1000, "unique_oracle_groups": 1200, "structure_signatures": 20,
            "profile_groups": {"eq": 334, "neq": 333, "add_mod": 333},
            "solution_value_counts": {"0": 2000, "1": 2000, "2": 2000},
            "given_counts": {"2": 500, "3": 500}, "sensitive_groups": 1000,
            "sensitivity_fraction": 1000 / 1200, "eligible_envelope_groups": 1000,
            "envelope_yield": 1000 / 1200, "groups_by_cell": {cell: 250 for cell in "ABCD"},
            "eligible_groups_by_cell": {cell: 250 for cell in "ABCD"},
            "four_cell_mass": {"numerator": 1000, "denominator": 1000, "fraction": 1.0}}
        closure = {"status": "PASS_SOURCE_CLOSURE", "reason": None,
            "source_quota_closed": True, "source_quota_counts": source_counts,
            "source_quota_targets": source_targets, "train_projection": source_projection,
            "source_support": {"cells": source_projection["eligible_groups_by_cell"],
                               "cell_min_pass": True, "mass_pass": True}}
        replay = {"status": "PASS_REPLAY", "quota_counts": {"source": source_counts,
            "development": dev_counts}, "quota_targets": {"source": source_targets,
            "development": dev_targets}, "quota_closed": {"source": True,
            "development": {key: True for key in dev_targets}},
            "partition_attempts": {"source": sum(sum(c.values()) for c in source_counts.values()),
                                    "development": sum(sum(c.values()) for c in dev_counts.values())},
            "artifact_sha256": "c" * 64}
        dev = {}
        for key in dev_targets:
            dev[key] = {"status": "complete", "eligible_groups": 120,
                "unique_oracle_groups": 120, "eligible_signatures": 12,
                "scientifically_eligible_groups": 120, "scientific_eligibility_fraction": 1.0,
                "eligible_envelope_groups": 100, "envelope_yield": 100 / 120,
                "groups_by_cell": {cell: 25 for cell in "ABCD"},
                "eligible_groups_by_cell": {cell: 30 for cell in "ABCD"}}
        def cohort(count):
            cells = {cell: count // 4 for cell in "ABCD"}
            return {"group_count": count, "groups_by_cell": cells,
                "common_cell_groups": dict(cells), "mass_fraction": 1.0, "pass": True}
        proxy = {"cohorts": {"source_train_eligible": cohort(1000),
            "development_validation_eq_neq_primary_eligible": cohort(120),
            "development_validation_eq_add_mod_primary_eligible": cohort(120),
            "development_validation_neq_add_mod_primary_eligible": cohort(120)}, "pass": True}
        for key in dev_targets:
            proxy["cohorts"]["development_validation_" + key.replace("/", "_") + "_primary_eligible"]["groups_by_cell"] = {cell: 30 for cell in "ABCD"}
            proxy["cohorts"]["development_validation_" + key.replace("/", "_") + "_primary_eligible"]["common_cell_groups"] = {cell: 30 for cell in "ABCD"}
        audit = {"status": "PASS_AUDIT", "integrity": {"status": "pass", "failures": []},
            "public_projection": {"source_train": source_projection,
                "development_validation_by_profile": dev, "four_cohort_proxy": proxy},
            "artifact_sha256": "d" * 64}
        source = {"commit": "a" * 40, "dirty": False,
            "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(), "checkout_root": "synthetic-doc-only"}
        guard = ResourceGuard(900, 1024)
        guard.check()
        report = _terminal_report(self.config, closure, replay, audit, None,
            "PASS_DEVELOPMENT_PROPOSAL", guard, "e" * 64, "f" * 64, source,
            {key: "pass" for key in dev_targets})
        _verify_terminal_report(report, self.config, source, "e" * 64, "f" * 64,
                                closure, replay, audit)
        altered = copy.deepcopy(report)
        altered_closure, altered_audit = copy.deepcopy(closure), copy.deepcopy(audit)
        altered_closure["train_projection"]["structure_signatures"] = 11
        altered_audit["public_projection"]["source_train"]["structure_signatures"] = 11
        altered["public_floors"]["source_train"]["values"]["structure_signatures"] = 11
        with self.assertRaises(ValueError):
            _verify_terminal_report(altered, self.config, source, "e" * 64, "f" * 64,
                                    altered_closure, replay, altered_audit)
        altered_dev = copy.deepcopy(report)
        altered_audit = copy.deepcopy(audit)
        altered_audit["public_projection"]["development_validation_by_profile"]["eq/neq"]["eligible_signatures"] = 11
        altered_dev["public_floors"]["development_validation_by_profile"]["eq/neq"]["values"]["eligible_signatures"] = 11
        with self.assertRaises(ValueError):
            _verify_terminal_report(altered_dev, self.config, source, "e" * 64, "f" * 64,
                                    closure, replay, altered_audit)
        bad_closure = copy.deepcopy(closure)
        bad_closure["source_quota_closed"] = False
        bad_replay = copy.deepcopy(replay)
        bad_replay["quota_closed"]["development"]["eq/neq"] = False
        bad_audit = copy.deepcopy(audit)
        bad_audit["status"] = "NOT_ASSESSED"
        bad_report = copy.deepcopy(report)
        bad_report["public_floors"]["development_validation_by_profile"]["eq/neq"] = {
            "status": "not_attempted", "values": None}
        bad_proxy_report = copy.deepcopy(report)
        bad_proxy_report["four_cohort_proxy"] = None
        for altered_report, altered_closure, altered_replay, altered_audit in (
                (report, bad_closure, replay, audit),
                (report, closure, bad_replay, audit),
                (report, closure, replay, bad_audit),
                (bad_report, closure, replay, audit),
                (bad_proxy_report, closure, replay, audit)):
            with self.assertRaises(ValueError):
                _verify_terminal_report(altered_report, self.config, source, "e" * 64,
                    "f" * 64, altered_closure, altered_replay, altered_audit)

    def test_synthetic_production_shape_not_assessed_bundle_is_accepted(self):
        """A fixed, empty NOT_ASSESSED bundle exercises production schemas without draws."""
        from datetime import datetime, timezone
        from dwr.tasks.probe_v4 import (
            _artifact_ref, _ensure_failure_artifacts, _initial_manifest,
            _not_assessed_report, _producer_identity, _source_identity,
            _verify_audit_document, _verify_initial_binding, _verify_replay_document,
            _verify_source_closure, _verify_terminal_manifest, _verify_terminal_report,
        )
        from dwr.tasks.run_support import sha256_file
        root = Path(__file__).parents[1]
        source = {"commit": "a" * 40, "dirty": False,
            "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(),
            "checkout_root": "synthetic-fixed-document-only"}
        producer = _producer_identity(self.config, root)
        error = {"type": "RuntimeError", "message": "NOT_ASSESSED_RESOURCE_CAP",
            "phase": "controller_preflight", "partition": None,
            "candidate_index": None, "reason": "NOT_ASSESSED_RESOURCE_CAP"}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "synthetic-production-shaped"
            output.mkdir()
            state = ProbeState(self.config)
            guard = ResourceGuard(900, 1024)
            guard.check()
            initial = _initial_manifest(self.config, CONFIG, output, ["synthetic-document-test"],
                root, producer, source)
            initial_path = output / ARTIFACTS[0]
            initial_path.write_text(json.dumps(initial, sort_keys=True) + "\n", encoding="utf-8")
            _ensure_failure_artifacts(output, self.config, source, producer, state, initial, guard)
            closure = json.loads((output / ARTIFACTS[3]).read_text(encoding="utf-8"))
            replay = json.loads((output / ARTIFACTS[4]).read_text(encoding="utf-8"))
            audit = json.loads((output / ARTIFACTS[5]).read_text(encoding="utf-8"))
            report = _not_assessed_report(self.config, source, initial_path,
                output / ARTIFACTS[3], guard, error["reason"], state)
            report.update({"initial_manifest_sha256": sha256_file(initial_path),
                "source_closure_sha256": sha256_file(output / ARTIFACTS[3]),
                "replay_sha256": sha256_file(output / ARTIFACTS[4]),
                "audit_sha256": sha256_file(output / ARTIFACTS[5])})
            from dwr.tasks.run_support import atomic_json
            atomic_json(output / ARTIFACTS[6], report)
            terminal = copy.deepcopy(initial)
            terminal.update({"status": "not_assessed",
                "finished_utc": datetime.now(timezone.utc).isoformat(), "error": error,
                "exit_code": 1, "elapsed_wall_seconds": report["resource_snapshot"]["wall_seconds"],
                "cpu_seconds": report["resource_snapshot"]["cpu_seconds"],
                "peak_resident_bytes": report["resource_snapshot"]["peak_rss_bytes"],
                "accounting_end_boundary": "post_first_terminal_write_resource_check",
                "artifacts": {name: _artifact_ref(output / name) for name in ARTIFACTS[:-1]}})
            atomic_json(output / ARTIFACTS[-1], terminal)
            _verify_initial_binding(initial, self.config, source, producer, CONFIG)
            _verify_source_closure(closure, output, self.config, source, producer)
            _verify_replay_document(replay, output, self.config, source, producer)
            _verify_audit_document(audit, output, self.config, source, producer)
            _verify_terminal_report(report, self.config, source, sha256_file(initial_path),
                sha256_file(output / ARTIFACTS[3]), closure,
                {**replay, "artifact_sha256": sha256_file(output / ARTIFACTS[4])},
                {**audit, "artifact_sha256": sha256_file(output / ARTIFACTS[5])}, error=error)
            _verify_terminal_manifest(terminal, initial, self.config, output, replay, report)

    def test_synthetic_production_shape_source_no_go_is_accepted(self):
        """A typed source-shortfall document is accepted as NO_GO, not PASS or error."""
        state = ProbeState(self.config)
        source_targets, empty_source = _profile_maps(self.config, "source", state)
        dev_targets, empty_dev = _profile_maps(self.config, "development", state)
        closure = {"status": "NO_GO_SOURCE_CLOSURE", "reason": "SOURCE_QUOTA_SHORTFALL",
            "source_quota_closed": False, "source_quota_counts": empty_source,
            "source_quota_targets": source_targets, "train_projection": None,
            "source_support": None}
        replay = {"status": "PASS_REPLAY", "quota_counts": {"source": empty_source,
            "development": empty_dev}, "quota_targets": {"source": source_targets,
            "development": dev_targets}, "quota_closed": {"source": False,
            "development": {key: False for key in dev_targets}},
            "partition_attempts": {"source": 0, "development": 0}, "artifact_sha256": "c" * 64}
        incomplete = {"status": "not_attempted", "eligible_groups": None,
            "unique_oracle_groups": None, "eligible_signatures": None,
            "scientifically_eligible_groups": None, "scientific_eligibility_fraction": None,
            "eligible_envelope_groups": None, "envelope_yield": None,
            "groups_by_cell": None, "eligible_groups_by_cell": None}
        audit = {"status": "PASS_AUDIT", "integrity": {"status": "pass", "failures": []},
            "public_projection": {"source_train": None,
                "development_validation_by_profile": {key: dict(incomplete) for key in dev_targets},
                "four_cohort_proxy": None}, "artifact_sha256": "d" * 64}
        source = {"commit": "a" * 40, "dirty": False,
            "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(), "checkout_root": "synthetic-no-go-doc"}
        guard = ResourceGuard(900, 1024)
        guard.check()
        report = _terminal_report(self.config, closure, replay, audit,
            "SOURCE_QUOTA_SHORTFALL", "NO_GO_PROPOSAL", guard, "e" * 64, "f" * 64, source)
        _verify_terminal_report(report, self.config, source, "e" * 64, "f" * 64,
                                closure, replay, audit)

    def test_terminal_source_links_reject_independent_count_projection_and_gate_mutations(self):
        state = ProbeState(self.config)
        _, source_counts = _profile_maps(self.config, "source", state)
        projection = {"accepted_groups": 0}
        source_doc = {"status": "NO_GO_SOURCE_CLOSURE", "source_attempts": 2,
            "source_quota_counts": source_counts, "train_projection": projection}
        replay_doc = {"quota_counts": {"source": copy.deepcopy(source_counts)},
            "partition_attempts": {"source": 2, "development": 0},
            "quota_closed": {"source": False}}
        audit_doc = {"public_projection": {"source_train": projection}}
        _verify_source_terminal_links(source_doc, replay_doc, audit_doc)
        mutations = []
        counts_bad = copy.deepcopy(replay_doc)
        counts_bad["quota_counts"]["source"]["eq"]["A"] = 1
        mutations.append((source_doc, counts_bad, audit_doc))
        attempts_bad = copy.deepcopy(replay_doc)
        attempts_bad["partition_attempts"]["source"] = 1
        mutations.append((source_doc, attempts_bad, audit_doc))
        projection_bad = {"public_projection": {"source_train": {"accepted_groups": 1}}}
        mutations.append((source_doc, replay_doc, projection_bad))
        dev_before_source = copy.deepcopy(replay_doc)
        dev_before_source["partition_attempts"]["development"] = 1
        mutations.append((source_doc, dev_before_source, audit_doc))
        for closure, replay, audit in mutations:
            with self.subTest(replay_attempts=replay["partition_attempts"],
                              projection=audit["public_projection"]), self.assertRaises(ValueError):
                _verify_source_terminal_links(closure, replay, audit)

    def test_consume_proposal_uses_one_canonical_content_traversal_for_lineage(self):
        import dwr.tasks.canonical as canonical
        original = canonical.canonical_content
        calls = []
        def count_calls(task):
            calls.append(task)
            return original(task)
        with patch.object(canonical, "canonical_content", side_effect=count_calls):
            outcome, _ = consume_proposal(self.config, ProbeState(self.config),
                "source", 2, ("add_mod",), "C")
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(outcome["content_id_or_null"])
        parsed_task = parse_task_json(json.dumps(outcome["task_or_null"]))
        self.assertEqual(outcome["content_id_or_null"], content_id(parsed_task))
        actual_task = parse_task_json(json.dumps(outcome["task_or_null"]))
        self.assertEqual(outcome["lineage_id_or_null"], lineage_id(actual_task))

    def test_failure_artifact_bundle_uses_null_incomplete_replay_and_profile_audit_shapes(self):
        from dwr.tasks.probe_v4 import _producer_identity, _source_identity
        root = Path(__file__).parents[1]
        source = _source_identity(root, "incomplete-schema")
        producer = _producer_identity(self.config, root)
        state = ProbeState(self.config, _fixture_policy=_FixturePolicy("incomplete-schema"))
        guard = ResourceGuard(900, 1024)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "incomplete"
            output.mkdir()
            (output / ARTIFACTS[1]).write_text("", encoding="utf-8")
            (output / ARTIFACTS[2]).write_text("", encoding="utf-8")
            initial = _initial_manifest(self.config, CONFIG, output, [], root, producer, source,
                                        _FixturePolicy("incomplete-schema"))
            _ensure_failure_artifacts(output, self.config, source, producer, state, initial, guard)
            replay = json.loads((output / ARTIFACTS[4]).read_text(encoding="utf-8"))
            audit = json.loads((output / ARTIFACTS[5]).read_text(encoding="utf-8"))
            self.assertIsNone(replay["outcome_prefix_bytes"])
            self.assertIsNone(replay["outcome_prefix_sha256"])
            self.assertIsNone(replay["record_prefix_bytes"])
            self.assertIsNone(replay["record_prefix_sha256"])
            self.assertIsNone(replay["denominator_projection_sha256"])
            self.assertIsNone(replay["quota_closed"]["source"])
            self.assertEqual(set(replay["quota_closed"]["development"]),
                             {"eq/neq", "eq/add_mod", "neq/add_mod"})
            self.assertTrue(all(value is None for value in replay["quota_closed"]["development"].values()))
            profiles = audit["public_projection"]["development_validation_by_profile"]
            self.assertEqual(set(profiles), {"eq/neq", "eq/add_mod", "neq/add_mod"})
            self.assertTrue(all(value["status"] == "not_attempted" and
                                all(value[key] is None for key in (
                                    "eligible_groups", "unique_oracle_groups", "eligible_signatures",
                                    "scientifically_eligible_groups", "scientific_eligibility_fraction",
                                    "eligible_envelope_groups", "envelope_yield", "groups_by_cell",
                                    "eligible_groups_by_cell")) for value in profiles.values()))

    def test_independent_audit_rejects_ineligible_accepted_outcome_record_pair(self):
        state = ProbeState(self.config)
        outcome, _ = consume_proposal(self.config, state, "source", 2, ("add_mod",), "C")
        task = parse_task_json(json.dumps(outcome["task_or_null"]))
        prompt = serialize_model_input(task)
        solution = outcome["oracle_solution_or_null"]
        record = {"record_version": "dwr-v4-probe-accepted-record-v1",
            "generator_id": "dwr-generator-v4-probe", "partition": "source",
            "candidate_index": 2, "profile": ["add_mod"], "task": outcome["task_or_null"],
            "content_id": outcome["content_id_or_null"], "lineage_id": outcome["lineage_id_or_null"],
            "structure_signature": outcome["structure_signature_or_null"],
            "split": outcome["split_or_null"], "witness": solution, "solution": solution,
            "solver_stats": outcome["solver_stats_or_null"], "model_input": prompt,
            "model_input_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
        outcome["state"], outcome["reason"] = "accepted_for_quota", None
        outcome["quota_cell_count_after_or_null"] = 1
        self.assertFalse(outcome["private_eligibility_or_null"]["eligible"])
        with self.assertRaises(ValueError):
            audit_records([record], [outcome], self.config,
                          _fixture_targets=_FixturePolicy("ineligible-acceptance").quotas)

    def test_cap_after_initial_manifest_write_is_preserved_as_terminal_not_assessed(self):
        guard = PersistentFailureGuard()
        real_write_text = Path.write_text
        def set_cap_after_initial(path, *args, **kwargs):
            result = real_write_text(path, *args, **kwargs)
            if Path(path).name == ARTIFACTS[0]:
                guard.fail = True
            return result
        with tempfile.TemporaryDirectory() as temp, patch.object(Path, "write_text",
                                                                    set_cap_after_initial):
            output = Path(temp) / "initial-cap"
            try:
                code = _run_controller(CONFIG, output, ["internal-fixture", "initial-cap"],
                    guard=guard, _fixture_policy=_FixturePolicy("initial-cap"),
                    _proposal_provider=_fixed_semantic_fixture_provider)
            except Exception as exc:
                self.fail(f"controller escaped instead of preserving initial-write cap: {exc!r}")
            self.assertEqual(code, 1)
            self.assertTrue((output / ARTIFACTS[-1]).exists())
            report = json.loads((output / ARTIFACTS[6]).read_text(encoding="utf-8"))
            terminal = json.loads((output / ARTIFACTS[-1]).read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(terminal["status"], "not_assessed")

    def test_preterminal_persistent_cap_is_flushed_and_returns_failure(self):
        guard = PersistentFailureGuard()
        from dwr.tasks import probe_v4
        real_atomic_json = probe_v4.atomic_json
        def set_cap_after_report(path, value):
            real_atomic_json(path, value)
            if Path(path).name == ARTIFACTS[6]:
                guard.fail = True
        with tempfile.TemporaryDirectory() as temp, patch.object(probe_v4, "atomic_json",
                                                                    side_effect=set_cap_after_report):
            output = Path(temp) / "preterminal-cap"
            try:
                code = _run_controller(CONFIG, output, ["internal-fixture", "preterminal-cap"],
                    guard=guard, _fixture_policy=_FixturePolicy("preterminal-cap"),
                    _proposal_provider=_fixed_semantic_fixture_provider)
            except Exception as exc:
                self.fail(f"controller escaped instead of preserving preterminal cap: {exc!r}")
            self.assertEqual(code, 1)
            report = json.loads((output / ARTIFACTS[6]).read_text(encoding="utf-8"))
            terminal = json.loads((output / ARTIFACTS[-1]).read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(terminal["status"], "not_assessed")
            from dwr.tasks.run_support import strict_json_loads
            from dwr.tasks.probe_v4 import (_verify_audit_document, _verify_replay_document,
                _verify_source_closure, _verify_terminal_manifest, _verify_terminal_report)
            from dwr.tasks.run_support import sha256_file
            initial = strict_json_loads((output / ARTIFACTS[0]).read_text(encoding="utf-8"))
            closure = strict_json_loads((output / ARTIFACTS[3]).read_text(encoding="utf-8"))
            replay = strict_json_loads((output / ARTIFACTS[4]).read_text(encoding="utf-8"))
            audit = strict_json_loads((output / ARTIFACTS[5]).read_text(encoding="utf-8"))
            self.assertEqual(terminal["partitions"]["source"]["attempts"], replay["partition_attempts"]["source"])
            self.assertEqual(terminal["partitions"]["development"]["attempts"], replay["partition_attempts"]["development"])
            policy = _FixturePolicy("preterminal-cap")
            _verify_source_closure(closure, output, self.config, initial["source"], initial["producer"], policy)
            _verify_replay_document(replay, output, self.config, initial["source"], initial["producer"], policy)
            _verify_audit_document(audit, output, self.config, initial["source"], initial["producer"], policy)
            _verify_terminal_report(report, self.config, initial["source"], sha256_file(output / ARTIFACTS[0]),
                sha256_file(output / ARTIFACTS[3]), closure,
                {**replay, "artifact_sha256": sha256_file(output / ARTIFACTS[4])},
                {**audit, "artifact_sha256": sha256_file(output / ARTIFACTS[5])}, policy, terminal["error"])
            _verify_terminal_manifest(terminal, initial, self.config, output, replay, report,
                                      policy)

    def test_emergency_terminal_write_samples_live_rss_and_revokes_decision(self):
        from dwr.tasks import probe_v4
        over = {"value": False}
        guard = OneShotGuard()
        guard.memory_reader = lambda: 2 * 1024 * 1024 * 1024 if over["value"] else 100
        real_atomic_json = probe_v4.atomic_json
        fired = {"value": False}
        def trigger_preterminal_then_raise_rss(path, value):
            real_atomic_json(path, value)
            if Path(path).name == ARTIFACTS[6] and not fired["value"]:
                fired["value"] = True
                guard.trip = True
            if Path(path).name == ARTIFACTS[-1]:
                over["value"] = True
        with tempfile.TemporaryDirectory() as temp, patch.object(probe_v4, "atomic_json",
                                                                    side_effect=trigger_preterminal_then_raise_rss):
            output = Path(temp) / "emergency-postwrite-rss"
            code = _run_controller(CONFIG, output, ["internal-fixture", "emergency-postwrite-rss"],
                guard=guard, _fixture_policy=_FixturePolicy("emergency-postwrite-rss"),
                _proposal_provider=_fixed_semantic_fixture_provider)
            report = json.loads((output / ARTIFACTS[6]).read_text(encoding="utf-8"))
            terminal = json.loads((output / ARTIFACTS[-1]).read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertEqual(report["reason"], "NOT_ASSESSED_RESOURCE_CAP")
            self.assertEqual(terminal["error"]["phase"], "post_terminal_write")
            self.assertEqual(report["resource_snapshot"]["peak_rss_bytes"], 2 * 1024 * 1024 * 1024)
            self.assertEqual(report["resource_snapshot"]["wall_seconds"], terminal["elapsed_wall_seconds"])
            self.assertEqual(report["resource_snapshot"]["cpu_seconds"], terminal["cpu_seconds"])
            self.assertEqual(report["resource_snapshot"]["peak_rss_bytes"], terminal["peak_resident_bytes"])
            from dwr.tasks.run_support import strict_json_loads, sha256_file
            from dwr.tasks.probe_v4 import (_verify_audit_document, _verify_replay_document,
                _verify_source_closure, _verify_terminal_manifest, _verify_terminal_report)
            initial = strict_json_loads((output / ARTIFACTS[0]).read_text(encoding="utf-8"))
            closure = strict_json_loads((output / ARTIFACTS[3]).read_text(encoding="utf-8"))
            replay = strict_json_loads((output / ARTIFACTS[4]).read_text(encoding="utf-8"))
            audit = strict_json_loads((output / ARTIFACTS[5]).read_text(encoding="utf-8"))
            policy = _FixturePolicy("emergency-postwrite-rss")
            _verify_source_closure(closure, output, self.config, initial["source"], initial["producer"], policy)
            _verify_replay_document(replay, output, self.config, initial["source"], initial["producer"], policy)
            _verify_audit_document(audit, output, self.config, initial["source"], initial["producer"], policy)
            _verify_terminal_report(report, self.config, initial["source"], sha256_file(output / ARTIFACTS[0]),
                sha256_file(output / ARTIFACTS[3]), closure,
                {**replay, "artifact_sha256": sha256_file(output / ARTIFACTS[4])},
                {**audit, "artifact_sha256": sha256_file(output / ARTIFACTS[5])}, policy, terminal["error"])
            _verify_terminal_manifest(terminal, initial, self.config, output, replay, report, policy)

    def test_postwrite_cap_returns_nonzero_and_refreshes_resource_accounting(self):
        import time
        guard = OneShotGuard()
        from dwr.tasks import probe_v4
        real_atomic_json = probe_v4.atomic_json
        def cap_after_first_terminal_write(path, value):
            real_atomic_json(path, value)
            if Path(path).name == ARTIFACTS[-1] and not getattr(cap_after_first_terminal_write, "done", False):
                cap_after_first_terminal_write.done = True
                time.sleep(0.04)
                for _ in range(100000):
                    _ = 7 * 9
                guard.trip = True
        with tempfile.TemporaryDirectory() as temp, patch.object(probe_v4, "atomic_json",
                                                                    side_effect=cap_after_first_terminal_write):
            output = Path(temp) / "postwrite-cap"
            code = _run_controller(CONFIG, output, ["internal-fixture", "postwrite-cap"],
                guard=guard, _fixture_policy=_FixturePolicy("postwrite-cap"),
                _proposal_provider=_fixed_semantic_fixture_provider)
            terminal = json.loads((output / ARTIFACTS[-1]).read_text(encoding="utf-8"))
            report = json.loads((output / ARTIFACTS[6]).read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertGreaterEqual(terminal["elapsed_wall_seconds"] + 1e-5, guard.elapsed_at_trip)
            self.assertGreaterEqual(terminal["cpu_seconds"], guard.cpu_at_trip - 0.01)
            self.assertEqual(terminal["peak_resident_bytes"], guard.peak_at_trip)
            self.assertEqual(report["resource_snapshot"]["wall_seconds"],
                             terminal["elapsed_wall_seconds"])
            self.assertEqual(report["resource_snapshot"]["cpu_seconds"], terminal["cpu_seconds"])
            self.assertEqual(report["resource_snapshot"]["peak_rss_bytes"],
                             terminal["peak_resident_bytes"])
            from dwr.tasks.probe_v4 import _verify_terminal_manifest, _verify_terminal_report
            from dwr.tasks.run_support import sha256_file, strict_json_loads
            initial = strict_json_loads((output / ARTIFACTS[0]).read_text(encoding="utf-8"))
            closure = strict_json_loads((output / ARTIFACTS[3]).read_text(encoding="utf-8"))
            replay = strict_json_loads((output / ARTIFACTS[4]).read_text(encoding="utf-8"))
            audit = strict_json_loads((output / ARTIFACTS[5]).read_text(encoding="utf-8"))
            policy = _FixturePolicy("postwrite-cap")
            self.assertEqual(terminal["artifacts"][ARTIFACTS[6]]["sha256"], sha256_file(output / ARTIFACTS[6]))
            _verify_terminal_report(report, self.config, initial["source"],
                sha256_file(output / ARTIFACTS[0]), sha256_file(output / ARTIFACTS[3]),
                closure, {**replay, "artifact_sha256": sha256_file(output / ARTIFACTS[4])},
                {**audit, "artifact_sha256": sha256_file(output / ARTIFACTS[5])},
                policy, terminal["error"])
            _verify_terminal_manifest(terminal, initial, self.config, output, replay, report, policy)


if __name__ == "__main__":
    unittest.main()
