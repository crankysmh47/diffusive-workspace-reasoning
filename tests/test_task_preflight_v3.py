import json
import tempfile
import unittest
import copy
import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

from dwr.tasks.config_v3 import load_config_v3
from dwr.tasks.dataset_v2 import DatasetLimits, validate_preflight_v3
from dwr.tasks.run_support import ResourceGuard, sha256_file, stable_json

ROOT = Path(__file__).resolve().parents[1]


class PreflightV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config_v3(ROOT / "configs/tasks/constraint_v3.json")
        cls.limits = DatasetLimits({"eq": 1, "neq": 1, "add_mod": 1}, {}, {}, 2, 600, 1024)

    def test_v2_and_unknown_preflight_versions_are_rejected(self):
        source = {"commit": "fixture", "dirty": False, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        base = {"report_version": "dwr-task-v2-feasibility-preflight-v1",
                "status": "PASS_FEASIBILITY", "mode": "feasibility",
                "config_sha256": self.config.sha256, "source": source,
                "generation_dir": str(ROOT), "generation_manifest_sha256": "a" * 64,
                "generation_artifacts": {}, "audit_artifact_path": str(ROOT / "gate"),
                "audit_artifact_sha256": "b" * 64, "audit_source": source}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "preflight.json"
            for version in ("dwr-task-v2-feasibility-preflight-v1", "unknown"):
                base["report_version"] = version
                path.write_text(stable_json(base), encoding="utf-8")
                with self.subTest(version=version), self.assertRaisesRegex(ValueError, "not PASS_FEASIBILITY"):
                    validate_preflight_v3(path, self.config, source, limits=self.limits)

    def test_generation_hash_substitution_fails_before_artifact_acceptance(self):
        source = {"commit": "fixture", "dirty": False, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        wrapper = {"report_version": "dwr-task-v3-feasibility-preflight-v1",
                   "status": "PASS_FEASIBILITY", "mode": "feasibility",
                   "config_sha256": self.config.sha256, "source": source,
                   "generation_dir": str(ROOT), "generation_manifest_sha256": "a" * 64,
                   "generation_artifacts": {}, "audit_artifact_path": str(ROOT / "gate"),
                   "audit_artifact_sha256": "b" * 64, "audit_source": source}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "preflight.json"
            path.write_text(stable_json(wrapper), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "manifest.json"):
                validate_preflight_v3(path, self.config, source, limits=self.limits,
                                      guard=ResourceGuard(20, 1024, memory_reader=lambda: 0))

    def test_live_guard_is_checked_before_wrapper_read(self):
        source = {"commit": "fixture", "dirty": False, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        times = iter((0.0, 1.0))
        guard = ResourceGuard(0.1, 1024, clock=lambda: next(times), memory_reader=lambda: 0)
        guard.started = 0.0
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "preflight.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(TimeoutError):
                validate_preflight_v3(path, self.config, source, limits=self.limits, guard=guard)

    def _accepted_handoff(self, root, mutate=None):
        from dwr.tasks import dataset_v2
        from dwr.tasks.audit_v2 import _audit_producer_identity
        from dwr.tasks.stream_contract import resolve_stream
        contract = resolve_stream("dwr-generator-v3")
        source = {"commit": "fixture", "dirty": False, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        generation_dir = root / "generation"
        generation_dir.mkdir()
        outcomes = b'{"synthetic":1}\n{"synthetic":2}\n{"synthetic":3}\n'
        records = b'{"synthetic":1}\n{"synthetic":2}\n{"synthetic":3}\n'
        generation_producer = dataset_v2._producer_identity("dwr-generator-v3")
        context = {"context_version": "dwr-task-v3-source-prefix-context-v1",
            "generator_id": contract.generator_id, "config_version": contract.config_version,
            "config_sha256": self.config.sha256, "source": source, "producer": generation_producer,
            "mode": "feasibility", "source_attempts": 3,
            "source_outcome_prefix_bytes": len(outcomes),
            "source_outcome_prefix_sha256": hashlib.sha256(outcomes).hexdigest(),
            "source_accepted_record_count": 3,
            "source_record_prefix_bytes": len(records),
            "source_record_prefix_sha256": hashlib.sha256(records).hexdigest(),
            "source_quota_targets": {"eq": 1, "neq": 1, "add_mod": 1},
            "source_quota_counts": {"eq": 1, "neq": 1, "add_mod": 1},
            "source_quota_closed": True, "development_attempts": 0, "final_attempts": 0}
        context_bytes = (stable_json(context) + "\n").encode("utf-8")
        context_sha = hashlib.sha256(context_bytes).hexdigest()
        prefix_replay = {"replay_version": "dwr-task-v3-source-prefix-replay-v1",
            "status": "PASS_SOURCE_CLOSURE", "context_sha256": context_sha,
            "generator_id": contract.generator_id, "config_sha256": self.config.sha256,
            "source": source, "producer": generation_producer,
            "source_attempts": 3, "source_accepted_record_count": 3,
            "source_quota_counts": {"eq": 1, "neq": 1, "add_mod": 1},
            "source_quota_closed": True, "development_attempts": 0, "final_attempts": 0,
            "source_outcome_prefix_bytes": len(outcomes),
            "source_outcome_prefix_sha256": hashlib.sha256(outcomes).hexdigest(),
            "source_record_prefix_bytes": len(records),
            "source_record_prefix_sha256": hashlib.sha256(records).hexdigest()}
        prefix = {"context_file": "source_prefix_context.json",
            "context_sha256": context_sha,
            "replay_file": "source_prefix_replay.json",
            "replay_sha256": hashlib.sha256((stable_json(prefix_replay) + "\n").encode("utf-8")).hexdigest()}
        identities = {}
        for name, body in (("outcomes.jsonl", outcomes), ("records.jsonl", records),
                           ("source_capacity_bound.json", b"{}\n"),
                           ("source_prefix_context.json", context_bytes),
                           ("source_prefix_replay.json", (stable_json(prefix_replay) + "\n").encode("utf-8"))):
            (generation_dir / name).write_bytes(body)
            identities[name] = {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
        capacity = {"bound_version": "dwr-task-v3-source-capacity-bound-v1",
            "status": "PASS_CAPACITY_BOUND", "reason": None,
            "source_prefix_replay_sha256": identities["source_prefix_replay.json"]["sha256"], "top_k_cells": 10,
            "minimum_source_train_mass": 0.8, "source_train_group_count": 100,
            "top_cells": [{"cell": [3, 2, 4, 0, 0], "count": 80}], "top_k_mass": 0.8}
        generation = {"status": "complete", "mode": "feasibility", "source": source,
                      "artifacts": identities, "source_capacity_bound": capacity,
                      "source_prefix": prefix, "producer": generation_producer,
                      "generation_elapsed_seconds": 5.0, "generation_cpu_seconds": 4.0,
                      "peak_resident_bytes": 10}
        (generation_dir / "manifest.json").write_text(stable_json(generation), encoding="utf-8")
        generation_hash = sha256_file(generation_dir / "manifest.json")
        report = {"audit_version": contract.audit_version, "status": "complete", "mode": "feasibility",
                  "g2_status": "PASS_FEASIBILITY", "proposal_status": "PASS_FEASIBILITY",
                  "proposal_reason": None, "config_sha256": self.config.sha256, "source": source,
                  "generation_manifest_sha256": generation_hash, "generation_artifacts": identities}
        gate = {"gate_version": contract.audit_gate_version, **{k: copy.deepcopy(v) for k, v in report.items()
                if k != "source"}, "source": source, "source_prefix": prefix, "capacity_bound": capacity}
        report.pop("source")
        report["generation_source"] = source
        report["audit_source"] = source
        report.update({"generation_complete": True, "generation_status": "complete",
            "replay": {"status": "complete", "outcomes_sha256": identities["outcomes.jsonl"]["sha256"],
                       "records_sha256": identities["records.jsonl"]["sha256"]},
            "replay_pass": True, "integrity_pass": True, "capacity_bound": capacity,
            "source_prefix_replay": prefix_replay,
            "four_cohort_common_support_proxy": {"pass": True}, "final_assessment": "not_assessed",
            "metrics": {"structural_pass": True},
            "denominator_profiles": {}, "per_example_outcomes": [],
            "independent_unique_candidate_count": 0, "accepted_record_count": 0,
            "unassessed": [], "resource": {"audit_elapsed_seconds": 1.0,
                "combined_generation_audit_elapsed_seconds": 6.0, "audit_cpu_seconds": 1.0,
                "combined_generation_audit_cpu_seconds": 5.0, "peak_resident_bytes": 10,
                "combined_peak_resident_bytes": 10, "remaining_global_seconds": 595.0,
                "accounting_snapshot_boundary": "before_detailed_report_serialization",
                "authoritative_final_accounting": "audit_manifest.json", "accounting_is_final": False}})
        wrapper = {"report_version": contract.preflight_version, "status": "PASS_FEASIBILITY",
                   "mode": "feasibility", "config_sha256": self.config.sha256, "source": source,
                   "generation_dir": str(generation_dir.resolve()),
                   "generation_manifest_sha256": generation_hash, "generation_artifacts": identities,
                   "audit_artifact_path": str((generation_dir / "audit_gate.json").resolve()),
                   "audit_artifact_sha256": None,
                   "audit_source": source}
        run_manifest = {"audit_manifest_version": contract.audit_run_manifest_version,
            "status": "complete", "command": [sys.executable, "-m", "dwr.tasks.audit_cli_v3",
            "--config", str(ROOT / "configs/tasks/constraint_v3.json"), "--mode", "feasibility",
            "--data", str(generation_dir)],
            "argv": ["--config", str(ROOT / "configs/tasks/constraint_v3.json"), "--mode", "feasibility",
                     "--data", str(generation_dir)], "generation_dir": str(generation_dir.resolve()),
            "mode": "feasibility", "cwd": str(ROOT), "os": "fixture", "python": "fixture",
            "cpu_count": 1, "rng": "none; deterministic audit", "precision": "integer CPU",
            "audit_cpu_started_seconds": 1.0, "started_utc": "2026-10-08T00:00:00+00:00",
            "finished_utc": "2026-10-08T00:00:01+00:00", "producer": _audit_producer_identity(),
            "error": None, "finalization_errors": [], "config_path": str((ROOT / "configs/tasks/constraint_v3.json").resolve()),
            "config_sha256": self.config.sha256, "generation_manifest_sha256": generation_hash,
            "generation_artifacts": identities, "generation_source": source, "audit_source": source,
            "remaining_global_seconds_at_start": 595.0, "audit_elapsed_seconds": 1.0,
            "combined_generation_audit_elapsed_seconds": 6.0, "audit_cpu_seconds": 1.0,
            "combined_generation_audit_cpu_seconds": 5.0, "peak_resident_bytes": 10,
            "combined_peak_resident_bytes": 10,
            "accounting_end_boundary": "post_first_terminal_manifest_write_resource_check",
            "accounting_note": "one bounded final metadata flush follows and is excluded", "exit_code": 0,
            "artifacts": {}}
        def artifact(name):
            path = generation_dir / name
            return {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        (generation_dir / "audit_report.json").write_text(stable_json(report), encoding="utf-8")
        (generation_dir / "audit_gate.json").write_text(stable_json(gate), encoding="utf-8")
        run_manifest["artifacts"] = {name: artifact(name) for name in ("audit_report.json", "audit_gate.json")}
        (generation_dir / "audit_manifest.json").write_text(stable_json(run_manifest), encoding="utf-8")
        wrapper_path = generation_dir / "preflight_report.json"
        original_gate_hash = sha256_file(generation_dir / "audit_gate.json")
        wrapper["audit_artifact_sha256"] = original_gate_hash
        if mutate is not None:
            mutate(wrapper, report, gate, run_manifest, generation_dir)
        (generation_dir / "audit_report.json").write_text(stable_json(report), encoding="utf-8")
        (generation_dir / "audit_gate.json").write_text(stable_json(gate), encoding="utf-8")
        for name in ("audit_report.json", "audit_gate.json"):
            if run_manifest["artifacts"][name].get("bytes") is not True:
                run_manifest["artifacts"][name] = artifact(name)
        if wrapper["audit_artifact_sha256"] == original_gate_hash:
            wrapper["audit_artifact_sha256"] = sha256_file(generation_dir / "audit_gate.json")
        wrapper_path.write_text(stable_json(wrapper), encoding="utf-8")
        run_manifest["artifacts"]["preflight_report.json"] = artifact("preflight_report.json")
        (generation_dir / "audit_manifest.json").write_text(stable_json(run_manifest), encoding="utf-8")
        return source, wrapper_path

    def _rewrite_linked_prefix_replay(self, wrapper, report, gate, run_manifest, generation_dir,
                                      field, value):
        replay_path = generation_dir / "source_prefix_replay.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        if field == "source.commit":
            replay["source"]["commit"] = value
        elif field.startswith("producer."):
            replay["producer"][field.split(".", 1)[1]] = value
        else:
            replay[field] = value
        replay_bytes = (stable_json(replay) + "\n").encode("utf-8")
        replay_path.write_bytes(replay_bytes)
        # Keep the report and linked replay equally substituted. The consumer must
        # compare both against the generation/context identities, not merely each other.
        report["source_prefix_replay"] = copy.deepcopy(replay)
        generation_path = generation_dir / "manifest.json"
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        replay_identity = {"sha256": hashlib.sha256(replay_bytes).hexdigest(), "bytes": len(replay_bytes)}
        generation["artifacts"]["source_prefix_replay.json"] = replay_identity
        generation["source_prefix"]["replay_sha256"] = replay_identity["sha256"]
        generation["source_capacity_bound"]["source_prefix_replay_sha256"] = replay_identity["sha256"]
        capacity_bytes = (stable_json(generation["source_capacity_bound"]) + "\n").encode("utf-8")
        (generation_dir / "source_capacity_bound.json").write_bytes(capacity_bytes)
        generation["artifacts"]["source_capacity_bound.json"] = {
            "sha256": hashlib.sha256(capacity_bytes).hexdigest(), "bytes": len(capacity_bytes)}
        generation_path.write_text(stable_json(generation), encoding="utf-8")
        generation_hash = sha256_file(generation_path)
        wrapper["generation_manifest_sha256"] = generation_hash
        wrapper["generation_artifacts"] = copy.deepcopy(generation["artifacts"])
        report["generation_manifest_sha256"] = generation_hash
        report["generation_artifacts"] = copy.deepcopy(generation["artifacts"])
        report["capacity_bound"] = copy.deepcopy(generation["source_capacity_bound"])
        gate["generation_manifest_sha256"] = generation_hash
        gate["generation_artifacts"] = copy.deepcopy(generation["artifacts"])
        gate["capacity_bound"] = copy.deepcopy(generation["source_capacity_bound"])
        gate["source_prefix"] = copy.deepcopy(generation["source_prefix"])
        run_manifest["generation_manifest_sha256"] = generation_hash
        run_manifest["generation_artifacts"] = copy.deepcopy(generation["artifacts"])

    def test_closed_synthetic_v3_handoff_accepts_and_mutations_reject(self):
        mutations = {
            "unbound replay outcomes hash": lambda w, r, g, m, d: r["replay"].update(outcomes_sha256="e" * 64),
            "unbound replay records hash": lambda w, r, g, m, d: r["replay"].update(records_sha256="f" * 64),
            "prefix replay version substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(
                replay_version="dwr-task-v2-source-prefix-replay-v1"),
            "prefix replay source substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(
                source={**r["source_prefix_replay"]["source"], "commit": "other"}),
            "prefix replay config substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(config_sha256="f" * 64),
            "prefix replay producer substitution": lambda w, r, g, m, d: r["source_prefix_replay"]["producer"].update(
                {next(iter(r["source_prefix_replay"]["producer"])): "f" * 64}),
            "prefix replay context substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(context_sha256="e" * 64),
            "prefix replay count substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(source_accepted_record_count=4),
            "prefix replay prefix hash substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(
                source_record_prefix_sha256="d" * 64),
            "prefix replay status substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(status="NO_GO"),
            "prefix replay generator substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(generator_id="dwr-generator-v2"),
            "prefix replay source attempts substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(source_attempts=4),
            "prefix replay quota counts substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(
                source_quota_counts={"eq": 2, "neq": 1, "add_mod": 1}),
            "prefix replay quota closure substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(source_quota_closed=False),
            "prefix replay development attempts substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(development_attempts=1),
            "prefix replay final attempts substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(final_attempts=1),
            "prefix replay outcome bytes substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(source_outcome_prefix_bytes=1),
            "prefix replay outcome hash substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(source_outcome_prefix_sha256="e" * 64),
            "prefix replay record bytes substitution": lambda w, r, g, m, d: r["source_prefix_replay"].update(source_record_prefix_bytes=1),
            "rehash linked wrong replay version": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "replay_version", "dwr-task-v2-source-prefix-replay-v1"),
            "rehash linked wrong source": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "source.commit", "other"),
            "rehash linked wrong config": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "config_sha256", "f" * 64),
            "rehash linked wrong context": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "context_sha256", "e" * 64),
            "rehash linked wrong source producer": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "producer.generator_v3_sha256", "f" * 64),
            "rehash linked wrong count": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "source_accepted_record_count", 4),
            "rehash linked wrong prefix hash": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "source_record_prefix_sha256", "d" * 64),
            "rehash linked wrong source quota": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "source_quota_counts", {"eq": 2, "neq": 1, "add_mod": 1}),
            "rehash linked wrong replay status": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "status", "NO_GO"),
            "rehash linked wrong generator": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "generator_id", "dwr-generator-v2"),
            "rehash linked wrong source attempts": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "source_attempts", 4),
            "rehash linked wrong quota closure": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "source_quota_closed", False),
            "rehash linked wrong development attempts": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "development_attempts", 1),
            "rehash linked wrong final attempts": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "final_attempts", 1),
            "rehash linked wrong outcome prefix bytes": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "source_outcome_prefix_bytes", 1),
            "rehash linked wrong outcome prefix hash": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "source_outcome_prefix_sha256", "e" * 64),
            "rehash linked wrong record prefix bytes": lambda w, r, g, m, d: self._rewrite_linked_prefix_replay(
                w, r, g, m, d, "source_record_prefix_bytes", 1),
            "alternate gate basename": lambda w, r, g, m, d: w.update(audit_artifact_path=str(d / "alternate_gate.json")),
            "wrong gate hash with rehashed wrapper": lambda w, r, g, m, d: w.update(audit_artifact_sha256="f" * 64),
            "wrong generation hash wrapper": lambda w, r, g, m, d: w.update(generation_manifest_sha256="f" * 64),
            "wrong source commit wrapper": lambda w, r, g, m, d: w.update(source={**w["source"], "commit": "other"}),
            "wrong config identity wrapper": lambda w, r, g, m, d: w.update(config_sha256="f" * 64),
            "v2 report identity": lambda w, r, g, m, d: r.update(audit_version="dwr-task-v2-audit-v1"),
            "nonpassing report": lambda w, r, g, m, d: r.update(g2_status="NO_GO"),
            "unknown report field": lambda w, r, g, m, d: r.update(unregistered=True),
            "report generation identity": lambda w, r, g, m, d: r.update(generation_manifest_sha256="e" * 64),
            "wrong mode": lambda w, r, g, m, d: m.update(mode="full"),
            "wrong command": lambda w, r, g, m, d: m.update(command=["python", "-m", "other"]),
            "argv command mode substitution": lambda w, r, g, m, d: (m.update(argv=["--config", str(ROOT / "configs/tasks/constraint_v3.json"),
                "--mode", "full", "--data", str(d)]), m.update(command=[sys.executable, "-m", "dwr.tasks.audit_cli_v3", *m["argv"]])),
            "wrong generation directory": lambda w, r, g, m, d: m.update(generation_dir=str(d.parent)),
            "unknown audit field": lambda w, r, g, m, d: m.update(unregistered=True),
            "audit source substitution": lambda w, r, g, m, d: m.update(audit_source={**m["audit_source"], "commit": "other"}),
            "boolean exit code": lambda w, r, g, m, d: m.update(exit_code=False),
            "boolean artifact size": lambda w, r, g, m, d: m["artifacts"]["audit_gate.json"].update(bytes=True),
            "primary error": lambda w, r, g, m, d: m.update(error={"type": "Error"}),
            "finalization error": lambda w, r, g, m, d: m.update(finalization_errors=[{"type": "Error"}]),
            "dirty wrapper source": lambda w, r, g, m, d: w.update(source={**w["source"], "dirty": True}),
            "gate wrong version": lambda w, r, g, m, d: g.update(audit_version="dwr-task-v2-audit-v1"),
            "gate nonpassing status": lambda w, r, g, m, d: g.update(proposal_status="NO_GO"),
        }
        from dwr.tasks.stream_contract import resolve_stream
        for producer_field in resolve_stream("dwr-generator-v3").audit_producer_fields:
            mutations[f"producer field {producer_field}"] = lambda w, r, g, m, d, key=producer_field: m["producer"].update({key: "f" * 64})
        for producer_field in resolve_stream("dwr-generator-v3").generation_producer_fields:
            mutations[f"rehash linked generation producer field {producer_field}"] = (
                lambda w, r, g, m, d, key=producer_field: self._rewrite_linked_prefix_replay(
                    w, r, g, m, d, f"producer.{key}", "f" * 64))
        with tempfile.TemporaryDirectory() as temp:
            source, wrapper = self._accepted_handoff(Path(temp))
            with patch("dwr.tasks.manifest_v2.validate_manifest_v3"):
                result = validate_preflight_v3(wrapper, self.config, source, limits=self.limits)
            self.assertEqual(result["linkage_version"], "dwr-task-v3-consumed-preflight-v1")
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                source, wrapper = self._accepted_handoff(Path(temp), mutate)
                with patch("dwr.tasks.manifest_v2.validate_manifest_v3"), self.assertRaises((ValueError, FileNotFoundError)):
                    validate_preflight_v3(wrapper, self.config, source, limits=self.limits)

    def test_actual_v3_cli_producer_handoff_is_consumable(self):
        from dwr.tasks import audit_cli_v2
        from dwr.tasks.audit_cli_v3 import main as audit_main
        from dwr.tasks.stream_contract import resolve_stream
        with tempfile.TemporaryDirectory() as temp:
            source, wrapper = self._accepted_handoff(Path(temp))
            generation_dir = wrapper.parent
            report = json.loads((generation_dir / "audit_report.json").read_text(encoding="utf-8"))
            report["gate"] = json.loads((generation_dir / "audit_gate.json").read_text(encoding="utf-8"))
            for name in ("audit_manifest.json", "audit_report.json", "audit_gate.json", "preflight_report.json"):
                (generation_dir / name).unlink()
            config_path = ROOT / "configs/tasks/constraint_v3.json"
            with patch("dwr.tasks.audit_v2.run_audit_v3", return_value=report), \
                 patch.object(audit_cli_v2, "source_identity", return_value=source), \
                 patch("dwr.tasks.manifest_v2.validate_manifest_v3"):
                exit_code = audit_main(["--config", str(config_path), "--mode", "feasibility",
                                        "--data", str(generation_dir)])
            self.assertEqual(exit_code, 0)
            produced_wrapper = generation_dir / "preflight_report.json"
            self.assertTrue(produced_wrapper.is_file())
            with patch("dwr.tasks.manifest_v2.validate_manifest_v3"):
                consumed = validate_preflight_v3(produced_wrapper, self.config, source, limits=self.limits)
            self.assertEqual(consumed["linkage_version"], resolve_stream("dwr-generator-v3").consumed_preflight_version)
            terminal = json.loads((generation_dir / "audit_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(terminal["artifacts"]["preflight_report.json"]["sha256"],
                             sha256_file(produced_wrapper))

    def test_terminal_cap_downgrades_synthetic_pass_and_invalidates_wrapper(self):
        from dwr.tasks import audit_cli_v2
        from dwr.tasks.run_support import ResourceGuard
        with tempfile.TemporaryDirectory() as temp:
            source, wrapper = self._accepted_handoff(Path(temp))
            generation_dir = wrapper.parent
            report = json.loads((generation_dir / "audit_report.json").read_text(encoding="utf-8"))
            gate = json.loads((generation_dir / "audit_gate.json").read_text(encoding="utf-8"))
            manifest = json.loads((generation_dir / "audit_manifest.json").read_text(encoding="utf-8"))
            generation = json.loads((generation_dir / "manifest.json").read_text(encoding="utf-8"))
            now = [0.0]
            guard = ResourceGuard(0.5, 1024, clock=lambda: now[0], memory_reader=lambda: 0)
            original_atomic_json = audit_cli_v2.atomic_json
            advanced = [False]
            def charged_write(path, value):
                original_atomic_json(path, value)
                if Path(path).name == "audit_manifest.json" and value.get("status") == "complete" and not advanced[0]:
                    advanced[0] = True
                    now[0] = 1.0
            with patch("dwr.tasks.audit_cli_v2.atomic_json", side_effect=charged_write):
                error = audit_cli_v2._finalize_manifest(generation_dir, manifest, report, gate, guard,
                    generation, 0.0, None, [], charge_guard=True)
            final_report = json.loads((generation_dir / "audit_report.json").read_text(encoding="utf-8"))
            final_gate = json.loads((generation_dir / "audit_gate.json").read_text(encoding="utf-8"))
            final_manifest = json.loads((generation_dir / "audit_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(error["type"], "TimeoutError")
            self.assertEqual(final_report["status"], "incomplete")
            self.assertEqual(final_report["g2_status"], "not_assessed")
            self.assertEqual(final_report["proposal_status"], "NOT_ASSESSED")
            self.assertEqual(final_gate["status"], "incomplete")
            self.assertFalse((generation_dir / "preflight_report.json").exists())
            self.assertTrue((generation_dir / "preflight_report.invalidated.json").exists())
            self.assertTrue((generation_dir / "audit_report.provisional.json").exists())
            self.assertEqual(final_manifest["error"]["type"], "TimeoutError")
            self.assertEqual(final_manifest["accounting_end_boundary"],
                "post_first_terminal_manifest_write_resource_check")
            self.assertGreaterEqual(final_manifest["combined_generation_audit_cpu_seconds"],
                                    generation["generation_cpu_seconds"])
            self.assertEqual(final_manifest["combined_peak_resident_bytes"],
                             generation["peak_resident_bytes"])

    def test_failed_gate_downgrade_records_sidecar_and_preserves_primary_error(self):
        from dwr.tasks import audit_cli_v2
        from dwr.tasks.run_support import ResourceGuard, atomic_json
        with tempfile.TemporaryDirectory() as temp:
            source, wrapper = self._accepted_handoff(Path(temp))
            generation_dir = wrapper.parent
            report = json.loads((generation_dir / "audit_report.json").read_text(encoding="utf-8"))
            gate = json.loads((generation_dir / "audit_gate.json").read_text(encoding="utf-8"))
            manifest = json.loads((generation_dir / "audit_manifest.json").read_text(encoding="utf-8"))
            generation = json.loads((generation_dir / "manifest.json").read_text(encoding="utf-8"))
            now = [0.0]
            guard = ResourceGuard(0.5, 1024, clock=lambda: now[0], memory_reader=lambda: 0)
            advanced = [False]
            def failing_gate_write(path, value):
                if Path(path).name == "audit_gate.json" and value.get("status") == "incomplete":
                    raise OSError("injected gate downgrade fault")
                atomic_json(path, value)
                if Path(path).name == "audit_manifest.json" and value.get("status") == "complete" and not advanced[0]:
                    advanced[0] = True
                    now[0] = 1.0
            with patch("dwr.tasks.audit_cli_v2.atomic_json", side_effect=failing_gate_write):
                error = audit_cli_v2._finalize_manifest(generation_dir, manifest, report, gate, guard,
                    generation, 0.0, None, [], charge_guard=True)
            sidecar = json.loads((generation_dir / "terminal_failure.json").read_text(encoding="utf-8"))
            summary = json.loads((generation_dir / "audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(error["type"], "TimeoutError")
            self.assertEqual(sidecar["primary_error"]["type"], "TimeoutError")
            self.assertIn("OSError", [item["type"] for item in sidecar["finalization_errors"]])
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["error"]["type"], "TimeoutError")
            self.assertTrue((generation_dir / "audit_report.provisional.json").exists())
            with patch("dwr.tasks.manifest_v2.validate_manifest_v3"), self.assertRaisesRegex(
                    ValueError, "terminal failure sidecar"):
                validate_preflight_v3(wrapper, self.config, source, limits=self.limits)
