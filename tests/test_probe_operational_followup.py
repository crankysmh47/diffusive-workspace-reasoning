"""Closed operational admission checks; fixtures only, never a production run."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dwr.tasks.config_probe_v4 import load_probe_config
from dwr.tasks.probe_v4 import (
    ARTIFACTS, PROBE_ID, _FixturePolicy, _initial_manifest, _verify_initial_binding,
    _verify_terminal_report, _run_controller, run_fixture_probe,
)
from dwr.tasks.run_support import ResourceGuard, sha256_file
from dwr.tasks.audit_probe_v4 import audit_run
from tests.test_probe_driver_v4 import _fixed_semantic_fixture_provider


ROOT = Path(__file__).parents[1]
OLD_CONFIG = ROOT / "configs" / "tasks" / "constraint_probe_v4.json"
NEW_ID = "dwr-constraint-development-probe-cpu-followup-v1"


def _followup_bytes() -> bytes:
    raw = OLD_CONFIG.read_bytes()
    old_id = b'"config_version": "dwr-constraint-development-probe-v4-r2"'
    new_id = f'"config_version": "{NEW_ID}"'.encode("ascii")
    old_cap = b'"combined_active_wall_seconds_generation_replay_audit_finalization": 900'
    new_cap = b'"combined_active_wall_seconds_generation_replay_audit_finalization": 1800'
    assert raw.count(old_id) == raw.count(old_cap) == 1
    return raw.replace(old_id, new_id, 1).replace(old_cap, new_cap, 1)


def _diff_paths(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        paths = []
        for key in left.keys() | right.keys():
            paths.extend(_diff_paths(left.get(key), right.get(key), f"{prefix}.{key}".strip(".")))
        return paths
    if left != right:
        return [prefix]
    return []


class ProbeOperationalFollowupTests(unittest.TestCase):
    def test_admitted_config_changes_only_registered_paths(self):
        original = json.loads(OLD_CONFIG.read_text(encoding="utf-8"))
        changed = json.loads(_followup_bytes().decode("utf-8"))
        self.assertEqual(set(_diff_paths(original, changed)), {
            "config_version",
            "runtime.combined_active_wall_seconds_generation_replay_audit_finalization",
        })
        self.assertEqual(original["runtime"]["combined_active_wall_seconds_generation_replay_audit_finalization"], 900)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "followup.json"
            path.write_bytes(_followup_bytes())
            config = load_probe_config(path)
            self.assertEqual(config.config_version, NEW_ID)
            self.assertEqual(config.wall_seconds, 1800)
            self.assertEqual(config.config_sha256, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_only_exact_registered_bytes_are_admitted(self):
        valid = _followup_bytes()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "candidate.json"
            for label, candidate in (
                ("byte", valid + b" "),
                ("unknown", valid.replace(b'"config_version":', b'"unknown": 1, "config_version":', 1)),
                ("bool-budget", valid.replace(b'": 1800', b'": true', 1)),
                ("duplicate", valid.replace(b'"root_seed": 20261007,',
                    b'"root_seed": 20261007, "root_seed": 20261007,', 1)),
                ("nonfinite", valid.replace(b'"root_seed": 20261007,', b'"root_seed": NaN,', 1)),
            ):
                with self.subTest(candidate=label):
                    path.write_bytes(candidate)
                    with self.assertRaises((ValueError, TypeError)):
                        load_probe_config(path)

    def test_original_and_followup_policy_identities_remain_distinct(self):
        original = load_probe_config(OLD_CONFIG)
        self.assertEqual(original.config_version, "dwr-constraint-development-probe-v4-r2")
        self.assertEqual(original.wall_seconds, 900)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "followup.json"
            path.write_bytes(_followup_bytes())
            followup = load_probe_config(path)
        self.assertNotEqual(original.config_sha256, followup.config_sha256)
        self.assertEqual(original.generator_id, followup.generator_id)
        self.assertEqual(original.source_profiles, followup.source_profiles)
        self.assertEqual(original.development_profiles, followup.development_profiles)

    def test_initial_manifest_binds_each_identity_and_rejects_cross_config(self):
        old = load_probe_config(OLD_CONFIG)
        with tempfile.TemporaryDirectory() as temp:
            new_path = Path(temp) / "followup.json"
            new_path.write_bytes(_followup_bytes())
            new = load_probe_config(new_path)
            source = {"commit": "a" * 40, "dirty": False,
                      "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(),
                      "checkout_root": str(ROOT)}
            producer = [{"path": item, "sha256": "b" * 64}
                        for item in new.raw_config["producer_hash_paths"]]
            manifest = _initial_manifest(new, new_path, Path(temp) / "out", ["fixture"],
                                         ROOT, producer, source)
            self.assertEqual(manifest["probe_id"], NEW_ID)
            self.assertNotEqual(manifest["probe_id"], PROBE_ID)
            self.assertEqual(manifest["limits"]["active_wall_seconds"], 1800)
            _verify_initial_binding(manifest, new, source, producer, new_path)
            with self.assertRaises(ValueError):
                _verify_initial_binding(manifest, old, source, producer, new_path)

    def test_driver_default_guard_uses_policy_without_restarting_elapsed_clock(self):
        for config_path, expected_cap in (
            (OLD_CONFIG, 900),
            (ROOT / "configs" / "tasks" / "constraint_probe_cpu_followup.json", 1800),
        ):
            with self.subTest(expected_cap=expected_cap):
                clock = {"now": 40.0}
                captured = []

                def make_guard(wall, memory):
                    guard = ResourceGuard(wall, memory, clock=lambda: clock["now"], memory_reader=lambda: 1024)
                    captured.append(guard)
                    return guard

                def stop_after_guard(*args, **kwargs):
                    clock["now"] = 45.0
                    raise RuntimeError("stop before candidate work")

                with tempfile.TemporaryDirectory(dir=ROOT / "data") as temp:
                    output = Path(temp) / "not-created"
                    with patch("dwr.tasks.probe_v4.ResourceGuard", side_effect=make_guard), \
                         patch("dwr.tasks.probe_v4._producer_identity", side_effect=stop_after_guard):
                        with self.assertRaisesRegex(RuntimeError, "stop before candidate work"):
                            _run_controller(config_path, output)
                self.assertEqual([guard.wall_seconds for guard in captured], [expected_cap])
                self.assertEqual(captured[0].elapsed, 5.0)

    def test_audit_default_guard_uses_followup_cap(self):
        followup_path = ROOT / "configs" / "tasks" / "constraint_probe_cpu_followup.json"
        captured = []

        class ProbeGuard(ResourceGuard):
            def __init__(self, wall, memory):
                super().__init__(wall, memory, memory_reader=lambda: 1024)
                captured.append(self)

        with tempfile.TemporaryDirectory(dir=ROOT / "data") as temp:
            output = Path(temp)
            initial = {"config_path": str(followup_path.resolve()),
                       "config_sha256": hashlib.sha256(_followup_bytes()).hexdigest(),
                       "output_dir": str(output)}
            (output / ARTIFACTS[0]).write_text(json.dumps(initial), encoding="utf-8")
            for name in ARTIFACTS[1:]:
                payload = {"status": "NOT_ASSESSED", "error": None}
                (output / name).write_text(json.dumps(payload), encoding="utf-8")
            with patch("dwr.tasks.probe_v4._source_identity", return_value={}), \
                 patch("dwr.tasks.probe_v4._producer_identity", return_value=[]), \
                 patch("dwr.tasks.probe_v4._verify_initial_binding"), \
                 patch("dwr.tasks.probe_v4._verify_source_closure"), \
                 patch("dwr.tasks.probe_v4._verify_replay_document"), \
                 patch("dwr.tasks.probe_v4._verify_audit_document"), \
                 patch("dwr.tasks.audit_probe_v4._verify_source_terminal_links"), \
                 patch("dwr.tasks.probe_v4._verify_terminal_report"), \
                 patch("dwr.tasks.probe_v4._verify_terminal_manifest"), \
                 patch("dwr.tasks.probe_v4._read_jsonl", return_value=[]), \
                 patch("dwr.tasks.audit_probe_v4._verify_reconstructed_source_prefix"), \
                 patch("dwr.tasks.run_support.ResourceGuard", ProbeGuard):
                audit_run(output, load_probe_config(followup_path))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].wall_seconds, 1800)

    def test_terminal_pass_at_followup_cap_is_rejected_and_timeout_stays_operational(self):
        followup_path = ROOT / "configs" / "tasks" / "constraint_probe_cpu_followup.json"
        config = load_probe_config(followup_path)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "pass-fixture"
            run_fixture_probe(followup_path, output, fixture_id="followup-pass-boundary",
                              _proposal_provider=_fixed_semantic_fixture_provider,
                              _guard=ResourceGuard(5000, 1024, memory_reader=lambda: 1024))
            report = json.loads((output / ARTIFACTS[6]).read_text(encoding="utf-8"))
            initial = json.loads((output / ARTIFACTS[0]).read_text(encoding="utf-8"))
            self.assertEqual(initial["limits"]["active_wall_seconds"], 1800)
            closure = json.loads((output / ARTIFACTS[3]).read_text(encoding="utf-8"))
            replay = json.loads((output / ARTIFACTS[4]).read_text(encoding="utf-8"))
            audit = json.loads((output / ARTIFACTS[5]).read_text(encoding="utf-8"))
            replay["artifact_sha256"] = sha256_file(output / ARTIFACTS[4])
            audit["artifact_sha256"] = sha256_file(output / ARTIFACTS[5])
            report["resource_snapshot"]["wall_seconds"] = 1800.0
            with self.assertRaisesRegex(ValueError, "resource accounting"):
                _verify_terminal_report(report, config, initial["source"],
                    sha256_file(output / ARTIFACTS[0]), sha256_file(output / ARTIFACTS[3]),
                    closure, replay, audit, fixture_policy=_FixturePolicy("followup-pass-boundary"))

        clock = {"now": 0.0}
        guard = ResourceGuard(1800, 1024, clock=lambda: clock["now"], memory_reader=lambda: 1024)

        def provider(current_config, partition, index, active):
            outcome = _fixed_semantic_fixture_provider(current_config, partition, index, active)
            if partition == "development" and index == 11:
                clock["now"] = 1800.0
            return outcome

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "fixture"
            with self.assertRaises(ValueError):
                run_fixture_probe(followup_path, output, fixture_id="followup-clock-boundary",
                                  _proposal_provider=provider, _guard=guard)
            report = json.loads((output / "probe_report.json").read_text(encoding="utf-8"))
            terminal = json.loads((output / "probe_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(report["reason"], "NOT_ASSESSED_RESOURCE_CAP")
            self.assertEqual(terminal["error"]["type"], "TimeoutError")
            self.assertEqual(terminal["limits"]["active_wall_seconds"], 1800)



if __name__ == "__main__":
    unittest.main()
