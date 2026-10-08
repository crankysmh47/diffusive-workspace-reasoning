import tempfile
import unittest
from pathlib import Path

from dwr.tasks.config_v3 import load_config_v3
from dwr.tasks.dataset_v2 import DatasetLimits, validate_preflight_v3
from dwr.tasks.run_support import ResourceGuard, stable_json

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
