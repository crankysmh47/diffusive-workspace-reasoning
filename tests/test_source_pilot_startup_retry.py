"""Regression checks for the environment-only, zero-model-work startup retry."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from dwr.experiments.source_pilot_registration import (
    CONFIG_PATH, CONFIG_SHA256, STARTUP_RETRY_CONFIG_PATH,
    STARTUP_RETRY_CONFIG_SHA256, _validate_startup_retry_config,
    REGISTRATION_REPAIR_CONFIG_PATH, REGISTRATION_REPAIR_CONFIG_SHA256,
    TOKENIZER_REPAIR_CONFIG_PATH, TOKENIZER_REPAIR_CONFIG_SHA256,
)


class StartupRetryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.old = json.loads((self.root / CONFIG_PATH).read_text(encoding="utf-8"))
        self.new = json.loads((self.root / STARTUP_RETRY_CONFIG_PATH).read_text(encoding="utf-8"))

    def test_only_run_identity_changes_and_both_raw_pins_hold(self):
        self.assertEqual({k for k in self.old if self.old[k] != self.new[k]},
            {"config_version", "output_dir"})
        for path, digest in ((CONFIG_PATH, CONFIG_SHA256),
                (STARTUP_RETRY_CONFIG_PATH, STARTUP_RETRY_CONFIG_SHA256)):
            self.assertEqual(hashlib.sha256((self.root / path).read_bytes()).hexdigest(), digest)
        self.assertEqual(_validate_startup_retry_config(self.new), self.new)
        repair_raw = (self.root / REGISTRATION_REPAIR_CONFIG_PATH).read_bytes()
        repair = json.loads(repair_raw)
        self.assertEqual(hashlib.sha256(repair_raw).hexdigest(), REGISTRATION_REPAIR_CONFIG_SHA256)
        self.assertEqual({k for k in self.old if self.old[k] != repair[k]}, {"config_version", "output_dir"})
        self.assertEqual(_validate_startup_retry_config(repair), repair)
        raw = (self.root / TOKENIZER_REPAIR_CONFIG_PATH).read_bytes()
        tokenizer_repair = json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), TOKENIZER_REPAIR_CONFIG_SHA256)
        self.assertEqual({k for k in self.old if self.old[k] != tokenizer_repair[k]}, {"config_version", "output_dir"})
        self.assertEqual(_validate_startup_retry_config(tokenizer_repair), tokenizer_repair)

    def test_memory_learning_and_unknown_key_changes_still_fail(self):
        for section, key, value in (("limits", "minimum_initial_free_mib", 14000),
                ("training", "optimizer_steps", 513), ("training", "unknown", True)):
            changed = copy.deepcopy(self.new)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                _validate_startup_retry_config(changed)

    def test_captured_v2_startup_failure_accepts_only_original_or_pinned_watchdog_successor(self):
        from dwr.experiments.source_pilot_registration import _verify_v2_startup_failure
        import tempfile
        source = self.root / "runs/source-validation-pilot-20261019-startup-retry-v2"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = root / "configs/models/source_validation_pilot_v2_startup_failure.json"
            proof.parent.mkdir(parents=True)
            proof.write_bytes((self.root / "configs/models/source_validation_pilot_v2_startup_failure.json").read_bytes())
            run = root / "runs/source-validation-pilot-20261019-startup-retry-v2"
            import shutil
            shutil.copytree(source, run)
            _verify_v2_startup_failure(root)
            status = json.loads((run / "status.json").read_text(encoding="utf-8"))
            status["error"] = "unrelated drift"
            (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "known watchdog successor"):
                _verify_v2_startup_failure(root)
            (run / "status.json").write_bytes((source / "status.json").read_bytes())
            receipt = json.loads((run / "watchdog-failure.json").read_text(encoding="utf-8"))
            receipt["complete"] = False
            (run / "watchdog-failure.json").write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "known watchdog successor"):
                _verify_v2_startup_failure(root)

    def test_stale_watchdog_preserves_terminal_failure_but_live_owner_still_enforced(self):
        from dwr.experiments import source_pilot_watchdog as watchdog
        dead = {"pid": 1, "creation_time": "old", "image": "python"}
        with patch.object(watchdog, "_same_process", return_value=False):
            self.assertFalse(watchdog._has_live_owned_process(dead, None, dead))
        with patch.object(watchdog, "_same_process", side_effect=lambda ref: ref is dead):
            self.assertTrue(watchdog._has_live_owned_process(dead, None, None))
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = {"state": "operational_failure", "error": "captured failure"}
            status_path = root / "status.json"
            status_path.write_bytes(watchdog._canonical(status) + b"\n")
            before = status_path.read_bytes()
            with patch.object(watchdog, "_deadline", return_value=(0.0, None, None)), \
                    patch.object(watchdog, "_owned_references", return_value=(None, None, None)), \
                    patch.object(watchdog.time, "time", return_value=1.0):
                watchdog.watch(root)
            self.assertEqual(status_path.read_bytes(), before)
            self.assertFalse((root / "watchdog-failure.json").exists())

    def test_watchdog_deadlines_are_capped_by_immutable_native_anchors(self):
        from dwr.experiments import source_pilot_watchdog as watchdog
        import tempfile
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(watchdog._canonical(value) + b"\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "manifest.initial.json", {"started_utc": "2026-01-01T00:00:00Z",
                "settings": {"original_deadlines": {"matrix_watchdog_epoch": 5000.0}}})
            arm = root / "arms/K0"
            write(arm / "manifest.initial.json", {"started_utc": "2026-01-01T00:01:00Z",
                "settings": {"original_deadlines": {"arm_watchdog_epoch": 4000.0}}})
            config = {"limits": {"matrix_external_watchdog_seconds": 16260,
                "arm_external_watchdog_seconds": 5460}, "arms": [{"arm_id": "K0"}]}
            matrix, arm_deadline, arm_id = watchdog._deadline(root, config)
            self.assertEqual((matrix, arm_deadline, arm_id), (5000.0, 4000.0, "K0"))
            write(root / "manifest.initial.json", {"started_utc": "2026-01-01T00:00:00Z",
                "settings": {"original_deadlines": {"matrix_watchdog_epoch": "not-an-epoch"}}})
            with self.assertRaisesRegex(ValueError, "immutable watchdog deadline"):
                watchdog._deadline(root, config)

    def test_worker_uses_the_same_registration_router_as_supervisor(self):
        from dwr.experiments import source_pilot_job as job
        fields = ("matrix_root arm_root segment_dir arm_id arm_index segment_index nonce config_sha256 "
            "cohort_sha256 source_commit physical_source_receipt physical_source_receipt_sha256 "
            "runtime_receipt runtime_receipt_sha256 matrix_initial_manifest_sha256 arm_initial_manifest_sha256 "
            "matrix_event_prefix arm_event_prefix segment_initial_manifest_sha256 segment_event_prefix "
            "matrix_clock arm_clock deadlines supervisor launcher resume_descriptor").split()
        descriptor = dict.fromkeys(fields)
        descriptor["schema_version"] = "dwr-source-validation-pilot-worker-descriptor-v1"
        with patch.object(job, "_read_descriptor_file", return_value=(self.root / "descriptor.json", descriptor)), \
                patch.object(job, "_registration", side_effect=RuntimeError("registration selected")) as selected:
            with self.assertRaisesRegex(RuntimeError, "registration selected"):
                job.read_worker_authorization(self.root, "descriptor.json", "a" * 64)
            selected.assert_called_once_with(self.root)

    def test_full_vocabulary_includes_added_tokens_without_changing_eos(self):
        from dwr.experiments.source_pilot import _tokenizer_vocabulary_size
        class Tokenizer:
            vocab_size = 2
            eos_token_id = 1
            def get_vocab(self):
                return {"base0": 0, "eos": 1, "added": 2}
        tokenizer = Tokenizer()
        self.assertEqual(_tokenizer_vocabulary_size(tokenizer), 3)
        self.assertEqual(tokenizer.vocab_size, 2)
        self.assertEqual(tokenizer.eos_token_id, 1)
