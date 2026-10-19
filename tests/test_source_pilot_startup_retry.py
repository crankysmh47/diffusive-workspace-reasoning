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

    def test_memory_learning_and_unknown_key_changes_still_fail(self):
        for section, key, value in (("limits", "minimum_initial_free_mib", 14000),
                ("training", "optimizer_steps", 513), ("training", "unknown", True)):
            changed = copy.deepcopy(self.new)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                _validate_startup_retry_config(changed)

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
