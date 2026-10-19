"""Regression checks for the environment-only, zero-model-work startup retry."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from dwr.experiments.source_pilot_registration import (
    CONFIG_PATH, CONFIG_SHA256, STARTUP_RETRY_CONFIG_PATH,
    STARTUP_RETRY_CONFIG_SHA256, _validate_startup_retry_config,
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

    def test_memory_learning_and_unknown_key_changes_still_fail(self):
        for section, key, value in (("limits", "minimum_initial_free_mib", 14000),
                ("training", "optimizer_steps", 513), ("training", "unknown", True)):
            changed = copy.deepcopy(self.new)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                _validate_startup_retry_config(changed)
