"""Closed config checks for the fixed v4 proposal fixture."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dwr.tasks.config_probe_v4 import load_probe_config


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "tasks" / "constraint_probe_v4.json"


class ProbeConfigV4Tests(unittest.TestCase):
    def test_approved_fixture_loads_with_registered_quota_shapes(self) -> None:
        config = load_probe_config(CONFIG_PATH)
        self.assertEqual(config.raw_config["config_version"],
                         "dwr-constraint-development-probe-v4-r2")
        self.assertEqual(config.raw_config["producer_hash_paths"][-5:], (
            "dwr/tasks/generator.py", "dwr/tasks/dataset.py", "dwr/tasks/verifier.py",
            "dwr/tasks/__init__.py", "dwr/__init__.py"))
        self.assertEqual(config.root_seed, 20261007)
        self.assertEqual(config.source_profiles, (("eq",), ("neq",), ("add_mod",)))
        self.assertEqual(config.development_profiles,
                         (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod")))
        self.assertEqual(config.strata["A"], (2, 8))
        self.assertEqual(config.strata["D"], (3, 8))
        self.assertEqual(sum(sum(cells.values()) for cells in config.source_quotas.values()), 1000)

    def test_duplicate_keys_and_nonfinite_constants_are_rejected(self) -> None:
        approved = CONFIG_PATH.read_text(encoding="utf-8")
        duplicate = approved.replace('"root_seed": 20261007,', '"root_seed": 20261007, "root_seed": 20261007,', 1)
        nonfinite = approved.replace('"root_seed": 20261007,', '"root_seed": NaN,', 1)
        for payload in (duplicate, nonfinite):
            with self.subTest(payload=payload[:40]), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.json"
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_probe_config(path)

    def test_unknown_fields_bool_counts_and_changed_registered_constants_are_rejected(self) -> None:
        original = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = []
        unknown = dict(original)
        unknown["unregistered"] = 1
        mutations.append(unknown)
        bool_seed = dict(original)
        bool_seed["root_seed"] = True
        mutations.append(bool_seed)
        altered_stratum = json.loads(json.dumps(original))
        altered_stratum["strata"][0]["given_count"] = True
        mutations.append(altered_stratum)
        mismatched_stratum = json.loads(json.dumps(original))
        mismatched_stratum["strata"][0]["constraint_count"] = 7
        mutations.append(mismatched_stratum)
        altered_quota = json.loads(json.dumps(original))
        altered_quota["quotas"]["source_accepted_by_profile_cell"]["eq"]["A"] += 1
        mutations.append(altered_quota)
        wrong_type = json.loads(json.dumps(original))
        wrong_type["rng"]["seed_version"] = "2"
        mutations.append(wrong_type)
        for value in mutations:
            with self.subTest(mutation=value.keys()), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises((TypeError, ValueError)):
                    load_probe_config(path)


if __name__ == "__main__":
    unittest.main()
