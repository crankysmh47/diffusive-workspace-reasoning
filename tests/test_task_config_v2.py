import copy
import json
import tempfile
import unittest
from pathlib import Path

from dwr.tasks.config_v2 import _same_typed_value, load_config_v2
from dwr.tasks.run_support import ConfigError, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tasks" / "constraint_v2.json"


class PinnedV2ConfigTests(unittest.TestCase):
    def test_registered_config_loads_with_exact_bytes_and_identity(self):
        loaded = load_config_v2(CONFIG)
        self.assertEqual(loaded.path, CONFIG.resolve())
        self.assertEqual(loaded.sha256, __import__("hashlib").sha256(CONFIG.read_bytes()).hexdigest())
        self.assertEqual(loaded.resolved["config_version"], "constraint-generation-v2")
        self.assertEqual(loaded.resolved["seed"], 20261007)
        self.assertEqual(loaded.resolved["generation"]["candidate_rng_domain"], "dwr-generator-v2")

    def test_v1_semantic_contract_sections_are_copied_exactly(self):
        v1 = json.loads((ROOT / "configs" / "tasks" / "constraint_v1.json").read_text(encoding="utf-8"))
        v2 = load_config_v2(CONFIG).resolved
        for key in ("grammar", "composition_partition", "solver", "oracle", "canonicalization", "audit_gate"):
            self.assertEqual(v2[key], v1[key])

    def test_closed_comparison_distinguishes_booleans_from_integers(self):
        self.assertFalse(_same_typed_value({"limit": True}, {"limit": 1}))
        self.assertFalse(_same_typed_value({"nested": {"x": 1, "extra": 2}}, {"nested": {"x": 1}}))
        self.assertTrue(_same_typed_value({"nested": [1, False]}, {"nested": [1, False]}))

    def test_unknown_missing_or_altered_nested_values_are_rejected(self):
        original = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = []
        value = copy.deepcopy(original); value["new"] = 1; mutations.append(value)
        value = copy.deepcopy(original); del value["seed"]; mutations.append(value)
        value = copy.deepcopy(original); value["grammar"]["domain"][0] = True; mutations.append(value)
        value = copy.deepcopy(original); value["generation"]["mode_quotas"]["feasibility"]["source_by_profile"]["eq"] += 1; mutations.append(value)
        value = copy.deepcopy(original); value["provenance"]["extra"] = "x"; mutations.append(value)
        for value in mutations:
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        load_config_v2(path)

    def test_baseline_bytes_are_pinned_and_utf8_bom_is_rejected(self):
        data = CONFIG.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bom.json"
            path.write_bytes(b"\xef\xbb\xbf" + data)
            with self.assertRaises((ConfigError, UnicodeError)):
                load_config_v2(path)

    def test_v1_loader_remains_closed_to_v2_contract(self):
        with self.assertRaises(ConfigError):
            load_config(CONFIG)


if __name__ == "__main__":
    unittest.main()
