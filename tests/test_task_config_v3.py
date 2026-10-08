import copy
import json
import tempfile
import unittest
from pathlib import Path

from dwr.tasks.config_v2 import _same_typed_value
from dwr.tasks.config_v3 import load_config_v3
from dwr.tasks.run_support import ConfigError


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "configs" / "tasks" / "constraint_v2.json"
V3 = ROOT / "configs" / "tasks" / "constraint_v3.json"


class PinnedV3ConfigTests(unittest.TestCase):
    def test_registered_config_loads_and_retains_frozen_v2_sections(self):
        loaded = load_config_v3(V3)
        v2 = json.loads(V2.read_text(encoding="utf-8"))
        for key in ("task_schema_version", "seed", "grammar", "composition_partition", "solver",
                    "oracle", "canonicalization", "audit_gate", "artifacts"):
            self.assertTrue(_same_typed_value(loaded.resolved[key], v2[key]), key)
        v2_generation = v2["generation"]
        v3_generation = loaded.resolved["generation"]
        removed_v2_proposal_keys = {"profile_sampling", "quota_check_order", "quota_skip_status",
                                    "quota_skip_consumes_candidate_index", "quota_skip_additional_rng_calls"}
        replaced_v2_identity_keys = {"candidate_rng_domain", "seed_derivation"}
        for key, value in v2_generation.items():
            if key not in removed_v2_proposal_keys | replaced_v2_identity_keys:
                self.assertIn(key, v3_generation, key)
                self.assertTrue(_same_typed_value(v3_generation[key], value), key)
        self.assertEqual(v3_generation["candidate_rng_domain"], "dwr-generator-v3")
        self.assertEqual(v3_generation["seed_derivation"], "sha256-prefix64-v3")
        self.assertEqual(loaded.resolved["config_version"], "constraint-generation-v3")
        self.assertEqual(loaded.resolved["seed_derivation"], "sha256-prefix64-v3")
        self.assertEqual(loaded.resolved["rng"]["seed_material"],
                         "dwr-generator-v3-domain+0x00+root-seed-u64be+0x00+partition-ascii+0x00+candidate-index-u64be")
        self.assertEqual(loaded.resolved["generation"]["profile_selection"], {
            "algorithm": "active-registered-order-minus-full-mod-index-before-n-v1",
            "empty_active_behavior": "raise-value-error-before-rng"})

    def test_source_neq_mechanics_match_registered_typed_literals(self):
        config = load_config_v3(V3).resolved["generation"]["source_neq_v3"]
        expected = {
            "n_min": 4, "n_max": 6, "given_min": 2, "given_max": "n_minus_2",
            "witness_order": "lexicographic-product-values-0-1-2",
            "require_all_three_values": True, "min_distinct_given_values": 2,
            "witness_selection": "uniform-rng-choice",
            "tuple_pool": "normalized-v1-neq-satisfied-by-witness-touching-unknown",
            "min_constraint_count": "2*(n-g)", "max_constraint_count": "min(8,pool_capacity)",
            "constraint_total_sampling": "uniform-inclusive",
            "tuple_sampling": "uniform-without-replacement", "empty_witness_reason": "no_neq_witness",
            "empty_total_reason": "no_feasible_neq_constraint_total",
        }
        self.assertTrue(_same_typed_value(config, expected))
        support = load_config_v3(V3).resolved["generation"]["feasibility_source_support_capacity_bound"]
        self.assertTrue(_same_typed_value(support, {
            "enabled": True, "top_k_cells": 10, "minimum_source_train_mass": 0.8,
            "source_cohort": "accepted-eligible-unique-source_train",
            "cell_helper": "dwr.tasks.audit._cell", "requires_source_quota_closure": True,
            "requires_source_prefix_replay": True, "target_attempts_at_boundary": 0}))

    def test_unknown_key_changed_type_changed_floor_and_raw_byte_drift_fail_closed(self):
        original = json.loads(V3.read_text(encoding="utf-8"))
        mutations = []
        value = copy.deepcopy(original); value["unexpected"] = 1; mutations.append(value)
        value = copy.deepcopy(original); value["generation"]["source_neq_v3"]["require_all_three_values"] = 1; mutations.append(value)
        value = copy.deepcopy(original); value["audit_gate"]["min_accepted_groups_per_partition"] += 1; mutations.append(value)
        with tempfile.TemporaryDirectory() as directory:
            for index, value in enumerate(mutations):
                path = Path(directory) / f"mutated-{index}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(index=index), self.assertRaises(ConfigError):
                    load_config_v3(path)
            byte_path = Path(directory) / "byte-drift.json"
            byte_path.write_bytes(V3.read_bytes() + b" ")
            with self.assertRaises(ConfigError):
                load_config_v3(byte_path)


if __name__ == "__main__":
    unittest.main()
