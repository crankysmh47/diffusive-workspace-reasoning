import unittest
from copy import deepcopy
from dataclasses import fields

from dwr.tasks.stream_contract import require_matching_config, resolve_stream
from dwr.tasks.config_v3 import load_config_v3
from pathlib import Path


class StreamContractTests(unittest.TestCase):
    def test_resolves_only_the_two_registered_streams(self):
        v2 = resolve_stream("dwr-generator-v2")
        v3 = resolve_stream("dwr-generator-v3")
        self.assertEqual(v2.config_version, "constraint-generation-v2")
        self.assertEqual(v2.seed_derivation, "sha256-prefix64-v2")
        self.assertEqual(v2.outcome_version, "dwr-task-outcome-v2")
        self.assertIsNone(v2.accepted_record_version)
        self.assertEqual(v2.manifest_version, "dwr-task-run-v2")
        self.assertEqual(v2.audit_version, "dwr-task-v2-audit-v1")
        self.assertIsNone(v2.audit_gate_version)
        self.assertEqual(v3.config_version, "constraint-generation-v3")
        self.assertEqual(v3.seed_derivation, "sha256-prefix64-v3")
        self.assertEqual(v3.outcome_version, "dwr-task-outcome-v3")
        self.assertEqual(v3.accepted_record_version, "dwr-task-accepted-record-v3")
        self.assertEqual(v3.manifest_version, "dwr-task-run-v3")
        self.assertEqual(v3.audit_gate_version, "dwr-task-v3-audit-gate-v1")
        self.assertEqual(v3.generation_producer_fields, (
            "generator_v2_sha256", "generator_v3_sha256", "config_v3_sha256",
            "dataset_v2_sha256", "replay_v2_sha256", "manifest_v2_sha256",
            "stream_contract_sha256", "cli_v2_sha256", "cli_v3_sha256",
            "canonical_sha256", "oracle_sha256", "schema_sha256", "solver_sha256",
            "run_support_sha256"))
        self.assertEqual(v2.audit_producer_fields, ("audit_v2_sha256", "audit_cli_v2_sha256"))
        self.assertEqual(v3.audit_producer_fields, (
            "audit_py_sha256", "audit_v2_sha256", "audit_cli_v2_sha256", "audit_cli_v3_sha256",
            "generator_v2_sha256", "generator_v3_sha256", "dataset_v2_sha256", "replay_v2_sha256",
            "manifest_v2_sha256", "config_v3_sha256", "stream_contract_sha256", "canonical_sha256",
            "oracle_sha256", "schema_sha256", "solver_sha256", "run_support_sha256"))
        self.assertEqual(tuple(field.name for field in fields(v3)), (
            "generator_id", "config_version", "seed_derivation", "outcome_version",
            "accepted_record_version", "manifest_version", "audit_version",
            "audit_run_manifest_version", "audit_gate_version", "preflight_version",
            "consumed_preflight_version", "candidate_fn", "generation_producer_fields",
            "audit_producer_fields"))

    def test_rejects_unknown_stream_id(self):
        with self.assertRaises(ValueError):
            resolve_stream("dwr-generator-v4")

    def test_rejects_config_with_substituted_rng_domain_or_seed_label(self):
        root = Path(__file__).resolve().parents[1]
        resolved = load_config_v3(root / "configs/tasks/constraint_v3.json").resolved
        contract = resolve_stream("dwr-generator-v3")
        for path in (("seed_derivation",), ("generation", "seed_derivation"),
                     ("generation", "candidate_rng_domain"), ("rng", "seed_material")):
            changed = deepcopy(resolved)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = "substituted-domain"
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "RNG identity"):
                require_matching_config(contract, changed)


if __name__ == "__main__":
    unittest.main()
