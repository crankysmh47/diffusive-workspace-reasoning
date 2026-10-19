"""Controller contracts for source-pilot worker authorization and arm audits."""
from __future__ import annotations

import unittest
import tempfile
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _skip_if_ml_dependencies_absent(testcase, *module_names: str) -> None:
    import importlib

    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name:
                testcase.skipTest(f"optional {module_name} dependency is not installed")
            raise


def _metrics(vsr: float, ce: float | None = None, *, add: float = 0.5,
             eq: float | None = 0.5, neq: float | None = 0.5) -> dict:
    primitive = {
        "add_mod": {"rate": add},
        "eq": {"rate": eq},
        "neq": {"rate": neq},
    }
    return {"vsr": vsr,
            "ce": None if ce is None else {"token_weighted_mean": ce},
            "primitive": primitive}


def _arm(arm_id: str, final: dict, *, initial: dict | None = None,
         diagnostics: dict | None = None, learning: bool | None = None) -> dict:
    shared = {"source": "1" * 64, "data": "2" * 64, "model": "3" * 64,
        "tokenizer": "4" * 64, "schedule": "5" * 64}
    shared["config"] = {"K0": "a", "K1": "b", "K4": "c"}[arm_id] * 64
    return {
        "schema_version": "dwr-source-pilot-audited-arm-v1",
        "arm_id": arm_id,
        "identities": shared,
        "progress": {"optimizer_step": 512},
        "initial_state_sha256": "a" * 64,
        "initial": _metrics(0.1, 2.0) if initial is None else initial,
        "final": final,
        "diagnostics": {} if diagnostics is None else diagnostics,
        "learning_gate_passed": learning,
        "pairing": [{"optimizer_step": step, "example_id": f"ex-{step}",
            "batch_sha256": f"{step:064x}", "noise_sha256": f"{step + 1000:064x}"} for step in range(1, 513)],
        "counts": {}, "costs": {}, "evidence": {},
    }


def _passing_arms() -> dict:
    return {
        "K0": _arm("K0", _metrics(0.2)),
        "K1": _arm("K1", _metrics(0.25)),
        "K4": _arm("K4", _metrics(0.4, 0.8), diagnostics={
            "zero": _metrics(0.0), "noise": _metrics(0.1), "shuffle": _metrics(0.2)}, learning=True),
    }


def _matrix_config(*, steps: int) -> dict:
    return {"training": {"optimizer_steps": steps}, "evaluation": {
        "learning_gate": {
            "K4_final_vsr_min": 0.2,
            "K4_final_minus_initial_vsr_min": 0.1,
            "K4_final_valid_each_represented_primitive_min": 0.2,
            "K4_final_ce_max_ratio": 0.5,
        },
        "workspace_screen": {
            "K4_minus_trained_K0_final_vsr_min": 0.1,
            "K4_unablated_minus_shuffled_final_vsr_min": 0.1,
        },
    }}


class SourcePilotMatrixDecisionTests(unittest.TestCase):
    def test_matrix_decision_accepts_absent_validated_primitive_as_learning_nogo(self):
        from dwr.experiments.source_pilot import matrix_decision

        arms = _passing_arms()
        arms["K4"]["final"]["primitive"] = {
            "eq": {"valid": 1, "denominator": 1, "rate": 1.0},
            "neq": {"valid": 0, "denominator": 0, "rate": None},
            "add_mod": {"valid": 0, "denominator": 0, "rate": None},
        }
        arms["K4"]["learning_gate_passed"] = False
        result = matrix_decision(_matrix_config(steps=512), arms)
        self.assertEqual(result["status"], "NO_GO_SOURCE_VALIDATION_PILOT")
        self.assertFalse(result["learning_gate_passed"])

        arms["K4"]["final"]["primitive"]["neq"] = {"valid": 0, "denominator": 1, "rate": None}
        with self.assertRaisesRegex(ValueError, "primitive counts and rates"):
            matrix_decision(_matrix_config(steps=512), arms)

    def test_k4_learning_miss_has_registered_nogo_even_if_specificity_passes(self):
        from dwr.experiments.source_pilot import matrix_decision

        config = {
            "training": {"optimizer_steps": 512},
            "evaluation": {
                "learning_gate": {
                    "K4_final_vsr_min": 0.2,
                    "K4_final_minus_initial_vsr_min": 0.1,
                    "K4_final_valid_each_represented_primitive_min": 0.2,
                    "K4_final_ce_max_ratio": 0.5,
                },
                "workspace_screen": {
                    "K4_minus_trained_K0_final_vsr_min": 0.1,
                    "K4_unablated_minus_shuffled_final_vsr_min": 0.1,
                },
            }
        }
        k0 = _arm("K0", _metrics(0.2))
        k1 = _arm("K1", _metrics(0.2))
        k4 = _arm("K4", _metrics(0.15, 0.8), diagnostics={"zero": _metrics(0.0),
            "noise": _metrics(0.0), "shuffle": _metrics(0.0)}, learning=False)

        result = matrix_decision(config, {"K0": k0, "K1": k1, "K4": k4})

        self.assertEqual(result["status"], "NO_GO_SOURCE_VALIDATION_PILOT")
        self.assertFalse(result["learning_gate_passed"])
        self.assertFalse(result["workspace_screen_passed"])

    def test_learning_pass_but_missing_shuffle_floor_has_specificity_nogo(self):
        from dwr.experiments.source_pilot import matrix_decision

        config = {
            "training": {"optimizer_steps": 512},
            "evaluation": {
                "learning_gate": {
                    "K4_final_vsr_min": 0.2,
                    "K4_final_minus_initial_vsr_min": 0.1,
                    "K4_final_valid_each_represented_primitive_min": 0.2,
                    "K4_final_ce_max_ratio": 0.5,
                },
                "workspace_screen": {
                    "K4_minus_trained_K0_final_vsr_min": 0.1,
                    "K4_unablated_minus_shuffled_final_vsr_min": 0.1,
                },
            }
        }
        k0 = _arm("K0", _metrics(0.4))
        k1 = _arm("K1", _metrics(0.3))
        k4 = _arm("K4", _metrics(0.4, 0.8),
            diagnostics={"zero": _metrics(0.0), "noise": _metrics(0.0),
                         "shuffle": _metrics(0.35)}, learning=True)

        result = matrix_decision(config, {"K0": k0, "K1": k1, "K4": k4})

        self.assertEqual(result["status"], "NO_GO_WORKSPACE_SPECIFICITY_PILOT")
        self.assertTrue(result["learning_gate_passed"])
        self.assertFalse(result["workspace_screen_passed"])

    def test_both_registered_screens_pass_only_for_exact_three_audited_arms(self):
        from dwr.experiments.source_pilot import matrix_decision

        config = {
            "training": {"optimizer_steps": 512},
            "evaluation": {
                "learning_gate": {
                    "K4_final_vsr_min": 0.2,
                    "K4_final_minus_initial_vsr_min": 0.1,
                    "K4_final_valid_each_represented_primitive_min": 0.2,
                    "K4_final_ce_max_ratio": 0.5,
                },
                "workspace_screen": {
                    "K4_minus_trained_K0_final_vsr_min": 0.1,
                    "K4_unablated_minus_shuffled_final_vsr_min": 0.1,
                },
            }
        }
        arms = {
            "K0": _arm("K0", _metrics(0.2)),
            "K1": _arm("K1", _metrics(0.25)),
            "K4": _arm("K4", _metrics(0.4, 0.8), diagnostics={
                "zero": _metrics(0.0), "noise": _metrics(0.1), "shuffle": _metrics(0.2)}, learning=True),
        }

        result = matrix_decision(config, arms)

        self.assertEqual(result["status"], "PASS_SOURCE_VALIDATION_PILOT")
        self.assertTrue(result["learning_gate_passed"])
        self.assertTrue(result["workspace_screen_passed"])
        with self.assertRaises(ValueError):
            matrix_decision(config, {"K0": arms["K0"], "K4": arms["K4"]})

    def test_rejects_unpaired_initial_module_bytes_across_arms(self):
        from dwr.experiments.source_pilot import matrix_decision

        config = {
            "training": {"optimizer_steps": 512},
            "evaluation": {
                "learning_gate": {
                    "K4_final_vsr_min": 0.2,
                    "K4_final_minus_initial_vsr_min": 0.1,
                    "K4_final_valid_each_represented_primitive_min": 0.2,
                    "K4_final_ce_max_ratio": 0.5,
                },
                "workspace_screen": {
                    "K4_minus_trained_K0_final_vsr_min": 0.1,
                    "K4_unablated_minus_shuffled_final_vsr_min": 0.1,
                },
            }
        }
        arms = {
            "K0": _arm("K0", _metrics(0.2)),
            "K1": _arm("K1", _metrics(0.25)),
            "K4": _arm("K4", _metrics(0.4, 0.8), diagnostics={
                "zero": _metrics(0.0), "noise": _metrics(0.1), "shuffle": _metrics(0.2)}, learning=True),
        }
        arms["K1"]["initial_state_sha256"] = "b" * 64

        with self.assertRaisesRegex(ValueError, "initial module"):
            matrix_decision(config, arms)

    def test_rejects_unmatched_ordered_batch_and_noise_histories(self):
        from dwr.experiments.source_pilot import matrix_decision

        config = _matrix_config(steps=512)
        for field, value in (("example_id", "other-example"),
                ("batch_sha256", "e" * 64), ("noise_sha256", "f" * 64)):
            with self.subTest(field=field):
                arms = _passing_arms()
                arms["K1"]["pairing"][17][field] = value
                with self.assertRaisesRegex(ValueError, "paired|matched|history"):
                    matrix_decision(config, arms)

    def test_rejects_changed_shared_identity_but_keeps_arm_config_distinct(self):
        from dwr.experiments.source_pilot import matrix_decision

        arms = _passing_arms()
        arms["K4"]["identities"]["model"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "shared|identity|identit"):
            matrix_decision(_matrix_config(steps=512), arms)

    def test_finite_zero_final_ce_passes_but_initial_ce_must_be_positive(self):
        from dwr.experiments.source_pilot import _recompute_learning_gate

        config = {"evaluation": {"learning_gate": {
            "K4_final_vsr_min": 0.2,
            "K4_final_minus_initial_vsr_min": 0.1,
            "K4_final_valid_each_represented_primitive_min": 0.2,
            "K4_final_ce_max_ratio": 0.5,
        }}}
        initial = _metrics(0.1, 2.0)
        final = _metrics(0.3, 0.0)
        self.assertTrue(_recompute_learning_gate(config, initial, final))
        with self.assertRaisesRegex(ValueError, "initial CE"):
            _recompute_learning_gate(config, _metrics(0.1, 0.0), final)

    def test_zero_denominator_primitive_is_learning_miss_but_positive_denominator_must_be_valid(self):
        from dwr.experiments.source_pilot import _recompute_learning_gate

        config = {"evaluation": {"learning_gate": {
            "K4_final_vsr_min": 0.2,
            "K4_final_minus_initial_vsr_min": 0.1,
            "K4_final_valid_each_represented_primitive_min": 0.2,
            "K4_final_ce_max_ratio": 0.5,
        }}}
        initial, final = _metrics(0.1, 2.0), _metrics(0.3, 0.5)
        final["primitive"]["neq"] = {"valid": 0, "denominator": 0, "rate": None}
        self.assertFalse(_recompute_learning_gate(config, initial, final))

        final["primitive"]["neq"] = {"valid": 0, "denominator": 1, "rate": None}
        with self.assertRaisesRegex(ValueError, "primitive rates|primitive counts"):
            _recompute_learning_gate(config, initial, final)

        final["primitive"]["neq"] = {"valid": 2, "denominator": 1, "rate": 2.0}
        with self.assertRaisesRegex(ValueError, "primitive rates|primitive counts"):
            _recompute_learning_gate(config, initial, final)


class _Tokenizer:
    eos_token_id = 151643
    vocab_size = 151665

    def __init__(self, decoded: str = "answer"):
        self.decoded = decoded
        self.calls = []

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        self.calls.append((list(token_ids), skip_special_tokens, clean_up_tokenization_spaces))
        return self.decoded


class SourcePilotTokenAuditTests(unittest.TestCase):
    def test_redecodes_using_model_vocabulary_and_strips_only_terminal_eos(self):
        from dwr.experiments.source_pilot import _verify_generated_text

        tokenizer = _Tokenizer()
        envelope = {
            "generated_token_ids": [42, 151643], "eos_token_id": 151643,
            "vocab_size": 151936,
            "score": {"answer_text": "answer", "termination": "eos", "generated_tokens": 2},
        }

        _verify_generated_text(envelope, tokenizer, model_vocab_size=151936)

        self.assertEqual(tokenizer.calls, [([42], False, False)])

    def test_scaled_and_production_token_caps_are_exact_and_interior_eos_fails(self):
        from dwr.experiments.source_pilot import _verify_generated_text

        tokenizer = _Tokenizer()
        capped = {"generated_token_ids": [42] * 8, "eos_token_id": tokenizer.eos_token_id,
            "vocab_size": 151936, "score": {"answer_text": "answer", "termination": "token_cap",
                "generated_tokens": 8}}
        _verify_generated_text(capped, tokenizer, model_vocab_size=151936, token_cap=8)
        with self.assertRaisesRegex(ValueError, "termination"):
            _verify_generated_text(capped, tokenizer, model_vocab_size=151936, token_cap=128)
        bad_ids = [42] * 8; bad_ids[3] = tokenizer.eos_token_id
        capped["generated_token_ids"] = bad_ids
        with self.assertRaisesRegex(ValueError, "before the terminal"):
            _verify_generated_text(capped, tokenizer, model_vocab_size=151936, token_cap=8)
        production = {"generated_token_ids": [42] * 128, "eos_token_id": tokenizer.eos_token_id,
            "vocab_size": 151936, "score": {"answer_text": "answer", "termination": "token_cap",
                "generated_tokens": 128}}
        _verify_generated_text(production, tokenizer, model_vocab_size=151936, token_cap=128)


class SourcePilotAuditRegressionTests(unittest.TestCase):
    def test_pinned_metadata_uses_actual_closed_registration_shape_and_rejects_mutations(self):
        from dwr.experiments import gpu_feasibility
        from dwr.experiments.source_pilot import _verify_pinned_tokenizer_metadata

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "models" / "tiny"
            model_dir.mkdir(parents=True)
            fetch = root / "runs" / "fetch"
            fetch.mkdir(parents=True)
            config_path = model_dir / "config.json"
            tokenizer_path = model_dir / "tokenizer.json"
            config_raw = b'{"vocab_size":128,"eos_token_id":127}'
            tokenizer_raw = b'{"tokenizer":"tiny"}'
            config_path.write_bytes(config_raw)
            tokenizer_path.write_bytes(tokenizer_raw)
            inventory = {"files": [
                {"name": "config.json", "size_bytes": len(config_raw), "sha256": hashlib.sha256(config_raw).hexdigest()},
                {"name": "tokenizer.json", "size_bytes": len(tokenizer_raw), "sha256": hashlib.sha256(tokenizer_raw).hexdigest()},
            ]}
            inventory_raw = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
            (fetch / "artifact-inventory.json").write_bytes(inventory_raw)
            manifest_raw = b'{"schema":"fixture"}'
            manifest_path = fetch / "manifest.json"
            manifest_path.write_bytes(manifest_raw)
            seed_names = ("python_global", "numpy_global", "torch_cpu_global", "torch_cuda_global",
                "workspace_initialization", "reader_initialization")
            seeds = {name: index + 1 for index, name in enumerate(seed_names)}
            registration = {"config": {"seeds": seeds}, "model_pins": {
                    "artifact_manifest_path": "runs/fetch/manifest.json",
                    "artifact_inventory_sha256": hashlib.sha256(inventory_raw).hexdigest(),
                    "model_config_sha256": hashlib.sha256(config_raw).hexdigest()},
                "feasibility_config": {"model": {"local_dir": "models/tiny"}, "seeds": dict(seeds)}}
            policy = {"expected_model_vocab_size": 128, "expected_eos_token_id": 127}
            with patch.object(gpu_feasibility, "MODEL_MANIFEST_SHA256", hashlib.sha256(manifest_raw).hexdigest()):
                _verify_pinned_tokenizer_metadata(root, registration, policy)
                config_path.write_bytes(b'{"vocab_size":129,"eos_token_id":127}')
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    _verify_pinned_tokenizer_metadata(root, registration, policy)
                config_path.write_bytes(config_raw)
                tokenizer_path.write_bytes(b'{"tokenizer":"changed"}')
                with self.assertRaisesRegex(ValueError, "tokenizer file changed"):
                    _verify_pinned_tokenizer_metadata(root, registration, policy)

    def test_same_producer_requires_resumed_worker_initial_and_final_refs(self):
        from dwr.experiments.source_pilot import _require_same_producing_segment

        _require_same_producing_segment(
            {"path": "segment-001/controller-audit/initial-frozen.json", "sha256": "a" * 64},
            {"path": "segment-001/controller-audit/final-controller.json", "sha256": "b" * 64})
        with self.assertRaisesRegex(ValueError, "same producing segment"):
            _require_same_producing_segment(
                {"path": "segment-000/controller-audit/initial-frozen.json", "sha256": "a" * 64},
                {"path": "segment-001/controller-audit/final-controller.json", "sha256": "b" * 64})

    def test_checkpoint_derived_counts_must_match_controller_details(self):
        from dwr.experiments.source_pilot import _verify_final_parameter_counts

        policy = {"expected_parameter_counts": {"original_parameter_elements": 100,
            "workspace_parameter_elements": 20, "reader_parameter_elements": 30,
            "new_parameter_elements": 50, "retained_total_parameter_elements": 150}}
        details = {"parameter_counts": dict(policy["expected_parameter_counts"]), "trainable_parameters": 30}
        result = {"trainable_parameters": 30, "retained_total_parameters": 150}
        _verify_final_parameter_counts(details, result, policy, "K0")
        result["retained_total_parameters"] = 149
        with self.assertRaisesRegex(ValueError, "checkpoint-derived"):
            _verify_final_parameter_counts(details, result, policy, "K0")

    def test_rejects_forged_model_vocabulary_in_receipt(self):
        from dwr.experiments.source_pilot import _verify_generated_text

        envelope = {
            "generated_token_ids": [42, 151643], "eos_token_id": 151643,
            "vocab_size": 151665,
            "score": {"answer_text": "answer", "termination": "eos", "generated_tokens": 2},
        }

        with self.assertRaisesRegex(ValueError, "vocabulary"):
            _verify_generated_text(envelope, _Tokenizer(), model_vocab_size=151936)

    def test_rejects_stored_text_that_does_not_match_live_tokenizer(self):
        from dwr.experiments.source_pilot import _verify_generated_text

        envelope = {
            "generated_token_ids": [42, 151643], "eos_token_id": 151643,
            "vocab_size": 151936,
            "score": {"answer_text": "forged", "termination": "eos", "generated_tokens": 2},
        }

        with self.assertRaisesRegex(ValueError, "re-decode"):
            _verify_generated_text(envelope, _Tokenizer(), model_vocab_size=151936)


class SourcePilotGuardTests(unittest.TestCase):
    def test_live_cuda_memory_guard_uses_explicit_inherited_device(self):
        _skip_if_ml_dependencies_absent(self, "numpy", "torch", "transformers")
        from dwr.experiments.source_pilot import _check_live_memory

        calls = []
        fake_cuda = SimpleNamespace(
            is_initialized=lambda: True,
            synchronize=lambda device: calls.append(("synchronize", device)),
            max_memory_allocated=lambda device: calls.append(("allocated", device)) or 2 * 1024**2,
            max_memory_reserved=lambda device: calls.append(("reserved", device)) or 3 * 1024**2,
        )
        fake_torch = SimpleNamespace(cuda=fake_cuda)
        with patch("dwr.training.source_pilot._memory_measurement",
                   return_value={"scope": "fixture", "value_bytes": 4 * 1024**2}):
            result = _check_live_memory({"limits": {"resident_memory_mib": 12,
                "maximum_peak_allocated_mib": 8, "maximum_peak_reserved_mib": 8}}, fake_torch,
                device="cuda:7")
        self.assertIn(("synchronize", "cuda:7"), calls)
        self.assertEqual(result["peak_allocated_bytes"], 2 * 1024**2)
        self.assertEqual(result["peak_reserved_bytes"], 3 * 1024**2)

    def test_hash_helper_uses_caller_selected_work_or_seal_guard(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU torch fixture is unavailable")
        from types import SimpleNamespace
        from dwr.experiments.source_pilot import _hash_original_parameters_live

        module = torch.nn.Linear(8, 8, bias=False)
        host = SimpleNamespace(backbone=module)
        stages = []
        _hash_original_parameters_live(host, lambda stage: stages.append(stage),
            chunk_bytes=1024, guard_stage="seal")
        self.assertTrue(stages)
        self.assertEqual(set(stages), {"seal"})

    def test_private_fixture_policy_is_scaled_but_closed_to_registered_shapes(self):
        _skip_if_ml_dependencies_absent(self, "numpy", "torch", "transformers")
        from dwr.experiments.source_pilot import _validate_controller_policy
        from tests.source_pilot_controller_fixture import scaled_policy, tiny_host, tiny_tokenizer

        host, tokenizer = tiny_host(), tiny_tokenizer()
        policy = scaled_policy(host, tokenizer, steps=2, validation_examples=5, decoder_max_new_tokens=8)
        config = {"data": {"validation_examples": 5}, "training": {"optimizer_steps": 2},
            "decoding": {"embedding_vocab_size": 128, "eos_token_id": 127,
                "max_new_tokens_including_eos": 8}}
        self.assertEqual(_validate_controller_policy(policy, config), policy)
        config["decoding"]["max_new_tokens_including_eos"] = 128
        with self.assertRaisesRegex(ValueError, "exact resolved"):
            _validate_controller_policy(policy, config)
        config["decoding"]["max_new_tokens_including_eos"] = 8
        bad = dict(policy); bad["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "unknown or missing"):
            _validate_controller_policy(bad, config)

    def test_controller_observation_is_checked_against_registered_memory_caps(self):
        from dwr.experiments.source_pilot import _verify_controller_observation

        limits = {"resident_memory_mib": 2, "maximum_peak_allocated_mib": 1,
            "maximum_peak_reserved_mib": 1}
        observation = {"schema_version": "dwr-source-pilot-controller-observation-v1", "stage": "complete",
            "observed_utc": "2026-10-19T00:00:00+00:00", "elapsed_seconds": 1.25,
            "memory": {"resident": {"scope": "fixture", "value_bytes": 1024},
                "peak_allocated_bytes": 512, "peak_reserved_bytes": 768,
                "scope": "controller_high_water_observation"}}
        _verify_controller_observation(observation, limits, require_cuda=True)
        observation["memory"]["peak_reserved_bytes"] = 2 * 1024**2
        with self.assertRaisesRegex(ValueError, "registered cap"):
            _verify_controller_observation(observation, limits, require_cuda=True)


class SourcePilotEventAuditTests(unittest.TestCase):
    def test_reads_actual_fsynced_runrecord_event_envelope(self):
        from dwr.experiments.records import RunRecord
        from dwr.experiments.source_pilot import _read_events

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            record = RunRecord.create(root / "segment-000", command=["python", "worker.py"],
                config_path=config, expected_config_sha256=__import__("hashlib").sha256(config.read_bytes()).hexdigest(),
                identities={"model": {"name": "fixture", "revision": "v1", "sha256": "a" * 64}},
                rng_streams={"fixture": 1}, settings={"record_kind": "fixture"},
                budgets={"wall_seconds": 10, "memory_mib": 128, "attempts": 1},
                confirmatory=False, source_root=Path(__file__).resolve().parents[1])
            record.append_event("worker_receipt_published", {"nonce": "n", "descriptor_sha256": "b" * 64,
                "receipt": {"path": "segment-000/worker-receipt.json", "sha256": "c" * 64}})

            events = _read_events(root / "segment-000" / "events.jsonl")

        self.assertEqual(events[0]["kind"], "worker_receipt_published")
        self.assertEqual(events[0]["payload"]["nonce"], "n")

    def test_interrupted_hard_loss_access_prefixes_require_verified_loss_only_marker(self):
        from dwr.experiments.source_pilot import _verify_access_order

        registration = {"cohort_sha256": "c" * 64,
            "cohort": {"selected": [{"example_id": "val-1"}]},
            "token_rows": [{"example_id": "train-1", "split": "source_train"}],
            "config": {"data": {"source_train_records": 1, "validation_examples": 1}}}
        config = registration["config"]
        marker = {"descriptor_sha256": "a" * 64, "worker_receipt_sha256": "b" * 64,
            "worker_identity": {"pid": 22, "creation_time": "worker-created", "image": "python.exe"},
            "launcher_identity": {"pid": 11, "creation_time": "launcher-created", "image": "python.exe"},
            "source_commit": "d" * 40, "runtime_receipt_sha256": "e" * 64,
            "worker_lock_released": True,
            "postloss_observation": {"available": False, "reason": "process_exited_before_observation"}}
        receipt = {"kind": "worker_receipt_published", "payload": {"nonce": "n",
            "descriptor_sha256": marker["descriptor_sha256"], "receipt": {"path": "segment-000/worker-receipt.json",
                "sha256": marker["worker_receipt_sha256"]}}}
        loss = {"kind": "worker_loss_finalized", "payload": marker}
        validation = {"kind": "source_examples_access_authorized", "payload": {
            "split": "source_validation", "record_count": 1,
            "ordered_ids_sha256": __import__("hashlib").sha256(b'["val-1"]').hexdigest(),
            "cohort_sha256": registration["cohort_sha256"],
            "purpose": "registered_source_validation_for_kernel_and_arm_audit"}}
        training = {"kind": "source_examples_access_authorized", "payload": {
            "split": "source_train", "record_count": 1,
            "ordered_ids_sha256": __import__("hashlib").sha256(b'["train-1"]').hexdigest(),
            "cohort_sha256": registration["cohort_sha256"],
            "purpose": "registered_source_training_for_kernel"}}

        for prefix in ([], [validation]):
            _verify_access_order([receipt, *prefix, loss], config, registration,
                terminal_status="interrupted", verified_loss=marker)
        _verify_access_order([receipt, validation, training, loss], config, registration,
            terminal_status="interrupted", verified_loss=marker)
        _verify_access_order([receipt, validation, training], config, registration,
            terminal_status="completed", verified_loss=None)

        with self.assertRaisesRegex(ValueError, "loss|prefix|access"):
            _verify_access_order([receipt, loss], config, registration,
                terminal_status="interrupted", verified_loss=None)
        with self.assertRaisesRegex(ValueError, "completed segments|completed|prefix"):
            _verify_access_order([receipt, validation], config, registration,
                terminal_status="completed", verified_loss=None)
        forged = dict(marker); forged["worker_lock_released"] = False
        with self.assertRaisesRegex(ValueError, "loss|prefix|access"):
            _verify_access_order([receipt, loss], config, registration,
                terminal_status="interrupted", verified_loss=forged)
        with self.assertRaisesRegex(ValueError, "loss|prefix|access"):
            _verify_access_order([receipt, {"kind": "controller_host_initialized", "payload": {}}, loss],
                config, registration, terminal_status="interrupted", verified_loss=marker)
        with self.assertRaisesRegex(ValueError, "order|validation"):
            _verify_access_order([receipt, training, loss], config, registration,
                terminal_status="interrupted", verified_loss=marker)

    def test_pairs_only_committed_updates_across_segments(self):
        from dwr.experiments.source_pilot import _pair_committed_updates

        event_segments = []
        for segment_index, steps in enumerate((range(1, 4), range(4, 513))):
            events = []
            for step in steps:
                events.append({"kind": "update_completed", "payload": {"net_step": step,
                    "example_id": f"ex-{step}", "batch_sha256": f"{step:064x}", "noise_sha256": f"{step + 1000:064x}"}})
                events.append({"kind": "recovery_committed", "payload": {"progress": {"optimizer_step": step}}})
            event_segments.append((segment_index, events))

        rows = _pair_committed_updates(event_segments)

        self.assertEqual(len(rows), 512)
        self.assertEqual(rows[0]["optimizer_step"], 1)
        self.assertEqual(rows[-1]["optimizer_step"], 512)
        self.assertEqual(rows[256]["example_id"], "ex-257")

    def test_does_not_invent_pairing_for_uncommitted_update(self):
        from dwr.experiments.source_pilot import _pair_committed_updates

        events = [{"kind": "update_completed", "payload": {"net_step": 1,
            "example_id": "ex-1", "batch_sha256": "a" * 64, "noise_sha256": "b" * 64}}]
        with self.assertRaisesRegex(ValueError, "registered unique committed"):
            _pair_committed_updates([(0, events)])

    def test_terminal_known_fault_is_fsynced_before_fallible_seal(self):
        from dwr.experiments.source_pilot import _seal_worker_failure

        class Record:
            def __init__(self):
                self.events = []

            def append_event(self, kind, payload):
                self.events.append((kind, payload))

            def finish(self, *args, **kwargs):
                raise OSError("disk full")

        with tempfile.TemporaryDirectory() as temporary:
            record = Record()
            _seal_worker_failure(record, Path(temporary), ValueError("bad config"), status="failed",
                check=lambda stage: None, phase="preflight")

        self.assertEqual(record.events[0][0], "known_fault")
        self.assertEqual(record.events[0][1]["exception_type"], "ValueError")
        self.assertEqual(record.events[0][1]["phase"], "preflight")
        self.assertEqual(len(record.events[0][1]["message_sha256"]), 64)

    def test_chunked_live_hash_matches_canonical_frozen_parameter_hash(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU torch fixture is unavailable")
        from types import SimpleNamespace
        from dwr.experiments.gpu_feasibility import hash_original_parameters
        from dwr.experiments.source_pilot import _hash_original_parameters_live

        module = torch.nn.Module()
        module.register_parameter("first", torch.nn.Parameter(torch.arange(4096, dtype=torch.float32).reshape(64, 64),
            requires_grad=False))
        module.register_parameter("second", torch.nn.Parameter(torch.arange(2048, dtype=torch.bfloat16),
            requires_grad=False))
        host = SimpleNamespace(backbone=module)
        checks = []

        observed = _hash_original_parameters_live(host, lambda stage: checks.append(stage), chunk_bytes=1024)

        self.assertEqual(observed, hash_original_parameters(host))
        self.assertGreater(len(checks), 2)
        self.assertTrue(all(stage == "work" for stage in checks))


if __name__ == "__main__":
    unittest.main()
