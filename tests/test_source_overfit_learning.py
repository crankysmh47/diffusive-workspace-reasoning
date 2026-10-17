"""Behavioral CPU checks for the tiny native learning kernel."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import subprocess

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ModuleNotFoundError as error:
    if error.name not in {"torch", "transformers"}:
        raise
    torch = Qwen2Config = Qwen2ForCausalLM = None

from test_training_batch_loss import AsciiTokenizer, fixture_task

if torch is not None:
    from dwr.models.qwen2 import Qwen2WorkspaceLM
    from dwr.models.workspace import FixedWorkspace
    from dwr.training.batches import SourceExample, TokenizerIdentity
    from dwr.training.overfit import run_overfit

_ML_READY = torch is not None and Qwen2Config is not None

IDS = tuple(hashlib.sha256(f"overfit-{name}".encode()).hexdigest()
            for name in ("eq", "neq", "add"))
IDENTITIES = {key: hashlib.sha256(key.encode()).hexdigest()
              for key in ("source", "config", "data", "model", "tokenizer", "schedule")}


class RecordingTokenizer(AsciiTokenizer):
    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return "".join(chr(value) for value in ids)


def _new_segment(directory):
    path = Path(directory) / "segment-000"
    path.mkdir(exist_ok=True)
    return path


def _fixtures(seed=10):
    examples = []
    for example_id, kind in zip(IDS, ("eq", "neq", "add_mod")):
        task, solution = fixture_task(kind)
        examples.append(SourceExample(example_id, "source_train", task, solution))
    config = Qwen2Config(vocab_size=128, hidden_size=12, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=3, num_key_value_heads=1,
        attention_dropout=0.0, tie_word_embeddings=True, pad_token_id=127,
        eos_token_id=127, _attn_implementation="eager")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = Qwen2ForCausalLM(config).to(dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        workspace = FixedWorkspace(12, num_slots=4, latent_dim=8, num_heads=2,
                                   max_steps=2).to(dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 2)
        host = Qwen2WorkspaceLM(backbone, workspace, prompt_layer=1,
                                adapter_layers=(0, 1, 2), reader_heads=3)
    study = {
        "training": {"method": "EX", "steps": 2, "optimizer_steps": 9, "batch_size": 1,
            "checkpoint_steps": [3, 9], "diagnose_state_gradient_steps": [1, 9],
            "gradient_clip_norm": 1.0, "num_slots": 4, "latent_dim": 8,
            "optimizer": {"learning_rate": 0.001, "betas": [0.9, 0.999],
                "epsilon": 1e-8, "weight_decay": 0.01, "amsgrad": False,
                "capturable": False, "differentiable": False, "foreach": False,
                "fused": False, "maximize": False},
            "schedule": {"warmup_steps": 8, "first_update_factor": 0.125,
                "function": "min(1.0,(scheduler_step+1)/8)"}},
        "decoding": {"max_prompt_tokens": 4096, "max_answer_tokens": 128, "max_new_tokens_including_eos": 8,
            "max_total_tokens": 4200, "eos_token_id": 127, "pad_token_id": 127,
            "embedding_vocab_size": 128},
        "seeds": {"source_sampler": 71, "workspace_noise_cuda": 72,
                  "evaluation_noise_base": 20261209},
        "resume": {"checkpoint_step": 3, "replayed_step": 4,
            "float_atol": 1e-12, "float_rtol": 1e-12},
        "evaluation": {"steps": [0, 9], "gate_max_final_to_initial_CE_ratio": 1e9},
    }
    metadata = {example.example_id: {"lineage_id": hashlib.sha256((example.example_id+"l").encode()).hexdigest(),
        "structure_signature": hashlib.sha256((example.example_id+"s").encode()).hexdigest(),
        "profile": kind, "fine_stratum": "A2"}
        for example, kind in zip(examples, ("eq", "neq", "add_mod"))}
    return host, tuple(examples), study, metadata


TEST_TOKENIZER_IDENTITY = TokenizerIdentity("ascii-fixture", "fixture-v1") if torch is not None else None


@unittest.skipIf(torch is None, "optional ML stack unavailable")
class SourceOverfitLearningTests(unittest.TestCase):
    @unittest.skipUnless(_ML_READY, "native model fixture requires optional Torch/transformers")
    def test_fixture_new_parameters_are_seeded_without_consuming_ambient_rng(self):
        before = torch.get_rng_state().clone()
        host_a, _, _, _ = _fixtures(77)
        middle = torch.get_rng_state().clone()
        host_b, _, _, _ = _fixtures(77)
        after = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(before, middle))
        self.assertTrue(torch.equal(middle, after))
        self.assertEqual(__import__("dwr.training.overfit", fromlist=["_initial_state_digest"])
            ._initial_state_digest(host_a.workspace, host_a.readers)[0],
            __import__("dwr.training.overfit", fromlist=["_initial_state_digest"])
            ._initial_state_digest(host_b.workspace, host_b.readers)[0])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(host_a.backbone.parameters(), host_b.backbone.parameters())))

    @unittest.skipUnless(_ML_READY, "native model fixture requires optional Torch/transformers")
    def test_training_updates_frozen_backbone_and_logs_preupdate_warmup_lr(self):
        host, examples, config, metadata = _fixtures()
        frozen = {n: p.detach().clone() for n, p in host.backbone.named_parameters()}
        learned = {n: p.detach().clone() for n, p in host.workspace.named_parameters()}
        original_modes = {module: module.training for module in host.modules()}
        tokenizer = RecordingTokenizer()
        events = []
        with tempfile.TemporaryDirectory() as directory:
            result = run_overfit(host, tokenizer, TEST_TOKENIZER_IDENTITY, examples, config=config,
                identities=IDENTITIES, metadata=metadata, output_dir=_new_segment(directory),
                check=lambda: None, on_event=lambda kind, payload: events.append((kind, payload)))
            updates = [payload for kind, payload in events if kind == "update"]
            main_updates = [row for row in updates if row["role"] == "main"]
            committed_steps = {payload["progress"]["optimizer_step"] for kind, payload in events
                               if kind == "recovery_committed"}
            self.assertTrue(set(range(10)).issubset(committed_steps))
            self.assertEqual([row["learning_rate"] for row in main_updates[:8]],
                             [0.001 * factor for factor in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)])
            self.assertEqual(result["physical_optimizer_updates"], 10)
            self.assertTrue(all(torch.equal(frozen[n], p) for n, p in host.backbone.named_parameters()))
            self.assertTrue(any(not torch.equal(learned[n], p) for n, p in host.workspace.named_parameters()))
            self.assertEqual(original_modes, {module: module.training for module in host.modules()})
            self.assertTrue(tokenizer.calls)
            self.assertEqual(tokenizer.calls[0][0], __import__("dwr.tasks", fromlist=["serialize_model_input"]).serialize_model_input(examples[0].task) + "\n")
            self.assertTrue(all(len(row["recurrent_state_norms"]) == 3 for row in updates))
            diagnostic_rows = [row for row in updates if row["reader_gradient_norms"]]
            self.assertTrue(diagnostic_rows)
            self.assertTrue(all(set(row["reader_gradient_norms"]) == set(host.readers) for row in diagnostic_rows))
            self.assertEqual(result["predictions"][:3][0]["checkpoint_sha256"], result["initial_state_sha256"])
            self.assertTrue(all(row["checkpoint_sha256"] == result["checkpoint_hashes"]["9"]
                                for row in result["predictions"][3:]))
            self.assertEqual(result["initial_state_hash_recipe"]["name"],
                             "dwr-source-overfit-initial-state-v1")
            self.assertTrue(Path(result["artifact_paths"]["retired_recovery_receipts"]).is_file())

    @unittest.skipUnless(_ML_READY, "native model fixture requires optional Torch/transformers")
    def test_injected_budget_failure_keeps_update_log(self):
        host, examples, config, metadata = _fixtures(25)
        saw_update = False
        def on_event(kind, _payload):
            nonlocal saw_update
            if kind == "update": saw_update = True
        def check():
            if saw_update: raise RuntimeError("injected budget stop")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "injected budget stop"):
                run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples, config=config,
                    identities=IDENTITIES, metadata=metadata, output_dir=_new_segment(directory),
                    check=check, on_event=on_event)
            log = Path(directory) / "segment-000" / "updates.jsonl"
            self.assertTrue(log.is_file())
            self.assertIn('"role":"main"', log.read_text(encoding="utf-8"))

    @unittest.skipUnless(_ML_READY, "native model fixture requires optional Torch/transformers")
    def test_numerical_divergence_is_distinct_and_has_partial_counters(self):
        from dwr.training.overfit import NumericalDivergence
        host, examples, config, metadata = _fixtures(26)
        with tempfile.TemporaryDirectory() as directory, patch("dwr.training.overfit._all_finite", return_value=False):
            with self.assertRaises(NumericalDivergence) as caught:
                run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples, config=config,
                    identities=IDENTITIES, metadata=metadata, output_dir=_new_segment(directory),
                    check=lambda: None, on_event=lambda *_: None)
            self.assertEqual(caught.exception.partial["last_good_step"], 0)
            self.assertEqual(caught.exception.partial["physical_optimizer_updates"], 1)
            self.assertTrue(Path(caught.exception.partial["artifact_paths"]["updates"]).is_file())
            self.assertIn('"failure_kind":"NUMERICAL_DIVERGENCE"',
                          Path(caught.exception.partial["artifact_paths"]["updates"]).read_text(encoding="utf-8"))

    @unittest.skipUnless(_ML_READY, "native model fixture requires optional Torch/transformers")
    def test_native_nonfinite_loss_is_numerical_divergence(self):
        from dwr.training.overfit import NumericalDivergence
        host, examples, config, metadata = _fixtures(27)
        forward = host.forward
        def nonfinite_forward(*args, **kwargs):
            output = forward(*args, **kwargs)
            output.logits[0, 0, 0] = torch.nan
            return output
        with tempfile.TemporaryDirectory() as directory, patch.object(host, "forward", side_effect=nonfinite_forward):
            with self.assertRaises(NumericalDivergence) as caught:
                run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                    config=config, identities=IDENTITIES, metadata=metadata, output_dir=_new_segment(directory),
                    check=lambda: None, on_event=lambda *_: None)
            self.assertEqual(caught.exception.partial["physical_optimizer_updates"], 0)
            self.assertIn('"failure_kind":"NUMERICAL_DIVERGENCE"',
                          Path(caught.exception.partial["artifact_paths"]["updates"]).read_text(encoding="utf-8"))

    @unittest.skipUnless(_ML_READY, "native model fixture requires optional Torch/transformers")
    def test_missing_active_gradient_is_operational_and_persisted(self):
        host, examples, config, metadata = _fixtures(28)
        forward = host.forward
        def disconnected_forward(*args, **kwargs):
            output = forward(*args, **kwargs)
            return replace(output, logits=torch.zeros_like(output.logits, requires_grad=True))
        with tempfile.TemporaryDirectory() as directory, patch.object(host, "forward", side_effect=disconnected_forward):
            with self.assertRaisesRegex(RuntimeError, "has no gradient"):
                run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                    config=config, identities=IDENTITIES, metadata=metadata, output_dir=_new_segment(directory),
                    check=lambda: None, on_event=lambda *_: None)
            record = Path(directory) / "segment-000" / "updates.jsonl"
            self.assertIn('"failure_kind":"gradient_invariant"', record.read_text(encoding="utf-8"))

    @unittest.skipUnless(_ML_READY, "native model fixture requires optional Torch/transformers")
    def test_evaluation_noise_is_stable_and_state_isolation_is_recorded(self):
        host, examples, config, metadata = _fixtures(20)
        events = []
        with tempfile.TemporaryDirectory() as directory:
            result = run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples, config=config,
                identities=IDENTITIES, metadata=metadata, output_dir=_new_segment(directory),
                check=lambda: None, on_event=lambda *args: events.append(args))
            self.assertEqual(len(result["predictions"]), 6)
            before, after = result["predictions"][:3], result["predictions"][3:]
            self.assertEqual([x["noise_seed"] for x in before], [x["noise_seed"] for x in after])
            self.assertEqual([x["noise_sha256"] for x in before], [x["noise_sha256"] for x in after])
            evaluations = [payload for kind, payload in events if kind == "evaluation"]
            self.assertEqual(len(evaluations), 2)
            for audit in evaluations:
                self.assertEqual(audit["before"], audit["after"])
            self.assertEqual(evaluations[0]["evaluation_noise_sha256_by_id"],
                             evaluations[1]["evaluation_noise_sha256_by_id"])
            # Recreate draws in reverse example order: per-ID noise must stay identical.
            from dwr.training.overfit import _noise_seed
            reordered = {}
            for example in reversed(examples):
                generator = torch.Generator(device="cpu").manual_seed(
                    _noise_seed(config["seeds"]["evaluation_noise_base"], example.example_id))
                noise = host.workspace.sample_noise(1, generator=generator, device="cpu", dtype=torch.float64)
                reordered[example.example_id] = hashlib.sha256(
                    noise.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            self.assertEqual(reordered, evaluations[0]["evaluation_noise_sha256_by_id"])

    @unittest.skipUnless(_ML_READY, "native model fixture requires optional Torch/transformers")
    def test_resume_replays_epoch_boundary_and_records_actual_probe(self):
        host, examples, config, metadata = _fixtures(30)
        with tempfile.TemporaryDirectory() as directory:
            result = run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples, config=config,
                identities=IDENTITIES, metadata=metadata, output_dir=_new_segment(directory),
                check=lambda: None, on_event=lambda *_: None)
            resume = result["resume"]
            self.assertEqual(resume["first_replayed_ids"], resume["second_replayed_ids"])
            self.assertEqual(resume["max_parameter_abs_error"], 0.0)
            self.assertEqual(resume["max_optimizer_abs_error"], 0.0)
            self.assertGreater(resume["optimizer_moment_count"], 0)
            self.assertEqual(resume["saved_sampler_epoch"], 0)
            self.assertEqual(resume["saved_sampler_position"], 3)
            self.assertEqual(resume["replayed_sampler_epoch"], 1)
            self.assertEqual(resume["replayed_sampler_position"], 1)
            self.assertTrue(Path(resume["probe_state_path"]).is_file())
            self.assertTrue(Path(resume["checkpoint_path"]).is_file())
            rows = [json.loads(line) for line in Path(result["artifact_paths"]["updates"]).read_text(encoding="utf-8").splitlines()]
            probe = next(row for row in rows if row["role"] == "discarded_resume_probe")
            replay = next(row for row in rows if row["role"] == "main" and row["net_step"] == 4)
            self.assertEqual(probe["training_noise_sha256"], replay["training_noise_sha256"])
            self.assertTrue(resume["checkpoint_sha256"])
            self.assertTrue(resume["probe_state_sha256"])

    @unittest.skipUnless(_ML_READY, "native model fixture requires optional Torch/transformers")
    def test_replay_tolerance_combines_absolute_and_relative_terms(self):
        from dwr.training.overfit import _within_tolerance
        left = torch.tensor([1.000009], dtype=torch.float64)
        right = torch.tensor([1.0], dtype=torch.float64)
        absolute, relative, within = _within_tolerance(left, right, 1e-6, 1e-5)
        self.assertTrue(within)
        self.assertGreater(absolute, 1e-6)
        self.assertGreater(relative, 0.0)
        self.assertFalse(_within_tolerance(torch.tensor([1.000012]), torch.tensor([1.0]), 1e-6, 1e-5)[2])

    @unittest.skipUnless(_ML_READY, "native model fixture requires optional Torch/transformers")
    def test_final_checkpoint_ex_and_batch_layout_are_required(self):
        host, examples, config, metadata = _fixtures(35)
        variants = []
        missing_final = copy.deepcopy(config); missing_final["training"]["checkpoint_steps"] = [3]
        variants.append(missing_final)
        wrong_method = copy.deepcopy(config); wrong_method["training"]["method"] = "IDX"
        variants.append(wrong_method)
        wrong_batch = copy.deepcopy(config); wrong_batch["training"]["batch_size"] = 2
        variants.append(wrong_batch)
        wrong_layout = copy.deepcopy(config); wrong_layout["training"]["steps"] = 4
        variants.append(wrong_layout)
        wrong_eval = copy.deepcopy(config); wrong_eval["evaluation"]["steps"] = [0, 8]
        variants.append(wrong_eval)
        with tempfile.TemporaryDirectory() as directory:
            for candidate in variants:
                with self.subTest(candidate=candidate["training"]), self.assertRaises(ValueError):
                    run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                        config=candidate, identities=IDENTITIES, metadata=metadata, output_dir=_new_segment(directory),
                        check=lambda: None, on_event=lambda *_: None)

    @unittest.skipUnless(_ML_READY, "native model fixture requires optional Torch/transformers")
    def test_indexed_host_is_rejected_even_when_config_says_ex(self):
        host, examples, config, metadata = _fixtures(41)
        host.workspace = FixedWorkspace(12, num_slots=4, latent_dim=8, num_heads=2,
            max_steps=2, slot_identity="indexed").to(dtype=torch.float64)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                config=config, identities=IDENTITIES, metadata=metadata, output_dir=_new_segment(directory),
                check=lambda: None, on_event=lambda *_: None)

    @unittest.skipUnless(_ML_READY, "native model fixture requires optional Torch/transformers")
    def test_bad_membership_rejected_before_any_update(self):
        host, examples, config, metadata = _fixtures(40)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples[:2], config=config,
                    identities=IDENTITIES, metadata=metadata, output_dir=_new_segment(directory),
                    check=lambda: None, on_event=lambda *_: None)


class OptionalDependencyImportTests(unittest.TestCase):
    def test_training_package_reports_missing_optional_ml_dependency(self):
        root = Path(__file__).resolve().parents[1]
        source = """\
try:
    import dwr.training
except ModuleNotFoundError as error:
    assert error.name == 'torch', error.name
    print('optional torch dependency reported')
else:
    raise SystemExit('expected optional torch import to be unavailable under -S')
"""
        result = subprocess.run([sys.executable, "-S", "-c", source],
            cwd=root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn("optional torch dependency reported", result.stdout)


if __name__ == "__main__":
    unittest.main()
