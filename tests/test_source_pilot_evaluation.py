from __future__ import annotations

import dataclasses
import hashlib
import tempfile
import unittest
from pathlib import Path

from dwr.evaluation.generation import greedy_workspace
from dwr.evaluation.source_pilot import build_workspace_bank, greedy_fixed_workspace, score_pilot_stage
from dwr.tasks.run_support import stable_json
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, Solution, RelationKind

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
if torch is not None:
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except ModuleNotFoundError as error:
        if error.name != "transformers":
            raise
        Qwen2Config = Qwen2ForCausalLM = None
    if Qwen2Config is not None:
        from dwr.models.qwen2 import Qwen2WorkspaceLM
        from dwr.models.workspace import FixedWorkspace
        from dwr.training.batches import SourceExample, TokenBatch, TokenizerIdentity
    else:
        Qwen2WorkspaceLM = FixedWorkspace = SourceExample = TokenBatch = TokenizerIdentity = None
else:
    Qwen2Config = Qwen2ForCausalLM = Qwen2WorkspaceLM = FixedWorkspace = None
    SourceExample = TokenBatch = TokenizerIdentity = None


def _task() -> ConstraintTask:
    return ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
        (Constraint(RelationKind.EQ, ("v0", "v1")),
         Constraint(RelationKind.EQ, ("v1", "v2"))), (GivenAssignment("v0", 1),))


def _example(example_id: str) -> SourceExample:
    return SourceExample(example_id, "source_validation", _task(),
                         Solution((("v0", 1), ("v1", 1), ("v2", 1))))


@unittest.skipIf(torch is None or Qwen2WorkspaceLM is None,
                 "optional torch/transformers dependencies are unavailable")
class NativeSourcePilotEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_host(self):
        config = Qwen2Config(vocab_size=19, hidden_size=12, intermediate_size=24,
            num_hidden_layers=3, num_attention_heads=3, num_key_value_heads=1,
            attention_dropout=0.0, tie_word_embeddings=True, pad_token_id=18,
            eos_token_id=17, _attn_implementation="eager")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1801)
            backbone = Qwen2ForCausalLM(config).to(dtype=torch.float32)
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1802)
            workspace = FixedWorkspace(12, num_slots=2, latent_dim=8, num_heads=2,
                max_steps=4, slot_identity="exchangeable").to(dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1803)
            host = Qwen2WorkspaceLM(backbone, workspace, prompt_layer=1,
                adapter_layers=(0, 1, 2), reader_heads=3)
        return host.eval()

    def _batch(self, example_id: str, label_value: int) -> TokenBatch:
        prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
        decoder_ids = torch.tensor([[1, 2, 3, label_value]], dtype=torch.long)
        decoder_mask = torch.ones_like(decoder_ids, dtype=torch.bool)
        labels = torch.tensor([[-100, -100, -100, label_value]], dtype=torch.long)
        return TokenBatch((example_id,), ("source_validation",), TokenizerIdentity("fixture", "v1"),
                          prompt_ids, prompt_mask, decoder_ids, decoder_mask, labels)

    def test_fixed_workspace_decoder_matches_native_generation_and_preserves_rng_state(self):
        host = self.make_host()
        noise = torch.randn((1, 2, 8), generator=torch.Generator().manual_seed(1901))
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
        before_rng = torch.random.get_rng_state().clone()
        before_params = [parameter.detach().clone() for parameter in host.parameters()]
        expected = greedy_workspace(host, prompt_ids, prompt_mask, noise=noise, steps=4,
            eos_token_id=17, max_new_tokens=3, max_total_tokens=10)
        encoded = host.encode_prompt(prompt_ids, prompt_mask).to(dtype=torch.float32)
        with torch.no_grad():
            workspace = host.workspace(encoded, noise=noise, prompt_padding_mask=~prompt_mask, steps=4)
        actual = greedy_fixed_workspace(host, prompt_ids, prompt_mask, workspace=workspace,
            eos_token_id=17, max_new_tokens=3, max_total_tokens=10)
        self.assertEqual(actual.token_ids, expected.token_ids)
        self.assertEqual(actual.prompt_encoder_calls, 0)
        self.assertEqual(actual.workspace_updates, 0)
        self.assertEqual(actual.decoder_calls, expected.decoder_calls)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_params, host.parameters())))
        with self.assertRaises(ValueError):
            greedy_fixed_workspace(host, prompt_ids, prompt_mask,
                workspace=torch.zeros((1, 3, 8)), eos_token_id=17,
                max_new_tokens=3, max_total_tokens=10)

    def test_fixed_workspace_exception_preserves_model_workspace_modes_hooks_and_rng(self):
        host = self.make_host()
        prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
        workspace = torch.full((1, 2, 8), 0.25, dtype=torch.float32)
        rng_before = torch.random.get_rng_state().clone()
        params_before = [parameter.detach().clone() for parameter in host.parameters()]
        workspace_before = workspace.clone()
        modes_before = [module.training for module in host.modules()]
        hooks_before = [(dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in host.modules()]
        config_before = dict(host.backbone.config.to_dict())
        calls = 0
        def fail_after_native_decode():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("synthetic budget stop after native decode")
        with self.assertRaisesRegex(RuntimeError, "after native decode"):
            greedy_fixed_workspace(host, prompt_ids, prompt_mask, workspace=workspace,
                eos_token_id=17, max_new_tokens=3, max_total_tokens=10,
                check=fail_after_native_decode)
        self.assertEqual(calls, 3)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertTrue(torch.equal(workspace, workspace_before))
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(params_before, host.parameters())))
        self.assertEqual([module.training for module in host.modules()], modes_before)
        self.assertEqual([(dict(module._forward_hooks), dict(module._forward_pre_hooks))
                          for module in host.modules()], hooks_before)
        self.assertEqual(host.backbone.config.to_dict(), config_before)

    def test_workspace_bank_uses_prompt_only_and_has_stable_cpu_bytes_and_donors(self):
        host = self.make_host()
        ids = tuple(hashlib.sha256(f"validation-{i}".encode()).hexdigest() for i in range(3))
        batches = tuple(self._batch(eid, 7 + i) for i, eid in enumerate(ids))
        before_rng = torch.random.get_rng_state().clone()
        bank = build_workspace_bank(host, batches, steps=4, noise_seed_base=20261219)
        changed_labels = tuple(dataclasses.replace(batch,
            decoder_ids=torch.full_like(batch.decoder_ids, 11),
            labels=torch.full_like(batch.labels, 11)) for batch in batches)
        again = build_workspace_bank(host, changed_labels, steps=4, noise_seed_base=20261219)
        self.assertEqual(set(bank), {"schema_version", "steps", "example_ids", "workspace_bank",
                                    "rows", "donor_map", "counts"})
        self.assertEqual(bank["schema_version"], "dwr-source-pilot-workspace-bank-v1")
        self.assertEqual(tuple(bank["workspace_bank"].shape), (3, 2, 8))
        self.assertEqual(bank["workspace_bank"].dtype, torch.float32)
        self.assertEqual(bank["counts"], {"prompt_encoder_calls": 3, "workspace_updates": 12})
        self.assertTrue(torch.equal(bank["workspace_bank"], again["workspace_bank"]))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))
        sorted_ids = sorted(ids)
        donors = {eid: sorted_ids[(i + 1) % len(sorted_ids)] for i, eid in enumerate(sorted_ids)}
        self.assertEqual(bank["donor_map"], donors)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bank.pt"
            torch.save(bank["workspace_bank"], path)
            restored = torch.load(path, map_location="cpu", weights_only=True)
            self.assertTrue(torch.equal(restored, bank["workspace_bank"]))

    def test_scorer_recomputes_verifier_and_keeps_failed_rows_in_fine_denominator(self):
        from dwr.tasks.verifier import verify_answer
        ids = tuple(hashlib.sha256(f"score-{i}".encode()).hexdigest() for i in range(5))
        examples = tuple(_example(eid) for eid in ids)
        strata = ("A2", "A3", "B2", "C2", "D2")
        token_counts = (1, 2, 3, 4, 5)
        metadata = {eid: {"profile": "eq", "fine_stratum": strata[i],
                          "answer_tokens_including_eos": token_counts[i]} for i, eid in enumerate(ids)}
        correct = stable_json({"assignment": {"v0": 1, "v1": 1, "v2": 1}})
        rows = []
        for i, eid in enumerate(ids):
            answer = correct if i == 0 else "{}"
            verdict = verify_answer(examples[i].task, answer)
            rows.append({"example_id": eid, "answer_text": answer, "valid": verdict.valid,
                "verifier_reason": verdict.reason.value, "termination": "eos", "generated_tokens": 3,
                "mean_ce": float(i + 1), "supervised_tokens": token_counts[i]})
        result = score_pilot_stage(tuple(rows), expected_ids=ids, metadata=metadata,
                                  examples=examples, include_ce=True)
        self.assertEqual(result["valid_denominator"], 5)
        self.assertEqual(result["valid_numerator"], 1)
        self.assertAlmostEqual(result["vsr"], 0.2)
        self.assertEqual(result["fine_strata"]["A3"]["valid"], 0)
        self.assertEqual(result["ce"]["token_denominator"], 15)
        self.assertAlmostEqual(result["ce"]["weighted_numerator"], 55.0)
        self.assertAlmostEqual(result["ce"]["token_weighted_mean"], 55.0 / 15)
        self.assertIsNone(result["primitive"]["neq"]["rate"])
        self.assertEqual(result["primitive"]["neq"]["denominator"], 0)
        self.assertIsNone(result["primitive_by_fine"]["neq"]["A2"]["rate"])
        forged = list(rows)
        forged[1] = {**forged[1], "valid": True}
        with self.assertRaises(ValueError):
            score_pilot_stage(tuple(forged), expected_ids=ids, metadata=metadata,
                              examples=examples, include_ce=True)
        with self.assertRaises(ValueError):
            score_pilot_stage(tuple(rows[:-1]), expected_ids=ids, metadata=metadata,
                              examples=examples, include_ce=True)
        wrong_ce_tokens = list(rows)
        wrong_ce_tokens[0] = {**rows[0], "supervised_tokens": 2}
        with self.assertRaises(ValueError):
            score_pilot_stage(tuple(wrong_ce_tokens), expected_ids=ids, metadata=metadata,
                              examples=examples, include_ce=True)

    def test_scorer_enforces_registered_eos_and_token_cap_boundaries(self):
        from dwr.tasks.verifier import verify_answer
        ids = tuple(hashlib.sha256(f"boundary-{i}".encode()).hexdigest() for i in range(5))
        examples = tuple(_example(eid) for eid in ids)
        strata = ("A2", "A3", "B2", "C2", "D2")
        metadata = {eid: {"profile": "eq", "fine_stratum": strata[i]} for i, eid in enumerate(ids)}
        correct = stable_json({"assignment": {"v0": 1, "v1": 1, "v2": 1}})

        def records_for(first_termination, first_count):
            rows = []
            for i, eid in enumerate(ids):
                answer = correct if i == 0 else "{}"
                termination = first_termination if i == 0 else "eos"
                count = first_count if i == 0 else 1
                verdict = verify_answer(examples[i].task, answer)
                valid = bool(verdict.valid and termination == "eos")
                rows.append({"example_id": eid, "answer_text": answer, "valid": valid,
                    "verifier_reason": verdict.reason.value, "termination": termination,
                    "generated_tokens": count})
            return tuple(rows)

        capped_boundary = score_pilot_stage(records_for("token_cap", 128), expected_ids=ids,
            metadata=metadata, examples=examples, include_ce=False)
        self.assertEqual(capped_boundary["valid_denominator"], 5)
        self.assertEqual(capped_boundary["valid_numerator"], 0)
        eos_boundary = score_pilot_stage(records_for("eos", 128), expected_ids=ids,
            metadata=metadata, examples=examples, include_ce=False)
        self.assertEqual(eos_boundary["valid_numerator"], 1)
        with self.assertRaises(ValueError):
            score_pilot_stage(records_for("eos", 129), expected_ids=ids,
                metadata=metadata, examples=examples, include_ce=False)
        with self.assertRaises(ValueError):
            score_pilot_stage(records_for("token_cap", 127), expected_ids=ids,
                metadata=metadata, examples=examples, include_ce=False)


if __name__ == "__main__":
    unittest.main()
