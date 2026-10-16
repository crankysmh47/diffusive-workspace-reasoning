"""Scripted algorithm checks and actual tiny-native CPU generation checks."""

from __future__ import annotations

import hashlib
import json
import math
import random
import unittest
from contextlib import nullcontext
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - optional ML extra
    if error.name != "torch":
        raise
    torch = None

from dwr.evaluation.generation import (
    GenerationResult,
    _autocast_off,
    _greedy_loop,
    _validate_model_device,
    _synchronize_device,
    evaluation_noise_seed,
    greedy_frozen,
    greedy_workspace,
    score_generation,
)
from dwr.tasks.schema import (
    Constraint,
    ConstraintTask,
    GivenAssignment,
    RelationKind,
    VerificationReason,
)

if torch is not None:
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except ModuleNotFoundError as error:  # pragma: no cover - optional backbone extra
        if error.name != "transformers":
            raise
        Qwen2Config = Qwen2ForCausalLM = None
    if Qwen2Config is not None:
        from dwr.models.qwen2 import Qwen2WorkspaceLM
        from dwr.models.workspace import FixedWorkspace
    else:
        Qwen2WorkspaceLM = FixedWorkspace = None
else:
    Qwen2Config = Qwen2ForCausalLM = Qwen2WorkspaceLM = FixedWorkspace = None


def _task() -> ConstraintTask:
    return ConstraintTask(
        "constraint-task-v1", 3, ("v0", "v1", "v2"),
        (Constraint(RelationKind.EQ, ("v0", "v1")),
         Constraint(RelationKind.NEQ, ("v1", "v2"))),
        (GivenAssignment("v0", 1),),
    )


class ScriptedGreedyTests(unittest.TestCase):
    """Unit checks of the shared loop; these do not claim native-model evidence."""

    @unittest.skipIf(torch is None, "scripted logits require the optional torch dependency")
    def test_greedy_loop_uses_last_position_tie_break_and_stops_at_first_eos(self):
        prefixes: list[tuple[int, ...]] = []
        rows = [[0.0, 2.0, 2.0, -1.0], [0.0, 0.0, 0.0, 9.0]]

        def scripted(prefix, mask):
            self.assertEqual(tuple(mask.shape), (1, len(prefix[0])))
            prefixes.append(tuple(prefix[0].tolist()))
            logits = torch.zeros(1, len(prefix[0]), 4)
            logits[0, -1] = torch.tensor(rows[len(prefixes) - 1])
            return logits

        ids = torch.tensor([[1, 2]], dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)
        generated = _greedy_loop(scripted, ids, mask, vocab_size=4,
                                 eos_token_id=3, max_new_tokens=4)
        self.assertEqual(generated, (1, 3))
        self.assertEqual(prefixes, [(1, 2), (1, 2, 1)])

    @unittest.skipIf(torch is None, "scripted logits require the optional torch dependency")
    def test_greedy_loop_records_token_cap_without_eos(self):
        ids = torch.tensor([[1]], dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)

        def scripted(prefix, _mask):
            logits = torch.zeros(1, prefix.shape[1], 5)
            logits[0, -1, 2] = 1
            return logits

        self.assertEqual(
            _greedy_loop(scripted, ids, mask, vocab_size=5,
                         eos_token_id=4, max_new_tokens=2),
            (2, 2),
        )

    @unittest.skipIf(torch is None, "scripted logits require the optional torch dependency")
    def test_greedy_loop_rejects_malformed_and_nonfinite_logits(self):
        ids = torch.tensor([[1]], dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)
        callbacks = (
            lambda _ids, _mask: torch.zeros(1, 2, 5),
            lambda _ids, _mask: torch.full((1, 1, 5), math.nan),
            lambda _ids, _mask: torch.zeros(1, 1, 4),
        )
        for decode in callbacks:
            with self.subTest(decode=decode), self.assertRaises(ValueError):
                _greedy_loop(decode, ids, mask, vocab_size=5,
                             eos_token_id=4, max_new_tokens=1)

    @unittest.skipIf(torch is None, "scripted logits require the optional torch dependency")
    def test_greedy_loop_rejects_meta_inputs_and_wrong_device_logits(self):
        ids = torch.tensor([[1]], dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)
        with self.assertRaises(ValueError):
            _greedy_loop(lambda *_: torch.zeros(1, 1, 3), ids.to("meta"), mask.to("meta"),
                         vocab_size=3, eos_token_id=2, max_new_tokens=1)
        with self.assertRaises(ValueError):
            _greedy_loop(lambda *_: torch.empty(1, 1, 3, device="meta"), ids, mask,
                         vocab_size=3, eos_token_id=2, max_new_tokens=1)

    @unittest.skipIf(torch is None, "scripted logits require the optional torch dependency")
    def test_appended_token_tensor_explicitly_uses_prefix_device(self):
        ids = torch.tensor([[1]], dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)
        original_tensor = torch.tensor
        token_devices = []

        def observe_tensor(data, *args, **kwargs):
            token_devices.append(kwargs.get("device"))
            return original_tensor(data, *args, **kwargs)

        with patch.object(torch, "tensor", side_effect=observe_tensor):
            _greedy_loop(lambda prefix, _mask: torch.zeros(1, prefix.shape[1], 3),
                         ids, mask, vocab_size=3, eos_token_id=2, max_new_tokens=1)
        self.assertEqual(token_devices, [ids.device])

    @unittest.skipIf(torch is None, "device checks require the optional torch dependency")
    def test_model_tensors_on_different_cuda_indices_are_rejected_without_cuda_calls(self):
        class DeviceOnlyTensor:
            def __init__(self, device):
                self.device = torch.device(device)

        class Model:
            def parameters(self):
                return [DeviceOnlyTensor("cuda:0")]

            def buffers(self):
                return [DeviceOnlyTensor("cuda:1")]

        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")):
            with self.assertRaises(ValueError):
                _validate_model_device(Model(), label="fake model")

    def test_noise_seed_is_order_independent_and_does_not_touch_rng(self):
        examples = [hashlib.sha256(f"example-{i}".encode()).hexdigest() for i in range(3)]
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state().clone() if torch is not None else None
        forward = {item: evaluation_noise_seed(21, item) for item in examples}
        reverse = {item: evaluation_noise_seed(21, item) for item in reversed(examples)}
        self.assertEqual(forward, reverse)
        self.assertEqual(len(set(forward.values())), len(examples))
        self.assertNotEqual(evaluation_noise_seed(22, examples[0]), forward[examples[0]])
        self.assertNotEqual(evaluation_noise_seed(21, examples[0], draw_id=1), forward[examples[0]])
        self.assertEqual(random.getstate(), python_state)
        if torch_state is not None:
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_noise_seed_rejects_noncanonical_id_and_boolean_integer(self):
        valid_id = "a" * 64
        for seed in (True, -1, 1.5):
            with self.subTest(seed=seed), self.assertRaises((TypeError, ValueError)):
                evaluation_noise_seed(seed, valid_id)
        for example_id in ("A" * 64, "a" * 63, "g" * 64):
            with self.subTest(example_id=example_id), self.assertRaises(ValueError):
                evaluation_noise_seed(1, example_id)


class ScoreGenerationTests(unittest.TestCase):
    def setUp(self):
        self.task = _task()

    def test_scorer_removes_only_final_eos_and_uses_strict_decode_flags(self):
        class Tokenizer:
            def __init__(self):
                self.calls = []

            def decode(self, token_ids, **kwargs):
                self.calls.append((tuple(token_ids), kwargs))
                return '{"assignment":{"v0":1,"v1":1,"v2":0}}'

        tokenizer = Tokenizer()
        result = GenerationResult((7, 2), 2, True, False, 3, 2, 1, 4, 0.25)
        scored = score_generation(self.task, result, tokenizer)
        self.assertEqual(tokenizer.calls, [((7,), {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        })])
        self.assertEqual(scored, {
            "answer_text": '{"assignment":{"v0":1,"v1":1,"v2":0}}',
            "valid": True,
            "verifier_reason": VerificationReason.VALID.value,
            "termination": "eos",
            "generated_tokens": 2,
        })

    def test_syntactically_valid_answer_at_cap_is_still_a_failure(self):
        class Tokenizer:
            def decode(self, token_ids, **kwargs):
                self.assert_flags = kwargs
                return '{"assignment":{"v0":1,"v1":1,"v2":0}}'

        result = GenerationResult((7, 8), 2, False, True, 3, 2, 1, 4, 0.25)
        scored = score_generation(self.task, result, Tokenizer())
        self.assertFalse(scored["valid"])
        self.assertEqual(scored["verifier_reason"], VerificationReason.VALID.value)
        self.assertEqual(scored["termination"], "token_cap")

    def test_malformed_text_and_wrong_assignment_keep_verifier_diagnostics(self):
        class Tokenizer:
            def __init__(self, answer):
                self.answer = answer

            def decode(self, token_ids, **kwargs):
                return self.answer

        base = dict(eos_token_id=2, ended_with_eos=True, truncated=False,
                    prompt_tokens=3, decoder_calls=1, prompt_encoder_calls=0,
                    workspace_updates=0, elapsed_wall_seconds=0.0)
        malformed = score_generation(
            self.task, GenerationResult(token_ids=(2,), **base), Tokenizer("not json")
        )
        incorrect = score_generation(
            self.task, GenerationResult(token_ids=(2,), **base),
            Tokenizer('{"assignment":{"v0":0,"v1":0,"v2":1}}'),
        )
        self.assertEqual(malformed["verifier_reason"], VerificationReason.MALFORMED_JSON.value)
        self.assertEqual(incorrect["verifier_reason"], VerificationReason.GIVEN_MISMATCH.value)
        self.assertFalse(malformed["valid"])
        self.assertFalse(incorrect["valid"])

    def test_result_metadata_invariants_are_strict(self):
        good = dict(token_ids=(1, 2), eos_token_id=2, ended_with_eos=True,
                    truncated=False, prompt_tokens=3, decoder_calls=2,
                    prompt_encoder_calls=0, workspace_updates=0,
                    elapsed_wall_seconds=0.1)
        bad_values = [
            {**good, "token_ids": (True, 2)},
            {**good, "token_ids": (1, 2, 2)},
            {**good, "decoder_calls": 1},
            {**good, "truncated": True},
            {**good, "elapsed_wall_seconds": math.inf},
        ]
        for values in bad_values:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                GenerationResult(**values)

    def test_scorer_rechecks_result_metadata_after_construction(self):
        class Tokenizer:
            def decode(self, *_args, **_kwargs):
                return "{}"

        result = GenerationResult((1,), 2, False, True, 3, 1, 0, 0, 0.0)
        object.__setattr__(result, "decoder_calls", 7)
        with self.assertRaises(ValueError):
            score_generation(self.task, result, Tokenizer())


@unittest.skipIf(torch is None or Qwen2WorkspaceLM is None,
                 "install the optional backbone extra")
class NativeGreedyTests(unittest.TestCase):
    atol = 1e-6
    rtol = 1e-6

    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def make_backbone(self):
        config = Qwen2Config(
            vocab_size=17, hidden_size=12, intermediate_size=24,
            num_hidden_layers=3, num_attention_heads=3,
            num_key_value_heads=1, attention_dropout=0.0,
            tie_word_embeddings=True, pad_token_id=16, eos_token_id=15,
            _attn_implementation="eager",
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(701)
            model = Qwen2ForCausalLM(config).to(dtype=torch.float64)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model.eval()

    def make_host(self):
        from dwr.models.workspace import FixedWorkspace

        backbone = self.make_backbone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(709)
            workspace = FixedWorkspace(
                12, num_slots=2, latent_dim=8, num_heads=2,
                max_steps=3, slot_identity="exchangeable",
            ).to(dtype=torch.float64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(719)
            host = Qwen2WorkspaceLM(
                backbone, workspace, prompt_layer=1,
                adapter_layers=(0, 1, 2), reader_heads=3,
            )
        host.backbone.config.output_hidden_states = True
        host.backbone.config.output_attentions = True
        host.backbone.config.use_cache = True
        return host.eval()

    def sample(self):
        ids = torch.tensor([[1, 3, 4]], dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)
        noise = torch.randn(1, 2, 8, generator=torch.Generator().manual_seed(727),
                            dtype=torch.float64)
        return ids, mask, noise

    def test_workspace_generation_matches_independent_manual_native_loop_and_reuses_x_z(self):
        host = self.make_host()
        prompt_ids, prompt_mask, noise = self.sample()
        expected_prompt = host.encode_prompt(prompt_ids, prompt_mask)
        expected_x = expected_prompt.to(dtype=next(host.workspace.parameters()).dtype)
        with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
            expected_z = host.workspace(expected_x, noise=noise, steps=2)
        prefix = prompt_ids.clone()
        expected_tokens = []
        for _ in range(3):
            with torch.no_grad():
                logits = host.decode_with_workspace(
                    prefix, torch.ones_like(prefix, dtype=torch.bool), expected_z
                )
            token = int(logits[0, -1].argmax().item())
            expected_tokens.append(token)
            prefix = torch.cat((prefix, torch.tensor([[token]], dtype=torch.long)), dim=1)
            if token == 15:
                break

        before_rng = torch.random.get_rng_state().clone()
        before_python_rng = random.getstate()
        before_parameters = [parameter.detach().clone() for parameter in host.parameters()]
        before_grad_flags = [parameter.requires_grad for parameter in host.parameters()]
        before_modes = [module.training for module in host.modules()]
        before_config = (host.backbone.config.return_dict,
                         host.backbone.config.output_hidden_states,
                         host.backbone.config.output_attentions,
                         host.backbone.config.use_cache)
        before_noise = noise.clone()
        counts = {"x": 0, "z": 0, "decode": 0}
        seen_workspaces = []
        seen_prefix_lengths = []
        encode = host.encode_prompt
        workspace_forward = host.workspace.forward
        decode = host.decode_with_workspace

        def count_encode(*args, **kwargs):
            counts["x"] += 1
            return encode(*args, **kwargs)

        def count_workspace(*args, **kwargs):
            counts["z"] += 1
            return workspace_forward(*args, **kwargs)

        def count_decode(ids, mask, workspace):
            counts["decode"] += 1
            seen_workspaces.append(workspace.detach().clone())
            seen_prefix_lengths.append(ids.shape[1])
            return decode(ids, mask, workspace)

        host.encode_prompt = count_encode
        host.workspace.forward = count_workspace
        host.decode_with_workspace = count_decode
        try:
            result = greedy_workspace(
                host, prompt_ids, prompt_mask, noise=noise,
                steps=2, eos_token_id=15, max_new_tokens=3, max_total_tokens=6,
            )
        finally:
            host.encode_prompt = encode
            host.workspace.forward = workspace_forward
            host.decode_with_workspace = decode
        self.assertEqual(result.token_ids, tuple(expected_tokens))
        self.assertEqual(counts, {"x": 1, "z": 1, "decode": len(expected_tokens)})
        self.assertEqual(seen_prefix_lengths, list(range(prompt_ids.shape[1],
                                                         prompt_ids.shape[1] + len(expected_tokens))))
        self.assertEqual(len(seen_workspaces), len(expected_tokens))
        self.assertTrue(all(torch.equal(seen_workspaces[0], item) for item in seen_workspaces))
        self.assertEqual(result.prompt_encoder_calls, 1)
        self.assertEqual(result.workspace_updates, 2)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))
        self.assertEqual(random.getstate(), before_python_rng)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_parameters, host.parameters())))
        self.assertEqual(before_grad_flags, [parameter.requires_grad for parameter in host.parameters()])
        self.assertEqual(before_modes, [module.training for module in host.modules()])
        self.assertEqual(before_config, (host.backbone.config.return_dict,
                                         host.backbone.config.output_hidden_states,
                                         host.backbone.config.output_attentions,
                                         host.backbone.config.use_cache))
        self.assertTrue(torch.equal(noise, before_noise))
        self.assertFalse(host.training)

    def test_frozen_generation_matches_manual_full_prefix_native_forwards(self):
        backbone = self.make_backbone()
        prompt_ids, prompt_mask, _ = self.sample()
        prefix = prompt_ids.clone()
        expected = []
        for _ in range(3):
            with torch.no_grad():
                output = backbone(
                    input_ids=prefix, attention_mask=torch.ones_like(prefix, dtype=torch.bool),
                    use_cache=False, output_hidden_states=False,
                    output_attentions=False, logits_to_keep=0, return_dict=True,
                )
            token = int(output.logits[0, -1].argmax().item())
            expected.append(token)
            prefix = torch.cat((prefix, torch.tensor([[token]], dtype=torch.long)), dim=1)
            if token == 15:
                break
        before_rng = torch.random.get_rng_state().clone()
        before_parameters = [parameter.detach().clone() for parameter in backbone.parameters()]
        result = greedy_frozen(
            backbone, prompt_ids, prompt_mask, eos_token_id=15,
            max_new_tokens=3, max_total_tokens=6,
        )
        self.assertEqual(result.token_ids, tuple(expected))
        self.assertEqual(result.prompt_encoder_calls, 0)
        self.assertEqual(result.workspace_updates, 0)
        self.assertEqual(result.decoder_calls, len(expected))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_parameters, backbone.parameters())))

    def test_budget_check_runs_at_entry_and_each_decode_boundary(self):
        backbone = self.make_backbone()
        prompt_ids, prompt_mask, _ = self.sample()
        events = []
        original_forward = backbone.forward

        def observe_forward(*args, **kwargs):
            events.append("decode")
            return original_forward(*args, **kwargs)

        backbone.forward = observe_forward
        result = greedy_frozen(
            backbone, prompt_ids, prompt_mask, eos_token_id=15,
            max_new_tokens=2, max_total_tokens=5,
            check=lambda: events.append("check"),
        )
        self.assertEqual(events, ["check", "check", "decode", "check", "decode"])
        self.assertEqual(result.decoder_calls, 2)

    def test_budget_check_failure_after_decode_propagates_without_changing_model_state(self):
        backbone = self.make_backbone()
        prompt_ids, prompt_mask, _ = self.sample()
        events = []
        parameters_before = [p.detach().clone() for p in backbone.parameters()]

        def check():
            events.append("check")
            if len(events) == 3:
                raise RuntimeError("budget exhausted")

        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            greedy_frozen(backbone, prompt_ids, prompt_mask, eos_token_id=15,
                          max_new_tokens=2, max_total_tokens=5, check=check)
        self.assertEqual(events, ["check", "check", "check"])
        self.assertTrue(all(torch.equal(before, after) for before, after in
                            zip(parameters_before, backbone.parameters())))

    def test_workspace_budget_failure_cleans_native_hooks(self):
        host = self.make_host()
        prompt_ids, prompt_mask, noise = self.sample()
        original_decode = host.decode_with_workspace

        def non_eos_decode(*args, **kwargs):
            logits = original_decode(*args, **kwargs).clone()
            logits[..., 15] = -100
            return logits

        host.decode_with_workspace = non_eos_decode
        layers = host.backbone.model.layers
        hooks_before = [len(layer._forward_hooks) for layer in layers]
        checks = []

        def check():
            checks.append(None)
            if len(checks) == 3:
                raise RuntimeError("budget exhausted")

        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            greedy_workspace(host, prompt_ids, prompt_mask, noise=noise, steps=1,
                             eos_token_id=15, max_new_tokens=2, max_total_tokens=5,
                             check=check)
        host.decode_with_workspace = original_decode
        self.assertEqual(len(checks), 3)
        self.assertEqual([len(layer._forward_hooks) for layer in layers], hooks_before)

    def test_cpu_generation_does_not_query_or_initialize_cuda(self):
        backbone = self.make_backbone()
        prompt_ids, prompt_mask, _ = self.sample()
        with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA queried")), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")), \
             patch.object(torch.cuda, "current_device", side_effect=AssertionError("CUDA queried")):
            result = greedy_frozen(backbone, prompt_ids, prompt_mask, eos_token_id=15,
                                   max_new_tokens=1, max_total_tokens=4)
        self.assertEqual(result.decoder_calls, 1)

    def test_workspace_rejects_meta_noise_and_meta_model_buffer_before_tensor_values(self):
        host = self.make_host()
        prompt_ids, prompt_mask, noise = self.sample()
        with self.assertRaises(ValueError):
            greedy_workspace(host, prompt_ids, prompt_mask, noise=noise.to("meta"),
                             steps=1, eos_token_id=15, max_new_tokens=1,
                             max_total_tokens=4)
        with self.assertRaises(ValueError):
            greedy_workspace(host, prompt_ids.to("meta"), prompt_mask.to("meta"), noise=noise,
                             steps=1, eos_token_id=15, max_new_tokens=1,
                             max_total_tokens=4)
        host.register_buffer("meta_probe", torch.empty(1, device="meta"))
        with self.assertRaises(ValueError):
            greedy_workspace(host, prompt_ids, prompt_mask, noise=noise,
                             steps=1, eos_token_id=15, max_new_tokens=1,
                             max_total_tokens=4)

    def test_workspace_checks_all_devices_before_any_noise_value_check(self):
        host = self.make_host()
        prompt_ids, prompt_mask, noise = self.sample()
        original_isfinite = torch.isfinite
        checked = []

        def observe_isfinite(value, *args, **kwargs):
            checked.append(value.device)
            return original_isfinite(value, *args, **kwargs)

        cases = (
            (prompt_ids.to("meta"), prompt_mask.to("meta"), noise),
            (prompt_ids, prompt_mask.to("meta"), noise),
            (prompt_ids, prompt_mask, noise.to("meta")),
        )
        for ids, mask, sample_noise in cases:
            checked.clear()
            with self.subTest(ids=ids.device, mask=mask.device, noise=sample_noise.device):
                with patch.object(torch, "isfinite", side_effect=observe_isfinite):
                    with self.assertRaises(ValueError):
                        greedy_workspace(host, ids, mask, noise=sample_noise, steps=1,
                                         eos_token_id=15, max_new_tokens=1,
                                         max_total_tokens=4)
                self.assertEqual(checked, [])

    def test_public_checks_must_be_callable(self):
        backbone = self.make_backbone()
        prompt_ids, prompt_mask, _ = self.sample()
        with self.assertRaises(TypeError):
            greedy_frozen(backbone, prompt_ids, prompt_mask, eos_token_id=15,
                          max_new_tokens=1, max_total_tokens=4, check=1)

    def test_mocked_cuda_timing_syncs_actual_device_and_uses_device_type_policy(self):
        device = torch.device("cuda:3")
        with patch.object(torch.cuda, "synchronize") as synchronize:
            _synchronize_device(torch, device)
            synchronize.assert_called_once_with(device)
            synchronize.reset_mock()
            _synchronize_device(torch, torch.device("cpu"))
            synchronize.assert_not_called()
        with patch.object(torch, "autocast", return_value=nullcontext()) as autocast:
            with _autocast_off(torch, device):
                pass
            autocast.assert_called_once_with(device_type="cuda", enabled=False)

    def test_forced_native_decoder_exception_cleans_hooks_and_restores_config(self):
        host = self.make_host()
        prompt_ids, prompt_mask, noise = self.sample()
        host.backbone.config.return_dict = False
        original_return_dict = host.backbone.config.return_dict
        layer = host.backbone.model.layers[1]
        original_forward = layer.forward
        original_encode = host.encode_prompt
        original_decode = host.decode_with_workspace
        original_backbone_forward = host.backbone.forward
        original_model_forward = host.backbone.model.forward
        hooks_before = [len(layer._forward_hooks) for layer in host.backbone.model.layers]
        modes_before = [module.training for module in host.modules()]
        python_rng_before = random.getstate()
        torch_rng_before = torch.random.get_rng_state().clone()
        parameters_before = [parameter.detach().clone() for parameter in host.parameters()]
        stage = {"prompt_completed": False, "decoder_started": False,
                 "reader_hook_present": False}

        def observe_encode(*args, **kwargs):
            result = original_encode(*args, **kwargs)
            stage["prompt_completed"] = True
            return result

        def observe_decode(*args, **kwargs):
            stage["decoder_started"] = True
            return original_decode(*args, **kwargs)

        def fail(*args, **kwargs):
            if stage["prompt_completed"]:
                stage["reader_hook_present"] = len(layer._forward_hooks) > hooks_before[1]
                raise RuntimeError("forced native decoder failure")
            return original_forward(*args, **kwargs)

        host.encode_prompt = observe_encode
        host.decode_with_workspace = observe_decode
        layer.forward = fail
        try:
            with self.assertRaisesRegex(RuntimeError, "forced native decoder failure"):
                greedy_workspace(
                    host, prompt_ids, prompt_mask, noise=noise,
                    steps=1, eos_token_id=15, max_new_tokens=1, max_total_tokens=4,
                )
        finally:
            host.encode_prompt = original_encode
            host.decode_with_workspace = original_decode
            layer.forward = original_forward
        self.assertEqual(stage, {"prompt_completed": True, "decoder_started": True,
                                 "reader_hook_present": True})
        self.assertEqual([len(layer._forward_hooks) for layer in host.backbone.model.layers], hooks_before)
        self.assertEqual(host.backbone.forward, original_backbone_forward)
        self.assertEqual(host.backbone.model.forward, original_model_forward)
        self.assertIs(layer.forward, original_forward)
        self.assertIs(host.encode_prompt, original_encode)
        self.assertIs(host.decode_with_workspace, original_decode)
        self.assertEqual(host.backbone.config.return_dict, original_return_dict)
        self.assertEqual([module.training for module in host.modules()], modes_before)
        self.assertEqual(random.getstate(), python_rng_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng_before))
        self.assertTrue(all(torch.equal(before, after) for before, after in
                            zip(parameters_before, host.parameters())))
        retry = greedy_workspace(
            host, prompt_ids, prompt_mask, noise=noise,
            steps=1, eos_token_id=15, max_new_tokens=1, max_total_tokens=4,
        )
        self.assertEqual(retry.decoder_calls, 1)

    def test_invalid_cpu_shapes_caps_eos_and_modes_fail_closed(self):
        host = self.make_host()
        prompt_ids, prompt_mask, noise = self.sample()
        common = dict(noise=noise, steps=1, eos_token_id=15,
                      max_new_tokens=2, max_total_tokens=5)
        with self.assertRaises(ValueError):
            greedy_workspace(host, prompt_ids, prompt_mask, **{**common, "eos_token_id": 17})
        with self.assertRaises(ValueError):
            greedy_workspace(host, prompt_ids, prompt_mask, **{**common, "max_total_tokens": 4})
        with self.assertRaises(ValueError):
            greedy_workspace(host, prompt_ids, prompt_mask, **{**common, "steps": 4})
        with self.assertRaises(ValueError):
            greedy_workspace(host, torch.tensor([[1, 15]]), prompt_mask,
                             **{**common, "max_total_tokens": 5})
        with self.assertRaises(ValueError):
            greedy_workspace(host, torch.tensor([[1, 16, 4]]), prompt_mask, **common)
        with self.assertRaises(ValueError):
            greedy_workspace(host, prompt_ids, torch.tensor([[True, False, True]]), **common)
        with self.assertRaises(ValueError):
            greedy_workspace(host, prompt_ids, prompt_mask,
                             **{**common, "noise": torch.full_like(noise, math.nan)})
        with self.assertRaises(ValueError):
            greedy_workspace(host, prompt_ids, prompt_mask,
                             **{**common, "noise": torch.zeros(1, 2, 7, dtype=torch.float64)})
        host.train()
        with self.assertRaises(ValueError):
            greedy_workspace(host, prompt_ids, prompt_mask, **common)

    def test_frozen_path_requires_frozen_eval_native_model_and_return_dict(self):
        backbone = self.make_backbone()
        prompt_ids, prompt_mask, _ = self.sample()
        args = dict(eos_token_id=15, max_new_tokens=1, max_total_tokens=4)
        backbone.config.return_dict = False
        with self.assertRaises(ValueError):
            greedy_frozen(backbone, prompt_ids, prompt_mask, **args)
        backbone.config.return_dict = True
        parameter = next(backbone.parameters())
        parameter.requires_grad_(True)
        with self.assertRaises(ValueError):
            greedy_frozen(backbone, prompt_ids, prompt_mask, **args)
        parameter.requires_grad_(False)
        backbone.train()
        with self.assertRaises(ValueError):
            greedy_frozen(backbone, prompt_ids, prompt_mask, **args)

    def test_score_requires_actual_task_result_and_string_decode(self):
        class BadTokenizer:
            def decode(self, *_args, **_kwargs):
                return 17

        result = GenerationResult((1,), 2, False, True, 3, 1, 0, 0, 0.0)
        with self.assertRaises(TypeError):
            score_generation(object(), result, BadTokenizer())
        with self.assertRaises(TypeError):
            score_generation(_task(), result, object())
        with self.assertRaises(TypeError):
            score_generation(_task(), result, BadTokenizer())


if __name__ == "__main__":
    unittest.main()
