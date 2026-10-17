"""CPU contract tests for the actual pinned Qwen2 host."""

from __future__ import annotations

import unittest
import pathlib
import subprocess
import sys
import textwrap

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - only optional dependency absence
    if error.name != "torch":
        raise
    torch = None
if torch is not None:
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except ModuleNotFoundError as error:  # pragma: no cover - only optional dependency absence
        if error.name != "transformers":
            raise
        Qwen2Config = Qwen2ForCausalLM = None
    if Qwen2Config is not None:
        from dwr.models.qwen2 import Qwen2WorkspaceLM
    else:
        Qwen2WorkspaceLM = None
else:
    Qwen2Config = Qwen2ForCausalLM = Qwen2WorkspaceLM = None


@unittest.skipIf(
    torch is None or Qwen2WorkspaceLM is None,
    "install the optional backbone extra",
)
class Qwen2WorkspaceTests(unittest.TestCase):
    atol = 1e-6
    rtol = 1e-6

    @classmethod
    def setUpClass(cls) -> None:
        cls.original_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_num_threads)

    def make_host(self, mode: str = "exchangeable"):
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=12,
            intermediate_size=24,
            num_hidden_layers=3,
            num_attention_heads=3,
            num_key_value_heads=1,
            attention_dropout=0.0,
            tie_word_embeddings=True,
            pad_token_id=31,
            eos_token_id=31,
            _attn_implementation="eager",
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            backbone = Qwen2ForCausalLM(config).to(dtype=torch.float64)
        self.assertEqual(backbone.config._attn_implementation, "eager")
        backbone.config.output_hidden_states = True
        backbone.config.output_attentions = True
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(89)
            workspace = __import__("dwr.models.workspace", fromlist=["FixedWorkspace"]).FixedWorkspace(
                12, num_slots=4, latent_dim=8, num_heads=2, max_steps=4,
                slot_identity=mode,
            ).to(dtype=torch.float64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101)
            host = Qwen2WorkspaceLM(
                backbone, workspace, prompt_layer=1,
                adapter_layers=(0, 1, 2), reader_heads=3,
            )
        return host

    def batch(self):
        prompt_ids = torch.tensor([[1, 2, 3], [4, 5, 31]], dtype=torch.long)
        prompt_mask = torch.tensor([[True, True, True], [True, True, False]])
        decoder_ids = torch.tensor([[1, 2, 3, 6], [4, 5, 7, 31]], dtype=torch.long)
        decoder_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
        return prompt_ids, prompt_mask, decoder_ids, decoder_mask

    def noise(self, batch_size=2):
        return torch.randn(batch_size, 4, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(8))

    def layer_hooks(self, host):
        return [len(layer._forward_hooks) for layer in host.backbone.model.layers]

    def test_prompt_capture_is_answer_independent_and_matches_raw_layer_output(self):
        host = self.make_host()
        prompt_ids, prompt_mask, decoder_ids, decoder_mask = self.batch()
        layer_functions = [layer.forward.__func__ for layer in host.backbone.model.layers]
        captured = []
        layer = host.backbone.model.layers[0]
        layer_forward = layer.forward
        def capture_forward(*args, **kwargs):
            output = layer_forward(*args, **kwargs)
            captured.append(output.detach().clone())
            return output
        layer.forward = capture_forward
        try:
            x = host.encode_prompt(prompt_ids, prompt_mask)
        finally:
            layer.forward = layer_forward
        self.assertTrue(torch.equal(x, captured[0]))
        altered = decoder_ids.clone()
        altered[:, -1] = torch.tensor([9, 10])
        fixed_noise = self.noise()
        host(prompt_ids, prompt_mask, decoder_ids, decoder_mask, noise=fixed_noise)
        host(prompt_ids, prompt_mask, altered, decoder_mask, noise=fixed_noise)
        self.assertTrue(torch.equal(x, host.encode_prompt(prompt_ids, prompt_mask)))
        self.assertFalse(x.requires_grad)
        self.assertEqual(layer_functions, [layer.forward.__func__ for layer in host.backbone.model.layers])

    def test_matched_ex_idx_host_initialization_preserves_global_rng(self):
        before_rng = torch.random.get_rng_state().clone()
        exchangeable = self.make_host("exchangeable")
        indexed = self.make_host("indexed")
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
        for name, value in exchangeable.backbone.state_dict().items():
            self.assertTrue(torch.equal(value, indexed.backbone.state_dict()[name]), name)
        for name, value in exchangeable.readers.state_dict().items():
            self.assertTrue(torch.equal(value, indexed.readers.state_dict()[name]), name)
        indexed_workspace = indexed.workspace.state_dict()
        for name, value in exchangeable.workspace.state_dict().items():
            self.assertTrue(torch.equal(value, indexed_workspace[name]), name)

    def test_all_learned_parameters_can_be_frozen_for_eval_without_flag_changes(self):
        host = self.make_host()
        for parameter in (*host.workspace.parameters(), *host.readers.parameters()):
            parameter.requires_grad_(False)
        host.eval()
        pids, pmask, dids, dmask = self.batch()
        flags_before = [parameter.requires_grad for parameter in host.parameters()]
        first = host(pids, pmask, dids, dmask, noise=self.noise())
        second = host(pids, pmask, dids, dmask, noise=self.noise())
        self.assertTrue(torch.equal(first.logits, second.logits))
        self.assertFalse(first.logits.requires_grad)
        self.assertEqual(flags_before, [parameter.requires_grad for parameter in host.parameters()])

    def test_k0_frozen_workspace_keeps_reader_trainable_and_gradient_active(self):
        host = self.make_host()
        for parameter in host.workspace.parameters():
            parameter.requires_grad_(False)
        pids, pmask, dids, dmask = self.batch()
        result = host(pids, pmask, dids, dmask, noise=self.noise(), steps=0)
        result.logits.square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in host.workspace.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in host.workspace.parameters()))
        reader_parameters = list(host.readers.parameters())
        self.assertTrue(all(parameter.requires_grad for parameter in reader_parameters))
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in reader_parameters
        ))

    def test_native_calls_force_return_dict_even_when_config_disables_it(self):
        host = self.make_host()
        host.backbone.config.return_dict = False
        pids, pmask, dids, dmask = self.batch()
        encoded = host.encode_prompt(pids, pmask)
        logits = host.decode_with_workspace(dids, dmask, self.noise())
        output = host(pids, pmask, dids, dmask, noise=self.noise())
        self.assertEqual(encoded.shape, (2, 3, 12))
        self.assertEqual(logits.shape, (2, 4, 32))
        self.assertEqual(output.logits.shape, (2, 4, 32))
        self.assertFalse(host.backbone.config.return_dict)

    def test_original_backbone_unfreeze_is_still_rejected(self):
        host = self.make_host()
        next(host.backbone.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "original Qwen2 parameters must remain frozen"):
            host.encode_prompt(*self.batch()[:2])

    def test_prompt_adapters_are_not_active_and_endpoint_is_raw(self):
        host = self.make_host()
        prompt_ids, prompt_mask, _, _ = self.batch()
        baseline = host.encode_prompt(prompt_ids, prompt_mask)
        for reader in host.readers.values():
            with torch.no_grad():
                reader.gate.fill_(9.0)
        self.assertTrue(torch.equal(baseline, host.encode_prompt(prompt_ids, prompt_mask)))
        endpoint = self.make_host()
        endpoint.prompt_layer = 3
        endpoint_raw = []
        layer = endpoint.backbone.model.layers[2]
        layer_forward = layer.forward
        def capture_endpoint(*args, **kwargs):
            output = layer_forward(*args, **kwargs)
            endpoint_raw.append(output.detach().clone())
            return output
        layer.forward = capture_endpoint
        try:
            actual = endpoint.encode_prompt(prompt_ids, prompt_mask)
        finally:
            layer.forward = layer_forward
        self.assertTrue(torch.equal(actual, endpoint_raw[0]))

    def test_causal_future_decoder_change_preserves_earlier_logits(self):
        host = self.make_host()
        prompt_ids, prompt_mask, decoder_ids, decoder_mask = self.batch()
        z = self.noise()
        first = host(prompt_ids, prompt_mask, decoder_ids, decoder_mask, noise=z).logits
        changed = decoder_ids.clone()
        changed[:, -1] = torch.tensor([11, 12])
        second = host(prompt_ids, prompt_mask, changed, decoder_mask, noise=z).logits
        self.assertTrue(torch.allclose(first[:, :-1], second[:, :-1], atol=self.atol, rtol=self.rtol))
        self.assertFalse(torch.allclose(first[:, -1], second[:, -1], atol=self.atol, rtol=self.rtol))

    def test_padding_and_batching_preserve_valid_states_and_logits(self):
        host = self.make_host()
        prompt_ids, prompt_mask, decoder_ids, decoder_mask = self.batch()
        batched = host(prompt_ids, prompt_mask, decoder_ids, decoder_mask, noise=self.noise())
        for row, plen, dlen in ((0, 3, 4), (1, 2, 3)):
            single = host(
                prompt_ids[row:row+1, :plen], torch.ones(1, plen, dtype=torch.bool),
                decoder_ids[row:row+1, :dlen], torch.ones(1, dlen, dtype=torch.bool),
                noise=self.noise()[row:row+1],
            )
            self.assertTrue(torch.allclose(batched.workspace[row], single.workspace[0], atol=self.atol, rtol=self.rtol), f"workspace row={row}, maxdiff={(batched.workspace[row] - single.workspace[0]).abs().max().item()}")
            self.assertTrue(torch.allclose(batched.logits[row, :dlen], single.logits[0], atol=self.atol, rtol=self.rtol), f"row={row}, maxdiff={(batched.logits[row, :dlen] - single.logits[0]).abs().max().item()}")

    def test_complete_workspace_permutation_preserves_logits_for_ex_and_idx(self):
        for mode in ("exchangeable", "indexed"):
            with self.subTest(mode=mode):
                host = self.make_host(mode)
                prompt_ids, prompt_mask, decoder_ids, decoder_mask = self.batch()
                output = host(prompt_ids, prompt_mask, decoder_ids, decoder_mask, noise=self.noise())
                permuted = host.decode_with_workspace(
                    decoder_ids, decoder_mask, output.workspace[:, [2, 0, 3, 1]]
                )
                self.assertTrue(torch.allclose(output.logits, permuted, atol=self.atol, rtol=self.rtol))

    def test_k4_all_states_and_new_parameter_groups_update_while_backbone_stays_frozen(self):
        for mode in ("exchangeable", "indexed"):
            with self.subTest(mode=mode):
                host = self.make_host(mode)
                prompt_ids, prompt_mask, decoder_ids, decoder_mask = self.batch()
                before = [p.detach().clone() for p in host.backbone.parameters()]
                optimized = [p for p in host.parameters() if p.requires_grad]
                initial = [p.detach().clone() for p in optimized]
                noise = self.noise().requires_grad_()
                result = host(prompt_ids, prompt_mask, decoder_ids, decoder_mask, noise=noise, steps=4, return_states=True)
                for state in result.states:
                    state.retain_grad()
                result.logits.square().mean().backward()
                self.assertEqual(len(result.states), 5)
                for state_index, state in enumerate(result.states):
                    if state_index:
                        self.assertIsNotNone(state.grad_fn)
                    self.assertIsNotNone(state.grad)
                    self.assertTrue(torch.isfinite(state.grad).all())
                self.assertIsNotNone(noise.grad)
                self.assertTrue(torch.isfinite(noise.grad).all() and noise.grad.abs().sum() > 0)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in optimized))
                torch.optim.SGD(optimized, lr=1e-4).step()
                self.assertTrue(all(not torch.equal(a, b) for a, b in zip(initial, optimized)))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, host.backbone.parameters())))

    def test_invalid_inputs_fail_strictly(self):
        host = self.make_host()
        pids, pmask, dids, dmask = self.batch()
        cases = [
            (pids.to(torch.int32), pmask, dids, dmask),
            (pids, pmask.to(torch.int64), dids, dmask),
            (pids, torch.tensor([[True, False, True], [True, True, False]]), dids, dmask),
            (pids, pmask, dids, torch.tensor([[True, False, True, False], [True, True, True, False]])),
        ]
        for args in cases:
            with self.subTest(args=args[0].dtype):
                with self.assertRaises((TypeError, ValueError)):
                    host(*args, noise=self.noise())
        bad = dids.clone()
        bad[0, 0] = 32
        with self.assertRaises(ValueError):
            host(pids, pmask, bad, dmask, noise=self.noise())
        with self.assertRaises(TypeError):
            host(pids, pmask, dids, dmask, noise=self.noise(), mystery=True)
        with self.assertRaises(ValueError):
            host.decode_with_workspace(dids, dmask, torch.zeros(2, 3, 8, dtype=torch.float64))

    def test_prompt_failure_removes_capture_hook_and_clears_guard(self):
        host = self.make_host()
        pids, pmask, dids, dmask = self.batch()
        layers = host.backbone.model.layers
        counts = self.layer_hooks(host)
        host.backbone.config.return_dict = False
        original_model_forward = host.backbone.model.forward
        failing_layer = layers[1]
        original_layer_forward = failing_layer.forward
        def fail(*args, **kwargs):
            raise RuntimeError("forced prompt failure")
        failing_layer.forward = fail
        try:
            with self.assertRaisesRegex(RuntimeError, "forced prompt failure"):
                host(pids, pmask, dids, dmask, noise=self.noise())
        finally:
            failing_layer.forward = original_layer_forward
        self.assertEqual(self.layer_hooks(host), counts)
        self.assertEqual(host.backbone.model.forward, original_model_forward)
        self.assertEqual(failing_layer.forward, original_layer_forward)
        self.assertFalse(host.backbone.config.return_dict)
        host(pids, pmask, dids, dmask, noise=self.noise())
        self.assertFalse(host.backbone.config.return_dict)

    def test_decoder_failure_removes_reader_hooks_and_clears_guard(self):
        host = self.make_host()
        pids, pmask, dids, dmask = self.batch()
        counts = self.layer_hooks(host)
        layer = host.backbone.model.layers[1]
        host.backbone.config.return_dict = False
        original_model_forward = host.backbone.model.forward
        original = layer.forward
        def fail(*args, **kwargs):
            raise RuntimeError("forced decoder failure")
        layer.forward = fail
        try:
            with self.assertRaisesRegex(RuntimeError, "forced decoder failure"):
                host.decode_with_workspace(dids, dmask, self.noise())
        finally:
            layer.forward = original
        self.assertEqual(self.layer_hooks(host), counts)
        self.assertEqual(host.backbone.model.forward, original_model_forward)
        self.assertEqual(layer.forward, original)
        self.assertFalse(host.backbone.config.return_dict)
        host.decode_with_workspace(dids, dmask, self.noise())
        self.assertFalse(host.backbone.config.return_dict)

    def test_forward_rejects_existing_hooks_and_reentrancy(self):
        host = self.make_host()
        pids, pmask, dids, dmask = self.batch()
        foreign = host.backbone.model.layers[0].register_forward_hook(lambda *_: None)
        try:
            with self.assertRaises(ValueError):
                host(pids, pmask, dids, dmask, noise=self.noise())
        finally:
            foreign.remove()
        host._execution_lock.acquire()
        try:
            with self.assertRaises(RuntimeError):
                host.encode_prompt(pids, pmask)
        finally:
            host._execution_lock.release()


class OptionalBackboneImportTests(unittest.TestCase):
    module_path = pathlib.Path(__file__).resolve()
    project_root = module_path.parents[1]

    def run_child(self, script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=self.project_root,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_simulated_transformers_absence_skips_native_class_cases(self):
        result = self.run_child(
            f"""
            import importlib.util, pathlib, sys, types, unittest
            fake_torch = types.ModuleType('torch')
            fake_torch.get_num_threads = lambda: 1
            fake_torch.set_num_threads = lambda count: None
            sys.modules['torch'] = fake_torch
            sys.modules['transformers'] = None
            path = pathlib.Path({str(self.module_path)!r})
            spec = importlib.util.spec_from_file_location('isolated_qwen2_test_module', path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.Qwen2WorkspaceTests)
            result = unittest.TextTestRunner(verbosity=0).run(suite)
            print(f'SIMULATED_TORCH_PRESENT_TRANSFORMERS_ABSENT tests={{result.testsRun}} skipped={{len(result.skipped)}} errors={{len(result.errors)}} failures={{len(result.failures)}}')
            if not result.wasSuccessful() or result.testsRun <= 0 or len(result.skipped) != result.testsRun:
                raise SystemExit(1)
            """
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "SIMULATED_TORCH_PRESENT_TRANSFORMERS_ABSENT tests=15 skipped=15 errors=0 failures=0",
            result.stdout,
        )

    def test_broken_transformers_api_import_error_propagates(self):
        result = self.run_child(
            f"""
            import importlib.util, pathlib, sys, types
            fake_torch = types.ModuleType('torch')
            fake_torch.get_num_threads = lambda: 1
            fake_torch.set_num_threads = lambda count: None
            sys.modules['torch'] = fake_torch
            sys.modules['transformers'] = types.ModuleType('transformers')
            path = pathlib.Path({str(self.module_path)!r})
            spec = importlib.util.spec_from_file_location('isolated_qwen2_test_module', path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
            except ImportError as error:
                print(f'EXPECTED_IMPORT_ERROR:{{type(error).__name__}}:{{error}}')
            else:
                raise SystemExit('broken Transformers API was silently accepted')
            """
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("EXPECTED_IMPORT_ERROR:ImportError:", result.stdout)


if __name__ == "__main__":
    unittest.main()
