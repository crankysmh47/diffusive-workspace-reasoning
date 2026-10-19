"""Behavioral checks for the native K0/K1/K4 source-control primitive."""

from __future__ import annotations

import hashlib
import copy
import math
import random
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "torch", "transformers"}:
        raise
    torch = Qwen2Config = Qwen2ForCausalLM = None

if torch is not None and Qwen2Config is not None:
    from dwr.models.qwen2 import Qwen2WorkspaceLM
    from dwr.models.workspace import FixedWorkspace
    from dwr.training.batches import SourceExample, TokenizerIdentity, collate_source_examples
    from dwr.training.controls import configure_source_control, source_control_step, validate_source_control
    from dwr.training.checkpoints import load_training_checkpoint, save_training_checkpoint
    from dwr.training.sampler import SourceSampler
    from test_training_batch_loss import AsciiTokenizer, fixture_task
else:
    Qwen2WorkspaceLM = FixedWorkspace = SourceExample = TokenizerIdentity = None
    collate_source_examples = configure_source_control = source_control_step = validate_source_control = None
    fixture_task = AsciiTokenizer = None
    load_training_checkpoint = save_training_checkpoint = SourceSampler = None


def _host(seed: int = 1) -> Qwen2WorkspaceLM:
    config = Qwen2Config(vocab_size=128, hidden_size=12, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=3, num_key_value_heads=1,
        attention_dropout=0.0, tie_word_embeddings=True, pad_token_id=127,
        eos_token_id=127, _attn_implementation="eager")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = Qwen2ForCausalLM(config).to(dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 100)
        workspace = FixedWorkspace(12, num_slots=4, latent_dim=8, num_heads=2,
            max_steps=4).to(dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 200)
        return Qwen2WorkspaceLM(backbone, workspace, prompt_layer=1,
            adapter_layers=(0, 1, 2), reader_heads=3)


_IDS = tuple(hashlib.sha256(f"source-control-{index}".encode()).hexdigest() for index in range(4))
_IDENTITIES = {name: hashlib.sha256(name.encode()).hexdigest()
               for name in ("source", "config", "data", "model", "tokenizer", "schedule")}


def _tree_equal(left, right, *, atol=1e-12, rtol=1e-12):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and (
            torch.equal(left, right) if not left.is_floating_point()
            else torch.allclose(left, right, atol=atol, rtol=rtol)
        )
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _tree_equal(left[key], right[key], atol=atol, rtol=rtol) for key in left
        )
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _tree_equal(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right)
        )
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, abs_tol=atol, rel_tol=rtol)
    return left == right


def _batch(example_id: str | None = None):
    task, solution = fixture_task("neq")
    example = SourceExample(example_id or hashlib.sha256(b"control fixture").hexdigest(),
        "source_train", task, solution)
    return collate_source_examples([example], AsciiTokenizer(),
        tokenizer_identity=TokenizerIdentity("fixture", "v1"), vocab_size=128,
        max_prompt_tokens=4096, max_answer_tokens=128, max_total_tokens=4224, device="cpu")


@unittest.skipIf(torch is None or Qwen2Config is None, "optional native ML dependencies unavailable")
class SourceControlTests(unittest.TestCase):
    def _optimizer(self, host, steps):
        groups = []
        if steps:
            groups.append({"params": list(host.workspace.parameters()), "name": "workspace"})
        groups.append({"params": list(host.readers.parameters()), "name": "readers"})
        optimizer = torch.optim.AdamW(groups, lr=0.002, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lr_lambda=lambda step: 1.0 / (step + 1))
        return optimizer, scheduler

    def _run_parts(self, steps):
        host = _host(53)
        control = configure_source_control(host, steps=steps)
        optimizer, scheduler = self._optimizer(host, steps)
        sampler = SourceSampler(_IDS, seed=313)
        noise_generator = torch.Generator(device="cpu").manual_seed(727)
        progress = {"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0}
        return host, control, optimizer, scheduler, sampler, noise_generator, progress

    def _run_updates(self, parts, count):
        host, control, optimizer, scheduler, sampler, noise_generator, progress = parts
        for _ in range(count):
            random.random()
            np.random.random()
            torch.rand((), dtype=torch.float64)
            ids = sampler.next_ids(1)
            batch = _batch(ids[0])
            noise = host.workspace.sample_noise(1, generator=noise_generator,
                device="cpu", dtype=torch.float64)
            result = source_control_step(host, batch, noise, control=control,
                optimizer=optimizer, scheduler=scheduler, gradient_clip_norm=1.0)
            progress["optimizer_step"] += 1
            progress["examples_seen"] += len(ids)
            progress["supervised_tokens_seen"] += result["supervised_tokens"]

    def _reset_global_rng(self):
        random.seed(101)
        np.random.seed(103)
        torch.manual_seed(107)

    def test_policy_and_real_native_updates_preserve_matched_initialization(self):
        hosts = [_host(13) for _ in range(3)]
        for left, right in zip(hosts[0].workspace.parameters(), hosts[1].workspace.parameters()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(hosts[1].readers.parameters(), hosts[2].readers.parameters()):
            self.assertTrue(torch.equal(left, right))
        controls = [configure_source_control(host, steps=k) for host, k in zip(hosts, (0, 1, 4))]
        self.assertEqual(controls[1].active_parameter_names, controls[2].active_parameter_names)
        batch = _batch()
        results = []
        before = [{name: value.detach().clone() for name, value in host.state_dict().items()}
                  for host in hosts]
        backbones = [{name: value.detach().clone() for name, value in host.backbone.state_dict().items()}
                     for host in hosts]
        for left, right in zip(hosts[0].readers.parameters(), hosts[2].readers.parameters()):
            self.assertTrue(torch.equal(left, right))
        for index, (host, control, steps) in enumerate(zip(hosts, controls, (0, 1, 4))):
            optimizer, scheduler = self._optimizer(host, steps)
            noise = torch.Generator(device="cpu").manual_seed(700)
            latent = host.workspace.sample_noise(1, generator=noise, device="cpu", dtype=torch.float64)
            supplied_noise = latent.clone()
            result = source_control_step(host, batch, latent, control=control,
                optimizer=optimizer, scheduler=scheduler, gradient_clip_norm=1.0, diagnose=True)
            results.append(result)
            self.assertTrue(all(p.grad is None for p in host.parameters()))
            self.assertGreater(result["loss"], 0.0)
            self.assertTrue(result["gradient_norms"])
            self.assertTrue(all(v > 0.0 for v in result["gradient_norms"].values()))
            self.assertTrue(all(v > 0.0 for v in result["state_gradient_norms"]))
            self.assertTrue(all(v > 0.0 for v in result["reader_gradient_norms"].values()))
            self.assertTrue(result["initial_state_equal_noise"])
            self.assertTrue(torch.equal(latent, supplied_noise))
            self.assertTrue(all(torch.equal(backbones[index][name], value)
                                for name, value in host.backbone.state_dict().items()))
            workspace_changed = any(
                not torch.equal(before[index][f"workspace.{name}"], value)
                for name, value in host.workspace.state_dict().items()
            )
            readers_changed = any(
                not torch.equal(before[index][f"readers.{name}"], value)
                for name, value in host.readers.state_dict().items()
            )
            if steps:
                self.assertTrue(workspace_changed)
                self.assertTrue(readers_changed)
            else:
                self.assertFalse(workspace_changed)
                self.assertTrue(readers_changed)
        self.assertEqual(len(results[0]["state_gradient_norms"]), 1)
        self.assertEqual(len(results[1]["state_gradient_norms"]), 2)
        self.assertEqual(len(results[2]["state_gradient_norms"]), 5)

    def test_policy_rejects_unsupported_horizons_idx_drift_and_dirty_entry(self):
        host = _host(31)
        for bad in (True, 2, -1):
            with self.assertRaises((TypeError, ValueError)):
                configure_source_control(host, steps=bad)
        indexed = _host(32)
        indexed.workspace.identity_mode = "indexed"
        with self.assertRaises(ValueError):
            configure_source_control(indexed, steps=1)
        control = configure_source_control(host, steps=0)
        optimizer, scheduler = self._optimizer(host, 0)
        batch = _batch()
        noise = torch.zeros((1, 4, 8), dtype=torch.float64)
        host.readers[next(iter(host.readers))].gate.grad = torch.ones_like(
            host.readers[next(iter(host.readers))].gate)
        with self.assertRaisesRegex(ValueError, "gradient must be None at step entry"):
            source_control_step(host, batch, noise, control=control, optimizer=optimizer,
                scheduler=scheduler, gradient_clip_norm=1.0)
        host.readers[next(iter(host.readers))].gate.grad = None
        host.readers[next(iter(host.readers))].gate.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "active flag drifted"):
            source_control_step(host, batch, noise, control=control, optimizer=optimizer,
                scheduler=scheduler, gradient_clip_norm=1.0)

    def test_none_horizon_rejects_before_any_policy_flag_changes(self):
        host = _host(35)
        initial_flags = tuple(parameter.requires_grad for parameter in host.parameters())
        with self.assertRaises((TypeError, ValueError)):
            configure_source_control(host, steps=None)
        self.assertEqual(initial_flags, tuple(parameter.requires_grad for parameter in host.parameters()))

    def test_control_rejects_prompt_and_workspace_shape_drift(self):
        host = _host(36)
        control = configure_source_control(host, steps=1)
        host.prompt_layer = 2
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_source_control(host, control)

        other = _host(37)
        other_control = configure_source_control(other, steps=1)
        other.workspace.num_slots = other.workspace.num_slots + 1
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_source_control(other, other_control)

        precision = _host(38)
        precision_control = configure_source_control(precision, steps=1)
        precision.to(dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_source_control(precision, precision_control)

    def test_control_rejects_same_count_workspace_size_drift(self):
        host = _host(39)
        control = configure_source_control(host, steps=1)
        host.workspace.num_slots += 1
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_source_control(host, control)

    def test_control_rejects_actual_reader_head_drift_before_update(self):
        host = _host(40)
        control = configure_source_control(host, steps=1)
        optimizer, scheduler = self._optimizer(host, 1)
        reader = host.readers["0"]
        self.assertEqual(reader.num_heads, 3)
        self.assertEqual(reader.cross_attention.num_heads, 3)
        # Keep the declared architecture metadata and all parameter shapes
        # unchanged while altering the head partition used by attention.
        reader.cross_attention.num_heads = 1
        parameters_before = [parameter.detach().clone() for parameter in host.parameters()]
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        scheduler_before = copy.deepcopy(scheduler.state_dict())
        noise = torch.Generator(device="cpu").manual_seed(741)
        latent = host.workspace.sample_noise(1, generator=noise,
            device="cpu", dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "reader.*head"):
            source_control_step(host, _batch(), latent, control=control,
                optimizer=optimizer, scheduler=scheduler, gradient_clip_norm=1.0)
        self.assertTrue(all(torch.equal(before, after)
                            for before, after in zip(parameters_before, host.parameters())))
        self.assertEqual(optimizer.state_dict(), optimizer_before)
        self.assertEqual(scheduler.state_dict(), scheduler_before)
        self.assertTrue(all(parameter.grad is None for parameter in host.parameters()))

    def test_ordinary_updates_return_finite_norms_for_every_workspace_state(self):
        for steps in (0, 1, 4):
            with self.subTest(steps=steps):
                host = _host(61 + steps)
                control = configure_source_control(host, steps=steps)
                optimizer, scheduler = self._optimizer(host, steps)
                noise = torch.Generator(device="cpu").manual_seed(701)
                latent = host.workspace.sample_noise(1, generator=noise,
                    device="cpu", dtype=torch.float64)
                result = source_control_step(host, _batch(), latent, control=control,
                    optimizer=optimizer, scheduler=scheduler, gradient_clip_norm=1.0,
                    diagnose=False)
                self.assertEqual(len(result["state_norms"]), steps + 1)
                self.assertTrue(all(math.isfinite(value) and value > 0
                                    for value in result["state_norms"]))

    def test_zero_state_or_reader_diagnostic_gradient_rejects_before_update(self):
        for target in ("state", "reader"):
            with self.subTest(target=target):
                host = _host(71)
                control = configure_source_control(host, steps=0)
                optimizer, scheduler = self._optimizer(host, 0)
                before = [parameter.detach().clone() for parameter in host.parameters()]
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                scheduler_state = copy.deepcopy(scheduler.state_dict())
                handles = []
                if target == "reader":
                    for parameter in host.readers.parameters():
                        handles.append(parameter.register_hook(lambda grad: torch.zeros_like(grad)))
                else:
                    original_forward = host.forward

                    def zero_initial_state_gradient(*args, **kwargs):
                        output = original_forward(*args, **kwargs)
                        output.states[0].register_hook(lambda grad: torch.zeros_like(grad))
                        return output

                    host.forward = zero_initial_state_gradient
                noise = torch.Generator(device="cpu").manual_seed(711)
                latent = host.workspace.sample_noise(1, generator=noise,
                    device="cpu", dtype=torch.float64)
                with self.assertRaisesRegex(ValueError, "diagnostic gradient norm must be positive"):
                    source_control_step(host, _batch(), latent, control=control,
                        optimizer=optimizer, scheduler=scheduler,
                        gradient_clip_norm=1.0, diagnose=True)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, host.parameters())))
                self.assertEqual(optimizer.state_dict(), optimizer_state)
                self.assertEqual(scheduler.state_dict(), scheduler_state)
                for handle in handles:
                    handle.remove()
                optimizer.zero_grad(set_to_none=True)

    def test_optimizer_membership_foreign_scheduler_and_nonfinite_noise_fail_closed(self):
        host = _host(41)
        control = configure_source_control(host, steps=0)
        updater = list(host.workspace.parameters())
        readers = list(host.readers.parameters())
        bad_optimizer = torch.optim.AdamW([
            {"params": updater + readers, "name": "readers"}], lr=0.001)
        bad_scheduler = torch.optim.lr_scheduler.LambdaLR(bad_optimizer, lambda _: 1.0)
        k0_optimizer, k0_scheduler = self._optimizer(host, 0)
        k0_with_workspace = torch.optim.AdamW([
            {"params": updater, "name": "workspace"},
            {"params": readers, "name": "readers"},
        ], lr=0.001)
        k0_with_workspace_scheduler = torch.optim.lr_scheduler.LambdaLR(
            k0_with_workspace, lambda _: 1.0)
        noise = torch.zeros((1, 4, 8), dtype=torch.float64)
        before = [p.detach().clone() for p in host.parameters()]
        with self.assertRaises(ValueError):
            source_control_step(host, _batch(), noise, control=control,
                optimizer=bad_optimizer, scheduler=bad_scheduler, gradient_clip_norm=1.0)
        with self.assertRaisesRegex(ValueError, "exactly the active named modules"):
            source_control_step(host, _batch(), noise, control=control,
                optimizer=k0_with_workspace, scheduler=k0_with_workspace_scheduler,
                gradient_clip_norm=1.0)
        good_optimizer, good_scheduler = self._optimizer(host, 0)
        foreign = torch.optim.AdamW(list(host.readers.parameters()), lr=0.001)
        foreign_scheduler = torch.optim.lr_scheduler.LambdaLR(foreign, lambda _: 1.0)
        with self.assertRaisesRegex(ValueError, "must belong"):
            source_control_step(host, _batch(), noise, control=control,
                optimizer=good_optimizer, scheduler=foreign_scheduler, gradient_clip_norm=1.0)
        nonfinite = noise.clone()
        nonfinite[0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "noise must be finite"):
            source_control_step(host, _batch(), nonfinite, control=control,
                optimizer=good_optimizer, scheduler=good_scheduler, gradient_clip_norm=1.0)
        with patch("dwr.training.controls.answer_only_loss",
                   return_value=SimpleNamespace(loss=torch.tensor(float("nan")), supervised_tokens=1)):
            with self.assertRaisesRegex(ValueError, "loss is nonfinite"):
                source_control_step(host, _batch(), noise, control=control,
                    optimizer=k0_optimizer, scheduler=k0_scheduler, gradient_clip_norm=1.0)
        with patch("dwr.training.controls.answer_only_loss",
                   return_value=SimpleNamespace(loss=torch.ones((), requires_grad=True),
                                                supervised_tokens=1)):
            with self.assertRaisesRegex(ValueError, "active parameter is missing a gradient"):
                source_control_step(host, _batch(), noise, control=control,
                    optimizer=k0_optimizer, scheduler=k0_scheduler, gradient_clip_norm=1.0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, host.parameters())))
        self.assertTrue(all(p.grad is None for p in host.parameters()))

    def test_optimizer_membership_and_frozen_gradient_rejections_precede_mutation(self):
        host = _host(81)
        control = configure_source_control(host, steps=0)
        readers = list(host.readers.parameters())
        before = [parameter.detach().clone() for parameter in host.parameters()]
        for params in (readers[:-1], readers + [readers[0]]):
            with self.subTest(parameter_count=len(params)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    optimizer = torch.optim.AdamW([{"params": params, "name": "readers"}], lr=0.001)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                scheduler_state = copy.deepcopy(scheduler.state_dict())
                with self.assertRaises(ValueError):
                    source_control_step(host, _batch(), torch.zeros((1, 4, 8), dtype=torch.float64),
                        control=control, optimizer=optimizer, scheduler=scheduler,
                        gradient_clip_norm=1.0)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, host.parameters())))
                self.assertEqual(optimizer.state_dict(), optimizer_state)
                self.assertEqual(scheduler.state_dict(), scheduler_state)
        stray = next(host.backbone.parameters())
        stray.grad = torch.ones_like(stray)
        optimizer, scheduler = self._optimizer(host, 0)
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict())
        with self.assertRaisesRegex(ValueError, "excluded parameter has a gradient"):
            source_control_step(host, _batch(), torch.zeros((1, 4, 8), dtype=torch.float64),
                control=control, optimizer=optimizer, scheduler=scheduler,
                gradient_clip_norm=1.0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, host.parameters())))
        self.assertTrue(_tree_equal(optimizer.state_dict(), optimizer_state))
        self.assertTrue(_tree_equal(scheduler.state_dict(), scheduler_state))
        stray.grad = None

    def test_nonfinite_active_gradient_rejects_before_optimizer_mutation(self):
        host = _host(82)
        control = configure_source_control(host, steps=0)
        optimizer, scheduler = self._optimizer(host, 0)
        parameter = next(host.readers.parameters())
        handle = parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        before = [item.detach().clone() for item in host.parameters()]
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict())
        noise = torch.Generator(device="cpu").manual_seed(821)
        latent = host.workspace.sample_noise(1, generator=noise, device="cpu", dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            source_control_step(host, _batch(), latent, control=control,
                optimizer=optimizer, scheduler=scheduler, gradient_clip_norm=1.0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, host.parameters())))
        self.assertEqual(optimizer.state_dict(), optimizer_state)
        self.assertEqual(scheduler.state_dict(), scheduler_state)
        self.assertTrue(torch.isinf(parameter.grad).any())
        handle.remove()
        optimizer.zero_grad(set_to_none=True)

    def test_nonfinite_post_update_is_terminal_and_never_retried_by_primitive(self):
        host = _host(83)
        control = configure_source_control(host, steps=0)
        optimizer, scheduler = self._optimizer(host, 0)
        poison = next(host.readers.parameters())
        before = [parameter.detach().clone() for parameter in host.parameters()]

        def inject_nonfinite(_optimizer, _args, _kwargs):
            poison.data.fill_(float("inf"))

        hook = optimizer.register_step_post_hook(inject_nonfinite)
        noise = torch.Generator(device="cpu").manual_seed(831)
        latent = host.workspace.sample_noise(1, generator=noise, device="cpu", dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "nonfinite learned parameter"):
            source_control_step(host, _batch(), latent, control=control,
                optimizer=optimizer, scheduler=scheduler, gradient_clip_norm=1.0)
        self.assertTrue(torch.isinf(poison).all())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, host.parameters())))
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict())
        hook.remove()
        with self.assertRaisesRegex(ValueError, "gradient must be None at step entry"):
            source_control_step(host, _batch(), latent, control=control,
                optimizer=optimizer, scheduler=scheduler, gradient_clip_norm=1.0)
        self.assertTrue(_tree_equal(optimizer.state_dict(), optimizer_state))
        self.assertTrue(_tree_equal(scheduler.state_dict(), scheduler_state))
        optimizer.zero_grad(set_to_none=True)

    def test_k0_and_k1_checkpoint_fresh_process_continuation(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            root = Path(__file__).resolve().parents[1]
            for steps in (0, 1):
                with self.subTest(steps=steps), tempfile.TemporaryDirectory() as directory:
                    partial_path = Path(directory) / f"k{steps}-partial.pt"
                    child_path = Path(directory) / f"k{steps}-child.pt"
                    self._reset_global_rng()
                    full = self._run_parts(steps)
                    self._run_updates(full, 3)
                    full_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())

                    self._reset_global_rng()
                    partial = self._run_parts(steps)
                    self._run_updates(partial, 1)
                    digest, _ = save_training_checkpoint(partial_path, host=partial[0],
                        optimizer=partial[2], scheduler=partial[3], scaler=None,
                        sampler=partial[4], noise_generator=partial[5],
                        identities=_IDENTITIES, progress=partial[6])
                    command = [sys.executable, str(root / "tests" / "source_control_resume_child.py"),
                        str(steps), str(partial_path), digest, str(child_path)]
                    child = subprocess.run(command, cwd=root, capture_output=True,
                        text=True, timeout=60, check=False)
                    self.assertEqual(child.returncode, 0, child.stdout + child.stderr)

                    resumed = self._run_parts(steps)
                    resumed_progress = load_training_checkpoint(child_path,
                        expected_sha256=hashlib.sha256(child_path.read_bytes()).hexdigest(),
                        host=resumed[0], optimizer=resumed[2], scheduler=resumed[3], scaler=None,
                        sampler=resumed[4], noise_generator=resumed[5], identities=_IDENTITIES)
                    self.assertEqual(full[6], resumed_progress)
                    self.assertTrue(_tree_equal(torch.get_rng_state(), full_rng[2]))
                    self.assertTrue(_tree_equal(random.getstate(), full_rng[0]))
                    self.assertTrue(_tree_equal(np.random.get_state(), full_rng[1]))
                    self.assertTrue(_tree_equal(full[4].state_dict(), resumed[4].state_dict()))
                    self.assertTrue(_tree_equal(full[5].get_state(), resumed[5].get_state()))
                    sampler_full = SourceSampler(_IDS, seed=313)
                    sampler_resumed = SourceSampler(_IDS, seed=313)
                    sampler_full.load_state_dict(full[4].state_dict())
                    sampler_resumed.load_state_dict(resumed[4].state_dict())
                    self.assertEqual(tuple(sampler_full.next_ids(1) for _ in range(7)),
                                     tuple(sampler_resumed.next_ids(1) for _ in range(7)))
                    full_optimizer_state = full[2].state_dict()
                    resumed_optimizer_state = resumed[2].state_dict()
                    self.assertTrue(full_optimizer_state["state"])
                    self.assertTrue(all({"step", "exp_avg", "exp_avg_sq"}.issubset(state)
                                        for state in full_optimizer_state["state"].values()))
                    for parameter_id, full_state in full_optimizer_state["state"].items():
                        resumed_state = resumed_optimizer_state["state"][parameter_id]
                        self.assertEqual(float(full_state["step"].item()), 3.0)
                        self.assertTrue(torch.equal(full_state["step"], resumed_state["step"]))
                    self.assertTrue(_tree_equal(full_optimizer_state, resumed_optimizer_state))
                    self.assertTrue(_tree_equal(full[3].state_dict(), resumed[3].state_dict()))
                    self.assertTrue(all(p.grad is None for p in resumed[0].parameters()))
                    for left, right in zip(full[0].workspace.parameters(), resumed[0].workspace.parameters()):
                        self.assertTrue(torch.allclose(left, right, atol=1e-12, rtol=1e-12))
                    for left, right in zip(full[0].readers.parameters(), resumed[0].readers.parameters()):
                        self.assertTrue(torch.allclose(left, right, atol=1e-12, rtol=1e-12))
                    max_errors = []
                    for module_name in ("workspace", "readers"):
                        left_module = getattr(full[0], module_name)
                        right_module = getattr(resumed[0], module_name)
                        errors = [float((left.detach() - right.detach()).abs().max().item())
                                  for left, right in zip(left_module.parameters(), right_module.parameters())]
                        max_errors.append(max(errors, default=0.0))
                    counts = {
                        "total": sum(parameter.numel() for parameter in full[0].parameters()),
                        "workspace": sum(parameter.numel() for parameter in full[0].workspace.parameters()),
                        "readers": sum(parameter.numel() for parameter in full[0].readers.parameters()),
                        "active": full[1].active_parameter_count,
                    }
                    print(f"continuation K{steps}: max_abs_errors={max_errors}; tiny_parameter_counts={counts}")
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
