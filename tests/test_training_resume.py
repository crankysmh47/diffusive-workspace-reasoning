"""CPU learned-state checkpoint replay and strict sampler contract tests."""

from __future__ import annotations

import copy
import hashlib
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as error:  # Optional for standard-library test discovery.
    if error.name != "numpy":
        raise
    np = None

torch = None
if np is not None:
    try:
        import torch
    except ModuleNotFoundError as error:  # Optional for standard-library test discovery.
        if error.name != "torch":
            raise
        torch = None

Qwen2Config = Qwen2ForCausalLM = None
SourceSampler = None
if torch is not None:
    from dwr.training.sampler import SourceSampler

if torch is not None and np is not None:
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except ModuleNotFoundError as error:  # Optional model package only; transitive failures remain errors.
        if error.name != "transformers":
            raise

if torch is not None and np is not None and Qwen2Config is not None:
    from dwr.models.qwen2 import Qwen2WorkspaceLM
    from dwr.models.workspace import FixedWorkspace
    from dwr.training.batches import SourceExample, TokenizerIdentity, collate_source_examples
    from dwr.training.checkpoints import load_training_checkpoint, save_training_checkpoint
    from dwr.training.loss import answer_only_loss
    from test_training_batch_loss import AsciiTokenizer, fixture_task
else:
    Qwen2WorkspaceLM = FixedWorkspace = SourceExample = TokenizerIdentity = None
    collate_source_examples = load_training_checkpoint = save_training_checkpoint = None
    answer_only_loss = AsciiTokenizer = fixture_task = None


TOL = 1e-12
IDS = tuple(hashlib.sha256(f"fixture-{i}".encode()).hexdigest() for i in range(4))
IDENTITIES = {name: hashlib.sha256(name.encode()).hexdigest() for name in
              ("source", "config", "data", "model", "tokenizer", "schedule")}


def _example(example_id: str) -> SourceExample:
    task, solution = fixture_task("eq")
    return SourceExample(example_id, "source_train", task, solution)


def _host(seed: int, *, heads: int = 2) -> Qwen2WorkspaceLM:
    config = Qwen2Config(vocab_size=128, hidden_size=12, intermediate_size=24,
                         num_hidden_layers=3, num_attention_heads=3, num_key_value_heads=1,
                         attention_dropout=0.0, tie_word_embeddings=True, pad_token_id=127,
                         eos_token_id=127, _attn_implementation="eager")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = Qwen2ForCausalLM(config).to(dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 100)
        workspace = FixedWorkspace(12, num_slots=4, latent_dim=8, num_heads=heads,
                                   max_steps=2).to(dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 200)
        return Qwen2WorkspaceLM(backbone, workspace, prompt_layer=1,
                                adapter_layers=(0, 1, 2), reader_heads=3)


def _components(seed: int):
    host = _host(seed)
    groups = [
        {"params": list(host.workspace.parameters()), "name": "workspace"},
        {"params": list(host.readers.parameters()), "name": "readers"},
    ]
    optimizer = torch.optim.AdamW(groups, lr=0.002, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1 / (step + 1))
    sampler = SourceSampler(IDS, seed=313)
    noise = torch.Generator(device="cpu").manual_seed(727)
    return host, optimizer, scheduler, sampler, noise


def _components_k0(seed: int):
    host = _host(seed)
    for parameter in host.workspace.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [{"params": list(host.readers.parameters()), "name": "readers"}],
        lr=0.002, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=lambda step: 1 / (step + 1))
    sampler = SourceSampler(IDS, seed=313)
    noise = torch.Generator(device="cpu").manual_seed(727)
    return host, optimizer, scheduler, sampler, noise


def _one_step(host, optimizer, scheduler, sampler, noise, progress, *, steps=2):
    random.random()
    np.random.random()
    torch.rand((), dtype=torch.float64)
    ids = sampler.next_ids(1)
    batch = collate_source_examples([_example(ids[0])], AsciiTokenizer(),
        tokenizer_identity=TokenizerIdentity("fixture", "v1"), vocab_size=128,
        max_prompt_tokens=4096, max_answer_tokens=128, max_total_tokens=4224, device="cpu")
    latent_noise = host.workspace.sample_noise(1, generator=noise, device="cpu", dtype=torch.float64)
    output = host(batch.prompt_ids, batch.prompt_mask, batch.decoder_ids, batch.decoder_mask,
                  noise=latent_noise, steps=steps)
    loss = answer_only_loss(output.logits, batch.labels)
    loss.loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    progress["optimizer_step"] += 1
    progress["examples_seen"] += 1
    progress["supervised_tokens_seen"] += loss.supervised_tokens


def _base_bytes(host):
    return {name: value.detach().cpu().contiguous().numpy().tobytes()
            for name, value in host.backbone.state_dict().items()}


def _resign(path: Path, edit):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    edit(payload)
    with path.open("wb") as handle:
        torch.save(payload, handle)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


def _global_rng_state():
    numpy_state = np.random.get_state()
    return (random.getstate(), (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
            torch.get_rng_state().clone())


def _next_global_draws(state):
    python_rng = random.Random()
    python_rng.setstate(state[0])
    numpy_rng = np.random.RandomState()
    numpy_rng.set_state(state[1])
    torch_rng = torch.Generator(device="cpu")
    torch_rng.set_state(state[2].clone())
    return python_rng.random(), numpy_rng.random_sample(), torch.rand(3, generator=torch_rng)


@unittest.skipIf(torch is None or SourceSampler is None, "install the optional torch extra")
class SourceSamplerTests(unittest.TestCase):
    def test_partial_epoch_state_is_detached_and_replays_exact_ids(self):
        sampler = SourceSampler(IDS, seed=11)
        first = sampler.next_ids(2)
        state = sampler.state_dict()
        state["generator_state"].zero_()
        self.assertFalse(torch.equal(state["generator_state"], sampler.state_dict()["generator_state"]))
        restored = SourceSampler(IDS, seed=11)
        restored.load_state_dict(sampler.state_dict())
        self.assertEqual(restored.next_ids(9), sampler.next_ids(9))
        self.assertEqual(len(first), 2)

    def test_epoch_boundary_and_invalid_identity_or_cursor(self):
        sampler = SourceSampler(IDS, seed=11)
        sampler.next_ids(len(IDS))
        boundary = sampler.state_dict()
        restored = SourceSampler(IDS, seed=11)
        restored.load_state_dict(boundary)
        self.assertEqual(restored.next_ids(2), sampler.next_ids(2))
        altered = copy.deepcopy(boundary)
        altered["position"] = len(IDS) + 1
        with self.assertRaises((TypeError, ValueError)):
            restored.load_state_dict(altered)
        altered = copy.deepcopy(boundary)
        altered["unknown"] = 1
        with self.assertRaises(ValueError):
            restored.load_state_dict(altered)
        altered = copy.deepcopy(boundary)
        del altered["seed"]
        with self.assertRaises(ValueError):
            restored.load_state_dict(altered)
        other = SourceSampler(tuple(reversed(IDS)), seed=11)
        with self.assertRaises(ValueError):
            other.load_state_dict(boundary)


@unittest.skipIf(torch is None or np is None or Qwen2Config is None,
                 "install the optional NumPy, torch and transformers extras")
class LearnedResumeTests(unittest.TestCase):
    def setUp(self):
        caller_python = random.getstate()
        caller_numpy = _global_rng_state()[1]
        caller_torch = torch.get_rng_state().clone()
        caller_threads = torch.get_num_threads()

        def restore_caller_state():
            random.setstate(caller_python)
            np.random.set_state(caller_numpy)
            torch.set_rng_state(caller_torch)
            torch.set_num_threads(caller_threads)

        self.addCleanup(restore_caller_state)

    def test_k0_readers_only_optimizer_resumes_with_complete_frozen_workspace(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            random.seed(1101)
            np.random.seed(1103)
            torch.manual_seed(1107)
            full = _components_k0(1211)
            full_workspace = {name: value.detach().clone() for name, value in
                              full[0].workspace.state_dict().items()}
            full_progress = {"optimizer_step": 0, "examples_seen": 0,
                             "supervised_tokens_seen": 0}
            for _ in range(2):
                _one_step(*full, full_progress, steps=0)

            random.seed(1101)
            np.random.seed(1103)
            torch.manual_seed(1107)
            partial = _components_k0(1211)
            partial_workspace = {name: value.detach().clone() for name, value in
                                 partial[0].workspace.state_dict().items()}
            progress = {"optimizer_step": 0, "examples_seen": 0,
                        "supervised_tokens_seen": 0}
            _one_step(*partial, progress, steps=0)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "k0.pt"
                digest, _ = save_training_checkpoint(path, host=partial[0], optimizer=partial[1],
                    scheduler=partial[2], scaler=None, sampler=partial[3],
                    noise_generator=partial[4], identities=IDENTITIES, progress=progress)
                saved = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual(len(saved["optimizer_groups"]), 1)
                self.assertTrue(all(name.startswith("readers.") for name in saved["optimizer_groups"][0]))
                self.assertEqual(saved["workspace"].keys(), full_workspace.keys())
                resumed = _components_k0(1211)
                resumed_workspace = {name: value.detach().clone() for name, value in
                                     resumed[0].workspace.state_dict().items()}
                resumed_progress = load_training_checkpoint(path, expected_sha256=digest,
                    host=resumed[0], optimizer=resumed[1], scheduler=resumed[2], scaler=None,
                    sampler=resumed[3], noise_generator=resumed[4], identities=IDENTITIES)
                _one_step(*resumed, resumed_progress, steps=0)

            self.assertEqual(full_progress, resumed_progress)
            self.assertTrue(all(not parameter.requires_grad for parameter in resumed[0].workspace.parameters()))
            self.assertTrue(_nested_equal(full[0].workspace.state_dict(), full_workspace))
            self.assertTrue(_nested_equal(partial[0].workspace.state_dict(), partial_workspace))
            self.assertTrue(_nested_equal(resumed[0].workspace.state_dict(), resumed_workspace))
            self.assertTrue(_nested_equal(full[0].readers.state_dict(), resumed[0].readers.state_dict()))
            self.assertTrue(_nested_equal(full[1].state_dict(), resumed[1].state_dict()))
            self.assertTrue(_nested_equal(full[2].state_dict(), resumed[2].state_dict()))
            self.assertTrue(all("exp_avg" in state for state in full[1].state.values()))
        finally:
            torch.set_num_threads(old_threads)

    def test_four_steps_equal_two_save_restart_two(self):
        cuda_initialized_before = torch.cuda.is_initialized()
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            random.seed(101)
            np.random.seed(103)
            torch.manual_seed(107)
            full = _components(211)
            full_base = _base_bytes(full[0])
            full_progress = {"optimizer_step": 0, "examples_seen": 0,
                             "supervised_tokens_seen": 0}
            for _ in range(4):
                _one_step(*full[:5], full_progress)
            full_global_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())

            random.seed(101)
            np.random.seed(103)
            torch.manual_seed(107)
            partial = _components(211)
            partial_base = _base_bytes(partial[0])
            progress = {"optimizer_step": 0, "examples_seen": 0,
                        "supervised_tokens_seen": 0}
            for _ in range(2):
                _one_step(*partial[:5], progress)

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state.pt"
                digest, size = save_training_checkpoint(path, host=partial[0], optimizer=partial[1],
                    scheduler=partial[2], scaler=None, sampler=partial[3], noise_generator=partial[4],
                    identities=IDENTITIES, progress=progress)
                self.assertGreater(size, 0)
                saved = torch.load(path, map_location="cpu", weights_only=True)
                self.assertNotIn("backbone", saved)
                self.assertTrue(any(row[1].startswith("backbone.") for row in saved["tensor_signature"]))
                self.assertTrue(all(not isinstance(value, torch.Tensor)
                                    for row in saved["tensor_signature"] for value in row))
                restarted = _components(211)
                restarted_base = _base_bytes(restarted[0])
                checkpoint_global_rng = _global_rng_state()
                random.seed(1201)
                np.random.seed(1203)
                torch.manual_seed(1207)
                perturbed_global_rng = _global_rng_state()
                self.assertFalse(_nested_equal(checkpoint_global_rng, perturbed_global_rng))
                resumed_progress = load_training_checkpoint(path, expected_sha256=digest,
                    host=restarted[0], optimizer=restarted[1], scheduler=restarted[2], scaler=None,
                    sampler=restarted[3], noise_generator=restarted[4], identities=IDENTITIES)
                restored_global_rng = _global_rng_state()
                self.assertTrue(_nested_equal(checkpoint_global_rng, restored_global_rng))
                self.assertTrue(_nested_equal(_next_global_draws(checkpoint_global_rng),
                                              _next_global_draws(restored_global_rng)))
                for _ in range(2):
                    _one_step(*restarted[:5], resumed_progress)
                resumed_global_rng = (random.getstate(), np.random.get_state(),
                                      torch.get_rng_state().clone())

            self.assertEqual(full_progress, resumed_progress)
            self.assertTrue(_nested_equal(full[3].state_dict(), restarted[3].state_dict()))
            self.assertEqual(full[4].get_state().tolist(), restarted[4].get_state().tolist())
            self.assertEqual(full[3].next_ids(3), restarted[3].next_ids(3))
            self.assertTrue(torch.equal(torch.randn(5, generator=full[4]),
                                        torch.randn(5, generator=restarted[4])))
            self.assertTrue(_nested_equal(full[1].state_dict(), restarted[1].state_dict()))
            self.assertTrue(_nested_equal(full[2].state_dict(), restarted[2].state_dict()))
            self.assertTrue(_nested_equal(full_global_rng, resumed_global_rng))
            for left, right in zip(full[0].workspace.parameters(), restarted[0].workspace.parameters()):
                self.assertTrue(torch.allclose(left, right, atol=TOL, rtol=TOL))
            for left, right in zip(full[0].readers.parameters(), restarted[0].readers.parameters()):
                self.assertTrue(torch.allclose(left, right, atol=TOL, rtol=TOL))
            for key, left in full[0].workspace.state_dict().items():
                self.assertTrue(torch.allclose(left, restarted[0].workspace.state_dict()[key],
                                               atol=TOL, rtol=TOL))
            for key, left in full[0].readers.state_dict().items():
                self.assertTrue(torch.allclose(left, restarted[0].readers.state_dict()[key],
                                               atol=TOL, rtol=TOL))
            self.assertEqual(full_base, _base_bytes(full[0]))
            self.assertEqual(partial_base, _base_bytes(partial[0]))
            self.assertEqual(restarted_base, _base_bytes(restarted[0]))
            self.assertTrue(any("exp_avg" in state for state in full[1].state.values()))
            self.assertEqual(torch.cuda.is_initialized(), cuda_initialized_before)
        finally:
            torch.set_num_threads(old_threads)

    def test_rejects_head_mismatch_and_foreign_scheduler(self):
        with tempfile.TemporaryDirectory() as directory:
            left = _components(301)
            path = Path(directory) / "state.pt"
            digest, _ = save_training_checkpoint(path, host=left[0], optimizer=left[1],
                scheduler=left[2], scaler=None, sampler=left[3], noise_generator=left[4],
                identities=IDENTITIES, progress={"optimizer_step": 0, "examples_seen": 0,
                                                   "supervised_tokens_seen": 0})
            changed = _components(301)
            changed_host = _host(301, heads=1)
            # Preserve optimizer construction from the incompatible host.
            changed_groups = [{"params": list(changed_host.workspace.parameters()), "name": "workspace"},
                              {"params": list(changed_host.readers.parameters()), "name": "readers"}]
            changed_optimizer = torch.optim.AdamW(changed_groups, lr=0.002, weight_decay=0.01)
            changed_scheduler = torch.optim.lr_scheduler.LambdaLR(changed_optimizer,
                lr_lambda=lambda step: 1 / (step + 1))
            with self.assertRaises(ValueError):
                load_training_checkpoint(path, expected_sha256=digest, host=changed_host,
                    optimizer=changed_optimizer, scheduler=changed_scheduler, scaler=None,
                    sampler=changed[3], noise_generator=changed[4], identities=IDENTITIES)

            changed_dtype = _components(301)
            changed_dtype[0].to(dtype=torch.float32)
            with self.assertRaises(ValueError):
                load_training_checkpoint(path, expected_sha256=digest, host=changed_dtype[0],
                    optimizer=changed_dtype[1], scheduler=changed_dtype[2], scaler=None,
                    sampler=changed_dtype[3], noise_generator=changed_dtype[4], identities=IDENTITIES)
            with self.assertRaises(ValueError):
                load_training_checkpoint(path, expected_sha256=digest, host=changed[0],
                    optimizer=changed[1], scheduler=changed[2],
                    scaler=torch.amp.GradScaler("cpu", enabled=False), sampler=changed[3],
                    noise_generator=changed[4], identities=IDENTITIES)
            altered_identities = dict(IDENTITIES)
            altered_identities["data"] = "f" * 64
            with self.assertRaises(ValueError):
                load_training_checkpoint(path, expected_sha256=digest, host=changed[0],
                    optimizer=changed[1], scheduler=changed[2], scaler=None,
                    sampler=changed[3], noise_generator=changed[4], identities=altered_identities)

            foreign = torch.optim.AdamW([p for p in left[0].parameters() if p.requires_grad], lr=0.002)
            foreign_scheduler = torch.optim.lr_scheduler.LambdaLR(foreign, lr_lambda=lambda step: 1 / (step + 1))
            with self.assertRaises(ValueError):
                save_training_checkpoint(Path(directory) / "bad.pt", host=left[0], optimizer=left[1],
                    scheduler=foreign_scheduler, scaler=None, sampler=left[3],
                    noise_generator=left[4], identities=IDENTITIES,
                    progress={"optimizer_step": 0, "examples_seen": 0,
                              "supervised_tokens_seen": 0})
            crossed_groups = torch.optim.AdamW([
                {"params": list(left[0].workspace.parameters()), "name": "readers"},
                {"params": list(left[0].readers.parameters()), "name": "workspace"},
            ], lr=0.002)
            crossed_scheduler = torch.optim.lr_scheduler.LambdaLR(crossed_groups,
                lr_lambda=lambda step: 1 / (step + 1))
            with self.assertRaises(ValueError):
                save_training_checkpoint(Path(directory) / "crossed.pt", host=left[0],
                    optimizer=crossed_groups, scheduler=crossed_scheduler, scaler=None,
                    sampler=left[3], noise_generator=left[4], identities=IDENTITIES,
                    progress={"optimizer_step": 0, "examples_seen": 0,
                              "supervised_tokens_seen": 0})

    def test_rejects_gradients_existing_dest_and_corrupt_digest(self):
        components = _components(401)
        progress = {"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            digest, _ = save_training_checkpoint(path, host=components[0], optimizer=components[1],
                scheduler=components[2], scaler=None, sampler=components[3],
                noise_generator=components[4], identities=IDENTITIES, progress=progress)
            with self.assertRaises(FileExistsError):
                save_training_checkpoint(path, host=components[0], optimizer=components[1],
                    scheduler=components[2], scaler=None, sampler=components[3],
                    noise_generator=components[4], identities=IDENTITIES, progress=progress)
            with self.assertRaises(ValueError):
                load_training_checkpoint(path, expected_sha256="0" * 64, host=components[0],
                    optimizer=components[1], scheduler=components[2], scaler=None,
                    sampler=components[3], noise_generator=components[4], identities=IDENTITIES)
            self.assertTrue(components[0]._execution_lock.acquire(blocking=False))
            try:
                with self.assertRaises(ValueError):
                    save_training_checkpoint(Path(directory) / "busy.pt", host=components[0],
                        optimizer=components[1], scheduler=components[2], scaler=None,
                        sampler=components[3], noise_generator=components[4], identities=IDENTITIES,
                        progress=progress)
            finally:
                components[0]._execution_lock.release()
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                load_training_checkpoint(path, expected_sha256=digest, host=components[0],
                    optimizer=components[1], scheduler=components[2], scaler=None,
                    sampler=components[3], noise_generator=components[4], identities=IDENTITIES)
            next(components[0].workspace.parameters()).grad = torch.ones_like(
                next(components[0].workspace.parameters()))
            with self.assertRaises(ValueError):
                save_training_checkpoint(Path(directory) / "grad.pt", host=components[0],
                    optimizer=components[1], scheduler=components[2], scaler=None,
                    sampler=components[3], noise_generator=components[4], identities=IDENTITIES,
                    progress=progress)

    def test_rejects_malformed_and_nonfinite_checkpoint_before_apply(self):
        components = _components(501)
        progress = {"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_training_checkpoint(path, host=components[0], optimizer=components[1],
                scheduler=components[2], scaler=None, sampler=components[3],
                noise_generator=components[4], identities=IDENTITIES, progress=progress)
            cases = (
                lambda payload: payload.update(unexpected=True),
                lambda payload: payload.pop("scheduler"),
                lambda payload: payload["workspace"]["step_norm.weight"].fill_(float("nan")),
                lambda payload: payload["sampler"].update(example_ids=tuple(reversed(IDS))),
            )
            for index, edit in enumerate(cases):
                digest = _resign(path, edit)
                before = {name: value.detach().clone() for name, value in
                          components[0].workspace.state_dict().items()}
                with self.subTest(index=index), self.assertRaises(ValueError):
                    load_training_checkpoint(path, expected_sha256=digest, host=components[0],
                        optimizer=components[1], scheduler=components[2], scaler=None,
                        sampler=components[3], noise_generator=components[4], identities=IDENTITIES)
                self.assertTrue(all(torch.equal(before[name], value) for name, value in
                                    components[0].workspace.state_dict().items()))
                # Restore a valid artifact for the next mutation.
                save_path = Path(directory) / f"valid-{index}.pt"
                save_training_checkpoint(save_path, host=components[0], optimizer=components[1],
                    scheduler=components[2], scaler=None, sampler=components[3],
                    noise_generator=components[4], identities=IDENTITIES, progress=progress)
                path.write_bytes(save_path.read_bytes())

    def test_rejects_nonfinite_adam_moments_and_reordered_group_mapping(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            components = _components(601)
            progress = {"optimizer_step": 0, "examples_seen": 0,
                        "supervised_tokens_seen": 0}
            _one_step(*components, progress)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state.pt"
                digest, _ = save_training_checkpoint(path, host=components[0], optimizer=components[1],
                    scheduler=components[2], scaler=None, sampler=components[3],
                    noise_generator=components[4], identities=IDENTITIES, progress=progress)
                def corrupt_moment(payload):
                    state = next(iter(payload["optimizer"]["state"].values()))
                    state["exp_avg"].fill_(float("inf"))
                digest = _resign(path, corrupt_moment)
                target = _components(601)
                with self.assertRaises(ValueError):
                    load_training_checkpoint(path, expected_sha256=digest, host=target[0],
                        optimizer=target[1], scheduler=target[2], scaler=None,
                        sampler=target[3], noise_generator=target[4], identities=IDENTITIES)

                valid_path = Path(directory) / "valid.pt"
                valid_digest, _ = save_training_checkpoint(valid_path, host=components[0],
                    optimizer=components[1], scheduler=components[2], scaler=None,
                    sampler=components[3], noise_generator=components[4], identities=IDENTITIES,
                    progress=progress)
                reordered_host = _host(601)
                reordered = torch.optim.AdamW([
                    {"params": list(reordered_host.readers.parameters()), "name": "readers"},
                    {"params": list(reordered_host.workspace.parameters()), "name": "workspace"},
                ], lr=0.002, weight_decay=0.01)
                reordered_scheduler = torch.optim.lr_scheduler.LambdaLR(reordered,
                    lr_lambda=lambda step: 1 / (step + 1))
                with self.assertRaises(ValueError):
                    load_training_checkpoint(valid_path, expected_sha256=valid_digest, host=reordered_host,
                        optimizer=reordered, scheduler=reordered_scheduler, scaler=None,
                        sampler=target[3], noise_generator=target[4], identities=IDENTITIES)
        finally:
            torch.set_num_threads(old_threads)

    def test_save_rejects_trainable_original_weights(self):
        components = _components(701)
        next(components[0].backbone.parameters()).requires_grad_(True)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            save_training_checkpoint(Path(directory) / "state.pt", host=components[0],
                optimizer=components[1], scheduler=components[2], scaler=None,
                sampler=components[3], noise_generator=components[4], identities=IDENTITIES,
                progress={"optimizer_step": 0, "examples_seen": 0,
                          "supervised_tokens_seen": 0})

    def test_positive_progress_requires_complete_integral_adamw_state_on_save_and_load(self):
        bad_steps = {
            "fractional": torch.tensor(1.5),
            "boolean": torch.tensor(True),
            "stale": torch.tensor(0.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            for index, defect in enumerate(("missing", "none", *bad_steps)):
                components = _components(801 + index)
                progress = {"optimizer_step": 0, "examples_seen": 0,
                            "supervised_tokens_seen": 0}
                _one_step(*components, progress)
                parameter = next(parameter for group in components[1].param_groups
                                 for parameter in group["params"])
                state = components[1].state[parameter]
                if defect == "missing":
                    del components[1].state[parameter]
                elif defect == "none":
                    components[1].state[parameter] = None
                else:
                    state["step"] = bad_steps[defect].clone()
                path = Path(directory) / f"bad-save-{index}.pt"
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    save_training_checkpoint(path, host=components[0], optimizer=components[1],
                        scheduler=components[2], scaler=None, sampler=components[3],
                        noise_generator=components[4], identities=IDENTITIES, progress=progress)
                self.assertFalse(path.exists())

            for index, defect in enumerate(("missing", "none", *bad_steps)):
                source = _components(901 + index)
                progress = {"optimizer_step": 0, "examples_seen": 0,
                            "supervised_tokens_seen": 0}
                _one_step(*source, progress)
                path = Path(directory) / f"valid-{index}.pt"
                save_training_checkpoint(path, host=source[0], optimizer=source[1],
                    scheduler=source[2], scaler=None, sampler=source[3],
                    noise_generator=source[4], identities=IDENTITIES, progress=progress)

                def damage(payload):
                    group_ids = payload["optimizer"]["param_groups"][0]["params"]
                    parameter_id = group_ids[0]
                    if defect == "missing":
                        del payload["optimizer"]["state"][parameter_id]
                    elif defect == "none":
                        payload["optimizer"]["state"][parameter_id] = None
                    else:
                        payload["optimizer"]["state"][parameter_id]["step"] = bad_steps[defect].clone()

                digest = _resign(path, damage)
                target = _components(901 + index)
                before = {name: value.detach().clone() for name, value in
                          target[0].workspace.state_dict().items()}
                before_optimizer = copy.deepcopy(target[1].state_dict())
                before_scheduler = copy.deepcopy(target[2].state_dict())
                before_sampler = target[3].state_dict()
                before_noise = target[4].get_state().clone()
                before_global = _global_rng_state()
                with self.subTest(load_defect=defect), self.assertRaises(ValueError):
                    load_training_checkpoint(path, expected_sha256=digest, host=target[0],
                        optimizer=target[1], scheduler=target[2], scaler=None,
                        sampler=target[3], noise_generator=target[4], identities=IDENTITIES)
                self.assertTrue(all(torch.equal(before[name], value) for name, value in
                                    target[0].workspace.state_dict().items()))
                self.assertTrue(_nested_equal(before_optimizer, target[1].state_dict()))
                self.assertTrue(_nested_equal(before_scheduler, target[2].state_dict()))
                self.assertTrue(_nested_equal(before_sampler, target[3].state_dict()))
                self.assertTrue(torch.equal(before_noise, target[4].get_state()))
                self.assertTrue(_nested_equal(before_global, _global_rng_state()))


class OptionalDependencyImportTests(unittest.TestCase):
    def test_import_absence_and_transitive_failures_are_narrowly_handled_simulations(self):
        root = Path(__file__).resolve().parents[1]
        stdlib_discovery = subprocess.run(
            [sys.executable, "-S", "-m", "unittest", "discover", "-s", "tests",
             "-p", "test_training_resume.py", "-k",
             "test_partial_epoch_state_is_detached_and_replays_exact_ids", "-v"],
            cwd=root, capture_output=True, text=True, timeout=25)
        self.assertEqual(stdlib_discovery.returncode, 0,
                         stdlib_discovery.stdout + stdlib_discovery.stderr)
        self.assertIn("skipped", stdlib_discovery.stdout + stdlib_discovery.stderr)

        direct_absence = subprocess.run([sys.executable, "-c", r'''
import importlib.abc, pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd() / "tests"))
class MissingOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy":
            raise ModuleNotFoundError("SIMULATED direct optional absence", name="numpy")
sys.meta_path.insert(0, MissingOptional())
import test_training_resume as module
assert module.np is None
print("SIMULATED direct optional numpy absence was skipped")
'''], cwd=root, capture_output=True, text=True, timeout=25)
        self.assertEqual(direct_absence.returncode, 0,
                         direct_absence.stdout + direct_absence.stderr)
        self.assertIn("SIMULATED", direct_absence.stdout)

        transitive_failure = subprocess.run([sys.executable, "-c", r'''
import importlib.abc, pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd() / "tests"))
class BrokenOptionalImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy":
            raise ModuleNotFoundError("SIMULATED broken NumPy dependency", name="numpy.core")
sys.meta_path.insert(0, BrokenOptionalImport())
try:
    import test_training_resume
except ModuleNotFoundError as error:
    if error.name != "numpy.core":
        raise
    print("SIMULATED transitive numpy.core failure propagated")
else:
    raise SystemExit("transitive optional import failure was swallowed")
'''], cwd=root, capture_output=True, text=True, timeout=25)
        self.assertEqual(transitive_failure.returncode, 0,
                         transitive_failure.stdout + transitive_failure.stderr)
        self.assertIn("SIMULATED transitive numpy.core failure propagated", transitive_failure.stdout)

        # These simulated source/API failures are reachable only with the ML stack importable.
        if np is None or torch is None or Qwen2Config is None:
            return

        source_failure = subprocess.run([sys.executable, "-c", r'''
import importlib.abc, pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd() / "tests"))
class BrokenSourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "dwr.models.qwen2":
            raise RuntimeError("SIMULATED source import failure")
sys.meta_path.insert(0, BrokenSourceFinder())
try:
    import test_training_resume
except RuntimeError as error:
    if str(error) != "SIMULATED source import failure":
        raise
    print("SIMULATED source import failure propagated")
else:
    raise SystemExit("source import failure was swallowed")
'''], cwd=root, capture_output=True, text=True, timeout=25)
        self.assertEqual(source_failure.returncode, 0,
                         source_failure.stdout + source_failure.stderr)
        self.assertIn("SIMULATED source import failure propagated", source_failure.stdout)

        broken_api = subprocess.run([sys.executable, "-c", r'''
import pathlib, sys, types
sys.path.insert(0, str(pathlib.Path.cwd() / "tests"))
sys.modules["transformers"] = types.ModuleType("transformers")
try:
    import test_training_resume
except ImportError as error:
    if "Qwen2Config" not in str(error):
        raise
    print("SIMULATED broken transformers API propagated")
else:
    raise SystemExit("broken transformers API was swallowed")
'''], cwd=root, capture_output=True, text=True, timeout=25)
        self.assertEqual(broken_api.returncode, 0, broken_api.stdout + broken_api.stderr)
        self.assertIn("SIMULATED broken transformers API propagated", broken_api.stdout)


if __name__ == "__main__":
    unittest.main()
