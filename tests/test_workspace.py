"""CPU contract tests for the fixed-N recurrent workspace core."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("optional torch dependency is not installed") from exc

from dwr.models.workspace import FixedWorkspace


DTYPE = torch.float64
ATOL = 1e-9
RTOL = 1e-9


def make_workspace(kind: str = "exchangeable", *, steps: int = 4) -> FixedWorkspace:
    return FixedWorkspace(
        context_dim=6,
        num_slots=4,
        latent_dim=8,
        num_heads=2,
        max_steps=steps,
        slot_identity=kind,
    ).to(dtype=DTYPE)


class FixedWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._previous_threads)

    def setUp(self) -> None:
        torch.manual_seed(701)
        self.context = torch.randn(2, 5, 6, dtype=DTYPE)
        self.noise = torch.randn(2, 4, 8, dtype=DTYPE)

    def test_noise_sampler_uses_only_explicit_generator_and_fixed_shape(self) -> None:
        workspace = make_workspace()
        generator = torch.Generator(device="cpu").manual_seed(19)
        before = torch.random.get_rng_state().clone()
        first = workspace.sample_noise(3, generator=generator, device="cpu", dtype=DTYPE)
        second = workspace.sample_noise(3, generator=generator, device="cpu", dtype=DTYPE)
        replay_generator = torch.Generator(device="cpu").manual_seed(19)
        replay_first = workspace.sample_noise(
            3, generator=replay_generator, device="cpu", dtype=DTYPE
        )
        replay_second = workspace.sample_noise(
            3, generator=replay_generator, device="cpu", dtype=DTYPE
        )
        self.assertEqual(first.shape, (3, 4, 8))
        self.assertEqual(first.dtype, DTYPE)
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(first.view(torch.uint8), replay_first.view(torch.uint8)))
        self.assertTrue(torch.equal(second.view(torch.uint8), replay_second.view(torch.uint8)))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_ex_and_idx_share_base_weights_and_idx_adds_exactly_one_table(self) -> None:
        caller_rng = torch.random.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(33)
            exchangeable = make_workspace("exchangeable")
        self.assertTrue(torch.equal(caller_rng, torch.random.get_rng_state()))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(33)
            indexed = make_workspace("indexed")
        self.assertTrue(torch.equal(caller_rng, torch.random.get_rng_state()))

        ex_parameters = dict(exchangeable.named_parameters())
        idx_parameters = dict(indexed.named_parameters())
        self.assertEqual(set(idx_parameters) - set(ex_parameters), {"slot_identity"})
        self.assertEqual(idx_parameters["slot_identity"].numel(), 4 * 8)
        for name, value in ex_parameters.items():
            self.assertTrue(torch.equal(value, idx_parameters[name]), name)
        self.assertFalse(hasattr(exchangeable, "slot_identity"))

    def test_index_embeddings_are_added_once_and_k0_skips_update(self) -> None:
        workspace = make_workspace("indexed")
        calls = 0
        original_forward = workspace.delta.forward

        def counted(*args: object, **kwargs: object) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return original_forward(*args, **kwargs)

        workspace.delta.forward = counted  # type: ignore[method-assign]
        initial = self.noise + workspace.slot_identity.unsqueeze(0)
        at_zero = workspace(self.context, noise=self.noise, steps=0)
        self.assertTrue(torch.equal(at_zero, initial))
        self.assertEqual(calls, 0)
        at_two, states = workspace(
            self.context, noise=self.noise, steps=2, return_states=True
        )
        expected = initial
        for _ in range(2):
            expected = expected + original_forward(workspace.step_norm(expected), self.context)
        torch.testing.assert_close(at_two, expected, atol=ATOL, rtol=RTOL)
        self.assertEqual(calls, 2)
        self.assertEqual(len(states), 3)
        self.assertTrue(torch.equal(states[0], initial))

    def test_exchangeable_updates_are_equivariant_at_multiple_depths(self) -> None:
        workspace = make_workspace()
        permutation = torch.tensor([2, 0, 3, 1])
        for steps in (1, 4):
            original = workspace(self.context, noise=self.noise, steps=steps)
            permuted = workspace(
                self.context, noise=self.noise[:, permutation], steps=steps
            )
            torch.testing.assert_close(
                permuted, original[:, permutation], atol=ATOL, rtol=RTOL
            )

    def test_padding_is_ignored_and_padded_context_perturbations_are_inert(self) -> None:
        workspace = make_workspace()
        mask = torch.tensor([[False, False, True, True, False], [False, True, False, True, True]])
        altered = self.context.clone()
        altered[mask] = 10_000.0
        actual = workspace(self.context, noise=self.noise, prompt_padding_mask=mask, steps=2)
        perturbed = workspace(altered, noise=self.noise, prompt_padding_mask=mask, steps=2)
        torch.testing.assert_close(actual, perturbed, atol=ATOL, rtol=RTOL)

    def test_every_returned_recurrent_state_receives_terminal_loss_gradient(self) -> None:
        workspace = make_workspace(steps=4)
        output, states = workspace(
            self.context,
            noise=self.noise.clone().requires_grad_(),
            steps=4,
            return_states=True,
        )
        loss = output.square().mean()
        gradients = torch.autograd.grad(loss, states)
        self.assertEqual(len(gradients), 5)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_k1_is_one_residual_application_and_returned_states_are_live(self) -> None:
        workspace = make_workspace()
        result, states = workspace(
            self.context, noise=self.noise, steps=1, return_states=True
        )
        initial = self.noise
        expected = initial + workspace.delta(workspace.step_norm(initial), self.context)
        torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)
        self.assertEqual(len(states), 2)
        self.assertIs(states[-1], result)
        self.assertTrue(states[-1].requires_grad)

    def test_strict_inputs_reject_invalid_counts_shapes_masks_and_values(self) -> None:
        workspace = make_workspace()
        with self.assertRaises((TypeError, ValueError)):
            make_workspace(steps=True)
        with self.assertRaises((TypeError, ValueError)):
            FixedWorkspace(context_dim=6, num_slots=True)
        with self.assertRaises((TypeError, ValueError)):
            FixedWorkspace(context_dim=6, latent_dim=0)
        with self.assertRaises((TypeError, ValueError)):
            FixedWorkspace(context_dim=6, latent_dim=7, num_heads=2)
        with self.assertRaises((TypeError, ValueError)):
            FixedWorkspace(context_dim=6, max_steps=0)
        with self.assertRaises((TypeError, ValueError)):
            make_workspace("typo")
        with self.assertRaises((TypeError, ValueError)):
            workspace(self.context, noise=self.noise, steps=True)
        with self.assertRaises((TypeError, ValueError)):
            workspace(self.context, noise=self.noise[:, :3], steps=1)
        with self.assertRaises((TypeError, ValueError)):
            workspace(self.context, noise=self.noise, prompt_padding_mask=torch.zeros(2, 5))
        with self.assertRaises((TypeError, ValueError)):
            workspace(
                self.context,
                noise=self.noise,
                prompt_padding_mask=torch.ones(2, 5, dtype=torch.bool),
            )
        with self.assertRaises((TypeError, ValueError)):
            workspace(self.context[:, :0], noise=self.noise)
        with self.assertRaises((TypeError, ValueError)):
            workspace(self.context, noise=self.noise, steps=5)
        with self.assertRaises((TypeError, ValueError)):
            workspace(self.context, noise=self.noise.float())
        with self.assertRaises((TypeError, ValueError)):
            workspace(self.context.requires_grad_(), noise=self.noise, steps=1)
        bad_context = self.context.detach().clone()
        bad_context[0, 0, 0] = float("inf")
        with self.assertRaises((TypeError, ValueError)):
            workspace(bad_context, noise=self.noise)
        bad_noise = self.noise.clone()
        bad_noise[0, 0, 0] = float("nan")
        with self.assertRaises((TypeError, ValueError)):
            workspace(self.context.detach(), noise=bad_noise, steps=1)

    def test_batched_and_individual_calls_agree(self) -> None:
        workspace = make_workspace()
        batched = workspace(self.context, noise=self.noise, steps=4)
        individual = torch.cat(
            [
                workspace(self.context[i : i + 1], noise=self.noise[i : i + 1], steps=4)
                for i in range(self.context.shape[0])
            ],
            dim=0,
        )
        torch.testing.assert_close(batched, individual, atol=ATOL, rtol=RTOL)

    def test_nonfinite_indexed_initial_state_fails_even_at_k0(self) -> None:
        workspace = make_workspace("indexed")
        with torch.no_grad():
            workspace.slot_identity[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "initial workspace state"):
            workspace(self.context, noise=self.noise, steps=0)


if __name__ == "__main__":
    unittest.main()
