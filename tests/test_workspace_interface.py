"""CPU checks for the invariant set reader and frozen-tail gradients."""

from __future__ import annotations

import unittest
import inspect

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None

if torch is not None:
    from dwr.models import FixedWorkspace, SetCrossAttentionAdapter


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class WorkspaceInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.old_threads)

    def setUp(self) -> None:
        torch.manual_seed(314159)
        self.reader = SetCrossAttentionAdapter(
            decoder_dim=8, latent_dim=6, num_heads=2
        ).to(dtype=torch.float64)
        self.queries = torch.randn(2, 3, 8, dtype=torch.float64)
        self.memory = torch.randn(2, 4, 6, dtype=torch.float64)

    def test_gate_and_output_shape(self) -> None:
        self.assertAlmostEqual(self.reader.gate.item(), 0.1, places=7)
        actual = self.reader(self.queries, self.memory)
        self.assertEqual(actual.shape, self.queries.shape)
        self.assertEqual(actual.dtype, self.queries.dtype)

    def test_whole_slot_permutation_invariance_for_ex_and_idx(self) -> None:
        permutation = torch.tensor([2, 0, 3, 1])
        for mode in ("exchangeable", "indexed"):
            with self.subTest(mode=mode):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(442)
                    workspace = FixedWorkspace(
                        5, num_slots=4, latent_dim=6, num_heads=2,
                        max_steps=2, slot_identity=mode,
                    ).to(dtype=torch.float64)
                context = torch.randn(2, 5, 5, dtype=torch.float64)
                noise = torch.randn(2, 4, 6, dtype=torch.float64)
                memory = workspace(context, noise=noise, steps=2)
                actual = self.reader(self.queries, memory)
                permuted = self.reader(self.queries, memory[:, permutation, :])
                torch.testing.assert_close(actual, permuted, atol=1e-9, rtol=1e-9)

    def test_query_positions_are_processed_independently(self) -> None:
        changed = self.queries.clone()
        changed[:, 2, :] += torch.tensor(
            [31.0, -7.0, 13.0, 2.0, -19.0, 5.0, 23.0, -11.0],
            dtype=torch.float64,
        )
        before = self.reader(self.queries, self.memory)
        after = self.reader(changed, self.memory)
        torch.testing.assert_close(before[:, :2], after[:, :2], atol=1e-9, rtol=1e-9)
        self.assertGreater((before[:, 2] - after[:, 2]).abs().max().item(), 1e-6)

    def test_reader_has_no_mask_or_slot_order_argument(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(self.reader.forward).parameters),
            ("queries", "workspace"),
        )
        with self.assertRaises(TypeError):
            self.reader(self.queries, self.memory, key_padding_mask=None)

    def test_rejects_invalid_constructor_dimensions_and_gate(self) -> None:
        for bad in (True, 0, -1, 2.5):
            with self.subTest(decoder_dim=bad):
                with self.assertRaises((TypeError, ValueError)):
                    SetCrossAttentionAdapter(bad, 6, num_heads=2)
        with self.assertRaises(ValueError):
            SetCrossAttentionAdapter(7, 6, num_heads=2)
        for bad_gate in (float("nan"), float("inf"), 1e308):
            with self.subTest(gate_init=bad_gate):
                with self.assertRaises(ValueError):
                    SetCrossAttentionAdapter(8, 6, num_heads=2, gate_init=bad_gate)
        with self.assertRaises(TypeError):
            SetCrossAttentionAdapter(8, 6, num_heads=2, gate_init=True)

    def test_rejects_nonfinite_reader_output(self) -> None:
        with torch.no_grad():
            self.reader.gate.fill_(float("inf"))
        with self.assertRaisesRegex(ValueError, "output must be finite"):
            self.reader(self.queries, self.memory)

    def test_rejects_bad_tensor_types_shapes_empty_axes_and_nonfinite_values(self) -> None:
        bad_cases = (
            (self.queries.tolist(), self.memory, TypeError),
            (self.queries[:, :, :7], self.memory, ValueError),
            (self.queries[:, :0], self.memory, ValueError),
            (self.queries, self.memory[:, :0], ValueError),
            (self.queries, self.memory[:, :, :5], ValueError),
            (self.queries, self.memory[:1], ValueError),
            (self.queries.to(torch.int64), self.memory.to(torch.int64), TypeError),
            (self.queries, self.memory.to(torch.float32), ValueError),
            (self.queries, torch.empty((2, 4, 6), dtype=torch.float64, device="meta"), ValueError),
            (self.queries.clone().index_put((torch.tensor([0]), torch.tensor([0]), torch.tensor([0])), torch.tensor([float("nan")], dtype=torch.float64)), self.memory, ValueError),
            (self.queries, self.memory.clone().index_put((torch.tensor([0]), torch.tensor([0]), torch.tensor([0])), torch.tensor([float("inf")], dtype=torch.float64)), ValueError),
        )
        for queries, memory, error in bad_cases:
            with self.subTest(error=error, query_shape=getattr(queries, "shape", None)):
                with self.assertRaises(error):
                    self.reader(queries, memory)

    def test_state_dict_strict_roundtrip_preserves_shapes_and_ex_idx_distinction(self) -> None:
        reader_copy = SetCrossAttentionAdapter(8, 6, num_heads=2).to(dtype=torch.float64)
        reader_copy.load_state_dict(self.reader.state_dict(), strict=True)
        self.assertEqual(
            {key: tuple(value.shape) for key, value in self.reader.state_dict().items()},
            {key: tuple(value.shape) for key, value in reader_copy.state_dict().items()},
        )
        torch.testing.assert_close(
            self.reader(self.queries, self.memory),
            reader_copy(self.queries, self.memory),
            atol=0,
            rtol=0,
        )

        for mode in ("exchangeable", "indexed"):
            with self.subTest(mode=mode):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(2718)
                    original = FixedWorkspace(
                        5, num_slots=4, latent_dim=6, num_heads=2,
                        max_steps=4, slot_identity=mode,
                    ).to(dtype=torch.float64)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(999)
                    restored = FixedWorkspace(
                        5, num_slots=4, latent_dim=6, num_heads=2,
                        max_steps=4, slot_identity=mode,
                    ).to(dtype=torch.float64)
                restored.load_state_dict(original.state_dict(), strict=True)
                self.assertEqual(
                    {k: tuple(v.shape) for k, v in original.state_dict().items()},
                    {k: tuple(v.shape) for k, v in restored.state_dict().items()},
                )
                self.assertEqual("slot_identity" in original.state_dict(), mode == "indexed")
                for key, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, restored.state_dict()[key]))

    def test_reader_initialization_uses_a_matched_independent_rng_stream(self) -> None:
        readers = []
        for mode in ("exchangeable", "indexed"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7001)
                workspace = FixedWorkspace(
                    5, num_slots=4, latent_dim=6, num_heads=2,
                    max_steps=4, slot_identity=mode,
                )
                # The reader gets its own seed after each differently sized
                # workspace constructor and must match across the two methods.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(8812)
                    readers.append(
                        SetCrossAttentionAdapter(8, 6, num_heads=2)
                    )
                del workspace
        for key, value in readers[0].state_dict().items():
            self.assertTrue(torch.equal(value, readers[1].state_dict()[key]), key)

    def test_real_workspace_reader_frozen_tail_optimizer_step_keeps_gradients_and_freeze(self) -> None:
        for mode in ("exchangeable", "indexed"):
            with self.subTest(mode=mode):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(1201)
                    workspace = FixedWorkspace(
                        5, num_slots=4, latent_dim=6, num_heads=2,
                        max_steps=4, slot_identity=mode,
                    ).to(dtype=torch.float64)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(8891)
                    reader = SetCrossAttentionAdapter(
                        8, 6, num_heads=2
                    ).to(dtype=torch.float64)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(3407)
                    frozen_tail = torch.nn.Linear(8, 2, bias=True).to(dtype=torch.float64)
                frozen_tail.requires_grad_(False)
                frozen_before = {
                    k: v.detach().clone() for k, v in frozen_tail.state_dict().items()
                }
                context = torch.randn(2, 5, 5, dtype=torch.float64)
                noise = torch.randn(2, 4, 6, dtype=torch.float64, requires_grad=True)
                decoder_queries = torch.randn(2, 3, 8, dtype=torch.float64)
                trainable = list(workspace.named_parameters()) + [
                    (f"reader.{name}", parameter)
                    for name, parameter in reader.named_parameters()
                ]
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in trainable
                }
                optimizer = torch.optim.SGD(
                    [parameter for _, parameter in trainable], lr=0.01
                )

                optimizer.zero_grad(set_to_none=True)
                final, states = workspace(
                    context, noise=noise, steps=4, return_states=True
                )
                for state in states:
                    state.retain_grad()
                decoded = frozen_tail(reader(decoder_queries, final)).square().mean()
                decoded.backward()

                for index, state in enumerate(states):
                    self.assertIsNotNone(state.grad, f"Z{index} did not receive a gradient")
                    self.assertTrue(torch.isfinite(state.grad).all().item())
                    self.assertGreater(state.grad.abs().sum().item(), 0.0)
                self.assertIsNotNone(noise.grad)
                self.assertTrue(torch.isfinite(noise.grad).all().item())
                self.assertGreater(noise.grad.abs().sum().item(), 0.0)
                for name, parameter in trainable:
                    self.assertIsNotNone(parameter.grad, f"{name} did not receive a gradient")
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)
                optimizer.step()
                for name, parameter in trainable:
                    self.assertFalse(
                        torch.equal(parameter.detach(), before[name]),
                        f"{name} received a gradient but did not update",
                    )
                for key, value in frozen_tail.state_dict().items():
                    self.assertTrue(torch.equal(value, frozen_before[key]), key)


if __name__ == "__main__":
    unittest.main()
