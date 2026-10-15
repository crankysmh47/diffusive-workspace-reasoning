from __future__ import annotations

import unittest
import math

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from dwr.tasks import Constraint, ConstraintTask, GivenAssignment, RelationKind, Solution
    from dwr.tasks import enumerate_solutions
    try:
        from dwr.training.batches import SourceExample, TokenizerIdentity, collate_source_examples
        from dwr.training.loss import answer_only_loss
    except ModuleNotFoundError as error:
        if error.name not in ("dwr.training", "dwr.training.batches", "dwr.training.loss"):
            raise
        SourceExample = TokenizerIdentity = collate_source_examples = answer_only_loss = None
else:
    Constraint = ConstraintTask = GivenAssignment = RelationKind = Solution = enumerate_solutions = None
    SourceExample = TokenizerIdentity = collate_source_examples = answer_only_loss = None


class AsciiTokenizer:
    eos_token_id = 127
    pad_token_id = 127
    vocab_size = 127

    def __init__(self):
        self.calls = []

    def encode(self, text, *, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return [ord(char) for char in text]


def fixture_task(kind= "eq"):
    relation = RelationKind(kind)
    if relation is RelationKind.EQ:
        constraints = (Constraint(relation, ("v0", "v1")), Constraint(relation, ("v1", "v2")), Constraint(relation, ("v2", "v3")))
        solution = Solution((("v0", 0), ("v1", 0), ("v2", 0), ("v3", 0)))
    elif relation is RelationKind.NEQ:
        constraints = (Constraint(relation, ("v0", "v1")), Constraint(relation, ("v2", "v3")))
        solution = Solution((("v0", 0), ("v1", 1), ("v2", 0), ("v3", 1)))
    else:
        constraints = (Constraint(relation, ("v0", "v1", "v2")), Constraint(relation, ("v1", "v2", "v3")))
        solution = Solution((("v0", 0), ("v1", 1), ("v2", 1), ("v3", 2)))
    task = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2", "v3"), constraints, (GivenAssignment("v0", solution.assignments[0][1]),))
    return task, solution


def example(task=None, solution=None, *, example_id=None, split="source_train"):
    if task is None:
        task, solution = fixture_task()
    return SourceExample(example_id or "a" * 64, split, task, solution)


@unittest.skipIf(torch is None, "install the optional torch extra")
class TokenBatchTests(unittest.TestCase):
    def setUp(self):
        if collate_source_examples is None:
            self.fail("training batch API is not implemented")

    def collate(self, examples, tokenizer=None, **kwargs):
        options = dict(max_prompt_tokens=4096, max_answer_tokens=128, max_total_tokens=4224)
        options.update(kwargs)
        return collate_source_examples(
            examples, tokenizer or AsciiTokenizer(),
            tokenizer_identity=TokenizerIdentity("synthetic-ascii", "fixture-v1"),
            vocab_size=128, device="cpu", **options,
        )

    def test_exact_separate_prompt_gold_encoding_and_answer_labels(self):
        task, solution = fixture_task()
        tokenizer = AsciiTokenizer()
        batch = self.collate([example(task, solution)], tokenizer)
        expected_prompt = __import__("dwr.tasks", fromlist=["serialize_model_input"]).serialize_model_input(task) + "\n"
        expected_answer = '{"assignment":{"v0":0,"v1":0,"v2":0,"v3":0}}'
        self.assertEqual(tokenizer.calls, [(expected_prompt, False), (expected_answer, False)])
        self.assertEqual(batch.example_ids, ("a" * 64,))
        self.assertEqual(batch.splits, ("source_train",))
        self.assertEqual(batch.tokenizer_identity, TokenizerIdentity("synthetic-ascii", "fixture-v1"))
        self.assertEqual(batch.prompt_ids[0, :len(expected_prompt)].tolist(), list(map(ord, expected_prompt)))
        self.assertEqual(batch.decoder_ids[0, :len(expected_prompt)].tolist(), batch.prompt_ids[0, :len(expected_prompt)].tolist())
        self.assertEqual(batch.decoder_ids[0, len(expected_prompt):].tolist(), list(map(ord, expected_answer)) + [127])
        self.assertTrue(batch.prompt_mask[0, :len(expected_prompt)].all())
        self.assertTrue(batch.decoder_mask[0, :len(expected_prompt) + len(expected_answer) + 1].all())
        self.assertEqual(batch.labels[0, :len(expected_prompt)].tolist(), [-100] * len(expected_prompt))
        self.assertEqual(batch.labels[0, len(expected_prompt):].tolist(), list(map(ord, expected_answer)) + [127])
        self.assertEqual(tokenizer.vocab_size, 127)  # The model embedding size is explicitly 128.

    def test_right_padding_uses_lengths_even_when_pad_equals_eos(self):
        task1, sol1 = fixture_task("eq")
        task2, sol2 = fixture_task("neq")
        batch = self.collate([example(task1, sol1, example_id="a"*64), example(task2, sol2, example_id="b"*64)])
        self.assertEqual(batch.prompt_mask.dtype, torch.bool)
        self.assertEqual(batch.decoder_mask.dtype, torch.bool)
        self.assertEqual(batch.prompt_mask.sum(1).tolist(), [len(batch.prompt_ids[0][batch.prompt_mask[0]]), int(batch.prompt_mask[1].sum())])
        self.assertTrue((batch.prompt_ids[~batch.prompt_mask] == 127).all())
        self.assertTrue((batch.decoder_ids[~batch.decoder_mask] == 127).all())
        self.assertTrue((batch.labels[~batch.decoder_mask] == -100).all())
        self.assertTrue((batch.labels[0, batch.decoder_mask[0]][-1] == 127).item())
        for row in range(2):
            prompt_len = int(batch.prompt_mask[row].sum())
            self.assertTrue(torch.equal(batch.prompt_ids[row, :prompt_len], batch.decoder_ids[row, :prompt_len]))

    def test_rejects_duplicates_protected_bad_types_mixed_kind_invalid_gold_and_caps(self):
        item = example()
        with self.assertRaises(ValueError):
            self.collate([item, item])
        with self.assertRaises(ValueError):
            self.collate([example(split="development")])
        with self.assertRaises(TypeError):
            self.collate([{"example_id": "a" * 64}])
        task, solution = fixture_task()
        mixed = ConstraintTask(task.schema_version, task.modulus, task.variables,
            (task.constraints[0], Constraint(RelationKind.NEQ, ("v1", "v2"))), task.givens)
        with self.assertRaises(ValueError):
            self.collate([example(mixed, solution)])
        wrong = Solution((("v0", 0), ("v1", 1), ("v2", 1), ("v3", 1)))
        with self.assertRaises(ValueError):
            self.collate([example(task, wrong)])
        with self.assertRaises(ValueError):
            self.collate([item], max_prompt_tokens=3)
        with self.assertRaises(ValueError):
            self.collate([item], max_answer_tokens=2)
        with self.assertRaises(ValueError):
            self.collate([item], max_total_tokens=4)

    def test_rejects_invalid_tokenizer_ids_identity_and_unknown_config(self):
        class BadTokenizer(AsciiTokenizer):
            def encode(self, text, *, add_special_tokens):
                return [128]
        with self.assertRaises(ValueError):
            self.collate([example()], BadTokenizer())
        with self.assertRaises(TypeError):
            collate_source_examples([example()], AsciiTokenizer(), tokenizer_identity=TokenizerIdentity("x", "y"), vocab_size=128, max_prompt_tokens=10, max_answer_tokens=10, max_total_tokens=20, device="cpu", typo=True)


@unittest.skipIf(torch is None, "install the optional torch extra")
class AnswerLossTests(unittest.TestCase):
    def setUp(self):
        if answer_only_loss is None:
            self.fail("answer loss API is not implemented")

    def test_shifted_sum_divided_by_count_first_answer_and_eos(self):
        logits = torch.tensor([[[0.0, 1.0, -1.0], [2.0, 0.0, 1.0], [0.0, 0.0, 2.0], [1.0, 2.0, 0.0]]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[-100, 1, 2, 0]], dtype=torch.long)
        result = answer_only_loss(logits, labels)
        expected = sum(
            math.log(sum(math.exp(value) for value in logits[0, index].tolist()))
            - logits[0, index, target].item()
            for index, target in enumerate((1, 2, 0))
        ) / 3
        self.assertEqual(result.supervised_tokens, 3)
        self.assertAlmostEqual(result.loss.item(), expected, places=12)
        result.loss.backward()
        self.assertTrue(torch.equal(logits.grad[:, 0], torch.zeros_like(logits.grad[:, 0])) is False)
        self.assertTrue(torch.equal(logits.grad[:, 3], torch.zeros_like(logits.grad[:, 3])))

    def test_ignored_prediction_positions_do_not_affect_loss_but_answer_and_eos_do(self):
        logits = torch.randn(1, 5, 7, dtype=torch.float64)
        labels = torch.tensor([[-100, -100, 3, 4, 6]], dtype=torch.long)
        base = answer_only_loss(logits, labels).loss
        changed_ignored = logits.clone(); changed_ignored[0, 0, 0] += 100
        self.assertFalse(torch.equal(changed_ignored[0, 0], logits[0, 0]))
        self.assertEqual(answer_only_loss(changed_ignored, labels).loss.item(), base.item())
        changed_first = logits.clone(); changed_first[:, 1, 3] += 10
        changed_eos = logits.clone(); changed_eos[:, 3, 6] += 10
        self.assertNotEqual(answer_only_loss(changed_first, labels).loss.item(), base.item())
        self.assertNotEqual(answer_only_loss(changed_eos, labels).loss.item(), base.item())

    def test_preserves_declared_loss_dtype_and_rejects_invalid_inputs(self):
        for dtype, expected in ((torch.float16, torch.float32), (torch.bfloat16, torch.float32), (torch.float32, torch.float32), (torch.float64, torch.float64)):
            result = answer_only_loss(torch.randn(1, 3, 5, dtype=dtype), torch.tensor([[-100, 2, 4]]))
            self.assertEqual(result.loss.dtype, expected)
        valid = torch.randn(1, 3, 5)
        labels = torch.tensor([[-100, 2, 4]])
        for bad_logits, bad_labels in ((valid[0], labels), (valid, labels[:, :2]), (valid, labels.to(torch.int32))):
            with self.assertRaises((TypeError, ValueError)):
                answer_only_loss(bad_logits, bad_labels)
        with self.assertRaises(ValueError):
            answer_only_loss(valid, torch.full((1, 3), -100, dtype=torch.long))
        with self.assertRaises(ValueError):
            answer_only_loss(valid, torch.tensor([[-100, 5, 4]]))
        for invalid_label in (-99, -101):
            invalid = labels.clone(); invalid[0, 1] = invalid_label
            with self.assertRaises(ValueError):
                answer_only_loss(valid, invalid)
        nonfinite = valid.clone(); nonfinite[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            answer_only_loss(nonfinite, labels)


@unittest.skipIf(torch is None, "install the optional torch extra")
class NativeIntegrationTests(unittest.TestCase):
    def test_tiny_qwen2_k4_collator_loss_backward_and_frozen_weights(self):
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ModuleNotFoundError as error:
            if error.name != "transformers":
                raise
            self.skipTest("install the optional transformers extra")
        from dwr.models.qwen2 import Qwen2WorkspaceLM
        from dwr.models.workspace import FixedWorkspace
        task, solution = fixture_task("eq")
        self.assertEqual(len(enumerate_solutions(task).solutions), 1)
        config = Qwen2Config(vocab_size=128, hidden_size=12, intermediate_size=24, num_hidden_layers=3, num_attention_heads=3, num_key_value_heads=1, attention_dropout=0.0, tie_word_embeddings=True, pad_token_id=127, eos_token_id=127, _attn_implementation="eager")
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(53)
                backbone = Qwen2ForCausalLM(config).to(dtype=torch.float64)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(61)
                workspace = FixedWorkspace(12, num_slots=4, latent_dim=8, num_heads=2, max_steps=4).to(dtype=torch.float64)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(73)
                host = Qwen2WorkspaceLM(backbone, workspace, prompt_layer=1, adapter_layers=(0, 1, 2), reader_heads=3)
            before = {name: value.detach().clone() for name, value in backbone.state_dict().items()}
            batch = collate_source_examples([example(task, solution)], AsciiTokenizer(), tokenizer_identity=TokenizerIdentity("synthetic-ascii", "fixture-v1"), vocab_size=128, max_prompt_tokens=4096, max_answer_tokens=128, max_total_tokens=4224, device="cpu")
            noise = torch.randn(1, 4, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
            output = host(batch.prompt_ids, batch.prompt_mask, batch.decoder_ids, batch.decoder_mask, noise=noise, steps=4)
            result = answer_only_loss(output.logits, batch.labels)
            result.loss.backward()
            self.assertEqual(result.supervised_tokens, int((batch.labels[:, 1:] != -100).sum()))
            self.assertTrue(torch.isfinite(result.loss))
            groups = {"workspace": list(workspace.parameters()), "readers": list(host.readers.parameters())}
            for parameters in groups.values():
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
                self.assertTrue(any(torch.count_nonzero(p.grad).item() > 0 for p in parameters))
            before_new = {name: [p.detach().clone() for p in parameters] for name, parameters in groups.items()}
            optimizer = torch.optim.SGD([p for parameters in groups.values() for p in parameters], lr=1e-3)
            optimizer.step()
            for name, parameters in groups.items():
                self.assertTrue(any(not torch.equal(before, parameter) for before, parameter in zip(before_new[name], parameters)))
            self.assertTrue(all(torch.equal(value, backbone.state_dict()[name]) for name, value in before.items()))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
