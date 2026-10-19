"""Strict prompt/answer token collation for source examples."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

import torch
from torch import Tensor

from dwr.tasks import ConstraintTask, Solution, serialize_model_input, verify_answer
from dwr.tasks.run_support import stable_json


_ID = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_SPLITS = frozenset(("source_train", "source_validation"))
_SOURCE_KINDS = frozenset(("eq", "neq", "add_mod"))


@dataclass(frozen=True)
class SourceExample:
    example_id: str
    split: str
    task: ConstraintTask
    solution: Solution


@dataclass(frozen=True)
class TokenizerIdentity:
    name: str
    revision: str


@dataclass(frozen=True)
class TokenBatch:
    example_ids: tuple[str, ...]
    splits: tuple[str, ...]
    tokenizer_identity: TokenizerIdentity
    prompt_ids: Tensor
    prompt_mask: Tensor
    decoder_ids: Tensor
    decoder_mask: Tensor
    labels: Tensor


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-boolean integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _token_id(name: str, value: object, vocab_size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-boolean integer")
    if not 0 <= value < vocab_size:
        raise ValueError(f"{name} is outside the model embedding vocabulary")
    return value


def _encode(tokenizer: Any, text: str, *, name: str, vocab_size: int,
            eos_id: int, pad_id: int) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(encoded, (list, tuple)) or not encoded:
        raise ValueError(f"{name} encoding must be a nonempty integer-ID sequence")
    result = []
    for token in encoded:
        token_id = _token_id(f"{name} token ID", token, vocab_size)
        if token_id == eos_id or token_id == pad_id:
            raise ValueError(f"{name} encoding contains an interior EOS or padding token")
        result.append(token_id)
    return result


def _validate_example(example: object) -> SourceExample:
    if not isinstance(example, SourceExample):
        raise TypeError("each example must be a SourceExample")
    if not isinstance(example.example_id, str) or _ID.fullmatch(example.example_id) is None:
        raise ValueError("example_id must be lowercase SHA-256 text")
    if example.split not in _SOURCE_SPLITS:
        raise ValueError("only source_train and source_validation examples are permitted")
    if not isinstance(example.task, ConstraintTask) or not isinstance(example.solution, Solution):
        raise TypeError("task and solution must use the strict task schema types")
    kinds = {constraint.kind.value for constraint in example.task.constraints}
    if len(kinds) != 1 or not kinds <= _SOURCE_KINDS:
        raise ValueError("task must contain exactly one source primitive kind")
    answer_object = {"assignment": {name: value for name, value in example.solution.assignments}}
    answer = stable_json(answer_object)
    if not verify_answer(example.task, answer).valid:
        raise ValueError("solution does not pass the strict task verifier")
    return example


def collate_source_examples(
    examples: Sequence[SourceExample],
    tokenizer: Any,
    *,
    tokenizer_identity: TokenizerIdentity,
    vocab_size: int,
    max_prompt_tokens: int,
    max_answer_tokens: int,
    max_total_tokens: int,
    device: str | torch.device,
) -> TokenBatch:
    """Frame, encode, and right-pad strict source examples without truncation."""
    if not isinstance(examples, (list, tuple)) or not examples:
        raise ValueError("examples must be a nonempty list or tuple")
    if not isinstance(tokenizer_identity, TokenizerIdentity):
        raise TypeError("tokenizer_identity must be a TokenizerIdentity")
    if (not isinstance(tokenizer_identity.name, str) or not tokenizer_identity.name.strip()
            or not isinstance(tokenizer_identity.revision, str) or not tokenizer_identity.revision.strip()):
        raise ValueError("tokenizer identity name and revision must be nonempty strings")
    vocab_size = _positive_int("vocab_size", vocab_size)
    max_prompt_tokens = _positive_int("max_prompt_tokens", max_prompt_tokens)
    max_answer_tokens = _positive_int("max_answer_tokens", max_answer_tokens)
    max_total_tokens = _positive_int("max_total_tokens", max_total_tokens)
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be explicit torch device text or torch.device")
    target_device = torch.device(device)
    if not isinstance(target_device, torch.device):
        raise TypeError("device must identify a torch device")
    if not callable(getattr(tokenizer, "encode", None)):
        raise TypeError("tokenizer must expose encode(text, add_special_tokens=False)")
    eos_id = _token_id("eos_token_id", getattr(tokenizer, "eos_token_id", None), vocab_size)
    pad_id = _token_id("pad_token_id", getattr(tokenizer, "pad_token_id", None), vocab_size)

    checked = [_validate_example(example) for example in examples]
    ids = [example.example_id for example in checked]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate example_id in batch")

    prompts: list[list[int]] = []
    answers: list[list[int]] = []
    for example in checked:
        prompt = _encode(tokenizer, serialize_model_input(example.task) + "\n",
                         name="prompt", vocab_size=vocab_size, eos_id=eos_id, pad_id=pad_id)
        answer_text = stable_json({"assignment": {
            name: value for name, value in example.solution.assignments
        }})
        answer = _encode(tokenizer, answer_text, name="answer", vocab_size=vocab_size,
                         eos_id=eos_id, pad_id=pad_id)
        answer.append(eos_id)
        if len(prompt) > max_prompt_tokens:
            raise ValueError("prompt exceeds max_prompt_tokens")
        if len(answer) > max_answer_tokens:
            raise ValueError("answer including EOS exceeds max_answer_tokens")
        if len(prompt) + len(answer) > max_total_tokens:
            raise ValueError("prompt plus answer exceeds max_total_tokens")
        prompts.append(prompt)
        answers.append(answer)

    prompt_width = max(map(len, prompts))
    decoder_width = max(len(prompt) + len(answer) for prompt, answer in zip(prompts, answers))
    prompt_ids = torch.full((len(checked), prompt_width), pad_id, dtype=torch.long, device=target_device)
    prompt_mask = torch.zeros((len(checked), prompt_width), dtype=torch.bool, device=target_device)
    decoder_ids = torch.full((len(checked), decoder_width), pad_id, dtype=torch.long, device=target_device)
    decoder_mask = torch.zeros((len(checked), decoder_width), dtype=torch.bool, device=target_device)
    labels = torch.full((len(checked), decoder_width), -100, dtype=torch.long, device=target_device)
    for row, (prompt, answer) in enumerate(zip(prompts, answers)):
        prompt_length = len(prompt)
        total_length = prompt_length + len(answer)
        prompt_ids[row, :prompt_length] = torch.tensor(prompt, dtype=torch.long, device=target_device)
        prompt_mask[row, :prompt_length] = True
        decoder_ids[row, :prompt_length] = torch.tensor(prompt, dtype=torch.long, device=target_device)
        decoder_ids[row, prompt_length:total_length] = torch.tensor(answer, dtype=torch.long, device=target_device)
        decoder_mask[row, :total_length] = True
        labels[row, prompt_length:total_length] = torch.tensor(answer, dtype=torch.long, device=target_device)

    return TokenBatch(tuple(ids), tuple(example.split for example in checked), tokenizer_identity,
                      prompt_ids, prompt_mask, decoder_ids, decoder_mask, labels)
