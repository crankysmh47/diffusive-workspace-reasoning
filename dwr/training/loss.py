"""Strict answer-only causal language-model loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class AnswerLoss:
    loss: Tensor
    supervised_tokens: int


def answer_only_loss(logits: Tensor, labels: Tensor) -> AnswerLoss:
    """Shift once, select supervised prediction positions, and average token CE."""
    if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("logits and labels must be tensors")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("logits must use float16, bfloat16, float32, or float64")
    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, sequence, vocabulary)")
    batch, length, vocab = logits.shape
    if batch <= 0 or length < 2 or vocab <= 0:
        raise ValueError("logits batch/vocabulary must be nonempty and sequence length at least two")
    if labels.dtype != torch.long:
        raise TypeError("labels must use torch.long")
    if labels.shape != (batch, length):
        raise ValueError("labels shape must match logits batch and sequence dimensions")
    if labels.device != logits.device:
        raise ValueError("labels and logits must share a device")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must be finite")
    invalid_negative = (labels < 0) & (labels != -100)
    if invalid_negative.any().item() or (labels >= vocab).any().item():
        raise ValueError("labels must be -100 or a vocabulary ID")
    if (labels[:, 0] != -100).any().item():
        raise ValueError("label column zero must be ignored because no preceding logit predicts it")

    shifted_labels = labels[:, 1:]
    supervised = shifted_labels != -100
    row_counts = supervised.sum(dim=1)
    if (row_counts == 0).any().item():
        raise ValueError("every batch row must contain a supervised shifted token")
    selected_logits = logits[:, :-1, :][supervised]
    selected_labels = shifted_labels[supervised]
    loss_dtype = torch.float64 if logits.dtype is torch.float64 else torch.float32
    summed = F.cross_entropy(selected_logits.to(dtype=loss_dtype), selected_labels, reduction="sum")
    count = int(supervised.sum().item())
    loss = summed / count
    if not torch.isfinite(loss).item():
        raise ValueError("answer-only loss is nonfinite")
    return AnswerLoss(loss, count)
