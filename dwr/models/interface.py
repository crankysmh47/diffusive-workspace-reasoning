"""Permutation-invariant cross-attention from decoder queries to workspace slots."""

from __future__ import annotations

import math
import numbers

import torch
from torch import Tensor, nn


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-boolean integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class SetCrossAttentionAdapter(nn.Module):
    """Apply ``H + g * CA(LN_H(H), LN_Z(Z), LN_Z(Z))``.

    Workspace slots are treated as a set: cross-attention has no positional
    encoding, slot mask, or order-dependent pooling. Query positions are
    independent because attention only reduces over workspace keys/values.
    """

    def __init__(
        self,
        decoder_dim: int,
        latent_dim: int,
        *,
        num_heads: int,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.decoder_dim = _positive_int("decoder_dim", decoder_dim)
        self.latent_dim = _positive_int("latent_dim", latent_dim)
        self.num_heads = _positive_int("num_heads", num_heads)
        if self.decoder_dim % self.num_heads:
            raise ValueError("decoder_dim must be divisible by num_heads")
        if isinstance(gate_init, bool) or not isinstance(gate_init, numbers.Real):
            raise TypeError("gate_init must be a finite real number")
        if not math.isfinite(float(gate_init)):
            raise ValueError("gate_init must be finite")

        self.query_norm = nn.LayerNorm(self.decoder_dim)
        self.workspace_norm = nn.LayerNorm(self.latent_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.decoder_dim,
            self.num_heads,
            dropout=0.0,
            batch_first=True,
            kdim=self.latent_dim,
            vdim=self.latent_dim,
        )
        gate = torch.tensor(float(gate_init))
        if not torch.isfinite(gate).item():
            raise ValueError("gate_init is not finite in the module parameter dtype")
        self.gate = nn.Parameter(gate)

    def _validate_inputs(self, queries: Tensor, workspace: Tensor) -> None:
        if not isinstance(queries, Tensor) or not isinstance(workspace, Tensor):
            raise TypeError("queries and workspace must be tensors")
        if queries.ndim != 3:
            raise ValueError("queries must have shape (batch, length, decoder_dim)")
        if workspace.ndim != 3:
            raise ValueError("workspace must have shape (batch, slots, latent_dim)")
        if queries.shape[0] <= 0 or queries.shape[1] <= 0:
            raise ValueError("queries must have nonempty batch and sequence axes")
        if workspace.shape[0] != queries.shape[0]:
            raise ValueError("queries and workspace batch sizes must match")
        if workspace.shape[1] <= 0:
            raise ValueError("workspace must contain at least one slot")
        if queries.shape[2] != self.decoder_dim:
            raise ValueError("query width does not match decoder_dim")
        if workspace.shape[2] != self.latent_dim:
            raise ValueError("workspace width does not match latent_dim")
        if not queries.is_floating_point() or not workspace.is_floating_point():
            raise TypeError("queries and workspace must have floating dtypes")
        if queries.device != workspace.device:
            raise ValueError("queries and workspace must be on the same device")
        if queries.dtype != workspace.dtype:
            raise ValueError("queries and workspace must have the same dtype")
        parameter = self.gate
        if queries.device != parameter.device or queries.dtype != parameter.dtype:
            raise ValueError("inputs must match the adapter parameter device and dtype")
        if not torch.isfinite(queries).all().item():
            raise ValueError("queries must be finite")
        if not torch.isfinite(workspace).all().item():
            raise ValueError("workspace must be finite")

    def forward(self, queries: Tensor, workspace: Tensor) -> Tensor:
        """Read the complete workspace set for each query position."""
        self._validate_inputs(queries, workspace)
        normalized_workspace = self.workspace_norm(workspace)
        update, _ = self.cross_attention(
            self.query_norm(queries),
            normalized_workspace,
            normalized_workspace,
            need_weights=False,
        )
        output = queries + self.gate * update
        if not torch.isfinite(output).all().item():
            raise ValueError("reader output must be finite")
        return output
