"""Fixed-size recurrent latent workspace and its EX/IDX update block.

To match initial parameters across methods, construct each method inside its
own ``torch.random.fork_rng`` context and call ``torch.manual_seed`` with the
same model-initialization seed. Workspace noise uses ``sample_noise`` and a
separate explicit generator; it never consumes the model initialization
stream.
"""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor, nn


States: TypeAlias = tuple[Tensor, ...]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-boolean integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class WorkspaceDelta(nn.Module):
    """One shared realization of ``F(U, X) = a + b + c``."""

    def __init__(
        self,
        context_dim: int,
        latent_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        context_dim = _positive_int("context_dim", context_dim)
        latent_dim = _positive_int("latent_dim", latent_dim)
        num_heads = _positive_int("num_heads", num_heads)
        if latent_dim % num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")

        self.self_attention = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.context_attention = nn.MultiheadAttention(
            latent_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim,
        )
        self.feed_forward_norm = nn.LayerNorm(latent_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )

    def forward(
        self,
        normalized_workspace: Tensor,
        prompt_states: Tensor,
        prompt_padding_mask: Tensor | None = None,
    ) -> Tensor:
        a, _ = self.self_attention(
            normalized_workspace,
            normalized_workspace,
            normalized_workspace,
            need_weights=False,
        )
        b, _ = self.context_attention(
            normalized_workspace + a,
            prompt_states,
            prompt_states,
            key_padding_mask=prompt_padding_mask,
            need_weights=False,
        )
        c = self.feed_forward(self.feed_forward_norm(normalized_workspace + a + b))
        return a + b + c


class FixedWorkspace(nn.Module):
    """A fixed-N EX or IDX workspace updated by one shared residual block."""

    def __init__(
        self,
        context_dim: int,
        *,
        num_slots: int = 64,
        latent_dim: int = 768,
        num_heads: int = 12,
        max_steps: int = 4,
        slot_identity: str = "exchangeable",
    ) -> None:
        super().__init__()
        self.context_dim = _positive_int("context_dim", context_dim)
        self.num_slots = _positive_int("num_slots", num_slots)
        self.latent_dim = _positive_int("latent_dim", latent_dim)
        self.num_heads = _positive_int("num_heads", num_heads)
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("max_steps must be a non-boolean integer")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        if slot_identity not in ("exchangeable", "indexed"):
            raise ValueError("slot_identity must be 'exchangeable' or 'indexed'")
        if self.latent_dim % self.num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.identity_mode = slot_identity

        # Construct all shared components before the IDX-only table so the
        # common parameters have identical initialization under matched seeds.
        self.step_norm = nn.LayerNorm(self.latent_dim)
        self.delta = WorkspaceDelta(
            self.context_dim, self.latent_dim, self.num_heads
        )
        if slot_identity == "indexed":
            self.slot_identity = nn.Parameter(
                torch.empty(self.num_slots, self.latent_dim)
            )
            nn.init.normal_(self.slot_identity, mean=0.0, std=0.02)

    @property
    def identity_embeddings(self) -> Tensor:
        """Return the IDX table; EX intentionally has no such parameter."""
        if self.identity_mode != "indexed":
            raise AttributeError("exchangeable workspaces have no identity embeddings")
        return self.slot_identity

    def sample_noise(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """Sample iid Gaussian ``B,N,d`` noise using only ``generator``."""
        batch_size = _positive_int("batch_size", batch_size)
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be an explicit torch.Generator")
        device = torch.device(device)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating torch dtype")
        if torch.device(generator.device) != device:
            raise ValueError("generator device must match requested noise device")
        return torch.randn(
            batch_size,
            self.num_slots,
            self.latent_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def _validate_inputs(
        self,
        prompt_states: Tensor,
        noise: Tensor,
        prompt_padding_mask: Tensor | None,
        steps: int,
    ) -> None:
        if not isinstance(prompt_states, Tensor) or not isinstance(noise, Tensor):
            raise TypeError("prompt_states and noise must be tensors")
        if prompt_states.ndim != 3:
            raise ValueError("prompt_states must have shape (batch, tokens, context_dim)")
        if noise.ndim != 3:
            raise ValueError("noise must have shape (batch, slots, latent_dim)")
        if prompt_states.shape[0] <= 0 or prompt_states.shape[1] <= 0:
            raise ValueError("prompt_states must have a nonempty batch and token axis")
        if prompt_states.shape[2] != self.context_dim:
            raise ValueError("prompt_states context width does not match context_dim")
        if noise.shape != (prompt_states.shape[0], self.num_slots, self.latent_dim):
            raise ValueError("noise shape must match batch size and this fixed workspace")
        if not prompt_states.is_floating_point() or not noise.is_floating_point():
            raise TypeError("prompt_states and noise must have floating dtypes")
        if prompt_states.device != noise.device:
            raise ValueError("prompt_states and noise must be on the same device")
        if prompt_states.dtype != noise.dtype:
            raise ValueError("prompt_states and noise must have the same dtype")
        parameter = next(self.parameters())
        if prompt_states.device != parameter.device or prompt_states.dtype != parameter.dtype:
            raise ValueError("inputs must match the workspace parameter device and dtype")
        if prompt_states.requires_grad:
            raise ValueError("prompt_states must come from the frozen prompt-only pass")
        if not torch.isfinite(prompt_states).all().item():
            raise ValueError("prompt_states must be finite")
        if not torch.isfinite(noise).all().item():
            raise ValueError("noise must be finite")
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be a non-boolean integer")
        if steps < 0 or steps > self.max_steps:
            raise ValueError("steps must be between zero and max_steps")
        if prompt_padding_mask is not None:
            if not isinstance(prompt_padding_mask, Tensor):
                raise TypeError("prompt_padding_mask must be a tensor")
            if prompt_padding_mask.dtype != torch.bool:
                raise TypeError("prompt_padding_mask must be boolean")
            if prompt_padding_mask.shape != prompt_states.shape[:2]:
                raise ValueError("prompt_padding_mask must have shape (batch, tokens)")
            if prompt_padding_mask.device != prompt_states.device:
                raise ValueError("prompt_padding_mask must be on the input device")
            if prompt_padding_mask.all(dim=1).any().item():
                raise ValueError("every batch row must contain at least one valid prompt token")

    def forward(
        self,
        prompt_states: Tensor,
        *,
        noise: Tensor,
        prompt_padding_mask: Tensor | None = None,
        steps: int = 4,
        return_states: bool = False,
    ) -> Tensor | tuple[Tensor, States]:
        """Apply ``Z <- Z + F(LN(Z), X)`` exactly ``steps`` times.

        Mask entries set to True are ignored padding. When requested, the
        returned tuple contains the live initial state and every recurrent
        state, without detaching their computation graphs.
        """
        if not isinstance(return_states, bool):
            raise TypeError("return_states must be boolean")
        self._validate_inputs(prompt_states, noise, prompt_padding_mask, steps)

        state = noise
        if self.identity_mode == "indexed":
            state = state + self.slot_identity.unsqueeze(0)
        if not torch.isfinite(state).all().item():
            raise ValueError("initial workspace state is nonfinite")
        states: list[Tensor] = [state]
        for _ in range(steps):
            delta = self.delta(
                self.step_norm(state), prompt_states, prompt_padding_mask
            )
            state = state + delta
            if not torch.isfinite(state).all().item():
                raise ValueError("workspace state became nonfinite")
            states.append(state)
        if return_states:
            return state, tuple(states)
        return state
