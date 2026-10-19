"""Frozen native Qwen2 host with prompt-only workspace conditioning."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Iterator

import torch
from torch import Tensor, nn
from transformers import Qwen2ForCausalLM

from .interface import SetCrossAttentionAdapter
from .workspace import FixedWorkspace, States


@dataclass(frozen=True)
class Qwen2WorkspaceOutput:
    """Logits and the terminal workspace, with optional live recurrent states."""

    logits: Tensor
    workspace: Tensor
    states: States | None = None


def _index(name: str, value: int, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-boolean integer")
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


class Qwen2WorkspaceLM(nn.Module):
    """One frozen Qwen2 model augmented by temporary set-reader hooks.

    Readers are initialized by the caller inside an isolated, matched RNG
    stream. The prompt encoder receives no decoder/answer tensors.
    """

    def __init__(
        self,
        backbone: Qwen2ForCausalLM,
        workspace: FixedWorkspace,
        *,
        prompt_layer: int,
        adapter_layers: tuple[int, ...],
        reader_heads: int,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, Qwen2ForCausalLM):
            raise TypeError("backbone must be a native Qwen2ForCausalLM")
        if not isinstance(workspace, FixedWorkspace):
            raise TypeError("workspace must be a FixedWorkspace")
        depth = len(backbone.model.layers)
        width = backbone.config.hidden_size
        prompt_layer = _index("prompt_layer", prompt_layer, lower=1, upper=depth)
        if not isinstance(adapter_layers, tuple):
            raise TypeError("adapter_layers must be a tuple of sorted layer indices")
        if not adapter_layers:
            raise ValueError("adapter_layers must not be empty")
        for layer in adapter_layers:
            _index("adapter layer", layer, lower=0, upper=depth - 1)
        if tuple(sorted(set(adapter_layers))) != adapter_layers:
            raise ValueError("adapter_layers must be distinct and sorted")
        if workspace.context_dim != width:
            raise ValueError("workspace context_dim must match Qwen2 hidden_size")
        if workspace.max_steps < 0:
            raise ValueError("workspace max_steps is invalid")
        _index("reader_heads", reader_heads, lower=1, upper=width)
        if width % reader_heads:
            raise ValueError("reader_heads must divide Qwen2 hidden_size")

        self.backbone = backbone
        self.workspace = workspace
        self.prompt_layer = prompt_layer
        self.adapter_layers = adapter_layers
        self.readers = nn.ModuleDict(
            {
                str(layer): SetCrossAttentionAdapter(
                    width, workspace.latent_dim,
                    num_heads=reader_heads, gate_init=gate_init,
                )
                for layer in adapter_layers
            }
        ).to(
            device=next(workspace.parameters()).device,
            dtype=next(workspace.parameters()).dtype,
        )
        self._execution_lock = threading.Lock()

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self._validate_parameter_contract()

    def train(self, mode: bool = True) -> Qwen2WorkspaceLM:
        super().train(mode)
        self.backbone.eval()
        return self

    @contextmanager
    def _execution(self) -> Iterator[None]:
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("reentrant or concurrent Qwen2 host execution is forbidden")
        original_return_dict = self.backbone.config.return_dict
        self.backbone.config.return_dict = True
        try:
            self._validate_parameter_contract()
            self._reject_foreign_hooks()
            yield
        finally:
            self.backbone.config.return_dict = original_return_dict
            self._execution_lock.release()

    def _validate_parameter_contract(self) -> None:
        if self.backbone.training:
            raise ValueError("the original Qwen2 backbone must remain in eval mode")
        if getattr(self.backbone.model, "gradient_checkpointing", False):
            raise ValueError("Qwen2 gradient checkpointing is unsupported")
        original = list(self.backbone.parameters())
        added = list(self.workspace.parameters()) + [p for p in self.readers.parameters()]
        if not original or not added:
            raise ValueError("backbone and new modules must have parameters")
        if any(p.requires_grad for p in original):
            raise ValueError("all original Qwen2 parameters must remain frozen")
        original_devices = {p.device for p in original}
        original_dtypes = {p.dtype for p in original}
        added_devices = {p.device for p in added}
        added_dtypes = {p.dtype for p in added}
        if len(original_devices) != 1 or len(added_devices) != 1:
            raise ValueError("each parameter group must use one explicit device")
        if original_devices != added_devices:
            raise ValueError("original and new parameters must share a device")
        if len(original_dtypes) != 1 or len(added_dtypes) != 1:
            raise ValueError("each parameter group must use one explicit dtype")
        if not next(self.backbone.parameters()).is_floating_point():
            raise TypeError("Qwen2 parameters must use floating-point dtype")
        if not next(self.workspace.parameters()).is_floating_point():
            raise TypeError("workspace parameters must use floating-point dtype")

    def _reject_foreign_hooks(self) -> None:
        for layer in self.backbone.model.layers:
            if layer._forward_hooks or layer._forward_pre_hooks:
                raise ValueError("foreign hooks on native decoder layers are unsupported")

    def _validate_tokens(self, ids: Tensor, mask: Tensor, name: str) -> None:
        if not isinstance(ids, Tensor) or not isinstance(mask, Tensor):
            raise TypeError(f"{name} IDs and mask must be tensors")
        if ids.dtype != torch.long:
            raise TypeError(f"{name} IDs must use torch.long")
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} mask must use torch.bool")
        if ids.ndim != 2 or mask.shape != ids.shape:
            raise ValueError(f"{name} IDs and mask must have matching shape (batch, length)")
        if ids.shape[0] <= 0 or ids.shape[1] <= 0:
            raise ValueError(f"{name} batch and sequence axes must be nonempty")
        if ids.device != mask.device:
            raise ValueError(f"{name} IDs and mask must share a device")
        if ids.device != next(self.backbone.parameters()).device:
            raise ValueError(f"{name} tensors must be on the backbone device")
        if not mask[:, 0].all().item():
            raise ValueError(f"every {name} row must start with a valid token")
        if ((~mask[:, :-1]) & mask[:, 1:]).any().item():
            raise ValueError(f"{name} mask must be contiguous right padding")
        vocab = self.backbone.config.vocab_size
        if ((ids < 0) | (ids >= vocab)).any().item():
            raise ValueError(f"{name} token ID is outside the Qwen2 vocabulary")

    def _validate_prefix(
        self, prompt_ids: Tensor, prompt_mask: Tensor, decoder_ids: Tensor,
        decoder_mask: Tensor,
    ) -> None:
        self._validate_tokens(prompt_ids, prompt_mask, "prompt")
        self._validate_tokens(decoder_ids, decoder_mask, "decoder")
        if prompt_ids.shape[0] != decoder_ids.shape[0]:
            raise ValueError("prompt and decoder batch sizes must match")
        for row in range(prompt_ids.shape[0]):
            prompt_length = int(prompt_mask[row].sum().item())
            if int(decoder_mask[row].sum().item()) < prompt_length:
                raise ValueError("decoder sequence is shorter than its prompt")
            if not torch.equal(prompt_ids[row, :prompt_length], decoder_ids[row, :prompt_length]):
                raise ValueError("decoder IDs must begin with the valid prompt tokens")

    def _encode_prompt(self, prompt_ids: Tensor, prompt_mask: Tensor) -> Tensor:
        self._validate_tokens(prompt_ids, prompt_mask, "prompt")
        captured: list[Tensor] = []

        def capture(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            if not isinstance(output, Tensor):
                raise TypeError("pinned Qwen2 decoder layer output must be a Tensor")
            captured.append(output)

        handle = self.backbone.model.layers[self.prompt_layer - 1].register_forward_hook(capture)
        try:
            with torch.no_grad():
                self.backbone.model(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("native prompt capture did not observe exactly one selected layer output")
        x = captured[0]
        if x.shape != (*prompt_ids.shape, self.backbone.config.hidden_size):
            raise ValueError("captured prompt state has an incompatible shape")
        if not torch.isfinite(x).all().item():
            raise ValueError("captured prompt state contains nonfinite values")
        return x

    def encode_prompt(self, prompt_ids: Tensor, prompt_mask: Tensor) -> Tensor:
        """Return detached raw output after the configured original layer."""
        with self._execution():
            return self._encode_prompt(prompt_ids, prompt_mask)

    def _decode(self, decoder_ids: Tensor, decoder_mask: Tensor, workspace: Tensor) -> Tensor:
        self._validate_tokens(decoder_ids, decoder_mask, "decoder")
        if not isinstance(workspace, Tensor) or workspace.ndim != 3:
            raise TypeError("workspace must be a rank-three tensor")
        if workspace.shape[0] != decoder_ids.shape[0] or workspace.shape[1:] != (
            self.workspace.num_slots, self.workspace.latent_dim
        ):
            raise ValueError("workspace shape must match batch size and configured slots/dimension")
        if workspace.device != next(self.workspace.parameters()).device:
            raise ValueError("workspace must be on the new-module device")
        if workspace.dtype != next(self.workspace.parameters()).dtype:
            raise ValueError("workspace dtype must match the new-module dtype")
        if not torch.isfinite(workspace).all().item():
            raise ValueError("workspace must be finite")

        handles = []
        try:
            for layer_index in self.adapter_layers:
                reader = self.readers[str(layer_index)]

                def read(_module: nn.Module, _inputs: tuple[object, ...], output: object,
                         *, _reader: SetCrossAttentionAdapter = reader) -> Tensor:
                    if not isinstance(output, Tensor):
                        raise TypeError("pinned Qwen2 decoder layer output must be a Tensor")
                    original_dtype = output.dtype
                    added_dtype = next(_reader.parameters()).dtype
                    with torch.autocast(device_type=output.device.type, enabled=False):
                        updated = _reader(output.to(added_dtype), workspace.to(added_dtype))
                    return updated.to(original_dtype)

                handles.append(
                    self.backbone.model.layers[layer_index].register_forward_hook(read)
                )
            output = self.backbone(
                input_ids=decoder_ids,
                attention_mask=decoder_mask,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                logits_to_keep=0,
                return_dict=True,
            )
        finally:
            for handle in reversed(handles):
                handle.remove()
        logits = output.logits
        if not torch.isfinite(logits).all().item():
            raise ValueError("native decoder produced nonfinite logits")
        return logits

    def decode_with_workspace(
        self, decoder_ids: Tensor, decoder_mask: Tensor, workspace: Tensor
    ) -> Tensor:
        """Decode through the native LM head with temporary workspace readers."""
        with self._execution():
            return self._decode(decoder_ids, decoder_mask, workspace)

    def forward(
        self,
        prompt_ids: Tensor,
        prompt_mask: Tensor,
        decoder_ids: Tensor,
        decoder_mask: Tensor,
        *,
        noise: Tensor,
        steps: int = 4,
        return_states: bool = False,
    ) -> Qwen2WorkspaceOutput:
        """Encode prompt only, update the workspace, then decode autoregressively."""
        with self._execution():
            self._validate_prefix(prompt_ids, prompt_mask, decoder_ids, decoder_mask)
            x_original = self._encode_prompt(prompt_ids, prompt_mask)
            new_parameter = next(self.workspace.parameters())
            x = x_original.to(device=new_parameter.device, dtype=new_parameter.dtype)
            if x.requires_grad:
                raise ValueError("frozen prompt representation must not require gradients")
            if not isinstance(noise, Tensor):
                raise TypeError("noise must be a tensor")
            if noise.device != new_parameter.device or noise.dtype != new_parameter.dtype:
                raise ValueError("noise device and dtype must match new module parameters")
            with torch.autocast(device_type=x.device.type, enabled=False):
                result = self.workspace(
                    x, noise=noise, prompt_padding_mask=~prompt_mask,
                    steps=steps, return_states=return_states,
                )
            if return_states:
                final_workspace, states = result
            else:
                final_workspace, states = result, None
            logits = self._decode(decoder_ids, decoder_mask, final_workspace)
            return Qwen2WorkspaceOutput(logits, final_workspace, states)
