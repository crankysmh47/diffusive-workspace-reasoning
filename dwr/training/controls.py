"""Checked source-control policy and native optimizer updates.

This module is intentionally imported explicitly: task and data utilities do
not need the optional torch/transformers dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn

from dwr.models.qwen2 import Qwen2WorkspaceLM
from .batches import TokenBatch
from .checkpoints import _architecture as _checkpoint_architecture
from .checkpoints import _tensor_signature as _checkpoint_tensor_signature
from .loss import answer_only_loss


_UNSET = object()


@dataclass(frozen=True)
class SourceControl:
    """Resolved trainable-parameter policy; it does not identify model weights."""

    method: str
    steps: int
    active_parameter_names: tuple[str, ...]
    total_parameter_count: int
    original_parameter_count: int
    new_parameter_count: int
    active_parameter_count: int
    excluded_workspace_parameter_count: int
    architecture_signature: tuple[tuple[str, object], ...]
    tensor_shapes: tuple[tuple[str, str, tuple[int, ...], str], ...]


def _architecture(host: object, *, steps: object = _UNSET) -> Qwen2WorkspaceLM:
    if type(host) is not Qwen2WorkspaceLM:
        raise TypeError("host must be an actual Qwen2WorkspaceLM")
    if steps is not _UNSET:
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be a non-boolean integer")
        if steps not in (0, 1, 4):
            raise ValueError("source-control steps must be one of 0, 1, or 4")
    host._validate_parameter_contract()
    if host.workspace.max_steps != 4:
        raise ValueError("source controls require workspace max_steps=4")
    if host.workspace.identity_mode != "exchangeable":
        raise ValueError("source controls require an exchangeable workspace")
    for reader_name, reader in host.readers.items():
        actual_heads = getattr(reader.cross_attention, "num_heads", None)
        if actual_heads != reader.num_heads:
            raise ValueError(
                f"reader {reader_name} declared head count does not match actual cross-attention heads"
            )
    _checkpoint_architecture(host)
    return host


def _description(host: Qwen2WorkspaceLM, steps: int) -> SourceControl:
    active_modules = {"readers"}
    if steps:
        active_modules.add("workspace")
    active = tuple(name for name, parameter in host.named_parameters()
                   if name.split(".", 1)[0] in active_modules)
    original = sum(parameter.numel() for parameter in host.backbone.parameters())
    workspace = sum(parameter.numel() for parameter in host.workspace.parameters())
    readers = sum(parameter.numel() for parameter in host.readers.parameters())
    total = sum(parameter.numel() for parameter in host.parameters())
    active_count = sum(parameter.numel() for name, parameter in host.named_parameters()
                       if name in set(active))
    architecture_signature = tuple(sorted(_checkpoint_architecture(host).items())) + (
        ("actual_reader_cross_attention_heads",
         tuple((name, reader.cross_attention.num_heads)
               for name, reader in host.readers.items())),
    )
    tensor_shapes = tuple((str(row[0]), str(row[1]), tuple(row[2]), str(row[3]))
                          for row in _checkpoint_tensor_signature(host))
    return SourceControl(
        method="K0_NOISE_MEMORY" if steps == 0 else f"EX_K{steps}",
        steps=steps,
        active_parameter_names=active,
        total_parameter_count=total,
        original_parameter_count=original,
        new_parameter_count=workspace + readers,
        active_parameter_count=active_count,
        excluded_workspace_parameter_count=workspace if steps == 0 else 0,
        architecture_signature=architecture_signature,
        tensor_shapes=tensor_shapes,
    )


def configure_source_control(host: Qwen2WorkspaceLM, *, steps: int) -> SourceControl:
    """Apply the explicit K0/K1/K4 active-parameter policy."""
    host = _architecture(host, steps=steps)
    if any(parameter.requires_grad for parameter in host.backbone.parameters()):
        raise ValueError("original backbone parameters must already be frozen")
    if host.backbone.training:
        raise ValueError("original backbone must already be in eval mode")
    if any(parameter.grad is not None for parameter in host.parameters()):
        raise ValueError("cannot configure a host with pre-existing gradients")
    expected = _description(host, steps)
    if not expected.active_parameter_names:
        raise ValueError("source control has no active parameters")
    for parameter in host.workspace.parameters():
        parameter.requires_grad_(steps != 0)
    for parameter in host.readers.parameters():
        parameter.requires_grad_(True)
    return expected


def validate_source_control(host: Qwen2WorkspaceLM, control: SourceControl) -> None:
    """Reject architecture or active-set drift without repairing the host."""
    if not isinstance(control, SourceControl):
        raise TypeError("control must be a SourceControl")
    host = _architecture(host, steps=control.steps)
    expected = _description(host, control.steps)
    if control != expected:
        raise ValueError("source-control description does not match host architecture or policy")
    if host.backbone.training or any(p.requires_grad for p in host.backbone.parameters()):
        raise ValueError("original backbone must remain frozen and in eval mode")
    expected_active = set(control.active_parameter_names)
    for name, parameter in host.named_parameters():
        should_train = name in expected_active
        if parameter.requires_grad != should_train:
            raise ValueError(f"parameter active flag drifted: {name}")
        if not should_train and parameter.grad is not None:
            raise ValueError(f"excluded parameter has a gradient: {name}")


def _preflight_optimizer(host: Qwen2WorkspaceLM, control: SourceControl,
                         optimizer: object, scheduler: object) -> None:
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("only torch.optim.AdamW is supported")
    if type(scheduler) is not torch.optim.lr_scheduler.LambdaLR:
        raise TypeError("only torch.optim.lr_scheduler.LambdaLR is supported")
    if scheduler.optimizer is not optimizer:
        raise ValueError("LambdaLR must belong to the supplied optimizer")
    by_id = {id(parameter): name for name, parameter in host.named_parameters()}
    expected_groups = {"readers"}
    if control.steps:
        expected_groups.add("workspace")
    seen: set[int] = set()
    seen_names: set[str] = set()
    for group in optimizer.param_groups:
        name = group.get("name")
        if name not in expected_groups or name in seen_names:
            raise ValueError("AdamW groups must be exactly the active named modules")
        seen_names.add(name)
        params = group.get("params")
        if not params:
            raise ValueError("AdamW groups cannot be empty")
        for parameter in params:
            identity = id(parameter)
            if identity in seen:
                raise ValueError("AdamW parameter membership is duplicated")
            if identity not in by_id or not parameter.requires_grad:
                raise ValueError("AdamW contains a foreign, frozen or excluded parameter")
            if not by_id[identity].startswith(name + "."):
                raise ValueError("AdamW group members must belong to their named module")
            seen.add(identity)
    expected = {id(parameter) for parameter in host.parameters() if parameter.requires_grad}
    if seen != expected or seen_names != expected_groups:
        raise ValueError("AdamW groups must cover every active parameter exactly once")


def _norm(tensor: Tensor) -> float:
    value = torch.linalg.vector_norm(tensor.detach().to(dtype=torch.float64))
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("gradient or state norm is nonfinite")
    return result


def _iterated_norm(tensors) -> float:
    """Compute a joint norm without concatenating parameter gradients."""
    result = 0.0
    for tensor in tensors:
        component = _norm(tensor)
        result = math.hypot(result, component)
    if not math.isfinite(result):
        raise ValueError("gradient or state norm is nonfinite")
    return result


def _all_host_grads_none(host: Qwen2WorkspaceLM) -> bool:
    return all(parameter.grad is None for parameter in host.parameters())


def source_control_step(
    host: Qwen2WorkspaceLM,
    batch: TokenBatch,
    noise: Tensor,
    *,
    control: SourceControl,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    gradient_clip_norm: float,
    diagnose: bool = False,
) -> dict[str, Any]:
    """Run one finite, matched native update and return boundary diagnostics."""
    host = _architecture(host)
    validate_source_control(host, control)
    if not isinstance(batch, TokenBatch):
        raise TypeError("batch must be the existing collated TokenBatch")
    if not isinstance(noise, Tensor):
        raise TypeError("noise must be a tensor")
    if isinstance(gradient_clip_norm, bool) or not isinstance(gradient_clip_norm, (int, float)):
        raise TypeError("gradient_clip_norm must be a finite positive number")
    gradient_clip_norm = float(gradient_clip_norm)
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be finite and positive")
    if not isinstance(diagnose, bool):
        raise TypeError("diagnose must be boolean")
    _preflight_optimizer(host, control, optimizer, scheduler)
    if not _all_host_grads_none(host):
        raise ValueError("every host parameter gradient must be None at step entry")
    if not torch.isfinite(noise).all().item():
        raise ValueError("noise must be finite")

    # A private leaf preserves caller ownership while making K0 and recurrent
    # initial-state gradients observable when diagnostics are requested.
    step_noise = noise.detach().clone() if diagnose else noise
    if diagnose:
        step_noise.requires_grad_(True)
    output = host(
        batch.prompt_ids, batch.prompt_mask, batch.decoder_ids, batch.decoder_mask,
        noise=step_noise, steps=control.steps, return_states=True,
    )
    loss_result = answer_only_loss(output.logits, batch.labels)
    if not torch.isfinite(loss_result.loss).item():
        raise ValueError("source-control loss is nonfinite")
    states = output.states
    if states is None:
        raise RuntimeError("native host did not return requested workspace states")
    state_norms = tuple(_norm(state) for state in states)
    if diagnose:
        for state in states:
            state.retain_grad()
    learning_rates = tuple(float(group["lr"]) for group in optimizer.param_groups)
    if any(not math.isfinite(rate) or rate < 0 for rate in learning_rates):
        raise ValueError("optimizer learning rates must be finite and nonnegative")
    loss_result.loss.backward()

    active = [(name, parameter) for name, parameter in host.named_parameters()
              if name in set(control.active_parameter_names)]
    for name, parameter in active:
        if parameter.grad is None:
            raise ValueError(f"active parameter is missing a gradient: {name}")
        if not torch.isfinite(parameter.grad).all().item():
            raise ValueError(f"active parameter has a nonfinite gradient: {name}")
    for name, parameter in host.named_parameters():
        if name not in set(control.active_parameter_names) and parameter.grad is not None:
            raise ValueError(f"excluded parameter received a gradient: {name}")

    gradient_norms = {}
    for module_name, module in (("workspace", host.workspace), ("readers", host.readers)):
        module_parameters = [p for p in module.parameters() if p.requires_grad]
        if module_parameters:
            gradient_norms[module_name] = _iterated_norm(p.grad for p in module_parameters)
    total_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in active], gradient_clip_norm, error_if_nonfinite=True
    )
    if not math.isfinite(float(total_norm.item())):
        raise ValueError("active gradient norm is nonfinite")

    state_gradient_norms: tuple[float, ...] = ()
    reader_gradient_norms: dict[str, float] = {}
    initial_state_equal_noise = False
    if diagnose:
        initial_state_equal_noise = torch.equal(states[0].detach(), step_noise.detach())
        if any(state.grad is None for state in states):
            raise ValueError("a recurrent workspace state is missing its diagnostic gradient")
        state_gradient_norms = tuple(_norm(state.grad) for state in states)
        if any(value <= 0.0 for value in state_gradient_norms):
            raise ValueError("diagnostic gradient norm must be positive for every workspace state")
        for reader_name, reader in host.readers.items():
            grads = [p.grad for p in reader.parameters()]
            if any(grad is None for grad in grads):
                raise ValueError(f"reader {reader_name} is missing a diagnostic gradient")
            reader_norm = _iterated_norm(grads)
            if reader_norm <= 0.0:
                raise ValueError("diagnostic gradient norm must be positive for every reader")
            reader_gradient_norms[reader_name] = reader_norm
    optimizer.step()
    scheduler.step()
    updated_rates = tuple(float(group["lr"]) for group in optimizer.param_groups)
    if any(not math.isfinite(rate) or rate < 0 for rate in updated_rates):
        raise ValueError("scheduler produced a nonfinite or negative learning rate")
    for name, parameter in host.named_parameters():
        if name.startswith(("workspace.", "readers.")) and not torch.isfinite(parameter).all().item():
            raise ValueError(f"optimizer produced nonfinite learned parameter values: {name}")
    optimizer.zero_grad(set_to_none=True)
    if not _all_host_grads_none(host):
        raise RuntimeError("successful optimizer boundary did not clear every gradient")
    result: dict[str, Any] = {
        "loss": float(loss_result.loss.detach().item()),
        "supervised_tokens": loss_result.supervised_tokens,
        "learning_rates": learning_rates,
        "gradient_norms": gradient_norms,
        "total_gradient_norm": float(total_norm.item()),
        "state_norms": state_norms,
        "state_gradient_norms": state_gradient_norms,
        "reader_gradient_norms": reader_gradient_norms,
        "initial_state_equal_noise": initial_state_equal_noise if diagnose else None,
        "policy": {
            "method": control.method,
            "steps": control.steps,
            "total_parameter_count": control.total_parameter_count,
            "original_parameter_count": control.original_parameter_count,
            "new_parameter_count": control.new_parameter_count,
            "active_parameter_count": control.active_parameter_count,
            "excluded_workspace_parameter_count": control.excluded_workspace_parameter_count,
        },
    }
    return result
