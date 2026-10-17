"""Strict, single-example greedy evaluation for the native Qwen2 integration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import time
from typing import Callable

from dwr.tasks.schema import ConstraintTask
from dwr.tasks.verifier import verify_answer


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Immutable output and cost counts for one greedy decoding attempt."""

    token_ids: tuple[int, ...]
    eos_token_id: int
    ended_with_eos: bool
    truncated: bool
    prompt_tokens: int
    decoder_calls: int
    prompt_encoder_calls: int
    workspace_updates: int
    elapsed_wall_seconds: float

    def __post_init__(self) -> None:
        if type(self.token_ids) is not tuple or not self.token_ids:
            raise TypeError("token_ids must be a nonempty tuple")
        if any(type(token) is not int or token < 0 for token in self.token_ids):
            raise TypeError("token_ids must contain nonnegative non-boolean integers")
        _nonnegative_int("eos_token_id", self.eos_token_id)
        if type(self.ended_with_eos) is not bool or type(self.truncated) is not bool:
            raise TypeError("termination flags must be booleans")
        if self.truncated is self.ended_with_eos:
            raise ValueError("truncated must be the logical inverse of ended_with_eos")
        eos_positions = [index for index, token in enumerate(self.token_ids)
                         if token == self.eos_token_id]
        if self.ended_with_eos:
            if eos_positions != [len(self.token_ids) - 1]:
                raise ValueError("EOS must appear exactly once, at the end, when generation ended")
        elif eos_positions:
            raise ValueError("EOS may appear only as the final generated token")
        for name in ("prompt_tokens", "decoder_calls", "prompt_encoder_calls", "workspace_updates"):
            _nonnegative_int(name, getattr(self, name))
        if self.decoder_calls != len(self.token_ids):
            raise ValueError("decoder_calls must equal the number of generated tokens")
        if isinstance(self.elapsed_wall_seconds, bool) or not isinstance(
            self.elapsed_wall_seconds, (int, float)
        ):
            raise TypeError("elapsed_wall_seconds must be a finite nonnegative number")
        if not math.isfinite(self.elapsed_wall_seconds) or self.elapsed_wall_seconds < 0:
            raise ValueError("elapsed_wall_seconds must be a finite nonnegative number")


def _nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a non-boolean integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_int(name: str, value: object) -> int:
    value = _nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def evaluation_noise_seed(training_seed: int, example_id: str, *, draw_id: int = 0) -> int:
    """Derive a stable 63-bit seed from the versioned DWR evaluation-noise identity."""
    training_seed = _nonnegative_int("training_seed", training_seed)
    draw_id = _nonnegative_int("draw_id", draw_id)
    if not isinstance(example_id, str) or _SHA256_RE.fullmatch(example_id) is None:
        raise ValueError("example_id must be a lowercase 64-character SHA-256 hex digest")
    identity = {
        "domain": "dwr-evaluation-noise-v1",
        "training_seed": training_seed,
        "example_id": example_id,
        "draw_id": draw_id,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        if error.name == "torch":
            raise RuntimeError("native generation requires the optional torch dependency") from error
        raise
    return torch


def _validate_model_device(model: object, *, label: str):
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    if not parameters:
        raise ValueError(f"{label} must have parameters")
    devices = [tensor.device for tensor in (*parameters, *buffers)]
    if any(device.type not in ("cpu", "cuda") for device in devices):
        raise ValueError(f"{label} tensors must be on one supported CPU or CUDA device")
    if any(device != devices[0] for device in devices[1:]):
        raise ValueError(f"{label} tensors must share one device")
    return devices[0]


def _validate_supported_device(device, *, label: str):
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"{label} must be on a supported CPU or CUDA device")
    return device


def _validate_check(check: object) -> None:
    if check is not None and not callable(check):
        raise TypeError("check must be callable or None")


def _check_budget(check: Callable | None) -> None:
    if check is not None:
        check()


def _synchronize_device(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_off(torch, device):
    return torch.autocast(device_type=device.type, enabled=False)


def _validate_prompt_devices(prompt_ids: object, prompt_mask: object, *, device) -> None:
    torch = _torch()
    if not isinstance(prompt_ids, torch.Tensor) or not isinstance(prompt_mask, torch.Tensor):
        raise TypeError("prompt_ids and prompt_mask must be tensors")
    _validate_supported_device(prompt_ids.device, label="prompt_ids")
    _validate_supported_device(prompt_mask.device, label="prompt_mask")
    if prompt_ids.device != device or prompt_mask.device != device:
        raise ValueError("model, prompt_ids and prompt_mask must share one device")


def _validate_prompt(prompt_ids: object, prompt_mask: object, *, vocab_size: int,
                     eos_token_id: int, pad_token_id: int | None,
                     max_new_tokens: int, max_total_tokens: int, device):
    torch = _torch()
    _validate_prompt_devices(prompt_ids, prompt_mask, device=device)
    if prompt_ids.dtype != torch.long:
        raise TypeError("prompt_ids must use torch.long")
    if prompt_mask.dtype != torch.bool:
        raise TypeError("prompt_mask must use torch.bool")
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] <= 0:
        raise ValueError("prompt_ids must have shape (1, nonempty length)")
    if prompt_mask.shape != prompt_ids.shape:
        raise ValueError("prompt_mask must exactly match prompt_ids shape")
    if not prompt_mask.all().item():
        raise ValueError("single-example prompts must be unpadded with an all-true mask")
    if ((prompt_ids < 0) | (prompt_ids >= vocab_size)).any().item():
        raise ValueError("prompt token ID is outside the model vocabulary")
    if (prompt_ids == eos_token_id).any().item():
        raise ValueError("prompt must not contain EOS")
    if pad_token_id is not None and pad_token_id != eos_token_id:
        if (prompt_ids == pad_token_id).any().item():
            raise ValueError("prompt must not contain padding tokens")
    max_new_tokens = _positive_int("max_new_tokens", max_new_tokens)
    max_total_tokens = _positive_int("max_total_tokens", max_total_tokens)
    if prompt_ids.shape[1] + max_new_tokens > max_total_tokens:
        raise ValueError("prompt length plus max_new_tokens exceeds max_total_tokens")
    return prompt_ids, prompt_mask


def _model_vocab(model: object) -> int:
    embedding = model.get_input_embeddings()
    if embedding is None or not hasattr(embedding, "weight") or embedding.weight.ndim != 2:
        raise ValueError("model must expose a rank-two input embedding")
    width = int(embedding.weight.shape[0])
    configured = getattr(model.config, "vocab_size", None)
    if isinstance(configured, bool) or not isinstance(configured, int) or configured != width:
        raise ValueError("model vocabulary config must match the input embedding")
    return width


def _validate_generation_args(eos_token_id: int, max_new_tokens: int,
                              max_total_tokens: int, vocab_size: int) -> tuple[int, int, int]:
    eos_token_id = _nonnegative_int("eos_token_id", eos_token_id)
    if eos_token_id >= vocab_size:
        raise ValueError("eos_token_id is outside the model vocabulary")
    max_new_tokens = _positive_int("max_new_tokens", max_new_tokens)
    max_total_tokens = _positive_int("max_total_tokens", max_total_tokens)
    return eos_token_id, max_new_tokens, max_total_tokens


def _greedy_loop(decode: Callable, prompt_ids, prompt_mask, *, vocab_size: int,
                 eos_token_id: int, max_new_tokens: int,
                 check: Callable | None = None) -> tuple[int, ...]:
    """Run the shared full-prefix loop; scripted callbacks are algorithm tests only."""
    torch = _torch()
    if not callable(decode):
        raise TypeError("decode must be callable")
    _validate_check(check)
    vocab_size = _positive_int("vocab_size", vocab_size)
    eos_token_id = _nonnegative_int("eos_token_id", eos_token_id)
    if eos_token_id >= vocab_size:
        raise ValueError("eos_token_id is outside vocab_size")
    max_new_tokens = _positive_int("max_new_tokens", max_new_tokens)
    if not isinstance(prompt_ids, torch.Tensor) or not isinstance(prompt_mask, torch.Tensor):
        raise TypeError("prompt_ids and prompt_mask must be tensors")
    device = _validate_supported_device(prompt_ids.device, label="greedy loop inputs")
    if prompt_mask.device != device:
        raise ValueError("greedy loop inputs must share one device")
    if (prompt_ids.dtype != torch.long or prompt_mask.dtype != torch.bool or
            prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] <= 0 or
            prompt_mask.shape != prompt_ids.shape or not prompt_mask.all().item()):
        raise ValueError("greedy loop requires one nonempty unpadded prompt")
    prefix = prompt_ids
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            _check_budget(check)
            mask = torch.ones_like(prefix, dtype=torch.bool)
            logits = decode(prefix, mask)
            if not isinstance(logits, torch.Tensor):
                raise TypeError("decoder logits must be a tensor")
            if logits.device != device:
                raise ValueError("decoder logits must share the input device")
            if logits.ndim != 3 or logits.shape != (1, prefix.shape[1], vocab_size):
                raise ValueError("decoder logits must have shape (1, current_length, vocab_size)")
            if not logits.is_floating_point():
                raise TypeError("decoder logits must use a floating dtype")
            if not torch.isfinite(logits).all().item():
                raise ValueError("decoder logits must be finite")
            token = int(torch.argmax(logits[0, -1], dim=-1).item())
            generated.append(token)
            if token == eos_token_id:
                break
            prefix = torch.cat((prefix, torch.tensor(
                [[token]], dtype=torch.long, device=prefix.device
            )), dim=1)
    return tuple(generated)


def _generation_result(token_ids: tuple[int, ...], *, eos_token_id: int,
                       prompt_tokens: int, prompt_encoder_calls: int,
                       workspace_updates: int, started: float) -> GenerationResult:
    elapsed = time.perf_counter() - started
    ended = token_ids[-1] == eos_token_id
    return GenerationResult(
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        ended_with_eos=ended,
        truncated=not ended,
        prompt_tokens=prompt_tokens,
        decoder_calls=len(token_ids),
        prompt_encoder_calls=prompt_encoder_calls,
        workspace_updates=workspace_updates,
        elapsed_wall_seconds=elapsed,
    )


def greedy_workspace(host, prompt_ids, prompt_mask, *, noise, steps: int,
                     eos_token_id: int, max_new_tokens: int,
                     max_total_tokens: int,
                     check: Callable | None = None) -> GenerationResult:
    """Greedily decode through one actual Qwen2 workspace host on CPU or CUDA."""
    torch = _torch()
    from dwr.models.qwen2 import Qwen2WorkspaceLM

    _validate_check(check)
    _check_budget(check)
    if not isinstance(host, Qwen2WorkspaceLM):
        raise TypeError("host must be an actual Qwen2WorkspaceLM")
    if host.training or any(module.training for module in host.modules()):
        raise ValueError("workspace host and all its modules must already be in eval mode")
    device = _validate_model_device(host, label="workspace host")
    if not isinstance(noise, torch.Tensor):
        raise TypeError("noise must be a tensor")
    _validate_prompt_devices(prompt_ids, prompt_mask, device=device)
    _validate_supported_device(noise.device, label="workspace noise")
    if noise.device != device:
        raise ValueError("workspace noise must share the host device")
    if noise.ndim != 3 or noise.shape != (
        1, host.workspace.num_slots, host.workspace.latent_dim
    ):
        raise ValueError("noise must match the single-example workspace shape")
    parameter = next(host.workspace.parameters())
    if noise.dtype != parameter.dtype:
        raise ValueError("noise dtype must match workspace parameters")
    if not noise.is_floating_point() or not torch.isfinite(noise).all().item():
        raise ValueError("noise must be finite floating-point data")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be a non-boolean integer")
    if steps < 0 or steps > host.workspace.max_steps:
        raise ValueError("steps must be between zero and workspace.max_steps")
    vocab_size = _model_vocab(host.backbone)
    eos_token_id, max_new_tokens, max_total_tokens = _validate_generation_args(
        eos_token_id, max_new_tokens, max_total_tokens, vocab_size
    )
    _validate_prompt(
        prompt_ids, prompt_mask, vocab_size=vocab_size,
        eos_token_id=eos_token_id, pad_token_id=host.backbone.config.pad_token_id,
        max_new_tokens=max_new_tokens, max_total_tokens=max_total_tokens,
        device=device,
    )
    _synchronize_device(torch, device)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            x_original = host.encode_prompt(prompt_ids, prompt_mask)
            if x_original.device != device:
                raise ValueError("prompt representation must remain on the host device")
            if x_original.requires_grad:
                raise ValueError("prompt representation must be detached")
            x = x_original.to(device=parameter.device, dtype=parameter.dtype)
            with _autocast_off(torch, device):
                workspace = host.workspace(
                    x, noise=noise, prompt_padding_mask=~prompt_mask, steps=steps
                )

            def decode(prefix, mask):
                return host.decode_with_workspace(prefix, mask, workspace)

            token_ids = _greedy_loop(
                decode, prompt_ids, prompt_mask, vocab_size=vocab_size,
                eos_token_id=eos_token_id, max_new_tokens=max_new_tokens, check=check,
            )
    finally:
        _synchronize_device(torch, device)
    return _generation_result(
        token_ids, eos_token_id=eos_token_id, prompt_tokens=prompt_ids.shape[1],
        prompt_encoder_calls=1, workspace_updates=steps, started=started,
    )


def greedy_frozen(backbone, prompt_ids, prompt_mask, *, eos_token_id: int,
                  max_new_tokens: int, max_total_tokens: int,
                  check: Callable | None = None) -> GenerationResult:
    """Greedily decode with the frozen native Qwen2 model on CPU or CUDA."""
    torch = _torch()
    try:
        from transformers import Qwen2ForCausalLM
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        if error.name == "transformers":
            raise RuntimeError("native generation requires the optional transformers dependency") from error
        raise
    _validate_check(check)
    _check_budget(check)
    if not isinstance(backbone, Qwen2ForCausalLM):
        raise TypeError("backbone must be an actual Qwen2ForCausalLM")
    if any(module.training for module in backbone.modules()):
        raise ValueError("backbone and all its modules must already be in eval mode")
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("all frozen-backbone parameters must have requires_grad=False")
    if backbone.config.return_dict is not True:
        raise ValueError("backbone config.return_dict must already be True")
    if getattr(backbone.model, "gradient_checkpointing", False):
        raise ValueError("Qwen2 gradient checkpointing is unsupported")
    for module in backbone.modules():
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("foreign hooks on frozen Qwen2 modules are unsupported")
    device = _validate_model_device(backbone, label="frozen Qwen2 backbone")
    vocab_size = _model_vocab(backbone)
    eos_token_id, max_new_tokens, max_total_tokens = _validate_generation_args(
        eos_token_id, max_new_tokens, max_total_tokens, vocab_size
    )
    _validate_prompt(
        prompt_ids, prompt_mask, vocab_size=vocab_size,
        eos_token_id=eos_token_id, pad_token_id=backbone.config.pad_token_id,
        max_new_tokens=max_new_tokens, max_total_tokens=max_total_tokens,
        device=device,
    )
    _synchronize_device(torch, device)
    started = time.perf_counter()

    def decode(prefix, mask):
        output = backbone(
            input_ids=prefix,
            attention_mask=mask,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            logits_to_keep=0,
            return_dict=True,
        )
        return output.logits

    try:
        with torch.no_grad():
            token_ids = _greedy_loop(
                decode, prompt_ids, prompt_mask, vocab_size=vocab_size,
                eos_token_id=eos_token_id, max_new_tokens=max_new_tokens, check=check,
            )
    finally:
        _synchronize_device(torch, device)
    return _generation_result(
        token_ids, eos_token_id=eos_token_id, prompt_tokens=prompt_ids.shape[1],
        prompt_encoder_calls=0, workspace_updates=0, started=started,
    )


def score_generation(task: ConstraintTask, result: GenerationResult, tokenizer) -> dict[str, object]:
    """Decode and strictly verify one result, preserving truncation as failure."""
    if type(task) is not ConstraintTask:
        raise TypeError("task must be an actual ConstraintTask")
    if type(result) is not GenerationResult:
        raise TypeError("result must be a GenerationResult")
    result.__post_init__()
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer.decode must be callable")
    token_ids = result.token_ids[:-1] if result.ended_with_eos else result.token_ids
    answer_text = decode(
        list(token_ids), skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(answer_text, str):
        raise TypeError("tokenizer.decode must return str")
    verification = verify_answer(task, answer_text)
    return {
        "answer_text": answer_text,
        "valid": verification.valid and not result.truncated,
        "verifier_reason": verification.reason.value,
        "termination": "eos" if result.ended_with_eos else "token_cap",
        "generated_tokens": len(result.token_ids),
    }
