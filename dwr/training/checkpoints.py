"""Strict boundary-only checkpoints for the native Qwen2 workspace host."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import random
import re
import stat
import uuid

import numpy as np
import torch

from dwr.models.qwen2 import Qwen2WorkspaceLM
from .sampler import SourceSampler


_VERSION = 1
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_KEYS = {"source", "config", "data", "model", "tokenizer", "schedule"}
_PROGRESS_KEYS = {"optimizer_step", "examples_seen", "supervised_tokens_seen"}
_ROOT_KEYS = {"version", "architecture", "tensor_signature", "identities", "progress",
              "workspace", "readers", "optimizer", "optimizer_groups", "scheduler",
              "scaler", "sampler", "noise_generator", "rng"}
_RNG_KEYS = {"python", "numpy", "torch_cpu", "cuda_initialized", "cuda_devices", "cuda_states"}


def _int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-boolean integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _sha(name: str, value: object) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256 text")
    return value


def _checked_identities(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _IDENTITY_KEYS:
        raise ValueError("identities must contain exactly the six bound hashes")
    return {key: _sha(f"identity {key}", value[key]) for key in sorted(_IDENTITY_KEYS)}


def _checked_progress(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != _PROGRESS_KEYS:
        raise ValueError("progress must contain exactly the three counters")
    return {key: _int(key, value[key]) for key in sorted(_PROGRESS_KEYS)}


def _finite_tree(value: object, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} contains nonfinite values")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{name}[{index}]")
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{name} contains a nonfinite value")


def _path(path: os.PathLike[str] | str) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be filesystem text")
    result = Path(path)
    if not result.name or not result.parent.exists() or not result.parent.is_dir():
        raise ValueError("checkpoint parent directory must already exist")
    current = Path(result.anchor) if result.is_absolute() else Path.cwd()
    parts = result.parts[1:] if result.is_absolute() else result.parts
    for part in parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        attrs = getattr(info, "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(info.st_mode) or attrs & reparse:
            raise ValueError("checkpoint paths cannot traverse symlinks or reparse points")
    return result


def _assert_host(host: object) -> Qwen2WorkspaceLM:
    if not isinstance(host, Qwen2WorkspaceLM):
        raise TypeError("host must be Qwen2WorkspaceLM")
    host._validate_parameter_contract()
    if any(parameter.requires_grad for parameter in host.backbone.parameters()):
        raise ValueError("all original Qwen2 parameters must remain frozen")
    if not host._execution_lock.locked():
        raise RuntimeError("internal checkpoint boundary lock is not held")
    host._reject_foreign_hooks()
    for name, parameter in host.named_parameters():
        if parameter.grad is not None:
            raise ValueError(f"parameter {name} has a gradient; checkpoint only at zero_grad(None) boundary")
    return host


def _architecture(host: Qwen2WorkspaceLM) -> dict[str, object]:
    config = host.backbone.config
    workspace = host.workspace
    self_heads = workspace.delta.self_attention.num_heads
    context_heads = workspace.delta.context_attention.num_heads
    if workspace.num_heads != self_heads or workspace.num_heads != context_heads:
        raise ValueError("workspace attention head metadata is inconsistent")
    return {
        "host_type": f"{type(host).__module__}.{type(host).__qualname__}",
        "model_type": config.model_type,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "workspace_context_dim": workspace.context_dim,
        "workspace_num_slots": workspace.num_slots,
        "workspace_latent_dim": workspace.latent_dim,
        "workspace_num_heads": workspace.num_heads,
        "workspace_max_steps": workspace.max_steps,
        "slot_identity": workspace.identity_mode,
        "prompt_layer": host.prompt_layer,
        "adapter_layers": tuple(host.adapter_layers),
        "reader_indices": tuple(int(key) for key in host.readers.keys()),
        "reader_heads": tuple(host.readers[key].num_heads for key in host.readers),
    }


def _tensor_signature(host: Qwen2WorkspaceLM) -> tuple[tuple[object, ...], ...]:
    rows = []
    for name, parameter in host.named_parameters():
        rows.append(("parameter", name, tuple(parameter.shape), str(parameter.dtype), parameter.requires_grad))
    for name, buffer in host.named_buffers():
        rows.append(("buffer", name, tuple(buffer.shape), str(buffer.dtype), False))
    return tuple(rows)


def _learned_states(host: Qwen2WorkspaceLM) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    workspace = {name: value.detach().clone() for name, value in host.workspace.state_dict().items()}
    readers = {name: value.detach().clone() for name, value in host.readers.state_dict().items()}
    _finite_tree(workspace, "workspace")
    _finite_tree(readers, "readers")
    return workspace, readers


def _optimizer_groups(host: Qwen2WorkspaceLM, optimizer: torch.optim.Optimizer) -> tuple[tuple[str, ...], ...]:
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("only torch.optim.AdamW is supported")
    names_by_parameter = {id(parameter): name for name, parameter in host.named_parameters()}
    expected = {id(parameter) for parameter in host.parameters() if parameter.requires_grad}
    expected_group_names = {
        module_name for module_name, module in (("workspace", host.workspace), ("readers", host.readers))
        if any(parameter.requires_grad for parameter in module.parameters())
    }
    seen: set[int] = set()
    groups: list[tuple[str, ...]] = []
    group_names: set[str] = set()
    for group in optimizer.param_groups:
        name = group.get("name")
        if name not in expected_group_names or name in group_names:
            raise ValueError("AdamW groups must match the nonempty trainable workspace/readers modules")
        group_names.add(name)
        current = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity in seen or identity not in expected:
                raise ValueError("AdamW parameter membership is duplicated, frozen or foreign")
            if not names_by_parameter[identity].startswith(f"{name}."):
                raise ValueError("AdamW group members must belong to their named module")
            seen.add(identity)
            current.append(names_by_parameter[identity])
        if not current:
            raise ValueError("AdamW parameter groups cannot be empty")
        groups.append(tuple(current))
    if seen != expected or group_names != expected_group_names:
        raise ValueError("AdamW groups must contain the active new parameters exactly once")
    return tuple(groups)


def _optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, object]:
    state = optimizer.state_dict()
    _finite_tree(state, "optimizer")
    return state


def _scaler_state(scaler: object) -> dict[str, object]:
    if scaler is None:
        return {"kind": "none", "enabled": False, "state": {}}
    if isinstance(scaler, torch.amp.GradScaler):
        kind = f"{type(scaler).__module__}.{type(scaler).__qualname__}"
        enabled = scaler.is_enabled()
        state = scaler.state_dict()
        _finite_tree(state, "scaler")
        return {"kind": kind, "enabled": enabled, "state": state}
    raise TypeError("scaler must be None or torch.amp.GradScaler")


def _scaler_matches(saved: object, scaler: object) -> None:
    if not isinstance(saved, dict) or set(saved) != {"kind", "enabled", "state"}:
        raise ValueError("scaler checkpoint schema is malformed")
    expected = _scaler_state(scaler)
    if saved["kind"] != expected["kind"] or saved["enabled"] != expected["enabled"]:
        raise ValueError("scaler type or enabled policy does not match")
    if scaler is None and saved["state"] != {}:
        raise ValueError("disabled scaler checkpoint must have empty state")
    if scaler is not None:
        state = saved["state"]
        if not isinstance(state, dict):
            raise ValueError("scaler state must be a mapping")
        if not scaler.is_enabled():
            if state:
                raise ValueError("disabled scaler state must be empty")
            return
        expected_keys = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
        if set(state) != expected_keys:
            raise ValueError("enabled scaler state keys do not match GradScaler")
        for key in ("scale", "growth_factor", "backoff_factor"):
            if isinstance(state[key], bool) or not isinstance(state[key], (int, float)):
                raise ValueError(f"scaler {key} must be numeric")
        if (not np.isfinite(state["scale"]) or state["scale"] <= 0
                or state["growth_factor"] != scaler._growth_factor
                or state["backoff_factor"] != scaler._backoff_factor
                or type(state["growth_interval"]) is not int
                or state["growth_interval"] != scaler._growth_interval
                or type(state["_growth_tracker"]) is not int or state["_growth_tracker"] < 0):
            raise ValueError("GradScaler values or growth policy do not match")
        _finite_tree(state, "scaler")


def _numpy_state() -> dict[str, object]:
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return {"algorithm": algorithm, "keys": torch.from_numpy(keys.copy()),
            "position": int(position), "has_gauss": int(has_gauss), "cached": float(cached)}


def _rng_state(noise_generator: torch.Generator) -> dict[str, object]:
    if not isinstance(noise_generator, torch.Generator):
        raise TypeError("noise_generator must be an explicit torch.Generator")
    initialized = torch.cuda.is_initialized()
    devices = tuple(range(torch.cuda.device_count())) if initialized else ()
    cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all()) if initialized else ()
    return {
        "python": random.getstate(),
        "numpy": _numpy_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda_initialized": initialized,
        "cuda_devices": devices,
        "cuda_states": cuda_states,
        "noise_device": str(noise_generator.device),
        "noise_state": noise_generator.get_state().clone(),
    }


def _validate_rng(state: object, noise_generator: torch.Generator) -> None:
    keys = _RNG_KEYS | {"noise_device", "noise_state"}
    if not isinstance(state, dict) or set(state) != keys:
        raise ValueError("RNG checkpoint schema is malformed")
    if not isinstance(state["python"], tuple):
        raise ValueError("Python RNG state is malformed")
    try:
        probe = random.Random()
        probe.setstate(state["python"])
    except (TypeError, ValueError) as error:
        raise ValueError("Python RNG state is invalid") from error
    numpy_state = state["numpy"]
    if (not isinstance(numpy_state, dict) or set(numpy_state) !=
            {"algorithm", "keys", "position", "has_gauss", "cached"}):
        raise ValueError("NumPy RNG state is malformed")
    keys_array = numpy_state["keys"]
    if (not isinstance(numpy_state["algorithm"], str) or not isinstance(keys_array, torch.Tensor)
            or keys_array.dtype != torch.uint32 or keys_array.device.type != "cpu"
            or keys_array.shape != (624,)):
        raise ValueError("NumPy RNG state has incompatible components")
    try:
        np.random.RandomState().set_state((numpy_state["algorithm"], keys_array.numpy().copy(),
            _int("NumPy position", numpy_state["position"]), _int("NumPy has_gauss", numpy_state["has_gauss"]),
            float(numpy_state["cached"])))
    except (TypeError, ValueError) as error:
        raise ValueError("NumPy RNG state is invalid") from error
    if (not isinstance(state["torch_cpu"], torch.Tensor) or state["torch_cpu"].dtype != torch.uint8
            or state["torch_cpu"].device.type != "cpu"):
        raise ValueError("Torch CPU RNG state is malformed")
    cpu_probe = torch.Generator(device="cpu")
    try:
        cpu_probe.set_state(state["torch_cpu"].clone())
    except RuntimeError as error:
        raise ValueError("Torch CPU RNG state is invalid") from error
    if type(state["cuda_initialized"]) is not bool:
        raise ValueError("CUDA initialization flag is malformed")
    devices, cuda_states = state["cuda_devices"], state["cuda_states"]
    if not isinstance(devices, tuple) or not isinstance(cuda_states, tuple) or len(devices) != len(cuda_states):
        raise ValueError("CUDA RNG layout is malformed")
    if any(type(device) is not int or device < 0 for device in devices):
        raise ValueError("CUDA RNG device IDs are malformed")
    if state["cuda_initialized"] != torch.cuda.is_initialized():
        raise ValueError("CUDA initialization policy differs from checkpoint")
    live_devices = tuple(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else ()
    if devices != live_devices:
        raise ValueError("CUDA device layout differs from checkpoint")
    if not state["cuda_initialized"] and (devices or cuda_states):
        raise ValueError("uninitialized CUDA checkpoint must not contain CUDA states")
    for device, cuda_state in zip(devices, cuda_states):
        if not isinstance(cuda_state, torch.Tensor) or cuda_state.dtype != torch.uint8 or cuda_state.device.type != "cpu":
            raise ValueError("CUDA RNG state must be a CPU uint8 tensor")
        cuda_probe = torch.Generator(device=f"cuda:{device}")
        try:
            cuda_probe.set_state(cuda_state.clone())
        except RuntimeError as error:
            raise ValueError("CUDA RNG state is invalid") from error
    if state["noise_device"] != str(noise_generator.device):
        raise ValueError("explicit noise generator device does not match")
    if not isinstance(state["noise_state"], torch.Tensor) or state["noise_state"].dtype != torch.uint8:
        raise ValueError("explicit noise generator state is malformed")
    noise_probe = torch.Generator(device=noise_generator.device)
    try:
        noise_probe.set_state(state["noise_state"].clone())
    except RuntimeError as error:
        raise ValueError("explicit noise generator state is invalid") from error


def _validate_state_dict(saved: object, live: dict[str, torch.Tensor], name: str) -> None:
    if not isinstance(saved, dict) or set(saved) != set(live):
        raise ValueError(f"{name} state keys do not match")
    for key, expected in live.items():
        value = saved[key]
        if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"{name}.{key} shape or dtype does not match")
        _finite_tree(value, f"{name}.{key}")


def _validate_optimizer(saved: object, optimizer: torch.optim.Optimizer,
                        groups: tuple[tuple[str, ...], ...], host: Qwen2WorkspaceLM,
                        step_limit: int) -> None:
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise ValueError("AdamW state schema is malformed")
    live = optimizer.state_dict()
    saved_groups = saved["param_groups"]
    if not isinstance(saved_groups, list) or len(saved_groups) != len(live["param_groups"]):
        raise ValueError("AdamW group count does not match")
    saved_ids: set[int] = set()
    param_by_name = dict(host.named_parameters())
    for old, current, names in zip(saved_groups, live["param_groups"], groups):
        if not isinstance(old, dict) or set(old) != set(current):
            raise ValueError("AdamW parameter group keys do not match")
        if (not isinstance(old.get("params"), list)
                or any(type(parameter_id) is not int or parameter_id < 0 for parameter_id in old["params"])
                or old["params"] != current.get("params")):
            raise ValueError("AdamW parameter group order does not match")
        for key in set(current) - {"params", "lr"}:
            if old[key] != current[key]:
                raise ValueError(f"AdamW group setting {key} does not match")
        if (isinstance(old["lr"], bool) or not isinstance(old["lr"], (int, float))
                or not np.isfinite(old["lr"])):
            raise ValueError("AdamW learning rate is invalid")
        saved_ids.update(old["params"])
    state = saved["state"]
    if not isinstance(state, dict) or any(type(key) is not int or key not in saved_ids for key in state):
        raise ValueError("AdamW moment mapping is malformed")
    if step_limit == 0 and state:
        raise ValueError("zero optimizer progress requires empty AdamW state")
    if step_limit > 0 and set(state) != saved_ids:
        raise ValueError("positive optimizer progress requires moments for every active parameter")
    # Serialized IDs follow group order; map them to the live parameter names.
    for saved_group, names in zip(saved_groups, groups):
        group = next(group for group in optimizer.param_groups if group.get("name") ==
                     ("workspace" if names[0].startswith("workspace.") else "readers"))
        for parameter_id, name in zip(saved_group["params"], names):
            if parameter_id not in state:
                continue
            moment = state[parameter_id]
            parameter = param_by_name[name]
            amsgrad = bool(group.get("amsgrad"))
            expected_keys = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if amsgrad else set())
            if not isinstance(moment, dict) or set(moment) != expected_keys:
                raise ValueError("AdamW moment keys are malformed")
            tick = moment["step"]
            if isinstance(tick, torch.Tensor):
                if (tick.ndim != 0 or tick.dtype == torch.bool or tick.is_complex()
                        or not torch.isfinite(tick).all().item()):
                    raise ValueError("AdamW step is malformed")
                tick = float(tick.item())
            if (isinstance(tick, bool) or not isinstance(tick, (int, float)) or not np.isfinite(tick)
                    or int(tick) != tick or tick != step_limit):
                raise ValueError("AdamW step must be an integral count equal to optimizer progress")
            for key in expected_keys - {"step"}:
                value = moment[key]
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError("AdamW moment shape or dtype does not match its named parameter")
                _finite_tree(value, f"AdamW.{name}.{key}")


def _validate_scheduler(saved: object, scheduler: object, optimizer: torch.optim.Optimizer,
                        optimizer_step: int | None = None) -> None:
    if type(scheduler) is not torch.optim.lr_scheduler.LambdaLR:
        raise TypeError("only torch.optim.lr_scheduler.LambdaLR is supported")
    if scheduler.optimizer is not optimizer:
        raise ValueError("scheduler must own the supplied optimizer")
    if not isinstance(saved, dict) or set(saved) != set(scheduler.state_dict()):
        raise ValueError("LambdaLR state keys do not match")
    if not isinstance(saved.get("base_lrs"), list) or len(saved["base_lrs"]) != len(optimizer.param_groups):
        raise ValueError("LambdaLR base learning rates do not match groups")
    if not isinstance(saved.get("_last_lr"), list) or len(saved["_last_lr"]) != len(optimizer.param_groups):
        raise ValueError("LambdaLR current learning rates do not match groups")
    if any(isinstance(rate, bool) or not isinstance(rate, (int, float)) or not np.isfinite(rate)
           for rate in saved["base_lrs"] + saved["_last_lr"]):
        raise ValueError("LambdaLR rates must be finite numbers")
    if not isinstance(saved.get("lr_lambdas"), list) or len(saved["lr_lambdas"]) != len(optimizer.param_groups):
        raise ValueError("LambdaLR lambda count does not match groups")
    if optimizer_step is not None:
        if (type(saved.get("last_epoch")) is not int or type(saved.get("_step_count")) is not int
                or saved["last_epoch"] != optimizer_step or saved["_step_count"] != optimizer_step + 1):
            raise ValueError("LambdaLR counters do not match optimizer progress")
    _finite_tree(saved, "scheduler")


def _validate_scheduler_rates(optimizer_state: dict[str, object], scheduler_state: dict[str, object]) -> None:
    rates = [group["lr"] for group in optimizer_state["param_groups"]]
    if any(isinstance(rate, bool) or not isinstance(rate, (int, float)) for rate in rates):
        raise ValueError("AdamW learning rates must be numeric")
    if rates != scheduler_state["_last_lr"]:
        raise ValueError("AdamW and LambdaLR learning rates disagree")


def _validate_sampler(saved: object, sampler: SourceSampler) -> None:
    probe = SourceSampler(sampler.example_ids, seed=sampler.seed)
    probe.load_state_dict(saved)


def _safe_rng_restore(state: dict[str, object], noise_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], numpy_state["keys"].numpy().copy(),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached"]))
    torch.set_rng_state(state["torch_cpu"])
    if state["cuda_initialized"]:
        torch.cuda.set_rng_state_all(list(state["cuda_states"]))
    noise_generator.set_state(state["noise_state"])


def _save_training_checkpoint_locked(path, *, host, optimizer, scheduler, scaler, sampler,
                                     noise_generator, identities, progress) -> tuple[str, int]:
    """Write one new checkpoint at a zero-grad optimizer boundary."""
    host = _assert_host(host)
    if not isinstance(sampler, SourceSampler):
        raise TypeError("sampler must be SourceSampler")
    if type(scheduler) is not torch.optim.lr_scheduler.LambdaLR:
        raise TypeError("only torch.optim.lr_scheduler.LambdaLR is supported")
    if scheduler.optimizer is not optimizer:
        raise ValueError("scheduler must own the supplied optimizer")
    groups = _optimizer_groups(host, optimizer)
    identity = _checked_identities(identities)
    counters = _checked_progress(progress)
    _validate_rng(_rng_state(noise_generator), noise_generator)
    workspace, readers = _learned_states(host)
    optimizer_state = _optimizer_state(optimizer)
    _validate_optimizer(optimizer_state, optimizer, groups, host, counters["optimizer_step"])
    scheduler_state = scheduler.state_dict()
    _validate_scheduler(scheduler_state, scheduler, optimizer, counters["optimizer_step"])
    _validate_scheduler_rates(optimizer_state, scheduler_state)
    sampler_state = sampler.state_dict()
    payload = {
        "version": _VERSION,
        "architecture": _architecture(host),
        "tensor_signature": _tensor_signature(host),
        "identities": identity,
        "progress": counters,
        "workspace": workspace,
        "readers": readers,
        "optimizer": optimizer_state,
        "optimizer_groups": groups,
        "scheduler": scheduler_state,
        "scaler": _scaler_state(scaler),
        "sampler": sampler_state,
        "noise_generator": {"device": str(noise_generator.device), "state": noise_generator.get_state().clone()},
        "rng": _rng_state(noise_generator),
    }
    _finite_tree(payload, "checkpoint")
    target = _path(path)
    if target.exists():
        raise FileExistsError(target)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temp, flags, 0o600)
    with os.fdopen(descriptor, "w+b") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temp, target)
    temp.unlink()
    raw = target.read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _load_training_checkpoint_locked(path, *, expected_sha256, host, optimizer, scheduler,
                                     scaler, sampler, noise_generator, identities) -> dict[str, int]:
    """Validate a trusted local checkpoint fully before applying learned state."""
    host = _assert_host(host)
    target = _path(path)
    expected_digest = _sha("expected_sha256", expected_sha256)
    raw = target.read_bytes()
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("checkpoint SHA-256 does not match")
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("checkpoint bytes are corrupt or unsupported") from error
    if not isinstance(payload, dict) or set(payload) != _ROOT_KEYS:
        raise ValueError("checkpoint has missing or unknown top-level keys")
    _finite_tree(payload, "checkpoint")
    if type(payload["version"]) is not int or payload["version"] != _VERSION:
        raise ValueError("unsupported checkpoint version")
    identity = _checked_identities(identities)
    if _checked_identities(payload["identities"]) != identity:
        raise ValueError("checkpoint identities do not match this run")
    progress = _checked_progress(payload["progress"])
    if payload["architecture"] != _architecture(host):
        raise ValueError("checkpoint host architecture does not match")
    if payload["tensor_signature"] != _tensor_signature(host):
        raise ValueError("checkpoint named tensor signature does not match")
    live_workspace = host.workspace.state_dict()
    live_readers = host.readers.state_dict()
    _validate_state_dict(payload["workspace"], live_workspace, "workspace")
    _validate_state_dict(payload["readers"], live_readers, "readers")
    groups = _optimizer_groups(host, optimizer)
    if payload["optimizer_groups"] != groups:
        raise ValueError("ordered named AdamW parameter groups do not match")
    _validate_optimizer(payload["optimizer"], optimizer, groups, host, progress["optimizer_step"])
    _validate_scheduler(payload["scheduler"], scheduler, optimizer, progress["optimizer_step"])
    _validate_scheduler_rates(payload["optimizer"], payload["scheduler"])
    _scaler_matches(payload["scaler"], scaler)
    if not isinstance(sampler, SourceSampler):
        raise TypeError("sampler must be SourceSampler")
    _validate_sampler(payload["sampler"], sampler)
    noise = payload["noise_generator"]
    if not isinstance(noise, dict) or set(noise) != {"device", "state"}:
        raise ValueError("explicit noise generator checkpoint is malformed")
    if noise["device"] != str(noise_generator.device):
        raise ValueError("explicit noise generator device does not match")
    if (not isinstance(noise["state"], torch.Tensor) or noise["state"].dtype != torch.uint8
            or noise["state"].device.type != "cpu"):
        raise ValueError("explicit noise generator state is malformed")
    probe = torch.Generator(device=noise_generator.device)
    try:
        probe.set_state(noise["state"].clone())
    except RuntimeError as error:
        raise ValueError("explicit noise generator state is invalid") from error
    _validate_rng(payload["rng"], noise_generator)
    if not isinstance(payload["rng"]["noise_state"], torch.Tensor) or not torch.equal(
            payload["rng"]["noise_state"], noise["state"]):
        raise ValueError("explicit noise generator states disagree")

    host.workspace.load_state_dict(payload["workspace"], strict=True)
    host.readers.load_state_dict(payload["readers"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"]["state"])
    sampler.load_state_dict(payload["sampler"])
    _safe_rng_restore(payload["rng"], noise_generator)
    return dict(progress)


def _host_lock(host: object):
    if not isinstance(host, Qwen2WorkspaceLM):
        raise TypeError("host must be Qwen2WorkspaceLM")
    if not host._execution_lock.acquire(blocking=False):
        raise ValueError("checkpoint operation requires an idle host")
    return host._execution_lock


def save_training_checkpoint(path, *, host, optimizer, scheduler, scaler, sampler,
                             noise_generator, identities, progress) -> tuple[str, int]:
    """Write one new checkpoint at a zero-grad optimizer boundary."""
    lock = _host_lock(host)
    try:
        return _save_training_checkpoint_locked(path, host=host, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, sampler=sampler, noise_generator=noise_generator,
            identities=identities, progress=progress)
    finally:
        lock.release()


def load_training_checkpoint(path, *, expected_sha256, host, optimizer, scheduler,
                             scaler, sampler, noise_generator, identities) -> dict[str, int]:
    """Validate and restore a trusted local checkpoint while holding the host idle."""
    lock = _host_lock(host)
    try:
        return _load_training_checkpoint_locked(path, expected_sha256=expected_sha256,
            host=host, optimizer=optimizer, scheduler=scheduler, scaler=scaler, sampler=sampler,
            noise_generator=noise_generator, identities=identities)
    finally:
        lock.release()
