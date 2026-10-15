"""Closed, single-invocation pretrained GPU feasibility probe.

The runner has no fixture or fallback CLI. Its tensor helpers are also exercised
with a tiny real Qwen2 host on CPU; production configuration stays byte-pinned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping


CONFIG_SHA256 = "781beb8ab5c1a26c8bcaafb9804d6c2b3d20a70b620df8842bd16a617d45e743"
MODEL_MANIFEST_SHA256 = "166492a8c0ae52d5b0a766fa281528a2348eca43f2b128f86da70936a4289606"
EXPORT_MANIFEST_SHA256 = "7db0b61ba40bea60bf17bc5faf615bb2c8ae21320739c3bea589e4d62aea0069"
TOKEN_AUDIT_MANIFEST_SHA256 = "aec131295af1d015d9ef7619ad35d0ad03792a7f147cdf5164bc4c8e4e800826"
_CONFIG_PATH = "configs/models/pretrained_gpu_feasibility_v1.json"
_OUTCOME_PATH = "outcome.json"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_TOP_LEVEL = {
    "attention", "config_version", "data_preconditions", "environment", "hardware",
    "layout", "limits", "measurement", "model", "numerics", "optimizer",
    "output_dir", "precision", "primary_metric", "protected_payload_access",
    "schedule", "seeds", "source_base_commit", "source_policy", "stop_rule",
    "study_type", "synthetic_input", "task_training",
}


@dataclass(frozen=True)
class ProbeBatch:
    """Synthetic token tensors; this is not a source-task example."""

    prompt_ids: Any
    prompt_mask: Any
    decoder_ids: Any
    decoder_mask: Any
    labels: Any


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def seed_cpu_initialization(torch_module: Any, seed: int) -> None:
    """Seed only the CPU default generator; torch.manual_seed also touches CUDA."""
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("CPU initialization seed must be a nonnegative signed-64-bit integer")
    torch_module.random.default_generator.manual_seed(seed)


def _reject_links(path: Path, *, allow_missing_leaf: bool = False) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    chain = list(reversed((absolute, *absolute.parents)))
    for item in chain:
        if allow_missing_leaf and item == absolute and not item.exists():
            continue
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        reparse = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if item.is_symlink() or getattr(info, "st_file_attributes", 0) & reparse:
            raise ValueError(f"symlink or reparse path is forbidden: {item}")


def _read_small_json(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    _reject_links(path)
    if not path.is_file():
        raise ValueError(f"required metadata file is missing: {path}")
    raw = path.read_bytes()
    if expected_sha256 is not None and _sha256(raw) != expected_sha256:
        raise ValueError(f"metadata identity mismatch: {path}")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"metadata is not strict UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"metadata root must be an object: {path}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _no_nulls(value: Any, where: str = "config") -> None:
    if value is None:
        raise ValueError(f"null is forbidden in closed config: {where}")
    if isinstance(value, dict):
        for key, child in value.items():
            _no_nulls(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _no_nulls(child, f"{where}[{index}]")


def validate_config_bytes(raw: bytes) -> dict[str, Any]:
    """Validate the registered config's exact bytes and closed top-level schema."""
    if not isinstance(raw, bytes):
        raise TypeError("config bytes must be bytes")
    if _sha256(raw) != CONFIG_SHA256:
        raise ValueError("config bytes do not match the registered SHA-256")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("registered config is not strict UTF-8 JSON") from exc
    return validate_config_object(value)


def validate_config_object(value: Any) -> dict[str, Any]:
    """Check the exact closed schema independently of the byte pin."""
    if not isinstance(value, dict) or set(value) != _TOP_LEVEL:
        raise ValueError("config top-level schema has unknown or missing fields")
    _no_nulls(value)
    if value["config_version"] != "dwr-pretrained-gpu-feasibility-v1":
        raise ValueError("unsupported config version")
    if value["task_training"] is not False or value["protected_payload_access"] is not False:
        raise ValueError("config does not preserve the registered probe boundary")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _registered_config(path: str | os.PathLike[str], root: Path) -> tuple[Path, bytes, dict[str, Any]]:
    requested = Path(os.path.abspath(os.fspath(path)))
    expected = root.joinpath(*PurePosixPath(_CONFIG_PATH).parts)
    _reject_links(requested)
    if requested != expected:
        raise ValueError("only the registered config path is accepted")
    raw = requested.read_bytes()
    return requested, raw, validate_config_bytes(raw)


def validate_relative_path(relative: object, *, expected: str | None = None) -> PurePosixPath:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("registered paths must be nonempty forward-slash relative paths")
    path = PurePosixPath(relative)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", relative) or any(
        part in ("", ".", "..") for part in relative.split("/")
    ):
        raise ValueError("registered path is unsafe")
    if expected is not None and relative != expected:
        raise ValueError("registered path differs from the closed config")
    return path


def _resolve_registered(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    pure = validate_relative_path(relative)
    target = root.joinpath(*pure.parts)
    _reject_links(target, allow_missing_leaf=not must_exist)
    if must_exist and not target.exists():
        raise ValueError(f"registered path does not exist: {relative}")
    resolved_root = root.resolve(strict=True)
    resolved = target.resolve(strict=must_exist)
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError("registered path escapes repository root")
    return resolved


def validate_output_directory(root: Path, relative: str) -> Path:
    validate_relative_path(relative, expected="runs/pretrained-gpu-feasibility-20261015-v1")
    output = _resolve_registered(root, relative, must_exist=False)
    if output.exists():
        raise FileExistsError(f"registered output directory already exists: {relative}")
    if not output.parent.is_dir():
        raise ValueError("registered output parent must already exist")
    return output


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a tensor")
    return tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()


def _tensor_hash(tensor: Any) -> str:
    return _sha256(_tensor_bytes(tensor))


def make_probe_batch(
    settings: Mapping[str, Any], *, generator: Any | None = None,
    vocab_size: int | None = None,
) -> ProbeBatch:
    """Draw local-CPU synthetic prompt and answer IDs, then append EOS once."""
    import torch

    if not isinstance(settings, Mapping):
        raise TypeError("synthetic input settings must be a mapping")
    fields = {"algorithm", "answer_tokens_including_eos", "batch_size", "embedding_vocab_size",
              "eos_token_id", "id_high_exclusive", "id_low_inclusive", "mask_policy",
              "noise_requires_grad_diagnostic", "pad_token_id", "prompt_tokens", "total_tokens"}
    if set(settings) != fields:
        raise ValueError("synthetic input settings have unknown or missing fields")
    if settings["algorithm"] != "local_CPU_torch_randint_prompt_then_answer_append_EOS_v1":
        raise ValueError("unknown synthetic ID algorithm")
    if settings["mask_policy"] != "all_valid_no_padding":
        raise ValueError("synthetic padding policy differs from registration")
    if type(settings["noise_requires_grad_diagnostic"]) is not bool:
        raise TypeError("noise diagnostic policy must be a boolean")
    if generator is None or not isinstance(generator, torch.Generator) or str(generator.device) != "cpu":
        raise TypeError("synthetic IDs require an explicit local CPU torch.Generator")
    prompt_length = settings["prompt_tokens"]
    answer_count = settings["answer_tokens_including_eos"]
    batch_size = settings["batch_size"]
    low, high = settings["id_low_inclusive"], settings["id_high_exclusive"]
    vocab = settings["embedding_vocab_size"] if vocab_size is None else vocab_size
    eos = settings["eos_token_id"]
    if any(type(v) is not int for v in (prompt_length, answer_count, batch_size, low, high, vocab, eos)):
        raise TypeError("synthetic dimensions and token IDs must be integers")
    if (prompt_length <= 0 or answer_count < 2 or batch_size <= 0 or low < 0 or high <= low
            or high > vocab or not 0 <= eos < vocab):
        raise ValueError("synthetic dimensions or token range are invalid")
    if settings["pad_token_id"] != eos or settings["total_tokens"] != prompt_length + answer_count:
        raise ValueError("synthetic EOS or total length differs from registration")
    prompt = torch.randint(low, high, (batch_size, prompt_length), generator=generator, dtype=torch.long)
    answer = torch.randint(low, high, (batch_size, answer_count - 1), generator=generator, dtype=torch.long)
    answer = torch.cat((answer, torch.full((batch_size, 1), eos, dtype=torch.long)), dim=1)
    decoder = torch.cat((prompt, answer), dim=1)
    prompt_mask = torch.ones_like(prompt, dtype=torch.bool)
    decoder_mask = torch.ones_like(decoder, dtype=torch.bool)
    labels = torch.full_like(decoder, -100)
    labels[:, prompt_length:] = answer
    if (labels[:, :prompt_length] != -100).any().item() or not torch.equal(labels[:, -1], torch.full((batch_size,), eos)):
        raise RuntimeError("synthetic label framing failed")
    return ProbeBatch(prompt, prompt_mask, decoder, decoder_mask, labels)


def batch_identity(batch: ProbeBatch) -> dict[str, Any]:
    tensors = {}
    for name in ("prompt_ids", "prompt_mask", "decoder_ids", "decoder_mask", "labels"):
        value = getattr(batch, name)
        tensors[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _tensor_hash(value)}
    return tensors


def _named_tensor_bytes(items: Mapping[str, Any] | list[tuple[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, tensor in items.items() if isinstance(items, Mapping) else items:
        rows[name] = {"numel": int(tensor.numel()), "bytes": int(tensor.numel() * tensor.element_size()),
                      "dtype": str(tensor.dtype), "device": str(tensor.device), "shape": list(tensor.shape)}
    return rows


def hash_original_parameters(host: Any) -> dict[str, str]:
    """Hash original weights sequentially; never retain a second full base."""
    result = {}
    for name, tensor in host.backbone.named_parameters():
        digest = hashlib.sha256()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_tensor_bytes(tensor))
        result[name] = digest.hexdigest()
    return result


def _group_norm(parameters: list[Any]) -> tuple[float, dict[str, float]]:
    total_sq = 0.0
    each = {}
    for index, parameter in enumerate(parameters):
        if parameter.grad is None:
            raise RuntimeError(f"active parameter {index} has no gradient")
        if not parameter.grad.is_floating_point() or not bool(parameter.grad.isfinite().all().item()):
            raise RuntimeError(f"active parameter {index} has nonfinite gradient")
        norm = float(parameter.grad.detach().double().norm().item())
        each[str(index)] = norm
        total_sq += norm * norm
    return math.sqrt(total_sq), each


def create_optimizer(host: Any, config: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch

    settings = config["optimizer"]
    groups = [
        {"params": list(host.workspace.parameters()), "name": "workspace"},
        {"params": list(host.readers.parameters()), "name": "readers"},
    ]
    optimizer = torch.optim.AdamW(
        groups, lr=settings["learning_rate"], betas=tuple(settings["betas"]),
        eps=settings["epsilon"], weight_decay=settings["weight_decay"],
        amsgrad=settings["amsgrad"], foreach=settings["foreach"], fused=settings["fused"],
        capturable=settings["capturable"], differentiable=settings["differentiable"],
        maximize=settings["maximize"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 / (step + 1))
    return optimizer, scheduler


def run_shared_step(
    host: Any, batch: ProbeBatch, noise_generator: Any, optimizer: Any, scheduler: Any,
    *, steps: int, expected_original_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run one real host encode/workspace/decode/loss/backward/update step."""
    import torch
    from dwr.training.loss import answer_only_loss

    invocation_start = time.perf_counter()
    host.train()
    host.backbone.eval()
    device = next(host.backbone.parameters()).device
    stage = {}
    _synchronize(device)
    timer = time.perf_counter()
    batch = ProbeBatch(*(getattr(batch, name).to(device) for name in
                         ("prompt_ids", "prompt_mask", "decoder_ids", "decoder_mask", "labels")))
    original_before = dict(expected_original_hashes or hash_original_parameters(host))
    optimizer.zero_grad(set_to_none=True)
    stage["input_transfer_hash_and_zero_grad_seconds"] = time.perf_counter() - timer
    _synchronize(device)
    timer = time.perf_counter()
    x_original = host.encode_prompt(batch.prompt_ids, batch.prompt_mask)
    _synchronize(device)
    stage["prompt_encode_seconds"] = time.perf_counter() - timer
    new = next(host.workspace.parameters())
    x = x_original.to(device=new.device, dtype=new.dtype)
    noise = host.workspace.sample_noise(batch.prompt_ids.shape[0], generator=noise_generator,
                                       device=device, dtype=new.dtype).requires_grad_(True)
    _synchronize(device)
    timer = time.perf_counter()
    with torch.autocast(device_type=device.type, enabled=False):
        output, states = host.workspace(x, noise=noise, prompt_padding_mask=~batch.prompt_mask,
                                        steps=steps, return_states=True)
    _synchronize(device)
    stage["workspace_seconds"] = time.perf_counter() - timer
    if len(states) != steps + 1:
        raise RuntimeError("workspace did not retain the complete registered recurrent state chain")
    for state in states:
        state.retain_grad()
    _synchronize(device)
    timer = time.perf_counter()
    logits = host.decode_with_workspace(batch.decoder_ids, batch.decoder_mask, output)
    _synchronize(device)
    stage["answer_decode_seconds"] = time.perf_counter() - timer
    _synchronize(device)
    timer = time.perf_counter()
    loss_result = answer_only_loss(logits, batch.labels)
    _synchronize(device)
    stage["loss_seconds"] = time.perf_counter() - timer
    _synchronize(device)
    timer = time.perf_counter()
    loss_result.loss.backward()
    _synchronize(device)
    stage["backward_seconds"] = time.perf_counter() - timer

    workspace_norm, workspace_tensor_norms = _group_norm(list(host.workspace.parameters()))
    reader_norm, reader_tensor_norms = _group_norm(list(host.readers.parameters()))
    if workspace_norm <= 0 or reader_norm <= 0:
        raise RuntimeError("workspace and each-reader aggregate gradients must be nonzero")
    each_reader_norms = {}
    for layer, reader in host.readers.items():
        norm, _ = _group_norm(list(reader.parameters()))
        if norm <= 0:
            raise RuntimeError(f"reader {layer} aggregate gradient must be nonzero")
        each_reader_norms[layer] = norm
    state_norms = []
    for index, state in enumerate(states):
        if state.grad is None or not bool(state.grad.isfinite().all().item()):
            raise RuntimeError(f"retained Z{index} gradient is absent or nonfinite")
        norm = float(state.grad.detach().double().norm().item())
        if norm <= 0:
            raise RuntimeError(f"retained Z{index} gradient is zero")
        state_norms.append(norm)
    if noise.grad is None or not bool(noise.grad.isfinite().all().item()):
        raise RuntimeError("diagnostic noise gradient is absent or nonfinite")

    parameter_info = _named_tensor_bytes(list(host.named_parameters()))
    gradient_info = _named_tensor_bytes([(name, parameter.grad) for name, parameter in host.named_parameters()
                                         if parameter.grad is not None])
    state_info = _named_tensor_bytes([(f"Z{index}", state) for index, state in enumerate(states)])
    state_grad_info = _named_tensor_bytes([(f"Z{index}.grad", state.grad) for index, state in enumerate(states)])
    noise_grad_info = _named_tensor_bytes({"noise.grad": noise.grad})
    zero_parameters = [name for name, parameter in host.named_parameters()
                       if parameter.grad is not None and float(parameter.grad.detach().abs().sum().item()) == 0.0]
    _synchronize(device)
    timer = time.perf_counter()
    optimizer.step()
    scheduler.step()
    _synchronize(device)
    stage["optimizer_and_schedule_seconds"] = time.perf_counter() - timer
    new_parameter_bytes = sum(row["bytes"] for name, row in parameter_info.items() if not name.startswith("backbone."))
    optimizer_state_info = {}
    for parameter, state in optimizer.state.items():
        name = next(name for name, candidate in host.named_parameters() if candidate is parameter)
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state_info[f"{name}.{key}"] = _named_tensor_bytes({f"{name}.{key}": value})[f"{name}.{key}"]
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                optimizer_state_info[f"{name}.{key}"] = {"bytes": 8, "dtype": type(value).__name__,
                                                        "device": "cpu-scalar", "value": value}
            else:
                raise RuntimeError(f"unsupported optimizer state value for {name}.{key}")
    if hash_original_parameters(host) != original_before:
        raise RuntimeError("original frozen parameter bytes changed during probe")
    if any(parameter.grad is not None for parameter in host.backbone.parameters()):
        raise RuntimeError("original backbone gradients must remain None")
    if any(parameter.grad is None for parameter in (*host.workspace.parameters(), *host.readers.parameters())):
        raise RuntimeError("active new-module gradients must remain available through measurement")
    return {
        "loss": float(loss_result.loss.detach().item()), "supervised_tokens": loss_result.supervised_tokens,
        "stage_seconds": stage,
        "shared_step_seconds_including_transfers_hashes_optimizer": time.perf_counter() - invocation_start,
        "workspace_gradient_group_norm": workspace_norm, "reader_gradient_group_norm": reader_norm,
        "workspace_gradient_tensor_norms": workspace_tensor_norms,
        "reader_gradient_tensor_norms": reader_tensor_norms, "zero_gradient_parameter_names": zero_parameters,
        "each_reader_gradient_group_norms": each_reader_norms,
        "retained_state_gradient_norms": state_norms,
        "noise_gradient_norm": float(noise.grad.detach().double().norm().item()),
        "parameter_tensors": parameter_info, "gradient_tensors": gradient_info,
        "optimizer_state_tensors": optimizer_state_info, "retained_states": state_info,
        "retained_state_gradients": state_grad_info, "diagnostic_noise_gradient": noise_grad_info,
        "new_parameter_bytes": new_parameter_bytes,
        "optimizer_state_bytes": sum(row["bytes"] for row in optimizer_state_info.values()),
        "diagnostic_state_and_gradient_bytes": sum(row["bytes"] for row in state_info.values())
            + sum(row["bytes"] for row in state_grad_info.values())
            + sum(row["bytes"] for row in noise_grad_info.values()),
        "diagnostic_bytes_accounting": "logical tensor bytes; shared storage may be counted in Z0 and noise gradient",
        "optimizer_step": int(scheduler.last_epoch), "profiler_flops": None,
    }


def _deep_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.device == right.device and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_deep_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_deep_equal(a, b) for a, b in zip(left, right))
    return left == right


def _synchronize(device: Any) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def roundtrip_checkpoint(
    path: str | os.PathLike[str], *, host: Any, optimizer: Any, scheduler: Any,
    sampler: Any, noise_generator: Any, identities: Mapping[str, str], supervised_tokens: int,
) -> dict[str, Any]:
    """Save and restore one post-step boundary in the same host/base instance."""
    import random
    import torch
    import numpy as np
    from dwr.training.checkpoints import save_training_checkpoint, load_training_checkpoint
    from dwr.training.sampler import SourceSampler

    optimizer.zero_grad(set_to_none=True)
    # One consumed synthetic fixture example mirrors the update progress; this is
    # sampler state coverage only and does not imply source-corpus progress.
    sampler.next_ids(1)
    original = hash_original_parameters(host)
    digest, size = save_training_checkpoint(path, host=host, optimizer=optimizer, scheduler=scheduler,
        scaler=None, sampler=sampler, noise_generator=noise_generator, identities=dict(identities),
        progress={"optimizer_step": 1, "examples_seen": 1,
                  "supervised_tokens_seen": supervised_tokens})
    saved_state = {
        "workspace": {k: v.detach().clone() for k, v in host.workspace.state_dict().items()},
        "readers": {k: v.detach().clone() for k, v in host.readers.state_dict().items()},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(), "noise": noise_generator.get_state().clone(),
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda": tuple(x.clone() for x in torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else (),
    }
    with torch.no_grad():
        for parameter in (*host.workspace.parameters(), *host.readers.parameters()):
            parameter.add_(torch.randn(parameter.shape, generator=torch.Generator(device="cpu").manual_seed(9),
                                       dtype=parameter.dtype).to(parameter.device) * 0.01)
    torch.rand(7, generator=noise_generator, device=noise_generator.device)
    cuda_rng_perturbed = torch.cuda.is_initialized()
    if cuda_rng_perturbed:
        torch.rand(3, device="cuda:0")
    sampler.next_ids(1)
    random.random()
    np.random.random()
    torch.rand(3)

    fresh_optimizer, fresh_scheduler = create_optimizer(host, {
        "optimizer": {"learning_rate": 0.0001, "betas": [0.9, 0.999], "epsilon": 1e-8,
                      "weight_decay": 0.01, "amsgrad": False, "foreach": False, "fused": False,
                      "capturable": False, "differentiable": False, "maximize": False},
    })
    fresh_sampler = SourceSampler(sampler.example_ids, seed=sampler.seed)
    sampler_destination_differed = fresh_sampler.state_dict() != saved_state["sampler"]
    if not sampler_destination_differed:
        raise RuntimeError("fresh sampler destination already equals the saved sampler boundary")
    fresh_noise = torch.Generator(device=noise_generator.device)
    fresh_noise.manual_seed(1)
    progress = load_training_checkpoint(path, expected_sha256=digest, host=host,
        optimizer=fresh_optimizer, scheduler=fresh_scheduler, scaler=None, sampler=fresh_sampler,
        noise_generator=fresh_noise, identities=dict(identities))
    restored = {
        "workspace": host.workspace.state_dict(), "readers": host.readers.state_dict(),
        "optimizer": fresh_optimizer.state_dict(), "scheduler": fresh_scheduler.state_dict(),
        "sampler": fresh_sampler.state_dict(), "noise": fresh_noise.get_state(),
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "cuda": tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else (),
    }
    # NumPy's legacy state includes an ndarray; normalize it for strict equality.
    expected_numpy = (saved_state["numpy"][0], torch.from_numpy(saved_state["numpy"][1].copy()),
                      *saved_state["numpy"][2:])
    actual_numpy = (restored["numpy"][0], torch.from_numpy(restored["numpy"][1].copy()),
                   *restored["numpy"][2:])
    saved_state["numpy"] = expected_numpy
    restored["numpy"] = actual_numpy
    equal = _deep_equal(saved_state, restored)
    if not equal:
        raise RuntimeError("same-host checkpoint did not exactly restore learned and RNG state")
    if hash_original_parameters(host) != original:
        raise RuntimeError("original parameter bytes changed across checkpoint roundtrip")
    if any(parameter.grad is not None for parameter in host.backbone.parameters()):
        raise RuntimeError("original gradient boundary must remain None")
    return {"sha256": digest, "size_bytes": size, "progress": progress,
            "fresh_optimizer_scheduler_sampler_noise": True, "exact_state_match": True,
            "sampler_destination_differed_before_restore": sampler_destination_differed,
            "synthetic_sampler_fixture_bookkeeping": True,
            "same_host_no_second_base": True, "original_parameter_bytes_unchanged": True,
            "deliberately_perturbed_rng_streams": {
                "python": True, "numpy": True, "torch_cpu": True,
                "torch_cuda": cuda_rng_perturbed, "explicit_noise": True,
                "sampler_position": True,
            }}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> bytes:
    proc = subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=30)
    return proc.stdout


def validate_clean_source(root: Path, expected_commit: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    top = Path(os.fsdecode(_git(root, "rev-parse", "--show-toplevel")).strip()).resolve(strict=True)
    if root != top:
        raise ValueError("source root must be the Git top-level directory")
    head = os.fsdecode(_git(root, "rev-parse", "HEAD")).strip()
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status:
        raise ValueError("source worktree or index is not clean")
    _git(root, "cat-file", "-e", f"{expected_commit}^{{commit}}")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", expected_commit, head], cwd=root,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if ancestor.returncode != 0:
        raise ValueError("registered source base commit is not an ancestor of current clean HEAD")
    files = []
    names = _git(root, "ls-files", "-z", "--cached").split(b"\0")
    for raw_name in names:
        if not raw_name:
            continue
        name = os.fsdecode(raw_name)
        path = root / Path(name)
        _reject_links(path)
        physical = path.read_bytes()
        blob = _git(root, "cat-file", "blob", f"HEAD:{name}")
        if physical != blob:
            raise ValueError(f"tracked source physical bytes differ from HEAD: {name}")
        files.append({"path": name.replace("\\", "/"), "sha256": _sha256(physical)})
    return {"commit": head, "clean": True, "status_sha256": _sha256(status), "files": files}


def _artifact_inventory(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    directory = _resolve_registered(root, model["local_dir"])
    manifest = _resolve_registered(root, model["artifact_manifest"])
    if _hash_file(manifest) != MODEL_MANIFEST_SHA256:
        raise ValueError("model artifact manifest hash mismatch")
    inventory = _read_small_json(manifest.parent / "artifact-inventory.json", model["artifact_inventory_sha256"])
    if (inventory.get("repo_id") != model["repo_id"]
            or inventory.get("revision") != model["revision"]
            or inventory.get("corpus_access") is not False
            or inventory.get("model_tensors_loaded") is not False
            or inventory.get("dtypes") != ["BF16"]
            or type(inventory.get("parameter_elements")) is not int):
        raise ValueError("model fetch receipt does not match the pinned artifact contract")
    by_name = {entry["name"]: entry for entry in inventory.get("files", [])}
    if not by_name or len(by_name) != len(inventory.get("files", [])):
        raise ValueError("pinned model artifact inventory has missing or duplicate file names")
    for entry in inventory.get("files", []):
        name = entry["name"]
        file_path = _resolve_registered(root, f"{model['local_dir']}/{name}")
        if file_path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"pinned model artifact size mismatch: {name}")
        if _hash_file(file_path) != entry["sha256"]:
            raise ValueError(f"pinned model artifact bytes mismatch: {name}")
    config_file = _resolve_registered(directory, "config.json")
    model_config = _read_small_json(config_file, model["config_sha256"])
    layout = config["layout"]
    synthetic = config["synthetic_input"]
    if (model_config.get("model_type") != "qwen2"
            or model_config.get("vocab_size") != synthetic["embedding_vocab_size"]
            or model_config.get("eos_token_id") != synthetic["eos_token_id"]
            or model_config.get("hidden_size") != layout["context_dim"]
            or model_config.get("num_hidden_layers", 0) <= max(layout["adapter_layers"])):
        raise ValueError("pinned model metadata does not match the registered host and token contract")
    weight_files = [name for name in by_name if name.endswith(".safetensors")]
    if not weight_files:
        raise ValueError("pinned artifact inventory has no safetensors weights")
    return {"manifest_sha256": MODEL_MANIFEST_SHA256,
            "inventory_sha256": model["artifact_inventory_sha256"],
            "config_sha256": model["config_sha256"], "weight_sha256": model["weight_sha256"],
            "files": sorted(by_name), "model_type": model_config["model_type"],
            "vocab_size": model_config["vocab_size"], "hidden_size": model_config["hidden_size"],
            "num_hidden_layers": model_config["num_hidden_layers"],
            "eos_token_id": model_config["eos_token_id"],
            "parameter_elements": inventory["parameter_elements"], "original_dtype": "bfloat16"}


def _data_preconditions(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    from dwr.experiments.records import verify_artifacts

    conditions = config["data_preconditions"]
    export_dir = _resolve_registered(root, conditions["source_export_dir"])
    audit_dir = _resolve_registered(root, conditions["token_audit_dir"])
    export_manifest = export_dir / "manifest.json"
    audit_manifest = audit_dir / "manifest.json"
    if _hash_file(export_manifest) != EXPORT_MANIFEST_SHA256:
        raise ValueError("source export manifest identity mismatch")
    if _hash_file(audit_manifest) != TOKEN_AUDIT_MANIFEST_SHA256:
        raise ValueError("token audit manifest identity mismatch")
    export_terminal = verify_artifacts(export_dir)
    audit_terminal = verify_artifacts(audit_dir)
    export_summary = _read_small_json(export_dir / "summary.json")
    audit_summary = _read_small_json(audit_dir / "summary.json")
    counts = conditions["allowed_source_counts"]
    _validate_data_receipts(conditions, config["synthetic_input"], export_terminal,
                            audit_terminal, export_summary, audit_summary)
    return {"source_export_manifest_sha256": EXPORT_MANIFEST_SHA256,
            "token_audit_manifest_sha256": TOKEN_AUDIT_MANIFEST_SHA256,
            "counts": counts, "token_audit_status": audit_summary["status"],
            "source_export_summary_sha256": _hash_file(export_dir / "summary.json"),
            "token_audit_summary_sha256": _hash_file(audit_dir / "summary.json"),
            "task_semantics_read": False}


def _validate_data_receipts(conditions: Mapping[str, Any], synthetic: Mapping[str, Any],
                            export_terminal: Mapping[str, Any], audit_terminal: Mapping[str, Any],
                            export_summary: Mapping[str, Any], audit_summary: Mapping[str, Any]) -> None:
    counts = conditions["allowed_source_counts"]
    if export_terminal.get("status") != "completed" or export_summary.get("exported_counts") != counts:
        raise ValueError("source export precondition is incomplete or count-mismatched")
    if audit_terminal.get("status") != "completed" or audit_summary.get("status") != conditions["token_audit_status"]:
        raise ValueError("source token audit precondition did not pass")
    if audit_summary.get("counts") != counts or audit_summary.get("protected_semantic_payload_access") is not False:
        raise ValueError("source token audit counts or access boundary mismatch")
    maxima = audit_summary.get("maxima")
    if not isinstance(maxima, dict) or set(maxima) != set(counts):
        raise ValueError("source token audit split inventory mismatch")
    for split, values in maxima.items():
        if not isinstance(values, dict) or set(values) != {
            "prompt_tokens", "answer_tokens_including_eos", "total_tokens"
        }:
            raise ValueError(f"source token audit dimensions are malformed for {split}")
        caps = (synthetic["prompt_tokens"], synthetic["answer_tokens_including_eos"], synthetic["total_tokens"])
        observed = (values["prompt_tokens"], values["answer_tokens_including_eos"], values["total_tokens"])
        if any(type(item) is not int or item < 0 for item in observed):
            raise ValueError(f"source token audit dimensions are invalid for {split}")
        if any(value > cap for value, cap in zip(observed, caps)):
            raise ValueError(f"source token cap audit exceeds registered limit for {split}")


def _check_environment(config: Mapping[str, Any]) -> dict[str, str]:
    from importlib import metadata

    required = {"numpy": config["environment"]["numpy"],
                "torch": config["environment"]["torch"],
                "transformers": config["environment"]["transformers"]}
    observed = {}
    for package, expected in required.items():
        try:
            observed[package] = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"ENVIRONMENT_BLOCKED: required package is missing: {package}") from exc
        if observed[package] != expected:
            raise RuntimeError(f"ENVIRONMENT_BLOCKED: {package} version differs from registration")
    import torch
    if torch.version.cuda != config["environment"]["cuda_build"]:
        raise RuntimeError("ENVIRONMENT_BLOCKED: Torch CUDA build differs from registration")
    if not torch.cuda.is_available() or torch.cuda.device_count() != config["hardware"]["device_count"]:
        raise RuntimeError("ENVIRONMENT_BLOCKED: CUDA device availability or count differs from registration")
    return {**observed, "cuda_build": str(torch.version.cuda)}


def _hardware_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total,memory.free",
               "--format=csv,noheader,nounits"]
    proc = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", timeout=20)
    rows = [line.strip().split(", ") for line in proc.stdout.splitlines() if line.strip()]
    if len(rows) != config["hardware"]["device_count"] or any(len(row) != 5 for row in rows):
        raise RuntimeError("hardware inventory differs from registered device count")
    uuid, name, driver, total, free = rows[0]
    observed = {"uuid": uuid, "name": name, "driver_version": driver,
                "total_mib": int(total), "initial_free_mib": int(free),
                "initial_used_mib": int(total) - int(free)}
    expected = config["hardware"]
    for key in ("uuid", "name", "driver_version"):
        if observed[key] != expected[key]:
            raise RuntimeError(f"hardware {key} does not match registration")
    if observed["total_mib"] < expected["minimum_total_mib"] or observed["initial_free_mib"] < expected["minimum_initial_free_mib"]:
        raise RuntimeError("device total or initial free memory is below registered minimum")
    return observed


def _read_numerics_policy(config: Mapping[str, Any], torch_module: Any) -> dict[str, Any]:
    cuda_backend = torch_module.backends.cuda
    getters = {
        "flash": "flash_sdp_enabled", "memory_efficient": "mem_efficient_sdp_enabled",
        "math": "math_sdp_enabled", "cudnn": "cudnn_sdp_enabled",
    }
    observed = {}
    for key, name in getters.items():
        getter = getattr(cuda_backend, name, None)
        if not callable(getter):
            raise RuntimeError(f"actual SDPA policy getter is unavailable: {name}")
        observed[key] = getter()
        if type(observed[key]) is not bool:
            raise RuntimeError(f"actual SDPA policy getter returned a nonboolean value: {name}")
    sdpa = {
        "flash": config["attention"]["builtin_flash"],
        "memory_efficient": config["attention"]["builtin_memory_efficient"],
        "math": config["attention"]["builtin_math"],
        "cudnn": config["attention"]["builtin_cudnn"],
    }
    if observed != sdpa:
        raise RuntimeError(f"actual SDPA policy differs from registration: expected {sdpa}, observed {observed}")
    settings = config["numerics"]
    policy = {
        "sdpa": observed,
        "tf32_matmul": torch_module.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch_module.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch_module.backends.cudnn.benchmark,
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
        "deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
        "default_float_dtype": str(torch_module.get_default_dtype()),
    }
    if any(type(policy[key]) is not bool for key in (
            "tf32_matmul", "tf32_cudnn", "cudnn_benchmark", "deterministic_algorithms")):
        raise RuntimeError("actual numerics policy returned a nonboolean setting")
    if not isinstance(policy["float32_matmul_precision"], str) or not isinstance(policy["default_float_dtype"], str):
        raise RuntimeError("actual numerics policy returned a malformed precision setting")
    expected = {
        "tf32_matmul": settings["tf32_matmul"], "tf32_cudnn": settings["tf32_cudnn"],
        "cudnn_benchmark": settings["cudnn_benchmark"],
        "float32_matmul_precision": settings["float32_matmul_precision"],
        "deterministic_algorithms": settings["deterministic_algorithms"],
        "default_float_dtype": str(getattr(torch_module, settings["default_float_dtype"])),
    }
    if any(policy[key] != value for key, value in expected.items()):
        raise RuntimeError(f"actual numerics policy differs from registration: expected {expected}, observed {policy}")
    return policy


def _set_numerics(config: Mapping[str, Any], *, torch_module: Any | None = None) -> dict[str, Any]:
    if torch_module is None:
        import torch as torch_module

    settings = config["numerics"]
    cuda_backend = torch_module.backends.cuda
    torch_module.set_default_dtype(getattr(torch_module, settings["default_float_dtype"]))
    cuda_backend.matmul.allow_tf32 = settings["tf32_matmul"]
    torch_module.backends.cudnn.allow_tf32 = settings["tf32_cudnn"]
    torch_module.backends.cudnn.benchmark = settings["cudnn_benchmark"]
    torch_module.set_float32_matmul_precision(settings["float32_matmul_precision"])
    torch_module.use_deterministic_algorithms(settings["deterministic_algorithms"])
    # The selected version's getters and setters are part of the pinned policy.
    # Missing APIs or a changed backend policy fail closed; there is no fallback.
    cuda_backend.enable_flash_sdp(config["attention"]["builtin_flash"])
    cuda_backend.enable_mem_efficient_sdp(config["attention"]["builtin_memory_efficient"])
    cuda_backend.enable_math_sdp(config["attention"]["builtin_math"])
    cuda_backend.enable_cudnn_sdp(config["attention"]["builtin_cudnn"])
    return _read_numerics_policy(config, torch_module)


def _load_production_host(root: Path, config: Mapping[str, Any]) -> Any:
    import torch
    try:
        from transformers import Qwen2ForCausalLM
    except ModuleNotFoundError as exc:
        if exc.name != "transformers":
            raise
        raise RuntimeError("ENVIRONMENT_BLOCKED: required transformers package is absent") from exc
    from dwr.models.qwen2 import Qwen2WorkspaceLM
    from dwr.models.workspace import FixedWorkspace

    model = config["model"]
    local = _resolve_registered(root, model["local_dir"])
    original = Qwen2ForCausalLM.from_pretrained(
        local, torch_dtype=torch.bfloat16, use_safetensors=True, local_files_only=True,
        trust_remote_code=False, device_map=None, low_cpu_mem_usage=False,
        attn_implementation="sdpa",
    )
    if original.config._attn_implementation != "sdpa":
        raise RuntimeError("loaded Qwen2 host did not select registered SDPA implementation")
    layout = config["layout"]
    with torch.random.fork_rng(devices=[]):
        seed_cpu_initialization(torch, config["seeds"]["workspace_initialization"])
        workspace = FixedWorkspace(layout["context_dim"], num_slots=layout["num_slots"],
            latent_dim=layout["latent_dim"], num_heads=layout["workspace_heads"],
            max_steps=layout["max_steps"], slot_identity=layout["slot_identity"]).float()
    with torch.random.fork_rng(devices=[]):
        seed_cpu_initialization(torch, config["seeds"]["reader_initialization"])
        host = Qwen2WorkspaceLM(original, workspace, prompt_layer=layout["prompt_layer"],
            adapter_layers=tuple(layout["adapter_layers"]), reader_heads=layout["reader_heads"],
            gate_init=layout["gate_init"])
    host.to(device=config["hardware"]["device"])
    host.eval()
    return host


def classify_failure(exc: BaseException, *, phase: str) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if phase in {"metadata", "preflight", "load"} or any(token in text for token in
            ("cuda is not available", "nvidia-smi", "hardware", "environment_blocked", "artifact", "precondition")):
        return "ENVIRONMENT_BLOCKED"
    return "FEASIBILITY_FAILED"


def _controller(
    *, record_initial: Callable[[], Any], record_access: Callable[[], Any],
    query_hardware: Callable[[], Any], run_candidate: Callable[[Any], Any],
) -> Any:
    """Order-only controller seam; no CUDA or model API is called here."""
    record_initial()
    record_access()
    hardware = query_hardware()
    return run_candidate(hardware)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> bytes:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw


def _process_peak_working_set_bytes() -> int:
    """Return the Windows process high-water RSS; unsupported platforms fail closed."""
    if os.name != "nt":
        raise RuntimeError("process high-water working-set measurement is unsupported on this platform")
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
    kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
    get_info = psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    get_process = kernel32.GetCurrentProcess
    get_process.argtypes = []
    get_process.restype = wintypes.HANDLE
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_info(get_process(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


def make_probe_guard(wall_seconds: float, memory_mib: int, *, peak_reader: Any | None = None) -> Any:
    """Use OS-maintained process peak RSS without changing shared guard defaults."""
    from dwr.tasks.run_support import ResourceGuard

    return ResourceGuard(wall_seconds, memory_mib,
                         memory_reader=peak_reader or _process_peak_working_set_bytes)


def _production_candidate(root: Path, config: Mapping[str, Any], hardware: Mapping[str, Any],
                          output: Path, started: float, guard: Any,
                          artifact_receipt: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from dwr.training.sampler import SourceSampler

    actual_policy = _set_numerics(config)
    seed_cpu_initialization(torch, config["seeds"]["torch_cpu_global"])
    import random
    import numpy as np
    random.seed(config["seeds"]["python_global"])
    np.random.seed(config["seeds"]["numpy_global"])
    torch.cuda.manual_seed_all(config["seeds"]["torch_cuda_global"])
    source = config["synthetic_input"]
    id_generator = torch.Generator(device="cpu").manual_seed(config["seeds"]["synthetic_ids"])
    batch = make_probe_batch(source, generator=id_generator)
    noise_generator = torch.Generator(device="cuda:0").manual_seed(config["seeds"]["workspace_noise_cuda"])
    guard.check()

    # Peak counters start before loading/allocating any candidate tensors.
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    host = _load_production_host(root, config)
    guard.check()
    host.backbone.eval()
    original_parameters = list(host.backbone.parameters())
    parameter_count = sum(parameter.numel() for parameter in original_parameters)
    if parameter_count != artifact_receipt["parameter_elements"]:
        raise RuntimeError("loaded model parameter count differs from pinned artifact receipt")
    if {parameter.dtype for parameter in original_parameters} != {torch.bfloat16}:
        raise RuntimeError("loaded original model dtype differs from registered BF16")
    original_hashes = hash_original_parameters(host)
    guard.check()
    optimizer, scheduler = create_optimizer(host, config)
    guard.check()
    setup_seconds = time.monotonic() - started
    identities = {
        "source": _sha256(_git(root, "rev-parse", "HEAD").strip()),
        "config": CONFIG_SHA256, "data": TOKEN_AUDIT_MANIFEST_SHA256,
        "model": config["model"]["weight_sha256"],
        "tokenizer": config["model"]["artifact_inventory_sha256"],
        "schedule": _sha256(json.dumps(config["schedule"], sort_keys=True, separators=(",", ":")).encode("utf-8")),
    }
    batch_info = batch_identity(batch)
    torch.cuda.synchronize(0)
    step = run_shared_step(host, batch, noise_generator, optimizer, scheduler,
                           steps=config["layout"]["steps"], expected_original_hashes=original_hashes)
    guard.check()
    torch.cuda.synchronize(0)
    step_seconds = step["shared_step_seconds_including_transfers_hashes_optimizer"]
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    caps = config["hardware"]
    if peak_allocated > caps["maximum_peak_allocated_mib"] * 1024**2:
        raise RuntimeError("peak allocated memory exceeded the registered cap")
    if peak_reserved > caps["maximum_peak_reserved_mib"] * 1024**2:
        raise RuntimeError("peak reserved memory exceeded the registered cap")

    sampler_seed = config["seeds"]["source_sampler_fixture"]
    sampler = SourceSampler((_sha256(b"synthetic-gpu-feasibility-probe"),), seed=sampler_seed)
    checkpoint = output / "state.pt"
    roundtrip = roundtrip_checkpoint(checkpoint, host=host, optimizer=optimizer, scheduler=scheduler,
        sampler=sampler, noise_generator=noise_generator, identities=identities,
        supervised_tokens=step["supervised_tokens"])
    guard.check()
    torch.cuda.synchronize(0)
    peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(0)))
    peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(0)))
    if peak_allocated > caps["maximum_peak_allocated_mib"] * 1024**2 or peak_reserved > caps["maximum_peak_reserved_mib"] * 1024**2:
        raise RuntimeError("peak memory exceeded registered cap through checkpoint evidence")
    if time.monotonic() - started > config["limits"]["wall_seconds"]:
        raise TimeoutError("registered wall-time cap exceeded")
    if int(peak_allocated) > caps["maximum_peak_allocated_mib"] * 1024**2:
        raise RuntimeError("peak allocation cap exceeded")
    base_hashes_after = hash_original_parameters(host)
    torch.cuda.synchronize(0)
    peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(0)))
    peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(0)))
    if peak_allocated > caps["maximum_peak_allocated_mib"] * 1024**2 or peak_reserved > caps["maximum_peak_reserved_mib"] * 1024**2:
        raise RuntimeError("peak memory exceeded registered cap after final base hashing")
    guard.check()
    outcome = {
        "schema_version": "dwr-gpu-feasibility-outcome-v1", "status": "PASS_FEASIBILITY",
        "hardware": dict(hardware), "batch": batch_info,
        "model": {"parameter_elements": parameter_count, "original_dtype": "bfloat16",
                  "new_parameter_dtype": "float32", "attention": "sdpa"},
        "measurement": step | {
            "probe_setup_through_optimizer_construction_seconds": setup_seconds,
            "shared_step_seconds_first_update": step_seconds,
            "elapsed_through_first_update_including_probe_initialization": setup_seconds + step_seconds,
            **step["stage_seconds"],
            "profiler_flops": None, "profiler": "disabled", "latency_benchmark": False,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "process_peak_working_set_bytes_before_outcome_serialization": guard.peak_bytes,
            "process_memory_reader": "Windows GetProcessMemoryInfo PeakWorkingSetSize",
            "peak_memory_scope": "through final frozen-base hashing before outcome serialization; later evidence and terminal artifact checks are recorded by the terminal budget event",
            "actual_numeric_policy": actual_policy,
            "base_parameter_hashes_before": original_hashes,
            "base_parameter_hashes_after": base_hashes_after,
        },
        "checkpoint_roundtrip": roundtrip,
        "elapsed_seconds_before_terminal_record_finalization": time.monotonic() - started,
    }
    if outcome["measurement"]["base_parameter_hashes_before"] != outcome["measurement"]["base_parameter_hashes_after"]:
        raise RuntimeError("original frozen parameter bytes changed after checkpoint")
    if time.monotonic() - started > config["limits"]["wall_seconds"]:
        raise TimeoutError("registered wall-time cap exceeded through evidence serialization")
    if output.exists() and (output / _OUTCOME_PATH).exists():
        raise FileExistsError("outcome artifact already exists")
    _write_json_exclusive(output / _OUTCOME_PATH, outcome)
    guard.check()
    torch.cuda.synchronize(0)
    final_allocated = int(torch.cuda.max_memory_allocated(0))
    final_reserved = int(torch.cuda.max_memory_reserved(0))
    if final_allocated > caps["maximum_peak_allocated_mib"] * 1024**2 or final_reserved > caps["maximum_peak_reserved_mib"] * 1024**2:
        raise RuntimeError("peak memory exceeded registered cap after evidence serialization")
    return outcome


def run_probe(config_path: str | os.PathLike[str], *, command: list[str]) -> dict[str, Any]:
    """Run the one closed registered GPU invocation; no overrides are accepted."""
    started = time.monotonic()
    from dwr.experiments.records import RunRecord
    guard = make_probe_guard(900, 12288)
    guard.started = started
    guard.check()
    root = _repo_root()
    phase = "metadata"
    config_file, raw, config = _registered_config(config_path, root)
    guard.check()
    source = validate_clean_source(root, config["source_base_commit"])
    guard.check()
    output = validate_output_directory(root, config["output_dir"])
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    identities = {
        "source": _sha256(source["commit"].encode("ascii")), "config": _sha256(raw),
        "data": TOKEN_AUDIT_MANIFEST_SHA256, "model": config["model"]["weight_sha256"],
        "tokenizer": config["model"]["artifact_inventory_sha256"],
        "schedule": _sha256(json.dumps(config["schedule"], sort_keys=True, separators=(",", ":")).encode("utf-8")),
    }
    record = RunRecord.create(output, command=command, config_path=config_file,
        expected_config_sha256=CONFIG_SHA256, identities=identities, rng_streams=config["seeds"],
        settings={"attention": config["attention"], "layout": config["layout"],
                  "precision": config["precision"], "numerics": config["numerics"]},
        budgets=config["limits"], confirmatory=False, source_root=root)
    guard.check()
    phase = "preflight"
    outcome: dict[str, Any] | None = None
    data_receipt: dict[str, Any] = {}
    artifact_receipt: dict[str, Any] = {}

    def initial_record() -> None:
        record.append_event("initial_manifest_written", {"config_sha256": CONFIG_SHA256,
            "source_commit": source["commit"], "source_clean": source["clean"],
            "output_dir": str(output)})

    def access_record() -> None:
        nonlocal data_receipt, artifact_receipt
        record.append_event("precondition_access_started", {
            "source_export_manifest_sha256": EXPORT_MANIFEST_SHA256,
            "token_audit_manifest_sha256": TOKEN_AUDIT_MANIFEST_SHA256,
            "model_manifest_sha256": MODEL_MANIFEST_SHA256,
            "protected_task_semantics_read": False,
        })
        data_receipt = _data_preconditions(root, config)
        guard.check()
        artifact_receipt = _artifact_inventory(root, config)
        guard.check()
        record.append_event("precondition_access_verified", {
            "source_export_manifest_sha256": data_receipt["source_export_manifest_sha256"],
            "token_audit_manifest_sha256": data_receipt["token_audit_manifest_sha256"],
            "model_manifest_sha256": artifact_receipt["manifest_sha256"],
            "protected_task_semantics_read": False,
        })

    def candidate(hardware: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal phase
        phase = "step"
        result = _production_candidate(root, config, hardware, output, started, guard, artifact_receipt)
        record.append_event("probe_complete", {"outcome": result["status"],
            "elapsed_seconds_before_terminal_record_finalization": result["elapsed_seconds_before_terminal_record_finalization"],
            "checkpoint_sha256": result["checkpoint_roundtrip"]["sha256"]})
        return result

    def terminal_budget_check(stage: str) -> None:
        guard.check()
        if stage == "artifacts_complete":
            record.append_event("terminal_artifact_budget_check", {
                "elapsed_seconds_before_terminal_record_finalization": guard.elapsed,
                "process_peak_working_set_bytes_before_terminal_record_finalization": guard.peak_bytes,
                "resident_memory_cap_bytes": guard.memory_bytes,
                "wall_time_cap_seconds": guard.wall_seconds,
                "artifact_hashing_complete": True,
                "measurement": "process high-water working set, not sampled-current RSS",
            })
            guard.check()

    try:
        def hardware_preflight() -> dict[str, Any]:
            if python != config["environment"]["python_major_minor"]:
                raise RuntimeError("ENVIRONMENT_BLOCKED: Python version differs from registration")
            environment = _check_environment(config)
            hardware = _hardware_metadata(config)
            record.append_event("hardware_preflight", {"packages": environment, "hardware": hardware})
            guard.check()
            return hardware

        outcome = _controller(record_initial=initial_record, record_access=access_record,
                              query_hardware=hardware_preflight,
                              run_candidate=candidate)
        guard.check()
        record.finish("completed", counts={"PASS_FEASIBILITY": 1}, artifacts=[_OUTCOME_PATH, "state.pt"],
                      budget_check=terminal_budget_check)
        return outcome
    except Exception as exc:
        status = ("FEASIBILITY_FAILED" if isinstance(exc, (MemoryError, TimeoutError))
                  else classify_failure(exc, phase=phase))
        failure = {"schema_version": "dwr-gpu-feasibility-outcome-v1", "status": status,
                   "phase": phase, "error_type": type(exc).__name__, "error": str(exc),
                   "elapsed_seconds": time.monotonic() - started,
                   "profiler_flops": None, "task_semantics_read": False}
        try:
            outcome_path = output / _OUTCOME_PATH
            if outcome_path.exists():
                try:
                    partial = json.loads(outcome_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    partial = {}
                if partial.get("status") == "PASS_FEASIBILITY":
                    os.replace(outcome_path, output / "outcome.partial.json")
                    _write_json_exclusive(outcome_path, failure)
            else:
                _write_json_exclusive(outcome_path, failure)
            record.append_event("probe_stopped", failure)
            artifacts = [name for name in (_OUTCOME_PATH, "outcome.partial.json", "state.pt")
                         if (output / name).is_file()]
            record.finish("failed", counts={status: 1}, artifacts=artifacts, error=f"{type(exc).__name__}: {exc}")
        except Exception:
            # The original failure and any partial files remain in the unique output directory.
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_probe(args.config, command=[sys.executable, "-m", "dwr.experiments.gpu_feasibility", "--config", args.config])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
