"""Fixed-workspace evaluation utilities for the registered source pilot."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Mapping

from dwr.evaluation.generation import (
    GenerationResult,
    _generation_result,
    _greedy_loop,
    _model_vocab,
    _positive_int,
    _synchronize_device,
    _validate_check,
    _validate_generation_args,
    _validate_model_device,
    _validate_prompt,
)

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FINE = ("A2", "A3", "B2", "C2", "D2")
_PROFILES = ("eq", "neq", "add_mod")


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        if error.name == "torch":
            raise RuntimeError("source-pilot evaluation requires the optional torch dependency") from error
        raise
    return torch


def _tensor_digest(tensor) -> str:
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor digest requires a torch tensor")
    raw = tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _prompt_digest(prompt_ids, prompt_mask) -> str:
    torch = _torch()
    if not isinstance(prompt_ids, torch.Tensor) or not isinstance(prompt_mask, torch.Tensor):
        raise TypeError("prompt identity requires tensors")
    identity = {"prompt_ids": prompt_ids.detach().to(device="cpu").tolist(),
        "prompt_mask": prompt_mask.detach().to(device="cpu").tolist(),
        "prompt_dtype": str(prompt_ids.dtype), "mask_dtype": str(prompt_mask.dtype)}
    raw = __import__("json").dumps(identity, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def greedy_fixed_workspace(host, prompt_ids, prompt_mask, *, workspace,
                           eos_token_id: int, max_new_tokens: int,
                           max_total_tokens: int, check=None) -> GenerationResult:
    """Decode with an already computed workspace, without encoding or updating it."""
    torch = _torch()
    from dwr.models.qwen2 import Qwen2WorkspaceLM

    _validate_check(check)
    if check is not None:
        check()
    if not isinstance(host, Qwen2WorkspaceLM):
        raise TypeError("host must be an actual Qwen2WorkspaceLM")
    if host.training or any(module.training for module in host.modules()):
        raise ValueError("workspace host and all its modules must already be in eval mode")
    device = _validate_model_device(host, label="workspace host")
    if not isinstance(workspace, torch.Tensor):
        raise TypeError("workspace must be a tensor")
    if workspace.device != device:
        raise ValueError("workspace and host must share one device")
    parameter = next(host.workspace.parameters(), None)
    if parameter is None:
        raise ValueError("workspace module must have parameters")
    if workspace.shape != (1, host.workspace.num_slots, host.workspace.latent_dim):
        raise ValueError("workspace must match the registered single-example N by d shape")
    if workspace.dtype != parameter.dtype or not workspace.is_floating_point():
        raise ValueError("workspace dtype must match floating-point workspace parameters")
    if workspace.requires_grad or not torch.isfinite(workspace).all().item():
        raise ValueError("fixed workspace must be finite and detached")
    vocab_size = _model_vocab(host.backbone)
    eos_token_id, max_new_tokens, max_total_tokens = _validate_generation_args(
        eos_token_id, max_new_tokens, max_total_tokens, vocab_size)
    _validate_prompt(prompt_ids, prompt_mask, vocab_size=vocab_size,
        eos_token_id=eos_token_id, pad_token_id=host.backbone.config.pad_token_id,
        max_new_tokens=max_new_tokens, max_total_tokens=max_total_tokens, device=device)
    _synchronize_device(torch, device)
    started = __import__("time").perf_counter()
    try:
        with torch.no_grad():
            def decode(prefix, mask):
                return host.decode_with_workspace(prefix, mask, workspace)
            token_ids = _greedy_loop(decode, prompt_ids, prompt_mask, vocab_size=vocab_size,
                eos_token_id=eos_token_id, max_new_tokens=max_new_tokens, check=check)
    finally:
        _synchronize_device(torch, device)
    return _generation_result(token_ids, eos_token_id=eos_token_id,
        prompt_tokens=prompt_ids.shape[1], prompt_encoder_calls=0,
        workspace_updates=0, started=started)


def _noise_seed(base: int, example_id: str) -> int:
    if type(base) is not int or not 0 <= base < 2**63:
        raise ValueError("noise_seed_base must be a nonnegative 63-bit integer")
    if not isinstance(example_id, str) or _SHA.fullmatch(example_id) is None:
        raise ValueError("example_id must be a lowercase SHA-256 string")
    payload = f"dwr-source-pilot-eval-v1:{base}:{example_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def build_workspace_bank(host, batches: tuple, *, steps: int,
                         noise_seed_base: int, check=None) -> dict[str, Any]:
    """Build the prompt-only final K4 bank and deterministic cyclic ID donor map."""
    torch = _torch()
    from dwr.models.qwen2 import Qwen2WorkspaceLM
    from dwr.training.batches import TokenBatch

    _validate_check(check)
    if not isinstance(host, Qwen2WorkspaceLM):
        raise TypeError("host must be an actual Qwen2WorkspaceLM")
    if host.training or any(module.training for module in host.modules()):
        raise ValueError("workspace host and all its modules must already be in eval mode")
    if type(steps) is not int or steps != 4 or steps != host.workspace.max_steps:
        raise ValueError("the registered final workspace bank uses K4 and capacity four")
    if type(batches) is not tuple or not batches:
        raise TypeError("batches must be a nonempty tuple of single-example TokenBatch values")
    if type(noise_seed_base) is not int or not 0 <= noise_seed_base < 2**63:
        raise ValueError("noise_seed_base must be a nonnegative 63-bit integer")
    device = _validate_model_device(host, label="workspace host")
    parameter = next(host.workspace.parameters(), None)
    if parameter is None:
        raise ValueError("workspace module must have parameters")
    ids: list[str] = []
    bank_rows: list[Any] = []
    row_meta: list[dict[str, Any]] = []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for batch in batches:
            if type(batch) is not TokenBatch or len(batch.example_ids) != 1 or batch.splits != ("source_validation",):
                raise ValueError("workspace bank accepts single source_validation TokenBatch values only")
            example_id = batch.example_ids[0]
            if not isinstance(example_id, str) or _SHA.fullmatch(example_id) is None or example_id in ids:
                raise ValueError("workspace bank IDs must be unique lowercase SHA-256 values")
            prompt_ids, prompt_mask = batch.prompt_ids, batch.prompt_mask
            if prompt_ids.device != device or prompt_mask.device != device:
                raise ValueError("validation prompt tensors must share the host device")
            vocab_size = _model_vocab(host.backbone)
            _validate_prompt(prompt_ids, prompt_mask, vocab_size=vocab_size,
                eos_token_id=host.backbone.config.eos_token_id,
                pad_token_id=host.backbone.config.pad_token_id,
                max_new_tokens=128, max_total_tokens=1152, device=device)
            if check is not None:
                check()
            encoded = host.encode_prompt(prompt_ids, prompt_mask)
            if encoded.device != device or encoded.requires_grad or not torch.isfinite(encoded).all().item():
                raise ValueError("prompt representation must be finite, detached, and on the host device")
            x = encoded.to(device=parameter.device, dtype=parameter.dtype)
            seed = _noise_seed(noise_seed_base, example_id)
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn((1, host.workspace.num_slots, host.workspace.latent_dim),
                generator=generator, device=device, dtype=parameter.dtype)
            if not torch.isfinite(noise).all().item():
                raise ValueError("evaluation noise is nonfinite")
            workspace = host.workspace(x, noise=noise, prompt_padding_mask=~prompt_mask, steps=steps)
            expected_shape = (1, host.workspace.num_slots, host.workspace.latent_dim)
            if not isinstance(workspace, torch.Tensor) or workspace.shape != expected_shape or \
                    workspace.device != device or workspace.dtype != parameter.dtype or \
                    not torch.isfinite(workspace).all().item():
                raise ValueError("computed prompt-only final workspace is malformed or nonfinite")
            detached_cpu = workspace.detach().to(device="cpu").contiguous()[0].clone()
            bank_rows.append(detached_cpu)
            ids.append(example_id)
            row_meta.append({"example_id": example_id, "noise_seed": seed,
                "noise_sha256": _tensor_digest(noise),
                "prompt_sha256": _prompt_digest(prompt_ids, prompt_mask),
                "workspace_sha256": _tensor_digest(detached_cpu),
                "workspace_shape": list(detached_cpu.shape),
                "workspace_dtype": str(detached_cpu.dtype)})
    sorted_ids = sorted(ids)
    donor_map = {eid: sorted_ids[(index + 1) % len(sorted_ids)]
                 for index, eid in enumerate(sorted_ids)}
    if any(donor_map[eid] == eid for eid in ids):
        raise ValueError("workspace donor mapping must never use the example itself")
    return {"schema_version": "dwr-source-pilot-workspace-bank-v1", "steps": steps,
        "example_ids": tuple(ids), "workspace_bank": torch.stack(bank_rows, dim=0).contiguous(),
        "rows": tuple(row_meta), "donor_map": donor_map,
        "counts": {"prompt_encoder_calls": len(ids), "workspace_updates": len(ids) * steps}}


def score_pilot_stage(records: tuple[dict[str, Any], ...], *,
                      expected_ids: tuple[str, ...], metadata: Mapping[str, Any],
                      examples: tuple, include_ce: bool) -> dict[str, Any]:
    """Verify exact rows and score strict answers against their ID-matched task objects."""
    from dwr.tasks.verifier import verify_answer
    from dwr.training.batches import SourceExample

    if type(records) is not tuple or type(expected_ids) is not tuple or type(examples) is not tuple:
        raise TypeError("records, expected_ids, and examples must be tuples")
    if not expected_ids or any(not isinstance(eid, str) or _SHA.fullmatch(eid) is None for eid in expected_ids) or \
            len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected IDs must be nonempty unique lowercase SHA-256 strings")
    if type(include_ce) is not bool:
        raise TypeError("include_ce must be boolean")
    if len(records) != len(expected_ids) or len(examples) != len(expected_ids) or \
            not isinstance(metadata, Mapping) or set(metadata) != set(expected_ids):
        raise ValueError("stage records, examples, and metadata must exactly cover expected IDs")
    if any(type(example) is not SourceExample for example in examples) or \
            tuple(example.example_id for example in examples) != expected_ids:
        raise ValueError("examples must be SourceExample values in the exact registered ID order")
    by_id = dict(zip(expected_ids, examples))
    common = {"example_id", "answer_text", "valid", "verifier_reason", "termination", "generated_tokens"}
    expected_keys = common | ({"mean_ce", "supervised_tokens"} if include_ce else set())
    fine = {key: {"valid": 0, "denominator": 0} for key in _FINE}
    primitive = {key: {"valid": 0, "denominator": 0} for key in _PROFILES}
    primitive_fine = {profile: {key: {"valid": 0, "denominator": 0} for key in _FINE}
                      for profile in _PROFILES}
    total_valid = 0
    weighted_ce = 0.0
    token_denominator = 0
    for expected_id, row in zip(expected_ids, records):
        if type(row) is not dict or set(row) != expected_keys or row.get("example_id") != expected_id:
            raise ValueError("stage row has unknown/missing fields or wrong ordered example ID")
        text = row["answer_text"]
        if not isinstance(text, str) or type(row["valid"]) is not bool or \
                row["termination"] not in {"eos", "token_cap"} or \
                type(row["generated_tokens"]) is not int or not 1 <= row["generated_tokens"] <= 128 or \
                (row["termination"] == "token_cap" and row["generated_tokens"] != 128):
            raise ValueError("stage prediction/termination metadata is malformed")
        example = by_id[expected_id]
        verdict = verify_answer(example.task, text)
        valid = bool(verdict.valid and row["termination"] == "eos")
        if row["valid"] is not valid or row["verifier_reason"] != verdict.reason.value:
            raise ValueError("stage validity/reason differs from the strict ID-matched verifier")
        meta = metadata[expected_id]
        if type(meta) is not dict or not {"profile", "fine_stratum"} <= set(meta):
            raise ValueError("stage metadata is missing primitive/fine-stratum membership")
        profile, stratum = meta["profile"], meta["fine_stratum"]
        task_profiles = {constraint.kind.value for constraint in example.task.constraints}
        if profile not in _PROFILES or stratum not in _FINE or task_profiles != {profile}:
            raise ValueError("metadata profile/stratum disagrees with the ID-matched task")
        total_valid += int(valid)
        fine[stratum]["denominator"] += 1
        fine[stratum]["valid"] += int(valid)
        primitive[profile]["denominator"] += 1
        primitive[profile]["valid"] += int(valid)
        primitive_fine[profile][stratum]["denominator"] += 1
        primitive_fine[profile][stratum]["valid"] += int(valid)
        if include_ce:
            mean_ce = row["mean_ce"]
            tokens = row["supervised_tokens"]
            if isinstance(mean_ce, bool) or not isinstance(mean_ce, (int, float)) or \
                    not math.isfinite(mean_ce) or mean_ce < 0 or type(tokens) is not int or tokens <= 0:
                raise ValueError("per-example CE and supervised-token count are invalid")
            if meta.get("answer_tokens_including_eos") != tokens:
                raise ValueError("per-example CE token count differs from pinned EOS-inclusive metadata")
            weighted_ce += float(mean_ce) * tokens
            token_denominator += tokens
    if any(fine[key]["denominator"] == 0 for key in _FINE):
        raise ValueError("pilot stage must include all five requested fine strata")
    vsr = sum(fine[key]["valid"] / fine[key]["denominator"] for key in _FINE) / len(_FINE)
    for counts in primitive.values():
        counts["rate"] = counts["valid"] / counts["denominator"] if counts["denominator"] else None
    for counts in fine.values():
        counts["rate"] = counts["valid"] / counts["denominator"]
    for by_stratum in primitive_fine.values():
        for counts in by_stratum.values():
            counts["rate"] = counts["valid"] / counts["denominator"] if counts["denominator"] else None
    result = {"count": len(records), "valid_numerator": total_valid,
        "valid_denominator": len(records), "vsr": vsr, "fine_strata": fine,
        "primitive": primitive, "primitive_by_fine": primitive_fine}
    if include_ce:
        if token_denominator <= 0 or not math.isfinite(weighted_ce):
            raise ValueError("official stage CE denominator/numerator must be finite and positive")
        result["ce"] = {"weighted_numerator": weighted_ce,
            "token_denominator": token_denominator, "token_weighted_mean": weighted_ce / token_denominator}
    else:
        result["ce"] = None
    return result
