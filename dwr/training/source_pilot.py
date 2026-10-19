"""Native, arm-local learning and staged evaluation kernel for the source pilot."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import statistics
import time
import uuid
from typing import Any, Callable, Mapping

import numpy as np
import torch

from dwr.evaluation.generation import greedy_workspace, score_generation, _synchronize_device
from dwr.evaluation.source_pilot import (
    _noise_seed, _prompt_digest, _tensor_digest, build_workspace_bank,
    greedy_fixed_workspace, score_pilot_stage,
)
from dwr.models.qwen2 import Qwen2WorkspaceLM
from dwr.training.batches import SourceExample, TokenizerIdentity, collate_source_examples
from dwr.training.checkpoints import load_training_checkpoint, save_training_checkpoint
from dwr.training.controls import configure_source_control, source_control_step
from dwr.training.loss import answer_only_loss
from dwr.training.sampler import SourceSampler
from .pilot_recovery import PilotRecoveryError, PilotRecoveryStore


_HEX = re.compile(r"[0-9a-f]{64}\Z")
_FINE = ("A2", "A3", "B2", "C2", "D2")
_COUNTS = {"completed_cases", "greedy_answer_forwards", "teacher_forced_answer_forwards",
           "prompt_encodes", "workspace_updates"}
_ENVELOPE = {"schema_version", "arm_id", "stage", "step", "identities", "checkpoint_sha256",
    "example_id", "prompt_sha256", "noise_seed", "noise_sha256", "workspace_sha256",
    "donor_id", "bank_metadata_sha256", "ce_workspace_sha256", "generated_token_ids",
    "eos_token_id", "vocab_size", "reader_measurements", "score", "counts", "costs"}
_RECEIPT = {"schema_version", "arm_id", "stage", "step", "identities", "checkpoint_sha256",
    "checkpoint", "example_ids", "records", "metrics", "bank_metadata_sha256", "counts", "reader_summary"}


class PilotNumericalDivergence(ArithmeticError):
    """A numerical failure after a pilot boundary; callers must preserve it terminally."""

    def __init__(self, message: str, *, partial: dict[str, Any]):
        super().__init__(message)
        self.partial = partial


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False) + "\n").encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, raw: bytes) -> None:
    import os
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _append_canonical(path: Path, value: object) -> None:
    raw = _canonical(value)
    with path.open("ab") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _strict_json(raw: bytes, label: str) -> object:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate {label} key")
            value[key] = item
        return value
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid {label} constant {token}")))
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if _canonical(value) != raw:
        raise ValueError(f"{label} is not canonical JSON with one final LF")
    return value


def _safe_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact path must be nonempty relative POSIX text")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.as_posix() != value or any(part in {".", "..", ""} for part in relative.parts):
        raise ValueError("artifact path must be canonical and contained")
    root = root.resolve(strict=True)
    current = root
    for part in relative.parts:
        current = current / part
        info = current.lstat()
        attrs = getattr(info, "st_file_attributes", 0)
        import stat
        if stat.S_ISLNK(info.st_mode) or attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError("artifact path traverses a link or reparse point")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("artifact path escapes its arm root") from error
    if not resolved.is_file():
        raise ValueError("artifact reference must name a regular file")
    return resolved


def _reference(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"} or \
            not isinstance(value["path"], str) or not value["path"] or \
            not isinstance(value["sha256"], str) or not _HEX.fullmatch(value["sha256"]):
        raise ValueError(f"{label} must contain exactly a path and SHA-256")
    return value


def _number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{label} must be a finite number at least {minimum}")
    return float(value)


def _count_map(value: object, expected: set[str], label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != expected or any(type(item) is not int or item < 0 for item in value.values()):
        raise ValueError(f"{label} must contain exactly the registered nonnegative integer counters")
    return value


def _memory_measurement() -> dict[str, Any]:
    """Measure process high-water memory with stdlib-only platform APIs."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD)
        get_memory_info.restype = wintypes.BOOL
        handle = get_current_process()
        if not get_memory_info(handle, ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return {"scope": "process_peak_working_set_bytes", "value_bytes": int(counters.PeakWorkingSetSize)}
    try:
        import resource
    except ImportError as error:
        raise RuntimeError("pilot costs require a measurable process memory high-water mark") from error
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if os.uname().sysname == "Darwin" else 1024
    return {"scope": "process_peak_rss_bytes", "value_bytes": int(peak * scale)}


def _validate_stage(arm_id: str, stage: str, expected_step: int, expected_ids: tuple[str, ...],
                    expected_bank_metadata_sha256: str | None) -> bool:
    if arm_id not in {"K0", "K1", "K4"} or stage not in {"initial", "final", "zero", "noise", "shuffle"}:
        raise ValueError("pilot arm or stage is unknown")
    if type(expected_step) is not int or expected_step < 0 or (stage == "initial") != (expected_step == 0):
        raise ValueError("stage and expected optimizer step are incompatible")
    if type(expected_ids) is not tuple or not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected stage IDs must be a nonempty unique tuple")
    diagnostic = stage in {"zero", "noise", "shuffle"}
    if diagnostic and arm_id != "K4":
        raise ValueError("only K4 owns registered workspace diagnostic stages")
    needs_bank = arm_id == "K4" and stage in {"final", "zero", "noise", "shuffle"}
    if needs_bank != (expected_bank_metadata_sha256 is not None):
        raise ValueError("stage bank identity does not match the registered method/stage")
    if expected_bank_metadata_sha256 is not None and not _HEX.fullmatch(expected_bank_metadata_sha256):
        raise ValueError("expected bank metadata identity must be SHA-256")
    return stage in {"initial", "final"}


def verify_pilot_stage_receipt(jobroot, reference, *, arm_id, stage, expected_step,
        expected_ids, expected_checkpoint_sha256, identities, metadata, examples,
        expected_bank_metadata_sha256=None, expected_bank_reference=None) -> dict:
    """Read and semantically recompute one canonical, arm-local stage receipt."""
    include_ce = _validate_stage(arm_id, stage, expected_step, expected_ids,
        expected_bank_metadata_sha256)
    if not isinstance(expected_checkpoint_sha256, str) or not _HEX.fullmatch(expected_checkpoint_sha256):
        raise ValueError("expected checkpoint digest must be lowercase SHA-256")
    if not isinstance(identities, dict) or set(identities) != {"source", "config", "data", "model", "tokenizer", "schedule"} or \
            any(not isinstance(value, str) or not _HEX.fullmatch(value) for value in identities.values()):
        raise ValueError("stage identity map must contain exactly six SHA-256 values")
    if not isinstance(metadata, Mapping) or set(metadata) != set(expected_ids) or type(examples) is not tuple or \
            tuple(example.example_id for example in examples) != expected_ids:
        raise ValueError("stage examples and metadata must exactly match ordered expected IDs")
    reference = _reference(reference, "stage receipt reference")
    root = Path(jobroot).resolve(strict=True)
    receipt_path = _safe_path(root, reference["path"])
    receipt_raw = receipt_path.read_bytes()
    if _sha_bytes(receipt_raw) != reference["sha256"]:
        raise ValueError("stage receipt bytes do not match their reference")
    receipt = _strict_json(receipt_raw, "pilot stage receipt")
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT or \
            receipt.get("schema_version") != "dwr-source-pilot-stage-v1" or \
            receipt.get("arm_id") != arm_id or receipt.get("stage") != stage or \
            receipt.get("step") != expected_step or receipt.get("identities") != identities or \
            receipt.get("checkpoint_sha256") != expected_checkpoint_sha256 or \
            receipt.get("example_ids") != list(expected_ids) or receipt.get("bank_metadata_sha256") != expected_bank_metadata_sha256:
        raise ValueError("stage receipt method, checkpoint, identity or ordered IDs differ")
    checkpoint_ref = _reference(receipt["checkpoint"], "stage checkpoint reference")
    checkpoint_path = _safe_path(root, checkpoint_ref["path"])
    if checkpoint_ref["sha256"] != expected_checkpoint_sha256 or _hash_file(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("stage checkpoint reference bytes differ from its registered SHA")
    from .pilot_recovery import _checkpoint_header
    expected_progress = {"optimizer_step": expected_step, "examples_seen": expected_step,
        "supervised_tokens_seen": None}
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_progress = payload.get("progress") if isinstance(payload, dict) else None
        if not isinstance(checkpoint_progress, dict) or checkpoint_progress.get("optimizer_step") != expected_step or \
                checkpoint_progress.get("examples_seen") != expected_step or \
                type(checkpoint_progress.get("supervised_tokens_seen")) is not int or \
                checkpoint_progress["supervised_tokens_seen"] < 0 or \
                (expected_step == 0) != (checkpoint_progress["supervised_tokens_seen"] == 0):
            raise ValueError("checkpoint optimizer progress differs from stage")
        expected_progress = checkpoint_progress
        _checkpoint_header(checkpoint_path, expected_checkpoint_sha256, identities,
            expected_progress, arm_id, max(expected_step, 1))
    except Exception as error:
        raise ValueError("stage checkpoint failed complete-checkpoint verification") from error
    bank_metadata = bank_tensor = None
    checkpoint_architecture = payload.get("architecture")
    expected_reader_ids = {str(index) for index in checkpoint_architecture.get("reader_indices", ())} \
        if isinstance(checkpoint_architecture, dict) else set()
    checkpoint_reader_gates = {}
    for reader_id in expected_reader_ids:
        gate = payload.get("readers", {}).get(f"{reader_id}.gate")
        if not isinstance(gate, torch.Tensor) or gate.numel() != 1:
            raise ValueError("checkpoint reader gate state is unavailable for measurement binding")
        checkpoint_reader_gates[reader_id] = float(gate.detach().cpu().item())
    if expected_bank_metadata_sha256 is not None:
        bank_reference = _reference(expected_bank_reference, "expected bank metadata reference")
        if bank_reference["sha256"] != expected_bank_metadata_sha256:
            raise ValueError("bank reference and expected bank metadata digest differ")
        from .pilot_recovery import _verify_bank_metadata
        try:
            bank_metadata = _verify_bank_metadata(root, bank_reference, arm_id="K4",
                identities=identities, checkpoint_sha256=expected_checkpoint_sha256,
                selected_ids=expected_ids, steps=expected_step, checkpoint_path=checkpoint_path)
            bank_path = _safe_path(root, bank_metadata["tensor"]["path"])
            bank_tensor = torch.load(bank_path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError("stage workspace bank failed checkpoint binding") from error
    records_ref = _reference(receipt["records"], "stage records reference")
    records_path = _safe_path(root, records_ref["path"])
    records_raw = records_path.read_bytes()
    if _sha_bytes(records_raw) != records_ref["sha256"]:
        raise ValueError("stage prediction bytes do not match the receipt")
    if not records_raw.endswith(b"\n") or records_raw.endswith(b"\n\n"):
        raise ValueError("stage JSONL must end in exactly one LF per canonical envelope")
    raw_lines = records_raw.splitlines(keepends=True)
    if len(raw_lines) != len(expected_ids):
        raise ValueError("stage prediction rows do not cover the exact registered IDs")
    scores = []
    observed_counts = {key: 0 for key in ("completed_cases", "greedy_answer_forwards",
        "teacher_forced_answer_forwards", "prompt_encodes", "workspace_updates")}
    bank_digest = expected_bank_metadata_sha256
    for eid, line in zip(expected_ids, raw_lines):
        envelope = _strict_json(line, "pilot prediction envelope")
        if not isinstance(envelope, dict) or set(envelope) != _ENVELOPE or \
                envelope.get("schema_version") != "dwr-source-pilot-prediction-v1" or \
                envelope.get("arm_id") != arm_id or envelope.get("stage") != stage or \
                envelope.get("step") != expected_step or envelope.get("identities") != identities or \
                envelope.get("checkpoint_sha256") != expected_checkpoint_sha256 or envelope.get("example_id") != eid or \
                envelope.get("bank_metadata_sha256") != bank_digest:
            raise ValueError("stage prediction envelope has invalid binding or schema")
        for name in ("prompt_sha256", "noise_sha256", "workspace_sha256"):
            if not isinstance(envelope[name], str) or not _HEX.fullmatch(envelope[name]):
                raise ValueError(f"stage prediction {name} is not SHA-256")
        if type(envelope["noise_seed"]) is not int or not 0 <= envelope["noise_seed"] < 2**63:
            raise ValueError("stage prediction noise seed is invalid")
        if bank_metadata is not None:
            bank_row_index = bank_metadata["example_ids"].index(eid)
            bank_row = bank_metadata["rows"][bank_row_index]
            if envelope["noise_seed"] != bank_row["noise_seed"] or \
                    envelope["noise_sha256"] != bank_row["noise_sha256"] or \
                    envelope["prompt_sha256"] != bank_row["prompt_sha256"]:
                raise ValueError("K4 stage prompt/noise identities differ from the committed bank row")
            if stage == "final":
                expected_donor = None
                expected_workspace = _tensor_digest(bank_tensor[bank_row_index:bank_row_index + 1])
            elif stage == "zero":
                expected_donor = None
                expected_workspace = _tensor_digest(torch.zeros_like(bank_tensor[bank_row_index:bank_row_index + 1]))
            elif stage == "noise":
                expected_donor = None
                expected_workspace = bank_row["noise_sha256"]
            else:
                expected_donor = bank_metadata["donor_map"][eid]
                donor_index = bank_metadata["example_ids"].index(expected_donor)
                expected_workspace = _tensor_digest(bank_tensor[donor_index:donor_index + 1])
            if envelope["donor_id"] != expected_donor or envelope["workspace_sha256"] != expected_workspace:
                raise ValueError("K4 stage workspace/donor identity differs from its registered intervention")
        if stage == "shuffle":
            if not isinstance(envelope["donor_id"], str) or envelope["donor_id"] == eid:
                raise ValueError("shuffle stage must bind a different example's donor ID")
        elif envelope["donor_id"] is not None:
            raise ValueError("only shuffle predictions may name a donor")
        if include_ce:
            if not isinstance(envelope["ce_workspace_sha256"], str) or not _HEX.fullmatch(envelope["ce_workspace_sha256"]):
                raise ValueError("official prediction must bind its independently computed CE workspace")
        elif envelope["ce_workspace_sha256"] is not None:
            raise ValueError("greedy-only diagnostic cannot contain a CE workspace identity")
        score = envelope["score"]
        score_keys = {"example_id", "answer_text", "valid", "verifier_reason", "termination", "generated_tokens"}
        if include_ce:
            score_keys |= {"mean_ce", "supervised_tokens"}
        if not isinstance(score, dict) or set(score) != score_keys or score.get("example_id") != eid:
            raise ValueError("stage score row has unknown/missing keys or wrong ID")
        generated = score.get("generated_tokens")
        termination = score.get("termination")
        if type(generated) is not int or generated < 1 or generated > 128 or \
                (termination == "token_cap" and generated != 128) or termination not in {"eos", "token_cap"}:
            raise ValueError("generated token count does not match the registered 128-token termination policy")
        if termination == "token_cap" and score.get("valid") is not False:
            raise ValueError("a capped generation cannot be a strict-valid prediction")
        token_ids = envelope["generated_token_ids"]
        eos_id, vocab_size = envelope["eos_token_id"], envelope["vocab_size"]
        if type(eos_id) is not int or type(vocab_size) is not int or vocab_size <= 0 or not 0 <= eos_id < vocab_size or \
                type(token_ids) is not list or len(token_ids) != generated or any(
                type(token) is not int or token < 0 or token >= vocab_size for token in token_ids) or \
                (eos_id in token_ids[:-1]) or (termination == "eos") != (token_ids[-1] == eos_id) or \
                (termination == "token_cap" and generated != 128):
            raise ValueError("raw generated token IDs violate the pinned vocabulary/EOS/cap contract")
        reader_measurements = envelope["reader_measurements"]
        if arm_id == "K4" and stage == "final":
            if not isinstance(reader_measurements, dict) or set(reader_measurements) != expected_reader_ids:
                raise ValueError("K4 final rows require actual reader measurements")
            for reader_id, row in reader_measurements.items():
                fields = {"sample_count", "query_norm_sum", "query_norm_max", "residual_norm_sum",
                    "residual_norm_max", "ratio_sum", "ratio_max", "gate"}
                if not isinstance(reader_id, str) or not reader_id or not isinstance(row, dict) or set(row) != fields or \
                        type(row["sample_count"]) is not int or row["sample_count"] != generated:
                    raise ValueError("K4 final reader measurements have invalid IDs or sample counts")
                for field in fields - {"sample_count"}:
                    _number(row[field], f"reader measurement {field}", minimum=-math.inf if field == "gate" else 0.0)
                if row["gate"] != checkpoint_reader_gates[reader_id]:
                    raise ValueError("reader measurement gate differs from the evaluated checkpoint")
        elif reader_measurements is not None:
            raise ValueError("reader measurements are permitted only on K4 final rows")
        counts = _count_map(envelope["counts"], {"prompt_encodes", "workspace_updates",
            "greedy_answer_forwards", "teacher_forced_answer_forwards"}, "prediction counts")
        expected_teacher = int(include_ce)
        if counts["teacher_forced_answer_forwards"] != expected_teacher or \
                counts["greedy_answer_forwards"] != generated:
            raise ValueError("per-prediction answer-forward counts differ from scored calls")
        method_steps = {"K0": 0, "K1": 1, "K4": 4}[arm_id]
        if include_ce:
            expected_prompt = 1 if arm_id == "K4" and stage == "final" else 2
            expected_updates = method_steps if arm_id == "K4" and stage == "final" else 2 * method_steps
            if counts["prompt_encodes"] != expected_prompt or counts["workspace_updates"] != expected_updates:
                raise ValueError("official prediction prompt/workspace counts differ from the arm method")
        if stage in {"zero", "noise", "shuffle"} and (counts["prompt_encodes"] != 0 or counts["workspace_updates"] != 0):
            raise ValueError("fixed-state diagnostics cannot repeat prompt encoding or workspace updates")
        if stage == "final" and arm_id == "K4" and (counts["prompt_encodes"] != 1 or counts["workspace_updates"] <= 0):
            raise ValueError("K4 final CE must count only its independent CE prompt/workspace computation")
        if not include_ce and (score.get("mean_ce") is not None or score.get("supervised_tokens") is not None):
            raise ValueError("diagnostic score unexpectedly contains CE")
        costs = envelope["costs"]
        expected_elapsed_scope = "full_case_including_reader_measurement_hooks" if arm_id == "K4" and stage == "final" else "full_case"
        if not isinstance(costs, dict) or set(costs) != {"elapsed_seconds", "elapsed_scope", "memory"} or \
                costs.get("elapsed_scope") != expected_elapsed_scope:
            raise ValueError("prediction costs must contain elapsed time and scoped memory")
        _number(costs["elapsed_seconds"], "prediction elapsed time")
        memory = costs["memory"]
        if not isinstance(memory, dict) or set(memory) != {"scope", "value_bytes"} or \
                not isinstance(memory["scope"], str) or not memory["scope"] or \
                type(memory["value_bytes"]) is not int or memory["value_bytes"] < 0:
            raise ValueError("prediction memory cost lacks a measured scope/value")
        for key, value in counts.items():
            observed_counts[key] += value
        observed_counts["completed_cases"] += 1
        scores.append(score)
    metrics = score_pilot_stage(tuple(scores), expected_ids=expected_ids, metadata=metadata,
        examples=examples, include_ce=include_ce)
    expected_counts = receipt["counts"]
    _count_map(expected_counts, _COUNTS, "stage receipt counts")
    if expected_counts != observed_counts:
        raise ValueError("stage aggregate physical counts differ from its prediction envelopes")
    if receipt["metrics"] != metrics:
        raise ValueError("stage receipt metrics differ from semantic scorer recomputation")
    reader_summary = receipt["reader_summary"]
    expected_reader_summary = _combine_reader_measurements([
        _strict_json(line, "pilot prediction envelope") for line in raw_lines
    ]) if arm_id == "K4" and stage == "final" else None
    if reader_summary != expected_reader_summary:
        raise ValueError("stage reader summary differs from recomputed row measurements")
    return {**receipt, "metrics": metrics, "records_path": records_path,
        "receipt_path": receipt_path, "records_sha256": records_ref["sha256"],
        "receipt_sha256": reference["sha256"]}


def _value_digest(value: object) -> str:
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"T" + str(tensor.dtype).encode("ascii") + repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"A" + item.dtype.str.encode("ascii") + repr(item.shape).encode("ascii"))
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item):
                add(key); add(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(b"[")
            for child in item:
                add(child)
            digest.update(b"]")
        elif item is None:
            digest.update(b"N")
        elif isinstance(item, (str, int, float, bool)):
            digest.update(type(item).__name__.encode("ascii") + b":" + repr(item).encode("utf-8"))
        else:
            raise TypeError(f"unsupported audit value type {type(item).__name__}")
    add(value)
    return digest.hexdigest()


def _clone_modules(host):
    return tuple({name: value.detach().to(device="cpu").clone()
                  for name, value in module.state_dict().items()}
                 for module in (host.workspace, host.readers))


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    return copy.deepcopy(value)


def _global_rng():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].copy(), *state[2:]),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else ()}


def _state_snapshot(host, optimizer, scheduler, sampler, noise_generator):
    return {"modules": _clone_modules(host), "optimizer": _cpu_snapshot(optimizer.state_dict()),
        "scheduler": _cpu_snapshot(scheduler.state_dict()), "sampler": _cpu_snapshot(sampler.state_dict()),
        "noise": noise_generator.get_state().clone(), "global_rng": _global_rng(),
        "modes": tuple((id(module), module.training) for module in host.modules()),
        "config_return_dict": host.backbone.config.return_dict}


def _restore_modes(host, modes, return_dict):
    for module, mode in modes.items():
        module.training = mode
    host.backbone.config.return_dict = return_dict


def _assert_state_preserved(before, host, optimizer, scheduler, sampler, noise_generator):
    after = _state_snapshot(host, optimizer, scheduler, sampler, noise_generator)
    if _value_digest(before) != _value_digest(after):
        raise RuntimeError("pilot evaluation or audit callback changed live training state")


def _assert_frozen_original(host):
    for name, parameter in host.backbone.named_parameters():
        if parameter.requires_grad or parameter.grad is not None:
            raise RuntimeError(f"frozen original parameter boundary failed for {name}")


def _stable_module_digest(host):
    from dwr.training.overfit import _initial_state_digest
    return _initial_state_digest(host.workspace, host.readers)


def _example_maps(examples, metadata, *, split):
    if type(examples) is not tuple or not examples:
        raise TypeError(f"{split} examples must be a nonempty tuple")
    ids = tuple(example.example_id for example in examples)
    if len(set(ids)) != len(ids) or any(type(example) is not SourceExample or example.split != split for example in examples):
        raise ValueError(f"{split} examples must be unique strict SourceExample values for that split")
    if not isinstance(metadata, dict) or set(metadata) != set(ids):
        raise ValueError(f"{split} metadata must exactly match its examples")
    return ids, dict(zip(ids, examples))


def _collate(example, tokenizer, tokenizer_identity, config, device):
    return collate_source_examples((example,), tokenizer, tokenizer_identity=tokenizer_identity,
        vocab_size=config["decoding"]["embedding_vocab_size"],
        max_prompt_tokens=config["decoding"]["max_prompt_tokens"],
        max_answer_tokens=config["decoding"]["max_answer_tokens_including_eos"],
        max_total_tokens=config["decoding"]["max_total_tokens"], device=device)


def _module_digest_states(host):
    digest = hashlib.sha256()
    for module_name, module in (("readers", host.readers), ("workspace", host.workspace)):
        for name, tensor in sorted(module.state_dict().items()):
            raw_name = f"{module_name}.{name}".encode("utf-8")
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(raw_name).to_bytes(8, "big") + raw_name)
            digest.update(len(raw).to_bytes(8, "big") + raw)
    return digest.hexdigest()


def _init_recipe():
    return {"name": "dwr-source-overfit-initial-state-v1", "ordering": "module then sorted state name",
        "framing": "big-endian uint64 byte lengths for UTF-8 name, JSON shape, ASCII dtype, and tensor bytes",
        "bytes": "contiguous CPU tensor viewed as uint8"}


def _save_boundary(host, optimizer, scheduler, sampler, noise_generator, identities, progress,
                   path: Path):
    return save_training_checkpoint(path, host=host, optimizer=optimizer, scheduler=scheduler,
        scaler=None, sampler=sampler, noise_generator=noise_generator, identities=identities,
        progress=progress)


def _emit(on_event, kind, payload):
    if not callable(on_event):
        raise TypeError("on_event must be callable and durably append/fsync before returning")
    on_event(kind, payload)


def _device_memory_context(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _norm(v):
    value = float(torch.linalg.vector_norm(v.detach().to(torch.float64)).item())
    if not math.isfinite(value):
        raise ValueError("reader query instrumentation observed a nonfinite norm")
    return value


def _reader_measurement_hooks(host):
    for reader in host.readers.values():
        for module in reader.modules():
            if module._forward_hooks or module._forward_pre_hooks:
                raise ValueError("foreign reader hook is unsupported during registered diagnostics")
    accum = {name: {"sample_count": 0, "query_norm_sum": 0.0, "query_norm_max": 0.0,
        "residual_norm_sum": 0.0, "residual_norm_max": 0.0, "ratio_sum": 0.0,
        "ratio_max": 0.0, "gate": float(reader.gate.detach().cpu().item())}
        for name, reader in host.readers.items()}
    handles = []
    for name, reader in host.readers.items():
        def hook(_module, inputs, output, *, _name=name):
            q, o = inputs[0], output
            if not isinstance(q, torch.Tensor) or not isinstance(o, torch.Tensor) or q.ndim != 3 or o.shape != q.shape:
                raise ValueError("reader instrumentation observed malformed query/output tensors")
            query, result = q[:, -1, :].detach(), o[:, -1, :].detach()
            delta = result - query
            q_norm, delta_norm = _norm(query), _norm(delta)
            ratio = delta_norm / max(q_norm, 1e-12)
            if not math.isfinite(ratio):
                raise ValueError("reader instrumentation observed a nonfinite ratio")
            row = accum[_name]
            row["sample_count"] += 1
            row["query_norm_sum"] += q_norm
            row["query_norm_max"] = max(row["query_norm_max"], q_norm)
            row["residual_norm_sum"] += delta_norm
            row["residual_norm_max"] = max(row["residual_norm_max"], delta_norm)
            row["ratio_sum"] += ratio
            row["ratio_max"] = max(row["ratio_max"], ratio)
        handles.append(reader.register_forward_hook(hook))
    return accum, handles


def _finish_reader_metrics(values):
    result = {}
    for name, row in values.items():
        count = row["sample_count"]
        if count <= 0:
            raise ValueError("reader measurement stage produced no query samples")
        result[name] = dict(row)
    return result


def _combine_reader_measurements(envelopes):
    if not envelopes:
        raise ValueError("reader summary requires complete row measurements")
    names = set(envelopes[0]["reader_measurements"] or {})
    if not names:
        raise ValueError("reader summary cannot be built from missing measurements")
    summary = {}
    for name in sorted(names):
        rows = [envelope["reader_measurements"].get(name) for envelope in envelopes]
        if any(not isinstance(row, dict) or set(row) != {"sample_count", "query_norm_sum",
                "query_norm_max", "residual_norm_sum", "residual_norm_max", "ratio_sum", "ratio_max", "gate"}
                for row in rows):
            raise ValueError("reader row measurement schema is invalid")
        gates = {row["gate"] for row in rows}
        if len(gates) != 1:
            raise ValueError("reader gate changed across final-stage examples")
        summary[name] = {"sample_count": sum(row["sample_count"] for row in rows),
            "query_norm_sum": sum(row["query_norm_sum"] for row in rows),
            "query_norm_max": max(row["query_norm_max"] for row in rows),
            "residual_norm_sum": sum(row["residual_norm_sum"] for row in rows),
            "residual_norm_max": max(row["residual_norm_max"] for row in rows),
            "ratio_sum": sum(row["ratio_sum"] for row in rows),
            "ratio_max": max(row["ratio_max"] for row in rows), "gate": next(iter(gates))}
    return summary


def _greedy_with_reader_measurements(host, enabled, *args, **kwargs):
    if not enabled:
        return greedy_fixed_workspace(host, *args, **kwargs), None
    values, handles = _reader_measurement_hooks(host)
    try:
        result = greedy_fixed_workspace(host, *args, **kwargs)
        measurements = _finish_reader_metrics(values)
        if any(row["sample_count"] != result.decoder_calls for row in measurements.values()):
            raise ValueError("reader query observations differ from actual greedy answer forwards")
        return result, measurements
    finally:
        for handle in handles:
            handle.remove()


def _stage_receipt_paths(segment: Path, stage: str):
    directory = segment / "stages"
    directory.mkdir(exist_ok=True)
    nonce = uuid.uuid4().hex
    return directory / f"{stage}-{nonce}.jsonl", directory / f"{stage}-{nonce}.receipt.json"


def _write_bank(jobroot: Path, segment: Path, bank: dict, *, identities: dict,
                checkpoint_sha256: str) -> tuple[dict[str, str], dict[str, Any]]:
    directory = segment / "bank"
    directory.mkdir(exist_ok=True)
    nonce = uuid.uuid4().hex
    tensor_path = directory / f"workspace-bank-{nonce}.pt"
    tensor = bank["workspace_bank"].detach().to(device="cpu").contiguous()
    if not torch.isfinite(tensor).all().item():
        raise ValueError("pilot workspace bank is nonfinite")
    with tensor_path.open("xb") as stream:
        torch.save(tensor, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tensor_ref = {"path": tensor_path.relative_to(jobroot).as_posix(), "sha256": _hash_file(tensor_path)}
    metadata = {"schema_version": "dwr-source-pilot-bank-metadata-v1", "arm_id": "K4", "step": 512,
        "identities": identities, "checkpoint_sha256": checkpoint_sha256,
        "example_ids": list(bank["example_ids"]), "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "tensor": tensor_ref, "rows": list(bank["rows"]), "donor_map": bank["donor_map"],
        "counts": bank["counts"]}
    # Fixture cursors carry their explicit final step; production is fixed at 512.
    metadata["step"] = int(bank.get("step", 512))
    raw = _canonical(metadata)
    metadata_path = directory / f"workspace-bank-{nonce}.metadata.json"
    _write_exclusive(metadata_path, raw)
    return {"path": metadata_path.relative_to(jobroot).as_posix(), "sha256": _sha_bytes(raw)}, metadata


def _load_bank(jobroot: Path, metadata_ref: dict[str, str]):
    metadata_path = _safe_path(jobroot, metadata_ref["path"])
    raw = metadata_path.read_bytes()
    if _sha_bytes(raw) != metadata_ref["sha256"]:
        raise ValueError("committed bank metadata bytes changed")
    metadata = _strict_json(raw, "pilot bank metadata")
    tensor_path = _safe_path(jobroot, metadata["tensor"]["path"])
    if _hash_file(tensor_path) != metadata["tensor"]["sha256"]:
        raise ValueError("committed bank tensor bytes changed")
    tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or list(tensor.shape) != metadata["shape"] or \
            str(tensor.dtype) != metadata["dtype"] or not tensor.is_contiguous() or not torch.isfinite(tensor).all().item():
        raise ValueError("committed bank tensor failed shape/dtype/finiteness checks")
    return metadata, tensor


def _persist_stage(jobroot: Path, segment: Path, *, arm_id: str, stage: str, step: int,
        identities: dict, checkpoint_ref: dict[str, str], expected_ids: tuple[str, ...],
        envelopes: list[dict], metrics: dict, bank_metadata_sha256: str | None,
        counts: dict) -> tuple[dict[str, str], dict]:
    checkpoint_sha256 = checkpoint_ref["sha256"]
    records_path, receipt_path = _stage_receipt_paths(segment, stage)
    with records_path.open("xb") as stream:
        for envelope in envelopes:
            stream.write(_canonical(envelope))
            stream.flush()
            os.fsync(stream.fileno())
    records_ref = {"path": records_path.relative_to(jobroot).as_posix(), "sha256": _hash_file(records_path)}
    receipt = {"schema_version": "dwr-source-pilot-stage-v1", "arm_id": arm_id,
        "stage": stage, "step": step, "identities": identities,
        "checkpoint_sha256": checkpoint_sha256, "checkpoint": checkpoint_ref,
        "example_ids": list(expected_ids),
        "records": records_ref, "metrics": metrics,
        "bank_metadata_sha256": bank_metadata_sha256, "counts": counts,
        "reader_summary": _combine_reader_measurements(envelopes) if arm_id == "K4" and stage == "final" else None}
    raw = _canonical(receipt)
    _write_exclusive(receipt_path, raw)
    reference = {"path": receipt_path.relative_to(jobroot).as_posix(), "sha256": _sha_bytes(raw)}
    return reference, receipt


def _batch_noise(host, batch, generator, device):
    parameter = next(host.workspace.parameters())
    return host.workspace.sample_noise(len(batch.example_ids), generator=generator,
        device=device, dtype=parameter.dtype)


def _evaluation_snapshot(host, optimizer, scheduler, sampler, noise_generator):
    modes = {module: module.training for module in host.modules()}
    return _state_snapshot(host, optimizer, scheduler, sampler, noise_generator), modes, host.backbone.config.return_dict


def _run_stage_preserved(host, tokenizer, tokenizer_identity, examples, metadata, *,
        optimizer, scheduler, sampler, noise_generator, **stage_kwargs):
    before, modes, return_dict = _evaluation_snapshot(host, optimizer, scheduler, sampler, noise_generator)
    try:
        host.eval()
        host.backbone.config.return_dict = True
        return _run_stage(host, tokenizer, tokenizer_identity, examples, metadata, **stage_kwargs)
    finally:
        _restore_modes(host, modes, return_dict)
        _assert_state_preserved(before, host, optimizer, scheduler, sampler, noise_generator)


def _run_stage(host, tokenizer, tokenizer_identity, examples, metadata, *, config,
        arm_id, stage, step, identities, checkpoint_ref, jobroot, segment, check,
        on_event, bank_ref=None, bank_metadata=None, bank_tensor=None):
    from dwr.evaluation.generation import _validate_check
    _validate_check(check)
    expected_ids = tuple(example.example_id for example in examples)
    include_ce = stage in {"initial", "final"}
    diagnostic = stage in {"zero", "noise", "shuffle"}
    checkpoint_sha256 = checkpoint_ref["sha256"]
    bank_digest = None if bank_ref is None else bank_ref["sha256"]
    if stage in {"final", "zero", "noise", "shuffle"} and arm_id == "K4" and bank_tensor is None:
        raise ValueError("K4 final/diagnostic evaluation requires the committed tensor bank")
    prediction_path, _ = _stage_receipt_paths(segment, stage)
    # Rename the generated stage receipt path so envelope and receipt share a nonce.
    nonce = prediction_path.stem.split("-")[-1]
    receipt_path = prediction_path.with_name(f"{stage}-{nonce}.receipt.json")
    envelopes = []
    scores = []
    totals = {key: 0 for key in _COUNTS}
    reader_summary = None
    attempt_created = False
    for index, example in enumerate(examples):
        _emit(on_event, "evaluation_case_started", {"arm_id": arm_id, "stage": stage,
            "step": step, "example_id": example.example_id, "attempt": nonce,
            "include_ce": include_ce})
        if check is not None:
            check()
        started = time.perf_counter()
        batch = _collate(example, tokenizer, tokenizer_identity, config, next(host.parameters()).device)
        eval_seed = _noise_seed(config["seeds"]["evaluation_noise_base"], example.example_id)
        device = next(host.workspace.parameters()).device
        generator = torch.Generator(device=device).manual_seed(eval_seed)
        noise = _batch_noise(host, batch, generator, device)
        prompt_hash = _prompt_digest(batch.prompt_ids, batch.prompt_mask)
        noise_hash = _tensor_digest(noise)
        ce_workspace_hash = None
        per_counts = {"prompt_encodes": 0, "workspace_updates": 0,
            "greedy_answer_forwards": 0, "teacher_forced_answer_forwards": 0}
        if include_ce:
            with torch.no_grad():
                output = host(batch.prompt_ids, batch.prompt_mask, batch.decoder_ids,
                    batch.decoder_mask, noise=noise, steps={"K0": 0, "K1": 1, "K4": 4}[arm_id])
                ce = answer_only_loss(output.logits, batch.labels)
            if not torch.isfinite(ce.loss).item() or ce.supervised_tokens != metadata[example.example_id]["answer_tokens_including_eos"]:
                raise ValueError("official CE is nonfinite or differs from pinned EOS-inclusive token metadata")
            ce_workspace_hash = _tensor_digest(output.workspace)
            per_counts["prompt_encodes"] += 1
            per_counts["workspace_updates"] += {"K0": 0, "K1": 1, "K4": 4}[arm_id]
            per_counts["teacher_forced_answer_forwards"] = 1
            _emit(on_event, "teacher_forced_completed", {"arm_id": arm_id, "stage": stage,
                "step": step, "example_id": example.example_id, "supervised_tokens": ce.supervised_tokens,
                "loss": float(ce.loss.item())})
        donor_id = None
        if diagnostic:
            if stage == "zero":
                workspace = torch.zeros_like(bank_tensor[index:index + 1])
            elif stage == "noise":
                workspace = noise.to(device="cpu").clone()
            else:
                donor_id = bank_metadata["donor_map"][example.example_id]
                donor_index = bank_metadata["example_ids"].index(donor_id)
                workspace = bank_tensor[donor_index:donor_index + 1]
            workspace = workspace.to(device=device, dtype=next(host.workspace.parameters()).dtype)
            workspace_hash = _tensor_digest(workspace)
            row_noise_hash = bank_metadata["rows"][index]["noise_sha256"]
            if stage == "noise" and workspace_hash != row_noise_hash:
                raise RuntimeError("deterministic Z0 reconstruction differs from persisted bank noise identity")
            row_noise_hash = bank_metadata["rows"][index]["noise_sha256"]
            row_prompt_hash = bank_metadata["rows"][index]["prompt_sha256"]
            if row_prompt_hash != prompt_hash:
                raise ValueError("K4 diagnostic prompt differs from persisted bank identity")
            gen_result, row_reader_measurements = _greedy_with_reader_measurements(host,
                arm_id == "K4" and stage == "final", batch.prompt_ids, batch.prompt_mask,
                workspace=workspace, eos_token_id=config["decoding"]["eos_token_id"],
                max_new_tokens=128, max_total_tokens=config["decoding"]["max_total_tokens"], check=check)
        else:
            if arm_id == "K4" and stage == "final":
                bank_index = index
                workspace_cpu = bank_tensor[bank_index:bank_index + 1]
                bank_row = bank_metadata["rows"][index]
                if bank_row["example_id"] != example.example_id or bank_row["prompt_sha256"] != prompt_hash:
                    raise ValueError("K4 final prompt/order differs from the committed workspace bank")
                workspace = workspace_cpu.to(device=device, dtype=next(host.workspace.parameters()).dtype)
                workspace_hash = _tensor_digest(workspace)
                row_noise_hash = bank_row["noise_sha256"]
                row_prompt_hash = bank_row["prompt_sha256"]
                gen_result, row_reader_measurements = _greedy_with_reader_measurements(host,
                    True, batch.prompt_ids, batch.prompt_mask, workspace=workspace,
                    eos_token_id=config["decoding"]["eos_token_id"], max_new_tokens=128,
                    max_total_tokens=config["decoding"]["max_total_tokens"], check=check)
            else:
                # Recreate the same private per-ID draw for CE and generation.
                gen = torch.Generator(device=device).manual_seed(eval_seed)
                gen_noise = _batch_noise(host, batch, gen, device)
                noise_hash_gen = _tensor_digest(gen_noise)
                if noise_hash_gen != noise_hash:
                    raise RuntimeError("official CE and generation noise identities differ")
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                    x_original = host.encode_prompt(batch.prompt_ids, batch.prompt_mask)
                    parameter = next(host.workspace.parameters())
                    x = x_original.to(device=parameter.device, dtype=parameter.dtype)
                    workspace = host.workspace(x, noise=gen_noise,
                        prompt_padding_mask=~batch.prompt_mask, steps={"K0": 0, "K1": 1, "K4": 4}[arm_id])
                workspace_hash = _tensor_digest(workspace)
                row_noise_hash, row_prompt_hash = noise_hash_gen, prompt_hash
                per_counts["prompt_encodes"] += 1
                per_counts["workspace_updates"] += {"K0": 0, "K1": 1, "K4": 4}[arm_id]
                gen_result, row_reader_measurements = _greedy_with_reader_measurements(host,
                    arm_id == "K4" and stage == "final", batch.prompt_ids, batch.prompt_mask,
                    workspace=workspace, eos_token_id=config["decoding"]["eos_token_id"],
                    max_new_tokens=128, max_total_tokens=config["decoding"]["max_total_tokens"], check=check)
        per_counts["greedy_answer_forwards"] = gen_result.decoder_calls
        score = score_generation(example.task, gen_result, tokenizer)
        score_row = {"example_id": example.example_id, "answer_text": score["answer_text"],
            "valid": bool(score["valid"]), "verifier_reason": score["verifier_reason"],
            "termination": score["termination"], "generated_tokens": int(score["generated_tokens"])}
        generated_token_ids = list(gen_result.token_ids)
        if len(generated_token_ids) != score_row["generated_tokens"] or any(
                type(token) is not int or token < 0 or token >= config["decoding"]["embedding_vocab_size"]
                for token in generated_token_ids) or (config["decoding"]["eos_token_id"] in generated_token_ids[:-1]) or \
                (score_row["termination"] == "eos") != (generated_token_ids[-1] == config["decoding"]["eos_token_id"]) or \
                (score_row["termination"] == "token_cap" and score_row["generated_tokens"] != 128):
            raise ValueError("generated token IDs violate vocabulary, EOS, or token-cap bounds")
        if include_ce:
            score_row["mean_ce"] = float(ce.loss.item())
            score_row["supervised_tokens"] = ce.supervised_tokens
        memory = _memory_measurement()
        cost = {"elapsed_seconds": max(0.0, time.perf_counter() - started),
            "elapsed_scope": "full_case_including_reader_measurement_hooks"
                if arm_id == "K4" and stage == "final" else "full_case", "memory": memory}
        envelope = {"schema_version": "dwr-source-pilot-prediction-v1", "arm_id": arm_id,
            "stage": stage, "step": step, "identities": identities,
            "checkpoint_sha256": checkpoint_sha256, "example_id": example.example_id,
            "prompt_sha256": row_prompt_hash, "noise_seed": eval_seed,
            "noise_sha256": row_noise_hash, "workspace_sha256": workspace_hash,
            "donor_id": donor_id, "bank_metadata_sha256": bank_digest,
            "ce_workspace_sha256": ce_workspace_hash, "generated_token_ids": generated_token_ids,
            "eos_token_id": config["decoding"]["eos_token_id"],
            "vocab_size": config["decoding"]["embedding_vocab_size"],
            "reader_measurements": row_reader_measurements if arm_id == "K4" and stage == "final" else None,
            "score": score_row,
            "counts": per_counts, "costs": cost}
        if attempt_created:
            _append_canonical(prediction_path, envelope)
        else:
            _write_exclusive(prediction_path, _canonical(envelope))
            attempt_created = True
        envelopes.append(envelope)
        scores.append(score_row)
        totals["completed_cases"] += 1
        for key, value in per_counts.items():
            totals[key] += value
        _emit(on_event, "prediction_completed", {"arm_id": arm_id, "stage": stage,
            "step": step, "example_id": example.example_id, "greedy_answer_forwards": gen_result.decoder_calls,
            "valid": score_row["valid"], "workspace_sha256": workspace_hash})
    metrics = score_pilot_stage(tuple(scores), expected_ids=expected_ids, metadata=metadata,
        examples=examples, include_ce=include_ce)
    records_ref = {"path": prediction_path.relative_to(jobroot).as_posix(), "sha256": _hash_file(prediction_path)}
    reader_summary = _combine_reader_measurements(envelopes) if arm_id == "K4" and stage == "final" else None
    receipt = {"schema_version": "dwr-source-pilot-stage-v1", "arm_id": arm_id,
        "stage": stage, "step": step, "identities": identities,
        "checkpoint_sha256": checkpoint_sha256, "checkpoint": checkpoint_ref,
        "example_ids": list(expected_ids),
        "records": records_ref, "metrics": metrics, "bank_metadata_sha256": bank_digest,
        "reader_summary": reader_summary,
        "counts": totals}
    receipt_raw = _canonical(receipt)
    _write_exclusive(receipt_path, receipt_raw)
    reference = {"path": receipt_path.relative_to(jobroot).as_posix(), "sha256": _sha_bytes(receipt_raw)}
    verified = verify_pilot_stage_receipt(jobroot, reference, arm_id=arm_id, stage=stage,
        expected_step=step, expected_ids=expected_ids, expected_checkpoint_sha256=checkpoint_sha256,
        identities=identities, metadata=metadata, examples=examples,
        expected_bank_metadata_sha256=bank_digest, expected_bank_reference=bank_ref)
    return verified, reference, envelopes, totals


def run_pilot_arm(host, tokenizer, tokenizer_identity, training_examples, validation_examples, *,
        registration, arm_id, identities, metadata, output_dir, check, on_event, before_complete, resume=None) -> dict:
    """Train and assess one registered K0/K1/K4 arm with complete boundaries."""
    if not isinstance(host, Qwen2WorkspaceLM):
        raise TypeError("host must be an actual native Qwen2WorkspaceLM")
    if not isinstance(registration, Mapping) or not isinstance(registration.get("config"), Mapping):
        raise TypeError("registration must contain the closed pilot config")
    config = registration["config"]
    if arm_id not in {arm.get("arm_id") for arm in config.get("arms", ())}:
        raise ValueError("arm_id is not present in the pilot registration")
    arm = next(arm for arm in config["arms"] if arm["arm_id"] == arm_id)
    steps = config["training"]["optimizer_steps"]
    if type(steps) is not int or steps <= 0:
        raise ValueError("pilot training step count must be a positive integer")
    method_steps = arm["steps"]
    expected_method_steps = {"K0": 0, "K1": 1, "K4": 4}[arm_id]
    if method_steps != expected_method_steps:
        raise ValueError("registered arm method/horizon mismatch")
    if host.workspace.max_steps != 4 or host.workspace.identity_mode != "exchangeable" or \
            host.workspace.num_slots != config.get("host", {}).get("num_slots", host.workspace.num_slots) or \
            host.workspace.latent_dim != config.get("host", {}).get("latent_dim", host.workspace.latent_dim):
        raise ValueError("native host architecture differs from the registered EX capacity/fixed-N policy")
    if not isinstance(tokenizer_identity, TokenizerIdentity):
        raise TypeError("tokenizer_identity must be a TokenizerIdentity")
    if not callable(check) or not callable(on_event) or not callable(before_complete):
        raise TypeError("check, durable on_event and before_complete callbacks are required")
    if not isinstance(metadata, dict) or set(metadata) != {"training", "validation"}:
        raise ValueError("kernel metadata must contain exactly training and validation maps")
    train_ids, train_by_id = _example_maps(training_examples, metadata["training"], split="source_train")
    selected_ids, _ = _example_maps(validation_examples, metadata["validation"], split="source_validation")
    if train_ids != tuple(sorted(train_ids)):
        raise ValueError("pilot training examples must be lexicographically ordered before sampling")
    if len(selected_ids) < len(_FINE) or set(train_ids) & set(selected_ids):
        raise ValueError("training and validation ID sets must be disjoint and validation span all strata")
    if identities.keys() != {"source", "config", "data", "model", "tokenizer", "schedule"} or \
            any(not isinstance(value, str) or not _HEX.fullmatch(value) for value in identities.values()):
        raise ValueError("pilot identities must contain exactly six lowercase SHA-256 values")
    progress = {"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0}
    output_dir = Path(output_dir).resolve(strict=True)
    if not output_dir.is_dir() or not re.fullmatch(r"segment-[0-9]{3}", output_dir.name):
        raise ValueError("output_dir must be a pre-created segment-NNN directory")
    jobroot = output_dir.parent.resolve(strict=True)
    control = configure_source_control(host, steps=method_steps)
    group_specs = []
    if method_steps:
        group_specs.append({"params": list(host.workspace.parameters()), "name": "workspace"})
    group_specs.append({"params": list(host.readers.parameters()), "name": "readers"})
    opt = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(group_specs, lr=opt["learning_rate"], betas=tuple(opt["betas"]),
        eps=opt["epsilon"], weight_decay=opt["weight_decay"], amsgrad=opt["amsgrad"],
        capturable=opt["capturable"], differentiable=opt["differentiable"], foreach=opt["foreach"],
        fused=opt["fused"], maximize=opt["maximize"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=lambda scheduler_step: min(1.0, (scheduler_step + 1) / config["training"]["schedule"]["warmup_steps"]))
    sampler = SourceSampler(train_ids, seed=config["seeds"]["source_sampler"])
    device = next(host.workspace.parameters()).device
    noise_generator = torch.Generator(device=device).manual_seed(config["seeds"]["training_noise"])
    expected_initial_digest, recipe = _stable_module_digest(host)
    initial_audit = expected_initial_digest
    _assert_frozen_original(host)
    original_modes = {module: module.training for module in host.modules()}
    original_return_dict = host.backbone.config.return_dict
    store = PilotRecoveryStore(jobroot, output_dir, identities, arm_id=arm_id,
        selected_ids=selected_ids, metadata=metadata["validation"],
        examples=validation_examples, steps=steps)
    receipts: dict[str, dict[str, str]] = {}
    bank_ref = None
    bank_metadata = bank_tensor = None
    physical_started = 0
    updates = []
    stages = {}
    recovery_descriptor = None
    last_checkpoint_path = None
    last_checkpoint_sha = None
    recognized_boundaries = {}

    def save_and_commit(phase, facts):
        nonlocal recovery_descriptor, last_checkpoint_path, last_checkpoint_sha
        step = progress["optimizer_step"]
        path = output_dir / "recovery" / f"recovery-checkpoint-step-{step:06d}.pt"
        known = recognized_boundaries.get(step)
        if path.exists():
            if known is None or known[0] != path.resolve(strict=True) or \
                    known[2] != progress or last_checkpoint_path is None or \
                    last_checkpoint_path.resolve(strict=True) != known[0] or last_checkpoint_sha != known[1]:
                raise PilotRecoveryError("refusing to adopt an unrecognized existing checkpoint path")
            if _hash_file(path) != known[1]:
                raise PilotRecoveryError("recognized checkpoint bytes changed after their durable boundary")
            last_checkpoint_path = path
            last_checkpoint_sha = known[1]
        else:
            if known is not None:
                raise PilotRecoveryError("recognized committed checkpoint is missing at its durable boundary")
            last_checkpoint_sha, _ = _save_boundary(host, optimizer, scheduler, sampler,
                noise_generator, identities, progress, path)
            last_checkpoint_path = path
            recognized_boundaries[step] = (path.resolve(strict=True), last_checkpoint_sha, dict(progress))
        recovery_descriptor = store.commit(last_checkpoint_path, last_checkpoint_sha,
            progress, phase, facts)
        _emit(on_event, "recovery_committed", {"arm_id": arm_id, "phase": phase,
            "progress": dict(progress), "checkpoint_path": last_checkpoint_path.relative_to(jobroot).as_posix(),
            "checkpoint_sha256": last_checkpoint_sha, "cursor_sha256": recovery_descriptor["cursor_sha256"]})
        return last_checkpoint_path, last_checkpoint_sha

    facts = {"initial_state_sha256": expected_initial_digest, "initial_state_hash_recipe": recipe,
        "stage_receipts": {}, "bank": None, "frozen_audit_refs": None, "final_result": None}
    phase = "initial_evaluation_pending"
    try:
        if resume is None:
            initial_path, initial_sha = save_and_commit("initial_evaluation_pending", facts)
        else:
            previous = PilotRecoveryStore.verify_descriptor(resume, identities, arm_id=arm_id,
                selected_ids=selected_ids, metadata=metadata["validation"], examples=validation_examples, steps=steps)
            if expected_initial_digest != previous["facts"]["initial_state_sha256"] or \
                    recipe != previous["facts"]["initial_state_hash_recipe"]:
                raise PilotRecoveryError("fresh host initial state differs from the committed paired initialization")
            expected_initial_digest = previous["facts"]["initial_state_sha256"]
            recipe = copy.deepcopy(previous["facts"]["initial_state_hash_recipe"])
            if previous["phase"] == "complete":
                return {"status": "PILOT_ARM_COMPLETE", "arm_id": arm_id,
                    "progress": previous["progress"], "stage_receipts": previous["facts"]["stage_receipts"],
                    "recovery_descriptor": resume, "final_result": previous["facts"]["final_result"],
                    "resumed_complete": True}
            if previous["segment"] != output_dir.name:
                copied = output_dir / "recovery" / f"recovery-checkpoint-step-{previous['progress']['optimizer_step']:06d}.pt"
                with previous["checkpoint_path_absolute"].open("rb") as source, copied.open("xb") as target:
                    while data := source.read(1024 * 1024):
                        target.write(data)
                    target.flush(); os.fsync(target.fileno())
                if _hash_file(copied) != previous["checkpoint_sha256"]:
                    raise PilotRecoveryError("copied continuation checkpoint bytes changed")
                checkpoint_path = copied
            else:
                checkpoint_path = previous["checkpoint_path_absolute"]
            progress = load_training_checkpoint(checkpoint_path,
                expected_sha256=previous["checkpoint_sha256"], host=host, optimizer=optimizer,
                scheduler=scheduler, scaler=None, sampler=sampler, noise_generator=noise_generator,
                identities=identities)
            if progress != previous["progress"]:
                raise PilotRecoveryError("loaded checkpoint progress differs from committed cursor")
            recognized_boundaries[progress["optimizer_step"]] = (
                checkpoint_path.resolve(strict=True), previous["checkpoint_sha256"], dict(progress))
            facts = copy.deepcopy(previous["facts"])
            receipts = copy.deepcopy(facts["stage_receipts"])
            bank_ref = copy.deepcopy(facts["bank"])
            if bank_ref is not None:
                from .pilot_recovery import _verify_bank_metadata
                bank_metadata = _verify_bank_metadata(jobroot, bank_ref, arm_id=arm_id,
                    identities=identities, checkpoint_sha256=previous["checkpoint_sha256"],
                    selected_ids=selected_ids, steps=steps,
                    checkpoint_path=previous["checkpoint_path_absolute"])
                bank_path = _safe_path(jobroot, bank_metadata["tensor"]["path"])
                bank_tensor = torch.load(bank_path, map_location="cpu", weights_only=True)
            phase = previous["phase"]
            last_checkpoint_path = checkpoint_path
            last_checkpoint_sha = previous["checkpoint_sha256"]
            if previous["segment"] != output_dir.name:
                recovery_descriptor = store.commit(checkpoint_path, last_checkpoint_sha,
                    progress, phase, facts)
                _emit(on_event, "recovery_committed", {"arm_id": arm_id, "phase": phase,
                    "progress": dict(progress), "checkpoint_path": checkpoint_path.relative_to(jobroot).as_posix(),
                    "checkpoint_sha256": last_checkpoint_sha, "cursor_sha256": recovery_descriptor["cursor_sha256"]})

        # Reuse each complete receipt; an unfinished stage is rerun as a whole.
        if "initial" in receipts:
            _assert_frozen_original(host)
            stages["initial"] = verify_pilot_stage_receipt(jobroot, receipts["initial"],
                arm_id=arm_id, stage="initial", expected_step=0,
                expected_ids=selected_ids, expected_checkpoint_sha256=_stage_checkpoint_sha(jobroot, receipts["initial"]),
                identities=identities, metadata=metadata["validation"], examples=validation_examples)
        else:
            if phase != "initial_evaluation_pending":
                raise PilotRecoveryError("training cursor has no complete initial stage receipt")
            stages["initial"], reference, _, _ = _run_stage_preserved(host, tokenizer, tokenizer_identity,
                validation_examples, metadata["validation"], config=config, arm_id=arm_id, stage="initial",
                step=0, identities=identities,
                checkpoint_ref={"path": last_checkpoint_path.relative_to(jobroot).as_posix(),
                    "sha256": last_checkpoint_sha},
                jobroot=jobroot, segment=output_dir, check=check, on_event=on_event,
                optimizer=optimizer, scheduler=scheduler, sampler=sampler,
                noise_generator=noise_generator)
            receipts["initial"] = reference
            facts["stage_receipts"] = receipts
            phase = "training"
            save_and_commit(phase, facts)

        host.train()
        progress = dict(progress)
        while progress["optimizer_step"] < steps:
            check()
            _assert_frozen_original(host)
            next_step = progress["optimizer_step"] + 1
            ids = sampler.next_ids(1)
            example = train_by_id[ids[0]]
            batch = _collate(example, tokenizer, tokenizer_identity, config, device)
            noise = _batch_noise(host, batch, noise_generator, device)
            update_started = {"arm_id": arm_id, "net_step": next_step, "physical_index": physical_started + 1,
                "example_id": ids[0], "prompt_tokens": int(batch.prompt_mask.sum().item()),
                "supervised_tokens": int(batch.labels.ne(-100).sum().item()),
                "batch_sha256": _tensor_digest(batch.decoder_ids), "noise_sha256": _tensor_digest(noise)}
            _emit(on_event, "update_started", update_started)
            physical_started += 1
            learning_rates = tuple(float(group["lr"]) for group in optimizer.param_groups)
            diagnose = next_step in config["training"]["diagnose_state_gradient_steps"]
            try:
                result = source_control_step(host, batch, noise, control=control, optimizer=optimizer,
                    scheduler=scheduler, gradient_clip_norm=config["training"]["gradient_clip_norm"],
                    diagnose=diagnose)
            except (ArithmeticError, ValueError) as error:
                if "nonfinite" in str(error).lower() or "non-finite" in str(error).lower():
                    raise PilotNumericalDivergence(str(error), partial={"arm_id": arm_id,
                        "physical_started": physical_started, "progress": dict(progress),
                        "recovery_descriptor": recovery_descriptor}) from error
                raise
            _assert_frozen_original(host)
            progress["optimizer_step"] = next_step
            progress["examples_seen"] += 1
            progress["supervised_tokens_seen"] += result["supervised_tokens"]
            if result["supervised_tokens"] != metadata["training"][ids[0]]["answer_tokens_including_eos"]:
                raise ValueError("training supervised-token count differs from pinned source metadata")
            update = {"net_step": next_step, "physical_index": physical_started,
                "example_id": ids[0], "noise_sha256": update_started["noise_sha256"],
                "batch_sha256": update_started["batch_sha256"], "loss": result["loss"],
                "supervised_tokens": result["supervised_tokens"], "learning_rates": learning_rates,
                "gradient_norms": result["gradient_norms"], "total_gradient_norm": result["total_gradient_norm"],
                "state_norms": result["state_norms"], "state_gradient_norms": result["state_gradient_norms"],
                "reader_gradient_norms": result["reader_gradient_norms"], "policy": result["policy"]}
            updates.append(update)
            _emit(on_event, "update_completed", update)
            if progress["optimizer_step"] == steps:
                phase = "final_bank_pending" if arm_id == "K4" else "final_evaluation_pending"
            save_and_commit(phase, facts)

        if arm_id == "K4" and bank_ref is None:
            check()
            before_eval, modes_before, return_dict = _evaluation_snapshot(host, optimizer, scheduler, sampler, noise_generator)
            try:
                bank_batches = tuple(_collate(example, tokenizer, tokenizer_identity, config, device)
                                     for example in validation_examples)
                _emit(on_event, "bank_started", {"arm_id": arm_id, "step": steps,
                    "example_ids": list(selected_ids)})
                host.eval()
                bank = build_workspace_bank(host, bank_batches, steps=4,
                    noise_seed_base=config["seeds"]["evaluation_noise_base"], check=check)
                bank["step"] = steps
                bank_ref, bank_metadata = _write_bank(jobroot, output_dir, bank,
                    identities=identities, checkpoint_sha256=last_checkpoint_sha)
                from .pilot_recovery import _verify_bank_metadata
                bank_metadata = _verify_bank_metadata(jobroot, bank_ref, arm_id=arm_id,
                    identities=identities, checkpoint_sha256=last_checkpoint_sha,
                    selected_ids=selected_ids, steps=steps, checkpoint_path=last_checkpoint_path)
                bank_tensor = torch.load(_safe_path(jobroot, bank_metadata["tensor"]["path"]),
                    map_location="cpu", weights_only=True)
            finally:
                _restore_modes(host, modes_before, return_dict)
                _assert_state_preserved(before_eval, host, optimizer, scheduler, sampler, noise_generator)
            facts["bank"] = bank_ref
            phase = "final_evaluation_pending"
            save_and_commit(phase, facts)
            _emit(on_event, "bank_completed", {"arm_id": arm_id, "step": steps,
                "bank_metadata_sha256": bank_ref["sha256"], "counts": bank_metadata["counts"]})
        elif bank_ref is not None and bank_tensor is None:
            bank_metadata, bank_tensor = _load_bank(jobroot, bank_ref)

        for stage in ("final", "zero", "noise", "shuffle"):
            if stage not in ({"final", "zero", "noise", "shuffle"} if arm_id == "K4" else {"final"}):
                continue
            if stage in receipts:
                expected_step = 0 if stage == "initial" else steps
                expected_checkpoint = _stage_checkpoint_sha(jobroot, receipts[stage])
                stages[stage] = verify_pilot_stage_receipt(jobroot, receipts[stage],
                    arm_id=arm_id, stage=stage, expected_step=expected_step,
                    expected_ids=selected_ids, expected_checkpoint_sha256=expected_checkpoint,
                    identities=identities, metadata=metadata["validation"], examples=validation_examples,
                    expected_bank_metadata_sha256=(bank_ref["sha256"] if arm_id == "K4" and stage != "initial" else None),
                    expected_bank_reference=(bank_ref if arm_id == "K4" and stage != "initial" else None))
                continue
            required_phase = {"final": "final_evaluation_pending", "zero": "diagnostic_zero_pending",
                "noise": "diagnostic_noise_pending", "shuffle": "diagnostic_shuffle_pending"}[stage]
            if phase != required_phase:
                if stage == "final" and phase == "final_bank_pending" and arm_id == "K4":
                    raise PilotRecoveryError("K4 final evaluation cannot run before bank-ready cursor publication")
                if stage == "zero" and "final" in receipts:
                    phase = "diagnostic_zero_pending"
                elif stage == "noise" and "zero" in receipts:
                    phase = "diagnostic_noise_pending"
                elif stage == "shuffle" and "noise" in receipts:
                    phase = "diagnostic_shuffle_pending"
                else:
                    raise PilotRecoveryError(f"cannot resume {stage} stage from phase {phase}")
            result, reference, envelopes, count_values = _run_stage_preserved(host, tokenizer, tokenizer_identity,
                validation_examples, metadata["validation"], config=config, arm_id=arm_id, stage=stage,
                step=steps, identities=identities,
                checkpoint_ref={"path": last_checkpoint_path.relative_to(jobroot).as_posix(),
                    "sha256": last_checkpoint_sha},
                jobroot=jobroot, segment=output_dir, check=check, on_event=on_event,
                bank_ref=bank_ref if arm_id == "K4" else None,
                bank_metadata=bank_metadata, bank_tensor=bank_tensor,
                optimizer=optimizer, scheduler=scheduler, sampler=sampler,
                noise_generator=noise_generator)
            stages[stage] = result
            receipts[stage] = reference
            facts["stage_receipts"] = receipts
            if arm_id == "K0" or arm_id == "K1":
                phase_after = "final_evaluation_pending"
            else:
                phase_after = {"final": "diagnostic_zero_pending", "zero": "diagnostic_noise_pending",
                    "noise": "diagnostic_shuffle_pending", "shuffle": "diagnostic_shuffle_pending"}[stage]
            if stage == "shuffle":
                phase = "diagnostic_shuffle_pending"
            else:
                phase = phase_after
            save_and_commit(phase, facts)
            _emit(on_event, "stage_committed", {"arm_id": arm_id, "stage": stage,
                "step": steps, "receipt": reference, "counts": count_values})

        # Re-read every stage receipt before final result/callback binding.
        for stage in _expected_stage_names(arm_id):
            if stage not in receipts:
                raise PilotRecoveryError(f"pilot arm is missing required stage {stage}")
            stages[stage] = verify_pilot_stage_receipt(jobroot, receipts[stage], arm_id=arm_id,
                stage=stage, expected_step=0 if stage == "initial" else steps,
                expected_ids=selected_ids,
                expected_checkpoint_sha256=_stage_checkpoint_sha(jobroot, receipts[stage]),
                identities=identities, metadata=metadata["validation"], examples=validation_examples,
                expected_bank_metadata_sha256=bank_ref["sha256"] if arm_id == "K4" and stage != "initial" else None,
                expected_bank_reference=bank_ref if arm_id == "K4" and stage != "initial" else None)
        before_callback = _state_snapshot(host, optimizer, scheduler, sampler, noise_generator)
        audit_payload = {"arm_id": arm_id, "identities": identities,
            "initial_state_sha256": initial_audit,
            "initial_checkpoint": _receipt_checkpoint_ref(jobroot, receipts["initial"]),
            "final_checkpoint": {"path": last_checkpoint_path.relative_to(jobroot).as_posix(),
                "sha256": last_checkpoint_sha},
            "stage_receipts": receipts, "bank": bank_ref,
            "counts": _aggregate_counts(stages), "costs": _aggregate_costs(stages),
            "original_parameter_hashes": {name: _tensor_digest(value) for name, value in
                host.backbone.named_parameters()}}
        try:
            audit_refs = before_complete(audit_payload)
        except BaseException as callback_error:
            try:
                _assert_state_preserved(before_callback, host, optimizer, scheduler, sampler, noise_generator)
            except BaseException as preservation_error:
                raise callback_error from preservation_error
            raise
        else:
            _assert_state_preserved(before_callback, host, optimizer, scheduler, sampler, noise_generator)
        final_result = {"status": "PILOT_ARM_COMPLETE", "arm_id": arm_id,
            "optimizer_step": progress["optimizer_step"],
            "initial": stages["initial"]["metrics"], "final": stages["final"]["metrics"],
            "diagnostics": {name: stages[name]["metrics"] for name in ("zero", "noise", "shuffle") if name in stages},
            "learning_gate_passed": _learning_gate(stages["initial"]["metrics"], stages["final"]["metrics"])
                if arm_id == "K4" else None,
            "initial_module_sha256": expected_initial_digest, "final_module_sha256": _module_digest_states(host),
            "counts": _aggregate_counts(stages), "costs": _aggregate_costs(stages),
            "trainable_parameters": sum(p.numel() for p in host.parameters() if p.requires_grad),
            "retained_total_parameters": sum(p.numel() for p in host.parameters())}
        facts["frozen_audit_refs"] = audit_refs
        facts["final_result"] = final_result
        if arm_id == "K0" or arm_id == "K1":
            phase = "final_evaluation_pending"
        else:
            phase = "diagnostic_shuffle_pending"
        save_and_commit("complete", facts)
        _emit(on_event, "arm_complete", {"arm_id": arm_id, "result": final_result,
            "frozen_audit_refs": audit_refs})
        return {**final_result, "stage_receipts": receipts, "bank": bank_ref,
            "recovery_descriptor": recovery_descriptor, "updates": updates,
            "initial_state_sha256": expected_initial_digest,
            "checkpoint_sha256": last_checkpoint_sha,
            "resumed_complete": False}
    finally:
        # Keep the same live host and caller-visible mode/config state across all exits.
        _restore_modes(host, original_modes, original_return_dict)
        _assert_frozen_original(host)


def _expected_stage_names(arm_id):
    return ("initial", "final") if arm_id in {"K0", "K1"} else ("initial", "final", "zero", "noise", "shuffle")


def _stage_checkpoint_sha(jobroot: Path, ref: dict[str, str]) -> str:
    checkpoint = _receipt_checkpoint_ref(jobroot, ref)
    path = _safe_path(jobroot, checkpoint["path"])
    if _hash_file(path) != checkpoint["sha256"]:
        raise PilotRecoveryError("stage checkpoint changed after receipt publication")
    return checkpoint["sha256"]


def _receipt_checkpoint_ref(jobroot: Path, reference: dict[str, str]) -> dict[str, str]:
    receipt_path = _safe_path(jobroot, reference["path"])
    raw = receipt_path.read_bytes()
    if _sha_bytes(raw) != reference["sha256"]:
        raise PilotRecoveryError("committed stage receipt changed before audit handoff")
    value = _strict_json(raw, "pilot stage receipt")
    checkpoint = value.get("checkpoint") if isinstance(value, dict) else None
    return _reference(checkpoint, "committed stage checkpoint")


def _aggregate_counts(stages):
    totals = {key: 0 for key in ("completed_cases", "greedy_answer_forwards",
        "teacher_forced_answer_forwards", "prompt_encodes", "workspace_updates")}
    for stage in stages.values():
        for key, value in stage["counts"].items():
            totals[key] += value
    return totals


def _aggregate_costs(stages):
    return {"synchronized_elapsed_seconds": sum(
        cost["elapsed_seconds"] for stage in stages.values()
        for cost in _stage_cost_rows(stage)), "memory_scope": "per-prediction process peak; not additive"}


def _stage_cost_rows(stage):
    root = Path(stage["records_path"])
    raw = root.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("stage records are missing their final LF during cost aggregation")
    envelopes = [_strict_json(line, "pilot prediction envelope") for line in raw.splitlines(keepends=True)]
    return [envelope["costs"] for envelope in envelopes]


def _learning_gate(initial, final):
    primitive = final["primitive"]
    required = ("eq", "neq", "add_mod")
    initial_ce, final_ce = initial["ce"]["token_weighted_mean"], final["ce"]["token_weighted_mean"]
    return bool(final["vsr"] >= 0.20 and final["vsr"] - initial["vsr"] >= 0.10 and
        all(primitive[name]["denominator"] > 0 and primitive[name]["rate"] >= 0.20 for name in required) and
        math.isfinite(initial_ce) and initial_ce > 0 and final_ce <= 0.5 * initial_ce)


if __name__ == "__main__":
    raise SystemExit("source pilot training is called only through the registered Task 3 controller")
