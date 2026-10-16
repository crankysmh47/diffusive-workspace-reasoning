"""Closed original-Qwen2 source-validation reference; no training or overrides."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import statistics
import stat
import sys
import time
from typing import Any, Callable, Mapping

from dwr.evaluation.generation import greedy_frozen, score_generation
from dwr.experiments.records import RunRecord, verify_artifacts
from dwr.tasks import canonical
from dwr.tasks.run_support import stable_json
from dwr.tasks.schema import parse_task_json, serialize_model_input

CONFIG_SHA256 = "9af9196c59b4b65774ac82aebb57378cb2fc23513b382b40cdf5b6dfe55c824b"
COHORT_SHA256 = "bca9505b817a9b75b9d7aba33710ddc17b47596bdb9a9772924348d15b4b64bd"
FEASIBILITY_CONFIG_SHA256 = "840615e265a427946627f3320f667154e17bc56c53c15b6f90a453c10693144f"
FEASIBILITY_MODULE_SHA256 = "5c8c8f36f198f08ab906be86e88c2dbac04471f7e6f5585b93d1fb402a3c5139"
PASS_STATUS = "PASS_FROZEN_REFERENCE_OPERATIONAL_V1"
_CONFIG_PATH = "configs/models/frozen_source_reference_v1.json"
_HEX = frozenset("0123456789abcdef")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_output_directory(root: Path, relative: str) -> Path:
    if relative != "runs/frozen-source-reference-20261016-v1" or "\\" in relative:
        raise ValueError("output directory differs from the closed registration")
    target = root.joinpath(*PurePosixPath(relative).parts)
    resolved_root = root.resolve(strict=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError("registered frozen-reference output directory already exists")
    if not target.parent.is_dir() or target.parent.resolve(strict=True) != resolved_root / "runs":
        raise ValueError("registered output parent is missing or aliased")
    if target.parent.is_symlink():
        raise ValueError("registered output parent cannot be a symlink")
    return target


def _read(path: Path, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if any(item.is_symlink() or bool(getattr(item.lstat(), "st_file_attributes", 0) & reparse)
           for item in (absolute, *absolute.parents) if item.exists()):
        raise ValueError(f"{label} cannot be linked through a symlink or reparse path")
    if not absolute.is_file():
        raise ValueError(f"{label} must be a regular file")
    return absolute.read_bytes()


def _json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_registration(root: Path, config_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the sole config/cohort and validate their closed identities."""
    expected = root.joinpath(*PurePosixPath(_CONFIG_PATH).parts).resolve(strict=True)
    requested = Path(os.path.abspath(os.fspath(config_path)))
    if requested.resolve(strict=True) != expected:
        raise ValueError("only the registered frozen-reference config is accepted")
    config_raw = _read(expected, "frozen-reference config")
    if _sha(config_raw) != CONFIG_SHA256:
        raise ValueError("frozen-reference config bytes differ from the closed registration")
    config = _json(config_raw, "frozen-reference config")
    cohort_path = root.joinpath(*PurePosixPath(config["dependencies"]["cohort_path"]).parts)
    cohort_raw = _read(cohort_path, "frozen-reference cohort")
    if _sha(cohort_raw) != COHORT_SHA256 or config["dependencies"]["cohort_sha256"] != COHORT_SHA256:
        raise ValueError("cohort bytes differ from the closed registration")
    cohort = _json(cohort_raw, "frozen-reference cohort")
    selected = cohort.get("selected")
    if cohort.get("schema_version") != "dwr-frozen-reference-measurement-cohort-v1" or type(selected) is not list or len(selected) != 15:
        raise ValueError("registered cohort must contain exactly 15 selections")
    ids = [entry.get("example_id") for entry in selected]
    if len(set(ids)) != 15 or any(type(value) is not str or len(value) != 64 or not set(value) <= _HEX for value in ids):
        raise ValueError("registered cohort IDs must be unique lowercase SHA-256 values")
    profiles = {name: sum(entry.get("profile") == name for entry in selected)
                for name in ("eq", "neq", "add_mod")}
    if profiles != {"eq": 5, "neq": 5, "add_mod": 5}:
        raise ValueError("registered cohort profile counts differ from the fixed design")
    if any(type(item.get("prompt_tokens")) is not int or item["prompt_tokens"] <= 0
           or item["prompt_tokens"] > config["decoding"]["max_prompt_tokens"] for item in selected):
        raise ValueError("registered prompt lengths exceed the fixed token budget")
    if config["dependencies"]["feasibility_config_sha256"] != FEASIBILITY_CONFIG_SHA256 or config["dependencies"]["feasibility_module_sha256"] != FEASIBILITY_MODULE_SHA256:
        raise ValueError("feasibility helper/config pins differ from the reviewed dependency")
    dependency_path = root.joinpath(*PurePosixPath(config["dependencies"]["feasibility_config_path"]).parts)
    dependency_raw = _read(dependency_path, "full feasibility config")
    if _sha(dependency_raw) != FEASIBILITY_CONFIG_SHA256:
        raise ValueError("full feasibility config bytes differ from the reviewed pin")
    from dwr.experiments import gpu_feasibility
    dependency_config = gpu_feasibility.validate_config_bytes(dependency_raw)
    return {"config": config, "config_raw": config_raw, "config_path": expected,
            "cohort": cohort, "cohort_raw": cohort_raw, "selected": tuple(selected),
            "config_sha256": _sha(config_raw), "cohort_sha256": _sha(cohort_raw),
            "feasibility_config": dependency_config}


def _select_source_rows(rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
                        cohort: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    selected = cohort.get("selected")
    if type(selected) is not list or not isinstance(rows, (tuple, list)):
        raise ValueError("source rows or cohort are malformed")
    expected = {item["example_id"]: item for item in selected}
    found: dict[str, dict[str, Any]] = {}; seen: set[str] = set()
    for row in rows:
        if type(row) is not dict or row.get("split") != "source_validation":
            raise ValueError("source reader returned a non-validation row")
        identity = row.get("example_id")
        if type(identity) is not str or len(identity) != 64 or not set(identity) <= _HEX:
            raise ValueError("source reader returned a malformed example ID")
        if identity in seen:
            raise ValueError("source reader returned a duplicate example ID")
        seen.add(identity)
        if identity not in expected:
            continue
        registration = expected[identity]
        for field in ("profile", "fine_stratum"):
            if row.get(field) != registration[field]:
                raise ValueError(f"source row {field} differs from the frozen cohort")
        row_prompt_tokens = row.get("prompt_tokens")
        if row_prompt_tokens is not None and (type(row_prompt_tokens) is not int or row_prompt_tokens != registration["prompt_tokens"]):
            raise ValueError("source row prompt length differs from the frozen cohort")
        found[identity] = row
    if set(found) != set(expected):
        raise ValueError("source reader did not return every frozen cohort ID exactly once")
    return tuple(found[item["example_id"]] for item in selected)


def _read_validation_rows(root: Path, config: Mapping[str, Any], reader: Callable[..., Any] | None = None):
    if reader is None:
        from dwr.experiments.source_reader import read_source_export as reader
    return reader(root.joinpath(*PurePosixPath(config["data"]["source_export_dir"]).parts),
                  expected_manifest_sha256=config["data"]["source_export_manifest_sha256"],
                  split="source_validation")


def _frame_prompt(task: Any, tokenizer: Any) -> tuple[str, str, tuple[int, ...], dict[str, str]]:
    model_input = serialize_model_input(task)
    framed = model_input + "\n"
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer.encode must be callable")
    ids = encode(framed, add_special_tokens=False)
    if type(ids) not in (list, tuple) or not ids or any(type(item) is not int or item < 0 for item in ids):
        raise ValueError("tokenizer returned malformed prompt IDs")
    token_ids = tuple(ids)
    token_raw = json.dumps(token_ids, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return model_input, framed, token_ids, {
        "model_input_sha256": _sha(model_input.encode("utf-8")),
        "framed_text_sha256": _sha(framed.encode("utf-8")),
        "token_ids_sha256": _sha(token_raw),
    }


def _evaluate_one(model: Any, prompt_ids: Any, prompt_mask: Any, task: Any,
                  tokenizer: Any, *, eos_token_id: int, max_new_tokens: int,
                  max_total_tokens: int, check: Callable[[], None] | None) -> dict[str, Any]:
    generation = greedy_frozen(model, prompt_ids, prompt_mask, eos_token_id=eos_token_id,
                               max_new_tokens=max_new_tokens, max_total_tokens=max_total_tokens,
                               check=check)
    score_started = time.perf_counter()
    score = score_generation(task, generation, tokenizer)
    return {"generation": generation, "score": score,
            "scoring_elapsed_seconds": max(0.0, time.perf_counter() - score_started)}


def _require_repeatability(attempts: list[dict[str, Any]], *, expected_examples: int,
                           repetitions: int) -> None:
    if len(attempts) != expected_examples * repetitions:
        raise ValueError("attempt count differs from registered repetitions and cohort size")
    if any(attempts[index]["repetition"] != index // expected_examples
           for index in range(len(attempts))):
        raise ValueError("attempts are not ordered repetition-outermost")
    cohort_order = [attempt["example_id"] for attempt in attempts[:expected_examples]]
    for repetition in range(repetitions):
        chunk = attempts[repetition * expected_examples:(repetition + 1) * expected_examples]
        if [attempt.get("example_id") for attempt in chunk] != cohort_order:
            raise ValueError("attempt order differs from the frozen cohort order")
    by_id: dict[str, dict[int, dict[str, Any]]] = {}
    for attempt in attempts:
        example_id = attempt.get("example_id")
        rep = attempt.get("repetition")
        if type(example_id) is not str or type(rep) is not int or not 0 <= rep < repetitions:
            raise ValueError("attempt has an invalid example or repetition identity")
        reps = by_id.setdefault(example_id, {})
        if rep in reps:
            raise ValueError("attempt has a duplicate example/repetition pair")
        reps[rep] = attempt
    if len(by_id) != expected_examples or any(set(rows) != set(range(repetitions)) for rows in by_id.values()):
        raise ValueError("attempts do not cover each example in every repetition")
    for rows in by_id.values():
        baseline = rows[0]
        for rep in range(1, repetitions):
            if rows[rep].get("token_ids") != baseline.get("token_ids") or rows[rep].get("score") != baseline.get("score"):
                raise ValueError("repeated generated token IDs or strict scores differ")


def _summarize_attempts(attempts: list[dict[str, Any]], *, expected_examples: int,
                        repetitions: int) -> dict[str, Any]:
    _require_repeatability(attempts, expected_examples=expected_examples, repetitions=repetitions)
    rep0 = [item for item in attempts if item["repetition"] == 0]
    if len({item["example_id"] for item in rep0}) != expected_examples:
        raise ValueError("primary metric does not have one unique example per cohort entry")
    failure_categories: dict[str, dict[str, int]] = {}
    for repetition in range(repetitions):
        reasons: dict[str, int] = {}
        for item in attempts:
            if item["repetition"] != repetition:
                continue
            reason = item["score"]["verifier_reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
            if item["score"]["termination"] == "token_cap":
                reasons["token_cap"] = reasons.get("token_cap", 0) + 1
        failure_categories[str(repetition)] = reasons
    return {"attempt_count": len(attempts), "quality": {
        "repetition": 0, "numerator": sum(bool(item["score"]["valid"]) for item in rep0),
        "valid_count": sum(bool(item["score"]["valid"]) for item in rep0),
        "denominator": expected_examples,
        "strict_valid_solution_rate": sum(bool(item["score"]["valid"]) for item in rep0) / expected_examples,
    }, "repetitions": {str(rep): {"examples": sum(item["repetition"] == rep for item in attempts),
                                     "valid_count": sum(bool(item["score"]["valid"]) for item in attempts if item["repetition"] == rep)}
                       for rep in range(repetitions)},
        "failure_categories_by_repetition": failure_categories,
        "decoder_calls_total": sum(len(item["token_ids"]) for item in attempts)}


def _write_json(path: Path, value: Any) -> None:
    raw = (stable_json(value) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _append_jsonl(path: Path, value: Any) -> None:
    raw = (stable_json(value) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _preserve_failure(record: RunRecord, error: BaseException, *, interrupted: bool) -> None:
    """Keep partial artifacts and link a failed/interrupted terminal record."""
    root = record.output_dir
    try:
        summary = root / "summary.json"
        if summary.is_file():
            os.replace(summary, root / "summary.partial.json")
        for name in ("outcome.json",):
            candidate = root / name
            if candidate.is_file():
                os.replace(candidate, root / "outcome.partial.json")
        _write_json(root / "failure.json", {"status": "INTERRUPTED" if interrupted else "FAILED",
                    "error_type": type(error).__name__, "error": str(error), "flops": None})
        record.append_event("reference_stopped", {"error_type": type(error).__name__,
                                                   "error": str(error), "partial": True})
        artifacts = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                           if path.is_file() and path.name not in {"manifest.initial.json", "manifest.json", "events.jsonl"})
        status = "interrupted" if interrupted else "failed"
        record.finish(status, counts={"failed_invocation": 1}, artifacts=artifacts,
                      error=f"{type(error).__name__}: {error}")
    except Exception:
        # A partially written record remains available if a secondary filesystem error occurs.
        return


def _validate_feasibility_receipts(root: Path, dependency: Mapping[str, Any], helper: Any, *,
                                   expected_module_sha256: str, expected_config_sha256: str,
                                   expected_manifest_sha256: str, expected_outcome_sha256: str) -> dict[str, Any]:
    """Validate the pinned helper and its completed receipt chain."""
    module_path = Path(helper.__file__).resolve(strict=True)
    module_raw = _read(module_path, "feasibility helper module")
    if _sha(module_raw) != expected_module_sha256:
        raise ValueError("feasibility helper module bytes differ from the reviewed pin")
    full_path = root.joinpath(*PurePosixPath(dependency["feasibility_config_path"]).parts)
    full_raw = _read(full_path, "full feasibility config")
    if _sha(full_raw) != expected_config_sha256:
        raise ValueError("full feasibility config bytes differ from the reviewed pin")
    full = helper.validate_config_bytes(full_raw)
    run_dir = root.joinpath(*PurePosixPath(dependency["feasibility_run_dir"]).parts)
    terminal_path = run_dir / "manifest.json"
    terminal_raw = _read(terminal_path, "feasibility terminal manifest")
    if _sha(terminal_raw) != expected_manifest_sha256:
        raise ValueError("feasibility terminal manifest identity mismatch")
    terminal = verify_artifacts(run_dir)
    if terminal.get("status") != "completed":
        raise ValueError("pinned feasibility run did not complete")
    outcome_path = run_dir / "outcome.json"
    outcome_raw = _read(outcome_path, "feasibility outcome")
    if _sha(outcome_raw) != expected_outcome_sha256:
        raise ValueError("feasibility outcome identity mismatch")
    outcome = _json(outcome_raw, "feasibility outcome")
    if outcome.get("status") != "PASS_FEASIBILITY":
        raise ValueError("pinned feasibility outcome is not PASS_FEASIBILITY")
    return {"path": str(full_path), "config": full, "config_sha256": _sha(full_raw),
            "module_sha256": _sha(module_raw), "manifest_sha256": _sha(terminal_raw),
            "outcome_sha256": _sha(outcome_raw), "outcome": outcome}


def _validate_feasibility(root: Path, config: Mapping[str, Any], helper: Any) -> dict[str, Any]:
    dependency = config["dependencies"]
    return _validate_feasibility_receipts(root, dependency, helper,
        expected_module_sha256=FEASIBILITY_MODULE_SHA256,
        expected_config_sha256=FEASIBILITY_CONFIG_SHA256,
        expected_manifest_sha256=dependency["feasibility_manifest_sha256"],
        expected_outcome_sha256=dependency["feasibility_outcome_sha256"])


def _hash_original_parameters(model: Any, helper: Any) -> dict[str, str]:
    """Hash every parameter on the raw frozen model using the pinned tensor routine."""
    result = {name: helper._tensor_hash(parameter) for name, parameter in model.named_parameters()}
    if not result:
        raise ValueError("original model exposes no named parameters")
    return result


def _rng_snapshot(torch: Any, numpy: Any, *, cuda: bool) -> tuple[Any, dict[str, str]]:
    python_state = random.getstate()
    numpy_state = numpy.random.get_state()
    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()] if cuda else []
    serial = {
        "python": _sha(repr(python_state).encode("utf-8")),
        "numpy": _sha(numpy_state[1].tobytes() + repr(numpy_state[0:1] + numpy_state[2:]).encode("utf-8")),
        "torch_cpu": _sha(cpu_state.cpu().numpy().tobytes()),
        "torch_cuda": [_sha(state.cpu().numpy().tobytes()) for state in cuda_states],
    }
    return (python_state, numpy_state, cpu_state, cuda_states), serial


def _same_rng(snapshot: tuple[Any, ...], torch: Any, numpy: Any, *, cuda: bool) -> bool:
    python_state, numpy_state, cpu_state, cuda_states = snapshot
    current_numpy = numpy.random.get_state()
    numpy_equal = (numpy_state[0] == current_numpy[0] and
                   (numpy_state[1] == current_numpy[1]).all() and
                   numpy_state[2:] == current_numpy[2:])
    if not (python_state == random.getstate() and numpy_equal and
            torch.equal(cpu_state, torch.random.get_rng_state())):
        return False
    if cuda:
        current_cuda = torch.cuda.get_rng_state_all()
        return len(current_cuda) == len(cuda_states) and all(torch.equal(a, b) for a, b in zip(cuda_states, current_cuda))
    return True


def _decode_memory(torch: Any, device: Any) -> dict[str, int]:
    if getattr(device, "type", None) != "cuda":
        return {"peak_allocated_mib": 0, "peak_reserved_mib": 0}
    return {"peak_allocated_mib": math.ceil(torch.cuda.max_memory_allocated() / (1024 * 1024)),
            "peak_reserved_mib": math.ceil(torch.cuda.max_memory_reserved() / (1024 * 1024))}


def _make_guard(helper: Any, config: Mapping[str, Any], torch: Any, guard: Any,
                device: Any) -> tuple[Any, Callable[[], None]]:
    limits = config["limits"]
    gpu_caps = {"allocated": limits["maximum_peak_allocated_mib"], "reserved": limits["maximum_peak_reserved_mib"]}

    def check() -> None:
        guard.check()
        memory = _decode_memory(torch, device)
        if getattr(device, "type", None) == "cuda" and (memory["peak_allocated_mib"] > gpu_caps["allocated"] or memory["peak_reserved_mib"] > gpu_caps["reserved"]):
            raise MemoryError("registered GPU allocated/reserved peak cap exceeded")
    return guard, check


def _execution_device(full: Mapping[str, Any], torch: Any) -> Any:
    return torch.device(full["hardware"]["device"])


def _load_original_model(root: Path, full: Mapping[str, Any], helper: Any, device: Any) -> tuple[Any, Any]:
    from transformers import AutoTokenizer, Qwen2ForCausalLM
    local = helper._resolve_registered(root, full["model"]["local_dir"])
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
    model = Qwen2ForCausalLM.from_pretrained(local, torch_dtype=__import__("torch").bfloat16,
        use_safetensors=True, local_files_only=True, trust_remote_code=False, device_map=None,
        low_cpu_mem_usage=False, attn_implementation="sdpa")
    return model, tokenizer


def _resolved_settings(config: Mapping[str, Any], full: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    return {"operation": "original_frozen_source_validation_reference", "model": config["model_start"],
            "model_identity": full["model"], "runtime": {"precision": full["precision"],
            "attention": full["attention"], "numerics": full["numerics"],
            "cache": config["decoding"]["kv_cache"], "device": full["hardware"]["device"]},
            "generation": config["decoding"], "cohort_sha256": config["dependencies"]["cohort_sha256"],
            "feasibility_config_sha256": FEASIBILITY_CONFIG_SHA256,
            "feasibility_module_sha256": FEASIBILITY_MODULE_SHA256,
            "source_commit": source["commit"], "training": False, "optimizer": None,
            "flops": None, "profiler": None}


def _run_model(root: Path, reg: Mapping[str, Any], helper: Any, full: Mapping[str, Any],
               selected: tuple[dict[str, Any], ...], record: RunRecord,
               guard: Any, check: Callable[[], None], *, device: Any | None = None) -> dict[str, Any]:
    import numpy as np
    import torch
    config = reg["config"]
    model_config = full["model"]
    device = _execution_device(full, torch) if device is None else device
    _set_stage_start = time.perf_counter()
    actual_policy = helper._set_numerics(full, torch_module=torch)
    python = config["seeds"]["python_global"]
    random.seed(python); np.random.seed(config["seeds"]["numpy_global"])
    torch.manual_seed(config["seeds"]["torch_cpu_global"])
    if getattr(device, "type", None) == "cuda":
        torch.cuda.manual_seed_all(config["seeds"]["torch_cuda_global"])
        torch.cuda.reset_peak_memory_stats(device)
    check()
    load_started = time.perf_counter()
    model, tokenizer = _load_original_model(root, full, helper, device)
    if model.config._attn_implementation != "sdpa":
        raise RuntimeError("loaded original model did not select registered SDPA")
    model.to(device=device).eval()
    if any(parameter.device != device or parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ValueError("original model parameters differ from the registered device or BF16 dtype")
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("original model did not remain frozen")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    original_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    parameter_counts = {"total": total_parameters, "original": original_parameters,
                        "trainable": trainable_parameters, "new": total_parameters - original_parameters}
    if parameter_counts["original"] != parameter_counts["total"] or parameter_counts["new"] != 0:
        raise ValueError("loaded raw original model unexpectedly contains non-original parameters")
    if parameter_counts["trainable"] != 0:
        raise ValueError("original frozen model has trainable parameters")
    check()
    record.append_event("model_loaded", {"parameter_counts": parameter_counts,
        "device": str(device), "dtype": str(next(model.parameters()).dtype),
        "training": False, "optimizer": None})
    load_seconds = max(0.0, time.perf_counter() - load_started)
    original_hash_before = _hash_original_parameters(model, helper)

    tasks = []
    prompt_rows = []
    preparation_started = time.perf_counter()
    for row, registered in zip(selected, reg["selected"]):
        task = parse_task_json(row["task"])
        if canonical.content_id(task) != registered["example_id"]:
            raise ValueError("source task ID differs from registered cohort")
        _, framed, token_ids, hashes = _frame_prompt(task, tokenizer)
        if len(token_ids) != registered["prompt_tokens"]:
            raise ValueError("tokenizer prompt length differs from frozen cohort")
        if len(token_ids) > config["decoding"]["max_prompt_tokens"] or len(token_ids) + config["decoding"]["max_new_tokens_including_eos"] > config["decoding"]["max_total_tokens"]:
            raise ValueError("prompt exceeds registered prompt/total token caps")
        tasks.append(task); prompt_rows.append((framed, token_ids, hashes))
    preparation_seconds = max(0.0, time.perf_counter() - preparation_started)

    warm_rng = random.Random(config["seeds"]["warmup_ids_cpu"])
    warm_ids = tuple(warm_rng.randrange(config["measurement"]["warmup_id_low_inclusive"],
                                        config["measurement"]["warmup_id_high_exclusive"])
                     for _ in range(config["measurement"]["synthetic_warmup_prompt_tokens"]))
    warm_input = torch.tensor([warm_ids], dtype=torch.long, device=device)
    warm_mask = torch.ones_like(warm_input, dtype=torch.bool)
    warm_started = time.perf_counter()
    warm = greedy_frozen(model, warm_input, warm_mask, eos_token_id=config["decoding"]["eos_token_id"],
                         max_new_tokens=config["measurement"]["synthetic_warmup_max_new_tokens"],
                         max_total_tokens=len(warm_ids) + config["measurement"]["synthetic_warmup_max_new_tokens"],
                         check=check)
    warm_seconds = max(0.0, time.perf_counter() - warm_started)
    warm_record = {"synthetic": True, "prompt_token_ids": list(warm_ids),
                   "prompt_token_count": len(warm_ids), "generated_token_ids": list(warm.token_ids),
                   "termination": "eos" if warm.ended_with_eos else "token_cap",
                   "decoder_calls": warm.decoder_calls, "elapsed_seconds": warm.elapsed_wall_seconds,
                   "excluded_from_quality": True}
    _write_json(record.output_dir / "warmup.json", warm_record)
    record.append_event("synthetic_warmup_complete", {"decoder_calls": warm.decoder_calls,
        "prompt_tokens": len(warm_ids), "excluded_from_quality": True})
    use_cuda_rng = getattr(device, "type", None) == "cuda"
    baseline_rng, rng_before = _rng_snapshot(torch, np, cuda=use_cuda_rng)
    attempts: list[dict[str, Any]] = []
    predictions_path = record.output_dir / "predictions.jsonl"
    predictions_path.touch(exist_ok=False)
    scoring_seconds = 0.0; writing_seconds = 0.0
    reps = config["measurement"]["repetitions_per_example"]
    for repetition in range(reps):
        for row_index, (row, task, prompt_data) in enumerate(zip(selected, tasks, prompt_rows)):
            check()
            framed, token_ids, prompt_hashes = prompt_data
            ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            mask = torch.ones_like(ids, dtype=torch.bool)
            result = _evaluate_one(model, ids, mask, task, tokenizer,
                eos_token_id=config["decoding"]["eos_token_id"],
                max_new_tokens=config["decoding"]["max_new_tokens_including_eos"],
                max_total_tokens=config["decoding"]["max_total_tokens"], check=check)
            generation = result["generation"]; score = result["score"]
            scoring_seconds += result["scoring_elapsed_seconds"]
            memory = _decode_memory(torch, device)
            attempt = {"example_id": row["example_id"], "lineage_id": row["lineage_id"],
                "structure_signature": row["structure_signature"], "profile": row["profile"],
                "fine_stratum": row["fine_stratum"], "repetition": repetition,
                "prompt_tokens": len(token_ids), "prompt_hashes": prompt_hashes,
                "framed_text_sha256": prompt_hashes["framed_text_sha256"],
                "generated_token_ids": list(generation.token_ids), "answer_text": score["answer_text"],
                "termination": score["termination"], "strict_verifier": {
                    "valid": score["valid"], "reason": score["verifier_reason"]},
                "generated_tokens": generation.decoder_calls, "decoder_calls": generation.decoder_calls,
                "generation_elapsed_seconds": generation.elapsed_wall_seconds,
                "scoring_elapsed_seconds": result["scoring_elapsed_seconds"],
                "run_peak_so_far_mib": memory}
            # Internal aliases are consumed by the strict aggregate only.
            attempts.append({"example_id": row["example_id"], "repetition": repetition,
                             "token_ids": list(generation.token_ids), "score": score,
                             "latency": generation.elapsed_wall_seconds, "record": attempt})
            write_started = time.perf_counter(); _append_jsonl(predictions_path, attempt)
            writing_seconds += max(0.0, time.perf_counter() - write_started)
    if not _same_rng(baseline_rng, torch, np, cuda=use_cuda_rng):
        raise ValueError("global Python/NumPy/Torch RNG changed during measured repetitions")
    rng_after_snapshot, rng_after = _rng_snapshot(torch, np, cuda=use_cuda_rng)
    if rng_before != rng_after:
        raise ValueError("RNG fingerprints differ across the measured region")
    original_hash_after = _hash_original_parameters(model, helper)
    if original_hash_before != original_hash_after:
        raise ValueError("original frozen model parameters changed during evaluation")
    public_attempts = [{**item, "score": item["score"]} for item in attempts]
    summary = _summarize_attempts(public_attempts, expected_examples=len(selected), repetitions=reps)
    if warm.decoder_calls + summary["decoder_calls_total"] > config["limits"]["maximum_decoder_calls_including_warmup"]:
        raise ValueError("registered maximum decoder-call budget exceeded")
    latency = {}
    for example_id in (row["example_id"] for row in selected):
        values = [item["latency"] for item in attempts if item["example_id"] == example_id]
        latency[example_id] = {"seconds": values, "median": statistics.median(values),
                               "minimum": min(values), "maximum": max(values)}
    summary.update({"status": PASS_STATUS, "cohort_sha256": reg["cohort_sha256"],
        "parameter_counts": parameter_counts,
        "outcome_status_scope": "pre_terminal_metrics",
        "quality_scope": "repetition0_fixed15_descriptive_only", "latency_by_example": latency,
        "warmup_seconds": warm_seconds, "preparation_seconds": preparation_seconds,
        "scoring_seconds": scoring_seconds, "prediction_write_seconds": writing_seconds,
        "model_load_seconds": load_seconds, "setup_seconds": max(0.0, _set_stage_start - guard.started),
        "run_peak_memory_mib": _decode_memory(torch, device), "original_parameter_hashes_before": original_hash_before,
        "original_parameter_hashes_after": original_hash_after, "rng_before_repetitions": rng_before,
        "rng_after_repetitions": rng_after, "actual_numerics_policy": actual_policy,
        "profiler_flops": None, "profiler": None,
        "decoder_calls_including_warmup": warm.decoder_calls + summary["decoder_calls_total"],
        "claim_boundary": "operational frozen reference only; no transfer, accuracy-population, or efficiency claim"})
    return summary


def run_reference(config_path: str | os.PathLike[str], *, command: list[str]) -> dict[str, Any]:
    """Execute the one fixed source-validation original-backbone reference."""
    from dwr.experiments import gpu_feasibility as helper
    started = time.monotonic(); guard = helper.make_probe_guard(900, 12288); guard.started = started
    guard.check(); record: RunRecord | None = None; output: Path | None = None; phase = "metadata"
    try:
        if type(command) is not list or not command or any(type(item) is not str or not item for item in command):
            raise ValueError("command must be a nonempty list of strings")
        root = _repo_root()
        reg = _load_registration(root, config_path); config = reg["config"]
        full = reg["feasibility_config"]
        source = helper.validate_clean_source(root, config["source_base_commit"])
        output = _validate_output_directory(root, config["output_dir"])
        identity_meta = {
            "source": ("dwr-source-commit", source["commit"], _sha(source["commit"].encode("ascii"))),
            "config": ("dwr-frozen-reference-config", config["config_version"], reg["config_sha256"]),
            "cohort": ("frozen-source-validation-cohort", reg["cohort"]["schema_version"], reg["cohort_sha256"]),
            "model": (full["model"]["repo_id"], full["model"]["revision"], full["model"]["weight_sha256"]),
            "tokenizer": ("pinned-local-original-tokenizer", full["model"]["revision"], full["model"]["artifact_inventory_sha256"]),
            "data": ("source-validation-and-token-audit-receipts", "pinned-v1", config["data"]["source_export_manifest_sha256"]),
            "runtime": ("gpu-feasibility-operational-helpers", "reviewed-v2", FEASIBILITY_MODULE_SHA256),
        }
        identities = {key: {"name": name, "revision": revision, "sha256": digest}
                      for key, (name, revision, digest) in identity_meta.items()}
        # Resolve and record the complete original runtime contract before any CUDA or task access.
        full_path = root.joinpath(*PurePosixPath(config["dependencies"]["feasibility_config_path"]).parts)
        full_raw = _read(full_path, "full feasibility config")
        if _sha(full_raw) != FEASIBILITY_CONFIG_SHA256:
            raise ValueError("full feasibility config bytes differ from the reviewed pin")
        full = helper.validate_config_bytes(full_raw)
        settings = _resolved_settings(config, full, source)
        record = RunRecord.create(output, command=command, config_path=reg["config_path"],
            expected_config_sha256=CONFIG_SHA256, identities=identities, rng_streams=config["seeds"],
            settings=settings, budgets={"wall_seconds": config["limits"]["wall_seconds"],
                "memory_mib": config["limits"]["resident_memory_mib"], "attempts": config["limits"]["attempts"]},
            confirmatory=False, source_root=root)
        record.append_event("initial_manifest_written", {"source_commit": source["commit"],
            "config_sha256": CONFIG_SHA256, "cohort_sha256": COHORT_SHA256, "source_clean": source["clean"]})
        record.append_event("precondition_access_started", {"protected_task_semantics_read": False,
            "source_split": "source_validation", "feasibility_manifest_sha256": config["dependencies"]["feasibility_manifest_sha256"]})
        phase = "preflight"
        check = guard.check
        feasibility = _validate_feasibility(root, config, helper)
        data_receipt = helper._data_preconditions(root, feasibility["config"])
        artifact_receipt = helper._artifact_inventory(root, feasibility["config"])
        if data_receipt["source_export_manifest_sha256"] != config["data"]["source_export_manifest_sha256"] or data_receipt["token_audit_manifest_sha256"] != helper.TOKEN_AUDIT_MANIFEST_SHA256:
            raise ValueError("source or token-audit receipt differs from the reference registration")
        record.append_event("precondition_access_verified", {"source_export_manifest_sha256": data_receipt["source_export_manifest_sha256"],
            "token_audit_manifest_sha256": data_receipt["token_audit_manifest_sha256"],
            "model_manifest_sha256": artifact_receipt["manifest_sha256"], "protected_task_semantics_read": False})
        record.append_event("source_validation_access_started", {"split": "source_validation",
            "expected_manifest_sha256": config["data"]["source_export_manifest_sha256"], "expected_records": config["data"]["reader_expected_records"]})
        rows = _read_validation_rows(root, config)
        if len(rows) != config["data"]["reader_expected_records"]:
            raise ValueError("source validation reader count differs from the preregistered 265 records")
        selected = _select_source_rows(rows, reg["cohort"])
        phase = "hardware"
        if f"{sys.version_info.major}.{sys.version_info.minor}" != feasibility["config"]["environment"]["python_major_minor"]:
            raise RuntimeError("ENVIRONMENT_BLOCKED: Python version differs from registration")
        packages = helper._check_environment(feasibility["config"])
        hardware = helper._hardware_metadata(feasibility["config"])
        record.append_event("hardware_preflight", {"packages": packages, "hardware": hardware})
        phase = "load"
        import torch
        device = _execution_device(feasibility["config"], torch)
        _guard, check = _make_guard(helper, config, torch, guard, device)
        phase = "evaluation"
        summary = _run_model(root, reg, helper, feasibility["config"], selected, record, _guard, check, device=device)
        summary["elapsed_seconds_before_terminal_record_finalization"] = time.monotonic() - started
        summary["process_peak_working_set_bytes"] = _guard.peak_bytes
        summary["wall_cap_seconds"] = config["limits"]["wall_seconds"]
        summary["cpu_memory_cap_mib"] = config["limits"]["resident_memory_mib"]
        summary["elapsed_measurement_scope"] = "full invocation including preparation, writing, loading and warmup"
        _write_json(output / "summary.json", summary)
        record.append_event("reference_complete_pre_terminal", {"status": PASS_STATUS,
            "attempts": summary["attempt_count"], "quality_denominator": summary["quality"]["denominator"],
            "profiler_flops": None, "pre_terminal": True})

        def terminal_budget_check(stage: str) -> None:
            check()
            if stage == "artifacts_complete":
                record.append_event("terminal_artifact_budget_check", {"elapsed_seconds_before_terminal_record_finalization": time.monotonic() - started,
                    "process_peak_working_set_bytes_before_terminal_record_finalization": _guard.peak_bytes,
                    "artifact_hashing_complete": True, "status": "pre_terminal_metrics"})
                check()
        record.finish("completed", counts={PASS_STATUS: 1, "attempts": summary["attempt_count"]},
            artifacts=["warmup.json", "predictions.jsonl", "summary.json"], budget_check=terminal_budget_check)
        return summary
    except BaseException as exc:
        if record is not None:
            _preserve_failure(record, exc, interrupted=isinstance(exc, (KeyboardInterrupt, SystemExit, TimeoutError, MemoryError)))
        raise


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", "dwr.experiments.frozen_reference", "--config", args.config]
    run_reference(args.config, command=command)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
