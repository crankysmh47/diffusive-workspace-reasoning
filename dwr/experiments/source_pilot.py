"""Strict worker and audit controller for the registered source-validation pilot."""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


_ARM_ORDER = ("K0", "K1", "K4")
_IDENTITY_KEYS = {"source", "config", "data", "model", "tokenizer", "schedule"}
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in _HEX for ch in value)


def _verify_generated_text(envelope: Mapping[str, Any], tokenizer: Any, *, model_vocab_size: int,
                          token_cap: int = 128) -> None:
    """Re-decode a stored output with the live pinned tokenizer and model token bounds."""
    if not isinstance(envelope, Mapping) or type(model_vocab_size) is not int or model_vocab_size <= 0 or \
            type(token_cap) is not int or token_cap <= 0:
        raise ValueError("generated-token audit inputs are malformed")
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int or envelope.get("eos_token_id") != eos or envelope.get("vocab_size") != model_vocab_size:
        raise ValueError("stored generation EOS or model vocabulary differs from the live model/tokenizer")
    ids = envelope.get("generated_token_ids")
    score = envelope.get("score")
    if type(ids) is not list or not ids or any(type(token) is not int or not 0 <= token < model_vocab_size for token in ids):
        raise ValueError("stored generated token IDs are outside the pinned model vocabulary")
    if eos in ids[:-1]:
        raise ValueError("stored generated IDs contain EOS before the terminal position")
    if not isinstance(score, Mapping) or type(score.get("generated_tokens")) is not int or score["generated_tokens"] != len(ids):
        raise ValueError("stored generation token count differs from its token IDs")
    termination = score.get("termination")
    if termination not in {"eos", "token_cap"} or (termination == "eos") != (ids[-1] == eos) or \
            (termination == "token_cap" and len(ids) != token_cap):
        raise ValueError("stored generation termination differs from EOS/token-cap evidence")
    decoded_ids = ids[:-1] if ids[-1] == eos else ids
    try:
        decoded = tokenizer.decode(decoded_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
    except Exception as error:
        raise ValueError("live tokenizer could not re-decode stored generation") from error
    if not isinstance(decoded, str) or decoded != score.get("answer_text"):
        raise ValueError("live tokenizer re-decode differs from stored scored text")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> dict[str, str]:
    raw = _canonical(dict(value)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": path.as_posix(), "sha256": _sha_bytes(raw)}


def _arm_relative_ref(path: Path, root: Path) -> dict[str, str]:
    root = root.resolve(strict=True)
    lexical = Path(os.path.abspath(os.fspath(path)))
    relative = lexical.relative_to(root)
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise ValueError("audit artifact must be a contained regular file")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
            raise ValueError("audit artifact path contains a link or junction")
    absolute = lexical.resolve(strict=True)
    if not absolute.is_file():
        raise ValueError("audit artifact must be a contained regular file")
    return {"path": relative.as_posix(), "sha256": _sha_bytes(absolute.read_bytes())}


def _read_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON artifact: {path.name}") from error
    if not isinstance(value, dict) or _canonical(value) + b"\n" != raw:
        raise ValueError(f"JSON artifact is not canonical or has no final LF: {path.name}")
    return value, raw


def _ordered_id_digest(ids: list[str]) -> str:
    return _sha_bytes(_canonical(ids))


def _record_identities(registration: Mapping[str, Any], identities: Mapping[str, str],
                       source_commit: str, runtime_sha256: str) -> dict[str, dict[str, str]]:
    model = registration["model_pins"]
    return {
        "source_commit": {"name": "source-commit", "revision": source_commit,
            "sha256": _sha_bytes(source_commit.encode("ascii"))},
        "pilot_config": {"name": "source-validation-pilot-config", "revision": registration["config"]["config_version"],
            "sha256": registration["config_sha256"]},
        "cohort": {"name": "source-validation-pilot-cohort", "revision": "v1",
            "sha256": registration["cohort_sha256"]},
        "model": {"name": model["repo_id"], "revision": model["revision"], "sha256": identities["model"]},
        "tokenizer": {"name": model["repo_id"], "revision": model["revision"], "sha256": identities["tokenizer"]},
        "runtime": {"name": "source-validation-pilot-runtime", "revision": "registered-v1",
            "sha256": runtime_sha256},
    }


def _checked_host_config(registration: Mapping[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(registration["feasibility_config"])
    seeds = registration["config"]["seeds"]
    for name in ("python_global", "numpy_global", "torch_cpu_global", "torch_cuda_global",
                 "workspace_initialization", "reader_initialization"):
        if name not in seeds or name not in resolved["seeds"]:
            raise ValueError("pilot seed binding does not cover the inherited host policy")
        resolved["seeds"][name] = seeds[name]
    return resolved


def _install_registered_global_rngs(config: Mapping[str, Any], torch: Any, numpy: Any) -> None:
    seeds = config["seeds"]
    random.seed(seeds["python_global"])
    numpy.random.seed(seeds["numpy_global"])
    torch.manual_seed(seeds["torch_cpu_global"])
    torch.cuda.manual_seed_all(seeds["torch_cuda_global"])


_CONTROLLER_POLICY_KEYS = {"validation_examples", "optimizer_steps", "expected_original_parameter_count",
    "expected_parameter_counts", "expected_model_vocab_size", "expected_tokenizer_vocab_size",
    "expected_eos_token_id", "decoder_max_new_tokens"}


def _registered_controller_policy(registration: Mapping[str, Any]) -> dict[str, Any]:
    """Fixed production policy; the private worker core may receive scaled fixture policy."""
    config = registration["config"]
    model = registration["model_pins"]
    feasibility = registration["feasibility_config"]
    vocab = feasibility["synthetic_input"]["embedding_vocab_size"]
    policy = {"validation_examples": 45, "optimizer_steps": 512,
        "expected_original_parameter_count": 338,
        "expected_parameter_counts": {"original_parameter_elements": 1543714304,
            "workspace_parameter_elements": 10629888, "reader_parameter_elements": 21265923,
            "new_parameter_elements": 31895811, "retained_total_parameter_elements": 1575610115},
        "expected_model_vocab_size": vocab, "expected_tokenizer_vocab_size": 151665,
        "expected_eos_token_id": 151643, "decoder_max_new_tokens": 128}
    if config["data"]["validation_examples"] != policy["validation_examples"] or \
            config["training"]["optimizer_steps"] != policy["optimizer_steps"] or \
            model["model_config_sha256"] != feasibility["model"]["config_sha256"] or \
            config["decoding"]["embedding_vocab_size"] != vocab or \
            config["decoding"]["eos_token_id"] != policy["expected_eos_token_id"] or \
            config["decoding"].get("max_new_tokens", config["decoding"].get("max_new_tokens_including_eos")) != policy["decoder_max_new_tokens"]:
        raise ValueError("registered production controller policy differs from its fixed host contract")
    return policy


def _validate_controller_policy(policy: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, Mapping) or set(policy) != _CONTROLLER_POLICY_KEYS:
        raise ValueError("controller policy has unknown or missing fields")
    result = dict(policy)
    if type(result["validation_examples"]) is not int or result["validation_examples"] <= 0 or \
            type(result["optimizer_steps"]) is not int or result["optimizer_steps"] <= 0 or \
            type(result["expected_original_parameter_count"]) is not int or result["expected_original_parameter_count"] <= 0 or \
            type(result["expected_model_vocab_size"]) is not int or result["expected_model_vocab_size"] <= 0 or \
            type(result["expected_tokenizer_vocab_size"]) is not int or result["expected_tokenizer_vocab_size"] <= 0 or \
            type(result["expected_eos_token_id"]) is not int or result["expected_eos_token_id"] < 0 or \
            type(result["decoder_max_new_tokens"]) is not int or result["decoder_max_new_tokens"] <= 0:
        raise ValueError("controller policy contains invalid integer bounds")
    expected_counts = result["expected_parameter_counts"]
    count_keys = {"original_parameter_elements", "workspace_parameter_elements", "reader_parameter_elements",
        "new_parameter_elements", "retained_total_parameter_elements"}
    if not isinstance(expected_counts, Mapping) or set(expected_counts) != count_keys or any(
            type(value) is not int or value <= 0 for value in expected_counts.values()):
        raise ValueError("controller parameter-count policy is malformed")
    if expected_counts["new_parameter_elements"] != expected_counts["workspace_parameter_elements"] + expected_counts["reader_parameter_elements"] or \
            expected_counts["retained_total_parameter_elements"] != expected_counts["original_parameter_elements"] + expected_counts["new_parameter_elements"]:
        raise ValueError("controller parameter-count policy is internally inconsistent")
    decode_cap = config["decoding"].get("max_new_tokens", config["decoding"].get("max_new_tokens_including_eos"))
    if result["validation_examples"] != config["data"]["validation_examples"] or \
            result["optimizer_steps"] != config["training"]["optimizer_steps"] or \
            result["expected_model_vocab_size"] != config["decoding"]["embedding_vocab_size"] or \
            result["expected_eos_token_id"] != config["decoding"]["eos_token_id"] or \
            result["decoder_max_new_tokens"] != decode_cap:
        raise ValueError("controller policy differs from the exact resolved pilot decoder/config cap")
    return result


def _source_access(record: Any, registration: Mapping[str, Any], *, split: str) -> None:
    config, cohort = registration["config"], registration["cohort"]
    if split == "source_validation":
        ids = [row["example_id"] for row in cohort["selected"]]
        purpose = "registered_source_validation_for_kernel_and_arm_audit"
    elif split == "source_train":
        ids = sorted(row["example_id"] for row in registration["token_rows"] if row["split"] == "source_train")
        purpose = "registered_source_training_for_kernel"
    else:
        raise ValueError("controller attempted an unregistered source split")
    record.append_event("source_examples_access_authorized", {"split": split,
        "record_count": len(ids), "ordered_ids_sha256": _ordered_id_digest(ids),
        "cohort_sha256": registration["cohort_sha256"], "purpose": purpose})


def _check_live_memory(config: Mapping[str, Any], torch: Any | None, *, device: str) -> dict[str, Any]:
    from dwr.training.source_pilot import _memory_measurement
    resident = _memory_measurement()
    resident_mib = resident["value_bytes"] / (1024 * 1024)
    if resident_mib > config["limits"]["resident_memory_mib"]:
        raise MemoryError("registered pilot resident-memory cap exceeded")
    allocated = reserved = None
    if torch is not None and torch.cuda.is_initialized():
        if not isinstance(device, str) or not device:
            raise ValueError("live CUDA memory guard requires the inherited registered device")
        torch.cuda.synchronize(device)
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
        if allocated > config["limits"]["maximum_peak_allocated_mib"] * 1024**2:
            raise MemoryError("registered pilot allocated-memory cap exceeded")
        if reserved > config["limits"]["maximum_peak_reserved_mib"] * 1024**2:
            raise MemoryError("registered pilot reserved-memory cap exceeded")
    return {"resident": resident, "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved, "scope": "controller_high_water_observation"}


def _hash_original_parameters_live(host: Any, check: Any, *, chunk_bytes: int = 64 * 1024**2,
                                   guard_stage: str = "work") -> dict[str, str]:
    """Hash the pinned backbone in bounded transfers while servicing worker guards."""
    import torch

    if type(chunk_bytes) is not int or chunk_bytes < 1024:
        raise ValueError("parameter hash chunk must be a positive bounded byte count")
    result: dict[str, str] = {}
    for name, parameter in host.backbone.named_parameters():
        if not parameter.is_contiguous():
            raise ValueError(f"original parameter {name} is not contiguous for canonical byte hashing")
        digest = hashlib.sha256()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(str(parameter.dtype).encode("ascii"))
        flat_bytes = parameter.detach().view(-1).view(torch.uint8)
        for offset in range(0, flat_bytes.numel(), chunk_bytes):
            check(guard_stage)
            part = flat_bytes[offset:offset + chunk_bytes].to(device="cpu").numpy().tobytes()
            digest.update(part)
        result[name] = digest.hexdigest()
    check(guard_stage)
    return result


def run_worker(descriptor_path: str | os.PathLike[str], descriptor_sha256: str, *, command: list[str]) -> dict[str, Any]:
    """Authorize, publish a process receipt, then execute one registered arm segment."""
    root = Path(__file__).resolve().parents[2]
    from dwr.experiments import source_pilot_job as job
    authorization = job.read_worker_authorization(root, descriptor_path, descriptor_sha256)
    receipt = job.publish_worker_receipt(authorization)
    try:
        return _execute_authorized_worker(authorization, receipt, command=command)
    finally:
        receipt["worker_lock"].close()


def _execute_authorized_worker(authorization: Mapping[str, Any], receipt_publication: Mapping[str, Any], *,
                               command: list[str], factories: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Private execution core; test factories are deliberately unreachable from the CLI."""
    from dwr.experiments.source_pilot_registration import pilot_identities, read_pilot_examples

    root = Path(authorization["root"]).resolve(strict=True)
    arm_root = Path(authorization["arm_root"]).resolve(strict=True)
    segment = Path(authorization["segment_dir"]).resolve(strict=True)
    authorization_registration = authorization["registration"]
    registration = factories.get("registration", authorization_registration) if factories is not None else authorization_registration
    config = registration["config"]
    policy = _validate_controller_policy(factories["policy"], config) if factories is not None else _registered_controller_policy(registration)
    host_config_for_guard = factories["host_config"] if factories is not None else _checked_host_config(registration)
    live_guard_device = host_config_for_guard["hardware"]["device"]
    if not isinstance(live_guard_device, str) or not live_guard_device:
        raise ValueError("worker host policy lacks its exact registered device")
    arm_id = authorization["arm_id"]
    source_commit = authorization["bindings"]["source"]["commit"]
    runtime = authorization["bindings"]["runtime"]
    runtime_sha256 = runtime["receipt_sha256"]
    identities = authorization["identities"]
    identity_resolver = factories.get("pilot_identities", pilot_identities) if factories is not None else pilot_identities
    if identities != identity_resolver(registration, source_commit=source_commit, arm_id=arm_id):
        raise ValueError("worker checkpoint identities differ from the closed pilot registration")
    if set(authorization) != {"schema_version", "root", "descriptor_path", "descriptor_sha256", "descriptor",
            "registration", "matrix_root", "arm_root", "segment_dir", "arm_id", "arm_index", "arm",
            "segment_index", "nonce", "identities", "bindings", "clocks", "deadlines", "resume_descriptor",
            "expected_launcher"}:
        raise ValueError("worker authorization has unknown or missing fields")
    if arm_id not in _ARM_ORDER or authorization["arm_index"] != _ARM_ORDER.index(arm_id):
        raise ValueError("worker authorization arm order differs from the registered matrix")
    if registration["config_sha256"] != authorization["bindings"]["matrix"]["config_sha256"] or \
            registration["cohort_sha256"] != authorization["bindings"]["matrix"]["cohort_sha256"]:
        raise ValueError("worker matrix bindings differ from the loaded registration")
    if not isinstance(receipt_publication, Mapping) or not isinstance(receipt_publication.get("receipt"), Mapping):
        raise ValueError("worker receipt publication is malformed")
    if Path(receipt_publication.get("receipt_path", "")).resolve(strict=True) != segment / "worker-receipt.json":
        raise ValueError("worker receipt is outside its authorized segment")

    receipt_ref = _publish_job_receipt_inside_run(segment, arm_root, authorization, receipt_publication)
    from dwr.experiments import source_pilot_job as job
    record = job.open_segment_record(dict(authorization))
    record.append_event("worker_receipt_published", {"receipt": receipt_ref,
        "descriptor_sha256": authorization["descriptor_sha256"], "nonce": authorization["nonce"]})
    torch = None
    host = tokenizer = None
    started = time.perf_counter()
    initial_hashes = None
    observation = None
    phase = "preflight"
    job = __import__("dwr.experiments.source_pilot_job", fromlist=["check_worker_budget"])

    def check(stage: str = "work") -> dict[str, Any]:
        job.check_worker_budget(dict(authorization), stage)
        return _check_live_memory(config, torch, device=live_guard_device)

    try:
        # The authorized event is fsynced before either exact split is parsed.
        _source_access(record, registration, split="source_validation")
        if factories is None:
            validation_examples, validation_meta = read_pilot_examples(root, registration, split="source_validation")
        else:
            validation_examples, validation_meta = factories["read_examples"](root, registration, split="source_validation")
        val_ids = [example.example_id for example in validation_examples]
        if val_ids != [row["example_id"] for row in registration["cohort"]["selected"]] or len(val_ids) != policy["validation_examples"]:
            raise ValueError("authorized validation examples differ from the fixed metadata cohort")
        resume = None
        resume_reference = authorization["resume_descriptor"]
        if resume_reference is not None:
            if resume_reference.get("source_commit") != source_commit or resume_reference.get("arm_id") != arm_id:
                raise ValueError("resume descriptor source or arm binding differs")
            cursor_path = (arm_root / "recovery.latest.json").resolve(strict=True)
            if cursor_path != Path(resume_reference["path"]).resolve(strict=True) or _sha_bytes(cursor_path.read_bytes()) != resume_reference["sha256"]:
                raise ValueError("resume cursor differs from the authorized immutable reference")
            from dwr.training.pilot_recovery import PilotRecoveryStore
            resume = {"jobroot": str(arm_root), "cursor_path": str(cursor_path),
                "cursor_sha256": resume_reference["sha256"]}
            PilotRecoveryStore.verify_descriptor(resume, identities, arm_id=arm_id,
                selected_ids=tuple(val_ids), metadata=validation_meta,
                examples=tuple(validation_examples), steps=policy["optimizer_steps"])

        _source_access(record, registration, split="source_train")
        if factories is None:
            training_examples, training_meta = read_pilot_examples(root, registration, split="source_train")
        else:
            training_examples, training_meta = factories["read_examples"](root, registration, split="source_train")
        if len(training_examples) != config["data"]["source_train_records"] or \
                [example.example_id for example in training_examples] != sorted(example.example_id for example in training_examples):
            raise ValueError("authorized source-training rows are incomplete or not in frozen order")

        if factories is None:
            from dwr.experiments import gpu_feasibility
            host_config = _checked_host_config(registration)
            environment = gpu_feasibility._check_environment(host_config)
            hardware = gpu_feasibility._hardware_metadata(host_config)
            numerical_policy = gpu_feasibility._set_numerics(host_config)
            record.append_event("runtime_hardware_preflight", {"environment": environment,
                "hardware": hardware, "numerical_policy": numerical_policy})
            phase = "model_inventory"
            check()
            inventory = gpu_feasibility._artifact_inventory(root, host_config)
            if inventory["weight_sha256"] != registration["model_pins"]["weight_sha256"] or \
                    inventory["inventory_sha256"] != registration["model_pins"]["artifact_inventory_sha256"]:
                raise ValueError("actual local pretrained inventory differs from the registered model/tokenizer pin")
            import numpy as np
            import torch as torch_module
            torch = torch_module
            _install_registered_global_rngs(config, torch_module, np)
            torch.cuda.synchronize(host_config["hardware"]["device"])
            torch.cuda.reset_peak_memory_stats(host_config["hardware"]["device"])
            phase = "load"
            host = gpu_feasibility._load_production_host(root, host_config)
            from transformers import AutoTokenizer
            model_dir = gpu_feasibility._resolve_registered(root, host_config["model"]["local_dir"])
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        else:
            torch, host, tokenizer = factories["torch"], factories["host"](), factories["tokenizer"]()
            host_config = factories["host_config"]
            environment = hardware = numerical_policy = {"fixture": True}
            inventory = factories.get("model_inventory", {"weight_sha256": registration["model_pins"]["weight_sha256"],
                "inventory_sha256": registration["model_pins"]["artifact_inventory_sha256"]})
        check()
        if getattr(tokenizer, "eos_token_id", None) != policy["expected_eos_token_id"] or \
                getattr(tokenizer, "vocab_size", None) != policy["expected_tokenizer_vocab_size"]:
            raise ValueError("live tokenizer EOS differs from the registered EOS ID")
        vocab_size = int(host.backbone.config.vocab_size)
        if vocab_size != policy["expected_model_vocab_size"]:
            raise ValueError("loaded model vocabulary differs from the registered embedding vocabulary")
        if inventory["weight_sha256"] != identities["model"] or inventory["inventory_sha256"] != identities["tokenizer"]:
            raise ValueError("actual model/tokenizer artifact identities differ from the worker binding")
        initial_hashes = _hash_original_parameters_live(host, check)
        parameter_counts = _parameter_counts(host)
        if len(initial_hashes) != policy["expected_original_parameter_count"] or parameter_counts != policy["expected_parameter_counts"]:
            raise ValueError("loaded production host parameter counts differ from the registered Qwen/workspace contract")
        if factories is None:
            from dwr.training.batches import TokenizerIdentity
            tokenizer_identity = TokenizerIdentity(registration["model_pins"]["repo_id"], registration["model_pins"]["revision"])
        else:
            tokenizer_identity = factories["tokenizer_identity"]
        from dwr.training.source_pilot import _stable_module_digest
        initial_module_sha256, initial_module_recipe = _stable_module_digest(host)
        record.append_event("controller_host_initialized", {"arm_id": arm_id,
            "initial_module_sha256": initial_module_sha256, "initial_module_recipe": initial_module_recipe,
            "parameter_counts": parameter_counts, "original_parameter_count": len(initial_hashes),
            "model_vocab_size": vocab_size, "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
            "eos_token_id": tokenizer.eos_token_id})
        initial_audit_dir = segment / "controller-audit"
        initial_audit_path = initial_audit_dir / "initial-frozen.json"
        initial_audit_value = {"schema": "dwr-source-pilot-initial-frozen-v1", "arm_id": arm_id,
            "identities": dict(identities), "original_parameter_hashes": initial_hashes}
        _write_exclusive_json(initial_audit_path, initial_audit_value)
        initial_frozen_ref = _arm_relative_ref(initial_audit_path, arm_root)
        def before_complete(audit_payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
            check("seal")
            if audit_payload.get("arm_id") != arm_id or audit_payload.get("identities") != identities or \
                    audit_payload.get("initial_state_sha256") != initial_module_sha256:
                raise ValueError("kernel completion audit differs from the live initialized arm")
            final_hashes = _hash_original_parameters_live(host, check, guard_stage="seal")
            if final_hashes != initial_hashes or len(final_hashes) != len(initial_hashes):
                raise ValueError("same-live-host original parameter bytes changed before completion")
            _audit_prediction_tokens(arm_root, audit_payload["stage_receipts"], tokenizer, vocab_size,
                                     tokenizer.eos_token_id, token_cap=policy["decoder_max_new_tokens"])
            final_counts = _parameter_counts(host)
            expected_trainable = final_counts["reader_parameter_elements"] if arm_id == "K0" else final_counts["new_parameter_elements"]
            actual_trainable = sum(parameter.numel() for parameter in host.parameters() if parameter.requires_grad)
            if actual_trainable != expected_trainable or final_counts["retained_total_parameter_elements"] != parameter_counts["retained_total_parameter_elements"]:
                raise ValueError("live active/retained parameter counts differ from the registered control method")
            memory = check("seal")
            audit_dir = segment / "controller-audit"
            observation = _controller_observation(started, memory, stage="complete")
            observation_path = audit_dir / "controller-observation.json"
            _write_exclusive_json(observation_path, observation)
            observation_ref = _arm_relative_ref(observation_path, arm_root)
            final_audit_path = audit_dir / "final-controller.json"
            final_audit = {"schema": "dwr-source-pilot-controller-audit-v1", "arm_id": arm_id,
                "identities": dict(identities), "initial_frozen": initial_frozen_ref,
                "initial_state_sha256": initial_module_sha256,
                "initial_checkpoint": audit_payload["initial_checkpoint"],
                "final_checkpoint": audit_payload["final_checkpoint"],
                "stage_receipts": audit_payload["stage_receipts"], "bank": audit_payload["bank"],
                "original_parameter_hashes": final_hashes,
                "details": {"parameter_counts": final_counts, "trainable_parameters": actual_trainable,
                    "model_vocab_size": vocab_size, "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
                    "eos_token_id": tokenizer.eos_token_id, "initial_module_sha256": initial_module_sha256,
                    "initial_module_recipe": initial_module_recipe, "source_commit": source_commit,
                    "runtime_sha256": runtime_sha256, "controller_observation": observation_ref}}
            _write_exclusive_json(final_audit_path, final_audit)
            final_ref = _arm_relative_ref(final_audit_path, arm_root)
            record.append_event("controller_audit", {"arm_id": arm_id,
                "initial_frozen": initial_frozen_ref, "final_controller": final_ref,
                "original_parameter_count": len(final_hashes), "parameter_counts": final_counts,
                "model_vocab_size": vocab_size, "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None)})
            return {"initial_frozen": initial_frozen_ref, "final_controller": final_ref}

        if factories is None:
            from dwr.training.source_pilot import run_pilot_arm
            kernel = run_pilot_arm
        else:
            kernel = factories["run_pilot_arm"]
        metadata = {"training": training_meta, "validation": validation_meta}
        def on_kernel_event(kind: str, payload: Mapping[str, Any]) -> None:
            # This callback is exposed only through the private worker core. A
            # fixture may interrupt after the authoritative event has been
            # durably appended, then resume through the ordinary job protocol.
            job.guard_worker_event(dict(authorization), record, kind, payload)
            if factories is not None and callable(factories.get("after_kernel_event")):
                factories["after_kernel_event"](kind, payload, record, authorization)

        output = kernel(host, tokenizer, tokenizer_identity, tuple(training_examples), tuple(validation_examples),
            registration=registration, arm_id=arm_id, identities=identities, metadata=metadata,
            output_dir=segment, check=check, on_event=on_kernel_event, before_complete=before_complete, resume=resume)
        check("seal")
        if output.get("status") != "PILOT_ARM_COMPLETE" or output.get("optimizer_step") != policy["optimizer_steps"]:
            raise ValueError("pilot kernel returned without the exact completed arm boundary")
        observation = _controller_observation(started, check("seal"), stage="segment_complete")
        _write_exclusive_json(segment / "controller-audit" / "segment-observation.json", observation)
        artifacts = _segment_artifacts(segment)
        terminal = record.finish("completed", counts=output.get("counts", {}), artifacts=artifacts,
            budget_check=check)
        manifest_ref = _arm_relative_ref(segment / "manifest.json", arm_root)
        return {"status": "completed", "arm_id": arm_id, "segment_index": authorization["segment_index"],
            "progress": {"optimizer_step": output["optimizer_step"]},
            "recovery_descriptor": output["recovery_descriptor"], "result": output,
            "record_manifest": manifest_ref}
    except KeyboardInterrupt as error:
        _seal_worker_failure(record, segment, error, status="interrupted", check=check, phase=phase)
        raise
    except BaseException as error:
        _seal_worker_failure(record, segment, error, status="failed", check=check, phase=phase)
        raise


def _publish_job_receipt_inside_run(segment: Path, arm_root: Path, authorization: Mapping[str, Any],
                                    publication: Mapping[str, Any]) -> dict[str, str]:
    receipt = publication["receipt"]
    expected = {"schema_version", "descriptor_sha256", "nonce", "matrix_root", "arm_root", "segment_dir",
        "arm_id", "segment_index", "source_commit", "config_sha256", "runtime_sha256", "worker", "parent",
        "launcher", "issued_utc"}
    if not isinstance(receipt, Mapping) or set(receipt) != expected or receipt["arm_id"] != authorization["arm_id"]:
        raise ValueError("published worker receipt differs from the closed process protocol")
    receipt_path = segment / "worker-receipt.json"
    value, raw = _read_json(receipt_path)
    if value != dict(receipt) or _sha_bytes(raw) != _sha_bytes(_canonical(dict(receipt)) + b"\n"):
        raise ValueError("durable worker receipt bytes differ from the published process identity")
    return _arm_relative_ref(receipt_path, arm_root)


def _parameter_counts(host: Any) -> dict[str, int]:
    original = sum(parameter.numel() for parameter in host.backbone.parameters())
    workspace = sum(parameter.numel() for parameter in host.workspace.parameters())
    readers = sum(parameter.numel() for parameter in host.readers.parameters())
    new = workspace + readers
    return {"original_parameter_elements": original, "workspace_parameter_elements": workspace,
        "reader_parameter_elements": readers, "new_parameter_elements": new,
        "retained_total_parameter_elements": original + new}


def _controller_observation(started: float, memory: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    return {"schema_version": "dwr-source-pilot-controller-observation-v1", "stage": stage,
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": max(0.0, time.perf_counter() - started), "memory": dict(memory)}


def _verify_controller_observation(value: Mapping[str, Any], limits: Mapping[str, Any], *, require_cuda: bool,
                                  stage: str = "complete") -> None:
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "stage", "observed_utc", "elapsed_seconds", "memory"} or \
            value.get("schema_version") != "dwr-source-pilot-controller-observation-v1" or value.get("stage") != stage:
        raise ValueError("controller observation schema or stage is invalid")
    elapsed = value.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("controller observation elapsed time is invalid")
    memory = value.get("memory")
    if not isinstance(memory, Mapping) or set(memory) != {"resident", "peak_allocated_bytes", "peak_reserved_bytes", "scope"} or \
            memory.get("scope") != "controller_high_water_observation":
        raise ValueError("controller observation memory schema is invalid")
    resident = memory.get("resident")
    if not isinstance(resident, Mapping) or set(resident) != {"scope", "value_bytes"} or \
            type(resident.get("value_bytes")) is not int or resident["value_bytes"] < 0 or \
            resident["value_bytes"] > limits["resident_memory_mib"] * 1024**2:
        raise ValueError("controller resident-memory observation exceeds or omits its registered cap")
    for key, limit in (("peak_allocated_bytes", "maximum_peak_allocated_mib"),
                       ("peak_reserved_bytes", "maximum_peak_reserved_mib")):
        count = memory.get(key)
        if count is None and not require_cuda:
            continue
        if type(count) is not int or count < 0 or count > limits[limit] * 1024**2:
            raise ValueError(f"controller {key} exceeds or omits its registered cap")


def _segment_artifacts(segment: Path) -> list[str]:
    reserved = {"manifest.initial.json", "manifest.json", "events.jsonl"}
    result = []
    for path in sorted(segment.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("worker segment contains a symlink/reparse artifact")
        if path.is_file() and path.name not in reserved:
            result.append(path.relative_to(segment).as_posix())
    return result


def _seal_worker_failure(record: Any, segment: Path, error: BaseException, *, status: str, check: Any,
                         phase: str = "unknown") -> None:
    try:
        if not isinstance(error, KeyboardInterrupt):
            record.append_event("known_fault", {"exception_type": type(error).__name__, "phase": phase,
                "message_sha256": _sha_bytes(str(error).encode("utf-8", errors="replace"))})
    except BaseException:
        # Keep the original fault authoritative; sealing below is still attempted.
        pass
    try:
        if not (segment / "manifest.json").exists():
            record.finish(status, counts={}, artifacts=_segment_artifacts(segment),
                error=f"{type(error).__name__}: {error}", budget_check=check)
    except BaseException:
        # An unsealed record remains explicit failure evidence; do not mask the original fault.
        pass


def _contained_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("artifact path must be a nonempty POSIX relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("artifact path escapes its arm root")
    path = root.joinpath(*pure.parts)
    if not path.is_file():
        raise ValueError("artifact reference is not a regular file")
    cursor = root.resolve(strict=True)
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
            raise ValueError("artifact reference contains a link or junction")
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    return resolved


def _checked_reference(root: Path, value: object, label: str) -> tuple[Path, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"} or not _is_sha256(value.get("sha256")):
        raise ValueError(f"{label} reference has an invalid schema")
    path = _contained_file(root, value["path"])
    if _sha_bytes(path.read_bytes()) != value["sha256"]:
        raise ValueError(f"{label} bytes differ from their reference")
    return path, {"path": value["path"], "sha256": value["sha256"]}


def _audit_prediction_tokens(arm_root: Path, stage_references: Mapping[str, Any], tokenizer: Any,
                             vocab_size: int, eos_token_id: int, *, token_cap: int = 128) -> None:
    for stage, reference in stage_references.items():
        receipt_path, _ = _checked_reference(arm_root, reference, f"{stage} stage receipt")
        receipt, _ = _read_json(receipt_path)
        records_path, _ = _checked_reference(arm_root, receipt.get("records"), f"{stage} prediction records")
        raw = records_path.read_bytes()
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                raise ValueError("prediction envelope is missing its terminal LF")
            try:
                envelope = json.loads(line[:-1].decode("utf-8", errors="strict"),
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError("prediction envelope is not strict UTF-8 JSON") from error
            if not isinstance(envelope, dict) or _canonical(envelope) != line[:-1]:
                raise ValueError("prediction envelope is not canonical JSON")
            if envelope.get("eos_token_id") != eos_token_id or envelope.get("vocab_size") != vocab_size:
                raise ValueError("prediction EOS/model-vocabulary bounds differ from the live pinned host")
            _verify_generated_text(envelope, tokenizer, model_vocab_size=vocab_size, token_cap=token_cap)


def _read_events(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    events = []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ValueError("RunRecord event stream has an incomplete final row")
        try:
            value = json.loads(line[:-1].decode("utf-8", errors="strict"),
                parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("RunRecord event stream has invalid canonical JSON") from error
        if not isinstance(value, dict) or _canonical(value) != line[:-1] or set(value) != {"sequence", "utc", "kind", "payload"}:
            raise ValueError("RunRecord event row has an invalid canonical envelope")
        if type(value["sequence"]) is not int or value["sequence"] != len(events) + 1 or \
                not isinstance(value["utc"], str) or not isinstance(value["kind"], str) or not isinstance(value["payload"], dict):
            raise ValueError("RunRecord event kind or payload is malformed")
        events.append(value)
    return events


def _pair_committed_updates(event_segments: list[tuple[int, list[dict[str, Any]]]], *, expected_steps: int = 512) -> list[dict[str, Any]]:
    pairs: dict[int, dict[str, Any]] = {}
    for _, events in event_segments:
        pending: dict[int, list[dict[str, Any]]] = {}
        for event in events:
            kind, payload = event["kind"], event["payload"]
            if kind == "update_completed":
                step = payload.get("net_step")
                if type(step) is not int or step <= 0:
                    raise ValueError("completed update lacks a positive net step")
                if pending.get(step):
                    raise ValueError("segment contains duplicate completed updates for one optimizer step")
                pending.setdefault(step, []).append(payload)
            elif kind == "recovery_committed":
                progress = payload.get("progress")
                step = progress.get("optimizer_step") if isinstance(progress, Mapping) else None
                if type(step) is not int or step <= 0:
                    continue
                if step in pairs:
                    if pending.get(step):
                        raise ValueError("optimizer step has duplicate durable completion evidence")
                    continue
                updates = pending.get(step, [])
                if not updates:
                    raise ValueError("durable optimizer commit has no preceding completed update in its segment")
                selected = updates[-1]
                pending.pop(step, None)
                for field in ("example_id", "batch_sha256", "noise_sha256"):
                    if not isinstance(selected.get(field), str) or not selected[field]:
                        raise ValueError("committed update pairing evidence is incomplete")
                pairs[step] = {"optimizer_step": step, "example_id": selected["example_id"],
                    "batch_sha256": selected["batch_sha256"], "noise_sha256": selected["noise_sha256"]}
    if sorted(pairs) != list(range(1, expected_steps + 1)):
        raise ValueError("durable event streams do not prove all registered unique committed optimizer updates")
    return [pairs[step] for step in range(1, expected_steps + 1)]


def _recompute_learning_gate(config: Mapping[str, Any], initial: Mapping[str, Any], final: Mapping[str, Any]) -> bool:
    gate = config["evaluation"]["learning_gate"]
    primitive = final.get("primitive")
    if not isinstance(primitive, Mapping) or set(primitive) != {"eq", "neq", "add_mod"}:
        raise ValueError("K4 final metrics omit a registered primitive")
    rates = []
    primitive_missing = False
    for name in ("eq", "neq", "add_mod"):
        item = primitive[name]
        if not isinstance(item, Mapping):
            raise ValueError("K4 primitive rates are malformed")
        rate = item.get("rate")
        if "valid" in item or "denominator" in item:
            valid, denominator = item.get("valid"), item.get("denominator")
            if isinstance(valid, bool) or not isinstance(valid, int) or \
                    isinstance(denominator, bool) or not isinstance(denominator, int) or \
                    denominator < 0 or valid < 0 or valid > denominator:
                raise ValueError("K4 primitive counts are malformed")
            if denominator == 0:
                if valid != 0 or rate is not None:
                    raise ValueError("K4 primitive counts and rates are malformed")
                primitive_missing = True
                continue
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or \
                    not 0.0 <= rate <= 1.0 or not math.isclose(rate, valid / denominator, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("K4 primitive counts and rates are malformed")
        elif isinstance(rate, bool) or not isinstance(rate, (int, float)) or \
                not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("K4 primitive rates are malformed")
        rates.append(float(rate))
    initial_ce, final_ce = _metric_ce(initial, "K4 initial"), _metric_ce(final, "K4 final", allow_zero=True)
    if primitive_missing:
        return False
    return bool(_finite_rate(final, "K4 final") >= gate["K4_final_vsr_min"] and
        _finite_rate(final, "K4 final") - _finite_rate(initial, "K4 initial") >= gate["K4_final_minus_initial_vsr_min"] and
        min(rates) >= gate["K4_final_valid_each_represented_primitive_min"] and
        final_ce <= gate["K4_final_ce_max_ratio"] * initial_ce)


def _verify_pinned_tokenizer_metadata(root: Path, registration: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    """Rehash metadata/tokenizer files without touching pretrained tensor payloads."""
    from dwr.experiments.gpu_feasibility import MODEL_MANIFEST_SHA256, _read_small_json, _resolve_registered
    model = registration["model_pins"]
    host_config = _checked_host_config(registration)
    model_dir = _resolve_registered(root, host_config["model"]["local_dir"])
    model_config = _read_small_json(_resolve_registered(model_dir, "config.json"), model["model_config_sha256"])
    if model_config.get("vocab_size") != policy["expected_model_vocab_size"] or \
            model_config.get("eos_token_id") != policy["expected_eos_token_id"]:
        raise ValueError("current pinned model metadata vocabulary/EOS differs from registered policy")
    manifest_path = _resolve_registered(root, model["artifact_manifest_path"])
    manifest_raw = manifest_path.read_bytes()
    if _sha_bytes(manifest_raw) != MODEL_MANIFEST_SHA256:
        raise ValueError("pinned artifact manifest changed before tokenizer audit")
    inventory_path = manifest_path.parent / "artifact-inventory.json"
    inventory = _read_small_json(inventory_path, model["artifact_inventory_sha256"])
    entries = inventory.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("pinned tokenizer inventory is malformed")
    tokenizer_files = [entry for entry in entries if isinstance(entry, Mapping) and
        not str(entry.get("name", "")).endswith(".safetensors")]
    if not tokenizer_files:
        raise ValueError("pinned inventory contains no tokenizer metadata")
    for entry in tokenizer_files:
        name = entry.get("name")
        if not isinstance(name, str) or not _is_sha256(entry.get("sha256")) or type(entry.get("size_bytes")) is not int:
            raise ValueError("pinned tokenizer file entry is malformed")
        path = _resolve_registered(model_dir, name)
        if path.stat().st_size != entry["size_bytes"] or _sha_bytes(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"pinned tokenizer file changed before audit: {name}")


def _require_same_producing_segment(initial_ref: Mapping[str, Any], final_ref: Mapping[str, Any]) -> str:
    """Require both live-host frozen audits to belong to the same worker segment."""
    if not isinstance(initial_ref, Mapping) or not isinstance(final_ref, Mapping) or \
            set(initial_ref) != {"path", "sha256"} or set(final_ref) != {"path", "sha256"}:
        raise ValueError("same producing segment audit references are malformed")
    initial_parts = PurePosixPath(initial_ref["path"]).parts
    final_parts = PurePosixPath(final_ref["path"]).parts
    if not initial_parts or not final_parts or not re.fullmatch(r"segment-[0-9]{3}", initial_parts[0]) or \
            not re.fullmatch(r"segment-[0-9]{3}", final_parts[0]) or initial_parts[0] != final_parts[0]:
        raise ValueError("initial and final controller audits must share the same producing segment")
    return initial_parts[0]


def _verify_final_parameter_counts(details: Mapping[str, Any], final_result: Mapping[str, Any],
                                   policy: Mapping[str, Any], arm_id: str) -> None:
    counts = policy["expected_parameter_counts"]
    active = counts["reader_parameter_elements"] if arm_id == "K0" else counts["new_parameter_elements"]
    retained = counts["retained_total_parameter_elements"]
    if details.get("parameter_counts") != counts or details.get("trainable_parameters") != active or \
            final_result.get("trainable_parameters") != active or \
            final_result.get("retained_total_parameters") != retained:
        raise ValueError("controller active/retained parameter counts differ from policy and checkpoint-derived final result")


def audit_completed_arm(root: str | os.PathLike[str], registration: Mapping[str, Any], arm_root: str | os.PathLike[str], *,
                        arm_id: str, source_commit: str, validation_examples: tuple[Any, ...],
                        metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Strict registered production audit; fixture policy is private-core only."""
    from transformers import AutoTokenizer
    from dwr.experiments.source_pilot_registration import pilot_identities
    policy = _registered_controller_policy(registration)
    _verify_pinned_tokenizer_metadata(Path(root).resolve(strict=True), registration, policy)
    host_config = _checked_host_config(registration)
    from dwr.experiments.gpu_feasibility import _resolve_registered
    tokenizer_dir = _resolve_registered(Path(root).resolve(strict=True), host_config["model"]["local_dir"])
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    if getattr(tokenizer, "eos_token_id", None) != policy["expected_eos_token_id"] or \
            getattr(tokenizer, "vocab_size", None) != policy["expected_tokenizer_vocab_size"]:
        raise ValueError("current pinned tokenizer vocabulary/EOS differs from the production policy")
    identities = pilot_identities(registration, source_commit=source_commit, arm_id=arm_id)
    return _audit_completed_arm_core(root, registration, arm_root, arm_id=arm_id,
        source_commit=source_commit, validation_examples=validation_examples, metadata=metadata,
        identities=identities, policy=policy, tokenizer=tokenizer)


def _audit_completed_arm_core(root: str | os.PathLike[str], registration: Mapping[str, Any], arm_root: str | os.PathLike[str], *,
                              arm_id: str, source_commit: str, validation_examples: tuple[Any, ...],
                              metadata: Mapping[str, Any], identities: Mapping[str, Any],
                              policy: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Shared evidence verifier; test fixtures may supply explicitly scaled policy."""
    from dwr.experiments.records import verify_artifacts
    from dwr.training.pilot_recovery import PilotRecoveryStore
    from dwr.training.source_pilot import verify_pilot_stage_receipt

    source_root = Path(root).resolve(strict=True)
    arm_root_path = Path(arm_root).resolve(strict=True)
    if arm_id not in _ARM_ORDER or not isinstance(registration, Mapping):
        raise ValueError("arm audit requires a registered K0/K1/K4 method")
    config = registration["config"]
    policy = _validate_controller_policy(policy, config)
    if not isinstance(identities, Mapping) or set(identities) != _IDENTITY_KEYS or any(not _is_sha256(v) for v in identities.values()):
        raise ValueError("arm audit identities are malformed")
    if len(validation_examples) != policy["validation_examples"] or tuple(example.example_id for example in validation_examples) != tuple(
            row["example_id"] for row in registration["cohort"]["selected"]) or \
            set(metadata) != {example.example_id for example in validation_examples}:
        raise ValueError("arm audit inputs are not the already-authorized fixed validation cohort")

    segment_dirs = sorted((path for path in arm_root_path.glob("segment-[0-9][0-9][0-9]") if path.is_dir()),
        key=lambda path: path.name)
    if not segment_dirs or len(segment_dirs) > 3 or [path.name for path in segment_dirs] != [f"segment-{i:03d}" for i in range(len(segment_dirs))]:
        raise ValueError("arm segment history is missing, duplicated, or outside the recovery cap")
    manifests, event_refs, event_segments, observations, controller_observations = [], [], [], [], []
    arm_settings = None
    for segment_index, segment in enumerate(segment_dirs):
        terminal = verify_artifacts(segment)
        status = terminal["status"]
        initial_doc, _ = _read_json(segment / "manifest.initial.json")
        settings = initial_doc.get("settings")
        required_settings = {"record_kind", "study_type", "config_sha256", "cohort_sha256", "source_commit",
            "physical_source_receipt", "physical_source_receipt_sha256", "runtime_receipt", "runtime_sha256",
            "original_deadlines", "arm_id", "arm_index", "segment_index", "nonce"}
        if not isinstance(settings, Mapping) or set(settings) != required_settings or \
                settings.get("record_kind") != "segment" or settings.get("study_type") != config["study_type"] or \
                settings.get("arm_id") != arm_id or settings.get("arm_index") != _ARM_ORDER.index(arm_id) or \
                settings.get("segment_index") != segment_index or settings.get("source_commit") != source_commit or \
                settings.get("config_sha256") != registration["config_sha256"] or \
                settings.get("cohort_sha256") != registration["cohort_sha256"] or \
                not isinstance(settings.get("nonce"), str) or len(settings["nonce"]) != 32 or \
                any(character not in _HEX for character in settings["nonce"]) or \
                not isinstance(settings.get("original_deadlines"), Mapping) or \
                set(settings["original_deadlines"]) != {"matrix_deadline_epoch", "matrix_watchdog_epoch",
                    "effective_arm_deadline_epoch", "arm_work_deadline_epoch", "arm_watchdog_epoch"}:
            raise ValueError("segment source/arm bindings differ from the registered arm")
        expected_record_identities = _record_identities(registration, identities, source_commit,
            settings.get("runtime_sha256", ""))
        if initial_doc.get("identities") != expected_record_identities or initial_doc.get("rng_streams") != config["seeds"]:
            raise ValueError("segment checkpoint identities or independent RNG stream map differ from registration")
        physical_receipt = settings.get("physical_source_receipt")
        runtime_receipt = settings.get("runtime_receipt")
        if not isinstance(physical_receipt, Mapping) or not isinstance(runtime_receipt, Mapping) or \
                not _is_sha256(settings.get("physical_source_receipt_sha256")) or \
                not _is_sha256(settings.get("runtime_sha256")) or \
                settings.get("physical_source_receipt_sha256") != _sha_bytes(_canonical(dict(physical_receipt))) or \
                settings.get("runtime_sha256") != _sha_bytes(_canonical(dict(runtime_receipt))):
            raise ValueError("segment physical-source or runtime receipt SHA differs from its canonical object")
        if arm_settings is None:
            arm_settings = settings
        elif settings.get("nonce") == arm_settings.get("nonce") or \
                settings.get("physical_source_receipt") != arm_settings.get("physical_source_receipt") or \
                settings.get("physical_source_receipt_sha256") != arm_settings.get("physical_source_receipt_sha256") or \
                settings.get("runtime_receipt") != arm_settings.get("runtime_receipt") or \
                settings.get("runtime_sha256") != arm_settings.get("runtime_sha256") or \
                settings.get("original_deadlines") != arm_settings.get("original_deadlines"):
            raise ValueError("recovery segments must use distinct nonces and the same source/runtime receipts")
        events_path = segment / "events.jsonl"
        events = _read_events(events_path)
        from dwr.experiments.source_pilot_job import verify_worker_loss_record
        verified_loss = verify_worker_loss_record(arm_root_path, segment, registration, source_commit=source_commit)
        _verify_access_order(events, config, registration, terminal_status=status, verified_loss=verified_loss)
        receipt_event = next(event for event in events if event["kind"] == "worker_receipt_published")
        receipt_path, receipt_ref = _checked_reference(arm_root_path, receipt_event["payload"].get("receipt"), "worker receipt")
        receipt, _ = _read_json(receipt_path)
        if receipt.get("descriptor_sha256") != receipt_event["payload"].get("descriptor_sha256") or \
                receipt.get("nonce") != settings["nonce"] or receipt.get("arm_id") != arm_id or \
                receipt.get("segment_index") != segment_index:
            raise ValueError("worker receipt event differs from its immutable segment identity")
        event_segments.append((segment_index, events))
        manifests.append(_arm_relative_ref(segment / "manifest.json", arm_root_path))
        event_refs.append(_arm_relative_ref(events_path, arm_root_path))
        if status not in {"completed", "interrupted"} or (segment_index < len(segment_dirs) - 1 and status != "interrupted"):
            raise ValueError("arm history contains a failed segment or an invalid completed-segment boundary")
        observation_path = segment / "controller-audit" / "segment-observation.json"
        observation = None
        if observation_path.is_file():
            observation, _ = _read_json(observation_path)
            _verify_controller_observation(observation, config["limits"],
                require_cuda=(policy["expected_model_vocab_size"] == 151936), stage="segment_complete")
        elif status == "completed":
            raise ValueError("completed worker segment lacks its controller observation")
        if status != "completed" and observation is None:
            observations.append({"segment_index": segment_index, "status": status, "observation": None,
                "scope": "interrupted segment; controller observation unavailable"})
        else:
            observations.append({"segment_index": segment_index, "status": status,
                "observation": observation, "scope": "controller segment observation"})
        controller_observation_path = segment / "controller-audit" / "controller-observation.json"
        if controller_observation_path.is_file():
            controller_observation, _ = _read_json(controller_observation_path)
            _verify_controller_observation(controller_observation, config["limits"],
                require_cuda=(policy["expected_model_vocab_size"] == 151936))
            controller_observations.append({"segment_index": segment_index, "status": status,
                "observation": controller_observation, "scope": "controller live-memory and elapsed observation"})
        elif status == "completed":
            raise ValueError("completed worker segment lacks its final live-memory observation")
        else:
            controller_observations.append({"segment_index": segment_index, "status": status,
                "observation": None, "scope": "controller observation unavailable for interrupted segment"})

    cursor_path = arm_root_path / "recovery.latest.json"
    cursor_sha = _sha_bytes(cursor_path.read_bytes())
    resume_descriptor = {"jobroot": str(arm_root_path), "cursor_path": str(cursor_path), "cursor_sha256": cursor_sha}
    cursor = PilotRecoveryStore.verify_descriptor(resume_descriptor, identities, arm_id=arm_id,
        selected_ids=tuple(example.example_id for example in validation_examples), metadata=dict(metadata),
        examples=tuple(validation_examples), steps=policy["optimizer_steps"])
    if cursor.get("phase") != "complete" or cursor["progress"].get("optimizer_step") != policy["optimizer_steps"]:
        raise ValueError("arm recovery cursor is not terminal at the registered optimizer-step boundary")
    facts = cursor["facts"]
    final_result = facts.get("final_result")
    expected_stages = {"initial", "final"} if arm_id in {"K0", "K1"} else {"initial", "final", "zero", "noise", "shuffle"}
    references = facts.get("stage_receipts")
    if not isinstance(final_result, Mapping) or not isinstance(references, Mapping) or set(references) != expected_stages:
        raise ValueError("terminal cursor is missing an official or required diagnostic stage")
    bank_ref = facts.get("bank")
    if (arm_id == "K4") != isinstance(bank_ref, Mapping):
        raise ValueError("diagnostic-bank presence differs from the registered arm")
    stages: dict[str, Mapping[str, Any]] = {}
    for stage in ("initial", "final", "zero", "noise", "shuffle"):
        if stage not in expected_stages:
            continue
        receipt_path, receipt_ref = _checked_reference(arm_root_path, references[stage], f"{stage} receipt")
        raw_receipt, _ = _read_json(receipt_path)
        checkpoint = raw_receipt.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or not _is_sha256(checkpoint.get("sha256")):
            raise ValueError("stage receipt checkpoint binding is malformed")
        checkpoint_path, _ = _checked_reference(arm_root_path, checkpoint, f"{stage} checkpoint")
        verified = verify_pilot_stage_receipt(arm_root_path, receipt_ref, arm_id=arm_id, stage=stage,
            expected_step=0 if stage == "initial" else policy["optimizer_steps"],
            expected_ids=tuple(example.example_id for example in validation_examples),
            expected_checkpoint_sha256=checkpoint["sha256"], identities=identities,
            metadata=metadata, examples=tuple(validation_examples),
            expected_bank_metadata_sha256=bank_ref["sha256"] if arm_id == "K4" and stage != "initial" else None,
            expected_bank_reference=bank_ref if arm_id == "K4" and stage != "initial" else None)
        stages[stage] = verified
    if stages["initial"]["metrics"] != final_result.get("initial") or stages["final"]["metrics"] != final_result.get("final"):
        raise ValueError("terminal learning metrics differ from the independently verified stage receipts")
    final_stage_sha = stages["final"]["checkpoint"].get("sha256")
    if any(stages[stage]["checkpoint"].get("sha256") != final_stage_sha
            for stage in expected_stages - {"initial", "final"}):
        raise ValueError("diagnostic stage checkpoint bytes differ from the committed final checkpoint")
    diagnostics = {stage: stages[stage]["metrics"] for stage in ("zero", "noise", "shuffle") if stage in stages}
    if diagnostics != final_result.get("diagnostics"):
        raise ValueError("terminal diagnostic metrics differ from verified stage receipts")
    learning_passed = _recompute_learning_gate(config, stages["initial"]["metrics"], stages["final"]["metrics"]) if arm_id == "K4" else None
    if learning_passed is not final_result.get("learning_gate_passed") or learning_passed is not facts.get("final_result", {}).get("learning_gate_passed"):
        raise ValueError("terminal K4 learning gate differs from the registered metric recomputation")

    vocab_size = policy["expected_model_vocab_size"]
    _audit_prediction_tokens(arm_root_path, references, tokenizer, vocab_size, tokenizer.eos_token_id,
        token_cap=policy["decoder_max_new_tokens"])

    frozen_refs = facts.get("frozen_audit_refs")
    if not isinstance(frozen_refs, Mapping) or set(frozen_refs) != {"initial_frozen", "final_controller"}:
        raise ValueError("terminal cursor lacks both required controller frozen audits")
    initial_frozen_path, initial_frozen_ref = _checked_reference(arm_root_path, frozen_refs["initial_frozen"], "initial frozen audit")
    final_audit_path, final_audit_ref = _checked_reference(arm_root_path, frozen_refs["final_controller"], "final controller audit")
    initial_frozen, _ = _read_json(initial_frozen_path)
    final_audit, _ = _read_json(final_audit_path)
    if initial_frozen.get("schema") != "dwr-source-pilot-initial-frozen-v1" or \
            initial_frozen.get("arm_id") != arm_id or initial_frozen.get("identities") != identities:
        raise ValueError("initial frozen audit identity differs from the arm")
    original_hashes = initial_frozen.get("original_parameter_hashes")
    if not isinstance(original_hashes, Mapping) or len(original_hashes) != policy["expected_original_parameter_count"] or any(not _is_sha256(value) for value in original_hashes.values()):
        raise ValueError("initial frozen audit does not cover the registered original parameters")
    if final_audit.get("schema") != "dwr-source-pilot-controller-audit-v1" or \
            final_audit.get("original_parameter_hashes") != original_hashes or final_audit.get("identities") != identities or \
            final_audit.get("initial_frozen") != initial_frozen_ref or \
            final_audit.get("initial_checkpoint") != stages["initial"]["checkpoint"] or \
            not isinstance(final_audit.get("final_checkpoint"), Mapping) or \
            final_audit["final_checkpoint"].get("sha256") != final_stage_sha or \
            final_audit.get("stage_receipts") != references or final_audit.get("bank") != bank_ref:
        raise ValueError("same-host final frozen audit does not bind immutable initial hashes and stage evidence")
    _, audit_final_checkpoint_ref = _checked_reference(arm_root_path, final_audit.get("final_checkpoint"),
        "final controller checkpoint")
    if audit_final_checkpoint_ref["sha256"] != final_stage_sha:
        raise ValueError("final controller checkpoint is not byte-identical to the immutable final-stage checkpoint")
    details = final_audit.get("details")
    detail_keys = {"parameter_counts", "trainable_parameters", "model_vocab_size", "tokenizer_vocab_size",
        "eos_token_id", "initial_module_sha256", "initial_module_recipe", "source_commit", "runtime_sha256",
        "controller_observation"}
    if not isinstance(details, Mapping) or set(details) != detail_keys or \
            details.get("model_vocab_size") != vocab_size or \
            details.get("tokenizer_vocab_size") != getattr(tokenizer, "vocab_size", None) or \
            details.get("eos_token_id") != getattr(tokenizer, "eos_token_id", None) or \
            details.get("parameter_counts") != policy["expected_parameter_counts"]:
        raise ValueError("final live-host audit details/counts differ from the registered host policy")
    _verify_final_parameter_counts(details, final_result, policy, arm_id)
    final_controller_path = PurePosixPath(final_audit_ref["path"])
    if len(final_controller_path.parts) < 2 or not re.fullmatch(r"segment-[0-9]{3}", final_controller_path.parts[0]):
        raise ValueError("final controller audit is not owned by a producing segment")
    producer_index = int(final_controller_path.parts[0].split("-")[1])
    if not audit_final_checkpoint_ref["path"].startswith(f"segment-{producer_index:03d}/"):
        raise ValueError("final controller checkpoint is not the current-segment exact-copy relocation")
    initial_producer_segment = _require_same_producing_segment(initial_frozen_ref, final_audit_ref)
    if initial_producer_segment != f"segment-{producer_index:03d}":
        raise ValueError("initial frozen audit does not belong to the final audit's producing segment")
    producer = next(((index, events) for index, events in event_segments if index == producer_index), None)
    if producer is None or producer_index != len(segment_dirs) - 1:
        raise ValueError("final controller audit is not in the currently authorized producing segment")
    producer_events = producer[1]
    audit_events = [event for event in producer_events if event["kind"] == "controller_audit" and
        event["payload"].get("final_controller") == final_audit_ref]
    if len(audit_events) != 1 or audit_events[0]["payload"].get("initial_frozen") != initial_frozen_ref:
        raise ValueError("final controller audit is not bound by its producing segment event")
    receipt_events = [event for event in producer_events if event["kind"] == "worker_receipt_published"]
    if len(receipt_events) != 1 or producer_events.index(audit_events[0]) <= producer_events.index(receipt_events[0]):
        raise ValueError("final controller audit predates its producing worker receipt")
    host_events = [event["payload"] for event in producer_events if event["kind"] == "controller_host_initialized"]
    if not host_events:
        raise ValueError("producing segment has no live-host initialization evidence")
    host_event = host_events[-1]
    if details.get("source_commit") != source_commit or details.get("runtime_sha256") != arm_settings["runtime_sha256"] or \
            details.get("initial_module_sha256") != facts.get("initial_state_sha256") or \
            details.get("initial_module_recipe") != host_event.get("initial_module_recipe") or \
            details.get("initial_module_sha256") != host_event.get("initial_module_sha256") or \
            host_event.get("parameter_counts") != details.get("parameter_counts") or \
            host_event.get("model_vocab_size") != vocab_size or \
            host_event.get("tokenizer_vocab_size") != getattr(tokenizer, "vocab_size", None) or \
            host_event.get("eos_token_id") != getattr(tokenizer, "eos_token_id", None):
        raise ValueError("controller final details differ from the same-segment host initialization/source/runtime")
    observation_path, observation_ref = _checked_reference(arm_root_path, details.get("controller_observation"),
        "final controller observation")
    if not observation_ref["path"].startswith(f"segment-{producer_index:03d}/"):
        raise ValueError("controller observation is not in the final audit producing segment")
    observation, _ = _read_json(observation_path)
    _verify_controller_observation(observation, config["limits"], require_cuda=(policy["expected_model_vocab_size"] == 151936))
    if facts.get("initial_state_sha256") != final_audit.get("initial_state_sha256"):
        raise ValueError("final controller audit changed the immutable initial adapter identity")

    pairing = _pair_committed_updates(event_segments, expected_steps=policy["optimizer_steps"])
    counts = {key: 0 for key in ("completed_cases", "greedy_answer_forwards", "teacher_forced_answer_forwards",
        "prompt_encodes", "workspace_updates")}
    stage_costs = {}
    for stage, verified in stages.items():
        stage_counts = verified["counts"]
        for key in counts:
            counts[key] += stage_counts[key]
        rows = _read_prediction_envelopes(verified["records_path"])
        stage_costs[stage] = {"synchronized_elapsed_seconds": sum(row["costs"]["elapsed_seconds"] for row in rows),
            "memory_scope": "per-prediction process peak; not additive",
            "max_observed_memory_bytes": max((row["costs"]["memory"]["value_bytes"] for row in rows), default=0)}
    cursor_ref = {"path": "recovery.latest.json", "sha256": cursor_sha}
    initial_checkpoint = dict(stages["initial"]["checkpoint"])
    final_checkpoint = {"path": cursor["checkpoint_path"], "sha256": cursor["checkpoint_sha256"]}
    current_checkpoint_path, checked_current_checkpoint = _checked_reference(arm_root_path, final_checkpoint,
        "current cursor checkpoint")
    current_segment = segment_dirs[-1].name
    if checked_current_checkpoint["sha256"] != final_stage_sha or \
            not current_checkpoint_path.is_relative_to(segment_dirs[-1].resolve(strict=True)):
        raise ValueError("current cursor checkpoint is not an exact final-stage copy in the authorized current segment")
    return {"schema_version": "dwr-source-pilot-audited-arm-v1", "arm_id": arm_id,
        "identities": identities, "progress": dict(cursor["progress"]),
        "initial_state_sha256": facts["initial_state_sha256"],
        "initial": stages["initial"]["metrics"], "final": stages["final"]["metrics"],
        "diagnostics": diagnostics, "learning_gate_passed": learning_passed,
        "pairing": pairing, "counts": counts,
        "costs": {"stages": stage_costs, "segments": observations,
            "controller_observations": controller_observations,
            "physical_accounting": "job-owned aggregate; excluded from controller totals"},
        "evidence": {"segment_manifests": manifests, "event_streams": event_refs,
            "cursor": cursor_ref, "initial_checkpoint": initial_checkpoint,
            "final_checkpoint": final_checkpoint, "stage_receipts": {stage: dict(ref) for stage, ref in references.items()},
            "bank": None if bank_ref is None else dict(bank_ref),
            "frozen_audit_refs": {"initial_frozen": initial_frozen_ref, "final_controller": final_audit_ref}}}


def _verify_access_order(events: list[dict[str, Any]], config: Mapping[str, Any], registration: Mapping[str, Any], *,
                        terminal_status: str, verified_loss: Mapping[str, Any] | None) -> None:
    receipt_indices = [index for index, event in enumerate(events) if event["kind"] == "worker_receipt_published"]
    receipt_index = receipt_indices[0] if receipt_indices else None
    if receipt_index != 0 or len(receipt_indices) != 1:
        raise ValueError("segment events do not durably bind the issued worker receipt")
    receipt_payload = events[receipt_index]["payload"]
    receipt_ref = receipt_payload.get("receipt")
    if receipt_payload.get("nonce") is None or receipt_payload.get("descriptor_sha256") is None or \
            set(receipt_payload) != {"receipt", "descriptor_sha256", "nonce"} or \
            not isinstance(receipt_ref, Mapping) or set(receipt_ref) != {"path", "sha256"} or \
            not isinstance(receipt_ref.get("path"), str) or not _is_sha256(receipt_ref.get("sha256")):
        raise ValueError("worker receipt event lacks descriptor/nonce identity")
    accesses = [index for index, event in enumerate(events) if event["kind"] == "source_examples_access_authorized"]
    loss_indices = [index for index, event in enumerate(events) if event["kind"] == "worker_loss_finalized"]
    if terminal_status not in {"completed", "interrupted"}:
        raise ValueError("segment access audit requires a completed or interrupted terminal record")
    if verified_loss is None:
        if loss_indices:
            raise ValueError("worker loss marker has not been independently verified by the job")
    else:
        if terminal_status != "interrupted" or len(loss_indices) != 1 or loss_indices[0] != len(events) - 1 or \
                events[loss_indices[0]]["payload"] != verified_loss:
            raise ValueError("interrupted hard-loss marker is missing, nonterminal, or differs from job verification")
        _validate_worker_loss_marker(verified_loss, receipt_payload)
    if any(index <= receipt_index for index in accesses) or len(accesses) > 2:
        raise ValueError("source access authorizations are duplicated or precede the worker receipt")
    if terminal_status == "completed" and (verified_loss is not None or len(accesses) != 2):
        raise ValueError("completed segments require both full source access authorizations")
    if len(accesses) < 2 and (terminal_status != "interrupted" or verified_loss is None):
        raise ValueError("short source access prefix requires independently verified interrupted hard loss")
    if len(accesses) < 2 and any(event["kind"] not in {
            "worker_receipt_published", "source_examples_access_authorized", "worker_loss_finalized"} for event in events):
        raise ValueError("short hard-loss access prefix contains model, kernel, hardware, or host work")
    allowed = {"source_train": config["data"]["source_train_records"],
        "source_validation": config["data"]["validation_examples"]}
    expected_splits = ["source_validation", "source_train"][:len(accesses)]
    for index in accesses:
        payload = events[index]["payload"]
        split = payload.get("split")
        if split not in allowed:
            raise ValueError("source access event names an unknown split")
        if payload.get("record_count") != allowed[split] or payload.get("cohort_sha256") != registration["cohort_sha256"]:
            raise ValueError("source access event differs from registered split count or cohort identity")
        if payload.get("ordered_ids_sha256") != _ordered_id_digest(
                [row["example_id"] for row in registration["cohort"]["selected"]] if split == "source_validation"
                else sorted(row["example_id"] for row in registration["token_rows"] if row["split"] == "source_train")):
            raise ValueError("source access event ordered IDs differ from the frozen split")
        expected_purpose = ("registered_source_validation_for_kernel_and_arm_audit" if split == "source_validation"
            else "registered_source_training_for_kernel")
        if payload.get("purpose") != expected_purpose:
            raise ValueError("source access event purpose differs from the authorized operation")
    if [events[index]["payload"].get("split") for index in accesses] != expected_splits:
        raise ValueError("validation examples must be durably authorized before training examples")
    if verified_loss is not None and loss_indices[0] <= max(accesses, default=receipt_index):
        raise ValueError("worker loss finalization predates a recorded source access authorization")


def _validate_worker_loss_marker(payload: Mapping[str, Any], receipt_payload: Mapping[str, Any]) -> None:
    expected = {"descriptor_sha256", "worker_receipt_sha256", "worker_identity", "launcher_identity",
        "source_commit", "runtime_receipt_sha256", "worker_lock_released", "postloss_observation"}
    if not isinstance(payload, Mapping) or set(payload) != expected or \
            payload.get("descriptor_sha256") != receipt_payload.get("descriptor_sha256") or \
            payload.get("worker_receipt_sha256") != receipt_payload.get("receipt", {}).get("sha256") or \
            not _is_sha256(payload.get("descriptor_sha256")) or \
            not _is_sha256(payload.get("worker_receipt_sha256")) or \
            not _is_sha256(payload.get("runtime_receipt_sha256")) or \
            not isinstance(payload.get("source_commit"), str) or not re.fullmatch(r"[0-9a-f]{40,64}", payload["source_commit"]) or \
            payload.get("worker_lock_released") is not True or \
            payload.get("postloss_observation") != {"available": False, "reason": "process_exited_before_observation"}:
        raise ValueError("worker loss event does not match the closed verified-loss schema")
    for key in ("worker_identity", "launcher_identity"):
        identity = payload.get(key)
        if not isinstance(identity, Mapping) or set(identity) != {"pid", "creation_time", "image"} or \
                type(identity.get("pid")) is not int or identity["pid"] <= 0 or \
                not isinstance(identity.get("creation_time"), str) or not identity["creation_time"] or \
                not isinstance(identity.get("image"), str) or not identity["image"]:
            raise ValueError("worker loss event process identity is malformed")


def _read_prediction_envelopes(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ValueError("prediction records are missing a terminal LF")
        try:
            row = json.loads(line[:-1].decode("utf-8", errors="strict"),
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("prediction records contain invalid JSON") from error
        if not isinstance(row, dict) or _canonical(row) != line[:-1]:
            raise ValueError("prediction envelope is not canonical")
        rows.append(row)
    return rows


def _finite_rate(metrics: Mapping[str, Any], label: str) -> float:
    value = metrics.get("vsr")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} VSR is malformed")
    return float(value)


def _metric_ce(metrics: Mapping[str, Any], label: str, *, allow_zero: bool = False) -> float:
    value = metrics.get("ce")
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} CE metric is missing")
    mean = value.get("token_weighted_mean")
    if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean) or \
            mean < 0 or (mean == 0 and not allow_zero):
        domain = "finite and nonnegative" if allow_zero else "finite and positive"
        raise ValueError(f"{label} CE is not {domain}")
    return float(mean)


def _audited_arm(audits: Mapping[str, Any], arm_id: str, *, expected_steps: int) -> Mapping[str, Any]:
    item = audits.get(arm_id)
    expected_fields = {"schema_version", "arm_id", "identities", "progress", "initial_state_sha256",
        "initial", "final", "diagnostics", "learning_gate_passed", "pairing", "counts", "costs", "evidence"}
    if not isinstance(item, Mapping) or set(item) != expected_fields or \
            item.get("schema_version") != "dwr-source-pilot-audited-arm-v1" or item.get("arm_id") != arm_id:
        raise ValueError(f"{arm_id} is not a semantically audited pilot arm")
    identities = item.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != _IDENTITY_KEYS or any(not _is_sha256(v) for v in identities.values()):
        raise ValueError(f"{arm_id} identities are malformed")
    progress = item.get("progress")
    if not isinstance(progress, Mapping) or progress.get("optimizer_step") != expected_steps:
        raise ValueError(f"{arm_id} does not have the registered committed step count")
    if not _is_sha256(item.get("initial_state_sha256")):
        raise ValueError(f"{arm_id} lacks its paired initialization digest")
    pairing = item.get("pairing")
    if type(pairing) is not list or len(pairing) != expected_steps:
        raise ValueError(f"{arm_id} lacks the complete committed update-pairing history")
    for step, row in enumerate(pairing, 1):
        if not isinstance(row, Mapping) or set(row) != {"optimizer_step", "example_id", "batch_sha256", "noise_sha256"} or \
                row.get("optimizer_step") != step or not isinstance(row.get("example_id"), str) or \
                not row["example_id"] or not _is_sha256(row.get("batch_sha256")) or not _is_sha256(row.get("noise_sha256")):
            raise ValueError(f"{arm_id} committed update pairing history is malformed at step {step}")
    return item


def matrix_decision(config: Mapping[str, Any], arm_audits: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the registered learning and workspace screens from three audited arms."""
    if not isinstance(config, Mapping) or set(arm_audits) != set(_ARM_ORDER):
        raise ValueError("matrix decision requires the exact registered K0/K1/K4 audits")
    training = config.get("training")
    expected_steps = training.get("optimizer_steps") if isinstance(training, Mapping) else None
    if type(expected_steps) is not int or expected_steps <= 0:
        raise ValueError("registered optimizer-step count is missing or invalid")
    arms = {arm_id: _audited_arm(arm_audits, arm_id, expected_steps=expected_steps) for arm_id in _ARM_ORDER}
    if len({tuple(sorted(arm["identities"].items())) for arm in arms.values()}) != 3:
        raise ValueError("each arm must have its own complete checkpoint identity")
    if len({arm["initial_state_sha256"] for arm in arms.values()}) != 1:
        raise ValueError("initial module bytes are not paired identically across all three arms")
    shared_identity_keys = _IDENTITY_KEYS - {"config"}
    if any(len({arm["identities"][key] for arm in arms.values()}) != 1 for key in shared_identity_keys):
        raise ValueError("source/data/model/tokenizer/schedule identities must be shared across paired arms")
    if len({arm["identities"]["config"] for arm in arms.values()}) != 3:
        raise ValueError("each method must retain its distinct registered arm configuration identity")
    if arms["K1"]["pairing"] != arms["K0"]["pairing"] or arms["K4"]["pairing"] != arms["K0"]["pairing"]:
        raise ValueError("committed example, batch and noise histories are not paired across arms")
    if arms["K0"].get("learning_gate_passed") is not None or arms["K1"].get("learning_gate_passed") is not None:
        raise ValueError("control arms cannot claim the K4 learning gate")
    if arms["K0"].get("diagnostics") != {} or arms["K1"].get("diagnostics") != {}:
        raise ValueError("control arms cannot contain K4 diagnostic outcomes")

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("registered evaluation settings are missing")
    learning = evaluation.get("learning_gate")
    screen = evaluation.get("workspace_screen")
    required_learning = {"K4_final_vsr_min", "K4_final_minus_initial_vsr_min",
        "K4_final_valid_each_represented_primitive_min", "K4_final_ce_max_ratio"}
    required_screen = {"K4_minus_trained_K0_final_vsr_min", "K4_unablated_minus_shuffled_final_vsr_min"}
    if not isinstance(learning, Mapping) or set(learning) != required_learning or not isinstance(screen, Mapping) or set(screen) != required_screen:
        raise ValueError("registered pilot gate schema is invalid")
    for value in (*learning.values(), *screen.values()):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("registered gate thresholds must be finite nonnegative numbers")

    k4, k0 = arms["K4"], arms["K0"]
    initial = k4.get("initial")
    final = k4.get("final")
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
        raise ValueError("K4 official initial/final metrics are missing")
    initial_vsr, final_vsr = _finite_rate(initial, "K4 initial"), _finite_rate(final, "K4 final")
    initial_ce, final_ce = _metric_ce(initial, "K4 initial"), _metric_ce(final, "K4 final", allow_zero=True)
    learning_passed = _recompute_learning_gate(config, initial, final)
    if k4.get("learning_gate_passed") is not learning_passed:
        raise ValueError("K4 audited learning outcome differs from the registered recomputation")

    k0_final = k0.get("final")
    diagnostics = k4.get("diagnostics")
    if not isinstance(k0_final, Mapping) or not isinstance(diagnostics, Mapping) or set(diagnostics) != {"zero", "noise", "shuffle"}:
        raise ValueError("K0 final metrics or exact K4 diagnostics are missing")
    k0_vsr = _finite_rate(k0_final, "K0 final")
    shuffle_vsr = _finite_rate(diagnostics["shuffle"], "K4 shuffled final")
    specificity_passed = (
        final_vsr - k0_vsr >= screen["K4_minus_trained_K0_final_vsr_min"] and
        final_vsr - shuffle_vsr >= screen["K4_unablated_minus_shuffled_final_vsr_min"]
    )
    if learning_passed:
        status = "PASS_SOURCE_VALIDATION_PILOT" if specificity_passed else "NO_GO_WORKSPACE_SPECIFICITY_PILOT"
    else:
        status = "NO_GO_SOURCE_VALIDATION_PILOT"
    return {
        "status": status,
        "learning_gate_passed": bool(learning_passed),
        "workspace_screen_passed": bool(specificity_passed),
        "contrasts": {
            "K4_minus_K1_final_vsr": final_vsr - _finite_rate(arms["K1"]["final"], "K1 final"),
            "K4_minus_K0_final_vsr": final_vsr - k0_vsr,
            "K4_minus_shuffle_final_vsr": final_vsr - shuffle_vsr,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--authorization", required=True)
    worker.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "worker":
        run_worker(args.authorization, args.sha256, command=[sys.executable, "-m", "dwr.experiments.source_pilot", *sys.argv[1:]])
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
