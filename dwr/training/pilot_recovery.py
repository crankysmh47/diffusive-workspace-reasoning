"""Closed, arm-local recovery cursor for the registered source-validation pilot."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import uuid

import torch

from .checkpoints import _ROOT_KEYS


class PilotRecoveryError(RuntimeError):
    """A corrupt, incompatible or unsafe pilot recovery boundary."""


_PHASES = {"initial_evaluation_pending", "training", "final_bank_pending",
    "final_evaluation_pending", "diagnostic_zero_pending", "diagnostic_noise_pending",
    "diagnostic_shuffle_pending", "complete"}
_STAGES = ("initial", "final", "zero", "noise", "shuffle")
_IDS = {"source", "config", "data", "model", "tokenizer", "schedule"}
_PROGRESS = {"optimizer_step", "examples_seen", "supervised_tokens_seen"}
_FACTS = {"initial_state_sha256", "initial_state_hash_recipe", "stage_receipts", "bank",
          "frozen_audit_refs", "final_result"}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SEGMENT = re.compile(r"segment-[0-9]{3}\Z")
_CHECKPOINT = re.compile(r"recovery-checkpoint-step-([0-9]{6})\.pt\Z")
_REF_KEYS = {"path", "sha256"}
_CURSOR_KEYS = {"schema_version", "metadata_path", "metadata_sha256", "checkpoint_path",
    "checkpoint_sha256", "segment", "arm_id", "identities", "progress", "phase", "steps"}
_META_KEYS = {"schema_version", "segment", "arm_id", "identities", "progress", "phase",
    "checkpoint_path", "checkpoint_sha256", "steps", "facts"}


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


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_dir(path.parent)


def _replace_durable(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    _write_exclusive(temporary, raw)
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _append_durable_line(path: Path, value: object) -> None:
    raw = _canonical(value)
    with path.open("ab") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_dir(path.parent)


def _safe_relative(root: Path, value: object, *, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PilotRecoveryError("artifact paths must be nonempty relative POSIX paths")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise PilotRecoveryError("artifact path is absolute or traverses its arm root")
    if relative.as_posix() != value:
        raise PilotRecoveryError("artifact path is not canonical POSIX text")
    root = root.resolve(strict=True)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if must_exist:
                raise PilotRecoveryError("artifact path is missing")
            continue
        attrs = getattr(info, "st_file_attributes", 0)
        if stat.S_ISLNK(info.st_mode) or attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise PilotRecoveryError("artifact path traverses a link or reparse point")
    resolved = current.resolve(strict=must_exist)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PilotRecoveryError("artifact path escapes its arm root") from error
    return resolved


def _checked_ids(ids: object) -> tuple[str, ...]:
    if type(ids) is not tuple or not ids or any(not isinstance(item, str) or not _HEX.fullmatch(item) for item in ids):
        raise PilotRecoveryError("selected IDs must be a nonempty tuple of lowercase SHA-256 values")
    if len(set(ids)) != len(ids):
        raise PilotRecoveryError("selected IDs must be unique")
    return ids


def _checked_identities(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _IDS or any(
            not isinstance(item, str) or not _HEX.fullmatch(item) for item in value.values()):
        raise PilotRecoveryError("pilot recovery identities must contain exactly six SHA-256 values")
    return dict(sorted(value.items()))


def _checked_progress(value: object, *, steps: int) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != _PROGRESS:
        raise PilotRecoveryError("pilot progress schema mismatch")
    if any(type(item) is not int or item < 0 for item in value.values()):
        raise PilotRecoveryError("pilot progress counters must be nonnegative integers")
    result = dict(value)
    if result["optimizer_step"] > steps or result["examples_seen"] != result["optimizer_step"]:
        raise PilotRecoveryError("pilot progress is incompatible with batch-one registered steps")
    if result["optimizer_step"] == 0 and result["supervised_tokens_seen"] != 0:
        raise PilotRecoveryError("zero-step pilot progress cannot include supervised tokens")
    if result["optimizer_step"] > 0 and result["supervised_tokens_seen"] <= 0:
        raise PilotRecoveryError("positive pilot progress requires supervised tokens")
    return result


def _checked_ref(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _REF_KEYS or not isinstance(value["path"], str) or \
            not value["path"] or not isinstance(value["sha256"], str) or not _HEX.fullmatch(value["sha256"]):
        raise PilotRecoveryError(f"{name} must be a compact path/SHA reference")
    return value


def _load_json(raw: bytes, name: str) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate {name} key")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid {name} constant {token}")))
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise PilotRecoveryError(f"{name} is not strict UTF-8 JSON") from error
    if _canonical(value) != raw:
        raise PilotRecoveryError(f"{name} bytes are not canonical UTF-8 JSON with LF")
    return value


def _expected_stages(arm_id: str) -> tuple[str, ...]:
    if arm_id in {"K0", "K1"}:
        return ("initial", "final")
    return _STAGES


def _validate_metadata_examples(selected_ids, metadata, examples) -> None:
    if type(examples) is not tuple or tuple(getattr(example, "example_id", None) for example in examples) != selected_ids:
        raise PilotRecoveryError("selected examples must exactly match ordered selected IDs")
    if not isinstance(metadata, dict) or set(metadata) != set(selected_ids):
        raise PilotRecoveryError("validation metadata must exactly cover selected IDs")
    for example in examples:
        if getattr(example, "split", None) != "source_validation":
            raise PilotRecoveryError("recovery context accepts only source_validation examples")
    for eid in selected_ids:
        row = metadata[eid]
        if type(row) is not dict or not {"profile", "fine_stratum", "prompt_tokens", "answer_tokens_including_eos", "total_tokens"} <= set(row):
            raise PilotRecoveryError("validation metadata row is incomplete")


def _checkpoint_header(path: Path, expected_sha256: str, identities: dict, progress: dict,
                       arm_id: str, steps: int) -> None:
    raw = path.read_bytes()
    if _sha_bytes(raw) != expected_sha256:
        raise PilotRecoveryError("checkpoint bytes do not match the cursor SHA")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PilotRecoveryError("checkpoint is corrupt or unsupported") from error
    if not isinstance(payload, dict) or set(payload) != _ROOT_KEYS or payload.get("version") != 1:
        raise PilotRecoveryError("checkpoint schema is incompatible with the complete checkpoint API")
    if payload.get("identities") != identities or payload.get("progress") != progress:
        raise PilotRecoveryError("checkpoint progress or identities differ from the cursor")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict) or architecture.get("slot_identity") != "exchangeable" or \
            architecture.get("workspace_max_steps") != 4:
        raise PilotRecoveryError("checkpoint architecture is not the registered EX capacity-four host")
    groups = payload.get("optimizer_groups")
    if not isinstance(groups, tuple) or ("workspace" in {group[0].split(".", 1)[0]
            for group in groups if group}) != (arm_id != "K0"):
        raise PilotRecoveryError("checkpoint optimizer groups do not match the arm method")
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, dict) or scheduler.get("last_epoch") != progress["optimizer_step"]:
        raise PilotRecoveryError("checkpoint scheduler progress differs from the cursor")


def _verify_audit_refs(jobroot: Path, value: object, *, arm_id: str, identities: dict,
                       initial_state_sha256: str, receipts: dict, bank: object,
                       current_checkpoint: dict[str, str], progress: dict, steps: int) -> dict[str, dict[str, str]]:
    keys = {"initial_frozen", "final_controller"}
    if not isinstance(value, dict) or set(value) != keys:
        raise PilotRecoveryError("complete pilot requires both live-host frozen audit references")
    result = {}
    documents = {}
    for key in sorted(keys):
        ref = _checked_ref(value[key], f"frozen audit {key}")
        path = _safe_relative(jobroot, ref["path"])
        if not path.is_file() or _hash_file(path) != ref["sha256"]:
            raise PilotRecoveryError("frozen audit reference bytes do not match")
        result[key] = dict(ref)
        documents[key] = _load_json(path.read_bytes(), f"frozen audit {key}")
    initial = documents["initial_frozen"]
    initial_keys = {"schema", "arm_id", "identities", "original_parameter_hashes"}
    if not isinstance(initial, dict) or set(initial) != initial_keys or \
            initial.get("schema") != "dwr-source-pilot-initial-frozen-v1" or \
            initial.get("arm_id") != arm_id or initial.get("identities") != identities:
        raise PilotRecoveryError("initial frozen audit has an invalid common envelope")
    original = initial.get("original_parameter_hashes")
    if not isinstance(original, dict) or not original or any(not isinstance(name, str) or not name or
            not isinstance(digest, str) or not _HEX.fullmatch(digest) for name, digest in original.items()):
        raise PilotRecoveryError("initial frozen audit lacks actual named parameter hashes")
    final = documents["final_controller"]
    final_keys = {"schema", "arm_id", "identities", "initial_frozen", "initial_state_sha256",
        "initial_checkpoint", "final_checkpoint", "stage_receipts", "bank",
        "original_parameter_hashes", "details"}
    if not isinstance(final, dict) or set(final) != final_keys or \
            final.get("schema") != "dwr-source-pilot-controller-audit-v1" or \
            final.get("arm_id") != arm_id or final.get("identities") != identities or \
            final.get("initial_frozen") != result["initial_frozen"] or \
            final.get("initial_state_sha256") != initial_state_sha256 or \
            final.get("stage_receipts") != receipts or final.get("bank") != bank or \
            final.get("original_parameter_hashes") != original or not isinstance(final.get("details"), dict):
        raise PilotRecoveryError("final controller audit does not bind the complete cursor evidence")
    initial_checkpoint = _receipt_checkpoint_binding(jobroot, receipts["initial"])
    final_checkpoint = _receipt_checkpoint_binding(jobroot, receipts["final"])
    cursor_ref = _checked_ref(current_checkpoint, "current cursor checkpoint")
    final_path = _safe_relative(jobroot, cursor_ref["path"])
    final_stage_path = _safe_relative(jobroot, final_checkpoint["path"])
    match = _CHECKPOINT.fullmatch(final_path.name)
    if final.get("initial_checkpoint") != initial_checkpoint or final.get("final_checkpoint") != cursor_ref or \
            cursor_ref["sha256"] != final_checkpoint["sha256"] or final_path.parent.name != "recovery" or \
            match is None or int(match.group(1)) != steps or not final_path.is_file() or \
            _hash_file(final_path) != cursor_ref["sha256"] or final_stage_path.parent.name != "recovery":
        raise PilotRecoveryError("controller audit checkpoint references differ from committed stages")
    _checkpoint_header(final_path, cursor_ref["sha256"], identities, progress, arm_id, steps)
    return result


def _verify_bank_metadata(jobroot: Path, ref: dict[str, str], *, arm_id: str,
                          identities: dict, checkpoint_sha256: str,
                          selected_ids: tuple[str, ...], steps: int,
                          checkpoint_path: Path | None = None) -> dict:
    if arm_id != "K4":
        raise PilotRecoveryError("only K4 can own a final workspace bank")
    metadata_path = _safe_relative(jobroot, ref["path"])
    raw = metadata_path.read_bytes()
    if _sha_bytes(raw) != ref["sha256"]:
        raise PilotRecoveryError("bank metadata bytes do not match their reference")
    value = _load_json(raw, "pilot bank metadata")
    keys = {"schema_version", "arm_id", "step", "identities", "checkpoint_sha256",
        "example_ids", "shape", "dtype", "tensor", "rows", "donor_map", "counts"}
    if not isinstance(value, dict) or set(value) != keys or value.get("schema_version") != "dwr-source-pilot-bank-metadata-v1" or \
            value.get("arm_id") != "K4" or value.get("step") != steps or value.get("identities") != identities or \
            value.get("checkpoint_sha256") != checkpoint_sha256 or tuple(value.get("example_ids", ())) != selected_ids:
        raise PilotRecoveryError("bank metadata does not bind the registered K4 final boundary")
    if type(value["shape"]) is not list or len(value["shape"]) != 3 or value["shape"][0] != len(selected_ids) or \
            any(type(dim) is not int or dim <= 0 for dim in value["shape"]):
        raise PilotRecoveryError("bank tensor shape metadata is invalid")
    if value["dtype"] not in {"torch.float32", "torch.float64", "torch.bfloat16"}:
        raise PilotRecoveryError("bank tensor dtype metadata is unsupported")
    tensor_ref = _checked_ref(value["tensor"], "bank tensor")
    tensor_path = _safe_relative(jobroot, tensor_ref["path"])
    if _hash_file(tensor_path) != tensor_ref["sha256"]:
        raise PilotRecoveryError("bank tensor bytes do not match metadata")
    try:
        bank = torch.load(tensor_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PilotRecoveryError("bank tensor artifact is corrupt or unsupported") from error
    if not isinstance(bank, torch.Tensor) or list(bank.shape) != value["shape"] or str(bank.dtype) != value["dtype"] or \
            not bank.is_floating_point() or not torch.isfinite(bank).all().item() or not bank.is_contiguous():
        raise PilotRecoveryError("bank tensor shape, dtype, contiguity, or values are invalid")
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("identities") != identities or \
                payload.get("progress", {}).get("optimizer_step") != steps:
            raise PilotRecoveryError("bank checkpoint identity or optimizer step is incompatible")
        architecture = payload.get("architecture")
        expected_shape = [len(selected_ids), architecture.get("workspace_num_slots"),
            architecture.get("workspace_latent_dim")] if isinstance(architecture, dict) else None
        signature = payload.get("tensor_signature")
        workspace_dtypes = {row[3] for row in signature if isinstance(row, tuple) and len(row) == 5 and
            row[0] == "parameter" and isinstance(row[1], str) and row[1].startswith("workspace.")} \
            if isinstance(signature, tuple) else set()
        if value["shape"] != expected_shape or len(workspace_dtypes) != 1 or \
                value["dtype"] != next(iter(workspace_dtypes)):
            raise PilotRecoveryError("bank tensor shape or dtype differs from its checkpoint architecture")
    rows = value["rows"]
    row_keys = {"example_id", "noise_seed", "noise_sha256", "prompt_sha256", "workspace_sha256",
        "workspace_shape", "workspace_dtype"}
    if type(rows) is not list or len(rows) != len(selected_ids):
        raise PilotRecoveryError("bank task-1 row identities are malformed")
    from dwr.evaluation.source_pilot import _tensor_digest
    for index, (eid, row) in enumerate(zip(selected_ids, rows)):
        if not isinstance(row, dict) or set(row) != row_keys or row.get("example_id") != eid or \
                type(row.get("noise_seed")) is not int or any(not isinstance(row.get(field), str) or
                not _HEX.fullmatch(row[field]) for field in ("noise_sha256", "prompt_sha256", "workspace_sha256")):
            raise PilotRecoveryError("bank row identity is malformed")
        if row["workspace_shape"] != list(bank[index].shape) or row["workspace_dtype"] != str(bank.dtype) or \
                _tensor_digest(bank[index]) != row["workspace_sha256"]:
            raise PilotRecoveryError("bank row tensor identity differs from persisted bytes")
    sorted_ids = sorted(selected_ids)
    expected_donors = {eid: sorted_ids[(index + 1) % len(sorted_ids)] for index, eid in enumerate(sorted_ids)}
    if value["donor_map"] != expected_donors:
        raise PilotRecoveryError("bank donor map is not the registered cyclic ID-only map")
    counts = value["counts"]
    if counts != {"prompt_encoder_calls": len(selected_ids), "workspace_updates": len(selected_ids) * 4}:
        raise PilotRecoveryError("bank preparation counts are inconsistent")
    return value


def _checkpoint_module_digest(payload: dict) -> str:
    digest = hashlib.sha256()
    for module_name, state_key in (("readers", "readers"), ("workspace", "workspace")):
        state = payload.get(state_key)
        if not isinstance(state, dict):
            raise PilotRecoveryError("checkpoint lacks learned module tensors for result verification")
        for name, tensor in sorted(state.items()):
            if not isinstance(tensor, torch.Tensor):
                raise PilotRecoveryError("checkpoint learned state contains a non-tensor")
            raw_name = f"{module_name}.{name}".encode("utf-8")
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(raw_name).to_bytes(8, "big") + raw_name)
            digest.update(len(raw).to_bytes(8, "big") + raw)
    return digest.hexdigest()


def _verify_final_result(result: object, *, arm_id: str, steps: int, stages: dict,
                         checkpoint_path: Path, initial_state_sha256: str) -> None:
    if not isinstance(result, dict):
        raise PilotRecoveryError("complete pilot result summary is malformed")
    from dwr.training.source_pilot import _aggregate_counts, _learning_gate
    expected_stages = _expected_stages(arm_id)
    diagnostics = {name: stages[name]["metrics"] for name in ("zero", "noise", "shuffle")
        if name in stages}
    with checkpoint_path.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    signature = payload.get("tensor_signature")
    if type(signature) is not tuple:
        raise PilotRecoveryError("final checkpoint lacks its parameter signature")
    trainable = retained = 0
    for row in signature:
        if not isinstance(row, tuple) or len(row) != 5 or row[0] != "parameter":
            continue
        count = math.prod(row[2])
        retained += count
        if row[4] is True:
            trainable += count
    expected = {"status": "PILOT_ARM_COMPLETE", "arm_id": arm_id,
        "optimizer_step": steps, "initial": stages["initial"]["metrics"],
        "final": stages["final"]["metrics"], "diagnostics": diagnostics,
        "learning_gate_passed": _learning_gate(stages["initial"]["metrics"],
            stages["final"]["metrics"]) if arm_id == "K4" else None,
        "initial_module_sha256": initial_state_sha256,
        "final_module_sha256": _checkpoint_module_digest(payload),
        "counts": _aggregate_counts(stages), "trainable_parameters": trainable,
        "retained_total_parameters": retained}
    keys = set(expected) | {"costs"}
    if set(result) != keys or any(result.get(key) != value for key, value in expected.items()):
        raise PilotRecoveryError("complete result summary differs from verified stage evidence")
    costs = result["costs"]
    if not isinstance(costs, dict) or set(costs) != {"synchronized_elapsed_seconds", "memory_scope"} or \
            costs.get("memory_scope") != "per-prediction process peak; not additive":
        raise PilotRecoveryError("complete result cost summary has an invalid schema")
    total = 0.0
    for stage in expected_stages:
        receipt = stages[stage]
        raw = receipt["records_path"].read_bytes()
        rows = [_load_json(line, "pilot prediction envelope") for line in raw.splitlines(keepends=True)]
        total += sum(row["costs"]["elapsed_seconds"] for row in rows)
    if type(costs["synchronized_elapsed_seconds"]) not in (int, float) or \
            not math.isclose(costs["synchronized_elapsed_seconds"], total, rel_tol=1e-12, abs_tol=1e-12):
        raise PilotRecoveryError("complete result cost summary differs from committed rows")


def _receipt_checkpoint_binding(jobroot: Path, reference: dict[str, str]) -> dict[str, str]:
    receipt_path = _safe_relative(jobroot, reference["path"])
    raw = receipt_path.read_bytes()
    if _sha_bytes(raw) != reference["sha256"]:
        raise PilotRecoveryError("stage receipt bytes changed before checkpoint binding")
    value = _load_json(raw, "pilot stage receipt")
    checkpoint = value.get("checkpoint") if isinstance(value, dict) else None
    ref = _checked_ref(checkpoint, "stage checkpoint")
    path = _safe_relative(jobroot, ref["path"])
    if path.parent.name != "recovery" or not _CHECKPOINT.fullmatch(path.name) or not path.is_file() or \
            _hash_file(path) != ref["sha256"] or value.get("checkpoint_sha256") != ref["sha256"]:
        raise PilotRecoveryError("stage checkpoint reference is missing, orphaned or inconsistent")
    return ref


def _receipt_expected_checkpoint(jobroot: Path, reference: dict[str, str], step: int) -> str:
    ref = _receipt_checkpoint_binding(jobroot, reference)
    match = _CHECKPOINT.fullmatch(Path(ref["path"]).name)
    if match is None or int(match.group(1)) != step:
        raise PilotRecoveryError("stage checkpoint step differs from its registered stage")
    return ref["sha256"]


def _read_retirement_receipts(jobroot: Path, segment: str) -> dict[str, tuple[int, str]]:
    if not _SEGMENT.fullmatch(segment):
        raise PilotRecoveryError("retirement segment name is invalid")
    relative = f"{segment}/recovery/pilot-retirements.jsonl"
    path = jobroot / relative
    if not path.exists():
        return {}
    raw = _safe_relative(jobroot, relative).read_bytes()
    if not raw.endswith(b"\n"):
        raise PilotRecoveryError("retirement history is missing its final LF")
    retired = {}
    for line in raw.splitlines(keepends=True):
        row = _load_json(line, "pilot checkpoint retirement")
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "step"}:
            raise PilotRecoveryError("checkpoint retirement row is malformed")
        match = _CHECKPOINT.fullmatch(row["path"]) if isinstance(row["path"], str) else None
        if match is None or type(row["step"]) is not int or int(match.group(1)) != row["step"] or \
                not isinstance(row["sha256"], str) or not _HEX.fullmatch(row["sha256"]):
            raise PilotRecoveryError("checkpoint retirement identity is malformed")
        identity = (row["step"], row["sha256"])
        if row["path"] in retired and retired[row["path"]] != identity:
            raise PilotRecoveryError("checkpoint retirement history contains conflicting identities")
        retired[row["path"]] = identity
    return retired


def _validate_facts(jobroot: Path, facts: object, *, arm_id: str, identities: dict,
                    progress: dict, phase: str, checkpoint_sha256: str,
                    checkpoint_path: Path, selected_ids: tuple[str, ...], metadata: dict,
                    examples: tuple, steps: int) -> dict:
    if not isinstance(facts, dict) or set(facts) != _FACTS:
        raise PilotRecoveryError("pilot recovery facts have missing or unknown fields")
    if not isinstance(facts["initial_state_sha256"], str) or not _HEX.fullmatch(facts["initial_state_sha256"]):
        raise PilotRecoveryError("pilot recovery facts lack the initial module-state digest")
    recipe = facts["initial_state_hash_recipe"]
    recipe_expected = {"name": "dwr-source-overfit-initial-state-v1",
        "ordering": "module then sorted state name",
        "framing": "big-endian uint64 byte lengths for UTF-8 name, JSON shape, ASCII dtype, and tensor bytes",
        "bytes": "contiguous CPU tensor viewed as uint8"}
    if recipe != recipe_expected:
        raise PilotRecoveryError("pilot initial-state hash recipe is unsupported")
    receipts = facts["stage_receipts"]
    if not isinstance(receipts, dict) or set(receipts) - set(_expected_stages(arm_id)):
        raise PilotRecoveryError("pilot stage receipt map has an unknown stage")
    verified_stages = {}
    for stage, ref_value in receipts.items():
        ref = _checked_ref(ref_value, f"{stage} stage receipt")
        step = 0 if stage == "initial" else steps
        expected_checkpoint = _receipt_expected_checkpoint(jobroot, ref, step)
        if stage == "initial" and step == progress["optimizer_step"] == 0 and \
                expected_checkpoint != checkpoint_sha256:
            raise PilotRecoveryError("initial stage must bind the actual committed step-zero checkpoint")
        if stage != "initial" and step == progress["optimizer_step"] and expected_checkpoint != checkpoint_sha256:
            raise PilotRecoveryError("final stage must bind the actual committed final checkpoint")
        bank_ref = facts["bank"]
        bank_digest = None
        if arm_id == "K4" and stage in {"final", "zero", "noise", "shuffle"}:
            if bank_ref is None:
                raise PilotRecoveryError("K4 final and diagnostic stages require the committed workspace bank")
            bank_digest = _checked_ref(bank_ref, "workspace bank metadata")["sha256"]
        from dwr.training.source_pilot import verify_pilot_stage_receipt
        try:
            verified_stages[stage] = verify_pilot_stage_receipt(jobroot, ref, arm_id=arm_id, stage=stage,
                expected_step=step, expected_ids=selected_ids,
                expected_checkpoint_sha256=expected_checkpoint, identities=identities,
                metadata=metadata, examples=examples, expected_bank_metadata_sha256=bank_digest,
                expected_bank_reference=bank_ref)
        except Exception as error:
            if isinstance(error, PilotRecoveryError):
                raise
            raise PilotRecoveryError(f"{stage} stage receipt failed semantic verification") from error
    bank_value = facts["bank"]
    if bank_value is not None:
        bank_ref = _checked_ref(bank_value, "workspace bank metadata")
        _verify_bank_metadata(jobroot, bank_ref, arm_id=arm_id, identities=identities,
            checkpoint_sha256=checkpoint_sha256 if progress["optimizer_step"] == steps else
                _receipt_expected_checkpoint(jobroot, receipts.get("final", bank_ref), steps),
            selected_ids=selected_ids, steps=steps, checkpoint_path=checkpoint_path)
    audit = facts["frozen_audit_refs"]
    if phase == "complete":
        expected = set(_expected_stages(arm_id))
        if set(receipts) != expected or progress["optimizer_step"] != steps or \
                (arm_id == "K4") != (bank_value is not None):
            raise PilotRecoveryError("complete phase lacks the full registered set of stages/bank")
        _verify_audit_refs(jobroot, audit, arm_id=arm_id, identities=identities,
            initial_state_sha256=facts["initial_state_sha256"], receipts=receipts, bank=bank_value,
            current_checkpoint={"path": checkpoint_path.relative_to(jobroot).as_posix(),
                "sha256": checkpoint_sha256}, progress=progress, steps=steps)
        _verify_final_result(facts["final_result"], arm_id=arm_id, steps=steps,
            stages=verified_stages, checkpoint_path=checkpoint_path,
            initial_state_sha256=facts["initial_state_sha256"])
    elif audit is not None or facts["final_result"] is not None:
        raise PilotRecoveryError("frozen audit/result is only permitted in complete phase")
    elif audit is not None:
        raise PilotRecoveryError("frozen audit references must remain absent before completion")

    has_initial = "initial" in receipts
    has_final = "final" in receipts
    has_zero = "zero" in receipts
    has_noise = "noise" in receipts
    has_shuffle = "shuffle" in receipts
    step = progress["optimizer_step"]
    phase_ok = (
        phase == "initial_evaluation_pending" and step == 0 and not receipts and bank_value is None or
        phase == "training" and has_initial and not has_final and not any((has_zero, has_noise, has_shuffle)) and
            bank_value is None and step < steps or
        phase == "final_bank_pending" and arm_id == "K4" and step == steps and has_initial and not has_final and
            bank_value is None or
        phase == "final_evaluation_pending" and step == steps and has_initial and
            (arm_id == "K4" and bank_value is not None and not has_final and not has_zero and
                not has_noise and not has_shuffle or
             arm_id != "K4" and bank_value is None and not has_zero and not has_noise and not has_shuffle) or
        phase == "diagnostic_zero_pending" and arm_id == "K4" and step == steps and has_initial and has_final and
            not has_zero and not has_noise and not has_shuffle and bank_value is not None or
        phase == "diagnostic_noise_pending" and arm_id == "K4" and step == steps and has_initial and has_final and
            has_zero and not has_noise and not has_shuffle and bank_value is not None or
        phase == "diagnostic_shuffle_pending" and arm_id == "K4" and step == steps and has_initial and has_final and
            has_zero and has_noise and bank_value is not None or
        phase == "complete"
    )
    if not phase_ok:
        raise PilotRecoveryError(f"pilot phase, progress, receipts, bank and method are inconsistent: {phase}, step={step}, stages={sorted(receipts)}, bank={bank_value is not None}, method={arm_id}")
    if bank_value is not None:
        _verify_bank_metadata(jobroot, _checked_ref(bank_value, "workspace bank metadata"),
            arm_id=arm_id, identities=identities, checkpoint_sha256=_receipt_expected_checkpoint(
                jobroot, receipts["final"], steps) if "final" in receipts else checkpoint_sha256,
            selected_ids=selected_ids, steps=steps, checkpoint_path=checkpoint_path)
    return facts


def _read_cursor(descriptor: object, identities: dict, *, arm_id: str,
                 selected_ids: tuple[str, ...], metadata: dict, examples: tuple, steps: int) -> dict:
    if not isinstance(descriptor, dict) or set(descriptor) != {"jobroot", "cursor_path", "cursor_sha256"}:
        raise PilotRecoveryError("pilot recovery descriptor is malformed")
    jobroot = Path(descriptor["jobroot"]).resolve(strict=True)
    cursor = Path(descriptor["cursor_path"]).resolve(strict=True)
    if cursor != jobroot / "recovery.latest.json" or _hash_file(cursor) != descriptor["cursor_sha256"]:
        raise PilotRecoveryError("pilot cursor path or SHA does not match descriptor")
    pointer = _load_json(cursor.read_bytes(), "pilot cursor")
    if not isinstance(pointer, dict) or set(pointer) != _CURSOR_KEYS or pointer.get("schema_version") != 1:
        raise PilotRecoveryError("pilot cursor schema is malformed")
    expected_ids = _checked_identities(identities)
    if pointer["arm_id"] != arm_id or pointer["identities"] != expected_ids or pointer["steps"] != steps:
        raise PilotRecoveryError("pilot cursor method, identities, or step budget differs")
    segment = pointer["segment"]
    if not isinstance(segment, str) or not _SEGMENT.fullmatch(segment):
        raise PilotRecoveryError("pilot cursor segment name is invalid")
    metadata_path = _safe_relative(jobroot, pointer["metadata_path"])
    metadata_raw = metadata_path.read_bytes()
    if _sha_bytes(metadata_raw) != pointer["metadata_sha256"]:
        raise PilotRecoveryError("pilot metadata bytes do not match cursor")
    value = _load_json(metadata_raw, "pilot recovery metadata")
    if not isinstance(value, dict) or set(value) != _META_KEYS or value.get("schema_version") != 1 or \
            value.get("segment") != segment or value.get("arm_id") != arm_id or \
            value.get("identities") != expected_ids or value.get("steps") != steps:
        raise PilotRecoveryError("pilot metadata schema or binding is malformed")
    if any(pointer[key] != value[key] for key in ("checkpoint_path", "checkpoint_sha256", "progress", "phase")):
        raise PilotRecoveryError("pilot cursor and metadata bindings differ")
    progress = _checked_progress(value["progress"], steps=steps)
    checkpoint_path = _safe_relative(jobroot, value["checkpoint_path"])
    if checkpoint_path.parent != (jobroot / segment / "recovery").resolve(strict=True):
        raise PilotRecoveryError("pilot checkpoint must be a direct current-segment recovery child")
    _checkpoint_header(checkpoint_path, value["checkpoint_sha256"], expected_ids, progress, arm_id, steps)
    _validate_facts(jobroot, value["facts"], arm_id=arm_id, identities=expected_ids,
        progress=progress, phase=value["phase"], checkpoint_sha256=value["checkpoint_sha256"],
        checkpoint_path=checkpoint_path, selected_ids=selected_ids, metadata=metadata,
        examples=examples, steps=steps)
    return {**value, "metadata_sha256": pointer["metadata_sha256"],
        "checkpoint_path_absolute": checkpoint_path, "jobroot": jobroot,
        "cursor_path": cursor, "cursor_sha256": descriptor["cursor_sha256"]}


class PilotRecoveryStore:
    """Publish durable arm-local cursor metadata after complete checkpoint boundaries."""

    def __init__(self, jobroot, segment_dir, identities, *, arm_id, selected_ids,
                 metadata, examples, steps=512):
        self.jobroot = Path(jobroot).resolve(strict=True)
        self.segment_dir = Path(segment_dir).resolve(strict=True)
        try:
            relative = self.segment_dir.relative_to(self.jobroot)
        except ValueError as error:
            raise PilotRecoveryError("pilot segment must be inside its arm root") from error
        if len(relative.parts) != 1 or not _SEGMENT.fullmatch(relative.name):
            raise PilotRecoveryError("pilot segment must be an immediate segment-NNN arm-root child")
        if arm_id not in {"K0", "K1", "K4"}:
            raise PilotRecoveryError("pilot arm must be K0, K1, or K4")
        if type(steps) is not int or steps <= 0:
            raise PilotRecoveryError("pilot checkpoint step count must be a positive integer")
        if type(metadata) is not dict:
            raise PilotRecoveryError("validation metadata must be a plain ID-keyed dictionary")
        self.arm_id = arm_id
        self.steps = steps
        self.identities = _checked_identities(identities)
        self.selected_ids = _checked_ids(selected_ids)
        self.metadata = metadata
        self.examples = examples
        _validate_metadata_examples(self.selected_ids, metadata, examples)
        self.recovery_dir = self.segment_dir / "recovery"
        self.recovery_dir.mkdir(exist_ok=True)
        self.cursor_path = self.jobroot / "recovery.latest.json"

    def commit(self, checkpoint_path, checkpoint_sha256, progress, phase, facts) -> dict:
        if phase not in _PHASES:
            raise PilotRecoveryError("pilot recovery phase is unknown")
        checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        if checkpoint_path.parent != self.recovery_dir.resolve(strict=True) or \
                _CHECKPOINT.fullmatch(checkpoint_path.name) is None:
            raise PilotRecoveryError("pilot checkpoint must be a named direct recovery-directory child")
        if not isinstance(checkpoint_sha256, str) or not _HEX.fullmatch(checkpoint_sha256):
            raise PilotRecoveryError("pilot checkpoint SHA must be lowercase hexadecimal")
        progress = _checked_progress(progress, steps=self.steps)
        _checkpoint_header(checkpoint_path, checkpoint_sha256, self.identities,
            progress, self.arm_id, self.steps)
        facts = _validate_facts(self.jobroot, facts, arm_id=self.arm_id,
            identities=self.identities, progress=progress, phase=phase,
            checkpoint_sha256=checkpoint_sha256, checkpoint_path=checkpoint_path,
            selected_ids=self.selected_ids,
            metadata=self.metadata, examples=self.examples, steps=self.steps)
        if self.cursor_path.exists():
            previous = _read_cursor({"jobroot": str(self.jobroot), "cursor_path": str(self.cursor_path),
                "cursor_sha256": _hash_file(self.cursor_path)}, self.identities, arm_id=self.arm_id,
                selected_ids=self.selected_ids, metadata=self.metadata, examples=self.examples, steps=self.steps)
            if previous["phase"] == "complete":
                raise PilotRecoveryError("complete pilot arm cursor is terminal")
            if previous["segment"] == self.segment_dir.name:
                allowed = self._transition(previous, progress, phase, facts, checkpoint_sha256)
                if not allowed:
                    raise PilotRecoveryError("pilot cursor phase/progress transition is illegal")
            elif progress != previous["progress"] or phase != previous["phase"] or facts != previous["facts"] or \
                    checkpoint_sha256 != previous["checkpoint_sha256"]:
                raise PilotRecoveryError("a new worker segment must first re-publish the prior boundary unchanged")
        else:
            if phase != "initial_evaluation_pending" or progress["optimizer_step"] != 0 or facts["stage_receipts"]:
                raise PilotRecoveryError("first pilot cursor must publish the empty step-zero initial boundary")

        metadata = {"schema_version": 1, "segment": self.segment_dir.name,
            "arm_id": self.arm_id, "identities": self.identities, "progress": progress,
            "phase": phase, "checkpoint_path": checkpoint_path.relative_to(self.jobroot).as_posix(),
            "checkpoint_sha256": checkpoint_sha256, "steps": self.steps, "facts": facts}
        metadata_raw = _canonical(metadata)
        meta_name = f"pilot-metadata-step-{progress['optimizer_step']:06d}-{uuid.uuid4().hex}.json"
        metadata_path = self.recovery_dir / meta_name
        _write_exclusive(metadata_path, metadata_raw)
        pointer = {"schema_version": 1, "metadata_path": metadata_path.relative_to(self.jobroot).as_posix(),
            "metadata_sha256": _sha_bytes(metadata_raw), "checkpoint_path": metadata["checkpoint_path"],
            "checkpoint_sha256": checkpoint_sha256, "segment": self.segment_dir.name,
            "arm_id": self.arm_id, "identities": self.identities, "progress": progress,
            "phase": phase, "steps": self.steps}
        pointer_raw = _canonical(pointer)
        _replace_durable(self.cursor_path, pointer_raw)
        _append_durable_line(self.jobroot / "recovery-commits.jsonl", {
            "cursor_sha256": _sha_bytes(pointer_raw), "metadata_path": pointer["metadata_path"],
            "metadata_sha256": pointer["metadata_sha256"],
            "checkpoint_path": pointer["checkpoint_path"],
            "checkpoint_sha256": pointer["checkpoint_sha256"], "segment": self.segment_dir.name,
            "progress": progress, "phase": phase})
        descriptor = {"jobroot": str(self.jobroot), "cursor_path": str(self.cursor_path),
            "cursor_sha256": _sha_bytes(pointer_raw)}
        verified = _read_cursor(descriptor, self.identities, arm_id=self.arm_id,
            selected_ids=self.selected_ids, metadata=self.metadata, examples=self.examples, steps=self.steps)
        if verified["metadata_sha256"] != pointer["metadata_sha256"]:
            raise PilotRecoveryError("pilot cursor failed post-publication verification")
        self._retire_old_checkpoints()
        return descriptor

    def _transition(self, previous: dict, progress: dict, phase: str, facts: dict, checkpoint_sha256: str) -> bool:
        old_phase = previous["phase"]
        old_progress = previous["progress"]
        old_facts = previous["facts"]
        new_step, old_step = progress["optimizer_step"], old_progress["optimizer_step"]
        old_receipts, new_receipts = old_facts["stage_receipts"], facts["stage_receipts"]
        if facts["initial_state_sha256"] != old_facts["initial_state_sha256"] or \
                facts["initial_state_hash_recipe"] != old_facts["initial_state_hash_recipe"] or \
                any(new_receipts.get(name) != reference for name, reference in old_receipts.items()):
            return False
        if phase == "complete":
            return new_step == old_step and progress == old_progress and \
                checkpoint_sha256 == previous["checkpoint_sha256"] and facts["bank"] == old_facts["bank"] and \
                new_receipts == old_receipts and set(new_receipts) == set(_expected_stages(self.arm_id)) and \
                facts["frozen_audit_refs"] is not None and facts["final_result"] is not None and \
                old_facts["frozen_audit_refs"] is None and old_facts["final_result"] is None
        same_progress_checkpoint = progress == old_progress and checkpoint_sha256 == previous["checkpoint_sha256"]
        if old_phase == "initial_evaluation_pending":
            return phase == "training" and new_step == 0 and "initial" in facts["stage_receipts"] and \
                set(new_receipts) == {"initial"} and facts["bank"] is None and same_progress_checkpoint and \
                facts["frozen_audit_refs"] is None and facts["final_result"] is None
        if old_phase == "training":
            if phase == "training":
                return new_step == old_step + 1 and new_step < self.steps and facts == old_facts
            if new_step == self.steps and old_step == self.steps - 1 and phase in {
                    "final_bank_pending" if self.arm_id == "K4" else "final_evaluation_pending"}:
                return facts == old_facts and progress["examples_seen"] == old_progress["examples_seen"] + 1 and \
                    progress["supervised_tokens_seen"] > old_progress["supervised_tokens_seen"] and \
                    facts["frozen_audit_refs"] is None and facts["final_result"] is None
        if old_phase == "final_bank_pending":
            return self.arm_id == "K4" and phase == "final_evaluation_pending" and same_progress_checkpoint and \
                facts["bank"] is not None and new_receipts == old_receipts and \
                facts["frozen_audit_refs"] is None and facts["final_result"] is None
        if old_phase == "final_evaluation_pending":
            if self.arm_id == "K4":
                return phase == "diagnostic_zero_pending" and same_progress_checkpoint and \
                    set(new_receipts) == set(old_receipts) | {"final"} and facts["bank"] == old_facts["bank"] and \
                    facts["frozen_audit_refs"] is None and facts["final_result"] is None
            return phase == "final_evaluation_pending" and same_progress_checkpoint and \
                set(new_receipts) == set(old_receipts) | {"final"} and facts["bank"] is None and \
                facts["frozen_audit_refs"] is None and facts["final_result"] is None
        if old_phase == "diagnostic_zero_pending":
            return phase == "diagnostic_noise_pending" and same_progress_checkpoint and \
                set(new_receipts) == set(old_receipts) | {"zero"} and facts["bank"] == old_facts["bank"] and \
                facts["frozen_audit_refs"] is None and facts["final_result"] is None
        if old_phase == "diagnostic_noise_pending":
            return phase == "diagnostic_shuffle_pending" and same_progress_checkpoint and \
                set(new_receipts) == set(old_receipts) | {"noise"} and facts["bank"] == old_facts["bank"] and \
                facts["frozen_audit_refs"] is None and facts["final_result"] is None
        if old_phase == "diagnostic_shuffle_pending":
            return phase == "diagnostic_shuffle_pending" and same_progress_checkpoint and \
                set(new_receipts) == set(old_receipts) | {"shuffle"} and \
                facts["bank"] == old_facts["bank"] and facts["frozen_audit_refs"] is None and \
                facts["final_result"] is None
        return False

    def _retire_old_checkpoints(self) -> None:
        """Retire only committed, hash-verified rolling checkpoints in this live segment."""
        try:
            pointer = _load_json(self.cursor_path.read_bytes(), "pilot cursor")
            protected_steps = {0, self.steps}
            known = {}
            retired_by_segment = {}
            history_path = self.jobroot / "recovery-commits.jsonl"
            if history_path.exists():
                history_raw = history_path.read_bytes()
                if not history_raw.endswith(b"\n"):
                    raise PilotRecoveryError("committed recovery history is missing its final LF")
                for line in history_raw.splitlines(keepends=True):
                    row = _load_json(line, "pilot committed recovery history")
                    if not isinstance(row, dict) or set(row) != {"cursor_sha256", "metadata_path",
                            "metadata_sha256", "checkpoint_path", "checkpoint_sha256", "segment",
                            "progress", "phase"}:
                        raise PilotRecoveryError("committed recovery history row is malformed")
                    metadata_path = _safe_relative(self.jobroot, row["metadata_path"])
                    metadata_raw = metadata_path.read_bytes()
                    metadata = _load_json(metadata_raw, "pilot committed recovery metadata")
                    if _sha_bytes(metadata_raw) != row["metadata_sha256"] or \
                            metadata.get("checkpoint_path") != row["checkpoint_path"] or \
                            metadata.get("checkpoint_sha256") != row["checkpoint_sha256"] or \
                            metadata.get("segment") != row["segment"] or metadata.get("progress") != row["progress"] or \
                            metadata.get("phase") != row["phase"]:
                        raise PilotRecoveryError("committed history does not match its immutable metadata")
                    segment = row["segment"]
                    if not isinstance(segment, str) or not _SEGMENT.fullmatch(segment):
                        raise PilotRecoveryError("committed history has an invalid segment")
                    relative = row["checkpoint_path"]
                    if not isinstance(relative, str) or "\\" in relative:
                        raise PilotRecoveryError("committed checkpoint path is malformed")
                    pure = PurePosixPath(relative)
                    match = _CHECKPOINT.fullmatch(pure.name)
                    if pure.parts != (segment, "recovery", pure.name) or match is None:
                        raise PilotRecoveryError("committed checkpoint is outside its segment recovery directory")
                    step = int(match.group(1))
                    if step != row["progress"].get("optimizer_step"):
                        raise PilotRecoveryError("committed checkpoint filename differs from its progress")
                    retired = retired_by_segment.setdefault(segment,
                        _read_retirement_receipts(self.jobroot, segment))
                    retirement = retired.get(pure.name)
                    if retirement is not None and retirement != (step, row["checkpoint_sha256"]):
                        raise PilotRecoveryError("retirement receipt differs from its committed checkpoint")
                    path = _safe_relative(self.jobroot, relative, must_exist=False)
                    if path.exists():
                        if _hash_file(path) != row["checkpoint_sha256"]:
                            raise PilotRecoveryError("committed checkpoint bytes changed")
                    elif retirement is None:
                        raise PilotRecoveryError("committed checkpoint is missing without a retirement receipt")
                    if segment == self.segment_dir.name and retirement is None:
                        known[path] = (step, row["checkpoint_sha256"])
            # Include the latest atomic cursor in the event of a crash after
            # its replacement but before the append-only commit receipt.
            pointer = _load_json(self.cursor_path.read_bytes(), "pilot cursor")
            current_path = _safe_relative(self.jobroot, pointer["checkpoint_path"])
            current_match = _CHECKPOINT.fullmatch(current_path.name)
            if current_match and current_path.parent == self.recovery_dir.resolve(strict=True):
                known[current_path] = (int(current_match.group(1)), pointer["checkpoint_sha256"])
            rolling_steps = sorted({step for step, _ in known.values()
                if step not in protected_steps}, reverse=True)[:2]
            keep_steps = protected_steps | set(rolling_steps)
            receipt_path = self.recovery_dir / "pilot-retirements.jsonl"
            for path, (step, expected_sha) in sorted(known.items(), key=lambda item: item[0].name):
                if step in keep_steps or not path.exists():
                    continue
                if path.parent != self.recovery_dir or path.is_symlink() or _hash_file(path) != expected_sha:
                    raise PilotRecoveryError("refusing to retire an unsafe or changed checkpoint")
                attrs = getattr(path.lstat(), "st_file_attributes", 0)
                if attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                    raise PilotRecoveryError("refusing to retire a reparse-point checkpoint")
                line = _canonical({"path": path.name, "sha256": expected_sha, "step": step})
                with receipt_path.open("ab") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
                path.unlink()
                _fsync_dir(self.recovery_dir)
        except PilotRecoveryError:
            raise
        except Exception as error:
            raise PilotRecoveryError("pilot rolling checkpoint retention failed safely") from error

    @staticmethod
    def verify_descriptor(descriptor, identities, *, arm_id, selected_ids, metadata, examples, steps=512) -> dict:
        if arm_id not in {"K0", "K1", "K4"} or type(steps) is not int or steps <= 0:
            raise PilotRecoveryError("pilot verification method or step budget is invalid")
        selected_ids = _checked_ids(selected_ids)
        if type(metadata) is not dict:
            raise PilotRecoveryError("validation metadata must be a plain ID-keyed dictionary")
        _validate_metadata_examples(selected_ids, metadata, examples)
        return _read_cursor(descriptor, identities, arm_id=arm_id, selected_ids=selected_ids,
            metadata=metadata, examples=examples, steps=steps)
