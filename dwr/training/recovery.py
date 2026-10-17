"""Small, strict transactional cursor store for source-overfit recovery."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid


class RecoveryError(RuntimeError):
    """A corrupt, incompatible, or unsafe recovery boundary."""


_VERSION = 1
_PHASES = {"initial_evaluation_pending", "training", "resume_probe_pending",
          "final_evaluation_pending", "complete"}
_IDENTITIES = {"source", "config", "data", "model", "tokenizer", "schedule"}
_PROGRESS = {"optimizer_step", "examples_seen", "supervised_tokens_seen"}
_FACTS = {"initial_state_sha256", "initial_state_hash_recipe", "initial_evaluation_receipt",
          "resume_probe", "final_result"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_CURSOR_KEYS = {"schema_version", "metadata_path", "metadata_sha256", "checkpoint_path",
                "checkpoint_sha256", "segment", "identities", "progress", "phase"}
_METADATA_KEYS = {"schema_version", "segment", "identities", "progress", "phase",
                  "checkpoint_path", "checkpoint_sha256", "facts"}
_DESCRIPTOR_KEYS = {"jobroot", "cursor_path", "cursor_sha256"}


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_tree_path(root: Path, value: str, *, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise RecoveryError("recovery paths must be nonempty relative paths")
    root = root.resolve(strict=True)
    path = root / value
    try:
        resolved = path.resolve(strict=must_exist)
    except (OSError, RuntimeError) as error:
        raise RecoveryError("recovery path is missing or invalid") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RecoveryError("recovery path escapes the job root") from error
    current = root
    for part in Path(value).parts:
        if part in {".", ".."}:
            raise RecoveryError("recovery path contains traversal")
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if must_exist:
                raise RecoveryError("recovery path is missing")
            continue
        attrs = getattr(info, "st_file_attributes", 0)
        if stat.S_ISLNK(info.st_mode) or attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise RecoveryError("recovery path traverses a link or reparse point")
    return resolved


def _reject_links(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        attrs = getattr(info, "st_file_attributes", 0)
        if stat.S_ISLNK(info.st_mode) or attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise RecoveryError("recovery paths cannot contain links or reparse points")


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_write(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_dir(path.parent)


def _validate_progress(progress: object) -> dict[str, int]:
    if not isinstance(progress, dict) or set(progress) != _PROGRESS:
        raise RecoveryError("recovery progress schema mismatch")
    result = {}
    for key, value in progress.items():
        if type(value) is not int or value < 0:
            raise RecoveryError(f"{key} must be a nonnegative integer")
        result[key] = value
    if result["examples_seen"] != result["optimizer_step"]:
        raise RecoveryError("source-overfit progress must retain one example per net update")
    return result


def _validate_identities(identities: object) -> dict[str, str]:
    if not isinstance(identities, dict) or set(identities) != _IDENTITIES:
        raise RecoveryError("recovery identities must contain exactly six hashes")
    if any(not isinstance(value, str) or not _SHA.fullmatch(value) for value in identities.values()):
        raise RecoveryError("recovery identities must be lowercase SHA-256 values")
    return dict(sorted(identities.items()))


def _validate_metadata(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != _METADATA_KEYS:
        raise RecoveryError("recovery metadata has missing or unknown fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != _VERSION:
        raise RecoveryError("unsupported recovery metadata version")
    if not isinstance(value["segment"], str) or not re.fullmatch(r"segment-[0-9]{3}", value["segment"]):
        raise RecoveryError("recovery segment name is invalid")
    _validate_identities(value["identities"])
    _validate_progress(value["progress"])
    if value["phase"] not in _PHASES:
        raise RecoveryError("recovery phase is invalid")
    for key in ("checkpoint_path",):
        if not isinstance(value[key], str) or not value[key]:
            raise RecoveryError(f"{key} must be a path")
    if not isinstance(value["checkpoint_sha256"], str) or not _SHA.fullmatch(value["checkpoint_sha256"]):
        raise RecoveryError("checkpoint digest is invalid")
    facts = value["facts"]
    if not isinstance(facts, dict) or set(facts) - _FACTS:
        raise RecoveryError("recovery facts have unknown fields")
    _validate_facts(value["phase"], value["progress"], facts, value["segment"])
    return value


def _validate_facts(phase: str, progress: dict, facts: dict, segment: str) -> None:
    """Validate the closed phase-dependent fact shape before publication/use."""
    common = {"initial_state_sha256", "initial_state_hash_recipe"}
    if not isinstance(facts.get("initial_state_sha256"), str) or not _SHA.fullmatch(facts["initial_state_sha256"]):
        raise RecoveryError("recovery facts lack the initial-state digest")
    recipe = facts.get("initial_state_hash_recipe")
    if not isinstance(recipe, dict) or set(recipe) != {"name", "ordering", "framing", "bytes"} or \
            any(not isinstance(v, str) or not v for v in recipe.values()):
        raise RecoveryError("initial-state hash recipe is malformed")
    has_initial = "initial_evaluation_receipt" in facts
    has_probe = "resume_probe" in facts
    has_final = "final_result" in facts
    if phase == "initial_evaluation_pending":
        if set(facts) != common or progress["optimizer_step"] != 0:
            raise RecoveryError("initial-evaluation-pending facts/progress mismatch")
        return
    if phase == "training" and not has_initial:
        raise RecoveryError("training phase requires a committed initial evaluation")
    if phase in {"resume_probe_pending", "final_evaluation_pending", "complete"} and not has_initial:
        raise RecoveryError(f"{phase} phase requires a committed initial evaluation")
    initial = facts.get("initial_evaluation_receipt")
    if has_initial and not _valid_eval_ref(initial, 0):
        raise RecoveryError("initial evaluation receipt reference is malformed")
    initial_segment = Path(initial["records_path"]).parts[0] if has_initial else None
    if has_initial and (not re.fullmatch(r"segment-[0-9]{3}", initial_segment or "") or
            Path(initial["receipt_path"]).parts[0] != initial_segment or
            re.fullmatch(r"predictions-step-000000-[0-9a-f]{32}\.jsonl", Path(initial["records_path"]).name) is None or
            re.fullmatch(r"evaluation-step-000000-[0-9a-f]{32}\.receipt\.json", Path(initial["receipt_path"]).name) is None):
        raise RecoveryError("initial evaluation receipt paths do not bind this segment")
    if has_probe:
        probe = facts["resume_probe"]
        required = {"path", "sha256", "checkpoint_path", "checkpoint_sha256", "step", "result"}
        if not isinstance(probe, dict) or set(probe) != required or type(probe["step"]) is not int or probe["step"] <= 0:
            raise RecoveryError("resume-probe evidence reference is malformed")
        if not _valid_rel(probe["path"]) or not _valid_rel(probe["checkpoint_path"]):
            raise RecoveryError("resume-probe evidence path is malformed")
        probe_segment = Path(probe["path"]).parts[0]
        if not re.fullmatch(r"segment-[0-9]{3}", probe_segment) or Path(probe["checkpoint_path"]).parts[0] != probe_segment or \
                re.fullmatch(r"resume-probe-state-[0-9a-f]{32}\.pt", Path(probe["path"]).name) is None or \
                probe["checkpoint_path"] != f"{probe_segment}/recovery/recovery-checkpoint-step-{probe['step']:06d}.pt":
            raise RecoveryError("resume-probe paths do not bind their registered segment/step")
        if not _SHA.fullmatch(str(probe["sha256"])) or not _SHA.fullmatch(str(probe["checkpoint_sha256"])):
            raise RecoveryError("resume-probe evidence digest is malformed")
        result = probe["result"]
        result_keys = {"first_replayed_ids", "second_replayed_ids", "max_parameter_abs_error",
            "max_parameter_relative_error", "max_optimizer_abs_error", "max_optimizer_relative_error",
            "optimizer_moment_count", "max_scheduler_abs_error", "max_rng_abs_error",
            "max_sampler_abs_error", "max_loss_abs_error", "max_loss_relative_error"}
        if not isinstance(result, dict) or set(result) != result_keys or \
                not isinstance(result["first_replayed_ids"], list) or not result["first_replayed_ids"] or \
                result["first_replayed_ids"] != result["second_replayed_ids"] or \
                type(result["optimizer_moment_count"]) is not int or result["optimizer_moment_count"] <= 0:
            raise RecoveryError("resume-probe result is malformed or did not pass replay")
        for key in result_keys - {"first_replayed_ids", "second_replayed_ids", "optimizer_moment_count"}:
            number = result[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
                raise RecoveryError("resume-probe errors must be finite and nonnegative")
        if phase == "resume_probe_pending" or progress["optimizer_step"] <= probe["step"]:
            raise RecoveryError("resume-probe evidence is inconsistent with cursor progress")
    elif phase in {"final_evaluation_pending", "complete"}:
        raise RecoveryError("final phase requires completed resume-probe evidence")
    if phase == "resume_probe_pending" and has_probe:
        raise RecoveryError("resume-probe-pending cannot contain completed probe evidence")
    if phase == "complete":
        result = facts["final_result"]
        result_keys = {"status", "initial_ce", "final_ce", "final_to_initial_ce_ratio", "valid_numerator",
            "valid_denominator", "physical_optimizer_updates", "segment_physical_optimizer_updates",
            "net_optimizer_steps", "examples_seen", "supervised_tokens_seen", "updates", "predictions",
            "resume", "evaluation_state_preserved", "checkpoint_hashes", "initial_state_sha256",
            "initial_state_hash_recipe", "initial_evaluation_receipt", "final_evaluation_receipt",
            "recovery_phase", "artifact_paths", "elapsed_seconds", "costs", "trainable_parameters",
            "frozen_parameters", "latent_slots", "latent_dim", "workspace_states"}
        if set(facts) != common | {"initial_evaluation_receipt", "resume_probe", "final_result"} or \
                not isinstance(result, dict) or set(result) != result_keys:
            raise RecoveryError("complete phase result facts are malformed")
        final_receipt = result.get("final_evaluation_receipt")
        if not _valid_eval_ref(final_receipt, progress["optimizer_step"]):
            raise RecoveryError("complete result lacks a final evaluation receipt")
        for key in ("initial_evaluation_receipt", "final_evaluation_receipt"):
            if not _valid_eval_ref(result.get(key), 0 if key.startswith("initial") else progress["optimizer_step"]):
                raise RecoveryError("complete result receipt binding is malformed")
        final_segment = Path(final_receipt["records_path"]).parts[0]
        if result["initial_evaluation_receipt"] != initial or final_segment != segment or \
                Path(final_receipt["receipt_path"]).parts[0] != segment or \
                re.fullmatch(rf"predictions-step-{progress['optimizer_step']:06d}-[0-9a-f]{{32}}\.jsonl",
                    Path(final_receipt["records_path"]).name) is None or \
                re.fullmatch(rf"evaluation-step-{progress['optimizer_step']:06d}-[0-9a-f]{{32}}\.receipt\.json",
                    Path(final_receipt["receipt_path"]).name) is None:
            raise RecoveryError("complete evaluation receipts do not bind this segment")
        for key in ("initial_ce", "final_ce", "final_to_initial_ce_ratio"):
            number = result.get(key)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
                raise RecoveryError("complete result metrics must be finite and positive")
        if result["final_to_initial_ce_ratio"] != result["final_ce"] / result["initial_ce"]:
            raise RecoveryError("complete result CE ratio arithmetic mismatch")
        if type(result.get("valid_numerator")) is not int or result["valid_numerator"] not in (0, 1, 2, 3) or result.get("valid_denominator") != 3:
            raise RecoveryError("complete result validity counts are malformed")
        if result.get("net_optimizer_steps") != progress["optimizer_step"] or result.get("status") not in {
                "PASS_TINY_SOURCE_OVERFIT", "NO_GO_TINY_SOURCE_OVERFIT"}:
            raise RecoveryError("complete result status/progress mismatch")
    elif has_final:
        raise RecoveryError("final result is only valid in complete phase")


def _valid_rel(value) -> bool:
    return isinstance(value, str) and bool(value) and not Path(value).is_absolute() and \
        all(part not in {".", ".."} for part in Path(value).parts)


def _valid_eval_ref(value, step: int) -> bool:
    required = {"schema_version", "step", "example_ids", "checkpoint_sha256", "records_path",
        "records_sha256", "weighted_ce_numerator", "eos_inclusive_token_denominator",
        "strict_valid_numerator", "strict_valid_denominator", "per_example", "receipt_path", "receipt_sha256"}
    if not isinstance(value, dict) or set(value) != required or value.get("schema_version") != 1 or value.get("step") != step:
        return False
    if any(not _valid_rel(value.get(k)) for k in ("records_path", "receipt_path")):
        return False
    if any(not isinstance(value.get(k), str) or not _SHA.fullmatch(value[k]) for k in
           ("checkpoint_sha256", "records_sha256", "receipt_sha256")):
        return False
    record_segment = Path(value["records_path"]).parts[0]
    receipt_parts = Path(value["receipt_path"]).parts
    if not re.fullmatch(r"segment-[0-9]{3}", record_segment) or len(receipt_parts) != 2 or receipt_parts[0] != record_segment or \
            re.fullmatch(rf"predictions-step-{step:06d}-[0-9a-f]{{32}}\.jsonl", Path(value["records_path"]).name) is None or \
            re.fullmatch(rf"evaluation-step-{step:06d}-[0-9a-f]{{32}}\.receipt\.json", receipt_parts[1]) is None:
        return False
    if type(value.get("eos_inclusive_token_denominator")) is not int or value["eos_inclusive_token_denominator"] <= 0:
        return False
    if type(value.get("strict_valid_denominator")) is not int or value["strict_valid_denominator"] != 3 or \
            type(value.get("strict_valid_numerator")) is not int or not 0 <= value["strict_valid_numerator"] <= 3:
        return False
    if isinstance(value.get("weighted_ce_numerator"), bool) or not isinstance(value.get("weighted_ce_numerator"), (int,float)) or \
            not math.isfinite(value["weighted_ce_numerator"]) or value["weighted_ce_numerator"] <= 0:
        return False
    ids = value.get("example_ids")
    rows = value.get("per_example")
    return isinstance(ids, list) and len(ids) == 3 and len(set(ids)) == 3 and \
        isinstance(rows, list) and len(rows) == 3 and [r.get("example_id") for r in rows if isinstance(r, dict)] == ids and \
        all(set(r) == {"example_id", "noise_seed", "noise_sha256", "prompt_sha256", "prediction_sha256"} and
            type(r["noise_seed"]) is int and all(_SHA.fullmatch(str(r[k])) for k in
            ("noise_sha256", "prompt_sha256", "prediction_sha256")) for r in rows)


class RecoveryStore:
    """Publish one atomic authoritative cursor after checkpoint and metadata durability."""
    def __init__(self, jobroot: os.PathLike | str, segment_dir: os.PathLike | str, identities: dict,
                 archival_steps=()):
        _reject_links(Path(jobroot))
        _reject_links(Path(segment_dir))
        self.jobroot = Path(jobroot).resolve(strict=True)
        self.segment_dir = Path(segment_dir).resolve(strict=True)
        try:
            self.segment_dir.relative_to(self.jobroot)
        except ValueError as error:
            raise RecoveryError("segment must be inside job root") from error
        if not re.fullmatch(r"segment-[0-9]{3}", self.segment_dir.name):
            raise RecoveryError("segment directory must use segment-NNN naming")
        self.identities = _validate_identities(identities)
        if any(type(step) is not int or step < 0 for step in archival_steps):
            raise RecoveryError("archival steps must be nonnegative integers")
        self.archival_steps = frozenset(archival_steps)
        self._committed_pairs = []
        self.recovery_dir = self.segment_dir / "recovery"
        self.recovery_dir.mkdir(exist_ok=True)
        _safe_tree_path(self.jobroot, str(self.segment_dir.relative_to(self.jobroot)), must_exist=True)
        self.cursor_path = self.jobroot / "recovery.latest.json"

    def commit(self, checkpoint_path: os.PathLike | str, checkpoint_sha256: str, *,
               progress: dict, phase: str, facts: dict) -> dict:
        checkpoint_path = Path(checkpoint_path)
        try:
            lexical_relative = checkpoint_path.absolute().relative_to(self.jobroot)
        except ValueError as error:
            raise RecoveryError("checkpoint must be inside the job root") from error
        _safe_tree_path(self.jobroot, lexical_relative.as_posix(), must_exist=True)
        checkpoint_path = checkpoint_path.resolve(strict=True)
        try:
            checkpoint_path.relative_to(self.recovery_dir.resolve(strict=True))
        except ValueError as error:
            raise RecoveryError("checkpoint must be an immediate recovery-directory child") from error
        if checkpoint_path.parent != self.recovery_dir.resolve(strict=True):
            raise RecoveryError("checkpoint must be an immediate recovery-directory child")
        _safe_tree_path(self.jobroot, str(checkpoint_path.relative_to(self.jobroot)), must_exist=True)
        _fsync_dir(checkpoint_path.parent)
        if not _SHA.fullmatch(checkpoint_sha256) or _hash(checkpoint_path) != checkpoint_sha256:
            raise RecoveryError("checkpoint bytes do not match the supplied digest")
        progress = _validate_progress(progress)
        if phase not in _PHASES:
            raise RecoveryError("recovery phase is invalid")
        metadata = {"schema_version": _VERSION, "segment": self.segment_dir.name,
            "identities": self.identities, "progress": progress, "phase": phase,
            "checkpoint_path": checkpoint_path.relative_to(self.jobroot).as_posix(),
            "checkpoint_sha256": checkpoint_sha256, "facts": facts}
        _validate_metadata(metadata)
        if self.cursor_path.exists():
            previous = self.verify_descriptor({"jobroot": str(self.jobroot),
                "cursor_path": str(self.cursor_path), "cursor_sha256": _hash(self.cursor_path)}, self.identities)
            same_segment = previous["segment"] == self.segment_dir.name
            old_step, new_step = previous["progress"]["optimizer_step"], progress["optimizer_step"]
            old_phase = previous["phase"]
            if same_segment:
                allowed = ((old_phase == "initial_evaluation_pending" and phase == "training" and new_step == 0) or
                    (old_phase == "training" and phase == "training" and new_step == old_step + 1) or
                    (old_phase == "training" and phase == "resume_probe_pending" and new_step == old_step + 1) or
                    (old_phase == "resume_probe_pending" and phase == "training" and new_step == old_step + 1) or
                    (old_phase == "training" and phase == "final_evaluation_pending" and new_step == old_step) or
                    (old_phase == "final_evaluation_pending" and phase == "complete" and new_step == old_step))
                if not allowed:
                    raise RecoveryError("recovery phase/progress transition is invalid")
            elif (new_step != old_step or phase != old_phase):
                raise RecoveryError("new segment must first publish the verified prior boundary unchanged")
        generations = list(self.recovery_dir.glob(f"metadata-step-{progress['optimizer_step']:06d}-*.json"))
        generation = len(generations)
        metadata_path = self.recovery_dir / f"metadata-step-{progress['optimizer_step']:06d}-{generation:04d}.json"
        metadata_raw = _canonical(metadata)
        _exclusive_write(metadata_path, metadata_raw)
        pointer = {"schema_version": _VERSION, "metadata_path": metadata_path.relative_to(self.jobroot).as_posix(),
            "metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
            "checkpoint_path": metadata["checkpoint_path"], "checkpoint_sha256": checkpoint_sha256,
            "segment": self.segment_dir.name, "identities": self.identities,
            "progress": progress, "phase": phase}
        pointer_raw = _canonical(pointer)
        temp = self.cursor_path.with_name(f".recovery.latest.{uuid.uuid4().hex}.tmp")
        _exclusive_write(temp, pointer_raw)
        os.replace(temp, self.cursor_path)
        _fsync_dir(self.jobroot)
        verified = self.verify_descriptor({"jobroot": str(self.jobroot),
            "cursor_path": str(self.cursor_path), "cursor_sha256": hashlib.sha256(pointer_raw).hexdigest()},
            self.identities)
        if verified["metadata_sha256"] != pointer["metadata_sha256"]:
            raise RecoveryError("new cursor failed post-publication verification")
        self._committed_pairs.append({"step": progress["optimizer_step"],
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256,
            "metadata": str(metadata_path), "metadata_sha256": pointer["metadata_sha256"]})
        self._retire_old_rolling(progress["optimizer_step"], metadata_path)
        return {"jobroot": str(self.jobroot), "cursor_path": str(self.cursor_path),
                "cursor_sha256": hashlib.sha256(pointer_raw).hexdigest()}

    def _retire_old_rolling(self, current_step: int, current_metadata: Path) -> None:
        steps = sorted({pair["step"] for pair in self._committed_pairs
            if pair["step"] > 0 and pair["step"] not in self.archival_steps}, reverse=True)
        keep_steps = set(steps[:2]) | set(self.archival_steps) | {0, current_step}
        targets = {}
        for pair in self._committed_pairs:
            step = pair["step"]
            if step <= 0 or step in keep_steps:
                continue
            for key, digest_key in (("checkpoint", "checkpoint_sha256"), ("metadata", "metadata_sha256")):
                path = Path(pair[key])
                if path.exists():
                    targets[path] = pair[digest_key]
        if not targets:
            return
        receipt = self.recovery_dir / "retirements.jsonl"
        for path in sorted(targets):
            if path.parent != self.recovery_dir or path.is_symlink() or not path.is_file():
                raise RecoveryError("refusing unsafe rolling checkpoint retirement")
            attrs = getattr(path.lstat(), "st_file_attributes", 0)
            if attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise RecoveryError("refusing reparse point retirement")
            matching = [pair for pair in self._committed_pairs if path in {Path(pair["checkpoint"]), Path(pair["metadata"])}]
            if not matching:
                raise RecoveryError("retirement target is absent from committed-pair ledger")
            actual_hash = _hash(path)
            if actual_hash != targets[path]:
                raise RecoveryError("committed retirement pair changed since publication")
            record = {"path": path.name, "sha256": actual_hash, "step": matching[0]["step"]}
            with receipt.open("ab") as stream:
                stream.write(_canonical(record)); stream.flush(); os.fsync(stream.fileno())
            path.unlink()
        _fsync_dir(self.recovery_dir)

    @staticmethod
    def verify_descriptor(descriptor: object, identities: dict) -> dict:
        if not isinstance(descriptor, dict) or set(descriptor) != _DESCRIPTOR_KEYS:
            raise RecoveryError("resume descriptor has missing or unknown fields")
        _reject_links(Path(descriptor["jobroot"]))
        _reject_links(Path(descriptor["cursor_path"]))
        root = Path(descriptor["jobroot"]).resolve(strict=True)
        cursor_path = Path(descriptor["cursor_path"]).resolve(strict=True)
        if cursor_path != root / "recovery.latest.json":
            raise RecoveryError("resume cursor must be authoritative jobroot pointer")
        _safe_tree_path(root, cursor_path.name, must_exist=True)
        raw = cursor_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if not isinstance(descriptor["cursor_sha256"], str) or digest != descriptor["cursor_sha256"]:
            raise RecoveryError("resume cursor digest mismatch")
        try:
            cursor = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RecoveryError("resume cursor is invalid UTF-8 JSON") from error
        if not isinstance(cursor, dict) or set(cursor) != _CURSOR_KEYS:
            raise RecoveryError("resume cursor has missing or unknown fields")
        if raw != _canonical(cursor):
            raise RecoveryError("resume cursor is not stable canonical JSON")
        if type(cursor["schema_version"]) is not int or cursor["schema_version"] != _VERSION:
            raise RecoveryError("unsupported cursor version")
        wanted = _validate_identities(identities)
        if _validate_identities(cursor["identities"]) != wanted:
            raise RecoveryError("resume identities differ from this run")
        _validate_progress(cursor["progress"])
        if cursor["phase"] not in _PHASES:
            raise RecoveryError("resume cursor phase is invalid")
        metadata_path = _safe_tree_path(root, cursor["metadata_path"], must_exist=True)
        expected_metadata_parent = f"{cursor['segment']}/recovery"
        if (metadata_path.parent.relative_to(root).as_posix() != expected_metadata_parent or
            re.fullmatch(rf"metadata-step-{cursor['progress']['optimizer_step']:06d}-[0-9]{{4}}\.json",
                         metadata_path.name) is None):
            raise RecoveryError("metadata path does not bind this segment and progress")
        if _hash(metadata_path) != cursor["metadata_sha256"]:
            raise RecoveryError("resume metadata digest mismatch")
        metadata_raw = metadata_path.read_bytes()
        try:
            metadata = json.loads(metadata_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RecoveryError("resume metadata is invalid UTF-8 JSON") from error
        if metadata_raw != _canonical(metadata):
            raise RecoveryError("resume metadata is not stable canonical JSON")
        _validate_metadata(metadata)
        for key in ("segment", "identities", "progress", "phase", "checkpoint_path", "checkpoint_sha256"):
            if cursor[key] != metadata[key]:
                raise RecoveryError(f"cursor and metadata disagree on {key}")
        checkpoint_path = _safe_tree_path(root, cursor["checkpoint_path"], must_exist=True)
        expected_checkpoint = (Path(cursor["segment"]) / "recovery" /
            f"recovery-checkpoint-step-{cursor['progress']['optimizer_step']:06d}.pt").as_posix()
        if cursor["checkpoint_path"] != expected_checkpoint:
            raise RecoveryError("checkpoint path does not bind this segment and progress")
        if _hash(checkpoint_path) != cursor["checkpoint_sha256"]:
            raise RecoveryError("resume checkpoint digest mismatch")
        return {**metadata, "metadata_path": str(metadata_path), "metadata_sha256": cursor["metadata_sha256"],
                "checkpoint_path_absolute": str(checkpoint_path), "cursor_path": str(cursor_path),
                "cursor_sha256": digest, "jobroot": str(root)}
