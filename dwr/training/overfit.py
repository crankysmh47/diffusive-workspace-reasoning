"""Bounded native source-example learning, evaluation, and replay kernel."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import stat
import time
import uuid

import numpy as np
import torch

from dwr.evaluation.generation import greedy_workspace, score_generation
from .batches import collate_source_examples
from .checkpoints import load_training_checkpoint, save_training_checkpoint
from .loss import answer_only_loss
from .recovery import RecoveryStore
from .sampler import SourceSampler


class NumericalDivergence(RuntimeError):
    """A scientific nonfinite outcome with a usable partial-artifact handoff."""
    def __init__(self, message: str, *, partial: dict[str, object]):
        super().__init__(message)
        self.partial = partial


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _clone_state(module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _nested_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _nested_cpu(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_nested_cpu(v) for v in value)
    if isinstance(value, list):
        return [_nested_cpu(v) for v in value]
    return copy.deepcopy(value)


def _float_errors(left, right) -> tuple[float, float]:
    max_abs = max_rel = 0.0
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape:
            raise ValueError("resume tensor shape differs")
        a, b = left.detach().double().cpu(), right.detach().double().cpu()
        if a.numel():
            difference = (a-b).abs()
            max_abs = float(difference.max().item())
            max_rel = float((difference / b.abs().clamp_min(1e-30)).max().item())
        return max_abs, max_rel
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError("resume mapping keys differ")
        pairs = zip(left.values(), right.values())
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError("resume sequence shape differs")
        pairs = zip(left, right)
    else:
        if isinstance(left, float):
            return abs(left-right), abs(left-right)/max(abs(right), 1e-30)
        if left != right:
            raise ValueError("resume discrete state differs")
        return 0.0, 0.0
    for a, b in pairs:
        x, y = _float_errors(a, b)
        max_abs, max_rel = max(max_abs, x), max(max_rel, y)
    return max_abs, max_rel


def _within_tolerance(left, right, atol: float, rtol: float) -> tuple[float, float, bool]:
    """Compare all floating leaves elementwise; retain exact discrete leaves."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("resume tensor identity differs")
        if left.is_floating_point() or left.is_complex():
            a, b = left.detach().double().cpu(), right.detach().double().cpu()
            delta = (a-b).abs()
            absolute = float(delta.max().item()) if delta.numel() else 0.0
            relative = float((delta / b.abs().clamp_min(1e-30)).max().item()) if delta.numel() else 0.0
            within = bool((delta <= atol + rtol*b.abs()).all().item())
            return absolute, relative, within
        return 0.0, 0.0, torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError("resume mapping keys differ")
        pairs = zip(left.values(), right.values())
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError("resume sequence shape differs")
        pairs = zip(left, right)
    elif isinstance(left, float):
        delta = abs(left-right)
        return delta, delta/max(abs(right), 1e-30), delta <= atol + rtol*abs(right)
    else:
        return 0.0, 0.0, left == right
    absolute = relative = 0.0
    within = True
    for a, b in pairs:
        aa, rr, ok = _within_tolerance(a, b, atol, rtol)
        absolute, relative, within = max(absolute, aa), max(relative, rr), within and ok
    return absolute, relative, within


def _all_finite(parameters) -> bool:
    return all(torch.isfinite(p).all().item() for p in parameters)


def _noise_seed(base: int, example_id: str) -> int:
    raw = f"dwr-source-overfit-eval-v1:{base}:{example_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63)


def _initial_state_digest(workspace, readers) -> tuple[str, dict[str, object]]:
    digest = hashlib.sha256()
    recipe = "dwr-source-overfit-initial-state-v1"
    digest.update(recipe.encode("ascii") + b"\0")
    for module_name, module in (("readers", readers), ("workspace", workspace)):
        for name, tensor in sorted(module.state_dict().items()):
            full_name = f"{module_name}.{name}".encode("utf-8")
            tensor = tensor.detach().cpu().contiguous()
            shape = json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
            dtype = str(tensor.dtype).encode("ascii")
            for value in (full_name, shape, dtype):
                digest.update(struct.pack(">Q", len(value)))
                digest.update(value)
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(struct.pack(">Q", len(raw)))
            digest.update(raw)
    return digest.hexdigest(), {"name": recipe, "ordering": "module then sorted state name",
        "framing": "big-endian uint64 byte lengths for UTF-8 name, JSON shape, ASCII dtype, and tensor bytes",
        "bytes": "contiguous CPU tensor viewed as uint8"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_receipt_path(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute() or
        any(part in {".", ".."} for part in Path(relative).parts)):
        raise ValueError("receipt path is missing")
    root = root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("receipt path escapes the job root") from error
    current = root
    for part in Path(relative).parts:
        current = current / part
        info = current.lstat()
        attrs = getattr(info, "st_file_attributes", 0)
        if os.path.islink(current) or attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError("receipt path traverses a link or reparse point")
    return candidate


def verify_evaluation_receipt(jobroot, receipt_ref, *, expected_step=None,
                              expected_example_ids=None, expected_checkpoint_sha256=None) -> dict:
    """Verify a durable evaluation receipt and its complete prediction record set.

    This intentionally has no model dependency beyond JSON/hash checks, so the
    controller can reuse it before sealing a completed job.
    """
    root = Path(jobroot).resolve(strict=True)
    required = {"schema_version", "step", "example_ids", "checkpoint_sha256", "records_path",
        "records_sha256", "weighted_ce_numerator", "eos_inclusive_token_denominator",
        "strict_valid_numerator", "strict_valid_denominator", "per_example", "receipt_path", "receipt_sha256"}
    if not isinstance(receipt_ref, dict) or set(receipt_ref) != required:
        raise ValueError("evaluation receipt reference schema mismatch")
    if receipt_ref["schema_version"] != 1 or type(receipt_ref["step"]) is not int:
        raise ValueError("evaluation receipt version or step is invalid")
    if expected_step is not None and receipt_ref["step"] != expected_step:
        raise ValueError("evaluation receipt step mismatch")
    if expected_example_ids is not None and receipt_ref["example_ids"] != list(expected_example_ids):
        raise ValueError("evaluation receipt example IDs mismatch")
    if expected_checkpoint_sha256 is not None and receipt_ref["checkpoint_sha256"] != expected_checkpoint_sha256:
        raise ValueError("evaluation receipt checkpoint mismatch")
    receipt_path = _safe_receipt_path(root, receipt_ref["receipt_path"])
    raw = receipt_path.read_bytes()
    if _sha(raw) != receipt_ref["receipt_sha256"]:
        raise ValueError("evaluation receipt bytes do not match cursor reference")
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation receipt is invalid JSON") from error
    body = {key: value for key, value in receipt_ref.items() if key not in {"receipt_path", "receipt_sha256"}}
    if receipt != body or raw != (json.dumps(receipt, sort_keys=True, separators=(",", ":"),
            allow_nan=False)+"\n").encode("utf-8"):
        raise ValueError("evaluation receipt content is not canonical or disagrees with cursor")
    records_path = _safe_receipt_path(root, receipt_ref["records_path"])
    records_raw = records_path.read_bytes()
    if _sha(records_raw) != receipt_ref["records_sha256"]:
        raise ValueError("prediction record bytes do not match evaluation receipt")
    try:
        rows = [json.loads(line) for line in records_raw.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prediction record set is invalid JSONL") from error
    ids = receipt_ref["example_ids"]
    if not isinstance(ids, list) or len(ids) != 3 or len(set(ids)) != 3 or \
            [row.get("example_id") for row in rows if isinstance(row, dict)] != ids or len(rows) != 3:
        raise ValueError("prediction record IDs/order do not match receipt")
    per_example = receipt_ref["per_example"]
    if len(per_example) != 3:
        raise ValueError("evaluation receipt does not bind three examples")
    ce_numerator = 0.0
    token_count = 0
    valid_count = 0
    for row, bound in zip(rows, per_example):
        if row.get("step") != receipt_ref["step"] or row.get("checkpoint_sha256") != receipt_ref["checkpoint_sha256"]:
            raise ValueError("prediction row checkpoint/step binding mismatch")
        for key in ("example_id", "noise_seed", "noise_sha256", "prompt_sha256", "prediction_sha256"):
            if row.get(key) != bound.get(key):
                raise ValueError("prediction row identity differs from evaluation receipt")
        pred_hash = _sha(json.dumps(row.get("generated_token_ids"), separators=(",", ":")).encode("ascii"))
        if pred_hash != bound["prediction_sha256"]:
            raise ValueError("prediction tokens do not match bound digest")
        ce = row.get("teacher_forced_ce")
        supervised = row.get("supervised_tokens")
        if isinstance(ce, bool) or not isinstance(ce, (int, float)) or not math.isfinite(ce) or ce <= 0 or \
                type(supervised) is not int or supervised <= 0:
            raise ValueError("prediction row CE/token evidence is invalid")
        ce_numerator += float(ce) * supervised
        token_count += supervised
        valid_count += int(row.get("valid") is True)
    stored_ce = receipt_ref["weighted_ce_numerator"]
    if isinstance(stored_ce, bool) or not isinstance(stored_ce, (int, float)) or \
            not math.isclose(ce_numerator, stored_ce, rel_tol=1e-12, abs_tol=1e-12) or \
            token_count != receipt_ref["eos_inclusive_token_denominator"] or \
            valid_count != receipt_ref["strict_valid_numerator"] or receipt_ref["strict_valid_denominator"] != 3:
        raise ValueError("evaluation receipt aggregate arithmetic mismatch")
    return {"receipt": receipt, "rows": rows, "ce": stored_ce/token_count,
            "valid": valid_count, "token_count": token_count}


def _value_digest(value) -> str:
    """Stable recursive digest for small checkpoint/control states and RNG tuples."""
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            add(("tensor", str(tensor.dtype), tuple(tensor.shape)))
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(struct.pack(">Q", len(raw))); digest.update(raw)
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda x: str(x)):
                add(str(key)); add(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(b"tuple\0" if isinstance(item, tuple) else b"list\0")
            for element in item: add(element)
        else:
            raw = repr(item).encode("utf-8")
            digest.update(struct.pack(">Q", len(raw))); digest.update(raw)
    add(value)
    return digest.hexdigest()


_NONFINITE_MESSAGES = {
    "captured prompt state contains nonfinite values", "prompt_states must be finite",
    "noise must be finite", "initial workspace state is nonfinite",
    "workspace state became nonfinite", "queries must be finite", "workspace must be finite",
    "reader output must be finite", "native decoder produced nonfinite logits",
    "decoder logits must be finite", "logits must be finite", "answer-only loss is nonfinite",
}


def _raise_known_nonfinite(error: ValueError) -> None:
    if str(error) in _NONFINITE_MESSAGES:
        raise ArithmeticError(str(error)) from error


def _make_optimizer(host, training):
    opt = training["optimizer"]
    groups = [{"params": list(host.workspace.parameters()), "name": "workspace"},
              {"params": list(host.readers.parameters()), "name": "readers"}]
    optimizer = torch.optim.AdamW(groups, lr=opt["learning_rate"], betas=tuple(opt["betas"]),
        eps=opt["epsilon"], weight_decay=opt["weight_decay"], amsgrad=opt["amsgrad"],
        capturable=opt["capturable"], differentiable=opt["differentiable"],
        foreach=opt["foreach"], fused=opt["fused"], maximize=opt["maximize"])
    warmup = training["schedule"]["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda scheduler_step: min(1.0, (scheduler_step+1)/warmup))
    return optimizer, scheduler


def _make_sampler_noise(ids, config, device):
    sampler = SourceSampler(tuple(ids), seed=config["seeds"]["source_sampler"])
    noise = torch.Generator(device=device).manual_seed(config["seeds"]["workspace_noise_cuda"])
    return sampler, noise


def _global_rng():
    state = np.random.get_state()
    numpy_state = (state[0], torch.from_numpy(state[1].copy()), int(state[2]), int(state[3]), float(state[4]))
    return (random.getstate(), numpy_state, torch.get_rng_state().clone(),
            tuple(x.clone() for x in torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else ())


def _restore_global_rng(state):
    random.setstate(state[0])
    n = state[1]
    np.random.set_state((n[0], n[1].numpy().copy(), n[2], n[3], n[4]))
    torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(list(state[3]))


def run_overfit(host, tokenizer, tokenizer_identity, examples, *, config, identities, metadata,
                output_dir, check, on_event, resume=None) -> dict:
    """Run a strict three-example answer-only overfit on the caller's native host."""
    from dwr.models.qwen2 import Qwen2WorkspaceLM
    from dwr.training.batches import SourceExample, TokenizerIdentity as TokenizerIdentityType
    if not isinstance(host, Qwen2WorkspaceLM):
        raise TypeError("host must be an actual Qwen2WorkspaceLM")
    if type(examples) is not tuple or len(examples) != 3 or any(type(x) is not SourceExample for x in examples):
        raise ValueError("examples must be an ordered tuple of exactly three SourceExample values")
    if tuple(x.split for x in examples) != ("source_train",)*3 or len({x.example_id for x in examples}) != 3:
        raise ValueError("examples must be three distinct source_train rows")
    profiles = tuple(next(iter({c.kind.value for c in x.task.constraints}), None) for x in examples)
    if profiles != ("eq", "neq", "add_mod"):
        raise ValueError("examples must be ordered EQ, NEQ, ADD_MOD")
    if set(metadata) != {x.example_id for x in examples}:
        raise ValueError("metadata must exactly cover the selected source IDs")
    for ex in examples:
        row = metadata[ex.example_id]
        kinds = {c.kind.value for c in ex.task.constraints}
        profile = next(iter(kinds)) if len(kinds) == 1 else None
        if row.get("profile") != profile or not all(isinstance(row.get(k), str) for k in
                ("lineage_id", "structure_signature", "fine_stratum")):
            raise ValueError("example metadata does not match its strict task profile")
    if set(identities) != {"source", "config", "data", "model", "tokenizer", "schedule"}:
        raise ValueError("identities must contain exactly the six checkpoint hashes")
    if not isinstance(tokenizer_identity, TokenizerIdentityType):
        raise TypeError("tokenizer_identity must be the caller's TokenizerIdentity")
    if not tokenizer_identity.name.strip() or not tokenizer_identity.revision.strip():
        raise ValueError("tokenizer identity name and revision must be nonempty")
    if not callable(check) or not callable(on_event):
        raise TypeError("check and on_event must be callable")
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise ValueError("output_dir must already exist")
    t = config["training"]
    total_steps = t["optimizer_steps"]
    checkpoint_steps = tuple(t["checkpoint_steps"])
    resume_at = config["resume"]["checkpoint_step"]
    replay_step = config["resume"]["replayed_step"]
    if (total_steps < 1 or resume_at not in checkpoint_steps or total_steps not in checkpoint_steps or
        replay_step != resume_at+1 or total_steps < replay_step):
        raise ValueError("invalid training/resume step layout")
    if (t.get("method") != "EX" or host.workspace.identity_mode != "exchangeable" or
        t["batch_size"] != 1 or t["steps"] != host.workspace.max_steps or
        host.workspace.num_slots != t["num_slots"] or host.workspace.latent_dim != t["latent_dim"]):
        raise ValueError("host workspace layout differs from resolved config")
    if tuple(config["evaluation"]["steps"]) != (0, total_steps):
        raise ValueError("evaluation steps must bind initialization and final checkpoint")
    device = next(host.workspace.parameters()).device
    dtype = next(host.workspace.parameters()).dtype
    original_params = tuple(host.backbone.parameters())
    original_requires = tuple(p.requires_grad for p in original_params)
    if any(original_requires) or any(p.grad is not None for p in original_params):
        raise ValueError("original backbone must be frozen and gradient-free")
    host_mode = {m: m.training for m in host.modules()}
    sampler, noise_generator = _make_sampler_noise([x.example_id for x in examples], config, device)
    optimizer, scheduler = _make_optimizer(host, t)
    if re.fullmatch(r"segment-[0-9]{3}", output_dir.name) is None:
        raise ValueError("output_dir must be a fresh segment-NNN directory")
    entries = list(output_dir.iterdir())
    if entries:
        expected_record_files = {"manifest.initial.json", "events.jsonl"}
        if {path.name for path in entries} != expected_record_files:
            raise ValueError("output_dir must be empty or contain only RunRecord initial files")
        for path in entries:
            info = path.lstat()
            attrs = getattr(info, "st_file_attributes", 0)
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or \
                    attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ValueError("output_dir must be empty or contain only regular RunRecord initial files")
    jobroot = output_dir.parent
    store = RecoveryStore(jobroot, output_dir, identities, archival_steps=checkpoint_steps)
    log_path = output_dir / "updates.jsonl"
    progress = {"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0}
    physical = 0
    net_tokens = 0
    last_good = 0
    start = time.perf_counter()
    rows: list[dict] = []
    predictions: list[dict] = []
    resume_result = {}
    checkpoint_hashes = {}
    checkpoint_paths = {}
    recovery_checkpoint_paths = {}
    recovery_descriptor = None
    resume_metadata = None
    initial_receipt = None
    final_receipt = None
    phase = "initial_evaluation_pending"
    recovery_facts = {}

    def commit_boundary(checkpoint_path, digest, *, boundary_phase, facts):
        nonlocal recovery_descriptor, phase, recovery_facts
        recovery_descriptor = store.commit(checkpoint_path, digest, progress=progress,
                                           phase=boundary_phase, facts=facts)
        verified = RecoveryStore.verify_descriptor(recovery_descriptor, identities)
        if verified["progress"] != progress:
            raise RuntimeError("published recovery cursor failed its progress receipt")
        phase = boundary_phase
        recovery_facts = facts
        on_event("recovery_committed", {"phase": phase, "progress": dict(progress),
            "cursor_sha256": recovery_descriptor["cursor_sha256"],
            "checkpoint_path": verified["checkpoint_path_absolute"],
            "checkpoint_sha256": verified["checkpoint_sha256"],
            "metadata_path": verified["metadata_path"], "metadata_sha256": verified["metadata_sha256"]})

    def save_boundary(step, *, boundary_phase, facts):
        recovery_dir = output_dir / "recovery"
        path = recovery_dir / f"recovery-checkpoint-step-{step:06d}.pt"
        digest, _ = save_training_checkpoint(path, host=host, optimizer=optimizer,
            scheduler=scheduler, scaler=None, sampler=sampler, noise_generator=noise_generator,
            identities=identities, progress=progress)
        checkpoint_hashes[str(step)] = digest
        recovery_checkpoint_paths[str(step)] = str(path)
        checkpoint_paths[str(step)] = str(path)
        if step in checkpoint_steps:
            archival = output_dir / f"checkpoint-step-{step:06d}.pt"
            with path.open("rb") as source, archival.open("xb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk: break
                    target.write(chunk)
                target.flush(); os.fsync(target.fileno())
            checkpoint_paths[str(step)] = str(archival)
        commit_boundary(path, digest, boundary_phase=boundary_phase, facts=facts)
        return path, digest

    def write_row(path, row):
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)+"\n")
            stream.flush(); os.fsync(stream.fileno())

    def batch_for(example):
        return collate_source_examples([example], tokenizer,
            tokenizer_identity=tokenizer_identity,
            vocab_size=config["decoding"]["embedding_vocab_size"],
            max_prompt_tokens=config["decoding"]["max_prompt_tokens"],
            max_answer_tokens=config["decoding"].get("max_answer_tokens", config["decoding"]["max_new_tokens_including_eos"]),
            max_total_tokens=config["decoding"]["max_total_tokens"], device=device)

    def update(role: str, *, increment_net: bool):
        nonlocal physical, net_tokens, last_good, progress
        check()
        on_event("update_started", {"role": role, "net_step": progress["optimizer_step"] + 1,
            "physical_update_index": physical + 1})
        global_rng_before = _value_digest(_global_rng())
        ids = sampler.next_ids(t["batch_size"])
        selected = tuple(next(x for x in examples if x.example_id == id_) for id_ in ids)
        batch = batch_for(selected[0])
        batch_sha256 = _value_digest((batch.prompt_ids, batch.prompt_mask, batch.decoder_ids,
                                      batch.decoder_mask, batch.labels))
        train_noise = host.workspace.sample_noise(1, generator=noise_generator, device=device, dtype=dtype)
        train_noise.requires_grad_(progress["optimizer_step"]+1 in t["diagnose_state_gradient_steps"])
        train_noise_sha256 = _sha(train_noise.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        host.train(True)
        modes_sha256 = _value_digest(tuple((type(module).__qualname__, module.training)
                                           for module in host.modules()))

        def gradient_fault(message: str, stage: str):
            row = {"role": "failed_update", "physical_update": physical,
                "net_step": progress["optimizer_step"], "example_ids": list(ids),
                "learning_rate": float(lr), "supervised_tokens": None,
                "failure_kind": "gradient_invariant", "stage": stage, "message": message}
            write_row(log_path, row)
            on_event("gradient_invariant_failure", row)
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(message)

        diagnostic = progress["optimizer_step"]+1 in t["diagnose_state_gradient_steps"]
        try:
            output = host(batch.prompt_ids, batch.prompt_mask, batch.decoder_ids, batch.decoder_mask,
                          noise=train_noise, steps=host.workspace.max_steps, return_states=True)
        except ValueError as error:
            _raise_known_nonfinite(error)
            raise
        if diagnostic:
            for state in output.states:
                state.retain_grad()
        try:
            loss = answer_only_loss(output.logits, batch.labels)
        except ValueError as error:
            _raise_known_nonfinite(error)
            raise
        if not torch.isfinite(loss.loss).item():
            raise ArithmeticError("nonfinite loss")
        if not loss.loss.requires_grad:
            gradient_fault("answer loss is disconnected from trainable parameters", "loss")
        loss.loss.backward()
        active = [p for p in host.parameters() if p.requires_grad]
        if any(p.grad is None for p in active):
            gradient_fault("an active workspace or reader parameter has no gradient", "active_parameters")
        if any(not torch.isfinite(p.grad).all().item() for p in active):
            raise ArithmeticError("an active workspace or reader gradient is nonfinite")
        state_gradient_norms = []
        if diagnostic:
            for state in output.states:
                if state.grad is None:
                    gradient_fault("a recurrent state has no gradient at a diagnostic step", "recurrent_states")
                if not torch.isfinite(state.grad).all().item():
                    raise ArithmeticError("a recurrent-state gradient is nonfinite")
                norm = float(state.grad.detach().float().norm().item())
                if not math.isfinite(norm):
                    raise ArithmeticError("recurrent-state gradient norm is nonfinite")
                if norm <= 0:
                    gradient_fault("recurrent-state gradient is zero", "recurrent_states")
                state_gradient_norms.append(norm)
        group_norms = {}
        for name, params in (("workspace", host.workspace.parameters()), ("readers", host.readers.parameters())):
            grads = [p.grad.detach().float().norm(2) for p in params if p.grad is not None]
            group_norms[name] = float(torch.stack(grads).norm(2).item()) if grads else 0.0
            if not math.isfinite(group_norms[name]):
                raise ArithmeticError(f"{name} gradient norm is nonfinite")
        reader_gradient_norms = {}
        if diagnostic:
            for reader_name, reader in host.readers.items():
                reader_params = tuple(reader.parameters())
                if any(p.grad is None for p in reader_params):
                    gradient_fault(f"reader {reader_name} has a missing gradient", f"reader:{reader_name}")
                if any(not torch.isfinite(p.grad).all().item() for p in reader_params):
                    raise ArithmeticError(f"reader {reader_name} has a nonfinite gradient")
                norm = float(torch.stack([p.grad.detach().float().norm(2) for p in reader_params]).norm(2).item())
                reader_gradient_norms[reader_name] = norm
                if not math.isfinite(norm):
                    raise ArithmeticError(f"reader {reader_name} gradient norm is nonfinite")
                if norm <= 0:
                    gradient_fault(f"reader {reader_name} gradient is zero", f"reader:{reader_name}")
        total_gradient_norm = math.hypot(*group_norms.values())
        if not math.isfinite(total_gradient_norm):
            raise ArithmeticError("total active gradient norm is nonfinite")
        torch.nn.utils.clip_grad_norm_(active, t["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        physical += 1
        if increment_net:
            progress["optimizer_step"] += 1
            progress["examples_seen"] += len(ids)
            progress["supervised_tokens_seen"] += loss.supervised_tokens
            net_tokens += loss.supervised_tokens
        if any(p.grad is not None for p in original_params) or any(p.requires_grad != req for p, req in zip(original_params, original_requires)):
            raise RuntimeError("frozen base parameter contract changed")
        learned_finite = _all_finite([p for p in host.workspace.parameters()] + [p for p in host.readers.parameters()])
        state_norms = [float(s.detach().float().norm().item()) for s in output.states]
        if not all(math.isfinite(norm) for norm in state_norms):
            raise ArithmeticError("recurrent-state norm is nonfinite")
        row = {"role": role, "physical_update": physical, "net_step": progress["optimizer_step"],
            "example_ids": list(ids), "supervised_tokens": loss.supervised_tokens,
            "learning_rate": float(lr), "loss": float(loss.loss.detach().item()),
            "training_noise_sha256": train_noise_sha256,
            "training_batch_sha256": batch_sha256, "training_modes_sha256": modes_sha256,
            "learned_state_sha256": _value_digest((_clone_state(host.workspace), _clone_state(host.readers))),
            "global_rng_sha256_before_update": global_rng_before,
            "gradient_group_norms": group_norms, "recurrent_state_norms": state_norms,
            "recurrent_state_gradient_norms": state_gradient_norms,
            "reader_gradient_norms": reader_gradient_norms, "total_gradient_norm": total_gradient_norm,
            "learned_parameters_finite": learned_finite}
        write_row(log_path, row); rows.append(row); on_event("update", row)
        optimizer.zero_grad(set_to_none=True)
        if not learned_finite:
            raise ArithmeticError("nonfinite learned parameter")
        if increment_net:
            last_good = progress["optimizer_step"]
        return row

    def evaluate(step, checkpoint_hash):
        nonlocal predictions
        evaluation_nonce = uuid.uuid4().hex
        prediction_path = output_dir / f"predictions-step-{step:06d}-{evaluation_nonce}.jsonl"
        prediction_path.open("x", encoding="utf-8", newline="\n").close()
        saved_learned = (_clone_state(host.workspace), _clone_state(host.readers))
        saved_optimizer, saved_scheduler = _nested_cpu(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
        saved_sampler, saved_noise, saved_global = sampler.state_dict(), noise_generator.get_state().clone(), _global_rng()
        prior_modes = {m: m.training for m in host.modules()}
        before_audit = {"learned": _value_digest(saved_learned), "optimizer": _value_digest(saved_optimizer),
            "scheduler": _value_digest(saved_scheduler), "sampler": _value_digest(saved_sampler),
            "global_rng": _value_digest(saved_global), "training_noise_rng": _value_digest(saved_noise),
            "host_modes": _value_digest(tuple((type(m).__qualname__, mode) for m, mode in prior_modes.items()))}
        items = []
        total_loss = total_tokens = 0
        try:
            host.eval()
            for ex in examples:
                check()
                on_event("prediction_started", {"step": step, "example_id": ex.example_id})
                batch = batch_for(ex)
                seed = _noise_seed(config["seeds"]["evaluation_noise_base"], ex.example_id)
                gen = torch.Generator(device=device).manual_seed(seed)
                noise = host.workspace.sample_noise(1, generator=gen, device=device, dtype=dtype)
                noise_hash = _sha(noise.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                try:
                    with torch.no_grad():
                        out = host(batch.prompt_ids, batch.prompt_mask, batch.decoder_ids, batch.decoder_mask,
                                   noise=noise, steps=host.workspace.max_steps)
                        ce = answer_only_loss(out.logits, batch.labels)
                except ValueError as error:
                    _raise_known_nonfinite(error)
                    raise
                total_loss += float(ce.loss.item())*ce.supervised_tokens
                total_tokens += ce.supervised_tokens
                # Generation receives the same draw recreated from the same per-example seed.
                gen = torch.Generator(device=device).manual_seed(seed)
                gen_noise = host.workspace.sample_noise(1, generator=gen, device=device, dtype=dtype)
                generation_noise_hash = _sha(gen_noise.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                if noise_hash != generation_noise_hash:
                    raise RuntimeError("evaluation CE and generation noise draws differ")
                try:
                    result = greedy_workspace(host, batch.prompt_ids, batch.prompt_mask, noise=gen_noise,
                        steps=host.workspace.max_steps, eos_token_id=config["decoding"]["eos_token_id"],
                        max_new_tokens=config["decoding"]["max_new_tokens_including_eos"],
                        max_total_tokens=config["decoding"]["max_total_tokens"], check=check)
                except ValueError as error:
                    _raise_known_nonfinite(error)
                    raise
                scored = score_generation(ex.task, result, tokenizer)
                prompt_sha256 = _sha(batch.prompt_ids.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() +
                    batch.prompt_mask.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                prediction_sha256 = _sha(json.dumps(list(result.token_ids), separators=(",", ":")).encode("ascii"))
                item = {"step": step, "example_id": ex.example_id, "metadata": metadata[ex.example_id],
                    "noise_seed": seed, "noise_sha256": noise_hash,
                    "prompt_sha256": prompt_sha256, "prediction_sha256": prediction_sha256,
                    "generated_token_ids": list(result.token_ids), **scored,
                    "teacher_forced_ce": float(ce.loss.item()), "supervised_tokens": ce.supervised_tokens,
                    "timing_seconds": result.elapsed_wall_seconds, "checkpoint_sha256": checkpoint_hash}
                items.append(item); write_row(prediction_path, item)
                on_event("prediction_completed", {"step": step, "example_id": ex.example_id,
                    "decoder_calls": len(result.token_ids)})
            if _float_errors(saved_learned, (_clone_state(host.workspace), _clone_state(host.readers))) != (0.0, 0.0):
                raise RuntimeError("evaluation changed learned parameters")
            if _float_errors(saved_optimizer, _nested_cpu(optimizer.state_dict())) != (0.0, 0.0) or saved_scheduler != scheduler.state_dict():
                raise RuntimeError("evaluation changed optimizer or scheduler")
            if _float_errors(saved_sampler, sampler.state_dict()) != (0.0, 0.0):
                raise RuntimeError("evaluation changed sampler")
            if not torch.equal(saved_noise, noise_generator.get_state()) or _float_errors(saved_global, _global_rng()) != (0.0, 0.0):
                raise RuntimeError("evaluation changed global or training noise RNG")
        finally:
            for module, mode in prior_modes.items(): module.training = mode
            _restore_global_rng(saved_global)
        after_optimizer, after_scheduler = _nested_cpu(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
        after_sampler, after_noise, after_global = sampler.state_dict(), noise_generator.get_state().clone(), _global_rng()
        after_modes = tuple((type(m).__qualname__, m.training) for m in host.modules())
        after_audit = {"learned": _value_digest((_clone_state(host.workspace), _clone_state(host.readers))),
            "optimizer": _value_digest(after_optimizer), "scheduler": _value_digest(after_scheduler),
            "sampler": _value_digest(after_sampler), "global_rng": _value_digest(after_global),
            "training_noise_rng": _value_digest(after_noise), "host_modes": _value_digest(after_modes)}
        if before_audit != after_audit:
            raise RuntimeError("evaluation state audit changed")
        on_event("evaluation", {"step": step, "checkpoint_sha256": checkpoint_hash,
            "initial_state_sha256": start_hash, "before": before_audit, "after": after_audit,
            "evaluation_noise_sha256_by_id": {row["example_id"]: row["noise_sha256"] for row in items}})
        predictions.extend(items)
        metrics = {"ce": total_loss/total_tokens, "supervised_tokens": total_tokens,
                   "valid": sum(bool(x["valid"]) for x in items), "predictions": items}
        receipt = {"schema_version": 1, "step": step,
            "example_ids": [x.example_id for x in examples],
            "checkpoint_sha256": checkpoint_hash,
            "records_path": prediction_path.relative_to(jobroot).as_posix(),
            "records_sha256": _file_sha256(prediction_path),
            "weighted_ce_numerator": total_loss, "eos_inclusive_token_denominator": total_tokens,
            "strict_valid_numerator": metrics["valid"], "strict_valid_denominator": len(items),
            "per_example": [{"example_id": item["example_id"], "noise_seed": item["noise_seed"],
                "noise_sha256": item["noise_sha256"], "prompt_sha256": item["prompt_sha256"],
                "prediction_sha256": item["prediction_sha256"]} for item in items]}
        receipt_path = output_dir / f"evaluation-step-{step:06d}-{evaluation_nonce}.receipt.json"
        raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")
        with receipt_path.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        metrics["receipt"] = {**receipt, "receipt_path": receipt_path.relative_to(jobroot).as_posix(),
                               "receipt_sha256": _sha(raw)}
        return metrics

    try:
        if resume is None:
            start_hash, initial_hash_recipe = _initial_state_digest(host.workspace, host.readers)
            zero_path, zero_digest = save_boundary(0, boundary_phase="initial_evaluation_pending",
                facts={"initial_state_sha256": start_hash, "initial_state_hash_recipe": initial_hash_recipe})
        else:
            resume_metadata = RecoveryStore.verify_descriptor(resume, identities)
            previous_segment = Path(resume_metadata["metadata_path"]).parent.parent.name
            if previous_segment == output_dir.name or output_dir.parent != Path(resume_metadata["jobroot"]):
                raise ValueError("resume requires a fresh sibling segment in the same job root")
            phase = resume_metadata["phase"]
            saved_step = resume_metadata["progress"]["optimizer_step"]
            valid_phase = ((phase == "initial_evaluation_pending" and saved_step == 0) or
                (phase == "training" and 0 <= saved_step <= total_steps) or
                (phase == "resume_probe_pending" and saved_step == resume_at) or
                (phase in {"final_evaluation_pending", "complete"} and saved_step == total_steps))
            if not valid_phase:
                raise ValueError("recovery phase and optimizer progress are incompatible")
            recovery_facts = resume_metadata["facts"]
            saved_probe = recovery_facts.get("resume_probe")
            if saved_probe is not None:
                probe_path = _safe_receipt_path(jobroot, saved_probe.get("path"))
                if _file_sha256(probe_path) != saved_probe.get("sha256"):
                    raise ValueError("saved resume-probe evidence digest mismatch")
                probe_checkpoint = _safe_receipt_path(jobroot, saved_probe.get("checkpoint_path"))
                if _file_sha256(probe_checkpoint) != saved_probe.get("checkpoint_sha256"):
                    raise ValueError("saved resume-probe checkpoint digest mismatch")
            if phase == "complete":
                if not isinstance(recovery_facts.get("final_result"), dict):
                    raise ValueError("complete recovery cursor lacks its final result")
                stored = recovery_facts["final_result"]
                initial_ref = recovery_facts.get("initial_evaluation_receipt")
                final_ref = stored.get("final_evaluation_receipt")
                if stored.get("initial_evaluation_receipt") != initial_ref:
                    raise ValueError("complete result initial receipt differs from cursor")
                if stored.get("initial_state_sha256") != recovery_facts.get("initial_state_sha256"):
                    raise ValueError("complete result initial state digest differs from cursor fact")
                initial_verified = verify_evaluation_receipt(jobroot, initial_ref, expected_step=0,
                    expected_example_ids=[x.example_id for x in examples],
                    expected_checkpoint_sha256=recovery_facts["initial_state_sha256"])
                final_verified = verify_evaluation_receipt(jobroot, final_ref, expected_step=total_steps,
                    expected_example_ids=[x.example_id for x in examples],
                    expected_checkpoint_sha256=resume_metadata["checkpoint_sha256"])
                if not math.isclose(stored.get("initial_ce", math.nan), initial_verified["ce"], rel_tol=1e-12, abs_tol=1e-12) or \
                        not math.isclose(stored.get("final_ce", math.nan), final_verified["ce"], rel_tol=1e-12, abs_tol=1e-12) or \
                        stored.get("valid_numerator") != final_verified["valid"] or stored.get("valid_denominator") != 3:
                    raise ValueError("complete result metrics disagree with verified evaluation receipts")
                ratio = final_verified["ce"] / initial_verified["ce"]
                if not math.isclose(stored.get("final_to_initial_ce_ratio", math.nan), ratio, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("complete result CE ratio does not recompute")
                if stored.get("predictions") != initial_verified["rows"] + final_verified["rows"]:
                    raise ValueError("complete result predictions differ from verified evaluation records")
                threshold = config["evaluation"]["gate_max_final_to_initial_CE_ratio"]
                status = "PASS_TINY_SOURCE_OVERFIT" if final_verified["valid"] == 3 and ratio <= threshold else "NO_GO_TINY_SOURCE_OVERFIT"
                if stored.get("status") != status:
                    raise ValueError("complete result gate arithmetic/status mismatch")
                probe = recovery_facts["resume_probe"]
                probe_path = _safe_receipt_path(jobroot, probe["path"])
                probe_checkpoint = _safe_receipt_path(jobroot, probe["checkpoint_path"])
                if _file_sha256(probe_path) != probe["sha256"] or _file_sha256(probe_checkpoint) != probe["checkpoint_sha256"]:
                    raise ValueError("complete result resume-probe evidence hash mismatch")
                result = {**stored, "recovery_descriptor": dict(resume)}
                # A verified result advertises only artifacts that still exist.
                for key, value in result.get("artifact_paths", {}).items():
                    if isinstance(value, dict):
                        result["artifact_paths"][key] = {k: v for k, v in value.items() if Path(v).is_file()}
                return result
            loaded_progress = load_training_checkpoint(resume_metadata["checkpoint_path_absolute"],
                expected_sha256=resume_metadata["checkpoint_sha256"], host=host, optimizer=optimizer,
                scheduler=scheduler, scaler=None, sampler=sampler, noise_generator=noise_generator,
                identities=identities)
            if loaded_progress != resume_metadata["progress"]:
                raise ValueError("checkpoint progress disagrees with verified recovery cursor")
            progress = loaded_progress
            net_tokens = progress["supervised_tokens_seen"]
            last_good = progress["optimizer_step"]
            start_hash = recovery_facts["initial_state_sha256"]
            initial_hash_recipe = recovery_facts["initial_state_hash_recipe"]
            checkpoint_path, checkpoint_digest = save_boundary(progress["optimizer_step"],
                boundary_phase=phase, facts=recovery_facts)
            resume_result = recovery_facts.get("resume_probe", {}).get("result", {})
            if recovery_facts.get("resume_probe"):
                resume_result = {**resume_result,
                    "probe_state_path": str(jobroot / recovery_facts["resume_probe"]["path"])}

        def read_initial_receipt(receipt):
            verified = verify_evaluation_receipt(jobroot, receipt, expected_step=0,
                expected_example_ids=[x.example_id for x in examples],
                expected_checkpoint_sha256=start_hash)
            return {"ce": verified["ce"], "supervised_tokens": verified["token_count"],
                "valid": verified["valid"], "predictions": verified["rows"], "receipt": receipt}

        if recovery_facts.get("initial_evaluation_receipt"):
            initial = read_initial_receipt(recovery_facts["initial_evaluation_receipt"])
        else:
            initial = None
        if phase == "initial_evaluation_pending" or initial is None:
            initial = evaluate(0, start_hash)
            if not math.isfinite(initial["ce"]) or initial["ce"] <= 0 or initial["supervised_tokens"] <= 0:
                raise RuntimeError("initial evaluation CE must be finite, positive, and EOS-denominated")
            initial_receipt = initial["receipt"]
            recovery_facts = {"initial_state_sha256": start_hash,
                "initial_state_hash_recipe": initial_hash_recipe,
                "initial_evaluation_receipt": initial_receipt}
            commit_boundary(zero_path if resume is None else checkpoint_path,
                zero_digest if resume is None else checkpoint_digest,
                boundary_phase="training", facts=recovery_facts)
        else:
            initial_receipt = recovery_facts["initial_evaluation_receipt"]
            predictions.extend(initial["predictions"])

        def run_resume_probe():
            nonlocal progress, optimizer, scheduler, sampler, noise_generator, resume_result
            boundary_path = Path(recovery_checkpoint_paths.get(str(resume_at),
                store.verify_descriptor(recovery_descriptor, identities)["checkpoint_path_absolute"]))
            boundary_digest = checkpoint_hashes.get(str(resume_at),
                store.verify_descriptor(recovery_descriptor, identities)["checkpoint_sha256"])
            saved_sampler_boundary = sampler.state_dict()
            first = update("discarded_resume_probe", increment_net=False)
            discarded_progress = copy.deepcopy(progress)
            replay_progress = copy.deepcopy(progress)
            replay_progress["optimizer_step"] += 1
            replay_progress["examples_seen"] += len(first["example_ids"])
            replay_progress["supervised_tokens_seen"] += first["supervised_tokens"]
            branch_state = {"workspace": _clone_state(host.workspace), "readers": _clone_state(host.readers),
                "optimizer": _nested_cpu(optimizer.state_dict()), "scheduler": copy.deepcopy(scheduler.state_dict()),
                "sampler": sampler.state_dict(), "noise": noise_generator.get_state().clone(),
                "global": _global_rng(), "progress": replay_progress,
                "discarded_branch_progress": discarded_progress,
                "loss": first["loss"], "training_noise_sha256": first["training_noise_sha256"]}
            probe_path = output_dir / f"resume-probe-state-{uuid.uuid4().hex}.pt"
            with probe_path.open("xb") as stream:
                torch.save(branch_state, stream)
                stream.flush(); os.fsync(stream.fileno())
            probe_digest = _file_sha256(probe_path)
            fresh_optimizer, fresh_scheduler = _make_optimizer(host, t)
            fresh_sampler, fresh_noise = _make_sampler_noise([x.example_id for x in examples], config, device)
            progress = load_training_checkpoint(boundary_path, expected_sha256=boundary_digest, host=host,
                optimizer=fresh_optimizer, scheduler=fresh_scheduler, scaler=None, sampler=fresh_sampler,
                noise_generator=fresh_noise, identities=identities)
            optimizer, scheduler, sampler, noise_generator = fresh_optimizer, fresh_scheduler, fresh_sampler, fresh_noise
            replay = update("main", increment_net=True)
            actual = {"workspace": _clone_state(host.workspace), "readers": _clone_state(host.readers),
                "optimizer": _nested_cpu(optimizer.state_dict()), "scheduler": copy.deepcopy(scheduler.state_dict()),
                "sampler": sampler.state_dict(), "noise": noise_generator.get_state().clone(),
                "global": _global_rng(), "progress": copy.deepcopy(progress),
                "loss": replay["loss"], "training_noise_sha256": replay["training_noise_sha256"]}
            tol_a, tol_r = config["resume"]["float_atol"], config["resume"]["float_rtol"]
            pe = _within_tolerance(branch_state["workspace"], actual["workspace"], tol_a, tol_r)
            pe2 = _within_tolerance(branch_state["readers"], actual["readers"], tol_a, tol_r)
            oe = _within_tolerance(branch_state["optimizer"], actual["optimizer"], tol_a, tol_r)
            se = _float_errors(branch_state["scheduler"], actual["scheduler"])
            re = _float_errors(branch_state["global"], actual["global"])
            sam = _float_errors(branch_state["sampler"], actual["sampler"])
            loss_error = abs(branch_state["loss"] - actual["loss"])
            loss_relative = loss_error/max(abs(actual["loss"]), 1e-30)
            if (not pe[2] or not pe2[2] or not oe[2] or se != (0.0, 0.0) or
                re != (0.0, 0.0) or sam != (0.0, 0.0) or
                loss_error > tol_a + tol_r*abs(actual["loss"]) or
                first["example_ids"] != replay["example_ids"] or
                branch_state["training_noise_sha256"] != actual["training_noise_sha256"] or
                branch_state["progress"] != actual["progress"] or
                not torch.equal(branch_state["noise"], actual["noise"])):
                raise RuntimeError("serialized replay state exceeds registered tolerance")
            probe_relative = Path(probe_path).relative_to(jobroot).as_posix()
            probe_facts = {**recovery_facts, "resume_probe": {"path": probe_relative,
                "sha256": probe_digest, "checkpoint_path": Path(boundary_path).relative_to(jobroot).as_posix(),
                "checkpoint_sha256": boundary_digest, "step": resume_at,
                "result": {"first_replayed_ids": first["example_ids"],
                    "second_replayed_ids": replay["example_ids"],
                    "max_parameter_abs_error": max(pe[0], pe2[0]),
                    "max_parameter_relative_error": max(pe[1], pe2[1]),
                    "max_optimizer_abs_error": oe[0], "max_optimizer_relative_error": oe[1],
                    "optimizer_moment_count": len(branch_state["optimizer"]["state"]),
                    "max_scheduler_abs_error": se[0], "max_rng_abs_error": re[0],
                    "max_sampler_abs_error": sam[0], "max_loss_abs_error": loss_error,
                    "max_loss_relative_error": loss_relative}}}
            resume_result = {"first_replayed_ids": first["example_ids"], "second_replayed_ids": replay["example_ids"],
                "max_parameter_abs_error": max(pe[0], pe2[0]), "max_parameter_relative_error": max(pe[1], pe2[1]),
                "max_optimizer_abs_error": oe[0], "max_optimizer_relative_error": oe[1],
                "optimizer_moment_count": len(branch_state["optimizer"]["state"]),
                "max_scheduler_abs_error": se[0], "max_rng_abs_error": re[0],
                "max_sampler_abs_error": sam[0], "max_loss_abs_error": loss_error,
                "max_loss_relative_error": loss_relative, "checkpoint_sha256": boundary_digest,
                "checkpoint_path": str(boundary_path), "probe_state_sha256": probe_digest,
                "probe_state_path": str(probe_path), "saved_sampler_epoch": saved_sampler_boundary["epoch"],
                "saved_sampler_position": saved_sampler_boundary["position"],
                "replayed_sampler_epoch": actual["sampler"]["epoch"],
                "replayed_sampler_position": actual["sampler"]["position"],
                "verdict": True, "atol": config["resume"]["float_atol"],
                "rtol": config["resume"]["float_rtol"],
                "source_sha256": identities["source"], "config_sha256": identities["config"]}
            on_event("resume_verified", resume_result)
            save_boundary(progress["optimizer_step"], boundary_phase="training", facts=probe_facts)

        if phase == "resume_probe_pending":
            run_resume_probe()
        first_step = progress["optimizer_step"] + 1
        for step in range(first_step, total_steps+1):
            if step == replay_step and progress["optimizer_step"] >= step:
                continue
            update("main", increment_net=True)
            facts = dict(recovery_facts)
            boundary_phase = "resume_probe_pending" if step == resume_at else "training"
            path, digest = save_boundary(step, boundary_phase=boundary_phase, facts=facts)
            if step == resume_at:
                run_resume_probe()

        if progress["optimizer_step"] != total_steps:
            raise RuntimeError("training ended without the exact configured net-step count")
        if phase != "final_evaluation_pending":
            final_path = Path(recovery_checkpoint_paths.get(str(total_steps),
                store.verify_descriptor(recovery_descriptor, identities)["checkpoint_path_absolute"]))
            final_digest = checkpoint_hashes.get(str(total_steps),
                store.verify_descriptor(recovery_descriptor, identities)["checkpoint_sha256"])
            commit_boundary(final_path, final_digest, boundary_phase="final_evaluation_pending",
                facts=recovery_facts)
        else:
            final_path = Path(store.verify_descriptor(recovery_descriptor, identities)["checkpoint_path_absolute"])
            final_digest = store.verify_descriptor(recovery_descriptor, identities)["checkpoint_sha256"]
        final = evaluate(total_steps, final_digest)
        ratio = final["ce"]/initial["ce"] if initial["ce"] > 0 and math.isfinite(initial["ce"]) else math.inf
        status = "PASS_TINY_SOURCE_OVERFIT" if final["valid"] == 3 and ratio <= config["evaluation"]["gate_max_final_to_initial_CE_ratio"] else "NO_GO_TINY_SOURCE_OVERFIT"
        final_receipt = final["receipt"]
        result = {"status": status, "initial_ce": initial["ce"], "final_ce": final["ce"],
            "final_to_initial_ce_ratio": ratio, "valid_numerator": final["valid"], "valid_denominator": 3,
            "physical_optimizer_updates": physical, "segment_physical_optimizer_updates": physical,
            "net_optimizer_steps": progress["optimizer_step"], "examples_seen": progress["examples_seen"],
            "supervised_tokens_seen": net_tokens, "updates": rows,
            "predictions": initial["predictions"]+final["predictions"], "resume": resume_result,
            "evaluation_state_preserved": True, "checkpoint_hashes": checkpoint_hashes,
            "initial_state_sha256": start_hash, "initial_state_hash_recipe": initial_hash_recipe,
            "initial_evaluation_receipt": initial_receipt, "final_evaluation_receipt": final_receipt,
            "recovery_descriptor": recovery_descriptor, "recovery_phase": "complete",
            "artifact_paths": {"updates": str(log_path), "predictions_initial": str(jobroot / initial_receipt["records_path"]),
                "predictions_final": str(jobroot / final_receipt["records_path"]),
                "checkpoints": {key: path for key, path in checkpoint_paths.items() if Path(path).is_file()},
                "recovery_checkpoints": {key: path for key, path in recovery_checkpoint_paths.items()
                    if Path(path).is_file()},
                "retired_recovery_receipts": str(output_dir / "recovery" / "retirements.jsonl"),
                "resume_probe_state": resume_result["probe_state_path"]},
            "elapsed_seconds": time.perf_counter()-start,
            "costs": {"training_wall_seconds": time.perf_counter()-start,
                "supervised_training_tokens": net_tokens, "gpu_hours": None,
                "flops": None, "peak_allocated_mib": None, "peak_reserved_mib": None},
            "trainable_parameters": sum(p.numel() for p in host.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in original_params),
            "latent_slots": host.workspace.num_slots, "latent_dim": host.workspace.latent_dim,
            "workspace_states": host.workspace.max_steps+1}
        stored_result = dict(result)
        stored_result.pop("recovery_descriptor", None)
        complete_facts = {**recovery_facts, "final_result": stored_result}
        commit_boundary(final_path, final_digest, boundary_phase="complete", facts=complete_facts)
        result["recovery_descriptor"] = recovery_descriptor
        return result
    except ArithmeticError as error:
        partial = {"physical_optimizer_updates": physical, "net_optimizer_steps": progress["optimizer_step"],
                   "segment_physical_optimizer_updates": physical, "last_good_step": last_good,
                   "recovery_descriptor": recovery_descriptor,
                   "committed_checkpoint_sha256": (None if recovery_descriptor is None else
                       RecoveryStore.verify_descriptor(recovery_descriptor, identities)["checkpoint_sha256"]),
                   "artifact_paths": {"updates": str(log_path), "recovery_cursor": str(store.cursor_path),
                       "predictions": [str(path) for path in output_dir.glob("predictions-step-*.jsonl")],
                       "resume_probe_state": str(output_dir / "resume-probe-state.pt") if
                           (output_dir / "resume-probe-state.pt").is_file() else None}}
        write_row(log_path, {"role": "numerical_divergence", "physical_update": physical,
            "net_step": progress["optimizer_step"], "last_good_step": last_good,
            "message": str(error), "failure_kind": "NUMERICAL_DIVERGENCE"})
        on_event("numerical_divergence", partial)
        raise NumericalDivergence(str(error), partial=partial) from error
    finally:
        # Frozen parameters are checked by identity/value without hashing the base at each update.
        if any(p.requires_grad != req or p.grad is not None for p, req in zip(original_params, original_requires)):
            raise RuntimeError("original backbone freeze/gradient invariant failed")
        for module, mode in host_mode.items(): module.training = mode
