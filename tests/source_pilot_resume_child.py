"""Fresh native interpreter for Task 2 source-pilot process recovery checks."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import torch

import test_source_pilot_learning as fixture
from dwr.training.batches import TokenizerIdentity
from dwr.training.source_pilot import _canonical, run_pilot_arm


EXIT_AFTER_DURABLE_EVENT = 91


def _parameter_hashes(host) -> dict[str, str]:
    result = {}
    for name, parameter in sorted(host.backbone.named_parameters()):
        value = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        result[name] = hashlib.sha256(value).hexdigest()
    if not result:
        raise RuntimeError("fixture original host has no parameters to audit")
    return result


def _write(path: Path, value: object) -> dict[str, str]:
    raw = _canonical(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def _stop_matches(stop: str | None, kind: str, payload: dict) -> bool:
    if stop is None:
        return False
    if stop == "step1" and kind == "recovery_committed":
        return payload.get("phase") == "training" and payload.get("progress", {}).get("optimizer_step") == 1
    if stop == "bank_ready" and kind == "recovery_committed":
        return payload.get("phase") == "final_evaluation_pending" and payload.get("progress", {}).get("optimizer_step") == 2
    if stop == "final_committed" and kind == "stage_committed":
        return payload.get("stage") == "final"
    if stop == "shuffle_committed" and kind == "stage_committed":
        return payload.get("stage") == "shuffle"
    if stop == "zero_prediction" and kind == "prediction_completed":
        return payload.get("stage") == "zero"
    return False


def _training_examples(count: int):
    rows = list(fixture.fixture_examples(min(count, 5), "source_train", start=5))
    while len(rows) < count:
        index = len(rows)
        example_id = hashlib.sha256(f"task2-extra-source-train-{index}".encode("utf-8")).hexdigest()
        rows.append(replace(rows[index % len(rows)], example_id=example_id))
    return tuple(sorted(rows, key=lambda row: row.example_id))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 7:
        raise SystemExit("usage: source_pilot_resume_child.py ARM ROOT SEGMENT STOP RESUME STEPS PRESERVE_ORPHANS")
    arm_id, root_text, segment_name, stop, resume_flag, steps_text, preserve_orphans = args
    steps = int(steps_text)
    if steps <= 0:
        raise ValueError("fixture optimizer step count must be positive")
    root = Path(root_text).resolve(strict=True)
    segment = root / segment_name
    if not segment.is_dir() or segment.parent != root:
        raise RuntimeError("caller must pre-create an immediate segment-NNN directory")
    torch.set_num_threads(1)
    random.seed(20261211)
    np.random.seed(20261212)
    torch.manual_seed(20261213)
    host = fixture.fixture_host()
    train = _training_examples(max(5, steps))
    validation = fixture.fixture_examples(5)
    metadata = {"training": fixture.fixture_metadata(train),
                "validation": fixture.fixture_metadata(validation)}
    event_path = root / f"events-{segment_name}.jsonl"
    result_path = root / f"result-{segment_name}.json"
    initial_ref = None
    if resume_flag == "resume":
        cursor_path = root / "recovery.latest.json"
        resume = {"jobroot": str(root), "cursor_path": str(cursor_path),
                  "cursor_sha256": hashlib.sha256(cursor_path.read_bytes()).hexdigest()}
        initial_path = root / "initial-frozen.json"
        initial_ref = {"path": initial_path.name,
                       "sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest()}
    else:
        resume = None
        initial_ref = _write(root / "initial-frozen.json", {
            "schema": "dwr-source-pilot-initial-frozen-v1",
            "arm_id": arm_id,
            "identities": fixture._IDENTITIES,
            "original_parameter_hashes": _parameter_hashes(host),
        })

    def on_event(kind: str, payload: dict) -> None:
        raw = _canonical({"kind": kind, "payload": payload})
        with event_path.open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if preserve_orphans == "yes" and kind == "recovery_committed" and \
                payload.get("phase") == "training" and payload.get("progress", {}).get("optimizer_step") == 1:
            committed = root / payload["checkpoint_path"]
            orphan = segment / "recovery" / "recovery-checkpoint-step-000099.pt"
            with orphan.open("xb") as stream:
                stream.write(committed.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            unknown = segment / "recovery" / "unknown.payload"
            with unknown.open("xb") as stream:
                stream.write(b"unrecognized artifact survives later retire cycles\x00\xff")
                stream.flush()
                os.fsync(stream.fileno())
        if _stop_matches(None if stop == "-" else stop, kind, payload):
            os._exit(EXIT_AFTER_DURABLE_EVENT)

    def before_complete(payload: dict) -> dict:
        after_hashes = _parameter_hashes(host)
        initial_bytes = (root / initial_ref["path"]).read_bytes()
        initial = json.loads(initial_bytes.decode("utf-8"))
        if initial["original_parameter_hashes"] != after_hashes:
            raise RuntimeError("fixture original parameters changed during pilot")
        final = {
            "schema": "dwr-source-pilot-controller-audit-v1",
            "arm_id": arm_id,
            "identities": fixture._IDENTITIES,
            "initial_frozen": initial_ref,
            "initial_state_sha256": payload["initial_state_sha256"],
            "initial_checkpoint": payload["initial_checkpoint"],
            "final_checkpoint": payload["final_checkpoint"],
            "stage_receipts": payload["stage_receipts"],
            "bank": payload["bank"],
            "original_parameter_hashes": after_hashes,
            "details": {"fixture": "tiny-native-cpu", "actual_parameter_count": len(after_hashes)},
        }
        final_ref = _write(root / f"final-controller-{segment_name}.json", final)
        return {"initial_frozen": initial_ref, "final_controller": final_ref}

    result = run_pilot_arm(host, fixture.FixtureTokenizer(), TokenizerIdentity("fixture", "v1"),
        train, validation, registration=fixture.fixture_registration(arm_id, steps=steps),
        arm_id=arm_id, identities=fixture._IDENTITIES, metadata=metadata,
        output_dir=segment, check=lambda: None, on_event=on_event,
        before_complete=before_complete, resume=resume)
    with result_path.open("xb") as stream:
        stream.write(_canonical(result))
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
