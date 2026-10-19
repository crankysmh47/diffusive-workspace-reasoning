"""Private detached native worker entrypoint used only by Task 3 integration tests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from dwr.experiments import source_pilot_job as job
from dwr.experiments.source_pilot import _execute_authorized_worker


def _worker_authorization(descriptor_path: Path, descriptor_sha256: str, registration: dict,
                          identities: dict) -> dict:
    descriptor_path, descriptor = job._read_descriptor_file(descriptor_path, descriptor_sha256)
    matrix_root = Path(descriptor["matrix_root"]).resolve(strict=True)
    arm_root = Path(descriptor["arm_root"]).resolve(strict=True)
    segment_dir = Path(descriptor["segment_dir"]).resolve(strict=True)
    arm_id = descriptor["arm_id"]
    arm_index = descriptor["arm_index"]
    source_commit = descriptor["source_commit"]
    resume = job._verify_opaque_resume(descriptor["resume_descriptor"], arm_root, arm_id, source_commit)
    clocks = {"matrix": descriptor["matrix_clock"], "arm": descriptor["arm_clock"]}
    deadlines = job._derive_deadlines(registration["config"]["limits"], clocks)
    launch_rows = [row for row in job._event_rows(arm_root / "events.jsonl")
        if row["kind"] == "worker_launch_command_issued" and
        row["payload"].get("segment_index") == descriptor["segment_index"]]
    if len(launch_rows) != 1 or launch_rows[0]["payload"].get("descriptor_sha256") != descriptor_sha256:
        raise ValueError("private worker launch argv is not uniquely bound to the descriptor")
    launch_command = launch_rows[0]["payload"].get("command")
    return {"schema_version": "dwr-source-pilot-worker-authorization-v1",
        "root": str(job._root().resolve(strict=True)), "descriptor_path": str(descriptor_path),
        "descriptor_sha256": descriptor_sha256, "descriptor": descriptor,
        "registration": registration, "matrix_root": str(matrix_root), "arm_root": str(arm_root),
        "segment_dir": str(segment_dir), "arm_id": arm_id, "arm_index": arm_index,
        "arm": registration["config"]["arms"][arm_index], "segment_index": descriptor["segment_index"],
        "nonce": descriptor["nonce"], "identities": identities,
        "bindings": {"source": {"commit": source_commit,
                "physical_receipt": descriptor["physical_source_receipt"],
                "receipt_sha256": descriptor["physical_source_receipt_sha256"]},
            "runtime": {"receipt": descriptor["runtime_receipt"],
                "receipt_sha256": descriptor["runtime_receipt_sha256"]},
            "matrix": {"initial_manifest_sha256": descriptor["matrix_initial_manifest_sha256"],
                "config_sha256": registration["config_sha256"],
                "cohort_sha256": registration["cohort_sha256"]},
            "arm": {"initial_manifest_sha256": descriptor["arm_initial_manifest_sha256"],
                "arm_id": arm_id, "identities": identities}},
        "clocks": clocks, "deadlines": deadlines, "resume_descriptor": resume,
        "expected_launcher": descriptor["launcher"]}


def _factories(registration: dict, authorization: dict, interrupt_stage: str | None, *,
               hold_after_step: int | None = None, ready_marker: Path | None = None,
               release_marker: Path | None = None) -> dict:
    from tests import source_pilot_controller_fixture as fixture
    import torch

    torch.set_num_threads(1)
    # Every paired arm and recovery worker starts from the same frozen fixture state.
    host = fixture.tiny_host(seed=501)
    tokenizer = fixture.tiny_tokenizer()
    identities = fixture.identities(registration,
        source_commit=authorization["bindings"]["source"]["commit"], arm_id=authorization["arm_id"])
    train = tuple(sorted(fixture.examples(5, "source_train", start=5), key=lambda row: row.example_id))
    validation = tuple(sorted(fixture.examples(5, "source_validation", start=0), key=lambda row: row.example_id))
    training_metadata = fixture.metadata(train)
    validation_metadata = fixture.metadata(validation)

    def read_examples(_root: Path, _registration: dict, *, split: str):
        if split == "source_train":
            return train, training_metadata
        if split == "source_validation":
            return validation, validation_metadata
        raise ValueError("fixture worker received an unregistered split")

    def after_kernel_event(kind, payload, _record, auth):
        if auth["arm_id"] == "K4" and kind == "stage_committed" and \
                payload.get("stage") == interrupt_stage and \
                auth["segment_index"] == (0 if interrupt_stage == "final" else 1):
            raise KeyboardInterrupt(f"injected recovery after durable {interrupt_stage} stage commit")
        progress = payload.get("progress") if isinstance(payload, dict) else None
        if hold_after_step is not None and ready_marker is not None and release_marker is not None and \
                kind == "recovery_committed" and isinstance(progress, dict) and \
                progress.get("optimizer_step") == hold_after_step:
            ready_marker.parent.mkdir(parents=True, exist_ok=True)
            with ready_marker.open("xb") as stream:
                stream.write(job._canonical({"arm_id": auth["arm_id"],
                    "segment_index": auth["segment_index"], "worker": job._reference(os.getpid()),
                    "optimizer_step": hold_after_step}) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            deadline = time.monotonic() + 600
            while not release_marker.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("private worker hold exceeded its bounded release wait")
                time.sleep(0.05)

    return {"registration": registration,
        "policy": fixture.scaled_policy(host, tokenizer,
            steps=registration["config"]["training"]["optimizer_steps"],
            validation_examples=len(validation),
            decoder_max_new_tokens=registration["config"]["decoding"]["max_new_tokens_including_eos"]),
        "pilot_identities": fixture.identities,
        "host_config": {"hardware": {"device": "cpu"}},
        "read_examples": read_examples, "torch": torch,
        "host": lambda: host, "tokenizer": lambda: tokenizer,
        "tokenizer_identity": __import__("dwr.training.batches", fromlist=["TokenizerIdentity"]).TokenizerIdentity(
            "fixture", "v1"),
        "run_pilot_arm": __import__("dwr.training.source_pilot", fromlist=["run_pilot_arm"]).run_pilot_arm,
        "model_inventory": {"weight_sha256": identities["model"],
            "inventory_sha256": identities["tokenizer"]},
        "after_kernel_event": after_kernel_event}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--interrupt-stage")
    parser.add_argument("--hold-after-step", type=int)
    parser.add_argument("--ready-marker")
    parser.add_argument("--release-marker")
    args = parser.parse_args(argv)
    descriptor_path = Path(args.descriptor).resolve(strict=True)
    registration = json.loads(Path(args.registration).read_text(encoding="utf-8"))
    from tests import source_pilot_controller_fixture as fixture

    descriptor = job._read_descriptor_file(descriptor_path, args.sha256)[1]
    identities = fixture.identities(registration, source_commit=descriptor["source_commit"],
        arm_id=descriptor["arm_id"])
    authorization = _worker_authorization(descriptor_path, args.sha256, registration, identities)
    publication = job.publish_worker_receipt(authorization)
    try:
        _execute_authorized_worker(authorization, publication,
            command=[sys.executable, "-m", "tests.source_pilot_job_child"],
            factories=_factories(registration, authorization, args.interrupt_stage,
                hold_after_step=args.hold_after_step,
                ready_marker=Path(args.ready_marker) if args.ready_marker else None,
                release_marker=Path(args.release_marker) if args.release_marker else None))
    finally:
        publication["worker_lock"].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
