"""Private detached native supervisor fixture for the Task 3 integration proof."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from dwr.experiments import source_pilot_job as job
from dwr.experiments.source_pilot import _audit_completed_arm_core, matrix_decision
from tests import source_pilot_controller_fixture as fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--attach", action="store_true")
    args = parser.parse_args(argv)
    repo = job._root()
    config_path = Path(args.config).resolve(strict=True)
    registration_path = Path(args.registration).resolve(strict=True)
    registration = job._read_json_path(registration_path)
    source = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
        check=True, capture_output=True, text=True).stdout.strip()
    physical = {"commit": source, "fixture_source": True}
    runtime = job._runtime_receipt()
    examples = tuple(sorted(fixture.examples(5, "source_validation", start=0),
        key=lambda row: row.example_id))
    identities_by_arm = {arm: fixture.identities(registration, source_commit=source, arm_id=arm)
        for arm in ("K0", "K1", "K4")}
    host = fixture.tiny_host(seed=501)
    tokenizer = fixture.tiny_tokenizer()
    policy = fixture.scaled_policy(host, tokenizer, steps=2, validation_examples=5,
        decoder_max_new_tokens=128)
    model = registration["model_pins"]

    def record_identities(reg, source_commit, runtime_sha256, arm_id=None):
        arm_id = arm_id or "K0"
        resolved = identities_by_arm[arm_id]
        return {"source_commit": {"name": "source-commit", "revision": source_commit,
                "sha256": hashlib.sha256(source_commit.encode("ascii")).hexdigest()},
            "pilot_config": {"name": "source-validation-pilot-config",
                "revision": reg["config"]["config_version"],
                "sha256": reg["config_sha256"]},
            "cohort": {"name": "source-validation-pilot-cohort", "revision": "v1",
                "sha256": reg["cohort_sha256"]},
            "model": {"name": model["repo_id"], "revision": model["revision"],
                "sha256": resolved["model"]},
            "tokenizer": {"name": model["repo_id"], "revision": model["revision"],
                "sha256": resolved["tokenizer"]},
            "runtime": {"name": "source-validation-pilot-runtime", "revision": "registered-v1",
                "sha256": runtime_sha256}}

    def read_validation(_root, _registration, split):
        if split != "source_validation":
            raise ValueError("private native supervisor fixture received an unexpected split")
        matrix_root = repo / _registration["config"]["output_dir"]
        rows = job._event_rows(matrix_root / "events.jsonl")
        if not any(row["kind"] == "source_examples_access_authorized" and
                row["payload"].get("split") == split for row in rows):
            raise ValueError("matrix before-access event was not durable before native validation read")
        return examples, fixture.metadata(examples)

    def audit(root, reg, arm_root, *, arm_id, source_commit, validation_examples, metadata):
        return _audit_completed_arm_core(root, reg, arm_root, arm_id=arm_id,
            source_commit=source_commit, validation_examples=validation_examples,
            metadata=metadata, identities=identities_by_arm[arm_id], policy=policy,
            tokenizer=tokenizer)

    def worker_command(arm_id, descriptor_path, descriptor_sha):
        segment_index = int(descriptor_path.parent.name.removeprefix("segment-"))
        command = [sys.executable, "-m", "tests.source_pilot_job_child",
            "--descriptor", str(descriptor_path), "--sha256", descriptor_sha,
            "--registration", str(registration_path)]
        control = matrix_root.parent / (matrix_root.name + "-control")
        if (arm_id, segment_index) in {("K0", 0), ("K1", 0)}:
            command.extend(["--hold-after-step", "1",
                "--ready-marker", str(control / f"{arm_id}-worker-ready.json"),
                "--release-marker", str(control / f"{arm_id}-worker-release.json")])
        if arm_id == "K4" and segment_index in {0, 1}:
            command.extend(["--interrupt-stage", "final" if segment_index == 0 else "zero"])
        return command

    factories = {"registration": registration, "physical": physical, "runtime": runtime,
        "read_validation": read_validation, "audit": audit, "matrix_decision": matrix_decision,
        "worker_command": worker_command, "watchdog_factory": lambda _root: None,
        "record_identities": record_identities}
    matrix_root = repo / registration["config"]["output_dir"]
    if args.attach:
        if not matrix_root.is_dir():
            raise FileNotFoundError("private attachment requested for a missing original matrix")
    else:
        matrix_root.mkdir(parents=True, exist_ok=False)
        job._write_status(matrix_root, state="starting", supervisor=None, worker=None,
            config_sha256=registration["config_sha256"], completed_arms=[], current_arm=None)
    job._supervise_core(str(config_path), factories=factories)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
