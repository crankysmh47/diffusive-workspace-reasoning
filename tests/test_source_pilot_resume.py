"""Fresh-process CPU continuation proof for the native pilot arm kernel."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import torch
    from transformers import Qwen2Config
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "torch", "transformers"}:
        raise
    np = torch = Qwen2Config = None

if torch is not None and Qwen2Config is not None:
    from dwr.training.checkpoints import load_training_checkpoint
    from dwr.training.controls import configure_source_control
    from dwr.training.sampler import SourceSampler
    from dwr.training.source_pilot import _canonical
    from test_source_pilot_learning import (
        FixtureTokenizer, _IDENTITIES, fixture_examples, fixture_host,
        fixture_metadata, fixture_registration,
    )
else:
    load_training_checkpoint = configure_source_control = SourceSampler = None
    _canonical = FixtureTokenizer = _IDENTITIES = None
    fixture_examples = fixture_host = fixture_metadata = fixture_registration = None


_ML_READY = np is not None and torch is not None and Qwen2Config is not None
_ROOT = Path(__file__).resolve().parents[1]
_CHILD = Path(__file__).with_name("source_pilot_resume_child.py")
_OPS = _ROOT / ".superpowers" / "sdd" / "source-validation-pilot-20261017"


def _case_root(label: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = _OPS / f"task2-process-{label}-{stamp}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _run(root: Path, arm_id: str, segment: str, *, stop: str = "-", resume: bool = False,
         steps: int = 2, preserve_orphans: bool = False) -> subprocess.CompletedProcess:
    (root / segment).mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(_ROOT), str(_ROOT / "tests"), env.get("PYTHONPATH", "")))
    command = [sys.executable, str(_CHILD), arm_id, str(root), segment,
               stop, "resume" if resume else "fresh", str(steps),
               "yes" if preserve_orphans else "no"]
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()
    stdout = b""
    stderr = b""
    timed_out = False
    exception = None
    try:
        completed = subprocess.run(command, cwd=_ROOT, env=env, capture_output=True,
                                   timeout=180, check=False)
        stdout, stderr = completed.stdout or b"", completed.stderr or b""
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.output or b""
        stderr = error.stderr or b""
        returncode = None
        exception = f"TimeoutExpired after {error.timeout} seconds"
    except OSError as error:
        returncode = None
        exception = f"{type(error).__name__}: {error}"
    elapsed_seconds = time.perf_counter() - started_clock
    ended_at = datetime.now(timezone.utc).isoformat()
    prefix = f"process-{segment}"
    stdout_path = root / f"{prefix}.stdout.bin"
    stderr_path = root / f"{prefix}.stderr.bin"
    record_path = root / f"{prefix}.json"
    for path, raw in ((stdout_path, stdout), (stderr_path, stderr)):
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    record = {
        "command": command,
        "cwd": str(_ROOT),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "elapsed_seconds": elapsed_seconds,
        "exitcode": returncode,
        "timed_out": timed_out,
        "exception": exception,
        "stdout_path": stdout_path.name,
        "stderr_path": stderr_path.name,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }
    with record_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    diagnostic = stderr.decode("utf-8", errors="replace")
    if exception:
        diagnostic += ("\n" if diagnostic else "") + exception
    return subprocess.CompletedProcess(command, -1 if timed_out or exception else returncode,
        stdout.decode("utf-8", errors="replace"), diagnostic)


def _events(root: Path, segment: str) -> list[dict]:
    path = root / f"events-{segment}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _result(root: Path, segment: str) -> dict:
    return json.loads((root / f"result-{segment}.json").read_text(encoding="utf-8"))


def _facts(root: Path) -> tuple[dict, dict]:
    pointer = json.loads((root / "recovery.latest.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / pointer["metadata_path"]).read_text(encoding="utf-8"))
    return pointer, metadata["facts"]


def _final_payload(root: Path, result: dict) -> dict:
    pointer, _ = _facts(root)
    checkpoint = root / pointer["checkpoint_path"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != pointer["checkpoint_sha256"]:
        raise AssertionError("terminal cursor checkpoint hash is not bound to its bytes")
    if result["checkpoint_sha256"] != pointer["checkpoint_sha256"]:
        raise AssertionError("returned terminal checkpoint differs from the durable cursor")
    return payload


def _assert_logical_equal(test: unittest.TestCase, left, right, path="root") -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        test.assertIsInstance(left, torch.Tensor, path)
        test.assertIsInstance(right, torch.Tensor, path)
        test.assertEqual((left.dtype, tuple(left.shape)), (right.dtype, tuple(right.shape)), path)
        if left.is_floating_point() or left.is_complex():
            test.assertTrue(torch.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=False), path)
        else:
            test.assertTrue(torch.equal(left, right), path)
    elif isinstance(left, dict) or isinstance(right, dict):
        test.assertIsInstance(left, dict, path)
        test.assertIsInstance(right, dict, path)
        test.assertEqual(set(left), set(right), path)
        for key in left:
            _assert_logical_equal(test, left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        test.assertIsInstance(right, type(left), path)
        test.assertEqual(len(left), len(right), path)
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_logical_equal(test, a, b, f"{path}[{index}]")
    elif isinstance(left, float) or isinstance(right, float):
        test.assertIsInstance(left, (int, float), path)
        test.assertIsInstance(right, (int, float), path)
        test.assertAlmostEqual(left, right, delta=1e-12, msg=path)
    else:
        test.assertEqual(left, right, path)


def _fresh_draws(root: Path, payload: dict, arm_id: str) -> dict:
    """Restore the committed state into real objects, then consume every next stream."""
    random.seed(20261211)
    np.random.seed(20261212)
    torch.manual_seed(20261213)
    host = fixture_host()
    method_steps = {"K0": 0, "K1": 1, "K4": 4}[arm_id]
    control = configure_source_control(host, steps=method_steps)
    groups = []
    if method_steps:
        groups.append({"params": list(host.workspace.parameters()), "name": "workspace"})
    groups.append({"params": list(host.readers.parameters()), "name": "readers"})
    optimizer_config = fixture_registration(arm_id, steps=2)["config"]["training"]["optimizer"]
    optimizer = torch.optim.AdamW(groups,
        lr=optimizer_config["learning_rate"], betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["epsilon"], weight_decay=optimizer_config["weight_decay"],
        amsgrad=optimizer_config["amsgrad"], capturable=optimizer_config["capturable"],
        differentiable=optimizer_config["differentiable"], foreach=optimizer_config["foreach"],
        fused=optimizer_config["fused"], maximize=optimizer_config["maximize"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 8))
    train = tuple(sorted(fixture_examples(5, "source_train", start=5), key=lambda row: row.example_id))
    sampler = SourceSampler(tuple(row.example_id for row in train), seed=20261217)
    noise = torch.Generator(device="cpu").manual_seed(20261218)
    pointer = json.loads((root / "recovery.latest.json").read_text(encoding="utf-8"))
    checkpoint = root / pointer["checkpoint_path"]
    load_training_checkpoint(checkpoint, expected_sha256=pointer["checkpoint_sha256"], host=host,
        optimizer=optimizer, scheduler=scheduler, scaler=None, sampler=sampler,
        noise_generator=noise, identities=_IDENTITIES)
    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch_cpu": torch.rand(4, dtype=torch.float64),
        "private_noise": torch.rand(4, dtype=torch.float64, generator=noise),
        "next_sampler_ids": sampler.next_ids(3),
    }


def _event_rows(rows: list[dict], kind: str) -> list[dict]:
    return [row["payload"] for row in rows if row["kind"] == kind]


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file() and
            path.name not in {"recovery.latest.json", "recovery-commits.jsonl"}}


@unittest.skipUnless(_ML_READY, "optional native ML dependencies unavailable")
class SourcePilotFreshProcessResumeTests(unittest.TestCase):
    def setUp(self):
        self.caller_python = random.getstate()
        self.caller_numpy = np.random.get_state()
        self.caller_torch = torch.get_rng_state().clone()
        self.caller_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        random.setstate(self.caller_python)
        np.random.set_state(self.caller_numpy)
        torch.set_rng_state(self.caller_torch)
        torch.set_num_threads(self.caller_threads)

    def _assert_exit_after_committed_boundary(self, completed, root, segment, phase, step):
        self.assertEqual(completed.returncode, 91, completed.stdout + "\n" + completed.stderr)
        events = _events(root, segment)
        matching = [row["payload"] for row in events if row["kind"] == "recovery_committed"
                    and row["payload"].get("phase") == phase
                    and row["payload"].get("progress", {}).get("optimizer_step") == step]
        self.assertTrue(matching, "process exit was not after the requested durable recovery commit")
        return events

    def test_k0_k1_k4_fresh_process_continuation_matches_complete_logical_checkpoint(self):
        for arm_id in ("K0", "K1", "K4"):
            with self.subTest(arm=arm_id):
                baseline_root = _case_root(f"{arm_id.lower()}-baseline")
                baseline = _run(baseline_root, arm_id, "segment-000")
                self.assertEqual(baseline.returncode, 0, baseline.stdout + "\n" + baseline.stderr)
                baseline_result = _result(baseline_root, "segment-000")
                baseline_payload = _final_payload(baseline_root, baseline_result)

                resumed_root = _case_root(f"{arm_id.lower()}-step1-resume")
                interrupted = _run(resumed_root, arm_id, "segment-000", stop="step1")
                first_events = self._assert_exit_after_committed_boundary(
                    interrupted, resumed_root, "segment-000", "training", 1)
                step1_events = [row["payload"] for row in first_events if row["kind"] == "update_completed"]
                self.assertEqual(len(step1_events), 1)
                committed_checkpoint = resumed_root / "segment-000" / "recovery" / \
                    "recovery-checkpoint-step-000001.pt"
                orphan = committed_checkpoint.with_name("recovery-checkpoint-step-000099.pt")
                orphan.write_bytes(committed_checkpoint.read_bytes())
                unknown = resumed_root / "segment-000" / "unknown.payload"
                unknown.write_bytes(b"preserve this unrecognized artifact\x00\xff")
                old_files = _tree_bytes(resumed_root)
                continued = _run(resumed_root, arm_id, "segment-001", resume=True)
                self.assertEqual(continued.returncode, 0, continued.stdout + "\n" + continued.stderr)
                self.assertTrue(all((resumed_root / relative).read_bytes() == raw
                                    for relative, raw in old_files.items()),
                                "fresh continuation rewrote an interrupted-segment artifact")
                resumed_result = _result(resumed_root, "segment-001")
                resumed_payload = _final_payload(resumed_root, resumed_result)
                _assert_logical_equal(self, baseline_payload, resumed_payload)
                self.assertTrue(baseline_payload["optimizer"]["state"],
                                "fixture must establish real nonempty Adam moments")
                self.assertTrue(resumed_payload["optimizer"]["state"],
                                "fresh-process restore lost Adam moments")
                self.assertEqual(set(baseline_payload["progress"]),
                                 {"optimizer_step", "examples_seen", "supervised_tokens_seen"})

                self.assertEqual(baseline_result["optimizer_step"], 2)
                self.assertEqual(resumed_result["optimizer_step"], 2)
                for summary in (baseline_result["initial"], baseline_result["final"],
                                resumed_result["initial"], resumed_result["final"]):
                    self.assertGreaterEqual(summary["valid_denominator"], 5)
                self.assertEqual(len(baseline_payload["optimizer"]["param_groups"]),
                                 {"K0": 1, "K1": 2, "K4": 2}[arm_id])
                baseline_events = _events(baseline_root, "segment-000")
                resumed_events = first_events + _events(resumed_root, "segment-001")
                for kind in ("update_started", "update_completed"):
                    def paired(rows):
                        return [{key: value for key, value in payload.items()
                                 if key in {"net_step", "example_id", "batch_sha256",
                                            "noise_sha256", "loss"}}
                                for payload in _event_rows(rows, kind)]
                    self.assertEqual(len(paired(baseline_events)), 2)
                    self.assertEqual(len(paired(resumed_events)), 2)
                    for expected, actual in zip(paired(baseline_events), paired(resumed_events)):
                        _assert_logical_equal(self, expected, actual, f"{arm_id}.{kind}")
                initial_cases = [row for row in _event_rows(_events(resumed_root, "segment-001"),
                    "evaluation_case_started") if row.get("stage") == "initial"]
                self.assertEqual(initial_cases, [], "committed initial evaluation was replayed")
                _assert_logical_equal(self, _fresh_draws(baseline_root, baseline_payload, arm_id),
                                      _fresh_draws(resumed_root, resumed_payload, arm_id),
                                      f"{arm_id}.next-draws")

    def test_k4_bank_and_committed_last_stages_are_reused_without_more_predictions(self):
        for stop, phase, expected_prior_stages in (
            ("bank_ready", "final_evaluation_pending", ("initial",)),
            ("final_committed", "diagnostic_zero_pending", ("initial", "final")),
            ("shuffle_committed", "diagnostic_shuffle_pending", ("initial", "final", "zero", "noise", "shuffle")),
        ):
            with self.subTest(stop=stop):
                root = _case_root(f"k4-{stop}")
                interrupted = _run(root, "K4", "segment-000", stop=stop)
                events = self._assert_exit_after_committed_boundary(
                    interrupted, root, "segment-000", phase, 2)
                _, before_facts = _facts(root)
                before_refs = dict(before_facts["stage_receipts"])
                before_bank = before_facts["bank"]
                before_bank_bytes = (root / before_bank["path"]).read_bytes() if before_bank else None
                before_files = _tree_bytes(root)
                resumed = _run(root, "K4", "segment-001", resume=True)
                self.assertEqual(resumed.returncode, 0, resumed.stdout + "\n" + resumed.stderr)
                result = _result(root, "segment-001")
                _, after_facts = _facts(root)
                self.assertEqual({stage: after_facts["stage_receipts"][stage]
                                  for stage in expected_prior_stages},
                                 {stage: before_refs[stage] for stage in expected_prior_stages})
                self.assertEqual(after_facts["bank"], before_bank)
                if before_bank_bytes is not None:
                    self.assertEqual((root / before_bank["path"]).read_bytes(), before_bank_bytes)
                self.assertTrue(all((root / relative).read_bytes() == raw
                                    for relative, raw in before_files.items()),
                                "continuation modified an earlier committed artifact or orphan")
                new_events = _events(root, "segment-001")
                self.assertEqual(_event_rows(new_events, "bank_started"), [],
                                 "a durable K4 bank was recomputed after resume")
                self.assertEqual(_event_rows(new_events, "update_started"), [],
                                 "training resumed after all updates had committed")
                already_committed = set(expected_prior_stages)
                repeated = [row for row in _event_rows(new_events, "evaluation_case_started")
                            if row.get("stage") in already_committed]
                self.assertEqual(repeated, [], "a previously committed stage predicted again")
                self.assertEqual(result["optimizer_step"], 2)

    def test_native_k0_kernel_commits_multiple_retire_cycles_and_keeps_orphans(self):
        root = _case_root("k0-retention-six")
        completed = _run(root, "K0", "segment-000", steps=6, preserve_orphans=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)
        result = _result(root, "segment-000")
        self.assertEqual(result["optimizer_step"], 6)
        self.assertEqual(len(result["updates"]), 6)
        self.assertEqual(len({row["example_id"] for row in result["updates"]}), 6,
                         "six net updates must use six distinct native synthetic source IDs")
        self.assertGreaterEqual(result["initial"]["valid_denominator"], 5)
        self.assertGreaterEqual(result["final"]["valid_denominator"], 5)

        recovery = root / "segment-000" / "recovery"
        retained = {int(path.stem.rsplit("-", 1)[1]) for path in recovery.glob("recovery-checkpoint-step-*.pt")}
        self.assertEqual(retained, {0, 4, 5, 6, 99},
                         "retention must keep step 0, final 6 and the two newest other commits")
        history = recovery / "pilot-retirements.jsonl"
        self.assertTrue(history.is_file(), "kernel commits did not publish retirement receipts")
        retirement_rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({row["step"] for row in retirement_rows}, {1, 2, 3},
                         "multiple real kernel commits must complete several retirement cycles")

        events = _events(root, "segment-000")
        committed_step_one = [row["payload"] for row in events if row["kind"] == "recovery_committed"
                              and row["payload"].get("phase") == "training"
                              and row["payload"].get("progress", {}).get("optimizer_step") == 1]
        self.assertEqual(len(committed_step_one), 1)
        orphan = recovery / "recovery-checkpoint-step-000099.pt"
        self.assertEqual(hashlib.sha256(orphan.read_bytes()).hexdigest(),
                         committed_step_one[0]["checkpoint_sha256"],
                         "uncommitted checkpoint orphan was rewritten or substituted")
        unknown = recovery / "unknown.payload"
        self.assertEqual(unknown.read_bytes(), b"unrecognized artifact survives later retire cycles\x00\xff")

    def test_first_zero_diagnostic_prediction_partial_attempt_is_preserved_and_retried_as_whole_stage(self):
        root = _case_root("k4-zero-partial")
        interrupted = _run(root, "K4", "segment-000", stop="zero_prediction")
        self.assertEqual(interrupted.returncode, 91, interrupted.stdout + "\n" + interrupted.stderr)
        prior_events = _events(root, "segment-000")
        first_zero = [row for row in prior_events if row["kind"] == "prediction_completed"
                      and row["payload"].get("stage") == "zero"]
        self.assertEqual(len(first_zero), 1)
        first_start = [row["payload"] for row in prior_events if row["kind"] == "evaluation_case_started"
                       and row["payload"].get("stage") == "zero"]
        self.assertEqual(len(first_start), 1)
        self.assertEqual(first_start[0]["example_id"], first_zero[0]["payload"]["example_id"])
        partial_path = root / "segment-000" / "stages" / f"zero-{first_start[0]['attempt']}.jsonl"
        self.assertTrue(partial_path.is_file(), "first diagnostic envelope was not durably persisted")
        partial_raw = partial_path.read_bytes()
        self.assertTrue(partial_raw, "partial diagnostic JSONL is empty")
        partial_lines = partial_raw.splitlines()
        self.assertEqual(len(partial_lines), 1, "interruption should retain exactly the first completed row")
        partial = json.loads(partial_lines[0].decode("utf-8"))
        self.assertEqual(partial["schema_version"], "dwr-source-pilot-prediction-v1")
        self.assertEqual(partial["stage"], "zero")
        self.assertEqual(partial["example_id"], first_start[0]["example_id"])
        self.assertIsInstance(partial["generated_token_ids"], list)
        self.assertGreater(len(partial["generated_token_ids"]), 0)
        self.assertEqual(len(partial["generated_token_ids"]), partial["score"]["generated_tokens"])
        self.assertEqual(partial["counts"]["greedy_answer_forwards"],
                         len(partial["generated_token_ids"]))
        for digest_name in ("prompt_sha256", "noise_sha256", "workspace_sha256"):
            self.assertRegex(partial[digest_name], r"^[0-9a-f]{64}$")
        self.assertIn("reader_measurements", partial)
        self.assertIsNone(partial["reader_measurements"], "zero diagnostic has no reader-hook measurement")
        self.assertGreater(partial["costs"]["elapsed_seconds"], 0.0)
        self.assertIn("memory", partial["costs"])
        self.assertIsInstance(partial["costs"]["memory"]["value_bytes"], int)
        before_files = _tree_bytes(root)
        _, before_facts = _facts(root)
        committed_before = dict(before_facts["stage_receipts"])
        resumed = _run(root, "K4", "segment-001", resume=True)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + "\n" + resumed.stderr)
        self.assertTrue(all((root / relative).read_bytes() == raw
                            for relative, raw in before_files.items()),
                        "fresh worker altered the first zero attempt or another preserved file")
        _, after_facts = _facts(root)
        self.assertEqual({stage: after_facts["stage_receipts"][stage] for stage in committed_before},
                         committed_before)
        new_events = _events(root, "segment-001")
        zero_predictions = _event_rows(new_events, "prediction_completed")
        zero_predictions = [row for row in zero_predictions if row.get("stage") == "zero"]
        self.assertEqual(len(zero_predictions), 5,
                         "uncommitted fixed cohort must be rerun in full exactly once")
        starts = [row for row in _event_rows(new_events, "evaluation_case_started")
                  if row.get("stage") == "zero"]
        self.assertEqual(len(starts), 5)
        self.assertEqual(_event_rows(new_events, "bank_started"), [])
        self.assertEqual(_event_rows(new_events, "update_started"), [])
        result = _result(root, "segment-001")
        self.assertEqual(result["optimizer_step"], 2)
        self.assertTrue(all(result["diagnostics"][stage]["valid_denominator"] >= 5
                            for stage in ("zero", "noise", "shuffle")))


if __name__ == "__main__":
    unittest.main()
