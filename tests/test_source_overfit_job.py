"""Standard-library lifecycle checks for the recoverable source-overfit supervisor."""
from __future__ import annotations

import os
import hashlib
import json
import inspect
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from dwr.experiments import source_overfit_job as job

def _optional_ml_ready():
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ModuleNotFoundError as error:
        if error.name not in {"numpy", "torch", "transformers"}:
            raise
        return False

_ML_READY = _optional_ml_ready()


def _fixture_bundle(config_path: Path):
    from test_source_overfit_learning import _fixtures, RecordingTokenizer
    from dwr.tasks.schema import serialize_task
    from dwr.tasks import serialize_model_input
    from dwr.experiments import source_overfit
    from dwr.experiments.gpu_feasibility import hash_original_parameters
    host, examples, config, metadata = _fixtures(390)
    config["seeds"].update({"python_global": 401, "numpy_global": 402,
        "torch_cpu_global": 403, "torch_cuda_global": 404})
    config["training"]["schedule"]["type"] = "LambdaLR"
    config.update({"config_version": "fixture-registration", "model": {
        "repo_id": "fixture/model", "revision": "fixture-revision", "weight_sha256": "a"*64,
        "artifact_inventory_sha256": "b"*64}, "source_base_commit": "c"*40})
    config["data"] = {"source_export_manifest_sha256": "1"*64, "source_export_data_sha256": "2"*64,
        "source_token_audit_manifest_sha256": "3"*64, "token_lengths_sha256": "4"*64,
        "source_split": "source_train", "reader_expected_records": 3, "selected_examples": 3}
    config["dependencies"] = {"cohort_sha256": "5"*64, "feasibility_manifest_sha256": "6"*64,
        "feasibility_outcome_sha256": "7"*64, "reference_manifest_sha256": "8"*64,
        "reference_summary_sha256": "9"*64}
    config["limits"] = {"wall_seconds": 120, "resident_memory_mib": 4096, "attempts": 3,
        "work_seconds": 110, "maximum_physical_updates": 197, "maximum_predictions": 18,
        "maximum_decoder_calls": 2304, "maximum_worker_processes": 3,
        "maximum_operational_recoveries": 2, "maximum_peak_allocated_mib": 14336,
        "maximum_peak_reserved_mib": 14336}
    config["recovery"] = {"heartbeat_seconds": 30}
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    config_path.write_bytes(raw)
    selected = []; rows = []; token_rows = []
    for example, kind in zip(examples, ("eq", "neq", "add_mod")):
        entry = {"example_id": example.example_id, "split": "source_train", "profile": kind,
            "fine_stratum": metadata[example.example_id]["fine_stratum"]}
        selected.append(entry)
        prompt_count = len(RecordingTokenizer().encode(serialize_model_input(example.task)+"\n",
                                                        add_special_tokens=False))
        rows.append({**entry, "task": serialize_task(example.task),
            "solution": {"assignment": dict(example.solution.assignments)},
            "lineage_id": metadata[example.example_id]["lineage_id"],
            "structure_signature": metadata[example.example_id]["structure_signature"]})
        token_rows.append({**entry, "prompt_tokens": prompt_count, "answer_tokens_including_eos": 2,
            "total_tokens": prompt_count+2, "within_caps": True})
    source = {"commit": "a"*40, "clean": True, "files": [{"path": "dwr/fixture.py", "sha256": "d"*64}]}
    from dwr.experiments.source_overfit import _environment_receipt
    model = config["model"]
    registration = {"config": config, "config_raw": raw, "config_path": config_path,
        "config_sha256": hashlib.sha256(raw).hexdigest(), "selected": tuple(selected),
        "feasibility_config": {"model": model}}
    seams = {"source": lambda: source, "feasibility": lambda: {"outcome": {"status": "PASS_FEASIBILITY"}},
        "reference": lambda: ({"status": "completed"}, b'{"status":"PASS_FROZEN_REFERENCE_OPERATIONAL_V1"}',
            {"status": "PASS_FROZEN_REFERENCE_OPERATIONAL_V1", "attempt_count": 45,
             "quality": {"denominator": 15}, "repetitions": {
                "0": {"examples": 15, "valid_count": 0}, "1": {"examples": 15, "valid_count": 0},
                "2": {"examples": 15, "valid_count": 0}}}),
        "data_preconditions": lambda: {"task_semantics_read": False},
        "artifact_inventory": lambda: {"weight_sha256": model["weight_sha256"],
            "inventory_sha256": model["artifact_inventory_sha256"]}, "rows": rows, "token_rows": token_rows,
        "tokenizer": RecordingTokenizer(), "host": host, "original_hashes": hash_original_parameters(host),
        "original_hashes_after": hash_original_parameters(host), "training_config": config,
        "environment": _environment_receipt(), "runtime_costs": {"process_peak_working_set_mib": 800.0,
            "peak_allocated_mib": None, "peak_reserved_mib": None}, "check": lambda: None}
    registration["config_sha256"] = hashlib.sha256(raw).hexdigest()
    return registration, source, seams


def _install_fixture_cpu_rngs(seeds):
    # Tiny-native lifecycle fixtures stay strictly CPU-only; production's CUDA
    # stream handoff is independently checked by the mocked API contract test.
    import random
    import numpy as np
    import torch
    random.seed(seeds["python_global"])
    np.random.seed(seeds["numpy_global"])
    # torch.manual_seed also queues CUDA lazy-seeding callbacks; seed only the
    # CPU default generator so this fixture cannot alter CUDA init policy.
    torch.random.default_generator.manual_seed(seeds["torch_cpu_global"])


def _fixture_job_factory(config_path: Path, jobroot: Path, lock_path: Path, ready: Path, release: Path,
                         *, hold_mode: str):
    from dwr.experiments.records import RunRecord
    from dwr.experiments.source_overfit import _kernel_identities, _record_identities
    from dwr.experiments.source_overfit_job import _process_creation_time, _utc, _atomic_json
    registration, source, seams = _fixture_bundle(config_path)
    environment = seams["environment"]
    config = registration["config"]; model = registration["feasibility_config"]["model"]
    identities = _kernel_identities(registration["config_raw"], config, source_commit=source["commit"],
        weight_sha256=model["weight_sha256"], tokenizer_inventory_sha256=model["artifact_inventory_sha256"])

    def create_record(config_arg, output, supervisor_pid):
        record = RunRecord.create(output, command=["fixture-supervisor"], config_path=config_path,
            expected_config_sha256=registration["config_sha256"],
            identities=_record_identities(identities, config_version=config["config_version"],
                model_repo=model["repo_id"], model_revision=model["revision"],
                schedule_type=config["training"]["schedule"]["type"]),
            rng_streams=config["seeds"], settings={"jobroot": str(output), "supervisor_pid": supervisor_pid,
                "supervisor_creation_time": _process_creation_time(supervisor_pid), "source_commit": source["commit"],
                "physical_source_files": source["files"], "environment_receipt": environment,
                "environment_receipt_sha256": hashlib.sha256(job._canonical(environment)).hexdigest()},
            budgets={"wall_seconds": 5400, "memory_mib": 12288, "attempts": 3},
            confirmatory=False, source_root=job._root())
        initial_raw = (output/"manifest.initial.json").read_bytes()
        binding = {"schema_version": 1, "initial_manifest_sha256": hashlib.sha256(initial_raw).hexdigest(),
            "started_utc": record._started_utc, "source_commit": source["commit"],
            "physical_source_files": source["files"], "environment_receipt": environment,
            "environment_receipt_sha256": hashlib.sha256(job._canonical(environment)).hexdigest(),
            "config_sha256": job.CONFIG_SHA256,
            "budgets": {"work_seconds": 5220, "seal_seconds": 5400, "watchdog_seconds": 5460}}
        (output/"job-binding.json").write_bytes(job._canonical(binding)+b"\n")
        record.append_event("job_binding_created", {"sha256": hashlib.sha256(job._canonical(binding)+b"\n").hexdigest(),
            "initial_manifest_sha256": binding["initial_manifest_sha256"], "started_utc": record._started_utc})
        _atomic_json(output/"elapsed-budget.json", {"schema_version": 1, "started_utc": record._started_utc,
            "last_observed_utc_epoch": job._parse_utc(record._started_utc).timestamp(), "elapsed_seconds": 0.0})
        return record

    def open_record(output, config_arg):
        from dwr.experiments.records import _parse_json_file
        initial, raw = _parse_json_file(output/"manifest.initial.json", "fixture initial")
        binding = job._read_json(output/"job-binding.json")
        if hashlib.sha256(raw).hexdigest() != binding["initial_manifest_sha256"]:
            raise ValueError("fixture job initial hash mismatch")
        event_raw = (output/"events.jsonl").read_bytes()
        from dwr.experiments.records import _validate_events
        _validate_events(event_raw)
        job._persist_elapsed(output)
        return RunRecord(output, initial["started_utc"], time.monotonic()-(time.time()-job._parse_utc(initial["started_utc"]).timestamp()),
                         hashlib.sha256(raw).hexdigest())

    worker_code = ("import sys; sys.path.insert(0, " + repr(str(Path(__file__).resolve().parent)) + "); "
        "from test_source_overfit_job import _fixture_worker_main; _fixture_worker_main()")
    def worker_command(descriptor_path, descriptor_sha):
        return [sys.executable, "-c", worker_code, str(descriptor_path), descriptor_sha, str(jobroot),
            str(lock_path), str(config_path), str(ready), str(release), hold_mode]
    def after_identity(segment_index, receipt):
        if hold_mode == "identity_before_status":
            Path(str(ready)+".identity-receipt").write_text(str(receipt["pid"]), encoding="utf-8")
            deadline = time.monotonic()+30
            while not release.exists() and time.monotonic() < deadline: time.sleep(0.02)
    return {"config_path": config_path, "jobroot": jobroot, "lock_path": lock_path,
        "registration": registration, "source_receipt": source, "environment_receipt": environment,
        "create_record": create_record, "open_record": open_record, "worker_command": worker_command,
        "after_identity": after_identity}


def _fixture_worker_main():
    import sys
    from pathlib import Path
    from dwr.experiments import source_overfit
    from dwr.experiments import source_overfit_job as jobmod
    descriptor_path, descriptor_sha, jobroot_s, lock_path_s, config_path_s, ready_s, release_s, hold = sys.argv[1:]
    jobroot = Path(jobroot_s); lock_path = Path(lock_path_s)
    descriptor = jobmod.read_worker_descriptor(descriptor_path, descriptor_sha)
    lock = jobmod._ByteLock(lock_path, 1, wait=False)
    pid = os.getpid(); creation = jobmod._process_creation_time(pid)
    parent = jobmod._process_parent_pid(pid)
    identity = {"schema_version": 1, "pid": pid, "creation_time": creation, "parent_pid": parent,
        "parent_creation_time": jobmod._process_creation_time(parent), "image": jobmod._process_image(pid),
        "python_executable": sys.executable, "base_executable": getattr(sys, "_base_executable", sys.executable),
        "descriptor_nonce": descriptor["nonce"], "config_sha256": jobmod.CONFIG_SHA256,
        "source_commit": descriptor["source_commit"], "segment_index": descriptor["segment_index"]}
    identity_path = jobroot/f"worker-{descriptor['segment_index']:03d}.identity.json"
    identity_path.write_bytes(jobmod._canonical(identity)+b"\n")
    Path(ready_s).write_text(str(pid), encoding="utf-8")
    if hold in {"before", "identity_before_status"}:
        deadline = time.monotonic()+30
        while not Path(release_s).exists() and time.monotonic() < deadline:
            time.sleep(0.02)
    registration, _, seams = _fixture_bundle(Path(config_path_s))
    host = seams["host"]
    authorization = {"jobroot": str(jobroot), "segment_index": descriptor["segment_index"],
        "worker_pid": pid, "worker_creation_time": creation,
        "supervisor_pid": descriptor["supervisor_pid"],
        "supervisor_creation_time": descriptor["supervisor_creation_time"],
        "resume_descriptor": descriptor.get("resume_descriptor"), "physical_source_receipt": descriptor["physical_source_receipt"]}
    def observe(kind, payload):
        if hold == "fault" and kind == "update_started":
            raise RuntimeError("fixture injected invariant failure")
        if hold == "after_step" and kind == "recovery_committed" and \
                payload.get("progress", {}).get("optimizer_step", 0) >= 1:
            Path(ready_s + ".step").write_text("committed", encoding="utf-8")
            while not Path(release_s).exists():
                time.sleep(0.02)
        if hold == "after_probe" and kind == "recovery_committed" and \
                payload.get("progress", {}).get("optimizer_step") == registration["config"]["resume"]["checkpoint_step"]+1:
            Path(ready_s + ".probe").write_text("committed", encoding="utf-8")
            while not Path(release_s).exists():
                time.sleep(0.02)
        if hold == "after_complete" and kind == "recovery_committed" and payload.get("phase") == "complete":
            Path(ready_s + ".complete").write_text("committed", encoding="utf-8")
            while not Path(release_s).exists():
                time.sleep(0.02)
    seams["observe_kernel_event"] = observe
    try:
        _install_fixture_cpu_rngs(registration["config"]["seeds"])
        source_overfit._execute_registered(jobmod._root(), registration, authorization,
            command=["fixture-worker"], _test_seams=seams)
    finally:
        lock.close()


def _launch_fixture_supervisor(config: Path, jobroot: Path, lock_path: Path, ready: Path, release: Path,
                               hold_mode: str, directory: Path):
    root = Path(__file__).resolve().parents[1]
    script = ("import sys; sys.path.insert(0, " + repr(str(Path(__file__).resolve().parent)) + "); "
        "from pathlib import Path; from test_source_overfit_job import _fixture_job_factory; "
        "from dwr.experiments import source_overfit_job as job; "
        "config=Path(sys.argv[1]); jobroot=Path(sys.argv[2]); lock=Path(sys.argv[3]); "
        "ready=Path(sys.argv[4]); release=Path(sys.argv[5]); mode=sys.argv[6]; "
        "factory=_fixture_job_factory(config,jobroot,lock,ready,release,hold_mode=mode); "
        "raise SystemExit(job._supervise(config,_test_factory=factory))")
    stdout_path = directory/"supervisor.stdout.log"; stderr_path = directory/"supervisor.stderr.log"
    flags = (getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) |
             getattr(subprocess, "CREATE_NO_WINDOW", 0))
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen([sys.executable, "-c", script, str(config), str(jobroot), str(lock_path),
            str(ready), str(release), hold_mode], cwd=root, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, creationflags=flags)
    return process


def _wait_until(predicate, timeout=30):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def _assert_final_accounting(testcase, jobroot, terminal):
    ledger = job._read_json(jobroot/"physical-work-ledger.json")
    observed = ledger["observed"]
    testcase.assertGreaterEqual(observed["started_physical_updates"], observed["observed_physical_updates"])
    testcase.assertEqual(observed["conservative_physical_updates"],
        observed["observed_physical_updates"]+observed["unknown_inflight_worker_updates"])
    testcase.assertEqual(terminal["counts"]["observed_physical_updates"], observed["observed_physical_updates"])
    testcase.assertEqual(terminal["counts"]["started_physical_updates"], observed["started_physical_updates"])
    testcase.assertEqual(terminal["counts"]["inflight_physical_updates"], observed["unknown_inflight_worker_updates"])
    testcase.assertEqual(terminal["counts"]["prediction_attempts"], observed["prediction_attempts"])
    testcase.assertEqual(terminal["counts"]["official_evaluation_denominator"], 3)


class SourceOverfitJobTests(unittest.TestCase):
    @unittest.skipUnless(_ML_READY, "production-shaped finalization seam uses optional registered fixture data")
    def test_production_final_accounting_uses_resolved_config_for_job_seal(self):
        from dwr.experiments import source_overfit
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s); config_path = root/"registered.json"
            registration, _, _ = _fixture_bundle(config_path)
            jobroot = root.joinpath(*job.PurePosixPath(job.JOB_DIR).parts)
            started = "2026-10-17T00:00:00Z"
            class Record:
                def append_event(self, *args, **kwargs): pass
                def finish(self, status, **kwargs): self.finished = status
            record = Record()
            def create_record(config, output, supervisor_pid):
                output.mkdir(parents=True)
                binding = {"started_utc": started, "source_commit": "a"*40}
                (output/"job-binding.json").write_bytes(job._canonical(binding)+b"\n")
                (output/"elapsed-budget.json").write_bytes(job._canonical({"started_utc": started,
                    "last_observed_utc_epoch": job._parse_utc(started).timestamp(), "elapsed_seconds": 0.0})+b"\n")
                return record
            with patch.object(job, "_root", return_value=root), \
                    patch.object(job, "_config", return_value=config_path), \
                    patch.object(job, "_lock_path", return_value=root/"supervisor.lock"), \
                    patch.object(job, "_create_job_record", side_effect=create_record), \
                    patch.object(job, "_run_workers", return_value={"outcome": "PASS_TINY_SOURCE_OVERFIT",
                        "segments": 1, "recoveries": 0, "worker_processes": 1, "elapsed_seconds": 1.0}), \
                    patch.object(job, "_launch_outer_watchdog"), \
                    patch.object(job, "_persist_elapsed", return_value=1.0), \
                    patch.object(job, "_job_artifacts", return_value=[]), \
                    patch.object(source_overfit, "_load_registration", return_value=registration):
                self.assertEqual(job._supervise(config_path), 0)
            self.assertEqual(record.finished, "completed")

    def test_heartbeat_writes_are_serialized_across_real_threads(self):
        from dwr.experiments import source_overfit
        with tempfile.TemporaryDirectory() as directory_s:
            target = Path(directory_s)/"heartbeat.json"
            authorization = {"worker_heartbeat_path": str(target), "worker_creation_time": "worker-id",
                "worker_receipt": {"descriptor_nonce": "fixture-nonce"}, "worker_stage": "work"}
            first_entered = threading.Event(); release_first = threading.Event(); second_started = threading.Event()
            original_fsync = os.fsync; first = {"pending": True}; errors = []
            def blocked_fsync(fd):
                if first["pending"]:
                    first["pending"] = False; first_entered.set()
                    if not release_first.wait(5):
                        raise TimeoutError("fixture did not release first heartbeat writer")
                return original_fsync(fd)
            def write(label):
                try:
                    if label == "second": second_started.set()
                    source_overfit._write_worker_heartbeat(authorization, interval=0)
                except Exception as error:
                    errors.append((label, type(error).__name__, str(error)))
            with patch.object(source_overfit.os, "fsync", side_effect=blocked_fsync):
                first_thread = threading.Thread(target=write, args=("first",))
                first_thread.start()
                self.assertTrue(first_entered.wait(5), "first heartbeat did not reach fsync barrier")
                second_thread = threading.Thread(target=write, args=("second",))
                second_thread.start()
                self.assertTrue(second_started.wait(5), "second heartbeat writer did not start")
                release_first.set()
                first_thread.join(timeout=5); second_thread.join(timeout=5)
            self.assertFalse(first_thread.is_alive()); self.assertFalse(second_thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(job._read_json(target)["descriptor_nonce"], "fixture-nonce")

    @unittest.skipUnless(_ML_READY, "native startup recovery fixture requires optional ML dependencies")
    def test_receipt_before_status_crash_attaches_actual_worker_without_duplicate(self):
        from dwr.experiments.records import verify_artifacts
        with tempfile.TemporaryDirectory() as directory_s:
            directory = Path(directory_s); config = directory/"fixture.json"; _fixture_bundle(config)
            jobroot = directory/"job"; lock_path = directory/"lock"
            ready, release = directory/"worker-ready", directory/"release"
            first = _launch_fixture_supervisor(config, jobroot, lock_path, ready, release,
                "identity_before_status", directory)
            receipt_marker = Path(str(ready)+".identity-receipt")
            self.assertTrue(_wait_until(receipt_marker.exists, 30), "worker identity/status crash window not reached")
            self.assertIsNone(job._status_data(jobroot).get("worker_pid"), "fixture did not stop before status PID publication")
            identity = job._read_json(jobroot/"worker-000.identity.json")
            supervisor_status = job._status_data(jobroot)
            job._terminate_owned_process(supervisor_status["supervisor_pid"], supervisor_status["supervisor_creation_time"])
            first.wait(timeout=10)
            second = _launch_fixture_supervisor(config, jobroot, lock_path, ready, release,
                "identity_before_status", directory)
            self.assertTrue(_wait_until(lambda: "orphan_worker_attached" in
                (jobroot/"events.jsonl").read_text(encoding="utf-8"), 20), "successor did not reconcile durable receipts")
            release.write_text("continue", encoding="utf-8")
            second.wait(timeout=90)
            self.assertEqual(verify_artifacts(jobroot)["status"], "completed")
            events = [json.loads(line) for line in (jobroot/"events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(event["kind"] == "worker_started" for event in events), 1)
            self.assertEqual(sum(event["kind"] == "orphan_worker_attached" for event in events), 1)
            self.assertEqual(job._read_json(jobroot/"worker-000.identity.json")["pid"], identity["pid"])

    @unittest.skipUnless(_ML_READY, "detached native lifecycle fixture requires optional ML dependencies")
    def test_detached_supervisor_restart_attaches_to_live_native_worker_once(self):
        from dwr.experiments.records import verify_artifacts
        with tempfile.TemporaryDirectory() as directory_s:
            directory = Path(directory_s); config = directory/"fixture.json"
            _fixture_bundle(config)
            jobroot = directory/"job"; lock_path = directory/"lock"
            ready, release = directory/"worker-ready", directory/"release"
            first = _launch_fixture_supervisor(config, jobroot, lock_path, ready, release, "before", directory)
            if not _wait_until(ready.exists):
                stdout = (directory/"supervisor.stdout.log").read_text(encoding="utf-8", errors="replace")
                stderr = (directory/"supervisor.stderr.log").read_text(encoding="utf-8", errors="replace")
                self.fail(f"detached tiny-native worker did not publish identity; stdout={stdout!r}; stderr={stderr!r}")
            worker_identity = job._read_json(jobroot/"worker-000.identity.json")
            self.assertEqual(int(ready.read_text(encoding="utf-8")), worker_identity["pid"])
            self.assertTrue(job._pid_matches(worker_identity["pid"], worker_identity["creation_time"]))
            self.assertTrue(_wait_until(lambda: job._status_data(jobroot).get("worker_pid") == worker_identity["pid"]),
                "supervisor did not durably bind actual worker before work")
            first_status = job._status_data(jobroot)
            supervisor_pid = first_status["supervisor_pid"]
            supervisor_creation = first_status["supervisor_creation_time"]
            job._terminate_owned_process(supervisor_pid, supervisor_creation)
            try:
                first.wait(timeout=10)
            except subprocess.TimeoutExpired:
                wrapper_creation = job._process_creation_time(first.pid)
                if job._pid_matches(first.pid, wrapper_creation):
                    job._terminate_owned_process(first.pid, wrapper_creation)
                first.wait(timeout=10)
            second = _launch_fixture_supervisor(config, jobroot, lock_path, ready, release, "before", directory)
            if not _wait_until(lambda: "orphan_worker_attached" in
                    (jobroot/"events.jsonl").read_text(encoding="utf-8"), 20):
                stderr = (directory/"supervisor.stderr.log").read_text(encoding="utf-8", errors="replace")
                events = (jobroot/"events.jsonl").read_text(encoding="utf-8", errors="replace")
                release.write_text("continue", encoding="utf-8")
                second_status_path = jobroot/"status.json"
                if second_status_path.is_file():
                    second_status = job._status_data(jobroot)
                    if second_status.get("supervisor_pid") and job._pid_matches(
                            second_status["supervisor_pid"], second_status["supervisor_creation_time"]):
                        job._terminate_owned_process(second_status["supervisor_pid"],
                                                     second_status["supervisor_creation_time"])
                self.fail(f"restarted supervisor did not attach; events={events!r}; stderr={stderr!r}")
            release.write_text("continue", encoding="utf-8")
            try:
                second.wait(timeout=90)
            except subprocess.TimeoutExpired:
                second_creation = job._process_creation_time(second.pid)
                job._terminate_owned_process(second.pid, second_creation)
                self.fail("restarted supervisor did not seal the tiny-native lifecycle")
            terminal = verify_artifacts(jobroot)
            self.assertEqual(terminal["status"], "completed", str(terminal)+"\n"+
                (directory/"supervisor.stderr.log").read_text(encoding="utf-8", errors="replace")+"\n"+
                (jobroot/"events.jsonl").read_text(encoding="utf-8", errors="replace"))
            _assert_final_accounting(self, jobroot, terminal)
            job_events = [json.loads(line) for line in (jobroot/"events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(event["kind"] == "worker_started" for event in job_events), 1)
            self.assertEqual(sum(event["kind"] == "orphan_worker_attached" for event in job_events), 1)
            cursor = json.loads((jobroot/"recovery.latest.json").read_text(encoding="utf-8"))
            self.assertEqual(cursor["phase"], "complete")
            self.assertEqual(cursor["progress"]["optimizer_step"], 9)
            self.assertFalse((jobroot/"segment-001").exists())

    @unittest.skipUnless(_ML_READY, "native recovery fixture requires optional ML dependencies")
    def test_real_supervisor_recovers_native_worker_loss_at_checkpoint(self):
        from dwr.experiments import source_overfit
        from dwr.experiments.records import verify_artifacts
        from dwr.training.recovery import RecoveryStore
        with tempfile.TemporaryDirectory() as directory_s:
            directory = Path(directory_s); config = directory/"fixture.json"
            registration, source, seams = _fixture_bundle(config)
            _install_fixture_cpu_rngs(registration["config"]["seeds"])
            baseline_root = directory/"baseline"; baseline_root.mkdir()
            baseline_auth = {"jobroot": str(baseline_root), "segment_index": 0,
                "worker_pid": os.getpid(), "worker_creation_time": job._process_creation_time(os.getpid()),
                "supervisor_pid": os.getpid(), "supervisor_creation_time": job._process_creation_time(os.getpid()),
                "resume_descriptor": None}
            baseline = source_overfit._execute_registered(Path(__file__).resolve().parents[1], registration,
                baseline_auth, command=["baseline"], _test_seams=seams)
            model = registration["feasibility_config"]["model"]
            identities = source_overfit._kernel_identities(registration["config_raw"], registration["config"],
                source_commit=source["commit"], weight_sha256=model["weight_sha256"],
                tokenizer_inventory_sha256=model["artifact_inventory_sha256"])
            baseline_cursor = RecoveryStore.verify_descriptor(baseline["recovery_descriptor"], identities)

            jobroot = directory/"job"; lock_path = directory/"lock"
            ready, release = directory/"worker-ready", directory/"release"
            supervisor = _launch_fixture_supervisor(config, jobroot, lock_path, ready, release,
                "after_step", directory)
            step_ready = Path(str(ready)+".step")
            self.assertTrue(_wait_until(step_ready.exists, 45), "native worker did not commit a resumable boundary")
            status = job._status_data(jobroot)
            self.assertGreaterEqual(status["committed_step"], 0)
            worker_pid, worker_creation = status["worker_pid"], status["worker_creation_time"]
            job._terminate_owned_process(worker_pid, worker_creation)
            release.write_text("continue", encoding="utf-8")
            try:
                supervisor.wait(timeout=90)
            except subprocess.TimeoutExpired:
                supervisor_creation = job._process_creation_time(supervisor.pid)
                job._terminate_owned_process(supervisor.pid, supervisor_creation)
                self.fail("supervisor did not recover and seal after native worker loss")
            terminal = verify_artifacts(jobroot)
            self.assertEqual(terminal["status"], "completed", str(terminal)+"\n"+
                (directory/"supervisor.stderr.log").read_text(encoding="utf-8", errors="replace")+"\n"+
                (jobroot/"events.jsonl").read_text(encoding="utf-8", errors="replace"))
            _assert_final_accounting(self, jobroot, terminal)
            job_events = [json.loads(line) for line in (jobroot/"events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(event["kind"] == "worker_started" for event in job_events), 2)
            cursor_data = json.loads((jobroot/"recovery.latest.json").read_text(encoding="utf-8"))
            resumed = RecoveryStore.verify_descriptor(job._cursor_pointer(jobroot), identities)
            self.assertEqual(cursor_data["progress"]["optimizer_step"], 9)
            self.assertEqual(cursor_data["phase"], "complete")
            self.assertEqual(resumed["checkpoint_sha256"], baseline_cursor["checkpoint_sha256"])
            self.assertFalse((jobroot/"segment-000"/"manifest.json").exists())

    @unittest.skipUnless(_ML_READY, "native post-probe recovery fixture requires optional ML dependencies")
    def test_real_supervisor_recovers_after_probe_and_audits_probe_producing_segment(self):
        from dwr.experiments.records import verify_artifacts
        from dwr.training.recovery import RecoveryStore
        with tempfile.TemporaryDirectory() as directory_s:
            directory = Path(directory_s); config = directory/"fixture.json"
            jobroot = directory/"job"; lock_path = directory/"lock"
            ready, release = directory/"worker-ready", directory/"release"
            supervisor = _launch_fixture_supervisor(config, jobroot, lock_path, ready, release,
                "after_probe", directory)
            probe_ready = Path(str(ready)+".probe")
            self.assertTrue(_wait_until(probe_ready.exists, 60),
                "native worker did not commit the post-probe continuation cursor")
            first_pointer = json.loads((jobroot/"recovery.latest.json").read_text(encoding="utf-8"))
            self.assertEqual(first_pointer["phase"], "training")
            self.assertEqual(first_pointer["progress"]["optimizer_step"], 4)
            status = job._status_data(jobroot)
            worker_pid, worker_creation = status["worker_pid"], status["worker_creation_time"]
            job._terminate_owned_process(worker_pid, worker_creation)
            release.write_text("continue", encoding="utf-8")
            try:
                supervisor.wait(timeout=100)
            except subprocess.TimeoutExpired:
                supervisor_creation = job._process_creation_time(supervisor.pid)
                job._terminate_owned_process(supervisor.pid, supervisor_creation)
                self.fail("supervisor did not resume/audit the completed post-probe cursor")
            terminal = verify_artifacts(jobroot)
            self.assertEqual(terminal["status"], "completed", str(terminal)+"\n"+
                (directory/"supervisor.stderr.log").read_text(encoding="utf-8", errors="replace")+"\n"+
                (jobroot/"events.jsonl").read_text(encoding="utf-8", errors="replace"))
            pointer = json.loads((jobroot/"recovery.latest.json").read_text(encoding="utf-8"))
            cursor = RecoveryStore.verify_descriptor(job._cursor_pointer(jobroot), pointer["identities"])
            self.assertEqual(pointer["phase"], "complete")
            self.assertEqual(pointer["segment"], "segment-001")
            probe = cursor["facts"]["resume_probe"]
            self.assertEqual(PurePosixPath(probe["path"]).parts[0], "segment-000")
            first_segment_events = [json.loads(line) for line in
                (jobroot/"segment-000"/"events.jsonl").read_text(encoding="utf-8").splitlines()]
            second_segment_events = [json.loads(line) for line in
                (jobroot/"segment-001"/"events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(event["kind"] == "resume_verified" for event in first_segment_events), 1)
            self.assertFalse(any(event["kind"] == "resume_verified" for event in second_segment_events),
                "recovery must reuse the durable probe verdict without replaying it")

    @unittest.skipUnless(_ML_READY, "native complete-cursor fixture requires optional ML dependencies")
    def test_supervisor_audits_complete_cursor_after_worker_dies_before_segment_seal(self):
        from dwr.experiments.records import verify_artifacts
        with tempfile.TemporaryDirectory() as directory_s:
            directory = Path(directory_s); config = directory/"fixture.json"
            jobroot = directory/"job"; lock_path = directory/"lock"
            ready, release = directory/"worker-ready", directory/"release"
            supervisor = _launch_fixture_supervisor(config, jobroot, lock_path, ready, release,
                "after_complete", directory)
            complete_ready = Path(str(ready)+".complete")
            self.assertTrue(_wait_until(complete_ready.exists, 90),
                "native worker did not publish its complete cursor")
            status = job._status_data(jobroot)
            self.assertEqual(status["worker_pid"], int(ready.read_text(encoding="utf-8")))
            job._terminate_owned_process(status["worker_pid"], status["worker_creation_time"])
            release.write_text("continue", encoding="utf-8")
            try:
                supervisor.wait(timeout=60)
            except subprocess.TimeoutExpired:
                supervisor_creation = job._process_creation_time(supervisor.pid)
                job._terminate_owned_process(supervisor.pid, supervisor_creation)
                self.fail("supervisor did not finish the verified complete cursor")
            terminal = verify_artifacts(jobroot)
            self.assertEqual(terminal["status"], "completed", str(terminal)+"\n"+
                (directory/"supervisor.stderr.log").read_text(encoding="utf-8", errors="replace"))
            events = [json.loads(line) for line in (jobroot/"events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(event["kind"] == "worker_started" for event in events), 1)
            completed = [event for event in events if event["kind"] == "job_completed_from_verified_complete_cursor"]
            self.assertEqual(len(completed), 1)
            self.assertFalse(completed[0]["payload"]["segment_terminal_present"])
            self.assertFalse((jobroot/"segment-000"/"manifest.json").exists())

    @unittest.skipUnless(_ML_READY, "native invariant fixture requires optional ML dependencies")
    def test_native_worker_invariant_failure_is_sealed_without_recovery(self):
        from dwr.experiments.records import verify_artifacts
        with tempfile.TemporaryDirectory() as directory_s:
            directory = Path(directory_s); config = directory/"fixture.json"
            jobroot = directory/"job"; lock_path = directory/"lock"
            ready, release = directory/"worker-ready", directory/"release"
            supervisor = _launch_fixture_supervisor(config, jobroot, lock_path, ready, release,
                "fault", directory)
            try:
                supervisor.wait(timeout=60)
            except subprocess.TimeoutExpired:
                supervisor_creation = job._process_creation_time(supervisor.pid)
                job._terminate_owned_process(supervisor.pid, supervisor_creation)
                self.fail("supervisor did not seal the native invariant failure")
            terminal = verify_artifacts(jobroot)
            self.assertEqual(terminal["status"], "failed")
            segment_terminal = verify_artifacts(jobroot/"segment-000")
            self.assertEqual(segment_terminal["status"], "failed")
            self.assertTrue((jobroot/"segment-000"/"worker-failure.json").is_file())
            events = [json.loads(line) for line in (jobroot/"events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(event["kind"] == "worker_started" for event in events), 1)
            self.assertEqual(sum(event["kind"] == "worker_exited" for event in events), 1)
            self.assertFalse(any(event["kind"] == "worker_started" and
                event["payload"].get("segment_index") == 1 for event in events))

    @unittest.skipUnless(_ML_READY, "native cleanup fixture requires optional ML dependencies")
    def test_supervisor_exception_cleans_actual_worker_after_cursor_corruption(self):
        from dwr.experiments.records import verify_artifacts
        with tempfile.TemporaryDirectory() as directory_s:
            directory = Path(directory_s); config = directory/"fixture.json"
            jobroot = directory/"job"; lock_path = directory/"lock"
            ready, release = directory/"worker-ready", directory/"release"
            supervisor = _launch_fixture_supervisor(config, jobroot, lock_path, ready, release,
                "before", directory)
            self.assertTrue(_wait_until(ready.exists, 30), "native worker did not publish its identity receipt")
            self.assertTrue(_wait_until(lambda: job._status_data(jobroot).get("worker_pid") is not None, 15),
                "supervisor did not record the actual worker PID")
            status = job._status_data(jobroot)
            worker_pid, worker_creation = status["worker_pid"], status["worker_creation_time"]
            (jobroot/"recovery.latest.json").write_bytes(b"{corrupt cursor\n")
            try:
                supervisor.wait(timeout=45)
            except subprocess.TimeoutExpired:
                supervisor_creation = job._process_creation_time(supervisor.pid)
                job._terminate_owned_process(supervisor.pid, supervisor_creation)
                self.fail("supervisor failed to clean up after cursor verification error")
            self.assertFalse(job._pid_matches(worker_pid, worker_creation),
                "supervisor exception left the actual worker running")
            with job._ByteLock(lock_path, 1, wait=False):
                pass
            self.assertEqual(verify_artifacts(jobroot)["status"], "failed")
            events = [json.loads(line) for line in (jobroot/"events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(event["kind"] == "supervisor_failure" for event in events))
            self.assertFalse(any(event["kind"] == "worker_cleanup_failed" for event in events), repr(events))

    def test_utc_anchor_uses_timezone_aware_epoch_and_rejects_rollback(self):
        instant = job._parse_utc("2026-10-17T00:00:00Z")
        self.assertEqual(instant.timestamp(), 1792195200.0)
        with self.assertRaisesRegex(ValueError, "rollback|clock"):
            job._validated_elapsed(instant.timestamp(), instant.timestamp() - 1, 10.0)

    def test_elapsed_budget_reopen_accumulates_from_original_durable_receipt(self):
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s)
            started = "2026-10-17T00:00:00Z"
            receipt = {"schema_version": 1, "started_utc": started,
                "last_observed_utc_epoch": 1000.0, "elapsed_seconds": 500.0}
            (root/"elapsed-budget.json").write_bytes(job._canonical(receipt)+b"\n")
            with patch.object(job.time, "time", return_value=1015.0):
                self.assertEqual(job._persist_elapsed(root), 515.0)
            persisted = job._read_json(root/"elapsed-budget.json")
            self.assertEqual(persisted["started_utc"], started)
            self.assertEqual(persisted["elapsed_seconds"], 515.0)
            with patch.object(job.time, "time", return_value=1014.0):
                with self.assertRaisesRegex(ValueError, "rollback"):
                    job._persist_elapsed(root)

    def test_process_creation_identity_is_stable_for_current_process(self):
        first = job._process_creation_time(os.getpid())
        time.sleep(0.01)
        self.assertEqual(job._process_creation_time(os.getpid()), first)
        with self.assertRaises((OSError, ValueError)):
            job._process_creation_time(2**31 - 1)

    def test_atomic_status_write_retries_transient_windows_replace_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            replace = os.replace
            attempts = []
            def transient_then_replace(source, target):
                attempts.append((source, target))
                if len(attempts) == 1:
                    raise PermissionError("fixture transient Windows sharing lock")
                return replace(source, target)
            with patch.object(job.os, "name", "nt"), patch.object(job.os, "replace",
                    side_effect=transient_then_replace):
                job._atomic_json(path, {"state": "running"})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(job._read_json(path), {"state": "running"})

    def test_atomic_status_write_fails_closed_and_preserves_old_status_after_retry_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            path.write_bytes(job._canonical({"state": "old"}) + b"\n")
            attempts = []
            def permanently_locked(source, target):
                attempts.append((source, target))
                raise PermissionError("fixture persistent Windows sharing lock")
            with patch.object(job.os, "name", "nt"), patch.object(job.os, "replace",
                    side_effect=permanently_locked):
                with self.assertRaisesRegex(PermissionError, "persistent"):
                    job._atomic_json(path, {"state": "new"})
            self.assertEqual(len(attempts), 6)
            self.assertEqual(job._read_json(path), {"state": "old"})
            self.assertTrue(attempts[-1][0].exists(), "failed atomic replacement must preserve its fsynced temp")

    def test_supervisor_lock_byte_excludes_another_process(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "owner.lock"
            with job._ByteLock(lock_path, 0):
                code = ("from pathlib import Path; from dwr.experiments.source_overfit_job import _ByteLock; "
                        "import sys; "
                        "\ntry: _ByteLock(Path(sys.argv[1]), 0); raise SystemExit(9) "
                        "\nexcept RuntimeError: raise SystemExit(0)")
                child = subprocess.run([sys.executable, "-c", code, str(lock_path)],
                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
                self.assertEqual(child.returncode, 0, child.stdout + child.stderr)

    def test_whole_job_caps_count_started_updates_and_incomplete_prediction_attempts(self):
        from dwr.experiments.source_overfit import _check_physical_job_caps
        config = {"limits": {"maximum_physical_updates": 197, "maximum_predictions": 18,
            "maximum_decoder_calls": 2304}, "decoding": {"max_new_tokens_including_eos": 128}}
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s); segment = root/"segment-000"; segment.mkdir()
            events = [
                {"sequence": 1, "utc": "2026-10-17T00:00:00Z", "kind": "update_started", "payload": {}},
                {"sequence": 2, "utc": "2026-10-17T00:00:01Z", "kind": "update_started", "payload": {}},
            ]
            raw = b"".join(job._canonical(event)+b"\n" for event in events)
            (segment/"events.jsonl").write_bytes(raw)
            config["limits"]["maximum_physical_updates"] = 2
            with self.assertRaisesRegex(RuntimeError, "physical update/prediction/decoder cap"):
                _check_physical_job_caps(root, config, "update_started")
            events[0]["kind"] = "prediction_started"
            events[1]["kind"] = "prediction_started"
            (segment/"events.jsonl").write_bytes(b"".join(job._canonical(event)+b"\n" for event in events))
            config["limits"]["maximum_physical_updates"] = 197
            config["limits"]["maximum_predictions"] = 2
            with self.assertRaisesRegex(RuntimeError, "physical update/prediction/decoder cap"):
                _check_physical_job_caps(root, config, "prediction_started")

    def test_known_worker_fault_receipt_blocks_cursorless_recovery(self):
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s); segment = root/"segment-000"; segment.mkdir()
            (segment/"events.jsonl").write_bytes(b"")
            (segment/"worker-failure.json").write_bytes(b'{"error":"NUMERICAL_DIVERGENCE"}\n')
            self.assertIsNone(job._load_cursor(root, {key: "a"*64 for key in
                ("source", "config", "data", "model", "tokenizer", "schedule")}))
            self.assertEqual(job._known_worker_fault(root), "known_worker_fault_no_recovery")

    def test_kernel_fault_event_blocks_recovery_before_failure_terminal(self):
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s); segment = root/"segment-000"; segment.mkdir()
            events = [{"sequence": 1, "utc": "2026-10-17T00:00:00Z", "kind": "gradient_invariant_failure",
                       "payload": {"role": "workspace"}}]
            (segment/"events.jsonl").write_bytes(b"".join(job._canonical(row)+b"\n" for row in events))
            self.assertEqual(job._known_worker_fault(root), "known_kernel_fault_no_recovery")

    def test_probe_audit_uses_additive_registered_tolerance(self):
        self.assertTrue(job._within_registered_tolerance([0.1000015], [0.1], atol=1e-6, rtol=1e-5))
        self.assertFalse(job._within_registered_tolerance([0.100003], [0.1], atol=1e-6, rtol=1e-5))

    def test_job_evidence_path_rejects_absolute_reference(self):
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s); outside = root.parent/(root.name+"-outside.txt"); outside.write_text("evidence")
            try:
                with self.assertRaisesRegex(ValueError, "relative|contained"):
                    job._safe_job_file_sha(root, str(outside.resolve()))
            finally:
                outside.unlink(missing_ok=True)

    def test_whole_job_ledger_retains_unfinished_update_after_worker_exit(self):
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s); segment = root/"segment-000"; segment.mkdir()
            (segment/"events.jsonl").write_bytes(job._canonical({"sequence": 1, "utc": "2026-10-17T00:00:00Z",
                "kind": "update_started", "payload": {"net_step": 1}})+b"\n")
            job_events = [
                {"sequence": 1, "utc": "2026-10-17T00:00:00Z", "kind": "worker_started", "payload": {"segment_index": 0}},
                {"sequence": 2, "utc": "2026-10-17T00:00:01Z", "kind": "worker_exited", "payload": {"segment_index": 0}},
            ]
            (root/"events.jsonl").write_bytes(b"".join(job._canonical(row)+b"\n" for row in job_events))
            config = {"limits": {"maximum_physical_updates": 197, "maximum_predictions": 18,
                "maximum_decoder_calls": 2304, "maximum_worker_processes": 3, "maximum_operational_recoveries": 2}}
            ledger = job._aggregate_physical_work(root, config)
            self.assertEqual(ledger["observed"]["unknown_inflight_worker_updates"], 1)
            self.assertEqual(ledger["observed"]["conservative_physical_updates"], 1)

    def test_work_cutoff_allows_worker_seal_until_distinct_seal_deadline(self):
        work, seal, outer = 100.0, 110.0, 112.0
        self.assertEqual(job._worker_cutoff(100.1, "work", work, seal, outer), "work_cutoff")
        self.assertIsNone(job._worker_cutoff(100.1, "seal", work, seal, outer))
        self.assertEqual(job._worker_cutoff(110.0, "seal", work, seal, outer), "seal_cutoff")
        self.assertEqual(job._worker_cutoff(112.0, "seal", work, seal, outer), "outer_watchdog")

    def test_watchdog_clock_rollback_is_charged_as_expiry(self):
        from dwr.experiments.source_overfit_watchdog import _clock_rollback
        self.assertTrue(_clock_rollback(100.0, 99.0, 50.0, 51.0))
        self.assertTrue(_clock_rollback(100.0, 101.0, 50.0, 49.0))
        self.assertFalse(_clock_rollback(100.0, 101.0, 50.0, 51.0))

    def test_watchdog_hash_enumeration_stops_at_entry_cap(self):
        from dwr.experiments import source_overfit_watchdog as watchdog
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s)
            for index in range(5):
                (root/f"{index}.bin").write_bytes(str(index).encode("ascii"))
            with patch.object(watchdog, "MAX_HASH_ENTRIES", 2):
                receipt = watchdog.bounded_artifact_hashes(root, 5)
            self.assertEqual(len(receipt["files"]), 2)
            self.assertFalse(receipt["hash_scan_completed"])

    def test_failed_seal_retains_bounded_partial_artifact_hash_receipt(self):
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s); artifact = root/"partial.bin"; artifact.write_bytes(b"partial")
            job._record_seal_failure_receipt(root, "2026-10-17T00:00:00Z", 5401,
                "before_artifact_hashing", OSError("injected seal failure"), "a"*40)
            receipt = job._read_json(root/"watchdog-receipt.json")
            row = next(item for item in receipt["files"] if item["path"] == "partial.bin")
            self.assertEqual(row["sha256"], hashlib.sha256(b"partial").hexdigest())
            self.assertEqual(receipt["error_type"], "OSError")
            self.assertLessEqual(receipt["hash_budget_seconds"], 55)

    def test_failed_seal_retains_bounded_partial_artifact_hash_receipt(self):
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s); artifact = root/"partial.bin"; artifact.write_bytes(b"partial")
            job._record_seal_failure_receipt(root, "2026-10-17T00:00:00Z", 5401,
                "before_artifact_hashing", OSError("injected seal failure"), "a"*40)
            receipt = job._read_json(root/"watchdog-receipt.json")
            row = next(item for item in receipt["files"] if item["path"] == "partial.bin")
            self.assertEqual(row["sha256"], hashlib.sha256(b"partial").hexdigest())
            self.assertEqual(receipt["error_type"], "OSError")
            self.assertLessEqual(receipt["hash_budget_seconds"], 55)

    def test_independent_watchdog_process_enforces_accelerated_deadline_and_hashes_partial_files(self):
        from dwr.experiments.source_overfit_job import _process_creation_time
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s); (root/"artifact.bin").write_bytes(b"partial evidence")
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], cwd=root,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=(getattr(subprocess, "DETACHED_PROCESS", 0) |
                               getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)))
            creation = job._process_creation_time(child.pid)
            (root/"status.json").write_bytes(job._canonical({"supervisor_pid": child.pid,
                "supervisor_creation_time": creation, "jobroot": str(root.resolve())})+b"\n")
            watchdog = subprocess.Popen([sys.executable, "-m", "dwr.experiments.source_overfit_watchdog",
                "--jobroot", str(root), "--supervisor-pid", str(child.pid), "--supervisor-creation", creation,
                "--deadline-epoch", str(time.time()+0.7)], cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=(getattr(subprocess, "DETACHED_PROCESS", 0) |
                               getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)))
            try:
                watchdog.wait(timeout=15)
                self.assertEqual(watchdog.returncode, 0)
                self.assertFalse(job._pid_matches(child.pid, creation), "watchdog did not terminate owned supervisor")
                receipt = job._read_json(root/"outer-watchdog-receipt.json")
                row = next(item for item in receipt["files"] if item["path"] == "artifact.bin")
                self.assertEqual(row["sha256"], hashlib.sha256(b"partial evidence").hexdigest())
                child.wait(timeout=5)
            finally:
                if job._pid_matches(child.pid, creation): job._terminate_owned_process(child.pid, creation)
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: pass

    def test_watchdog_survives_supervisor_exit_while_verified_worker_remains_live(self):
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s); ready = root/"receipts-ready"
            code = ("import json,os,subprocess,sys,time; from pathlib import Path; "
                "from dwr.experiments import source_overfit_job as job; "
                "root=Path(sys.argv[1]); ready=Path(sys.argv[2]); "
                "spid=os.getpid(); screation=job._process_creation_time(spid); "
                "worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], "
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL, "
                "creationflags=getattr(subprocess,'DETACHED_PROCESS',0)|getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0)); "
                "nonce='fixture-nonce'; source='a'*40; creation=job._process_creation_time(worker.pid); "
                "descriptor={'nonce':nonce,'source_commit':source,'supervisor_pid':spid, "
                    "'supervisor_creation_time':screation,'segment_index':0}; "
                "launcher={'schema_version':1,'pid':worker.pid,'creation_time':creation,'image':job._process_image(worker.pid), "
                    "'descriptor_nonce':nonce,'supervisor_pid':spid,'supervisor_creation_time':screation}; "
                "identity={'pid':worker.pid,'creation_time':creation,'image':job._process_image(worker.pid), "
                    "'parent_pid':spid,'parent_creation_time':screation,'descriptor_nonce':nonce, "
                    "'config_sha256':job.CONFIG_SHA256,'source_commit':source,'segment_index':0}; "
                "(root/'worker-descriptor-000.json').write_bytes(job._canonical(descriptor)+b'\\n'); "
                "(root/'worker-000.launcher-receipt.json').write_bytes(job._canonical(launcher)+b'\\n'); "
                "(root/'worker-000.identity.json').write_bytes(job._canonical(identity)+b'\\n'); "
                "ready.write_text(str(spid)+','+screation+','+str(worker.pid)+','+creation); time.sleep(.35)")
            supervisor = subprocess.Popen([sys.executable, "-c", code, str(root), str(ready)],
                cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=(getattr(subprocess, "DETACHED_PROCESS", 0) |
                               getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)))
            self.assertTrue(_wait_until(ready.exists, 10), "fixture supervisor did not publish worker receipts")
            spid_s, screation, worker_s, worker_creation = ready.read_text(encoding="utf-8").split(",")
            supervisor.wait(timeout=5)
            started = time.monotonic(); deadline_epoch = time.time()+0.8
            watchdog = subprocess.Popen([sys.executable, "-m", "dwr.experiments.source_overfit_watchdog",
                "--jobroot", str(root), "--supervisor-pid", spid_s, "--supervisor-creation", screation,
                "--deadline-epoch", str(deadline_epoch)], cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=(getattr(subprocess, "DETACHED_PROCESS", 0) |
                               getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)))
            try:
                watchdog.wait(timeout=15)
                self.assertEqual(watchdog.returncode, 0)
                self.assertGreaterEqual(time.monotonic()-started, 0.65,
                    "watchdog exited with supervisor while a verified worker remained live")
                self.assertFalse(job._pid_matches(int(worker_s), worker_creation))
                receipt = job._read_json(root/"outer-watchdog-receipt.json")
                self.assertEqual(receipt["termination_errors"], [])
                self.assertEqual(receipt["workers"], [{"pid": int(worker_s), "creation_time": worker_creation}])
            finally:
                if job._pid_matches(int(worker_s), worker_creation): job._terminate_owned_process(int(worker_s), worker_creation)
                try: watchdog.wait(timeout=5)
                except subprocess.TimeoutExpired: pass

    def test_watchdog_expiry_targets_verified_launcher_when_actual_worker_exited(self):
        from dwr.experiments import source_overfit_watchdog as watchdog
        with tempfile.TemporaryDirectory() as directory_s:
            root = Path(directory_s); supervisor_pid, worker_pid, launcher_pid = 10, 11, 12
            supervisor_creation, worker_creation, launcher_creation = "sup-create", "worker-create", "launcher-create"
            nonce, source = "fixture-nonce", "a"*40
            descriptor = {"nonce": nonce, "source_commit": source, "supervisor_pid": supervisor_pid,
                "supervisor_creation_time": supervisor_creation, "segment_index": 0}
            launcher = {"schema_version": 1, "pid": launcher_pid, "creation_time": launcher_creation, "image": "launcher-image",
                "descriptor_nonce": nonce, "supervisor_pid": supervisor_pid,
                "supervisor_creation_time": supervisor_creation}
            identity = {"pid": worker_pid, "creation_time": worker_creation, "image": "worker-image",
                "parent_pid": supervisor_pid, "parent_creation_time": supervisor_creation,
                "descriptor_nonce": nonce, "config_sha256": job.CONFIG_SHA256,
                "source_commit": source, "segment_index": 0}
            for name, value in (("worker-descriptor-000.json", descriptor),
                                ("worker-000.launcher-receipt.json", launcher),
                                ("worker-000.identity.json", identity)):
                (root/name).write_bytes(job._canonical(value)+b"\n")
            alive = {supervisor_pid: False, worker_pid: False, launcher_pid: True}
            termination_calls = []; image_queries = []
            def matches(pid, creation):
                expected = {supervisor_pid: supervisor_creation, worker_pid: worker_creation,
                    launcher_pid: launcher_creation}.get(pid)
                return alive.get(pid, False) and creation == expected
            def image(pid):
                image_queries.append(pid)
                if pid == worker_pid:
                    raise AssertionError("dead actual worker image must not be queried")
                return "launcher-image" if pid == launcher_pid else "supervisor-image"
            def terminate(pid, creation):
                if matches(pid, creation):
                    termination_calls.append(pid); alive[pid] = False
            with patch.object(job, "_pid_matches", side_effect=matches), \
                    patch.object(job, "_process_image", side_effect=image), \
                    patch.object(job, "_terminate_owned_process", side_effect=terminate), \
                    patch.object(job, "_terminate_worker_tree", side_effect=AssertionError("dead worker tree")):
                code = watchdog._run(root, supervisor_pid, supervisor_creation, time.time()-1.0)
            self.assertEqual(code, 0)
            self.assertEqual(termination_calls, [launcher_pid])
            self.assertEqual(image_queries, [launcher_pid])
            self.assertFalse(alive[launcher_pid])
            receipt = job._read_json(root/"outer-watchdog-receipt.json")
            self.assertEqual(receipt["termination_errors"], [])

            alive[launcher_pid] = True; termination_calls.clear(); image_queries.clear()
            with patch.object(job, "_pid_matches", side_effect=lambda pid, creation:
                    matches(pid, "reused-pid" if pid == launcher_pid else creation)), \
                    patch.object(job, "_process_image", side_effect=image), \
                    patch.object(job, "_terminate_owned_process", side_effect=terminate), \
                    patch.object(job, "_terminate_worker_tree", side_effect=AssertionError("dead worker tree")):
                code = watchdog._run(root, supervisor_pid, supervisor_creation, time.time()-1.0)
            self.assertEqual(code, 0)
            self.assertEqual(termination_calls, [], "reused launcher PID must not be terminated")

    def test_detached_dummy_worker_outlives_its_launcher_process(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "finished.txt"
            child_code = ("import pathlib,time; time.sleep(0.4); "
                          f"pathlib.Path({str(marker)!r}).write_text('alive', encoding='utf-8')")
            launcher_code = ("import subprocess,sys; "
                "flags=getattr(subprocess,'DETACHED_PROCESS',0)|getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0); "
                "subprocess.Popen([sys.executable,'-c',sys.argv[1]], stdin=subprocess.DEVNULL, "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)")
            launcher = subprocess.run([sys.executable, "-c", launcher_code, child_code],
                capture_output=True, text=True, timeout=15)
            self.assertEqual(launcher.returncode, 0, launcher.stdout + launcher.stderr)
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(marker.read_text(encoding="utf-8"), "alive")

    def test_supervisor_restart_keeps_live_worker_lock_owned_and_prevents_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(__file__).resolve().parents[1]
            lock_path = Path(directory) / "job.lock"
            ready = Path(directory) / "worker-ready"
            done = Path(directory) / "worker-done"
            worker_code = ("import pathlib,sys,time; from dwr.experiments.source_overfit_job import _ByteLock; "
                "lock=_ByteLock(pathlib.Path(sys.argv[1]),1); "
                "pathlib.Path(sys.argv[2]).write_text('ready',encoding='utf-8'); time.sleep(4); "
                "pathlib.Path(sys.argv[3]).write_text('done',encoding='utf-8'); lock.close()")
            flags = (getattr(subprocess, "DETACHED_PROCESS", 0) |
                     getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            worker = subprocess.Popen([sys.executable, "-c", worker_code, str(lock_path), str(ready), str(done)],
                cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=flags)
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(ready.exists(), "worker did not acquire its process-lifetime lock")
            worker_creation = job._process_creation_time(worker.pid)
            first_supervisor = ("import pathlib,sys; from dwr.experiments.source_overfit_job import _ByteLock; "
                "lock=_ByteLock(pathlib.Path(sys.argv[1]),0); "
                "pathlib.Path(sys.argv[2]).write_text('stopped',encoding='utf-8'); lock.close()")
            first = subprocess.run([sys.executable, "-c", first_supervisor, str(lock_path),
                str(Path(directory) / "supervisor-stopped")], cwd=root, capture_output=True, text=True, timeout=15)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertTrue((Path(directory) / "supervisor-stopped").is_file())
            restarted = ("import pathlib,sys; from dwr.experiments.source_overfit_job import _ByteLock; "
                "from pathlib import Path; owner=_ByteLock(Path(sys.argv[1]),0); "
                "\ntry: _ByteLock(Path(sys.argv[1]),1); raise SystemExit(9) "
                "\nexcept RuntimeError: print('live worker lock retained')")
            second = subprocess.run([sys.executable, "-c", restarted, str(lock_path)], cwd=root,
                capture_output=True, text=True, timeout=15)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("live worker lock retained", second.stdout)
            self.assertEqual(job._process_creation_time(worker.pid), worker_creation)
            worker.wait(timeout=10)
            self.assertEqual(worker.returncode, 0)
            self.assertTrue(done.is_file())

    def test_only_registered_config_is_accepted_without_launching_job(self):
        with self.assertRaises(ValueError):
            job._config(Path(__file__).resolve())

    def test_status_is_read_only_when_job_has_not_started(self):
        with tempfile.TemporaryDirectory() as directory_s:
            base = Path(directory_s); root = base/"runs"/"tiny-job"; root.mkdir(parents=True)
            (root/"status.json").write_bytes(job._canonical({"state": "running", "jobroot": str(root.resolve())})+b"\n")
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            with patch.object(job, "_root", return_value=base), patch.object(job, "_config", return_value=base/"fixture.json"), \
                    patch.object(job, "JOB_DIR", "runs/tiny-job"):
                result = job.status(base/"fixture.json")
            after = {path.name: path.read_bytes() for path in root.iterdir()}
            self.assertEqual(result["state"], "running")
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
