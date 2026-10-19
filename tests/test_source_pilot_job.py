"""Focused tests for the detached source-validation pilot supervisor."""
from __future__ import annotations

import hashlib
import json
import os
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone

from dwr.experiments.source_pilot_job import (
    _canonical,
    _derive_deadlines,
    _physical_ledger,
    _read_status,
    _verify_opaque_resume,
    _verify_live_worker_receipt,
    _validated_elapsed,
    guard_worker_event,
    read_worker_authorization,
)


class SourcePilotAuthorizationTests(unittest.TestCase):
    def _record(self, root: Path, settings=None, run_dir: Path | None = None):
        from dwr.experiments.records import RunRecord

        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "pilot-test@example.invalid"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Pilot Test"], cwd=root, check=True)
        config = root / "pilot.json"
        config.write_bytes(b'{"fixture":true}\n')
        (root / "runs").mkdir()
        subprocess.run(["git", "add", "pilot.json"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=root, check=True)
        output_dir = run_dir or (root / "runs" / "segment-000")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        record = RunRecord.create(output_dir, command=["fixture"],
            config_path=config, expected_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            identities={"config": {"name": "fixture", "revision": "1", "sha256": "a" * 64}},
            rng_streams={"fixture": 1}, settings=settings or {"record_kind": "segment"},
            budgets={"wall_seconds": 10, "memory_mib": 10, "attempts": 1}, source_root=root)
        return record

    def test_postseal_child_stderr_does_not_change_segment_manifest(self):
        from dwr.experiments.records import verify_artifacts
        from dwr.experiments import source_pilot_job as job

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deadlines = {"wall_seconds": 10}
            matrix = root / "matrix"
            arm = matrix / "arms" / "K0"
            segment = arm / "segment-000"
            record = self._record(root, {"record_kind": "segment", "original_deadlines": deadlines}, segment)
            job._write_status(matrix, state="running", completed_arms=[], current_arm="K0")
            command = [sys.executable, "-c",
                "import sys; from pathlib import Path; "
                "from dwr.experiments.source_pilot_job import _reopen_record; "
                "from dwr.experiments.source_pilot import _segment_artifacts; "
                "p=Path(sys.argv[1]); r=_reopen_record(p, {'wall_seconds': 10}); "
                "r.finish('interrupted', counts={}, artifacts=_segment_artifacts(p), error='KeyboardInterrupt: injected'); "
                "sys.stderr.write('post-seal traceback marker\\n'); sys.stderr.flush()",
                str(segment)]
            current = job._process_identity(os.getpid())
            descriptor = {"launcher": {key: current[key] for key in ("pid", "creation_time", "image")},
                "nonce": "b" * 32, "segment_index": 0}
            descriptor_path = segment / "worker-descriptor.json"
            descriptor_path.write_bytes(_canonical(descriptor) + b"\n")
            descriptor_sha = job._sha(descriptor_path.read_bytes())
            process, stdout_path, stderr_path = job._launch_worker(descriptor_path, descriptor_sha,
                matrix, arm, segment, worker_command=lambda _path, _sha: command)
            return_code = process.wait(timeout=15)
            self.assertEqual(return_code, 0, stderr_path.read_text(encoding="utf-8"))
            self.assertNotIn(segment.resolve(), stdout_path.resolve().parents)
            self.assertNotIn(segment.resolve(), stderr_path.resolve().parents)
            self.assertIn("post-seal traceback marker", stderr_path.read_text(encoding="utf-8"))
            self.assertEqual(verify_artifacts(segment)["status"], "interrupted")

    def test_worker_log_receipt_drives_fault_scan_and_detects_changed_final_bytes(self):
        from dwr.experiments import source_pilot_job as job
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = root / "runs" / "matrix"
            arm = matrix / "arms" / "K0"
            segment = arm / "segment-000"
            record = self._record(root, {"record_kind": "arm"}, arm)
            segment.mkdir()
            stdout_path, stderr_path = job._worker_log_paths(arm, segment)
            stdout_path.parent.mkdir(parents=True)
            stdout_path.write_bytes(b"")
            stderr_path.write_text("Traceback (most recent call last):\nRuntimeError: known fault\n", encoding="utf-8")
            self.assertTrue(job._known_worker_fault(segment))
            physical_audit = {"schema_version": "dwr-source-pilot-physical-audit-v1", "worker_logs": [
                {"path": stdout_path.relative_to(arm).as_posix(), "sha256": job._sha(stdout_path.read_bytes())},
                {"path": stderr_path.relative_to(arm).as_posix(), "sha256": job._sha(stderr_path.read_bytes())}]}
            (arm / "physical-audit.json").write_bytes(job._canonical(physical_audit) + b"\n")
            record.finish("completed", counts={}, artifacts=["physical-audit.json"])
            job._write_status(matrix, state="running", completed_arms=[], current_arm="K0")
            registration = {"config_path": str((root / "pilot.json").resolve()),
                "config": {"output_dir": "runs/matrix"}}
            with patch.object(job, "_root", return_value=root), \
                    patch.object(job, "_registration", return_value=registration):
                self.assertEqual(job.status(root / "pilot.json")["verified_terminal_records"][1]["integrity"]["status"],
                    "completed")
            job._verify_worker_log_receipts(arm, physical_audit)
            stderr_path.write_text("changed after physical audit\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "worker log.*hash|worker log.*identity"):
                with patch.object(job, "_root", return_value=root), \
                        patch.object(job, "_registration", return_value=registration):
                    job.status(root / "pilot.json")

    def test_private_child_fixture_reuses_identical_initial_host_seed_per_arm(self):
        try:
            import torch
            from tests import source_pilot_controller_fixture as fixture
            from tests import source_pilot_job_child as child
        except ImportError as error:
            self.skipTest(f"optional native ML dependencies unavailable: {error}")
        from unittest.mock import patch

        torch.set_num_threads(1)
        registration = {"config": {"training": {"optimizer_steps": 2, "schedule": "fixture"},
                "decoding": {"max_new_tokens_including_eos": 128}},
            "config_sha256": "a" * 64, "cohort_sha256": "b" * 64,
            "model_pins": {"weight_sha256": "c" * 64, "artifact_inventory_sha256": "d" * 64}}
        seeds = []
        with patch.object(fixture, "tiny_host", side_effect=lambda seed: seeds.append(seed) or object()), \
                patch.object(fixture, "scaled_policy", return_value={}):
            for arm_index, arm_id in enumerate(("K0", "K1", "K4")):
                authorization = {"arm_id": arm_id, "arm_index": arm_index,
                    "bindings": {"source": {"commit": "e" * 40}}}
                child._factories(registration, authorization, None)
        self.assertEqual(seeds, [501, 501, 501])

    def test_descriptor_digest_is_checked_before_any_authorization_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = root / "issued.json"
            descriptor.write_bytes(_canonical({"schema_version": "wrong"}) + b"\n")
            with self.assertRaisesRegex(ValueError, "digest"):
                read_worker_authorization(root, descriptor, "0" * 64)

    def test_descriptor_must_be_canonical_and_exactly_hashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = root / "issued.json"
            raw = b'{ "schema_version": "dwr-source-pilot-worker-v1" }\n'
            descriptor.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "canonical"):
                read_worker_authorization(root, descriptor, hashlib.sha256(raw).hexdigest())

    def test_clock_recovery_charges_downtime_and_rejects_rollback(self):
        self.assertEqual(_validated_elapsed(100.0, 121.5, 40.0), 61.5)
        with self.assertRaisesRegex(ValueError, "rollback"):
            _validated_elapsed(121.5, 120.0, 61.5)

    def test_historical_clock_accepts_only_advanced_highwater_with_original_anchor(self):
        from dwr.experiments.source_pilot_job import _validate_issued_clock

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            started = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat().replace("+00:00", "Z")
            epoch = datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
            issued = {"started_utc": started, "started_epoch": epoch,
                "elapsed_high_water_seconds": 0.0, "last_utc_epoch": epoch}
            current = {"schema_version": "dwr-source-pilot-elapsed-v1", **issued,
                "elapsed_high_water_seconds": 2.0, "last_utc_epoch": epoch + 2.0}
            (root / "elapsed-budget.json").write_bytes(_canonical(current) + b"\n")
            self.assertEqual(_validate_issued_clock(issued, root), issued)
            current["started_epoch"] += 1
            (root / "elapsed-budget.json").write_bytes(_canonical(current) + b"\n")
            with self.assertRaisesRegex(ValueError, "original anchor"):
                _validate_issued_clock(issued, root)
            current = {"schema_version": "dwr-source-pilot-elapsed-v1", **issued,
                "started_utc": (datetime.now(timezone.utc) - timedelta(seconds=9)).isoformat().replace("+00:00", "Z"),
                "elapsed_high_water_seconds": 2.0, "last_utc_epoch": epoch + 2.0}
            (root / "elapsed-budget.json").write_bytes(_canonical(current) + b"\n")
            with self.assertRaisesRegex(ValueError, "original anchor"):
                _validate_issued_clock(issued, root)
            current = {"schema_version": "dwr-source-pilot-elapsed-v1", **issued,
                "elapsed_high_water_seconds": -1.0, "last_utc_epoch": epoch + 2.0}
            (root / "elapsed-budget.json").write_bytes(_canonical(current) + b"\n")
            with self.assertRaisesRegex(ValueError, "rolled back|high-water"):
                _validate_issued_clock(issued, root)
            current = {"schema_version": "dwr-source-pilot-elapsed-v1", **issued,
                "elapsed_high_water_seconds": 2.0, "last_utc_epoch": epoch - 1.0}
            (root / "elapsed-budget.json").write_bytes(_canonical(current) + b"\n")
            with self.assertRaisesRegex(ValueError, "rolled back|invalid"):
                _validate_issued_clock(issued, root)

    def test_reconstituted_segment_writer_checks_all_initial_identity_bindings(self):
        from dwr.experiments.source_pilot_job import open_segment_record

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "runs" / "matrix" / "arms" / "K1" / "segment-000"
            source_sha, runtime_sha, config_sha, cohort_sha = (
                "c" * 40, "e" * 64, "d" * 64, "f" * 64)
            self._record(root, {"record_kind": "segment", "arm_id": "K1", "segment_index": 0,
                "nonce": "b" * 32, "config_sha256": config_sha, "cohort_sha256": cohort_sha,
                "source_commit": source_sha, "runtime_sha256": runtime_sha}, segment)
            initial_raw = (segment / "manifest.initial.json").read_bytes()
            authorization = {"segment_dir": str(segment), "arm_id": "K1", "segment_index": 0,
                "nonce": "b" * 32, "descriptor": {"segment_initial_manifest_sha256": hashlib.sha256(
                    initial_raw).hexdigest(), "segment_event_prefix": {"byte_length": 0,
                    "sha256": hashlib.sha256(b"").hexdigest()}},
                "registration": {"config_sha256": config_sha, "cohort_sha256": cohort_sha},
                "bindings": {"source": {"commit": source_sha},
                    "runtime": {"receipt_sha256": runtime_sha}}}
            record = open_segment_record(authorization)
            record.append_event("worker_loss_finalized", {"fixture": True})
            self.assertEqual(len((segment / "events.jsonl").read_bytes().splitlines()), 1)

    def test_hard_loss_allows_live_issuing_supervisor_after_worker_exits(self):
        from dwr.experiments.source_pilot_job import _check_hard_loss_process_state
        supervisor = {"pid": 101, "creation_time": "s1", "image": "python.exe"}
        worker = {"pid": 202, "creation_time": "w1", "image": "python.exe"}
        authorization = {"descriptor": {"supervisor": supervisor}}
        receipt = {"worker": worker, "launcher": supervisor}
        _check_hard_loss_process_state(authorization, receipt,
            pid_matches=lambda pid, creation: (pid, creation) == (101, "s1"))
        receipt["launcher"] = {"pid": 303, "creation_time": "l1", "image": "launcher.exe"}
        with self.assertRaisesRegex(ValueError, "distinct worker-owned launcher"):
            _check_hard_loss_process_state(authorization, receipt,
                pid_matches=lambda pid, creation: (pid, creation) == (303, "l1"))

    def test_finalization_events_use_original_seal_reserve(self):
        from dwr.experiments import source_pilot_job as job
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_root = root / "matrix"
            arm_root = matrix_root / "arms" / "K0"
            segment_root = arm_root / "segment-000"
            matrix = self._record(root / "matrix-repo", {"record_kind": "matrix"}, matrix_root)
            arm = self._record(root / "arm-repo", {"record_kind": "arm"}, arm_root)
            segment = self._record(root / "segment-repo", {"record_kind": "segment"}, segment_root)
            job._clock_from_record(matrix_root)
            job._clock_from_record(arm_root)
            config = {"arms": [{"arm_id": "K0", "steps": 0}], "limits": {
                "maximum_physical_updates_per_arm": 514,
                "per_arm_physical_predictions": {"K0": 512},
                "per_arm_teacher_forced_ce_answer_forwards": {"K0": 512},
                "per_arm_greedy_decoder_calls": {"K0": 65536},
                "decoder_calls_per_prediction_bound": 128}}
            now = time.monotonic()
            authorization = {"matrix_root": str(matrix_root), "arm_root": str(arm_root),
                "segment_index": 0, "arm_id": "K0", "config": config,
                "deadlines": {"work_monotonic_deadline": now + 60,
                    "seal_monotonic_deadline": now + 120,
                    "matrix_monotonic_deadline": now + 180,
                    "matrix_watchdog_monotonic_deadline": now + 240}}
            job.guard_worker_event(authorization, segment, "controller_audit", {"schema_version": "fixture"})
            authorization["deadlines"]["work_monotonic_deadline"] = time.monotonic() - 1
            job.guard_worker_event(authorization, segment, "recovery_committed", {"phase": "complete"})
            job.guard_worker_event(authorization, segment, "arm_complete", {})
            with self.assertRaisesRegex(TimeoutError, "work or seal deadline"):
                job.guard_worker_event(authorization, segment, "update_started", {})
            from dwr.experiments.records import _validate_events
            stream = (segment_root / "events.jsonl").read_bytes()
            _validate_events(stream)
            rows = [json.loads(line) for line in stream.splitlines()]
            self.assertEqual([row["kind"] for row in rows],
                ["controller_audit", "recovery_committed", "arm_complete"])

    def test_watchdog_deadline_uses_immutable_arm_anchor_when_status_is_corrupt(self):
        from dwr.experiments import source_pilot_watchdog as watchdog
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "matrix"
            now = time.time()
            matrix_start = datetime.fromtimestamp(now - 5, timezone.utc).isoformat().replace("+00:00", "Z")
            matrix = self._record(Path(temporary) / "matrix-repo", {"record_kind": "matrix"}, root)
            arm_root = root / "arms" / "K1"
            arm_root.parent.mkdir(parents=True, exist_ok=True)
            arm = self._record(Path(temporary) / "arm-repo", {"record_kind": "arm"}, arm_root)
            matrix_root_initial = json.loads((root / "manifest.initial.json").read_text(encoding="utf-8"))
            arm_initial = json.loads((arm_root / "manifest.initial.json").read_text(encoding="utf-8"))
            for initial, started in ((matrix_root_initial, matrix_start), (arm_initial, matrix_start)):
                initial["started_utc"] = started
                raw = _canonical(initial) + b"\n"
                path = root / "manifest.initial.json" if initial is matrix_root_initial else arm_root / "manifest.initial.json"
                path.write_bytes(raw)
            (root / "status.json").write_bytes(b"{broken\n")
            config = {"limits": {"matrix_external_watchdog_seconds": 100,
                "arm_external_watchdog_seconds": 60},
                "arms": [{"arm_id": "K0"}, {"arm_id": "K1"}, {"arm_id": "K4"}]}
            matrix_outer, arm_outer, arm_id = watchdog._deadline(root, config)
            self.assertEqual(arm_id, "K1")
            self.assertAlmostEqual(matrix_outer - time.time(), 95, delta=2)
            self.assertAlmostEqual(arm_outer - time.time(), 55, delta=2)

    def test_readonly_progress_reports_event_hashes_counters_and_original_deadline(self):
        from dwr.experiments import source_pilot_job as job
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm_root = root / "arms" / "K1"
            segment = arm_root / "segment-000"
            now = time.time()
            limits = {"total_wall_seconds": 16200, "matrix_external_watchdog_seconds": 16260,
                "arm_wall_seconds": 5400, "arm_finalization_reserve_seconds": 180,
                "arm_external_watchdog_seconds": 5460}
            clocks = {"matrix": {"started_epoch": now - 3000, "last_utc_epoch": now,
                    "elapsed_high_water_seconds": 3000},
                "arm": {"started_epoch": now - 100, "last_utc_epoch": now,
                    "elapsed_high_water_seconds": 100}}
            original_deadlines = _derive_deadlines(limits, clocks)["epochs"]
            self._record(root / "arm-repo", {"record_kind": "arm",
                "original_deadlines": original_deadlines}, arm_root)
            record = self._record(root / "segment-repo", {"record_kind": "segment"}, segment)
            record.append_event("evaluation_case_started", {"stage": "initial", "example_id": "x"})
            record.append_event("update_started", {"net_step": 1, "example_id": "x"})
            record.append_event("update_completed", {"net_step": 1, "example_id": "x"})
            record.append_event("recovery_committed", {"phase": "zero", "progress": {"optimizer_step": 1}})
            progress = job._artifact_progress(arm_root)
            self.assertEqual(progress["progress"], {"optimizer_step": 1, "phase": "zero"})
            self.assertEqual(progress["physical_event_counters"]["updates_started"], 1)
            self.assertEqual(progress["physical_event_counters"]["evaluation_cases_started"], 1)
            self.assertEqual(progress["effective_original_deadlines"], original_deadlines)
            self.assertEqual(progress["event_streams"][0]["sha256"], job._sha((segment / "events.jsonl").read_bytes()))

    def test_effective_deadlines_keep_seal_reserve_inside_original_matrix_cap(self):
        limits = {"total_wall_seconds": 16200, "matrix_external_watchdog_seconds": 16260,
            "arm_wall_seconds": 5400, "arm_finalization_reserve_seconds": 180,
            "arm_external_watchdog_seconds": 5460}
        clocks = {"matrix": {"started_epoch": 1000, "last_utc_epoch": 1000,
                "elapsed_high_water_seconds": 0},
            "arm": {"started_epoch": 2000, "last_utc_epoch": 2000,
                "elapsed_high_water_seconds": 0}}
        result = _derive_deadlines(limits, clocks)
        self.assertEqual(result["epochs"], {"matrix_deadline_epoch": 17200,
            "matrix_watchdog_epoch": 17260, "effective_arm_deadline_epoch": 7400,
            "arm_work_deadline_epoch": 7220, "arm_watchdog_epoch": 7460})

    def test_resume_reference_is_normalized_to_its_verified_arm_contained_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            arm_root = Path(temporary) / "arms" / "K4"
            arm_root.mkdir(parents=True)
            cursor = arm_root / "recovery.latest.json"
            cursor.write_bytes(b"{}\n")
            value = {"path": "recovery.latest.json", "sha256": hashlib.sha256(cursor.read_bytes()).hexdigest(),
                "source_commit": "a" * 40, "arm_id": "K4"}
            result = _verify_opaque_resume(value, arm_root, "K4", "a" * 40)
            self.assertEqual(Path(result["path"]), cursor.resolve())
            self.assertEqual(Path(result["resolved_path"]), cursor.resolve())
            for raw in (b"not-json\n", b"{ }\n"):
                cursor.write_bytes(raw)
                invalid = {**value, "sha256": hashlib.sha256(raw).hexdigest()}
                with self.assertRaisesRegex(ValueError, "canonical|JSON"):
                    _verify_opaque_resume(invalid, arm_root, "K4", "a" * 40)

    def test_published_receipt_binds_real_python_and_launcher_processes(self):
        from dwr.experiments import source_pilot_job as job
        from dwr.experiments.source_overfit_job import _terminate_owned_process
        from dwr.experiments.source_pilot_job import _process_identity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = root / "matrix"
            arm = matrix / "arms" / "K0"
            segment = arm / "segment-000"
            arm_record = self._record(root / "arm-repo", {"record_kind": "arm"}, arm)
            segment.mkdir()
            python = Path(__file__).resolve().parents[1] / ".venv-ml" / "Scripts" / "python.exe"
            if not python.is_file():
                python = Path(sys.executable)
            code = ("import json,os,time,sys; from dwr.experiments.source_pilot_job import _process_identity; "
                "x=_process_identity(os.getpid()); x['python_executable']=os.path.realpath(sys.executable); "
                "x['base_executable']=os.path.realpath(getattr(sys,'_base_executable',sys.executable)); "
                "print(json.dumps(x),flush=True); time.sleep(30)")
            command = [str(python), "-c", code]
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            child = None
            try:
                line = process.stdout.readline()
                self.assertTrue(line, "native launcher child did not publish its OS identity")
                child = json.loads(line)
                issuer_full = _process_identity(os.getpid())
                issuer = {key: issuer_full[key] for key in ("pid", "creation_time", "image")}
                process_full = _process_identity(process.pid)
                process_ref = {key: process_full[key] for key in ("pid", "creation_time", "image")}
                process_parent = {"pid": process_full["parent_pid"],
                    "creation_time": process_full["parent_creation_time"],
                    "image": process_full["parent_image"]}
                descriptor_sha, nonce = "a" * 64, "b" * 32
                descriptor = {"launcher": issuer}
                receipt = {"schema_version": "dwr-source-pilot-launch-receipt-v1",
                    "descriptor_sha256": descriptor_sha, "nonce": nonce, "command": command,
                    "command_sha256": job._sha(job._canonical(command)), "issuer": issuer,
                    "process": process_ref, "process_parent": process_parent}
                receipt_path = job._launch_receipt_path(segment)
                digest = job._exclusive_json(receipt_path, receipt)
                arm_record.append_event("worker_launch_command_issued", {"segment_index": 0,
                    "descriptor_sha256": descriptor_sha, "command": command,
                    "command_sha256": job._sha(job._canonical(command))})
                arm_record.append_event(job._LAUNCH_EVENT, {"segment_index": 0,
                    "descriptor_sha256": descriptor_sha, "nonce": nonce,
                    "launch_receipt": {"path": "segment-000/worker-launch-receipt.json",
                        "sha256": digest}})
                auth = {"segment_dir": str(segment), "arm_root": str(arm),
                    "arm_id": "K0", "segment_index": 0, "matrix_root": str(matrix),
                    "descriptor_sha256": descriptor_sha, "nonce": nonce, "descriptor": descriptor,
                    "expected_launcher": issuer,
                    "bindings": {"source": {"commit": "c" * 40},
                        "runtime": {"receipt_sha256": "e" * 64}},
                    "registration": {"config_sha256": "d" * 64,
                        "config": {"arms": [{"arm_id": "K0"}]}}}
                parent = {"pid": child["parent_pid"], "creation_time": child["parent_creation_time"],
                    "image": child["parent_image"]}
                observed = job._worker_launcher_for_parent(auth, parent, live=True)
                self.assertIn(observed, (issuer, process_ref))
                unrelated = {**parent, "pid": parent["pid"] + 100000}
                with self.assertRaisesRegex(ValueError, "neither the issuer nor its exact Popen"):
                    job._worker_launcher_for_parent(auth, unrelated, live=False)
                self.assertEqual(process_parent, issuer)
                self.assertEqual(parent, observed)
                worker = {key: child[key] for key in ("pid", "creation_time", "image",
                    "python_executable", "base_executable")}
                live_receipt = {"schema_version": "dwr-source-pilot-worker-receipt-v1",
                    "descriptor_sha256": descriptor_sha, "nonce": nonce,
                    "matrix_root": str(matrix), "arm_root": str(arm), "segment_dir": str(segment),
                    "arm_id": "K0", "segment_index": 0, "source_commit": "c" * 40,
                    "config_sha256": "d" * 64, "runtime_sha256": "e" * 64,
                    "worker": worker, "parent": parent, "launcher": observed,
                    "issued_utc": "2026-10-19T00:00:00Z"}
                self.assertTrue(_verify_live_worker_receipt(live_receipt, auth))
                live_receipt["nonce"] = "f" * 32
                with self.assertRaisesRegex(ValueError, "nonce"):
                    _verify_live_worker_receipt(live_receipt, auth)
            finally:
                if child is not None:
                    try:
                        _terminate_owned_process(child["pid"], child["creation_time"])
                    except (OSError, ValueError):
                        pass
                process.terminate()
                process.wait(timeout=10)
                process.stdout.close()
                process.stderr.close()

    def test_live_receipt_requires_the_issued_nonce_and_actual_process_identity(self):
        self.test_published_receipt_binds_real_python_and_launcher_processes()

    def test_windows_venv_launcher_receipt_binds_exact_popen_parent_chain(self):
        self.test_published_receipt_binds_real_python_and_launcher_processes()

    def test_started_prediction_cap_is_reserved_before_actual_runrecord_append(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self._record(root)
            arm_root = root / "runs"
            matrix_root = root / "matrix"
            matrix_root.mkdir()
            now = time.time()
            started = datetime.fromtimestamp(now - 2, timezone.utc).isoformat().replace("+00:00", "Z")
            clock = {"schema_version": "dwr-source-pilot-elapsed-v1", "started_utc": started,
                "started_epoch": now - 2, "elapsed_high_water_seconds": 2.0, "last_utc_epoch": now}
            (matrix_root / "elapsed-budget.json").write_bytes(_canonical(clock) + b"\n")
            (arm_root / "elapsed-budget.json").write_bytes(_canonical(clock) + b"\n")
            auth = {"matrix_root": str(matrix_root), "arm_root": str(arm_root),
                "segment_index": 0, "arm_id": "K0",
                "config": {"arms": [{"arm_id": "K0", "steps": 0}], "limits": {"maximum_physical_updates_per_arm": 514,
                    "per_arm_physical_predictions": {"K0": 1},
                    "per_arm_teacher_forced_ce_answer_forwards": {"K0": 1},
                    "per_arm_greedy_decoder_calls": {"K0": 128},
                    "decoder_calls_per_prediction_bound": 128}},
                "deadlines": {"work_monotonic_deadline": time.monotonic() + 10,
                    "seal_monotonic_deadline": time.monotonic() + 10,
                    "matrix_monotonic_deadline": time.monotonic() + 10,
                    "matrix_watchdog_monotonic_deadline": time.monotonic() + 10}}
            first = {"stage": "initial", "example_id": "x", "attempt": 0, "include_ce": True}
            guard_worker_event(auth, record, "evaluation_case_started", first)
            before = (root / "runs" / "segment-000" / "events.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "prediction cap"):
                guard_worker_event(auth, record, "evaluation_case_started",
                    {"stage": "initial", "example_id": "y", "attempt": 0, "include_ce": True})
            after = (root / "runs" / "segment-000" / "events.jsonl").read_bytes()
            self.assertEqual(after, before)
            self.assertEqual(_physical_ledger(auth)["prediction_attempts_started"], 1)

    def test_update_event_tuple_sequences_become_json_arrays_without_mutating_kernel_payload(self):
        from dwr.tasks.run_support import strict_json_loads

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self._record(root)
            matrix_root = root / "matrix"
            arm_root = root / "runs"
            matrix_root.mkdir()
            now = time.time()
            started = datetime.fromtimestamp(now - 2, timezone.utc).isoformat().replace("+00:00", "Z")
            clock = {"schema_version": "dwr-source-pilot-elapsed-v1", "started_utc": started,
                "started_epoch": now - 2, "elapsed_high_water_seconds": 2.0, "last_utc_epoch": now}
            (matrix_root / "elapsed-budget.json").write_bytes(_canonical(clock) + b"\n")
            (arm_root / "elapsed-budget.json").write_bytes(_canonical(clock) + b"\n")
            auth = {"matrix_root": str(matrix_root), "arm_root": str(arm_root), "segment_index": 0,
                "arm_id": "K0", "config": {"arms": [{"arm_id": "K0", "steps": 0}], "limits": {
                    "maximum_physical_updates_per_arm": 514, "per_arm_physical_predictions": {"K0": 1},
                    "per_arm_teacher_forced_ce_answer_forwards": {"K0": 1},
                    "per_arm_greedy_decoder_calls": {"K0": 128}, "decoder_calls_per_prediction_bound": 128}},
                "deadlines": {"work_monotonic_deadline": time.monotonic() + 10,
                    "seal_monotonic_deadline": time.monotonic() + 10,
                    "matrix_monotonic_deadline": time.monotonic() + 10,
                    "matrix_watchdog_monotonic_deadline": time.monotonic() + 10}}
            payload = {"net_step": 1, "physical_index": 1, "example_id": "fixture",
                "batch_sha256": "a" * 64, "noise_sha256": "b" * 64,
                "state_norms": (1.25, (2.5, 5.0)), "state_gradient_norms": ()}
            original = copy.deepcopy(payload)
            guard_worker_event(auth, record, "update_completed", payload)
            self.assertEqual(payload, original)
            rows = (arm_root / "segment-000" / "events.jsonl").read_bytes().splitlines()
            self.assertEqual(strict_json_loads(rows[-1].decode("utf-8"))["payload"]["state_norms"], [1.25, [2.5, 5.0]])

            before = (arm_root / "segment-000" / "events.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "only JSON values"):
                guard_worker_event(auth, record, "update_completed", {**payload, "state_norms": (object(),)})
            with self.assertRaisesRegex(ValueError, "non-finite"):
                guard_worker_event(auth, record, "update_completed", {**payload, "state_norms": (float("nan"),)})
            self.assertEqual((arm_root / "segment-000" / "events.jsonl").read_bytes(), before)

    def test_matrix_and_worker_lock_bytes_are_independent_but_each_is_exclusive(self):
        from dwr.experiments.source_overfit_job import _ByteLock

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pilot.lock"
            matrix = _ByteLock(path, 0)
            worker = _ByteLock(path, 1)
            try:
                with self.assertRaisesRegex(RuntimeError, "byte 0"):
                    _ByteLock(path, 0)
                with self.assertRaisesRegex(RuntimeError, "byte 1"):
                    _ByteLock(path, 1)
            finally:
                worker.close()
                matrix.close()
            with _ByteLock(path, 0):
                with _ByteLock(path, 1):
                    pass

    def test_status_rejects_corrupt_or_out_of_order_arm_progression(self):
        from dwr.experiments.source_pilot_job import _write_status

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "matrix"
            root.mkdir()
            _write_status(root, state="running", completed_arms=["K0"], current_arm="K1")
            self.assertEqual(_read_status(root)["completed_arms"], ["K0"])
            _write_status(root, completed_arms=["K1"])
            with self.assertRaisesRegex(ValueError, "progression"):
                _read_status(root)
            (root / "status.json").write_bytes(b'{"state":"running"}\n')
            with self.assertRaisesRegex(ValueError, "schema or matrix binding"):
                _read_status(root)

    def test_unsealed_failed_seal_with_known_fault_cannot_autoresume(self):
        from unittest.mock import patch
        from dwr.experiments.source_pilot_job import _run_arm, _worker_log_paths

        with tempfile.TemporaryDirectory() as temporary:
            matrix = Path(temporary) / "matrix"
            arm = matrix / "arms" / "K0"
            segment = arm / "segment-000"
            segment.mkdir(parents=True)
            receipt = {"worker": {"pid": 99999999, "creation_time": "old", "image": "python.exe"}}
            (segment / "worker-receipt.json").write_bytes(_canonical(receipt) + b"\n")
            _, stderr = _worker_log_paths(arm, segment)
            stderr.parent.mkdir(parents=True, exist_ok=True)
            stderr.write_text(
                "FloatingPointError: nonfinite loss; record finish budget check failed\n", encoding="utf-8")
            registration = {"config": {"arms": [{"arm_id": "K0"}], "limits": {
                "maximum_workers_per_arm": 3, "total_wall_seconds": 16200,
                "matrix_external_watchdog_seconds": 16260, "arm_wall_seconds": 5400,
                "arm_finalization_reserve_seconds": 180, "arm_external_watchdog_seconds": 5460}}}
            clock = {"started_epoch": time.time(), "last_utc_epoch": time.time(),
                "elapsed_high_water_seconds": 0.0}
            with patch("dwr.experiments.source_pilot_job._issue_worker_descriptor") as issue, \
                    patch("dwr.experiments.source_pilot_job._launch_worker") as launch:
                with self.assertRaisesRegex(RuntimeError, "known numerical, OOM, CUDA or configuration fault"):
                    _run_arm(registration, {"commit": "a" * 40}, {}, matrix, arm, None, clock, clock, 0)
                issue.assert_not_called()
                launch.assert_not_called()
                stderr.write_text(
                    "Traceback (most recent call last):\nRuntimeError: controller seal failed\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "known numerical, OOM, CUDA or configuration fault"):
                    _run_arm(registration, {"commit": "a" * 40}, {}, matrix, arm, None, clock, clock, 0)
                issue.assert_not_called()
                launch.assert_not_called()

    def test_sealed_keyboard_interrupt_segment_uses_narrow_interruption_recovery(self):
        from unittest.mock import patch
        from dwr.experiments import source_pilot_job as job

        class ReachedNextSegment(Exception):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = root / "runs" / "matrix"
            arm = matrix / "arms" / "K1"
            segment = arm / "segment-000"
            self._record(root, {"record_kind": "matrix"}, matrix)
            from dwr.experiments.records import RunRecord
            config_path = root / "pilot.json"
            config_bytes = config_path.read_bytes()
            def make_record(path, kind):
                path.parent.mkdir(parents=True, exist_ok=True)
                return RunRecord.create(path, command=["fixture"], config_path=config_path,
                    expected_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    identities={"config": {"name": "fixture", "revision": "1", "sha256": "a" * 64}},
                    rng_streams={"fixture": 1}, settings={"record_kind": kind},
                    budgets={"wall_seconds": 10, "memory_mib": 10, "attempts": 1}, source_root=root)
            arm_record = make_record(arm, "arm")
            segment_record = make_record(segment, "segment")
            historical_issuer = {"pid": 987654321, "creation_time": "old", "image": "python.exe"}
            descriptor = {"nonce": "b" * 32, "segment_index": 0, "launcher": historical_issuer}
            (segment / "worker-descriptor.json").write_bytes(_canonical(descriptor) + b"\n")
            (segment / "worker-receipt.json").write_bytes(_canonical({
                "descriptor_sha256": hashlib.sha256(_canonical(descriptor) + b"\n").hexdigest(),
                "launcher": historical_issuer, "worker": {
                    "pid": 987654321, "creation_time": "old", "image": "python.exe"}}) + b"\n")
            _, stderr = job._worker_log_paths(arm, segment)
            stderr.parent.mkdir(parents=True, exist_ok=True)
            stderr.write_text("Traceback (most recent call last):\nKeyboardInterrupt: fixture interruption\n",
                encoding="utf-8")
            from dwr.experiments.source_pilot import _segment_artifacts
            segment_record.finish("interrupted", counts={}, artifacts=_segment_artifacts(segment),
                error="KeyboardInterrupt: fixture interruption")
            clock = {"started_epoch": time.time() - 1, "last_utc_epoch": time.time(),
                "elapsed_high_water_seconds": 0.0}
            registration = {"config": {"arms": [{"arm_id": "K1", "steps": 1}],
                "training": {"optimizer_steps": 1}, "limits": {
                    "maximum_workers_per_arm": 3, "total_wall_seconds": 16200,
                    "matrix_external_watchdog_seconds": 16260, "arm_wall_seconds": 5400,
                    "arm_finalization_reserve_seconds": 180, "arm_external_watchdog_seconds": 5460}}}
            with patch("dwr.experiments.source_pilot_job._issue_worker_descriptor",
                       side_effect=ReachedNextSegment), patch(
                       "dwr.experiments.source_pilot_job._known_worker_fault",
                       wraps=job._known_worker_fault) as fault_scan:
                with self.assertRaises(ReachedNextSegment):
                    job._run_arm(registration, {"commit": "c" * 40}, {}, matrix, arm,
                        arm_record, clock, clock, 0)
            self.assertTrue(any(call.kwargs.get("allow_keyboard_interrupt") is True
                for call in fault_scan.call_args_list))
            self.assertTrue(any(row["kind"] == "worker_interruption_recovered"
                for row in job._event_rows(arm / "events.jsonl")))

    def test_live_supervisor_finalizes_receipted_hard_worker_loss_and_recovers(self):
        from unittest.mock import Mock, patch
        from dwr.experiments import source_pilot_job as job
        from dwr.experiments.records import RunRecord

        class ReachedNextSegment(Exception):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = root / "runs" / "matrix"
            arm = matrix / "arms" / "K1"
            self._record(root, {"record_kind": "matrix"}, matrix)
            config_path = root / "pilot.json"
            config_bytes = config_path.read_bytes()
            def make_record(path, kind):
                path.parent.mkdir(parents=True, exist_ok=True)
                return RunRecord.create(path, command=["fixture"], config_path=config_path,
                    expected_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    identities={"config": {"name": "fixture", "revision": "1", "sha256": "a" * 64}},
                    rng_streams={"fixture": 1}, settings={"record_kind": kind},
                    budgets={"wall_seconds": 10, "memory_mib": 10, "attempts": 1}, source_root=root)
            arm_record = make_record(arm, "arm")
            issuer = {"pid": 101, "creation_time": "issuer", "image": "python.exe"}
            worker = {"pid": 202, "creation_time": "worker", "image": "python.exe"}
            segment_records = {}
            calls = {"issue": 0}

            def issue(_root, _registration, _physical, _runtime, _matrix, _arm, segment_root, **kwargs):
                calls["issue"] += 1
                if kwargs["segment_index"] != 0:
                    raise ReachedNextSegment()
                segment_record = make_record(segment_root, "segment")
                descriptor = {"nonce": "d" * 32, "arm_id": "K1", "segment_index": 0,
                    "launcher": issuer}
                descriptor_path = segment_root / "worker-descriptor.json"
                descriptor_path.write_bytes(_canonical(descriptor) + b"\n")
                descriptor_sha = job._sha(descriptor_path.read_bytes())
                segment_records[segment_root] = segment_record
                return descriptor_path, descriptor_sha, descriptor

            def launch(_descriptor_path, descriptor_sha, _matrix, arm_root, segment_root,
                       *, command=None, arm_record=None):
                _, stderr = job._worker_log_paths(arm_root, segment_root)
                stdout, _ = job._worker_log_paths(arm_root, segment_root)
                stdout.parent.mkdir(parents=True, exist_ok=True)
                stdout.write_bytes(b"")
                stderr.write_bytes(b"")
                receipt = {"descriptor_sha256": descriptor_sha, "worker": worker,
                    "launcher": issuer, "source_commit": "c" * 40}
                (segment_root / "worker-receipt.json").write_bytes(_canonical(receipt) + b"\n")
                return object(), stdout, stderr

            def finalize(_arm_root, segment_root, receipt, _registration, source_commit):
                record = segment_records[segment_root]
                loss = {"descriptor_sha256": receipt["descriptor_sha256"],
                    "worker_receipt_sha256": job._sha((segment_root / "worker-receipt.json").read_bytes()),
                    "worker_identity": worker, "launcher_identity": issuer, "source_commit": source_commit,
                    "runtime_receipt_sha256": "e" * 64, "worker_lock_released": True,
                    "postloss_observation": {"available": False,
                        "reason": "process_exited_before_observation"}}
                record.append_event(job._LOSS_EVENT, loss)
                record.finish("interrupted", counts={}, artifacts=[],
                    error="supervisor-observed hard worker loss; post-loss process observation unavailable")

            finalizer = Mock(side_effect=finalize)
            clock = {"started_epoch": time.time() - 1, "last_utc_epoch": time.time(),
                "elapsed_high_water_seconds": 0.0}
            registration = {"config": {"arms": [{"arm_id": "K1", "steps": 1}],
                "training": {"optimizer_steps": 1}, "limits": {
                    "maximum_workers_per_arm": 3, "total_wall_seconds": 16200,
                    "matrix_external_watchdog_seconds": 16260, "arm_wall_seconds": 5400,
                    "arm_finalization_reserve_seconds": 180, "arm_external_watchdog_seconds": 5460}}}
            with patch("dwr.experiments.source_pilot_job._issue_worker_descriptor", side_effect=issue), \
                    patch("dwr.experiments.source_pilot_job._launch_worker", side_effect=launch), \
                    patch("dwr.experiments.source_pilot_job._wait_worker", return_value=137), \
                    patch("dwr.experiments.source_pilot_job._finalize_worker_loss", finalizer):
                with self.assertRaises(ReachedNextSegment):
                    job._run_arm(registration, {"commit": "c" * 40}, {}, matrix, arm,
                        arm_record, clock, clock, 0)
            self.assertEqual(finalizer.call_count, 1)
            self.assertEqual(calls["issue"], 2)
            self.assertTrue(any(row["kind"] == "worker_interruption_recovered"
                for row in job._event_rows(arm / "events.jsonl")))

    def test_real_hard_loss_finalizer_seals_and_verifies_zero_access_prefix(self):
        from dwr.experiments import source_pilot_job as job
        from dwr.experiments.records import RunRecord, verify_artifacts
        from dwr.experiments.source_pilot_registration import load_pilot_registration
        from dwr.experiments.source_overfit_job import _pid_matches, _terminate_owned_process

        repository = job._root()
        registration = load_pilot_registration(repository,
            repository / "configs/experiments/source_validation_pilot_v1.json")
        config = registration["config"]
        config_sha = registration["config_sha256"]
        cohort_sha = registration["cohort_sha256"]
        source_commit = "d4d3a6664f245869604a01be91dfdd67175417d1"
        physical = {"commit": source_commit, "fixture": "no-model-loss-smoke"}
        runtime = {"python": sys.version, "executable": sys.executable}
        runtime_sha = job._sha(_canonical(runtime))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = root / "runs" / "matrix"
            arm = matrix / "arms" / "K1"
            segment = arm / "segment-000"

            def create_record(path, settings):
                path.parent.mkdir(parents=True, exist_ok=True)
                return RunRecord.create(path, command=["no-model-loss-smoke"],
                    config_path=registration["config_path"], expected_config_sha256=config_sha,
                    identities={"fixture": {"name": "loss-smoke", "revision": "1", "sha256": "a" * 64}},
                    rng_streams={"fixture": 1}, settings=settings,
                    budgets={"wall_seconds": 5400, "memory_mib": 1024, "attempts": 3},
                    source_root=repository)

            create_record(matrix, {"record_kind": "matrix"})
            matrix_clock = job._clock_from_record(matrix)
            create_record(arm, {"record_kind": "arm"})
            arm_clock = job._clock_from_record(arm)
            deadlines = job._derive_deadlines(config["limits"],
                {"matrix": matrix_clock, "arm": arm_clock})["epochs"]
            nonce = "e" * 32
            segment_settings = job._record_settings(registration, physical, runtime,
                record_kind="segment", original_deadlines=deadlines,
                arm_id="K1", arm_index=1, segment_index=0, nonce=nonce)
            create_record(segment, segment_settings)
            initial = {"matrix": job._sha((matrix / "manifest.initial.json").read_bytes()),
                "arm": job._sha((arm / "manifest.initial.json").read_bytes()),
                "segment": job._sha((segment / "manifest.initial.json").read_bytes())}
            issuer_full = job._process_identity(os.getpid())
            issuer = {key: issuer_full[key] for key in ("pid", "creation_time", "image")}
            descriptor = {"schema_version": "dwr-source-validation-pilot-worker-descriptor-v1",
                "matrix_root": str(matrix.resolve()), "arm_root": str(arm.resolve()),
                "segment_dir": str(segment.resolve()), "arm_id": "K1", "arm_index": 1,
                "segment_index": 0, "nonce": nonce, "config_sha256": config_sha,
                "cohort_sha256": cohort_sha, "source_commit": source_commit,
                "physical_source_receipt": physical,
                "physical_source_receipt_sha256": job._sha(_canonical(physical)),
                "runtime_receipt": runtime, "runtime_receipt_sha256": runtime_sha,
                "matrix_initial_manifest_sha256": initial["matrix"],
                "arm_initial_manifest_sha256": initial["arm"],
                "matrix_event_prefix": job._event_prefix(matrix),
                "arm_event_prefix": job._event_prefix(arm),
                "segment_initial_manifest_sha256": initial["segment"],
                "segment_event_prefix": job._event_prefix(segment),
                "matrix_clock": matrix_clock, "arm_clock": arm_clock,
                "deadlines": deadlines, "supervisor": issuer, "launcher": issuer,
                "resume_descriptor": None}
            descriptor_path = segment / "worker-descriptor.json"
            descriptor_sha = job._exclusive_json(descriptor_path, descriptor)
            arm_record = job._reopen_record(arm, {})
            arm_record.append_event("worker_descriptor_issued", {"segment_index": 0,
                "descriptor_sha256": descriptor_sha, "nonce": nonce,
                "descriptor_path": "arms/K1/segment-000/worker-descriptor.json"})
            reg_projection = {"config": config, "config_sha256": config_sha, "cohort_sha256": cohort_sha}
            reg_path = root / "registration.json"
            reg_path.write_bytes(_canonical(reg_projection) + b"\n")
            child_code = (
                "import json,sys,time; from pathlib import Path; "
                "from dwr.experiments import source_pilot_job as j; "
                "dp=Path(sys.argv[1]); sha=sys.argv[2]; rp=Path(sys.argv[3]); "
                "d=j._read_descriptor_file(dp,sha)[1]; reg=j.strict_json_loads(rp.read_text(encoding='utf-8')); "
                "a={'descriptor':d,'descriptor_sha256':sha,'descriptor_path':str(dp),"
                "'nonce':d['nonce'],'matrix_root':d['matrix_root'],'arm_root':d['arm_root'],"
                "'segment_dir':d['segment_dir'],'arm_id':d['arm_id'],'segment_index':d['segment_index'],"
                "'expected_launcher':d['launcher'],'registration':reg,"
                "'bindings':{'source':{'commit':d['source_commit']},"
                "'runtime':{'receipt_sha256':d['runtime_receipt_sha256']}}}; "
                "j.publish_worker_receipt(a); rec=j.open_segment_record(a); "
                "rec.append_event('worker_receipt_published',{'descriptor_sha256':sha,'nonce':d['nonce']}); "
                "print('receipt-published',flush=True); time.sleep(120)"
            )
            command = [sys.executable, "-c", child_code, str(descriptor_path), descriptor_sha, str(reg_path)]
            command_sha = job._sha(_canonical(command))
            arm_record.append_event("worker_launch_command_issued", {"segment_index": 0,
                "descriptor_sha256": descriptor_sha, "command": command, "command_sha256": command_sha})
            process, stdout_path, stderr_path = job._launch_worker(descriptor_path, descriptor_sha,
                matrix, arm, segment, command=command, arm_record=arm_record)
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if any(row["kind"] == "worker_receipt_published"
                           for row in job._event_rows(segment / "events.jsonl")):
                        break
                    if process.poll() is not None:
                        self.fail(f"no-model child exited before publishing receipt: {stderr_path.read_text(encoding='utf-8')}")
                    time.sleep(0.025)
                else:
                    self.fail("no-model child did not durably publish its receipt")
                receipt = job._read_json_path(segment / "worker-receipt.json")
                worker = receipt["worker"]
                self.assertFalse(any(row["kind"] == "source_examples_access_authorized"
                    for row in job._event_rows(segment / "events.jsonl")))
                self.assertTrue(_pid_matches(worker["pid"], worker["creation_time"]))
                _terminate_owned_process(worker["pid"], worker["creation_time"])
                process.wait(timeout=10)
                job._finalize_worker_loss(arm, segment, receipt, reg_projection, source_commit)
            finally:
                if process.poll() is None:
                    _terminate_owned_process(process.pid, job._process_identity(process.pid)["creation_time"])
                    process.wait(timeout=10)
            self.assertEqual(verify_artifacts(segment)["status"], "interrupted")
            self.assertEqual(job.verify_worker_loss_record(arm, segment, reg_projection,
                source_commit=source_commit)["postloss_observation"],
                {"available": False, "reason": "process_exited_before_observation"})
            self.assertFalse(any(row["kind"] == "source_examples_access_authorized"
                for row in job._event_rows(segment / "events.jsonl")))

    def test_detached_native_three_arm_matrix_resumes_two_k4_boundaries(self):
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name == "torch":
                self.skipTest("optional native ML dependency torch is unavailable")
            raise
        from tests import source_pilot_controller_fixture as fixture
        from dwr.experiments import source_pilot_job as job
        from dwr.experiments.source_pilot_registration import load_pilot_registration

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        repo = job._root()
        base = load_pilot_registration(repo, repo / "configs/experiments/source_validation_pilot_v1.json")
        registration = copy.deepcopy(base)
        config = copy.deepcopy(fixture.kernel_registration("K0", steps=2)["config"])
        config.update({"study_type": base["config"]["study_type"],
            "config_version": base["config"]["config_version"],
            "limits": copy.deepcopy(base["config"]["limits"]),
            "data": copy.deepcopy(base["config"]["data"])})
        config["evaluation"].update({key: copy.deepcopy(base["config"]["evaluation"][key])
            for key in ("learning_gate", "workspace_screen")})
        config["arms"] = [{"arm_id": "K0", "steps": 0}, {"arm_id": "K1", "steps": 1},
            {"arm_id": "K4", "steps": 4}]
        config["decoding"]["max_new_tokens_including_eos"] = 128
        config["data"]["source_train_records"] = 5
        config["data"]["validation_examples"] = 5
        registration["config"] = config
        validation = tuple(sorted(fixture.examples(5, "source_validation", start=0), key=lambda row: row.example_id))
        training = tuple(sorted(fixture.examples(5, "source_train", start=5), key=lambda row: row.example_id))
        selected = [{"example_id": row.example_id} for row in validation]
        registration["cohort"] = {"selected": selected}
        registration["cohort_sha256"] = hashlib.sha256(_canonical(selected)).hexdigest()
        registration["token_rows"] = ([{"example_id": row.example_id, "split": "source_validation"}
            for row in validation] + [{"example_id": row.example_id, "split": "source_train"} for row in training])
        matrix_parent = repo / ".superpowers/sdd/source-validation-pilot-20261017"
        matrix_parent.mkdir(parents=True, exist_ok=True)
        matrix_root = matrix_parent / f"task3-job-native-{__import__('uuid').uuid4().hex}"
        control_dir = matrix_parent / f"{matrix_root.name}-control"
        control_dir.mkdir()
        config["output_dir"] = matrix_root.relative_to(repo).as_posix()
        config_bytes = _canonical(config) + b"\n"
        registration["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        config_path = repo / "configs/experiments" / f"task3-job-native-{matrix_root.name}.json"
        registration["config_path"] = str(config_path.resolve())
        registration.pop("config_raw", None)
        registration.pop("cohort_raw", None)
        registration["token_rows"] = list(registration["token_rows"])
        config_path.write_bytes(config_bytes)
        registration_path = matrix_parent / f"{matrix_root.name}-registration.json"
        registration_path.write_bytes(_canonical(registration) + b"\n")
        test_dir = repo / ".superpowers/sdd/source-validation-pilot-20261017"
        receipt_path = test_dir / f"{matrix_root.name}.receipt.json"
        command = [sys.executable, "-m", "tests.source_pilot_job_supervisor_child",
            "--config", str(config_path), "--registration", str(registration_path)]
        started = time.monotonic()
        process_records = {}

        def launch_supervisor(attach: bool, index: int):
            args = command + (["--attach"] if attach else [])
            stdout = test_dir / f"{matrix_root.name}-supervisor-{index}.stdout.log"
            stderr = test_dir / f"{matrix_root.name}-supervisor-{index}.stderr.log"
            with stdout.open("wb") as out, stderr.open("wb") as err:
                process = subprocess.Popen(args, cwd=repo, stdin=subprocess.DEVNULL,
                    stdout=out, stderr=err,
                    creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) |
                        getattr(subprocess, "CREATE_NO_WINDOW", 0)))
            process_records[f"supervisor_{index}"] = {"pid": process.pid, "command": args,
                "stdout": str(stdout), "stderr": str(stderr)}
            return process

        def wait_until(predicate, description: str, process=None):
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if predicate():
                    return
                if process is not None and process.poll() is not None:
                    details = "\n".join(Path(item["stderr"]).read_text(encoding="utf-8", errors="replace")
                        for item in process_records.values() if Path(item["stderr"]).is_file())
                    raise RuntimeError(f"supervisor exited with {process.returncode} before {description}\n{details}")
                time.sleep(0.05)
            raise TimeoutError(f"bounded native recovery test timed out waiting for {description}")

        def verify_live_identity(identity: dict):
            from dwr.experiments.source_overfit_job import _pid_matches, _process_image
            self.assertTrue(_pid_matches(identity["pid"], identity["creation_time"]))
            self.assertEqual(_process_image(identity["pid"]), identity["image"])

        def actual_supervisor_reference(process, *, wait_for_publication=False):
            reference = job._read_status(matrix_root)["supervisor"]
            from dwr.experiments.source_overfit_job import _pid_matches
            if not _pid_matches(reference["pid"], reference["creation_time"]):
                if wait_for_publication:
                    return None
                self.fail("published supervisor identity is no longer live")
            verify_live_identity(reference)
            if reference["pid"] != process.pid:
                actual = job._process_identity(reference["pid"])
                launcher = job._process_identity(process.pid)
                exact_chain = actual["parent_pid"] == process.pid and \
                    actual["parent_creation_time"] == launcher["creation_time"] and \
                    actual["parent_image"] == launcher["image"]
                if not exact_chain:
                    if wait_for_publication:
                        return None
                    self.fail("published supervisor is not the exact launched Popen child")
            return reference

        def terminate_verified(identity: dict):
            from dwr.experiments.source_overfit_job import _pid_matches, _process_image, _terminate_owned_process
            verify_live_identity(identity)
            if _pid_matches(identity["pid"], identity["creation_time"]) and \
                    _process_image(identity["pid"]) == identity["image"]:
                _terminate_owned_process(identity["pid"], identity["creation_time"])

        try:
            first = launch_supervisor(False, 1)
            k0_ready = control_dir / "K0-worker-ready.json"
            k0_release = control_dir / "K0-worker-release.json"
            wait_until(lambda: k0_ready.is_file(), "K0 committed-step worker hold", first)
            original_matrix_manifest = job._sha((matrix_root / "manifest.initial.json").read_bytes())
            k0_segment = matrix_root / "arms/K0/segment-000"
            original_k0_manifest = job._sha((matrix_root / "arms/K0/manifest.initial.json").read_bytes())
            k0_receipt = job._read_json_path(k0_segment / "worker-receipt.json")
            k0_worker = k0_receipt["worker"]
            verify_live_identity(k0_worker)
            self.assertEqual(job._read_json_path(k0_ready)["worker"]["pid"], k0_worker["pid"])
            first_reference = actual_supervisor_reference(first)
            terminate_verified(first_reference)
            first.wait(timeout=10)
            process_records["supervisor_1"]["exit_code"] = first.returncode
            self.assertNotEqual(first.returncode, 0)
            from dwr.experiments.source_overfit_job import _pid_matches
            self.assertFalse(_pid_matches(first_reference["pid"], first_reference["creation_time"]))
            verify_live_identity(k0_worker)

            second = launch_supervisor(True, 2)
            def attached_and_status_written_twice():
                events = job._event_rows(matrix_root / "events.jsonl")
                second_reference = actual_supervisor_reference(second, wait_for_publication=True)
                if second_reference is None:
                    return False
                attached = any(row["kind"] == "matrix_supervisor_attached" and
                    row["payload"].get("supervisor") == second_reference for row in events)
                if not attached:
                    return False
                verify_live_identity(k0_worker)
                status = matrix_root / "status.json"
                modified = status.stat().st_mtime_ns
                seen = getattr(attached_and_status_written_twice, "seen", set())
                seen.add(modified)
                attached_and_status_written_twice.seen = seen
                return len(seen) >= 3
            wait_until(attached_and_status_written_twice,
                "surviving K0 worker reattachment and repeated status writes", second)
            k0_release.write_bytes(_canonical({"release": True}) + b"\n")

            k1_ready = control_dir / "K1-worker-ready.json"
            wait_until(lambda: k1_ready.is_file(), "K1 committed-step worker hold", second)
            k1_segment = matrix_root / "arms/K1/segment-000"
            original_k1_manifest = job._sha((matrix_root / "arms/K1/manifest.initial.json").read_bytes())
            k1_receipt = job._read_json_path(k1_segment / "worker-receipt.json")
            k1_worker = k1_receipt["worker"]
            self.assertEqual(job._read_json_path(k1_ready)["worker"]["pid"], k1_worker["pid"])
            self.assertTrue(any(row["kind"] == "recovery_committed" and
                row["payload"].get("progress", {}).get("optimizer_step") == 1
                for row in job._event_rows(k1_segment / "events.jsonl")))
            verify_live_identity(k1_worker)
            from dwr.experiments.source_overfit_job import _pid_matches
            self.assertTrue(_pid_matches(k1_worker["pid"], k1_worker["creation_time"]))
            terminate_verified(k1_worker)
            wait_until(lambda: not _pid_matches(k1_worker["pid"], k1_worker["creation_time"]),
                "verified K1 worker death and OS lock release")

            result_code = second.wait(timeout=900)
            process_records["supervisor_2"]["exit_code"] = result_code
            duration = time.monotonic() - started
            logs = {name: {"stdout_sha256": job._sha(Path(item["stdout"]).read_bytes()),
                "stderr_sha256": job._sha(Path(item["stderr"]).read_bytes())}
                for name, item in process_records.items()}
            receipt = {"command": command, "timeout_seconds": 900, "duration_seconds": duration,
                "supervisors": process_records, "logs": logs, "k0_worker": k0_worker,
                "k1_worker": k1_worker, "exit_code": result_code,
                "outcome": job._read_status(matrix_root).get("state")}
            receipt_path.write_bytes(_canonical(receipt) + b"\n")
            self.assertEqual(result_code, 0, "\n".join(Path(item["stderr"]).read_text(encoding="utf-8", errors="replace")
                for item in process_records.values()))
            self.assertEqual(job._sha((matrix_root / "manifest.initial.json").read_bytes()), original_matrix_manifest)
            self.assertEqual(job._sha((matrix_root / "arms/K0/manifest.initial.json").read_bytes()), original_k0_manifest)
            self.assertEqual(job._sha((matrix_root / "arms/K1/manifest.initial.json").read_bytes()), original_k1_manifest)
            from dwr.experiments.records import verify_artifacts
            self.assertEqual(verify_artifacts(matrix_root)["status"], "completed")
            audits = {arm: job._read_json_path(matrix_root / "arms" / arm / "arm-audit.json")
                for arm in ("K0", "K1", "K4")}
            decision = job._read_json_path(matrix_root / "matrix-decision.json")
            self.assertEqual([len(job._segments(matrix_root / "arms" / arm))
                for arm in ("K0", "K1", "K4")], [1, 2, 3])
            losses = [row["payload"] for row in job._event_rows(
                matrix_root / "arms/K1/segment-000/events.jsonl") if row["kind"] == "worker_loss_finalized"]
            self.assertEqual(len(losses), 1)
            self.assertTrue(losses[0]["worker_lock_released"])
            self.assertEqual(losses[0]["postloss_observation"],
                {"available": False, "reason": "process_exited_before_observation"})
            self.assertEqual(audits["K4"]["progress"]["optimizer_step"], 2)
            self.assertEqual(len(audits["K4"]["pairing"]), 2)
            self.assertEqual(decision["status"], "NO_GO_SOURCE_VALIDATION_PILOT")
            self.assertEqual(audits["K0"]["pairing"], audits["K1"]["pairing"])
            self.assertEqual(audits["K1"]["pairing"], audits["K4"]["pairing"])
            self.assertTrue(receipt["duration_seconds"] < 900)
        finally:
            for arm_id in ("K0", "K1"):
                (control_dir / f"{arm_id}-worker-release.json").write_bytes(_canonical({"release": True}) + b"\n")
            torch.set_num_threads(previous_threads)
            config_path.unlink(missing_ok=True)
            registration_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
