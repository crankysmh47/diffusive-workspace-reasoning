"""Durable store contract tests for recoverable source-overfit runs."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_recovery_spec = importlib.util.spec_from_file_location(
    "dwr_training_recovery_test_module", Path(__file__).resolve().parents[1] / "dwr" / "training" / "recovery.py")
_recovery_module = importlib.util.module_from_spec(_recovery_spec)
_recovery_spec.loader.exec_module(_recovery_module)
RecoveryStore, RecoveryError = _recovery_module.RecoveryStore, _recovery_module.RecoveryError

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch": raise
    torch = None

IDENTITIES = {key: hashlib.sha256(key.encode()).hexdigest() for key in
              ("source", "config", "data", "model", "tokenizer", "schedule")}
HASH_RECIPE = {"name": "dwr-source-overfit-initial-state-v1", "ordering": "module then sorted state name",
    "framing": "big-endian uint64 byte lengths for UTF-8 name, JSON shape, ASCII dtype, and tensor bytes",
    "bytes": "contiguous CPU tensor viewed as uint8"}
EXAMPLE_IDS = ["eq", "neq", "add_mod"]


def _eval_ref(step=0):
    return {"schema_version": 1, "step": step, "example_ids": EXAMPLE_IDS,
        "checkpoint_sha256": "b"*64, "records_path": f"segment-000/predictions-step-{step:06d}-{'0'*32}.jsonl",
        "records_sha256": "c"*64, "weighted_ce_numerator": 6.0,
        "eos_inclusive_token_denominator": 3, "strict_valid_numerator": 0,
        "strict_valid_denominator": 3, "per_example": [{"example_id": x, "noise_seed": 1,
            "noise_sha256": "d"*64, "prompt_sha256": "e"*64, "prediction_sha256": "f"*64} for x in EXAMPLE_IDS],
        "receipt_path": f"segment-000/evaluation-step-{step:06d}-{'0'*32}.receipt.json", "receipt_sha256": "a"*64}


def _base_facts():
    return {"initial_state_sha256": "a"*64, "initial_state_hash_recipe": HASH_RECIPE}


def _training_facts():
    return {**_base_facts(), "initial_evaluation_receipt": _eval_ref()}


class RecoveryStoreTests(unittest.TestCase):
    def test_cursor_commits_only_fsynced_payload_and_reopens_verified_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = root / "segment-000"
            segment.mkdir()
            checkpoint = segment / "recovery" / "recovery-checkpoint-step-000000.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"complete checkpoint bytes")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            store = RecoveryStore(root, segment, IDENTITIES)
            descriptor = store.commit(checkpoint, digest,
                progress={"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0},
                phase="initial_evaluation_pending", facts=_base_facts())
            self.assertEqual(RecoveryStore.verify_descriptor(descriptor, IDENTITIES)["phase"],
                             "initial_evaluation_pending")
            self.assertEqual(json.loads((root / "recovery.latest.json").read_text())["checkpoint_sha256"], digest)

    def test_corrupt_checkpoint_or_identity_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); segment = root / "segment-000"; segment.mkdir()
            checkpoint = segment / "recovery" / "recovery-checkpoint-step-000000.pt"
            checkpoint.parent.mkdir(); checkpoint.write_bytes(b"good")
            store = RecoveryStore(root, segment, IDENTITIES)
            descriptor = store.commit(checkpoint, hashlib.sha256(b"good").hexdigest(),
                progress={"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0},
                phase="initial_evaluation_pending", facts=_base_facts())
            checkpoint.write_bytes(b"bad")
            with self.assertRaises(RecoveryError): RecoveryStore.verify_descriptor(descriptor, IDENTITIES)
            checkpoint.write_bytes(b"good")
            wrong = dict(IDENTITIES); wrong["source"] = "f"*64
            with self.assertRaises(RecoveryError): RecoveryStore.verify_descriptor(descriptor, wrong)

    def test_unknown_metadata_fields_and_escaping_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); segment = root / "segment-000"; segment.mkdir()
            checkpoint = segment / "recovery" / "recovery-checkpoint-step-000000.pt"
            checkpoint.parent.mkdir(); checkpoint.write_bytes(b"good")
            store = RecoveryStore(root, segment, IDENTITIES)
            descriptor = store.commit(checkpoint, hashlib.sha256(b"good").hexdigest(),
                progress={"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0},
                phase="training", facts=_training_facts())
            cursor = json.loads(Path(descriptor["cursor_path"]).read_text())
            cursor["unknown"] = True
            Path(descriptor["cursor_path"]).write_text(json.dumps(cursor), encoding="utf-8")
            with self.assertRaises(RecoveryError): RecoveryStore.verify_descriptor(descriptor, IDENTITIES)

    def test_checkpoint_outside_recovery_child_and_cursor_path_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); segment = root / "segment-000"; segment.mkdir()
            checkpoint = segment / "recovery" / "recovery-checkpoint-step-000000.pt"
            checkpoint.parent.mkdir(); checkpoint.write_bytes(b"ok")
            store = RecoveryStore(root, segment, IDENTITIES)
            outside = root / "outside.pt"; outside.write_bytes(b"no")
            with self.assertRaises(RecoveryError):
                store.commit(outside, hashlib.sha256(b"no").hexdigest(),
                    progress={"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0},
                    phase="initial_evaluation_pending", facts=_base_facts())
            descriptor = store.commit(checkpoint, hashlib.sha256(b"ok").hexdigest(),
                progress={"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0},
                phase="initial_evaluation_pending", facts=_base_facts())
            cursor_path = Path(descriptor["cursor_path"])
            cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
            cursor["metadata_path"] = "../outside.json"
            cursor_path.write_text(json.dumps(cursor), encoding="utf-8")
            descriptor["cursor_sha256"] = hashlib.sha256(cursor_path.read_bytes()).hexdigest()
            with self.assertRaises(RecoveryError): RecoveryStore.verify_descriptor(descriptor, IDENTITIES)

    def test_failed_pointer_replace_keeps_previous_cursor_and_orphan_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); segment = root / "segment-000"; segment.mkdir()
            checkpoint = segment / "recovery" / "recovery-checkpoint-step-000000.pt"
            checkpoint.parent.mkdir(); checkpoint.write_bytes(b"zero")
            store = RecoveryStore(root, segment, IDENTITIES)
            first = store.commit(checkpoint, hashlib.sha256(b"zero").hexdigest(),
                progress={"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0},
                phase="initial_evaluation_pending", facts=_base_facts())
            checkpoint.write_bytes(b"zero")
            first = store.commit(checkpoint, hashlib.sha256(b"zero").hexdigest(),
                progress={"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0},
                phase="training", facts=_training_facts())
            before = Path(first["cursor_path"]).read_bytes()
            checkpoint1 = segment / "recovery" / "recovery-checkpoint-step-000001.pt"
            checkpoint1.write_bytes(b"one")
            with patch.object(_recovery_module.os, "replace", side_effect=OSError("injected before publish")):
                with self.assertRaises(OSError):
                    store.commit(checkpoint1, hashlib.sha256(b"one").hexdigest(),
                        progress={"optimizer_step": 1, "examples_seen": 1, "supervised_tokens_seen": 9},
                        phase="training", facts=_training_facts())
            self.assertEqual(Path(first["cursor_path"]).read_bytes(), before)
            self.assertTrue(list((segment / "recovery").glob("metadata-step-000001-*.json")))
            self.assertEqual(RecoveryStore.verify_descriptor(first, IDENTITIES)["progress"]["optimizer_step"], 0)

    def test_live_segment_keeps_latest_two_positive_rolling_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); segment = root / "segment-000"; segment.mkdir()
            store = RecoveryStore(root, segment, IDENTITIES, archival_steps=(3, 6))
            orphan = segment / "recovery" / "recovery-checkpoint-step-000099.pt"
            orphan.write_bytes(b"uncommitted orphan")
            for step in range(7):
                path = segment / "recovery" / f"recovery-checkpoint-step-{step:06d}.pt"
                path.write_bytes(f"checkpoint-{step}".encode())
                store.commit(path, hashlib.sha256(path.read_bytes()).hexdigest(),
                    progress={"optimizer_step": step, "examples_seen": step,
                              "supervised_tokens_seen": step*9}, phase="training", facts=_training_facts())
            kept = {int(path.stem.split("step-")[1]) for path in
                    (segment / "recovery").glob("recovery-checkpoint-step-*.pt")}
            self.assertEqual(kept, {0, 3, 4, 5, 6, 99})
            self.assertTrue(orphan.exists(), "uncommitted orphan must remain for audit")
            self.assertTrue((segment / "recovery" / "recovery-checkpoint-step-000000.pt").exists(),
                            "checkpoint zero must remain for audit")
            retirements = (segment / "recovery" / "retirements.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(retirements), 4)
            self.assertNotIn(orphan.name, "\n".join(retirements))

    def test_phase_facts_are_closed_and_require_initial_receipt_after_initial_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); segment = root / "segment-000"; segment.mkdir()
            checkpoint = segment / "recovery" / "recovery-checkpoint-step-000000.pt"
            checkpoint.parent.mkdir(); checkpoint.write_bytes(b"checkpoint")
            store = RecoveryStore(root, segment, IDENTITIES)
            with self.assertRaises(RecoveryError):
                store.commit(checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    progress={"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0},
                    phase="training", facts=_base_facts())
            malformed = {**_training_facts(), "resume_probe": {"result": {"unexpected": True}}}
            with self.assertRaises(RecoveryError):
                store.commit(checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    progress={"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0},
                    phase="training", facts=malformed)


@unittest.skipIf(torch is None, "optional ML stack unavailable")
class KernelRecoveryTests(unittest.TestCase):
    def test_runrecord_created_segment_directory_is_accepted_by_native_kernel(self):
        import tempfile
        from dwr.experiments.records import RunRecord
        from test_source_overfit_learning import (_fixtures, RecordingTokenizer,
            TEST_TOKENIZER_IDENTITY, IDENTITIES)
        from dwr.training.overfit import run_overfit
        host, examples, config, metadata = _fixtures(87)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "resolved.json"
            config_path.write_text('{"fixture":true}\n', encoding="utf-8")
            record = RunRecord.create(root / "segment-000", command=["python", "fixture.py"],
                config_path=config_path, expected_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                identities={"task": {"name": "tiny-fixture", "revision": "v1", "sha256": "a"*64}},
                rng_streams={"python": 17}, settings={"purpose": "tiny native integration"},
                budgets={"wall_seconds": 60, "memory_mib": 2048, "attempts": 1},
                confirmatory=False, source_root=Path.cwd())
            self.assertEqual({p.name for p in record.output_dir.iterdir()},
                             {"manifest.initial.json", "events.jsonl"})
            result = run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                config=config, identities=IDENTITIES, metadata=metadata, output_dir=record.output_dir,
                check=lambda: None, on_event=lambda *_: None)
            self.assertEqual(result["net_optimizer_steps"], config["training"]["optimizer_steps"])
            self.assertTrue((record.output_dir / "manifest.initial.json").is_file())
            self.assertTrue((record.output_dir / "events.jsonl").is_file())

    def test_kernel_rejects_record_directory_with_any_other_entry_or_terminal(self):
        from test_source_overfit_learning import (_fixtures, RecordingTokenizer,
            TEST_TOKENIZER_IDENTITY, IDENTITIES)
        from dwr.training.overfit import run_overfit
        host, examples, config, metadata = _fixtures(88)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, add in enumerate((
                    lambda d: (d / "manifest.initial.json").write_text("{}", encoding="utf-8"),
                    lambda d: (d / "unknown.txt").write_text("x", encoding="utf-8"),
                    lambda d: (d / "nested").mkdir(),
                    lambda d: (d / "manifest.json").write_text("{}", encoding="utf-8"))):
                output = root / f"segment-{index:03d}"; output.mkdir(); add(output)
                with self.assertRaisesRegex(ValueError, "empty or contain only"):
                    run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                        config=config, identities=IDENTITIES, metadata=metadata, output_dir=output,
                        check=lambda: None, on_event=lambda *_: None)

    def test_hash_valid_cursor_checkpoint_progress_mismatch_fails_before_successor_publish(self):
        from test_source_overfit_learning import (_fixtures, RecordingTokenizer,
            TEST_TOKENIZER_IDENTITY, IDENTITIES, _new_segment)
        from dwr.training.overfit import run_overfit
        from dwr.training.recovery import _canonical
        host, examples, config, metadata = _fixtures(84)
        with tempfile.TemporaryDirectory() as tmp:
            def interrupt(kind, payload):
                if kind == "recovery_committed" and payload["phase"] == "initial_evaluation_pending":
                    raise RuntimeError("stop after initial cursor")
            with self.assertRaisesRegex(RuntimeError, "after initial cursor"):
                run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                    config=config, identities=IDENTITIES, metadata=metadata,
                    output_dir=_new_segment(tmp), check=lambda: None, on_event=interrupt)
            cursor_path = Path(tmp) / "recovery.latest.json"
            cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
            metadata_path = Path(tmp) / cursor["metadata_path"]
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            saved_metadata["progress"]["supervised_tokens_seen"] = 1
            metadata_raw = _canonical(saved_metadata)
            metadata_path.write_bytes(metadata_raw)
            cursor["progress"] = saved_metadata["progress"]
            cursor["metadata_sha256"] = hashlib.sha256(metadata_raw).hexdigest()
            cursor_raw = _canonical(cursor)
            cursor_path.write_bytes(cursor_raw)
            descriptor = {"jobroot": tmp, "cursor_path": str(cursor_path),
                "cursor_sha256": hashlib.sha256(cursor_raw).hexdigest()}
            segment = Path(tmp) / "segment-001"; segment.mkdir()
            host, examples, config, metadata = _fixtures(84)
            with self.assertRaisesRegex(ValueError, "checkpoint progress disagrees"):
                run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                    config=config, identities=IDENTITIES, metadata=metadata, output_dir=segment,
                    check=lambda: None, on_event=lambda *_: None, resume=descriptor)
            self.assertFalse((segment / "recovery" / "recovery-checkpoint-step-000000.pt").exists())

    def test_complete_resume_rejects_corrupt_evaluation_records_and_paths_are_live(self):
        from test_source_overfit_learning import (_fixtures, RecordingTokenizer,
            TEST_TOKENIZER_IDENTITY, IDENTITIES, _new_segment)
        from dwr.training.overfit import run_overfit
        host, examples, config, metadata = _fixtures(85)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                config=config, identities=IDENTITIES, metadata=metadata, output_dir=_new_segment(tmp),
                check=lambda: None, on_event=lambda *_: None)
            for key, value in result["artifact_paths"].items():
                if isinstance(value, dict):
                    self.assertTrue(all(Path(path).is_file() for path in value.values()), key)
                elif isinstance(value, str):
                    self.assertTrue(Path(value).is_file(), key)
            Path(result["artifact_paths"]["predictions_final"]).write_text("corrupt\n", encoding="utf-8")
            segment = Path(tmp) / "segment-001"; segment.mkdir()
            host, examples, config, metadata = _fixtures(85)
            with self.assertRaisesRegex(ValueError, "prediction record bytes"):
                run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                    config=config, identities=IDENTITIES, metadata=metadata, output_dir=segment,
                    check=lambda: None, on_event=lambda *_: None, resume=result["recovery_descriptor"])

    def test_complete_cursor_rejects_coherently_rehashed_initial_digest_or_stale_predictions(self):
        from test_source_overfit_learning import (_fixtures, RecordingTokenizer,
            TEST_TOKENIZER_IDENTITY, IDENTITIES, _new_segment)
        from dwr.training.overfit import run_overfit
        from dwr.training.recovery import _canonical
        host, examples, config, metadata = _fixtures(86)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                config=config, identities=IDENTITIES, metadata=metadata, output_dir=_new_segment(tmp),
                check=lambda: None, on_event=lambda *_: None)
            cursor_path = Path(result["recovery_descriptor"]["cursor_path"])
            original_cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
            metadata_path = Path(tmp) / original_cursor["metadata_path"]
            original_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for offset, (mutate, message) in enumerate((
                    (lambda facts: facts["final_result"].__setitem__("initial_state_sha256", "0"*64),
                     "initial state digest differs"),
                    (lambda facts: facts["final_result"]["predictions"].append({"example_id": "stale"}),
                     "predictions differ")), start=1):
                cursor = dict(original_cursor)
                facts_metadata = json.loads(json.dumps(original_metadata))
                mutate(facts_metadata["facts"])
                metadata_raw = _canonical(facts_metadata)
                metadata_path.write_bytes(metadata_raw)
                cursor["metadata_sha256"] = hashlib.sha256(metadata_raw).hexdigest()
                cursor_raw = _canonical(cursor)
                cursor_path.write_bytes(cursor_raw)
                descriptor = {"jobroot": tmp, "cursor_path": str(cursor_path),
                    "cursor_sha256": hashlib.sha256(cursor_raw).hexdigest()}
                segment = Path(tmp) / f"segment-{offset:03d}"; segment.mkdir()
                host, examples, config, metadata = _fixtures(86)
                with self.assertRaisesRegex(ValueError, message):
                    run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                        config=config, identities=IDENTITIES, metadata=metadata, output_dir=segment,
                        check=lambda: None, on_event=lambda *_: None, resume=descriptor)
            metadata_path.write_bytes(_canonical(original_metadata))
            cursor_path.write_bytes(_canonical(original_cursor))
    def test_initial_evaluation_pending_resumes_without_losing_initial_checkpoint(self):
        import numpy as np
        from test_source_overfit_learning import (_fixtures, RecordingTokenizer,
            TEST_TOKENIZER_IDENTITY, IDENTITIES, _new_segment)
        from dwr.training.overfit import run_overfit
        host, examples, config, metadata = _fixtures(81)
        with tempfile.TemporaryDirectory() as tmp:
            observed = []
            def interrupt(kind, payload):
                if kind == "recovery_committed" and payload["phase"] == "initial_evaluation_pending":
                    raise RuntimeError("crash after initial cursor publication")
            with self.assertRaisesRegex(RuntimeError, "after initial cursor publication"):
                run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                    config=config, identities=IDENTITIES, metadata=metadata,
                    output_dir=_new_segment(tmp), check=lambda: None, on_event=interrupt)
            pointer = Path(tmp) / "recovery.latest.json"
            descriptor = {"jobroot": tmp, "cursor_path": str(pointer),
                "cursor_sha256": hashlib.sha256(pointer.read_bytes()).hexdigest()}
            segment = Path(tmp) / "segment-001"; segment.mkdir()
            host, examples, config, metadata = _fixtures(81)
            result = run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                config=config, identities=IDENTITIES, metadata=metadata, output_dir=segment,
                check=lambda: None, on_event=lambda kind, payload: observed.append((kind, payload)),
                resume=descriptor)
            self.assertEqual(result["net_optimizer_steps"], 9)
            self.assertEqual(len([x for x in observed if x[0] == "evaluation"]), 2)
            self.assertEqual(result["initial_evaluation_receipt"]["step"], 0)

    def test_resume_probe_pending_replays_probe_once_without_repeating_initial_evaluation(self):
        from test_source_overfit_learning import (_fixtures, RecordingTokenizer,
            TEST_TOKENIZER_IDENTITY, IDENTITIES, _new_segment)
        from dwr.training.overfit import run_overfit
        host, examples, config, metadata = _fixtures(82)
        with tempfile.TemporaryDirectory() as tmp:
            def interrupt(kind, payload):
                if kind == "recovery_committed" and payload["phase"] == "resume_probe_pending":
                    raise RuntimeError("crash before resume probe")
            with self.assertRaisesRegex(RuntimeError, "before resume probe"):
                run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                    config=config, identities=IDENTITIES, metadata=metadata,
                    output_dir=_new_segment(tmp), check=lambda: None, on_event=interrupt)
            pointer = Path(tmp) / "recovery.latest.json"
            descriptor = {"jobroot": tmp, "cursor_path": str(pointer),
                "cursor_sha256": hashlib.sha256(pointer.read_bytes()).hexdigest()}
            self.assertEqual(json.loads(pointer.read_text(encoding="utf-8"))["phase"], "resume_probe_pending")
            segment = Path(tmp) / "segment-001"; segment.mkdir()
            host, examples, config, metadata = _fixtures(82)
            events = []
            result = run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                config=config, identities=IDENTITIES, metadata=metadata, output_dir=segment,
                check=lambda: None, on_event=lambda kind, payload: events.append((kind, payload)),
                resume=descriptor)
            self.assertEqual(result["net_optimizer_steps"], 9)
            self.assertEqual(result["resume"]["first_replayed_ids"], result["resume"]["second_replayed_ids"])
            self.assertEqual(len([x for x in events if x[0] == "evaluation"]), 1)
            self.assertEqual(result["resume"]["optimizer_moment_count"] > 0, True)

    def test_final_evaluation_pending_finishes_without_an_extra_update(self):
        from test_source_overfit_learning import (_fixtures, RecordingTokenizer,
            TEST_TOKENIZER_IDENTITY, IDENTITIES, _new_segment)
        from dwr.training.overfit import run_overfit
        host, examples, config, metadata = _fixtures(83)
        with tempfile.TemporaryDirectory() as tmp:
            def interrupt(kind, payload):
                if kind == "recovery_committed" and payload["phase"] == "final_evaluation_pending":
                    raise RuntimeError("crash before final evaluation")
            with self.assertRaisesRegex(RuntimeError, "before final evaluation"):
                run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                    config=config, identities=IDENTITIES, metadata=metadata,
                    output_dir=_new_segment(tmp), check=lambda: None, on_event=interrupt)
            pointer = Path(tmp) / "recovery.latest.json"
            descriptor = {"jobroot": tmp, "cursor_path": str(pointer),
                "cursor_sha256": hashlib.sha256(pointer.read_bytes()).hexdigest()}
            self.assertEqual(json.loads(pointer.read_text(encoding="utf-8"))["phase"], "final_evaluation_pending")
            segment = Path(tmp) / "segment-001"; segment.mkdir()
            host, examples, config, metadata = _fixtures(83)
            events = []
            result = run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                config=config, identities=IDENTITIES, metadata=metadata, output_dir=segment,
                check=lambda: None, on_event=lambda kind, payload: events.append((kind, payload)),
                resume=descriptor)
            self.assertEqual(result["net_optimizer_steps"], 9)
            self.assertEqual(result["segment_physical_optimizer_updates"], 0)
            self.assertEqual(len([x for x in events if x[0] == "update"]), 0)
            self.assertEqual(len([x for x in events if x[0] == "evaluation"]), 1)
            self.assertEqual(len(result["predictions"]), 6)

    def test_fresh_process_resume_matches_uninterrupted_tiny_cpu_trajectory(self):
        import numpy as np
        import torch
        torch.set_num_threads(1)
        from test_source_overfit_learning import (_fixtures, RecordingTokenizer,
            TEST_TOKENIZER_IDENTITY, IDENTITIES, _new_segment)
        from dwr.training.overfit import run_overfit, _within_tolerance, _initial_state_digest

        def frozen_digest(host):
            digest = hashlib.sha256()
            for name, parameter in host.backbone.state_dict().items():
                digest.update(name.encode()); digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()

        def seed_streams():
            random.seed(616); np.random.seed(616); torch.manual_seed(616)

        host, examples, config, metadata = _fixtures(79)
        baseline_initial, _ = _initial_state_digest(host.workspace, host.readers)
        baseline_frozen = frozen_digest(host)
        with tempfile.TemporaryDirectory() as baseline_dir, tempfile.TemporaryDirectory() as interrupted_dir:
            seed_streams()
            baseline = run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                config=config, identities=IDENTITIES, metadata=metadata,
                output_dir=_new_segment(baseline_dir), check=lambda: None, on_event=lambda *_: None)
            baseline_payload = torch.load(baseline["artifact_paths"]["recovery_checkpoints"]["9"],
                                          map_location="cpu", weights_only=True)

            host, examples, config, metadata = _fixtures(79)
            interrupted_initial, _ = _initial_state_digest(host.workspace, host.readers)
            self.assertEqual(baseline_initial, interrupted_initial, "fresh workspace initialization differs")
            self.assertEqual(baseline_frozen, frozen_digest(host), "fresh frozen backbone initialization differs")
            expected_frozen = frozen_digest(host)
            seed_streams()
            observed = False
            def on_event(kind, _payload):
                nonlocal observed
                if kind == "update": observed = True
            def check():
                if observed: raise RuntimeError("simulated worker interruption")
            with self.assertRaisesRegex(RuntimeError, "simulated worker interruption"):
                run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
                    config=config, identities=IDENTITIES, metadata=metadata,
                    output_dir=_new_segment(interrupted_dir), check=check, on_event=on_event)
            baseline_updates = [json.loads(line) for line in
                Path(baseline["artifact_paths"]["updates"]).read_text(encoding="utf-8").splitlines()]
            interrupted_updates = [json.loads(line) for line in
                (Path(interrupted_dir) / "segment-000" / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
            baseline_first = next(row for row in baseline_updates if row["role"] == "main" and row["net_step"] == 1)
            interrupted_first = next(row for row in interrupted_updates if row["role"] == "main" and row["net_step"] == 1)
            for key in ("example_ids", "training_noise_sha256", "global_rng_sha256_before_update",
                        "training_batch_sha256", "training_modes_sha256"):
                self.assertEqual(baseline_first[key], interrupted_first[key], key)
            self.assertEqual(baseline_first["loss"], interrupted_first["loss"])

            cursor_path = Path(interrupted_dir) / "recovery.latest.json"
            descriptor = {"jobroot": interrupted_dir, "cursor_path": str(cursor_path),
                "cursor_sha256": hashlib.sha256(cursor_path.read_bytes()).hexdigest()}
            segment = Path(interrupted_dir) / "segment-001"; segment.mkdir()
            script = r'''import hashlib, json, random, sys
import numpy as np, torch
torch.set_num_threads(1)
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_source_overfit_learning import _fixtures, RecordingTokenizer, TEST_TOKENIZER_IDENTITY, IDENTITIES
from dwr.training.overfit import run_overfit
root = Path(sys.argv[1]); host, examples, config, metadata = _fixtures(79)
def frozen_digest(host):
    digest = hashlib.sha256()
    for name, parameter in host.backbone.state_dict().items():
        digest.update(name.encode()); digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
before = frozen_digest(host)
descriptor = json.loads(sys.argv[2])
result = run_overfit(host, RecordingTokenizer(), TEST_TOKENIZER_IDENTITY, examples,
    config=config, identities=IDENTITIES, metadata=metadata, output_dir=root / "segment-001",
    check=lambda: None, on_event=lambda *_: None, resume=descriptor)
after = frozen_digest(host)
print(json.dumps({"checkpoint": result["artifact_paths"]["recovery_checkpoints"]["9"],
    "frozen_before": before, "frozen_after": after}))
'''
            done = subprocess.run([sys.executable, "-c", script, interrupted_dir,
                json.dumps(descriptor)], cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stdout+done.stderr)
            child_result = json.loads(done.stdout.strip().splitlines()[-1])
            self.assertEqual(expected_frozen, child_result["frozen_before"])
            self.assertEqual(expected_frozen, child_result["frozen_after"])
            resumed_updates = [json.loads(line) for line in
                (segment / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
            expected_second = next(row for row in baseline_updates if row["role"] == "main" and row["net_step"] == 2)
            actual_second = next(row for row in resumed_updates if row["role"] == "main" and row["net_step"] == 2)
            self.assertEqual(expected_second["example_ids"], actual_second["example_ids"])
            self.assertEqual(expected_second["training_noise_sha256"], actual_second["training_noise_sha256"])
            self.assertEqual(expected_second["global_rng_sha256_before_update"],
                             actual_second["global_rng_sha256_before_update"])
            self.assertEqual(expected_second["training_batch_sha256"], actual_second["training_batch_sha256"])
            self.assertEqual(expected_second["training_modes_sha256"], actual_second["training_modes_sha256"])
            self.assertEqual(expected_second["loss"], actual_second["loss"])
            resumed_payload = torch.load(child_result["checkpoint"],
                                         map_location="cpu", weights_only=True)
            for key in ("workspace", "readers", "optimizer", "scheduler", "sampler", "rng"):
                error = _within_tolerance(baseline_payload[key], resumed_payload[key], 1e-12, 1e-12)
                self.assertTrue(error[2], f"{key}: abs={error[0]} rel={error[1]}")
            self.assertEqual(baseline_payload["progress"], resumed_payload["progress"])


if __name__ == "__main__": unittest.main()
