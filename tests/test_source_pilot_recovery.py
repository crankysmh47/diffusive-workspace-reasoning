"""Native CPU cursor, receipt and artifact recovery regressions for the pilot."""
from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
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
    from dwr.training.checkpoints import save_training_checkpoint
    from dwr.training.controls import configure_source_control
    from dwr.training.sampler import SourceSampler
    from dwr.training.source_pilot import verify_pilot_stage_receipt
    from dwr.training.pilot_recovery import PilotRecoveryStore, PilotRecoveryError
    from dwr.training.pilot_recovery import _canonical as recovery_canonical, _sha_bytes as recovery_sha
    from test_source_pilot_learning import fixture_examples, fixture_host, fixture_metadata, _IDENTITIES
else:
    save_training_checkpoint = configure_source_control = SourceSampler = None
    verify_pilot_stage_receipt = PilotRecoveryStore = PilotRecoveryError = None
    recovery_canonical = recovery_sha = None


_ML_READY = np is not None and torch is not None and Qwen2Config is not None


@unittest.skipUnless(_ML_READY, "optional native ML dependencies unavailable")
class PilotRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _ML_READY:
            cls._original_torch_threads = torch.get_num_threads()
            torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        if _ML_READY:
            torch.set_num_threads(cls._original_torch_threads)

    def _retention_store(self, root, segment_name="segment-000"):
        segment = root / segment_name
        (segment / "recovery").mkdir(parents=True)
        examples = fixture_examples(5)
        return PilotRecoveryStore(root, segment, _IDENTITIES, arm_id="K0",
            selected_ids=tuple(example.example_id for example in examples),
            metadata=fixture_metadata(examples), examples=examples, steps=6)

    def _publish_retention_boundary(self, store, step):
        from dwr.training.pilot_recovery import _append_durable_line
        root = store.jobroot
        checkpoint = store.recovery_dir / f"recovery-checkpoint-step-{step:06d}.pt"
        checkpoint.write_bytes(f"synthetic committed checkpoint {step}".encode("ascii"))
        digest = recovery_sha(checkpoint.read_bytes())
        progress = {"optimizer_step": step, "examples_seen": step,
            "supervised_tokens_seen": 0 if step == 0 else step * 3}
        metadata = {"schema_version": 1, "segment": store.segment_dir.name,
            "arm_id": "K0", "identities": _IDENTITIES, "progress": progress,
            "phase": "training", "checkpoint_path": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": digest, "steps": 6}
        metadata_raw = recovery_canonical(metadata)
        metadata_path = store.recovery_dir / f"synthetic-metadata-{step:06d}.json"
        metadata_path.write_bytes(metadata_raw)
        pointer = {"schema_version": 1,
            "metadata_path": metadata_path.relative_to(root).as_posix(),
            "metadata_sha256": recovery_sha(metadata_raw),
            "checkpoint_path": metadata["checkpoint_path"], "checkpoint_sha256": digest,
            "segment": store.segment_dir.name, "arm_id": "K0", "identities": _IDENTITIES,
            "progress": progress, "phase": "training", "steps": 6}
        pointer_raw = recovery_canonical(pointer)
        (root / "recovery.latest.json").write_bytes(pointer_raw)
        _append_durable_line(root / "recovery-commits.jsonl", {
            "cursor_sha256": recovery_sha(pointer_raw),
            "metadata_path": pointer["metadata_path"],
            "metadata_sha256": pointer["metadata_sha256"],
            "checkpoint_path": pointer["checkpoint_path"], "checkpoint_sha256": digest,
            "segment": store.segment_dir.name, "progress": progress, "phase": "training"})
        store._retire_old_checkpoints()
        return checkpoint

    def test_rolling_retirement_survives_two_cycles_and_reused_segment_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = self._retention_store(root)
            paths = {step: self._publish_retention_boundary(store, step) for step in range(7)}
            self.assertEqual({step for step, path in paths.items() if path.exists()}, {0, 4, 5, 6})
            next_segment = self._retention_store(root, "segment-001")
            relocated = next_segment.recovery_dir / "recovery-checkpoint-step-000006.pt"
            relocated.write_bytes(paths[6].read_bytes())
            self._publish_retention_boundary(next_segment, 6)
            self.assertTrue(relocated.exists())

    def test_unretired_missing_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = self._retention_store(root)
            paths = {step: self._publish_retention_boundary(store, step) for step in range(4)}
            paths[2].unlink()
            with self.assertRaises(PilotRecoveryError):
                self._publish_retention_boundary(store, 4)

    def test_transition_rejects_valid_alternative_prior_receipts_and_same_step_rebindings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = self._retention_store(root)
            progress = {"optimizer_step": 6, "examples_seen": 6, "supervised_tokens_seen": 18}
            old = {"initial_state_sha256": "a" * 64,
                "initial_state_hash_recipe": {"name": "recipe"},
                "stage_receipts": {name: {"path": f"{name}-old.json", "sha256": "1" * 64}
                    for name in ("initial", "final", "zero")},
                "bank": {"path": "bank-old.json", "sha256": "2" * 64},
                "frozen_audit_refs": None, "final_result": None}
            previous = {"phase": "diagnostic_noise_pending", "progress": progress,
                "facts": old, "checkpoint_sha256": "3" * 64}
            valid = copy.deepcopy(old)
            valid["stage_receipts"]["noise"] = {"path": "noise-valid.json", "sha256": "4" * 64}
            self.assertTrue(store._transition(previous, progress, "diagnostic_shuffle_pending",
                valid, "3" * 64))
            replacements = []
            changed = copy.deepcopy(valid)
            changed["stage_receipts"]["initial"] = {"path": "initial-valid-alternative.json", "sha256": "5" * 64}
            replacements.append((changed, "3" * 64))
            changed = copy.deepcopy(valid); changed["initial_state_sha256"] = "b" * 64
            replacements.append((changed, "3" * 64))
            changed = copy.deepcopy(valid); changed["bank"] = {"path": "bank-alternative.json", "sha256": "6" * 64}
            replacements.append((changed, "3" * 64))
            replacements.append((copy.deepcopy(valid), "7" * 64))
            for changed, checkpoint_sha in replacements:
                self.assertFalse(store._transition(previous, progress, "diagnostic_shuffle_pending",
                    changed, checkpoint_sha))

    def test_k4_shuffle_receipt_cannot_publish_a_backward_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            segment = root / "segment-000"; (segment / "recovery").mkdir(parents=True)
            examples = fixture_examples(5)
            store = PilotRecoveryStore(root, segment, _IDENTITIES, arm_id="K4",
                selected_ids=tuple(row.example_id for row in examples),
                metadata=fixture_metadata(examples), examples=examples, steps=6)
            progress = {"optimizer_step": 6, "examples_seen": 6, "supervised_tokens_seen": 18}
            receipts = {name: {"path": f"{name}.json", "sha256": "1" * 64}
                for name in ("initial", "final", "zero", "noise")}
            old_facts = {"initial_state_sha256": "a" * 64,
                "initial_state_hash_recipe": {"name": "recipe"},
                "stage_receipts": receipts, "bank": {"path": "bank.json", "sha256": "2" * 64},
                "frozen_audit_refs": None, "final_result": None}
            previous = {"phase": "diagnostic_shuffle_pending", "progress": progress,
                "facts": old_facts, "checkpoint_sha256": "3" * 64}
            next_facts = copy.deepcopy(old_facts)
            next_facts["stage_receipts"]["shuffle"] = {"path": "shuffle-valid.json", "sha256": "4" * 64}
            self.assertTrue(store._transition(previous, progress, "diagnostic_shuffle_pending",
                next_facts, "3" * 64))
            self.assertFalse(store._transition(previous, progress, "final_evaluation_pending",
                next_facts, "3" * 64))

    def _zero_checkpoint(self, root, *, arm_id="K0", identities=None):
        host = fixture_host(613)
        steps = {"K0": 0, "K1": 1, "K4": 4}[arm_id]
        configure_source_control(host, steps=steps)
        groups = []
        if steps:
            groups.append({"params": list(host.workspace.parameters()), "name": "workspace"})
        groups.append({"params": list(host.readers.parameters()), "name": "readers"})
        optimizer = torch.optim.AdamW(groups, lr=0.0003, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 8))
        examples = tuple(sorted(fixture_examples(10, "source_train"), key=lambda row: row.example_id))
        sampler = SourceSampler(tuple(example.example_id for example in examples), seed=20261217)
        noise = torch.Generator(device="cpu").manual_seed(20261218)
        progress = {"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0}
        checkpoint = root / "segment-000" / "recovery" / "recovery-checkpoint-step-000000.pt"
        checkpoint.parent.mkdir(parents=True)
        digest, _ = save_training_checkpoint(checkpoint, host=host, optimizer=optimizer,
            scheduler=scheduler, scaler=None, sampler=sampler, noise_generator=noise,
            identities=identities or _IDENTITIES, progress=progress)
        return host, checkpoint, digest

    def test_step_zero_cursor_roundtrips_and_binds_live_examples_and_metadata(self):
        examples = fixture_examples(5)
        metadata = fixture_metadata(examples)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            segment = root / "segment-000"
            segment.mkdir()
            _, checkpoint, digest = self._zero_checkpoint(root)
            store = PilotRecoveryStore(root, segment, _IDENTITIES, arm_id="K0",
                selected_ids=tuple(example.example_id for example in examples),
                metadata=metadata, examples=examples, steps=2)
            facts = {"initial_state_sha256": "a" * 64,
                "initial_state_hash_recipe": {"name": "dwr-source-overfit-initial-state-v1",
                    "ordering": "module then sorted state name",
                    "framing": "big-endian uint64 byte lengths for UTF-8 name, JSON shape, ASCII dtype, and tensor bytes",
                    "bytes": "contiguous CPU tensor viewed as uint8"},
                "stage_receipts": {}, "bank": None,
                "frozen_audit_refs": None, "final_result": None}
            descriptor = store.commit(checkpoint, digest,
                progress={"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0},
                phase="initial_evaluation_pending", facts=facts)
            verified = PilotRecoveryStore.verify_descriptor(descriptor, _IDENTITIES,
                arm_id="K0", selected_ids=tuple(example.example_id for example in examples),
                metadata=metadata, examples=examples, steps=2)
            self.assertEqual(verified["phase"], "initial_evaluation_pending")
            self.assertEqual(verified["progress"]["optimizer_step"], 0)
            self.assertEqual(verified["checkpoint_sha256"], digest)
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed committed bytes")
            with self.assertRaises(PilotRecoveryError):
                PilotRecoveryStore.verify_descriptor(descriptor, _IDENTITIES,
                    arm_id="K0", selected_ids=tuple(example.example_id for example in examples),
                    metadata=metadata, examples=examples, steps=2)

    def test_progress_method_identity_and_illegal_phase_fail_before_cursor_replace(self):
        examples = fixture_examples(5)
        metadata = fixture_metadata(examples)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            segment = root / "segment-000"
            segment.mkdir()
            _, checkpoint, digest = self._zero_checkpoint(root)
            store = PilotRecoveryStore(root, segment, _IDENTITIES, arm_id="K0",
                selected_ids=tuple(example.example_id for example in examples),
                metadata=metadata, examples=examples, steps=2)
            facts = {"initial_state_sha256": "a" * 64,
                "initial_state_hash_recipe": {"name": "dwr-source-overfit-initial-state-v1",
                    "ordering": "module then sorted state name",
                    "framing": "big-endian uint64 byte lengths for UTF-8 name, JSON shape, ASCII dtype, and tensor bytes",
                    "bytes": "contiguous CPU tensor viewed as uint8"},
                "stage_receipts": {}, "bank": None,
                "frozen_audit_refs": None, "final_result": None}
            descriptor = store.commit(checkpoint, digest,
                progress={"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0},
                phase="initial_evaluation_pending", facts=facts)
            cursor_path = Path(descriptor["cursor_path"])
            original = cursor_path.read_bytes()
            with self.assertRaises((ValueError, RuntimeError)):
                store.commit(checkpoint, digest,
                    progress={"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0},
                    phase="complete", facts=facts)
            self.assertEqual(cursor_path.read_bytes(), original)
            with self.assertRaises((ValueError, RuntimeError)):
                PilotRecoveryStore.verify_descriptor(descriptor, _IDENTITIES,
                    arm_id="K1", selected_ids=tuple(example.example_id for example in examples),
                    metadata=metadata, examples=examples, steps=2)
            self.assertEqual(cursor_path.read_bytes(), original)

    def test_receipt_verifier_rejects_method_stage_and_checkpoint_policy_mismatches(self):
        examples = fixture_examples(5)
        metadata = fixture_metadata(examples)
        bad_ref = {"path": "does-not-exist.json", "sha256": "0" * 64}
        with self.assertRaises(ValueError):
            verify_pilot_stage_receipt(Path.cwd(), bad_ref, arm_id="K0", stage="initial",
                expected_step=1, expected_ids=tuple(row.example_id for row in examples),
                expected_checkpoint_sha256="0" * 64, identities=_IDENTITIES,
                metadata=metadata, examples=examples)
        with self.assertRaises(ValueError):
            verify_pilot_stage_receipt(Path.cwd(), bad_ref, arm_id="K0", stage="shuffle",
                expected_step=2, expected_ids=tuple(row.example_id for row in examples),
                expected_checkpoint_sha256="0" * 64, identities=_IDENTITIES,
                metadata=metadata, examples=examples, expected_bank_metadata_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
