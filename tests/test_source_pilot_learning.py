"""Native CPU behavioral checks for the prospective pilot arm kernel."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "torch", "transformers"}:
        raise
    np = torch = Qwen2Config = Qwen2ForCausalLM = None

if torch is not None and Qwen2Config is not None:
    from dwr.models.qwen2 import Qwen2WorkspaceLM
    from dwr.models.workspace import FixedWorkspace
    from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, Solution
    from dwr.tasks.run_support import stable_json
    from dwr.training.batches import SourceExample, TokenizerIdentity
    from dwr.training.checkpoints import _learned_states
    from dwr.training.controls import configure_source_control
    from dwr.training.source_pilot import run_pilot_arm
    from test_training_batch_loss import AsciiTokenizer
else:
    Qwen2WorkspaceLM = FixedWorkspace = SourceExample = TokenizerIdentity = stable_json = None
    configure_source_control = run_pilot_arm = AsciiTokenizer = None


_ML_READY = np is not None and torch is not None and Qwen2Config is not None
_IDS = tuple(hashlib.sha256(f"pilot-fixture-{i}".encode()).hexdigest() for i in range(10))
_IDENTITIES = {key: hashlib.sha256(key.encode()).hexdigest() for key in
               ("source", "config", "data", "model", "tokenizer", "schedule")}
_STRATA = ("A2", "A3", "B2", "C2", "D2")


if AsciiTokenizer is not None:
    class FixtureTokenizer(AsciiTokenizer):
        def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            return "".join(chr(value) for value in ids)
else:
    class FixtureTokenizer:
        pass


def _read_receipt(root, reference):
    import json
    return json.loads((root / reference["path"]).read_text(encoding="utf-8"))


def _rewrite_stage(root, reference, edit_rows, edit_receipt=None):
    from dwr.training.source_pilot import _canonical
    receipt = _read_receipt(root, reference)
    rows_path = root / receipt["records"]["path"]
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    edit_rows(rows)
    raw_rows = b"".join(_canonical(row) for row in rows)
    records_path = root / f"tampered-{hashlib.sha256(raw_rows).hexdigest()}.jsonl"
    records_path.write_bytes(raw_rows)
    receipt["records"] = {"path": records_path.name,
        "sha256": hashlib.sha256(raw_rows).hexdigest()}
    if edit_receipt is not None:
        edit_receipt(receipt, rows)
    raw_receipt = _canonical(receipt)
    receipt_path = root / f"tampered-{hashlib.sha256(raw_receipt).hexdigest()}.receipt.json"
    receipt_path.write_bytes(raw_receipt)
    return {"path": receipt_path.name, "sha256": hashlib.sha256(raw_receipt).hexdigest()}


def _fixture_task():
    task = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
        (Constraint(RelationKind.EQ, ("v0", "v1")),
         Constraint(RelationKind.EQ, ("v1", "v2"))), (GivenAssignment("v0", 1),))
    solution = Solution((("v0", 1), ("v1", 1), ("v2", 1)))
    return task, solution


def fixture_host(seed=501):
    config = Qwen2Config(vocab_size=128, hidden_size=12, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=3, num_key_value_heads=1,
        attention_dropout=0.0, tie_word_embeddings=True, pad_token_id=127,
        eos_token_id=127, _attn_implementation="eager")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = Qwen2ForCausalLM(config).to(dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 10)
        workspace = FixedWorkspace(12, num_slots=4, latent_dim=8, num_heads=2,
            max_steps=4, slot_identity="exchangeable").to(dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 20)
        host = Qwen2WorkspaceLM(backbone, workspace, prompt_layer=1,
            adapter_layers=(0, 1, 2), reader_heads=3)
    with torch.no_grad():
        host.backbone.lm_head.weight[127].fill_(10.0)
    original_projection = host.backbone.lm_head.forward
    def eos_favored_projection(hidden):
        logits = original_projection(hidden)
        bias = torch.zeros(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        bias[127] = 20.0
        return logits + bias
    host.backbone.lm_head.forward = eos_favored_projection
    return host.eval()


def fixture_examples(count=5, split="source_validation", start=0):
    task, solution = _fixture_task()
    return tuple(SourceExample(_IDS[start + index], split, task, solution) for index in range(count))


def fixture_metadata(examples):
    return {example.example_id: {
        "profile": "eq", "fine_stratum": _STRATA[index % len(_STRATA)],
        "prompt_tokens": 4,
        "answer_tokens_including_eos": len(stable_json({"assignment": dict(example.solution.assignments)})) + 1,
        "total_tokens": 4 + len(stable_json({"assignment": dict(example.solution.assignments)})) + 1,
    } for index, example in enumerate(examples)}


def fixture_registration(arm_id, steps=2):
    policy_steps = {"K0": 0, "K1": 1, "K4": 4}[arm_id]
    config = {
        "arms": [{"arm_id": arm_id, "steps": policy_steps}],
        "training": {"optimizer_steps": steps, "batch_size": 1,
            "gradient_clip_norm": 1.0, "diagnose_state_gradient_steps": [1, steps],
            "optimizer": {"learning_rate": 0.0003, "betas": [0.9, 0.999],
                "epsilon": 1e-8, "weight_decay": 0.01, "amsgrad": False,
                "capturable": False, "differentiable": False, "foreach": False,
                "fused": False, "maximize": False},
            "schedule": {"warmup_steps": 8,
                "function": "min(1.0,(scheduler_step+1)/8)"}},
        "seeds": {"python_global": 20261211, "numpy_global": 20261212,
            "torch_cpu_global": 20261213, "torch_cuda_global": 20261214,
            "workspace_initialization": 20261215, "reader_initialization": 20261216,
            "source_sampler": 20261217, "training_noise": 20261218,
            "evaluation_noise_base": 20261219},
        "decoding": {"embedding_vocab_size": 128, "eos_token_id": 127,
            "pad_token_id": 127, "max_prompt_tokens": 4096,
            "max_answer_tokens_including_eos": 256, "max_new_tokens_including_eos": 128,
            "max_total_tokens": 4096},
        "evaluation": {"steps": [0, steps]},
        "diagnostics": {"steps": ["unablated", "zero", "noise", "cross_example_shuffle"],
            "noise_rule": "same_deterministic_per_id_Z0_as_official_evaluation",
            "shuffle": "next_example_learned_final_workspace_in_cyclic_lexicographic_ID_order",
            "donor_map_path": "configs/splits/source_validation_pilot_cohort_v1.json",
            "donor_map_key": "shuffle_donor_map"},
    }
    return {"config": config, "config_sha256": hashlib.sha256(b"fixture config").hexdigest()}


def _frozen_ref(root: Path, name: str, host, arm_id: str):
    from dwr.training.source_pilot import _canonical, _tensor_digest
    hashes = {name: _tensor_digest(value) for name, value in host.backbone.named_parameters()}
    document = {"schema": "dwr-source-pilot-initial-frozen-v1", "arm_id": arm_id,
        "identities": _IDENTITIES, "original_parameter_hashes": hashes}
    path = root / name
    path.write_bytes(_canonical(document))
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@unittest.skipUnless(_ML_READY, "optional native ML dependencies unavailable")
class SourcePilotLearningTests(unittest.TestCase):
    def test_fresh_boundary_rejects_valid_same_step_orphan_and_preserves_far_orphan(self):
        original_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, original_threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); segment = root / "segment-000"; segment.mkdir()
            from test_source_pilot_recovery import PilotRecoveryTests
            orphan_host, orphan_path, orphan_sha = PilotRecoveryTests()._zero_checkpoint(root)
            host = fixture_host(1901)
            orphan_state = _learned_states(orphan_host)
            live_state = _learned_states(host)
            self.assertFalse(all(torch.equal(a, b) for module_a, module_b in zip(orphan_state, live_state)
                for key in module_a for a, b in ((module_a[key], module_b[key]),)))
            far_orphan = segment / "recovery" / "recovery-checkpoint-step-000099.pt"
            far_orphan.write_bytes(b"uncommitted far-step orphan")
            train = tuple(sorted(fixture_examples(5, "source_train", start=5), key=lambda row: row.example_id))
            validation = fixture_examples(5)
            metadata = {"training": fixture_metadata(train), "validation": fixture_metadata(validation)}
            with self.assertRaisesRegex(Exception, "unrecognized existing checkpoint path"):
                run_pilot_arm(host, FixtureTokenizer(), TokenizerIdentity("fixture", "v1"),
                    train, validation, registration=fixture_registration("K0"), arm_id="K0",
                    identities=_IDENTITIES, metadata=metadata, output_dir=segment,
                    check=lambda: None, on_event=lambda *_: None,
                    before_complete=lambda _: self.fail("orphan collision must fail before evaluation"))
            self.assertEqual(hashlib.sha256(orphan_path.read_bytes()).hexdigest(), orphan_sha)
            self.assertEqual(far_orphan.read_bytes(), b"uncommitted far-step orphan")
            self.assertFalse((root / "recovery.latest.json").exists())

    def test_new_step_collision_rejects_valid_alternative_and_keeps_prior_cursor(self):
        original_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, original_threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); segment = root / "segment-000"; segment.mkdir()
            train = tuple(sorted(fixture_examples(5, "source_train", start=5), key=lambda row: row.example_id))
            validation = fixture_examples(5)
            metadata = {"training": fixture_metadata(train), "validation": fixture_metadata(validation)}
            target = segment / "recovery" / "recovery-checkpoint-step-000001.pt"
            far_orphan = segment / "recovery" / "recovery-checkpoint-step-000099.pt"
            far_orphan.parent.mkdir()
            far_orphan.write_bytes(b"uncommitted far-step orphan")
            injected = {"done": False, "step0_cursor": None, "step0_pointer": None, "collision_sha": None}

            def event(kind, payload):
                if kind != "recovery_committed" or injected["done"] or \
                        payload["phase"] != "training" or payload["progress"]["optimizer_step"] != 0:
                    return
                injected["step0_pointer"] = (root / "recovery.latest.json").read_bytes()
                cursor = json.loads(injected["step0_pointer"].decode("utf-8"))
                injected["step0_cursor"] = (root / cursor["checkpoint_path"]).read_bytes()
                # Build an independent, schema-valid K0 checkpoint at the exact next progress
                # boundary. Its alternate reader tensors make it unsafe to adopt by filename.
                from dwr.training.checkpoints import save_training_checkpoint
                from dwr.training.controls import configure_source_control, source_control_step
                from dwr.training.source_pilot import _batch_noise, _collate
                from dwr.training.sampler import SourceSampler
                alternate = fixture_host(2701)
                control = configure_source_control(alternate, steps=0)
                groups = [{"params": list(alternate.readers.parameters()), "name": "readers"}]
                optimizer = torch.optim.AdamW(groups, lr=0.0003, betas=(0.9, 0.999),
                    eps=1e-8, weight_decay=0.01, amsgrad=False, capturable=False,
                    differentiable=False, foreach=False, fused=False, maximize=False)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lr_lambda=lambda index: min(1.0, (index + 1) / 8))
                train_ids = tuple(row.example_id for row in train)
                sampler = SourceSampler(train_ids, seed=20261217)
                noise_generator = torch.Generator(device="cpu").manual_seed(20261218)
                example_id = sampler.next_ids(1)[0]
                example = next(row for row in train if row.example_id == example_id)
                config = fixture_registration("K0")["config"]
                batch = _collate(example, FixtureTokenizer(), TokenizerIdentity("fixture", "v1"), config,
                    next(alternate.parameters()).device)
                noise = _batch_noise(alternate, batch, noise_generator, next(alternate.parameters()).device)
                result = source_control_step(alternate, batch, noise, control=control,
                    optimizer=optimizer, scheduler=scheduler, gradient_clip_norm=1.0)
                progress = {"optimizer_step": 1, "examples_seen": 1,
                    "supervised_tokens_seen": result["supervised_tokens"]}
                digest, _ = save_training_checkpoint(target, host=alternate, optimizer=optimizer,
                    scheduler=scheduler, scaler=None, sampler=sampler, noise_generator=noise_generator,
                    identities=_IDENTITIES, progress=progress)
                injected["collision_sha"] = digest
                injected["done"] = True

            with self.assertRaisesRegex(Exception, "unrecognized existing checkpoint path"):
                run_pilot_arm(fixture_host(2711), FixtureTokenizer(), TokenizerIdentity("fixture", "v1"),
                    train, validation, registration=fixture_registration("K0"), arm_id="K0",
                    identities=_IDENTITIES, metadata=metadata, output_dir=segment,
                    check=lambda: None, on_event=event,
                    before_complete=lambda _: self.fail("collision must fail before evaluation"))
            self.assertTrue(injected["done"])
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), injected["collision_sha"])
            self.assertEqual((root / "recovery.latest.json").read_bytes(), injected["step0_pointer"])
            cursor = json.loads((root / "recovery.latest.json").read_text(encoding="utf-8"))
            self.assertEqual((root / cursor["checkpoint_path"]).read_bytes(), injected["step0_cursor"])
            self.assertEqual(far_orphan.read_bytes(), b"uncommitted far-step orphan")

    def test_audit_callback_mutation_and_exception_is_reported_with_state_guard(self):
        original_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, original_threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); segment = root / "segment-000"; segment.mkdir()
            host = fixture_host(881)
            train = tuple(sorted(fixture_examples(5, "source_train", start=5), key=lambda row: row.example_id))
            validation = fixture_examples(5)
            metadata = {"training": fixture_metadata(train), "validation": fixture_metadata(validation)}
            parameter = next(host.readers.parameters())
            before = parameter.detach().clone()
            def mutate_then_fail(_payload):
                with torch.no_grad():
                    parameter.add_(0.125)
                raise LookupError("injected live-audit callback failure")
            with self.assertRaisesRegex(LookupError, "injected live-audit callback failure") as raised:
                run_pilot_arm(host, FixtureTokenizer(), TokenizerIdentity("fixture", "v1"),
                    train, validation, registration=fixture_registration("K0"), arm_id="K0",
                    identities=_IDENTITIES, metadata=metadata, output_dir=segment,
                    check=lambda: None, on_event=lambda *_: None, before_complete=mutate_then_fail)
            self.assertIsInstance(raised.exception.__cause__, RuntimeError)
            self.assertIn("changed live training state", str(raised.exception.__cause__))
            self.assertFalse(torch.equal(parameter, before))

    def test_initial_evaluation_callback_failure_restores_mixed_modes_and_return_dict(self):
        original_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, original_threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            segment = root / "segment-000"; segment.mkdir()
            host = fixture_host(877)
            host.train(); next(iter(host.readers.values())).eval()
            host.backbone.config.return_dict = False
            modes = {module: module.training for module in host.modules()}
            train = tuple(sorted(fixture_examples(5, "source_train", start=5), key=lambda row: row.example_id))
            validation = fixture_examples(5)
            metadata = {"training": fixture_metadata(train), "validation": fixture_metadata(validation)}
            def fail_after_persisted_prediction(kind, payload):
                if kind == "evaluation_case_started":
                    self.started_attempt = payload["attempt"]
                    self.started_stage = payload["stage"]
                if kind == "prediction_completed":
                    raise RuntimeError("injected evaluation callback failure")
            with self.assertRaisesRegex(RuntimeError, "injected evaluation callback failure"):
                run_pilot_arm(host, FixtureTokenizer(), TokenizerIdentity("fixture", "v1"),
                    train, validation, registration=fixture_registration("K0"), arm_id="K0",
                    identities=_IDENTITIES, metadata=metadata, output_dir=segment,
                    check=lambda: None, on_event=fail_after_persisted_prediction,
                    before_complete=lambda payload: self.fail("audit must not run after callback failure"))
            self.assertEqual({module: module.training for module in host.modules()}, modes)
            self.assertFalse(host.backbone.config.return_dict)
            self.assertTrue((segment / "stages" / f"{self.started_stage}-{self.started_attempt}.jsonl").is_file())

    def test_all_three_arms_train_and_commit_semantic_stage_records(self):
        caller_python, caller_numpy = random.getstate(), np.random.get_state()
        caller_torch = torch.get_rng_state().clone()
        caller_threads = torch.get_num_threads()
        self.addCleanup(random.setstate, caller_python)
        self.addCleanup(np.random.set_state, caller_numpy)
        self.addCleanup(torch.set_rng_state, caller_torch)
        self.addCleanup(torch.set_num_threads, caller_threads)
        torch.set_num_threads(1)
        outcomes = {}
        for arm_id in ("K0", "K1", "K4"):
            with self.subTest(arm=arm_id), tempfile.TemporaryDirectory() as temp:
                arm_root = Path(temp)
                segment = arm_root / "segment-000"
                segment.mkdir()
                host = fixture_host()
                host.train()
                next(iter(host.readers.values())).eval()
                host.backbone.config.return_dict = False
                caller_modes = {module: module.training for module in host.modules()}
                caller_return_dict = host.backbone.config.return_dict
                control = configure_source_control(host, steps={"K0": 0, "K1": 1, "K4": 4}[arm_id])
                original = {name: value.detach().clone() for name, value in host.backbone.state_dict().items()}
                initial_module_state = _learned_states(host)
                train = tuple(sorted(fixture_examples(5, "source_train", start=5), key=lambda row: row.example_id))
                validation = fixture_examples(5)
                metadata = {"training": fixture_metadata(train), "validation": fixture_metadata(validation)}
                audit_before = _frozen_ref(arm_root, "initial-frozen.json", host, arm_id)
                events = []
                attempts = {}
                transition_previous = None
                shuffle_previous = None

                def event(kind, payload):
                    nonlocal transition_previous, shuffle_previous
                    events.append((kind, copy.deepcopy(payload)))
                    if kind == "recovery_committed":
                        if payload["phase"] == "final_evaluation_pending" and transition_previous is None:
                            pointer = json.loads((arm_root / "recovery.latest.json").read_text(encoding="utf-8"))
                            persisted = json.loads((arm_root / pointer["metadata_path"]).read_text(encoding="utf-8"))
                            if "initial" in persisted["facts"]["stage_receipts"] and \
                                    "final" not in persisted["facts"]["stage_receipts"]:
                                transition_previous = {"phase": persisted["phase"],
                                    "progress": persisted["progress"], "facts": persisted["facts"],
                                    "checkpoint_sha256": persisted["checkpoint_sha256"]}
                        if payload["phase"] == "diagnostic_shuffle_pending" and shuffle_previous is None:
                            pointer = json.loads((arm_root / "recovery.latest.json").read_text(encoding="utf-8"))
                            persisted = json.loads((arm_root / pointer["metadata_path"]).read_text(encoding="utf-8"))
                            if "noise" in persisted["facts"]["stage_receipts"] and \
                                    "shuffle" not in persisted["facts"]["stage_receipts"]:
                                shuffle_previous = {"phase": persisted["phase"],
                                    "progress": persisted["progress"], "facts": persisted["facts"],
                                    "checkpoint_sha256": persisted["checkpoint_sha256"]}
                    elif kind == "evaluation_case_started":
                        attempts[(payload["stage"], payload["example_id"])] = payload["attempt"]
                    elif kind == "prediction_completed":
                        attempt = attempts[(payload["stage"], payload["example_id"])]
                        durable_rows = segment / "stages" / f"{payload['stage']}-{attempt}.jsonl"
                        self.assertTrue(durable_rows.is_file())
                        raw = durable_rows.read_bytes()
                        self.assertTrue(raw.endswith(b"\n"))
                        self.assertEqual(json.loads(raw.splitlines()[-1])["example_id"], payload["example_id"])
                    elif kind == "stage_committed":
                        cursor = json.loads((arm_root / "recovery.latest.json").read_text(encoding="utf-8"))
                        metadata_path = arm_root / cursor["metadata_path"]
                        persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
                        self.assertEqual(persisted["facts"]["stage_receipts"].get(payload["stage"]),
                            payload["receipt"])

                def before_complete(payload):
                    from dwr.training.source_pilot import _canonical
                    initial_doc = json.loads((arm_root / audit_before["path"]).read_text(encoding="utf-8"))
                    final_doc = {"schema": "dwr-source-pilot-controller-audit-v1", "arm_id": arm_id,
                        "identities": _IDENTITIES, "initial_frozen": audit_before,
                        "initial_state_sha256": payload["initial_state_sha256"],
                        "initial_checkpoint": payload["initial_checkpoint"],
                        "final_checkpoint": payload["final_checkpoint"],
                        "stage_receipts": payload["stage_receipts"], "bank": payload["bank"],
                        "original_parameter_hashes": initial_doc["original_parameter_hashes"],
                        "details": {"counts": payload["counts"], "costs": payload["costs"]}}
                    raw = _canonical(final_doc)
                    audit_path = arm_root / "final-controller.json"
                    audit_path.write_bytes(raw)
                    audit_after = {"path": audit_path.name, "sha256": hashlib.sha256(raw).hexdigest()}
                    return {"initial_frozen": audit_before, "final_controller": audit_after}

                result = run_pilot_arm(host, FixtureTokenizer(), TokenizerIdentity("fixture", "v1"),
                    train, validation, registration=fixture_registration(arm_id), arm_id=arm_id,
                    identities=_IDENTITIES, metadata=metadata, output_dir=segment,
                    check=lambda: None, on_event=event, before_complete=before_complete)
                from dwr.training.source_pilot import _canonical, verify_pilot_stage_receipt
                verified_stages = {}
                for stage, reference in result["stage_receipts"].items():
                    receipt = _read_receipt(arm_root, reference)
                    verified_stages[stage] = verify_pilot_stage_receipt(arm_root, reference, arm_id=arm_id, stage=stage,
                        expected_step=0 if stage == "initial" else 2,
                        expected_ids=tuple(row.example_id for row in validation),
                        expected_checkpoint_sha256=receipt["checkpoint_sha256"],
                        identities=_IDENTITIES, metadata=metadata["validation"], examples=validation,
                        expected_bank_metadata_sha256=(result["bank"]["sha256"]
                            if arm_id == "K4" and stage != "initial" else None),
                        expected_bank_reference=(result["bank"]
                            if arm_id == "K4" and stage != "initial" else None))
                from dwr.training.pilot_recovery import PilotRecoveryStore, _verify_audit_refs, _verify_final_result
                verified_cursor = PilotRecoveryStore.verify_descriptor(result["recovery_descriptor"],
                    _IDENTITIES, arm_id=arm_id, selected_ids=tuple(row.example_id for row in validation),
                    metadata=metadata["validation"], examples=validation, steps=2)
                facts = verified_cursor["facts"]
                if arm_id in {"K0", "K1", "K4"}:
                    self.assertIsNotNone(transition_previous)
                    attempted = copy.deepcopy(transition_previous["facts"])
                    attempted["stage_receipts"]["initial"] = result["stage_receipts"]["final"]
                    attempted["stage_receipts"]["final"] = result["stage_receipts"]["final"]
                    transition_store = PilotRecoveryStore(arm_root, segment, _IDENTITIES, arm_id=arm_id,
                        selected_ids=tuple(row.example_id for row in validation),
                        metadata=metadata["validation"], examples=validation, steps=2)
                    next_phase = "diagnostic_zero_pending" if arm_id == "K4" else "final_evaluation_pending"
                    self.assertFalse(transition_store._transition(transition_previous,
                        transition_previous["progress"], next_phase, attempted,
                        transition_previous["checkpoint_sha256"]))
                _verify_audit_refs(arm_root, facts["frozen_audit_refs"], arm_id=arm_id,
                    identities=_IDENTITIES, initial_state_sha256=facts["initial_state_sha256"],
                    receipts=facts["stage_receipts"], bank=facts["bank"],
                    current_checkpoint={"path": verified_cursor["checkpoint_path"],
                        "sha256": verified_cursor["checkpoint_sha256"]},
                    progress=verified_cursor["progress"], steps=2)
                # A byte-identical checkpoint relocation is authoritative as the current cursor,
                # while the committed final stage continues to point at its historical artifact.
                relocated_dir = arm_root / "segment-999" / "recovery"
                relocated_dir.mkdir(parents=True)
                relocated_path = relocated_dir / verified_cursor["checkpoint_path_absolute"].name
                relocated_path.write_bytes(verified_cursor["checkpoint_path_absolute"].read_bytes())
                relocated_ref = {"path": relocated_path.relative_to(arm_root).as_posix(),
                    "sha256": hashlib.sha256(relocated_path.read_bytes()).hexdigest()}
                relocated_audit = json.loads((arm_root /
                    facts["frozen_audit_refs"]["final_controller"]["path"]).read_text(encoding="utf-8"))
                relocated_audit["final_checkpoint"] = relocated_ref
                relocated_raw = _canonical(relocated_audit)
                relocated_audit_path = arm_root / "relocated-final-controller.json"
                relocated_audit_path.write_bytes(relocated_raw)
                relocated_refs = copy.deepcopy(facts["frozen_audit_refs"])
                relocated_refs["final_controller"] = {"path": relocated_audit_path.name,
                    "sha256": hashlib.sha256(relocated_raw).hexdigest()}
                _verify_audit_refs(arm_root, relocated_refs, arm_id=arm_id,
                    identities=_IDENTITIES, initial_state_sha256=facts["initial_state_sha256"],
                    receipts=facts["stage_receipts"], bank=facts["bank"],
                    current_checkpoint=relocated_ref, progress=verified_cursor["progress"], steps=2)
                if arm_id == "K4":
                    self.assertIsNotNone(shuffle_previous)
                    shuffle_facts = copy.deepcopy(shuffle_previous["facts"])
                    shuffle_facts["stage_receipts"]["shuffle"] = result["stage_receipts"]["shuffle"]
                    transition_store = PilotRecoveryStore(arm_root, segment, _IDENTITIES, arm_id="K4",
                        selected_ids=tuple(row.example_id for row in validation),
                        metadata=metadata["validation"], examples=validation, steps=2)
                    self.assertFalse(transition_store._transition(shuffle_previous,
                        shuffle_previous["progress"], "final_evaluation_pending", shuffle_facts,
                        shuffle_previous["checkpoint_sha256"]))
                _verify_final_result(facts["final_result"], arm_id=arm_id, steps=2,
                    stages=verified_stages, checkpoint_path=verified_cursor["checkpoint_path_absolute"],
                    initial_state_sha256=facts["initial_state_sha256"])
                final_audit_path = arm_root / facts["frozen_audit_refs"]["final_controller"]["path"]
                final_audit = json.loads(final_audit_path.read_text(encoding="utf-8"))
                swapped_stages = copy.deepcopy(final_audit["stage_receipts"])
                swapped_stages["initial"] = swapped_stages["final"]
                for field, value in (("arm_id", "K9"), ("stage_receipts", swapped_stages),
                        ("original_parameter_hashes",
                        {**final_audit["original_parameter_hashes"], "forged": "0" * 64})):
                    forged = copy.deepcopy(final_audit); forged[field] = value
                    forged_raw = _canonical(forged)
                    forged_path = arm_root / f"forged-audit-{field}.json"; forged_path.write_bytes(forged_raw)
                    forged_refs = copy.deepcopy(facts["frozen_audit_refs"])
                    forged_refs["final_controller"] = {"path": forged_path.name,
                        "sha256": hashlib.sha256(forged_raw).hexdigest()}
                    with self.assertRaises(Exception):
                        _verify_audit_refs(arm_root, forged_refs, arm_id=arm_id, identities=_IDENTITIES,
                            initial_state_sha256=facts["initial_state_sha256"],
                            receipts=facts["stage_receipts"], bank=facts["bank"],
                            current_checkpoint={"path": verified_cursor["checkpoint_path"],
                                "sha256": verified_cursor["checkpoint_sha256"]},
                            progress=verified_cursor["progress"], steps=2)
                contradictory = copy.deepcopy(facts["final_result"])
                contradictory["final"] = copy.deepcopy(contradictory["final"])
                contradictory["final"]["vsr"] = min(1.0, contradictory["final"]["vsr"] + 0.1)
                with self.assertRaises(Exception):
                    _verify_final_result(contradictory, arm_id=arm_id, steps=2,
                        stages=verified_stages, checkpoint_path=verified_cursor["checkpoint_path_absolute"],
                        initial_state_sha256=facts["initial_state_sha256"])
                if arm_id == "K0":
                    resumed_segment = arm_root / "segment-001"; resumed_segment.mkdir()
                    mismatched = fixture_host(999)
                    parameter = next(mismatched.workspace.parameters())
                    with torch.no_grad():
                        parameter.add_(0.25)
                    mismatched_initial = parameter.detach().clone()
                    with self.assertRaisesRegex(Exception, "fresh host initial state differs"):
                        run_pilot_arm(mismatched, FixtureTokenizer(), TokenizerIdentity("fixture", "v1"),
                            train, validation, registration=fixture_registration(arm_id), arm_id=arm_id,
                            identities=_IDENTITIES, metadata=metadata, output_dir=resumed_segment,
                            check=lambda: None, on_event=lambda *_: None,
                            before_complete=lambda _: None, resume=result["recovery_descriptor"])
                    self.assertTrue(torch.equal(parameter, mismatched_initial))
                if arm_id == "K4":
                    from dwr.training.source_pilot import _canonical
                    bank_ref = result["bank"]
                    final_ref = result["stage_receipts"]["final"]

                    def reject_mutation(stage, mutator, receipt_mutator=None):
                        forged_ref = _rewrite_stage(arm_root, result["stage_receipts"][stage],
                            mutator, receipt_mutator)
                        forged_receipt = _read_receipt(arm_root, forged_ref)
                        with self.assertRaises(ValueError):
                            verify_pilot_stage_receipt(arm_root, forged_ref, arm_id="K4", stage=stage,
                                expected_step=2, expected_ids=tuple(row.example_id for row in validation),
                                expected_checkpoint_sha256=forged_receipt["checkpoint_sha256"],
                                identities=_IDENTITIES, metadata=metadata["validation"], examples=validation,
                                expected_bank_metadata_sha256=bank_ref["sha256"],
                                expected_bank_reference=bank_ref)

                    reject_mutation("final", lambda rows: rows[0].__setitem__("workspace_sha256", "0" * 64))
                    reject_mutation("zero", lambda rows: rows[0].__setitem__("workspace_sha256", "0" * 64))
                    reject_mutation("noise", lambda rows: rows[0].__setitem__("workspace_sha256", "0" * 64))
                    reject_mutation("shuffle", lambda rows: rows[0].__setitem__("donor_id",
                        next(eid for eid in _IDS[:5] if eid not in {rows[0]["example_id"], rows[0]["donor_id"]})))

                    def lower_final_count(rows):
                        for row in rows:
                            row["counts"]["workspace_updates"] -= 1
                    def sum_final_count(receipt, rows):
                        receipt["counts"]["workspace_updates"] = sum(
                            row["counts"]["workspace_updates"] for row in rows)
                    reject_mutation("final", lower_final_count, sum_final_count)

                    def forge_reader_gate(rows):
                        first_reader = next(iter(rows[0]["reader_measurements"]))
                        rows[0]["reader_measurements"][first_reader]["gate"] += 0.25
                    reject_mutation("final", forge_reader_gate)

                    from dwr.training.source_pilot import _combine_reader_measurements
                    def forge_reader_ids(rows):
                        for row in rows:
                            original_reader = next(iter(row["reader_measurements"]))
                            row["reader_measurements"] = {"999": row["reader_measurements"][original_reader]}
                    def rekey_reader_summary(receipt, rows):
                        receipt["reader_summary"] = _combine_reader_measurements(rows)
                    reject_mutation("final", forge_reader_ids, rekey_reader_summary)

                    from dwr.evaluation.source_pilot import _tensor_digest
                    bank_meta = json.loads((arm_root / bank_ref["path"]).read_text(encoding="utf-8"))
                    bank_tensor = torch.load(arm_root / bank_meta["tensor"]["path"], map_location="cpu", weights_only=True)
                    wrong_tensor = bank_tensor[:, :2, :4].contiguous()
                    tensor_raw_path = arm_root / "wrong-shape-bank.pt"
                    torch.save(wrong_tensor, tensor_raw_path)
                    bank_meta["shape"] = list(wrong_tensor.shape)
                    bank_meta["tensor"] = {"path": tensor_raw_path.name,
                        "sha256": hashlib.sha256(tensor_raw_path.read_bytes()).hexdigest()}
                    for index, row in enumerate(bank_meta["rows"]):
                        row["workspace_shape"] = list(wrong_tensor[index].shape)
                        row["workspace_sha256"] = _tensor_digest(wrong_tensor[index])
                    wrong_bank_raw = _canonical(bank_meta)
                    wrong_bank_path = arm_root / "wrong-shape-bank.metadata.json"
                    wrong_bank_path.write_bytes(wrong_bank_raw)
                    wrong_bank_ref = {"path": wrong_bank_path.name,
                        "sha256": hashlib.sha256(wrong_bank_raw).hexdigest()}
                    wrong_final_ref = _rewrite_stage(arm_root, final_ref,
                        lambda rows: [row.__setitem__("bank_metadata_sha256", wrong_bank_ref["sha256"]) for row in rows],
                        lambda receipt, rows: receipt.__setitem__("bank_metadata_sha256", wrong_bank_ref["sha256"]))
                    wrong_final = _read_receipt(arm_root, wrong_final_ref)
                    with self.assertRaises(ValueError):
                        verify_pilot_stage_receipt(arm_root, wrong_final_ref, arm_id="K4", stage="final",
                            expected_step=2, expected_ids=tuple(row.example_id for row in validation),
                            expected_checkpoint_sha256=wrong_final["checkpoint_sha256"], identities=_IDENTITIES,
                            metadata=metadata["validation"], examples=validation,
                            expected_bank_metadata_sha256=wrong_bank_ref["sha256"],
                            expected_bank_reference=wrong_bank_ref)
                if arm_id == "K0":
                    reference = result["stage_receipts"]["initial"]
                    receipt = _read_receipt(arm_root, reference)
                    receipt["metrics"]["valid_denominator"] = 999
                    altered = _canonical(receipt)
                    altered_path = arm_root / "altered-receipt.json"
                    altered_path.write_bytes(altered)
                    with self.assertRaises(ValueError):
                        verify_pilot_stage_receipt(arm_root,
                            {"path": altered_path.name, "sha256": hashlib.sha256(altered).hexdigest()},
                            arm_id="K0", stage="initial", expected_step=0,
                            expected_ids=tuple(row.example_id for row in validation),
                            expected_checkpoint_sha256=receipt["checkpoint_sha256"], identities=_IDENTITIES,
                            metadata=metadata["validation"], examples=validation)
                    with self.assertRaises(ValueError):
                        verify_pilot_stage_receipt(arm_root,
                            {"path": reference["path"], "sha256": "0" * 64},
                            arm_id="K0", stage="initial", expected_step=0,
                            expected_ids=tuple(row.example_id for row in validation),
                            expected_checkpoint_sha256=receipt["checkpoint_sha256"], identities=_IDENTITIES,
                            metadata=metadata["validation"], examples=validation)
                    records_ref = receipt["records"]
                    records_path = arm_root / records_ref["path"]
                    lines = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
                    previous_calls = lines[0]["counts"]["greedy_answer_forwards"]
                    lines[0]["score"]["termination"] = "token_cap"
                    lines[0]["score"]["generated_tokens"] = 127
                    lines[0]["score"]["valid"] = False
                    lines[0]["generated_token_ids"] = [0] * 127
                    lines[0]["counts"]["greedy_answer_forwards"] = 127
                    bad_records = b"".join(_canonical(row) for row in lines)
                    bad_records_path = arm_root / "bad-cap.jsonl"
                    bad_records_path.write_bytes(bad_records)
                    receipt["records"] = {"path": bad_records_path.name,
                        "sha256": hashlib.sha256(bad_records).hexdigest()}
                    receipt["counts"]["greedy_answer_forwards"] += 127 - previous_calls
                    bad_receipt = _canonical(receipt)
                    bad_receipt_path = arm_root / "bad-cap-receipt.json"
                    bad_receipt_path.write_bytes(bad_receipt)
                    with self.assertRaises(ValueError):
                        verify_pilot_stage_receipt(arm_root,
                            {"path": bad_receipt_path.name, "sha256": hashlib.sha256(bad_receipt).hexdigest()},
                            arm_id="K0", stage="initial", expected_step=0,
                            expected_ids=tuple(row.example_id for row in validation),
                            expected_checkpoint_sha256=receipt["checkpoint_sha256"], identities=_IDENTITIES,
                            metadata=metadata["validation"], examples=validation)
                self.assertEqual(result["optimizer_step"], 2)
                self.assertEqual(result["status"], "PILOT_ARM_COMPLETE")
                self.assertEqual({module: module.training for module in host.modules()}, caller_modes)
                self.assertEqual(host.backbone.config.return_dict, caller_return_dict)
                self.assertEqual([row["example_id"] for row in result["updates"]],
                    [event[1]["example_id"] for event in events if event[0] == "update_completed"])
                self.assertEqual(set(result["stage_receipts"]),
                    {"initial", "final"} | ({"zero", "noise", "shuffle"} if arm_id == "K4" else set()))
                self.assertEqual(tuple(p.grad for p in host.parameters()), tuple(None for _ in host.parameters()))
                self.assertTrue(all(torch.equal(original[name], value)
                                    for name, value in host.backbone.state_dict().items()))
                if arm_id == "K0":
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                        initial_module_state[0].values(), _learned_states(host)[0].values())))
                else:
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
                        initial_module_state[0].values(), _learned_states(host)[0].values())))
                outcomes[arm_id] = result
        self.assertEqual(set(outcomes), {"K0", "K1", "K4"})

    def test_128_token_boundary_and_false_stage_denominator_are_strict(self):
        from dwr.training.source_pilot import _validate_stage
        self.assertTrue(_validate_stage("K0", "initial", 0, _IDS[:5], None))
        with self.assertRaises(ValueError):
            _validate_stage("K0", "initial", 1, _IDS[:5], None)


if __name__ == "__main__":
    unittest.main()
