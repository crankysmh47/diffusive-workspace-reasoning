"""Scoped CPU contract tests for the closed GPU feasibility probe."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from dwr.experiments import gpu_feasibility as probe

ROOT = pathlib.Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - optional ML environment
    if error.name != "torch":
        raise
    torch = None
if torch is not None:
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except ModuleNotFoundError as error:  # pragma: no cover - optional backbone environment
        if error.name != "transformers":
            raise
        Qwen2Config = Qwen2ForCausalLM = None
else:
    Qwen2Config = Qwen2ForCausalLM = None


class GPUFeasibilityValidationTests(unittest.TestCase):
    def test_startup_repair_registration_changes_only_version_output_and_source_base(self):
        original = json.loads((ROOT / "configs/models/pretrained_gpu_feasibility_v1.json").read_text(encoding="utf-8"))
        current = json.loads((ROOT / "configs/models/pretrained_gpu_feasibility_startup_repair_v2.json").read_text(encoding="utf-8"))
        for key in ("config_version", "output_dir", "source_base_commit"):
            original[key] = current[key]
        self.assertEqual(current, original)

    def test_registered_config_has_exact_bytes_and_rejects_mutation(self):
        raw = (ROOT / "configs/models/pretrained_gpu_feasibility_startup_repair_v2.json").read_bytes()
        config = probe.validate_config_bytes(raw)
        self.assertEqual(config["config_version"], "dwr-pretrained-gpu-feasibility-startup-repair-v2")
        with self.assertRaisesRegex(ValueError, "registered SHA-256"):
            probe.validate_config_bytes(raw + b" ")

    def test_config_object_rejects_unknown_and_null_fields(self):
        value = json.loads((ROOT / "configs/models/pretrained_gpu_feasibility_startup_repair_v2.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "unknown or missing"):
            probe.validate_config_object(dict(value, unregistered=True))
        malformed = json.loads(json.dumps(value))
        malformed["optimizer"]["epsilon"] = None
        with self.assertRaisesRegex(ValueError, "null is forbidden"):
            probe.validate_config_object(malformed)

    def test_registered_paths_reject_aliases_traversal_and_existing_output(self):
        for path in ("../runs/out", "C:/runs/out", "/runs/out", "runs\\out", "runs/./out"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                probe.validate_relative_path(path)
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            expected = "runs/pretrained-gpu-feasibility-20261015-startup-repair-v2"
            (root / "runs").mkdir()
            self.assertEqual(probe.validate_output_directory(root, expected), root / expected)
            (root / expected).mkdir()
            with self.assertRaises(FileExistsError):
                probe.validate_output_directory(root, expected)

    def test_failure_classification_keeps_preflight_and_execution_distinct(self):
        self.assertEqual(probe.classify_failure(RuntimeError("hardware UUID mismatch"), phase="preflight"),
                         "ENVIRONMENT_BLOCKED")
        self.assertEqual(probe.classify_failure(RuntimeError("nonfinite retained state"), phase="step"),
                         "FEASIBILITY_FAILED")

    def test_resource_guard_enforces_rss_and_wall_caps(self):
        from dwr.tasks.run_support import ResourceGuard

        with self.assertRaisesRegex(MemoryError, "resident memory cap"):
            ResourceGuard(10, 1, memory_reader=lambda: 2 * 1024 * 1024).check()
        tick = [0.0]
        guard = ResourceGuard(1, 2, clock=lambda: tick[0], memory_reader=lambda: 1)
        tick[0] = 1.0
        with self.assertRaisesRegex(TimeoutError, "wall-time cap"):
            guard.check()

    def test_probe_guard_uses_process_high_water_and_fails_closed_when_unsupported(self):
        memory = {"current": 2 * 1024 * 1024, "high_water": 2 * 1024 * 1024}
        memory["current"] = 512 * 1024  # A transient excursion has ended before the check.
        guard = probe.make_probe_guard(900, 1, peak_reader=lambda: memory["high_water"])
        with self.assertRaisesRegex(MemoryError, "resident memory cap"):
            guard.check()
        with mock.patch.object(probe.os, "name", "posix"):
            with self.assertRaisesRegex(RuntimeError, "high-water.*unsupported"):
                probe._process_peak_working_set_bytes()
        if probe.os.name == "nt":
            self.assertGreater(probe._process_peak_working_set_bytes(), 0)

    def test_numerics_and_sdpa_policy_are_read_back_and_enforced(self):
        """Injected policy objects verify control flow only; no CUDA work occurs."""
        from types import SimpleNamespace

        config = probe.validate_config_bytes((ROOT / "configs/models/pretrained_gpu_feasibility_startup_repair_v2.json").read_bytes())
        state = {"flash": False, "efficient": False, "math": False, "cudnn": True,
                 "tf32_matmul": True, "tf32_cudnn": True, "benchmark": True,
                 "precision": "highest", "deterministic": False}
        backend_cuda = SimpleNamespace(
            enable_flash_sdp=lambda value: state.__setitem__("flash", value),
            enable_mem_efficient_sdp=lambda value: state.__setitem__("efficient", value),
            enable_math_sdp=lambda value: state.__setitem__("math", value),
            enable_cudnn_sdp=lambda value: state.__setitem__("cudnn", value),
            flash_sdp_enabled=lambda: state["flash"],
            mem_efficient_sdp_enabled=lambda: state["efficient"],
            math_sdp_enabled=lambda: state["math"],
            cudnn_sdp_enabled=lambda: state["cudnn"],
            matmul=SimpleNamespace(allow_tf32=True),
        )
        backend_cudnn = SimpleNamespace(allow_tf32=True, benchmark=True)
        fake_torch = SimpleNamespace(
            backends=SimpleNamespace(cuda=backend_cuda, cudnn=backend_cudnn),
            set_default_dtype=lambda value: None, float32="float32",
            get_default_dtype=lambda: "float32",
            set_float32_matmul_precision=lambda value: state.__setitem__("precision", value),
            get_float32_matmul_precision=lambda: state["precision"],
            use_deterministic_algorithms=lambda value: state.__setitem__("deterministic", value),
            are_deterministic_algorithms_enabled=lambda: state["deterministic"],
        )
        policy = probe._set_numerics(config, torch_module=fake_torch)
        self.assertEqual(policy["sdpa"], {"flash": True, "memory_efficient": True,
                                           "math": True, "cudnn": False})
        self.assertFalse(policy["tf32_matmul"] or policy["tf32_cudnn"] or policy["cudnn_benchmark"])
        backend_cuda.flash_sdp_enabled = lambda: False
        with self.assertRaisesRegex(RuntimeError, "actual SDPA policy"):
            probe._read_numerics_policy(config, fake_torch)

    def test_controller_records_before_hardware_or_candidate(self):
        order = []
        result = probe._controller(
            record_initial=lambda: order.append("initial"),
            record_access=lambda: order.append("access"),
            query_hardware=lambda: (order.append("hardware") or {"device": "mock-only"}),
            run_candidate=lambda hardware: (order.append("candidate") or hardware["device"]),
        )
        self.assertEqual(result, "mock-only")
        self.assertEqual(order, ["initial", "access", "hardware", "candidate"])

    def test_preconditions_check_only_receipt_metadata_and_opaque_artifact_hashes(self):
        config = probe.validate_config_bytes((ROOT / "configs/models/pretrained_gpu_feasibility_startup_repair_v2.json").read_bytes())
        result = probe._data_preconditions(ROOT, config)
        self.assertEqual(result["token_audit_status"], "PASS_SOURCE_TOKEN_CAPS")
        self.assertFalse(result["task_semantics_read"])

    def test_precondition_receipts_fail_closed_on_status_counts_and_caps(self):
        config = probe.validate_config_bytes((ROOT / "configs/models/pretrained_gpu_feasibility_startup_repair_v2.json").read_bytes())
        conditions = config["data_preconditions"]
        counts = conditions["allowed_source_counts"]
        export_terminal = {"status": "completed"}
        audit_terminal = {"status": "completed"}
        export_summary = {"exported_counts": counts}
        audit_summary = {
            "status": conditions["token_audit_status"], "counts": counts,
            "protected_semantic_payload_access": False,
            "maxima": {split: {"prompt_tokens": 407, "answer_tokens_including_eos": 34,
                               "total_tokens": 441} for split in counts},
        }
        probe._validate_data_receipts(conditions, config["synthetic_input"], export_terminal,
                                      audit_terminal, export_summary, audit_summary)
        with self.assertRaisesRegex(ValueError, "did not pass"):
            probe._validate_data_receipts(conditions, config["synthetic_input"], export_terminal,
                {"status": "failed"}, export_summary, audit_summary)
        with self.assertRaisesRegex(ValueError, "count-mismatched"):
            probe._validate_data_receipts(conditions, config["synthetic_input"], export_terminal,
                audit_terminal, {"exported_counts": {"source_train": 1}}, audit_summary)
        bad_caps = json.loads(json.dumps(audit_summary))
        bad_caps["maxima"]["source_train"]["total_tokens"] = 1153
        with self.assertRaisesRegex(ValueError, "exceeds registered limit"):
            probe._validate_data_receipts(conditions, config["synthetic_input"], export_terminal,
                audit_terminal, export_summary, bad_caps)

    def test_import_does_not_require_optional_torch(self):
        script = r'''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("blocked for optional-dependency test", name="torch")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import dwr.experiments.gpu_feasibility
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


class GPUFeasibilityControllerRecordTests(unittest.TestCase):
    """Real controller and RunRecord lifecycle with external execution boundaries stubbed."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.config_relative = pathlib.Path("configs/models/pretrained_gpu_feasibility_startup_repair_v2.json")
        self.config_path = self.root / self.config_relative
        self.config_path.parent.mkdir(parents=True)
        source_config = ROOT / self.config_relative
        self.config_path.write_bytes(source_config.read_bytes())
        (self.root / "runs").mkdir()
        for args in (
            ["init", "--quiet"],
            ["config", "user.name", "DWR CPU Fixture"],
            ["config", "user.email", "dwr-fixture@example.invalid"],
            ["add", str(self.config_relative)],
            ["commit", "--quiet", "-m", "controller fixture"],
        ):
            subprocess.run(["git", *args], cwd=self.root, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.output = self.root / self.config["output_dir"]

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, *, hardware_error=None):
        source_commit = "a" * 40

        def candidate(_root, _config, _hardware, output, _started, _guard, _artifact_receipt):
            (output / "outcome.json").write_bytes(b'{"status":"PASS_FEASIBILITY"}\n')
            (output / "state.pt").write_bytes(b"CPU controller fixture only")
            return {"status": "PASS_FEASIBILITY", "checkpoint_roundtrip": {"sha256": "b" * 64},
                    "elapsed_seconds_before_terminal_record_finalization": 0.01}

        patches = [
            mock.patch.object(probe, "_repo_root", return_value=self.root),
            mock.patch.object(probe, "validate_clean_source",
                              return_value={"commit": source_commit, "clean": True}),
            mock.patch.object(probe, "_data_preconditions", return_value={
                "source_export_manifest_sha256": "c" * 64,
                "token_audit_manifest_sha256": "d" * 64,
                "task_semantics_read": False,
            }),
            mock.patch.object(probe, "_artifact_inventory", return_value={"manifest_sha256": "e" * 64}),
            mock.patch.object(probe, "_check_environment", return_value={"mocked": True}),
            mock.patch.object(probe, "_hardware_metadata", side_effect=hardware_error or {"mocked": True}),
            mock.patch.object(probe, "_production_candidate", side_effect=candidate),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            try:
                outcome = probe.run_probe(
                    self.config_path,
                    command=[sys.executable, "-m", "dwr.experiments.gpu_feasibility",
                             "--config", self.config_relative.as_posix()],
                )
            except RuntimeError:
                outcome = None
        from dwr.experiments.records import verify_artifacts
        terminal = verify_artifacts(self.output)
        initial = json.loads((self.output / "manifest.initial.json").read_text(encoding="utf-8"))
        events = [json.loads(line) for line in (self.output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        return outcome, terminal, initial, events

    def test_actual_controller_projects_registered_contract_into_real_run_record(self):
        outcome, terminal, initial, events = self._run()

        self.assertEqual(outcome["status"], "PASS_FEASIBILITY")
        self.assertEqual(terminal["status"], "completed")
        self.assertEqual(initial["config"]["sha256"], "840615e265a427946627f3320f667154e17bc56c53c15b6f90a453c10693144f")
        self.assertEqual(set(initial["identities"]), {"source", "config", "data", "model", "tokenizer", "schedule"})
        self.assertEqual({key: (value["name"], value["revision"])
                          for key, value in initial["identities"].items()}, {
            "source": ("dwr-source-commit", "a" * 40),
            "config": ("dwr-gpu-feasibility-config", self.config["config_version"]),
            "data": ("source-token-cap-audit", self.config["data_preconditions"]["token_audit_status"]),
            "model": (self.config["model"]["repo_id"], self.config["model"]["revision"]),
            "tokenizer": ("pinned-model-tokenizer-inventory", self.config["model"]["revision"]),
            "schedule": ("optimizer-schedule", self.config["schedule"]["type"]),
        })
        for identity in initial["identities"].values():
            self.assertEqual(set(identity), {"name", "revision", "sha256"})
            self.assertRegex(identity["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(initial["budgets"], {"wall_seconds": 900, "memory_mib": 12288, "attempts": 1})
        kinds = [event["kind"] for event in events]
        self.assertEqual(kinds[:3], ["initial_manifest_written", "precondition_access_started",
                                     "precondition_access_verified"])
        self.assertIn("probe_complete", kinds)
        self.assertEqual(terminal["event_count"], len(events))
        self.assertEqual(terminal["initial_manifest_sha256"], hashlib.sha256(
            (self.output / "manifest.initial.json").read_bytes()).hexdigest())
        self.assertEqual(terminal["events_sha256"], hashlib.sha256(
            (self.output / "events.jsonl").read_bytes()).hexdigest())

    def test_actual_controller_records_hardware_preflight_failure(self):
        _, terminal, initial, events = self._run(hardware_error=RuntimeError("mock hardware preflight failure"))

        self.assertEqual(initial["budgets"], {"wall_seconds": 900, "memory_mib": 12288, "attempts": 1})
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["counts"], {"ENVIRONMENT_BLOCKED": 1})
        self.assertEqual(terminal["event_count"], len(events))
        self.assertIn("probe_stopped", [event["kind"] for event in events])
        self.assertEqual(terminal["initial_manifest_sha256"], hashlib.sha256(
            (self.output / "manifest.initial.json").read_bytes()).hexdigest())
        self.assertEqual(terminal["events_sha256"], hashlib.sha256(
            (self.output / "events.jsonl").read_bytes()).hexdigest())
        self.assertTrue((self.output / "outcome.json").is_file())


@unittest.skipIf(torch is None or Qwen2Config is None, "install the optional backbone extra")
class GPUFeasibilityCPUIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def make_host(self, attention="eager"):
        config = Qwen2Config(
            vocab_size=32, hidden_size=12, intermediate_size=24, num_hidden_layers=3,
            num_attention_heads=3, num_key_value_heads=1, attention_dropout=0.0,
            tie_word_embeddings=True, pad_token_id=31, eos_token_id=31,
            _attn_implementation=attention,
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(73)
            backbone = Qwen2ForCausalLM(config).float()
        from dwr.models.qwen2 import Qwen2WorkspaceLM
        from dwr.models.workspace import FixedWorkspace

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(89)
            workspace = FixedWorkspace(12, num_slots=4, latent_dim=8, num_heads=2,
                                       max_steps=4).float()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(101)
            return Qwen2WorkspaceLM(backbone, workspace, prompt_layer=1,
                                    adapter_layers=(0, 1, 2), reader_heads=3)

    def settings(self):
        return {
            "algorithm": "local_CPU_torch_randint_prompt_then_answer_append_EOS_v1",
            "answer_tokens_including_eos": 3, "batch_size": 1,
            "embedding_vocab_size": 32, "eos_token_id": 31,
            "id_high_exclusive": 30, "id_low_inclusive": 1,
            "mask_policy": "all_valid_no_padding", "noise_requires_grad_diagnostic": True,
            "pad_token_id": 31,
            "prompt_tokens": 3, "total_tokens": 6,
        }

    def batch(self, seed=8):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return probe.make_probe_batch(self.settings(), generator=generator, vocab_size=32)

    def test_synthetic_batch_uses_local_rng_and_strict_prompt_answer_eos_framing(self):
        before = torch.get_rng_state().clone()
        first = self.batch()
        second = self.batch()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for name in ("prompt_ids", "prompt_mask", "decoder_ids", "decoder_mask", "labels"):
            self.assertTrue(torch.equal(getattr(first, name), getattr(second, name)))
        self.assertTrue(torch.equal(first.decoder_ids[:, :3], first.prompt_ids))
        self.assertTrue((first.labels[:, :3] == -100).all())
        self.assertTrue(torch.equal(first.labels[:, -1], torch.tensor([31])))
        self.assertTrue(first.prompt_mask.all() and first.decoder_mask.all())
        with self.assertRaisesRegex(TypeError, "explicit local CPU"):
            probe.make_probe_batch(self.settings(), generator=None, vocab_size=32)
        with self.assertRaisesRegex(ValueError, "unknown or missing"):
            probe.make_probe_batch(dict(self.settings(), extra=True), generator=torch.Generator(), vocab_size=32)

    def test_exact_registered_synthetic_input_reaches_maximum_framing(self):
        config = probe.validate_config_bytes((ROOT / "configs/models/pretrained_gpu_feasibility_startup_repair_v2.json").read_bytes())
        settings = config["synthetic_input"]
        batch = probe.make_probe_batch(
            settings, generator=torch.Generator(device="cpu").manual_seed(config["seeds"]["synthetic_ids"]),
            vocab_size=config["model"]["embedding_vocab_size"] if "embedding_vocab_size" in config["model"]
            else settings["embedding_vocab_size"],
        )
        self.assertTrue(settings["noise_requires_grad_diagnostic"])
        self.assertEqual(tuple(batch.prompt_ids.shape), (1, 1024))
        self.assertEqual(tuple(batch.decoder_ids.shape), (1, 1152))
        self.assertEqual(tuple(batch.labels.shape), (1, 1152))
        self.assertTrue((batch.labels[:, :1024] == -100).all())
        self.assertTrue(torch.equal(batch.labels[:, -1], torch.tensor([151643])))

    def test_cpu_initialization_seed_policy_does_not_call_global_manual_seed(self):
        before = torch.get_rng_state().clone()
        probe.seed_cpu_initialization(torch, 1234)
        actual = torch.rand(4)
        torch.set_rng_state(before)
        torch.random.default_generator.manual_seed(1234)
        expected = torch.rand(4)
        torch.set_rng_state(before)
        self.assertTrue(torch.equal(actual, expected))

    def test_tiny_actual_shared_step_and_exact_same_host_checkpoint_roundtrip(self):
        from dwr.training.sampler import SourceSampler

        host = self.make_host("eager")
        original = probe.hash_original_parameters(host)
        settings = {"optimizer": {"learning_rate": 1e-4, "betas": [0.9, 0.999],
                    "epsilon": 1e-8, "weight_decay": 0.01, "amsgrad": False,
                    "foreach": False, "fused": False, "capturable": False,
                    "differentiable": False, "maximize": False}}
        optimizer, scheduler = probe.create_optimizer(host, settings)
        noise_generator = torch.Generator(device="cpu").manual_seed(19)
        result = probe.run_shared_step(host, self.batch(), noise_generator, optimizer, scheduler,
                                       steps=4, expected_original_hashes=original)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
        self.assertEqual(result["supervised_tokens"], 3)
        self.assertEqual(len(result["retained_state_gradient_norms"]), 5)
        self.assertEqual(set(result["each_reader_gradient_group_norms"]), {"0", "1", "2"})
        self.assertTrue(all(value > 0 for value in result["each_reader_gradient_group_norms"].values()))
        self.assertEqual(result["optimizer_step"], 1)
        self.assertTrue(result["stage_seconds"]["prompt_encode_seconds"] >= 0)
        self.assertEqual(original, probe.hash_original_parameters(host))
        self.assertTrue(all(p.grad is None for p in host.backbone.parameters()))
        self.assertTrue(all(p.grad is not None for p in (*host.workspace.parameters(), *host.readers.parameters())))

        identity = {key: hashlib.sha256(key.encode("ascii")).hexdigest()
                    for key in ("source", "config", "data", "model", "tokenizer", "schedule")}
        sampler = SourceSampler((hashlib.sha256(b"tiny-synthetic-fixture").hexdigest(),), seed=29)
        with tempfile.TemporaryDirectory() as temp:
            report = probe.roundtrip_checkpoint(pathlib.Path(temp) / "state.pt", host=host,
                optimizer=optimizer, scheduler=scheduler, sampler=sampler,
                noise_generator=noise_generator, identities=identity,
                supervised_tokens=result["supervised_tokens"])
        self.assertTrue(report["exact_state_match"])
        self.assertTrue(report["sampler_destination_differed_before_restore"])
        self.assertTrue(report["synthetic_sampler_fixture_bookkeeping"])
        self.assertTrue(report["same_host_no_second_base"])
        self.assertTrue(report["original_parameter_bytes_unchanged"])
        self.assertFalse(report["deliberately_perturbed_rng_streams"]["torch_cuda"])
        self.assertEqual(report["progress"]["supervised_tokens_seen"], 3)

    def test_actual_tiny_qwen_eager_and_sdpa_match_causal_padding_and_backward(self):
        eager = self.make_host("eager")
        sdpa = self.make_host("sdpa")
        sdpa.load_state_dict(eager.state_dict())
        self.assertEqual(sdpa.backbone.config._attn_implementation, "sdpa")
        prompt = torch.tensor([[1, 2, 3], [4, 5, 31]])
        prompt_mask = torch.tensor([[True, True, True], [True, True, False]])
        decoder = torch.tensor([[1, 2, 3, 7, 8], [4, 5, 6, 9, 31]])
        decoder_mask = torch.tensor([[True, True, True, True, True], [True, True, True, True, False]])
        noise = torch.randn((2, 4, 8), generator=torch.Generator().manual_seed(17))
        outputs = []
        for host in (eager, sdpa):
            x_noise = noise.clone().requires_grad_(True)
            result = host(prompt, prompt_mask, decoder, decoder_mask,
                          noise=x_noise, steps=4, return_states=True)
            loss = result.logits.square().mean()
            loss.backward()
            outputs.append((result.logits.detach(), result.workspace.detach(), loss.detach(),
                            [p.grad.detach().clone() for p in (*host.workspace.parameters(), *host.readers.parameters())],
                            x_noise.grad.detach().clone()))
            changed = decoder.clone()
            changed[0, -1] = 10
            changed_result = host(prompt, prompt_mask, changed, decoder_mask,
                                  noise=noise.clone(), steps=4)
            self.assertTrue(torch.allclose(result.logits[0, :-1], changed_result.logits[0, :-1],
                                           atol=1e-6, rtol=1e-6))
            self.assertFalse(torch.allclose(result.logits[0, -1], changed_result.logits[0, -1],
                                            atol=1e-6, rtol=1e-6))
            single = host(prompt[1:2, :2], torch.ones((1, 2), dtype=torch.bool),
                          decoder[1:2, :4], torch.ones((1, 4), dtype=torch.bool),
                          noise=noise[1:2].clone(), steps=4)
            self.assertTrue(torch.allclose(result.workspace[1], single.workspace[0], atol=1e-6, rtol=1e-6))
            self.assertTrue(torch.allclose(result.logits[1, :4], single.logits[0], atol=1e-6, rtol=1e-6))
        for left, right in zip(outputs[0][:3], outputs[1][:3]):
            self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-6),
                            f"eager/SDPA max diff {float((left-right).abs().max())}")
        self.assertEqual(len(outputs[0][3]), len(outputs[1][3]))
        for left, right in zip(outputs[0][3], outputs[1][3]):
            self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(outputs[0][4], outputs[1][4], atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
