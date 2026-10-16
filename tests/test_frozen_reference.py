"""Closed frozen-reference runner controller and tiny native CPU regressions."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

from dwr.experiments import frozen_reference as runner
from dwr.experiments.records import RunRecord, verify_artifacts
from dwr.evaluation.generation import GenerationResult
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, serialize_model_input, serialize_task

ROOT = Path(__file__).resolve().parents[1]


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=15).stdout


def task() -> ConstraintTask:
    return ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
                          (Constraint(RelationKind.EQ, ("v0", "v1")),
                           Constraint(RelationKind.NEQ, ("v1", "v2"))),
                          (GivenAssignment("v0", 1),))


class CohortAndPromptTests(unittest.TestCase):
    def setUp(self):
        self.cohort = json.loads((ROOT / "configs/splits/frozen_reference_source_validation_cohort_v1.json").read_text(encoding="utf-8"))

    def rows(self):
        return [{"example_id": item["example_id"], "profile": item["profile"],
                 "fine_stratum": item["fine_stratum"], "prompt_tokens": item["prompt_tokens"],
                 "lineage_id": "a" * 64, "structure_signature": "b" * 64,
                 "split": "source_validation", "task": serialize_task(task())}
                for item in self.cohort["selected"]]

    def test_exact_closed_registration_maps_full_feasibility_proof_and_cohort(self):
        resolved = runner._load_registration(ROOT, "configs/models/frozen_source_reference_startup_repair_v2.json")
        self.assertEqual(resolved["config_sha256"], "e3a67600b9bada872bd4523f319378a2088e25824df82d90c44d2fb00b42144f")
        self.assertEqual(resolved["cohort_sha256"], "bca9505b817a9b75b9d7aba33710ddc17b47596bdb9a9772924348d15b4b64bd")
        self.assertEqual(resolved["feasibility_config"]["config_version"], "dwr-pretrained-gpu-feasibility-startup-repair-v2")
        self.assertEqual(resolved["selected"], tuple(self.cohort["selected"]))
        with self.assertRaisesRegex(ValueError, "only the registered"):
            runner._load_registration(ROOT, "configs/models/frozen_source_reference_v1.json")

    def test_source_rows_are_exactly_cohort_ordered_and_source_validation_only(self):
        selected = runner._select_source_rows(self.rows(), self.cohort)
        self.assertEqual([row["example_id"] for row in selected],
                         [item["example_id"] for item in self.cohort["selected"]])
        bad = self.rows()
        bad[0] = {**bad[0], "split": "protected_test"}
        with self.assertRaises(ValueError):
            runner._select_source_rows(bad, self.cohort)

    def test_reader_adapter_requests_only_registered_source_validation(self):
        config = json.loads((ROOT / "configs/models/frozen_source_reference_startup_repair_v2.json").read_text(encoding="utf-8"))
        calls = []
        def reader(path, **kwargs):
            calls.append((path, kwargs)); return ()
        rows = runner._read_validation_rows(ROOT, config, reader=reader)
        self.assertEqual(rows, ())
        self.assertEqual(calls[0][1], {"expected_manifest_sha256": config["data"]["source_export_manifest_sha256"],
                                       "split": "source_validation"})

    def test_mixed_duplicate_missing_and_mismatched_metadata_fail_closed(self):
        cases = []
        duplicate = self.rows(); duplicate[-1] = dict(duplicate[0]); cases.append(duplicate)
        cases.append(self.rows()[:-1])
        mismatch = self.rows(); mismatch[0] = {**mismatch[0], "profile": "neq"}; cases.append(mismatch)
        for rows in cases:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                runner._select_source_rows(rows, self.cohort)

    def test_prompt_framing_is_canonical_lf_without_special_tokens(self):
        class Tokenizer:
            def __init__(self): self.calls = []
            def encode(self, value, *, add_special_tokens):
                self.calls.append((value, add_special_tokens))
                return [ord(ch) % 17 for ch in value]
        tokenizer = Tokenizer()
        model_input, framed, token_ids, hashes = runner._frame_prompt(task(), tokenizer)
        self.assertEqual(model_input, serialize_model_input(task()))
        self.assertEqual(framed, model_input + "\n")
        self.assertEqual(tokenizer.calls, [(framed, False)])
        self.assertEqual(token_ids, tuple(ord(ch) % 17 for ch in framed))
        self.assertEqual(hashes["model_input_sha256"], hashlib.sha256(model_input.encode("utf-8")).hexdigest())
        self.assertEqual(hashes["framed_text_sha256"], hashlib.sha256(framed.encode("utf-8")).hexdigest())
        self.assertEqual(hashes["token_ids_sha256"], hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode("ascii")).hexdigest())

    def test_repetition_accounting_is_45_attempts_and_15_primary_examples(self):
        attempts = [{"example_id": str(i), "repetition": rep, "valid": False,
                     "token_ids": [1], "score": {"valid": False, "verifier_reason": "malformed_json",
                                                     "termination": "eos"}}
                    for rep in range(3) for i in range(15)]
        summary = runner._summarize_attempts(attempts, expected_examples=15, repetitions=3)
        self.assertEqual(summary["attempt_count"], 45)
        self.assertEqual(summary["quality"]["denominator"], 15)
        self.assertEqual(summary["quality"]["repetition"], 0)
        self.assertEqual(summary["quality"]["valid_count"], 0)
        attempts[-1]["token_ids"] = [9]
        with self.assertRaises(ValueError):
            runner._require_repeatability(attempts, expected_examples=15, repetitions=3)


@unittest.skipUnless(__import__("importlib").util.find_spec("torch") and __import__("importlib").util.find_spec("transformers"),
                     "optional ML dependencies are unavailable")
class TinyCpuEvaluationTests(unittest.TestCase):
    def test_actual_tiny_native_generation_flows_through_strict_scorer(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=17, hidden_size=12, intermediate_size=24,
                          num_hidden_layers=1, num_attention_heads=3, num_key_value_heads=1,
                          pad_token_id=16, eos_token_id=15, _attn_implementation="eager")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(83)
            model = Qwen2ForCausalLM(cfg).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        ids = torch.tensor([[1, 3, 4]], dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)
        class Tokenizer:
            def decode(self, _token_ids, **_kwargs): return "not json"
        result = runner._evaluate_one(model, ids, mask, task(), Tokenizer(), eos_token_id=15,
                                     max_new_tokens=1, max_total_tokens=4, check=None)
        self.assertIsInstance(result["generation"], GenerationResult)
        self.assertEqual(result["generation"].decoder_calls, 1)
        self.assertIn(result["score"]["verifier_reason"], {"valid", "malformed_json", "wrong_container",
                         "unknown_key", "missing_assignment", "unknown_variable", "missing_variable",
                         "non_integer_value", "value_out_of_domain", "given_mismatch", "constraint_violation"})

        class ValidTokenizer:
            def decode(self, _token_ids, **_kwargs):
                return '{"assignment":{"v0":1,"v1":1,"v2":0}}'
        capped = GenerationResult((7, 8), 15, False, True, 3, 2, 0, 0, 0.01)
        cap_score = runner.score_generation(task(), capped, ValidTokenizer())
        self.assertFalse(cap_score["valid"])
        self.assertEqual(cap_score["termination"], "token_cap")
        self.assertEqual(cap_score["verifier_reason"], "valid")


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "optional ML dependency is unavailable")
class ModelStartupOrderTests(unittest.TestCase):
    class ModelLoadReached(Exception):
        pass

    def _run_until_model_load(self, device, events):
        import torch

        class Helper:
            @staticmethod
            def _set_numerics(_full, *, torch_module):
                return {"fixture": True}

        reg = {"config": {"seeds": {"python_global": 1, "numpy_global": 2,
                                     "torch_cpu_global": 3, "torch_cuda_global": 4}}}
        def load(*_args):
            events.append("load")
            raise self.ModelLoadReached
        with patch.object(torch, "manual_seed"), \
             patch.object(runner, "_load_original_model", side_effect=load), \
             patch.object(runner.random, "seed"), \
             patch("numpy.random.seed"):
            return runner._run_model(Path("."), reg, Helper(), {"model": {}}, (), None, object(),
                                     lambda: None, device=device)

    def test_cuda_startup_synchronizes_exact_device_then_resets_before_model_load(self):
        import torch
        device = torch.device("cuda:0")
        events = []
        initialized = False

        def synchronize(actual):
            nonlocal initialized
            self.assertEqual(actual, device)
            events.append("synchronize")
            initialized = True

        def reset(actual):
            self.assertEqual(actual, device)
            events.append("reset")
            if not initialized:
                raise RuntimeError("Invalid device argument: allocator is uninitialized")

        with patch.object(torch.cuda, "synchronize", side_effect=synchronize), \
             patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=reset):
            with self.assertRaises(self.ModelLoadReached):
                self._run_until_model_load(device, events)

        self.assertEqual(events, ["synchronize", "reset", "load"])

    def test_cpu_startup_does_not_call_runner_cuda_startup_or_measurement_apis(self):
        import torch
        device = torch.device("cpu")
        events = []
        cuda_apis = ("synchronize", "reset_peak_memory_stats", "manual_seed_all",
                     "memory_allocated", "memory_reserved", "max_memory_allocated",
                     "max_memory_reserved", "get_rng_state_all", "set_rng_state_all")
        with patch.multiple(torch.cuda, **{name: unittest.mock.Mock(side_effect=AssertionError(name))
                                           for name in cuda_apis}):
            with self.assertRaises(self.ModelLoadReached):
                self._run_until_model_load(device, events)
        self.assertEqual(events, ["load"])


@unittest.skipUnless(__import__("importlib").util.find_spec("torch") and __import__("importlib").util.find_spec("transformers"),
                     "optional ML dependencies are unavailable")
class RawOriginalHashTests(unittest.TestCase):
    def test_raw_tiny_qwen_hashes_every_parameter_and_detects_mutation(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from dwr.experiments import gpu_feasibility as helper
        cfg = Qwen2Config(vocab_size=17, hidden_size=12, intermediate_size=24,
                          num_hidden_layers=1, num_attention_heads=3, num_key_value_heads=1)
        model = Qwen2ForCausalLM(cfg)
        first = runner._hash_original_parameters(model, helper)
        self.assertEqual(set(first), {name for name, _ in model.named_parameters()})
        self.assertEqual(first, runner._hash_original_parameters(model, helper))
        with torch.no_grad():
            next(model.parameters()).view(-1)[0].add_(1)
        self.assertNotEqual(first, runner._hash_original_parameters(model, helper))


class FeasibilityValidatorTests(unittest.TestCase):
    def setUp(self):
        import dwr.experiments.gpu_feasibility as helper
        self.helper = helper
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"; self.root.mkdir()
        git(self.root, "init", "-q"); git(self.root, "config", "user.name", "Runner Test")
        git(self.root, "config", "user.email", "runner@example.invalid")
        source_config = ROOT / "configs/models/pretrained_gpu_feasibility_startup_repair_v2.json"
        raw = source_config.read_bytes()
        self.config_path = self.root / "configs/models/pretrained_gpu_feasibility_startup_repair_v2.json"
        self.config_path.parent.mkdir(parents=True); self.config_path.write_bytes(raw)
        git(self.root, "add", "."); git(self.root, "commit", "-qm", "fixture")
        run_dir = self.root / "runs/feasibility"; run_dir.parent.mkdir()
        config_sha = hashlib.sha256(raw).hexdigest()
        self.record = RunRecord.create(run_dir, command=["python", "fixture"], config_path=self.config_path,
            expected_config_sha256=config_sha,
            identities={"fixture": {"name": "fixture", "revision": "1", "sha256": "a" * 64}},
            rng_streams={"python": 4}, settings={"fixture": True},
            budgets={"wall_seconds": 10, "memory_mib": 512, "attempts": 1}, source_root=self.root)
        self.outcome = {"status": "PASS_FEASIBILITY"}
        (run_dir / "outcome.json").write_text(json.dumps(self.outcome) + "\n", encoding="utf-8")
        self.record.finish("completed", counts={"fixture": 1}, artifacts=["outcome.json"])
        self.dependency = {"feasibility_config_path": "configs/models/pretrained_gpu_feasibility_startup_repair_v2.json",
                           "feasibility_run_dir": "runs/feasibility"}
        self.pins = {"expected_module_sha256": hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(),
                     "expected_config_sha256": config_sha,
                     "expected_manifest_sha256": hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest(),
                     "expected_outcome_sha256": hashlib.sha256((run_dir / "outcome.json").read_bytes()).hexdigest()}

    def tearDown(self): self.temp.cleanup()

    def validate(self, **overrides):
        return runner._validate_feasibility_receipts(self.root, self.dependency, self.helper,
                                                     **(self.pins | overrides))

    def test_actual_receipt_validator_accepts_real_completed_fixture_chain(self):
        result = self.validate()
        self.assertEqual(result["outcome"], self.outcome)
        self.assertEqual(verify_artifacts(self.root / "runs/feasibility")["status"], "completed")

    def test_actual_receipt_validator_rejects_module_config_manifest_outcome_and_pass_mismatches(self):
        for key in self.pins:
            wrong = dict(self.pins); wrong[key] = "0" * 64
            with self.subTest(pin=key), self.assertRaises(ValueError):
                runner._validate_feasibility_receipts(self.root, self.dependency, self.helper, **wrong)
        outcome_path = self.root / "runs/feasibility/outcome.json"
        # Create a second valid terminal chain whose outcome is not PASS, so the
        # semantic status gate is reached after artifact verification succeeds.
        run_dir = self.root / "runs/nonpass"
        record = RunRecord.create(run_dir, command=["python", "fixture"], config_path=self.config_path,
            expected_config_sha256=self.pins["expected_config_sha256"],
            identities={"fixture": {"name": "fixture", "revision": "1", "sha256": "b" * 64}},
            rng_streams={"python": 5}, settings={"fixture": True},
            budgets={"wall_seconds": 10, "memory_mib": 512, "attempts": 1}, source_root=self.root)
        outcome_path = run_dir / "outcome.json"
        outcome_path.write_text('{"status":"NOT_PASS"}\n', encoding="utf-8")
        record.finish("completed", counts={"fixture": 1}, artifacts=["outcome.json"])
        self.dependency["feasibility_run_dir"] = "runs/nonpass"
        self.pins["expected_manifest_sha256"] = hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()
        self.pins["expected_outcome_sha256"] = hashlib.sha256(outcome_path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "not PASS_FEASIBILITY"):
            self.validate()


class RealRunRecordLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"; self.root.mkdir()
        git(self.root, "init", "-q"); git(self.root, "config", "user.name", "Runner Test")
        git(self.root, "config", "user.email", "runner@example.invalid")
        (self.root / "source.py").write_bytes(b"source = 1\n")
        git(self.root, "add", "source.py"); git(self.root, "commit", "-qm", "base")
        self.config = self.root / "config.json"; self.config.write_bytes(b'{"fixture":true}\n')
        git(self.root, "add", "config.json"); git(self.root, "commit", "-qm", "config")
        self.output_parent = self.root / "runs"; self.output_parent.mkdir()

    def tearDown(self): self.temp.cleanup()

    def create_record(self, name):
        config_sha = hashlib.sha256(self.config.read_bytes()).hexdigest()
        return RunRecord.create(self.output_parent / name, command=["python", "-m", "test"],
            config_path=self.config, expected_config_sha256=config_sha,
            identities={"source": {"name": "source", "revision": "test", "sha256": "a" * 64}},
            rng_streams={"python": 1}, settings={"resolved": True},
            budgets={"wall_seconds": 30, "memory_mib": 512, "attempts": 1}, source_root=self.root)

    def test_real_create_finish_and_verify_success_lifecycle(self):
        record = self.create_record("success")
        record.append_event("precondition_access_started", {"protected_task_semantics_read": False})
        (record.output_dir / "summary.json").write_text('{"status":"completed"}\n', encoding="utf-8")
        record.finish("completed", counts={"attempts": 1}, artifacts=["summary.json"])
        self.assertEqual(verify_artifacts(record.output_dir)["status"], "completed")

    def test_real_terminal_budget_failure_keeps_failed_record_and_artifact_linkage(self):
        record = self.create_record("late-failure")
        (record.output_dir / "summary.partial.json").write_bytes(b'{"status":"pre_terminal"}\n')
        def fail(stage):
            if stage == "before_seal": raise TimeoutError("late cap")
        with self.assertRaisesRegex(TimeoutError, "late cap"):
            record.finish("completed", counts={"attempts": 1}, artifacts=["summary.partial.json"], budget_check=fail)
        runner._preserve_failure(record, TimeoutError("late cap"), interrupted=True)
        terminal = verify_artifacts(record.output_dir)
        self.assertEqual(terminal["status"], "interrupted")
        partial = next(item for item in terminal["artifacts"] if item["path"] == "summary.partial.json")
        self.assertEqual(partial["size_bytes"], len(b'{"status":"pre_terminal"}\n'))
        self.assertRegex(partial["sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse((record.output_dir / "PASS_FROZEN_REFERENCE_OPERATIONAL_V1").exists())


@unittest.skipUnless(__import__("importlib").util.find_spec("torch") and __import__("importlib").util.find_spec("transformers"),
                     "optional ML dependencies are unavailable")
class RunReferenceControllerTests(unittest.TestCase):
    """Exercise the real registered controller and RunRecord in a temp Git descendant."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        temp_root = Path(self.temp.name)
        self.root = temp_root / "repo"
        subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--shared", "--no-checkout", str(ROOT), str(self.root)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        git(self.root, "config", "core.autocrlf", "false")
        git(self.root, "config", "core.eol", "lf")
        git(self.root, "-c", "core.autocrlf=false", "checkout", "--detach", "HEAD")
        for relative in ("configs/models/frozen_source_reference_startup_repair_v2.json",
                         "configs/models/pretrained_gpu_feasibility_startup_repair_v2.json",
                         "configs/splits/frozen_reference_source_validation_cohort_v1.json",
                         "dwr/experiments/frozen_reference.py"):
            destination = self.root / relative; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        (self.root / "runs").mkdir()
        (self.root / "runs/.keep").write_bytes(b"fixture\n")
        git(self.root, "add", "configs/models/frozen_source_reference_startup_repair_v2.json",
            "configs/models/pretrained_gpu_feasibility_startup_repair_v2.json",
            "configs/splits/frozen_reference_source_validation_cohort_v1.json",
            "dwr/experiments/frozen_reference.py")
        git(self.root, "add", "-f", "runs/.keep")
        git(self.root, "-c", "user.name=Runner Test", "-c", "user.email=runner@example.invalid",
            "commit", "-qm", "registered frozen-reference controller fixture")
        self.config = self.root / "configs/models/frozen_source_reference_startup_repair_v2.json"
        self.ref_config = json.loads(self.config.read_text(encoding="utf-8"))
        self.cohort = json.loads((self.root / self.ref_config["dependencies"]["cohort_path"]).read_text(encoding="utf-8"))

    def tearDown(self):
        self.temp.cleanup()

    def canned_rows(self):
        selected = [{"example_id": item["example_id"], "profile": item["profile"],
                     "fine_stratum": item["fine_stratum"], "prompt_tokens": item["prompt_tokens"],
                     "lineage_id": "a" * 64, "structure_signature": "b" * 64,
                     "split": "source_validation", "task": serialize_task(task())}
                    for item in self.cohort["selected"]]
        extras = [{"example_id": hashlib.sha256(f"extra-{i}".encode()).hexdigest(),
                   "split": "source_validation"} for i in range(250)]
        return selected + extras

    def apply_controller_stubs(self, *, late_failure=False):
        import torch
        import dwr.experiments.gpu_feasibility as helper
        import dwr.experiments.source_reader as source_reader
        from transformers import Qwen2Config, Qwen2ForCausalLM
        check_state = {"terminal": False, "calls": 0}
        class Guard:
            peak_bytes = 1024
            started = 0.0
            def check(self):
                check_state["calls"] += 1
                if late_failure and check_state["terminal"]:
                    raise TimeoutError("terminal cap")
        guard = Guard()
        config = Qwen2Config(vocab_size=17, hidden_size=12, intermediate_size=24,
            num_hidden_layers=1, num_attention_heads=3, num_key_value_heads=1, _attn_implementation="sdpa")
        model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        class Tokenizer:
            def __init__(self): self.encodes = 0
            def encode(self, _text, *, add_special_tokens):
                if add_special_tokens: raise AssertionError("special tokens must be disabled")
                n = self_outer.cohort["selected"][self.encodes]["prompt_tokens"]
                self.encodes += 1
                return [1] * n
            def decode(self, token_ids, **_kwargs):
                if tuple(token_ids) == (7,): return '{"assignment":{"v0":1,"v1":1,"v2":0}}'
                return "not json"
        self_outer = self
        tokenizer = Tokenizer()
        generation_calls = []
        def canned_generation(_model, ids, _mask, *, eos_token_id, max_new_tokens, max_total_tokens, check):
            check()
            prompt_size = int(ids.shape[1]); generation_calls.append(prompt_size)
            measured_index = len(generation_calls) - 1
            if measured_index == 0:
                return GenerationResult((eos_token_id,), eos_token_id, True, False, prompt_size, 1, 0, 0, 0.001)
            cohort_position = (measured_index - 1) % 15
            generated = (7,) if cohort_position == 0 else (eos_token_id,)
            if getattr(self_outer, "_mismatch", False) and measured_index == 18:
                generated = (8,)
            ended = generated[-1] == eos_token_id
            return GenerationResult(generated, eos_token_id, ended, not ended,
                prompt_size, 1, 0, 0, 0.001)
        full = runner._load_registration(self.root, self.config)["feasibility_config"]
        def load_model(_root, _full, _helper, device):
            run_dir = self.root / self.ref_config["output_dir"]
            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            kinds = [event["kind"] for event in events]
            self.assertLess(kinds.index("initial_manifest_written"), kinds.index("precondition_access_started"))
            self.assertLess(kinds.index("precondition_access_started"), kinds.index("precondition_access_verified"))
            self.assertLess(kinds.index("precondition_access_verified"), kinds.index("source_validation_access_started"))
            self.assertTrue(all(event["payload"].get("protected_task_semantics_read") is False
                                for event in events if event["kind"] in {"precondition_access_started", "precondition_access_verified"}))
            self.assertEqual(device.type, "cpu")
            return model, tokenizer
        receipt = {"source_export_manifest_sha256": self.ref_config["data"]["source_export_manifest_sha256"],
                   "token_audit_manifest_sha256": helper.TOKEN_AUDIT_MANIFEST_SHA256}
        artifact = {"manifest_sha256": helper.MODEL_MANIFEST_SHA256}
        original_write_json = runner._write_json
        def tracked_write_json(path, value):
            if Path(path).name == "summary.json": check_state["terminal"] = True
            return original_write_json(path, value)
        patches = [
            patch.object(runner, "_repo_root", return_value=self.root),
            patch.object(helper, "make_probe_guard", return_value=guard),
            patch.object(helper, "validate_clean_source", wraps=helper.validate_clean_source),
            patch.object(runner, "_validate_feasibility", return_value={"config": full}),
            patch.object(helper, "_data_preconditions", return_value=receipt),
            patch.object(helper, "_artifact_inventory", return_value=artifact),
            patch.object(helper, "_check_environment", return_value={"torch": "fixture"}),
            patch.object(helper, "_hardware_metadata", return_value={"uuid": "fixture"}),
            patch.object(source_reader, "read_source_export", return_value=self.canned_rows()),
            patch.object(runner, "_make_guard", wraps=runner._make_guard),
            patch.object(runner, "_execution_device", return_value=torch.device("cpu")),
            patch.object(runner, "_load_original_model", side_effect=load_model),
            patch.object(helper, "_set_numerics", return_value={"fixture": True}),
            patch.object(runner, "greedy_frozen", side_effect=canned_generation),
            patch.object(runner.canonical, "content_id", side_effect=[item["example_id"] for item in self.cohort["selected"]]),
            patch.object(runner, "_write_json", side_effect=tracked_write_json),
        ]
        self._canned_generation = canned_generation
        self._tiny_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        return patches, generation_calls, check_state, tokenizer

    def test_real_controller_runs_actual_candidate_warmup_45_loop_and_terminal_linkage(self):
        patches, generation_calls, check_state, tokenizer = self.apply_controller_stubs()
        from contextlib import ExitStack
        with ExitStack() as stack:
            reader = stack.enter_context(patches[8])
            for item in patches[:8] + patches[9:]: stack.enter_context(item)
            summary = runner.run_reference(str(self.config), command=["python", "-m", "frozen_reference"])
        self.assertEqual(summary["attempt_count"], 45)
        self.assertEqual(summary["quality"]["denominator"], 15)
        self.assertEqual(len(generation_calls), 46)
        self.assertEqual(generation_calls[0], 16)  # synthetic warmup is first
        self.assertEqual(generation_calls[1:], [item["prompt_tokens"] for _ in range(3) for item in self.cohort["selected"]])
        self.assertEqual(tokenizer.encodes, 15)
        self.assertGreater(check_state["calls"], 90)
        reader.assert_called_once()
        self.assertEqual(reader.call_args.kwargs["split"], "source_validation")
        run_dir = self.root / self.ref_config["output_dir"]
        terminal = verify_artifacts(run_dir)
        self.assertEqual(terminal["status"], "completed")
        self.assertEqual(terminal["counts"][runner.PASS_STATUS], 1)
        rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 45)
        self.assertTrue({"generated_token_ids", "strict_verifier", "run_peak_so_far_mib", "generation_elapsed_seconds"} <= rows[0].keys())
        self.assertEqual(rows[0]["termination"], "token_cap")
        self.assertFalse(rows[0]["strict_verifier"]["valid"])
        warm = json.loads((run_dir / "warmup.json").read_text(encoding="utf-8"))
        self.assertTrue(warm["excluded_from_quality"])
        summary_file = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        expected_parameters = self._tiny_parameter_count
        self.assertEqual(summary_file["parameter_counts"], {"total": expected_parameters,
            "original": expected_parameters, "trainable": 0, "new": 0})
        events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        loaded = next(event for event in events if event["kind"] == "model_loaded")
        self.assertEqual(loaded["payload"]["parameter_counts"], summary_file["parameter_counts"])
        self.assertEqual(summary_file["original_parameter_hashes_before"], summary_file["original_parameter_hashes_after"])
        self.assertEqual(summary_file["rng_before_repetitions"], summary_file["rng_after_repetitions"])
        self.assertEqual((run_dir / "PASS_FROZEN_REFERENCE_OPERATIONAL_V1").exists(), False)

    def test_real_controller_candidate_repeat_failure_preserves_completed_prediction_linkage(self):
        patches, _calls, _state, _tokenizer = self.apply_controller_stubs()
        from contextlib import ExitStack
        # Change one measured result for a repeated example in repetition one.
        self._mismatch = True
        with ExitStack() as stack:
            for item in patches: stack.enter_context(item)
            with self.assertRaisesRegex(ValueError, "repeated generated"):
                runner.run_reference(str(self.config), command=["python", "-m", "frozen_reference"])
        run_dir = self.root / self.ref_config["output_dir"]
        terminal = verify_artifacts(run_dir)
        self.assertIn(terminal["status"], {"failed", "interrupted"})
        linked = next(item for item in terminal["artifacts"] if item["path"] == "predictions.jsonl")
        raw = (run_dir / "predictions.jsonl").read_bytes()
        self.assertEqual(linked["size_bytes"], len(raw))
        self.assertEqual(linked["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(len(raw.splitlines()), 45)
        self.assertFalse((run_dir / "PASS_FROZEN_REFERENCE_OPERATIONAL_V1").exists())

    def test_real_controller_late_terminal_callback_seals_failed_linked_partial(self):
        patches, _calls, _state, _tokenizer = self.apply_controller_stubs(late_failure=True)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in patches: stack.enter_context(item)
            with self.assertRaisesRegex(TimeoutError, "terminal cap"):
                runner.run_reference(str(self.config), command=["python", "-m", "frozen_reference"])
        run_dir = self.root / self.ref_config["output_dir"]
        terminal = verify_artifacts(run_dir)
        self.assertEqual(terminal["status"], "interrupted")
        self.assertTrue(any(item["path"] == "summary.partial.json" for item in terminal["artifacts"]))
        events = [json.loads(line)["kind"] for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertIn("reference_stopped", events)


if __name__ == "__main__":
    unittest.main()
