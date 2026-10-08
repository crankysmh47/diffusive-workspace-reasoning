import tempfile
import unittest
import contextlib
import io
import hashlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from dwr.tasks.config_v3 import load_config_v3
from dwr.tasks.cli_v3 import main as cli_v3_main
from dwr.tasks.dataset_v2 import (DatasetLimits, accepted_record, outcome_row, run_dataset_v3,
                                  _split_and_ids, _source_closure_fixture)
from dwr.tasks.generator_v2 import generate_candidate
from dwr.tasks.generator_v3 import generate_candidate_v3
from dwr.tasks.generator_v2 import CandidatePrivateEvidence, SourceEligibilityEvidence
from dwr.tasks.generator_v2 import _finish_candidate
from dwr.tasks.schema import ConstraintTask, Constraint, RelationKind, GivenAssignment, Solution, canonical_constraint_key
from dwr.tasks.replay_v2 import replay_files_v3
from dwr.tasks.replay_v2 import replay_source_prefix_v3
from dwr.tasks.run_support import ResourceGuard, stable_json


ROOT = Path(__file__).resolve().parents[1]


class DatasetV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config_v3(ROOT / "configs/tasks/constraint_v3.json")

    def _run_small_closed_source(self, out: Path, *, guard=None):
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        limits = DatasetLimits({"eq": 0, "neq": 1, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 4, 30, 1024)
        with _source_closure_fixture(limits):
            code = run_dataset_v3(self.config, out, "feasibility", ["fixture"],
                                  limits=limits, source=source, guard=guard)
        return code, limits, source

    def _replay_small_closed_source(self, out: Path, limits, source, *, context=None,
                                    context_sha256=None, guard=None):
        import json
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        context = context or json.loads((out / "source_prefix_context.json").read_text(encoding="utf-8"))
        context_sha256 = context_sha256 or hashlib.sha256(
            (stable_json(context) + "\n").encode("utf-8")).hexdigest()
        return replay_source_prefix_v3(
            out / "outcomes.jsonl", out / "records.jsonl", context, context_sha256,
            self.config, source, manifest["producer"], limits=limits,
            guard=guard or ResourceGuard(30, 1024, memory_reader=lambda: 0))

    def test_v3_outcome_schema_is_explicit_and_rejects_a_v2_candidate(self):
        outcome = generate_candidate_v3(20261007, "source", 0)
        row = outcome_row(outcome, outcome.rejection_reason or outcome.eligibility_reason or "accepted",
                          split=None, identities=None, denominator_counted=False,
                          expected_generator_id="dwr-generator-v3")
        self.assertEqual(row["outcome_version"], "dwr-task-outcome-v3")
        legacy = generate_candidate(20261007, "source", 0)
        with self.assertRaisesRegex(ValueError, "generator identity"):
            outcome_row(legacy, legacy.rejection_reason or legacy.eligibility_reason or "accepted",
                        split=None, identities=None, denominator_counted=False,
                        expected_generator_id="dwr-generator-v3")

    def test_v3_accepted_record_adds_only_registered_identity_fields(self):
        outcome = generate_candidate_v3(20261007, "source", 1)
        self.assertTrue(outcome.oracle_unique and outcome.eligible)
        split, identities = _split_and_ids(outcome)
        record = accepted_record(outcome, split, identities, expected_generator_id="dwr-generator-v3")
        from dwr.tasks.audit import RECORD_KEYS
        self.assertEqual(record["record_version"], "dwr-task-accepted-record-v3")
        self.assertEqual(record["generator_id"], "dwr-generator-v3")
        self.assertEqual(set(record) - {"record_version", "generator_id"}, RECORD_KEYS)

    def test_v3_full_mode_fails_closed_before_creating_output(self):
        limits = DatasetLimits({"eq": 0, "neq": 0, "add_mod": 0}, {}, {}, 1, 30, 1024)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "full"
            with self.assertRaisesRegex(ValueError, "preflight consumption is not available"):
                run_dataset_v3(self.config, out, "full", ["test"], limits=limits)
            self.assertFalse(out.exists())

    def test_v3_cli_does_not_accept_v2_config(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_v3_main(["--config", str(ROOT / "configs/tasks/constraint_v2.json"),
                                    "--mode", "feasibility", "--out", str(out)])
            self.assertEqual(code, 1)
            self.assertIn("v3 generation failed", stderr.getvalue())
            self.assertFalse(out.exists())

    def test_small_incomplete_source_fixture_logs_v3_and_replays_as_partial(self):
        limits = DatasetLimits({"eq": 2, "neq": 0, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 1, 30, 1024)
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            code = run_dataset_v3(self.config, out, "feasibility", ["fixture"],
                                  limits=limits, source=source)
            self.assertEqual(code, 1)
            manifest = __import__("json").loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_version"], "dwr-task-run-v3")
            self.assertEqual(manifest["producer"].keys(), {
                "generator_v2_sha256", "generator_v3_sha256", "config_v3_sha256",
                "dataset_v2_sha256", "replay_v2_sha256", "manifest_v2_sha256",
                "stream_contract_sha256", "cli_v2_sha256", "cli_v3_sha256",
                "canonical_sha256", "oracle_sha256", "schema_sha256", "solver_sha256",
                "run_support_sha256"})
            self.assertIsNone(manifest["early_stop"])
            self.assertEqual(manifest["source_capacity_bound"]["status"], "NOT_ASSESSED")
            replay = replay_files_v3(out / "outcomes.jsonl", out / "records.jsonl",
                                     self.config, "feasibility", manifest,
                                     limits=limits, allow_partial=True)
            self.assertEqual(replay["status"], "partial")

    def test_v3_source_prefix_closure_is_a_valid_operational_error_phase(self):
        from dwr.tasks.manifest_v2 import validate_manifest_v3
        limits = DatasetLimits({"eq": 2, "neq": 0, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 1, 30, 1024)
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            run_dataset_v3(self.config, out, "feasibility", ["fixture"], limits=limits, source=source)
            import json
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            manifest["error"]["phase"] = "source_prefix_closure"
            manifest["source_capacity_bound"]["reason"] = "operational_error"
            validate_manifest_v3(manifest, config_sha256=self.config.sha256,
                                 config=self.config.resolved, mode="feasibility", limits=limits)

    def test_real_source_closure_replay_pass_advances_to_development(self):
        import json
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        limits = DatasetLimits({"eq": 0, "neq": 1, "add_mod": 0},
                               {("eq", "neq"): 1, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 3, 30, 1024)
        calls = []
        from dwr.tasks.dataset_v2 import _split_and_ids as original_split
        def fixture_split(outcome):
            if outcome.partition == "source":
                return "source_train", original_split(outcome)[1]
            return original_split(outcome)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            with _source_closure_fixture(limits), patch("dwr.tasks.dataset_v2._split_and_ids", side_effect=fixture_split), \
                 patch("dwr.tasks.replay_v2._split_and_ids", side_effect=fixture_split), \
                 patch("dwr.tasks.generator_v3.generate_candidate_v3", side_effect=lambda seed, part, idx, **kw: (
                     calls.append(part), generate_candidate_v3(seed, part, idx, **kw))[1]):
                code = run_dataset_v3(self.config, out, "feasibility", ["fixture"], limits=limits, source=source)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_capacity_bound"]["status"], "PASS_CAPACITY_BOUND")
            self.assertIn("development", calls)
            self.assertEqual(manifest["source_prefix"]["replay_file"], "source_prefix_replay.json")
            self.assertEqual(code, 1)  # the deliberately tiny development quota may cap

    def test_real_empty_source_train_closure_is_no_go_without_development(self):
        import json
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        limits = DatasetLimits({"eq": 0, "neq": 1, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 3, 30, 1024)
        calls = []
        original_split = _split_and_ids
        def fixture_split(outcome):
            if outcome.partition == "source":
                return "source_validation", original_split(outcome)[1]
            return original_split(outcome)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            with _source_closure_fixture(limits), patch("dwr.tasks.dataset_v2._split_and_ids", side_effect=fixture_split), \
                 patch("dwr.tasks.replay_v2._split_and_ids", side_effect=fixture_split), \
                 patch("dwr.tasks.generator_v3.generate_candidate_v3", side_effect=lambda seed, part, idx, **kw: (
                     calls.append(part), generate_candidate_v3(seed, part, idx, **kw))[1]):
                code = run_dataset_v3(self.config, out, "feasibility", ["fixture"], limits=limits, source=source)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            bound = manifest["source_capacity_bound"]
            self.assertEqual(bound["status"], "NO_GO", manifest.get("error"))
            self.assertEqual(bound["source_train_group_count"], 0)
            self.assertEqual(bound["top_k_mass"], 0.0)
            self.assertIsNotNone(bound["source_prefix_replay_sha256"])
            self.assertIsNotNone(manifest["early_stop"])
            self.assertNotIn("development", calls)
            self.assertEqual(code, 1)

    def test_real_bound_no_go_spans_more_than_ten_synthetic_verified_cells(self):
        import json
        from dataclasses import replace as dc_replace
        from itertools import combinations
        from dwr.tasks.stream_contract import _STREAMS
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        limits = DatasetLimits({"eq": 0, "neq": 15, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 15, 60, 1024)
        specifications = []
        for n, unknown_count in ((4, 2), (5, 2), (5, 3), (6, 2), (6, 3), (6, 4)):
            given_count = n - unknown_count
            if given_count < 2:
                continue
            values = [0, 1] + [i % 3 for i in range(given_count - 2)]
            givens = tuple(GivenAssignment(f"v{i}", values[i]) for i in range(given_count))
            base = []
            for unknown in range(given_count, n):
                base.extend((Constraint(RelationKind.NEQ, (f"v{unknown}", "v0")),
                              Constraint(RelationKind.NEQ, (f"v{unknown}", "v1"))))
            redundant = [Constraint(RelationKind.NEQ, (f"v{i}", f"v{j}"))
                         for i, j in combinations(range(given_count), 2) if values[i] != values[j]]
            for extras in range(min(len(redundant), 8 - len(base)) + 1):
                specifications.append((n, givens, tuple(sorted(base + redundant[:extras],
                                                               key=canonical_constraint_key))))
        specifications = specifications[:15]
        self.assertEqual(len(specifications), 15)
        candidates = {}
        for index, (n, givens, constraints) in enumerate(specifications):
            task = ConstraintTask("constraint-task-v1", 3, tuple(f"v{i}" for i in range(n)),
                                  constraints, givens)
            witness_values = dict((given.variable, given.value) for given in givens)
            witness_values.update((f"v{i}", 2) for i in range(len(givens), n))
            witness = Solution(tuple((f"v{i}", witness_values[f"v{i}"]) for i in range(n)))
            outcome = _finish_candidate("source", index, n, ("neq",), witness, task,
                                        generator_id="dwr-generator-v3")
            self.assertTrue(outcome.oracle_unique and outcome.eligible)
            candidates[index] = outcome
        def fixture_candidate(seed, partition, index, *, full_profiles=frozenset()):
            if partition == "source":
                return candidates[index]
            return generate_candidate_v3(seed, partition, index, full_profiles=full_profiles)
        contract = _STREAMS["dwr-generator-v3"]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            with _source_closure_fixture(limits), patch.dict(_STREAMS, {
                    "dwr-generator-v3": dc_replace(contract, candidate_fn=fixture_candidate)}), \
                 patch("dwr.tasks.generator_v3.generate_candidate_v3", side_effect=fixture_candidate), \
                 patch("dwr.tasks.dataset_v2._split_and_ids", side_effect=lambda outcome: (
                     "source_train", _split_and_ids(outcome)[1])), \
                 patch("dwr.tasks.replay_v2._split_and_ids", side_effect=lambda outcome: (
                     "source_train", _split_and_ids(outcome)[1])):
                code = run_dataset_v3(self.config, out, "feasibility", ["synthetic-fixture"],
                                      limits=limits, source=source)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            bound = manifest["source_capacity_bound"]
            self.assertEqual(manifest["source_prefix"]["replay_file"], "source_prefix_replay.json")
            self.assertEqual(bound["source_train_group_count"], 15)
            from dwr.tasks.audit import _cell
            records = [json.loads(line) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
            cells = set()
            for record in records:
                record["model_input_bytes"] = len(record["model_input"].encode("utf-8"))
                cells.add(_cell(record))
            self.assertGreater(len(cells), 10)
            self.assertLess(bound["top_k_mass"], 0.8)
            self.assertEqual(bound["status"], "NO_GO")
            self.assertEqual(manifest["partitions"]["development"]["status"], "not_attempted")
            self.assertEqual(code, 1)

    def test_source_prefix_replay_reconstructs_after_later_append(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            _, limits, source = self._run_small_closed_source(out)
            with (out / "outcomes.jsonl").open("ab") as stream:
                stream.write(b'{"later":"outcome"}\n')
            with (out / "records.jsonl").open("ab") as stream:
                stream.write(b'{"later":"record"}\n')
            report = self._replay_small_closed_source(out, limits, source)
            self.assertEqual(report["status"], "PASS_SOURCE_CLOSURE")

    def test_source_prefix_replay_rejects_missing_corrupt_postquota_and_bad_context(self):
        import json

        def fresh():
            temp = tempfile.TemporaryDirectory()
            out = Path(temp.name) / "run"
            _, limits, source = self._run_small_closed_source(out)
            context = json.loads((out / "source_prefix_context.json").read_text(encoding="utf-8"))
            ctx_hash = hashlib.sha256((stable_json(context) + "\n").encode("utf-8")).hexdigest()
            return temp, out, limits, source, context, ctx_hash

        temp, out, limits, source, context, ctx_hash = fresh()
        try:
            with self.assertRaisesRegex(ValueError, "context content"):
                self._replay_small_closed_source(out, limits, source, context=context,
                                                 context_sha256="0" * 64)
        finally:
            temp.cleanup()

        temp, out, limits, source, context, ctx_hash = fresh()
        try:
            outcomes_path = out / "outcomes.jsonl"
            raw = outcomes_path.read_bytes()
            first, tail = raw.split(b"\n", 1)
            changed = first + b" \n" + tail
            outcomes_path.write_bytes(changed)
            context["source_outcome_prefix_bytes"] = len(changed)
            ctx_hash = hashlib.sha256((stable_json(context) + "\n").encode("utf-8")).hexdigest()
            with self.assertRaisesRegex(ValueError, "byte count or hash mismatch"):
                self._replay_small_closed_source(out, limits, source, context=context,
                                                 context_sha256=ctx_hash)
        finally:
            temp.cleanup()

        temp, out, limits, source, context, ctx_hash = fresh()
        try:
            (out / "records.jsonl").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "longer than its artifact"):
                self._replay_small_closed_source(out, limits, source, context=context,
                                                 context_sha256=ctx_hash)
        finally:
            temp.cleanup()

        temp, out, limits, source, context, ctx_hash = fresh()
        try:
            records_path = out / "records.jsonl"
            raw = records_path.read_bytes()
            mutated = bytearray(raw)
            mutated[mutated.index(b"accepted") if b"accepted" in mutated else 5] ^= 1
            records_path.write_bytes(mutated)
            context["source_record_prefix_sha256"] = hashlib.sha256(mutated).hexdigest()
            ctx_hash = hashlib.sha256((stable_json(context) + "\n").encode("utf-8")).hexdigest()
            with self.assertRaisesRegex(ValueError, "differs from deterministic replay"):
                self._replay_small_closed_source(out, limits, source, context=context,
                                                 context_sha256=ctx_hash)
        finally:
            temp.cleanup()

        temp, out, limits, source, context, ctx_hash = fresh()
        try:
            extra_bytes = (stable_json({"post_quota": True}) + "\n").encode("utf-8")
            outcomes_path = out / "outcomes.jsonl"
            outcomes_path.write_bytes(outcomes_path.read_bytes() + extra_bytes)
            all_bytes = outcomes_path.read_bytes()
            context["source_attempts"] += 1
            context["source_outcome_prefix_bytes"] = len(all_bytes)
            context["source_outcome_prefix_sha256"] = hashlib.sha256(all_bytes).hexdigest()
            ctx_hash = hashlib.sha256((stable_json(context) + "\n").encode("utf-8")).hexdigest()
            with self.assertRaisesRegex(ValueError, "continue after source quota completion"):
                self._replay_small_closed_source(out, limits, source, context=context,
                                                 context_sha256=ctx_hash)
        finally:
            temp.cleanup()

    def test_source_prefix_corrupt_outcome_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            _, limits, source = self._run_small_closed_source(out)
            path = out / "outcomes.jsonl"
            raw = bytearray(path.read_bytes())
            raw[raw.index(b"\"source\"") + 2] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "outcome differs from deterministic replay"):
                self._replay_small_closed_source(out, limits, source)

    def test_driver_uses_same_live_guard_for_capture_and_prefix_replay(self):
        guard = ResourceGuard(30, 1024, memory_reader=lambda: 0)
        capture = __import__("dwr.tasks.dataset_v2", fromlist=["_capture_source_prefix_context_v3"])._capture_source_prefix_context_v3
        replay = __import__("dwr.tasks.replay_v2", fromlist=["replay_source_prefix_v3"]).replay_source_prefix_v3
        seen = []
        def capture_check(*args, **kwargs):
            seen.append(args[-1])
            return capture(*args, **kwargs)
        def replay_check(*args, **kwargs):
            seen.append(kwargs["guard"])
            return replay(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            with patch("dwr.tasks.dataset_v2._capture_source_prefix_context_v3", side_effect=capture_check), \
                 patch("dwr.tasks.replay_v2.replay_source_prefix_v3", side_effect=replay_check):
                _, _, _ = self._run_small_closed_source(Path(temp) / "run", guard=guard)
        self.assertEqual(seen, [guard, guard])

    def test_public_source_capture_keeps_registered_quota_guard(self):
        from dwr.tasks.dataset_v2 import capture_source_prefix_context_v3, _producer_identity
        limits = DatasetLimits({"eq": 0, "neq": 1, "add_mod": 0}, {}, {}, 4, 30, 1024)
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "registered complete 1,000-group quotas"):
                capture_source_prefix_context_v3(root / "outcomes", root / "records",
                                                 root / "context", self.config, source,
                                                 _producer_identity("dwr-generator-v3"), limits,
                                                 ResourceGuard(30, 1024, memory_reader=lambda: 0))

    def test_source_cap_and_closure_operational_error_remain_distinct_and_replayable(self):
        import json
        from dwr.tasks.manifest_v2 import validate_manifest_v3
        from dwr.tasks.dataset_v2 import _producer_identity
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        limits = DatasetLimits({"eq": 0, "neq": 1, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 1, 30, 1024)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            with _source_closure_fixture(limits), patch(
                    "dwr.tasks.dataset_v2._capture_source_prefix_context_v3",
                    side_effect=OSError("injected closure capture failure")):
                code = run_dataset_v3(self.config, out, "feasibility", ["fixture"],
                                      limits=limits, source=source)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertEqual(manifest["error"]["phase"], "source_prefix_closure")
            self.assertEqual(manifest["source_capacity_bound"]["reason"], "operational_error")
            self.assertIsNone(manifest["early_stop"])
            self.assertEqual(json.loads((out / "source_capacity_bound.json").read_text(encoding="utf-8")),
                             manifest["source_capacity_bound"])
            replay_files_v3(out / "outcomes.jsonl", out / "records.jsonl", self.config,
                            "feasibility", manifest, limits=limits, allow_partial=True)
            validate_manifest_v3(manifest, config_sha256=self.config.sha256,
                                 config=self.config.resolved, mode="feasibility", limits=limits,
                                 expected_producer=_producer_identity("dwr-generator-v3"))

    def test_bound_hash_failure_withdraws_decision_and_preserves_provisional(self):
        import json
        from dwr.tasks.dataset_v2 import sha256_file as original_hash
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        limits = DatasetLimits({"eq": 0, "neq": 1, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 4, 30, 1024)
        failed = False
        def fail_once(path):
            nonlocal failed
            if Path(path).name == "source_capacity_bound.json" and not failed:
                failed = True
                raise OSError("injected bound hash failure")
            return original_hash(path)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            with _source_closure_fixture(limits), patch("dwr.tasks.dataset_v2.sha256_file", side_effect=fail_once):
                code = run_dataset_v3(self.config, out, "feasibility", ["fixture"],
                                      limits=limits, source=source)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            report = json.loads((out / "source_capacity_bound.json").read_text(encoding="utf-8"))
            diagnostic = json.loads((out / "source_capacity_bound_provisional.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertEqual(manifest["source_capacity_bound"], report)
            self.assertEqual(report["reason"], "operational_error")
            self.assertIsNone(manifest["early_stop"])
            self.assertEqual(diagnostic["source_capacity_bound"]["status"], "PASS_CAPACITY_BOUND")
            self.assertEqual(manifest["artifacts"]["source_capacity_bound.json"]["sha256"], original_hash(out / "source_capacity_bound.json"))
            self.assertEqual(manifest["artifacts"]["source_capacity_bound_provisional.json"]["sha256"],
                             original_hash(out / "source_capacity_bound_provisional.json"))
            from dwr.tasks.manifest_v2 import validate_manifest_v3
            validate_manifest_v3(manifest, config_sha256=self.config.sha256,
                                 config=self.config.resolved, mode="feasibility", limits=limits)

    def test_bound_write_failure_is_terminally_operational(self):
        import json
        from dwr.tasks.dataset_v2 import atomic_json as original_atomic_json
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        limits = DatasetLimits({"eq": 0, "neq": 1, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 4, 30, 1024)
        failed = False
        def fail_once(path, value):
            nonlocal failed
            if Path(path).name == "source_capacity_bound.json" and not failed:
                failed = True
                raise OSError("injected bound report write failure")
            return original_atomic_json(path, value)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            with _source_closure_fixture(limits), patch("dwr.tasks.dataset_v2.atomic_json", side_effect=fail_once):
                code = run_dataset_v3(self.config, out, "feasibility", ["fixture"],
                                      limits=limits, source=source)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            report = json.loads((out / "source_capacity_bound.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertTrue(failed)
            self.assertEqual(report, manifest["source_capacity_bound"])
            self.assertEqual(report["reason"], "operational_error")
            self.assertIsNone(manifest["early_stop"])

    def test_terminal_manifest_write_failure_withdraws_no_go_and_keeps_diagnostic(self):
        import json
        from dwr.tasks.dataset_v2 import atomic_json as original_atomic_json
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        limits = DatasetLimits({"eq": 0, "neq": 1, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 4, 30, 1024)
        original_split = _split_and_ids
        def source_validation(outcome):
            if outcome.partition == "source":
                return "source_validation", original_split(outcome)[1]
            return original_split(outcome)
        failed = False
        def fail_terminal_once(path, value):
            nonlocal failed
            if Path(path).name == "manifest.json" and value.get("status") == "incomplete" and \
                    value.get("error", {}).get("type") == "ProposalNoGo" and not failed:
                failed = True
                raise OSError("injected terminal manifest replacement failure")
            return original_atomic_json(path, value)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            with _source_closure_fixture(limits), patch("dwr.tasks.dataset_v2._split_and_ids", side_effect=source_validation), \
                 patch("dwr.tasks.replay_v2._split_and_ids", side_effect=source_validation), \
                 patch("dwr.tasks.dataset_v2.atomic_json", side_effect=fail_terminal_once):
                code = run_dataset_v3(self.config, out, "feasibility", ["fixture"],
                                      limits=limits, source=source)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            diagnostic = json.loads((out / "source_capacity_bound_provisional.json").read_text(encoding="utf-8"))
            report = json.loads((out / "source_capacity_bound.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertTrue(failed)
            self.assertIsNone(manifest["early_stop"])
            self.assertEqual(manifest["source_capacity_bound"]["reason"], "operational_error")
            self.assertEqual(report, manifest["source_capacity_bound"])
            self.assertEqual(diagnostic["source_capacity_bound"]["status"], "NO_GO")
            self.assertEqual(diagnostic["prior_error"]["type"], "ProposalNoGo")
            from dwr.tasks.manifest_v2 import validate_manifest_v3
            validate_manifest_v3(manifest, config_sha256=self.config.sha256,
                                 config=self.config.resolved, mode="feasibility", limits=limits)
            self.assertTrue((out / "terminal_failure.json").exists())

    def test_first_ineligible_unique_owns_content_and_later_copy_is_duplicate(self):
        base = generate_candidate_v3(20261007, "source", 1)
        self.assertTrue(base.oracle_unique and base.eligible)
        ineligible_evidence = CandidatePrivateEvidence(
            True, SourceEligibilityEvidence(tuple(1 for _ in
                base.private_evidence.source.removal_solution_counts), False), None)
        limits = DatasetLimits({"eq": 2, "neq": 0, "add_mod": 0},
                               {("eq", "neq"): 0, ("eq", "add_mod"): 0,
                                ("neq", "add_mod"): 0}, {}, 3, 30, 1024)
        source = {"commit": "fixture", "dirty": True, "dirty_diff_sha256": "0" * 64,
                  "checkout_root": str(ROOT)}
        def injected(seed, partition, index, *, full_profiles=frozenset()):
            if partition == "source" and index in (1, 2):
                return replace(base, candidate_index=index,
                               private_evidence=ineligible_evidence if index == 1 else base.private_evidence)
            return generate_candidate_v3(seed, partition, index, full_profiles=full_profiles)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            with patch("dwr.tasks.generator_v3.generate_candidate_v3", side_effect=injected):
                code = run_dataset_v3(self.config, out, "feasibility", ["fixture"],
                                      limits=limits, source=source)
            self.assertEqual(code, 1)
            import json
            rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[1]["terminal_reason"], "source_insensitive")
            self.assertTrue(rows[1]["denominator_counted"])
            self.assertEqual(rows[2]["terminal_reason"], "duplicate_content")
            self.assertFalse(rows[2]["denominator_counted"])
            self.assertEqual((out / "records.jsonl").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
