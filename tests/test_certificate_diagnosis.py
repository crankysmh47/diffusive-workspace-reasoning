from __future__ import annotations

import unittest
import tempfile
import io
import json
import hashlib
from itertools import combinations
from unittest.mock import patch
from pathlib import Path

from dwr.tasks.certificate_diagnosis import (
    _analyze_rows,
    _compare_public_linkage,
    _counterfactual_arithmetic,
    _read_rows,
    _write_failure_terminal,
    run_diagnosis,
    _task_features,
    load_config,
)
from dwr.tasks import certificate_diagnosis as diagnosis_module
from dwr.tasks.schema import parse_task_json, serialize_model_input
from dwr.tasks.run_support import ResourceGuard


def row(index, *, split="development_validation", profile="eq/neq", cell="A",
        state=None, cid=None, eligible=True, envelope=True,
        task=None, values=None, prompt=None, nodes=None, gac=None):
    if state is None:
        state = ("scientifically_ineligible" if not eligible else
                 "accepted_for_support" if envelope else "outside_registered_support_envelope")
    vars_ = [f"v{i}" for i in range(6)]
    kinds = profile.split("/")
    target_constraints = {"A": 8, "B": 6, "C": 7, "D": 8}[cell]
    relations = {"eq": list(combinations(vars_, 2)), "neq": list(combinations(vars_, 2)),
                 "add_mod": list(combinations(vars_, 3))}
    kind_positions = {kind: 0 for kind in kinds}
    constraints = []
    for i in range(target_constraints):
        kind = kinds[i % len(kinds)]
        pairs = relations[kind]
        constraints.append({"kind": kind, "variables": list(pairs[kind_positions[kind] % len(pairs)])})
        kind_positions[kind] += 1
    given_count = {"A": 2, "B": 3, "C": 3, "D": 3}[cell]
    task_obj = {"schema_version": "constraint-task-v1", "modulus": 3,
                "variables": vars_, "constraints": constraints,
                "givens": [{"variable": vars_[i], "value": [0, 0, 1][i]} for i in range(given_count)]}
    task_json = json.dumps(task_obj, separators=(",", ":"))
    task_obj = parse_task_json(task_json)
    prompt = len(serialize_model_input(task_obj).encode("utf-8"))
    if nodes is None:
        nodes = 5 if envelope and prompt > 1023 else 3
    envelope = bool(envelope and prompt > 1023 and 4 <= nodes <= 7)
    duplicate = state == "duplicate_content"
    cid = cid or hashlib.sha256(f"content:{index}".encode()).hexdigest()
    if not (isinstance(cid, str) and len(cid) == 64 and all(ch in "0123456789abcdef" for ch in cid)):
        cid = hashlib.sha256(str(cid).encode()).hexdigest()
    return {
        "partition": "source" if split == "source_train" else "development",
        "candidate_index": index, "profile": profile,
        "cell": cell, "split": split, "state": state,
        "content_id": cid, "lineage_id": hashlib.sha256(f"lineage:{index}".encode()).hexdigest(),
        "structure_signature": hashlib.sha256(f"signature:{index}".encode()).hexdigest(),
        "oracle_unique": state != "duplicate_content",
        "eligible": eligible, "envelope": envelope, "sensitive": True,
        "oracle_solution_count": 1 if state != "oracle_non_unique" else 2,
        "duplicate_content": duplicate,
        "values": values or [0, 0, 1, 1, 2, 2], "given_count": given_count,
        "prompt_bytes": prompt, "first_solution_nodes": nodes,
        "gac": gac or {"solved": False, "stalled": True, "contradiction": False,
                       "unresolved_unknowns": 4, "rounds": 2, "support_checks": 20},
        "task": task or task_json,
        "witness": {"private": "must not leak"},
    }


class CertificateDiagnosisTests(unittest.TestCase):
    def test_protected_split_is_discarded_before_task_or_witness_parsing(self):
        poison = row(0, split="development_diagnostic_test")
        result = _analyze_rows([poison], partition="development")
        self.assertEqual(result["rows_seen"], 0)
        self.assertEqual(result["permitted_rows"], 0)
        self.assertEqual(result["feature_rows"], 0)
        self.assertNotIn("diagnostic_test", str(result))

    def test_duplicate_and_ineligible_rows_do_not_enter_eligible_denominator(self):
        rows = [
            row(0, cid="same", eligible=True, envelope=False),
            row(1, cid="same", state="duplicate_content", eligible=True),
            row(2, eligible=False, envelope=False, state="scientifically_ineligible"),
        ]
        result = _analyze_rows(rows, partition="development")
        counts = result["counts"]["development_validation"]["eq/neq"]
        self.assertEqual(counts["eligible_groups"], 1)
        self.assertEqual(counts["envelope_groups"], 0)
        self.assertEqual(counts["duplicates"], 1)

    def test_registered_row_contract_rejects_bool_indices_bad_cells_and_mismatched_task_metadata(self):
        mutations = [
            lambda item: item.update(candidate_index=True),
            lambda item: item.update(cell=""),
            lambda item: item.update(cell="A0"),
            lambda item: item.update(content_id="short"),
            lambda item: item.update(structure_signature="sig1"),
            lambda item: item.update(given_count=True),
            lambda item: item.update(prompt_bytes=item["prompt_bytes"] + 1),
            lambda item: item.update(values=[1, 0, 1, 1, 2, 2]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                item = row(10)
                mutate(item)
                with self.assertRaisesRegex(Exception, "row_invalid"):
                    _analyze_rows([item], partition="development")

    def test_duplicate_keeps_pre_dedup_envelope_metadata_but_not_group_counts(self):
        item = row(1, state="duplicate_content", cid="same")
        result = _analyze_rows([item], partition="development")
        counts = result["counts"]["development_validation"]["eq/neq"]
        self.assertTrue(item["envelope"])
        self.assertEqual(item["oracle_solution_count"], 1)
        self.assertFalse(item["oracle_unique"])
        self.assertEqual(counts["duplicates"], 1)
        self.assertEqual(counts["envelope_groups"], 0)

    def test_duplicate_nonunique_attempt_preserves_nonunique_pre_dedup_fields(self):
        item = row(1, state="duplicate_content", cid="repeat")
        item.update(oracle_solution_count=2, eligible=False, sensitive=False, envelope=False)
        result = _analyze_rows([item], partition="development")
        counts = result["counts"]["development_validation"]["eq/neq"]
        self.assertEqual(counts["duplicates"], 1)
        self.assertEqual(counts["unique"], 0)
        self.assertEqual(counts["eligible_groups"], 0)
        self.assertEqual(counts["envelope_groups"], 0)

    def test_empty_profiles_are_reported_as_zero(self):
        result = _analyze_rows([], partition="source")
        self.assertEqual(result["profiles"]["eq"]["constructed"], 0)
        self.assertEqual(result["profiles"]["neq"]["constructed"], 0)

    def test_splitless_failure_uses_only_registered_public_fields(self):
        failed = {"partition": "development", "candidate_index": 3, "profile": "eq/neq",
                  "cell": "B", "task": None, "split": None, "state": "rejected_proposal",
                  "rejection_reason": "empty_witness_pool", "work": {
                      "given_position_scans": 1, "witness_pool_scans": 2,
                      "alternative_pool_scans": 3, "constraint_pool_scans": 4,
                      "count_vector_scans": 5, "feasible_alternatives": 0,
                      "feasible_second_alternatives": 0, "feasible_count_vectors": 0,
                      "rng_calls": 6}, "witness": "must be ignored", "alternatives": "ignore"}
        result = _analyze_rows([failed], partition="development")
        self.assertEqual(result["profiles"]["eq/neq"]["construction_failures"], 1)
        self.assertEqual(result["profiles"]["eq/neq"]["constructed"], 0)
        self.assertEqual(result["construction_failures"]["eq/neq"]["reasons"],
                         {"empty_witness_pool": 1})
        self.assertNotIn("must be ignored", str(result))

    def test_eq_neq_features_cover_all_variables_and_anchored_components(self):
        task = parse_task_json(
            '{"schema_version":"constraint-task-v1","modulus":3,'
            '"variables":["v0","v1","v2","v3","v4","v5"],'
            '"constraints":[{"kind":"eq","variables":["v0","v1"]},'
            '{"kind":"eq","variables":["v1","v2"]},'
            '{"kind":"neq","variables":["v2","v3"]},'
            '{"kind":"neq","variables":["v1","v3"]}],'
            '"givens":[{"variable":"v0","value":0},'
            '{"variable":"v5","value":2}]}')
        features = _task_features(task, [0, 0, 0, 1, 2, 2])
        self.assertEqual(features["component_count"], 4)
        self.assertEqual(features["component_sizes"], [3, 1, 1, 1])
        self.assertEqual(features["anchored_components"], 2)
        self.assertEqual(features["unanchored_components"], 2)
        self.assertEqual(features["component_members"], [["v0", "v1", "v2"], ["v3"], ["v4"], ["v5"]])
        self.assertEqual(features["raw_inequality_edges"], 2)
        self.assertEqual(features["distinct_normalized_inequality_edges"], 1)
        self.assertEqual(features["witness_value_multiplicities"], [3, 2, 1])
        self.assertEqual(features["distinct_given_values"], 2)
        self.assertEqual(features["kind_counts"], {"eq": 2, "neq": 2})
        self.assertEqual(features["given_only_constraints"], 0)

    def test_counterfactual_arithmetic_is_conditional_and_exact(self):
        result = _counterfactual_arithmetic(eligible=569, stalled=60)
        self.assertEqual(result["plus_43_nonstalled"]["eligible"], 612)
        self.assertEqual(result["plus_43_nonstalled"]["stalled_fraction"], 60 / 612)
        self.assertEqual(result["plus_100_nonstalled"]["eligible"], 669)
        self.assertEqual(result["max_added_nonstalled_at_10_percent"], 31)
        self.assertEqual(result["added_stalls_required_for_10_percent"]["43"]["added_stalls_required"], 2)
        self.assertEqual(result["added_stalls_required_for_10_percent"]["100"]["added_stalls_required"], 8)

    def test_feature_histogram_keeps_duplicate_attempts_separate_from_group_counts(self):
        rows = [row(0, cid="same"), row(1, cid="same", state="duplicate_content")]
        result = _analyze_rows(rows, partition="development")
        counts = result["counts"]["development_validation"]["eq/neq"]
        self.assertEqual(counts["eligible_groups"], 1)
        self.assertEqual(counts["envelope_groups"], 1)
        self.assertEqual(result["feature_rows"], 2)
        self.assertEqual(sum(result["eq_neq_features"]["eq/neq"]["envelope"].values()), 1)
        self.assertEqual(sum(result["eq_neq_features"]["eq/neq"]["duplicate"].values()), 1)

    def test_permitted_projection_contains_no_task_witness_or_assignment_values(self):
        stream = io.StringIO()
        _analyze_rows([row(0)], partition="development", projection_stream=stream)
        projection = json.loads(stream.getvalue())
        self.assertIn("features", projection)
        self.assertNotIn("task", projection)
        self.assertNotIn("witness", projection)
        self.assertNotIn("values", projection)

    def test_protected_or_source_mixed_rows_do_not_parse_eqneq_task(self):
        poison = row(0, split="source_validation", profile="eq", task="opaque")
        result = _analyze_rows([poison], partition="source")
        self.assertEqual(result["feature_rows"], 0)

    def test_public_aggregate_disagreement_fails_closed(self):
        with self.assertRaisesRegex(Exception, "public_aggregate_mismatch"):
            _compare_public_linkage({"source": {}, "development": {}}, {}, {})

    def test_registration_is_closed_to_the_exact_config_path_bytes(self):
        self.assertEqual(load_config("configs/tasks/certificate_coverage_diagnosis_v1.json")["config_version"],
                         "dwr-certificate-coverage-diagnosis-v1")

    def test_modified_config_hash_is_refused(self):
        registered = Path("configs/tasks/certificate_coverage_diagnosis_v1.json").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "config.json"
            changed.write_text(registered.replace('"final_access": false', '"final_access": true'),
                               encoding="utf-8")
            with self.assertRaisesRegex(Exception, "configuration_identity_mismatch"):
                load_config(changed)

    def test_existing_output_is_refused_before_any_input_run_access(self):
        root = Path(__file__).resolve().parents[1]
        runs = root / "runs"
        with tempfile.TemporaryDirectory(prefix="certificate-coverage-diagnosis-test-", dir=runs) as directory:
            with self.assertRaisesRegex(Exception, "output_refused"):
                run_diagnosis("configs/tasks/certificate_coverage_diagnosis_v1.json", directory)

    def test_full_registered_prefix_routes_interleaved_schedule_and_partition_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcomes.jsonl"
            source_rows = {1: row(1, split="source_train", profile="eq", cell="B"),
                           3599: row(3599, split="source_train", profile="add_mod", cell="D")}
            development_rows = {3600: row(0, split="development_validation", profile="eq/neq", cell="A"),
                                5999: row(2399, split="development_validation", profile="neq/add_mod", cell="D")}
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for index in range(6000):
                    if index < 3600:
                        item = source_rows.get(index, {"partition": "source", "candidate_index": index,
                                                       "split": "source_validation"})
                    else:
                        item = development_rows.get(index, {"partition": "development",
                            "candidate_index": index - 3600, "split": "development_diagnostic_test"})
                    stream.write(json.dumps(item, separators=(",", ":")) + "\n")
            guard = ResourceGuard(30, 256)
            source = _read_rows(path, "source", guard, 3600, 3600, 6000)
            development = _read_rows(path, "development", guard, 2400, 3600, 6000)
            self.assertEqual(source["permitted_rows"], 2)
            self.assertEqual(development["permitted_rows"], 2)
            self.assertEqual(development["feature_rows"], 1)

    def test_late_guard_revocation_downgrades_partial_output_without_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            initial = {"status": "running", "command": ["python", "-m", "test"]}
            hashes = {"outcomes.jsonl": "a" * 64}
            source = {"git_commit": "fixture"}
            (output / "manifest.initial.json").write_text(json.dumps(initial), encoding="utf-8")
            (output / "permitted_rows.jsonl").write_text('{"profile":"eq"}\n', encoding="utf-8")
            (output / "report.json").write_text('{"status":"PASS_DIAGNOSIS"}', encoding="utf-8")
            (output / "manifest.json").write_text('{"status":"PASS_DIAGNOSIS"}', encoding="utf-8")
            _write_failure_terminal(output, initial, hashes, source, TimeoutError("expired"))
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["input_artifacts"], hashes)
            self.assertEqual((output / "manifest.initial.json").read_text(encoding="utf-8"),
                             json.dumps(initial))
            self.assertIn('"profile":"eq"', (output / "permitted_rows.jsonl").read_text(encoding="utf-8"))
            self.assertNotIn("expired", (output / "report.json").read_text(encoding="utf-8"))

    def test_run_wrapper_writes_initial_manifest_before_analysis_and_revokes_saved_pass(self):
        class ExpiringGuard:
            peak_bytes = 1024
            expired = False
            post_expiry_checks = 0

            def check(self):
                if self.expired:
                    self.post_expiry_checks += 1
                    raise RuntimeError("expired")

        config = load_config("configs/tasks/certificate_coverage_diagnosis_v1.json")
        hashes = config["input_artifact_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_run = root / config["input_run_path"]
            input_run.mkdir(parents=True)
            (input_run / "manifest.json").write_text(json.dumps({
                "status": "NO_GO_DEVELOPMENT_CAPACITY",
                "config_sha256": diagnosis_module.ORIGINAL_CONFIG_SHA256,
                "artifacts": {name: value for name, value in hashes.items() if name != "manifest.json"},
            }), encoding="utf-8")
            (input_run / "report.json").write_text(json.dumps({
                "status": "NO_GO_DEVELOPMENT_CAPACITY", "g2_status": "not_assessed",
                "preflight_emitted": False,
            }), encoding="utf-8")
            (input_run / "manifest.initial.json").write_text(json.dumps({
                "config_sha256": diagnosis_module.ORIGINAL_CONFIG_SHA256,
                "source": {"git_commit": diagnosis_module.SCIENTIFIC_COMMIT, "git_clean": True},
            }), encoding="utf-8")
            (input_run / "replay.json").write_text(json.dumps({"status": "PASS_REPLAY"}), encoding="utf-8")
            output = root / "runs" / "certificate-coverage-diagnosis-fixture"
            guard = ExpiringGuard()

            def expire_during_analysis(*args, **kwargs):
                initial_path = output / "manifest.initial.json"
                self.assertTrue(initial_path.is_file())
                self.assertEqual(json.loads(initial_path.read_text(encoding="utf-8"))["status"], "running")
                (output / "report.json").write_text('{"status":"PASS_DIAGNOSIS"}', encoding="utf-8")
                (output / "manifest.json").write_text('{"status":"PASS_DIAGNOSIS"}', encoding="utf-8")
                guard.expired = True
                raise RuntimeError("private row contents must not escape")

            with patch.object(diagnosis_module, "_source_identity", return_value={
                    "git_commit": "diagnosis-fixture", "git_clean": True, "producer_sha256": {}}), \
                 patch.object(diagnosis_module, "_input_identities", return_value=hashes), \
                 patch.object(diagnosis_module, "_verify_input_config",
                              return_value=config["input_config_sha256"]), \
                 patch.object(diagnosis_module, "_hash_file") as hash_file, \
                 patch.object(diagnosis_module, "_read_rows", side_effect=expire_during_analysis):
                with self.assertRaisesRegex(Exception, "diagnosis_failed"):
                    run_diagnosis("configs/tasks/certificate_coverage_diagnosis_v1.json", output,
                                  guard=guard, _root=root)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["status"], "NOT_ASSESSED")
            self.assertEqual(guard.post_expiry_checks, 0)
            hash_file.assert_not_called()
            self.assertNotIn("private row contents", str(report) + str(manifest))

    def test_run_wrapper_late_output_hash_io_failure_withdraws_pass(self):
        class LiveGuard:
            peak_bytes = 512

            def check(self):
                return None

        config = load_config("configs/tasks/certificate_coverage_diagnosis_v1.json")
        hashes = config["input_artifact_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_run = root / config["input_run_path"]
            input_run.mkdir(parents=True)
            (input_run / "manifest.json").write_text(json.dumps({
                "status": "NO_GO_DEVELOPMENT_CAPACITY",
                "config_sha256": diagnosis_module.ORIGINAL_CONFIG_SHA256,
                "artifacts": {name: value for name, value in hashes.items() if name != "manifest.json"},
            }), encoding="utf-8")
            (input_run / "report.json").write_text(json.dumps({
                "status": "NO_GO_DEVELOPMENT_CAPACITY", "g2_status": "not_assessed",
                "preflight_emitted": False,
            }), encoding="utf-8")
            (input_run / "manifest.initial.json").write_text(json.dumps({
                "config_sha256": diagnosis_module.ORIGINAL_CONFIG_SHA256,
                "source": {"git_commit": diagnosis_module.SCIENTIFIC_COMMIT, "git_clean": True},
            }), encoding="utf-8")
            (input_run / "replay.json").write_text(json.dumps({"status": "PASS_REPLAY"}), encoding="utf-8")
            output = root / "runs" / "certificate-coverage-diagnosis-io-fixture"
            source_result = {"partition": "source", "profiles": {}, "signature_coverage": {}}
            development_result = {"partition": "development", "profiles": {}, "signature_coverage": {}}
            with patch.object(diagnosis_module, "_source_identity", return_value={
                    "git_commit": "diagnosis-fixture", "git_clean": True, "producer_sha256": {}}), \
                 patch.object(diagnosis_module, "_input_identities", return_value=hashes), \
                 patch.object(diagnosis_module, "_verify_input_config",
                              return_value=config["input_config_sha256"]), \
                 patch.object(diagnosis_module, "_read_rows",
                              side_effect=[source_result, development_result]), \
                 patch.object(diagnosis_module, "_compare_public_linkage"), \
                 patch.object(diagnosis_module, "_hash_file", side_effect=OSError("private path detail")) as hash_file:
                with self.assertRaisesRegex(Exception, "diagnosis_failed"):
                    run_diagnosis("configs/tasks/certificate_coverage_diagnosis_v1.json", output,
                                  guard=LiveGuard(), _root=root)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["status"], "NOT_ASSESSED")
            self.assertTrue((output / "manifest.initial.json").is_file())
            self.assertTrue((output / "permitted_rows.jsonl").is_file())
            self.assertEqual(hash_file.call_count, 1)
            self.assertNotIn("private path detail", str(report) + str(manifest))

    def test_setup_initialization_failure_does_not_create_output_directory(self):
        class LiveGuard:
            peak_bytes = 0

            def check(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "runs" / "certificate-coverage-diagnosis-init-fixture"
            with patch.object(diagnosis_module, "_source_identity", return_value={
                    "git_commit": "fixture", "git_clean": True, "producer_sha256": {}}), \
                 patch.object(diagnosis_module.platform, "platform", side_effect=RuntimeError("setup failure")):
                with self.assertRaisesRegex(RuntimeError, "setup failure"):
                    run_diagnosis("configs/tasks/certificate_coverage_diagnosis_v1.json", output,
                                  guard=LiveGuard(), _root=root)
            self.assertFalse(output.exists())

    def test_real_terminal_atomic_write_guard_expiry_revokes_saved_pass(self):
        class ExpiringGuard:
            peak_bytes = 256
            expired = False

            def check(self):
                if self.expired:
                    raise RuntimeError("budget expired")

        config = load_config("configs/tasks/certificate_coverage_diagnosis_v1.json")
        hashes = config["input_artifact_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_run = root / config["input_run_path"]
            input_run.mkdir(parents=True)
            (input_run / "manifest.json").write_text(json.dumps({
                "status": "NO_GO_DEVELOPMENT_CAPACITY",
                "config_sha256": diagnosis_module.ORIGINAL_CONFIG_SHA256,
                "artifacts": {name: value for name, value in hashes.items() if name != "manifest.json"},
            }), encoding="utf-8")
            (input_run / "report.json").write_text(json.dumps({
                "status": "NO_GO_DEVELOPMENT_CAPACITY", "g2_status": "not_assessed",
                "preflight_emitted": False,
            }), encoding="utf-8")
            (input_run / "manifest.initial.json").write_text(json.dumps({
                "config_sha256": diagnosis_module.ORIGINAL_CONFIG_SHA256,
                "source": {"git_commit": diagnosis_module.SCIENTIFIC_COMMIT, "git_clean": True},
            }), encoding="utf-8")
            (input_run / "replay.json").write_text(json.dumps({"status": "PASS_REPLAY"}), encoding="utf-8")
            output = root / "runs" / "certificate-coverage-diagnosis-real-write-fixture"
            guard = ExpiringGuard()
            real_replace = Path.replace
            real_hash = diagnosis_module._hash_file
            post_expiry_hashes = []

            def replace_then_expire(path, target):
                result = real_replace(path, target)
                if Path(target).name == "manifest.json":
                    guard.expired = True
                return result

            def tracked_hash(path, active_guard):
                if guard.expired:
                    post_expiry_hashes.append(Path(path).name)
                return real_hash(path, active_guard)

            source_result = {"partition": "source", "profiles": {}, "signature_coverage": {}}
            development_result = {"partition": "development", "profiles": {}, "signature_coverage": {}}
            with patch.object(diagnosis_module, "_source_identity", return_value={
                    "git_commit": "diagnosis-fixture", "git_clean": True, "producer_sha256": {}}), \
                 patch.object(diagnosis_module, "_input_identities", return_value=hashes), \
                 patch.object(diagnosis_module, "_verify_input_config",
                              return_value=config["input_config_sha256"]), \
                 patch.object(diagnosis_module, "_read_rows",
                              side_effect=[source_result, development_result]), \
                 patch.object(diagnosis_module, "_compare_public_linkage"), \
                 patch.object(diagnosis_module, "_hash_file", side_effect=tracked_hash), \
                 patch.object(Path, "replace", new=replace_then_expire):
                with self.assertRaisesRegex(Exception, "resource_limit"):
                    run_diagnosis("configs/tasks/certificate_coverage_diagnosis_v1.json", output,
                                  guard=guard, _root=root)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["status"], "NOT_ASSESSED")
            self.assertTrue(guard.expired)
            self.assertEqual(post_expiry_hashes, [])


if __name__ == "__main__":
    unittest.main()
