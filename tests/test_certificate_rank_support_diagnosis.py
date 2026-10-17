import unittest
import tempfile
import hashlib
import json
from io import BytesIO
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dwr.tasks import certificate_rank_support_diagnosis as diagnosis


def row(index, *, partition="development", profile="eq/add_mod", cell="A", branch="ordinary",
        split="development_validation", prompt=1024, nodes=4, eligible=True, sensitive=True,
        envelope=None, stalled=False, signature=None, first=True, content=None):
    pbin, nbin = diagnosis._bin(prompt, diagnosis.PROMPT_BINS), diagnosis._bin(nodes, diagnosis.NODE_BINS)
    if envelope is None:
        envelope = pbin == 4 and nbin == 2
    return {"partition": partition, "candidate_index": index, "profile": profile, "cell": cell,
            "branch": branch, "split": split, "content_id": content or f"id-{index}",
            "structure_signature": signature or f"sig-{index}", "first_seen_content": first,
            "oracle_unique": True, "eligible": eligible, "sensitive": sensitive,
            "envelope": envelope, "given_count": diagnosis.CELLS[cell][0],
            "prompt_bytes": prompt, "first_solution_nodes": nodes,
            "gac": {"stalled": stalled, "solved": False, "contradiction": False}}


class DiagnosisTests(unittest.TestCase):
    def test_all_bin_boundaries_and_complete_zero_grid(self):
        for index, (lo, hi) in enumerate(diagnosis.PROMPT_BINS):
            self.assertEqual(diagnosis._bin(lo, diagnosis.PROMPT_BINS), index)
            if hi is not None:
                self.assertEqual(diagnosis._bin(hi, diagnosis.PROMPT_BINS), index)
        for index, (lo, hi) in enumerate(diagnosis.NODE_BINS):
            self.assertEqual(diagnosis._bin(lo, diagnosis.NODE_BINS), index)
            if hi is not None:
                self.assertEqual(diagnosis._bin(hi, diagnosis.NODE_BINS), index)
        result = diagnosis.analyze_rows([])
        self.assertEqual(len(result["complete_cell_bin_grid"]), 140)
        for counts in result["complete_cell_bin_grid"].values():
            self.assertEqual(set(counts["cohorts"]), {"source_train", *diagnosis.PROFILES})
            self.assertTrue(all(not any(vector.values()) for vector in counts["cohorts"].values()))

    def test_all_e_denominator_overlap_and_signature_counts(self):
        rows = [row(0, prompt=1024, nodes=4, signature="shared", stalled=True),
                row(1, prompt=511, nodes=4, signature="shared", stalled=True),
                row(2, prompt=1024, nodes=8, signature="excluded-node")]
        result = diagnosis.analyze_rows(rows)
        cohort = result["cohort_totals"]["eq/add_mod"]
        self.assertEqual((cohort["E"], cohort["A"], cohort["excluded_E"]), (3, 1, 2))
        self.assertEqual((cohort["stalled_E"], cohort["stalled_A"], cohort["stalled_excluded_E"]), (2, 1, 1))
        self.assertEqual((cohort["SigE"], cohort["SigA"], cohort["SigE_minus_SigA"]), (2, 1, 1))
        strata = result["strata"]["eq/add_mod/ordinary/A"]
        self.assertEqual((strata["prompt_only"], strata["node_only"]), (1, 1))

    def test_source_e_requires_sensitive_and_eligible(self):
        rows = [row(0, partition="source", profile="eq", split="source_train", eligible=True, sensitive=True),
                row(1, partition="source", profile="eq", split="source_train", eligible=False, sensitive=False)]
        result = diagnosis.analyze_rows(rows)
        self.assertEqual(result["cohort_totals"]["source_train"]["E"], 1)
        with self.assertRaises(diagnosis.DiagnosisError):
            diagnosis.analyze_rows([row(2, partition="source", profile="eq", split="source_train", eligible=True, sensitive=False)])

    def test_minimal_projection_retains_u_without_answer_fields(self):
        item = row(0, eligible=False)
        sink = BytesIO()
        result = diagnosis.analyze_rows([item], projection_stream=sink)
        projected = diagnosis.rank._strict_load(sink.getvalue().decode("utf-8"))
        self.assertEqual(result["cohort_totals"]["eq/add_mod"]["U"], 1)
        self.assertEqual(result["cohort_totals"]["eq/add_mod"]["E"], 0)
        self.assertEqual(set(projected), set(diagnosis.PROJECTION_FIELDS))
        self.assertNotIn("task", projected)
        self.assertNotIn("values", projected)
        self.assertNotIn("witness", projected)

    def test_protected_route_happens_before_semantic_metadata(self):
        class Poison(dict):
            def __getitem__(self, key):
                if key == "content_id":
                    raise AssertionError("protected semantic field accessed")
                return super().__getitem__(key)
            def get(self, key, default=None):
                if key == "content_id":
                    raise AssertionError("protected semantic field accessed")
                return super().get(key, default)
        protected = Poison(partition="development", candidate_index=7, split="development_diagnostic_test")
        self.assertIsNone(diagnosis._routed_projection(protected, "development", 7))

    def test_ordered_prefix_stream_routes_protected_and_rejects_extent_errors(self):
        records = [row(0, partition="source", profile="eq", split="source_train"),
                   row(1, partition="source", profile="eq", split="source_validation"),
                   row(0, profile="eq/add_mod", split="development_validation"),
                   row(1, profile="eq/add_mod", split="development_diagnostic_test")]
        stream = StringIO("".join(json.dumps(value) + "\n" for value in records))
        selected = list(diagnosis._iter_prefix(stream, 2, 2))
        self.assertEqual([item["candidate_index"] for item in selected], [0, 0])
        with self.assertRaises(diagnosis.DiagnosisError):
            list(diagnosis._iter_prefix(StringIO("{}\n"), 1, 0))
        with self.assertRaises(diagnosis.DiagnosisError):
            list(diagnosis._iter_prefix(StringIO("{}\n{}\n"), 1, 0))

    def test_first_seen_duplicates_are_not_resurrected_and_flags_consistent(self):
        first = row(0, first=True, content="same")
        excluded = row(1, first=False, content="same")
        # First-seen source truth is not recomputed within this diagnostic.
        self.assertEqual(diagnosis.analyze_rows([first, excluded])["cohort_totals"]["eq/add_mod"]["E"], 1)
        conflict = row(2, first=True, content="same")
        with self.assertRaises(diagnosis.DiagnosisError):
            diagnosis.analyze_rows([first, conflict])

    def test_bad_types_bins_and_public_mismatch_fail_closed(self):
        with self.assertRaises(diagnosis.DiagnosisError):
            diagnosis._bin(True, diagnosis.PROMPT_BINS)
        bad = row(0)
        bad["given_count"] = True
        with self.assertRaises(diagnosis.DiagnosisError):
            diagnosis.analyze_rows([bad])
        with self.assertRaises(diagnosis.DiagnosisError):
            diagnosis.compare_public({}, {"cohort_totals": {}})

    def test_selected_public_aggregates_match_exactly(self):
        cells = {"source_train": {"A": 253, "B": 250, "C": 302, "D": 351},
                 "eq/neq": {"A": 36, "B": 26, "C": 48, "D": 80},
                 "eq/add_mod": {"A": 123, "B": 129, "C": 116, "D": 94},
                 "neq/add_mod": {"A": 55, "B": 51, "C": 54, "D": 62}}
        values = {"source_train": (1317, 1317, 1156, 1071, 932),
                  "eq/neq": (279, 212, 190, 185, 169),
                  "eq/add_mod": (729, 593, 462, 560, 449),
                  "neq/add_mod": (345, 316, 222, 313, 220)}
        stalls = {"eq/neq": 90, "eq/add_mod": 250, "neq/add_mod": 112}
        analysis = {"cohort_totals": {key: {"U": v[0], "E": v[1], "A": v[2], "SigE": v[3],
                                             "SigA": v[4], "cell_A": cells[key],
                                             "sensitive_U": 1317 if key == "source_train" else 0,
                                             "stalled_E": stalls.get(key, 0)}
                                     for key, v in values.items()},
                    "branch_totals": {
                        "source_train": {"ordinary": {"E": 1317, "A": 1156}},
                        "eq/neq": {"ordinary": {"E": 212, "A": 190}},
                        "eq/add_mod": {"ordinary": {"E": 351, "A": 301}, "no_unary": {"E": 242, "A": 161}},
                        "neq/add_mod": {"ordinary": {"E": 168, "A": 136}, "no_unary": {"E": 148, "A": 86}}}}
        profile_values = {
            "eq/neq": (279, 212, 190, 185, 169, 22, 16, 0, 0),
            "eq/add_mod": (729, 593, 462, 560, 449, 131, 111, 242, 161),
            "neq/add_mod": (345, 316, 222, 313, 220, 94, 93, 148, 86)}
        profiles = {key: {"U": v[0], "E": v[1], "A": v[2], "SigE": v[3], "SigA": v[4],
                          "E_only": v[5], "SigE_only": v[6], "hard_E": v[7], "hard_A": v[8],
                          "cell_A": cells[key], "support": {"mass": v[2]/v[1]}}
                    for key, v in profile_values.items()}
        report = {"status": "NO_GO_DEVELOPMENT", "g2_status": "not_assessed", "source_gate": {
                      "first_seen_unique": 1317, "sensitive": 1317, "E": 1317, "A": 1156,
                      "signature_E": 1071, "signature_A": 932, "signature_E_only": 139,
                      "cell_counts": cells["source_train"], "support": {"mass": 1156/1317}},
                  "development_gate": {"profiles": profiles, "E": 1121, "A": 874,
                                       "stalled_E": 452, "stalled_fraction_E": 452/1121,
                                       "hard_E": 390, "ordinary_E": 731}}
        diagnosis.compare_public(report, analysis)

    def test_public_comparison_rejects_derived_stall_total_mismatch(self):
        # The public receipt's 452/1121 stall diagnostic must agree with the
        # newly derived pooled annotation inventory, independently of U/E/A.
        analysis = {"cohort_totals": {p: {"E": e, "stalled_E": s}
                                     for p, e, s in (("eq/neq", 212, 80),
                                                     ("eq/add_mod", 593, 250),
                                                     ("neq/add_mod", 316, 121))},
                    "branch_totals": {"eq/neq": {"ordinary": {"E": 212, "stalled_E": 80}},
                        "eq/add_mod": {"ordinary": {"E": 351, "stalled_E": 150},
                                       "no_unary": {"E": 242, "stalled_E": 100}},
                        "neq/add_mod": {"ordinary": {"E": 168, "stalled_E": 90},
                                        "no_unary": {"E": 148, "stalled_E": 32}}}}
        report = {"development_gate": {"E": 1121, "A": 874,
                      "stalled_E": 452, "stalled_fraction_E": 452/1121,
                      "hard_E": 390, "ordinary_E": 731}}
        with self.assertRaises(diagnosis.DiagnosisError):
            diagnosis._compare_pooled_stalls(report, analysis)
        analysis["cohort_totals"]["neq/add_mod"]["stalled_E"] = 122
        diagnosis._compare_pooled_stalls(report, analysis)

    def test_original_producer_binding_checks_original_blobs_and_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            receipt_path = root / diagnosis.SOURCE_RECEIPT_REL
            receipt_path.parent.mkdir(parents=True)
            original = {"dwr/__init__.py": b"old init\n", **{f"dwr/tasks/p{i}.py": f"old producer {i}\n".encode()
                                                                     for i in range(39)}}
            expected = {name: hashlib.sha256(blob).hexdigest() for name, blob in original.items()}
            receipt_path.write_bytes(diagnosis.rank._json_bytes({"source": {"producer_sha256": expected}}))
            current = {name: hashlib.sha256((blob + b"current")).hexdigest()
                       for name, blob in original.items()}
            def git_show(args, **kwargs):
                self.assertEqual(args[:2], ["git", "show"])
                commit_path = args[2].split(":", 1)
                self.assertEqual(commit_path[0], diagnosis.ORIGINAL_SOURCE_COMMIT)
                return SimpleNamespace(stdout=original[commit_path[1]])
            with patch.object(diagnosis.rank.subprocess, "run", side_effect=git_show):
                diagnosis._verify_original_producers({"producer_sha256": expected}, root, None)
                calls_after_valid = 40
                self.assertEqual(diagnosis.rank.subprocess.run.call_count, calls_after_valid)
                missing = dict(expected)
                missing.pop("dwr/tasks/p0.py")
                mutated = dict(expected)
                mutated["dwr/tasks/p0.py"] = "0" * 64
                for bad in ({}, missing, {**expected, "dwr/tasks/missing.py": "0" * 64}, mutated, current):
                    with self.assertRaises(diagnosis.DiagnosisError):
                        diagnosis._verify_original_producers({"producer_sha256": bad}, root, None)
                self.assertEqual(diagnosis.rank.subprocess.run.call_count, calls_after_valid)
                def current_git(args, **kwargs):
                    self.assertEqual(args[2].split(":", 1)[0], "b" * 40)
                    return SimpleNamespace(stdout=original[args[2].split(":", 1)[1]] + b"current")
                with patch.object(diagnosis, "ORIGINAL_SOURCE_COMMIT", "b" * 40), \
                        patch.object(diagnosis.rank.subprocess, "run", side_effect=current_git):
                    with self.assertRaises(diagnosis.DiagnosisError):
                        diagnosis._verify_original_producers({"producer_sha256": expected}, root, None)

    def test_runner_refusal_preserves_existing_registered_and_alternate_bytes(self):
        class Guard:
            peak_bytes = 1
            def check(self): return None
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path = root / diagnosis.CONFIG_REL
            config_path.parent.mkdir(parents=True)
            config_path.write_bytes((Path(diagnosis.__file__).resolve().parents[2] / diagnosis.CONFIG_REL).read_bytes())
            fake_module = root / "dwr" / "tasks" / "certificate_rank_support_diagnosis.py"
            fake_module.parent.mkdir(parents=True)
            # Root is the fake module's package root; use the registered relative path exactly.
            registered = root / "runs" / "certificate-rank-support-diagnosis-20261014-v1"
            alternate_paths = (registered, root / "runs" / "certificate-rank-qualification-20261012-v1")
            with patch.object(diagnosis, "__file__", str(fake_module)):
                for output in alternate_paths:
                    output.mkdir(parents=True)
                    sentinels = {name: f"untouched:{output.name}:{name}".encode()
                                 for name in ("report.json", "manifest.json")}
                    for name, value in sentinels.items(): (output / name).write_bytes(value)
                    argv = ["python", "-m", "dwr.tasks.certificate_rank_support_diagnosis",
                            "--config", str(config_path), "--output", str(output)]
                    with patch.object(diagnosis.rank, "_sha") as sha, \
                            patch.object(diagnosis, "_revoke") as revoke:
                        with self.assertRaises(diagnosis.DiagnosisError):
                            diagnosis.run_diagnosis(config_path, output, argv=argv, _guard=Guard())
                        sha.assert_not_called()
                        revoke.assert_not_called()
                    self.assertEqual({name: (output / name).read_bytes() for name in sentinels}, sentinels)
                output = registered
                sentinels = {name: (output / name).read_bytes() for name in ("report.json", "manifest.json")}
                argv = ["python", "-m", "dwr.tasks.certificate_rank_support_diagnosis",
                        "--config", str(config_path), "--output", str(output)]
                with patch.object(diagnosis.rank, "_safe_path", side_effect=ValueError("reparse path")):
                    with patch.object(diagnosis.rank, "_sha") as sha, \
                            patch.object(diagnosis, "_revoke") as revoke:
                        with self.assertRaises(diagnosis.DiagnosisError):
                            diagnosis.run_diagnosis(config_path, output, argv=argv, _guard=Guard())
                        sha.assert_not_called()
                        revoke.assert_not_called()
                self.assertEqual({name: (output / name).read_bytes() for name in sentinels}, sentinels)

    def test_canned_runner_binds_validated_input_config_into_final_boundary(self):
        """Fixture-only runner wiring check; no retained qualification artifacts are opened."""
        class Guard:
            peak_bytes = 0
            def check(self):
                return None

        public = {"status": "NO_GO_DEVELOPMENT", "g2_status": "not_assessed",
                  "source_gate": {"support": {"mass": 1.0, "pass": True}},
                  "development_gate": {"common_support": {}, "floors": {"support": False}}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path = root / diagnosis.CONFIG_REL
            config_path.parent.mkdir(parents=True)
            canonical_config = (Path(diagnosis.__file__).resolve().parents[2] / diagnosis.CONFIG_REL).read_bytes()
            config_path.write_bytes(canonical_config)
            fake_module = root / "dwr" / "tasks" / "certificate_rank_support_diagnosis.py"
            fake_module.parent.mkdir(parents=True)
            output = root / "runs" / "certificate-rank-support-diagnosis-20261014-v1"
            input_dir = root / diagnosis.INPUT_REL
            input_dir.mkdir(parents=True)
            (input_dir / "outcomes.jsonl").write_text("fixture-only\n", encoding="utf-8")
            argv = ["python", "-m", "dwr.tasks.certificate_rank_support_diagnosis",
                    "--config", str(config_path), "--output", str(output)]
            with patch.object(diagnosis, "__file__", str(fake_module)), \
                    patch.object(diagnosis.rank, "_source_identity", return_value={"fixture": True}), \
                    patch.object(diagnosis, "_verify_input_bindings",
                                 return_value=(public, {"support": {"cohort_mass_min": 0.8}})), \
                    patch.object(diagnosis, "_iter_prefix", return_value=iter(())), \
                    patch.object(diagnosis, "compare_public"), \
                    patch("dwr.tasks.run_support.ResourceGuard", return_value=Guard()):
                report = diagnosis.run_diagnosis(config_path, output, argv=argv)

            self.assertEqual(report["status"], "PASS_DIAGNOSIS")
            self.assertEqual(report["qualification_boundary"]["support_floor"], 0.8)
            manifest = diagnosis.rank._strict_load((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "PASS_DIAGNOSIS")

    def test_report_and_initial_manifest_preserve_registered_boundaries(self):
        public = {"status": "NO_GO_DEVELOPMENT", "g2_status": "not_assessed",
                  "source_gate": {"support": {"mass": 0.8777524677296887, "pass": True}},
                  "development_gate": {"floors": {"support": False},
                      "common_support": {"eq/add_mod": {"mass": 0.7790893760539629, "pass": False}}}}
        boundary = diagnosis._qualification_boundary(public, {"support": {"cohort_mass_min": 0.8}})
        self.assertEqual(boundary["original_terminal_status"], "NO_GO_DEVELOPMENT")
        self.assertEqual(boundary["original_g2_status"], "not_assessed")
        self.assertEqual(boundary["support_floor"], 0.8)
        self.assertEqual(boundary["source_support"], public["source_gate"]["support"])
        self.assertEqual(boundary["development_support"], public["development_gate"]["common_support"])
        self.assertEqual(boundary["floor_status"], public["development_gate"]["floors"])
        self.assertEqual((boundary["scope"], boundary["analysis"], boundary["new_samples"],
                          boundary["independent_reproduction"], boundary["archival_backup"]),
                         ("frozen_annotations", "agent_analysis", False, False, False))

    def test_projection_rejects_unknown_fields_and_wrong_index(self):
        item = row(0)
        item["task"] = "must not be projected"
        with self.assertRaises(diagnosis.DiagnosisError):
            diagnosis._validate_projection(item)
        with self.assertRaises(diagnosis.DiagnosisError):
            diagnosis._routed_projection(row(1), "development", 0)

    def test_late_failure_revokes_all_terminal_success_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            diagnosis._revoke(output, "TimeoutError")
            report = diagnosis.rank._strict_load((output / "report.json").read_text(encoding="utf-8"))
            manifest = diagnosis.rank._strict_load((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["status"], "NOT_ASSESSED")
            self.assertEqual(report["artifacts"], {})
            self.assertEqual(manifest["artifacts"], {})

    def test_closed_config_admission_and_fixture_cannot_pass(self):
        class Guard:
            peak_bytes = 1
            def check(self):
                return None
        canonical = Path(diagnosis.__file__).resolve().parents[2] / diagnosis.CONFIG_REL
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path = root / diagnosis.CONFIG_REL
            config_path.parent.mkdir(parents=True)
            config_path.write_bytes(canonical.read_bytes())
            fake_module = root / "dwr" / "tasks" / "certificate_rank_support_diagnosis.py"
            fake_module.parent.mkdir(parents=True)
            output = root / "runs" / "certificate-rank-support-diagnosis-20261014-v1"
            output.parent.mkdir(parents=True)
            argv = ["python", "-m", "dwr.tasks.certificate_rank_support_diagnosis",
                    "--config", str(config_path), "--output", str(output)]
            with patch.object(diagnosis, "__file__", str(fake_module)):
                with self.assertRaises(diagnosis.DiagnosisError):
                    diagnosis.run_diagnosis(config_path, output, argv=argv, _guard=Guard())
                wrong_output = root / "runs" / "wrong-output"
                wrong_argv = ["python", "-m", "dwr.tasks.certificate_rank_support_diagnosis",
                              "--config", str(config_path), "--output", str(wrong_output)]
                with self.assertRaises(diagnosis.DiagnosisError):
                    diagnosis.run_diagnosis(config_path, wrong_output, argv=wrong_argv, _guard=Guard())
                self.assertFalse(wrong_output.exists())
            report = diagnosis.rank._strict_load((output / "report.json").read_text(encoding="utf-8"))
            manifest = diagnosis.rank._strict_load((output / "manifest.json").read_text(encoding="utf-8"))
            initial_manifest = diagnosis.rank._strict_load((output / "manifest.initial.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["status"], "NOT_ASSESSED")
            self.assertNotEqual(report["status"], "PASS_DIAGNOSIS")
            self.assertEqual(initial_manifest["resolved_config"], diagnosis._config(config_path)[0])
            bad_config = root / "bad.json"
            bad_config.write_bytes(config_path.read_bytes() + b" ")
            with self.assertRaises(diagnosis.DiagnosisError):
                diagnosis._config(bad_config)

    def test_fake_input_metadata_and_hash_binding(self):
        root = Path(diagnosis.__file__).resolve().parents[2]
        input_config, input_sha, input_config_path = diagnosis.rank.load_config(
            root / "configs/tasks/certificate_rank_qualification_v1.json")
        receipt = diagnosis.rank._strict_load((root / diagnosis.SOURCE_RECEIPT_REL).read_text(encoding="utf-8"))
        source = {"git_commit": diagnosis.ORIGINAL_SOURCE_COMMIT, "git_clean": True,
                  "producer_sha256": receipt["source"]["producer_sha256"]}
        with tempfile.TemporaryDirectory() as folder:
            input_dir = Path(folder) / "input"
            input_dir.mkdir()
            requested = ["-m", "dwr.tasks.certificate_rank_study", "--config", str(input_config_path),
                         "--output", str(input_dir)]
            fake_initial = {"status": "running", "config_path": str(input_config_path),
                            "output_dir": str(input_dir.resolve()), "config_sha256": input_sha,
                            "resolved_config": input_config, "source": source,
                            "requested_argv": requested}
            fake_files = {
                "manifest.initial.json": diagnosis.rank._json_bytes(fake_initial),
                "outcomes.jsonl": b"minimal fake bytes\n",
                "records.jsonl": b"minimal fake records\n",
                "replay.json": diagnosis.rank._json_bytes({"status": "PASS_REPLAY", "attempts_rederived": 8400,
                                                             "exact_indices": True}),
                "report.json": diagnosis.rank._json_bytes({"status": "NO_GO_DEVELOPMENT", "g2_status": "not_assessed",
                                                             "raw_attempts": {"source": 3600, "development": 4800, "final": 0}}),
            }
            for name, contents in fake_files.items():
                (input_dir / name).write_bytes(contents)
            artifact_hashes = {name: hashlib.sha256(contents).hexdigest() for name, contents in fake_files.items()}
            fake_manifest = {"status": "NO_GO_DEVELOPMENT", "g2_status": "not_assessed",
                             "config_sha256": input_sha, "source": source, "artifacts": artifact_hashes}
            manifest_bytes = diagnosis.rank._json_bytes(fake_manifest)
            (input_dir / "manifest.json").write_bytes(manifest_bytes)
            bound = {**artifact_hashes, "manifest.json": hashlib.sha256(manifest_bytes).hexdigest()}
            closed, _, _ = diagnosis._config(root / diagnosis.CONFIG_REL)
            test_config = {**closed, "input_artifact_sha256": bound}
            with patch.object(diagnosis, "_verify_original_producers") as verify_original:
                bound_report, bound_config = diagnosis._verify_input_bindings(input_dir, test_config, root, None)
                self.assertEqual(bound_report["status"], "NO_GO_DEVELOPMENT")
                self.assertEqual(bound_config, input_config)
                verify_original.assert_called_once_with(source, root, None)
            (input_dir / "outcomes.jsonl").write_bytes(b"changed fake bytes\n")
            with self.assertRaises(diagnosis.DiagnosisError):
                diagnosis._verify_input_bindings(input_dir, test_config, root, None)

    def test_source_identity_checks_raw_blobs_with_fake_git(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks = root / "dwr" / "tasks"
            tasks.mkdir(parents=True)
            package = root / "dwr" / "__init__.py"
            package.write_bytes(b"package bytes\n")
            source = tasks / "fake.py"
            source.write_bytes(b"producer bytes\n")
            fake_rank = tasks / "certificate_rank_study.py"
            fake_rank.write_bytes(b"rank bytes\n")
            contents = {"dwr/__init__.py": package.read_bytes(),
                        "dwr/tasks/certificate_rank_study.py": fake_rank.read_bytes(),
                        "dwr/tasks/fake.py": source.read_bytes()}
            class Guard:
                def check(self):
                    return None
            def fake_run(args, **kwargs):
                if args[:3] == ["git", "rev-parse", "HEAD"]:
                    return SimpleNamespace(stdout="a" * 40 + "\n")
                if args[:3] == ["git", "status", "--porcelain"]:
                    return SimpleNamespace(stdout="")
                if args[:2] == ["git", "show"]:
                    return SimpleNamespace(stdout=contents[args[2].split(":", 1)[1]])
                raise AssertionError("unexpected Git command")
            with patch.object(diagnosis.rank, "__file__", str(fake_rank)), \
                    patch.object(diagnosis.rank.subprocess, "run", side_effect=fake_run):
                identity = diagnosis.rank._source_identity(Guard())
            self.assertEqual(identity["git_commit"], "a" * 40)
            self.assertTrue(identity["git_clean"])
            source.write_bytes(b"changed producer bytes\n")
            with patch.object(diagnosis.rank, "__file__", str(fake_rank)), \
                    patch.object(diagnosis.rank.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(diagnosis.rank.StudyError):
                    diagnosis.rank._source_identity(Guard())


if __name__ == "__main__":
    unittest.main()
