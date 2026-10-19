from __future__ import annotations

import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dwr.experiments.source_pilot_registration import (
    CONFIG_PATH,
    _json,
    _read_token_metadata,
    _select_pilot_rows,
    _validate_cohort,
    _validate_config,
    _validate_feasibility_outcome,
    _validation_selection,
    read_pilot_examples,
)
from dwr.tasks.run_support import stable_json

ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _tracked_config():
    raw = (ROOT / CONFIG_PATH).read_bytes()
    return _validate_config(_json(raw, "tracked pilot config"))


def _synthetic_cohort_fixture():
    rows = []
    for fine in ("A2", "A3", "B2", "C2", "D2"):
        for index in range(9):
            rows.append({"example_id": hashlib.sha256(f"{fine}-{index}".encode()).hexdigest(),
                "split": "source_validation", "profile": "eq", "fine_stratum": fine})
    selected = list(_validation_selection(rows))
    ids = sorted(row["example_id"] for row in selected)
    inventory = [{"count": sum(row["profile"] == profile and row["fine_stratum"] == fine for row in rows),
        "fine_stratum": fine, "profile": profile}
        for profile in ("eq", "neq", "add_mod") for fine in ("A2", "A3", "B2", "C2", "D2")]
    cohort = {"schema_version": "dwr-source-pilot-cohort-v1", "metadata_only_selection": True,
        "study": "dwr-source-validation-pilot-v1", "source_train_count": 2436,
        "source_validation_count": 265,
        "token_lengths_path": "runs/source-token-cap-audit-20261015-v1/token_lengths.jsonl",
        "token_lengths_sha256": "b26a54dc18099601d146c1c2a53609a72b848b282a5761a063972e72851c30de",
        "validation_primitive_fine_inventory": inventory,
        "selected": selected,
        "selection": {"hash": "SHA256 UTF8 prefix+example_id",
            "ordering": "fine A2 A3 B2 C2 D2 then digest then example_id",
            "per_pooled_fine_stratum": 9, "prefix": "dwr-source-pilot-validation-v1:", "refill": False},
        "frozen_reference_overlap_ids": [selected[0]["example_id"]],
        "shuffle_donor_map": {eid: ids[(index + 1) % len(ids)] for index, eid in enumerate(ids)}}
    return tuple(rows), cohort


class PilotRegistrationTests(unittest.TestCase):
    def test_tracked_config_is_closed_without_loading_ignored_artifacts(self):
        config = _tracked_config()
        self.assertEqual(config["arms"], [
            {"arm_id": "K0", "method": "noise_memory", "steps": 0},
            {"arm_id": "K1", "method": "EX", "steps": 1},
            {"arm_id": "K4", "method": "EX", "steps": 4}])
        unknown = copy.deepcopy(config)
        unknown["science_override"] = True
        with self.assertRaises(ValueError):
            _validate_config(unknown)
        null_value = copy.deepcopy(config)
        null_value["data"]["cohort_sha256"] = None
        with self.assertRaises(ValueError):
            _validate_config(null_value)
        changed_pin = copy.deepcopy(config)
        changed_pin["data"]["source_export_manifest_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            _validate_config(changed_pin)
        changed_budget = copy.deepcopy(config)
        changed_budget["limits"]["matrix_external_watchdog_seconds"] += 1
        with self.assertRaises(ValueError):
            _validate_config(changed_budget)

    def test_feasibility_outcome_fixture_binds_actual_bytes_and_status(self):
        good_raw = stable_json({"status": "PASS_FEASIBILITY", "fixture": "unchanged"}).encode()
        digest = hashlib.sha256(good_raw).hexdigest()
        manifest = {"artifacts": [{"path": "outcome.json", "sha256": digest}]}
        self.assertEqual(_validate_feasibility_outcome(manifest, good_raw, digest)["status"],
                         "PASS_FEASIBILITY")
        edited_pass = stable_json({"status": "PASS_FEASIBILITY", "fixture": "edited"}).encode()
        with self.assertRaises(ValueError):
            _validate_feasibility_outcome(manifest, edited_pass, digest)
        with self.assertRaises(ValueError):
            _validate_feasibility_outcome(manifest, None, digest)
        failed_raw = stable_json({"status": "FAIL_FEASIBILITY", "fixture": "unchanged"}).encode()
        failed_digest = hashlib.sha256(failed_raw).hexdigest()
        failed_manifest = {"artifacts": [{"path": "outcome.json", "sha256": failed_digest}]}
        with self.assertRaises(ValueError):
            _validate_feasibility_outcome(failed_manifest, failed_raw, failed_digest)

    def test_synthetic_cohort_selection_and_guardrails_recompute(self):
        rows, cohort = _synthetic_cohort_fixture()
        checked = _validate_cohort(cohort, rows)
        self.assertEqual(len(checked["selected"]), 45)
        self.assertEqual({item["fine_stratum"] for item in checked["selected"]},
                         {"A2", "A3", "B2", "C2", "D2"})
        changed = copy.deepcopy(cohort)
        changed["selected"][0]["profile"] = "neq"
        with self.assertRaises(ValueError):
            _validate_cohort(changed, rows)

    def test_token_metadata_reader_accepts_synthetic_complete_counts_only(self):
        config = copy.deepcopy(_tracked_config())
        rows = []
        for index in range(2701):
            split = "source_train" if index < 2436 else "source_validation"
            rows.append({"answer_tokens_including_eos": 3, "example_id": hashlib.sha256(
                    f"token-{index}".encode()).hexdigest(),
                "fine_stratum": ("A2", "A3", "B2", "C2", "D2")[index % 5],
                "profile": ("eq", "neq", "add_mod")[index % 3], "prompt_tokens": 10,
                "split": split, "total_tokens": 13, "within_caps": True})
        raw = "".join(stable_json(row) + "\n" for row in rows).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_dir = root / "synthetic-token-audit"
            token_dir.mkdir()
            (token_dir / "token_lengths.jsonl").write_bytes(raw)
            config["data"]["source_token_audit_dir"] = "synthetic-token-audit"
            config["data"]["token_lengths_sha256"] = hashlib.sha256(raw).hexdigest()
            parsed = _read_token_metadata(root, config)
        self.assertEqual(len(parsed), 2701)
        self.assertEqual(sum(row["split"] == "source_train" for row in parsed), 2436)
        self.assertEqual(sum(row["split"] == "source_validation" for row in parsed), 265)

    def test_split_selector_is_exact_ordered_and_never_refills(self):
        rows, selected = [], []
        for i in range(3):
            eid = hashlib.sha256(f"id-{i}".encode()).hexdigest()
            row = {"example_id": eid, "split": "source_validation", "profile": "eq", "fine_stratum": "A2"}
            selected.append({"example_id": eid, "split": "source_validation",
                             "profile": "eq", "fine_stratum": "A2"})
            rows.append(row)
        distractor = {"example_id": hashlib.sha256(b"distractor").hexdigest(),
                      "split": "source_validation", "profile": "add_mod", "fine_stratum": "A2"}
        ordered = _select_pilot_rows((distractor, rows[2], rows[0], rows[1]), selected,
                                     split="source_validation")
        self.assertEqual([row["example_id"] for row in ordered],
                         [row["example_id"] for row in selected])
        with self.assertRaises(ValueError):
            _select_pilot_rows((distractor, rows[0], rows[1]), selected, split="source_validation")
        with self.assertRaises(ValueError):
            _select_pilot_rows((rows[0], rows[0], rows[1], rows[2]), selected, split="source_validation")

    @staticmethod
    def _synthetic_source_row(example_id: str, *, split="source_train", profile="eq",
                              fine="A2", assignment=None, task_text=None):
        task = {"schema_version": "constraint-task-v1", "modulus": 3,
            "variables": ["v0", "v1", "v2"],
            "constraints": [{"kind": "eq", "variables": ["v0", "v1"]},
                {"kind": "eq", "variables": ["v1", "v2"]}],
            "givens": [{"variable": "v0", "value": 1}]}
        return {"example_id": example_id, "split": split, "profile": profile,
            "fine_stratum": fine, "task": stable_json(task) if task_text is None else task_text,
            "solution": {"assignment": assignment or {"v0": 1, "v1": 1, "v2": 1}}}

    @staticmethod
    def _registration_for_rows(train_ids, validation_ids):
        config = copy.deepcopy(_tracked_config())
        config["data"]["source_export_dir"] = "synthetic-export"
        config["data"]["source_train_records"] = len(train_ids)
        config["data"]["source_validation_records"] = len(validation_ids)
        token_rows = []
        for split, ids in (("source_train", train_ids), ("source_validation", validation_ids)):
            for eid in ids:
                token_rows.append({"example_id": eid, "split": split, "profile": "eq", "fine_stratum": "A2",
                    "prompt_tokens": 8, "answer_tokens_including_eos": 3, "total_tokens": 11})
        selected = [{"example_id": eid, "split": "source_validation", "profile": "eq", "fine_stratum": "A2"}
                    for eid in validation_ids]
        return {"config": config, "cohort": {"selected": selected}, "token_rows": tuple(token_rows)}

    @unittest.skipUnless(TORCH_AVAILABLE, "read_pilot_examples constructs SourceExample, which requires torch")
    def test_read_pilot_examples_wrapper_orders_and_converts_synthetic_rows(self):
        from dwr.training.batches import SourceExample
        train_ids = tuple(hashlib.sha256(f"train-{i}".encode()).hexdigest() for i in range(3))
        validation_ids = tuple(hashlib.sha256(f"validation-{i}".encode()).hexdigest() for i in range(2))
        registration = self._registration_for_rows(train_ids, validation_ids)
        train_rows = [self._synthetic_source_row(eid) for eid in reversed(train_ids)]
        val_rows = [self._synthetic_source_row(eid, split="source_validation") for eid in reversed(validation_ids)]
        with patch("dwr.experiments.source_reader.read_source_export", side_effect=[train_rows, val_rows]) as reader:
            train, train_meta = read_pilot_examples(ROOT, registration, split="source_train")
            validation, validation_meta = read_pilot_examples(ROOT, registration, split="source_validation")
        self.assertEqual(tuple(item.example_id for item in train), tuple(sorted(train_ids)))
        self.assertEqual(tuple(item.example_id for item in validation), validation_ids)
        self.assertTrue(all(type(item) is SourceExample for item in train + validation))
        self.assertEqual(set(train_meta), set(train_ids))
        self.assertEqual(tuple(validation_meta), validation_ids)
        self.assertEqual(reader.call_count, 2)

    @unittest.skipUnless(TORCH_AVAILABLE, "read_pilot_examples constructs SourceExample, which requires torch")
    def test_read_pilot_examples_rejects_missing_duplicate_metadata_and_bad_semantics(self):
        train_ids = tuple(hashlib.sha256(f"case-{i}".encode()).hexdigest() for i in range(2))
        registration = self._registration_for_rows(train_ids, ())
        good = [self._synthetic_source_row(eid) for eid in train_ids]
        bad_profile = [dict(good[0], profile="neq"), good[1]]
        duplicate = [good[0], good[0]]
        missing = [good[0]]
        invalid_solution = [dict(good[0], solution={"assignment": {"v0": 1, "v1": 2, "v2": 2}}), good[1]]
        malformed_task = [dict(good[0], task="{}"), good[1]]
        for rows in (bad_profile, duplicate, missing, invalid_solution, malformed_task):
            with self.subTest(rows=rows):
                with patch("dwr.experiments.source_reader.read_source_export", return_value=rows):
                    with self.assertRaises(ValueError):
                        read_pilot_examples(ROOT, registration, split="source_train")

    @unittest.skipUnless(TORCH_AVAILABLE, "read_pilot_examples constructs SourceExample, which requires torch")
    def test_read_pilot_examples_validation_rejects_missing_duplicate_or_changed_selected_ids(self):
        ids = tuple(hashlib.sha256(f"validation-case-{i}".encode()).hexdigest() for i in range(2))
        registration = self._registration_for_rows((), ids)
        valid = [self._synthetic_source_row(eid, split="source_validation") for eid in ids]
        missing = [valid[0]]
        duplicate = [valid[0], valid[0]]
        changed_metadata = [valid[0], dict(valid[1], fine_stratum="D2")]
        for rows in (missing, duplicate, changed_metadata):
            with self.subTest(rows=rows):
                with patch("dwr.experiments.source_reader.read_source_export", return_value=rows):
                    with self.assertRaises(ValueError):
                        read_pilot_examples(ROOT, registration, split="source_validation")

    def test_read_pilot_examples_rejects_protected_route_before_reader_access(self):
        registration = self._registration_for_rows((), ())
        with patch("dwr.experiments.source_reader.read_source_export") as reader:
            with self.assertRaises(ValueError):
                read_pilot_examples(ROOT, registration, split="development_test")
        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
