import copy
import json
from pathlib import Path
import unittest

from dwr.tasks.certificate_support import is_envelope, support_counts, support_stratum
from dwr.tasks.certificate_rank_audit import development_gate, full_stage_gate, source_gate
from tests.test_certificate_rank_audit import full_fixture, small_config


ROOT = Path(__file__).parents[1]
LEGACY = json.loads((ROOT / "configs/tasks/certificate_rank_qualification_v1.json").read_text(encoding="utf-8"))
FIVE = json.loads((ROOT / "configs/tasks/certificate_five_stratum_qualification_v1.json").read_text(encoding="utf-8"))


def meta(cell="A", nodes=5, prompt_bytes=1100, envelope=True, eligible=True):
    return {"cell": cell, "first_solution_nodes": nodes, "prompt_bytes": prompt_bytes,
            "envelope": envelope, "eligible": eligible}


def audit_row(content_id, *, split, profile, cell, nodes, signature=None):
    row = meta(cell=cell, nodes=nodes)
    row.update({"split": split, "content_id": content_id, "lineage_id": "L" + content_id,
        "oracle_unique": True, "first_seen_content": True, "structure_signature": signature or "S" + content_id,
        "profile": profile, "partition": "source" if split.startswith("source_") else "development",
        "sensitive": True, "values": [0, 1, 2], "given_count": 2,
        "gac": {"stalled": False}, "branch": "ordinary", "core": None})
    return row


def source_train_rows(*, include_all_a3_primitives=True):
    rows = []
    profiles = FIVE["source_profiles"]
    for cell, nodes in (("A", 5), ("A", 10), ("B", 5), ("C", 5), ("D", 5)):
        for i in range(10):
            profile = profiles[i % len(profiles)]
            rows.append(audit_row(f"train-{cell}-{i}", split="source_train", profile=profile,
                                  cell=cell, nodes=nodes))
    a3_profiles = profiles if include_all_a3_primitives else profiles[:2]
    for i in range(10):
        profile = a3_profiles[i % len(a3_profiles)]
        rows.append(audit_row(f"train-a3-{i}", split="source_train", profile=profile,
                              cell="A", nodes=10))
    return rows


def fine_rows(prefix, *, split, profile, strata=("A2", "A3", "B2", "C2", "D2"), count=10):
    cells_nodes = {"A2": ("A", 5), "A3": ("A", 10), "B2": ("B", 5),
                   "C2": ("C", 5), "D2": ("D", 5)}
    return [audit_row(f"{prefix}-{stratum}-{i}", split=split, profile=profile,
                      cell=cells_nodes[stratum][0], nodes=cells_nodes[stratum][1])
            for stratum in strata for i in range(count)]


def five_mode_config():
    c = copy.deepcopy(FIVE)
    c["source_floors"].update(train_groups=1, train_signatures=1, primitives=3,
        eligible_envelope_train_per_profile_cell=1, accepted_signatures_per_profile=1,
        eligible_envelope_train_per_all_raw_attempts=0, value_fraction_each=0,
        sensitivity=0, eligible_envelope_fraction=0, sensitive_per_accepted_signature=1,
        given_counts=[2])
    c["development_floors"].update(eligible_envelope_groups_per_profile=1,
        accepted_signatures_per_profile=1, scientific_eligibility_fraction=0,
        eligible_envelope_fraction=0, eligible_envelope_per_profile_cell=1,
        core_signatures_per_profile_given_count=1)
    c["difficulty"].update(development_stalled_groups_min=0,
        development_stalled_fraction_min=0, development_profiles_with_stall_min=0)
    c["full_floors"].update(source_quota=1, development_quota_per_profile=1,
        accepted_groups_per_partition=1, accepted_signatures_per_partition=1,
        source_split_minima={"source_train": 1, "source_validation": 1, "source_test": 1},
        development_split_minima={"development_validation": 1, "development_diagnostic_test": 1},
        development_split_minima_per_profile={"development_validation": 1, "development_diagnostic_test": 1},
        accepted_groups_per_mixed_profile=1, accepted_signatures_per_mixed_profile=1,
        mixed_eligibility_fraction=0, value_fraction_each=0, given_counts=[2], final_quota=1)
    return c


class CertificateFiveSupportTests(unittest.TestCase):
    def test_five_strata_use_exact_cell_node_boundaries_and_real_envelope_flag(self):
        expected = {7: "A2", 8: "A3", 15: "A3", 16: None}
        for nodes, stratum in expected.items():
            with self.subTest(nodes=nodes):
                row = meta(nodes=nodes)
                self.assertEqual(support_stratum(row, FIVE), stratum)
                self.assertEqual(is_envelope(row, FIVE), stratum is not None)
        self.assertEqual(support_stratum(meta("B", 7), FIVE), "B2")
        self.assertEqual(support_stratum(meta("C", 7), FIVE), "C2")
        self.assertEqual(support_stratum(meta("D", 7), FIVE), "D2")
        self.assertIsNone(support_stratum(meta(prompt_bytes=1023), FIVE))
        self.assertFalse(is_envelope(meta(envelope=False), FIVE))

    def test_support_counts_keep_all_eligible_as_denominator_and_separate_a3(self):
        rows = [meta(nodes=5) for _ in range(10)] + [meta(nodes=10) for _ in range(10)]
        rows += [meta(nodes=16) for _ in range(5)]
        rows += [meta(nodes=5, eligible=False) for _ in range(2)]
        result = support_counts(rows, FIVE)
        self.assertEqual(result["eligible"], 25)
        self.assertEqual(result["stratum_counts"], {"A2": 10, "A3": 10, "B2": 0, "C2": 0, "D2": 0})
        self.assertEqual(result["mass"], 0.8)
        self.assertFalse(result["pass"], "pooled A capacity must not satisfy missing fine strata")

    def test_legacy_defaults_retain_fixed_four_node_two_behavior(self):
        self.assertEqual(support_stratum(meta(nodes=7), LEGACY), "A")
        self.assertIsNone(support_stratum(meta(nodes=8), LEGACY))
        result = support_counts([meta(nodes=5)], LEGACY)
        self.assertEqual(result, {"eligible": 1, "cell_counts": {"A": 1, "B": 0, "C": 0, "D": 0},
                                  "mass": 1.0, "pass": False})

    def test_unknown_scheme_and_malformed_frozen_descriptors_fail_closed(self):
        bad = copy.deepcopy(FIVE)
        bad["support"]["scheme"] = "invented"
        with self.assertRaises(ValueError):
            support_stratum(meta(), bad)
        bad = copy.deepcopy(FIVE)
        bad["support"]["strata"][1]["first_solution_node_bin"] = 2
        with self.assertRaises(ValueError):
            support_stratum(meta(nodes=10), bad)

    def test_source_node2_capacity_cannot_be_boosted_by_a3_and_a3_presence_is_separate(self):
        c = copy.deepcopy(FIVE)
        c["source_floors"].update(train_groups=1, train_signatures=1, primitives=3,
            eligible_envelope_train_per_profile_cell=2, accepted_signatures_per_profile=1,
            eligible_envelope_train_per_all_raw_attempts=0, value_fraction_each=0,
            sensitivity=0, eligible_envelope_fraction=0, sensitive_per_accepted_signature=1)
        c["support"].update(cell_min=0, cohort_mass_min=0)
        rows = []
        for profile in c["source_profiles"]:
            for cell in "ABCD":
                row = meta(cell=cell)
                row.update(split="source_train", content_id=f"{profile}-{cell}", oracle_unique=True,
                    first_seen_content=True, sensitive=True, profile=profile,
                    structure_signature=f"sig-{profile}-{cell}", values=[0, 1, 2], given_count=2)
                rows.append(row)
            row = meta(nodes=10)
            row.update(split="source_train", content_id=f"a3-{profile}", oracle_unique=True,
                first_seen_content=True, sensitive=True, profile=profile,
                structure_signature=f"sig-a3-{profile}", values=[0, 1, 2], given_count=2)
            rows.append(row)
        ok, report = source_gate(rows, len(rows), c)
        self.assertFalse(ok)
        self.assertFalse(report["floors"]["profile_cell_capacity"])
        self.assertTrue(report["floors"]["a3_source_primitive_presence"])
        self.assertEqual(report["node2_cell_counts"]["A"], 3)
        self.assertEqual(report["source_primitive_by_fine_stratum"]["eq"]["A3"], 1)
        self.assertEqual(report["source_primitive_by_fine_stratum"]["neq"]["A3"], 1)

    def test_source_a3_primitive_presence_is_train_only_and_independent(self):
        c = five_mode_config()
        rows = source_train_rows(include_all_a3_primitives=False)
        protected = audit_row("protected-add-a3", split="source_test", profile="add_mod",
                              cell="A", nodes=10)
        rows.append(protected)
        ok, report = source_gate(rows, len(rows), c)
        self.assertFalse(ok)
        self.assertTrue(report["support"]["pass"], report["support"])
        self.assertTrue(report["floors"]["profile_cell_capacity"])
        self.assertFalse(report["floors"]["a3_source_primitive_presence"])
        self.assertEqual([name for name, passed in report["floors"].items() if not passed],
                         ["a3_source_primitive_presence"])
        self.assertNotIn("add_mod", report["a3_source_primitives"])

    def test_source_id_test_must_supply_its_own_a3_support(self):
        c = five_mode_config()
        train = source_train_rows()
        validation = [audit_row("source-val", split="source_validation", profile="eq",
                                 cell="A", nodes=5)]
        test = fine_rows("source-test", split="source_test", profile="eq")
        rows = train + validation + test
        ok, report = full_stage_gate(rows, "source", c, 100)
        self.assertTrue(report["source_id_test"]["pass"])
        self.assertTrue(report["qualification_floors"]["common_support"])
        self.assertTrue(report["floors"]["source_train"])
        self.assertTrue(all(report["floors"].values()), report["floors"])

        rows = train + validation + fine_rows("source-test-no-a3", split="source_test",
                                               profile="eq", strata=("A2", "B2", "C2", "D2"))
        ok, report = full_stage_gate(rows, "source", c, 100)
        self.assertTrue(report["qualification_floors"]["common_support"])
        self.assertFalse(report["source_id_test"]["pass"])
        self.assertFalse(report["floors"]["source_id_test_support"])
        self.assertEqual([name for name, passed in report["floors"].items() if not passed],
                         ["source_id_test_support"])

    def test_each_development_validation_profile_requires_its_own_a3(self):
        c = five_mode_config()
        source = source_train_rows()
        rows = []
        target = c["development_profiles"][0]
        for profile in c["development_profiles"]:
            strata = ("A2", "B2", "C2", "D2") if profile == target else ("A2", "A3", "B2", "C2", "D2")
            rows.extend(fine_rows("val-" + profile, split="development_validation",
                                  profile=profile, strata=strata))
        # A3 in this composition's diagnostic split must not rescue validation.
        rows.extend(fine_rows("diag-target", split="development_diagnostic_test",
                              profile=target, strata=("A3",), count=10))
        ok, report = development_gate(rows, source, c)
        self.assertFalse(report["profiles"][target]["support"]["pass"])
        self.assertTrue(all(report["profiles"][profile]["support"]["pass"]
                            for profile in c["development_profiles"] if profile != target))
        self.assertFalse(report["floors"][f"{target}/support"])

        ok, report = full_stage_gate(rows, "development", c, 100, source_rows=source)
        self.assertFalse(report["profiles"][target]["support"]["pass"])
        self.assertTrue(all(report["profiles"][profile]["support"]["pass"]
                            for profile in c["development_profiles"] if profile != target))
        self.assertFalse(report["floors"][f"{target}/support"])
        self.assertEqual(report["profiles"][target]["diagnostic_fine_stratum_counts"]["A3"], 10)

    def test_final_support_floor_is_per_fine_stratum(self):
        c = copy.deepcopy(FIVE)
        c["support"].update(cell_min=1, cohort_mass_min=0.8)
        c["full_floors"] = {"final_quota": 1, "accepted_groups_per_partition": 1,
            "accepted_signatures_per_partition": 1, "mixed_eligibility_fraction": 0,
            "value_fraction_each": 0, "given_counts": [2]}
        rows = []
        for i, (cell, nodes) in enumerate((("A", 5), ("A", 10), ("B", 5), ("C", 5), ("D", 5))):
            r = meta(cell=cell, nodes=nodes)
            r.update(split="final_test", content_id=f"f{i}", first_seen_content=True,
                     oracle_unique=True, structure_signature=f"s{i}", values=[0, 1, 2], given_count=2)
            rows.append(r)
        ok, report = full_stage_gate(rows, "final", c, len(rows))
        self.assertTrue(report["support"]["pass"])
        self.assertEqual(report["support"]["stratum_counts"], {"A2": 1, "A3": 1, "B2": 1, "C2": 1, "D2": 1})
        rows.pop(1)
        ok, report = full_stage_gate(rows, "final", c, len(rows))
        self.assertFalse(report["support"]["pass"])
        self.assertFalse(report["floors"]["support"])

    def test_development_diagnostic_nonzero_is_separate_from_validation_support(self):
        c = small_config()
        c["support"] = copy.deepcopy(FIVE["support"])
        c["support"].update(cell_min=1, cohort_mass_min=0.8)
        rows = full_fixture()
        development = [r for r in rows if r["partition"] == "development"]
        source = [r for r in rows if r["partition"] == "source"]
        ok, report = full_stage_gate(development, "development", c, 100, source_rows=source)
        for profile in c["development_profiles"]:
            self.assertFalse(report["floors"][f"{profile}/diagnostic_fine_stratum_nonzero"])
            self.assertEqual(report["profiles"][profile]["diagnostic_fine_stratum_counts"]["A2"], 1)


if __name__ == "__main__":
    unittest.main()
