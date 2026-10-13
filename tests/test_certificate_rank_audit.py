import copy
import json
from pathlib import Path
import unittest

from dwr.tasks.certificate_rank_audit import (
    development_gate, full_gate, source_gate,
)


CONFIG = json.loads((Path(__file__).parents[1] / "configs/tasks/certificate_rank_qualification_v1.json").read_text(encoding="utf-8"))


def small_config():
    c = copy.deepcopy(CONFIG)
    c["support"].update(cell_min=1, cohort_mass_min=0.5)
    c["source_floors"].update(train_groups=1, train_signatures=1, primitives=1,
        value_fraction_each=0.0, sensitivity=0.25, eligible_envelope_fraction=0.25,
        sensitive_per_accepted_signature=1, eligible_envelope_train_per_profile_cell=1,
        accepted_signatures_per_profile=1, eligible_envelope_train_per_all_raw_attempts=0.0)
    c["development_floors"].update(eligible_envelope_groups_per_profile=1,
        accepted_signatures_per_profile=1, scientific_eligibility_fraction=0.25,
        eligible_envelope_fraction=0.25, eligible_envelope_per_profile_cell=1,
        core_signatures_per_profile_given_count=1)
    c["difficulty"].update(development_stalled_groups_min=1,
        development_stalled_fraction_min=0.1, development_profiles_with_stall_min=1)
    c["full_floors"].update(source_quota=1, development_quota_per_profile=1, final_quota=1,
        accepted_groups_per_partition=1, accepted_signatures_per_partition=1,
        source_split_minima={"source_train": 1, "source_validation": 1, "source_test": 1},
        development_split_minima={"development_validation": 1, "development_diagnostic_test": 1},
        development_split_minima_per_profile={"development_validation": 1, "development_diagnostic_test": 1},
        accepted_groups_per_mixed_profile=1, accepted_signatures_per_mixed_profile=1,
        mixed_eligibility_fraction=0.25, value_fraction_each=0.0, given_counts=[2, 3])
    return c


def row(cid, *, split="development_validation", partition="development", profile="eq/add_mod",
        cell="A", eligible=True, envelope=True, gac=False, signature=None, given=2,
        hard=False, first=True):
    return {"partition": partition, "profile": profile, "cell": cell, "split": split,
        "content_id": cid, "lineage_id": "L"+cid, "structure_signature": signature or "S"+cid,
        "oracle_unique": first, "first_seen_content": first, "eligible": eligible, "envelope": envelope,
        "sensitive": True, "values": [0, 1, 2], "given_count": given,
        "prompt_bytes": 1100 if envelope else 500, "first_solution_nodes": 5 if envelope else 2,
        "gac": {"stalled": gac}, "branch": "no_unary" if hard else "ordinary",
        "core": {"signature": "C"+str(given)} if hard else None}


class OpaqueProtected(dict):
    protected = {"content_id", "oracle_unique", "eligible", "envelope", "sensitive",
                 "values", "given_count", "prompt_bytes", "first_solution_nodes", "gac",
                 "profile", "cell", "structure_signature", "core", "branch"}
    def get(self, key, default=None):
        if key in self.protected:
            raise AssertionError(f"protected field read before routing: {key}")
        return super().get(key, default)


class CertificateRankAuditTests(unittest.TestCase):
    def test_development_routes_protected_split_before_semantic_reads(self):
        protected = OpaqueProtected(split="development_diagnostic_test")
        ok, report = development_gate([protected], [], small_config())
        self.assertFalse(ok)
        self.assertEqual(report["E"], 0)

    def test_all_eligible_denominator_keeps_out_of_envelope_groups(self):
        c = small_config()
        c["support"]["cohort_mass_min"] = 0.75
        rs = [row("a", signature="a"), row("b", signature="b"),
              row("c", envelope=False, signature="c"), row("d", envelope=False, signature="d")]
        ok, report = development_gate(rs, [], c)
        self.assertFalse(ok)
        self.assertEqual(report["profiles"]["eq/add_mod"]["E"], 4)
        self.assertEqual(report["profiles"]["eq/add_mod"]["A"], 2)
        self.assertEqual(report["profiles"]["eq/add_mod"]["support"]["mass"], 0.5)

    def test_signature_floor_uses_sig_a_and_reports_e_only_attrition(self):
        c = small_config()
        c["development_floors"]["accepted_signatures_per_profile"] = 2
        rs = [row("a", signature="shared"), row("b", signature="shared"),
              row("c", envelope=False, signature="outside")]
        ok, report = development_gate(rs, [], c)
        p = report["profiles"]["eq/add_mod"]
        self.assertFalse(ok)
        self.assertEqual(p["SigA"], 1)
        self.assertEqual(p["SigE_only"], 1)

    def test_first_seen_dedup_and_hard_core_are_reported(self):
        c = small_config()
        rs = [row("a", signature="s", gac=True, hard=True, given=2),
              row("a", signature="s-duplicate", gac=True, hard=True, given=2, first=False),
              row("b", signature="t", gac=True, hard=True, given=3)]
        ok, report = development_gate(rs, [], c)
        self.assertEqual(report["E"], 2)
        self.assertEqual(report["stalled_E"], 2)
        self.assertEqual(report["profiles"]["eq/add_mod"]["hard_E"], 2)

    def test_qualification_does_not_require_impossible_eq_neq_core_floors(self):
        ok, report = development_gate([row("a", profile="eq/neq")], [], small_config())
        self.assertFalse(ok)
        self.assertFalse(any("eq/neq/hard_core" in key for key in report["floors"]))

    def test_source_uses_actual_raw_attempts_and_empty_denominators_fail(self):
        c = small_config()
        c["source_floors"]["eligible_envelope_train_per_all_raw_attempts"] = 0.5
        rs = [row("a", split="source_train", partition="source", profile="eq", signature="a")]
        ok, report = source_gate(rs, 3, c)
        self.assertFalse(ok)
        self.assertEqual(report["A_per_raw_attempt"], 1/3)
        ok, empty = source_gate([], 0, c)
        self.assertFalse(ok)
        self.assertFalse(empty["floors"]["raw_yield"])

    def test_first_seen_content_flag_is_authoritative(self):
        c = small_config()
        duplicate_only = row("dup", first=True)
        duplicate_only["first_seen_content"] = False
        ok, report = development_gate([duplicate_only], [], c)
        self.assertFalse(ok)
        self.assertEqual(report["E"], 0)

    def test_full_five_cohort_positive_allows_source_id_structure_overlap(self):
        c = small_config()
        rows = full_fixture()
        rows.append({"partition": "source", "candidate_index": 29, "profile": "eq", "cell": "A",
                     "split": None, "state": "construction_failed", "content_id": None,
                     "lineage_id": None, "structure_signature": None})
        ok, report = full_gate(rows, {"source": 30, "development": 30, "final": 6}, c)
        self.assertTrue(ok, report)
        self.assertTrue(report["floors"]["source_stage"])
        self.assertTrue(report["floors"]["development_stage"])
        self.assertTrue(report["floors"]["final_stage"])
        # ID splits are content-disjoint; a source-test structure can match train.
        train_sigs = {r["structure_signature"] for r in rows if r["split"] == "source_train"}
        test_sigs = {r["structure_signature"] for r in rows if r["split"] == "source_test"}
        self.assertTrue(train_sigs & test_sigs)

    def test_full_gate_rejects_pair_profile_in_source_train(self):
        c = small_config()
        rows = full_fixture()
        leaked = next(r for r in rows if r["split"] == "source_train")
        leaked["profile"] = "eq/neq"
        ok, report = full_gate(rows, {"source": 30, "development": 30, "final": 6}, c)
        self.assertFalse(ok)
        self.assertFalse(report["floors"]["row_schema"])
        self.assertFalse(report["floors"]["heldout_compositions_absent_from_source_train"])

    def test_full_gate_rejects_development_validation_diagnostic_structure_overlap(self):
        c = small_config()
        rows = full_fixture()
        validation = next(r for r in rows if r["split"] == "development_validation")
        diagnostic = next(r for r in rows if r["split"] == "development_diagnostic_test")
        diagnostic["structure_signature"] = validation["structure_signature"]
        ok, report = full_gate(rows, {"source": 30, "development": 30, "final": 6}, c)
        self.assertFalse(ok)
        self.assertFalse(report["floors"]["development_structure_closure"])

    def test_full_value_and_given_floors_use_accepted_population(self):
        c = small_config()
        rows = full_fixture()
        rows.append(row("final-outside", split="final_test", partition="final", profile="eq/neq/add_mod",
                        eligible=True, envelope=False, signature="outside", given=2))
        rows[-1]["values"] = [0] * 100
        ok, report = full_gate(rows, {"source": 30, "development": 30, "final": 6}, c)
        self.assertTrue(ok, report)

    def test_full_split_minimum_counts_A_not_raw_unique_rows(self):
        c = small_config()
        rows = full_fixture()
        rows.append(row("dev-diag-ineligible", split="development_diagnostic_test",
                        partition="development", profile="eq/neq", cell="B",
                        eligible=False, envelope=False, signature="dev-diag-ineligible"))
        c["full_floors"]["development_split_minima"]["development_diagnostic_test"] = 4
        ok, report = full_gate(rows, {"source": 30, "development": 30, "final": 6}, c)
        self.assertFalse(ok)
        self.assertEqual(sum(r.get("split") == "development_diagnostic_test" for r in rows), 4)
        self.assertEqual(report["development"]["floors"]["development_diagnostic_test_total_A_minimum"], False)
        self.assertFalse(report["development"]["floors"]["development_diagnostic_test_total_A_minimum"])

    def test_full_validation_capacity_has_explicit_100A_12_signature_gate(self):
        c = small_config()
        rows = full_fixture()
        c["development_floors"]["eligible_envelope_groups_per_profile"] = 5
        c["development_floors"]["accepted_signatures_per_profile"] = 5
        c["support"]["cell_min"] = 0
        c["support"]["cohort_mass_min"] = 0.0
        c["development_floors"]["eligible_envelope_per_profile_cell"] = 0
        for r in rows:
            if r.get("split") == "development_validation" and r.get("profile") == "eq/neq" and r.get("cell") != "A":
                r["envelope"] = False
                r["prompt_bytes"] = 500
                r["first_solution_nodes"] = 2
        ok, report = full_gate(rows, {"source": 30, "development": 30, "final": 6}, c)
        self.assertFalse(ok)
        self.assertFalse(report["development"]["floors"]["eq/neq/validation_A_min"])


def full_fixture():
    rows = []
    # Source train has each source primitive in each support cell.
    for profile in CONFIG["source_profiles"]:
        for cell in "ABCD":
            rows.append(row(f"src-tr-{profile}-{cell}", split="source_train", partition="source",
                            profile=profile, cell=cell, signature=f"src-tr-{profile}-{cell}",
                            given=2 if cell in "AC" else 3))
    # ID source test deliberately repeats a train structure with distinct content.
    rows.append(row("src-val", split="source_validation", partition="source", profile="eq",
                    cell="A", signature="src-tr-eq-A", given=2))
    for cell in "ABCD":
        rows.append(row(f"src-test-{cell}", split="source_test", partition="source", profile="neq",
                        cell=cell, signature=f"src-tr-neq-{cell}", given=2 if cell in "AC" else 3))
    # All three validation profiles have fixed-cell support; ADD profiles carry
    # the required g2/g3 no-unary core signatures.
    for profile in CONFIG["development_profiles"]:
        for i, cell in enumerate("ABCD"):
            hard = profile in ("eq/add_mod", "neq/add_mod") and i < 2
            rows.append(row(f"dev-val-{profile}-{cell}", split="development_validation", partition="development",
                            profile=profile, cell=cell, signature=f"dev-val-{profile}-{cell}",
                            given=2 if i % 2 == 0 else 3, hard=hard, gac=(i == 0)))
        rows.append(row(f"dev-diag-{profile}", split="development_diagnostic_test", partition="development",
                        profile=profile, cell="A", signature=f"dev-diag-{profile}", given=2))
    for i, cell in enumerate("ABCD"):
        rows.append(row(f"final-{cell}", split="final_test", partition="final",
                        profile="eq/neq/add_mod", cell=cell, signature=f"final-{cell}",
                        given=2 if i % 2 == 0 else 3))
    return rows


if __name__ == "__main__":
    unittest.main()
