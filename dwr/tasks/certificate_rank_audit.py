"""Pure qualification and full-stage gates for rank-certificate study rows.

Rows are already frozen audit projections. Split routing happens before any
scientific field is inspected; this module never opens artifacts or creates
tasks.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .certificate_support import FIVE_STRATA, is_envelope, support_counts, support_stratum

CELLS = ("A", "B", "C", "D")
SOURCE_SPLITS = ("source_train", "source_validation", "source_test")
DEV_SPLITS = ("development_validation", "development_diagnostic_test")


def _routed(rows: Iterable[dict], split: str) -> list[dict]:
    # Intentionally inspect only split until the permitted cohort is known.
    return [row for row in rows if row.get("split") == split]


def _first_unique(rows: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for row in rows:
        cid = row.get("content_id")
        if not isinstance(cid, str) or cid in seen:
            continue
        seen.add(cid)
        if row.get("first_seen_content", True) is True and row.get("oracle_unique") is True:
            out.append(row)
    return out


def _support(rows: list[dict], config: dict) -> dict:
    return support_counts(rows, config)


def _node2_envelope(row: dict, config: dict) -> bool:
    return is_envelope(row, config) and support_stratum(row, config) in ("A", "B", "C", "D", "A2", "B2", "C2", "D2")


def five_row_schema_valid(row: dict, config: dict) -> bool:
    """Validate new-only support annotations and the row states that omit them."""
    if config.get("support", {}).get("scheme") != "five-fixed-strata-v1":
        return True
    if type(row) is not dict:
        return False
    state = row.get("state")
    if state == "construction_failed":
        return (row.get("split") is None and "fine_stratum" not in row and row.get("envelope") is False and
                row.get("oracle_unique") is False and row.get("eligible") is False and
                row.get("sensitive") is False and row.get("first_seen_content") is False)
    if state == "protected_integrity_only":
        return (row.get("split") in SOURCE_SPLITS + DEV_SPLITS and "fine_stratum" not in row and
                row.get("envelope") is False and row.get("oracle_unique") is False and
                row.get("eligible") is False and row.get("sensitive") is False and
                type(row.get("first_seen_content")) is bool and
                type(row.get("duplicate_content")) is bool and
                row["first_seen_content"] is (not row["duplicate_content"]))
    if state not in ("first_seen", "duplicate_content") or "fine_stratum" not in row:
        return False
    if (row.get("split") not in SOURCE_SPLITS + DEV_SPLITS + ("final_test",) or
            row.get("cell") not in CELLS or type(row.get("first_solution_nodes")) is not int or
            type(row.get("prompt_bytes")) is not int or type(row.get("oracle_unique")) is not bool or
            type(row.get("envelope")) is not bool or
            row.get("branch") not in ("ordinary", "no_unary") or
            not isinstance(row.get("model_input_sha256"), str) or len(row["model_input_sha256"]) != 64 or
            any(ch not in "0123456789abcdef" for ch in row["model_input_sha256"])):
        return False
    expected = support_stratum(row, config)
    fine = row["fine_stratum"]
    return fine == expected and row["envelope"] is (row["oracle_unique"] and expected is not None)


def row_schema_valid(row: dict, config: dict) -> bool:
    """Check the full public audit row shape, including five-mode annotation rules."""
    if type(row) is not dict or not five_row_schema_valid(row, config):
        return False
    source_profiles = set(config["source_profiles"])
    development_profiles = set(config["development_profiles"])
    final_profiles = set(config["final_profiles"])
    expected_partition = {**{s: "source" for s in SOURCE_SPLITS},
                          **{s: "development" for s in DEV_SPLITS}, "final_test": "final"}
    split = row.get("split")
    if split is None and row.get("state") == "construction_failed":
        part, profile = row.get("partition"), row.get("profile")
        allowed = source_profiles if part == "source" else development_profiles if part == "development" else final_profiles if part == "final" else ()
        return profile in allowed and type(row.get("candidate_index")) is int and row["candidate_index"] >= 0
    if split not in expected_partition:
        return False
    cid, lid, part = (row.get(k) for k in ("content_id", "lineage_id", "partition"))
    profile = row.get("profile")
    allowed_profiles = source_profiles if split in SOURCE_SPLITS else development_profiles if split in DEV_SPLITS else final_profiles
    required_bools = ("oracle_unique", "eligible", "envelope", "sensitive", "first_seen_content")
    valid = (all(isinstance(x, str) for x in (cid, lid, part, profile, row.get("structure_signature")))
             and part == expected_partition[split] and profile in allowed_profiles
             and all(type(row.get(k)) is bool for k in required_bools)
             and row.get("cell") in CELLS and type(row.get("given_count")) is int
             and isinstance(row.get("values"), list)
             and all(type(v) is int and v in (0, 1, 2) for v in row.get("values", ())))
    if split in DEV_SPLITS and row.get("state") != "protected_integrity_only":
        valid = valid and isinstance(row.get("gac"), dict)
    if config.get("support", {}).get("scheme") == "five-fixed-strata-v1":
        valid = valid and row.get("branch") in ("ordinary", "no_unary")
        if row.get("state") != "protected_integrity_only":
            valid = valid and isinstance(row.get("gac"), dict)
    return bool(valid)


def _value_given(rows: list[dict], config: dict) -> tuple[bool, dict]:
    vals = Counter(v for r in rows for v in r.get("values", ()))
    total = sum(vals.values())
    floors = config["full_floors"]
    fractions = {str(v): vals[v] / total if total else 0.0 for v in (0, 1, 2)}
    givens = sorted({r.get("given_count") for r in rows if type(r.get("given_count")) is int})
    return (total > 0 and all(fractions[str(v)] >= floors["value_fraction_each"] for v in (0, 1, 2))
            and set(floors["given_counts"]).issubset(givens),
            {"value_counts": {str(v): vals[v] for v in (0, 1, 2)},
             "value_fractions": fractions, "given_counts": givens})


def _dedup_by_content(rows: Iterable[dict]) -> list[dict]:
    return _first_unique(rows)


def _common_core_counts(rows: list[dict]) -> dict:
    out: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if (r.get("eligible") is True and r.get("branch") == "no_unary" and
                r.get("profile") in ("eq/add_mod", "neq/add_mod") and
                r.get("given_count") in (2, 3) and isinstance(r.get("core"), dict)):
            sig = r["core"].get("signature")
            if isinstance(sig, str):
                out[f"{r['profile']}/g{r['given_count']}"] .add(sig)
    return {k: len(v) for k, v in sorted(out.items())}


def source_gate(rows: list[dict], raw_attempts: int, config: dict) -> tuple[bool, dict]:
    train = _first_unique(_routed(rows, "source_train"))
    sensitive = [r for r in train if r.get("sensitive") is True]
    eligible = [r for r in sensitive if r.get("eligible") is True]
    accepted = [r for r in eligible if is_envelope(r, config)]
    floors, profiles = config["source_floors"], config["source_profiles"]
    node2_accepted = [r for r in accepted if _node2_envelope(r, config)]
    cells = Counter(r["cell"] for r in node2_accepted)
    pcells = Counter((r.get("profile"), r.get("cell")) for r in node2_accepted)
    sigs = {r.get("structure_signature") for r in accepted if isinstance(r.get("structure_signature"), str)}
    psigs = {p: {r.get("structure_signature") for r in accepted if r.get("profile") == p} for p in profiles}
    sig_support = Counter(r.get("structure_signature") for r in accepted)
    values, valmeta = _value_given(accepted, config)
    checks = {
        "train_groups": len(accepted) >= floors["train_groups"],
        "train_signatures": len(sigs) >= floors["train_signatures"],
        "primitives": len({k for r in accepted for k in str(r.get("profile", "")).split("/")}) >= floors["primitives"],
        "cell_min": all(cells[c] >= config["support"]["cell_min"] for c in CELLS),
        "profile_cell_capacity": all(pcells[p, c] >= floors["eligible_envelope_train_per_profile_cell"] for p in profiles for c in CELLS),
        "profile_signatures": all(len(psigs[p]) >= floors["accepted_signatures_per_profile"] for p in profiles),
        "raw_yield": type(raw_attempts) is int and raw_attempts > 0 and len(accepted) / raw_attempts >= floors["eligible_envelope_train_per_all_raw_attempts"],
        "value_mass": values,
        "given_counts": set(floors["given_counts"]).issubset({r.get("given_count") for r in accepted}),
        "sensitivity": bool(train) and len(sensitive) / len(train) >= floors["sensitivity"],
        "eligible_envelope_fraction": bool(train) and len(accepted) / len(train) >= floors["eligible_envelope_fraction"],
        "sensitive_signatures": all(sig_support[s] >= floors["sensitive_per_accepted_signature"] for s in sigs),
    }
    if config["support"].get("scheme") == "five-fixed-strata-v1":
        a3_primitives = {primitive for r in accepted if support_stratum(r, config) == "A3"
                         for primitive in str(r.get("profile", "")).split("/")}
        required_primitives = {primitive for profile in profiles for primitive in str(profile).split("/")}
        checks["a3_source_primitive_presence"] = required_primitives.issubset(a3_primitives)
        primitive_strata: dict[str, Counter] = defaultdict(Counter)
        for r in accepted:
            stratum = support_stratum(r, config)
            if stratum is not None:
                for primitive in str(r.get("profile", "")).split("/"):
                    primitive_strata[primitive][stratum] += 1
    else:
        a3_primitives = set()
    support = _support(sensitive, config)
    checks["common_support"] = support["pass"]
    report = {"floors": checks, "first_seen_unique": len(train), "sensitive": len(sensitive),
              "eligible": len(eligible), "E": len(eligible), "A": len(accepted),
              "E_only": len(eligible) - len(accepted), "signature_E": len({r.get("structure_signature") for r in eligible}),
              "signature_A": len(sigs), "signature_E_only": len({r.get("structure_signature") for r in eligible} - sigs),
              "sensitive_fraction": len(sensitive) / len(train) if train else 0.0,
              "eligible_fraction": len(eligible) / len(train) if train else 0.0,
              "A_fraction": len(accepted) / len(train) if train else 0.0,
              "A_per_raw_attempt": len(accepted) / raw_attempts if type(raw_attempts) is int and raw_attempts > 0 else 0.0,
              "cell_counts": {c: cells[c] for c in CELLS}, "support": support, **valmeta}
    if config["support"].get("scheme") == "five-fixed-strata-v1":
        report["a3_source_primitives"] = sorted(a3_primitives)
        report["node2_cell_counts"] = {c: cells[c] for c in CELLS}
        report["source_primitive_by_fine_stratum"] = {
            primitive: {stratum: primitive_strata[primitive][stratum] for stratum in FIVE_STRATA}
            for primitive in sorted(primitive_strata)}
    return all(checks.values()), report


def development_gate(rows: list[dict], source_rows: list[dict], config: dict) -> tuple[bool, dict]:
    # Protected validation/test cohorts are discarded before content or metric reads.
    profs = config["development_profiles"]
    by_profile = {p: _first_unique(_routed(rows, "development_validation")) for p in profs}
    by_profile = {p: [r for r in rs if r.get("profile") == p] for p, rs in by_profile.items()}
    source = _first_unique(_routed(source_rows, "source_train"))
    floors = config["development_floors"]
    checks: dict[str, bool] = {}
    report_profiles = {}
    for p, rs in by_profile.items():
        E = [r for r in rs if r.get("eligible") is True]
        A = [r for r in E if is_envelope(r, config)]
        sigE = {r.get("structure_signature") for r in E}
        sigA = {r.get("structure_signature") for r in A}
        cells = Counter(r.get("cell") for r in A if _node2_envelope(r, config))
        checks[f"{p}/A_groups"] = len(A) >= floors["eligible_envelope_groups_per_profile"]
        checks[f"{p}/SigA"] = len(sigA) >= floors["accepted_signatures_per_profile"]
        checks[f"{p}/cell_min"] = all(cells[c] >= floors["eligible_envelope_per_profile_cell"] for c in CELLS)
        checks[f"{p}/E_fraction"] = bool(rs) and len(E) / len(rs) >= floors["scientific_eligibility_fraction"]
        checks[f"{p}/A_fraction"] = bool(rs) and len(A) / len(rs) >= floors["eligible_envelope_fraction"]
        support = _support(E, config)
        checks[f"{p}/support"] = support["pass"]
        hardE = [r for r in E if r.get("branch") == "no_unary"]
        cores = _common_core_counts(hardE)
        if p in ("eq/add_mod", "neq/add_mod"):
            for g in (2, 3):
                checks[f"{p}/hard_core_g{g}"] = cores.get(f"{p}/g{g}", 0) >= floors["core_signatures_per_profile_given_count"]
        report_profiles[p] = {"U": len(rs), "E": len(E), "A": len(A), "E_only": len(E)-len(A),
                              "SigE": len(sigE), "SigA": len(sigA), "SigE_only": len(sigE-sigA),
                              "cell_A": {c: cells[c] for c in CELLS}, "support": support, "hard_E": len(hardE),
                              "hard_A": sum(r.get("branch") == "no_unary" for r in A), "core_signatures": cores}
    allE = [r for rs in by_profile.values() for r in rs if r.get("eligible") is True]
    stalled = [r for r in allE if isinstance(r.get("gac"), dict) and r["gac"].get("stalled") is True]
    stalled_ids = {r.get("content_id") for r in stalled}
    diff = config["difficulty"]
    checks["pooled_stall_count"] = len(stalled_ids) >= diff["development_stalled_groups_min"]
    checks["pooled_stall_fraction"] = bool(allE) and len(stalled_ids) / len(allE) >= diff["development_stalled_fraction_min"]
    checks["profiles_with_stalls"] = len({r.get("profile") for r in stalled}) >= diff["development_profiles_with_stall_min"]
    support_cohorts = {"source_train": [r for r in source if r.get("sensitive") is True and r.get("eligible") is True]}
    support_cohorts.update({f"development_validation_{p}": [r for r in rs if r.get("eligible") is True] for p, rs in by_profile.items()})
    common = {k: _support(v, config) for k, v in support_cohorts.items()}
    checks["common_support"] = all(v["pass"] for v in common.values())
    report = {"floors": checks, "profiles": report_profiles, "E": len(allE), "A": sum(x["A"] for x in report_profiles.values()),
              "stalled_E": len(stalled_ids), "stalled_fraction_E": len(stalled_ids)/len(allE) if allE else 0.0,
              "ordinary_E": sum(r.get("branch") != "no_unary" for r in allE),
              "hard_E": sum(r.get("branch") == "no_unary" for r in allE), "common_support": common}
    return all(checks.values()), report


def full_stage_gate(rows: list[dict], partition: str, config: dict, raw_attempts: int,
                    *, source_rows: Iterable[dict] = ()) -> tuple[bool, dict]:
    if partition not in ("source", "development", "final"):
        raise ValueError("partition must be source, development or final")
    if partition == "source":
        permitted = [r for s in SOURCE_SPLITS for r in _routed(rows, s)]
        source_train = [r for r in permitted if r.get("split") == "source_train"]
        qual, qreport = source_gate(source_train, raw_attempts, config)
        all_source = _first_unique(permitted)
        source_E = [r for r in all_source if r.get("sensitive") is True and r.get("eligible") is True]
        source_A = [r for r in source_E if is_envelope(r, config)]
        source_sensitive = [r for r in all_source if r.get("sensitive") is True]
        full = config["full_floors"]
        value_ok, value_meta = _value_given(source_A, config)
        checks = {
            "source_quota": len(source_A) >= full["source_quota"],
            "accepted_groups": len(source_A) >= full["accepted_groups_per_partition"],
            "accepted_signatures": len({r.get("structure_signature") for r in source_A}) >= full["accepted_signatures_per_partition"],
            "source_sensitivity_all_splits": bool(all_source) and len(source_sensitive)/len(all_source) >= config["source_floors"]["sensitivity"],
            "values_and_givens": value_ok,
        }
        for split, minimum in full["source_split_minima"].items():
            checks[f"{split}_minimum_A"] = sum(r.get("split") == split and r.get("sensitive") is True and
                                                  r.get("eligible") is True and is_envelope(r, config)
                                                  for r in all_source) >= minimum
        # Full source's source-train floors remain, and ID-test ALL-E support is mandatory.
        testE = [r for r in _first_unique(_routed(permitted, "source_test"))
                 if r.get("sensitive") is True and r.get("eligible") is True]
        support = _support(testE, config)
        checks["source_train"] = qual
        checks["source_id_test_support"] = support["pass"]
        return all(checks.values()), {"partition": partition, "qualification_floors": qreport["floors"], "source_id_test": support,
                    "floors": checks, "source_E": len(source_E), "source_A": len(source_A), **value_meta}
    if partition == "development":
        permitted = _routed(rows, "development_validation") + _routed(rows, "development_diagnostic_test")
        unique = _first_unique(permitted)
        full = config["full_floors"]
        checks: dict[str, bool] = {}
        profiles_report = {}
        for p in config["development_profiles"]:
            vals = [r for r in unique if r.get("profile") == p]
            validation = [r for r in vals if r.get("split") == "development_validation"]
            diagnostic = [r for r in vals if r.get("split") == "development_diagnostic_test"]
            E = [r for r in vals if r.get("eligible") is True]
            A = [r for r in E if is_envelope(r, config)]
            E_validation = [r for r in validation if r.get("eligible") is True]
            A_validation = [r for r in E_validation if is_envelope(r, config)]
            counts = Counter(r.get("cell") for r in A)
            bysplit = Counter(r.get("split") for r in vals)
            validation_cells = Counter(r.get("cell") for r in A_validation if _node2_envelope(r, config))
            support = _support(E_validation, config)
            checks[f"{p}/quota"] = len(A) >= full["development_quota_per_profile"]
            checks[f"{p}/support"] = support["pass"]
            checks[f"{p}/split_validation_A"] = sum(r.get("split") == "development_validation" and
                                                      r.get("eligible") is True and is_envelope(r, config) for r in vals) >= full["development_split_minima_per_profile"]["development_validation"]
            checks[f"{p}/split_diagnostic_A"] = sum(r.get("split") == "development_diagnostic_test" and
                                                     r.get("eligible") is True and is_envelope(r, config) for r in vals) >= full["development_split_minima_per_profile"]["development_diagnostic_test"]
            checks[f"{p}/A_groups"] = len(A) >= full["accepted_groups_per_mixed_profile"]
            checks[f"{p}/SigA"] = len({r.get("structure_signature") for r in A}) >= full["accepted_signatures_per_mixed_profile"]
            checks[f"{p}/eligibility"] = bool(vals) and len(E)/len(vals) >= full["mixed_eligibility_fraction"]
            checks[f"{p}/qualification_E_fraction"] = bool(validation) and len(E_validation)/len(validation) >= config["development_floors"]["scientific_eligibility_fraction"]
            checks[f"{p}/qualification_A_fraction"] = bool(validation) and len(A_validation)/len(validation) >= config["development_floors"]["eligible_envelope_fraction"]
            checks[f"{p}/qualification_A_cell_min"] = all(validation_cells[c] >= config["development_floors"]["eligible_envelope_per_profile_cell"] for c in CELLS)
            checks[f"{p}/validation_A_min"] = len(A_validation) >= config["development_floors"]["eligible_envelope_groups_per_profile"]
            checks[f"{p}/validation_SigA_min"] = len({r.get("structure_signature") for r in A_validation}) >= config["development_floors"]["accepted_signatures_per_profile"]
            profiles_report[p] = {"U": len(vals), "E": len(E), "A": len(A), "validation_U": len(validation),
                                  "validation_E": len(E_validation), "validation_A": len(A_validation),
                                  "support": support, "splits": dict(bysplit)}
            if config["support"].get("scheme") == "five-fixed-strata-v1":
                diagnostic_fine = Counter(support_stratum(r, config) for r in diagnostic
                                          if r.get("eligible") is True and is_envelope(r, config))
                diagnostic_nonzero = all(diagnostic_fine[stratum] > 0 for stratum in FIVE_STRATA)
                checks[f"{p}/diagnostic_fine_stratum_nonzero"] = diagnostic_nonzero
                profiles_report[p]["diagnostic_fine_stratum_counts"] = {
                    stratum: diagnostic_fine[stratum] for stratum in FIVE_STRATA}
        for split, minimum in full["development_split_minima"].items():
            checks[f"{split}_total_A_minimum"] = sum(r.get("split") == split and r.get("eligible") is True and
                                                       is_envelope(r, config) for r in unique) >= minimum
        allE = [r for r in unique if r.get("eligible") is True]
        validationE = [r for r in unique if r.get("split") == "development_validation" and r.get("eligible") is True]
        source_train_E = [r for r in _first_unique(_routed(source_rows, "source_train"))
                          if r.get("sensitive") is True and r.get("eligible") is True]
        source_support = _support(source_train_E, config)
        checks["source_train_common_support"] = source_support["pass"]
        stalled = [r for r in validationE if isinstance(r.get("gac"), dict) and r["gac"].get("stalled") is True]
        ids = {r.get("content_id") for r in stalled}
        d = config["difficulty"]
        checks["pooled_stalls"] = len(ids) >= d["development_stalled_groups_min"] and bool(validationE) and len(ids)/len(validationE) >= d["development_stalled_fraction_min"] and len({r.get("profile") for r in stalled}) >= d["development_profiles_with_stall_min"]
        hardE = [r for r in validationE if r.get("branch") == "no_unary"]
        cores = _common_core_counts(hardE)
        for p in ("eq/add_mod", "neq/add_mod"):
            for g in (2, 3):
                checks[f"{p}/core_g{g}"] = cores.get(f"{p}/g{g}", 0) >= config["development_floors"]["core_signatures_per_profile_given_count"]
        allA = [r for r in unique if r.get("eligible") is True and is_envelope(r, config)]
        value_ok, value_meta = _value_given(allA, config)
        checks["values_and_givens"] = value_ok
        return all(checks.values()), {"partition": partition, "floors": checks, "profiles": profiles_report, "E": len(allE), "A": sum(v["A"] for v in profiles_report.values()), "stalled_E": len(ids), "core_signatures": cores, "source_train_support": source_support, **value_meta}
    # Final is a single triple-kind test cohort. It is structurally read only here.
    final_rows = _first_unique(_routed(rows, "final_test"))
    E = [r for r in final_rows if r.get("eligible") is True]
    A = [r for r in E if is_envelope(r, config)]
    full = config["full_floors"]
    support = _support(E, config)
    value_ok, valmeta = _value_given(A, config)
    checks = {"quota": len(A) >= full["final_quota"],
              "accepted_groups": len(A) >= full["accepted_groups_per_partition"],
              "accepted_signatures": len({r.get("structure_signature") for r in A}) >= full["accepted_signatures_per_partition"],
              "support": support["pass"],
              "eligibility": bool(final_rows) and len(E)/len(final_rows) >= full["mixed_eligibility_fraction"],
              "values_and_givens": value_ok}
    return all(checks.values()), {"partition": partition, "floors": checks, "U": len(final_rows), "E": len(E), "A": len(A), "support": support, **valmeta}


def full_gate(all_rows: list[dict], raw_attempts_by_partition: dict[str, int], config: dict) -> tuple[bool, dict]:
    # Canonical ownership, held-out composition leakage and global first-seen
    # content uniqueness are structural full-audit checks.
    lineage_content: dict[str, str] = {}
    content_owner: dict[str, tuple[str, str]] = {}
    checks = {"row_schema": True, "lineage_consistency": True, "content_split_closure": True,
              "heldout_compositions_absent_from_source_train": True}
    expected_partition = {**{s: "source" for s in SOURCE_SPLITS},
                          **{s: "development" for s in DEV_SPLITS}, "final_test": "final"}
    for r in all_rows:
        if type(r) is not dict:
            checks["row_schema"] = False
            continue
        if not row_schema_valid(r, config):
            checks["row_schema"] = False
        split = r.get("split")
        if split is None and r.get("state") == "construction_failed":
            # A consumed failed construction is an attempt, not a content row.
            continue
        if split not in expected_partition:
            checks["row_schema"] = False
            continue
        cid, lid, part = (r.get(k) for k in ("content_id", "lineage_id", "partition"))
        profile = r.get("profile")
        if not all(isinstance(x, str) for x in (cid, lid, part, split)):
            continue
        if lid in lineage_content and lineage_content[lid] != cid:
            checks["lineage_consistency"] = False
        lineage_content[lid] = cid
        owner = content_owner.get(cid)
        if owner and owner != (part, split):
            checks["content_split_closure"] = False
        content_owner.setdefault(cid, (part, split))
    # Source ID split closure is content-based; composition holding applies to
    # held-out pair/triple profiles. Development structure hashes close across
    # validation and diagnostic test, while final remains a separate profile.
    train_profiles = {r.get("profile") for r in _routed(all_rows, "source_train")}
    heldout_target_profiles = {r.get("profile") for r in all_rows
                               if r.get("split") in DEV_SPLITS + ("final_test",)}
    checks["heldout_compositions_absent_from_source_train"] = not (train_profiles & heldout_target_profiles)
    validation_structures = {r.get("structure_signature") for r in _routed(all_rows, "development_validation")}
    diagnostic_structures = {r.get("structure_signature") for r in _routed(all_rows, "development_diagnostic_test")}
    checks["development_structure_closure"] = not (validation_structures & diagnostic_structures)
    source = [r for r in all_rows if r.get("partition") == "source"]
    dev = [r for r in all_rows if r.get("partition") == "development"]
    final = [r for r in all_rows if r.get("partition") == "final"]
    sg, sr = full_stage_gate(source, "source", config, raw_attempts_by_partition.get("source", 0))
    dg, dr = full_stage_gate(dev, "development", config, raw_attempts_by_partition.get("development", 0), source_rows=source)
    fg, fr = full_stage_gate(final, "final", config, raw_attempts_by_partition.get("final", 0))
    checks.update({"source_stage": sg, "development_stage": dg, "final_stage": fg})
    return all(checks.values()), {"floors": checks, "source": sr, "development": dr, "final": fr,
                                  "global_first_seen_content": len(content_owner)}
