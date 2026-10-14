"""Task-free descriptive inventory for the closed rank qualification."""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
import time
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path

from . import certificate_rank_study as rank

CONFIG_SHA256 = "63145f650d17dc5bd205bfd1dc0db38b7c2b68dadc8ea445ddf8805e296bd394"
CONFIG_REL = Path("configs/tasks/certificate_rank_support_diagnosis_v1.json")
INPUT_REL = Path("runs/certificate-rank-qualification-20261012-v1")
SOURCE_RECEIPT_REL = Path("configs/splits/certificate-rank-qualification-result.json")
ORIGINAL_SOURCE_COMMIT = "41bd7846f8435d02616a526ef6dbdbae9663271d"
PROMPT_BINS = ((0, 511), (512, 639), (640, 767), (768, 1023), (1024, None))
NODE_BINS = ((1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, None))
CELLS = {"A": (2, 8), "B": (3, 6), "C": (3, 7), "D": (3, 8)}
PROFILES = ("eq/neq", "eq/add_mod", "neq/add_mod")
PARTITIONS = ("source", "development")
PROJECTION_FIELDS = ("partition", "candidate_index", "profile", "cell", "branch", "split",
                     "content_id", "structure_signature", "first_seen_content", "oracle_unique",
                     "eligible", "sensitive", "envelope", "given_count", "prompt_bytes",
                     "first_solution_nodes", "gac")


class DiagnosisError(ValueError):
    pass


def _bin(value, bins):
    if type(value) is not int:
        raise DiagnosisError("invalid permitted integer")
    for i, (lo, hi) in enumerate(bins):
        if value >= lo and (hi is None or value <= hi):
            return i
    raise DiagnosisError("permitted value outside bins")


def _guard(check_budget):
    if check_budget is not None:
        check_budget()


def _validate_projection(row):
    if type(row) is not dict or set(row) != set(PROJECTION_FIELDS):
        raise DiagnosisError("invalid permitted projection")
    if row["partition"] not in PARTITIONS or type(row["candidate_index"]) is not int or row["candidate_index"] < 0:
        raise DiagnosisError("invalid permitted index")
    if row["split"] not in ("source_train", "development_validation"):
        raise DiagnosisError("invalid permitted split")
    if row["partition"] == "source" and row["split"] != "source_train":
        raise DiagnosisError("invalid source split")
    if row["partition"] == "development" and row["split"] != "development_validation":
        raise DiagnosisError("invalid development split")
    if row["partition"] == "source":
        if row["profile"] not in ("eq", "neq", "add_mod") or row["branch"] != "ordinary":
            raise DiagnosisError("invalid source stratum")
    elif (row["profile"] not in PROFILES or row["branch"] not in ("ordinary", "no_unary") or
          (row["profile"] == "eq/neq" and row["branch"] != "ordinary")):
        raise DiagnosisError("invalid development stratum")
    if row["cell"] not in CELLS:
        raise DiagnosisError("invalid cell")
    if type(row["content_id"]) is not str or not row["content_id"] or type(row["structure_signature"]) is not str:
        raise DiagnosisError("invalid permitted identity")
    for name in ("first_seen_content", "oracle_unique", "eligible", "sensitive", "envelope"):
        if type(row[name]) is not bool:
            raise DiagnosisError("invalid permitted flag")
    for name in ("given_count", "prompt_bytes", "first_solution_nodes"):
        if type(row[name]) is not int or row[name] < (1 if name == "first_solution_nodes" else 0):
            raise DiagnosisError("invalid permitted count")
    if row["given_count"] != CELLS[row["cell"]][0]:
        raise DiagnosisError("given count does not match scheduled cell")
    if type(row["gac"]) is not dict or any(type(row["gac"].get(k)) is not bool for k in ("stalled", "solved", "contradiction")):
        raise DiagnosisError("invalid permitted stall annotation")


def _routed_projection(raw, partition, index):
    """Check prefix identity, route first, and only then touch permitted metadata."""
    if not isinstance(raw, dict) or raw.get("partition") != partition or type(raw.get("candidate_index")) is not int or raw["candidate_index"] != index:
        raise DiagnosisError("input prefix index mismatch")
    split = raw.get("split")
    if split is None:
        return None
    if partition == "source":
        if split in ("source_validation", "source_test"):
            return None
        if split != "source_train":
            raise DiagnosisError("invalid source routing label")
    else:
        if split == "development_diagnostic_test":
            return None
        if split != "development_validation":
            raise DiagnosisError("invalid development routing label")
    projection = {k: raw.get(k) for k in PROJECTION_FIELDS}
    projection["partition"] = partition
    projection["candidate_index"] = index
    _validate_projection(projection)
    return projection


def _iter_prefix(stream, source_count, development_count, guard=None):
    """Read the exact registered ordered prefixes, routing before projection fields."""
    for partition, limit in (("source", source_count), ("development", development_count)):
        for index in range(limit):
            rank._check(guard)
            line = stream.readline()
            if not line:
                raise DiagnosisError("truncated input prefix")
            raw = rank._strict_load(line)
            projection = _routed_projection(raw, partition, index)
            if projection is not None:
                yield projection
    if stream.readline():
        raise DiagnosisError("unexpected rows beyond registered prefixes")


def analyze_rows(rows, *, projection_stream=None, check_budget=None):
    """Analyze only already-routed allowlisted annotations; rows can be streamed."""
    totals = Counter()
    strata = defaultdict(Counter)
    bins = defaultdict(Counter)
    contingencies = defaultdict(Counter)
    cell_contingencies = defaultdict(Counter)
    cell_cohort_bins = defaultdict(Counter)
    source_primitives = defaultdict(Counter)
    for primitive in ("eq", "neq", "add_mod"):
        source_primitives[primitive]
    raw_nodes = defaultdict(Counter)
    sigs = defaultdict(lambda: {"E": set(), "A": set()})
    sig_excluded = defaultdict(Counter)
    sig_accepted = defaultdict(set)
    signature_scopes = defaultdict(lambda: {"E": set(), "A": set(), "excluded": Counter()})
    seen = {}
    count = 0
    for row in rows:
        _guard(check_budget)
        _validate_projection(row)
        count += 1
        if row["split"] not in ("source_train", "development_validation"):
            continue
        key = row["content_id"]
        if key in seen and row["first_seen_content"]:
            raise DiagnosisError("inconsistent first-seen annotations")
        if key not in seen:
            seen[key] = row["first_seen_content"]
        if row["first_seen_content"] is True and row["oracle_unique"] is True:
            u_cohort = "source_train" if row["partition"] == "source" else row["profile"]
            totals[(u_cohort, "U")] += 1
            totals[(u_cohort, "sensitive_U")] += int(row["sensitive"])
            if row["partition"] == "source":
                source_primitives[row["profile"]]["U"] += 1
                source_primitives[row["profile"]]["sensitive_U"] += int(row["sensitive"])
            if projection_stream is not None:
                projection_stream.write(rank._json_bytes({k: row[k] for k in PROJECTION_FIELDS}))
        if row["first_seen_content"] is not True or row["oracle_unique"] is not True:
            continue
        E = row["eligible"] is True and (row["partition"] != "source" or row["sensitive"] is True)
        if row["partition"] == "source" and row["eligible"] and not row["sensitive"]:
            raise DiagnosisError("source eligible row lacks sensitivity")
        if not E:
            continue
        prompt_bin, node_bin = _bin(row["prompt_bytes"], PROMPT_BINS), _bin(row["first_solution_nodes"], NODE_BINS)
        A = row["envelope"] is True
        if A != (prompt_bin == 4 and node_bin == 2):
            raise DiagnosisError("envelope annotation disagrees with frozen bins")
        cohort = "source_train" if row["partition"] == "source" else row["profile"]
        branch = row["branch"]
        cell = row["cell"]
        group = (cohort, branch, cell)
        totals[(cohort, "E")] += 1
        totals[(cohort, "A")] += int(A)
        totals[(cohort, "stalled_E")] += int(row["gac"]["stalled"])
        totals[(cohort, "stalled_A")] += int(A and row["gac"]["stalled"])
        totals[(cohort, "stalled_excluded_E")] += int(not A and row["gac"]["stalled"])
        raw_nodes[(cohort, branch, cell)][row["first_solution_nodes"]] += 1
        if row["partition"] == "source":
            raw_nodes[(f"source_train/{row['profile']}", branch, cell)][row["first_solution_nodes"]] += 1
            for metric, value in (("E", 1), ("A", int(A)), ("stalled_E", int(row["gac"]["stalled"])),
                                  ("stalled_A", int(A and row["gac"]["stalled"])),
                                  ("stalled_excluded_E", int(not A and row["gac"]["stalled"]))):
                source_primitives[row["profile"]][metric] += value
        strata[group]["E"] += 1
        strata[group]["A"] += int(A)
        strata[group]["stalled_E"] += int(row["gac"]["stalled"])
        strata[group]["stalled_A"] += int(A and row["gac"]["stalled"])
        strata[group]["stalled_excluded_E"] += int(not A and row["gac"]["stalled"])
        strata[group]["nonstalled_A"] += int(A and not row["gac"]["stalled"])
        strata[group]["nonstalled_excluded_E"] += int(not A and not row["gac"]["stalled"])
        stratum = bins[(cohort, branch, cell)]
        stratum[(prompt_bin, node_bin, "E")] += 1
        stratum[(prompt_bin, node_bin, "A")] += int(A)
        stratum[(prompt_bin, node_bin, "stalled_E")] += int(row["gac"]["stalled"])
        stratum[(prompt_bin, node_bin, "stalled_A")] += int(A and row["gac"]["stalled"])
        if row["partition"] == "source":
            primitive_bins = bins[(f"source_train/{row['profile']}", branch, cell)]
            for metric, value in (("E", 1), ("A", int(A)), ("stalled_E", int(row["gac"]["stalled"])),
                                  ("stalled_A", int(A and row["gac"]["stalled"]))):
                primitive_bins[(prompt_bin, node_bin, metric)] += value
        for metric, value in (("E", 1), ("A", int(A)), ("stalled_E", int(row["gac"]["stalled"])),
                              ("stalled_A", int(A and row["gac"]["stalled"]))):
            contingencies[(prompt_bin, node_bin, cohort)][metric] += value
            cell_contingencies[(cell, prompt_bin, node_bin, cohort)][metric] += value
            cell_cohort_bins[(cell, prompt_bin, node_bin, cohort)][metric] += value
        sigs[cohort]["E"].add(row["structure_signature"])
        scopes = (cohort, f"{cohort}/{branch}/{cell}")
        if row["partition"] == "source":
            scopes += (f"source_train/{row['profile']}",)
        for scope in scopes:
            signature_scopes[scope]["E"].add(row["structure_signature"])
        if not A:
            sig_excluded[cohort][row["structure_signature"]] += 1
            for scope in scopes:
                signature_scopes[scope]["excluded"][row["structure_signature"]] += 1
        if A:
            sigs[cohort]["A"].add(row["structure_signature"])
            sig_accepted[cohort].add(row["structure_signature"])
            for scope in scopes:
                signature_scopes[scope]["A"].add(row["structure_signature"])
        category = "both_fail" if prompt_bin != 4 and node_bin != 2 else "prompt_only" if prompt_bin != 4 else "node_only" if node_bin != 2 else "accepted"
        strata[group][category] += 1
        strata[(cohort, branch, cell, "bins", prompt_bin, node_bin)][category] += 1
    # Explicit zero-inclusive unselected grids for every cohort/branch/cell stratum.
    branch_grid = {}
    cohorts = ("source_train", *PROFILES)
    for cohort in cohorts:
        branches = ("ordinary", "no_unary") if cohort != "source_train" else ("ordinary",)
        for branch in branches:
            for cell in CELLS:
                key = (cohort, branch, cell)
                branch_grid["/".join(key)] = {f"p{p}-n{n}": {metric: bins[key][(p, n, metric)]
                                            for metric in ("E", "A", "stalled_E", "stalled_A")}
                                               for p in range(5) for n in range(7)}
    source_primitive_grid = {}
    for primitive in ("eq", "neq", "add_mod"):
        for cell in CELLS:
            key = (f"source_train/{primitive}", "ordinary", cell)
            source_primitive_grid[f"{primitive}/{cell}"] = {f"p{p}-n{n}": {metric: bins[key][(p, n, metric)]
                                                   for metric in ("E", "A", "stalled_E", "stalled_A")}
                                                               for p in range(5) for n in range(7)}
    cohorts = ("source_train", *PROFILES)
    complete_grid = {}
    for cell in CELLS:
        for p in range(5):
            for n in range(7):
                key = f"{cell}/p{p}-n{n}"
                vectors = {cohort: {metric: cell_cohort_bins[(cell, p, n, cohort)][metric]
                                    for metric in ("E", "A", "stalled_E", "stalled_A")}
                           for cohort in cohorts}
                complete_grid[key] = {"cohorts": vectors,
                                      "minimum_E_capacity": min(v["E"] for v in vectors.values()),
                                      "capacity_at_least_10_every_cohort": all(v["E"] >= 10 for v in vectors.values())}
    all_bins = {f"p{p}-n{n}": {cohort: {metric: contingencies[(p, n, cohort)][metric]
                                            for metric in ("E", "A", "stalled_E", "stalled_A")}
                                for cohort in cohorts} for p in range(5) for n in range(7)}
    cell_bins = {f"{cell}/p{p}-n{n}": {cohort: {metric: cell_contingencies[(cell, p, n, cohort)][metric]
                                                       for metric in ("E", "A", "stalled_E", "stalled_A")}
                                         for cohort in cohorts}
                 for cell in CELLS for p in range(5) for n in range(7)}
    totals_out = {}
    for cohort in cohorts:
        e, a = totals[(cohort, "E")], totals[(cohort, "A")]
        stalled_e, stalled_a = totals[(cohort, "stalled_E")], totals[(cohort, "stalled_A")]
        def fraction(numerator, denominator):
            return {"numerator": numerator, "denominator": denominator,
                    "value": numerator / denominator if denominator else None}
        totals_out[cohort] = {"U": totals[(cohort, "U")], "sensitive_U": totals[(cohort, "sensitive_U")],
                              "E": e, "A": a, "excluded_E": e-a,
                              "cell_A": {cell: sum(counter["A"] for key, counter in strata.items()
                                                   if len(key) == 3 and key[0] == cohort and key[2] == cell)
                                         for cell in CELLS},
                              "A_over_E": {"numerator": a, "denominator": e,
                                           "value": a / e if e else None},
                              "stalled_E": stalled_e, "stalled_A": stalled_a,
                              "stalled_excluded_E": totals[(cohort, "stalled_excluded_E")],
                              "nonstalled_A": totals[(cohort, "A")] - stalled_a,
                              "nonstalled_excluded_E": (e-a) - totals[(cohort, "stalled_excluded_E")],
                              "stalled_fraction_E": fraction(stalled_e, e),
                              "stalled_fraction_A": fraction(stalled_a, a),
                              "SigE": len(sigs[cohort]["E"]), "SigA": len(sigs[cohort]["A"]),
                              "SigE_minus_SigA": len(sigs[cohort]["E"]-sigs[cohort]["A"]),
                              "excluded_E_on_absent_signatures": sum(n for sig, n in sig_excluded[cohort].items() if sig not in sig_accepted[cohort]),
                              "excluded_E_on_represented_signatures": sum(n for sig, n in sig_excluded[cohort].items() if sig in sig_accepted[cohort])}
    branch_summary = {}
    for cohort in cohorts:
        branch_summary[cohort] = {}
        for branch in (("ordinary", "no_unary") if cohort != "source_train" else ("ordinary",)):
            cells_for_branch = [value for key, value in strata.items() if len(key) == 3 and key[0] == cohort and key[1] == branch]
            branch_summary[cohort][branch] = {name: sum(value[name] for value in cells_for_branch)
                                              for name in ("E", "A", "stalled_E", "stalled_A")}
    all_signature_scopes = list(cohorts)
    all_signature_scopes.extend(f"{cohort}/{branch}/{cell}"
                                for cohort in cohorts
                                for branch in (("ordinary", "no_unary") if cohort != "source_train" else ("ordinary",))
                                for cell in CELLS)
    all_signature_scopes.extend(f"source_train/{primitive}" for primitive in ("eq", "neq", "add_mod"))
    signature_summary = {}
    for scope in all_signature_scopes:
        values = signature_scopes[scope]
        signature_summary[scope] = {"SigE": len(values["E"]), "SigA": len(values["A"]),
                                    "SigE_minus_SigA": len(values["E"]-values["A"]),
                                    "excluded_E_on_absent_signatures": sum(n for sig, n in values["excluded"].items()
                                                                           if sig not in values["A"]),
                                    "excluded_E_on_represented_signatures": sum(n for sig, n in values["excluded"].items()
                                                                                if sig in values["A"])}
    applicability = {"/".join((cohort, branch)): ("not_applicable" if branch == "no_unary" and cohort in ("source_train", "eq/neq") else "available")
                     for cohort in cohorts for branch in (("ordinary", "no_unary") if cohort != "source_train" else ("ordinary",))}
    return {"status": "ANALYZED_FIXTURE" if check_budget is not None else "ANALYZED",
            "row_count": count, "cohort_totals": totals_out,
            "source_primitive_totals": {key: dict(value) for key, value in source_primitives.items()},
            "source_primitive_cell_bin_grid": source_primitive_grid,
            "branch_totals": branch_summary,
            "signature_inventories": signature_summary,
            "raw_node_histograms": {"/".join(key): {str(n): value[n] for n in sorted(value)} for key, value in raw_nodes.items()},
            "strata": {"/".join(map(str, k)): dict(v) for k, v in strata.items()},
            "complete_cell_bin_grid": complete_grid, "branch_cell_bin_grid": branch_grid,
            "unselected_bin_contingency": all_bins,
            "cell_unselected_bin_contingency": cell_bins, "branch_applicability": applicability,
            "cell_contract": {cell: {"given_count": values[0], "constraint_count": values[1]}
                              for cell, values in CELLS.items()},
            "signature_counts_are_distinct_from_content_counts": True}


def compare_public(report, analysis):
    """Check preserved public totals without revising any gate or status."""
    expected = {"source_train": (1317, 1317, 1156, 1071, 932), "eq/neq": (279, 212, 190, 185, 169),
                "eq/add_mod": (729, 593, 462, 560, 449), "neq/add_mod": (345, 316, 222, 313, 220)}
    expected_cells = {"source_train": {"A": 253, "B": 250, "C": 302, "D": 351},
                      "eq/neq": {"A": 36, "B": 26, "C": 48, "D": 80},
                      "eq/add_mod": {"A": 123, "B": 129, "C": 116, "D": 94},
                      "neq/add_mod": {"A": 55, "B": 51, "C": 54, "D": 62}}
    try:
        for cohort, values in expected.items():
            item = analysis["cohort_totals"][cohort]
            got = (item["U"], item["E"], item["A"], item["SigE"], item["SigA"])
            if (not rank._same_typed(got, values) or not rank._same_typed(item["cell_A"], expected_cells[cohort]) or
                    (cohort == "source_train" and item["sensitive_U"] != 1317)):
                raise DiagnosisError("public aggregate mismatch")
        expected_branches = {"source_train": {"ordinary": (1317, 1156)},
                             "eq/neq": {"ordinary": (212, 190)},
                             "eq/add_mod": {"ordinary": (351, 301), "no_unary": (242, 161)},
                             "neq/add_mod": {"ordinary": (168, 136), "no_unary": (148, 86)}}
        for cohort, branches in expected_branches.items():
            for branch, values in branches.items():
                actual = analysis["branch_totals"][cohort][branch]
                if not rank._same_typed((actual["E"], actual["A"]), values):
                    raise DiagnosisError("public aggregate mismatch")
        dev = report["development_gate"]
        expected_profile = {
            "eq/neq": (279, 212, 190, 185, 169, 22, 16, 0, 0),
            "eq/add_mod": (729, 593, 462, 560, 449, 131, 111, 242, 161),
            "neq/add_mod": (345, 316, 222, 313, 220, 94, 93, 148, 86),
        }
        for profile, values in expected_profile.items():
            item = dev["profiles"][profile]
            got = (item["U"], item["E"], item["A"], item["SigE"], item["SigA"], item["E_only"],
                   item["SigE_only"], item["hard_E"], item["hard_A"])
            if not rank._same_typed(got, values) or not rank._same_typed(item["cell_A"], expected_cells[profile]):
                raise DiagnosisError("public aggregate mismatch")
        if (report.get("status") != "NO_GO_DEVELOPMENT" or
                not rank._same_typed((dev["E"], dev["A"], dev["stalled_E"], dev["hard_E"], dev["ordinary_E"]),
                                     (1121, 874, 452, 390, 731))):
            raise DiagnosisError("public terminal mismatch")
        source_report = report["source_gate"]
        if not rank._same_typed((source_report["first_seen_unique"], source_report["sensitive"],
                                 source_report["E"], source_report["A"], source_report["signature_E"],
                                 source_report["signature_A"], source_report["signature_E_only"],
                                 source_report["cell_counts"]),
                                (1317, 1317, 1317, 1156, 1071, 932, 139, expected_cells["source_train"])):
            raise DiagnosisError("public aggregate mismatch")
        masses = {"source_train": report["source_gate"]["support"]["mass"],
                  **{f"development_validation_{p}": value["support"]["mass"]
                     for p, value in dev["profiles"].items()}}
        for cohort, item in (("source_train", analysis["cohort_totals"]["source_train"]),
                             *((f"development_validation_{p}", analysis["cohort_totals"][p]) for p in PROFILES)):
            observed = masses[cohort]
            if abs(observed - (item["A"] / item["E"])) > 1e-15:
                raise DiagnosisError("public support-mass mismatch")
        _compare_pooled_stalls(report, analysis)
    except (KeyError, TypeError) as exc:
        raise DiagnosisError("public aggregate mismatch") from exc


def _compare_pooled_stalls(report, analysis):
    """Bind the independently derived development stall inventory to its public receipt."""
    dev = report["development_gate"]
    cohorts = [analysis["cohort_totals"][profile] for profile in PROFILES]
    if any(type(item.get(name)) is not int for item in cohorts for name in ("E", "stalled_E")):
        raise DiagnosisError("invalid pooled stall counts")
    e = sum(item["E"] for item in cohorts)
    stalled = sum(item["stalled_E"] for item in cohorts)
    branches = analysis["branch_totals"]
    ordinary = sum(branches[p]["ordinary"]["E"] for p in PROFILES)
    hard = sum(branches[p]["no_unary"]["E"] for p in ("eq/add_mod", "neq/add_mod"))
    derived = (e, stalled, hard, ordinary, stalled / e if e else None)
    public = (dev["E"], dev["stalled_E"], dev["hard_E"], dev["ordinary_E"], dev["stalled_fraction_E"])
    if not rank._same_typed(derived, public):
        raise DiagnosisError("public pooled stall mismatch")


def _qualification_boundary(report, config):
    """Copy the original public decision boundary as context; calculate no new gate."""
    if report.get("status") != "NO_GO_DEVELOPMENT" or report.get("g2_status") != "not_assessed":
        raise DiagnosisError("original qualification boundary mismatch")
    dev = report["development_gate"]
    source = report["source_gate"]
    return {"original_terminal_status": report["status"], "original_g2_status": report["g2_status"],
            "support_floor": config["support"]["cohort_mass_min"],
            "source_support": source["support"], "development_support": dev["common_support"],
            "floor_status": dev["floors"], "scope": "frozen_annotations", "analysis": "agent_analysis",
            "new_samples": False, "independent_reproduction": False, "archival_backup": False}


def _verify_original_producers(initial_source, root, guard):
    """Check the registered original producer map against its tracked receipt and raw Git blobs."""
    receipt = rank._strict_load((root / SOURCE_RECEIPT_REL).read_text(encoding="utf-8"))
    expected = receipt.get("source", {}).get("producer_sha256")
    observed = initial_source.get("producer_sha256") if type(initial_source) is dict else None
    if (type(expected) is not dict or len(expected) != 40 or type(observed) is not dict or
            not rank._same_typed(observed, expected) or
            any(type(name) is not str or not name.startswith("dwr/") or not name.endswith(".py") or
                type(digest) is not str or len(digest) != 64 or
                any(char not in "0123456789abcdef" for char in digest)
                for name, digest in expected.items())):
        raise DiagnosisError("original producer map mismatch")
    for name, digest in expected.items():
        rank._check(guard)
        blob = rank.subprocess.run(["git", "show", f"{ORIGINAL_SOURCE_COMMIT}:{name}"], cwd=root,
                                   check=True, capture_output=True).stdout
        rank._check(guard)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise DiagnosisError("original producer blob mismatch")


def _config(path):
    p = Path(path).resolve(strict=True)
    raw = p.read_bytes()
    value = rank._strict_load(raw.decode("utf-8"))
    expected = (Path(__file__).resolve().parents[2] / CONFIG_REL).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256 or hashlib.sha256(expected).hexdigest() != CONFIG_SHA256:
        raise DiagnosisError("closed config identity mismatch")
    parsed = rank._strict_load(expected.decode("utf-8"))
    if not rank._same_typed(value, parsed):
        raise DiagnosisError("closed config contract mismatch")
    return value, CONFIG_SHA256, p


def _verify_input_bindings(input_dir, config, root, guard):
    """Verify only registered hashes and public identity metadata; never parse ledger payloads here."""
    hashes = config["input_artifact_sha256"]
    if type(hashes) is not dict or set(hashes) != {"manifest.initial.json", "manifest.json", "outcomes.jsonl",
                                                   "records.jsonl", "replay.json", "report.json"}:
        raise DiagnosisError("bound input artifact map mismatch")
    for name, digest in hashes.items():
        rank._check(guard)
        if rank._sha(input_dir / name, guard) != digest:
            raise DiagnosisError("bound input hash mismatch")
    rank._check(guard)
    public_report = rank._strict_load((input_dir / "report.json").read_text(encoding="utf-8"))
    rank._check(guard)
    replay = rank._strict_load((input_dir / "replay.json").read_text(encoding="utf-8"))
    rank._check(guard)
    manifest = rank._strict_load((input_dir / "manifest.json").read_text(encoding="utf-8"))
    rank._check(guard)
    initial_input = rank._strict_load((input_dir / "manifest.initial.json").read_text(encoding="utf-8"))
    rank._check(guard)
    input_config, _, input_config_path = rank.load_config(root / config["input_config_path"])
    source_count, dev_count = config["registered_attempts"]["source"], config["registered_attempts"]["development"]
    manifest_artifacts = {name: digest for name, digest in hashes.items() if name != "manifest.json"}
    if (public_report.get("status") != config["input_terminal_status"] or
            manifest.get("status") != config["input_terminal_status"] or
            initial_input.get("status") != "running" or
            manifest.get("config_sha256") != config["input_config_sha256"] or
            initial_input.get("config_sha256") != config["input_config_sha256"] or
            initial_input.get("config_path") != str(input_config_path) or
            initial_input.get("output_dir") != str(input_dir.resolve(strict=True)) or
            not rank._same_typed(initial_input.get("resolved_config"), input_config) or
            initial_input.get("source", {}).get("git_commit") != config["input_scientific_source_commit"] or
            not rank._same_typed(manifest.get("source"), initial_input.get("source")) or
            initial_input.get("source", {}).get("git_clean") is not True or
            not rank._argv_binds(initial_input.get("requested_argv"), input_config_path, input_dir) or
            not rank._same_typed(manifest.get("artifacts"), manifest_artifacts) or
            replay.get("status") != "PASS_REPLAY" or
            replay.get("attempts_rederived") != source_count + dev_count or replay.get("exact_indices") is not True or
            manifest.get("g2_status") != "not_assessed" or public_report.get("g2_status") != "not_assessed" or
            not rank._same_typed(public_report.get("raw_attempts"), config["registered_attempts"])):
        raise DiagnosisError("bound input terminal mismatch")
    _verify_original_producers(initial_input.get("source"), root, guard)
    return public_report, input_config


def run_diagnosis(config_path, output_path, *, argv=None, _guard=None):
    """Bounded production runner. A supplied guard is fixture-only and cannot pass."""
    fixture = _guard is not None
    guard = _guard
    if guard is None:
        from .run_support import ResourceGuard
        guard = ResourceGuard(wall_seconds=60, memory_mib=256)
    out = Path(output_path).absolute()
    active_started, cpu_started = time.monotonic(), time.process_time()
    owns_output = False
    try:
        rank._safe_path(out)
        rank._check(guard)
        config, cfg_hash, cfg_path = _config(config_path)
        root = Path(__file__).resolve().parents[2]
        if cfg_path != (root / CONFIG_REL).resolve(strict=True):
            raise DiagnosisError("config path is not the registered file")
        expected_out = (root / config["output_path"]).absolute()
        if out != expected_out or out.exists():
            raise DiagnosisError("output path is not the unused registered path")
        if not rank._argv_binds(list(argv or sys.argv), cfg_path, expected_out):
            raise DiagnosisError("command does not bind registered config and output")
        out.mkdir(parents=True)
        owns_output = True
        source = {"fixture_only": True} if fixture else rank._source_identity(guard)
        initial = {"status": "RUNNING", "requested_argv": list(argv or sys.argv), "config_path": str(cfg_path),
                   "config_sha256": cfg_hash, "resolved_config": config,
                   "source": source, "output_dir": str(out),
                   "budgets": config["limits"], "rng": "not used"}
        initial.update({"input_run_path": config["input_run_path"],
                        "input_artifact_sha256": config["input_artifact_sha256"],
                        "input_config_sha256": config["input_config_sha256"],
                        "input_source_commit": config["input_scientific_source_commit"],
                        "python": sys.version, "platform": platform.platform(),
                        "cwd": str(Path.cwd()), "started_utc": datetime.now(timezone.utc).isoformat()})
        rank._write(out / "manifest.initial.json", initial, guard)
        if fixture:
            raise DiagnosisError("fixture guard cannot certify production diagnosis")
        input_dir = root / INPUT_REL
        if input_dir.absolute() != (root / config["input_run_path"]).absolute():
            raise DiagnosisError("input run path differs from registered binding")
        public_report, input_config = _verify_input_bindings(input_dir, config, root, guard)
        source_count, dev_count = config["registered_attempts"]["source"], config["registered_attempts"]["development"]
        identity_seen = set()
        with (input_dir / "outcomes.jsonl").open("r", encoding="utf-8") as stream, (out / "permitted_rows.jsonl").open("wb") as projected:
            def permitted_rows():
                for allowed in _iter_prefix(stream, source_count, dev_count, guard):
                    ident = allowed["content_id"]
                    if allowed["first_seen_content"]:
                        if ident in identity_seen:
                            raise DiagnosisError("inconsistent global first-seen annotation")
                        identity_seen.add(ident)
                    yield allowed
            analysis = analyze_rows(permitted_rows(), projection_stream=projected, check_budget=lambda: rank._check(guard))
            projected.flush()
            os.fsync(projected.fileno())
            compare_public(public_report, analysis)
            analysis["qualification_boundary"] = _qualification_boundary(public_report, input_config)
            analysis["resources"] = {"elapsed_seconds": time.monotonic() - active_started,
                                     "cpu_seconds": time.process_time() - cpu_started,
                                     "peak_rss_bytes": getattr(guard, "peak_bytes", None),
                                     "accounting_boundary": "through permitted projection; report and manifest writes remain guard-checked"}
            rank._write(out / "report.json", analysis, guard)
        hashes_out = {name: rank._sha(out/name, guard) for name in ("manifest.initial.json", "permitted_rows.jsonl")}
        analysis["status"] = "PASS_DIAGNOSIS"
        analysis["artifacts"] = hashes_out
        rank._write(out / "report.json", analysis, guard)
        manifest_artifacts = {**hashes_out, "report.json": rank._sha(out / "report.json", guard)}
        rank._check(guard)
        rank._write(out / "manifest.json", {"status": "PASS_DIAGNOSIS", "config_sha256": cfg_hash,
                    "source": source, "artifacts": manifest_artifacts,
                    "input_artifacts": config["input_artifact_sha256"],
                    "resources": {"elapsed_seconds": time.monotonic() - active_started,
                                  "cpu_seconds": time.process_time() - cpu_started,
                                  "peak_rss_bytes": getattr(guard, "peak_bytes", None),
                                  "accounting_boundary": "through report hash; final manifest write remains guard-checked"}}, guard)
        return analysis
    except Exception as exc:
        if owns_output:
            _revoke(out, type(exc).__name__)
        if isinstance(exc, DiagnosisError):
            raise
        raise DiagnosisError(type(exc).__name__) from exc


def _revoke(output, reason):
    """Revoke every provisional success marker without rescanning artifacts."""
    report = {"status": "NOT_ASSESSED", "phase": "runner", "reason": reason, "artifacts": {}}
    rank._write(output / "report.json", report, None)
    rank._write(output / "manifest.json", {"status": "NOT_ASSESSED", "artifacts": {}, "reason": reason}, None)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        run_diagnosis(args.config, args.output, argv=sys.argv)
        return 0
    except Exception as exc:
        print(f"diagnosis failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
