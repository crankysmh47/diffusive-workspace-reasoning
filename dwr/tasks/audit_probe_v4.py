"""Independent semantic and aggregate audit for the v4 development proposal."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .audit import audit_interactions
from .canonical import content_id, development_split, lineage_id, source_split, structure_signature
from .oracle import enumerate_assignments, enumerate_solutions
from .run_support import stable_json, strict_json_loads
from .schema import (ConstraintTask, RelationKind, parse_task_json,
                     serialize_model_input, serialize_task)
from .solver import solve

AUDIT_VERSION = "dwr-v4-independent-audit-v1"
RECORD_VERSION = "dwr-v4-probe-accepted-record-v1"
GENERATOR_ID = "dwr-generator-v4-probe"
_RECORD_KEYS = {"record_version", "generator_id", "partition", "candidate_index", "profile", "task",
                "content_id", "lineage_id", "structure_signature", "split", "witness", "solution",
                "solver_stats", "model_input", "model_input_sha256"}
_ASSIGNMENT_KEYS = {f"v{i}" for i in range(6)}
_SOLVER_STAT_KEYS = {"first_solution_nodes", "uniqueness_nodes", "uniqueness_solution_count_capped"}


def _validate_assignment_payload(value: object, field: str) -> None:
    if (not isinstance(value, dict) or set(value) != _ASSIGNMENT_KEYS or
            any(type(item) is not int or item not in (0, 1, 2) for item in value.values())):
        raise ValueError(f"{field} assignment payload is invalid")


def _validate_solver_stats_payload(value: object, field: str) -> None:
    if (not isinstance(value, dict) or set(value) != _SOLVER_STAT_KEYS or
            any(type(item) is not int or item < 0 for item in value.values()) or
            value["uniqueness_solution_count_capped"] not in (1, 2)):
        raise ValueError(f"{field} solver statistics are invalid")


def _validate_private_evidence(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or type(value.get("eligible")) is not bool:
        raise ValueError("outcome private eligibility is malformed")
    kind = value.get("kind")
    if kind == "source":
        counts = value.get("removal_solution_counts")
        if (set(value) != {"kind", "eligible", "removal_solution_counts"} or type(counts) is not list or
                any(type(item) is not int or not 0 <= item <= 729 for item in counts)):
            raise ValueError("outcome source private eligibility is malformed")
    elif kind == "mixed":
        expected = {"kind", "eligible", "kinds", "kind_removal_solution_counts",
                    "factor_graph_connected", "factor_graph_kinds", "all_present_kinds_in_graph", "reason"}
        counts = value.get("kind_removal_solution_counts")
        if (set(value) != expected or type(value.get("kinds")) is not list or
                any(not isinstance(item, str) for item in value["kinds"]) or type(counts) is not list or
                any(type(item) is not int or not 0 <= item <= 729 for item in counts) or
                type(value.get("factor_graph_connected")) is not bool or
                type(value.get("factor_graph_kinds")) is not list or
                any(not isinstance(item, str) for item in value["factor_graph_kinds"]) or
                type(value.get("all_present_kinds_in_graph")) is not bool or
                (value.get("reason") is not None and not isinstance(value.get("reason"), str))):
            raise ValueError("outcome mixed private eligibility is malformed")
    else:
        raise ValueError("outcome private eligibility kind is invalid")


def _assignment(solution) -> dict[str, int]:
    return dict(solution.assignments)


def _bin_prompt(size: int) -> int:
    return 0 if size <= 511 else 1 if size <= 639 else 2 if size <= 767 else 3 if size <= 1023 else 4


def _bin_nodes(nodes: int) -> int:
    return 0 if nodes == 1 else 1 if nodes <= 3 else 2 if nodes <= 7 else 3 if nodes <= 15 else 4 if nodes <= 31 else 5 if nodes <= 63 else 6


def _primitive_holds(kind: str, values: tuple[int, ...]) -> bool:
    if kind == "eq":
        return values[0] == values[1]
    if kind == "neq":
        return values[0] != values[1]
    if kind == "add_mod":
        return (values[0] + values[1]) % 3 == values[2]
    raise ValueError("unknown relation kind")


def _independent_interaction_projection(task: ConstraintTask, partition: str) -> dict[str, Any]:
    """Recompute the private eligibility claim without sampler predicates."""
    # The accepted private evidence schema binds removal counts to sorted kind
    # names, independently from the producer's profile-order convention.
    kinds = tuple(sorted({constraint.kind.value for constraint in task.constraints}))
    if partition == "source":
        counts = []
        for index in range(len(task.constraints)):
            retained = task.constraints[:index] + task.constraints[index + 1:]
            counts.append(len(enumerate_assignments(task.variables, task.givens, retained).solutions))
        return {"kind": "source", "eligible": any(count >= 2 for count in counts),
                "removal_solution_counts": counts}

    counts = []
    for kind in kinds:
        retained = tuple(c for c in task.constraints if c.kind.value != kind)
        counts.append(len(enumerate_assignments(task.variables, task.givens, retained).solutions))
    unknown = set(task.variables) - {given.variable for given in task.givens}
    adjacency: dict[str, set[str]] = {name: set() for name in unknown}
    graph_kinds = set()
    for constraint in task.constraints:
        touched = [name for name in constraint.variables if name in unknown]
        if touched:
            graph_kinds.add(constraint.kind.value)
            for name in touched:
                adjacency[name].update(other for other in touched if other != name)
    reached: set[str] = set()
    stack = [next(iter(unknown))] if unknown else []
    while stack:
        name = stack.pop()
        if name not in reached:
            reached.add(name)
            stack.extend(adjacency[name] - reached)
    connected = bool(unknown) and reached == unknown
    all_kinds = set(kinds) <= graph_kinds
    necessary = all(count >= 2 for count in counts)
    eligible = necessary and connected and all_kinds
    reason = None if eligible else "mixed_kind_not_necessary" if not necessary else "mixed_unknown_factor_disconnected"
    return {"kind": "mixed", "eligible": eligible, "kinds": list(kinds),
            "kind_removal_solution_counts": counts, "factor_graph_connected": connected,
            "factor_graph_kinds": sorted(graph_kinds), "all_present_kinds_in_graph": all_kinds,
            "reason": reason}


def validate_accepted_record(record: dict, config, *, interactions: bool = True) -> dict[str, Any]:
    """Verify record identity, strict solution, solver evidence and privacy projection."""
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        raise ValueError("accepted record has wrong fields")
    if record["record_version"] != RECORD_VERSION or record["generator_id"] != GENERATOR_ID:
        raise ValueError("accepted record version or producer mismatch")
    part = record["partition"]
    if part not in ("source", "development"):
        raise ValueError("accepted record partition invalid")
    _validate_assignment_payload(record["witness"], "accepted record witness")
    _validate_assignment_payload(record["solution"], "accepted record solution")
    _validate_solver_stats_payload(record["solver_stats"], "accepted record")
    index = record["candidate_index"]
    if type(index) is not int or not 0 <= index < 2**64:
        raise ValueError("accepted record candidate index invalid")
    task_value = record["task"]
    if not isinstance(task_value, dict):
        raise ValueError("accepted record task must be an object")
    task = parse_task_json(stable_json(task_value))
    if stable_json(strict_json_loads(serialize_task(task))) != stable_json(task_value):
        raise ValueError("accepted record task serialization is not canonical")
    if len(task.variables) != 6 or len(task.givens) not in (2, 3) or len(task.constraints) not in (6, 7, 8):
        raise ValueError("accepted record violates fixed v4 task sizes")
    kinds = tuple(kind.value for kind in RelationKind if any(c.kind is kind for c in task.constraints))
    profile = tuple(record["profile"])
    expected_profiles = config.profiles_for(part)
    if profile not in expected_profiles or profile != kinds:
        raise ValueError("accepted record profile differs from task relations")
    cid = content_id(task)
    lid = hashlib.sha256(b"dwr-lineage-v1\0" + cid.encode("ascii")).hexdigest()
    signature = structure_signature(task)
    split = source_split(cid) if part == "source" else development_split(signature)
    if (record["content_id"], record["lineage_id"], record["structure_signature"], record["split"]) != (cid, lid, signature, split):
        raise ValueError("accepted record canonical identity or split mismatch")
    oracle = enumerate_solutions(task)
    if len(oracle.solutions) != 1:
        raise ValueError("accepted record is not independently oracle-unique")
    solution = oracle.solutions[0]
    solution_json = stable_json(_assignment(solution))
    if (stable_json(record["solution"]) != solution_json or
            stable_json(record["witness"]) != solution_json):
        raise ValueError("accepted record witness or solution mismatch")
    solved = solve(task)
    stats = {"first_solution_nodes": solved.stats.first_solution_nodes,
             "uniqueness_nodes": solved.stats.uniqueness_nodes,
             "uniqueness_solution_count_capped": solved.stats.uniqueness_solution_count_capped}
    if stable_json(record["solver_stats"]) != stable_json(stats):
        raise ValueError("accepted record solver statistics mismatch")
    model_input = serialize_model_input(task)
    if (record["model_input"] != model_input or
            record["model_input_sha256"] != hashlib.sha256(model_input.encode("utf-8")).hexdigest()):
        raise ValueError("accepted record model input or hash mismatch")
    projected = strict_json_loads(model_input)
    forbidden = {"witness", "solution", "oracle", "solver", "eligibility", "content_id", "lineage_id",
                 "signature", "split", "stratum", "quota_state", "audit_metadata"}
    if forbidden.intersection(projected):
        raise ValueError("audit metadata leaked into model input")
    interaction = _independent_interaction_projection(task, part)
    if interaction["eligible"] is not True:
        raise ValueError("accepted record fails independently recomputed eligibility")
    if interactions:
        checked = audit_interactions(task, part)
        if part == "source":
            if checked["removal_solution_counts"] != interaction["removal_solution_counts"] or checked["sensitive"] != interaction["eligible"]:
                raise ValueError("independent source interaction audit mismatch")
        else:
            if (checked["removal_solution_counts"] != interaction["kind_removal_solution_counts"] or
                    checked["factor_graph_connected"] != interaction["factor_graph_connected"] or
                    checked["factor_graph_kinds"] != interaction["factor_graph_kinds"] or
                    checked["all_present_kinds_in_graph"] != interaction["all_present_kinds_in_graph"] or
                    checked["sensitive"] != interaction["eligible"]):
                raise ValueError("independent mixed interaction audit mismatch")
    prompt_size = len(model_input.encode("utf-8"))
    prompt_bin = _bin_prompt(prompt_size)
    node_bin = _bin_nodes(stats["first_solution_nodes"])
    cell = _cell(task, prompt_bin, node_bin)
    if cell is None:
        raise ValueError("accepted record falls outside the registered support envelope")
    return {"task": task, "partition": part, "candidate_index": index,
            "profile": profile, "content_id": cid, "lineage_id": lid, "signature": signature,
            "split": split, "solution": solution, "interaction": interaction,
            "prompt_utf8_bytes": prompt_size, "prompt_bin": prompt_bin,
            "first_solution_node_bin": node_bin, "cell": cell}


def _cell(task: ConstraintTask, prompt_bin: int, node_bin: int) -> str | None:
    g, c = len(task.givens), len(task.constraints)
    for name, expected in (("A", (2, 8)), ("B", (3, 6)), ("C", (3, 7)), ("D", (3, 8))):
        if (g, c, prompt_bin, node_bin) == (*expected, 4, 2):
            return name
    return None


def validate_outcome(row: dict) -> None:
    fields = {"outcome_version", "partition", "candidate_index", "profile", "stratum", "n", "given_count",
              "constraint_count", "state", "reason", "task_or_null", "witness_or_null", "oracle_solution_or_null",
              "oracle_count_capped_or_null", "solver_stats_or_null", "content_id_or_null", "lineage_id_or_null",
              "structure_signature_or_null", "split_or_null", "private_eligibility_or_null", "denominator_counted",
              "prompt_utf8_bytes_or_null", "prompt_byte_bin_or_null", "first_solution_node_bin_or_null",
              "quota_cell_count_after_or_null"}
    if not isinstance(row, dict) or set(row) != fields:
        raise ValueError("outcome has wrong fields")
    if row["outcome_version"] != "dwr-v4-probe-outcome-v1":
        raise ValueError("outcome version mismatch")
    if (row["partition"] not in ("source", "development") or type(row["candidate_index"]) is not int or
            not 0 <= row["candidate_index"] < 2**64):
        raise ValueError("outcome identity is invalid")
    profiles = {"source": (("eq",), ("neq",), ("add_mod",)),
                "development": (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod"))}
    if row["profile"] not in [list(profile) for profile in profiles[row["partition"]]]:
        raise ValueError("outcome profile is invalid")
    cells = {"A": (2, 8), "B": (3, 6), "C": (3, 7), "D": (3, 8)}
    if (row["stratum"] not in cells or type(row["n"]) is not int or row["n"] != 6 or
            type(row["given_count"]) is not int or type(row["constraint_count"]) is not int or
            (row["given_count"], row["constraint_count"]) != cells[row["stratum"]]):
        raise ValueError("outcome task header is invalid")
    if row["state"] not in ("rejected_proposal", "oracle_non_unique", "fatal_error", "duplicate_content",
                              "scientifically_ineligible", "outside_registered_support_envelope",
                              "retained_diagnostic", "accepted_for_quota"):
        raise ValueError("outcome state is invalid")
    if type(row["denominator_counted"]) is not bool:
        raise ValueError("outcome denominator flag is invalid")
    for field in ("witness_or_null", "oracle_solution_or_null"):
        if row[field] is not None:
            _validate_assignment_payload(row[field], field)
    if row["task_or_null"] is not None:
        if not isinstance(row["task_or_null"], dict):
            raise ValueError("outcome task payload must be an object or null")
        task = parse_task_json(stable_json(row["task_or_null"]))
        if stable_json(strict_json_loads(serialize_task(task))) != stable_json(row["task_or_null"]):
            raise ValueError("outcome task payload is not canonical")
    if row["solver_stats_or_null"] is not None:
        _validate_solver_stats_payload(row["solver_stats_or_null"], "outcome")
    _validate_private_evidence(row["private_eligibility_or_null"])
    reason_sets = {"rejected_proposal": {"no_source_eq_witness", "no_neq_witness",
            "no_feasible_constraint_vector", "no_feasible_neq_constraint_total"},
        "oracle_non_unique": {None}, "fatal_error": {"oracle_witness_disagreement", "solver_oracle_disagreement"},
        "duplicate_content": {"duplicate_content"},
        "scientifically_ineligible": {"source_insensitive", "mixed_kind_not_necessary", "mixed_unknown_factor_disconnected"},
        "outside_registered_support_envelope": {"outside_registered_support_envelope"},
        "retained_diagnostic": {"development_diagnostic_split"}, "accepted_for_quota": {None}}
    if row["reason"] not in reason_sets[row["state"]]:
        raise ValueError("outcome reason does not match its state")
    if row["state"] == "retained_diagnostic" and row["denominator_counted"]:
        raise ValueError("diagnostic outcome cannot enter a denominator")
    if row["state"] == "accepted_for_quota" and row["quota_cell_count_after_or_null"] is None:
        raise ValueError("accepted outcome lacks post-quota count")
    for field in ("oracle_count_capped_or_null", "prompt_utf8_bytes_or_null", "prompt_byte_bin_or_null",
                  "first_solution_node_bin_or_null", "quota_cell_count_after_or_null"):
        if row[field] is not None and type(row[field]) is not int:
            raise ValueError("outcome integer field is invalid")
    for field in ("content_id_or_null", "lineage_id_or_null", "structure_signature_or_null"):
        value = row[field]
        if value is not None and (not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("outcome hash is malformed")
    count = row["oracle_count_capped_or_null"]
    if count is not None and count not in (0, 1, 2):
        raise ValueError("outcome oracle count is outside the capped domain")
    if count == 1 and not (row["state"] == "fatal_error" and
                            row["reason"] == "solver_oracle_disagreement"):
        for field in ("task_or_null", "witness_or_null", "oracle_solution_or_null", "solver_stats_or_null",
                      "content_id_or_null", "lineage_id_or_null", "structure_signature_or_null",
                      "split_or_null", "private_eligibility_or_null", "prompt_utf8_bytes_or_null",
                      "prompt_byte_bin_or_null", "first_solution_node_bin_or_null"):
            if row[field] is None:
                raise ValueError("unique outcome lacks a required nested payload")
    if row["state"] == "rejected_proposal":
        expect_witness = row["reason"] in ("no_feasible_constraint_vector", "no_feasible_neq_constraint_total")
        if (row["task_or_null"] is not None or (row["witness_or_null"] is not None) != expect_witness or
                count is not None or any(row[field] is not None for field in (
                    "oracle_solution_or_null", "solver_stats_or_null", "content_id_or_null",
                    "lineage_id_or_null", "structure_signature_or_null", "split_or_null",
                    "private_eligibility_or_null", "prompt_utf8_bytes_or_null", "prompt_byte_bin_or_null",
                    "first_solution_node_bin_or_null", "quota_cell_count_after_or_null"))):
            raise ValueError("proposal rejection has a state-incompatible nested payload")
    elif row["state"] == "oracle_non_unique":
        if (count != 2 or row["task_or_null"] is None or row["witness_or_null"] is None or
                any(row[field] is not None for field in ("oracle_solution_or_null", "solver_stats_or_null",
                    "content_id_or_null", "lineage_id_or_null", "structure_signature_or_null", "split_or_null",
                    "private_eligibility_or_null", "prompt_utf8_bytes_or_null", "prompt_byte_bin_or_null",
                    "first_solution_node_bin_or_null", "quota_cell_count_after_or_null"))):
            raise ValueError("nonunique outcome has a state-incompatible nested payload")
    elif row["state"] == "fatal_error":
        if (count not in (0, 1) or row["task_or_null"] is None or row["witness_or_null"] is None or
                any(row[field] is not None for field in ("oracle_solution_or_null", "solver_stats_or_null",
                    "content_id_or_null", "lineage_id_or_null", "structure_signature_or_null", "split_or_null",
                    "private_eligibility_or_null", "prompt_utf8_bytes_or_null", "prompt_byte_bin_or_null",
                    "first_solution_node_bin_or_null", "quota_cell_count_after_or_null"))):
            raise ValueError("fatal outcome has a state-incompatible nested payload")
    elif row["state"] in ("duplicate_content", "scientifically_ineligible",
                           "outside_registered_support_envelope", "retained_diagnostic", "accepted_for_quota"):
        if count != 1:
            raise ValueError("unique post-oracle outcome lacks a unique oracle count")


def read_jsonl(path: Path, *, check=None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_no, line in enumerate(stream, 1):
            if check:
                check()
            if not line.endswith("\n"):
                raise ValueError(f"JSONL row {line_no} is unterminated")
            value = strict_json_loads(line)
            rows.append(value)
    return rows


def audit_records(records: list[dict], outcomes: list[dict], config, *, check=None,
                  _fixture_targets: dict | None = None) -> dict:
    """Return allowed projections only; opaque cohorts are checked but never aggregated."""
    info_by_key = {}
    seen_content, seen_lineage = set(), set()
    for record in records:
        if check: check()
        if not isinstance(record, dict):
            raise ValueError("accepted record must be an object")
        key = (record.get("partition"), record.get("candidate_index"))
        if key in info_by_key:
            raise ValueError("duplicate accepted record key")
        # Held-out records are opaque here; deterministic replay checks their
        # exact bytes and semantics without exposing their task or statistics.
        if record.get("partition") == "source" and record.get("split") != "source_train":
            info_by_key[key] = {"partition": "source", "candidate_index": record.get("candidate_index"),
                                "split": record.get("split"), "profile": tuple(record.get("profile", ())) }
            continue
        info = validate_accepted_record(record, config)
        if info["content_id"] in seen_content or info["lineage_id"] in seen_lineage:
            raise ValueError("accepted record content or lineage duplicate")
        seen_content.add(info["content_id"])
        seen_lineage.add(info["lineage_id"])
        info_by_key[key] = info
    first_seen_content: set[str] = set()
    first_seen_lineage: set[str] = set()
    denominator_groups = []
    attempted_dev_profiles = set()
    for row in outcomes:
        if check: check()
        validate_outcome(row)
        if row["partition"] == "development":
            attempted_dev_profiles.add(tuple(row["profile"]))
        if row["oracle_count_capped_or_null"] != 1 or row["state"] in ("duplicate_content", "fatal_error"):
            continue
        # Source validation/test and development diagnostic rows stay opaque.
        split_hint = row["split_or_null"]
        permitted = ((row["partition"] == "source" and split_hint == "source_train") or
                     (row["partition"] == "development" and split_hint == "development_validation"))
        if not permitted:
            continue
        task = parse_task_json(stable_json(row["task_or_null"]))
        part = row["partition"]
        cid = content_id(task)
        lid = hashlib.sha256(b"dwr-lineage-v1\0" + cid.encode("ascii")).hexdigest()
        signature = structure_signature(task)
        split = source_split(cid) if part == "source" else development_split(signature)
        if (cid, lid, signature, split) != (row["content_id_or_null"], row["lineage_id_or_null"],
                                             row["structure_signature_or_null"], row["split_or_null"]):
            raise ValueError("first-seen outcome identity or split mismatch")
        if cid in first_seen_content or lid in first_seen_lineage:
            raise ValueError("repeated unique outcome was not marked duplicate")
        first_seen_content.add(cid)
        first_seen_lineage.add(lid)
        oracle = enumerate_solutions(task)
        if (len(oracle.solutions) != 1 or
                stable_json(_assignment(oracle.solutions[0])) != stable_json(row["oracle_solution_or_null"])):
            raise ValueError("unique outcome oracle claim mismatch")
        interaction = _independent_interaction_projection(task, part)
        if stable_json(row["private_eligibility_or_null"]) != stable_json(interaction):
            raise ValueError("unique outcome private eligibility evidence mismatch")
        solved = solve(task)
        stats = {"first_solution_nodes": solved.stats.first_solution_nodes,
                 "uniqueness_nodes": solved.stats.uniqueness_nodes,
                 "uniqueness_solution_count_capped": solved.stats.uniqueness_solution_count_capped}
        if stable_json(stats) != stable_json(row["solver_stats_or_null"]):
            raise ValueError("unique outcome solver claim mismatch")
        prompt = serialize_model_input(task)
        pbytes = len(prompt.encode("utf-8"))
        pbin = _bin_prompt(pbytes)
        nbin = _bin_nodes(stats["first_solution_nodes"])
        if (row["prompt_utf8_bytes_or_null"], row["prompt_byte_bin_or_null"],
                row["first_solution_node_bin_or_null"]) != (pbytes, pbin, nbin):
            raise ValueError("unique outcome prompt or solver bin mismatch")
        expected_denominator = ((part == "source" and split == "source_train") or
                                (part == "development" and split == "development_validation"))
        if row["denominator_counted"] is not expected_denominator:
            raise ValueError("unique outcome denominator timing or cohort mismatch")
        if not interaction["eligible"]:
            expected_state = "scientifically_ineligible"
            expected_reason = "source_insensitive" if part == "source" else interaction["reason"]
        elif _cell(task, pbin, nbin) is None:
            expected_state = "outside_registered_support_envelope"
            expected_reason = "outside_registered_support_envelope"
        else:
            expected_state, expected_reason = "accepted_for_quota", None
        if (row["state"], row["reason"]) != (expected_state, expected_reason):
            raise ValueError("unique outcome state or reason disagrees with independent eligibility precedence")
        denominator_groups.append({"partition": part, "split": split, "profile": tuple(row["profile"]),
            "task": task, "content_id": cid, "lineage_id": lid, "signature": signature,
            "interaction": interaction, "solution": oracle.solutions[0],
            "cell": _cell(task, pbin, nbin), "state": row["state"]})
    accepted_keys = {(row["partition"], row["candidate_index"])
                     for row in outcomes if row["state"] == "accepted_for_quota"}
    if accepted_keys != set(info_by_key):
        raise ValueError("accepted outcome and record keys disagree")
    for row in outcomes:
        key = (row["partition"], row["candidate_index"])
        info = info_by_key.get(key)
        if row["state"] == "accepted_for_quota" and (
                info is None or row["split_or_null"] != info["split"] or
                row["stratum"] not in ("A", "B", "C", "D") or
                tuple(row["profile"]) != info.get("profile") or
                ("task" in info and (info.get("cell") != row["stratum"] or
                    row["content_id_or_null"] != info["content_id"] or
                    row["structure_signature_or_null"] != info["signature"] or
                    row["lineage_id_or_null"] != info["lineage_id"] or
                    stable_json(row["oracle_solution_or_null"]) != stable_json(_assignment(info["solution"])) or
                    stable_json(row["task_or_null"]) !=
                    stable_json(strict_json_loads(serialize_task(info["task"])))))):
            raise ValueError("accepted outcome identity does not match its record")
    grouped = defaultdict(list)
    for info in denominator_groups:
        if (info["partition"], info["split"]) == ("source", "source_train"):
            grouped[("source",)].append(info)
        elif info["partition"] == "development" and info["split"] == "development_validation":
            grouped[("development", *info["profile"])].append(info)
    accepted_groups = defaultdict(list)
    for info in info_by_key.values():
        if info["partition"] == "source" and info["split"] == "source_train":
            accepted_groups[("source",)].append(info)
        elif info["partition"] == "development" and info["split"] == "development_validation":
            accepted_groups[("development", *info["profile"])].append(info)
    return _public_projection(grouped, accepted_groups, _fixture_targets, attempted_dev_profiles)


def _public_projection(grouped: dict, accepted_groups: dict,
                       fixture_targets: dict | None = None,
                       attempted_dev_profiles: set[tuple[str, ...]] | None = None) -> dict:
    source = grouped.get(("source",), [])
    accepted_source = accepted_groups.get(("source",), [])
    source_sensitive = [i for i in source if i["interaction"]["eligible"]]
    source_cells = Counter(i["cell"] for i in accepted_source)
    eligible_cells = Counter(i["cell"] for i in source_sensitive if i["cell"] is not None)
    signatures = {i["signature"] for i in accepted_source}
    values = Counter()
    for info in accepted_source:
        values.update(_assignment(info["solution"]).values())
    total_values = sum(values.values())
    eligible_cell_total = sum(eligible_cells.values())
    source_projection = {"accepted_groups": len(accepted_source), "unique_oracle_groups": len(source),
        "structure_signatures": len(signatures), "profile_groups": {name: sum(i["profile"] == (name,) for i in accepted_source)
                                                                            for name in ("eq", "neq", "add_mod")},
        "solution_value_counts": {str(v): values[v] for v in range(3)},
        "given_counts": {str(v): sum(len(i["task"].givens) == v for i in accepted_source) for v in (2, 3)},
        "sensitive_groups": len(source_sensitive),
        "sensitivity_fraction": len(source_sensitive) / len(source) if source else 0.0,
        "eligible_envelope_groups": len(accepted_source),
        "envelope_yield": len(accepted_source) / len(source) if source else 0.0,
        "groups_by_cell": {c: source_cells[c] for c in "ABCD"},
        "eligible_groups_by_cell": {c: eligible_cells[c] for c in "ABCD"},
        "four_cell_mass": {"numerator": eligible_cell_total,
                           "denominator": len(source_sensitive),
                           "fraction": eligible_cell_total / len(source_sensitive) if source_sensitive else 0.0}}
    dev = {}
    for profile in (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod")):
        denominator_rows = grouped.get(("development", *profile), [])
        rows = accepted_groups.get(("development", *profile), [])
        eligible = [i for i in denominator_rows if i["interaction"]["eligible"]]
        sigs = {i["signature"] for i in eligible}
        cell_counts = Counter(i["cell"] for i in rows if i["cell"] is not None)
        eligible_cells = Counter(i["cell"] for i in eligible if i["cell"] is not None)
        target_profile = None if fixture_targets is None else fixture_targets["development"].get(profile)
        target_total = 100 if target_profile is None else sum(target_profile.values())
        target_cells = {c: 25 for c in "ABCD"} if target_profile is None else target_profile
        complete = len(rows) == target_total and all(cell_counts[c] == target_cells[c] for c in "ABCD")
        if complete:
            dev["/".join(profile)] = {"status": "complete",
                "eligible_groups": len(eligible), "unique_oracle_groups": len(denominator_rows),
                "eligible_signatures": len(sigs), "scientifically_eligible_groups": len(eligible),
                "scientific_eligibility_fraction": len(eligible) / len(denominator_rows) if denominator_rows else 0.0,
                "eligible_envelope_groups": len(rows),
                "envelope_yield": len(rows) / len(denominator_rows) if denominator_rows else 0.0,
                "groups_by_cell": {c: cell_counts[c] for c in "ABCD"},
                "eligible_groups_by_cell": {c: eligible_cells[c] for c in "ABCD"}}
        else:
            attempted = (profile in (attempted_dev_profiles or set()) or denominator_rows or rows)
            status = "not_assessed" if attempted else "not_attempted"
            dev["/".join(profile)] = {"status": status, "eligible_groups": None,
                "unique_oracle_groups": None, "eligible_signatures": None,
                "scientifically_eligible_groups": None, "scientific_eligibility_fraction": None,
                "eligible_envelope_groups": None, "envelope_yield": None,
                "groups_by_cell": None, "eligible_groups_by_cell": None}
    cohorts = {"source_train_eligible": accepted_source,
        "development_validation_eq_neq_primary_eligible": [i for i in grouped.get(("development", "eq", "neq"), [])
                                                              if i["interaction"]["eligible"]],
        "development_validation_eq_add_mod_primary_eligible": [i for i in grouped.get(("development", "eq", "add_mod"), [])
                                                                  if i["interaction"]["eligible"]],
        "development_validation_neq_add_mod_primary_eligible": [i for i in grouped.get(("development", "neq", "add_mod"), [])
                                                                   if i["interaction"]["eligible"]]}
    proxy = None
    complete_keys = (("source",), ("development", "eq", "neq"),
                     ("development", "eq", "add_mod"), ("development", "neq", "add_mod"))
    minimum_source = 100 if fixture_targets is None else 1
    all_development_complete = all(
        dev["/".join(profile)]["status"] == "complete"
        for profile in (("eq", "neq"), ("eq", "add_mod"), ("neq", "add_mod")))
    if (len(accepted_groups.get(("source",), [])) >= minimum_source and
            (all_development_complete if fixture_targets is not None else all(
                len(accepted_groups.get(key, [])) == 100 for key in complete_keys[1:]))):
        counts = {name: Counter(i["cell"] for i in rows) for name, rows in cohorts.items()}
        minimum_cell = 10 if fixture_targets is None else 1
        common = [cell for cell in "ABCD" if all(counts[name][cell] >= minimum_cell for name in cohorts)]
        proxy_cohorts = {}
        for name, rows in cohorts.items():
            n = len(rows)
            in_common = sum(counts[name][cell] for cell in common)
            fraction = in_common / n if n else 0.0
            proxy_cohorts[name] = {"group_count": n, "groups_by_cell": {c: counts[name][c] for c in "ABCD"},
                "common_cell_groups": {c: counts[name][c] for c in common}, "mass_fraction": fraction,
                "pass": bool(n and len(common) and fraction >= 0.8)}
        proxy = {"cohorts": proxy_cohorts, "pass": all(x["pass"] for x in proxy_cohorts.values())}
    return {"source_train": source_projection, "development_validation_by_profile": dev,
            "four_cohort_proxy": proxy}


def _verify_source_terminal_links(source_doc: dict, replay_doc: dict, audit_doc: dict) -> None:
    """Bind saved source closure facts to the independently replayed/audited stream."""
    if (source_doc["source_quota_counts"] != replay_doc["quota_counts"]["source"] or
            source_doc["source_attempts"] != replay_doc["partition_attempts"]["source"] or
            (source_doc["train_projection"] is not None and
             audit_doc["public_projection"]["source_train"] != source_doc["train_projection"])):
        raise ValueError("source closure facts differ from replay or independent audit")
    if (source_doc["status"] == "PASS_SOURCE_CLOSURE" and
            (replay_doc["quota_closed"]["source"] is not True or
             audit_doc["public_projection"]["source_train"] != source_doc["train_projection"])):
        raise ValueError("passing source closure is not independently supported")
    if (source_doc["status"] != "PASS_SOURCE_CLOSURE" and
            replay_doc["partition_attempts"]["development"] != 0):
        raise ValueError("development stream exists without passing source closure")


def _verify_reconstructed_source_prefix(source_doc: dict, outcomes: list[dict], records: list[dict]) -> None:
    source_outcomes = [row for row in outcomes if row["partition"] == "source"]
    source_records = [row for row in records if row["partition"] == "source"]
    expected_outcome = "".join(stable_json(row) + "\n" for row in source_outcomes).encode("utf-8")
    expected_record = "".join(stable_json(row) + "\n" for row in source_records).encode("utf-8")
    if (source_doc["source_attempts"] != len(source_outcomes) or
            source_doc["outcome_prefix_bytes"] != len(expected_outcome) or
            source_doc["outcome_prefix_sha256"] != hashlib.sha256(expected_outcome).hexdigest() or
            source_doc["record_prefix_bytes"] != len(expected_record) or
            source_doc["record_prefix_sha256"] != hashlib.sha256(expected_record).hexdigest()):
        raise ValueError("source closure prefixes are not the reconstructed source-only stream")


def audit_run(output_dir: Path, config, *, guard=None) -> dict:
    """Validate a production terminal bundle against current source, then re-audit it."""
    from .config_probe_v4 import load_probe_config
    from .probe_v4 import (ARTIFACTS, _producer_identity, _read_jsonl, _source_identity,
        _verify_audit_document, _verify_initial_binding, _verify_replay_document,
        _verify_source_closure, _verify_terminal_manifest, _verify_terminal_report,
        replay_prefix)
    from .run_support import ResourceGuard, sha256_file, strict_json_loads

    output_dir = Path(output_dir).resolve()
    if (output_dir / "fixture_identity.json").exists():
        raise ValueError("fixture artifacts are not valid production audit input")
    initial_path = output_dir / "probe_manifest.initial.json"
    if not initial_path.is_file():
        raise ValueError("production audit requires a closed initial manifest")
    initial = strict_json_loads(initial_path.read_text(encoding="utf-8"))
    if "fixture_id" in initial.get("source", {}):
        raise ValueError("fixture source identity is not valid production audit input")
    if {path.name for path in output_dir.iterdir() if path.is_file()} != set(ARTIFACTS):
        raise ValueError("production terminal artifact set is incomplete or has unexpected files")
    root = Path(__file__).resolve().parents[2]
    try:
        output_dir.relative_to((root / "data").resolve())
    except ValueError as exc:
        raise ValueError("production audit artifacts must be under the ignored data directory") from exc
    current_source = _source_identity(root)
    current_producer = _producer_identity(config, root)
    config_path = Path(initial.get("config_path", "")).resolve()
    actual_config = load_probe_config(config_path)
    if actual_config.config_sha256 != config.config_sha256:
        raise ValueError("production audit config differs from the approved caller config")
    if initial.get("output_dir") != str(output_dir):
        raise ValueError("production audit output path differs from the initial manifest")
    if guard is None:
        guard = ResourceGuard(900, 1024)
    _verify_initial_binding(initial, config, current_source, current_producer, config_path)
    source_doc = strict_json_loads((output_dir / ARTIFACTS[3]).read_text(encoding="utf-8"))
    replay_doc = strict_json_loads((output_dir / ARTIFACTS[4]).read_text(encoding="utf-8"))
    audit_doc = strict_json_loads((output_dir / ARTIFACTS[5]).read_text(encoding="utf-8"))
    report = strict_json_loads((output_dir / ARTIFACTS[6]).read_text(encoding="utf-8"))
    terminal = strict_json_loads((output_dir / ARTIFACTS[7]).read_text(encoding="utf-8"))
    _verify_source_closure(source_doc, output_dir, config, current_source, current_producer)
    _verify_replay_document(replay_doc, output_dir, config, current_source, current_producer)
    _verify_audit_document(audit_doc, output_dir, config, current_source, current_producer)
    # The source closure is a prefix of the same stream that replay and audit
    # certify. Bind its exact assessed facts before considering terminal status.
    _verify_source_terminal_links(source_doc, replay_doc, audit_doc)
    _verify_terminal_report(report, config, current_source, sha256_file(initial_path),
        sha256_file(output_dir / ARTIFACTS[3]), source_doc,
        {**replay_doc, "artifact_sha256": sha256_file(output_dir / ARTIFACTS[4])},
        {**audit_doc, "artifact_sha256": sha256_file(output_dir / ARTIFACTS[5])},
        error=terminal["error"])
    _verify_terminal_manifest(terminal, initial, config, output_dir, replay_doc, report)
    outcomes = _read_jsonl(output_dir / "outcomes.jsonl", check=guard.check)
    records = _read_jsonl(output_dir / "accepted_records.jsonl", check=guard.check)
    _verify_reconstructed_source_prefix(source_doc, outcomes, records)
    if replay_doc["status"] != "PASS_REPLAY":
        if report["status"] != "NOT_ASSESSED":
            raise ValueError("nonpassing replay must have a NOT_ASSESSED terminal decision")
        return audit_doc
    if audit_doc["status"] != "PASS_AUDIT":
        if report["status"] != "NOT_ASSESSED":
            raise ValueError("nonpassing independent audit must have a NOT_ASSESSED terminal decision")
        return audit_doc
    replayed, denominator_digest = replay_prefix(config, outcomes, records, guard, capture_rows=False)
    if (replayed.quota_counts != replay_doc["quota_counts"] or
            denominator_digest != replay_doc["denominator_projection_sha256"]):
        raise ValueError("production replay reconstruction differs from terminal replay document")
    projection = audit_records(records, outcomes, config,
                               check=guard.check)
    if projection != audit_doc["public_projection"]:
        raise ValueError("production semantic projection differs from terminal audit document")
    return audit_doc
