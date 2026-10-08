"""Independent v2 generation audit and decision metrics."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from . import audit as v1
from .dataset_v2 import _producer_identity, registered_limits
from .manifest_v2 import _same_typed, validate_manifest_v2
from .replay_v2 import replay_files_v2
from .run_support import (ResourceGuard, remaining_global_seconds, sha256_file,
                          stable_json, strict_json_loads)
from .schema import parse_task_json

AUDIT_VERSION = "dwr-task-v2-audit-v1"
GATE_FIELDS = {"audit_version", "status", "mode", "g2_status", "config_sha256",
               "source", "generation_manifest_sha256", "generation_artifacts"}
FEASIBILITY_COHORTS = ("source_train", "development_validation_eq_neq",
                       "development_validation_eq_add_mod", "development_validation_neq_add_mod")
PROFILE_COHORT = {("eq", "neq"): FEASIBILITY_COHORTS[1],
                  ("eq", "add_mod"): FEASIBILITY_COHORTS[2],
                  ("neq", "add_mod"): FEASIBILITY_COHORTS[3]}
FULL_PROFILE_COHORT = {profile: v1.COHORTS[index + 1]
                       for index, profile in enumerate(v1.PROFILES["development"])}


def _check(guard: ResourceGuard | None) -> None:
    if guard is not None:
        guard.check()


def _read_jsonl(path: Path, guard: ResourceGuard | None) -> list[dict]:
    rows = []
    _check(guard)
    with Path(path).open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            _check(guard)
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"malformed JSONL line {number}")
            row = strict_json_loads(line)
            if type(row) is not dict:
                raise ValueError(f"JSONL line {number} must be an object")
            rows.append(row)
    _check(guard)
    return rows


def _private_matches(row: dict, interaction: dict) -> None:
    """Compare producer annotations against the independent v1 counterfactual audit."""
    private = row["private_evidence"]
    if (type(private) is not dict or set(private) != {"oracle_unique", "source", "mixed"} or
            private.get("oracle_unique") is not True):
        raise ValueError("unique outcome lacks private oracle evidence")
    if row["partition"] == "source":
        actual = private.get("source")
        expected = {"removal_solution_counts": interaction["removal_solution_counts"],
                    "sensitive": interaction["sensitive"]}
        if not _same_typed(actual, expected) or private.get("mixed") is not None:
            raise ValueError("private source sensitivity annotation disagrees with independent audit")
        return
    mixed = private.get("mixed")
    necessary = all(item["necessary"] for item in interaction["removals"])
    eligible = interaction["sensitive"]
    reason = None if eligible else ("mixed_kind_not_necessary" if not necessary else
                                    "mixed_unknown_factor_disconnected")
    expected = {"kinds": sorted({item["kind"] for item in interaction["removals"]}),
                "kind_removal_solution_counts": interaction["removal_solution_counts"],
                "factor_graph_connected": interaction["factor_graph_connected"],
                "factor_graph_kinds": interaction["factor_graph_kinds"],
                "all_present_kinds_in_graph": interaction["all_present_kinds_in_graph"],
                "primary_eligible": eligible, "ineligibility_reason": reason}
    if not _same_typed(mixed, expected) or private.get("source") is not None:
        raise ValueError("private mixed eligibility annotation disagrees with independent audit")


def _v1_record(task, partition: str, candidate_index: int) -> dict:
    return v1.make_record(task, partition, candidate_index)


def _validate_v2_artifacts(outcomes: list[dict], records: list[dict], config: dict,
                           guard: ResourceGuard | None = None) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    """Validate every first-seen unique task and bind eligible records to those outcomes."""
    accepted_records = {}
    for record in records:
        key = (record.get("partition"), record.get("candidate_index"))
        if key in accepted_records:
            raise ValueError("duplicate accepted record identity")
        accepted_records[key] = record
    infos = []
    for row in outcomes:
        _check(guard)
        if row.get("denominator_counted") is not True:
            continue
        if row.get("oracle_unique") is not True or type(row.get("eligible")) is not bool:
            raise ValueError("denominator row is not a typed unique candidate")
        task = parse_task_json(stable_json(row["task"]))
        if len(set(task.variables) - {given.variable for given in task.givens}) < 2:
            raise ValueError("unique candidate has fewer than two unassigned variables")
        record = _v1_record(task, row["partition"], row["candidate_index"])
        info = v1.validate_record(record, config)
        interaction = v1.audit_interactions(info["task"], info["partition"])
        _private_matches(row, interaction)
        if row["eligible"] != interaction["sensitive"]:
            raise ValueError("eligible flag disagrees with independent interaction audit")
        identity = row.get("identities")
        if (type(identity) is not dict or set(identity) != {"content_id", "lineage_id", "structure_signature", "model_input_sha256"} or
                not _same_typed(identity.get("content_id"), info["content_id"]) or
                not _same_typed(identity.get("lineage_id"), info["lineage_id"]) or
                not _same_typed(identity.get("structure_signature"), info["signature"]) or
                not _same_typed(identity.get("model_input_sha256"), record["model_input_sha256"]) or
                not _same_typed(row.get("split"), info["split"]) or
                not _same_typed(row.get("solution"), record["solution"]) or
                not _same_typed(row.get("witness"), record["witness"]) or
                not _same_typed(row.get("solver_stats"), record["solver_stats"])):
            raise ValueError("outcome identity, answer, or solver metadata fails independent validation")
        should_have_record = row.get("terminal_reason") == "accepted"
        if should_have_record != row["eligible"]:
            raise ValueError("accepted terminal reason disagrees with eligibility")
        key = (row["partition"], row["candidate_index"])
        actual = accepted_records.pop(key, None)
        if should_have_record:
            if actual is None:
                raise ValueError("eligible outcome has no accepted record")
            for field in ("partition", "candidate_index", "profile", "task", "content_id", "lineage_id",
                          "structure_signature", "split", "witness", "solution", "solver_stats",
                          "model_input", "model_input_sha256"):
                if not _same_typed(actual.get(field), record.get(field)):
                    raise ValueError(f"accepted record field {field} disagrees with independent reconstruction")
        elif actual is not None:
            raise ValueError("ineligible outcome unexpectedly has an accepted record")
        infos.append({**record, **info, "interaction": interaction,
                      "primary_eligible": interaction["sensitive"] if info["partition"] != "source" else False})
    if accepted_records:
        raise ValueError("accepted record has no corresponding unique eligible outcome")
    v1.validate_corpus_closure(infos)
    return infos


def _counts(rows: list[dict], guard=None) -> dict:
    result = defaultdict(lambda: {"unique_pre_eligibility": 0, "eligible": 0, "eligible_signatures": set(),
                                  "all_signatures": set(), "eligible_by_signature": Counter()})
    for row in rows:
        _check(guard)
        profile = "/".join(row["profile"])
        cell = result[(row["partition"], row["split"], profile)]
        cell["unique_pre_eligibility"] += 1
        cell["all_signatures"].add(row["signature"])
        eligible = row["interaction"]["sensitive"] if row["partition"] == "source" else row["primary_eligible"]
        if eligible:
            cell["eligible"] += 1
            cell["eligible_signatures"].add(row["signature"])
            cell["eligible_by_signature"][row["signature"]] += 1
    out = {}
    for key, value in sorted(result.items()):
        out["|".join(key)] = {"unique_pre_eligibility": value["unique_pre_eligibility"],
            "eligible": value["eligible"], "eligible_signatures": len(value["eligible_signatures"]),
            "rejected_only_signatures": len(value["all_signatures"] - value["eligible_signatures"]),
            "min_eligible_per_eligible_signature": min(value["eligible_by_signature"].values(), default=0)}
    return out


def _floor(failures: list[str], condition: bool, name: str) -> None:
    if not condition:
        failures.append(name)


def _strata(rows: list[dict]) -> dict:
    dimensions = {"n": [len(r["task"].variables) for r in rows],
                  "given_count": [len(r["task"].givens) for r in rows],
                  "constraint_count": [len(r["task"].constraints) for r in rows],
                  "model_input_utf8_bytes": [r["model_input_bytes"] for r in rows],
                  "first_solution_nodes": [r["solver_stats"]["first_solution_nodes"] for r in rows]}
    result = {}
    for name, values in dimensions.items():
        hist = Counter(values)
        result[name] = {"histogram": {str(value): hist[value] for value in sorted(hist)},
                        "min": min(values) if values else None,
                        "max": max(values) if values else None}
    return result


def _source_summary(rows: list[dict], config: dict, failures: list[str], *, feasibility: bool) -> dict:
    train = [r for r in rows if r["partition"] == "source" and r["split"] == "source_train"]
    source = [r for r in rows if r["partition"] == "source"]
    selected = train if feasibility else source
    accepted_train = [r for r in train if r["interaction"]["sensitive"]]
    accepted_selected = [r for r in selected if r["interaction"]["sensitive"]]
    gate = config["audit_gate"]
    values = Counter(value for row in accepted_train for _, value in row["solution"].assignments)
    total_values = sum(values.values())
    value_fractions = {str(value): values[value] / total_values if total_values else 0.0 for value in range(3)}
    signatures = {r["signature"] for r in accepted_train}
    sensitive = [r for r in selected if r["interaction"]["sensitive"]]
    sensitivity = len(sensitive) / len(selected) if selected else 0.0
    counts = Counter(r["signature"] for r in sensitive)
    _floor(failures, len(accepted_train) >= 100 if feasibility else len(accepted_selected) >= gate["min_accepted_groups_per_partition"], "source_train_min_groups" if feasibility else "source_min_groups")
    _floor(failures, len(signatures) >= gate["source_train_min_structure_signatures"], "source_train_signatures")
    if feasibility:
        _floor(failures, all(value_fractions[str(v)] >= gate["min_solution_value_fraction_each_partition"] for v in range(3)), "source_train_solution_value_balance")
        _floor(failures, len({len(r["task"].givens) for r in accepted_train}) >= 2, "source_train_given_count_variation")
    coverage = v1.source_train_primitive_coverage(accepted_train)
    _floor(failures, coverage["pass"], "source_train_primitive_coverage")
    composition = v1.source_train_composition_absence(accepted_train)
    _floor(failures, composition["heldout_compositions_absent"], "heldout_composition_absent_from_source_train")
    _floor(failures, sensitivity >= gate["min_source_constraint_sensitive_fraction"], "source_sensitive_fraction")
    accepted_signatures = {r["signature"] for r in accepted_selected}
    _floor(failures, all(counts[sig] >= gate["min_source_sensitive_examples_per_signature"] for sig in accepted_signatures), "source_sensitive_per_signature")
    return {"source_train_unique_groups": len(train), "source_train_groups": len(accepted_train),
            "source_groups_in_denominator": len(selected),
            "sensitive_groups": len(sensitive), "sensitivity_fraction": sensitivity,
            "source_train_signatures": len(signatures), "source_train_value_fractions": value_fractions,
            "source_train_given_counts": sorted({len(r["task"].givens) for r in accepted_train}),
            "rejected_only_source_signatures": len({r["signature"] for r in selected} - accepted_signatures),
            "source_train_primitive_coverage": coverage, "source_composition_absence": composition}


def _feasibility_support_proxy(rows: list[dict], *, eligible_targets: bool = False) -> dict:
    """Build feasibility cohorts while preserving the registered v2 default behavior."""
    if type(eligible_targets) is not bool:
        raise TypeError("eligible_targets must be a boolean")
    cohorts = {FEASIBILITY_COHORTS[0]: [r for r in rows
                if r["partition"] == "source" and r["split"] == "source_train" and
                r["interaction"]["sensitive"]]}
    for profile, name in PROFILE_COHORT.items():
        cohorts[name] = [r for r in rows if r["partition"] == "development" and
                         r["split"] == "development_validation" and tuple(r["profile"]) == profile and
                         (not eligible_targets or r["primary_eligible"])]
    return v1.common_support(cohorts, 10, 0.8, cohort_names=FEASIBILITY_COHORTS)


def _feasibility_metrics(rows: list[dict], config: dict, guard=None, *,
                         eligible_support_targets: bool = False) -> dict:
    failures: list[str] = []
    decision_rows = [r for r in rows if (r["partition"] == "source" and r["split"] == "source_train") or
                     (r["partition"] == "development" and r["split"] == "development_validation")]
    source = _source_summary(decision_rows, config, failures, feasibility=True)
    dev = [r for r in decision_rows if r["partition"] == "development"]
    per_profile = {}
    for profile, cohort_name in PROFILE_COHORT.items():
        _check(guard)
        pr = [r for r in dev if tuple(r["profile"]) == profile]
        eligible = [r for r in pr if r["primary_eligible"]]
        sigs = {r["signature"] for r in eligible}
        all_sigs = {r["signature"] for r in pr}
        per_profile["/".join(profile)] = {"unique_pre_eligibility": len(pr), "eligible": len(eligible),
            "eligible_signatures": len(sigs), "rejected_only_signatures": len(all_sigs - sigs),
            "min_eligible_per_signature": min(Counter(r["signature"] for r in eligible).values(), default=0)}
        _floor(failures, len(eligible) >= 100, f"development_validation_{'/'.join(profile)}_eligible_groups")
        _floor(failures, len(sigs) >= 12, f"development_validation_{'/'.join(profile)}_eligible_signatures")
        _floor(failures, all(Counter(r["signature"] for r in eligible)[sig] >= 1 for sig in sigs), f"development_validation_{'/'.join(profile)}_per_signature")
    total = len(dev); eligible_total = sum(r["primary_eligible"] for r in dev)
    fraction = eligible_total / total if total else 0.0
    _floor(failures, eligible_total >= config["audit_gate"]["development_target_min_groups"]["development_validation"], "development_validation_groups")
    _floor(failures, fraction >= config["audit_gate"]["min_primary_eligible_fraction_per_mixed_partition"], "development_validation_raw_eligible_fraction")
    support = _feasibility_support_proxy(decision_rows, eligible_targets=eligible_support_targets)
    _floor(failures, support["pass"], "four_cohort_common_support_proxy")
    return {"source_train": source, "development_validation": {"groups": total,
            "eligible_groups": eligible_total, "eligible_fraction": fraction, "profiles": per_profile},
            "four_cohort_common_support_proxy": support, "floor_failures": sorted(set(failures)),
            "structural_pass": not failures}


def _full_metrics(rows: list[dict], config: dict, guard=None) -> dict:
    failures: list[str] = []
    gate = config["audit_gate"]
    partitions = {}
    aggregate_source_sensitivity = None
    for part in ("source", "development", "final"):
        _check(guard)
        pr = [r for r in rows if r["partition"] == part]
        accepted_rows = [r for r in pr if r["interaction"]["sensitive"]]
        sigs = {r["signature"] for r in accepted_rows}
        profiles = {}
        for profile in v1.PROFILES[part]:
            pp = [r for r in pr if tuple(r["profile"]) == profile]
            eligible = [r for r in pp if r["interaction"]["sensitive"]] if part == "source" else [r for r in pp if r["primary_eligible"]]
            ecounts = Counter(r["signature"] for r in eligible)
            profiles["/".join(profile)] = {"unique_pre_eligibility": len(pp), "eligible": len(eligible),
                "eligible_fraction": len(eligible) / len(pp) if pp else 0.0,
                "eligible_signatures": len(ecounts),
                "rejected_only_signatures": len({r["signature"] for r in pp} - set(ecounts)),
                "min_eligible_per_signature": min(ecounts.values(), default=0)}
            if part != "source":
                _floor(failures, len(eligible) >= gate["min_primary_eligible_groups_per_mixed_profile"], f"{part}_{'/'.join(profile)}_eligible_groups")
                _floor(failures, len(ecounts) >= gate["min_primary_eligible_structure_signatures_per_mixed_profile"], f"{part}_{'/'.join(profile)}_eligible_signatures")
                _floor(failures, min(ecounts.values(), default=0) >= gate["min_primary_eligible_examples_per_heldout_signature"], f"{part}_{'/'.join(profile)}_eligible_per_signature")
        value_counts = Counter(value for r in accepted_rows for _, value in r["solution"].assignments)
        denom = sum(value_counts.values())
        value_fractions = {str(v): value_counts[v] / denom if denom else 0.0 for v in range(3)}
        eligible_count = len(accepted_rows)
        frac = eligible_count / len(pr) if pr else 0.0
        split_stats = {}
        for split in (v1.SPLITS if part != "source" else v1.SPLITS[:3]):
            split_rows = [r for r in pr if r["split"] == split]
            split_stats[split] = {"unique_oracle_groups": len(split_rows),
                "eligible_groups": sum(r["interaction"]["sensitive"] if part == "source" else r["primary_eligible"] for r in split_rows),
                "signatures": len({r["signature"] for r in split_rows}), "strata": _strata(split_rows)}
        partitions[part] = {"unique_oracle_groups": len(pr), "signatures": len(sigs),
            "profiles": profiles, "eligible_groups": eligible_count, "eligible_fraction": frac,
            "solution_value_fractions": value_fractions,
            "given_counts": sorted({len(r["task"].givens) for r in accepted_rows}),
            "splits": split_stats}
        _floor(failures, eligible_count >= gate["min_accepted_groups_per_partition"], f"{part}_min_groups")
        _floor(failures, len(sigs) >= gate["min_structure_signatures_per_partition"], f"{part}_min_signatures")
        partitions[part]["given_counts"] = sorted({len(r["task"].givens) for r in accepted_rows})
        _floor(failures, len(partitions[part]["given_counts"]) >= gate["min_distinct_given_counts_per_partition"], f"{part}_given_count_variation")
        _floor(failures, all(value_fractions[str(v)] >= gate["min_solution_value_fraction_each_partition"] for v in range(3)), f"{part}_solution_value_balance")
        if part == "source":
            source_summary = _source_summary(rows, config, failures, feasibility=False)
            aggregate_source_sensitivity = {
                "numerator": source_summary["sensitive_groups"],
                "denominator": source_summary["source_groups_in_denominator"],
                "fraction": source_summary["sensitivity_fraction"]}
        else:
            _floor(failures, frac >= gate["min_primary_eligible_fraction_per_mixed_partition"], f"{part}_eligible_fraction")
    source = [r for r in rows if r["partition"] == "source"]
    for split, minimum in gate["source_id_groups_min"].items():
        _floor(failures, sum(r["split"] == f"source_{split}" and r["interaction"]["sensitive"] for r in source) >= minimum, f"source_{split}_groups")
    dev = [r for r in rows if r["partition"] == "development"]
    for split, minimum in gate["development_target_min_groups"].items():
        _floor(failures, sum(r["split"] == split and r["primary_eligible"] for r in dev) >= minimum, f"{split}_groups")
    split_cfg = config["composition_partition"]["development_target_split"]
    for profile in v1.PROFILES["development"]:
        for split, minimum in (("development_validation", split_cfg["min_validation_groups_per_profile"]),
                               ("development_diagnostic_test", split_cfg["min_diagnostic_test_groups_per_profile"])):
            _floor(failures, sum(r["split"] == split and tuple(r["profile"]) == profile and r["primary_eligible"] for r in dev) >= minimum,
                   f"{split}_{'/'.join(profile)}_groups")
    accepted_source = [r for r in source if r["interaction"]["sensitive"]]
    coverage = v1.source_train_primitive_coverage(accepted_source)
    composition = v1.source_train_composition_absence(accepted_source)
    _floor(failures, coverage["pass"], "source_train_primitive_coverage")
    _floor(failures, composition["heldout_compositions_absent"], "heldout_composition_absent_from_source_train")
    cohorts = {name: [] for name in v1.COHORTS}
    for r in rows:
        if r["split"] == "source_test" and r["interaction"]["sensitive"]:
            cohorts[v1.COHORTS[0]].append(r)
        elif r["partition"] == "development" and r["split"] == "development_validation" and r["primary_eligible"]:
            cohorts[FULL_PROFILE_COHORT[tuple(r["profile"])]].append(r)
        elif r["partition"] == "final" and r["primary_eligible"]:
            cohorts[v1.COHORTS[4]].append(r)
    support = v1.common_support(cohorts, gate["common_support"]["min_groups_per_cell_per_cohort"],
                                gate["common_support"]["min_group_fraction_per_cohort_in_common_cells"])
    _floor(failures, support["pass"], "common_support")
    return {"partitions": partitions, "aggregate_source_sensitivity": aggregate_source_sensitivity,
            "source_train_primitive_coverage": coverage,
            "source_composition_absence": composition, "five_cohort_common_support": support,
            "floor_failures": sorted(set(failures)), "structural_pass": not failures}


def run_audit_v2(data_dir: Path, config, source: dict, mode: str, *,
                 guard: ResourceGuard | None = None) -> dict:
    """Replay a v2 run, independently validate all unique candidates, and aggregate floors."""
    data_dir = Path(data_dir).resolve()
    if (data_dir / "terminal_failure.json").exists():
        raise ValueError("generation has a terminal_failure.json sidecar; audit refused")
    manifest_path = data_dir / "manifest.json"
    generation = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    limits = registered_limits(config.resolved, mode)
    validate_manifest_v2(generation, config_sha256=config.sha256, config=config.resolved,
                         mode=mode, limits=limits, expected_producer=_producer_identity())
    generation_hash = sha256_file(manifest_path)
    if guard is not None:
        guard.wall_seconds = min(guard.wall_seconds, remaining_global_seconds(generation, limits.wall_seconds))
        _check(guard)
    replay = replay_files_v2(data_dir / "outcomes.jsonl", data_dir / "records.jsonl", config,
                             mode, generation, limits=limits, guard=guard, allow_partial=True)
    _check(guard)
    outcomes = _read_jsonl(data_dir / "outcomes.jsonl", guard)
    records = _read_jsonl(data_dir / "records.jsonl", guard)
    rows = _validate_v2_artifacts(outcomes, records, config.resolved, guard)
    complete = generation["status"] == "complete" and replay["status"] == "complete"
    metrics = _feasibility_metrics(rows, config.resolved, guard) if mode == "feasibility" else _full_metrics(rows, config.resolved, guard)
    if not complete:
        metrics["observed_prefix_floor_shortfalls"] = metrics.pop("floor_failures")
        metrics["unassessed_reason"] = "generation prefix is incomplete; unattempted targets are not floor failures"
        metrics["floor_failures"] = []
        metrics["structural_pass"] = False
    clean_match = source == generation["source"] and source.get("dirty") is False
    g2_status = "PASS_FEASIBILITY" if mode == "feasibility" and complete and metrics["structural_pass"] and clean_match else (
        "measured_pass" if mode == "full" and complete and metrics["structural_pass"] and clean_match else "NO_GO")
    if not complete:
        g2_status = "not_assessed"
    visible_rows = rows if mode == "full" else [r for r in rows if (r["partition"] == "source" and r["split"] == "source_train") or
                                                  (r["partition"] == "development" and r["split"] == "development_validation")]
    visible_keys = {(r["partition"], r["candidate_index"]) for r in visible_rows}
    visible_records = [r for r in records if (r["partition"], r["candidate_index"]) in visible_keys]
    report_replay = replay if mode == "full" else {key: replay[key] for key in ("status", "outcomes_sha256", "records_sha256")}
    per_example = [{"partition": r["partition"], "candidate_index": r["candidate_index"],
                    "split": r["split"], "profile": r["profile"], "content_id": r["content_id"],
                    "structure_signature": r["signature"], "eligible": r["interaction"]["sensitive"] if r["partition"] == "source" else r["primary_eligible"],
                    "interaction": r["interaction"]} for r in visible_rows]
    return {"generation_complete": complete, "replay": report_replay, "replay_pass": True,
            "integrity_pass": True, "generation_manifest_sha256": generation_hash,
            "generation_artifacts": generation["artifacts"], "generation_source": generation["source"],
            "audit_source": source, "config_sha256": config.sha256, "mode": mode,
            "g2_status": g2_status, "metrics": metrics,
            "denominator_profiles": _counts(visible_rows, guard),
            "per_example_outcomes": per_example,
            "independent_unique_candidate_count": len(visible_rows),
            "accepted_record_count": len(visible_records)}


def _validate_v3_artifacts(outcomes: list[dict], records: list[dict], config: dict,
                           guard: ResourceGuard | None = None) -> list[dict]:
    """Validate v3's closed identity envelope, then use the unchanged v1 projection."""
    from .audit import RECORD_KEYS
    from .stream_contract import resolve_stream
    contract = resolve_stream("dwr-generator-v3")
    outcome_keys = {"outcome_version", "generator_id", "partition", "candidate_index", "n",
                    "profile", "terminal_reason", "oracle_unique", "eligible", "private_evidence",
                    "split", "denominator_counted", "identities", "task", "witness", "solution",
                    "solver_stats"}
    for row in outcomes:
        _check(guard)
        if type(row) is not dict or set(row) != outcome_keys:
            raise ValueError("v3 outcome has unknown or missing fields")
        if row["outcome_version"] != contract.outcome_version or row["generator_id"] != contract.generator_id:
            raise ValueError("v3 outcome identity does not match selected stream")
    expected_record_keys = set(RECORD_KEYS) | {"record_version", "generator_id"}
    for record in records:
        _check(guard)
        if (type(record) is not dict or set(record) != expected_record_keys or
                record.get("record_version") != contract.accepted_record_version or
                record.get("generator_id") != contract.generator_id):
            raise ValueError("v3 accepted record has invalid version or fields")
    projected_outcomes = [{key: value for key, value in row.items()
                           if key not in {"outcome_version", "generator_id"}} for row in outcomes]
    # Preserve the legacy validator's exact schema, eligibility recomputation and first-seen
    # duplicate/denominator behavior. The only v3 record fields are removed explicitly.
    projected_records = [{key: value for key, value in row.items()
                          if key not in {"record_version", "generator_id"}} for row in records]
    return _validate_v2_artifacts(projected_outcomes, projected_records, config, guard)


def _audit_source_capacity(records: list[dict], replay_sha256: str,
                           guard: ResourceGuard | None = None) -> dict:
    """Recompute the capacity histogram independently of the generation helper."""
    import math
    import re
    if type(replay_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", replay_sha256):
        raise ValueError("source prefix replay identity is malformed")
    counts = Counter()
    total = 0
    for record in records:
        _check(guard)
        if record.get("partition") != "source" or record.get("split") != "source_train":
            continue
        task = parse_task_json(stable_json(record["task"]))
        size = len(record["model_input"].encode("utf-8"))
        prompt_bin = 0 if size <= 511 else 1 if size <= 639 else 2 if size <= 767 else 3 if size <= 1023 else 4
        nodes = record["solver_stats"]["first_solution_nodes"]
        node_bin = 0 if nodes == 1 else 1 if nodes <= 3 else 2 if nodes <= 7 else 3 if nodes <= 15 else 4 if nodes <= 31 else 5 if nodes <= 63 else 6
        cell = (len(task.variables), len(task.givens), len(task.constraints), prompt_bin, node_bin)
        counts[cell] += 1
        total += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    mass = sum(count for _, count in ordered) / total if total else 0.0
    if not math.isfinite(mass) or not 0.0 <= mass <= 1.0:
        raise ValueError("independent source capacity mass is invalid")
    return {"bound_version": "dwr-task-v3-source-capacity-bound-v1",
            "status": "NO_GO" if mass < 0.8 else "PASS_CAPACITY_BOUND",
            "reason": "source_support_capacity_bound" if mass < 0.8 else None,
            "source_prefix_replay_sha256": replay_sha256,
            "top_k_cells": 10, "minimum_source_train_mass": 0.8,
            "source_train_group_count": total,
            "top_cells": [{"cell": list(cell), "count": count} for cell, count in ordered],
            "top_k_mass": mass}


def _audit_producer_identity() -> dict:
    """Hash every independent v3 audit dependency named by the stream contract."""
    from .stream_contract import resolve_stream
    root = Path(__file__).resolve().parents[2]
    task_dir = Path(__file__).resolve().parent
    paths = {
        "audit_py_sha256": task_dir / "audit.py",
        "audit_v2_sha256": task_dir / "audit_v2.py",
        "audit_cli_v2_sha256": task_dir / "audit_cli_v2.py",
        "audit_cli_v3_sha256": task_dir / "audit_cli_v3.py",
        "generator_v2_sha256": task_dir / "generator_v2.py",
        "generator_v3_sha256": task_dir / "generator_v3.py",
        "dataset_v2_sha256": task_dir / "dataset_v2.py",
        "replay_v2_sha256": task_dir / "replay_v2.py",
        "manifest_v2_sha256": task_dir / "manifest_v2.py",
        "config_v3_sha256": root / "configs/tasks/constraint_v3.json",
        "stream_contract_sha256": task_dir / "stream_contract.py",
        "canonical_sha256": task_dir / "canonical.py",
        "oracle_sha256": task_dir / "oracle.py",
        "schema_sha256": task_dir / "schema.py",
        "solver_sha256": task_dir / "solver.py",
        "run_support_sha256": task_dir / "run_support.py",
    }
    contract = resolve_stream("dwr-generator-v3")
    result = {name: sha256_file(path) for name, path in paths.items()}
    if set(result) != set(contract.audit_producer_fields):
        raise ValueError("v3 audit producer fields differ from closed stream contract")
    return result


def _feasibility_visible_projection(replay: dict, records: list[dict], rows: list[dict], mode: str):
    """Keep feasibility report aggregates opaque outside source TRAIN and DEV VAL."""
    if mode == "full":
        visible_rows = rows
        visible_records = records
        visible_replay = replay
    else:
        visible_rows = [r for r in rows if
            (r["partition"] == "source" and r["split"] == "source_train") or
            (r["partition"] == "development" and r["split"] == "development_validation")]
        visible_keys = {(r["partition"], r["candidate_index"]) for r in visible_rows}
        visible_records = [r for r in records if (r["partition"], r["candidate_index"]) in visible_keys]
        visible_replay = {key: replay[key] for key in ("status", "outcomes_sha256", "records_sha256")}
    return visible_replay, visible_rows, visible_records


def run_audit_v3(data_dir: Path, config, source: dict, mode: str, *,
                 guard: ResourceGuard | None = None, allow_partial: bool = True) -> dict:
    """Independently replay and audit the closed v3 stream using shared v2 metrics."""
    from .dataset_v2 import registered_limits
    from .manifest_v2 import validate_manifest_v3
    from .replay_v2 import replay_files_v3, replay_source_prefix_v3
    from .stream_contract import require_matching_config, resolve_stream

    contract = resolve_stream("dwr-generator-v3")
    require_matching_config(contract, config.resolved)
    caller_supplied_guard = guard is not None
    if mode not in ("feasibility", "full"):
        raise ValueError("mode must be feasibility or full")
    data_dir = Path(data_dir).resolve()
    if (data_dir / "terminal_failure.json").exists():
        raise ValueError("generation has a terminal_failure.json sidecar; audit refused")
    manifest_path = data_dir / "manifest.json"
    limits = registered_limits(config.resolved, mode)
    if guard is None:
        guard = ResourceGuard(limits.wall_seconds, limits.memory_mib)
    _check(guard)
    generation = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    _check(guard)
    producer = _producer_identity("dwr-generator-v3")
    validate_manifest_v3(generation, config_sha256=config.sha256, config=config.resolved,
                         mode=mode, limits=limits, expected_producer=producer)
    if source != generation.get("source"):
        raise ValueError("audit source identity differs from generation source")
    # The CLI sets the inherited budget at startup. Direct callers get the same setup here;
    # prefix reconstruction and terminal replay never subtract generation time again.
    if not caller_supplied_guard:
        guard.wall_seconds = min(guard.wall_seconds,
                                 remaining_global_seconds(generation, limits.wall_seconds))
    _check(guard)
    prefix_replay = None
    capacity = {"status": "NOT_APPLICABLE", "reason": "full_mode_uses_registered_five_cohort_gate"}
    if mode == "feasibility":
        prefix_link = generation.get("source_prefix")
        if prefix_link is not None:
            context_path = data_dir / prefix_link["context_file"]
            replay_path = data_dir / prefix_link["replay_file"]
            _check(guard)
            context = strict_json_loads(context_path.read_text(encoding="utf-8"))
            _check(guard)
            context_sha = sha256_file(context_path)
            _check(guard)
            if context_sha != prefix_link["context_sha256"]:
                raise ValueError("source prefix context hash differs from generation manifest")
            prefix_replay = replay_source_prefix_v3(
                data_dir / "outcomes.jsonl", data_dir / "records.jsonl", context, context_sha,
                config, source, producer, limits=limits, guard=guard)
            _check(guard)
            stored_replay = strict_json_loads(replay_path.read_text(encoding="utf-8"))
            _check(guard)
            replay_sha = sha256_file(replay_path)
            _check(guard)
            if prefix_replay != stored_replay or replay_sha != prefix_link["replay_sha256"]:
                raise ValueError("independent source-prefix replay differs from the linked report")
            _check(guard)
        elif generation.get("source_capacity_bound", {}).get("status") not in ("NOT_ASSESSED",):
            raise ValueError("feasibility bound claims an unlinked source closure")
    replay = replay_files_v3(data_dir / "outcomes.jsonl", data_dir / "records.jsonl", config,
                             mode, generation, limits=limits, guard=guard,
                             allow_partial=allow_partial)
    _check(guard)
    outcomes = _read_jsonl(data_dir / "outcomes.jsonl", guard)
    records = _read_jsonl(data_dir / "records.jsonl", guard)
    rows = _validate_v3_artifacts(outcomes, records, config.resolved, guard)
    generation_complete = generation["status"] == "complete" and replay["status"] == "complete"
    audit_complete = True  # Reaching this point means integrity and deterministic replay completed.
    if mode == "feasibility":
        bound_file = strict_json_loads((data_dir / "source_capacity_bound.json").read_text(encoding="utf-8"))
        if prefix_replay is not None:
            replay_sha = generation["source_prefix"]["replay_sha256"]
            independent = _audit_source_capacity(records, replay_sha, guard)
            # Incomplete or early-stop datasets may only audit the closed source prefix.
            if not generation_complete:
                prefix_count = generation.get("partitions", {}).get("development", {}).get("attempts", 0)
                if generation.get("early_stop") is not None and prefix_count != 0:
                    raise ValueError("early capacity stop contains target attempts")
            if bound_file != generation.get("source_capacity_bound"):
                raise ValueError("capacity report file differs from generation manifest")
            if bound_file.get("status") in ("PASS_CAPACITY_BOUND", "NO_GO"):
                if independent != bound_file:
                    raise ValueError("independent source capacity recomputation disagrees with producer report")
                capacity = independent
            else:
                # A recorded operational failure withdraws the producer's provisional bound;
                # audit must retain that state even when it could now recompute the histogram.
                capacity = bound_file
        else:
            capacity = bound_file
            if capacity.get("status") not in ("NOT_ASSESSED",):
                raise ValueError("unclosed source prefix cannot establish capacity")
    metrics = None
    proxy = None
    if generation_complete:
        metrics = (_feasibility_metrics(rows, config.resolved, guard,
                                        eligible_support_targets=True) if mode == "feasibility"
                   else _full_metrics(rows, config.resolved, guard))
        if mode == "feasibility":
            proxy = metrics["four_cohort_common_support_proxy"]
        if mode == "feasibility" and capacity.get("status") == "NO_GO":
            g2, proposal, reason = "not_assessed", "NO_GO", "source_support_capacity_bound"
        else:
            passed = metrics["structural_pass"] and source.get("dirty") is False
            if mode == "feasibility":
                g2 = "PASS_FEASIBILITY" if passed else "NO_GO"
                proposal, reason = ("PASS_FEASIBILITY", None) if passed else ("NO_GO", "measured_feasibility_floors")
            else:
                g2 = "PASS_G2" if passed else "NO_GO"
                proposal, reason = ("PASS_G2", None) if passed else ("NO_GO", "measured_full_g2_floors")
    else:
        g2, proposal, reason = "not_assessed", "NO_GO" if generation.get("early_stop") else "NOT_ASSESSED", (
            "source_support_capacity_bound" if generation.get("early_stop") else None)
        if rows:
            partial_metrics = (_feasibility_metrics(rows, config.resolved, guard,
                                                    eligible_support_targets=True) if mode == "feasibility"
                               else _full_metrics(rows, config.resolved, guard))
            partial_metrics["observed_prefix_floor_shortfalls"] = partial_metrics.pop("floor_failures")
            partial_metrics["unassessed_reason"] = "generation prefix is incomplete; unattempted targets are not floor failures"
            partial_metrics["floor_failures"] = []
            partial_metrics["structural_pass"] = False
            metrics = partial_metrics
    visible_replay, visible, visible_records = _feasibility_visible_projection(replay, records, rows, mode)
    gate = {"gate_version": contract.audit_gate_version, "audit_version": contract.audit_version,
            "status": "complete" if audit_complete else "incomplete", "mode": mode,
            "g2_status": g2, "proposal_status": proposal, "proposal_reason": reason,
            "config_sha256": config.sha256, "source": source,
            "generation_manifest_sha256": sha256_file(manifest_path),
            "generation_artifacts": generation["artifacts"],
            "source_prefix": generation.get("source_prefix"), "capacity_bound": capacity}
    return {"audit_version": contract.audit_version, "status": "complete" if audit_complete else "incomplete",
            "generation_complete": generation_complete, "generation_status": generation["status"],
            "replay": visible_replay, "replay_pass": True, "integrity_pass": True,
            "generation_manifest_sha256": gate["generation_manifest_sha256"],
            "generation_artifacts": generation["artifacts"], "generation_source": generation["source"],
            "audit_source": source, "config_sha256": config.sha256, "mode": mode,
            "g2_status": g2, "proposal_status": proposal, "proposal_reason": reason,
            "capacity_bound": capacity, "source_prefix_replay": prefix_replay,
            "four_cohort_common_support_proxy": proxy,
            "final_assessment": "not_assessed" if mode == "feasibility" else "assessed",
            "metrics": metrics, "denominator_profiles": _counts(visible, guard),
            "per_example_outcomes": [{"partition": r["partition"], "candidate_index": r["candidate_index"],
                "split": r["split"], "profile": r["profile"], "content_id": r["content_id"],
                "structure_signature": r["signature"],
                "eligible": r["interaction"]["sensitive"] if r["partition"] == "source" else r["primary_eligible"],
                "interaction": r["interaction"]} for r in visible],
            "independent_unique_candidate_count": len(visible),
            "accepted_record_count": len(visible_records), "gate": gate,
            "unassessed": [] if generation_complete else ["unattempted generation targets", "G2"]}
