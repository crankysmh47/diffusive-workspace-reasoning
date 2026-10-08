"""Strict streaming replay of versioned v2 orchestration artifacts."""
from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .dataset_v2 import (DatasetLimits, _full_profiles, _quota_counted, _quota_tables,
                         _split_and_ids, accepted_record, outcome_row, registered_limits)
from .generator_v2 import PARTITIONS, PROFILES, generate_candidate
from .manifest_v2 import validate_manifest_v2
from .run_support import (LoadedConfig, ResourceGuard, remaining_global_seconds, sha256_file,
                          strict_json_loads, stable_json)


def _jsonl(path: Path, guard: ResourceGuard | None = None) -> Iterator[dict]:
    if guard is not None: guard.check()
    with Path(path).open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if guard is not None: guard.check()
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"malformed JSONL line {number}")
            row = strict_json_loads(line)
            if type(row) is not dict:
                raise ValueError(f"JSONL line {number} must be an object")
            if guard is not None: guard.check()
            yield row


def _quota_target(mode: str, partition: str, limits: DatasetLimits) -> int:
    if partition == "source": return sum(limits.source_by_profile.values())
    if partition == "development": return sum(limits.development_by_profile.values())
    return sum(limits.final_by_profile.values()) if mode == "full" else 0


def _complete(partition: str, mode: str, counts: tuple[dict, dict, dict], limits: DatasetLimits) -> bool:
    if partition == "source": return all(counts[0][(p,)] >= n for p, n in limits.source_by_profile.items())
    if partition == "development": return all(counts[1][p] >= n for p, n in limits.development_by_profile.items())
    if mode == "feasibility": return True
    return all(counts[2][p] >= n for p, n in limits.final_by_profile.items())


def _expected_row(seed: int, partition: str, index: int, mode: str,
                  counts: tuple[dict, dict, dict], limits: DatasetLimits,
                  seen: set[str], candidate_fn: Callable,
                  expected_generator_id: str = "dwr-generator-v2") -> tuple[dict, dict | None]:
    full = _full_profiles(partition, mode, counts, limits)
    outcome = candidate_fn(seed, partition, index, full_profiles=full)
    if outcome.partition != partition or outcome.candidate_index != index:
        raise ValueError("candidate generator returned an invalid index or partition")
    split = identities = None
    counted = False
    reason = outcome.rejection_reason or outcome.eligibility_reason
    if outcome.oracle_unique:
        split, identities = _split_and_ids(outcome)
        cid = identities["content_id"]
        if cid in seen:
            reason = "duplicate_content"
        else:
            seen.add(cid)
            counted = True
            if reason is None:
                reason = "accepted"
                if _quota_counted(mode, partition, split, True):
                    if partition == "source": counts[0][outcome.profile] += 1
                    elif partition == "development": counts[1][outcome.profile] += 1
                    else: counts[2][outcome.profile] += 1
    row = outcome_row(outcome, reason, split=split, identities=identities,
                      denominator_counted=counted, expected_generator_id=expected_generator_id)
    record = (accepted_record(outcome, split, identities,
                              expected_generator_id=expected_generator_id)
              if reason == "accepted" else None)
    return row, record


def replay_outcomes_v2(rows: Iterable[dict], config: LoadedConfig, mode: str = "feasibility", *,
                       check: Callable[[dict], None] | None = None,
                       partition_check: Callable[[str, bool, int], None] | None = None,
                       limits: DatasetLimits | None = None,
                       expected_attempts: dict[str, int] | None = None,
                       allow_partial: bool = False,
                       guard: ResourceGuard | None = None,
                       candidate_fn: Callable = generate_candidate,
                       expected_generator_id: str = "dwr-generator-v2") -> list[dict]:
    """Replay an iterable exactly; callers can pass check to avoid retaining outputs."""
    if mode not in ("feasibility", "full"):
        raise ValueError("mode must be feasibility or full")
    from .stream_contract import require_matching_config, resolve_stream
    contract = resolve_stream(expected_generator_id)
    require_matching_config(contract, config.resolved)
    if candidate_fn is not contract.candidate_fn and expected_generator_id != "dwr-generator-v2":
        raise ValueError("v3 replay does not permit public candidate-function injection")
    limits = limits or registered_limits(config.resolved, mode)
    counts = _quota_tables(limits); seen: set[str] = set(); accepted: list[dict] = []
    iterator = iter(rows)
    if expected_attempts is not None:
        if type(expected_attempts) is not dict or set(expected_attempts) != set(PARTITIONS):
            raise ValueError("expected_attempts must bind every registered partition")
        for part, attempts in expected_attempts.items():
            if type(attempts) is not int or not 0 <= attempts <= limits.attempts_per_partition:
                raise ValueError("manifest attempt counts must be bounded integers")
            if part == "final" and mode == "feasibility" and attempts != 0:
                raise ValueError("feasibility replay cannot contain final candidates")
    for partition in PARTITIONS:
        if partition == "final" and mode == "feasibility":
            if partition_check is not None: partition_check(partition, True, 0)
            continue
        index = 0
        count = expected_attempts[partition] if expected_attempts is not None else limits.attempts_per_partition
        while index < count and (expected_attempts is not None or not _complete(partition, mode, counts, limits)):
            if index >= limits.attempts_per_partition:
                raise ValueError(f"replay attempt cap exceeded in {partition}")
            if _complete(partition, mode, counts, limits):
                raise ValueError(f"outcomes continue after {partition} quota completion")
            if guard is not None: guard.check()
            try: observed = next(iterator)
            except StopIteration as exc: raise ValueError(f"outcomes ended before manifest prefix in {partition}") from exc
            if type(observed) is not dict:
                raise ValueError("outcome rows must be objects")
            if guard is not None: guard.check()
            expected, record = _expected_row(config.resolved["seed"], partition, index,
                                             mode, counts, limits, seen, candidate_fn,
                                             expected_generator_id)
            if guard is not None: guard.check()
            if set(observed) != set(expected) or stable_json(observed) != stable_json(expected):
                raise ValueError(f"outcome differs from deterministic replay at {partition}:{index}")
            if record is not None:
                if check is None: accepted.append(record)
                else: check(record)
            if guard is not None: guard.check()
            index += 1
        if expected_attempts is not None and not _complete(partition, mode, counts, limits):
            if not allow_partial:
                raise ValueError(f"manifest prefix ends before {partition} quotas complete")
            if any(expected_attempts[later] > 0 for later in PARTITIONS[PARTITIONS.index(partition) + 1:]):
                raise ValueError("partial outcomes are not one global partition prefix")
        if expected_attempts is None and not _complete(partition, mode, counts, limits):
            raise ValueError(f"outcome stream ended before {partition} quotas completed")
        if partition_check is not None:
            partition_check(partition, _complete(partition, mode, counts, limits), index)
    try: extra = next(iterator)
    except StopIteration: extra = None
    if extra is not None:
        raise ValueError("outcomes contain rows after all mode quotas were complete")
    return accepted


def replay_outcomes_v3(rows: Iterable[dict], config: LoadedConfig, mode: str = "feasibility", *,
                       check: Callable[[dict], None] | None = None,
                       partition_check: Callable[[str, bool, int], None] | None = None,
                       limits: DatasetLimits | None = None,
                       expected_attempts: dict[str, int] | None = None,
                       allow_partial: bool = False,
                       guard: ResourceGuard | None = None) -> list[dict]:
    """Replay v3 rows with the closed v3 producer selected from the config."""
    from .stream_contract import resolve_stream
    candidate = resolve_stream("dwr-generator-v3").candidate_fn
    return replay_outcomes_v2(rows, config, mode, check=check, partition_check=partition_check,
                              limits=limits, expected_attempts=expected_attempts,
                              allow_partial=allow_partial, guard=guard,
                              candidate_fn=candidate,
                              expected_generator_id="dwr-generator-v3")


def _prefix_rows(path: Path, count: int, expected_sha256: str,
                 guard: ResourceGuard) -> Iterator[dict]:
    if type(count) is not int or count < 0:
        raise ValueError("source prefix byte count must be a nonnegative integer")
    digest = hashlib.sha256()
    consumed = 0
    with Path(path).open("rb") as stream:
        remaining = count
        number = 0
        while remaining:
            guard.check()
            raw = stream.readline(remaining + 1)
            if not raw or len(raw) > remaining:
                raise ValueError("source prefix is longer than its artifact")
            if not raw.endswith(b"\n"):
                raise ValueError("source prefix must end at a complete JSONL row")
            digest.update(raw)
            consumed += len(raw)
            remaining -= len(raw)
            number += 1
            try:
                line = raw[:-1].decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise ValueError(f"source JSONL row {number} is not UTF-8") from exc
            if not line.strip():
                raise ValueError(f"source JSONL row {number} is empty")
            row = strict_json_loads(line)
            if type(row) is not dict:
                raise ValueError(f"source JSONL row {number} must be an object")
            yield row
    if consumed != count or digest.hexdigest() != expected_sha256:
        raise ValueError("source prefix byte count or hash mismatch")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def replay_source_prefix_v3(outcomes_path: Path, records_path: Path, context: dict,
                            context_sha256: str, config: LoadedConfig, source: dict,
                            producer: dict, *, limits: DatasetLimits,
                            guard: ResourceGuard) -> dict:
    """Reconstruct and replay the exact closed source prefix, ignoring later suffixes."""
    from .stream_contract import resolve_stream, require_matching_config
    contract = resolve_stream("dwr-generator-v3")
    require_matching_config(contract, config.resolved)
    if not re.fullmatch(r"[0-9a-f]{64}", context_sha256):
        raise ValueError("source context SHA-256 is malformed")
    if _sha256_bytes((stable_json(context) + "\n").encode("utf-8")) != context_sha256:
        raise ValueError("source context content does not match its declared SHA-256")
    required = {"context_version", "generator_id", "config_version", "config_sha256", "source", "producer",
                "mode", "source_attempts", "source_outcome_prefix_bytes", "source_outcome_prefix_sha256",
                "source_accepted_record_count", "source_record_prefix_bytes", "source_record_prefix_sha256",
                "source_quota_targets", "source_quota_counts", "source_quota_closed",
                "development_attempts", "final_attempts"}
    if type(context) is not dict or set(context) != required:
        raise ValueError("source prefix context has unknown or missing fields")
    if (context["context_version"] != "dwr-task-v3-source-prefix-context-v1" or
        context["generator_id"] != contract.generator_id or
        context["config_version"] != contract.config_version or
        context["config_sha256"] != config.sha256 or context["source"] != source or
        context["producer"] != producer or context["mode"] != "feasibility" or
        context["development_attempts"] != 0 or context["final_attempts"] != 0 or
        context["source_quota_closed"] is not True):
        raise ValueError("source prefix context identity or phase mismatch")
    if (type(source) is not dict or set(source) != {"commit", "dirty", "dirty_diff_sha256", "checkout_root"} or
        type(source["commit"]) is not str or type(source["dirty"]) is not bool or
        type(source["dirty_diff_sha256"]) is not str or
        not re.fullmatch(r"[0-9a-f]{64}", source["dirty_diff_sha256"]) or
        type(source["checkout_root"]) is not str or
        type(context["source_accepted_record_count"]) is not int or
        type(context["development_attempts"]) is not int or context["development_attempts"] != 0 or
        type(context["final_attempts"]) is not int or context["final_attempts"] != 0):
        raise ValueError("source prefix context provenance or counts are malformed")
    if (type(context["source_quota_closed"]) is not bool or
        type(context["source_outcome_prefix_bytes"]) is not int or context["source_outcome_prefix_bytes"] < 0 or
        type(context["source_record_prefix_bytes"]) is not int or context["source_record_prefix_bytes"] < 0 or
        type(context["source_accepted_record_count"]) is not int or context["source_accepted_record_count"] < 0 or
        any(type(context[name]) is not str or not re.fullmatch(r"[0-9a-f]{64}", context[name])
            for name in ("source_outcome_prefix_sha256", "source_record_prefix_sha256")) or
        type(producer) is not dict or set(producer) != set(contract.generation_producer_fields) or
        any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in producer.values())):
        raise ValueError("source prefix context counts, hashes, or producer are malformed")
    from .dataset_v2 import _producer_identity
    guard.check()
    if producer != _producer_identity("dwr-generator-v3"):
        raise ValueError("source prefix producer differs from current generation code")
    guard.check()
    targets = {"eq": limits.source_by_profile["eq"], "neq": limits.source_by_profile["neq"],
               "add_mod": limits.source_by_profile["add_mod"]}
    for field in ("source_quota_targets", "source_quota_counts"):
        value = context[field]
        if (type(value) is not dict or set(value) != set(targets) or
            any(type(value[name]) is not int for name in targets)):
            raise ValueError("source prefix quota counts are malformed")
    if context["source_quota_targets"] != targets or context["source_quota_counts"] != targets:
        raise ValueError("source prefix quota closure does not match the registered source quotas")
    attempts = context["source_attempts"]
    if type(attempts) is not int or not 1 <= attempts <= limits.attempts_per_partition:
        raise ValueError("source prefix attempt count is invalid")
    rows = _prefix_rows(Path(outcomes_path), context["source_outcome_prefix_bytes"],
                        context["source_outcome_prefix_sha256"], guard)
    actual_record_iter = _prefix_rows(Path(records_path), context["source_record_prefix_bytes"],
                                      context["source_record_prefix_sha256"], guard)
    replayed_count = 0
    profile_counts = {name: 0 for name in targets}
    def compare(expected: dict) -> None:
        nonlocal replayed_count
        try:
            actual = next(actual_record_iter)
        except StopIteration as exc:
            raise ValueError("source accepted-record prefix ended early") from exc
        if set(actual) != set(expected) or stable_json(actual) != stable_json(expected):
            raise ValueError("source accepted record differs from deterministic replay")
        if expected["partition"] != "source":
            raise ValueError("source prefix replay produced a non-source accepted record")
        profile_counts[expected["profile"][0]] += 1
        replayed_count += 1
    replay_outcomes_v3(rows, config, "feasibility", check=compare, limits=limits,
                       expected_attempts={"source": attempts, "development": 0, "final": 0},
                       allow_partial=False, guard=guard)
    try:
        next(actual_record_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("source accepted-record prefix has unreferenced rows")
    if replayed_count != context["source_accepted_record_count"] or profile_counts != targets:
        raise ValueError("source accepted-record count or profile quota closure mismatch")
    report = {"replay_version": "dwr-task-v3-source-prefix-replay-v1", "status": "PASS_SOURCE_CLOSURE",
              "context_sha256": context_sha256, "generator_id": contract.generator_id,
              "config_sha256": config.sha256, "source": source, "producer": producer,
              "source_attempts": attempts, "source_accepted_record_count": replayed_count,
              "source_quota_counts": profile_counts, "source_quota_closed": True,
              "development_attempts": 0, "final_attempts": 0,
              "source_outcome_prefix_bytes": context["source_outcome_prefix_bytes"],
              "source_outcome_prefix_sha256": context["source_outcome_prefix_sha256"],
              "source_record_prefix_bytes": context["source_record_prefix_bytes"],
              "source_record_prefix_sha256": context["source_record_prefix_sha256"]}
    guard.check()
    return report


def source_support_capacity_bound(records: Iterable[dict], source_prefix_replay_sha256: str,
                                  *, top_k: int = 10, min_fraction: float = 0.8) -> dict:
    """Calculate the registered source-train cell-mass necessary capacity bound."""
    import math
    import re
    from .audit import _cell
    if type(source_prefix_replay_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", source_prefix_replay_sha256):
        raise ValueError("source prefix replay SHA-256 is required")
    if type(top_k) is not int or top_k != 10 or type(min_fraction) not in (int, float) or isinstance(min_fraction, bool) or float(min_fraction) != 0.8:
        raise ValueError("source capacity bound parameters are closed to the registered constants")
    counts = Counter()
    total = 0
    for record in records:
        if type(record) is not dict or record.get("partition") != "source" or record.get("split") != "source_train":
            raise ValueError("capacity bound accepts source_train records only")
        normalized = dict(record)
        if "model_input_bytes" not in normalized:
            normalized["model_input_bytes"] = len(record["model_input"].encode("utf-8"))
        counts[tuple(_cell(normalized))] += 1
        total += 1
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    mass = None if total == 0 else sum(count for _, count in top) / total
    return {"bound_version": "dwr-task-v3-source-capacity-bound-v1",
            "status": "NOT_ASSESSED" if total == 0 else ("PASS_CAPACITY_BOUND" if mass >= min_fraction else "NO_GO"),
            "reason": "source_partition_incomplete" if total == 0 else
                      (None if mass >= min_fraction else "source_support_capacity_bound"),
            "source_prefix_replay_sha256": source_prefix_replay_sha256,
            "top_k_cells": top_k, "minimum_source_train_mass": float(min_fraction),
            "source_train_group_count": total,
            "top_cells": [{"cell": list(cell), "count": count} for cell, count in top],
            "top_k_mass": mass}


def replay_files_v2(outcomes_path: Path, records_path: Path, config: LoadedConfig,
                    mode: str, generation_manifest: dict, *, limits: DatasetLimits | None = None,
                    guard: ResourceGuard | None = None, allow_partial: bool = False,
                    candidate_fn: Callable = generate_candidate,
                    expected_generator_id: str = "dwr-generator-v2") -> dict:
    """Compare outcome and accepted-record streams without retaining corpus rows."""
    if type(generation_manifest) is not dict:
        raise ValueError("generation_manifest must be an object")
    if (Path(outcomes_path).parent / "terminal_failure.json").exists():
        raise ValueError("generation has a terminal failure sidecar")
    from .stream_contract import resolve_stream, require_matching_config
    contract = resolve_stream(expected_generator_id)
    require_matching_config(contract, config.resolved)
    if (generation_manifest.get("manifest_version") != contract.manifest_version or
        generation_manifest.get("mode") != mode or
        generation_manifest.get("config_sha256") != config.sha256):
        raise ValueError("generation manifest version, mode, or config identity mismatch")
    limits = limits or registered_limits(config.resolved, mode)
    from .dataset_v2 import _producer_identity
    if expected_generator_id == "dwr-generator-v2":
        validate_manifest_v2(generation_manifest, config_sha256=config.sha256,
                             config=config.resolved, mode=mode, limits=limits,
                             expected_producer=_producer_identity())
    else:
        from .manifest_v2 import validate_manifest_v3
        validate_manifest_v3(generation_manifest, config_sha256=config.sha256,
                             config=config.resolved, mode=mode, limits=limits,
                             expected_producer=_producer_identity(expected_generator_id))
    remaining = remaining_global_seconds(generation_manifest, limits.wall_seconds)
    if guard is None:
        guard = ResourceGuard(remaining, limits.memory_mib)
    else:
        guard.wall_seconds = min(guard.wall_seconds, remaining)
        guard.memory_bytes = min(guard.memory_bytes, limits.memory_mib * 1024 * 1024)
        # A caller-supplied guard may already have charged preceding audit work.
    guard.check()
    artifacts = generation_manifest.get("artifacts")
    expected_artifact_names = {Path(outcomes_path).name, Path(records_path).name}
    if expected_generator_id == "dwr-generator-v3" and mode == "feasibility":
        expected_artifact_names.add("source_capacity_bound.json")
        allowed_artifact_names = expected_artifact_names | {"source_prefix_context.json", "source_prefix_replay.json"}
        if generation_manifest.get("source_prefix") is not None:
            expected_artifact_names |= {"source_prefix_context.json", "source_prefix_replay.json"}
    else:
        allowed_artifact_names = expected_artifact_names
    if (type(artifacts) is not dict or not expected_artifact_names <= set(artifacts) or
        not set(artifacts) <= allowed_artifact_names):
        raise ValueError("generation manifest artifact set is invalid")
    for artifact_name, identity in artifacts.items():
        artifact_path = Path(outcomes_path).parent / artifact_name
        if (type(identity) is not dict or set(identity) != {"sha256", "bytes"} or
            type(identity["bytes"]) is not int or identity["bytes"] < 0 or
            type(identity["sha256"]) is not str or len(identity["sha256"]) != 64):
            raise ValueError("generation manifest artifact identity is malformed")
        if artifact_path.stat().st_size != identity["bytes"] or sha256_file(artifact_path) != identity["sha256"]:
            raise ValueError("generation artifact differs from manifest identity")
        guard.check()
    states = generation_manifest.get("partitions")
    if type(states) is not dict or set(states) != set(PARTITIONS):
        raise ValueError("generation manifest partition accounting is invalid")
    attempts = {}
    for part in PARTITIONS:
        state = states[part]
        if type(state) is not dict or set(state) - {"status", "attempts", "accepted", "not_attempted"}:
            raise ValueError("generation partition state is malformed")
        n = state.get("attempts")
        if type(n) is not int or not 0 <= n <= limits.attempts_per_partition:
            raise ValueError("generation manifest attempt count is invalid")
        if type(state.get("accepted")) is not int or state["accepted"] < 0:
            raise ValueError("generation manifest accepted count is invalid")
        if type(state.get("not_attempted")) is not bool or state["not_attempted"] != (n == 0 and state.get("status") == "not_attempted"):
            raise ValueError("generation not_attempted flag disagrees with its prefix")
        attempts[part] = n
    status = generation_manifest.get("status")
    if status not in ("complete", "incomplete"):
        raise ValueError("generation manifest has a nonterminal status")
    if status == "complete":
        for part in PARTITIONS:
            expected_status = "not_attempted" if part == "final" and mode == "feasibility" else "complete"
            if states[part].get("status") != expected_status:
                raise ValueError("complete generation status disagrees with partition status")
    if mode == "feasibility" and (attempts["final"] != 0 or states["final"].get("status") != "not_attempted"):
        raise ValueError("feasibility manifest contains a final partition attempt")
    record_iter = iter(_jsonl(records_path, guard)); accepted_count = 0
    visible_accepted = {part: 0 for part in PARTITIONS}
    def compare(expected: dict) -> None:
        nonlocal accepted_count
        guard.check()
        try: actual = next(record_iter)
        except StopIteration as exc: raise ValueError("accepted record stream ended early") from exc
        guard.check()
        if set(actual) != set(expected) or stable_json(actual) != stable_json(expected):
            raise ValueError("accepted record differs from replay")
        accepted_count += 1
        part = expected["partition"]
        split = expected["split"]
        if (mode == "full" or (part == "source" and split == "source_train") or
            (part == "development" and split == "development_validation")):
            visible_accepted[part] += 1
    def partition_check(part: str, quota_complete: bool, attempts_count: int) -> None:
        state = states[part]
        if state["status"] == "complete" and not quota_complete:
            raise ValueError(f"manifest marks {part} complete before its registered quotas")
        if state["status"] == "not_attempted" and attempts_count != 0:
            raise ValueError(f"manifest marks {part} not attempted after replayed candidates")
    rows = _jsonl(outcomes_path, guard)
    # Exhaust and compare incrementally through a generator so the production path is bounded.
    if expected_generator_id == "dwr-generator-v2":
        outcome_replay = replay_outcomes_v2
        replay_kw = {"candidate_fn": candidate_fn}
    else:
        outcome_replay = replay_outcomes_v3
        replay_kw = {}
    outcome_replay(rows, config, mode, check=compare, limits=limits,
                   expected_attempts=attempts, partition_check=partition_check,
                   allow_partial=allow_partial and status == "incomplete",
                   guard=guard, **replay_kw)
    try: next(record_iter)
    except StopIteration: pass
    else: raise ValueError("accepted record stream contains extra rows")
    if any(visible_accepted[part] != states[part]["accepted"] for part in PARTITIONS):
        raise ValueError("generation manifest accepted counts disagree with replayed records")
    guard.check()
    outcomes_hash = sha256_file(Path(outcomes_path))
    records_hash = sha256_file(Path(records_path))
    guard.check()
    complete = status == "complete"
    return {"status": "complete" if complete else "partial", "accepted_records": accepted_count,
            "outcomes_sha256": outcomes_hash, "records_sha256": records_hash}


def replay_files_v3(outcomes_path: Path, records_path: Path, config: LoadedConfig,
                    mode: str, generation_manifest: dict, *, limits: DatasetLimits | None = None,
                    guard: ResourceGuard | None = None, allow_partial: bool = False) -> dict:
    """Replay the closed v3 stream using the shared bounded file runner."""
    return replay_files_v2(outcomes_path, records_path, config, mode, generation_manifest,
                           limits=limits, guard=guard, allow_partial=allow_partial,
                           expected_generator_id="dwr-generator-v3")
