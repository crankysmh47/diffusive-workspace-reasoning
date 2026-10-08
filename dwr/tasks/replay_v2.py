"""Strict streaming replay of versioned v2 orchestration artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .dataset_v2 import (DatasetLimits, _full_profiles, _quota_counted, _quota_tables,
                         _split_and_ids, accepted_record, outcome_row, registered_limits)
from .generator_v2 import PARTITIONS, PROFILES, generate_candidate
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
                  seen: set[str], candidate_fn: Callable) -> tuple[dict, dict | None]:
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
                      denominator_counted=counted)
    record = accepted_record(outcome, split, identities) if reason == "accepted" else None
    return row, record


def replay_outcomes_v2(rows: Iterable[dict], config: LoadedConfig, mode: str = "feasibility", *,
                       check: Callable[[dict], None] | None = None,
                       limits: DatasetLimits | None = None,
                       expected_attempts: dict[str, int] | None = None,
                       allow_partial: bool = False,
                       guard: ResourceGuard | None = None,
                       candidate_fn: Callable = generate_candidate) -> list[dict]:
    """Replay an iterable exactly; callers can pass check to avoid retaining outputs."""
    if mode not in ("feasibility", "full"):
        raise ValueError("mode must be feasibility or full")
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
                                             mode, counts, limits, seen, candidate_fn)
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
        if expected_attempts is None and not _complete(partition, mode, counts, limits):
            raise ValueError(f"outcome stream ended before {partition} quotas completed")
    try: extra = next(iterator)
    except StopIteration: extra = None
    if extra is not None:
        raise ValueError("outcomes contain rows after all mode quotas were complete")
    return accepted


def replay_files_v2(outcomes_path: Path, records_path: Path, config: LoadedConfig,
                    mode: str, generation_manifest: dict, *, limits: DatasetLimits | None = None,
                    guard: ResourceGuard | None = None, allow_partial: bool = False,
                    candidate_fn: Callable = generate_candidate) -> dict:
    """Compare outcome and accepted-record streams without retaining corpus rows."""
    if type(generation_manifest) is not dict:
        raise ValueError("generation_manifest must be an object")
    if (generation_manifest.get("manifest_version") != "dwr-task-run-v2" or
        generation_manifest.get("mode") != mode or
        generation_manifest.get("config_sha256") != config.sha256):
        raise ValueError("generation manifest version, mode, or config identity mismatch")
    limits = limits or registered_limits(config.resolved, mode)
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
    if type(artifacts) is not dict or set(artifacts) != expected_artifact_names:
        raise ValueError("generation manifest artifact set is invalid")
    for artifact_path in (Path(outcomes_path), Path(records_path)):
        identity = artifacts[artifact_path.name]
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
    rows = _jsonl(outcomes_path, guard)
    # Exhaust and compare incrementally through a generator so the production path is bounded.
    replay_outcomes_v2(rows, config, mode, check=compare, limits=limits,
                       expected_attempts=attempts,
                       allow_partial=allow_partial and status == "incomplete",
                       guard=guard, candidate_fn=candidate_fn)
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
