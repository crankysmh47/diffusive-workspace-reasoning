# Controller Task 3 / Implementation Plan Task 2 Report

Date: 2026-10-08
Worktree: `C:\sem5\diffusive-workspace-reasoning\.worktrees\transfer-cpu-v1`
Branch: `research/transfer-cpu-v1`
Base: `188f6efd4568be85d88ed0d1ab454e9f820ab441`

Commit: `34bcc858eba2f02d06b845efeb3ea5ad9ea8e5bd` (`Add shared v3 task stream pipeline`). Worktree is clean after commit.

## Delivered

- Added the frozen, exact two-stream `StreamContract`, with closed v2/v3 generation and audit producer field lists. Config selection checks the config version, top-level and generation seed labels, candidate RNG domain, and exact seed material.
- Refactored generation through one shared internal runner. The public v2 wrapper keeps its signature and candidate injection; v3 uses its registered candidate producer. Outcome rows and accepted records enforce the selected generator ID. V2 serialization remains unchanged; v3 rows add `generator_id`, and v3 records add `record_version` plus `generator_id`.
- Added shared closed manifest dispatch and v3 terminal validation. V3 feasibility manifests bind the source-prefix context/replay and capacity report; full-mode source capacity is `NOT_APPLICABLE`.
- Added source context capture after flushing both streams, bounded prefix hashing/replay under the same live guard, and a top-ten source-train `_cell` mass report. An incomplete source remains `NOT_ASSESSED`; an operational closure error is typed as operational; only a replay-closed, hashed bound report below 0.8 creates the typed `ProposalNoGo` and stops before target candidates.
- Added shared v3 replay wrappers. Prefix replay reads exactly the recorded complete JSONL byte boundaries, validates their hashes, reconstructs outcome and record order, and ignores later appended rows.
- Added `cli_v3.py`; the shared v2 CLI entry remains the default v2 entry point. V3 full-mode preflight consumption fails closed before creating an output directory until the independent audit/preflight work in Task 3.
- Added unit-only fixtures for stream identity, serializer cross-version rejection, duplicate/ineligible first ownership, source-prefix byte boundary/tamper checks, and top-ten mass below/equal/above 0.8. No corpus, model, training, dependency, or scientific-sampling run was performed.

## Verification

- Initial RED: `python -m unittest tests.test_task_stream_contract -v` failed because `dwr.tasks.stream_contract` did not exist. Log: `.superpowers/sdd/transfer-v3/task-3-red.txt`.
- Functional RED: `python -m unittest tests.test_task_stream_contract.StreamContractTests.test_rejects_config_with_substituted_rng_domain_or_seed_label -v` showed all four wrong-domain/seed-label substitutions were accepted. Log: `.superpowers/sdd/transfer-v3/task-3-rng-red.txt`.
- GREEN: `python -m unittest tests.test_task_stream_contract tests.test_task_dataset_v3 tests.test_task_replay_v3 -v` — **17 tests passed**. Log: `.superpowers/sdd/transfer-v3/task-3-green.txt`.
- V2 regression: `python -m unittest tests.test_task_dataset_v2 tests.test_task_replay_v2 tests.test_task_run_support tests.test_task_config_v2 tests.test_task_generator_v2 -v` — **67 tests passed**, including v2 byte/default identity and replay behavior. Log: `.superpowers/sdd/transfer-v3/task-3-v2-green.txt`.
- `python -m py_compile dwr\tasks\stream_contract.py dwr\tasks\cli_v2.py dwr\tasks\cli_v3.py dwr\tasks\dataset_v2.py dwr\tasks\manifest_v2.py dwr\tasks\replay_v2.py` — passed.
- `git diff --check` — passed.

## Limits and handoff

The registered 1,000-source-group 334/333/333 sequence was not run; completing it would require scientific sampling, which this task excludes. Round 1 uses tiny deterministic fixtures to execute the real driver, prefix capture/replay, bound calculation, and PASS/NO_GO gates. The v3 audit aggregation/CLI and preflight consumer remain for the next task; the current public v3 full path rejects before output creation.

Changed files in the original implementation: `dwr/tasks/stream_contract.py`, `dwr/tasks/cli_v3.py`, `dwr/tasks/cli_v2.py`, `dwr/tasks/dataset_v2.py`, `dwr/tasks/replay_v2.py`, `dwr/tasks/manifest_v2.py`, `tests/test_task_stream_contract.py`, `tests/test_task_dataset_v3.py`, and `tests/test_task_replay_v3.py`. No `todo.md`, README, protocol, or config changes were made.

## Controller review fix round 1

Date: 2026-10-08. Review: `.superpowers/sdd/transfer-v3/task-3-review.md` (four reproduced findings).

- Finding 1: source-only replay now permits unfinished target quotas only inside `replay_source_prefix_v3`; it still requires exact source profile closure/counts, zero target attempts in the context, exact prefix exhaustion, no unreferenced accepted records, and current producer/config identities. Real small-fixture driver closure tests cover PASS advancing to development, appended suffix reconstruction, missing/corrupt records, tampered outcome/context hashes, and post-quota rows. The reviewer’s original pre-fix `ValueError: manifest prefix ends before development quotas complete` remains in the review evidence. I made the `allow_partial=True` production change before adding the end-to-end fixture assertion, so no separate RED was captured for this fix; the GREEN exercises actual capture and deterministic replay.
- Finding 2: a replay-closed source prefix with zero source-train records yields `NO_GO`, mass `0.0`, count `0`, and a nonnull replay hash; it stops before development. A separate 15-record synthetic fixture builds 15 unique source cells using real task validation, oracle, solver, canonical serialization, deterministic prefix replay, and the bound helper; it demonstrates a top-ten mass below 0.8 without scientific sampling. Attempt-cap `NOT_ASSESSED/source_partition_incomplete` remains distinct.
- Finding 3: `source_prefix_closure` is valid in the typed v3 manifest and maps to the shared v2 `generation` phase only in the private v2-validator projection. V2's public accepted phase list is unchanged. An injected operational closure failure is persisted and accepted by partial v3 replay.
- Finding 4: operational report/hash/finalization faults clear `early_stop`, replace assessed capacity with `NOT_ASSESSED/operational_error`, reconcile the authoritative bound file and manifest artifact hash, and preserve the prior report/error in `source_capacity_bound_provisional.json`. Injected bound write/hash and terminal-manifest replacement failures are covered, including v3 validation of the resulting operational terminal state.
- The public capture API still hard-requires 334/333/333. A private `ContextVar` fixture seam supplies small nonnegative profile quotas only to deterministic unit fixtures. No config, quota, threshold, generator selection, corpus, model, training, or dependency was changed.

RED evidence: `.superpowers/sdd/transfer-v3/task-3-fix1-red.txt` preserves the actual empty-bound and v3 closure-phase failures. The earlier module-availability and RNG-domain RED logs remain unchanged at `task-3-red.txt` and `task-3-rng-red.txt`.

GREEN evidence and exact commands:

- `python -m unittest tests.test_task_stream_contract tests.test_task_dataset_v3 tests.test_task_replay_v3 -v` — **30 passed**. Raw output: `.superpowers/sdd/transfer-v3/task-3-fix1-green.txt`.
- `python -m unittest tests.test_task_dataset_v2 tests.test_task_replay_v2 tests.test_task_run_support tests.test_task_config_v2 tests.test_task_generator_v2 -q` — **67 passed**. Output appended to the same log.
- `python -m py_compile dwr\tasks\dataset_v2.py dwr\tasks\manifest_v2.py dwr\tasks\replay_v2.py tests\test_task_dataset_v3.py tests\test_task_replay_v3.py` — passed; output appended to the same log.
- `git diff --check` — passed; output appended to the same log.

Self-review: public v2 wrappers and v2 validation rules are unchanged; the v3 validator is strict about operational phases withdrawing the bound; `source_capacity_bound_provisional.json` is allowed only in the closed v3 feasibility artifact set; full-mode behavior and the unimplemented public v3 preflight path remain fail-closed. Production 334/333/333 source completion is unverified by design. No actual corpus or scientific result is claimed.
