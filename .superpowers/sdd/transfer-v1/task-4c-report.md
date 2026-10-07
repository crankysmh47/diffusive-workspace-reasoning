# Task 4c independent suitability audit

Date: 2026-10-07
Branch: `research/transfer-cpu-v1`
Base: `2b7697ed1bb5efb3e79dd13faf4bc11c5ca3b338`

## Scope delivered

Added `dwr.tasks.audit` with strict generation-manifest and JSONL readers; task, answer, identity, split, model-input, oracle-solution and solver-stat reconstruction; bounded candidate-stream replay; source singleton sensitivity; mixed all-kind necessity and unassigned-variable factor-graph checks; corpus closure and source TRAIN coverage/composition checks; realized strata and five-cohort common-support accounting; prospective G2 aggregation; and the `--config/--data` CLI.

The audit inherits the generation manifest's remaining share of the global 600-second allowance, includes setup through the first terminal report write and resource check, and records the final report/manifest flush boundary. A zero remaining budget writes a `blocked` / `not_assessed` report without replay. Incomplete generation cannot reach G2 pass. Partial generation artifacts remain available for later inspection.

## Consumer API

- `read_records` and `read_outcomes` enforce closed JSONL field sets and strict JSON values.
- `validate_record`, `validate_record_set`, `validate_corpus_closure`, and `validate_complete_generation` verify record, corpus and producer-manifest invariants.
- `replay_outcomes` deterministically reconstructs candidate outcomes, quota exits and duplicate handling. `check=` allows the controller to enforce the shared budget during replay.
- `audit_interactions` returns source constraint-removal counts or mixed all-kind removal counts and factor-graph evidence.
- `source_train_primitive_coverage` and `source_train_composition_absence` keep primitive exposure separate from instantiated target compositions; universal prompt rule definitions do not count as task compositions.
- `common_support`, `compute_metrics`, and `assess_g2` provide pure cohort support, structural floor and prospective G2 summaries. `run_audit` persists the audit manifest, per-example results, report, time, RSS, hashes and terminal status.

The required `n=3`, `given(v0=0)`, three-edge equality triangle is retained as a measured source-sensitivity failure: all three edge removals leave one solution. Its signature remains subject to the per-signature floor. Mixed positive and negative witnesses also cover all-kind removal and anchor-only relation exclusion.

## Verification

Exact commands and results in this worktree:

1. `python -m unittest tests.test_task_audit -v` — 14 tests passed, including record tampering, registered mixed-profile order, lineage/signature closure, source TRAIN versus corpus coverage, incomplete and zero-budget terminal behavior, strict manifest fields, positive/negative interaction fixtures, empty and primary-only common support, and G2 pass/no-go helper fixtures.
2. `python -m dwr.tasks.audit --help` — displayed required `--config` and `--data` options (exit 0).
3. `python -m unittest discover -s tests -v` — 62 tests passed in 0.308s (exit 0).
4. `git diff --check` — exit 0 before final report creation.

No production corpus, long replay, training run, or scientific suitability claim was made. Passing synthetic aggregate fixtures only test the reporting helper; they are not corpus evidence. A future controller must run the audit on complete records from the frozen reviewed generator commit. The report/manifest accounting note excludes the ultimate bounded terminal flush tail, following Task 4b's reviewed boundary.

## Review round 1 follow-up

Date: 2026-10-08. Review base: `14991b18bbefe3485fc88c31b43df2b30e2a5554`.

### Regression evidence

The retained pre-fix behavioral probes in `task-4c-review.md` are the evidence for the original defects: partial producer rows were not validated or measured, an updated artifact hash allowed a wrong partial solution to remain unexamined, a rerun replaced prior audit metadata and rehashed stale results, the audit source was copied from the generator, and six nested-manifest mutations were accepted. The first local regression run against the pre-fix implementation failed because the new tests referenced the not-yet-existing `expected_limits` and `source_identity` hooks (TypeError/missing attribute). That run is recorded only as API/harness RED; it is not claimed as independent behavioral RED evidence.

### Fixes and consumer API

- `run_audit` now validates, measures and replays the bounded prefix of valid records from incomplete generation when budget remains. It streams completed interaction/replay results into `audit_results.jsonl`, persists assessed progress, and retains partial diagnostics through caps/errors. Generation status `COMPLETE` and completed replay remain required for G2. Zero remaining budget still emits blocked/not-assessed without replay; a finalization cap is recorded separately and does not overwrite the primary error.
- `run_audit` refuses any existing audit output or temporary artifact before creating outputs. It permits one audit per corpus and hashes only outputs created by that invocation.
- Audit metadata now records executing `source` independently from `generation_source`, `platform`, and `audit_cpu_seconds` (null when unavailable), alongside the existing wall/RSS accounting boundary.
- `_verify_manifest(data_dir, config=None, expected_limits=None)` strictly validates producer top-level and nested schemas, scalar types, source/config/seed identities, registered limits, partition/profile counters and artifact identities. `expected_limits` is an internal function injection for small producer fixtures; the CLI exposes no limit override and `load_config` remains closed. `replay_outcomes(..., on_row=None)` supports streaming completed replay rows; `compute_metrics(..., validated_infos=..., interaction_results=...)` accepts aligned prefix results so completed measurements are not repeated.

### Verification

- `python -m unittest tests.test_task_audit_review1 tests.test_task_audit -v` — 19 passed in 0.433s, including actual small `run_dataset` incomplete/cap/strict-manifest fixtures and the retained-behavior regressions.
- `python -m unittest discover -s tests -v` — 67 passed in 0.690s.
- `python -m dwr.tasks.audit --help` — exit 0; CLI remains `--config` and `--data` only.
- `git diff --check` — exit 0.

No corpus or training was run. These tests establish controller and consumer-contract behavior only; they do not establish corpus suitability. The final bounded report/manifest flush remains outside the shared wall-time accounting boundary as documented in the original Task 4b contract. No production scientific limits or generation rules changed.
