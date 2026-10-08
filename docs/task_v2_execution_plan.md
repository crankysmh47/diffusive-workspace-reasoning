# Task design v2 implementation and feasibility plan

> **For agentic workers:** implement each task in order after this plan and [task_spec_v2.md](task_spec_v2.md) are reviewed. Tasks 1–5 are CPU data-gate work and contain no model code. A complete G2 pass clears the data gate for a separate fixed-N EX/IDX implementation task; training remains separately gated.

**Goal:** Implement and evaluate one versioned CPU-only task-generation proposal that preserves the v1 grammar and numeric G2 floors while testing whether sensitivity and mixed-task eligibility can support fixed-N composition transfer.

**Architecture:** Keep v1 generator/config/audit replayable. First fix bounded Windows manifest/resource monitoring. Add a pinned v2 loader and a pure candidate generator that owns private proposal predicates but no corpus state. Add orchestration with run-wide deduplication, strict replay, feasibility and conditionally gated full modes; independently audit private predicate claims through the existing `dwr.tasks.audit.audit_interactions` function. Implement and test both modes before the single feasibility run.

**Tech Stack:** Python 3.11/3.12 standard library, `pathlib`, `hashlib`, `random.Random`, JSONL, SHA-256, and `unittest`; no ML dependency or GPU.

**Spec:** [docs/task_spec_v2.md](task_spec_v2.md), under [docs/research_protocol.md](research_protocol.md) and the fixed-N contract in [docs/transfer_research_direction.md](transfer_research_direction.md).

## Global Constraints

- Preserve the v1 spec/config, interrupted run, partial metrics, and every numeric G2 floor.
- Preserve v1 grammar, parser/verifier, exhaustive oracle, solver, IDs, split hashes, prompt projection, and model-input leakage boundary.
- Keep candidate generation pure and deterministic. The generator owns private source/mixed eligibility predicates; orchestration owns the run-wide seen-content set, duplicate precedence, quotas, and terminal outcomes.
- Use only source-train and development-validation aggregates for feasibility choices. Deterministic acceptance/replay may process held-out rows, which remain opaque and excluded from decision statistics.
- Implement and test feasibility and full modes before starting feasibility. Full mode requires a matching `PASS_FEASIBILITY` report and frozen clean source/config identity.
- One worker, at most 30,000 candidates per partition, at most 600 active seconds for each generation-plus-audit invocation, and 1 GiB resident memory. Preserve evidence at any cap; never resume under changed rules.
- Write resolved manifests before work, use unique artifact directories, stable UTF-8 serialization, all candidate outcomes and SHA-256 identities. Never overwrite v1 artifacts.
- No tuning, padding, weakened floors, model implementation, or training. A feasibility failure is a documented no-go.

## Review Focus

- A witness-first draw with no feasible tuple/count vector rejects deterministically without inventing a task; test exact draw order and replay in Task 2.
- A source structure whose every constraint deletion stays unique is rejected; use the equality triangle fixture in Task 2.
- A mixed task missing kind necessity or unknown-variable connectivity is rejected; test each possible independent failure and the logically coupled condition in Task 2.
- Repeated content must enter the seen set on its first oracle-unique occurrence even if eligibility rejects it; test rejected-first and accepted-first duplicate streams in Task 3.
- Incorrect private eligibility annotations must be caught by independent `audit_interactions`; test a deliberately corrupted annotation in Task 4.
- Source validation/test and development diagnostic rows may be privately processed for acceptance/replay but cannot affect feasibility aggregates or choices; prove this with opaque-row mutation tests in Task 4.

---

### Task 1: Bounded Windows replacement and cached resource monitoring

**Files:**
- Modify: `dwr/tasks/run_support.py`
- Modify: `tests/test_task_run_support.py`
- Test: `tests/test_task_audit_review1.py`

**Interfaces:**
- Consumes: existing `atomic_json(path, value)` and `resident_bytes_windows()` interfaces.
- Produces: at most four `Path.replace` attempts, with delays 0.01/0.02/0.04 seconds for Windows replace error codes 5/32/33; cached/module-level ctypes signatures and `PROCESS_MEMORY_COUNTERS` definition. Terminal errors retain their original type/context.

- [ ] Write RED tests with injected replacement: three retryable errors then success; four errors then propagation; other Windows error code immediate; non-Windows error immediate; open/write failure immediate; delays exact. Verify the caller's resource checks bracket retries so elapsed time is charged to the monotonic guard.
- [ ] Add a Windows regression that warms monitoring initialization, performs 252 reads including a post-GC boundary read, and confirms ctypes type/cache identities do not grow after warm-up. Do not assert an RSS target.
- [ ] Run `python -m unittest discover -s tests -p test_task_run_support.py -v`; confirm new retry/cache tests fail before editing.
- [ ] Implement only bounded replacement retry and cached monitoring declarations. Code 5 may represent permanent access denial, so four failures still fail; do not claim it proves a sharing lock. The existing final metadata flush accounting boundary remains unchanged.
- [ ] Run `python -m unittest discover -s tests -p test_task_run_support.py -v` and `python -m unittest discover -s tests -p test_task_audit_review1.py -v`.
- [ ] Gate: maximum four replace attempts; unrelated, non-Windows, and open/write errors fail immediately; repeated monitoring type/cache identities stabilize. No causal statement about v1 is added.

### Task 2: Pinned v2 loader and pure witness-first candidate generator

**Files:**
- Create: `configs/tasks/constraint_v2.json`
- Create: `dwr/tasks/config_v2.py`
- Create: `dwr/tasks/generator_v2.py`
- Test: `tests/test_task_config_v2.py`
- Test: `tests/test_task_generator_v2.py`

**Interfaces:**
- Consumes: v1 schema/oracle/solver/canonical helpers and the existing `LoadedConfig` record shape.
- Produces: `load_config_v2(path: Path) -> LoadedConfig`, which accepts only the exact closed v2 contract (unknown/missing/nested keys and altered registered values fail; there are no mutable scientific overrides); pure `sample_candidate_header(root_seed, partition, candidate_index)` and `generate_candidate(root_seed, partition, candidate_index) -> CandidateOutcome`. The candidate outcome includes task/IDs-ready inputs, private witness and private predicate evidence; candidate generation has no seen-set or duplicate handling.
- Private predicate ownership: `generator_v2.py` implements `_source_sensitive(task)` and `_mixed_primary_eligibility(task)` for proposal acceptance. These must not be imported by the independent audit.

- [ ] Write strict-loader tests for exact config acceptance, unknown/missing/nested keys, altered generation settings, and v1 `load_config` behavior remaining closed to v2.
- [ ] Write generator tests for seed bytes/domain, profile schedule, n-before-quota header, witness-first tuple feasibility, no feasible vector, constraint and given draw order, deterministic replay, and solver/oracle agreement. Use `python -m unittest discover -s tests -p test_task_config_v2.py -v` and `python -m unittest discover -s tests -p test_task_generator_v2.py -v`; confirm RED first.
- [ ] Implement the exact v2 contract and keep root seed `20261007`; do not alter v1 loader/config/generator. The pure generator returns unique-but-ineligible tasks with private predicate diagnostics rather than owning a corpus-level rejection reason.
- [ ] Test source sensitivity with an independent equality-triangle fixture. Test mixed kind necessity and disconnected graph. Also test the implication that if a present kind has no factor touching an unknown variable, removing that kind cannot add solutions, so it necessarily fails kind necessity; do not demand an impossible fixture where this is the sole failing predicate.
- [ ] Add prompt-byte leakage checks: adding/changing private witness, solution, or predicate metadata leaves `serialize_model_input(task)` exactly unchanged.
- [ ] Run both named discovery commands above, then `python -m unittest discover -s tests -v`.
- [ ] Gate: config is strictly pinned, v1 replay behavior is unchanged, candidate stream is deterministic/pure, and every accepted candidate is unique with solver/oracle agreement.

### Task 3: Versioned orchestration, run-wide deduplication, strict replay, and both modes

**Files:**
- Create: `dwr/tasks/dataset_v2.py`
- Create: `dwr/tasks/replay_v2.py`
- Create: `dwr/tasks/cli_v2.py`
- Test: `tests/test_task_dataset_v2.py`
- Test: `tests/test_task_replay_v2.py`

**Interfaces:**
- Consumes: `LoadedConfig` from `config_v2`, pure `CandidateOutcome`, v1 IDs/split functions, stable JSON/hash helpers, `ResourceGuard`, and atomic manifest writer.
- Produces: `python -m dwr.tasks.cli_v2 --config configs/tasks/constraint_v2.json --mode feasibility --out <new-dir>` and, conditionally, `--mode full --preflight-report <PASS-report> --out <new-dir>`; `replay_outcomes(rows, config, *, check=None) -> list[dict]`.
- Dedup contract: one empty `seen_content_ids` set per run, shared across partitions. For every oracle-unique candidate, compute identities/split, check set, then insert the first ID before denominator counting and terminal eligibility. Thus a first ineligible unique candidate enters the set; later repeats are `duplicate_content`. Oracle-nonunique candidates and profile-quota skips do not enter the set. Replay starts empty and uses the same order/reason precedence.
- Denominators: per partition/profile, retain counts for each first-seen unique oracle-unique candidate before eligibility. Feasibility includes source-train and development-validation only; full mode includes every registered split for its frozen structural audit. Duplicate IDs count nowhere.

- [ ] Write RED tests for output-directory collision, candidate-index order, deterministic run-wide seen set, repeated ineligible-first and eligible-first content, cross-partition set scope, per-profile unique pre-eligibility denominators, and exact terminal reason precedence.
- [ ] Write feasibility-mode tests for 1,000 source groups (334/333/333), 100 eligible development-validation groups per pair, opaque held-out rows, and no final-target sampling.
- [ ] Write full-mode tests for 1,000/1,800/600 eligible groups, 30,000 attempt caps, all split records, exact replay, and rejection of missing/NO_GO/mismatched/dirty-source preflight identities. A matching synthetic PASS fixture may exercise dispatch; do not generate a real corpus in these tests.
- [ ] Run `python -m unittest discover -s tests -p test_task_dataset_v2.py -v` and `python -m unittest discover -s tests -p test_task_replay_v2.py -v`; confirm RED before implementation.
- [ ] Implement both modes and exact replay. Feasibility artifacts preserve held-out source/diagnostic rows for integrity replay but do not expose their counts, task/eligibility/solver aggregates, or strata. Full mode requires a matching PASS report, clean frozen source identity, and new output directory.
- [ ] Assert serialized model input excludes all task identities, witness, solution, solver trace, and eligibility metadata. Write manifest before candidate work; preserve all terminal failures.
- [ ] Run both named discovery commands and `python -m unittest discover -s tests -v`.
- [ ] Gate: synthetic runs replay exactly; the feasibility CLI cannot sample final targets; full dispatch cannot run without a matching PASS report; duplicate behavior matches the exact stateful contract.

### Task 4: Independent aggregation audit and feasibility/full reports

**Files:**
- Create: `dwr/tasks/audit_v2.py`
- Create: `dwr/tasks/audit_cli_v2.py`
- Modify: `dwr/tasks/audit.py` (add explicit cohort-name option to `common_support`; existing five-cohort default unchanged)
- Create: `docs/task_suitability_v2.md`
- Test: `tests/test_task_audit_v2.py`
- Test: `tests/test_task_audit.py`

**Interfaces:**
- Consumes: v2 outcomes/records and replay; existing `dwr.tasks.audit.audit_interactions(task, partition)` as an independent semantic check; v2 config floor values; generation manifest's `generation_elapsed_seconds`.
- Produces: independent per-candidate cross-check plus feasibility/full aggregation. It may not import either private generator predicate. For every unique candidate, recompute interaction evidence with `audit_interactions` and fail if its result disagrees with private annotations. Retain unique pre-eligibility denominator and eligible numerator counts per mixed profile; apply the registered 0.25 fraction threshold to the combined development-validation cohort in feasibility and to each complete mixed partition in the full audit. Count unique eligible signatures for the registered 12-signature floor and the one-example-per-represented-eligible-signature floor. Report rejected-only source/mixed structure signatures separately; rejected-only signatures do not fail an eligible-signature floor. Inherit `max(0, 600 - generation_elapsed_seconds)` as the audit time allowance; generation, replay and audit together cannot exceed 600 active seconds per invocation.

- [ ] Write RED tests for deliberate corruption of private sensitivity/kind-removal/connectivity annotations; independent `audit_interactions` must reject each.
- [ ] Test exact per-profile denominator/numerator accounting with one eligible, one ineligible, and a duplicate unique task in each profile. Require duplicates excluded; test the 0.25 fraction at the combined partition threshold; source and mixed per-signature floors use accepted signature universes, while rejected-only structure coverage is a separate diagnostic.
- [ ] Test feasibility opaque-row behavior by changing source validation/test and development diagnostic task metrics while holding IDs/replay integrity constant; `PASS/NO_GO`, all feasibility counts, and common-support proxy must not change. Test only source-train and three development-validation cohorts feed the four-cohort proxy.
- [ ] Test source-train floors and development-validation floors at exact boundaries; test the four-cohort proxy at 10 groups/cell and 80% mass. Test full mode against all original G2 floors and the unchanged five cohorts: source ID-test all groups, three development-validation eligible cohorts, and final-test eligible cohort.
- [ ] Extend `common_support` with the optional explicit cohort names and test its default output is unchanged for all existing five-cohort fixtures; test the four-cohort proxy against hand-counted cells.
- [ ] Run `python -m unittest discover -s tests -p test_task_audit_v2.py -v`; confirm RED first.
- [ ] Implement aggregation by calling existing `audit_interactions`; do not duplicate its oracle counterfactual/graph implementation. Extend `common_support` with an optional explicit ordered cohort-name tuple so the source-train four-cohort proxy and frozen five-cohort final check share the same cell/mass calculation; its omitted-argument default must remain the exact v1 five-cohort behavior. Treat the first ordered cohort as the all-groups source cohort and filter `primary_eligible` on each later cohort. Keep aggregation/reporting separate from private candidate acceptance.
- [ ] Preserve complete-v2 G2 floors: partition and split group minima; signature minima; source-train kinds and source composition absence; value and clue variation; source 25% sensitivity over unique source groups before eligibility and one per accepted source signature; per-profile mixed 100 eligible groups and 12 eligible signatures; 25% eligible fraction per mixed partition using raw unique pre-eligibility outcomes while retaining per-profile numerators/denominators; one example per represented eligible signature; development 80/20 total and 20/5 per pair; five-cohort support with ≥10 groups/cell and ≥80% cohort mass; replay, leakage, all other strata/audit conditions, caps.
- [ ] Run `python -m unittest discover -s tests -p test_task_audit_v2.py -v`, `python -m unittest discover -s tests -p test_task_audit.py -v`, and `python -m unittest discover -s tests -v`.
- [ ] Gate: no feasibility choice consumes source validation/test or development diagnostic aggregates; full report distinguishes every unchanged floor and computes mixed fractions from raw unique pre-eligibility outcomes, not only accepted records.

### Task 5: One feasibility run and conditional full corpus

**Files:**
- Modify only after evidence: `docs/task_suitability_v2.md`
- Artifacts: unique ignored `data/constraint-v2-feasibility-<run-id>/` and, only after PASS, unique `data/constraint-v2-full-<run-id>/`

**Commands and gates:**

- [ ] Freeze and commit source/config/tests; require clean `git status`. Record source/config hashes, exact command, Python/runtime, RNG, caps, output path, and start time in the resolved manifest.
- [ ] Run exactly once: `python -m dwr.tasks.cli_v2 --config configs/tasks/constraint_v2.json --mode feasibility --out data/constraint-v2-feasibility-<unique-id>`.
- [ ] Run exactly once: `python -m dwr.tasks.audit_cli_v2 --config configs/tasks/constraint_v2.json --mode feasibility --data data/constraint-v2-feasibility-<same-id>`. Generation plus audit share one 600-second active-time budget, one worker, and 1 GiB. Preserve outputs and record actual checks/results.
- [ ] If any fixed applicable floor, independent replay, source-train/development-validation proxy, wall cap, or memory cap fails, record `NO_GO`, leave G2 unchecked, and stop. Do not rerun with changed settings.
- [ ] If the report is `PASS_FEASIBILITY`, the already authorized continuous sequence proceeds using the implemented full mode and matching clean-source report identity. Run one full generation with 1,000/1,800/600 accepted eligible groups and 30,000 attempts per partition; run one full audit over source ID-test, three development-validation eligible cohorts, and final-test eligible cohort. Generation plus audit share a separate 600-second budget, one worker, and 1 GiB.
- [ ] Check off G2 only if the complete five-cohort audit passes and exact commands/outcomes/evidence paths are recorded. Any incomplete or failed full run is preserved; no changed-rule retry.

## Research stop

After a complete G2 PASS, proceed to the next authorized implementation stage: build the fixed-size exchangeable workspace and its indexed control with the matched decoder interface, preserving the residual update equation and gradient flow through frozen decoder computations. Keep N/K and all other comparison settings matched; test permutation behavior and control identities. This CPU plan does not include that model code. Training remains blocked until a separate study registration fills the hypothesis, primary contrast, effect size, sample/seed count, checkpoint rule, compute caps, and stop rule. No feasibility or corpus task authorizes training.
# Prospective clarification — 2026-10-08

For both the four-cohort feasibility proxy and the five-cohort full common-support comparison, source cohorts contain all independently accepted source groups in the applicable split. “All groups” means all accepted source groups regardless of the mixed-primary predicate; independently verified `source_insensitive` rejected proposals are excluded from common-support cells and mass. Those raw first-seen oracle-unique source proposals remain in the registered sensitivity denominator and raw diagnostics. This interpretation applies identically to EX and IDX comparisons, changes no numeric threshold, and does not alter the generator, eligibility predicates, or split definitions.
