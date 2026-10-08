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

Status: complete and review-clean. See the Windows regression and RED/GREEN logs summarized in `.superpowers/sdd/transfer-v2/task-2-report.md` (local ignored evidence).

**Files:**
- Modify: `dwr/tasks/run_support.py`
- Modify: `tests/test_task_run_support.py`
- Test: `tests/test_task_audit_review1.py`

**Interfaces:**
- Consumes: existing `atomic_json(path, value)` and `resident_bytes_windows()` interfaces.
- Produces: at most four `Path.replace` attempts, with delays 0.01/0.02/0.04 seconds for Windows replace error codes 5/32/33; cached/module-level ctypes signatures and `PROCESS_MEMORY_COUNTERS` definition. Terminal errors retain their original type/context.

- [x] Write RED tests with injected replacement: three retryable errors then success; four errors then propagation; other Windows error code immediate; non-Windows error immediate; open/write failure immediate; delays exact. Verify the caller's resource checks bracket retries so elapsed time is charged to the monotonic guard.
- [x] Add a Windows regression that warms monitoring initialization, performs 252 reads including a post-GC boundary read, and confirms ctypes type/cache identities do not grow after warm-up. Do not assert an RSS target.
- [x] Run `python -m unittest discover -s tests -p test_task_run_support.py -v`; confirm new retry/cache tests fail before editing.
- [x] Implement only bounded replacement retry and cached monitoring declarations. Code 5 may represent permanent access denial, so four failures still fail; do not claim it proves a sharing lock. The existing final metadata flush accounting boundary remains unchanged.
- [x] Run `python -m unittest discover -s tests -p test_task_run_support.py -v` and `python -m unittest discover -s tests -p test_task_audit_review1.py -v`.
- [x] Gate: maximum four replace attempts; unrelated, non-Windows, and open/write errors fail immediately; repeated monitoring type/cache identities stabilize. No causal statement about v1 is added.

### Task 2: Pinned v2 loader and pure witness-first candidate generator

Status: implementation and focused tests complete. The initial implementation has no retained functional preimplementation RED; a later constructor-boundary fix has functional RED/GREEN evidence. See `.superpowers/sdd/transfer-v2/task-3-report.md`.

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

- [x] Write strict-loader tests for exact config acceptance, unknown/missing/nested keys, altered generation settings, and v1 `load_config` behavior remaining closed to v2.
- [x] Write generator tests for seed bytes/domain, profile schedule, n-before-quota header, witness-first tuple feasibility, no feasible vector, constraint and given draw order, deterministic replay, and solver/oracle agreement.
- [x] Implement the exact v2 contract and keep root seed `20261007`; do not alter v1 loader/config/generator. The pure generator returns unique-but-ineligible tasks with private predicate diagnostics rather than owning a corpus-level rejection reason.
- [x] Test source sensitivity with an independent equality-triangle fixture. Test mixed kind necessity and disconnected graph. Also test the implication that if a present kind has no factor touching an unknown variable, removing that kind cannot add solutions, so it necessarily fails kind necessity; do not demand an impossible fixture where this is the sole failing predicate.
- [x] Add prompt-byte leakage checks: adding/changing private witness, solution, or predicate metadata leaves `serialize_model_input(task)` exactly unchanged.
- [x] Run `python -m unittest discover -s tests -p test_task_config_v2.py -v` and `python -m unittest discover -s tests -p test_task_generator_v2.py -v`; run `python -m unittest discover -s tests -v` (91 passed at the task commit, with a later focused 16-test boundary regression).
- [x] Gate: config is strictly pinned, v1 replay behavior is unchanged, candidate stream is deterministic/pure, and every accepted candidate is unique with solver/oracle agreement.

TDD boundary: the initial config/generator test run did not retain a functional RED before implementation. Do not treat import absence as functional RED. The later constructor-boundary regression did record functional failures before its fix.

### Task 3: Versioned orchestration, run-wide deduplication, strict replay, and both modes

Status: implementation and review-clean focused tests complete; Task 5's controller suite later passed 138 tests. The initial implementation's RED was import absence only; review fixes have retained functional RED/GREEN evidence. See `.superpowers/sdd/transfer-v2/task-4-report.md`.

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

- [x] Write tests for output-directory collision, candidate-index order, deterministic run-wide seen set, repeated ineligible-first and eligible-first content, cross-partition set scope, per-profile unique pre-eligibility denominators, and exact terminal reason precedence. Functional RED was captured against the reviewed implementation during review fixes; initial import-absence failures are not treated as semantic RED.
- [x] Write feasibility-mode tests for source/development quotas, opaque held-out rows, and no final-target sampling.
- [x] Write full-mode tests for quotas, attempt caps, all split records, exact replay, and rejection of missing/NO_GO/mismatched/dirty-source preflight identities. Synthetic dispatch fixtures only; no real corpus was generated by these tests.
- [x] Run focused dataset/replay tests; 22 passed after the final review fix. The initial implementation's test-first evidence was import absence only.
- [x] Implement both modes and exact replay. Feasibility artifacts preserve held-out rows for replay but omit their aggregates; full mode requires matching clean-source PASS evidence.
- [x] Assert model-input projection excludes private task evidence. Write the manifest before candidate work and preserve terminal failures.
- [x] Run focused tests and full suite; the controller's later fresh full suite passed 138 tests at the frozen source commit.
- [x] Gate: synthetic runs replay exactly; feasibility cannot sample final targets; full dispatch requires PASS; duplicate handling follows the registered stateful contract.

### Task 4: Independent aggregation audit and feasibility/full reports

Status: implementation, review, and tests complete. See `.superpowers/sdd/transfer-v2/task-5-report.md`; the controller's frozen-source suite later passed 138 tests.

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

- [x] Write tests that corrupt private sensitivity/kind-removal/connectivity annotations and require independent `audit_interactions` to reject them. The missing-API `TypeError` was a functional RED for the four-cohort path.
- [x] Test exact per-profile denominator/numerator accounting, duplicate exclusion, the 0.25 fraction, accepted-signature floors, and rejected-only diagnostics.
- [x] Test feasibility opacity and restrict feasibility choices/proxy to source-train and development-validation cohorts.
- [x] Test source/development floors, four-cohort support thresholds, and full-mode five-cohort behavior.
- [x] Extend `common_support` with explicit ordered cohort names while retaining the v1 default output.
- [x] Run v2 and v1 audit suites; 10 and 14 tests passed respectively. The fresh controller suite later passed 138 tests.
- [x] Implement independent aggregation through `audit_interactions`; preserve the registered full G2 floors and separate feasibility from full-mode aggregation.
- [x] Gate: feasibility choices omit held-out aggregates; full reports retain all registered floors and raw pre-eligibility denominators.

### Task 5: One feasibility run and conditional full corpus

Status: the single registered feasibility generation and audit ran. The attempt cap caused NO-GO; stop rule applied, G2 remains unchecked, and full mode was not invoked. See [the actual result](task_suitability_v2.md).

**Files:**
- Modify only after evidence: `docs/task_suitability_v2.md`
- Artifacts: unique ignored `data/constraint-v2-feasibility-<run-id>/` and, only after PASS, unique `data/constraint-v2-full-<run-id>/`

**Commands and gates:**

- [x] Freeze and commit source/config/tests at clean source identity `9f08efddb60537c61a89a90325bc9c832c32ca1c`; record config hash, commands, Python/runtime, RNG, caps, output path, and start time in the manifests.
- [x] Run exactly once: `python -m dwr.tasks.cli_v2 --config configs/tasks/constraint_v2.json --mode feasibility --out data/constraint-v2-feasibility-20261008T062621Z`. It stopped at source candidate 29,999; development/final were not attempted. The public audit reports 734 visible source-train examples, not the total source quota.
- [x] Run exactly once: `python -m dwr.tasks.audit_cli_v2 --config configs/tasks/constraint_v2.json --mode feasibility --data data/constraint-v2-feasibility-20261008T062621Z`. Integrity passed, replay passed for the recorded prefix, and the report left G2 not assessed. Resource use was within caps.
- [x] Record NO-GO at the fixed attempt cap, leave G2 unchecked, and stop. Do not retry with changed settings.
- [ ] Conditional full generation and audit: not invoked because feasibility ended NO-GO.
- [ ] Check off G2 only if the complete five-cohort audit passes and exact commands/outcomes/evidence paths are recorded. Any incomplete or failed full run is preserved; no changed-rule retry.

## Research stop

After a complete G2 PASS, proceed to the next authorized implementation stage: build the fixed-size exchangeable workspace and its indexed control with the matched decoder interface, preserving the residual update equation and gradient flow through frozen decoder computations. Keep N/K and all other comparison settings matched; test permutation behavior and control identities. This CPU plan does not include that model code. Training remains blocked until a separate study registration fills the hypothesis, primary contrast, effect size, sample/seed count, checkpoint rule, compute caps, and stop rule. No feasibility or corpus task authorizes training.
# Prospective clarification — 2026-10-08

For both the four-cohort feasibility proxy and the five-cohort full common-support comparison, source cohorts contain all independently accepted source groups in the applicable split. “All groups” means all accepted source groups regardless of the mixed-primary predicate; independently verified `source_insensitive` rejected proposals are excluded from common-support cells and mass. Those raw first-seen oracle-unique source proposals remain in the registered sensitivity denominator and raw diagnostics. This interpretation applies identically to EX and IDX comparisons, changes no numeric threshold, and does not alter the generator, eligibility predicates, or split definitions.
