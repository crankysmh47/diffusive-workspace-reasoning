# Task design v2 implementation and feasibility plan

> **For agentic workers:** implement each checked step only after the v2 specification and this plan are reviewed. Keep work on `research/transfer-cpu-v1`; do not start model work or training.

**Goal:** Implement and evaluate one versioned CPU-only task-generation proposal that preserves the v1 grammar and all numeric G2 floors while testing whether source sensitivity and mixed-task eligibility can support fixed-N composition transfer.

**Architecture:** Keep the frozen v1 generator/config/audit replayable. Add an explicit v2 config and pure candidate generator, then add isolated v2 orchestration/replay and independent feasibility reporting. A single source-train plus development-validation feasibility command is the decision gate; only a pass can be followed by one separately manifested complete-corpus generation and five-cohort audit.

**Tech Stack:** Python 3.11/3.12 standard library, `pathlib`, `hashlib`, `random.Random`, JSONL, SHA-256, and `unittest`; no ML dependency or GPU.

**Spec:** [docs/task_spec_v2.md](task_spec_v2.md), under [docs/research_protocol.md](research_protocol.md) and the fixed-N contract in [docs/transfer_research_direction.md](transfer_research_direction.md).

## Global Constraints

- Keep v1 spec/config, interrupted run, partial metrics, and all v1 numeric G2 floors unchanged.
- Preserve the v1 grammar, schema/parser/verifier, independent oracle, solver, IDs, split hashes, prompt projection, and model-input leakage boundary.
- Use only the v2 balanced profile schedule, witness-first sampling, and prespecified sensitivity/eligibility acceptance in the dated proposal.
- Use source train and development validation only for the one feasibility decision; diagnostic-test and all final-target records/metrics are unavailable to choices.
- Use one worker, at most 30,000 candidate indices per attempted partition, at most 600 total active seconds for generation plus audit, and 1 GiB resident memory. Stop and preserve evidence at any cap.
- Write resolved manifests before work, use unique artifact directories, stable UTF-8 serialization, per-candidate outcomes and SHA-256 identities. Never overwrite or delete the v1 artifacts.
- Do not tune, pad strata, replace failures, weaken floors, or launch model/training work. A feasibility failure is a documented no-go.

## Review Focus

- A witness-first draw that has no feasible tuple/count vector must reject deterministically without inventing a task; pin exact draw order, reason, and replay in Task 1 tests.
- A source structure whose every single-constraint removal stays unique must fail source acceptance; pin with the equality triangle fixture in Task 2.
- A mixed task missing any-kind necessity or a connected unknown-variable factor graph must fail eligibility; pin each independent predicate in Task 2.
- Witness, solution, solver counters, and sensitivity diagnostics could leak through prompt or serialized model-input bytes; compare bytes and verify annotations are absent in Tasks 1 and 3.
- Windows monitoring initialization could retain a new ctypes type per poll or mask a permanent write error; pin type/cache stability and sharing-only bounded retry behavior in Task 4.

---

### Task 1: Frozen v2 config and witness-first candidate stream

**Files:**
- Create: `configs/tasks/constraint_v2.json`
- Create: `dwr/tasks/generator_v2.py`
- Test: `tests/test_task_generator_v2.py`

**Interfaces:**
- Consumes: existing `Constraint`, `ConstraintTask`, `GivenAssignment`, `Solution`, `enumerate_solutions`, and `solve` interfaces from `dwr/tasks/`.
- Produces: `sample_candidate_header(root_seed: int, partition: str, candidate_index: int) -> CandidateHeader`; `generate_candidate(root_seed: int, partition: str, candidate_index: int, *, full_profiles: frozenset[tuple[str,...]] = frozenset()) -> CandidateOutcome`; immutable candidate outcomes with one documented rejection reason and private witness/counterfactual metadata.

- [ ] Write tests first for config strictness, seed derivation, profile round-robin, and n-before-quota behavior. Assert the exact v2 seed bytes and first candidate outputs against fixed small golden fixtures; do not use v1 goldens.
- [ ] Run `python -m unittest tests.test_task_generator_v2 -v`; confirm the new import/API fails before implementation.
- [ ] Implement only the v2 config and pure generator. Follow spec draw order exactly: `n`, profile/quota, witness, witnessed feasible tuple/count choices, then givens; use local RNG only. Keep v1 generator untouched.
- [ ] Add RED/GREEN cases for no witnessed feasible vector, witness-valid relations, canonical constraint order, duplicate candidate rejection behavior, deterministic repeat, and exact solver/oracle agreement for every accepted fixture.
- [ ] Add a leakage assertion that `serialize_model_input(task)` is byte-identical whether private witness/solution metadata is present or absent and contains none of those fields.
- [ ] Run `python -m unittest tests.test_task_generator_v2 tests.test_task_schema tests.test_task_oracle tests.test_task_solver -v`.
- [ ] Gate: every new candidate is deterministic, replays exactly, and any accepted task is independently unique with solver agreement; no v1 file/config changes.

### Task 2: Independent sensitivity and mixed eligibility checks

**Files:**
- Create: `dwr/tasks/audit_v2.py`
- Test: `tests/test_task_audit_v2.py`

**Interfaces:**
- Consumes: v1 exhaustive `enumerate_assignments` oracle for counterfactual constraints, v1 `enumerate_solutions`, `ConstraintTask`, and v2 candidate metadata.
- Produces: `source_sensitivity(task: ConstraintTask) -> dict`; `mixed_primary_eligibility(task: ConstraintTask) -> dict`; outputs include per-removal solution counts, factor connectivity, kinds in connected factors, and a boolean predicate. These functions do not call the solver or generator acceptance helper.

- [ ] Write independent fixture tests for sensitive and insensitive source tasks, including the n=3 equality triangle whose three edge removals each leave one solution.
- [ ] Write mixed fixtures that fail exactly one condition at a time: one present kind unnecessary, unknown-variable factor graph disconnected, and a present kind absent from the connected graph; add one passing fixture.
- [ ] Run `python -m unittest tests.test_task_audit_v2 -v`; confirm expected failures before implementation.
- [ ] Implement counterfactual enumeration with the independent oracle and explicit deterministic ordering. Treat oracle disagreement or invalid task shape as fatal.
- [ ] Cross-check the v2 report against independently hand-enumerated tiny fixtures; assert the generator cannot certify its own eligibility result.
- [ ] Run `python -m unittest tests.test_task_audit_v2 tests.test_task_audit tests.test_task_oracle -v`.
- [ ] Gate: exact deletion counts and graph diagnostics match fixtures; v1 audit outputs remain untouched.

### Task 3: v2 orchestration, outcomes, and exact replay

**Files:**
- Create: `dwr/tasks/dataset_v2.py`
- Create: `dwr/tasks/replay_v2.py`
- Create: `dwr/tasks/cli_v2.py`
- Test: `tests/test_task_dataset_v2.py`
- Test: `tests/test_task_replay_v2.py`

**Interfaces:**
- Consumes: `LoadedConfig`, `ResourceGuard`, `atomic_json`, stable JSON/hash helpers, v2 candidate generator, v1 canonical IDs/split assignment, and Task 2 independent eligibility functions.
- Produces: CLI `python -m dwr.tasks.cli_v2 --config <v2-config> --mode feasibility --out <new-dir>`; replay API `replay_outcomes(rows, config, *, check=None) -> list[dict]`; a manifest written before the first candidate, JSONL of all outcomes, accepted records with private annotations outside `model_input`, aggregate counts, and artifact hashes.

- [ ] Write tests for output-directory collision, partition/profile quotas, candidate-index ordering, duplicate content accounting, source hash splits, development structure splits, unknown config keys, and missing/extra outcome fields.
- [ ] Run `python -m unittest tests.test_task_dataset_v2 tests.test_task_replay_v2 -v`; confirm failures before implementation.
- [ ] Implement only `feasibility` mode: 1,000 source groups total (profile quotas 334/333/333) and 100 eligible development-validation groups per pair. Keep source validation/test and development diagnostic-test rows opaque; exclude their counts, task/eligibility/solver metrics, and strata from the feasibility decision. Integrity-only replay/hash checks are allowed. Do not permit final-target generation in this command.
- [ ] Store every candidate attempt and rejection, including unique-but-ineligible mixed tasks and their eligibility diagnostics. Accepted source/development records contain only the exact unchanged v1 model-input projection plus separately stored private metadata.
- [ ] Replay every recorded candidate from seed/config and compare typed outcomes, reason precedence, candidate index, accepted task, IDs, split, and counts exactly.
- [ ] Add per-artifact SHA-256 and strict `UTF-8 no BOM`, newline, and stable serialization tests.
- [ ] Run `python -m unittest tests.test_task_dataset_v2 tests.test_task_replay_v2 tests.test_task_canonical tests.test_task_schema -v`.
- [ ] Gate: a synthetic small run fully replays; a nonempty preexisting output directory fails without mutation; no final target can be sampled in feasibility mode.

### Task 4: Windows-safe manifest replacement and stable RSS monitoring

**Files:**
- Modify: `dwr/tasks/run_support.py`
- Test: `tests/test_task_run_support.py`
- Test: `tests/test_task_audit_review1.py`

**Interfaces:**
- Consumes: existing `atomic_json(path, value)` and `resident_bytes_windows()` interfaces.
- Produces: atomic replacement that retries at most four total attempts with 0.01, 0.02, 0.04 second delays only for the specified Windows replace sharing/lock errors; module-level or cached Windows ctypes signatures and `PROCESS_MEMORY_COUNTERS`; terminal errors retain their original failure context.

- [ ] Write tests first with an injected replace function: three retryable Windows sharing failures then success; four failures propagate; non-Windows failure propagates immediately; unrelated error propagates immediately; delays are exact and consume the supplied guard/time accounting.
- [ ] Add a repeated-monitor-read regression that warms initialization, performs 252 reads (including a final post-GC check), and asserts ctypes type/cache identity counts do not grow after warm-up. Do not assert a particular RSS change.
- [ ] Run `python -m unittest tests.test_task_run_support -v`; confirm the retry and type-stability cases fail before changes.
- [ ] Implement only the bounded retry and cached/module-level monitoring declarations. Never retry resource failures or non-replace operations. The controller must read live manifests with compatible sharing or wait for terminal completion.
- [ ] Run `python -m unittest tests.test_task_run_support tests.test_task_audit_review1 -v`.
- [ ] Gate: exactly four attempts maximum; permanent, unrelated, and non-Windows errors stay failures; monitoring type/cache identities stabilize. The policy does not attribute the historical WinError 5 to this monitor issue.

### Task 5: Independent v2 audit and one feasibility decision

**Files:**
- Modify: `dwr/tasks/audit_v2.py`
- Create: `dwr/tasks/audit_cli_v2.py`
- Create: `docs/task_suitability_v2.md`
- Test: `tests/test_task_audit_v2.py`

**Interfaces:**
- Consumes: completed feasibility output, replay implementation, v1 floor definitions, source-train content split and three development-validation pair cohorts.
- Produces: one audit report with source sensitivity yield, mixed eligibility denominators/fractions, source and pair strata, unchanged-floor results, a four-cohort source-train/development-validation common-support proxy, outcome/artifact hashes, and a terminal `PASS_FEASIBILITY` or `NO_GO` status.

- [ ] Write tests that fail on any source-train floor (100 groups, 12 signatures, all three kinds, value fractions 0.1, two given counts, sensitivity fraction 0.25, and one sensitive example per train signature), pair eligibility floor (fraction 0.25, 100 groups, 12 signatures, one per eligible signature), fewer than 80 development-validation groups, fewer than 20 per pair, or failure of unchanged common-support cell/mass thresholds.
- [ ] Add tests proving the proxy uses exactly source-train and the three development-validation cohorts; source validation/test and development diagnostic-test rows are opaque and cannot affect counts or metrics; final cohorts are rejected as inputs.
- [ ] Run `python -m unittest tests.test_task_audit_v2 -v`; confirm each floor test fails before implementing aggregation.
- [ ] Implement independent replay before metrics. Preserve raw source-train/development-validation counts and failures; held-out source/diagnostic rows carry integrity-only status in the feasibility report. Empty/unattempted cohorts are `not_assessed`, never numeric zero evidence. A source/development proxy pass must not report five-cohort G2 pass.
- [ ] Run `python -m unittest discover -s tests -v` and record the exact count/outcome and evidence path in `docs/task_suitability_v2.md` only after the synthetic and audit tests pass.
- [ ] Gate: code, config, analysis, and source commit are frozen before the single feasibility invocation. Write a fresh resolved manifest, run once in a unique ignored artifact directory, audit once under the same remaining global budget, and preserve all outputs. Any failure/cap gives `NO_GO`; no retry or changed setting.
- [ ] If `NO_GO`, record the result and stop with G2 unchecked. If `PASS_FEASIBILITY`, continue under the already authorized sequence: create a separate clean-source manifest and unique directory for one full-corpus generation using 1,000/1,800/600, 30,000 attempts/partition, 600 seconds total, 1 GiB, one worker; generate all partitions from the frozen v2 streams and run the full unchanged five-cohort audit once. Check off no gate unless that complete audit passes and exact commands/results/evidence are recorded.

## Stop conditions and handoff

Stop before execution if independent oracle checks, replay, source or pair feasibility, source-train/development-validation proxy common support, the attempt limit, wall cap, or memory cap fails. Keep the run and report its measured state; never resume a partial candidate stream under a changed config. Final-target generation occurs only in the separate complete-corpus invocation after feasibility and clean-source freeze. Even a complete G2 pass does not authorize training: the model registration's hypothesis, primary contrast, effect size, seed count, checkpoint rule, compute caps and stop rule must all be populated and reviewed first.
