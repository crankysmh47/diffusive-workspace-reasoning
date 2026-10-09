# V4 CPU development-proposal execution plan

Date: 2026-10-09. Binding specification: [task_spec_v4.md](task_spec_v4.md); exact proposal config: [constraint_probe_v4.json](../configs/tasks/constraint_probe_v4.json). Scope: implement and, only after review, execute one bounded development-only proposal probe. This plan does not authorize final generation, a formal G2 preflight, model code, or training. The documentation milestone currently underway is Task 1 of the controller work; the numbered implementation tasks below are separate sampler milestones and do not imply any sampler code has been written.

## Goal

Determine with one bounded CPU development proposal whether the frozen v4 source and three development-validation cohorts satisfy their registered quotas, floors, support, replay and independent-audit gates. A successful result is only `PASS_DEVELOPMENT_PROPOSAL`.

## Architecture

Use one versioned stateless proposal generator and a small v4-specific driver. The driver owns quotas, run-wide identity/deduplication state, denominator projections and resource accounting. It calls existing schema/canonicalization/oracle/solver primitives through explicit interfaces. A separate v4 audit implementation independently verifies records and eligibility interactions. Historical v1/v2/v3 pipelines remain unchanged. One driver process performs source generation and source closure checks, conditionally performs development generation, then performs the terminal replay/audit and report finalization under one resource guard.

## Tech Stack

Python 3.11/3.12 standard library for config, streams, generation, manifests, serialization, replay and logs. Reuse the repository's existing task schema, canonicalizer, solver, oracle and independent interaction checker. No ML dependency, GPU, model code or external dataset.

## Spec

The exact grammar, quotas, RNG, outcomes, denominators, support envelope, privacy, floors, resource caps and stop behavior are pinned in [task_spec_v4.md](task_spec_v4.md) and the closed config. This plan is subordinate to those files.

## GlobalConstraints

Use the clean `research/transfer-cpu-v1` branch. Preserve original evidence and unrelated user files. Do not weaken thresholds, inspect protected cohort aggregates, generate final data, or retune after results. Log each attempted candidate and preserve incomplete/negative evidence. No todo item is checked by a plan or proposal probe.

## ReviewFocus

Review version isolation and strict schema closure; exact candidate draw/rejection order; denominator insertion before private eligibility/envelope decisions; prompt-answer separation; held-out opacity; source-before-development gate; independent replay/audit; and one shared 900-second/1-GiB guard through terminal report writes. Keep the implementation minimal and the decision label distinct from G2.

## Shared implementation rules

Use only standard-library task/generation/replay/manifest code plus the repository's existing `constraint-task-v1` schema, canonicalizer, solver, oracle, and independent interaction audit through explicit functions. Do not alter v1/v2/v3 dispatch, candidate outcomes, configs, accepted records, stream contracts, loaders, audit reports, or tests except narrowly necessary backwards-compatible shared interfaces identified in task review. Do not clone the historical pipelines. New v4 objects remain separately versioned and reject unknown keys/types, duplicate JSON keys, nonfinite numbers, malformed hashes, identities from another version, invalid shapes, incompatible output paths, and incomplete resumes. During sampler implementation, unit tests may call real proposal functions on tiny fixed synthetic candidate fixtures to prove RNG, oracle, solver, eligibility, and rejection behavior. Those fixtures are correctness checks, not registered candidate sampling. The actual v4 quotas, production candidate pool, source closure, probe replay/audit, final partition and training remain forbidden until the reviewed single run in Task 3.

Use source branch `research/transfer-cpu-v1` and require the documented clean base before code. Keep all real artifacts under a unique ignored `data/constraint-probe-v4-<UTC>/` directory. Save small config/split/hash manifests in Git, not generated rows or logs. Run meaningful pure contract/arithmetic checks for these documents only; do not repeat the repository suite during this docs milestone. Later code tasks require scoped tests and a fresh full suite only after integration.

## Implementation Task 1 — pinned development configuration and stateless proposal

- [ ] Add the closed config and stateless candidate proposal.
- [ ] Add fixture-based correctness checks; record exact commands and outcomes.
- [ ] Self-review the complete contract and obtain an independent task review; fix findings.

Files: add `configs/tasks/constraint_probe_v4.json`; add one real proposal module under `dwr/tasks/` (suggested `generator_probe_v4.py`); add focused `tests/` for the closed config, candidate stream, cell schedule, witness populations, normalized tuple/count-vector order, exact sample order, typed proposal failures, and separation from historical outcome dispatch. Synthetic fixed fixtures may invoke the proposal/oracle/solver as described above. No production candidate-pool sampling, manifest, run driver, corpus replay/audit or registered quota execution in this task.

Interface: `load_probe_config(path) -> ProbeConfig` validates exactly the documented v4 config and rejects duplicate/unknown keys, bool-as-int, NaN/Infinity, wrong types, invalid quota sums, mismatched strata, and altered constants. `candidate_stream(config, partition, candidate_index) -> random.Random` implements the exact domain bytes. `next_active_cell(config, partition, candidate_index, remaining_quota) -> (profile,stratum)` is pure and errors before stream construction for empty/malformed state. `propose_candidate(config, partition, candidate_index, active_cell) -> ProbeCandidateOutcome` is stateless and returns one fully closed proposal outcome before orchestration identity/dedup state. It implements exact source-EQ and source-NEQ witness populations, IID witness order for other profiles, exact pools, count vectors, tuple selection, witness-as-givens, independent oracle uniqueness, frozen solver agreement, and private eligibility. The orchestrator later owns run-wide content/lineage/signature sets, denominators, envelope rejection, quotas, and stream terminal precedence.

Review checkpoint: task owner self-checks contract coverage against every config field and outcome reason; an independent task reviewer inspects diff and specific scope. Fix all actionable findings before the next task. Exact commands and RED/GREEN outcomes are recorded by the implementer; tests must state fixture identity and must not claim corpus yield or feasibility.

## Implementation Task 2 — bounded probe orchestration, replay, and independent audit

- [ ] Add one guarded driver, source closure replay/audit, full replay, and independent terminal audit.
- [ ] Add focused orchestration, privacy, corruption, failure, and accounting checks; record exact outcomes.
- [ ] Self-review and obtain an independent task review; fix findings before integration.

Files: add a distinct `dwr/tasks/probe_v4.py`, `audit_probe_v4.py` and CLI entry points, reusing only audited low-level helpers; add tests for identity/split/dedup order, denominator timing, envelope accounting, privacy projections, quota transitions, source-before-development stop, partial/operational/floor failure, atomic writes, resume rejection, accounting, replay corruption, independent interaction mismatch, source/four-cohort calculations and held-out opacity. Avoid changes to v3 production branch dispatch. Any shared change needs explicit compatibility tests against affected v2/v3 consumers.

The closed probe artifact set is: pre-run `probe_manifest.json`; `outcomes.jsonl`; `accepted_records.jsonl`; `source_closure.json`; `candidate_replay.json`; `independent_audit.json`; `probe_report.json`; and terminal `probe_manifest.json` with exact sizes/SHA-256. The manifest is written before draws and binds clean source commit/diff identity, all producer dependency hashes, config bytes/hash, Python/OS, RNG, quotas, command, unique path, resource limits and artifact versions. Replay starts from empty seen/denominator/quota state, regenerates candidate outcomes exactly, recalculates hashes and split identities, verifies every outcome/accepted row, and consumes the same live resource guard. A separate audit implementation reopens validated files, independently verifies tasks, oracle/solver claims, eligibility interactions, model-input annotation exclusion, canonical grouping/splits, allowed denominator projections, cell bins, floors and support. It must not import private source/mixed eligibility predicates or use hidden split rows to compute proposal choices.

Terminal source boundary: only after source quotas and source-prefix replay/independent source checks pass may the controller begin development. If source support or any source-train floor fails, seal a scientific NO_GO without a development draw. The opaque source validation/test rows must be checksum/replay only. The same rule keeps diagnostic development aggregates out of choices. Never open or construct final rows. One live guard covers candidate sampling, source-prefix replay, complete replay/audit, artifact hashing and report finalization; no budget reset. Resource caps become `NOT_ASSESSED`, with partial artifacts preserved.

Review checkpoint: fresh implementer self-review, then independent task review. Do not advance until the reviewer sees the full consumer contract and no unresolved blocking finding. Tests here are synthetic correctness checks; they do not consume the one scientific attempt and do not establish task yield.

## Implementation Task 3 — integration freeze and the sole conditional scientific probe

- [ ] Freeze reviewed code on a clean source commit; run focused checks and one fresh full CPU suite; resolve source/config/producer identities.
- [ ] Obtain final independent review of the exact driver and consumer contracts.
- [ ] Run the single conditional probe once; preserve the complete terminal status, artifacts, hashes, resource use, and evidence path.

Before any candidate draw, freeze implementation at a clean commit; rerun focused v4 checks and the full existing CPU suite once on that root commit; run config schema/arithmetic checks; calculate all producer SHA-256 identities; inspect `git status`; require no generated/scratch files staged. Save a resolved manifest and exact PowerShell commands before execution. Reviewer performs a final independent review of the integration diff and probe-run consumer, including the no-G2/no-final guarantees.

Only after that review, execute exactly once as one end-to-end driver invocation:

```powershell
python -m dwr.tasks.probe_cli_v4 --config configs/tasks/constraint_probe_v4.json --mode development-probe --out data/constraint-probe-v4-<UTC>
```

Resolve `<UTC>` to one new never-before-used directory in the pre-run manifest. This one process writes the initial manifest before draws; samples source; flushes a source prefix; independently replays it and computes only permitted source-train closure/floors/support; stops before development on any source failure; otherwise samples development; then independently replays the complete outcome/record streams, runs the separate semantic audit and four-cohort proxy, hashes artifacts, and finalizes the terminal report. The source replay/audit and final replay/audit each consume the same monotonic 900-second wall guard and 1-GiB RSS limit; there is no second audit CLI or fresh budget. Record CPU separately. Use one worker and <=30,000 candidate indices in each of source/development. No retry, extension or resume. If completing the audit cannot fit, preserve evidence and report `NOT_ASSESSED`.

The only successful label is `PASS_DEVELOPMENT_PROPOSAL`. Any quota/floor/support/proxy failure is terminal `NO_GO_PROPOSAL`; resource, environment, serialization, solver/oracle, replay, audit or integrity failure is `NOT_ASSESSED`. Do not call either outcome G2 or formal feasibility. Record exact commands, source/config/dependency/data hashes, accepted/attempted counts by only permitted cohorts, all status/reason fields, active wall and CPU time, peak RSS, and evidence location. Preserve raw outcomes in ignored storage and back up evidence before reproducibility claims. Update `todo.md` only after a separately verified full milestone; this probe never checks off G2.

## Conditional work after a real PASS

After and only after complete `PASS_DEVELOPMENT_PROPOSAL`, open a separate bounded design/review task for the minimal full-run configuration, closed v4 manifests/records/audit/preflight consumer and immutable probe-result linkage. No full schema is prebuilt in this plan. Its reviewed implementation and clean-source full manifest must precede the single separately budgeted 2,400-second full generation and actual five-cohort G2 audit. The five cohorts, all original numeric floors, independent oracle/interaction/replay, and hidden-test access restrictions remain binding. A complete actual G2 PASS is required before model implementation. Before training, fill and review the separate experiment registration: hypothesis/primary contrast, practical effect and precision rationale, >=3 training seeds, checkpoint rule, tuning/training/inference budgets, memory measurements, paired/family-grouped uncertainty and terminal stop rule. Null settings block training; the development probe is not model evidence.
