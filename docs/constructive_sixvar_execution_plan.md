# Six-variable certificate study implementation plan

> For agentic workers: use superpowers:subagent-driven-development. Luna implements; controller decides and reviews evidence.

Goal: implement and execute the single prospectively specified capacity/difficulty study.
Architecture: separate candidate construction, synchronous local propagation, and bounded study orchestration; existing schema, canonicalizer, verifier/oracle and reference solver remain unchanged.
Tech stack: Python 3.11 standard library, unittest, Windows paths.
Spec: [constructive_sixvar_study.md](constructive_sixvar_study.md).

Global constraints: preserve historical modules/artifacts, task semantics, splits and 25%/80% floors. Fixed source3600 then conditional development2400; seed20261011;1800 seconds/1GiB/one worker; tests120 seconds; no retry or final tasks. No model/training. Owner delegation covers review/design decisions without repeated approval.

Review focus: private split metrics entering public summaries; oracle-guided hidden construction retries; full eligible-cohort support denominators; terminal hashing beyond guard; bad config/types or path collision; shared predicate bugs; nondeterministic set iteration/RNG order. Preserve failed tests and typed errors.

## Current operational blocker — 2026-10-11

Both requested gpt-6-luna implementers terminated with a tool-reported usage limit before either production module existed. The constructor agent left one partial test file; it is preserved as ignored `.superpowers/sdd/sixvar/test_certificate_generator.partial-usage-limit.py`, outside unittest discovery because its implementation is absent. No meaningful red/green result was observed, no implementation milestone is complete, and no scientific candidate was drawn. Root reviewed the design/registration for internal consistency and parsed its JSON; those are document checks only. Resume Tasks 1/2 with Luna when available, inspect the preserved test draft rather than treating it as verified, then complete runner/reviews/clean-source freeze before the sole production invocation. Do not silently switch the owner's requested implementation model or consume an experiment retry for this pre-execution tooling failure.

## Task 1: local propagation diagnostic

Files: create dwr/tasks/local_propagation.py and tests/test_local_propagation.py.
Interface: propagate(task: ConstraintTask) returns immutable domains in task variable order, solved/stalled, unresolved unknown count, synchronous rounds and local support-check count. It may not read answers.

- [ ] Write tests before code; verify missing-module failure. Test anchored EQ chain, unsupported ADD value pruning, synchronous order invariance, contradiction, disconnected variables and a fixed-point stall without falsely labeling uniqueness.
- [ ] Implement finite support enumeration with snapshot domains and simultaneous intersections. Reject wrong containers and serialize/validate task first.
- [ ] Cross-check a separate brute-force local support reference over tiny fixtures; ensure every exhaustive solution remains within final domains.
- [ ] Run python -m unittest discover -s tests -p test_local_propagation.py -v; retain exact red/green commands and outcomes in ignored .superpowers/sdd/sixvar/; controller review before integration.

## Task 2: certificate constructor

Files: create dwr/tasks/certificate_generator.py and tests/test_certificate_generator.py.
Interface: construct(root_seed:int, partition:str, candidate_index:int, *, check_budget:Callable[[],None]|None=None) returns profile/cell/given/witness/alternative/protected-constraint/work metadata and task-or-null/rejection reason. Construction does not call oracle/solver or look at splits.

- [ ] Write tests and retain valid missing-module red output; pin deterministic candidate metadata, no global RNG, exact cell scheduling, empty feasible pools and invalid indices/partition/seeds.
- [ ] Implement the exact single-path pools/count-vector/RNG order in the spec, optional periodic budget checks, and normalized constraint sorting.
- [ ] Validate protected alternatives independently with scalar tests, including cross-kind invalidation, givens preservation and canonical tuple order. Check no oracle/solver imports/calls in the constructor.
- [ ] Run python -m unittest discover -s tests -p test_certificate_generator.py -v and retain evidence; review the component before real candidate execution.

## Task 3: bounded runner and registration

Files: create dwr/tasks/certificate_study.py, tests/test_certificate_study.py and configs/tasks/certificate_sixvar_v1.json.
Interface: python -m dwr.tasks.certificate_study --config configs/tasks/certificate_sixvar_v1.json --output runs/certificate-sixvar-<UTC>; no alternate scientific settings accepted silently. Runner consumes construct and propagate; output conforms to spec.

- [ ] Start from failing closed-config/public-projection/resource/collision/failure-preservation tests. Use injected canned rows/guards for runner tests, never a population trial. Fixture-only limits are separately marked and cannot emit production PASS.
- [ ] Implement independent task/certificate counting, existing oracle/solver identity agreement, canonical IDs and unchanged split assignment, private outcomes, source/development gating, GAC and full eligible support denominators. Pure source EQ difficulty is diagnostic only.
- [ ] Implement initial/terminal manifests, all producer hashes, stable JSONL, exact per-attempt accounting, deterministic replay, typed terminal failures, resource guard and bounded final flush; fail if opaque metrics are published.
- [ ] Run scoped tests, then full CPU suite under declared test allowance. Review whole diff and dependency/hash/privacy/floor closure. Freeze config/source on clean branch before any real draw.

## Task 4: controller execution and decision

- [ ] Record exact invocation/command budget and launch ONE production study under the separate watchdog; no standalone ignored launcher import path.
- [ ] Verify hashes/identities, permitted aggregates, actual costs and replay outcome without using protected rows for decisions.
- [ ] Publish a compact tracked result and update todo/execution state only for verified milestones. Preserve negative/partial outcomes. If PASS, specify next full G2 study; if failure, stop at that genuine blocker without changing floors or seed.
