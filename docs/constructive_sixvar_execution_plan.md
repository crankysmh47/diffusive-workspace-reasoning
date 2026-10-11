# Six-variable certificate study implementation plan

> For agentic workers: use superpowers:subagent-driven-development. Luna implements; controller decides and reviews evidence.

Goal: implement and execute the single prospectively specified capacity/difficulty study.
Architecture: separate candidate construction, synchronous local propagation, and bounded study orchestration; existing schema, canonicalizer, verifier/oracle and reference solver remain unchanged.
Tech stack: Python 3.11 standard library, unittest, Windows paths.
Spec: [constructive_sixvar_study.md](constructive_sixvar_study.md).

Global constraints: preserve historical modules/artifacts, task semantics, splits and 25%/80% floors. Fixed source3600 then conditional development2400; seed20261011;1800 seconds/1GiB/one worker; tests120 seconds; no retry or final tasks. No model/training. Owner delegation covers review/design decisions without repeated approval.

Review focus: private split metrics entering public summaries; oracle-guided hidden construction retries; full eligible-cohort support denominators; terminal hashing beyond guard; bad config/types or path collision; shared predicate bugs; nondeterministic set iteration/RNG order. Preserve failed tests and typed errors.

## Preserved service interruption — 2026-10-11

Both requested gpt-6-luna implementers terminated with a tool-reported usage limit before either production module existed. The constructor agent left one partial test file; it is preserved as ignored `.superpowers/sdd/sixvar/test_certificate_generator.partial-usage-limit.py`, outside unittest discovery because its implementation is absent. No meaningful red/green result was observed, no implementation milestone is complete, and no scientific candidate was drawn. Root reviewed the design/registration for internal consistency and parsed its JSON; those are document checks only. Resume Tasks 1/2 with Luna when available, inspect the preserved test draft rather than treating it as verified, then complete runner/reviews/clean-source freeze before the sole production invocation. Do not silently switch the owner's requested implementation model or consume an experiment retry for this pre-execution tooling failure.

On 2026-10-12 the owner requested another service attempt. Both Luna tasks resumed successfully; the constructor and propagation components are now implemented and undergoing final review/fixes, and the runner is in development. The prior interruption is preserved as historical evidence, not a current quota blocker. No population/study invocation has occurred; its allowance remains unused.

## Task 1: local propagation diagnostic

Files: create dwr/tasks/local_propagation.py and tests/test_local_propagation.py.
Interface: propagate(task: ConstraintTask) returns immutable domains in task variable order, solved/stalled, unresolved unknown count, synchronous rounds and local support-check count. It may not read answers.

- [x] Write tests before code; verify missing-module failure. Test anchored EQ chain, unsupported ADD value pruning, synchronous order invariance, contradiction, disconnected variables and a fixed-point stall without falsely labeling uniqueness.
- [x] Implement finite support enumeration with snapshot domains and simultaneous intersections. Reject wrong containers and serialize/validate task first.
- [x] Cross-check a separate brute-force local support reference over tiny fixtures; ensure every exhaustive solution remains within final domains.
- [x] Run python -m unittest discover -s tests -p test_local_propagation.py -v; retain exact red/green commands and outcomes in ignored .superpowers/sdd/sixvar/; controller review before integration.

## Task 2: certificate constructor

Files: create dwr/tasks/certificate_generator.py and tests/test_certificate_generator.py.
Interface: construct(root_seed:int, partition:str, candidate_index:int, *, check_budget:Callable[[],None]|None=None) returns profile/cell/given/witness/alternative/protected-constraint/work metadata and task-or-null/rejection reason. Construction does not call oracle/solver or look at splits.

- [x] Write tests and retain valid missing-module red output; pin deterministic candidate metadata, no global RNG, exact cell scheduling, empty feasible pools and invalid indices/partition/seeds.
- [x] Implement the exact single-path pools/count-vector/RNG order in the spec, optional periodic budget checks, and normalized constraint sorting.
- [x] Validate protected alternatives independently with scalar tests, including cross-kind invalidation, givens preservation and canonical tuple order. Check no oracle/solver imports/calls in the constructor.
- [x] Run python -m unittest discover -s tests -p test_certificate_generator.py -v and retain evidence; review the component before real candidate execution.

## Task 3: bounded runner and registration

Files: create dwr/tasks/certificate_study.py, tests/test_certificate_study.py and configs/tasks/certificate_sixvar_v1.json.
Interface: python -m dwr.tasks.certificate_study --config configs/tasks/certificate_sixvar_v1.json --output runs/certificate-sixvar-<UTC>; no alternate scientific settings accepted silently. Runner consumes construct and propagate; output conforms to spec.

- [x] Start from failing closed-config/public-projection/resource/collision/failure-preservation tests. Use injected canned rows/guards for runner tests, never a population trial. Fixture-only limits are separately marked and cannot emit production PASS.
- [x] Implement independent task/certificate counting, existing oracle/solver identity agreement, canonical IDs and unchanged split assignment, private outcomes, source/development gating, GAC and full eligible support denominators. Pure source EQ difficulty is diagnostic only.
- [x] Implement initial/terminal manifests, all producer hashes, stable JSONL, exact per-attempt accounting, deterministic replay, typed terminal failures, resource guard and bounded final flush; fail if opaque metrics are published.
- [x] Run scoped tests, then full CPU suite under declared test allowance. Review whole diff and dependency/hash/privacy/floor closure. Freeze config/source on clean branch before any real draw.

## Task 4: controller execution and decision

- [ ] Record exact invocation/command budget and launch ONE production study under the separate watchdog; no standalone ignored launcher import path.
- [ ] Verify hashes/identities, permitted aggregates, actual costs and replay outcome without using protected rows for decisions.
- [ ] Publish a compact tracked result and update todo/execution state only for verified milestones. Preserve negative/partial outcomes. If PASS, specify next full G2 study; if failure, stop at that genuine blocker without changing floors or seed.


## Verified implementation release — 2026-10-12

Luna completed all three real modules and regression suites without a model substitution. Constructor 18 and propagation 10 scoped tests passed; the final runner suite passed 31 in 0.498 seconds. Full static review and rereview closed ten runner findings plus a failed-candidate preservation regression. Final decision: READY_FOR_CPU_STUDY in ignored .superpowers/sdd/sixvar/branch-release.md.

Fresh controller command: `C:\Users\Hank47\AppData\Local\Programs\Python\Python311\python.exe -m unittest discover -s tests -v`, run from this worktree via .superpowers/sdd/sixvar/controller_launch.py tests. Result: 347 tests passed in 40.625 seconds; wrapper 40.765 seconds, exit 0, no watchdog timeout. controller-tests-command.initial.json/controller-tests-command.json bind the pre-freeze commit, actual config and all producer/test bytes; controller-tests.txt stores the complete output. Scoped tests plus this command remain within the separate 120-second allowance. `git diff --check` passed.

Config SHA-256: 4f8b76404295828afe7b2079784fca1a2cbf2c782bfe9378f5571383ffe089e6. Final runner SHA-256: 072fec8951a4d91734cbe00d379921451a170af5777081ed0004597dbe7b0e75; runner-test SHA-256: 744bfd5d68794c56151a8d2e43c406134af809242dc1a3148e95011023541199. Component release notes bind the unchanged constructor/propagation files. Freeze these exact tested bytes before the one scientific invocation; no corpus/G2/model conclusion follows from this implementation release.

The original service interruption, import-only runner RED, intermediate fixture expectation error and all negative reviews remain preserved. The larger runner-fix batch has no preserved behavior RED; do not claim one. The final fatal-evidence preservation regression has a meaningful targeted RED and final GREEN in retry4-runner logs/meta. Local ignored evidence is not archival backup or independent reproduction.

Before any scientific draw on 2026-10-12, the clean-freeze controller check found CRLF/LF differences in four pre-existing files: dwr/tasks/audit_cli_v3.py, tests/canonical_reference.py, tests/test_canonical_equivalence.py and tests/test_task_generator_v2.py. Normalizing CRLF to LF yields their exact committed blobs; no token, scientific behavior, setting or comparison changes. Luna normalizes those files to the repository's declared LF convention, preserving old/new hashes in ignored eol-normalization.json. The first passing full-suite receipt remains intact; a second full suite binds the exact normalized bytes under a 70-second watchdog within the remaining combined 120-second test allowance. Freeze provenance is not accepted until that verification finishes.

The normalized-byte controller suite passed 347 tests in 39.921 seconds (40.062 seconds wrapper, exit 0, no watchdog timeout). Its authoritative receipts are controller-tests-lf-command.initial.json/controller-tests-lf-command.json and controller-tests-lf.txt. Both full-suite wrappers total 80.827 seconds; recorded scoped test executions add less than 10 seconds, within the combined 120-second allowance. Independent review confirmed all four normalized files equal the exact HEAD blobs and recorded old/new hashes; refreshing their index stat entries left no tracked content changes. The new constructor/propagation/runner bytes and config hash are unchanged from release. Bind the normalized-byte receipt to the final clean scientific source; preserve the original failed freeze check and first suite as historical evidence.
