# Certificate coverage diagnosis implementation plan

> For agentic workers: use superpowers:subagent-driven-development. Luna implements; the controller decides and reviews evidence.

Goal: explain the closed study's development shortfall using its permitted retained outcomes, then choose a justified prospective design.

Architecture: one small standard-library diagnostic module streams the bound existing artifact, discards opaque rows before task inspection, and writes deterministic permitted aggregates/row diagnostics. Existing generator, solver, oracle, canonicalizer and study gate are unchanged. This is a retained-data analysis, not a replacement population.

Tech stack: CPython 3.11/3.12, pathlib, JSON/SHA-256, unittest, existing schema/resource helpers.

Spec: [coverage_diagnosis.md](../../coverage_diagnosis.md); closed config: [certificate_coverage_diagnosis_v1.json](../../../configs/tasks/certificate_coverage_diagnosis_v1.json).

## Global constraints

- Exactly one retained-data invocation; 60 combined active seconds, 256 MiB sampled RSS, one worker, external watchdog65 seconds. Tests120 seconds separately.
- Bind all input artifact hashes and original scientific source/config; clean diagnosis source before invocation.
- Source-train and development-validation only; splitless construction failures separately. No opaque-row metrics, historical v1–v4 rows, final access, candidate generation, RNG, seed search or semantic experiment replay.
- Preserve original six-variable NO-GO, counts, predicates, splits and all five artifacts. No model or training.
- Fail on malformed/unknown config, identity/hash/aggregate disagreements, wrong types/nonfinite fields, output collision or resource failure. Safe public typed failures, retained partial directories.
- Stable UTF-8 serialization, bounded streaming/hashes/terminal writes; no dependencies or mandatory Bash.

## Review focus

- Protected rows with poisoned task/metric payload must be routed away before parsing/aggregation/error text.
- Duplicate and ineligible rows must stay outside unique eligible denominators; envelope failures must remain inside eligible support/difficulty denominators.
- Missing-family zero counts and feature histograms must use the exact documented populations, rather than accidentally only usable rows.
- Late guard/I/O failures must revoke PASS_DIAGNOSIS and preserve evidence without large post-cap rehashes.
- Input identity/config/hash changes and previous output directories must fail before analysis; fixture tests must not access real outcomes or emit scientific gate success.

## Task 1: retained-data reader and descriptive analyzer

Files: create `dwr/tasks/certificate_diagnosis.py`, `tests/test_certificate_diagnosis.py`. Do not edit historical scientific producers.

Interface: `python -m dwr.tasks.certificate_diagnosis --config configs/tasks/certificate_coverage_diagnosis_v1.json --output runs/certificate-coverage-diagnosis-<UTC>` accepts only the exact registration. Internal helpers may accept canned streams for tests; production artifact identity remains closed and there is no fixture argument on the CLI.

- [x] Write failing tests first and save command/output/hash metadata in ignored `.superpowers/sdd/coverage/`. Test protected poison, ordered stage counts, duplicate/eligibility/envelope populations, exact graph features on handcrafted EQ/NEQ cases, empty profiles, public aggregate mismatch, config/hash/output refusal and counterfactual arithmetic.
- [x] Implement typed row routing, counts and feature aggregation without constructor/solver/oracle/canonicalizer calls or RNG. Graph components include all six variables and anchored status; private task objects are not retained across rows.
- [x] Implement independent source/config/input identities, initial/terminal manifests, streamed permitted-row projections, guarded stable hashing/writes and safe typed partial failures. Include source/runtime/command and full task-producer hashes; report G2 unassessed and original NO-GO.
- [x] Run `python -m unittest discover -s tests -p test_certificate_diagnosis.py -v` with durable RED/GREEN evidence. Only canned fixtures; no real retained-data invocation before controller freeze.

Verification on 2026-10-12: Luna's final scoped suite passed 22 tests in 0.120 seconds. Controller full suite `python -m unittest discover -s tests -v` passed 370 tests in 38.495 seconds (wrapper 38.641 seconds, exit 0, no timeout). The preceding 368-test pass in 39.410 seconds is preserved; the second run followed the concrete setup-order fix and new terminal-boundary regressions. Both wrapper totals plus focused checks remain within the separate 120-second test allowance. Evidence: ignored `.superpowers/sdd/coverage/task1-implementation-evidence.md`, `controller-final-tests-command.json`, `controller-final-tests.txt`, and `diagnosis-release-r3.md` (`READY_FOR_RETAINED_DIAGNOSIS`). These are CPU implementation checks, not a retained diagnosis, G2 or model validation. No historical outcome was analyzed before registration/freeze.

Pre-freeze byte check found CRLF in the new module while Git's staged blob was LF. Controller verified exact equivalence after only CRLF-to-LF conversion and wrote the staged blob back before freezing; no executable content changed. The resulting module SHA-256 is `f7cf96bb63fdf0555908b170a3cc81c61769785aa4b0c21ee5c5082372e988c1`. All other producer/test bytes already matched. The final-byte scoped command `python -m unittest discover -s tests -p test_certificate_diagnosis.py -v` passed 22 tests in 0.114 seconds. Evidence: `controller-eol-normalization.json` and `controller-eol-scoped.txt`; the full-suite receipt honestly retains its pre-normalization byte identity. This follows the earlier Windows line-ending failure shield without rerunning an unchanged full scientific suite.

## Task 2: review, verification and sole diagnosis

- [x] Review spec/implementation/config/privacy/denominator/resource closure with a fresh reviewer; resolve concrete findings with Luna and regression checks.
- [x] Controller runs the full CPU suite under the separate test allowance; verify exact file/config identities, freeze clean source and record the one invocation.
- [x] Execute the bound retained-data diagnostic under the watchdog; verify output/input hashes, actual costs and permitted aggregate linkage without reading protected tasks.
- [x] Save a small tracked diagnosis/result note and decision. Distinguish observed associations, conditional arithmetic, formal guarantees and unknowns; no independent reproduction/backup claim from local artifacts.

Sole invocation: clean `3e16ca31abf843bbf76e45e5890bba8845028eb1`, output `runs/certificate-coverage-diagnosis-20261012T013816950316Z`, PASS_DIAGNOSIS, 0.688 seconds internal wall / 0.609375 CPU / 24,981,504 sampled RSS bytes; controller wrapper0.781 seconds, exit0, no timeout. All four output/five input hashes verified. See [result](../../coverage_diagnosis_result.md) and [compact manifest](../../../configs/splits/certificate-coverage-diagnosis-result.json). Original scientific NO-GO unchanged.

## Task 3: prospective research decision

- [x] Explicitly choose universal family-envelope coverage or the supported eligible-envelope estimand, with inherited-contract rationale and narrower claim if changed. Preserve the old study's strict representation failure.
- [x] Use the capacity and pooled-difficulty diagnosis jointly to select one mechanism and bounded analytical/fixture question. Register any next diagnostic or population before code/execution, with fixed sample/seed/cost/stop settings and untouched test/final boundaries.
- [ ] Continue only along a justified reviewed path. A passing new capacity/difficulty study would still lead to separate complete G2 qualification before fixed-N EX/IDX implementation; no immediate training.

The owner's continuous-work delegation covers controller decisions/reviews; it does not reopen the closed six-variable population or permit an outcome-selected retry. Record any ruling and its scientific cost before proceeding.
