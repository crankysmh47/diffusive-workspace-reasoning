# Retained rank support diagnosis implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development task by task, with independent reviews. Root owns scientific decisions, freeze/invocation and reporting; Luna implements. Owner authorized autonomous continuation and routine design decisions.

**Goal:** Explain the closed qualification's support mismatch from permitted retained annotations without generating tasks or selecting another envelope.

**Architecture:** One thin separate standard-library diagnostic reuses existing reviewed JSON/resource/hash/raw-Git/safe-write helpers and streams split-routed metadata. Old scientific producers, gates and artifacts remain untouched.

**Tech Stack:** Python, pathlib, UTF-8 stable JSON/JSONL, SHA-256, unittest, Windows-safe subprocess/resource checks.

**Spec:** `docs/rank_support_diagnosis.md`; closed config `configs/tasks/certificate_rank_support_diagnosis_v1.json`, SHA256 `63145f650d17dc5bd205bfd1dc0db38b7c2b68dadc8ea445ddf8805e296bd394`.

## Global constraints

- Input qualification41bd784,3600source/4800development/final0, NO_GO_DEVELOPMENT/PASS_REPLAY8400; exact six input artifact hashes/config/source/command linkage.
- Only source_train/development_validation; route before scientific metadata, never parse tasks/witnesses/private certificates/cores or protected semantics. No protected counts, resurrection of protected-first duplicate content or raw records parsing.
- No generation/RNG/solver/oracle/GAC/canonicalization/candidate replay, no new cutoffs/selected support search. Complete fixed4x5x7 grid and coarser5x7 contingency, all-E denominators, no G2/model PASS.
- Sole path `runs/certificate-rank-support-diagnosis-20261014-v1`,60s/256MiB/oneworker/watchdog65,zero retries. Canned full tests120s; clean raw-Git producer source before real analysis.
- Preserve negative/inconclusive evidence and original PR record; owner mergedPR#1 externally. No local main edits/agent merge/visibility/license change.

## Review focus

- Protected poison rows cannot contribute features, aggregates, skipped counts or public error values.
- Global first-seen annotations are authoritative; do not recalculate local dedup and resurrect protected-first content.
- Source E requires sensitivity as well as eligibility; U/E/A and signature sets remain distinct; nontrivial E may fall outside A.
- Fixed strata are derived from cell metadata, not task parsing; unknown shapes/bool-as-integer and annotation/public-count mismatches fail closed.
- Late write/resource failure revokes diagnostic success, clears stale hashes and preserves partial evidence without post-cap large rehash.

## Task1: metadata analysis and strict bounded runner

Files create `dwr/tasks/certificate_rank_support_diagnosis.py`, `tests/test_certificate_rank_support_diagnosis.py`. No other code/config edits without a root ruling.

Interfaces: `analyze_rows(rows, *, projection_stream=None, check_budget=None) -> dict`; `compare_public(report, analysis) -> None`; `run_diagnosis(config_path, output_path, *, argv=None, _guard=None) -> dict`; CLI `python -m dwr.tasks.certificate_rank_support_diagnosis --config ... --output ...`. Private test guard marks fixture-only and cannot certify production success. Root may settle helper names when existing interfaces are clearer; record departures before real execution.

- [x] Write meaningful canned RED tests: bin boundaries, U/E/A/signatures, branch/cell/stall contingencies/zero bins, protected poison/global first-seen routing, scalar-type errors, public mismatch, config/output/input admission and late revocation.
- [x] Implement thin reusable analysis and runner; no task parsing or calls. Initial identity manifest before analysis, exact8400 ordered prefix checks, permitted-only projection, bound report linkage, guarded hashes/writes/resources and explicit terminal claim/cost boundaries.
- [x] Run scoped unittest and diff check; save self-review/RED/GREEN/hash evidence under ignored `.superpowers/sdd/rank-support-diagnosis-20261014/`; normalize LF. No whole-suite/production call by implementer; no root/old producer edits or source commit.
- [x] Independent task/release review, root scoped fixes as needed, full stable-byte CPU suite once, source/config/links verified.

## Task2: one registered diagnosis and evidence-based design decision

- [ ] Root freeze clean verified source and execute exactly one registered CLI with external watchdog65. Do not retry or reset budgets; preserve every terminal.
- [ ] Verify allowed report/opaque hashes and input/source binding; record result/compact manifest and update live todo/execution/README/history.
- [ ] Use complete unselected bins to choose a justified prospective next design or record substantive impossibility/insufficient support. No new population or model follows without a separately reviewed registration; do not relabel original NO-GO.
- [ ] Push/create the next owner-requested checkpoint PR to current remote main without merging. Preserve local ignored artifacts and evidence; do not claim archival backup or independent reproduction.

## Controller rulings

2026-10-14: Routine design/implementation authorization persists from owner delegation and resume. Selected diagnosis rather than another seed or post-result envelope change. It costs one retained-data analysis; if overlap is poor it may justify rejecting the current composition-only task comparison, not another tune-until-PASS cycle. No raw permitted data were read at registration creation. Read-only scientific review agrees the existing Sig(A)/ALL-E support contract is coherent; implementation feasibility recommends reuse of pure helpers without old-module edits.

2026-10-14 release corrections: independent review found an unowned-output revocation bug, missing original Git proof, missing derived stall comparison and incomplete output boundaries. The scoped wave closes the first three; rereview caught an undefined input_config at the report-boundary insertion. Ruling: explicitly bind the already validated original configuration and add a fake-metadata runner regression that reaches final reporting before any retained-data invocation. This is a dependent correctness fix, not another diagnostic/population allowance or gate change. Preserve all RED/negative reviews and documented implementation departures.

2026-10-14 implementation verified: root493 CPU tests passed43.653s (wrapper43.828), tracked bytes stable; final scoped review READY_FOR_SINGLE_DIAGNOSIS. All41 runtime producers match staged raw Git bytes/LF. Config unchanged. See diagnosis implementation verification for exact log/hash and evidence limits; actual analysis remains unattempted until clean freeze.
