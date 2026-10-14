# Five-stratum study implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development task by task. Luna implements; root makes scientific decisions, verifies/freeze source, runs the sole registered invocations and opens checkpoint PRs. Owner delegated routine design and continuous execution.

**Goal:** Test the new fixed five-stratum population on fresh seeds while preserving all previous negative studies and exact legacy-config behavior.

**Architecture:** Small explicit support-policy module, configuration-aware rank audit/runner extension, existing unchanged rank constructor. No production monkeypatching, rewritten metrics or placeholder framework.

**Tech stack:** Python standard library, pathlib/UTF-8/SHA-256, unittest, Windows-safe guards and Git blob identities.

**Spec:** `docs/five_stratum_study_design.md`. Closed qualification SHA256 `d66ad33da9111287223e7e94ab849e8b710692db32f71f94410f305ce1cf1002`; full `3aff6fcaba6143a26ad6d8a86cc83f9782ff63d689912f5fb04ffc4e9cc81afa`.

## Global constraints

- Old study/config/artifacts unchanged. Constructor/RNG/branch law/canonicalization/solver/oracle/GAC/splits unchanged. Five-support annotations are outside model inputs and retain real node/byte counts.
- A2/A3/B2/C2/D2 only; all-E mass>=.8, EVERY fine support cell>=10 in each required cohort. Original node2 source primitive/cell floors10 remain; A3 must contain each source primitive. Core/difficulty use E before envelope.
- Qualification root20261016 source3600/development4800/final0,1800s/1GiB/oneworker/watchdog1805. Full root20261017 source3000A5/development600A5 per pair/final600A5,30000 raw maximum per partition,3600s/1GiB/oneworker/watchdog3605. Zero retries.
- Exact sole paths in configs; qualification failure blocks full; source/development full stage gates before final. No final production draws/unit populations or models/training/dependencies.
- Legacy configurations keep their original support/schema/path/linkage behavior; hashes remain closed. Full requires exact passing NEW qualification/config/producers/replay/artifacts/nonempty floor maps, never an old result.
- Canned full CPU suite120s; clean committed raw-Git/LF producers before any real invocation. Preserve all failures and implementation departures.

## Review focus

- New A3 can never be pooled with A2 to satisfy its fine-cell floor or counted by rewriting first_solution_nodes.
- Legacy source primitive capacities are measured on node2, not silently boosted by new A3; source primitive presence in A3 is separate.
- Protected qualification rows route before fine-support choice annotation/core/histograms; duplicate raw facts cannot increment accepted quotas.
- New support metadata does not alter prompt hash, witness/answer encoding, independent oracle/deletion/GAC checks or replay identity.
- New config/path/qualification linkage cannot vacuously pass empty floor maps, reuse old PASS/status, overwrite existing paths or certify mocks as production.

## Task1: explicit support policy and audit

Own create `dwr/tasks/certificate_support.py`, modify `dwr/tasks/certificate_rank_audit.py`, create `tests/test_certificate_five_support.py`. Do not edit runner, generator, old tests, configs/docs or artifacts.

Interfaces: `support_stratum(row, config) -> str | None` (legacy A/B/C/D or new A2/A3/B2/C2/D2); `is_envelope(row, config) -> bool` uses real metadata plus strict existing flag; `support_counts(rows, config) -> dict`. Legacy no-scheme config uses unchanged fixed4/node2; five scheme validates exact frozen descriptors and refuses unknown policy. Audit gates consume config explicitly; retain old default helper behavior where needed. Root settles minor API names before the dependent task; no generic configurable-search system.

- [ ] Meaningful canned RED for A3 membership, exact boundary nodes7/8/15/16, all-E denominators, per-fine-cell floor failure despite pooled-A capacity, source node2 primitive floors, A3 primitive presence, full stage/final support, nonzero diagnostic fine strata, and legacy behavior.
- [ ] Implement policy and config-aware gates with unchanged old-config return/schema behavior. Extra fine counts/metadata are versioned new-only. Required full support cohorts are source-ID-test, each3 development-validation pairs and final; new diagnostic-test per-composition fine nonzero check is separately labelled.
- [ ] Scoped tests, self-review/hashes/LF/diff check, no full suite or source commit by implementer. No task construction/solver calls or raw artifact access; fake metadata only.
- [ ] Independent task review and scoped fixes. Record stable public interfaces for Task2.

## Task2: strict new runner/admission and fixtures

Own modify `dwr/tasks/certificate_rank_study.py`, create `tests/test_certificate_five_study.py`. Read Task1 interface receipt; do not modify its files, constructor, old tests or configurations without root ruling.

Interfaces: extend `load_config`/registered-path and output admission to the two exact new trees alongside untouched legacy trees; `run_study` uses the shared configuration-aware policy/audit. Old call signatures/defaults remain where possible. New full links only to the new qualification. Source `_build_row` assigns fine-support annotation after protected routing and independently checked metrics; `_a_projection` includes new-only metadata; `_is_A` governs records/quotas/replay consistently. CLI unchanged.

- [ ] RED canned strict config/admission/linkage cases, mismatch with old qualification, protected route, unchanged constructor delegation, new A3/duplicate quotas, answer-free prompt projection, replay/late cap and rejected-path preservation.
- [ ] Implement small explicit version/mode registry and guard new fields while retaining legacy behavior. New full success status is `PASS_G2_FIVE_STRATUM_V1` with g2_status=pass; qualification status remains PASS_QUALIFICATION with exact new config identity. Late resource/integrity failure revokes every success.
- [ ] Canned temp-path mocked runner tests reach report/manifest/qualification-to-full interface without real candidate/final draws; private injection cannot produce scientific PASS. Keep typed outcomes and independent scalar/oracle/solver/verifier checks intact.
- [ ] Scoped tests/self-review/diff/hashes; independent task review plus scoped whole-branch release review. Root runs full stable-byte CPU suite once after code stops changing; verify closed hashes, raw Git bytes and doc links.

## Task3: registered fresh replication and terminal decision

- [ ] Freeze verified clean source and execute ONE new qualification under1805 watchdog. Public choices only from source_train/development_validation; preserve all raw/permitted/protected integrity rows and exact replay.
- [ ] Verify actual terminal public reports/artifact/source/command/config/count/resource identities; record result and checkpoint PR without merging. A scientific NO-GO is a terminal valid result, not permission for another seed.
- [ ] Only new PASS: execute the predeclared full once under3605 watchdog, with exact linkage and prior source/development stage gates plus final structural access log.
- [ ] Record full five-cohort G2 PASS or retained NO-GO/operationally unassessed; no automatic retries or support adjustments. Only G2 PASS permits a separately concrete fixed-N infrastructure plan; model training remains unregistered until its other prerequisites.

## Controller rulings

2026-10-14: The complete predeclared diagnostic inventory, reviewed scientific rationale and fresh replication distinguish this from retroactively accepting an enlarged four-cell envelope. A3 is an explicit fifth stratum, not a global high-node pooling exception. Source full quota3000 is frozen for prospective ID-test scarcity risk, not after protected counts. Equal1/5 evaluation weights change given-count mixture and are explicit; source primitive balance/exact within-bin matching are not claimed. Preserve old negative evidence and the current original local main edits. Owner merged PR#1; new checkpoint PRs target remote main and are never auto-merged.
