# Bounded CPU performance diagnosis

Date: 2026-10-10. Authorized by the owner's request to resume the proposed diagnosis and operational amendment, using Luna for implementation and the controller for decisions.

## Question and limits

Determine which deterministic computations dominate the existing six-variable source pipeline, before choosing an optimization or a future verification budget. The v4 result remains `NOT_ASSESSED_RESOURCE_CAP`; its artifacts, registration, quotas, splits, scientific floors and resource result remain unchanged. This is an engineering fixture study, not a new corpus, development proposal or G2 assessment.

Use exactly the twelve existing handcrafted source fixtures in `tests/test_probe_driver_v4.py`: profiles in EQ, NEQ, ADD order, and cells A, B, C, D. No production candidate sampling, prior raw outcome reads, development/final tasks, model selection or training. Execute one baseline measurement and at most one post-change measurement, each with one pass per stage, one worker, a 180-second combined wall cap and 1-GiB peak resident cap. Correctness tests have a separate 180-second suite cap; a performance comparison is descriptive, not a statistical speedup claim.

Stages: construct independent fixture proposals; canonical content/structure identities; consume the fixed twelve-source prefix; replay the prefix independently; audit the prefix independently; and stable JSONL write/read/hash. Record wall and CPU time per stage plus cProfile function counts/times. Record the fixture corpus hash, producer/test-helper hashes, source commit/dirty state, exact command, CPython/platform identity and actual peak resident memory in a resolved manifest before execution. Store results under unique ignored `runs/cpu-performance-*` directories. Preserve failures and partial profiling output.

The synthetic provider preserves the driver and audit paths but replaces random candidate construction. It cannot reconstruct the original v4 phase timings or establish production yield/capacity. Report this limitation explicitly. No completion or gate status is fabricated from fixtures.

## Decision rule

After the baseline, choose at most one local exact-equivalent optimization justified by the measured hotspot. Prefer pure canonicalization/serialization work over weakening any oracle, replay, audit or acceptance check. Require byte-identical identities and fixture outcomes/projections against the existing implementation, including renamed variables, constraint/given order, all relation kinds and tiny exhaustive cases. Retain an independent legacy implementation in tests for equivalence. Do not reuse generated proposal answers to certify replay or independent oracle facts.

After verification and the post-change measurement, prospectively register one follow-up source/development feasibility study with finite resources, unchanged scientific thresholds and leakage rules. Its results cannot replace v4. If measurements do not justify a tractable follow-up, record a task-construction decision rather than allocate unbounded retries. Full-corpus schema/linkage, five-cohort G2, model and training gates remain binding.

## Execution tasks

1. Luna: implement the ignored profiling harness and run the sole baseline; controller checks evidence and decides the optimization.
2. Luna: implement the selected bounded optimization and meaningful equivalence tests; controller reviews and runs verification; one post-change measurement.
3. Controller: write the prospective operational amendment and explicit study contract; Luna implements only its reviewed minimal execution path, with failure preservation and provenance tests.
4. Run the single new bounded attempt, preserve its outcome, and continue only through satisfied gates. A scientific failure or exhausted registered diagnostic allowance is the next research decision boundary.

The implementation helper remains ignored because it imports private test fixtures; production code belongs under `dwr/`. Existing scientific producers may change only after the byte-equivalence checks and prospective note, with changed producer hashes bound in every future manifest. Historical source identities are never relabeled.

## Baseline decision — 2026-10-10, before optimization

The sole fixed-fixture baseline at clean source `b363e02ca26dc945a0f43fc8353c675698df139a` took 2.500 seconds (cProfile enabled). Isolated canonicalization took 0.438 seconds; consume/replay/audit took 0.523/0.522/0.922 seconds, with canonicalization prominent in each. The isolated twelve-task identity stage called JSON serialization 17,280 times. Evidence is ignored `runs/cpu-performance-baseline-20261009T201932Z/`; this is source-fixture evidence only, not original-run phase attribution or production-scale timing.

Selected change: keep exhaustive enumeration of every variable permutation, but rank normalized immutable tuples and serialize only the winning permutation. For the supported schema, variable names are exactly `v0` through `v5`, given values are 0/1/2, and list lengths are fixed within one task's permutation orbit. Content JSON compares constraints first, then givens (given object keys compare value before variable); structure JSON compares anchors first, then constraints. Tuple ranking must follow precisely this order. No pruning, caching of proposal/oracle answers, RNG changes, relaxed grouping, schema or hash-domain changes are authorized.

Equivalence requires all normalized constraint subsets of sizes 2 through 8 on three variables, every legal single given location/value, fixed fixtures spanning six-variable relation/cell combinations, reorderings/operand symmetries, and supported-size checks against the original exhaustive JSON implementation. The original implementation is retained in a test helper. Outcome, record, identity, denominator and audit projection hashes from the sole post-change fixture measurement must match the baseline. A mismatch blocks production work.
