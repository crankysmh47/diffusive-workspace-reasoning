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
