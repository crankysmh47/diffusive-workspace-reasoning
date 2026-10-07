# Task 1 report

Status: DONE — CPU task contract and generation/audit configuration are specified. No production modules, tests, datasets, training, or TODO completion flags were changed.

Files:

- `docs/task_spec.md` — immutable API records; grammar and strict task/answer serialization; verifier reasons; independent solver/oracle contract; canonical IDs/signatures; seeded generation; split and suitability rules; module map and next correctness checks.
- `configs/tasks/constraint_v1.json` — versioned seed, grammar, source/development/final composition partition, RNG/canonicalization identity, generation target and caps, audit thresholds and artifact policy.

Exact decisions: domain `{0,1,2}` with `eq`, `neq`, and `add_mod(a,b,c): (a+b) % 3 == c`; `n=3..6`; 2–8 distinct constraints; 1–`n-2` givens. Source has one relation kind, development targets have each two-kind combination, and final targets contain all three. Content IDs preserve given values; structure signatures omit values but preserve anchor positions and relation structure. Variable renaming, constraint ordering, and only true operand symmetries are canonicalized; value relabeling is not assumed.

The audit targets 300 unique groups per partition, with at most 30,000 candidates per partition, 600 seconds, one worker, and 1 GiB. Its thresholds are structural suitability floors only. Passing G2 does not provide power, an EX–IDX practical effect, or authorization for confirmatory training. No `docs/decisions/task-v1.md` was needed because this freezes and operationalizes the selected direction without a deliberate departure.

Checks: Python 3 parsed `configs/tasks/constraint_v1.json`; asserted the configured 300 accepted-group target and 30,000 candidate cap; staged `git diff --cached --check` was clean. No tests or generation were run because implementation is explicitly out of scope for Task 1.

Ambiguities resolved: “ternary add_mod” means two commutative addends and one distinguished result variable; source ID splits group exact canonical content and lineage by a stable SHA-256 bucket; development-composition data can support choices and diagnostics, while final-composition data is audit-only until frozen final evaluation. The structural audit may pass while confirmatory precision remains unresolved.
