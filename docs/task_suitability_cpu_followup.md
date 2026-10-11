# CPU follow-up task suitability result

Date: 2026-10-11. Status: `NO_GO_PROPOSAL` (`SOURCE_TRAIN_FLOOR_FAILED`). This is the registered CPU operational follow-up study, with identity `dwr-constraint-development-probe-cpu-followup-v1`; it is not a continuation or reclassification of the original v4 run.

The source quota closed at 1,000 accepted rows after 20,299 attempts. Replay passed and the independent audit passed. Source verification and quota bookkeeping completed. The public source-train projection reports 795 accepted groups, 493 structure signatures, and 4,499 unique oracle groups in the sensitivity and envelope-yield denominators.

Two source-train floors failed:

- Sensitivity was 893 / 4,499 = 0.1984885530, below the registered 0.25 floor.
- Accepted support-envelope yield was 795 / 4,499 = 0.1767059347, below the registered 0.25 floor.

The remaining source-train floors passed. All four registered support cells passed their minimum, and their 795 groups were 0.8902575588 of the 893 sensitive groups, above the 0.8 support-mass floor. This does not offset the two failed floors. Development had zero attempts and was not started; final composition tasks were never sampled. Preflight was false, and G2 was not assessed.

The exact invocation, source/config identities, eight run-artifact SHA-256 values, all 17 producer hashes, and controller receipt hash are recorded in [constraint-cpu-followup-nogo.json](../configs/splits/constraint-cpu-followup-nogo.json). Controller review/test evidence is bound there as well: final code review was `READY`; the final CPU suite passed 288 tests in 39.772 seconds. The root controller performed metadata/hash verification only: it did not parse task rows or perform a fresh semantic replay or audit. The original run artifacts remain local and ignored; no archival backup was verified.

The run used 951.718 seconds wall time, 950.09375 CPU seconds, and 533,737,472 peak resident bytes, then exited 1 with no error type because the terminal status was a completed scientific no-go. These measurements are below the separately registered 1,800-second engineering cap and are not production-completion estimates.

The original v4 run remains `NOT_ASSESSED_RESOURCE_CAP`. Its source prefix is byte-identical to the prefix replayed here, but that does not change its historical terminal result. The CPU follow-up changes the registered engineering cap under a new study identity; it does not change the EX hypothesis or any model comparison. This outcome is a task-construction/corpus-suitability failure, not evidence against the EX hypothesis.

Do not lower the floors, retry, or resume this study. Any next work requires a new bounded constructive task-generation design and a new registration before execution.
