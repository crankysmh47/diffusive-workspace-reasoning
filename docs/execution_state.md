# Execution state: transfer v1

Updated: 2026-10-08. Branch: `research/transfer-cpu-v1`. Worktree: `C:\sem5\diffusive-workspace-reasoning\.worktrees\transfer-cpu-v1`. The implementation and the bounded data attempt were performed on the clean source commit `800f9cdb0d44c2b0ca54fd54039c5383821a3c2b`. This document records the state at that commit; its own documentation commit follows it.

## Verified state

- G1 passed: frozen grammar/task contract, strict parser/verifier, independent exhaustive tiny-instance oracle, deterministic solver, generator and audit implementation are present. The controller's fresh check at source commit `800f9cd` ran `python -m unittest discover -s tests -v`: 69 passed in 0.815 seconds. This is CPU correctness evidence, not model validation.
- The only registered corpus run is `data/constraint-v1-20261007220128`. It is incomplete after a Windows manifest replacement `PermissionError [WinError 5]`; source generation stopped after 22,450 attempts and 185 accepted examples, below the 30,000 cap. Development and final generation are 0 / `not_attempted`.
- Independent audit validated 185 records and replayed 22,450 candidate outcomes. Record integrity and replay passed. Structural floors did not pass; the report status is `incomplete_generation`, G2 is `not_assessed`, and common support is a measured failure on the available prefix.
- Full resource accounting: generation 17.891 s wall / 17.640625 s CPU / 379,232,256 peak bytes; audit 19.875 s wall / 19.640625 s CPU / 433,459,200 peak bytes; combined wall 37.766 s of the single 600 s cap.
- The bounded temporary-file probe confirms an incompatible Windows sharing handle can cause the atomic replacement failure. The real run's lock owner remains unknown; the coincident controller read is not proven causal.
- No model code, ML dependencies, GPU work, training, or transfer benchmark was performed. The local ignored corpus is not archived; no independent reproduction is claimed.

Evidence details and all partial-data artifact hashes are in [task_suitability.md](task_suitability.md) and [the compact split evidence manifest](../configs/splits/constraint-v1-partial.json). Exact task command interfaces are `python -m dwr.tasks.generator --config configs/tasks/constraint_v1.json --out <new-output-dir>` and `python -m dwr.tasks.audit --config configs/tasks/constraint_v1.json --data <generated-output-dir>`. Do not rerun against the existing output directory or alter the frozen scientific config as an operational workaround.

## Scientific decision

The main exchangeable workspace transfer question remains active. Fixed-N EX/IDX implementation is conditional on G2, which is not assessed and not passed. The current corpus cannot establish zero target yield, validate the whole held-out benchmark, or show that the attempt cap was reached. The source sample does establish specific blockers: source validation/test groups are under their floors, 51 observed source signatures have no constraint-sensitive example, and the final-target equality-triangle signature occurs three times with every edge removal leaving one solution. Mixed development/final groups and common support are absent.

Next, create a versioned **development-only design task** for the measured sensitivity/common-support blockers and for Windows progress reading and atomic replacement. Define acceptance behavior prospectively, retain the current thresholds and final-target definitions until that design review, and do not tune on final-target outcomes. This is a planning/human scientific decision point, not approval to generate again. Only after the revised development contract is frozen may a separately versioned run be considered; EX/IDX remains blocked until a complete independent G2 pass.

## Durable ledger decisions

| Decision | Reason and trade-off |
| --- | --- |
| Use an isolated linked worktree for the authorized implementation and documentation. | Preserves the original main checkout; cost is that the user reviews a second checkout. |
| Use native PowerShell and Python for ledger/review operations. | Meets Windows support requirements; cost is losing helper-specific Bash conveniences. |
| Require mixed targets to depend on every present primitive kind. | Preserves the transferable-computation question; cost is lower acceptance yield, which may expose a real data gate. |
| Use only primary-eligible target cohorts for common support; keep diagnostic cohorts reporting-only; skip full profiles after sampling the profile. | Prevents invalid overlap denominators and quota overshoot; cost is fewer accepted groups under the fixed cap. |
| Reject solver partial assignments only when a formula is fully grounded and false; detect contradictory grounded givens before DFS. | Keeps node difficulty reproducible rather than dependent on arbitrary lookahead; cost is less pruning for the bounded `n<=6` task. |
| Split Task 4 into canonical/candidate work (4a), bounded orchestration (4b), and independent audit (4c). | Narrows review surfaces; cost is one extra task review. The corpus run waited until all three were reviewed. |
| Classify JSON literal `NaN`/infinities as `non_finite_constant`; classify syntactically finite exponent reals such as `1e400` as `non_integer_value`. | Preserves the integer-only grammar and diagnostic precedence; cost is a diagnostic-label distinction, with no non-finite value accepted. |
| Use one 600-second active-command cap across generation plus audit/replay; audit inherits the remaining time. | Avoids doubling the allowance across two CLIs; cost is less replay time after expensive generation. |
| Keep the main exchangeable transfer question and make fixed-N EX/IDX conditional on a complete passing G2 audit. | Preserves the selected research direction while preventing architecture work on unqualified data; cost is delayed model work. |
| Preserve the failed partial corpus and do not retry under changed rules. | Keeps interrupted and negative evidence visible and prevents outcome-driven tuning; cost is that the current benchmark cannot proceed. |
| Register only one short temporary Windows sharing-lock probe. | Tests a narrow I/O mechanism without a scientific rerun; it cannot identify the real run's lock owner, so causality remains unresolved. |

The authoritative implementation ledger remains in the ignored `.superpowers/sdd/transfer-v1/progress.md`; this table carries its design rulings, rationale and trade-offs into tracked project documentation.
The authoritative implementation ledger remains in the ignored `.superpowers/sdd/transfer-v1/progress.md`; this table carries the relevant rationale and costs into tracked project documentation.
