# V3 task suitability: one feasibility attempt

Date: 2026-10-08. Status: **NO-GO at the registered source-support capacity bound; G2 not assessed.** This is one agent-run CPU feasibility attempt and its independent audit. It reports no model behavior or transfer result.

## Frozen identities and commands

The run used clean source commit `57c953f2f01841be3391e4027e7c8514f59f14e9`, config `configs/tasks/constraint_v3.json` SHA-256 `e5d7de77fc576b56847e7a55e401cd6c318f4bc35f5aa7f843ef41d1af9ee025`, seed `20261007`, and CPython 3.11.9 on 64-bit Windows 10 (build 26200). It used one worker, a 30,000-attempt-per-partition cap, and one combined generation-plus-audit budget of 600 active seconds and 1 GiB resident memory.

```powershell
python -m dwr.tasks.cli_v3 --config configs/tasks/constraint_v3.json --mode feasibility --out data/constraint-v3-feasibility-20261008T173012Z
python -m dwr.tasks.audit_cli_v3 --config configs/tasks/constraint_v3.json --mode feasibility --data data/constraint-v3-feasibility-20261008T173012Z
```

Generation started at `2026-10-08T17:30:12.682158Z`, exited 1 at `17:31:42.541935Z`, and recorded the expected `ProposalNoGo` reason `source_support_capacity_bound`. The independent audit started immediately and exited 0 at `17:34:03.746514Z`. It completed with integrity and source-prefix replay passing. The report's replay status is `partial` because this feasibility run deliberately stopped before target sampling; it is not an incomplete or failed source closure. The proposal status is `NO_GO`; G2 is `not_assessed`.

## Public feasibility evidence

The source quota closed at 1,000 accepted records: 334 `eq`, 333 `neq`, and 333 `add_mod`, after 13,262 attempts. Public source-train metrics cover 807 accepted groups, 440 structure signatures, and 1,361 unique source-train groups in the sensitivity denominator. The sensitivity fraction was 807/1,361 = 0.5929463629684056. These source-train counts are not the total source quota.

| Measure | Observed result |
| --- | ---: |
| Source quota | 1,000 / 1,000; closed |
| Source-train accepted groups | 807 |
| Source-train structure signatures | 440 |
| Source-train unique groups in sensitivity denominator | 1,361 |
| Constraint-sensitive source-train groups | 807 / 1,361 = 0.5929463629684056 |
| Source-train primitive coverage | `eq`, `neq`, `add_mod`; pass |
| Source-train solution-value fractions (0, 1, 2) | 0.334250, 0.332875, 0.332875 |
| Source-train given counts | 1, 2, 3, 4 |
| Development-validation attempts | 0; not attempted |
| Final attempts | 0; not attempted |

The replay-validated top ten source-train support cells contain 321 of 807 groups, a mass of 0.39776951672862454. The fixed feasibility quota is exactly 100 eligible development-validation groups for each of three profiles. With the registered minimum of ten groups per common-support cell, no more than ten cells can contribute to each cohort's common set. Therefore the source top-ten mass is a necessary upper bound for the four-cohort feasibility proxy. Since 0.39776951672862454 is below the required 0.8, this registered proposal is mathematically unable to pass its proxy for this source sample. The check is a source capacity bound, not an observation of actual source-to-target support.

Development and final groups were never sampled. The public `four_cohort_common_support_proxy` is null, and the audit marks unattempted targets and G2 unassessed. A nested legacy prefix diagnostic reports `measured_fail` with empty target placeholders; this is not measured target yield, target mismatch, or an actual proxy result. No candidate-level held-out aggregates were computed or used here. Final-target candidates and metrics were not accessed.

## Cost, checks, and evidence limits

The authoritative `audit_manifest.json` records 230.89 seconds combined active wall time, 229.78125 seconds combined CPU time, and 134,852,608 bytes combined peak resident memory. Generation recorded 89.781 seconds wall / 89.375 CPU seconds; audit recorded 141.109 seconds wall / 140.40625 CPU seconds. The combined budget was not reset between generation and audit/replay. Accounting ends at the post-first-terminal-manifest-write resource check; one bounded final metadata flush is excluded. The run remained within the 600-second and 1-GiB limits.

The controller's fresh CPU suite at the frozen run source commit reported 209 passed in 9.720 seconds with `python -m unittest discover -s tests -v`. These are agent-run implementation checks. The separate audit is corpus integrity and replay evidence; neither is model validation or independent reproduction. No archival backup or independent reproduction exists. Source sensitivity does not measure learned difficulty, and no novelty claim is made.

The test-driven review history has one evidence limitation: the raw Task 4 fix-round-1 RED log for finding F2 was lost and was not reconstructed. Raw F5, fix-round-2, and fix-round-3 RED logs were retained. F1 did not establish prior reporting contamination: the shared audit helper already filtered primary-eligible names; v3 added and checked an explicit caller projection. Other review/test details and the honest log provenance are in the ignored [v3 Task 4 rereview](../.superpowers/sdd/transfer-v3/task-4-rereview3.md) and [progress ledger](../.superpowers/sdd/transfer-v3/progress.md). Those CPU review checks do not change the run's source identity or establish a scientific result.

Artifact sizes and SHA-256 values are in the [compact feasibility evidence manifest](../configs/splits/constraint-v3-feasibility-nogo.json). The ignored run directory is `data/constraint-v3-feasibility-20261008T173012Z/`. Preserve this NO-GO. Do not retry or retune this proposal. Any next task-generation hypothesis needs a separately versioned prospective design and review. A complete passing five-cohort G2 audit is still required before fixed-size EX/IDX implementation; training requires its separate complete preregistration.

## Post-review code correction — 2026-10-09

The whole-branch review found two integration defects in the conditional full pathway and two included minor reporting/maintenance issues. The code now validates and preserves genuine v3 consumed-preflight linkage before passing a v2-shaped private projection to shared legacy accounting checks; rejects independently rehashed preflight reports whose combined wall time reaches/exceeds the registered cap or whose combined peak memory exceeds the cap; uses the shared pure `_cell` extractor for the independent source-capacity histogram; and marks final assessment `not_assessed` for incomplete full generation. Complete full audits may still mark it assessed; feasibility never accesses final candidates.

Focused unit/integration verification on 2026-10-09 passed: 100 affected v3/v2 tests in 8.005 seconds, including the real synthetic full-manifest validator, actual CLI-to-consumer preflight handoff, independent rehashed wall/memory mutations and inclusive memory boundary, cell-bin boundaries, and an incomplete-full report. The broader final branch suite is pending with the root controller. These corrected-code checks do not revise the historical 209-test result at source `57c953f`, change that source identity, touch the compact manifest or its 16 artifact hashes, or constitute a data-generation/replay/audit/model experiment. The preserved scientific result remains the same source-sample NO-GO with G2 `not_assessed`.
