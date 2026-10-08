# V2 task suitability: one feasibility attempt

Date: 2026-10-08. Status: **NO-GO at the registered feasibility attempt cap; G2 not assessed.** This records the real agent-run CPU attempt. It does not establish model behavior or target transfer.

## Frozen identities and commands

The attempt used clean source commit `9f08efddb60537c61a89a90325bc9c832c32ca1c` and `configs/tasks/constraint_v2.json` SHA-256 `cc9fb0a827d27e4f00425ca9364365026bb6c70c75dc8d6dc0eebcc1f6865222`. Runtime was CPython 3.11.9 on 64-bit Windows, seed `20261007`, one worker, one invocation, a 30,000-candidate cap per partition, and combined generation-plus-audit caps of 600 seconds and 1 GiB resident memory.

```powershell
python -m dwr.tasks.cli_v2 --config configs/tasks/constraint_v2.json --mode feasibility --out data/constraint-v2-feasibility-20261008T062621Z
python -m dwr.tasks.audit_cli_v2 --config configs/tasks/constraint_v2.json --mode feasibility --data data/constraint-v2-feasibility-20261008T062621Z
```

Generation exited 1 after reaching the source partition's 30,000-attempt cap at candidate index 29,999. The public audit reports 734 visible `source_train` records; that is not the total source quota. Development and final generation were not attempted. No full-mode command ran. The generation manifest records an incomplete run; the independent audit completed with integrity and recorded-prefix replay passing, but replay was partial and `g2_status` is `not_assessed`. The proposal is NO-GO because generation hit its fixed attempt cap. There was no changed-rule retry.

## Public feasibility metrics

Only `source_train` and `development_validation` metrics are included here. Feasibility mode omits source-validation, source-test, and development-diagnostic aggregates; they were not used to make choices and are not inspected in this report.

| Measure | Observed result |
| --- | ---: |
| Accepted source-train groups | 734 |
| Accepted source-train structure signatures | 454 |
| Raw unique source candidates in the sensitivity denominator | 1,239 |
| Constraint-sensitive source groups | 734 / 1,239 = 0.5924132365 |
| Rejected-only source signatures | 422 |
| Source-train primitive coverage | `eq`, `neq`, and `add_mod`; pass |
| Source-train value fractions (0, 1, 2) | 0.319012, 0.333934, 0.347054 |
| Source-train given counts | 1, 2, 3, 4 |
| Development-validation groups | 0; generation was not reached |
| Final targets | Not attempted |

The observed constraint-sensitivity fraction gives limited evidence that the accepted source tasks are not all insensitive under the registered single-constraint-removal check. It does not measure learned difficulty or establish a useful hard regime. The source sample contained 454 accepted structural signatures, while 422 signatures occurred only among rejected proposals; these counts do not establish target coverage.

The four-cohort common-support proxy has no common cells because its three development-validation cohorts contain no generated groups. This is unavailable support evidence, not an observed source-to-target mismatch or target-yield failure. Final-target support, target signatures, and the full five-cohort G2 comparison remain unassessed.

## Resource accounting and verification limits

The authoritative terminal audit manifest records 125.985 seconds combined wall time, 125.578125 seconds combined CPU time, and 183,894,016 peak resident bytes. Generation used 47.375 seconds wall / 47.125 seconds CPU and peaked at 25,817,088 bytes. The audit used 78.61 seconds wall / 78.453125 seconds CPU; its terminal combined peak was 183,894,016 bytes. The run stayed within the 600-second and 1-GiB caps. The generation cap, rather than a resource cap, caused the NO-GO.

The controller's fresh `python -m unittest discover -s tests -v` run at the frozen source commit passed 138 tests in 2.250 seconds. This is CPU implementation evidence only. It does not convert the partial generation into a passing suitability audit. No model, training, final corpus, archival backup, or independent reproduction was performed.

Artifact sizes and SHA-256 hashes for the generation and audit outputs, plus exact command-log identities, are in the [compact feasibility manifest](../configs/splits/constraint-v2-feasibility-nogo.json). The ignored output remains under `data/constraint-v2-feasibility-20261008T062621Z/`. Preserve this attempt. The next step is a prospective development-only design for bounded source-generation yield/quota feasibility, followed by the composition-support question. This does not authorize an arbitrary rerun. Any new run requires a separately versioned and reviewed contract.
