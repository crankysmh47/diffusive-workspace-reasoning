# Transfer v1 bounded suitability result

Date: 2026-10-08. Status: **G2 not assessed and not passed.** This is a bounded partial-corpus diagnostic, not a validated benchmark or a model result. The active exchangeable-workspace transfer hypothesis remains unchanged; the conditional EX/IDX work was not started.

## Registered run and provenance

The run used the frozen source commit `800f9cdb0d44c2b0ca54fd54039c5383821a3c2b` (clean checkout), seed `20261007`, and unchanged [registered config](../configs/tasks/constraint_v1.json), SHA-256 `26e43360a005fc41fd00981d8ff8471ab9a4bbc90494f7ea312958c0e428e8f3`. The generator command was:

```powershell
python -m dwr.tasks.generator --config configs/tasks/constraint_v1.json --out data/constraint-v1-20261007220128
```

It exited 1 with `PermissionError [WinError 5]` while atomically replacing the run manifest after 22,450 source candidates and 185 accepted groups. The registered source cap was 30,000 attempts; it was **not reached**. The accepted source profile counts were `eq=93`, `neq=23`, `add_mod=69`. Development and final were each `0 / not_attempted`. No run was retried and no scientific setting was changed.

The independent audit command was:

```powershell
python -m dwr.tasks.audit --config configs/tasks/constraint_v1.json --data data/constraint-v1-20261007220128
```

It exited 1 with `incomplete_generation`. It independently reconciled and validated all 185 records, evaluated interaction metrics on all 185, and replayed 22,450 candidates. Integrity and replay checks passed. Generation was incomplete, structural floors failed, and `g2.status` is `not_assessed`; partial metrics cannot establish suitability.

The local, ignored evidence is under `data/constraint-v1-20261007220128/`. It is agent-run evidence and is not archived or independently reproduced. The compact hash manifest is [configs/splits/constraint-v1-partial.json](../configs/splits/constraint-v1-partial.json). The full audit report is 94,332 bytes and is deliberately not copied into Git.

## Measured structure and blockers

The 185 accepted records comprise 185 unique content IDs and 137 structural signatures. Source splits assigned 153 groups to train, 13 to validation, and 19 to test; validation and test are below the registered 20-group floors. Train contains all three primitive kinds, so the source primitive-exposure check passed for this partial sample. The profile totals were 93 equality, 23 inequality, and 69 modular-addition groups.

Of the 185 source examples, 120 were sensitive to removal of at least one constraint (64.86%). However, 51 observed signatures had no sensitive example, failing the registered one-sensitive-example-per-signature requirement. The `n=3` equality-triangle signature `b4df0ccea1b640e9ca2bba3a314a4340f4487249cbe1f7fbb91dae73f0f47625` appeared three times. Every one of its edge removals still had exactly one solution, so this structure cannot satisfy that sensitivity floor under the current criterion.

There were no development or final target examples. All mixed-profile eligibility counts are therefore zero/not attempted, and no source-to-target common-support comparison exists. The audit reports `structural_pass=false`; source validation/test group floors, mixed-target floors, common support, value balance and given-count variation also fail or are unassessed. These are observed design blockers, not evidence that the model yields zero on a held-out benchmark.

## Cost and bounded Windows diagnostic

Generation used 17.891 seconds elapsed, 17.640625 CPU seconds, and peak resident memory 379,232,256 bytes. Audit used 19.875 seconds elapsed, 19.640625 CPU seconds, and peak resident memory 433,459,200 bytes. Combined generation plus audit was 37.766 seconds, under the single registered 600-second active-command budget. Replay stopped with the failed incomplete run; 22,450 candidates is below the 30,000 per-partition cap.

A separate registered temporary-file probe reproduced the Windows mechanism: a reader handle that shared read/write but omitted delete sharing caused `atomic_json` replacement to fail with WinError 5; closing the handle allowed replacement. This establishes a sufficient sharing-lock mechanism, but does not identify the lock owner or prove that the controller's coincident `Get-Content` caused this run's failure. See the ignored [diagnostic record](../.superpowers/sdd/transfer-v1/windows-manifest-diagnostic.md). No source or corpus change followed the probe.

## Gate decision and next work

G2 remains conditional and unmet. The main exchangeable transfer hypothesis stays active; EX/IDX tensors, model integration, registration, and training remain unimplemented. The next work is a **versioned development-only design** addressing the source-sensitivity/support blocker and safe Windows progress/manifest handling. It must be decided without selecting parameters on final-target outcomes. This document does not authorize a corpus retry. Preserve the failed run and its partial diagnostics; any future study must version its decision and retain these results.
