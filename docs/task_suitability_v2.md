# V2 task audit implementation

Date: 2026-10-08. Status: implementation and CPU fixtures only. No v2 corpus has been audited here, and this document records no dataset result or G2 decision.

The independent audit entry point is:

```powershell
python -m dwr.tasks.audit_cli_v2 --config configs/tasks/constraint_v2.json --mode feasibility --data <generation-directory>
```

`--mode full` uses the same replay and integrity path and checks the registered five-cohort G2 floors. Each corpus permits one audit attempt: the CLI refuses known audit artifacts and temporary files already present. It writes `audit_manifest.json` before replay, a detailed `audit_report.json`, and the closed eight-field `audit_gate.json`. A complete, matching, clean-source `PASS_FEASIBILITY` also writes the closed Task 4 `preflight_report.json` wrapper. A partial generation can pass replay of its recorded prefix but remains `g2_status: not_assessed`.

The CLI exits successfully only for `PASS_FEASIBILITY` or a complete full audit with `measured_pass`; a complete audit with `NO_GO` remains preserved and exits nonzero. It records combined generation-plus-audit wall time, CPU time and peak resident memory. Report, gate and first terminal-manifest writes are charged before a post-write resource check; one bounded final metadata flush follows that check. A terminal-write or cap failure changes the gate to `not_assessed`, removes a provisional preflight wrapper, and records a terminal failure sidecar if recovery writes fail.

The audit uses `manifest_v2.validate_manifest_v2`, `replay_files_v2`, `audit.validate_record`, and `audit.audit_interactions`. It independently reconstructs each first-seen unique oracle task, solution, identity, input projection, solver statistics, source sensitivity, and mixed-kind/connectivity annotation. Generation-private eligibility labels are checked as annotations and never used as the audit's proof. Raw mixed denominators count unique oracle-unique outcomes before eligibility; duplicate content is excluded. Rejected-only signatures are reported separately from accepted-signature floors.

Feasibility aggregates only source-train and development-validation metrics. Other recorded splits remain available for deterministic replay and integrity checks, while their aggregate counts, strata, yields, and metrics are omitted from feasibility output. The four-cohort support result is explicitly a proxy; it does not establish final-target support. Full mode checks source ID-test and the three eligible development-validation cohorts plus eligible final-test, with the registered common-support thresholds. Both modes share the same remaining generation-plus-audit resource guard.

The common-support helper accepts an ordered `cohort_names` tuple. Its omitted-argument path retains the v1 five-cohort behavior. Synthetic unit fixtures verify the four-cohort proxy, the ten-groups-per-cell and 80%-mass thresholds, private annotation rejection, held-out aggregate opacity, CLI collision refusal, and full-mode cohort naming. These fixtures are implementation checks only; they are not corpus, feasibility, or model evidence.
