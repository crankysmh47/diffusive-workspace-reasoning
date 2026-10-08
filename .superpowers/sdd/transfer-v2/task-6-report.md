# Task 6 report — v2 feasibility durable state

Date: 2026-10-08. Worktree: `C:\sem5\diffusive-workspace-reasoning\.worktrees\transfer-cpu-v1`. Branch: `research/transfer-cpu-v1`.

## Result

Recorded one registered feasibility sequence: generation stopped at the 30,000 source-attempt cap; the independent audit completed with integrity and recorded-prefix replay passing. G2 is not assessed. Development and final generation, full mode, model work, and training did not occur. The visible 734 source-train records are not the total source quota. The source-sensitivity floor passed; development support is unavailable because its cohorts were not generated.

## Durable files

- `docs/task_suitability_v2.md`: run identities and commands, public source-train/development-validation metrics, unassessed dimensions, resource actuals, limitations, and next gate.
- `configs/splits/constraint-v2-feasibility-nogo.json`: compact SHA-256 and byte records for generation/audit outputs and controller evidence; no raw records.
- `docs/execution_state.md`: current v2 status and all current v2 ledger rulings, with v1 history, metrics, and rulings retained.
- `todo.md`, `README.md`, `docs/task_v2_execution_plan.md`, `docs/task_spec_v2.md`, and `docs/research_protocol.md`: current verified status, unchecked G2/full/model milestones, and next-step pointers.
- `AGENTS.md`: first-read pointers for the active v2 spec, plan, and suitability record.

## Evidence and checks

The frozen source is `9f08efddb60537c61a89a90325bc9c832c32ca1c`; config SHA-256 is `cc9fb0a827d27e4f00425ca9364365026bb6c70c75dc8d6dc0eebcc1f6865222`. Commands and artifact hashes are in the compact evidence manifest. The controller log `.superpowers/sdd/transfer-v2/controller-final-suite.txt` records `python -m unittest discover -s tests -v`: 138 passed in 2.250 seconds at the frozen source commit; this suite was not rerun for this documentation-only task.

This task ran no project tests, corpus generation, audit, model, or training command. Documentation checks completed:

- PowerShell validation of JSON parsing, config SHA-256, and every recorded artifact/log SHA-256 and byte length: passed.
- Local Markdown link-target check across edited docs: passed.
- `git diff --check`: passed.

## Limits

This is local agent-run evidence. G2, the full five-cohort support comparison, final-target yield, target mismatch, model behavior, archival backup, and independent reproduction remain unverified. The source-only prefix does not support conclusions about those quantities. No changed-rule retry was made.

Durable-state commit: `41fc48eb2037fdc147bd1b49d3f452793686d686` (`docs: record v2 feasibility attempt outcome`).
