# Resolved run records implementation plan

> **For agentic workers:** use superpowers:subagent-driven-development. Owner requests Luna implementation and minimal consolidated commits.

**Goal:** Durable byte-bound initial/terminal run records before native experiment launchers exist.

**Architecture:** One standard-library module, no training framework. Consumer-owned scientific validation remains separate from run metadata validation.

**Spec:** [run_record_design.md](../../run_record_design.md).

## Global constraints

No scientific run, corpus/model/GPU access, new dependency, legacy task change or per-task commit. Use existing isolated research worktree. Preserve all RED/GREEN evidence under ignored `.superpowers/sdd/run-records-20261015/`. No automatic evidence deletion because raw checks must remain preserved under the protocol.

## Task1 — single-owner run records

- [ ] Read spec and protocol. Own only new `dwr/experiments/{__init__.py,records.py}` and `tests/test_experiment_records.py`.
- [ ] Write meaningful failing stdlib tests for strict metadata/config/Git identity, lifecycle and artifact integrity using temporary real repos. Capture actual RED.
- [ ] Implement the minimal specified API; no universal scientific config schema or launcher. Capture actual GREEN from `.venv-task/Scripts/python.exe -m unittest discover -s tests -p test_experiment_records.py -v`, cap90s.
- [ ] Self-review, source hashes, command/outcomes and preserved negative logs. Root dispatches independent task review and fixes.

## Task2 — root release and continuation

- [ ] Stop writer, independent component review, resolve load-bearing findings; record verification before status updates.
- [ ] Full stage verification stays in separate stdlib/ML environments under combined180s; no relaxed legacy memory guards. Consolidate substantial checkpoint only, per owner.
- [ ] Continue exact allowed-source export, native autoregressive evaluation, full training resume and separately registered GPU feasibility. Do not stop at component completion.

Preflight: Task1 creates the only source/test files; Task2 consumes their API and does not overlap writer ownership. Initial and terminal schema/link validation use one common definition; scientific config remains the future consumer's obligation. Design does not certify model/training readiness.
