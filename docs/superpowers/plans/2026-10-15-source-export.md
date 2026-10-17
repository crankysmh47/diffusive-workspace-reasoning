# Permitted-source export implementation plan

> **For agentic workers:** use superpowers:subagent-driven-development; Luna implements and root decides. No per-task commit.

**Goal:** Convert only the exact accepted source-training/validation population into model-consumable records without regenerating tasks.

**Spec:** [source_export_design.md](../../source_export_design.md).

**Architecture:** Standard-library conversion core binds immutable producer artifacts and routes metadata before task semantics. A subsequent strict registered wrapper uses the run-record API. No eager ML import.

## Global constraints

Preserve frozen data/splits/earlier failures. No actual retained task payload read during component implementation/tests, model/GPU work, dependency installation, full suites, commits or nested agents. Preserve all evidence under ignored `.superpowers/sdd/source-export-20261015/`. Public original terminal metadata may be inspected to resolve producer schema, never actual outcome/task/answer lines.

## Task1 — conversion core

- [ ] Read spec, existing producer projection/schema and independent solver/oracle APIs; implement only `dwr/experiments/source_corpus.py` and `tests/test_source_corpus.py`. Do not modify experiments package initializer or run-record code owned by the other worker.
- [ ] Real tiny fixture ledgers, actual strict solver/oracle, poisoned forbidden payloads and valid construction_failed/null split regression. Add hash/status/projection/identity/routing/duplicate/uniqueness/output/midrun mutation negatives; capture meaningful RED.
- [ ] Implement pure converter. Keep all six hashes and original config bytes/typed resolved config bound. Protect nonallowed task semantics and exact source counts/order. Capture GREEN with `.venv-task/Scripts/python.exe -m unittest discover -s tests -p test_source_corpus.py -v`, cap90s. Return exact API and source/evidence hashes.
- [ ] Independent task review and Luna fixes; no claimed real corpus conversion from fixture success.

## Task2 — registered production wrapper and loader

- [x] Root writes closed exact config and wrapper/loader contract after Task1/run-record APIs are reviewed. See [runner design](../../source_export_runner_design.md), independently READY_FOR_IMPLEMENTATION.
- [ ] Luna Task2 owns only new dwr/experiments/source_export.py, source_reader.py and tests/test_source_export_runner.py, test_source_reader.py. Use actual converter/run-record integration in temporary real Git fixture repos. Root owns config/design; no runtime fixture bypass. Capture genuine RED/GREEN from unittest pattern test_source_* under90s, without reading actual retained corpus. Independent review/fixes follow.
- [ ] Full root suites on stable bytes, clean source freeze in one substantial checkpoint, then sole300s/512MiB source conversion plus305s watchdog. Initial manifest and access event precede payload work; zero automatic retries, preserved failures.
- [ ] Root verifies summary/counts/hash/access boundary; update checklist only after actual passed conversion. Continue evaluation/resume/GPU feasibility prerequisites.

Preflight: Task1 owns new converter/test files only; Task2 consumes converter plus run-record APIs, waits for both gates, no shared writer. Source-validation export permits scoring, not optimization; no protected task gold enters export.
