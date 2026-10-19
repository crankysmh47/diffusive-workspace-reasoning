# Frozen-reference startup-repair implementation plan

> Use subagent-driven development. Fresh Luna implements; root decides and releases. Minimal consolidated commits, original v1 allowance closed, no agent merge.

**Spec:** [frozen_reference_startup_repair.md](../../frozen_reference_startup_repair.md), amending only operational startup and active registration. ConfigSHA e3a67600b9bada872bd4523f319378a2088e25824df82d90c44d2fb00b42144f. Same scientific gate/cohort/budgets; exactly config_version/output/source_base change.

## Task1 — two-file bounded startup repair

Own only dwr/experiments/frozen_reference.py and tests/test_frozen_reference.py plus ignored `.superpowers/sdd/frozen-reference-startup-repair-20261016/task1/`. No pinned helper/generation/record edits, dependency changes, actual corpus/pretrained/GPU operations, full suite, commits or nested agents.

- [x] Read spec/existing exact initializer and closed config; capture behavioral RED reproducing reset-before-init with stateful CPU mocks and exact-device/order assertion.
- [x] Synchronize registered CUDA device before the sole pre-model peak reset; CPU path makes no runner CUDA startup/measurement calls; preserve Torch seeding and distinguish its deferred third-party CUDA-seed dispatch from initialization. No dummy tensor, fallback, cache clear or post-load reset.
- [x] Retarget every closed config SHA/version/path/output adapter to the new v2 registration only; preserve original v1 config/artifacts/source and all scientific settings.
- [x] Run both scoped test_frozen_reference.py suites under90s, preserving real tiny CPU, actual candidate loop, record/proof/failure coverage and initialization-order regression; save exact logs/snapshots/hashes/report, stop writes.
- [x] Independent review and bounded fixes; root full separate CPU release on stable bytes. CPU mocked CUDA order is not actual GPU evidence.

## Task2 — root release and sole additional invocation

- [x] Verify failed original terminal links, three-key config diff and unchanged scientific pins; preserve all failures/service interruptions.
- [x] Consolidate/push one clean reviewed checkpoint, update existingPR#2, verify every physical tracked byte against HEAD. Keep original main checkout untouched.
- [x] Recheck GPU; close only owner-authorized unused apps if required, preserving Zen/terminal/activeCodex/essentialWindows. Invoke only v2 once under905s, no automatic retry.
- [x] Verify terminal hashes/45 records/15 quality and repeat/base/RNG/parameter/cost invariants; record actual outcome. G3/learning follow their gates. A further failure closes this allowance and requires separate root assessment, not an automatic second repair.

Preflight: Task1 consumes passed reference/candidate/record/proof and exact new root config, produces same public functions with active closedCLI retargeting and one CUDA startup-order correction; Task2 consumes stopped cleared source. No shared writer. Owner-authorized routine operational correction, with no model prediction guiding the change.
