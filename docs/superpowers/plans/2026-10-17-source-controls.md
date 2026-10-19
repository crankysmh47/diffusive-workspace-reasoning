# Trained source controls implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Execute the single task below with Luna implementation and root decisions/release. Owner already requested autonomous execution and minimal commits; no additional approval or per-task commit is needed.

**Goal:** Implement and verify the shared training-step policy for K0 noise-memory, EX K1 and EX K4 before a separately registered GPU pilot.

**Architecture:** A small explicit control description configures active parameters on the existing native host. A single checked step consumes caller-owned optimizer, schedule, batch and noise. Existing overfit/provenance/recovery foundations stay unchanged.

**Tech Stack:** Existing Python3.11, PyTorch2.11, Transformers4.57.1 and unittest; no new dependency.

**Spec:** [source_controls_design.md](../../source_controls_design.md).

## Global constraints

- Exactly steps {0,1,4}, exchangeable `max_steps=4`; original backbone frozen/eval, matched reader/updater construction.
- K0 trains only readers; K1/K4 train updater plus readers. No silent parameter removal or inference-compute matching claim.
- Explicit caller-owned noise/optimizer/schedule and existing answer loss; preserve prompt-only X and all recurrent gradients.
- Exactly named-module AdamW groups and LambdaLR owned by that optimizer, matching existing checkpoint validation. All parameter gradients None on step entry and successful exit; stale accumulation is rejected.
- Unit checks use tiny random actual native CPU hosts, not pretrained weights or retained tasks. CUDA pilot is not authorized by this implementation plan alone.
- No existing overfit runner/config changes; no source/test writer during final scoped verification; no commit until root consolidates.

## Review focus

- An optimizer accidentally includes the K0 updater or duplicates a parameter: reject before mutation.
- Same-shape host with changed active flags or stray frozen gradients: reject, never repair in step validation.
- K0 diagnostic noise handling: live gradient evidence without caller noise mutation or hidden RNG consumption.
- Loss/gradients/update produce NaN/Inf or disconnect an active reader: fail closed and preserve the caller's failure evidence.
- A resume reintroduces workspace optimizer moments into K0 or drops sampler/RNG state: strict existing checkpoint validation and actual fresh-process comparison.

## Task 1: Native source-control policy and checked updates

**Files:** Create `dwr/training/controls.py`, `tests/test_source_controls.py`; optionally a tiny child helper in `tests/` only if fresh-process proof needs one. Record verification in `docs/source_controls_verification.md`. Do not alter unrelated runtime/config/test files.

**Interfaces:** Consume `Qwen2WorkspaceLM`, existing collated batch, `answer_only_loss`, `SourceSampler`, `save_training_checkpoint` and `load_training_checkpoint`. Produce `SourceControl`, `configure_source_control`, `validate_source_control` and `source_control_step` with exact signatures/semantics from the spec. Future pilot runner will import these explicitly.

- [ ] Read spec and existing native/optimizer/checkpoint tests; record the exact test command and expected behavioral RED before writing implementation.
- [ ] Add actual native failing tests for K0 updater exclusion, K1/K4 active updates and matched initial state; run and preserve RED. Include original byte immutability and state/reader gradient assertions.
- [ ] Implement the small control description, strict configuration/validation and checked step; do not duplicate checkpoint or job machinery.
- [ ] Add concrete fail-closed cases from Review Focus and actual K0/K1 AdamW/LambdaLR uninterrupted versus fresh-process resumed trajectories at `atol=rtol=1e-12`, with nonempty moments, nonconstant LR and complete RNG/sampler state.
- [ ] Run focused unittest in `.venv-task` (optional ML skips only) and `.venv-ml` (all ML cases execute). Capture exact commands/results and stable file hashes; preserve failures. Run relevant existing checkpoint/native tests once to catch integration regressions.
- [ ] Self-review, write bounded verification/report and stop all source/test writes for root's scoped independent review. No training, pretrained load, commit or push.

Root then verifies the stopped diff and scoped evidence, rules on findings and records completion only after checks. The subsequent source-validation pilot requires its own closed registration, implementation and release; this plan does not invent its hyperparameters or use protected data.
