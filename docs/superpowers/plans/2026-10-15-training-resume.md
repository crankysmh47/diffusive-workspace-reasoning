# Learned-state resume implementation plan

> **For agentic workers:** use superpowers:subagent-driven-development. Owner requires Luna implementation, root decisions and minimum consolidated commits.

**Goal:** Actual CPU AdamW training can resume with model, moments, schedule, RNG streams and exact source sampler position preserved.

**Spec:** [training_resume_design.md](../../training_resume_design.md).

**Architecture:** One explicit-ID local-generator sampler and a checkpoint helper for the real native host. Save all new workspace/reader states and original model identity; avoid repeated multi-GB frozen-weight copies. Strict known schema, no general training framework.

## Global constraints

No scientific training run, actual corpus/pretrained model/GPU access, dependency installation, verified host/core edits, full suite, commits or nested agents. Required optimizer/schedule/scaler policies and dtype/head/identity checks are in the spec. Isolate initialization/RNG streams and restore fixture environment/thread settings. Preserve ignored evidence under `.superpowers/sdd/training-resume-20261015/`.

## Task1 — sampler and complete CPU resume

- [ ] Own only `dwr/training/sampler.py`, `dwr/training/checkpoints.py`, `tests/test_training_resume.py`; no training initializer edits while other work continues.
- [ ] Write meaningful failing sampler/identity/cursor/group/schedule/hash/checkpoint tests. Declare resume tolerance1e-12 before observations; capture RED.
- [ ] Implement explicit local sampler and full boundary-only save/load. Actual tiny native+collator/loss fixture with nonempty AdamW moments and nonconstant LambdaLR compares4 uninterrupted steps against2+restart+2. Compare all learned parameters/moments/scheduler/global+local RNG/sampler/progress and frozen base bytes; no mock substitutes for native evidence.
- [ ] Test same-shape different workspace heads and foreign scheduler optimizer, reordered parameters/IDs, malformed state and finite/gradient/scaler/device/hash/path failures. Capture GREEN from `.venv-ml/Scripts/python.exe -m unittest discover -s tests -p test_training_resume.py -v`, cap90s. No CUDA initialization.
- [ ] Self-review/hash/API/evidence receipt, independent review and Luna fixes. Root release full checks and later actual GPU replay remain separate gates.

Preflight: sampler produces exact ordered state consumed by checkpoint; checkpoint consumes existing native host/training batch/loss without changing them. Task source files do not overlap concurrent experiments modules. Initializer remains unchanged to preserve lazy optional dependency behavior.
