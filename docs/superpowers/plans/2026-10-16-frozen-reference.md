# GPU frozen-reference implementation plan

> **For agentic workers:** use superpowers:subagent-driven-development. Luna implements; root decides scientific settings and release. Owner authorizes routine design choices and continuation; no per-task commits or agent merge.

**Goal:** Extend the existing native greedy evaluator to a consistent CUDA device and run one preregistered, fully recorded frozen-reference measurement cohort.

**Architecture:** Preserve generation/scoring contracts, add observational budget callbacks and device synchronization, then a closed runner reusing the exact verified candidate's operational helpers. Actual controller tests use real run records, with external/model seams stubbed separately from tiny actual CPU integration.

**Tech stack:** Python3.11, standard-library task/manifests/logging, existing pinned Torch2.11.0+cu128/Transformers4.57.1/NumPy2.4.6. No new dependencies.

**Spec:** [frozen_reference_design.md](../../frozen_reference_design.md). Closed config SHA `9af9196c59b4b65774ac82aebb57378cb2fc23513b382b40cdf5b6dfe55c824b`; cohort SHA `bca9505b817a9b75b9d7aba33710ddc17b47596bdb9a9772924348d15b4b64bd`.

## Global constraints

- Exact frozen original BF16 Qwen2, SDPA/numeric/cache/framing/greedy/strict verifier policies from the passed feasibility candidate; no synthetic-updated checkpoint, new modules, optimizer or target task semantics.
- CPU/CUDA single-device consistency without silent moves, all original mode/weight/RNG/hook/finite checks, workspace X/Z once and final Z unchanged across generation.
-15 fixed source-validation IDs,5per primitive,3 repetitions; quality denominator15/repetition0 only; no accuracy minimum or confirmatory claim. No refilling, skipping or truncation.
-900s internal/905s external,12GiB CPU,14GiB allocated/reserved,14500MiB initially free, one attempt/zero automatic retries. Source clean and exact bytes at invocation.
- Actual structured RunRecord identities and projected budgets, per-example fsynced predictions and terminal pre-seal resource checks. CPU mocks are not GPU evidence.
- Luna owns only each task's explicit files; root owns config/spec/cohort/status. No corpus/model/GPU operations, full suites, commits or nested agents during implementation.

## Review focus

- Mixed/meta device, wrong-device logits/noise or CPU code accidentally initializing CUDA must fail/preserve state.
- A budget callback failing after native decoding must clean hooks and preserve completed predictions.
- Exact closed config and real RunRecord/source/cohort adapters must be exercised together, avoiding the previous strict-API startup gap.
- Source metadata cohort has no NEQ/A3; never invent/refill a cell or reinterpret balanced15 accuracy as full validation performance.
- Late artifact hashing, repeat mismatch or cap failure must leave no completed/PASS terminal seal.

## Task1 — device-aware greedy component

Files: modify `dwr/evaluation/generation.py` and `tests/test_native_generation.py` only. Consumes native host/core and current generation/scoring interfaces; produces same interfaces plus `check=None` on both public greedy functions, with CPU/CUDA supported.

- [x] Read the exact spec/current component and tests; capture genuine behavioral REDs for callback/device contracts before modifying implementation.
- [x] Implement uniform device validation, device-correct token tensors/autocast policy, synchronized CUDA timing and observational callback boundaries; preserve CPU behavior and complete answer semantics.
- [x] Run scoped `test_native_generation.py` under `.venv-ml`,90s watchdog, with meaningful tiny native/manual-forward/workspace checks and clearly separated controller mocks. Test callback failure cleanup, unsupported/meta/mixed devices and absence of CUDA calls on CPU.
- [x] Save snapshots/hashes/RED/GREEN/report under ignored `.superpowers/sdd/frozen-reference-20261016/task1/`; stop writes. Independent review and bounded fixes before Task2 depends on the new API.

## Task2 — closed source-reference runner

Files: create `dwr/experiments/frozen_reference.py` and `tests/test_frozen_reference.py` only. Consumes cleared Task1, source reader, strict scorer/serializer, real run records and byte-bound feasibility operational helpers. Produces `run_reference(config_path, *, command)` and fixed `--config` CLI, predictions JSONL, warmup record, summary and linked manifests.

- [ ] Write behavior-first CPU regressions for exact registration/cohort/proof mapping and real RunRecord success/failure, using temporary Git and stubbing only external/model boundaries. Preserve shared strict APIs. Initial strict test-first chronology was missed and remains unchecked as historical evidence; meaningful regression coverage is complete. See implementation verification.
- [x] Implement no-bypass config/source/proof/model/data checks and initial/access records; exact validation-only15 IDs, prompt-only framing and local original model load/policy.
- [x] Implement synthetic warmup plus45 attempts, complete IDs/output/termination/verifier/timing/memory records, exact repeat equivalence,15-quality denominator, original/RNG invariance, full-scope terminal caps and no automatic retries.
- [x] Test actual tiny CPU native evaluation/scoring, canned rows/selection/prompt hashes, cap/invalid outputs, missing/duplicate/mismatched IDs, repeat mismatch and late finish budget failure. Run only covering scopes under90s; save report/snapshots/hashes/RED/GREEN in task2 folder and stop writes.
- [x] Independent review/fix loop, then root whole-new-stage review on stable source and full separate CPU release. Do not reinterpret component tests as model validation.


## Task3 — root release and sole reference invocation

- [x] Root bind actual feasibility PASS/source/cohort/export/token/model hashes, review proof and exact runtime/config/test bytes. Preserve prior failed startup and all negative reviews.
- [ ] Consolidate source/config/cohort/result docs in one clean substantive checkpoint, push/update existing PR#2 without merge. Original main checkout stays untouched.
- [ ] Recheck hardware; identity-check/close unused apps only under owner's cleanup authorization if required. Execute once under905s watchdog, no fallback/refill/retry.
- [ ] Independently verify terminal links,45 predictions/15 unique examples/repetition0 quality, exact repeated IDs/scores, original/RNG state and full costs/caps. Record the actual result without learned/transfer claims.
- [ ] If operational reference passes, continue bounded source overfit/pilot registration and implementation, with fresh matched initializations; no target evaluation or diffusion. Stop only at a genuine blocker, preserving a durable next action.
