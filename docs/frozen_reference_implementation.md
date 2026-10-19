# Frozen-reference implementation verification

2026-10-16. **READY_FOR_CLEAN_CHECKPOINT_AND_SOLE_REFERENCE** after independent component and whole-stage reviews, one scoped final fix-wave review, and root full CPU release on stopped bytes. This is CPU implementation/controller evidence. The actual original-backbone reference has not run; no task-learning/transfer or G3/G4 PASS is claimed.

## Scope and resolved findings

The [design](frozen_reference_design.md) and [plan](superpowers/plans/2026-10-16-frozen-reference.md) implement device-aware native greedy generation and the closed15-example source-validation reference. Generation preserves full-prefix/cache-off decoding, strict EOS/cap/finite checks, one prompt X/final Z, caller devices and optional budget callbacks. CPU paths do not call CUDA; actual CUDA generation remains the next invocation.

The runner reuses the byte-bound feasibility operational helpers with the validated full v2 config, loads only the original frozen Qwen2, records initial/access events before CUDA/task semantics, measures actual original/total and zero trainable/new parameters, and emits one warmup plus45 ordered predictions. Quality uses repetition0/15; repeated token IDs and scores must match. It preserves global RNG and original weights, fsyncs predictions, measures cumulative peaks/full costs and checks budgets through terminal artifact hashing. There is no accuracy minimum or protected evaluation.

Review caught and fixed device-validation ordering, the raw-Qwen versus workspace-host hashing adapter, missing candidate/proof integration coverage, optional-ML test isolation and missing measured parameter counts. The candidate tests now retain the actual45-attempt loop, hashing/scoring/RNG/weight checks, record formatting/writes, aggregates and real RunRecord lifecycle. External loading/device/CUDA/corpus boundaries and generated tokens are canned; a separate tiny random CPU native model exercises real generation/scoring. These tests are not pretrained/GPU evidence.

## Root release evidence

Exact controller command from the active worktree:

```text
.venv-task/Scripts/python.exe .superpowers/sdd/frozen-reference-20261016/verify_separate_suites.py
```

It runs standard-library discovery with Torch absent, followed by the ML subset for workspace, native host, loss, generation, resume, probe and reference. Each exact child command and method ID is retained in the receipt below.

- Standard-library process:686 reported,611 passed,75 optional-ML skips; exit0,53.637s unittest/53.890s wrapper.
- ML process:117 passed,zero skips; exit0,26.002s unittest/29.188s wrapper.
-34 executed methods overlap:611+117-34=**694 distinct executed checks**.
- Combined83.110s under180s watchdog; runtime/config/test bytes stable before/after.
- Standard-library log SHA-256 `b777c2d0698c571268b0a0fb25c639422e7c19419cdb3db9f64a512e7d64dc7b`; ML log `9e0d0df8ea4c44d72a9fc8b4f5d0125ee3cb89e5dbfdacdbdc091c8b03f7eb54`.

Ignored evidence under `.superpowers/sdd/frozen-reference-20261016/`: `controller-separated-suites-20261016T142551948879Z.json`, both method lists/logs, `final-cpu-method-accounting.json`, `design-review.md`, component reviews/rereviews, `final-review.md` and `final-rereview.md`. The original root release failure (`20261016T042136598846Z`:three unguarded Torch imports) remains preserved; no guard was loosened or dependency installed into the standard-library environment.

Final physical SHA-256 values:

- `dwr/evaluation/generation.py`: `d132ffd95c1aee6b87010da1f6102f844cbf1b05fe1f89e9ab58a14c88e970f8`.
- `dwr/experiments/frozen_reference.py`: `03b755b5f4eeb4d24965f5ebc194f29f1c82067ff57136e64c073799bcef933f`.
- `tests/test_native_generation.py`: `e8a746f8646731583aeeaf068a4ff0164612c94200db577d90e93c5800d0aaf6`.
- `tests/test_frozen_reference.py`: `f6773f0c505d4b4605eb4734ec3bc7f73f075c4cd05cf3eab8433eef9c6ed2d1`.

The reference config/cohort hashes remain `9af9196c59b4b65774ac82aebb57378cb2fc23513b382b40cdf5b6dfe55c824b` / `bca9505b817a9b75b9d7aba33710ddc17b47596bdb9a9772924348d15b4b64bd`. The passed GPU feasibility and all source/model/token bindings are unchanged.

## Evidence limits and next gate

Task1 has preserved behavioral RED/GREEN. Task2 initially recorded only an absent-module ImportError; this is startup evidence, not behavioral TDD. Its first expanded regressions were added after production edits began, so strict test-first chronology is not claimed. Later genuine missing-parameter-count and optional-dependency REDs, negative reviews and failed fixtures remain intact. Root retained honest history rather than deleting/recreating it; independent reviews and actual behavioral/controller checks provide the available assurance.

The release review service stopped after delivering its complete two-finding list; recovery saved that already-completed review, then the single Luna fix wave and scoped re-review passed. No experiment was attempted during that interruption. Source/config/docs will be consolidated once; actual execution must bind the clean current descendant of83406bb and exact physical bytes. Then invoke the sole900s/905s-watchdog reference after hardware preflight, without automatic retry. Artifacts/review controllers remain local and ignored, not archival backup or independent-person reproduction. Use fresh matched initialization for later separately registered learning pilots; never reuse the synthetic-updated probe checkpoint.

Final formatting correction removed only extra test-file EOF newlines after the review, with equal AST recorded in `task2/final-wave/eof-format.json`. Root reran both complete suites on the final exact bytes before amending its own unpublished checkpoint. The earlier passing release and the initially missed cached whitespace warning remain preserved; no scientific setting changed.
