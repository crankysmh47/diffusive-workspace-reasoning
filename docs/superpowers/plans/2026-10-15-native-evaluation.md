# Native greedy evaluation implementation plan

> **For agentic workers:** use superpowers:subagent-driven-development; Luna implements after independent design review. Owner delegated routine design and requests minimum commits.

**Goal:** Strict single-draw EOS-bounded generation/scoring on the actual native model, first verified on CPU fixtures.

**Spec:** [native_evaluation_design.md](../../native_evaluation_design.md).

**Architecture:** Small real-native workspace/frozen generation entry points share one greedy loop. Strict scorer uses existing task verifier without answer repair. Later run wrapper supplies protected-access registration, per-example identity and measured GPU costs.

## Global constraints

CPU-only component, no pretrained/actualdata/protected access, GPU operation, dependencies, host/core changes, full suites, commits or nested agents. Keep exact X once/Z once and cache-off policy. Preserve ignored RED/GREEN/snapshots in `.superpowers/sdd/native-evaluation-20261015/`.

## Task1 — actual native generation and strict score

- [ ] Own only `dwr/evaluation/{__init__.py,generation.py}` and `tests/test_native_generation.py`; read protocol/native/evaluation specs.
- [ ] Meaningful RED for exact EOS/cap/tie/strict decoder behavior with scripted algorithm fixtures; distinguish these from actual native model checks. Test fixed-X/Z call counts, noise seed independence, invalid inputs and score/cap failures.
- [ ] Implement real Qwen2 host and unaugmented frozen entry points with explicit CPU/no-grad/eval/finite/shape/cap checks. No mode/flag/global RNG mutation, answer conditioning, noise resampling or output repair.
- [ ] Actual tiny native forwards compared to independent manual loop, stable parameters/RNG, forced native exception cleanup. Isolated streams and restored thread settings. Scoped `.venv-ml/Scripts/python.exe -m unittest discover -s tests -p test_native_generation.py -v`, cap90s. Record exact evidence/API/hashes; independent task review and fixes.

Preflight: generation consumes existing host APIs, scorer consumes generation/result/task verifier, seed helper is independent and pure. One owner, no overlapping source files. Component completion does not complete frozen pretrained benchmark, per-example run records, GPU timing or G3.
