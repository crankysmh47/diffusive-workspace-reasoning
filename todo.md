# DWR research and implementation checklist

Scientific authority: [v2 plan](docs/diffusive_workspace_reasoning_project_plan_v2.md), amended by the binding [v2.2 protocol](docs/research_protocol.md). This is the sole live checklist. Check an item only when its output and recorded verification exist. Planning does not count as implementing a gate.

## 0. Repository setup

- [x] Preserve the supplied v2 plan in `docs/`.
- [x] Add README, checklist, contribution instructions, and draft prototype specification.
- [x] Add a minimal installable Python package and ignore generated artifacts/secrets.
- [x] Create the private GitHub repository and push the initial commit.

## 1. Transfer task specification and correctness — next milestone, CPU only

- [ ] Write `docs/task_spec.md` from the [main direction](docs/transfer_research_direction.md): exact finite-domain primitive semantics, source-task mixture, held-out composition signatures, strict prompt/answer syntax and verifier metric. Candidate primitives are not yet a validated benchmark.
- [ ] Fix the distinctions among composition, size/depth and cross-family transfer; define development target signatures separately from untouched final target signatures.
- [ ] Implement board-independent constraint parsing/verifying under `dwr/tasks/`; test malformed assignments, unknown variables, missing/duplicate values, domain violations and every primitive's semantics in `tests/`.
- [ ] Implement a deterministic reference solver and independent exhaustive tiny-instance oracle; cross-check zero/one/multiple solutions, counting uniqueness only up to two.
- [ ] Implement seeded generation with source lineage and separate first-solution/uniqueness statistics; test regeneration, annotation exclusion and answer validity.
- [ ] Record nonzero test discovery and actual results for `python -m unittest discover -s tests -v`. Optional 4×4 Sudoku fixtures must not become a mandatory larger-Sudoku experiment.

Gate G1: an explicit transfer question, task contract and independent CPU correctness evidence.

## 2. Composition suitability and immutable transfer splits

- [ ] Implement and test canonical structure/lineage grouping, including variable renaming and constraint-order changes; define which transformations preserve each primitive's semantics.
- [ ] Audit source primitive coverage, withheld composition coverage, isomorphic/augmented duplicates and generation lineage intersections in `docs/task_suitability.md`.
- [ ] Check nontrivial uniquely solvable instances and independent group counts; match or stratify input length, size and solver difficulty across the composition comparison.
- [ ] Use development data to choose sample/group counts and prospective precision; register an absolute transfer criterion, ID-learning guardrails and a practical EX-IDX effect size.
- [ ] Freeze source train/validation/ID-test plus composition-transfer validation/test, with distinct development/final composition signatures. Keep size/depth and later cross-family tests separately identified.
- [ ] Commit small generator/split manifests and hashes under `configs/splits/`; keep generated records in ignored `data/`. Verify regeneration and model-input leakage checks.

Gate G2: the benchmark genuinely withholds compositions while exposing their primitives. No target-answer tuning or relaxed grouping to fill splits.

## 3. Run infrastructure and frozen-backbone reference

- [ ] Implement resolved run manifests/status logging under `dwr/experiments/` with strict configuration validation, source/data identities, separate RNG streams, environment and unique run directories; add CPU tests for invalid/missing config and artifact identity.
- [ ] Select a candidate base backbone and tokenizer, pin revisions/terms and exact environment; add ML dependencies only here. Check frozen encoding/decoding memory; keep workspace/adapters provisional until the full training probe in G4.
- [ ] Fix prompt-only encoding with adapters disabled and cache identity; verify answer changes cannot affect X, and padding/batching do not alter valid prompt states.
- [ ] Implement strict autoregressive evaluation and per-example records under `dwr/evaluation/`; fix greedy decoding, output cap, parser, single-draw noise seeds and invalid-output accounting.
- [ ] Benchmark the unaugmented frozen backbone with the same prompt/answer rules on development/validation data; record valid-solution rate, latency and memory.
- [ ] Implement checkpoint identity and RNG/sampler-state serialization with CPU tests; validate complete learned-model resume after the optimizer and adapters exist in G4.
- [ ] Establish warmup/synchronization/repetition settings and FLOPs scope; include prompt encoding, workspace and answer decoding, with cache policy fixed.

Gate G3: run identities, repeatable frozen-backbone evaluation and measurement infrastructure. Full training feasibility remains a G4 prerequisite to the pilot.

## 4. One-shot workspace and interface controls

- [ ] Implement trainable set cross-attention adapters and the shared residual block `Z_next = Z + F(LN(Z), X)` under `dwr/models/`; fix layers, dimensions, initializations and gates across methods.
- [ ] Train only new parameters using shifted answer-token cross-entropy, masking prompt/padding tokens; retain gradients through frozen decoder computations and all workspace updates.
- [ ] Add tensor tests for EX equivariance, decoder invariance, causality, masks, shapes, finite gradients and frozen weights unchanged after an optimizer step. Report CPU versus GPU checks separately.
- [ ] Implement K=4 shared recurrence for the pilot and full planned K support for the memory probe; test gradients through every update and finite latent norms.
- [ ] Measure full forward/backward/optimizer memory at maximum planned training K, sequence length and batch, plus any separately registered extrapolation horizon; freeze model/workspace/adapter/precision settings consistently across methods before the pilot.
- [ ] Verify complete optimizer/scheduler/scaler/RNG/sampler resume against uninterrupted tiny training with stated tolerances.
- [ ] Overfit a tiny verified batch; diagnose integration failures before development comparisons.
- [ ] Implement separately trained K=0 noise-memory adapter control and K=1 EX using the same initialization/interface; record parameter differences.
- [ ] Run a bounded one-seed validation pilot of K=0, EX K=1 and EX K=4. Freeze pilot step/GPU-hour limits and retry allowance first; save learning curves and latent norms.
- [ ] Measure workspace zero/noise/cross-example shuffle ablations and adapter magnitudes; distinguish distribution-shift diagnostics from trained controls.

Gate G4: interpretable learning or a documented failure/no-go. Do not force anti-bypass losses into the first model.

## 5. Registered fixed-size transfer study — main experiment

- [ ] Complete the closest-method audit before comparative training, explicitly including Latent Recurrent Thoughts (arXiv:2609.01117), Slot Attention and relevant compositional-generalization work; distinguish custom controls from faithful reproductions.
- [ ] Freeze a study registration: primary EX-IDX macro valid-solution-rate difference on held-out composition groups, absolute transfer and ID criteria, group weights, practical delta, uncertainty, seeds, finite tuning/training costs and stop rules.
- [ ] Train one EX and one IDX workspace/interface on the same source mixture at fixed N/d/K (provisional 64/768/4), matching all settings except learned initial identities and reporting their extra parameters.
- [ ] Train K=0 noise-memory interface and K=1 EX controls on the same source mixture. Freeze every learned component for target evaluation; no target-specific modules or checkpoint selection.
- [ ] Specify and implement matched-interface feedforward and sequential controls using development-only cost matching; include them before claiming a joint-workspace transfer advantage.
- [ ] Complete at least three independent training seeds; freeze one source/development-validation-selected checkpoint per method for all final target groups. Report failures and insufficient-seed studies explicitly.
- [ ] Run the frozen test batch with test-access logging. Report absolute performance, ID/OOD gaps, each composition group's results, macro effects and family-aware uncertainty; do not infer success from a difference between two failing models.
- [ ] Evaluate separately registered size/depth and cross-family targets if pursuing those claims. A new composition is not automatically a new domain; no target-task updates are allowed for the frozen-transfer claim.

Gate G5: an interpretable conclusion about reusable computation and whether exchangeability improves it. Equal transfer across EX/IDX is not an exchangeability advantage; failed ID learning does not establish failed transfer.

## 6. Variable slot count — supporting experiment after G5

- [ ] Register fixed-N-trained/new-N evaluation separately from variable-N training. Freeze seen/interpolated/extrapolated counts after memory probes; illustrative counts in the direction document are not executable settings.
- [ ] Make projections/updates independent of N; test slot permutation and padding equivalence for every supported size, without changing the residual update or leaking answer information.
- [ ] Hold K fixed while varying N on the same examples, then cross with task size. Use one checkpoint across N selected by a registered validation aggregate; do not tune or select N from test answers.
- [ ] Compare against fixed-N training, a larger fixed workspace and increased K at measured comparable cost; report attention/padding overhead and whole-system accuracy/cost frontiers.
- [ ] Register a fair indexed-size comparison where appropriate. A finite identity table's inability to instantiate unseen indices is not evidence that EX learned superior reasoning.
- [ ] Decide whether variable N supports useful capacity generalization. Leave learned allocation and slot creation/pruning parked unless a new study is justified.
- [ ] Run wider fixed-K or variable-K diagnostics only through a separate bounded registration when needed to interpret the transfer result.

Gate G6: demonstrated useful generalization across workspace sizes, or a bounded negative result. Shape compatibility and permutation invariance alone do not pass this gate.

## 7. Reproducible research report

- [ ] Generate learning curves, ID/OOD accuracy, seen/unseen K curves, cost frontiers and all-seed tables from saved per-example outputs using versioned analysis code.
- [ ] Report training/tuning/failed-run cost, inference measurement settings, statistical limitations, leakage-audit scope and pretraining-contamination uncertainty.
- [ ] Write the actual supported claim, methods, related work, positive/negative results and limitations; link each table/figure to immutable run/checkpoint/data identities.
- [ ] Archive artifacts with hashes and storage locations; document regeneration/train/evaluate/report commands and exact environment.
- [ ] Reproduce data hashes and at least one representative evaluation in a fresh environment. Record whether performed by an agent or an independent person; do not claim independent replication from self-checks.

Gate G7: a reproducible technical report, not guaranteed venue acceptance. Choose title and contribution from evidence.

## 8. Parked alternatives

See [spare_research_ideas.md](docs/spare_research_ideas.md) for task-structure theory, persistent repair, learned slot allocation, flow distillation and adaptive stopping. These are not active milestones. Public datasets and larger backbones also require a new evidence-backed scope decision; preserve problem-level grouping and usage-term checks if activated.

No implementation, data, model, run-infrastructure or experimental gate above is complete at this revision.
