# DWR implementation checklist

Source: [research plan v2](docs/diffusive_workspace_reasoning_project_plan_v2.md). Check items only when their outputs and verification exist. Repository setup does not imply a working research implementation.

## 0. Repository setup

- [x] Preserve the complete v2 plan in `docs/`.
- [x] Add README, checklist, contribution instructions, and draft prototype specification.
- [x] Add a minimal installable Python package and ignore generated artifacts/secrets.
- [x] Create the private GitHub repository and push the initial commit.

## 1. Verified task and immutable splits — next milestone

- [ ] Fix the first family as 4×4 mini-Sudoku with 2×2 subgrids; document symbols, empty-cell encoding, prompt, answer format, and primary metric before tuning.
- [ ] Implement a constraint verifier: complete assignments, symbol range, rows, columns, subgrids, and preserved clues. Test valid, incomplete, malformed, duplicate, and clue-violating outputs.
- [ ] Implement a deterministic reference solver and uniqueness check that stops after two solutions. Verify unsatisfiable, unique, and ambiguous puzzles.
- [ ] Generate valid complete boards, remove clues under a seeded procedure, and retain uniquely solvable puzzles.
- [ ] Record generator version/seed, source-template ID, solution, clue count, propagation rounds, branch points, search depth, and backtracking nodes.
- [ ] Define measured difficulty buckets; check that the 4×4 generator actually produces useful variation. Do not invent a hard bucket if solver statistics do not support one.
- [ ] Prevent identical and symmetry-equivalent/near-template instances crossing splits; document canonicalization and family grouping.
- [ ] Freeze train/validation/test manifests plus held-out harder instances, hashes, and split IDs. Audit intersections and never retune the test set after inspection.

Checkpoint: reproducible generation, independently verified solutions, difficulty distribution, and a passing leakage audit.

## 2. Frozen backbone baseline

- [ ] Select and record a 1.5B–2B decoder-only base checkpoint, revision, terms, tokenizer, and dependencies.
- [ ] Verify installation and memory on Windows with the target CUDA/GPU environment; record dtype, precision, maximum sequence length, batch size, and peak VRAM.
- [ ] Fix decoding policy and benchmark the unaugmented frozen backbone using the same prompts and verifier.
- [ ] Extract prompt-only hidden states at a fixed early/middle layer shared by all core methods. Prevent answer-token leakage into X.

Checkpoint: repeatable baseline accuracy, latency, memory, and documented prompt representation.

## 3. Set interface and one-shot exchangeable workspace

- [ ] Insert trainable set cross-attention adapters; fix adapter layers, dimensions, initialization, and optional gates across methods.
- [ ] Freeze original backbone parameters while preserving autograd through the frozen decoder operations used after trainable adapters.
- [ ] Sample iid Gaussian slots; implement the residual workspace update with self-attention, prompt cross-attention, and MLP. No fixed slot index embeddings in the exchangeable variant.
- [ ] Use K=1 and final-answer cross-entropy only; mask prompt and padding tokens in the loss.
- [ ] Verify shapes, gradient flow to workspace/interface, and absence of gradients/updates to pretrained parameters.
- [ ] Check update permutation equivariance and decoder permutation invariance at appropriate numeric tolerances.
- [ ] Overfit a tiny verified batch as an integration check, then establish nontrivial validation learning.
- [ ] Ablate/zero/noise-replace the workspace and record adapter magnitudes to diagnose latent bypass before considering interventions.

Checkpoint: functioning answer-only training, architectural checks, and a meaningful validation curve.

## 4. Recurrent and indexed core experiment

- [ ] Reuse shared residual-update parameters at K=2,4,8; check gradients through all updates and log latent norms.
- [ ] Train separate fixed-K checkpoints for K=1,2,4,8 using matched data, interface, and recorded budgets.
- [ ] Add indexed initialization `Z0[i] = epsilon[i] + embedding[i]`; keep other settings matched and report the extra parameters.
- [ ] Compare one-shot, exchangeable recurrent, and indexed recurrent variants across difficulty buckets.
- [ ] Implement the variable-K training protocol with support {1,2,4,8}; evaluate the same checkpoint at {1,2,4,8,12,16}, labeling seen and unseen horizons.
- [ ] Target at least three seeds for a claimed effect; report mean/spread and investigate instability.

Checkpoint: assess held-out iterative benefit before expanding architecture scope. Negative findings remain reportable.

## 5. Measurement and reporting — integrate into every run

- [ ] Save commit, exact config, backbone revision, dataset hash/version, generator version, split IDs, seed, N/d/K, fixed/sampled K, slot identity, and environment/dependency versions.
- [ ] Log total/trainable parameter counts, train/validation loss, structured accuracy, verifier pass rate, hard-bucket accuracy, latency, peak memory, inference FLOPs, and training compute where practical.
- [ ] Define warmup, CUDA synchronization, batch/token lengths, device, and repeats for latency; document FLOPs estimation scope.
- [ ] Save resumable checkpoints with optimizer, scheduler, RNG state, training step, and config/data identities; verify resumed training on a small run.
- [ ] Produce accuracy/loss learning curves, accuracy by K (seen/unseen separated), accuracy versus FLOPs/latency, and indexed/exchangeable difficulty comparisons.
- [ ] Keep same-checkpoint inference scaling separate from separately trained architecture-at-budget tables.

## 6. Later controls — after the recurrent result is understood

- [ ] Add an unshared deeper feedforward workspace; report both parameter and compute differences rather than assuming simultaneous exact matching.
- [ ] Add ordered sequential latent computation with the same set cross-attention decoder interface.
- [ ] Add a deterministic sequential-depth task with controlled chain depth and a proper verifier.
- [ ] Evaluate transfer and compute-matched alternatives; do not require a preconceived crossover.
- [ ] Add a faithful external LRT-style comparison only if feasible, identifying the additional architectural axes it changes.

## 7. Flow matching — gated on an understood recurrent teacher

- [ ] Measure terminal-latent and output diversity across noise seeds, accounting for slot permutations.
- [ ] Freeze teacher, trained decoder interface, and backbone; version the teacher targets and coordinate system.
- [ ] Implement straight-line conditional flow matching, with explicit pairing/permutation handling. Do not claim optimal transport without an OT coupling.
- [ ] Compare teacher-quality retention, NFE, latency, memory, and FLOPs. Treat NFE as numerical/inference efficiency.
- [ ] If adding answer-loss fine-tuning, give the recurrent comparator comparable extra optimization.

## 8. Scale only when evidence supports it

- [ ] Consider 6×6 tasks, public math data, code tasks, and a larger backbone after controlled results justify the cost.
- [ ] Split public data by underlying problem, keeping all trajectories together.
- [ ] Revisit anti-bypass losses, posterior encoders, or gradient gating only if diagnostics justify them.
- [ ] Choose the final paper claim/title from the mechanism supported by the experiments, including negative results.
