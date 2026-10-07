# Working in DWR

Read `README.md`, `todo.md`, and the relevant sections of the v2 plan before changing scientific behavior. The plan is authoritative; record deliberate departures with their rationale.

- Implement the next unchecked milestone. Do not introduce diffusion, posterior encoders, dependency-gated optimization, or public datasets before their prerequisites.
- Keep changes small. Use Python's standard library for generation, reference solving, manifests, and logging where practical. Add ML dependencies only when needed.
- Put implementation in `dwr/`; create modules as they become real, not placeholder architectures.
- Preserve matched prompt encoding, stochastic initialization, decoder interfaces, splits, and reporting across comparisons.
- Preserve the residual update equation and explicit exchangeable/indexed distinction.
- Frozen parameters do not mean all decoder computation belongs in `no_grad()`: gradients must reach the newly inserted interface and workspace.
- Never condition the v1 workspace on the ground-truth answer. Never alter held-out splits based on results.
- Add meaningful correctness checks for solvers/verifiers, leakage, gradient flow, and permutation behavior when those implementations land. Use `unittest` initially; no GPU training required for basic task checks.
- Log reproducibility metadata and report actual test/check outcomes. Update `todo.md` only for completed and verified work.
- Keep generated datasets, checkpoints, model weights, credentials, and run artifacts out of Git. Commit small configurations and split manifests/hashes, not large data.
- Support Windows paths; avoid mandatory Bash scripts, symlinks, or platform-specific ML dependencies.
- Do not choose an open-source license or change repository visibility without owner instruction.
