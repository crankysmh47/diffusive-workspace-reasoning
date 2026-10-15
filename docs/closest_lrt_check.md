# Closest LRT reference check

2026-10-15. Targeted primary-paper reading, not an exhaustive novelty audit or reproduction. This identifies the September 2026 LRT paper separately from the older MobiusDai repository in the initial review.

[Latent Recurrent Thoughts, arXiv:2609.01117v1](https://arxiv.org/html/2609.01117v1) uses a task-specific proposer, recurrent refinement and a frozen decoder. Sections 3.2–3.5 describe learned latent queries, two recurrent states, bounded residual corrections, soft-token injection, staged training and truncated-gradient unrolling. Its limitations explicitly leave cross-task and difficulty transfer untested. The paper therefore already covers frozen-decoder recurrent latent reasoning; that mechanism cannot be DWR's novelty claim.

Our inference for DWR: retain the shared source-trained module, frozen transfer and matched EX–IDX contrast as the research question. Full recurrent backpropagation, Gaussian exchangeable initialization and set readers are implementation distinctions, with different costs; they do not establish superiority or novelty. A custom matched sequential control is not an LRT reproduction.

Before external baseline implementation, pin the authors' code revision and audit supervision, interface, training schedule and complete costs. This reading clears no model gate and does not complete the broader closest-method audit in [todo.md](../todo.md). It changes no registered experiment, split or architecture.
