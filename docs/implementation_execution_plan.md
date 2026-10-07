# Transfer v1 execution plan

> Execute via subagent-driven-development. GPT-6 Luna performs implementation; task reviews are separate. User authorized continuous execution through evidence gates, including task specification, CPU implementation/audit and then fixed-size workspace/control if prerequisites pass.

Goal: an independently checked finite-domain composition benchmark, followed by EX/IDX model components only when data and environment gates support them.
Spec: docs/research_protocol.md and docs/transfer_research_direction.md; Task 1 fixes docs/task_spec.md before production task code.

## Global constraints
- Python >=3.11,<3.13. CPU task generation/solver/verification/manifests use the standard library.
- Source under dwr/, unittest under tests/, generated artifacts ignored under data/ or runs/, small manifests/configs under configs/.
- Never feed answers, solver statistics or split identifiers into model inputs. Never tune on final test outcomes.
- Shared residual update Z_next = Z + F(LN(Z), X). IID Gaussian initialization, identities added once for IDX, matched decoder access.
- Freeze original backbone parameters without suppressing gradients through operations after trainable interfaces.
- Preserve failed evidence; fail rather than silently dropping cases or changing scientific settings.
- No GPU training until data suitability, provenance, backbone and full forward/backward memory gates pass.

## Tasks
1. Fix task spec, schema/interfaces, composition/source split definitions, initial CPU audit configuration and explicit gate thresholds. No code. Check independent small oracle strategy and nontriviality before fixing generation choices.
2. Implement immutable task structures/strict serialization/verifier and independent exhaustive oracle with red-green unittest checks.
3. Implement deterministic CSP reference solver with capped solution counting, first-solution and uniqueness statistics, and oracle comparison tests.
4. Implement seeded generator, canonical structure/content identities, family-safe development split generation and bounded suitability audit; log source/env/config identities before work, hashes and coverage after work. Preserve failures. Check meaningful nontriviality and absent held-out compositions independently.
5. Evaluate G1/G2 evidence. If failed, retain no-go and stop architecture work. If passed, inspect ML environment, pin dependencies and implement needed provenance infrastructure/fixed-N EX/IDX components plus CPU gradient/permutation tests; do not claim a frozen-LM end-to-end implementation from tensor components alone.
6. Broad review, actual verification, durable execution-state handoff at the first unmet scientific/environment gate. Update TODO only for verified outputs.

## Review focus
Strict numeric types (bool is not int); unknown keys and malformed assignments; independent truth evaluation; given anchors and disconnected/free variables; commutative addition semantics; renamed/reordered duplicates; deterministic local RNG; audit gate failure not disguised as success; whole-branch evidence consistency.
