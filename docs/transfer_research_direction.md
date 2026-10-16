# Main direction: transferable computation with exchangeable workspaces

2026-10-16 status update: native integration/run/export/evaluation/CPU-resume infrastructure and the [actual pretrained GPU feasibility](pretrained_gpu_feasibility_result.md) pass. Device-aware frozen reference is the next operational stage; no task-learning or transfer result exists. G3/G4 and confirmatory precision remain incomplete. This note supersedes older status summaries below without changing the binding scientific protocol.

Decision date: 2026-10-07. User selected reusable computation across tasks as the main direction. Gate status updated 2026-10-14: the separately registered [five-stratum study](five_stratum_study_result.md) passed qualification/full structural G2; every prior NO-GO/unassessed result remains unchanged. Fixed-N [tensor components](fixed_workspace_core_verification.md) are independently reviewed and verified on CPU. Native backbone integration, training infrastructure, GPU feasibility and prospective confirmatory precision remain outstanding. No model training or transfer result exists. See [execution state](execution_state.md). This status note does not amend v2.2. Other directions remain parked in [spare_research_ideas.md](spare_research_ideas.md).

## Central question

Current continuation: the actual shared residual EX/IDX workspace and invariant set reader pass strict CPU permutation/gradient/mask checks; implement the [native host](native_backbone_design.md) and run/evaluation/resume machinery next. Synthetic frozen-tail fixtures do not certify native decoder integration, complete resume, learning or GPU memory. Those checks and non-null training registration follow; fixed-N transfer remains the first scientific model study.

Can one trained, exchangeable latent workspace attached to a frozen language model reuse learned constraint computation on unseen compositions and task structures without task-specific retraining?

Exchangeability is the main architectural hypothesis supporting this question. Removing fixed slot identities may encourage content-dependent binding and reusable operations instead of position-specific roles. This is a testable hypothesis, not a consequence of permutation equivariance. IDX may transfer equally well; preserve that outcome.

Train one shared workspace and one shared decoder interface on a mixture of source tasks. At evaluation freeze all learned parameters, including adapters, projections, gates and format embeddings. Use one validation-selected checkpoint across target families. No target-specific adapters, finetuning, checkpoint selection or answer-conditioned initialization. Few-shot adaptation, if studied later, is a different experiment.

## What counts as transfer

- **Compositional transfer (first target):** known primitive rules appear in combinations or dependency structures excluded from training. Every primitive must occur in training; a new name or surface template is insufficient.
- **Size/depth transfer (secondary):** more variables, constraints or longer compositions than trained. Keep it separate from novel compositions at familiar sizes.
- **Cross-family transfer (stronger follow-up):** a structurally distinct task family expressed through the shared representation, with no updates on that family. Success on held-out compositions alone does not establish arbitrary cross-domain reasoning.

Report each regime separately, including an ID test and a frozen-backbone reference. Training on a mixture then testing familiar task families is multitask generalization, not automatically transfer to an unseen family. New rule semantics require an explicit additional study; the initial target is reuse of learned primitives.

## First task-design gate

Design a small finite-domain constraint grammar before building a large Sudoku dataset. Candidate primitives are equality, inequality and modular-addition relations over named variables with explicit domains and given assignments. These are candidates, not a frozen benchmark: first verify that their compositions produce uniquely solvable, nontrivial cases and distinguish the desired generalization regimes. Use strict full-assignment answers and an independent constraint verifier. Do not introduce learned solvers to generate ground truth.

The task specification must fix primitive semantics, serialization, source mixtures, held-out composition signatures, structural canonicalization, generation lineage and the exact train/validation/test assignment algorithm. Audit variable-renamed, constraint-reordered and isomorphic copies, not just textual duplicates. Canonicalize with respect to actual rule semantics; arbitrary value relabeling need not preserve modular arithmetic. Keep related instances/augmentations in one partition. Match or stratify length, variable count and measured difficulty so composition shift is not simply a length shift.

Choose development-held-out compositions disjoint from final held-out composition signatures for validation. Freeze the final target definitions before architecture tuning. Record primitive/composition coverage and group counts. All models receive identical input information, including any explicit rule definitions. No target solution, solver trace or hidden graph metadata enters X. Generator metadata is available to audits and analysis, not extra model conditioning.

4×4 Sudoku is an optional small correctness fixture. The earlier mandatory 6×6 study is removed from the critical path: larger Sudoku alone cannot establish the new central claim. A small exhaustive enumerator for the selected grammar is the immediate solver oracle. Solver/generation/leakage/statistical checks from the protocol still apply.

## First decisive comparison

Begin at fixed N and K: provisional N=64, d=768, K=4, subject to the same memory probe for every method. The primary architectural contrast is EX minus IDX on the equally weighted mean valid-solution rate over preregistered held-out composition groups. Register group weights, practical delta, uncertainty and ID-performance guardrails before confirmatory training. Report group-level results as well as the macro average.

Also include K=0 trained noise-memory interface and EX K=1 controls. Bring measured-cost feedforward and sequential controls into the main transfer study before claiming a joint-workspace advantage. Train each comparator once on the same source mixture, with matched selection/tuning opportunity and recorded costs. Full external reproductions require a method-level review and pinned revisions; an adapted matched-interface baseline is not a faithful reproduction.

Interpretation: robust OOD performance shows transfer within the specified grammar. An EX advantage over IDX supports an identity-prior effect. An advantage over the other controls supports a stronger joint-computation explanation. If every model transfers similarly, there is no evidence that exchangeability caused the benefit. If ID learning fails, OOD failure is not an interpretable transfer result. Predefine a nontrivial absolute transfer criterion relative to the task/reference controls as well as the EX-IDX effect; two failing models are not a successful transfer result.

K=4 vs K=1 becomes a secondary iteration diagnostic, replacing the old primary ID contrast. Defer a full K sweep and variable-K study until this fixed-size transfer comparison is interpretable. A small task-structure analysis may explain transfer failures; a standalone topology/mechanism project remains parked.

## Exchangeability is not redundancy

Slots are scratch-space vectors, not reasoning blocks or predefined textual steps. The shared reasoning block updates them. They start as IID random noise rather than blank text; their learned contents need not map one-to-one to constraints or steps. Increasing N increases workspace capacity; increasing K repeats the computation over that workspace.

For any permutation P of whole slot vectors, EX requires `F(PZ, X) = P F(Z, X)` and `D(x, PZ) = D(x, Z)`. Slot order carries no fixed identity. The slots can still carry different information and specialize by content. Replacing, deleting or averaging a slot is not a permutation and need not preserve the answer. IID noise supplies symmetry breaking without assigning persistent index roles.

Slot permutation is distinct from renaming input variables, reordering constraints, or changing the number of slots. Each requires its own test. In particular, permutation symmetry does not prove semantic decomposition into one slot per variable, cross-task transfer, or successful evaluation at a new N.

## Variable slot count: a supporting hypothesis, staged

1. **Fixed-N control first.** Establish the transfer comparison without changing workspace size or adding allocation losses.
2. **Externally selected N using one checkpoint.** After the fixed-N result is interpretable, register a secondary study training a shared model across a finite set of N values, then evaluate seen and unseen counts at fixed K. Illustrative training counts are {16,32,64}, with unseen interpolation N=48 and extrapolation N=96; these are not launch settings until memory/cost probes and registration. Separately label a fixed-N-trained model tested at new N. Do not mix those protocols.
3. **Input-adaptive N or slot creation/removal during recurrence.** Parked pending evidence that variable N gives a useful accuracy/cost frontier. These require an allocation policy and extra training choices; they are not implied by accepting different tensor sizes.

For step 2, projections and update weights must be shared across slots and independent of N; no flattened fixed-N layers or per-slot decoder parameters. Test masked padding against the corresponding unpadded set and permutation behavior at every supported N. Choose N before seeing the answer; a verifier-selected best-N sweep is additional search with additional cost. Freeze a common validation selection rule averaged across registered training counts, not a different checkpoint per test N.

Keep task size and N as separate evaluation axes: vary N on the same examples, then examine transfer to larger tasks. Compare against fixed-N models, a large fixed workspace and increasing K at comparable measured cost. Dense slot self-attention has a quadratic N term; N×K is not a complete FLOPs model. Include initialization, padding, prompt encoding and decoding in measured costs. More slots can dilute attention or destabilize learned computation; accepting a larger tensor is not capacity generalization.

The fixed-N EX-IDX study isolates learned identity. A learned table limited to training slot indices is not a fair indexed extrapolation baseline: register how new identities are generated/trained or restrict that comparison to trained indices. Do not claim EX superiority merely because the baseline cannot instantiate the requested shape.

## Prior work and intended contribution

- [Slot Attention](https://arxiv.org/abs/2006.15055) already uses exchangeable slots and studies compositional generalization in object-centric learning. Its [paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf) also studies more objects/slots. DWR is not inventing exchangeability or variable cardinality.
- [AdaSlot](https://arxiv.org/abs/2406.09196) dynamically selects slot number for object discovery. Adaptive N alone is not a new idea.
- [Latent Recurrent Thoughts](https://arxiv.org/html/2609.01117v1) is a close frozen-decoder recurrent-latent comparison. Its stated per-task training and untested cross-task/difficulty transfer motivate investigating shared reusable computation. This is a different LRT reference from the candidate in the initial review; disambiguate names explicitly.

The candidate contribution is evidence that exchangeable workspaces learn reusable computation, with controlled compositional/cross-family transfer and, if supported, useful capacity scaling with N without target-task retraining. No first-of-its-kind or top-tier publication claim is established by this design. Complete the closest-method audit before expensive runs.
