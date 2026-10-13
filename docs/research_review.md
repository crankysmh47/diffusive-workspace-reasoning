# Research-readiness review

Date: 2026-10-07. Scope: all tracked project files and the complete v2 narrative, plus a targeted check of related primary sources. This is a design review, not an experimental result or an exhaustive novelty search.

## Current status amendment

The user subsequently selected [transferable computation](transfer_research_direction.md) as the main objective, now governed by protocol v2.2. The v2.1 review below is historical where its Sudoku-first sequence or primary contrast differs. A closer prior-work check also identified [Latent Recurrent Thoughts](https://arxiv.org/html/2609.01117v1), which was missing from the initial map: frozen-decoder recurrent latent refinement already overlaps the original project. Exchangeable slots and adaptive counts also have precedents in [Slot Attention](https://arxiv.org/abs/2006.15055) and [AdaSlot](https://arxiv.org/abs/2406.09196). None of these discoveries establishes that the new transfer proposal is novel; a full closest-method audit remains required.

## Assessment

The direction is coherent: isolate iteration, slot identity and computation style; hold decoder access fixed; start with verifiable tasks; keep flow conditional. The repository is only a package bootstrap (`dwr/__init__.py`); generator, solver, training and evaluation do not exist. The old status “ready to implement” was too broad: task correctness work can start, while confirmatory experiments still need data and hardware gates.

The plausible contribution is a controlled study of exchangeable joint workspaces attached to a frozen LM, with useful positive or negative evidence under explicit costs. Iteration, latent reasoning, set attention, and flow matching alone are not new research contributions. A distinctive combination is a hypothesis to investigate, not proof of novelty.

## Findings and deliberate changes

| Finding in v2/bootstrap | Resolution in v2.1 | Reason |
| --- | --- | --- |
| 4×4 first experiment plus strict template/symmetry separation | 4×4 correctness fixture; 6×6 data-feasibility gate before modeling | Only two completed-grid symmetry classes cannot populate three family-disjoint splits |
| “Near-template” leakage and “hard” buckets undefined | Canonical completed-board grouping, generation lineage, puzzle audit, fixed solver statistics and development-derived cutoffs | Split guarantees and difficulty claims must be implementable and testable |
| Frozen baseline has no task-trained interface | Add separately trained K=0 noise-memory control | Interface adaptation can explain gains over the frozen baseline |
| Fixed-K runs described as automatically fair architecture-at-budget evidence | Distinguish data-exposure and compute-budget comparisons | Equal examples/steps do not equal training or inference FLOPs |
| Feedforward displayed without the residual used elsewhere | Preserve residual form; change only weight sharing first | Otherwise the baseline changes two mechanisms |
| “Persistent” indexed embeddings, prompt-state extraction, variable-K schedule ambiguous | Add identities once; separate prompt-only adapter-disabled encoding; uniform sampled terminal K | Prevent hidden architectural and leakage differences |
| Three seeds and mean/spread without a test protocol | Register primary contrast, practical delta, clustered uncertainty, checkpoint selection and test access | Avoid selection bias, pseudoreplication and treating a null as equivalence |
| Logging appears after training in TODO | Run identity and resume/measurement checks precede comparative training | Reproducibility cannot be reconstructed reliably after results |
| Sequential/feedforward only after positive results; vague stop/rethink loop | Mandatory for their corresponding claims; bounded diagnostics and reportable negatives | Avoid assuming success and endless retuning |
| Flow endpoint pairing unspecified | Teacher-induced pairing with joint permutations; simple distillation control | Reproducible latent coordinates and a fair efficiency attribution |
| Duplicate “first-week” checklist and proposed code under ignored data paths | One live TODO; source under `dwr/`, small manifests under `configs/` | Prevent inconsistent progress and ignored source code |

These are explicit amendments to the original plan, documented in [research_protocol.md](research_protocol.md). The original narrative is retained with amendment notices. Historical bootstrap checks in [bootstrap_review.md](bootstrap_review.md) are not evidence that the scientific design had no gaps.

## Related-work starting map

Sources checked on 2026-10-07. The descriptions below use the linked paper abstracts and, for LRT, the authors' repository overview. They are a starting map, not a full reproduction assessment; numerical results are intentionally not imported as DWR evidence. Before implementing an external baseline, read the methods, pin paper/code revisions, verify training/supervision and compute, and record deviations.

| Primary source | Relevance and consequence for DWR |
| --- | --- |
| [Coconut: Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) | Continuous hidden-state feedback already exists. Distinguish DWR's joint workspace, frozen original backbone, interface and answer-only supervision; a custom sequential control is not a faithful Coconut reproduction. |
| [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171) | Recurrent latent depth and extra inference computation are prior art. “More latent steps help” alone is insufficient novelty. |
| [Set Transformer](https://arxiv.org/abs/1810.00825) | Attention over sets and permutation invariance are established tools. Test the property; do not claim its invention. |
| [Perceiver](https://arxiv.org/abs/2103.03206) | Iterative attention through a latent bottleneck is a close architectural precedent. DWR must explain its frozen-language-model and experimental distinctions. |
| [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734) and [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) | Recurrence on verified puzzles has strong prior work. These are external context, with different model/training assumptions; establish simple matched controls before expensive reproductions. |
| [LRT authors' repository: Rethinking LLM Reasoning](https://github.com/MobiusDai/LRT) | The overview describes latent embeddings concatenated back into a frozen base model. v2 did not identify its intended “LRT” source. This is a candidate reference; confirm the exact intended paper/revision before reproduction, and do not assume it is the matched sequential baseline. |
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | Straight-line flow matching is an established generative method. DWR's later question is controlled teacher distillation/efficiency; flow and diffusion must not be conflated. |
| [Cornell: the 4×4 Sudoku case](https://pi.math.cornell.edu/~mec/Summer2009/Mahmood/Four.html) | 288 completed grids and two symmetry classes expose the small-task split problem. This does not count clue-mask puzzles or establish 6×6 suitability. |

## Remaining decisions and when to resolve them

1. **Before research data freeze:** establish that the 6×6 candidate supplies enough independent families and a measurable harder regime; implement and audit canonicalization; choose sample counts using the intended precision. These need task code and a data audit.
2. **Before learned-model comparisons:** choose backbone/revision, extraction and adapter layers, dtype, attention implementation and batch size from a full training-memory probe. No hardware feasibility is currently demonstrated.
3. **Before confirmatory training:** freeze practical effect size, seed/sample counts, training/tuning budgets, GPU-hour cap and diagnostic allowance in an executable registration. Null fields in the prototype deliberately block this gate.
4. **Before H2 or efficiency claims:** specify the sequential implementation and measured feedforward/sequential budget matches. The current equation sketches are not complete baseline specifications.
5. **Before a novelty claim/submission:** complete a targeted full-method literature comparison, including identifying the intended LRT reference, and position the result that actually survives. No venue or guaranteed publishability is assumed.

These are scheduled empirical/design decisions, not reasons to add diffusion or redesign the whole project now. The immediate next milestone is the CPU-only task specification, verifier and reference solver with an independent small-grid oracle.

## Verification boundary

This revision changes documentation and the draft configuration only. No datasets were generated, models downloaded, or training launched. Documentation/configuration checks are recorded after the edits; solver correctness, leakage control, gradients, GPU feasibility, research effects and independent reproduction remain unverified.

Checks executed on 2026-10-07:

- `git diff --check`: passed after removing new trailing whitespace and normalizing TODO line endings.
- Inline Python standard-library audit (`python -`, using `pathlib`, `json`, `tomllib`, `re` and `subprocess`): all 19 local Markdown links across seven documents resolve; JSON/TOML parse; protocol identity, task role, horizons, seed count and intentionally incomplete registration agree with the revised scope.
- Compared package source, packaging metadata and ignore rules against `git show HEAD:<path>`: unchanged. The four historical completed setup items remain the only completed TODO items.
- No research test suite exists yet. No solver, leakage, gradient, permutation, GPU, training or scientific-result check was run or claimed.
