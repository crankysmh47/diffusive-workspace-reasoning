# DWR research protocol — v2.2

Date: 2026-10-08. Status updated 2026-10-12: binding protocol; the [six-variable study](task_suitability_sixvar.md) passed source suitability/local difficulty but ended `NO_GO_DEVELOPMENT_CAPACITY`. The [retained diagnosis](coverage_diagnosis_result.md) is complete; the separately registered [sampler v2 mechanism fixture](certificate_sampler_v2_design.md) is being implemented/reviewed and has not run. Its supported-population departure is explicit and prospective; historical NO-GOs remain unchanged. Final tasks/G2/model/training are unattempted. See [execution state](execution_state.md) and [todo.md](../todo.md). The original [v4 result](task_suitability_v4.md) remains `NOT_ASSESSED_RESOURCE_CAP`; all prior attempts are preserved. This status note does not amend the v2.2 scientific protocol.

This is the binding amendment to the [v2 plan](diffusive_workspace_reasoning_project_plan_v2.md). It resolves the ambiguities identified in the [research review](research_review.md). Where the original narrative differs, this protocol takes precedence; [todo.md](../todo.md) is the sole progress checklist. Changes below are deliberate scientific decisions, not experimental findings.

## 1. Research objective and claim boundaries

**2026-10-07 focus decision:** [Transferable computation with exchangeable workspaces](transfer_research_direction.md) is the active scientific objective and specifies its task/transfer contract. Train one shared workspace/interface across source tasks, then freeze it for unseen compositions and structures. Exchangeable slots are the central architectural hypothesis; variable slot count is a staged supporting study. Other directions are [parked](spare_research_ideas.md). This deliberately replaces the single-Sudoku-first research study; all reproducibility safeguards below still apply.

Study whether a jointly updated latent workspace attached to a frozen language model provides useful computation under controlled task, interface, data, and budget conditions. A credible outcome is a reproducible positive effect, a bounded negative result, or an informative characterization. Publication and novelty are not guaranteed by completing the implementation.

| Question | Required comparison | Maximum supported interpretation |
| --- | --- | --- |
| H1: does iteration help? | Separately trained EX K=4 vs EX K=1, matched data exposure and tuning opportunity | Benefit of additional computation in this setup; training/inference compute differ |
| Inference scaling | One variable-K checkpoint at K=1,2,4,8 and separately labeled 12,16 | Within-checkpoint horizon sensitivity, not architecture superiority |
| H2: does joint computation help? | Joint recurrence vs ordered sequential computation at measured comparable budgets, with the same decoder interface | Task-specific computation-style evidence |
| H3: does slot identity matter? | EX vs IDX at the same K, N, d and training protocol | Effect of learned initial slot identities, including their extra parameters |
| Is recurrence cost-effective? | Shared recurrence vs unshared feedforward at comparable end-to-end inference cost | Evidence about weight sharing; report parameter/training-cost differences |
| H4: does flow help? | Frozen recurrent teacher vs distilled flow and a simpler direct distillation control | Quality/cost tradeoff for distillation, not inherently better reasoning |

The primary confirmatory architectural contrast is EX minus IDX at the same fixed N and K on macro-averaged valid-solution rate across preregistered held-out composition groups. Use one checkpoint per method across target groups, without target-task retraining. Register absolute transfer and ID-learning criteria as well as the between-method practical effect. The K=4 vs K=1 ID contrast is now secondary. The table above lists supporting questions, not the execution order. Cross-family transfer, size/depth extrapolation and new slot-count extrapolation require separate evidence and labels; none follows from compositional transfer alone.

## 2. Task suitability comes before training

The active data gate is the finite-domain constraint-grammar specification in the main-direction document: known primitives, unseen compositions, canonical structure/lineage grouping, and separate development/final target signatures. Before implementation, fix its exact syntax, semantics, verifier, exhaustive small-instance oracle and prospective split sizes in `docs/task_spec.md`. The Sudoku-specific rules below apply only if Sudoku fixtures or later Sudoku studies are used; they do not impose a mandatory detour.

4×4 Sudoku with 2×2 boxes is a solver/verifier and integration fixture. It is not the default confirmatory research dataset. There are only 288 completed grids and two equivalence classes under standard Sudoku symmetries ([Cornell enumeration](https://pi.math.cornell.edu/~mec/Summer2009/Mahmood/Four.html)). This concerns completed grids; clue-mask puzzles have a different count. Two solution families cannot support three nonempty family-disjoint splits.

4×4 remains an optional correctness fixture. The v2.1 mandatory 6×6 research-data step is superseded by the composition-transfer task gate. A later 6×6 study would still need diversity/difficulty evidence; never relax leakage grouping to fill any split.

Freeze a task specification before generation:

- Symbols 1..n; 0 denotes an empty input cell. Input and output use row-major, space-separated integer grids with one row per line. The answer is the complete grid; extra commentary, missing/extra cells, non-integer symbols, and changed clues fail strict parsing/verification. Normalize line endings and surrounding whitespace only.
- Store solution and solver annotations outside model input. Prompts contain only the puzzle, fixed rules, and a fixed answer-format instruction. Version the exact prompt text and tokenizer serialization before model tuning.
- Primary metric: fraction of all requested puzzles yielding one complete, valid, clue-preserving solution. Malformed outputs and truncations count as failures. For unique instances, structured exact match must agree with verifier acceptance. Report parsing rate and blank-cell accuracy only as diagnostics.
- Fix deterministic solver propagation rules, branch heuristic, tie order, and counting conventions. Distinguish finding the first solution from searching for a second to prove uniqueness. Record their statistics separately; use first-solution structural counts for difficulty, not uniqueness-proof runtime.

Define explicit equivalence transformations for each board geometry. For 4×4 these include digit relabeling, rows within bands, band swaps, columns within stacks, stack swaps, and transpose. For 6×6 use only transformations preserving the declared 2×3 box geometry; do not blindly reuse transpose. Version and test canonicalization.

Group all clue removals and augmentations from the same canonical completed-board orbit in one split. Also group any larger known generation lineage; audit canonical puzzle duplicates independently. Generate genuinely different source boards rather than only shuffling one template. Document what the audit covers: arbitrary semantic similarity is not a proven guarantee.

Use a disposable development pool to choose solver-statistic cutoffs and generator settings. Then freeze an ID pool and a harder OOD pool, retaining a harder validation subset for development. Assign whole families before augmentation and reserve whole families for OOD; do not let one board's easy masks enter training and its hard masks enter test. Freeze sample counts and group counts per split/bucket. Report realized proportions; an 80/10/10 target must not break groups.

Data gate: independent oracle checks pass, regeneration reproduces hashes, all required splits have nonzero independent family counts, a prospective precision analysis supports the intended effect size, and the designated hard distribution differs on the registered difficulty measure. Inspect these structural properties before training. If no meaningful hard regime exists, record it and limit the claim or reject the task. Never manufacture buckets from ties.

## 3. Architectural contract

Keep `Z_next = Z + F(LN(Z), X)` with weights shared across recurrent steps. Recurrence is deterministic conditional on the input, initial noise, and evaluation settings. Training randomness remains separately logged.

- Compute X in a separate prompt-only pass through the original frozen encoder layers, with workspace adapters disabled. Cache only by backbone/tokenizer revision, exact token IDs/mask, extraction layer and precision. Never cache answer-containing states or states computed through changing adapters.
- The answer decoder can attend causally to previous answer tokens during teacher-forced training. Those tokens must never flow back into X or workspace updates. At evaluation, generate autoregressively from the prompt; teacher-forced token accuracy is not solution accuracy.
- Match adapter layers, heads, dimensions, gates, initialization and decoding policy across learned workspace methods. Original parameters stay frozen; autograd through frozen decoder operations after trainable insertions must remain enabled. Separate train/eval module modes from `requires_grad`.
- IID Gaussian noise initializes all slots. IDX adds its learned identity embeddings **once at initialization**, not at every recurrence. This resolves v2's ambiguous word “persistent.” Report the N×d added parameters.
- EX has no slot positions, slot-specific projections, causal slot masks, or order-dependent pooling. IDX's identities distinguish initialization; its set decoder must still be invariant to permutation of whole final slot vectors. Permuting noise alone while leaving identity embeddings in place is not the EX equivariance test.
- Test EX update equivariance and decoder logit invariance with dropout disabled and dtype-specific tolerances. Test causal answer masking, prompt-padding behavior, and batched versus single-example outputs. Check finite gradients through every update and verify frozen weights are unchanged after an optimizer step.
- K=1 EX is exactly the one-shot implementation, not a separate architecture. Use the same residual block for unshared feedforward, with separate weights per depth; v2's non-residual feedforward example is superseded.

## 4. Controls, budgets and staged execution

The frozen unaugmented backbone is a reference, not a parameter-matched control. Add an adapter-only **K=0 noise-memory** control: train the same interface/projections on unchanged IID Gaussian slots. This tests task adaptation through the interface without prompt-conditioned workspace computation. It has fewer trainable parameters; report that difference. Zero/noise/shuffle ablations of a trained workspace are diagnostics and do not substitute for this separately trained control.

Start with tiny-batch integration, then a one-seed development pilot of K=0, EX K=1 and EX K=4. Run pilots only on development/train/validation data. Diagnose failures before expanding the matrix. Use fixed K=4 for the initial transfer comparison, with K=1 as a control. The full {1,2,4,8} grid and variable-K protocol are deferred secondary diagnostics, not prerequisites to the transfer result.

For data-exposure comparisons, hold examples/tokens, effective batch size, optimizer schedule and tuning trial count fixed; report that larger K costs more training compute. For compute-efficiency comparisons, separately freeze a training-compute cap and inference-cost target; report different data exposure if induced. Never label equal optimizer steps as equal FLOPs, nor assume simultaneous parameter and compute matching.

Choose unshared feedforward and ordered sequential controls using measured development costs, without test accuracy. Define sequential state visibility, ordering, number of generated slots, and serial update count before implementation; use the same N/d and decoder access where feasible. Match total inference FLOPs within a registered tolerance (default ±10%); report actual deviations and latency. If no match is feasible, report a cost/accuracy frontier and omit “matched-compute winner” language. Baseline tuning gets equal opportunity, with exceptions documented.

Feedforward is required before recurrence-specific efficiency claims. Sequential is required before joint-over-sequential claims. These controls may be omitted after an uninformative pilot, but then so must the corresponding claims. A well-trained null pilot can support stopping; it does not prove recurrence never helps.

Variable-K training samples uniformly from {1,2,4,8} per batch by default, uses loss only at the sampled terminal horizon, and uses full backpropagation through those updates. Do not detach steps, add intermediate supervision, or change K sampling without a protocol amendment.

Before training, measure full forward/backward/optimizer memory at the maximum planned K, sequence length and batch size. Frozen weights still incur activation costs. Freeze backbone revision, dtype, attention implementation, adapter layout, N/d and batch settings after this probe. A smaller model or workspace is an allowed documented feasibility decision applied consistently, not a hidden per-method fallback.

## 5. Registration, statistics and test access

Each study has a small versioned registration under `configs/experiments/` and a dated decision note under `docs/`. Create these when the study becomes executable; the draft prototype is not a registration. Required fields:

- Hypothesis, primary contrast/metric, secondary analyses, effect size of practical interest (delta in percentage points), uncertainty method, and decision rule.
- Dataset/generator/canonicalizer hashes, split/group counts, difficulty boundaries, and prospective sample/seed precision calculation using only development data.
- Exact methods, configuration IDs, training seeds, evaluation noise draws, tuning trials, training steps/examples/tokens, maximum GPU-hours, memory limits, and maximum diagnostic reruns.
- Validation checkpoint-selection rule, evaluation schedule, decoding/length settings, compute-matching tolerance, and planned handling of failure/missingness.

Delta, sample counts and GPU-hour limits are intentionally not guessed here: fill them from task/feasibility probes and freeze them before comparative training. Missing values block a confirmatory run. Default checkpoint selection is highest validation valid-solution rate, ties broken by the earliest step, evaluated at fixed intervals. For variable-K use the equally weighted mean over seen horizons only. Freeze selection before testing; never select a separate checkpoint by test K or test bucket.

Use at least three independent training seeds for a confirmatory effect; fewer is exploratory. Three is a minimum, not evidence of sufficient power. Pair data order, compatible parameter initializations, and evaluation noise across methods where possible. Use separate RNG streams for data, model initialization, training workspace noise, K sampling and evaluation. Stable per-example evaluation noise must not depend on batch order.

Primary decoding is greedy with a fixed maximum answer length and one predetermined workspace-noise draw per example per training seed. Multiple-noise diagnostics average correctness over a registered draw set; they are not best-of-N. Report pass@N separately only with total sampling and verifier-selection cost.

Report per-seed numerators/denominators, mean and standard deviation, paired effect sizes and uncertainty (default two-sided 95% intervals). Cluster resampling at the split's family level rather than treating symmetry-related puzzles as independent; account for training-seed variation separately or hierarchically and state the method. No significance claim from pooling all seed×puzzle observations as independent. Prespecify simultaneous intervals or multiplicity correction for families of confirmatory tests; otherwise label secondary comparisons exploratory.

Decision language: an interval excluding zero supports a directional effect; a lower bound exceeding registered delta supports a practically meaningful improvement. An interval entirely inside [-delta,+delta] can support practical equivalence for this setup. Wider intervals are inconclusive. A point estimate near zero or a nonsignificant test is not proof of equivalence.

Test evaluation is a terminal batch after configurations, checkpoint selection and analysis code are frozen. Log every access, including exploratory access. Any adaptation to test feedback makes subsequent analysis exploratory on that set. Keep the original test and results; a new confirmatory study requires a newly versioned untouched test, not replacement of inconvenient cases. Fixes to solver/data/scoring invalidate affected evidence and require a dated correction plus reruns of all affected methods.

## 6. Reproducibility and code care

Every run writes a resolved manifest **before** work and has a unique directory. Include run/study ID; UTC start/end/status; exact command; source commit and dirty diff/hash (confirmatory runs use a clean commit); full resolved config; model/tokenizer revisions; data and split hashes; prompt/parser/verifier versions; all RNG seeds; Python/package lock or exact package inventory; OS/GPU/driver/CUDA/library versions; dtype, determinism/TF32/kernel settings; parameters; steps/examples/tokens and training cost. Record unavailable measurements explicitly, never as zero.

Persist per-example predictions, example/family IDs, noise-draw IDs, verifier outcomes/reasons, checkpoint hash, counts and aggregate metrics. Avoid logging secrets. Keep generated datasets, caches, raw predictions and checkpoints in ignored artifact directories; commit small manifests, aggregate reports and reproduction instructions. Hash artifacts and maintain a durable storage/backup location before claiming a reproducible result. An ignored local directory alone is not archival preservation.

Checkpoint optimizer, scheduler, gradient scaler if used, all RNG states, step, sampler position and identities. Test uninterrupted versus resumed small runs with declared tolerances; claim bitwise replay only where actually demonstrated. Log nondeterministic operations rather than promising determinism from a seed alone.

Use fail-fast config and shape validation. Reject unknown scientific config keys and incompatible resumes. Do not silently change precision, truncate data, substitute a model, skip examples, loosen verifiers, catch-and-ignore errors or detach gradients to get a run through. Numerical divergence is an outcome; stop, preserve evidence and diagnose. Retrying OOM or hardware failures gets a new run ID and a link to the failed attempt; state any scientifically material change.

Use standard-library `unittest` for task logic. Test the verifier independently of the generator; cross-check the selected task grammar with an independent exhaustive tiny-instance oracle. If a 4×4 Sudoku path is implemented, separately check solution counting against all 288 valid completed grids. Include invalid inputs, zero/one/multiple solutions, clue preservation, transformations, deterministic regeneration and leakage audits. Add tiny tensor gradient/permutation/causality tests when ML code lands; CPU checks do not establish GPU correctness or training efficacy.

Implement only modules needed by the next milestone under `dwr/`. Use `pathlib`, explicit UTF-8, local RNG objects, stable serialization and SHA-256 identities. Keep test fixtures small and checked in. No mandatory Bash, symlinks, speculative frameworks, unneeded ML dependencies or unrelated refactors. A milestone is complete only when commands, actual outcomes and evidence paths are recorded.

## 7. Measurement and publication package

Measure prompt encoding + workspace + autoregressive answer decoding end to end. State whether prompt caches are warm; give every method the same cache policy and include cache-build/storage costs separately. Report workspace-only costs as a secondary decomposition. Include repeated answer-token adapter calls, generated lengths, all attempted samples and verifier-selection costs where applicable.

Record device, batch size, prompt/output lengths, warmups, timed repetitions, synchronization, median/tail latency, throughput and peak allocated/reserved VRAM. Freeze repetitions after a development timing probe. Estimate FLOPs with an explicit operation scope and convention; do not substitute latent block counts for total cost. Report training, tuning, failed attempts and later teacher-target generation costs separately from inference.

A research report needs: related-work positioning with exact citations/versions; frozen methods and data protocols; complete seed results including negatives; uncertainty and cost plots; limitations; artifact provenance and environment; commands to regenerate data, train, evaluate and rebuild tables/figures. At least one fresh-environment reproduction is required before an artifact-reproducibility claim. A single task supports a narrow task claim; broad generalization requires an additional controlled task, not only more samples from the same generator. Strong architecture claims also need the controls above.

Family-disjoint generated data limits leakage from this project's training set; it cannot certify absence from a pretrained model's corpus. Solver difficulty is relative to the specified solver, not a universal measure of reasoning depth. State both limitations rather than treating generation or a verifier as proof of novelty or mechanism.

## 8. Parked flow gate and general stop rules

Flow is a spare direction, not an active milestone; reactivation requires a dated study decision. The technical safeguards below remain binding if it is activated.

Flow remains optional and blocked until the recurrent study is understood and its targets/interface are stable. Unique-solution Sudoku has only one correct answer per puzzle: distinct outputs do not demonstrate diverse correct solutions. Report correctness alongside latent and output diversity, and do not equate latent distance with useful diversity.

Initial flow target pairing is `Z0 = epsilon`, `Z1 = frozen_teacher(x, epsilon)` for an exchangeable teacher; preserve the teacher's slot correspondence and apply any permutation jointly to both endpoints. Use the straight-line path and squared velocity loss from v2. This is a teacher-induced coupling, not optimal transport. Freeze pairing, teacher/interface hashes, target-data split and time-sampling distribution in the later registration. Collect training targets only from the training split. Independent source/target pairing is a separately labeled ablation, not an interchangeable implementation detail.

Compare against a direct one-shot student trained on the same teacher targets/interface and budget. Count target-generation/training cost and declare quality-retention tolerance before benchmarking. A straight-line flow result supports a flow-distillation claim, not automatically a diffusion-model claim. More integration evaluations need not mean more learned reasoning.

At each gate, choose proceed, bounded diagnostic, or stop and report. Pre-register a finite diagnostic-run allowance and total compute cap; do not keep tuning until positive. Distinguish implementation failure, task failure, underpowered/inconclusive evidence, and a well-controlled negative result. Failure against a feedforward control narrows the recurrence claim; it does not erase a valid finding about extra compute. Preserve every decision and its evidence.
