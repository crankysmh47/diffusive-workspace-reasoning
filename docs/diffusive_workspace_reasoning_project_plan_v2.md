# Diffusive Workspace Reasoning
## Revised implementation-first research plan (v2)

**Status:** Ready to implement  
**Revision date:** 2026-10-04  
**Working title:** *Diffusive Workspace Reasoning: Testing Joint Latent Workspaces Against Sequential Latent Computation*  
**Short name:** DWR

---

# 0. What changed in this revision

This revision deliberately narrows the first experiment.

The project is no longer organized around the assumption that diffusion is the correct reasoning mechanism. The first scientific question is now:

> **Does iterative joint computation over a latent workspace help reasoning, especially on problems with interacting constraints, compared with one-shot and ordered alternatives?**

Diffusion / flow matching is a later experiment that must earn its place.

The main changes from v1 are:

1. **No posterior encoder in v1.**
2. **No answer-conditioned latent targets in v1.**
3. **No dependency-gated backprop in v1.**
4. **The decoder backbone is frozen, but newly added latent-interface layers are trainable.**
5. **The first working system is a randomly initialized recurrent latent workspace.**
6. **All recurrent and later diffusion/flow variants receive the same stochastic slot initialization.**
7. **The core comparison holds the decoder interface fixed.**
8. **Slot identity, computation style, decoder access, and training mechanism are treated as separate experimental axes.**
9. **A deterministic recurrent workspace is built before diffusion.**
10. **The first task family must have a proper verifier and measured difficulty.**
11. **Inference-time compute scaling and architecture-compute efficiency are evaluated separately.**
12. **Flow matching is initially treated as distillation from a frozen recurrent teacher.**
13. **Solver function evaluations are reported as numerical/inference efficiency, not automatically interpreted as “more reasoning.”**
14. **The stop rule is softened: a deterministic workspace result is still valuable even if diffusion does not win.**

---

# 1. Project thesis

The project investigates the following hypothesis:

> Language-model reasoning may benefit from representing intermediate computation as a jointly updated collection of latent states rather than forcing all computation into a sequential latent trajectory.

The project will test increasingly strong hypotheses:

## H1 — Iteration

Repeated latent updates improve performance relative to a one-update latent model.

## H2 — Joint workspace computation

Jointly refining multiple latent states is more effective than ordered/sequential latent computation on some reasoning problems.

## H3 — Slot identity

Removing fixed slot identities changes generalization or compute efficiency relative to indexed latent states.

## H4 — Stochastic generative refinement

A genuine diffusion / flow model provides an additional benefit over deterministic recurrence, such as:

- fewer inference operations for the same accuracy,
- better stochastic coverage of useful latent states,
- or higher task performance under matched compute.

The project should not assume H4 is true.

---

# 2. What the first experiment can and cannot prove

If an 8-update recurrent workspace beats a 1-update recurrent workspace, the supported claim is:

> Additional iterative computation is useful for this trained architecture.

It does **not** establish:

- that recurrence is the optimal use of the extra compute,
- that exchangeable slots are necessary,
- that the model is performing human-like reasoning,
- that diffusion would be better,
- or that the architecture generalizes beyond the tested task family.

Two distinct compute comparisons are required.

## Comparison A — Same checkpoint, different update counts

Question:

> Does the same trained recurrent model benefit from extra inference-time updates?

Report separately:

- update counts seen during training,
- update counts not seen during training.

Example training support:

\[
K \in \{1,2,4,8\}
\]

Example extrapolation tests:

\[
K \in \{12,16\}
\]

Do not mix the two categories in plots without labeling them.

## Comparison B — Separately trained architectures at comparable budgets

Question:

> Which architecture uses a comparable compute budget most effectively?

This comparison should eventually include:

- one-shot latent model,
- recurrent workspace,
- indexed recurrent workspace,
- ordinary deeper feedforward alternative,
- sequential latent computation,
- later, flow/diffusion.

Report:

- task accuracy,
- inference FLOPs,
- latency,
- trainable parameters,
- total parameters,
- training compute where practical.

---

# 3. Core architectural axes

Do not conflate these choices.

| Axis | Alternatives |
|---|---|
| Slot identity | indexed / exchangeable |
| Computation | one-shot / joint recurrent / sequential |
| Decoder access | fixed set cross-attention for core experiments |
| Training mechanism | answer-loss recurrence initially / flow matching later |

A faithful LRT-style reproduction should be treated as an **external architecture comparison**, not as the sole “sequential baseline,” because it changes more than one axis at once.

---

# 4. v1 model: recurrent latent workspace

## 4.1 Backbone

Use one pretrained decoder-only base model.

Recommended prototype scale:

**~1.5B–2B parameters**

Purpose:

- fast iteration,
- many seeds,
- ablations,
- controlled compute experiments.

Only move to ~3B after the prototype produces a convincing signal.

## 4.2 Frozen versus trainable parameters

### Frozen

- pretrained token embeddings,
- pretrained self-attention blocks,
- pretrained MLPs,
- pretrained LM head,
- all original backbone weights.

### Trainable

- latent workspace module,
- latent projections,
- newly inserted cross-attention adapters,
- optional small adapter gates,
- task-format embeddings if required.

“Frozen decoder” always means **frozen pretrained weights**, not “no trainable interface.”

---

# 5. Prompt representation

Let the frozen backbone encode the problem.

Extract hidden states from an early or middle layer:

\[
X = E_{\text{frozen}}(x)
\]

Use the same \(X\) for all core methods.

Do not give one method a better prompt encoder.

Initial recommendation:

- extract from approximately the first third of the backbone,
- keep that extraction point fixed across all core comparisons.

---

# 6. Latent workspace

Use a fixed slot count in v1.

Recommended starting point:

\[
N = 32 \text{ or } 64
\]

Latent dimension:

\[
d = 512 \text{ to } 1024
\]

Initial configuration:

\[
Z_0 \in \mathbb{R}^{64\times 768}
\]

or:

\[
Z_0 \in \mathbb{R}^{64\times 1024}
\]

depending on memory.

---

# 7. Random initialization

Every joint-workspace method receives the same type of stochastic initialization:

\[
Z_0 \sim \mathcal N(0,I)
\]

independently across slots.

This applies to:

- one-shot workspace,
- exchangeable recurrent workspace,
- indexed recurrent workspace,
- later flow/diffusion workspace.

Do **not** give random symmetry breaking only to the diffusion model.

---

# 8. Consistent recurrent update equation

Use one update form throughout v1.

Recommended residual update:

\[
\boxed{
Z_{k+1}
=
Z_k
+
F_\theta(\operatorname{LN}(Z_k),X)
}
\]

where \(F_\theta\):

- uses shared parameters across recurrent steps,
- performs self-attention over latent slots,
- cross-attends to prompt representation \(X\),
- contains an MLP block,
- has no slot positional embeddings in the exchangeable variant.

Do not alternate between

\[
Z_{k+1}=F(Z_k,X)
\]

and

\[
Z_{k+1}=Z_k+F(Z_k,X)
\]

in code, paper, or documentation.

The residual form above is the default.

---

# 9. Exchangeable recurrent variant

The exchangeable model should not contain fixed slot-index embeddings.

The update should be permutation-equivariant with respect to slot ordering.

If \(\pi\) permutes the slots:

\[
F_\theta(\pi Z,X)=\pi F_\theta(Z,X)
\]

The decoder interface should be invariant to slot permutation.

This removes **predefined index identity**.

It does **not** prove the model is nonsequential internally.

Slots may still specialize by content and may represent internally ordered roles.

Therefore do not claim:

> exchangeability prevents sequential reasoning.

The correct claim is:

> exchangeability removes fixed slot-index identity as an architectural prior.

---

# 10. Indexed recurrent variant

Use the same recurrent architecture and same stochastic initialization, but add persistent learned slot identity embeddings:

\[
Z_0^{(i)} = \epsilon_i + e_i
\]

where

\[
\epsilon_i\sim\mathcal N(0,I)
\]

and \(e_i\) is a learned slot embedding.

Everything else should remain as similar as possible.

This isolates:

\[
\text{exchangeable}
\quad \text{vs} \quad
\text{indexed}
\]

without simultaneously changing the update mechanism.

---

# 11. One-shot variant

Use the same workspace initialization and same latent block, but only one update:

\[
Z_1
=
Z_0
+
F_\theta(\operatorname{LN}(Z_0),X)
\]

Then decode.

This is the first baseline.

It answers:

> Is iteration useful beyond a single latent transformation?

---

# 12. Fixed decoder interface for core comparisons

Do **not** append latent states as soft tokens in the core experiments.

Use a fixed set cross-attention interface for every core method.

At selected frozen-backbone layers:

\[
H_\ell'
=
H_\ell
+
A_\ell(H_\ell,Z_K)
\]

where \(A_\ell\) is a **trainable newly added cross-attention adapter**.

Use the same:

- number of adapters,
- adapter locations,
- projection dimensions,
- initialization,

for all core methods.

This prevents a computation comparison from being confounded by decoder access.

---

# 13. Permutation test

For the exchangeable variant, randomly permute slots before decoder cross-attention.

The output distribution should be numerically equivalent up to floating-point noise:

\[
D(x,\pi Z)=D(x,Z)
\]

Test this directly.

If performance changes materially under permutation, there is an implementation bug or hidden order dependence.

This is an architectural sanity check, not evidence that internal computation is conceptually nonsequential.

---

# 14. v1 training objective

There is no posterior encoder.

There are no target latents.

There is no diffusion loss.

Train directly from final-answer supervision:

\[
\boxed{
\mathcal L_{\text{answer}}
=
-\log p(y\mid x,Z_K)
}
\]

Backpropagate through:

- decoder cross-attention adapters,
- latent projections,
- all \(K\) recurrent updates.

The pretrained backbone remains frozen.

---

# 15. Update-count training protocols

Two protocols should be run.

## Protocol A — fixed-K checkpoints

Train separate checkpoints with:

\[
K=1,2,4,8
\]

This supports fair architecture-at-budget comparisons.

## Protocol B — variable-K checkpoint

Train one checkpoint where:

\[
K \sim \{1,2,4,8\}
\]

sampled per batch or according to a controlled schedule.

Evaluate the same checkpoint at:

### Seen update counts

\[
1,2,4,8
\]

### Unseen update counts

\[
12,16
\]

This distinguishes:

- interpolation across trained horizons,
- extrapolation beyond trained horizons.

Do not describe performance at unseen \(K\) as reliable test-time scaling unless it is stable across seeds and tasks.

---

# 16. Feedforward compute baseline

Eventually add an ordinary deeper feedforward alternative.

Purpose:

> Determine whether recurrence is a particularly effective use of additional computation.

Construct a model that uses approximately comparable trainable parameters and inference FLOPs, but does not share weights recurrently.

For example:

\[
Z_1=F_1(Z_0,X),\quad Z_2=F_2(Z_1,X),\ldots
\]

with unshared blocks.

Compare:

- accuracy,
- FLOPs,
- latency,
- trainable parameters,
- training stability.

This comparison is separate from same-checkpoint update scaling.

---

# 17. Sequential latent-computation baseline

After the initial recurrent prototype is working, build an ordered/sequential latent method while keeping the decoder interface fixed.

Example:

\[
z_1=G(x,\epsilon_1)
\]

\[
z_2=G(x,\epsilon_2,z_1)
\]

\[
\ldots
\]

\[
z_N=G(x,\epsilon_N,z_{<N})
\]

Then feed the resulting collection through the **same set cross-attention interface** used by the joint models.

The purpose is to isolate:

\[
\text{sequential computation}
\]

versus

\[
\text{joint iterative computation}
\]

without simultaneously changing the decoder access mechanism.

A faithful LRT reproduction can be added separately as an external baseline.

---

# 18. First task family

Do not begin with several public reasoning datasets.

Begin with **one verified controlled task family**.

Recommended first family:

## Mini-Sudoku / Latin-square-style constraint completion

Why:

- interacting constraints,
- deterministic verifier,
- easy generation,
- controllable size,
- supports unique-solution filtering,
- suitable for studying iterative constraint propagation.

Exact task format should be fixed before architecture tuning.

Possible initial sizes:

- 4×4,
- then 6×6 if practical.

Do not assume larger size automatically means harder.

---

# 19. Task generation requirements

For each generated instance:

1. Generate a complete valid structure.
2. Remove entries according to a controlled procedure.
3. Use a reference solver to verify solvability.
4. Prefer uniquely solvable instances for the first experiment.
5. Record solver-derived difficulty.
6. Store generator seed.
7. Store template / family identifier.
8. Ensure no identical or near-identical instance leaks across splits.

---

# 20. Difficulty measurement

Do not use only:

- board size,
- number of blanks,
- constraint density,

as difficulty proxies.

Measure difficulty using a reference solver.

Possible metrics:

- number of propagation rounds,
- number of backtracking nodes,
- search-tree depth,
- number of branch points,
- solver runtime under a fixed implementation,
- size of largest unresolved interacting component.

Create difficulty buckets using measured solver behavior.

For example:

- easy,
- medium,
- hard,
- extrapolation-hard.

---

# 21. Verifier

Do not use exact string match if multiple valid solutions are allowed.

Preferred v1 approach:

**Generate uniquely solvable instances.**

Then evaluation can use exact structured solution comparison.

If multiple valid solutions are later allowed, use a constraint checker that verifies:

- all required variables assigned,
- all hard constraints satisfied,
- no invalid symbols,
- no duplicate assignment where prohibited.

Report:

- valid-solution rate,
- exact match where meaningful.

---

# 22. Train / validation / test split

Prevent template leakage.

Recommended:

- train: 80–90%,
- validation: 5–10%,
- test: 5–10%.

Additionally create a held-out difficulty split.

Examples:

- train on lower solver-depth range,
- test on higher solver-depth range,
- train on one clue-density range,
- test on harder verified instances.

Do not revise the held-out test set after inspecting results.

---

# 23. First implementation sequence

The next useful evidence should come from experiments, not redesign.

Implement in this order:

## Step 1

Build the generator and reference verifier.

## Step 2

Create fixed train/validation/test splits.

## Step 3

Benchmark the frozen backbone without latent augmentation.

## Step 4

Implement the trainable set cross-attention interface.

## Step 5

Implement the one-shot exchangeable latent model.

## Step 6

Verify end-to-end answer-loss training.

## Step 7

Implement the exchangeable recurrent model with shared residual updates.

## Step 8

Train fixed-\(K\) models for:

\[
K=1,2,4,8
\]

## Step 9

Implement indexed recurrent slots.

## Step 10

Compare:

- one-shot,
- exchangeable recurrent,
- indexed recurrent.

All on the same first task family.

## Step 11

Run several random seeds.

## Step 12

Only after a real iterative signal appears, add the feedforward and sequential baselines.

## Step 13

Only after the recurrent result is understood, implement flow matching.

---

# 24. Minimum first experiment

The minimum experiment that justifies continuing is:

- one frozen ~1.5B–2B backbone,
- one fixed task family,
- fixed slot count,
- same random initialization distribution,
- trainable set cross-attention interface,
- final-answer supervision only,
- one-shot exchangeable model,
- exchangeable recurrent model,
- indexed recurrent model,
- update counts 1, 2, 4, 8,
- at least 3 seeds if compute permits,
- held-out difficulty evaluation,
- latency and FLOPs measurement.

Do not add diffusion before this experiment is interpretable.

---

# 25. What counts as a useful v1 signal

A useful result could be any of the following:

## Result A

Recurrent workspace consistently beats one-shot under comparable conditions.

Interpretation:

Iterative latent computation is useful.

## Result B

Exchangeable and indexed variants differ meaningfully in held-out generalization.

Interpretation:

Slot identity prior matters.

## Result C

A recurrent workspace beats a compute-matched deeper feedforward alternative.

Interpretation:

Weight-shared iterative computation may be a particularly effective compute mechanism.

## Result D

Joint recurrence beats sequential latent computation on the controlled task.

Interpretation:

Workspace-style computation is promising.

None of these requires diffusion.

---

# 26. What does not count as strong evidence

Do not overinterpret:

- training accuracy,
- posterior-like latent reconstruction,
- large gains only on train-like difficulty,
- one lucky seed,
- unverified generated outputs,
- gains caused only by more parameters,
- gains caused only by more inference FLOPs,
- performance that disappears when the decoder interface is matched.

---

# 27. Same-checkpoint inference scaling experiment

For the variable-\(K\) checkpoint:

Evaluate:

\[
K=1,2,4,8,12,16
\]

Label each point as:

- trained horizon,
- untrained extrapolation horizon.

Plot:

- accuracy vs \(K\),
- latency vs \(K\),
- FLOPs vs \(K\),
- accuracy vs FLOPs.

If performance improves only for trained update counts, report that honestly.

If performance extrapolates to unseen \(K\), verify:

- multiple seeds,
- no divergence in latent norms,
- no numerical instability,
- consistent task-level gains.

---

# 28. Architecture-at-budget experiment

Train separate models for selected budgets.

Example:

| Method | Approx. compute budget |
|---|---:|
| one-shot | 1 latent block |
| recurrent K=2 | 2 block evaluations |
| recurrent K=4 | 4 block evaluations |
| recurrent K=8 | 8 block evaluations |
| deeper feedforward | matched block evaluations |
| sequential latent | matched approximate FLOPs |

This answers a different question from same-checkpoint scaling.

Keep these analyses separate in the paper.

---

# 29. Second task family

Only add a second family after the first is working.

Recommended second family:

## Sequential-depth symbolic task

Examples:

- pointer chasing,
- function composition,
- deterministic state transitions,
- symbolic rewrite chains.

Control true chain depth explicitly:

\[
d=2,4,8,16,\ldots
\]

Use a deterministic verifier.

The purpose is not to force a “crossover” where sequential wins one task and workspace wins another.

That crossover would be interesting, but it is not required.

The purpose is to test whether architectural behavior changes with task structure.

---

# 30. Do not require a preconceived crossover

Do not make the paper's validity depend on:

> sequential must win Family S and workspace must win Family C.

Possible valid findings include:

- workspace wins both,
- sequential wins both,
- indexed recurrence wins both,
- recurrence helps only on difficult instances,
- one-shot is sufficient,
- task structure interacts with architecture only weakly.

The experiment should discover the result.

---

# 31. Flow matching comes later

Only proceed after a recurrent teacher has been trained and understood.

At this stage, the recurrent teacher defines useful terminal latent states:

\[
Z_K = R_\theta(x,\epsilon)
\]

with

\[
\epsilon\sim\mathcal N(0,I)
\]

Collect terminal states from the trained teacher.

---

# 32. Measure teacher latent diversity first

Do not assume different noise seeds produce meaningfully different latent solutions.

For each input \(x\), sample several seeds:

\[
\epsilon_1,\ldots,\epsilon_m
\]

and obtain:

\[
Z_K^{(1)},\ldots,Z_K^{(m)}
\]

Measure:

- pairwise latent distance,
- decoder-output diversity,
- solution diversity,
- answer correctness,
- whether diversity survives after slot matching/permutation handling.

If all terminal states are effectively equivalent, do not sell multimodality as a motivation.

---

# 33. Flow teacher and interface freezing

For the initial flow experiment:

**Freeze:**

- recurrent teacher,
- trained decoder cross-attention interface,
- frozen language-model backbone.

This fixes the meaning of the target latent coordinate system.

Train only the flow model.

Do not allow the target interface to drift while fitting the flow.

---

# 34. Conditional flow-matching target

Let:

\[
Z_1 \sim p_R(Z\mid x)
\]

be a recurrent-teacher terminal latent.

Let:

\[
Z_0 \sim \mathcal N(0,I)
\]

be the source sample.

Use a valid conditional probability path.

A simple straight-line conditional path is:

\[
Z_t=(1-t)Z_0+tZ_1
\]

with target conditional velocity:

\[
u_t=Z_1-Z_0
\]

Train:

\[
v_\phi(Z_t,x,t)
\]

with:

\[
\boxed{
\mathcal L_{\text{FM}}
=
\mathbb E
\left[
\|v_\phi(Z_t,x,t)-u_t\|^2
\right]
}
\]

Do not call independently paired endpoints “optimal transport” unless an actual OT coupling is used.

---

# 35. Initial purpose of the flow experiment

The first flow question is:

> **Can flow matching reproduce the recurrent teacher's answer quality with fewer or cheaper inference operations?**

This is initially a **distillation / amortization experiment**.

Primary flow metrics:

- answer accuracy,
- function evaluations,
- latency,
- memory,
- FLOPs,
- teacher-quality retention.

A successful result does not need to beat the teacher's accuracy.

Matching teacher quality at substantially lower inference cost would already be useful.

---

# 36. Flow solver interpretation

For a fixed learned vector field, fixed initial condition, and fixed integration interval, additional solver steps primarily improve numerical approximation of the same continuous trajectory.

Therefore do not write:

> more ODE solver steps = more reasoning.

Report:

- accuracy vs NFE,
- latency vs NFE,
- estimated solver error if practical.

Treat NFE as an inference-efficiency variable.

If the project later wants “more learned computation,” explicitly vary:

- computation horizon,
- additional refinement blocks,
- repeated learned refinement cycles,
- or another supported dynamical mechanism.

Do not infer extra reasoning solely from finer integration.

---

# 37. Flow fine-tuning fairness

After pure flow distillation, an optional next stage is answer-loss fine-tuning:

\[
\mathcal L_{\text{answer}}
=
-\log p(y\mid x,Z(1))
\]

where gradients pass through the flow solver and trainable interface.

If the flow model receives this additional optimization:

**give the recurrent comparator a comparable additional training opportunity.**

For example:

- equal number of extra answer-loss updates,
- same train/validation data,
- comparable optimizer schedule.

Do not compare a post-tuned flow against an earlier recurrent checkpoint and attribute the gain to flow dynamics.

---

# 38. Later anti-collapse interventions

Do not implement dependency-gated optimization in v1.

First observe whether the frozen-backbone setup actually suffers latent bypass.

Diagnostics:

- ablate \(Z\),
- replace \(Z\) with noise,
- compare one-shot versus recurrence,
- measure cross-attention magnitudes.

If the backbone can bypass the workspace through the original prompt path and this prevents useful training, then evaluate anti-bypass interventions.

Possible later interventions:

- prompt-state dropout,
- restricted upper-layer prompt access,
- dependency diagnostics,
- carefully designed contrastive dependence losses,
- gradient gating.

These are contingency tools, not part of the initial claim.

---

# 39. Why no answer-conditioned posterior in v1

Posterior-based latent training is legitimate.

The reason for excluding it from v1 is experimental simplicity.

An answer-conditioned posterior:

\[
q(Z\mid x,y)
\]

can learn highly useful latent codes, including codes that directly contain answer information.

That is not inherently wrong.

However, strong reconstruction through such a posterior does not establish that the model learned a useful **question-only reasoning process**.

Therefore v1 evaluates question-only latent computation from the beginning.

Posterior methods may return later as:

- teachers,
- auxiliary analyses,
- or alternative training mechanisms.

---

# 40. Later public-dataset phase

Only after the controlled mechanism is understood should the project move to broad natural-language reasoning data.

Candidate sources retained from v1:

## OpenThoughts3

Use primarily for:

- math reasoning,
- multiple trajectories per problem,
- later analysis of latent solution diversity.

## OpenMathInstruct-2

Use for:

- large-scale math diversity,
- answer-supervised transfer.

## Bespoke-Stratos

Use for:

- smaller high-quality hard reasoning examples.

Do not begin the project with hundreds of thousands of examples from all three.

The controlled prototype should determine whether the architecture is worth scaling.

---

# 41. Public-data split rule

When public reasoning data is introduced:

**split by unique underlying problem, never by reasoning trajectory.**

All traces for one problem must remain in the same split.

Otherwise train/test leakage can occur through alternate trajectories.

---

# 42. Code domain later

After math / symbolic reasoning is established, code is a strong second domain because it has an objective verifier.

Possible sources can include:

- OpenThoughts code subset,
- programming datasets with test cases,
- generated algorithmic tasks.

Evaluation should be based primarily on unit-test correctness, not exact source-code match.

Do not add code until the core latent-computation result is stable.

---

# 43. Logging requirements

Every run must record:

- git commit,
- exact config,
- backbone checkpoint,
- dataset version/hash,
- generator version,
- train/validation/test split IDs,
- random seed,
- slot count \(N\),
- latent dimension,
- update count \(K\),
- whether \(K\) was fixed or sampled,
- indexed/exchangeable status,
- trainable parameter count,
- total parameter count,
- inference FLOPs estimate,
- training compute estimate where practical,
- latency,
- peak memory,
- accuracy,
- verifier pass rate,
- loss curves.

Later flow runs also record:

- NFE,
- solver type,
- tolerance,
- integration interval.

---

# 44. First plots

The first prototype should produce these plots automatically.

## Plot 1

Training and validation accuracy versus optimization step.

## Plot 2

Held-out accuracy versus update count \(K\).

Separate:

- trained horizons,
- extrapolated horizons.

## Plot 3

Accuracy versus inference FLOPs.

## Plot 4

Accuracy versus latency.

## Plot 5

Accuracy by solver-derived task difficulty bucket.

## Plot 6

Exchangeable versus indexed recurrence across difficulty.

Do not start with topology visualizations.

First establish that the system actually works.

---

# 45. First tables

## Table A — core prototype

| Method | K | Indexed? | Accuracy | Hard accuracy | FLOPs | Latency |
|---|---:|---|---:|---:|---:|---:|
| frozen backbone | 0 | n/a | | | | |
| one-shot latent | 1 | no | | | | |
| recurrent | 2 | no | | | | |
| recurrent | 4 | no | | | | |
| recurrent | 8 | no | | | | |
| recurrent indexed | 4 | yes | | | | |
| recurrent indexed | 8 | yes | | | | |

## Table B — architecture at matched compute

Include:

- recurrent,
- feedforward,
- sequential,
- later flow.

## Table C — same-checkpoint scaling

One variable-\(K\) checkpoint evaluated across:

- trained horizons,
- unseen horizons.

---

# 46. Seeds and statistical discipline

For early debugging:

- 1 seed is acceptable.

For any claimed effect:

- target at least 3 seeds.

For small differences:

- use more seeds if affordable,
- report mean and spread,
- avoid declaring victory from one best run.

Predefine the primary metric before the final comparison.

---

# 47. Failure conditions

## Failure 1

Recurrent workspace does not beat one-shot on held-out difficulty.

Interpretation:

No evidence that iterative latent computation helps this task/setup.

## Failure 2

Recurrent gains disappear against a compute-matched feedforward alternative.

Interpretation:

Extra compute helps, but recurrence may not be special.

## Failure 3

Exchangeable and indexed variants behave identically.

Interpretation:

Slot identity may not matter for the tested task.

This is still informative.

## Failure 4

Joint recurrence does not outperform sequential computation.

Interpretation:

No evidence yet for workspace-style computation.

## Failure 5

Flow cannot reproduce teacher quality efficiently.

Interpretation:

Flow may be an unnecessary mechanism.

## Failure 6

All architecture differences vanish on a second task family.

Interpretation:

The first result may be task-specific.

---

# 48. Success conditions

A strong progression would be:

1. recurrence beats one-shot,
2. gains persist on held-out harder instances,
3. gains are competitive against a compute-matched feedforward model,
4. joint recurrence differs meaningfully from sequential computation,
5. the effect transfers to a second task family,
6. flow later matches or improves the recurrent system under a useful efficiency metric.

Only the final step supports a specifically diffusion/flow-centered paper.

Earlier steps can still support a latent-workspace paper.

---

# 49. Revised project decision tree

```text
Does iteration beat one-shot?
        │
   ┌────┴────┐
   no        yes
   │          │
rethink     Does recurrence beat/compete with
setup       compute-matched feedforward?
                   │
              ┌────┴────┐
              no        yes
              │          │
        extra compute   Does joint workspace differ
        explains gain   from sequential computation?
                             │
                        ┌────┴────┐
                        no        yes
                        │          │
                  weaker claim   workspace result
                                   │
                              Does exchangeability matter?
                                   │
                           characterize, do not force
                                   │
                              Does flow help efficiency
                              or quality beyond recurrence?
                               /                 \
                             no                   yes
                             │                     │
                   recurrent workspace      DWR / flow result
                   remains the result
```

---

# 50. Revised project naming

Keep **Diffusive Workspace Reasoning** as the working project name because diffusion remains the ambitious endpoint.

But internal experiment names should not presuppose diffusion.

Use:

- `RWS` — recurrent workspace,
- `RWS-EX` — exchangeable recurrent workspace,
- `RWS-IDX` — indexed recurrent workspace,
- `FF-WS` — feedforward workspace,
- `SEQ-LAT` — sequential latent computation,
- `FM-WS` — flow-matched workspace.

This prevents the codebase from treating diffusion as inevitable.

---

# 51. Suggested repository structure

```text
dwr/
├── README.md
├── configs/
│   ├── backbone/
│   ├── tasks/
│   ├── workspace/
│   └── experiments/
├── data/
│   ├── generators/
│   ├── solvers/
│   ├── splits/
│   └── public/
├── dwr/
│   ├── models/
│   │   ├── prompt_encoder.py
│   │   ├── cross_attention_interface.py
│   │   ├── recurrent_workspace.py
│   │   ├── indexed_workspace.py
│   │   ├── feedforward_workspace.py
│   │   ├── sequential_latent.py
│   │   └── flow_workspace.py
│   ├── tasks/
│   │   ├── mini_sudoku.py
│   │   ├── sequential_depth.py
│   │   └── verifiers.py
│   ├── training/
│   │   ├── train_workspace.py
│   │   ├── train_variable_k.py
│   │   └── train_flow.py
│   ├── eval/
│   │   ├── accuracy.py
│   │   ├── difficulty.py
│   │   ├── compute.py
│   │   └── permutation.py
│   └── utils/
├── experiments/
│   ├── prototype/
│   ├── fixed_k/
│   ├── variable_k/
│   ├── indexed_vs_exchangeable/
│   ├── matched_compute/
│   ├── sequential/
│   └── flow/
└── paper/
    ├── figures/
    ├── tables/
    └── notes/
```

---

# 52. Prototype configuration

Recommended starting configuration:

```yaml
backbone:
  size: 1.5B-2B
  pretrained_weights: frozen
  prompt_state_layer: fixed_across_methods

workspace:
  num_slots: 32_or_64
  latent_dim: 768
  init: iid_gaussian
  update: residual_shared
  slot_positional_embeddings: false

interface:
  type: set_cross_attention
  trainable: true
  same_across_core_methods: true

training:
  objective: final_answer_cross_entropy
  fixed_k_runs: [1, 2, 4, 8]
  variable_k_train_support: [1, 2, 4, 8]
  variable_k_eval_extrapolation: [12, 16]

task:
  family: verified_constraint_completion
  unique_solution_preferred: true
  difficulty_from_reference_solver: true

evaluation:
  seeds_claimed_result: 3
  report_flops: true
  report_latency: true
  report_difficulty_buckets: true
```

---

# 53. Exact first-week implementation checklist

## Data and verifier

- [ ] Implement first task generator.
- [ ] Implement reference solver.
- [ ] Implement uniqueness checking.
- [ ] Record solver difficulty statistics.
- [ ] Freeze train/validation/test splits.
- [ ] Unit-test verifier on valid and invalid solutions.

## Backbone and interface

- [ ] Load frozen base model.
- [ ] Extract prompt hidden states.
- [ ] Add set cross-attention adapters.
- [ ] Confirm original backbone weights receive no gradients.
- [ ] Confirm new interface weights do receive gradients.

## One-shot model

- [ ] Sample iid latent slots.
- [ ] Apply one residual workspace block.
- [ ] Decode answer.
- [ ] Train to nontrivial validation accuracy.

## Recurrent model

- [ ] Share workspace-block parameters across steps.
- [ ] Implement K=2,4,8.
- [ ] Confirm gradient flows through all updates.
- [ ] Log latent norms by step.
- [ ] Check for exploding/vanishing states.

## Indexed variant

- [ ] Add learned slot identity embeddings.
- [ ] Keep everything else matched.
- [ ] Train under same schedules.

## Evaluation

- [ ] Run held-out accuracy.
- [ ] Run hard-bucket accuracy.
- [ ] Measure latency.
- [ ] Estimate FLOPs.
- [ ] Run at least 3 seeds for the first claimed effect.

---

# 54. Questions deliberately postponed

Do not solve these before the recurrent prototype produces data:

- Which diffusion schedule is best?
- Should we use DDPM or flow matching?
- Should we use a posterior encoder?
- Should we add dependency-gated backprop?
- Should latent slot count vary at inference?
- Should we add RL?
- Should we use OpenThoughts3 immediately?
- Should we scale to 3B or 7B?
- Should we analyze emergent graph topology?
- Should we train multimodal latent distributions?

Those questions matter later.

They are not blockers for v1.

---

# 55. Later scaling plan

If the recurrent prototype succeeds:

## Phase 2

Add:

- second controlled task family,
- feedforward matched-compute baseline,
- sequential latent baseline,
- faithful external LRT comparison where feasible.

## Phase 3

Implement recurrent-teacher latent collection and flow matching.

## Phase 4

Move to ~3B backbone.

## Phase 5

Introduce public math datasets:

- OpenThoughts3,
- OpenMathInstruct-2,
- Bespoke-Stratos.

## Phase 6

Evaluate code tasks with executable verifiers.

Only then consider larger-scale training.

---

# 56. Revised paper possibilities

The eventual paper title depends on what survives.

## If recurrence is the main result

**Latent Workspace Reasoning: Joint Iterative Computation for Frozen Language Models**

## If exchangeability is important

**Exchangeable Latent Workspaces for Iterative Language-Model Reasoning**

## If flow adds a clear efficiency or quality advantage

**Diffusive Workspace Reasoning: Generative Flow over Latent Reasoning Workspaces**

Do not decide the final title before the experiments.

---

# 57. Claims discipline

Allowed early claim:

> Repeated updates to a latent workspace improve performance on held-out verified constraint tasks.

Stronger claim requiring evidence:

> Joint workspace computation is more effective than sequential latent computation under matched compute.

Stronger still:

> Exchangeable slot structure improves generalization.

Diffusion-specific claim requiring the later flow experiment:

> A learned flow over latent workspace states improves inference efficiency or task performance relative to deterministic recurrence.

Avoid:

- “human-like thought,”
- “brain-inspired cognition,”
- “true reasoning,”
- “diffusion inherently thinks better,”
- “nonsequential thought” solely because slots are exchangeable,
- “more solver steps means more reasoning.”

---

# 58. Final stop rule

Do **not** stop simply because diffusion loses.

Continue if the experiments reveal a robust and interesting workspace effect.

Stop or substantially rethink the project if:

1. iteration does not reliably beat one-shot,
2. gains disappear under compute-matched feedforward controls,
3. results fail to transfer beyond one narrow generator,
4. or the latent interface contributes no measurable task benefit.

The mechanism that survives the experiments becomes the paper.

---

# 59. Immediate next action

The project is now ready for implementation.

The immediate target is:

> **One frozen backbone + one verified constraint task + one trainable set cross-attention interface + fixed slot count + answer-only training + one-shot, exchangeable recurrent, and indexed recurrent variants at K ∈ {1,2,4,8}.**

Do not add diffusion before this result exists.

The next useful evidence should be:

- learning curves,
- held-out verifier accuracy,
- hard-instance accuracy,
- accuracy versus update count,
- accuracy versus FLOPs,
- latency,
- seed variance.

That evidence decides the next architectural step.

---

# 60. One-sentence project rule

**Lock the scientific question, not the mechanism: first establish whether iterative joint latent workspaces help, then let the experiments decide whether recurrence, slot identity, sequential computation, or flow dynamics are actually responsible.**
