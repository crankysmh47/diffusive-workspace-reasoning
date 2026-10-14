# Working in DWR

Read `README.md`, `todo.md`, `docs/research_protocol.md`, and relevant v2 plan sections before changing scientific behavior. The v2.2 protocol amends v2 and takes precedence. The current structural gate is the passing separately registered `docs/five_stratum_study_result.md`; its qualification/full allowances are closed. Fixed-N tensor-core work follows `docs/fixed_workspace_core_design.md`, with Luna implementation and root decisions. Earlier six-variable, sampler and rank NO-GOs remain unchanged and cannot be retried. Read `docs/transfer_research_direction.md` for the active transfer/slot hypothesis; `docs/spare_research_ideas.md` is not an implementation backlog. Record deliberate departures with date, rationale, affected comparisons and evidence in `docs/` before running a changed experiment.

- When implementation is requested, implement the next unchecked milestone; a planning/review request does not authorize training. Do not introduce diffusion, posterior encoders, dependency-gated optimization, or public datasets before their prerequisites.
- Read `docs/constructive_task_design.md` for the earlier fixture-only certificate check. Its mechanism PASS alone does not pass corpus suitability; the later six-variable study has its own registered population and preserved terminal result.
- The six-variable study and implementation in `docs/constructive_sixvar_execution_plan.md` are complete and closed at a development coverage/representation failure. Preserve its source/config/artifacts; no retry, full G2 or model stage follows. A future construction/support change needs a separately versioned prospective design and review before another population. The owner-requested Luna implementation model remains binding.
- Keep changes small. Use Python's standard library for generation, reference solving, manifests, and logging where practical. Add ML dependencies only when needed.
- Put implementation in `dwr/`; create modules as they become real, not placeholder architectures.
- Preserve matched prompt encoding, stochastic initialization, decoder interfaces, splits, and reporting across comparisons.
- Preserve the residual update equation and explicit exchangeable/indexed distinction.
- Frozen parameters do not mean all decoder computation belongs in `no_grad()`: gradients must reach the newly inserted interface and workspace.
- Never condition the v1 workspace on the ground-truth answer. Never alter held-out splits based on results.
- Add meaningful correctness checks for solvers/verifiers, leakage, gradient flow, and permutation behavior when those implementations land. Use `unittest` initially; no GPU training required for basic task checks.
- Log reproducibility metadata and report actual test/check outcomes. Update `todo.md` only for completed and verified work.
- Consolidate verified checkpoints; the owner requests minimal commits. Do not make per-task/status commits or merge the research PR without owner instruction.
- Keep generated datasets, checkpoints, model weights, credentials, and run artifacts out of Git. Commit small configurations and split manifests/hashes, not large data.
- Support Windows paths; avoid mandatory Bash scripts, symlinks, or platform-specific ML dependencies.
- Do not choose an open-source license or change repository visibility without owner instruction.

## Research discipline

- Keep one trained workspace/interface shared across source tasks and frozen across transfer targets; no target-specific tuning. Treat 4×4 Sudoku as an optional correctness fixture. Pass the research-task diversity, difficulty and family-split gate before model comparisons; do not relax leakage controls to fill a dataset.
- Register the hypothesis, primary contrast/metric, practical effect size, sample/seed counts, checkpoint rule, tuning/compute caps and stop rule before confirmatory training. Null required settings block execution.
- Use development/validation data for choices. Freeze test data and log every test access. Test-driven changes make subsequent results exploratory; retain the original results and version any new study.
- Distinguish equal data exposure, equal training compute and equal inference compute. Report actual parameters, costs and deviations; no claim of simultaneous matching without measurements.
- Use at least three training seeds for confirmatory effects, paired evaluation where possible and uncertainty respecting family grouping. Fewer seeds are exploratory; nonsignificance is not equivalence.
- Require feedforward controls for recurrence-specific efficiency claims and sequential controls for joint-over-sequential claims. Diffusion/flow success is not required for a useful result.
- Preserve negative, inconclusive, diverged and interrupted runs. Bound diagnostics and retries in advance; never tune until a positive result appears.
- Check prior work before novelty claims. Label paper evidence, repository checks, agent-run experiments and independent reproduction accurately.

- Establish fixed-N transfer before variable-N experiments. Distinguish slot permutation, input symmetry, unseen N and adaptive allocation; none proves the others. Keep K controlled when testing N.

## Reproducibility and code care

- Write a resolved manifest before each run: command, source/config/data/model identities, environment, independent RNG streams, precision/determinism settings and budgets. Confirmatory runs require a clean source commit.
- Use unique artifact directories, stable serialization, SHA-256 hashes, explicit UTF-8 and Windows-safe paths. Save per-example outcomes and aggregate counts; back up artifacts before reproducibility claims.
- Checkpoint optimizer/scheduler/scaler, all RNG states, sampler position and identities; test resume behavior. Do not assume a seed guarantees bitwise determinism.
- Fail on unknown config keys, invalid shapes, nonfinite values or incompatible resumes. Never silently skip examples, truncate inputs, swap models/dtypes, weaken verification or detach recurrent gradients.
- Add a regression check with each scientific bug fix. Cross-check solver/verifier logic independently; a shared generator/verifier bug must not certify itself.
- Keep prompt-only X isolated from answers and trainable adapters. Test causal masks, loss shifting/masking, frozen-weight immutability, gradients and permutation behavior when ML components land.
- Measure full forward/backward memory before launching a run matrix. Include prompt encoding and answer decoding in cost reports; block counts alone are not FLOPs.
- Record exact verification commands, outcomes and evidence paths before checking off a milestone. An empty test suite, configuration parse or CPU mock is not model validation.
