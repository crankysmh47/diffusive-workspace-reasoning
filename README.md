# Diffusive Workspace Reasoning (DWR)

Studying whether exchangeable latent workspaces learn reusable computation that transfers to unseen task compositions and structures.

**Status (2026-10-19):** The bounded [K0/K1/K4 source-validation pilot is running on the GPU](docs/source_validation_pilot_launch.md), with actual forward passes and persisted predictions confirmed. Each arm has512 registered training updates and45 fixed validation examples. The earlier [tiny-source run](docs/source_overfit_result.md) passed3/3 on its three memorized examples; broader learning, workspace specificity, transfer and novelty remain unestablished. Full end-to-end recovery acceptance is still partial and disclosed. Read the [execution state](docs/execution_state.md) and [todo.md](todo.md) for completed and remaining work.

The main question is whether one trained workspace and decoder interface can transfer without task-specific retraining. Exchangeability is the central architectural hypothesis. Variable slot count is a supporting experiment after a fixed-size transfer study; diffusion and other alternatives are parked.

The earlier [four-variable mechanism check](docs/constructive_task_design.md) verified necessity certificates on 12 total variants, including six fixed bases, and motivated the broader constructor. Historical results remain preserved: the [CPU follow-up](docs/task_suitability_cpu_followup.md) failed its source sensitivity/yield floors, and the [original v4 run](docs/task_suitability_v4.md) remains operationally unassessed.

## Start here

- [Project history and current status](docs/project_history_and_status.md): original ideas, architecture choices, implemented infrastructure, every completed study, interrupted work and the GitHub handoff.
- [Rank-certified design](docs/certificate_rank_design.md) and [implementation plan](docs/superpowers/plans/2026-10-12-certificate-rank.md): prospectively registered path; implementation verification is recorded separately.

- [Research plan v2](docs/diffusive_workspace_reasoning_project_plan_v2.md) and [binding v2.2 protocol](docs/research_protocol.md): scientific scope, experiment design and evidence gates.
- [Main research direction](docs/transfer_research_direction.md): transfer, exchangeability and variable-slot-count hypotheses.
- [Sampler v2 design](docs/certificate_sampler_v2_design.md), [fixture registration](configs/tasks/certificate_sampler_v2_fixture.json), [plan](docs/superpowers/plans/2026-10-12-certificate-sampler-v2.md) and [result](docs/certificate_sampler_v2_fixture_result.md): closed mechanism fixture; hard-cell coverage blocks a capacity study or model work.
- [Retained coverage diagnosis](docs/coverage_diagnosis_result.md) and [compact manifest](configs/splits/certificate-coverage-diagnosis-result.json): completed descriptive analysis; original scientific NO-GO unchanged.
- [Six-variable study](docs/constructive_sixvar_study.md), [implementation plan](docs/constructive_sixvar_execution_plan.md) and [result](docs/task_suitability_sixvar.md): completed constructor diagnostic and current development-coverage blocker; its single population allowance is closed.
- [CPU follow-up](docs/cpu_operational_followup.md) and [result](docs/task_suitability_cpu_followup.md): closed single-attempt study with unchanged scientific floors and a source suitability NO-GO.
- [V4 task specification](docs/task_spec_v4.md): preserved development-probe proposal and terminal limits. Its sole probe is operationally unassessed; see the [suitability report](docs/task_suitability_v4.md) and [compact result manifest](configs/splits/constraint-v4-development-probe-result.json). No full-run or model stage is authorized by this result.
- [V3 task specification](docs/task_spec_v3.md) and [v3 implementation plan](docs/task_v3_execution_plan.md): historical registered generation proposal and stop rules. The v3 feasibility attempt ended NO-GO; see its [suitability report](docs/task_suitability_v3.md) and [compact evidence manifest](configs/splits/constraint-v3-feasibility-nogo.json).
- [V2 task specification](docs/task_spec_v2.md) and [v2 implementation plan](docs/task_v2_execution_plan.md): preserved prior generation proposal and result.
- [Spare ideas](docs/spare_research_ideas.md): alternatives outside the active roadmap.
- [Research-readiness review](docs/research_review.md): design corrections, related-work starting map and unresolved decisions.
- [Implementation checklist](todo.md): ordered work and verification checkpoints.
- [Prototype specification](configs/prototype.json): initial settings and decisions still to resolve; not an executable training configuration yet.
- [Contribution instructions](AGENTS.md): implementation and experiment rules.

## First experiment

| Component | Initial scope |
| --- | --- |
| Task | Controlled finite-domain constraint compositions; five-stratum structural G2 passed, model precision remains unassessed; Sudoku stays an optional fixture |
| Backbone | One frozen 1.5B–2B decoder-only base model; pinned Qwen2.5-1.5B native integration is prospective |
| Workspace | 64 slots, dimension 768, iid Gaussian initialization |
| Update | `Z_next = Z + F(LN(Z), X)`, shared parameters across steps |
| Interface | Newly added trainable set cross-attention adapters, fixed across core methods |
| Training | Final-answer cross-entropy; original pretrained weights frozen |
| Methods | Frozen reference, trained K=0 noise-memory interface control, one-shot, exchangeable recurrent, indexed recurrent |
| Fixed horizons | Primary fixed K=4 with K=1 control; wider sweep deferred |
| Variable horizons | Deferred secondary study; distinguish trained and unseen horizons |
| Evaluation | Valid-solution rate, supported hard-distribution accuracy, paired effects/uncertainty, end-to-end latency/FLOPs and peak memory |

Target hardware: one RTX 5060 Ti with 16 GB VRAM, 32 GB system RAM, Windows 11. Measure memory before committing to model/dtype/batch settings. Smaller slot counts are permitted only as documented experimental decisions applied consistently across comparisons.

## Local setup

Python 3.11 or 3.12 is supported. Task generation/verification uses the standard library; neural components require the optional `ml` extra with pinned torch2.11.0. The current isolated Windows `.venv-ml` uses the official2.11.0+cu128 build. Global environments are untouched. The optional `backbone` extra additionally pins transformers4.57.1; native CPU integration is verified.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -c "import dwr; print(dwr.__version__)"
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -c 'import dwr; print(dwr.__version__)'
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `dwr/` | Python package; add task, model, training, and evaluation modules as implemented |
| `configs/` | Versioned experimental settings |
| `docs/` | Research plan and subsequent decisions |
| `todo.md` | Ordered implementation milestones |
| `data/`, `runs/`, `checkpoints/` | Local generated artifacts; ignored by Git |

## Research rules

Keep dataset splits and decoder access matched. Check exchangeable updates for permutation equivariance and decoder outputs for permutation invariance. Verify backbone freezing without blocking gradient flow through the trainable adapters and workspace. Keep same-checkpoint horizon scaling separate from architecture-at-budget comparisons.

Every experiment must capture source/configuration/model/data identities, environment, RNG streams, per-example outcomes, parameter counts and full costs. Confirmatory effects require a preregistered comparison, at least three training seeds, suitable uncertainty analysis and a fixed test-access policy. Preserve failed and negative runs. See the protocol for exact requirements.

The active pilot runs independently of this terminal. Use its frozen producer checkout and pinned local artifacts; keep that checkout unchanged while the job runs. The [launch record](docs/source_validation_pilot_launch.md) gives its exact source, artifact paths, limitations and recovery policy:

```powershell
.venv-ml\Scripts\python.exe -m dwr.experiments.source_pilot_job status --config configs/experiments/source_validation_pilot_tokenizer_repair_v4.json
.venv-ml\Scripts\python.exe -m dwr.experiments.source_pilot_job start --config configs/experiments/source_validation_pilot_tokenizer_repair_v4.json
```

`start` attaches/reports rather than launching a competing worker. Each completed optimizer boundary is checkpointed. At most two eligible process-interruption recoveries per arm share the original budget; known faults stop. Local checkpoints do not provide disk-failure or archival protection, and complete native recovery acceptance remains outstanding.

**Next work:** finish the registered pilot and its workspace diagnostics, audit the results and apply its predeclared gates. Fixed-N EX/IDX transfer and confirmatory training follow only if justified by that evidence.

## Licensing

No open-source license is assigned at bootstrap. Redistribution and licensing decisions remain with the owner; third-party models and datasets retain their own terms.
