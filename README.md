# Diffusive Workspace Reasoning (DWR)

Studying whether exchangeable latent workspaces learn reusable computation that transfers to unseen task compositions and structures.

**Status (2026-10-12):** The six-variable study remains `NO_GO_DEVELOPMENT_CAPACITY`: source suitability passed, but EQ/NEQ has only 57 usable groups against 100. A registered [retained diagnosis](docs/coverage_diagnosis_result.md) reproduced the permitted counts and identified nonuniqueness as its main loss. The next step is the separately registered [sampler v2 mechanism fixture](docs/certificate_sampler_v2_design.md): necessary EQ/NEQ filters and a fixed ADD branch designed to require interacting constraints. Implementation and review are in progress; the fixture has not run. Final tasks, full G2, workspace and training remain unattempted. The exchangeable workspace hypothesis is untested. See [execution state](docs/execution_state.md) and [todo.md](todo.md) for verified milestones.

The main question is whether one trained workspace and decoder interface can transfer without task-specific retraining. Exchangeability is the central architectural hypothesis. Variable slot count is a supporting experiment after a fixed-size transfer study; diffusion and other alternatives are parked.

The earlier [four-variable mechanism check](docs/constructive_task_design.md) verified necessity certificates on 12 total variants, including six fixed bases, and motivated the broader constructor. Historical results remain preserved: the [CPU follow-up](docs/task_suitability_cpu_followup.md) failed its source sensitivity/yield floors, and the [original v4 run](docs/task_suitability_v4.md) remains operationally unassessed.

## Start here

- [Research plan v2](docs/diffusive_workspace_reasoning_project_plan_v2.md) and [binding v2.2 protocol](docs/research_protocol.md): scientific scope, experiment design and evidence gates.
- [Main research direction](docs/transfer_research_direction.md): transfer, exchangeability and variable-slot-count hypotheses.
- [Sampler v2 design](docs/certificate_sampler_v2_design.md), [fixture registration](configs/tasks/certificate_sampler_v2_fixture.json) and [plan](docs/superpowers/plans/2026-10-12-certificate-sampler-v2.md): active bounded mechanism/diversity qualification, before another capacity study or model work.
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
| Task | Controlled finite-domain constraint compositions; implemented grammar and verifier, with suitability still gated; Sudoku remains optional fixture work |
| Backbone | One frozen 1.5B–2B decoder-only base model; checkpoint selection pending |
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

Python 3.11 or 3.12 is the initial supported runtime. The bootstrap package has no runtime dependencies. PyTorch/CUDA and model dependencies will be added when the backbone is selected; there are no training commands yet.

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

**Next decision:** specify a broader six-variable constructor using the checked necessity-certificate principle, with capacity, raw-attempt yield and difficulty diagnostics registered before any population run. The fixed fixture motifs are correctness examples. Preserve the failed study and its thresholds; it has no retry allowance. Workspace/model work requires a complete passing five-cohort G2 audit, and training requires a separate study registration.

## Licensing

No open-source license is assigned at bootstrap. Redistribution and licensing decisions remain with the owner; third-party models and datasets retain their own terms.
