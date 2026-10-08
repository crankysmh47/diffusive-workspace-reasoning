# Diffusive Workspace Reasoning (DWR)

Studying whether exchangeable latent workspaces learn reusable computation that transfers to unseen task compositions and structures.

**Status:** The v2 CPU generation and independent audit implementation is complete. Its single feasibility attempt stopped at the registered source attempt cap; the audit passed integrity and partial replay, while G2 remains unassessed. No model, training, or transfer result is claimed. See [v2 suitability and attempt result](docs/task_suitability_v2.md), [execution state](docs/execution_state.md), and the [v2 execution plan](docs/task_v2_execution_plan.md). The earlier v1 evidence remains recorded separately in [v1 suitability](docs/task_suitability.md).

The main question is whether one trained workspace and decoder interface can transfer without task-specific retraining. Exchangeability is the central architectural hypothesis. Variable slot count is a supporting experiment after a fixed-size transfer study; diffusion and other alternatives are parked.

## Start here

- [Research plan v2](docs/diffusive_workspace_reasoning_project_plan_v2.md) and [binding v2.2 protocol](docs/research_protocol.md): scientific scope, experiment design and evidence gates.
- [Main research direction](docs/transfer_research_direction.md): transfer, exchangeability and variable-slot-count hypotheses.
- [V2 task specification](docs/task_spec_v2.md) and [v2 implementation plan](docs/task_v2_execution_plan.md): registered generation proposal, execution steps and stop rules.
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

**Next action:** prepare a versioned development-only design for bounded source-generation yield/quota feasibility, then address composition support. Preserve final-target definitions and this attempt. Its empty development cohorts leave support unavailable; they do not show target mismatch or zero target yield. Do not rerun under altered rules. Workspace/model work remains blocked until a complete G2 audit passes, and training also requires a separate study registration.

## Licensing

No open-source license is assigned at bootstrap. Redistribution and licensing decisions remain with the owner; third-party models and datasets retain their own terms.
