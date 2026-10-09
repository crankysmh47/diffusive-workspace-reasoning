# Diffusive Workspace Reasoning (DWR)

Studying whether exchangeable latent workspaces learn reusable computation that transfers to unseen task compositions and structures.

**Status:** The v4 CPU proposal implementation passed the frozen-source full suite (275 tests in 46.451 seconds) and its scoped final review was READY. The sole registered development probe recorded 1,000 source quota rows after 20,299 attempts, then exited 1 at its 900-second resource boundary with `NOT_ASSESSED_RESOURCE_CAP`. Source verification, replay, and audit were not assessed; development remained unattempted (0), no final or full run occurred, and G2 remains unassessed. This is an operationally unassessed proposal, not a scientific NO-GO or PASS. The current v4 proposal is closed: no retry, extension, retuning, full run, or model work follows from it. Any next experiment requires a new bounded prospective design decision. See the [v4 suitability report](docs/task_suitability_v4.md), [compact result manifest](configs/splits/constraint-v4-development-probe-result.json), [execution state](docs/execution_state.md), and [v4 specification](docs/task_spec_v4.md). Historical v1, v2, and v3 evidence remains recorded in their [suitability reports](docs/task_suitability.md), [v2 report](docs/task_suitability_v2.md), and [v3 report](docs/task_suitability_v3.md).

The main question is whether one trained workspace and decoder interface can transfer without task-specific retraining. Exchangeability is the central architectural hypothesis. Variable slot count is a supporting experiment after a fixed-size transfer study; diffusion and other alternatives are parked.

## Start here

- [Research plan v2](docs/diffusive_workspace_reasoning_project_plan_v2.md) and [binding v2.2 protocol](docs/research_protocol.md): scientific scope, experiment design and evidence gates.
- [Main research direction](docs/transfer_research_direction.md): transfer, exchangeability and variable-slot-count hypotheses.
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

**Next action:** stop the v4 proposal at `NOT_ASSESSED_RESOURCE_CAP`. Do not retry, extend the budget, retune, allocate v5, or proceed to the conditional full/G2/model stage from this result. A `PASS_DEVELOPMENT_PROPOSAL` would only permit a separate full-run design/schema/linkage task and one 2,400-second full G2 gate; the actual probe was not PASS. Workspace/model work requires a complete passing five-cohort G2 audit, and training requires a separate study registration.

## Licensing

No open-source license is assigned at bootstrap. Redistribution and licensing decisions remain with the owner; third-party models and datasets retain their own terms.
