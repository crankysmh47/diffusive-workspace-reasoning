# Diffusive Workspace Reasoning (DWR)

Testing joint iterative latent workspaces against one-shot, indexed, and sequential latent computation for frozen language models.

**Status:** Repository bootstrap. The generator, solver, models, training, and evaluation are not implemented yet. No experimental results are claimed.

The first question is whether repeated joint updates to a latent workspace improve verified constraint completion. Diffusion/flow matching is a later experiment, conditional on understanding a working recurrent teacher.

## Start here

- [Research plan v2](docs/diffusive_workspace_reasoning_project_plan_v2.md): authoritative scientific scope, preserved from the supplied plan.
- [Implementation checklist](todo.md): ordered work and verification checkpoints.
- [Prototype specification](configs/prototype.json): initial settings and decisions still to resolve; not an executable training configuration yet.
- [Contribution instructions](AGENTS.md): implementation and experiment rules.

## First experiment

| Component | Initial scope |
| --- | --- |
| Task | Verified 4×4 mini-Sudoku, unique solutions, solver-derived difficulty |
| Backbone | One frozen 1.5B–2B decoder-only base model; checkpoint selection pending |
| Workspace | 64 slots, dimension 768, iid Gaussian initialization |
| Update | `Z_next = Z + F(LN(Z), X)`, shared parameters across steps |
| Interface | Newly added trainable set cross-attention adapters, fixed across core methods |
| Training | Final-answer cross-entropy; original pretrained weights frozen |
| Methods | Frozen baseline, one-shot, exchangeable recurrent, indexed recurrent |
| Fixed horizons | Separate checkpoints at K = 1, 2, 4, 8 |
| Variable horizons | Train on K = 1, 2, 4, 8; label K = 12, 16 as unseen extrapolation |
| Evaluation | Verifier accuracy, hard-bucket accuracy, latency, FLOPs, peak memory, seed spread |

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

Every experiment must capture commit, exact configuration, model revision, dataset hash, generator version, split IDs, seed, N/d/K, slot identity, parameter counts, accuracy, verifier pass rate, latency, memory, and compute estimates. Use at least three seeds for claimed effects where feasible. Test sets remain fixed after results are inspected.

**Next action:** fix the 4×4 mini-Sudoku format, implement the reference solver and verifier, then generate unique puzzles and freeze leakage-controlled splits. Follow the checklist before adding model complexity.

## Licensing

No open-source license is assigned at bootstrap. Redistribution and licensing decisions remain with the owner; third-party models and datasets retain their own terms.
