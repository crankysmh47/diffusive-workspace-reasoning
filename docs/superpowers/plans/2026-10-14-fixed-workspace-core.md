# Fixed-N workspace tensor core implementation plan

> For agentic workers: Use subagent-driven-development, Luna implementation and independent reviews. Root owns design/environment/verification. Owner delegated routine design; no further permission handoff, and minimal commits.

Goal: Build the real fixed-N EX/IDX recurrent tensor core and invariant set reader after registered structural G2, without native-model training.
Architecture: two small PyTorch modules, no generic training framework, no soft tokens or placeholder backbone wrapper.
Tech stack: Python3.11,torch2.11.0+cu128,unittest,Windows-safe isolated environment.
Spec: docs/fixed_workspace_core_design.md.

## Global constraints

- Z_next=Z+F(LN(Z),X), same block every step, F=a+b+c as exactly defined in spec. EX no IDs; IDX learned N*d IDs added once at initialization. Same shared weights/interface/noise; explicit local noise RNG.
- Fixed instance N64/d768/K4 defaults; 12 heads,4dMLP,dropout0. Tiny tests declared smaller. Context/decoder widths explicit. Reject unknown kinds, bad nonboolean dimensions/K/shapes/devices/dtypes, nonfinite values, all-padding, or trainable X; no detach/fallback/truncation.
- Reader H+gCA(LN_H(H),LN_Z(Z),LN_Z(Z)),gate0.1; same future methods, set invariant even for IDX. Native causal-mask/encoder/loss/cache integration not claimed.
- No source/test tasks, pretrained backbone, datasets, generator, GPU probe or scientific training. Only synthetic CPU unit optimizer/gradient checks. Existing G2 artifacts/closed allowances untouched.
- Optional torch only; global nightly untouched; .venv-ml isolated pinned stablecu128. Stdlib task tests can run withoutML via explicit skips; actualML verification requires real executedtests. No worker installs/updatesdeps, commits or fullsuite; no worker subagents.
- Keep code under dwr/models; root package no mandatory torch import. No separate status commits; retain all evidence ignored per-plan.

## Task1: fixed workspace and indexed control

Own create dwr/models/workspace.py,dwr/models/__init__.py,tests/test_workspace.py; modify pyproject.toml only adding project.optional-dependencies ml=['torch==2.11.0']. No reader/docs/task-code/config/artifact edits.

- [x] Read spec, write meaningful RED for noise/shape/once-ID/residual/shared weights/equivariance/padding/all-K-state gradient properties; scoped tests with .venv-ml/Scripts/python.exe.
- [x] Implement WorkspaceDelta,FixedWorkspace interfaces exactly, identities after shared initialization. Mask semantics True=padding. Return optional live intermediate states. No answer parameter or backbone dummy architecture.
- [x] Record current torch environment, exact commands/output, filehash/LF/diff; scopedtests only. Self-review.
- [x] Independent task review and scoped fixes; document API for dependent reader task. No commit.

## Task2: invariant set reader and frozen-gradient integration

Own create dwr/models/interface.py,tests/test_workspace_interface.py and update dwr/models/__init__.py exports. Task1 files untouched unless root ruling.

- [x] RED strict tensor shapes/finite/dtype/device, final whole-vector permutation invariance for EX/IDX, independent query outputs, real gradient through frozen tail and immutable frozen weights after optimizer step. Reader API has no mask; prompt-mask checks are in Task1.
- [x] Implement SetCrossAttentionAdapter exact API/equation. Synthetic optimizer fixture uses actual Task1workspace and a frozen linear tail; no native-LM claim. State_dict strict roundtrip keys/shapes test.
- [x] Scoped task1+reader tests, saved actualRED/GREEN/hashes/LF/diff/selfreview. Independent task review/scoped fixes. No commit/fullsuite.

## Task3: core release and next real-model plan

- [x] Root whole-core review with complete diff and task evidence; resolve load-bearing findings; verify all checks under the amended separated180s watchdog on stable bytes and exact environments. See [verification](../../fixed_workspace_core_verification.md).
- [ ] Update active docs/checklist, preserve G2/power/backup/host limitations. Consolidate G2 result+design+verifiedcore into one source checkpoint and push/update existing PR; nevermerge.
- [ ] Continue G3 native-backbone/run-data/evaluation/checkpoint infrastructure and a separately registered GPU memory/integration probe. No modeltraining until its prerequisites/budgets/registration are concrete. G2structuralPASS alone does not certify precision or trained transfer.

## Controller rulings

2026-10-14: Owner asked for EX/IDX after data gate, so this core is the next real component. Existing G3 native integration remains necessary afterward. Parameter-count differences are measured, not padded with dummy weights. K0forward initialization is tested but is not a trained-control registration. Torchstablecu128 isolated environment replaces no global package; selected from official available wheels. All tensor checks CPU-only, distinct from GPU/native-backbone/modelvalidation. Minimal commits overrides skill frequent-commit defaults.

2026-10-14 verification ruling: preserve the first monolithic548-test failure (two unchanged256MiB legacy guards exceeded by Torch import). Perform complete stdlib discovery with explicitMLskips in isolated `.venv-task`, and actual20 tensor checks withoutskips in `.venv-ml`, separate subprocesses under one combined180s cap. No guard/config/scientific allowance changes; see the design's dated departure. Report separate outcomes and old failure accurately.
