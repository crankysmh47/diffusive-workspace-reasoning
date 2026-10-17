# Native backbone and answer-loss CPU verification

2026-10-15. Implements the [native design](native_backbone_design.md) and [token/loss contract](token_loss_design.md) on `research/transfer-cpu-v1`, after base commit `ecb05c6`. This is synthetic CPU integration evidence, not pretrained inference, GPU feasibility, task learning, transfer or independent-person reproduction.

Luna implemented the actual pinned Qwen2 host, source collator and shifted answer-only loss. Independent task reviews are READY, including fixes for optional-dependency import behavior and exception cleanup. A separate whole-stage review checks native/training integration. Original negative reviews, failing attempts, snapshots and review receipts remain under ignored `.superpowers/sdd/native-host-20261014/`; these are local evidence, not archival backup.

## Root release check

No source writer was active during verification. Root ran `.venv-task/Scripts/python.exe .superpowers/sdd/native-host-20261014/verify_separate_suites.py` under one combined180-second watchdog, preserving the separate-process policy established by the [core verification](fixed_workspace_core_verification.md). No historical memory guard was relaxed.

- Fresh `.venv-task` process asserted Torch absent, then `-m unittest discover -s tests -v`: **564 reported, 530 passed and34 explicit optional-ML skips**,41.782s unittest /42.000s wrapper, exit0.
- `.venv-ml` loaded discovery patterns `test_workspace*.py`, `test_qwen2_workspace.py`, `test_training_batch_loss.py` into one unittest suite, explicitly requiring **45 methods and zero skips**: **45 passed**,0.566s unittest /3.812s wrapper, exit0. These comprise20 core/reader,17 native/import-regression and8 batching/loss checks. Two import-regression methods also run in the stdlib process; there are573 distinct executed checks across the processes.
- Combined46.313s, no watchdog timeout. All `dwr/**/*.py`, `tests/**/*.py`, `configs/**/*.json` and `pyproject.toml` SHA256 inventories matched before/after.

Receipt: `.superpowers/sdd/native-host-20261014/controller-separated-suites-20261015T052403039400Z.json`. Stdlib log `-1.log` SHA256 `0802643cd7d099e651bb4c27968431b5326ee328dfaeb820ec4e34d9faab8538`; ML log `-2.log` SHA256 `9a5c55ef11307d532394afbf64d7aadb1e671a2a1d3253d23b9386056bfe58c4`. Commands, byte inventories, environment and individual original logs are retained with the receipt.

## What this establishes

Actual native CPU fixtures check prompt-only raw-state extraction with adapters disabled; answer isolation; valid-state padding/batching agreement; causal logits; whole-final-vector EX/IDX invariance; explicit original/new precision boundaries; strict shapes and active/frozen parameter contracts; and temporary hook/config cleanup after prompt/decoder exceptions with retry. K4 gradients and actual SGD changes reach the new workspace/readers while original weights remain byte-identical. Fully frozen target evaluation and frozen EX workspace in the K0 reader control are allowed; later drivers must enforce exact trainable groups.

Token checks establish separate prompt/gold encoding, no answer in X, exact assignment JSON plus EOS, length-based right-padding with EOS=PAD, no truncation, single loss shifting, first-answer/EOS supervision, independent hand-computed CE and ignored-position gradient behavior. A synthetic unique EQ-chain fixture is independently counted by the exhaustive oracle. A real random native model uses the collator/loss for backward and SGD. This does not establish retained-corpus membership or pretrained tokenizer behavior.

The isolated environment is Python3.11.9, torch2.11.0+cu128 and transformers4.57.1; full25-package inventory is retained in `environment.json`. Global Python/Torch and the stdlib-only environment were untouched. Native random CPU comparisons use prospective1e-6 tolerance because pinned native RMSNorm/RoPE/softmax contain FP32 internals; core1e-9 remains unchanged.

## Remaining prerequisites

Exact allowed-source export and identities; resolved run records; strict autoregressive per-example evaluation and costs; complete optimizer/scheduler/scaler/global+local RNG/sampler resume; pinned pretrained tokenizer/weights; and a separately registered full pretrained GPU forward/backward/optimizer memory probe. No GPU arithmetic, actual corpus model consumption, task training, protected model evaluation or G3/G4 PASS occurred here. Diagnostic A3 sample limitations and all earlier negative studies remain unchanged.

Later same-day preparation: pinned pretrained files and one real-tokenizer synthetic framing check now pass; see [artifact preparation](pretrained_artifact_preparation.md). No pretrained model tensors were loaded and no GPU/model execution follows from that preparation. All native CPU release evidence above retains its original scope.
