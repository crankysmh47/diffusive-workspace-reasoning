# Native Qwen2 host and token/loss implementation plan

2026-10-14. Root decisions; Luna implementation with independent task/release review. Specification: [native_backbone_design.md](../../native_backbone_design.md). Structural G2 and the fixed-N tensor core pass; no training is authorized by this implementation plan. Owner delegated routine design and requests minimal commits.

## Order and scope ruling

Implement the native graph/token/loss contract before the general G3 run/evaluation/resume consumer so those interfaces and precision boundaries are real rather than speculative. This deliberately reorders the G3 helper bullets without removing any prerequisite. Affected comparisons: future frozen/K0/EX/IDX all use the same native host/framing/layout. Evidence: tensor core verification and pinned Qwen2 metadata/source; no model experiment has run. Full manifests, exact corpus consumption, evaluation/checkpoint/resume and registered GPU feasibility still precede task training. No raw corpus, public dataset, pretrained weight download, GPU math or scientific run in this component scope.

Root creates an ignored `native-host-20261014` evidence directory. Preserve actual RED/GREEN, failing intermediates, snapshots and hashes. `.venv-ml` remains isolated; root alone installs pinned optional Transformers4.57.1 and captures the exact updated inventory. Stdlib-only task environment stays untouched. Workers never install dependencies, commit, spawn agents or run the full suite. Genuine missing optional dependencies skip explicitly; missing transitive dependencies/source errors fail. No model import is added eagerly to `dwr.models`.

## Task1: actual native Qwen2 host

Own new `dwr/models/qwen2.py`, new `tests/test_qwen2_workspace.py`, and optional extra `backbone=['torch==2.11.0','transformers==4.57.1']` in pyproject. No workspace/reader/task code or status docs. Root is the only environment writer.

- [x] Read binding protocol, native spec and actual pinned Qwen2 source. Declare precise APIs/return dataclass and CPU test tolerance before implementation.
- [x] Actual TDD RED, then real Qwen2ForCausalLM tiny random-config CPU tests, no mocks or weights. Tiny config: vocab32,hidden12,intermediate24,layers3,queryheads3,KVheads1,attention_dropout0,tiedembeddings; pad/eos31. Workspace context12,N4,d8,heads2,K4; prompt_layer1; readers after layers0,1,2,readerheads3. These are fixture dimensions, not scientific backbone settings.
- [x] Implement Qwen2WorkspaceLM and typed output. Constructor freezes originals/eval, enforces actual Qwen2 and compatible layer/width/device/dtype contracts, registers new readers and honors separately matched reader initialization. No global workspace cache.
- [x] Separate no-grad prompt-only pass/adapters disabled; host-owned raw-output hook on layer(completed_layer-1), original backbone.model with cache/hidden-state/attention recording disabled, finally removal. Explicit prefix agreement. Strict right-padding Boolean mask/torch.long ID/shape/vocabulary/device and unknown arguments. Answer changes cannot affect X.
- [x] Temporary per-call reader hooks with `finally` cleanup; explicit original/new dtype conversions, autocast off inside new modules, graph through frozen layers. No KV cache/gradient checkpointing/foreign hooks/reentrant calls or fallback output guessing.
- [x] Meaningful native tests: prompt X isolation, adapter changes cannot affect X, raw selected-layer extraction including depth endpoint, actual future-token causal logits, padded-vs-unpadded and batched/single valid states/logits, whole-final-vector EX/IDX logit invariance, invalid contracts and forced prompt/decoder exception cleanup, allK4state/reader/workspace-group gradients and actual SGD changes with original weights unchanged. Tokenwise RMSNorm/RoPE and eager softmax use FP32 internally in pinned native code even for float64 outer tensors; preregister native CPU atol/rtol1e-6, without changing existing core1e-9 tolerance.
- [x] Scoped unittest only, evidence/API receipt/hash/LF/self-review, independent task review and fixes. No commit/full suite.

## Task2: strict token collation and answer-only loss

Own new `dwr/training/batches.py`, new `dwr/training/loss.py`, new `dwr/training/__init__.py`, new `tests/test_training_batch_loss.py`. No native/core/task code unless root ruling. Depends on cleared Task1 API.

- [x] TDD using an explicit tiny tokenizer fixture for serialization and the actual native model for gradient/loss integration. No pretrained tokenizer download in this task.
- [x] Exact existing task JSON plus newline prompt; separate answer JSON encoding plus terminal EOS; no chat/BOS or annotation/gold in prompt. Validate tokenizer integer IDs/EOS/pad, unique example identity, strict task/solution and no unknown payload keys. Caller provides pinned tokenizer identity and explicit length caps; overlength fails without skip/truncation.
- [x] Right-padded prompt and decoder IDs/Boolean masks; prompt/pad labels=-100, answer JSON/EOS supervised. Verify prefix equality, source examples retain their identity, inputs not mutated. No protected-corpus load by fixture collator.
- [x] Shift loss once, divide token-summed CE by actual supervised count, fail if none; check first-answer/EOS predictions and invariance to ignored-label logits. Reject nonfinite or malformed shapes/IDs/masks. No hidden fallback.
- [x] Actual native K4 forward/backward integration uses this collator/loss on synthetic tasks and frozen original model; verify prompt exclusion/finite new gradients. Scoped tests and independent task review, preserved actual logs/hashes. No training CLI/run or target evaluation.

## Task3: root release and remaining G3 infrastructure

- [x] Whole new-stage review; stopped writers; complete stdlib discovery plus actual ML suites in separate environments under combined180s cap. Preserve all failures and exact updated environment. No inference/training/GPU claim from native random CPU fixtures.
- [ ] Update todo/execution/design verification only for passed outputs. Consolidate the next substantial verified checkpoint; no per-task/status commits, no merge.
- [ ] Implement strict resolved manifests/corpus permission routing, autoregressive scoring/cost records and complete optimizer/scheduler/scaler/global+localRNG/sampler resume. All necessary before registered real-pretrained feasibility/pilots. Never launch a null registration.

Terminal of this plan is verified native CPU integration and strict token/loss contracts. Owner's broader authorization continues into remaining infrastructure; do not stop merely because this component plan ends. Scientific/CPU/GPU caps and genuine blockers retain precedence.
