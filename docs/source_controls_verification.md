# Source-control implementation verification

Root release check, 2026-10-17: all original scoped findings are closed in local `fix2-review.md`. Fresh command `.venv-ml/Scripts/python.exe .superpowers/sdd/source-controls-20261017/verify_stopped_controls.py` passed13 tests with zero errors/failures/skips in12.577 seconds including imports. All eight checked runtime/test/foundation files stayed byte-identical, and CUDA remained uninitialized. Receipt `root-focused-20261017T182731511375Z.json` and raw log live in that ignored directory; log SHA256 `38b587e683367ad2f26f3922768c78f997c6eea368c3125923b019317db30ee0`. This closes the trained-control implementation milestone, not the pretrained pilot or G4. No unchanged integration suite was repeated after the final head-consistency-only correction.

2026-10-17. Task 1 implementation evidence following the scoped root review. This records CPU implementation checks only; it is not a training result, pretrained-model result, GPU feasibility result, or pilot authorization.

## Implemented behavior

dwr.training.controls provides a frozen SourceControl description, strict configuration and validation, and one caller-owned update primitive for K0, EX K1, and EX K4. Every explicit horizon, including None, is type/range checked before any policy flag changes. The policy description records the checkpoint architecture metadata and named tensor shapes/dtypes as structural metadata, not a model-weight identity. Validation rejects same-count prompt placement drift, workspace slot-count drift, and precision drift.

The step validates exact native EX/max_steps=4 architecture, frozen/eval original weights, active flags, zero-gradient entry, exact named AdamW module groups, and an owned LambdaLR before backward or optimizer mutation. It checks finite loss and active gradients, rejects missing active/state/reader gradients, clips active gradients, performs AdamW then LambdaLR, checks updated learned parameters and resulting learning rates, and clears every gradient at successful exit. K0 has a reader-only optimizer and preserves the workspace. It returns finite Z0..ZK norms on every update; optional state and per-reader gradient diagnostics must be present, finite, and positive before the optimizer update. Joint parameter-gradient norms are streamed per parameter to avoid a concatenation-sized temporary.

The tests use independently initialized, tiny random native Qwen2 hosts in CPU float64, dropout zero, and synthetic verified source examples. No pretrained tensors, retained task payloads, dataset artifacts, or GPU were accessed. K0/K1/K4 initialization, K0 updater/original immutability, K1/K4 learned updates, optimizer membership, frozen stray gradients, active nonfinite gradients, invalid finite checks, and terminal post-update nonfinite behavior are covered.

K0 and K1 each compare three uninterrupted updates against one update plus a fresh-interpreter restore and two resumed updates through the complete checkpoint API. Both interpreters pin one CPU thread. Comparisons cover all named optimizer groups and state (nonempty AdamW moments and exact step counters), full sampler state and seven future IDs, complete NumPy/Python/Torch RNG state, private noise-generator state, scheduler, and progress. Learned tensors are compared at atol=rtol=1e-12 and the test reports measured maximum absolute error and fixture parameter counts.

## Verification

Run from C:\sem5\diffusive-workspace-reasoning\.worktrees\transfer-cpu-v1:

| Command | Outcome |
|---|---|
| .venv-task\Scripts\python.exe -m unittest discover -s tests -p test_source_controls.py -v | Exit 0; 12 optional native-ML tests skipped because this environment lacks the dependencies. |
| .venv-ml\Scripts\python.exe -m unittest discover -s tests -p test_source_controls.py -v | Exit 0; 12 tests passed. |
| .venv-ml\Scripts\python.exe -m unittest discover -s tests -p test_qwen2_workspace.py -v | Exit 0; 17 tests passed. |
| .venv-ml\Scripts\python.exe -m unittest discover -s tests -p test_training_resume.py -v | Exit 0; 11 tests passed. |
| .venv-ml\Scripts\python.exe -m unittest discover -s tests -p test_native_generation.py -v | Exit 0; 27 tests passed. |

Fresh-process continuation output:

| Control | Maximum absolute error: workspace | Maximum absolute error: readers | Tiny total parameters | Workspace | Readers | Active |
|---|---:|---:|---:|---:|---:|---:|
| K0 | 0.0 | 0.0 | 8,355 | 1,224 | 1,707 | 1,707 |
| K1 | 0.0 | 0.0 | 8,355 | 1,224 | 1,707 | 2,931 |

The initial pre-implementation run was only an API/import failure: .venv-ml could not import the not-yet-created dwr.training.controls module, while .venv-task skipped. No native update or policy behavior was exercised in that run. The original pre-implementation behavioral-RED requirement was missed; this chronology is retained accurately and is not represented as a behavioral RED.

Round-one review regressions were run before their corresponding corrections. Natural REDs: steps=None was accepted without error; same-count prompt-layer drift was accepted; dtype drift was accepted; ordinary K0/K1/K4 updates returned zero state norms; forced zero state and per-reader gradients did not reject; and same-count workspace slot drift was accepted under an intentionally omitted signature field. The K0/K1 continuation assertion was fault-injected with finite zeroed Adam exp_avg moments; the new complete optimizer-state comparison failed for both controls. The named optimizer preflight was fault-injected off; missing/duplicate membership tests detected parameter/gradient boundary violations. The post-update learned-parameter check was fault-injected off; its regression failed because no error was raised. These temporary injections were removed before final verification.

Other preserved implementation-stage failures: the first synthetic example used an invalid relation constructor and was replaced with the existing validated fixture; an initial assertion incorrectly expected EX Z0 to differ from supplied noise and was corrected; one dirty-gradient regex did not match the actual error text; a state-dict comparison initially used raw dict equality on tensors and was replaced with recursive exact/tolerance-aware comparison. An attempted active-gradient-check bypass was still caught by the streamed finite gradient norm before optimizer mutation. All final commands above pass or skip as expected; no unresolved failure remains.

Final SHA-256 of implementation/test inputs:

| File | SHA-256 |
|---|---|
| dwr/training/controls.py | 8126cd797a69a06f6771605534627b08d4f5538cd131fe11ddbd4e372b75bd6f |
| tests/test_source_controls.py | cebc12e476f18c0d80a89a1ee494ee3f342a6034d099d83cee9ade009396c0e4 |
| tests/source_control_resume_child.py | 9e7256576ba1d863e89c64983a9ec8bb5b0842f60659feb6cbb6a6245b5d324c |

The retained tiny-source overfit was not rerun or changed. No runner, config, existing test, README, todo, pilot registration, pretrained loader, checkpoint artifact, or experiment artifact was changed. Task 1 is implemented with all five scoped findings addressed and awaits root's focused rereview. No commit or push was made.

## Round-two focused residual

Root's second scoped review found that mutating only `reader.cross_attention.num_heads` left the declaration and tensor signature unchanged. Before the correction, the new native regression configured EX K1, changed actual heads from 3 to 1 (still a valid divisor of hidden size 12), then called the update primitive. The focused 13-test run failed only this case because no `ValueError` was raised; the other 12 cases passed, including K0/K1 continuation at exact zero error. This is the genuine behavioral RED for this residual.

The runtime guard now rejects any reader whose actual cross-attention head count differs from its declared count before policy configuration or step execution. The actual head-count tuple is also included in the immutable policy signature. The regression asserts rejection before parameters, optimizer state, scheduler state, or gradients change.

Final round-two focused commands:

| Command | Outcome |
|---|---|
| `.venv-task\Scripts\python.exe -m unittest discover -s tests -p test_source_controls.py -v` | Exit 0; 13 optional native-ML tests skipped because dependencies are unavailable in this environment. |
| `.venv-ml\Scripts\python.exe -m unittest discover -s tests -p test_source_controls.py -v` | Exit 0; 13 passed. K0/K1 continuation max absolute errors remained `[0.0, 0.0]`. |

Round-two hashes:

| File | SHA-256 |
|---|---|
| `dwr/training/controls.py` | `4cf542f09af4bef2e29e036d5c12f42b80d88cd1bcf5ccd3891b9f44332d26ea` |
| `tests/test_source_controls.py` | `7525f2331308104bb3b58015e104822e1921358ce18fcd6e74546fe7f4983dba` |
| `tests/source_control_resume_child.py` | `9e7256576ba1d863e89c64983a9ec8bb5b0842f60659feb6cbb6a6245b5d324c` |

No integration suites were repeated because round two only changes source-control architecture validation and its focused regression. No training, pretrained load, GPU, pilot execution, commit, or push occurred.
