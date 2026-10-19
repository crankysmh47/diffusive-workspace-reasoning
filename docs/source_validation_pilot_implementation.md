# Source-validation pilot implementation evidence

Actual execution update, 2026-10-19: the [registered GPU pilot is active](source_validation_pilot_launch.md) at clean `a708fb3`, with real GPU forward passes/predictions confirmed. The startup fixes, targeted checks and preserved failures are documented [separately](source_validation_pilot_startup_retry.md). This launch does not close the remaining full native recovery-acceptance gate or establish a scientific pilot result.

2026-10-19 launch decision: the owner explicitly directed skipping further review cycles and starting the registered GPU pilot as soon as possible, then updating and merging PR2. Registration, evaluation, native learning and checkpoint continuation have the CPU evidence below. The production worker/supervisor are implemented; complete native matrix recovery acceptance remains partial. No completed GPU pilot or scientific outcome is claimed here. Scientific settings remain the registered 512 updates, 45 validation examples and K0/K1/K4 comparisons. This does not reopen the completed tiny-source overfit.

The expedited launch deliberately proceeds without another full integration/review cycle. Root's task suite passed850 collected tests (153 optional skips); the remaining native learning/recovery/registration/resume suite passed26 tests with no skips in471.437 wall seconds on stable bytes. A subsequent affected scope passed48 task tests (six skips) and47 ML tests (zero skips). Later native integration completed K0 after actual supervisor loss and same-worker attachment, then exposed concrete hard-loss finalization defects. Those fixes and focused regressions are retained; the last `_segment_artifacts` import correction passes compilation and Ruff F821, but its complete hard-loss/native rerun has not passed. Automatic recovery is therefore implemented but not fully validated. All failed fixture matrices remain preserved under `.superpowers/sdd/source-validation-pilot-20261017/` and are not relabeled successful.

This owner-directed operational departure changes no data, model, optimizer, decoding, evaluation, numerical gate, source-integrity requirement, memory limit or retry/compute allowance. A fault in the actual pilot must stop and preserve its evidence; this decision grants no scientific retry. Full loss-path acceptance remains an explicit follow-up, alongside the pilot's measured learning and workspace-use results.

## Closed prerequisites

The [trained controls](source_controls_verification.md) passed independent review and 13 root-executed native CPU checks. Task1 registration/readers/fixed-workspace evaluation passed independent fix review and root checks: 14 ML methods executed with zero skips; the task environment executed six methods and skipped eight optional ML methods. The executable config remains SHA256 `125ebe2ee9672b58f07f6511f29be28b5eef04382ee8a71cc7a9951e30503a31`, and the fixed cohort remains `615e357ceff2a7b0a32e9509d68d1be2e881a233f6387088746b9f58bf32afe7`. Dependency and cohort selection checks did not open retained task semantics or load pretrained tensors.

## Native kernel and recovery gate

The implementation uses the actual tiny Qwen2 host, checked control update, answer loss, optimizer/scheduler, sampler, private noise, checkpoint APIs, full-prefix greedy decoder and semantic task verifier. CPU fixtures scale dimensions/counts explicitly; they do not certify real-model learning or GPU costs.

Independent review found seven original defects covering retirement, intervention binding, immutable prior evidence, partial/last-stage durability, completion audits, evaluation preservation and initialization identity. Two scoped fix rounds closed them and their concrete residuals, including rejection of valid but uncommitted checkpoint collisions. No scientific settings, cohort, noise policy, training allowance or numerical advancement gate changed. All negative test attempts remain local.

Final kernel-focused ML commands:

```powershell
.venv-ml\Scripts\python.exe -m unittest discover -s tests -p test_source_pilot_learning.py -v
.venv-ml\Scripts\python.exe -m unittest discover -s tests -p test_source_pilot_recovery.py -v
```

Results: learning 6/6 passed in98.565s; recovery 7/7 passed in0.249s. Their four-file frozen/post inventories match. Log SHA256 values are `106e905d0e599464e847ecfe7c64433c85bcb1b37ebbbfecd4a0f407f069922e` and `e09a8f76b05d93c80cd09fea2857838720ce10f575aa81d91cebe59a03b63bda`. These are implementer-executed checks; root inspected the recorded bytes, logs and hashes without relabeling that inspection as an independent rerun.

## Actual process proof and retained failure

A detached four-method CPU suite executed real fresh child interpreters and actual process loss via `os._exit(91)` after durable events. The first complete run took363.530 wall seconds (352.450 unittest seconds). Partial-zero persistence/full-stage redo, K4 committed bank/final/shuffle reuse and native six-update K0 retention/orphan preservation passed. Its K0/K1/K4 continuation subcases matched complete checkpoint payloads, then failed the parent's additional next-draw restore because the harness omitted the registered AdamW flags. The unchanged checkpoint loader correctly rejected `foreach=None` versus required `False`.

Only the parent test helper changed: it now reconstructs every AdamW option from the existing fixture registration. Independent scoped review accepted that correction. Read-only replay of the preserved actual checkpoint pairs matched future sampler IDs and Python/NumPy/TorchCPU/private-noise draws. The affected method then passed a new detached fresh-process run for all three arms:

```powershell
.venv-ml\Scripts\python.exe -m unittest tests.test_source_pilot_resume.SourcePilotFreshProcessResumeTests.test_k0_k1_k4_fresh_process_continuation_matches_complete_logical_checkpoint -v
```

Result: one method/three arm subcases passed in175.501 unittest seconds,178.732 wall seconds, exit0; raw stderr SHA256 `faf3cde6df947ae12c3a8150123dda5d9a8197c463853ba6f9733c578e7bb107`. Full checkpoint trees, nonempty Adam moments, exact discrete state, numeric tolerance1e-12 and future draw assertions remain intact. The other three methods were unchanged; root's AST comparison confirmed `_fresh_draws` was the only changed definition. This is successful evidence across scoped checks, **not a single-invocation full-suite PASS**. The later integrated release must run the current complete suite.

The task environment collected all four methods and skipped each for genuinely absent direct ML dependencies; exit0,0.074 wall seconds. The runner's own CUDA initialization probes were false before/after. The original helper used a1200s internal timeout despite the requested900s CPU verification cap; its actual363.530s run remained below900s, and the corrected affected-method runner used900s. This verification-only deviation changed no scientific budget.

## Source identities and local evidence

- `dwr/training/source_pilot.py`: `d769b8d80f30f6c80ffd30c55f801dd0e4fd316565660141173a7a0906f67b80`.
- `dwr/training/pilot_recovery.py`: `9be94caac508a6cdea365f240ae0a5f48b25f91422ee1ba62868c302d91989ba`.
- `tests/test_source_pilot_learning.py`: `9d5d12e730e22f73d72cae4d15288c0839e88cb7e4d08ecda66cd1d91714c85b`.
- `tests/test_source_pilot_recovery.py`: `72b0f6154458278d1ab3185afa0e3e99661b2866e9f93f7aa09dc05f59aef4b1`.
- `tests/source_pilot_resume_child.py`: `7a597e33edde060e10c0f6fac2a3581d9c3131fef4808d578758937f8a18b762`.
- `tests/test_source_pilot_resume.py`: `aaaef782415923d87b064ff55c6db18de141e22c3cdca6c8332c60b7569ab2bc`.

Local ignored evidence lives under `.superpowers/sdd/source-validation-pilot-20261017/`: `task2-fix2-report.md`, `task2-fix2-review.md`, `task2-probe-review.md`, `task2-process-report.md`, `root-task2-process-evidence-audit.json`, and both detached process run directories. The root audit checked receipt/log hashes, unchanged source bytes and the narrow AST delta. Initial incomplete/harness-failed runs and all child artifacts remain preserved. They are local agent-run evidence, not a Git backup or independent-person reproduction.

## Remaining production gate

The controller's two scoped fix reviews closed the identified memory-device, cross-arm pairing, checkpoint-relocation, same-producer audit, pinned metadata, seal-budget and CE-domain defects. The final correction review passed its static scope; its 23 focused tests passed in2.906s. This remains separate from native supervisor acceptance. Earlier missing raw-log/RED evidence was disclosed, and subsequent fix logs were retained.

Root separately checked the actual closed registration and pinned model/tokenizer metadata in1.516s with stable source/config bytes. The check resolved the inherited device and exact production512/45/338/128 policy; an audit hook prohibited pretrained tensor-file opens and CUDA stayed uninitialized. Receipt: `.superpowers/sdd/source-validation-pilot-20261017/root-production-metadata-20261019T051807077435Z.json`; raw log SHA256 `ffd2bacdb049bba1eeb9a91399b87d6d71ec1eca57a09ff98335e882d14009f6`. This is metadata-wrapper evidence, not model validation or learning.

The first detached tiny-native worker reached a real K0 update, then failed strict event serialization on tuple-valued norm observations. The job stopped rather than retrying the known fault. The job-owned serialization adapter now has an actual `RunRecord` regression; the closed Task2 kernel is unchanged. All failed setup/native attempts remain preserved. A second integration issue was exposed when stderr changed after the worker sealed its record; the launcher review and correction remain pending. No tiny-native matrix or actual pilot GPU result is claimed.

Task3 must implement/review actual pinned host/data loading, source/runtime and same-live-host frozen audits, live-tokenizer re-decode/EOS/vocabulary checks, physical-work ledgers, original clocks/caps, sequential arm records, actual Windows process ownership, watchdogs and detached recovery. Full CPU release and physical/index/commit byte equality precede the sole registered GPU launch. G4, generalization, transfer, workspace specificity and paper novelty remain unestablished.
