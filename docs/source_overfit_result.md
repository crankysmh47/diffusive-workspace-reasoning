# Recoverable tiny-source overfit result

**Result: `PASS_TINY_SOURCE_OVERFIT`.** The single registered run learned the three selected source examples: official strict validity rose from 0/3 at initialization to 3/3 at step 192, and CE fell from `0.4202528993288676` to `0.000005453095809571096` (ratio `0.000012975748217988598`). This is evidence of tiny-batch memorization and successful end-to-end training, not task-family generalization, transfer, or a G4 result.

## Run identity and scope

- Run directory: `runs/source-overfit-20261017-recoverable-v2` (ignored local artifacts).
- Registered command: `.venv-ml\Scripts\python.exe -m dwr.experiments.source_overfit_job start --config configs/experiments/source_overfit_recoverable_v2.json`.
- Start/end: `2026-10-17T17:53:28.909561+00:00` to `2026-10-17T18:02:28.601790+00:00`; whole-job elapsed time was `539.7030000000086` seconds. The worker segment took `505.4539999999979` seconds.
- Terminal job and worker statuses are both `completed`; job outcome is `PASS_TINY_SOURCE_OVERFIT`. One worker, one segment, zero operational recoveries, and a single training seed were used. The run was registered as non-confirmatory.
- Physical source was clean at commit `745a74241a832946579b487f839acab383a455bc`; the launch receipt recorded 266 source files. Its source identity is SHA-256 `e3fbfb939a5e59639179bc48aea75c7eff20788047fde23050046f1a164dc740`.
- Config SHA-256: `e6c3de7e59e58c494b89052086c0840dcd3421d17396fd5a55cac40611049594`. Cohort SHA-256: `b7319af014508a13dea46c19169ff36800c5db17c6d22d7c28c6e72d44b46932`. The data/proof bundle identity is `10a1895bf8838c5870091b1ba0cd7dda20d1b6bb04871ee7fb3dca1a0f4a1dbc`.
- Model: `Qwen/Qwen2.5-1.5B`, revision `8faed761d45a263340a0528343f099c05c9a4323`, weight identity SHA-256 `a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796`. Tokenizer inventory identity is separately pinned to `4d5c001069a7c1e225b72a5723fc4a23e6ab316ec940454bbd42b02440ad2dea`.

## Learning, replay, and costs

The run completed 192 net optimizer steps and 193 physical optimizer updates (one extra update was used by the registered replay probe). It evaluated three examples at each of two checkpoints, for six predictions and 486 conservative decoder calls. There were no inflight updates, recoveries, or extra workers. The final gate required 3/3 valid and a final-to-initial CE ratio at most 0.2; both conditions passed.

The step-96 to step-97 serialized replay probe passed the registered elementwise rule `abs(actual-reference) <= 1e-6 + 1e-5*abs(reference)`. Discrete state, scheduler, sampler, and RNG comparisons were exact; the replayed example IDs matched. Maximum parameter absolute/relative errors were `3.725290298461914e-09` / `3.6540134304276826e-05`; optimizer absolute/relative maxima were `5.002220859751105e-12` / `17.678696087698754`. The large relative optimizer maximum occurs near zero and is not an independent tolerance failure: the registered additive elementwise comparison passed. Replay was not bitwise equality.

The controller counted 1,543,714,304 original parameter elements and 31,895,811 newly trained parameter elements (1,575,610,115 total). The frozen-base guard passed; all 338 before/after original parameter hashes matched. Peak GPU memory was 4,670.041 MiB allocated and 13,412 MiB reserved; process peak working set was 3,821.711 MiB. The run stayed within its registered limits of 197 physical updates, 18 predictions, 2,304 decoder calls, 14,336 MiB allocated/reserved, and 12,288 MiB resident memory.

## Evidence and limits

The [compact result manifest](../configs/models/source_overfit_recoverable_v2_result.json) records the principal receipt, checkpoint, evaluation, replay, and terminal hashes. The local run directory contains the source/job binding, segment `RunRecord`, controller audit, evaluation receipts and prediction records, physical-work ledger, complete recovery cursor, final checkpoint, and replay evidence. The root's local independent terminal audit, `.superpowers/sdd/source-overfit-recovery-20261017/actual-terminal-independent-audit.json`, SHA-256 `5f735f94bc626203d4f162fd260c85f4a61d191e29a69542198efa53f538ff72`, checked job and worker artifact links, the six recovery identities and complete phase, evaluation arithmetic and validity, all 338 frozen-parameter hashes, memory caps, and replay evidence. Both terminal records also pass `verify_artifacts`.

The run establishes a tiny source-overfit result only. G4, general task learning, transfer, novelty, and confirmatory precision remain unestablished. A separate pilot requires its own registration before execution. The run artifacts are ignored by Git and remain local; they are not an archival backup or an independent reproduction.
