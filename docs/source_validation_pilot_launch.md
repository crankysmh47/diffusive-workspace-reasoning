# Active GPU source-validation pilot — 2026-10-19

The registered K0/K1/K4 pilot launched at17:05:42 UTC from clean source `a708fb399eca01c42e12bcc9f4bc80f7695dbfde`. Actual GPU forward passes and persisted initial-validation predictions were confirmed at17:07:37 UTC. K0 is first; K1 and K4 follow sequentially. This is an active exploratory run, not a completed learning or transfer result.

The active [v4 configuration](../configs/experiments/source_validation_pilot_tokenizer_repair_v4.json) has SHA256 `297fbd72496dd2706b85d6ad03b7cf3e265cd35aa46769af41e28581de191570`. It retains512 optimizer updates per arm,45 fixed validation examples, the original seeds, decoder, model, memory caps and scientific gates. The [startup-repair record](source_validation_pilot_startup_retry.md) preserves every earlier failure and explains the unchanged scientific settings. V1 failed memory preflight, v2 failed registration routing before data access, and v3 failed a tokenizer metadata check after model loading but before predictions or optimization. No failed run was relabeled or reused.

The actual pinned Qwen2.5-1.5B is running on the RTX5060Ti. V4 hardware preflight recorded14814MiB free, driver616.92, Torch2.11.0+cu128, Transformers4.57.1 and NumPy2.4.6. Original weights remain frozen by policy; the terminal comparison of all338 original parameter hashes is still required. Current utilization or allocated memory is not a terminal cost/memory result.

## Status and recovery

The independent supervisor, worker, watchdog, per-update checkpointing and bounded interruption recovery are implemented. Component continuation tests passed, but complete end-to-end hard-loss acceptance remains incomplete. The owner explicitly directed GPU execution without further review cycles; this limitation is retained in the [implementation evidence](source_validation_pilot_implementation.md). Known numerical, configuration, CUDA, memory or integrity faults stop rather than silently retry.

The producer checkout is `C:/sem5/diffusive-workspace-reasoning/.worktrees/transfer-cpu-v1`. Keep it at the source commit above until the job is terminal; do not pull, merge or edit its tracked files while it runs. Documentation and PR publication use a separate checkout. Run these commands from the producer:

```powershell
.venv-ml\Scripts\python.exe -m dwr.experiments.source_pilot_job status --config configs/experiments/source_validation_pilot_tokenizer_repair_v4.json
.venv-ml\Scripts\python.exe -m dwr.experiments.source_pilot_job start --config configs/experiments/source_validation_pilot_tokenizer_repair_v4.json
```

`status` reports artifact integrity and persisted progress without new model predictions. `start` attaches to an existing live job or attempts only eligible interruption recovery under its original deadlines; it is not permission for a second scientific run. The matrix has a16200s whole-job limit, with5400s per arm, a180s seal reserve and at most two verified interruption recoveries per arm. Local checkpointing does not protect against disk failure.

Artifacts remain local under `runs/source-validation-pilot-20261019-tokenizer-repair-v4/`; supervisor stdout/stderr are in `runs/source-pilot-supervisor.*.log`. Data, weights, checkpoints, raw outputs and these logs are ignored by Git. Publishing source is not an artifact backup or independent reproduction.

## What remains

- Finish K0/K1/K4 and their fixed initial/final evaluations; preserve failures and every per-example outcome.
- Finish K4's zero/noise/shuffled-workspace diagnostics and verify terminal records, paired exposure, full costs and original-weight hashes.
- Apply the registered learning and workspace-use gates. Either scientific NO_GO blocks automatic transfer training; a one-seed PASS is still exploratory.
- Complete the outstanding native recovery acceptance, artifact backup and fresh-environment reproduction.
- Only after an interpretable pilot, register the fixed-N EX-versus-IDX transfer study with adequate family/seed precision and matched controls. Variable N, adaptive allocation and diffusion remain deferred.
