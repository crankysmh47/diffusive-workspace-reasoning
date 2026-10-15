# Pretrained GPU probe: implementation verification

2026-10-15. Luna implemented the closed [probe](gpu_feasibility_design.md), its real tiny-model CPU checks and a narrow optional run-record budget callback. Independent task and release review found two blockers before any hardware invocation: the exact registered synthetic settings failed the input factory, and sampled RAM/terminal artifact work could escape the budget gate. Both were corrected with failing regressions, along with sampler restoration sensitivity and actual numeric/SDPA policy readback. Independent fix review is **READY_FOR_ROOT_CPU_RELEASE**. The config and sole-run allowance are unchanged.

Root then ran the complete separate CPU release on stopped, stable runtime/config/test bytes:

```powershell
.venv-task/Scripts/python.exe -m unittest discover -s tests -v
```

This reported 657 methods: 598 passed and 59 skipped, in 58.842 seconds. One skip is the existing Windows directory-symlink privilege fixture; the remaining skips reflect absent optional ML dependencies in this environment.

The `.venv-ml` process ran discovery for `test_workspace*.py`, `test_qwen2_workspace.py`, `test_training_batch_loss.py`, `test_native_generation.py`, `test_training_resume.py` and `test_gpu_feasibility.py`: 88 passed, zero skipped, in 9.079 seconds. With 21 methods shared between processes, **665 distinct methods executed**. Combined wrapper time was 71.891 seconds under one 180-second watchdog, with all commands exiting zero and all tracked runtime/config/test inputs unchanged.

The ignored root receipt is `.superpowers/sdd/gpu-feasibility-20261015/controller-separated-suites-20261015T182045872415Z.json`; method accounting is `final-cpu-method-accounting.json`. Task/ML log SHA-256 hashes are respectively `a31770c343a9b71cd3427ce62c7619a42398a76d91d94dd40fd1bc36bb592d47` and `47286d7de50d402a7967a2c33279d4869195050298b52904d446350b4b9e50d5`. Prior 660-method success on the defective pre-review code remains preserved; it never authorized a GPU launch. A first log-accounting script assumed status text stayed on the method line and failed; counting unique method starts and explicit skips matched both successful unittest totals and the stated overlap.

Reviewed and tested source SHA-256 hashes:

- `dwr/experiments/gpu_feasibility.py`: `7af8e916d4544e78b2c56cb4b13a7c17d0c3d63e4ad0c9319aa199b825f58c1a`
- `tests/test_gpu_feasibility.py`: `231ca630cddd66bf425c07d1917a20b00737147cbc9dc4175346fa4b22eb8698`
- `dwr/experiments/records.py`: `1a2f98e92e5062358241ac4a2e9aae50520bba79ff9763d6e2b0950f99b72a06`
- `tests/test_experiment_records.py`: `13a0e0895dc2d54ae0bc1d7821a23d0d7a075d3ffedb2dbfeb6ca755b59307c7`

Root's scope decision added only `records.py` and its focused tests to the original two-file task so the existing lifecycle could enforce pre-seal budgets. Existing callers retain the default behavior. The risk was a shared run-record regression; focused lifecycle/integrity checks and the full release passed. No general trainer or fallback was introduced.

The original implementation RED was a module-presence check, not behavioral TDD. Review-driven fixes have genuine failing regressions; their logs, snapshots, earlier failures and independent reviews remain under the ignored task folder. Actual tiny CPU Qwen2 computation is distinct from mocked controller/policy coverage. A Windows process high-water API smoke was also run, without GPU computation.

This verifies implementation and CPU behavior. A clean source checkpoint, exact artifact/environment checks and at least 14,500 MiB free on the registered GPU remain prerequisites for the sole 900-second probe with a 905-second watchdog. No pretrained GPU math, task training, GPU replay trajectory, model quality, transfer effect or G3/G4 PASS is established. Local ignored artifacts are not archival backup or independent-person reproduction.

Launch availability observation, 2026-10-15: the registered GPU UUID/name/driver still match, but a read-only NVIDIA query reported13,745MiB free against the unchanged14,500MiB minimum. Free memory is volatile. The probe has not been invoked and its single attempt remains unused. Finish the clean-source handoff and recheck this prerequisite before launch; do not terminate unrelated applications or silently lower the gate.

Subsequent actual startup: owner-authorized cleanup cleared the memory gate, but v1 failed at RunRecord.create before model/CUDA operations. Preserve its [failure receipt](../configs/models/pretrained_gpu_feasibility_v1_startup_failure.json); the allowance is closed. The separately registered [startup repair](gpu_feasibility_startup_repair.md) now passes independent review and668 distinct root CPU methods, including the real controller/run-record boundary previously missed. No original CPU evidence is rewritten as model validation.
