# Permitted-source export result

2026-10-15. The sole registered source-only conversion completed at clean source `6f0ea98c83356b7ebb53f887a394726a81f9edb6`, with source unchanged before and after. The command was:

```powershell
.venv-task/Scripts/python.exe -m dwr.experiments.source_export --config configs/tasks/source_export_v1.json
```

The reviewed converter exported all 2,436 source-training and 265 source-validation records. Independent reference solving, exhaustive tiny-instance oracle checks, strict verification and identity checks passed for every exported task. It semantically parsed zero protected source-test, development or final task payloads. Full-file opaque hashing and strict raw JSON parsing materialize fields before routing; the access claim concerns task semantics, not absence of raw file reads. No model or GPU operation occurred.

The run consumed 49.814 seconds internally (46.719 CPU seconds), with 51,703,808 bytes peak RSS. The external wrapper completed in 49.984 seconds, exit 0, without reaching its 305-second watchdog; the registered internal caps were 300 seconds and 512 MiB. Separate artifact-write costs and FLOPs were unmeasured. The [compact receipt](../configs/splits/source-export-v1-result.json) binds the exact configuration, six upstream artifacts, run manifest, output and summary hashes. The ignored run directory is `runs/source-export-20261015-v1`; its manifest SHA-256 is `7db0b61ba40bea60bf17bc5faf615bb2c8ae21320739c3bea589e4d62aea0069`.

The subsequent [complete token-cap audit](source_token_cap_audit.md) passed all 2,701 exported records without truncation or skipping. Both splits have maximum lengths of 407 prompt tokens, 34 answer tokens including EOS and 441 total tokens. The prospective GPU limits remain 1,024 prompt, 128 answer and 1,152 total tokens. This validates source framing and input caps; it establishes no neural learning or transfer effect.

The [pretrained GPU feasibility design](gpu_feasibility_design.md), closed configuration and implementation still require their CPU release and clean source checkpoint before the sole hardware invocation. Local ignored artifacts are not archival backup, and no independent-person reproduction is claimed.
