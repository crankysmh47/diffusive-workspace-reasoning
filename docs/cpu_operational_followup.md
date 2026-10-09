# Prospective CPU operational follow-up

Date: 2026-10-10. Status: registered execution contract; implementation/review/equivalence gates must pass before the sole invocation. This is a new study, not a resumed or reclassified v4 run. The owner's instruction authorizes the bounded diagnosis, operational amendment and continuation until a genuine gate. The controller owns this decision.

## Rationale and affected comparisons

The v4 study filled its source quota but exhausted 900 seconds before source verification was assessed. Fixed source-fixture profiling identifies exhaustive JSON canonicalization as a local CPU hotspot. The selected optimization preserves the exact canonical strings while enumerating all permutations. No optimization speedup or original-run phase attribution is inferred before verification.

Register one fresh source/development invocation with a combined 1,800-second wall cap, one worker, 1-GiB peak resident cap, and the existing 30,000 candidates per partition. The larger engineering cap provides finite headroom for required generation, source replay/audit, terminal replay/audit and finalization. It is a resource feasibility decision, not an estimated production completion time. Neither a passing floor nor a favorable target result determined this cap. There is no retry, renewed budget, resume, seed search or allocation to a further proposal in this study.

The new config is `configs/tasks/constraint_probe_cpu_followup.json`, identity `dwr-constraint-development-probe-cpu-followup-v1`. Relative to the original immutable v4 config, change exactly `config_version` and `runtime.combined_active_wall_seconds_generation_replay_audit_finalization` (900 -> 1800). Retain the generator identity/domain, root seed, candidate draw order, grammar, cells, quota/profile ordering, source/development hash splits, primitive necessity/connectivity tests, all denominators, original numeric floors, outcome/record/artifact schemas, producer path order, max attempts, one worker, memory cap, final-task prohibition and absence of preflight. The retained v4 artifact version strings identify reused schemas, not the new study identity.

All future EX/IDX and controls must use the same eventual frozen corpus, so this pre-model operational change affects no already-trained comparison. Historical v1-v4 results/source/config/artifacts remain immutable and independently identified. The v2.2 model contract and research gates are unchanged. No protected source-test or development-diagnostic statistics may enter decisions; final composition tasks are never sampled by this proposal.

## Release conditions and minimal implementation

Before production execution, require exact canonicalization equivalence tests, a full nonempty CPU suite, post-change fixed-fixture equality of identities/outcomes/records/denominators/audit projections, independent code review, and a clean committed source. Preserve the imperfect original profiling baseline and qualify its retrospective provenance/reporting gaps; do not rerun it. Failure-injection correctness checks for the profiling helper are separate from performance measurements.

Reuse the existing driver and CLI rather than duplicate scientific logic. The closed config loader may admit only two exact SHA-256 config identities: historical v4 and this follow-up. Every other byte change, unknown key, duplicate key or nonfinite constant remains rejected. Derive manifest probe identity and resource wall limit from the admitted config; historical v4 remains exactly its original identity and 900-second policy. Use the same configured wall bound in default generation/controller/audit guards, initial-manifest readers, terminal PASS resource checks and emergency failure manifests. No resource override CLI is exposed. Existing exact schema/hash/status/replay/audit checks remain required.

The exact CLI is:

```powershell
& 'C:\Users\Hank47\AppData\Local\Programs\Python\Python311\python.exe' -m dwr.tasks.probe_cli_v4 --config configs/tasks/constraint_probe_cpu_followup.json --mode development-probe --out data/constraint-probe-cpu-followup-<unique-UTC>
```

Write the resolved manifest before drawing the first candidate. Bind the new config hash, clean source and complete ordered producer hashes. Save all candidate outcomes and actual wall/CPU/RSS costs in a unique ignored directory. Terminal resource accounting includes replay, independent audit and post-write checks. Retain partial artifacts and typed failure causes. Report only allowed source-train and development-validation projections.

## Decision rules and continuation

Use the original source closure rules before any development candidate; use the original development floors and four-cohort proxy after full replay and independent audit. A required scientific floor/support failure or quota shortfall at exactly its attempt cap is `NO_GO_PROPOSAL`. Resource, I/O, solver/oracle disagreement, replay or audit failure is `NOT_ASSESSED` with its original typed cause. Do not call an operational failure a scientific negative.

Only a complete `PASS_DEVELOPMENT_PROPOSAL` permits the separately reviewed full-corpus schema/consumer/linkage task and one registered full G2 invocation. Probe PASS does not pass G2. A complete passing actual five-cohort G2 audit remains required before model implementation/backbone selection; training additionally needs its own non-null registration and memory/resume/gradient gates. At any failed scientific gate or exhausted allowance, preserve evidence and report the concrete blocker.
