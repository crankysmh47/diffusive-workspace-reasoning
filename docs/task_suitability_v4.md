# V4 development-probe suitability result

Date: 2026-10-10. Status: **operationally unassessed at the registered resource cap**. The sole v4 probe is closed. Its terminal state is neither scientific `NO_GO_PROPOSAL` nor `PASS_DEVELOPMENT_PROPOSAL`; it does not assess corpus suitability or pass G2.

## Frozen identities and execution

The probe used clean source commit `169740fb7c1c4f995526ba18c28a8c82797df6fe`, config `configs/tasks/constraint_probe_v4.json` SHA-256 `6f86fa7d91a0b469fbd9f657eedfe7de87a6934005054059fe246878ec52f104`, generator `dwr-generator-v4-probe`, config identity `dwr-constraint-development-probe-v4-r2`, the registered 17 producer files, and CPython 3.11.9 on 64-bit Windows 10 build 26200. The registered limits were one worker, 30,000 candidates per partition, a combined 900-second wall budget, and 1 GiB peak resident memory.

The exact command was:

```powershell
& 'C:\Users\Hank47\AppData\Local\Programs\Python\Python311\python.exe' -m dwr.tasks.probe_cli_v4 --config configs/tasks/constraint_probe_v4.json --mode development-probe --out data/constraint-probe-v4-20261009T192624974846Z
```

The process exited 1 with `TimeoutError`, reason `NOT_ASSESSED_RESOURCE_CAP`, during the post-terminal-write resource check. Its run interval was `2026-10-09T19:26:24.974846Z` to `2026-10-09T19:41:25.755624Z`. The authoritative accounting boundary reports 900.234 seconds wall, 897.34375 seconds CPU, and 285,614,080 peak resident bytes; the external wrapper reported 900.75 seconds wall.

## Observed state and gate interpretation

| Item | Observed result | Assessment |
| --- | ---: | --- |
| Source quota bookkeeping | 1,000 rows; 20,299 attempts; quota counts closed | Count recorded; source closure/suitability not assessed |
| Source verification | — | `NOT_ASSESSED` |
| Candidate replay | — | `NOT_ASSESSED` |
| Independent audit / integrity | — | `NOT_ASSESSED` |
| Development attempts | 0 | Not started |
| Final attempts / full run | 0 | Not attempted |
| Preflight / G2 | None / — | Not emitted / not assessed |
| Model or training | None | Not run |

The controller observation says root rehashed all eight artifacts and cross-bound small public accounting, attempt, and status metadata. That check did not inspect rows and did not perform semantic replay or an independent audit. The source quota count alone cannot establish source verification, floors, support, task suitability, or transfer potential. Empty development targets are unattempted data, not measured zeros or failures.

This result must remain `NOT_ASSESSED_RESOURCE_CAP`. It is not a scientific NO-GO, because the required checks did not finish; it is not a PASS, because source verification and the development proposal checks were not completed. The actual probe did not satisfy the conditional requirement for a full-run design task, and no full/G2/model stage follows from it. The proposal is closed without retry, budget extension, retuning, or a new v5 allocation. A future experimental path requires a new bounded prospective design decision.

## Implementation checks and evidence limits

At the frozen source commit, the controller's fresh CPU suite passed 275 tests in 46.451 seconds, exit 0; log SHA-256 is `967d8a9bf9f327bbdf08738c79bf626ff0ced335207f9091af1b658d5dda130f`. The scoped independent final rereview was READY, SHA-256 `c913ab95b69b7287649b0748787bb368eaca5880e6907d54620f76deb100b31d`. These are agent-run implementation and review evidence. They are not corpus validation, semantic audit, model evidence, or independent reproduction.

The complete root observation is `.superpowers/sdd/transfer-v4/controller-observation.json` (SHA-256 `df328fb5374cc307226d67b0ed9f201090c094118146d94c7bd4349376971fe5`). Its bound wrapper receipt is `.superpowers/sdd/transfer-v4/controller-probe-finalfix-command.json` (SHA-256 `b5f28f15201014cb619c6b588e2ad90a9a18be0139c1b25d81101a6fb25697f6`). The compact [result manifest](../configs/splits/constraint-v4-development-probe-result.json) records the exact command, identities, resource values, public status/counts, and eight artifact hashes. Run files remain in the ignored `data/constraint-probe-v4-20261009T192624974846Z/` directory. Local ignored preservation is not an archival backup; no independent reproduction was performed.

The README status wording was manually reviewed for plain language and bounded claims because the README Writer's required humanizer subskill was unavailable; this follows the existing v2 documented fallback. Package metadata was checked against `pyproject.toml` and the package entry points before updating links and status text. This documentation does not change task semantics, thresholds, splits, or the v2.2 protocol.
