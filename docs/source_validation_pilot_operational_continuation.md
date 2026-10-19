# Source-validation pilot operational continuation — 2026-10-19

Owner authorized repairing the bookkeeping failure, preserving K0/K1 and completing K4. No scientific result is being repaired or retried.

## Failure and preserved evidence

V4 source a708fb399eca01c42e12bcc9f4bc80f7695dbfde completed K0 and K1 model work (512 updates each, initial/final 45-case evaluations). K1 arm auditing succeeded but its final record sealing failed. A surviving V2 watchdog overwrote V2 status.json at its old deadline; V4 registration incorrectly depended on the mutable status hash. Preserve that successor, its watchdog receipt, the original captured startup proof, every worker segment and V4 operational_failure status. Do not restore or fabricate historical status bytes.

The correction authenticates the immutable original startup proof and exact watchdog successor, including zero-access evidence, and prevents a watchdog from rewriting an already failed job with no live owned processes. It still enforces deadlines against live owned processes.

## Recovery contract

Finalize K1 from its existing audited records and checkpoint only, after checking worker terminal artifacts and matching audit hashes against the existing arm_audited event. Append the recovery source/command/time to its arm record before sealing; report the later seal separately. No new model prediction or optimizer update is allowed for K0/K1. The original V4 matrix remains an operational failure.

Create a uniquely named V5 continuation matrix. Import hash-bound audited K0/K1 results from V4 explicitly; do not copy them into fresh arm records or represent them as newly trained. Run only the never-started K4 arm, from the original fresh model and the same initialization/noise seeds. Retain 512 batch1 updates, exact cohort, task framing, dtype, optimizer, sampling, model, workspace, gates, diagnostics and strict verifier. Training batch changes and evaluation batching are deferred until a separately registered future study.

Apply the owner's requested minimum initial free VRAM of 13000 MiB instead of14500 MiB. This is a startup availability threshold, not a13GB consumption cap; retain14336 MiB allocated/reserved and12288 MiB process-resident caps.

The V5 recovery budget is at most5400 whole-matrix seconds (5220 work plus180 seal), outer watchdog5460, with the same5400 per-arm cap. Launch no later than2026-10-19T20:05:42.301480Z so all5400 seconds fit inside the original V4 absolute deadline2026-10-19T21:35:42.301480Z. Include actual V4 elapsed, downtime, finalization and V5 elapsed in reporting; this is not a reset of the original allowance. The original600s inter-arm overhead budget has been exceeded by repair downtime, so record this explicit operational departure rather than claim unchanged overhead compliance. Do not exceed the original4.5h absolute deadline. No K0/K1 rerun, tuning, new seed or additional K4 scientific retry.

## Provenance and gates

V4 controls retain their actual a708 source identities; K4 records the clean repaired commit. This is mixed source provenance. Verify computational source files against a708: only operational supervision/registration/watchdog, continuation/reporting and their focused tests/config/docs may differ. Never rewrite original source identities to satisfy pairing. Explicitly authorize the two source identities at result aggregation, while still enforcing equal data/model/tokenizer/schedule, initialization and all512 example/batch/noise pairs.

Verify focused regressions for terminal watchdog behavior, authenticated original evidence/successor, tamper rejection, import integrity, changed scientific settings, source provenance and deadline enforcement. Freeze a clean commit before GPU launch. These checks do not establish full hard-loss recovery acceptance, which remains incomplete.

After all K4 diagnostics complete, recompute the original learning and workspace gates. On a miss, stop transfer advancement; use existing recorded failure evidence first and register at most one justified bounded diagnostic before new training. A passing one-seed pilot is exploratory and is not exchangeability or transfer evidence.

## Verified implementation and commands

Root verified the actual K1 worker terminal record and both existing audit hashes against the pre-failure arm_audited event, then appended operational_finalization_recovered with exact finalizer file hashes/dirty status and sealed the arm under its original deadline at2026-10-19T18:46:58.815394Z. The recovered arm manifest SHA256 is27b9fb73b796f32793c1d713608f202826f69686c4fe7d899df2305b5c63cf5a. Its model work retains a708; the later metadata-only seal explicitly records mixed finalization provenance.

Root ran `.venv-ml/Scripts/python.exe -m unittest tests.test_source_pilot_controller tests.test_source_pilot_continuation tests.test_source_pilot_continuation_registration tests.test_source_pilot_startup_retry`:39 tests passed,0 skipped,3.377s. The broader74-test task run initially exposed a missing original_deadlines field in a hard-loss test fixture; retain this observed failure, repair the fixture rather than weakening production validation, and record the subsequent exact result below.

The closed V5 configuration SHA256 is826004053ef194a6e817c9f660eef9ec36cacdb350e1e86a2ed6c063d03e0232; parent continuation manifest SHA256 is89e832a0ffcf48899035ba4c79a8053604f81049729523d2cf41e44961f990f6. Real local registration/import/scientific-source checks and prospective launch-window validation pass. Prelaunch GPU snapshot14719MiB free; no user application was terminated. The stale V4 watchdog was stopped after checking its PID/creation/command identity; the snapshot is retained at runs/pilot-v4-watchdog-stop-20261019.json.

Launch after freezing a clean source checkpoint:

```powershell
.venv-ml/Scripts/python.exe -m dwr.experiments.source_pilot_job start --config configs/experiments/source_validation_pilot_operational_continuation_v5.json
.venv-task/Scripts/python.exe -m dwr.experiments.source_pilot_job status --config configs/experiments/source_validation_pilot_operational_continuation_v5.json
```

These use the detached supervisor/worker/watchdog and durable optimizer-boundary checkpoint machinery; they do not establish complete hard-loss recovery acceptance. Do not edit the producer after launch. Capture launch/progress/results in ignored receipts or a separate publication checkout until the job ends.

Final prelaunch verification on the frozen candidate: `.venv-task/Scripts/python.exe -m unittest tests.test_source_pilot_startup_retry tests.test_source_pilot_continuation tests.test_source_pilot_continuation_registration tests.test_source_pilot_registration tests.test_source_pilot_controller tests.test_source_pilot_job` completed75 methods in6.490s,66 passed and9 optional-ML skips. The corrected hard-loss fixture regression executed and passed; this is not the outstanding full end-to-end hard-loss matrix. The separate actual ML command above completed40 tests in3.458s with0 skips after the additional import-content tamper regression. `git diff --check` passes. No scientific model/controller/kernel code changed; source_pilot.py changes are restricted to declared-source aggregation.

Prelaunch physical-byte validation caught Windows CRLF normalization in newly created registrations before any V5 launch. New V5 JSON/source/docs were normalized to committed LF and their new receipt/config pins recomputed; original V1 mixed-newline registration and all historical pins remain untouched.
