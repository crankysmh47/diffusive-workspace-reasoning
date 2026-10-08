# Fixed-N transfer corpus v3 Implementation Plan

> **For agentic workers:** Execute this plan sequentially with the authorized Luna method. Do not spawn child agents. Each task ends with a controller review of the diff and its recorded checks.

**Goal:** Implement and audit the registered v3 CPU proposal for bounded fixed-N source and target task generation, including a replay-validated source-support capacity stop.

**Architecture:** Add a strict v3 config and candidate generator while retaining the reviewed v2 schema, candidate evidence types, solver, oracle, and corpus bookkeeping. Resolve exactly two stream versions through a closed dispatcher, share v2 pipeline/replay/manifest/audit code, and bind all records and handoffs to config, source, producer, and stream identities. The feasibility path stops before target sampling only after complete source quotas and source replay closure establish a top-ten support bound below 0.8.

**Tech Stack:** Python standard library, existing `dwr.tasks` modules, `unittest`, strict UTF-8 JSON/JSONL, SHA-256.

**Spec:** [task_spec_v3.md](task_spec_v3.md)

## Global Constraints

- Keep grammar `constraint-task-v1`, three values/modulus 3, `n=3..6` outside source `neq` (`n=4..6`), 2–8 constraints, 1–`n-2` givens outside source `neq` (`g=2..n-2`), parser/verifier/oracle/solver and model-input projection fixed.
- Use seed `20261007`, `random.Random.seed(..., version=2)`, v3 SHA-prefix64 domain, and independent candidate streams by partition and index.
- Preserve source quotas 334/333/333 (1,000 total), feasibility development-validation quotas 100 per pair, full quotas 600 per development pair and 600 final, all v2 floors/bins/splits, 30,000 candidates per partition, one worker, 600 combined active seconds, and 1 GiB.
- Select a remaining active profile before `n`; fail closed on an empty active list; draw no quota-skip or idle attempts.
- Do not introduce templates, padding, target-conditioned choices, weakened floors, retries, final-target feasibility access, ML dependencies, model code, or training.
- Keep all generated data, checkpoints, and run artifacts out of Git; preserve v1/v2 files and all previous outcomes.
- Only a complete full-audit G2 PASS opens fixed-size EX/IDX tensor implementation. Training remains blocked until a separate non-null preregistration exists.

## Review Focus

- A quota context that marks every profile complete must fail before consuming RNG; Task 1 pins the exception and unchanged v2 behavior.
- A `neq` witness with only two colors or givens with only one color must never pass the registered eligibility gate; Task 1 tests each mathematical necessity with the independent oracle.
- A duplicate or ineligible oracle-unique candidate must retain v2 deduplication and denominator precedence; Task 2 tests the replayed stream order.
- A source cap/incomplete prefix must never become a bound-based no-go; Task 2 tests that only complete quotas plus source replay closure enable the gate.
- A passing top-ten bound must never be reported as actual common-support success; Task 3 tests both the capacity report and the unchanged measured four/five-cohort gates.

---

### Task 1: Pin v3 config, stream, and `neq` candidate proposal

**Files:**
- Create: `configs/tasks/constraint_v3.json`
- Create: `dwr/tasks/config_v3.py`
- Create: `dwr/tasks/generator_v3.py`
- Modify: `dwr/tasks/generator_v2.py` only to expose shared pure proposal helpers without changing v2 draws or identities
- Test: `tests/test_task_config_v3.py`
- Test: `tests/test_task_generator_v3.py`
- Test: `tests/test_neq_necessary_conditions_v3.py`

**Interfaces:**
- `load_config_v3(path: str | Path) -> LoadedConfig` accepts only exact reviewed v3 bytes/types and reports `ConfigError` on key, type, value, or hash drift.
- `generate_candidate_v3(root_seed: int, partition: str, candidate_index: int, *, full_profiles: frozenset[tuple[str, ...]]) -> CandidateOutcome` returns the existing typed outcome/evidence classes from `generator_v2.py` and makes no corpus/dedup state.
- Shared public pure helpers from `generator_v2.py`: `normalized_tuples(n: int, kind: RelationKind) -> tuple[tuple[str, ...], ...]`, `source_sensitive(task: ConstraintTask) -> SourceEligibilityEvidence`, and `mixed_primary_eligibility(task: ConstraintTask) -> MixedEligibilityEvidence`. Keep v2 wrappers/results byte-identical.

- [ ] Copy the registered v2 config to v3, change only version/domain/seed labels and the exact v3 schedule/proposal fields; retain every floor, bin, quota, and cap. Compute the final approved raw SHA-256 after review and pin it in `config_v3.py`.
- [ ] Add config-loader cases for the reviewed file (PASS), one unknown key, one changed integer-to-bool value, one changed floor, and one raw-byte change (each must raise `ConfigError`).
- [ ] Write deterministic candidate tests with fixtures for each partition; assert exact candidate header, profile, task serialization, witness, private evidence, rejection reason, and repeat-call equality for v3.
- [ ] Pin the v3 seed material with known-answer SHA-256/prefix64 vectors for `source:0`, `development:17`, and `final:9`; assert v2 golden vectors and v2 config loading remain unchanged.
- [ ] Test active-profile selection before `n`, dynamic order after an earlier quota fills, zero extra RNG calls for active-profile selection, and fail-closed behavior when no profile remains.
- [ ] Add independent exhaustive-oracle fixtures for (a) swapping two unanchored colors when givens use one color, (b) recoloring an unknown to the unused third value when a witness uses two colors, and (c) a missing-anchor bichromatic component. For every fixture where a unique completion is asserted, enumerate all `3**n` assignments independently and verify uniqueness; do not call private generator predicates from the fixture oracle.
- [ ] Assert on bounded small generated all-`neq` tasks that every oracle-unique task satisfies all-three-witness-values, two-distinct-given-values, pair-component anchoring, and `unknown_touching_constraint_count >= 2*(n-g)`. Include the connected-component edge-count argument in the test comments and spec link.
- [ ] Run `python -m unittest tests.test_task_config_v3 tests.test_task_generator_v3 tests.test_neq_necessary_conditions_v3 -v`; expected: all named tests pass, with nonzero discovered count.
- [ ] Review v2 golden tests and exact config hash before commit; commit only task-1 source/config/tests when executing this plan.

### Task 2: Add closed two-stream dispatch, shared pipeline/replay, and source-closure bound

**Files:**
- Create: `dwr/tasks/stream_contract.py`
- Create: `dwr/tasks/cli_v3.py`
- Modify: `dwr/tasks/cli_v2.py` only to expose a shared closed-version CLI runner while preserving the v2 entry point
- Modify: `dwr/tasks/dataset_v2.py` (factor one shared internal runner; keep `run_dataset_v2` default and signature behavior)
- Modify: `dwr/tasks/replay_v2.py` (factor shared replay; keep v2 wrappers/defaults)
- Modify: `dwr/tasks/manifest_v2.py` (closed v2/v3 validation dispatch; preserve `validate_manifest_v2`)
- Modify: `dwr/tasks/audit.py` only if a shared `_cell` import/export is needed; do not change its bins
- Test: `tests/test_task_stream_contract.py`
- Test: `tests/test_task_dataset_v3.py`
- Test: `tests/test_task_replay_v3.py`

**Interfaces:**
- `resolve_stream(generator_id: str) -> StreamContract` accepts only `dwr-generator-v2` or `dwr-generator-v3`; each contract contains exact `config_version`, `outcome_version`, `manifest_version`, RNG domain, candidate callable, and producer module names. Unknown identities raise `ValueError`.
- `run_dataset_v3(config: LoadedConfig, outdir: Path, mode: str, argv: list[str], *, limits: DatasetLimits | None = None, guard: ResourceGuard | None = None, source: dict | None = None, preflight_report: Path | None = None, cpu_origin: float | None = None) -> int` uses the v3 contract. `run_dataset_v2` continues resolving v2 by default; keep its existing `candidate_fn` test injection.
- `replay_outcomes_v3(rows, config, mode="feasibility", *, check=None, partition_check=None, limits=None, expected_attempts=None, allow_partial=False, guard=None) -> list[dict]` and `replay_files_v3(...)` replay with only the resolved v3 candidate producer. Existing v2 entry points continue to resolve v2.
- `replay_source_closure_v3(outcomes_path: Path, records_path: Path, config: LoadedConfig, generation_manifest: dict, *, limits: DatasetLimits, guard: ResourceGuard) -> dict` accepts only a complete source quota prefix, replays every source row/record, and requires development/final to remain unattempted.
- `source_support_capacity_bound(records: Iterable[dict], *, top_k: int = 10, min_fraction: float = 0.8) -> dict` accepts replay-validated accepted source-train records only and calls the exact existing `_cell` implementation.

- [ ] Implement a two-branch dispatch, not a plugin registry. Validate that config identity, RNG domain, candidate function, outcome schema, manifest version, and producer filenames resolve to the same stream.
- [ ] Factor v2 orchestration/replay internals only as needed; leave v2 config loader bytes, defaults, candidate behavior, output fields, quotas, and golden replay records unchanged. Ensure v3 record/outcome schemas are explicit and closed.
- [ ] Add tests for both exact stream IDs, unknown IDs, mixed config/generator IDs, v3 outcome passed to v2 replay, v2 outcome passed to v3 replay, and attempted v2 PASS preflight linked to v3 config (all cross-version cases fail closed).
- [ ] Add integration fixtures for duplicate content, first-seen ineligible content, profile quota completion, source/development/final partition prefixes, and exact resume/replay equality; assert raw denominator counts exclude duplicates and preserve reason precedence.
- [ ] Implement `replay_source_closure_v3` so the generation driver can prove the source prefix reaches every registered quota, every outcome and accepted record matches deterministic replay, and development/final have zero attempts before it considers the bound. It must reject a prefix that continues after source quota completion or omits an expected record.
- [ ] Implement the early gate only after `replay_source_closure_v3` succeeds. Use the ten most common registered `_cell` values from accepted source-train records, lexicographic cell order for ties, and report source training mass. If source quotas or replay closure are incomplete, leave `capacity_bound_status` as `not_assessed` and preserve the actual cap/integrity cause.
- [ ] For a completed source prefix with mass `<0.8`, write the closed v3 terminal extension `{ "early_stop": {"status":"NO_GO", "reason":"source_support_capacity_bound", "source_closure_replay_sha256":"<64 lowercase hex>", "top10_source_train_mass": <finite number in [0,1]> }}` with source complete, development/final not attempted, and global generation incomplete; emit no target candidate. The v3 manifest validator accepts this intentional early terminal state only when the source-closure replay identity and bound are present and valid. For mass `>=0.8`, record `capacity_bound_passed` and continue to registered development validation; do not label actual support passed.
- [ ] Test mass `<0.8`, exactly `0.8`, `>0.8`, tied cells, fewer than ten cells, empty train, source quota not complete, tampered replay row, wrong source partition split, missing accepted record, post-quota extra outcome, and incorrect early-stop linkage. The cap/incomplete test must show no bound-based no-go.
- [ ] Run `python -m unittest tests.test_task_stream_contract tests.test_task_dataset_v3 tests.test_task_replay_v3 -v`; expected: all named tests pass and all cross-version/tamper cases fail at the asserted boundary.
- [ ] Review all shared-code diffs against v2 tests, then commit task-2 changes.

### Task 3: Add independent v3 audit, reports, linkage, and budget enforcement

**Files:**
- Create: `dwr/tasks/audit_cli_v3.py`
- Modify: `dwr/tasks/audit_cli_v2.py` only to expose a shared closed-version audit CLI runner while preserving the v2 entry point
- Modify: `dwr/tasks/audit_v2.py` through shared version-aware aggregation only; retain `run_audit_v2` behavior
- Modify: `dwr/tasks/manifest_v2.py` if audit linkage needs an additional closed v3-only field
- Modify: `dwr/tasks/dataset_v2.py` only for exact v3 preflight consumption/binding
- Test: `tests/test_task_audit_v3.py`
- Test: `tests/test_task_preflight_v3.py`
- Test: `tests/test_task_budget_v3.py`

**Interfaces:**
- `run_audit_v3(data_dir: Path, config: LoadedConfig, source: dict, mode: str, *, guard: ResourceGuard | None = None, allow_partial: bool = False) -> dict` independently validates/replays v3 artifacts, recomputes eligibility with `audit_interactions`, and returns a version-tagged audit report.
- `validate_preflight_v3(path: Path, config: LoadedConfig, source: dict, *, limits: DatasetLimits | None = None) -> dict` accepts only one closed `PASS_FEASIBILITY` report whose config/source/producer/generation/audit identities exactly match v3 full-mode input.
- The v3 feasibility report fields include `g2_status`, `proposal_status`, `proposal_reason`, `capacity_bound`, `source_closure_replay`, `four_cohort_common_support_proxy`, and explicit unassessed held-out/final status. For a capacity stop, `g2_status` is `not_assessed`, `proposal_status` is `NO_GO`, and `proposal_reason` is `source_support_capacity_bound`.

- [ ] Implement an independent source-capacity recomputation from replay-validated accepted `source_train` records, using the existing cell function and separate audit aggregation. Do not call the generator's bound helper or private eligibility predicates.
- [ ] Test the source-bound derivation: each common cell needs at least 10 rows from each of three feasibility target profiles; each target has a 100-group quota; thus there can be at most 10 qualifying cells. Compute source cell mass as each accepted `source_train` cell count divided by all accepted `source_train` groups. Show that source top-ten mass `<0.8` makes the four-cohort floor impossible, while mass `>=0.8` leaves actual support unresolved.
- [ ] For early capacity no-go, assert source quota/replay closure, no development/final rows, incomplete global generation, proposal no-go reason, `g2_status=not_assessed`, and no statement of observed mismatch or measured proxy failure. For incomplete source, assert capacity `not_assessed` and the registered cap/incomplete reason only.
- [ ] For a bound pass, generate and independently evaluate the unchanged feasibility floors and measured four-cohort proxy. Keep source train as the source cohort and three eligible development-validation cohorts, each with exact existing 10-row/0.8 thresholds. A proxy failure is a measured proposal no-go; a proxy pass is feasibility only.
- [ ] For complete mode, independently recompute all unchanged five-cohort floors, denominators, stratified reports, uniqueness, integrity, annotation exclusion, and source composition checks. Preserve all negative/inconclusive results.
- [ ] Bind preflight and full manifest to exact `dwr-generator-v3`, config SHA, clean source HEAD, producer hashes, source identity, generation manifest/artifacts, and independent feasibility audit artifact. Test every field independently by substitution and reject a v2 linkage.
- [ ] Enforce the single 600-second/1-GiB/one-worker invocation accounting across generation, source replay, independent audit, and finalization. Preserve artifacts and typed failures at every cap.
- [ ] Run `python -m unittest tests.test_task_audit_v3 tests.test_task_preflight_v3 tests.test_task_budget_v3 -v`; expected: all named tests pass, including terminal no-go and cross-version handoff cases.
- [ ] Review the report schema and the v2 behavior-preservation suite, then commit task-3 changes.

### Task 4: Run one registered gate, retain evidence, and close the documentation handoff

**Files:**
- Create: one unique ignored feasibility artifact directory under `data/constraint-v3-feasibility-<UTC>/`
- Create: one unique ignored full artifact directory only if the complete feasibility gate passes
- Create: `configs/splits/constraint-v3-feasibility-<outcome>.json` with compact identities/hashes only
- Create: `docs/task_suitability_v3.md` with exact commands, observed results, evidence paths, and limits
- Modify: `README.md` and `todo.md` only after evidence/report review; leave G2 unchecked unless full independent audit passes
- Test/verification: `python -m unittest discover -s tests -v`

**Interfaces:**
- Feasibility command sequence, with a fresh Windows UTC run ID:

  ```powershell
  $feasibilityRun = "data/constraint-v3-feasibility-$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))"
  python -m dwr.tasks.cli_v3 --config configs/tasks/constraint_v3.json --mode feasibility --out $feasibilityRun
  python -m dwr.tasks.audit_cli_v3 --config configs/tasks/constraint_v3.json --mode feasibility --data $feasibilityRun
  ```

- Conditional full command sequence, only after matching independent `PASS_FEASIBILITY`:

  ```powershell
  $fullRun = "data/constraint-v3-full-$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))"
  python -m dwr.tasks.cli_v3 --config configs/tasks/constraint_v3.json --mode full --out $fullRun --preflight-report "$feasibilityRun/preflight_report.json"
  python -m dwr.tasks.audit_cli_v3 --config configs/tasks/constraint_v3.json --mode full --data $fullRun
  ```

- [ ] Commit implementation and tests; verify clean worktree, exact branch, source HEAD, v3 config hash, producer hashes, and available budgets before the gate. Write the resolved manifest before any candidate is sampled.
- [ ] Run exactly one feasibility generation/audit sequence. It uses 1,000 source groups at 334/333/333 and 100 eligible development-validation groups per pair only if the replay-validated source top-ten bound is at least 0.8. Final remains absent. Never overwrite an artifact directory or rerun with changed rules.
- [ ] If source caps before quotas complete, preserve all rows, report the cap cause, and mark the source capacity bound not assessed. If complete source top-ten mass is below 0.8, preserve the terminal early no-go and do not sample targets. If bound passes, require the actual measured four-cohort proxy and every computable feasibility floor to pass.
- [ ] Only after complete independent `PASS_FEASIBILITY`, freeze the same clean HEAD/config/source identity and run one full corpus generation plus independent full audit with the matching v3 preflight link. A full-mode attempt or audit failure is preserved; never retry under changed rules.
- [ ] Only a complete independent full G2 PASS makes fixed-size EX/IDX tensor implementation the next authorized milestone. Do not implement models or train in this CPU plan; training requires a separate non-null study registration.
- [ ] Run the entire unit suite once after integration with `python -m unittest discover -s tests -v`; capture the exact result. Review full branch diff and evidence identities; make one bounded fix wave only for concrete defects, then rerun the directly affected checks and final suite if changed.
- [ ] Write `docs/task_suitability_v3.md` from saved manifests, audit reports, and per-example evidence, separating implementation checks, generation outcome, gate result, and unassessed claims. Preserve v1/v2 status and artifacts; update README/todo only to describe verified v3 outcomes and leave G2 unchecked unless full PASS is recorded.
- [ ] Create a compact split/hash manifest without task data; verify generated data, checkpoints, and run artifacts remain ignored. Record exact verification commands/outcomes and artifact paths before checking any milestone.
- [ ] Commit task-4 documentation and compact manifest only after the evidence/report audit; do not change repository visibility or licensing.

---

## Plan self-review

The four tasks cover the v3 config/stream/proposal, strict two-version reuse, source-closure capacity gate, independent feasibility/full audit and preflight linkage, exact budgets, one feasibility sequence, conditional full sequence, and durable evidence. Cross-version config/outcome/audit/PASS linkage, incomplete source prefixes, duplicate/ineligible denominator behavior, tied cells, empty histograms, and the distinction between capacity and observed support are assigned explicit checks. This design phase creates no tests or data and does not execute any generation command.
