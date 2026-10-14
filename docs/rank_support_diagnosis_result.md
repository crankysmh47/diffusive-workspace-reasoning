# Retained rank support diagnosis result

Recorded2026-10-14. Terminal **PASS_DIAGNOSIS** means complete bound-input integrity and descriptive reporting only. The original qualification remains **NO_GO_DEVELOPMENT**, G2 remains unassessed, and no full corpus/final/model/training ran. No new scientific examples, seeds, solver calls or candidate replay were introduced by this analysis.

## Frozen execution

Clean source `bb9540e2f92ea05c4ba502f2bbbcf2c469b32402`; config SHA256 `63145f650d17dc5bd205bfd1dc0db38b7c2b68dadc8ea445ddf8805e296bd394`. Sole run `runs/certificate-rank-support-diagnosis-20261014-v1`:

```powershell
python -m dwr.tasks.certificate_rank_support_diagnosis --config configs/tasks/certificate_rank_support_diagnosis_v1.json --output runs/certificate-rank-support-diagnosis-20261014-v1
```

The input is exactly the original3600-source/4800-development/final0 qualification at41bd784. All six input artifact hashes and original40 raw Git producers are bound; current41 runtime producers match the new clean commit. Rows route before semantic metadata: only source_train/development_validation annotations are used. Protected/splitless rows create no summaries; answers/tasks/private certificates are not parsed. The full suite passed493 tests in43.653 seconds on stable bytes, and independent review cleared the implementation. See [verification](rank_support_diagnosis_implementation.md) and [scope/departures](rank_support_diagnosis.md).

The terminal manifest records1.734 seconds active wall,0.5 seconds CPU and30,220,288 bytes sampled peak RSS. The receipt covers report hashing; the small final manifest write is guard-checked but excluded. External wrapper1.796 seconds, exit0/no timeout under65 seconds. One60-second/256MiB invocation was consumed, with zero retries. Root independently checked four output hashes, original input hashes unchanged, current raw Git identities/config/resolved manifest/budgets and unchanged NO-GO/G2 boundary. The [compact record](../configs/splits/certificate-rank-support-diagnosis-result.json) preserves exact identities and aggregates.

## What the inventories show

Every E group lies in prompt bin4 (>1023 bytes). Attrition is solver-node-related, not a prompt-bin mismatch. Across source / EQ-NEQ / EQ-ADD / NEQ-ADD, E counts in node bin2 (4..7) are1156/190/462/222; bin3 (8..15)153/22/125/81; bin4 (16..31)6/0/6/13; bin5 (32..63)2/0/0/0. All other bins are zero. These are the frozen reference solver's annotations, not a measure of neural reasoning complexity.

The complete140 cell/bin vectors have exactly five cells with at least10 E in every permitted cohort:

- A/g2c8, prompt4/node2: source253, EQ-NEQ36, EQ-ADD123, NEQ-ADD55.
- A/g2c8, prompt4/node3: source86, EQ-NEQ16, EQ-ADD64, NEQ-ADD56.
- B/g3c6, prompt4/node2: source250, EQ-NEQ26, EQ-ADD129, NEQ-ADD51.
- C/g3c7, prompt4/node2: source302, EQ-NEQ48, EQ-ADD116, NEQ-ADD54.
- D/g3c8, prompt4/node2: source351, EQ-NEQ80, EQ-ADD94, NEQ-ADD62.

The other node3 B/C/D cells have only2 EQ-NEQ E groups each. Globally widening the node bound to15 would pool these sparse strata and would not establish fine-cell matching. Higher-node A bins have zero EQ-NEQ support. An observed zero/small count is not proof of a universal structural impossibility or exhausted population.

Source's additional A/node3 cell is imbalanced: EQ9, NEQ14, ADD63. Aggregate source overlap is not balanced primitive support. Source GAC stalls159/1317;158 are ADD and1 NEQ. The method comparison must report these locality/primitive/difficulty differences rather than claim a pure composition-only shift.

## Nontriviality and filtering

Development E1121 contains452 stalled groups. Original A874 contains286 stalled groups; the247 excluded E groups contain166 stalls. Thus excluding E outside the old envelope removes36.7% of the observed stalls, and stall fraction changes from40.3% in E to32.7% in A. Both ADD no_unary branches stall on every eligible group, as their input run checked. Source A has114 stalls versus159 in source E. These diagnostics do not certify a model effect.

For the jointly populated A/node3 cell, EQ-ADD has42 stalls among64 E and NEQ-ADD44 among56. This identifies a plausible additional supported hard regime. It does not retroactively satisfy the original fixed-bin qualification, select a new model population or authorize another generator invocation.

## Decision boundary

Root selects a separately reviewed prospective five-stratum design using all cells meeting the diagnostic's predeclared joint-capacity rule, with fixed weights, fresh registration/replication and original all-E denominators. That is a substantive population/estimand departure from the four-cell study, not a correction to its NO-GO. A globally enlarged cutoff, dropping the hard branch or silently accepting a favorable union is rejected. The new design must freeze its exact strata and fine-cell floors before any draw, address source-ID-test scarcity and disclose source imbalance/locality/GAC shifts. It still requires a new qualification and complete five-cohort G2.

Raw annotation projections/report remain ignored; this is an agent-run descriptive analysis using independently checked frozen annotations. No archival raw-artifact backup or independent-person reproduction exists. Implementation-boundary and test-order departures remain documented and are not erased by PASS_DIAGNOSIS.
