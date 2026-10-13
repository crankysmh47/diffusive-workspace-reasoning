# Rank qualification: development support NO-GO

Recorded 2026-10-14. The sole registered qualification is closed **NO_GO_DEVELOPMENT**. Source passed every floor. Development passed capacity, signature diversity, eligibility, accepted fraction, per-cell counts, pooled difficulty and hard-core diversity, but two ADD compositions failed the fixed common-support mass floor. This is a scientific negative result, not a resource failure. Full corpus/final generation and model work are blocked. No retry, seed search, prefix extension or threshold change follows this result.

## Identity and verification

Clean scientific source `41bd7846f8435d02616a526ef6dbdbae9663271d`; qualification config SHA256 `8fa592fe2bedc012a1a10fb5ecd74bbd871df2f0cedcf5026562f5b518c2d601`; root20261014. Run path `runs/certificate-rank-qualification-20261012-v1`. Command:

```powershell
python -m dwr.tasks.certificate_rank_study --config configs/tasks/certificate_rank_qualification_v1.json --output runs/certificate-rank-qualification-20261012-v1
```

All3600 source and4800 development attempts were retained; final0. **PASS_REPLAY**,8400 exact indices rederived. No new scientific attempt was introduced by replay. Production is explicitly not fixture-only. G2 remains `not_assessed`; no full-study preflight was emitted. Before invocation the controller passed474 CPU tests in40.367 seconds on stable tracked bytes and the release reviewer cleared the integrated source. See [implementation verification](certificate_rank_implementation.md).

The controller subsequently checked all six opaque artifact hashes, initial/final manifest source equality, all40 runtime producer hashes against the recorded Git blobs, config and command identities, exact counts, replay and the preserved terminal status. No raw/protected task payload was parsed for these checks, and no second replay/population was run. The source commit and clean working state remained unchanged during invocation. Small tracked evidence: [result manifest](../configs/splits/certificate-rank-qualification-result.json).

## Permitted aggregate results

Source-train U=1317, all1317 sensitive and scientifically eligible E. Accepted envelope A=1156, Sig(A)=932, Sig(E)=1071,139 E signatures outside A. A/all raw source=0.321111; accepted mass E=0.877752. Each cell has253,250,302,351 A groups respectively. All source floors passed. Value fractions are0.341984/0.332612/0.325404, with both given counts represented. This preserves the prospective Sig(A) estimand rather than the stronger historical universal Sig(E) promise.

Development-validation aggregates:

- EQ/NEQ: U279,E212,A190,Sig(A)169. Cell A/B/C/D counts36/26/48/80; supported mass190/212=**0.896226**, PASS.
- EQ/ADD: U729,E593,A462,Sig(A)449. Cell counts123/129/116/94; supported mass462/593=**0.779089**, FAIL against0.8. E-only groups131, E-only signatures111. Hard E242, hard A161; core signatures g2=77,g3=28.
- NEQ/ADD: U345,E316,A222,Sig(A)220. Cell counts55/51/54/62; supported mass222/316=**0.702532**, FAIL against0.8. E-only groups94, E-only signatures93. Hard E148, hard A86; core signatures g2=68,g3=11.

Every pair passed A>=100, Sig(A)>=12, each cell>=10, E/U>=0.25 and A/U>=0.25. Both ADD profiles passed each given-count hard-core diversity floor>=2. Across validation E1121,A874, hard E390 and ordinary E731. GAC-stalled E452/1121=**0.403211**, exceeding both the10-group and10% floors in at least two profiles. GAC stalling is a registered diagnostic, not evidence of learned reasoning difficulty or architectural benefit.

The failed floor map contains only `eq/add_mod/support`, `neq/add_mod/support` and the corresponding `common_support`. Source and EQ/NEQ satisfy the envelope mass condition. All E, including the247 development groups outside A, remain in the required denominators; reporting only A would conceal this failure.

## Resources and preservation

The terminal manifest records300.437 seconds active wall and299.140625 seconds CPU, sampled peak RSS128,884,736 bytes (about123MiB). Its precise accounting boundary is through report hashing; the small final manifest write is guard-checked but excluded from that resource receipt. The external wrapper records300.578 seconds, exit1 (scientific NO-GO), no timeout under1805 seconds. The1800-active-second/1GiB/one-worker limits were respected. Artifact-write timing fields report0.0 at the runtime clock resolution; do not interpret them as zero-cost I/O.

Local raw outcome ledger20,207,054 bytes and accepted projection ledger1,930,256 bytes remain ignored. All prior negative, interrupted and diagnostic evidence is preserved. Local ignored files are not an archival backup. This is an agent-run qualification with independent logic cross-checks, not independent-person reproduction or model validation.

## Decision and next research boundary

The qualification establishes adequate accepted per-cell capacity, hard-core diversity and pooled nontriviality on its permitted cohorts, but does not pass the required supported-population comparison. It does not establish the previous fixture's discarded all-cell hard requirement. The next question is how to obtain a defensible matched task population while retaining all eligible outcomes and nontriviality. A new bounded prospective design would be required; no automatic filtering, support expansion, seed replacement or post-result floor relaxation is authorized by this closed study. The existing model/EX-IDX plan remains conditional on full G2 PASS.

This result does not reject the exchangeable workspace hypothesis: no workspace was built or trained. Final grammar was checked on explicit hand states only, and final production tasks remain ungenerated. A complete held-out five-cohort comparison, full leakage audit and model transfer conclusions remain unavailable. The owner-requested PR checkpoint is reached by reviewed code, passing CPU checks and this recorded terminal result; PR creation does not mean the research gate passed or authorize merging.
