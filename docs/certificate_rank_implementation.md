# Rank implementation verification

Recorded 2026-10-14, before the sole registered population. Authority: [rank design](certificate_rank_design.md), [plan](superpowers/plans/2026-10-12-certificate-rank.md), protocol v2.2. Luna implemented; the controller made research decisions and performed release verification. No new production source/development/final population or model training occurred during implementation.

## Completed code and checks

Finite F3 arithmetic, strict rank-certified pair outcomes, the frozen unary-permitted final grammar, U/E/A ledgers, independent scalar/oracle/solver/verifier checks, replay, protected routing, bounded writes/resources, qualification linkage and the full five-cohort audit are implemented. Old source and ordinary/EQ-NEQ development constructors retain their registered delegation laws. Final tests use explicit hand states only; final production construction remains gated. Canned tests establish conditional correctness, not yield, population diversity or G2.

The controller ran `python -m unittest discover -s tests -v`: **474 passed in 40.367 seconds**, wrapper40.532 seconds, exit0, no timeout under the120-second test cap. SHA-256 snapshots of every tracked file before/after were equal. Log SHA256 `4d6689050c96a40518a206cf55aefe9cc8b4946c3874aa8033bab48278a8e3f6`. Local evidence: `.superpowers/sdd/rank-resume-20261013/controller-full-suite-20261013T190129427593Z.log` and matching `.json`. `git diff --check` passed.

Earlier controller component checks passed27 F3/pair tests and21 study tests; the later stable suite above covers the final integrated bytes. Luna's final reporting check passed22 study tests in0.217 seconds with retained RED evidence. Independent Task2 rereview and integrated release review are retained in the same ignored evidence directory. The release review found one terminal reporting defect, then `release-rereview-20261014.md` closed it with **READY_FOR_SINGLE_QUALIFICATION**, subject to clean committed source. Review note SHA256 `0b796d008331ff7bed88983c6b53f239a5b7a4038c93dfb59783c74d03c857d7`.

Earlier interrupted and overlapping-worker full-suite failures are preserved as diagnostic evidence. An isolated historical v4 fixture passed independently; the stable complete suite also passed. No proven cause is assigned to the earlier intermittent errors, and no historical study artifact or threshold was changed.

## Source and operational decisions

Runtime scientific producers must match raw committed Git blobs, not filtered working-tree identities. UTF-8/LF is verified before freeze. Both closed configuration hashes remain unchanged: qualification `8fa592fe2bedc012a1a10fb5ecd74bbd871df2f0cedcf5026562f5b518c2d601`; full `c01056ece9e4d63981301b2cff2c224df952499a54d8b746ac1d73b2df15be6f`.

The 2026-10-14 reporting correction makes a complete replay-valid production full audit explicitly `g2_status=pass` or `fail`, copies it into the manifest and resets it on late revocation. Qualification, fixtures, incomplete stages, replay mismatch and exceptions remain `not_assessed`. This changes reporting only; comparisons, constructor law, RNG, samples, gates and budgets are unchanged. It was reviewed before any population.

Controller ruling: preserve this plan's ignored evidence/worktree instead of deleting it after implementation, because the binding research protocol requires preservation of failures and run evidence. Use README Writer with manual prose review under the previously documented missing-humanizer fallback. Source backup, agent checks and local artifact retention do not establish archival backup or independent reproduction.

## Next admitted action

Freeze a clean source commit, then execute exactly once:

```powershell
python -m dwr.tasks.certificate_rank_study --config configs/tasks/certificate_rank_qualification_v1.json --output runs/certificate-rank-qualification-20261012-v1
```

Root20261014; source3600, development4800 only after source PASS, final0;1800 active seconds,1GiB,oneworker, external watchdog1805. Zero retries. Preserve every terminal result. Qualification alone cannot pass G2. Only PASS admits the already frozen conditional full path; only full G2 PASS admits fixed-N infrastructure. The qualification's verified terminal record is the owner-requested PR checkpoint, including a scientific NO-GO. Do not merge the PR or reinterpret a negative result as a workspace result.
