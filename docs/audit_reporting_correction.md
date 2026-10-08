# V2 audit and accounting correction

Date: 2026-10-08. This bounded correction addresses two reporting/accounting defects identified in the final review. It does not change scientific settings, corpus contents, splits, floors, or the preserved feasibility attempt.

## F1: full-mode source eligibility and sensitivity

Full-mode source `eligible_fraction` previously returned 1.0 unconditionally, including when source proposals were insensitive or the source population was empty. The metric now divides accepted sensitive source groups by the raw unique source-group count, yielding 0.0 for an empty denominator. Full-mode results also retain `aggregate_source_sensitivity` with numerator, denominator, and measured fraction from `_source_summary`. The registered floors and populations remain unchanged. A focused pure aggregation regression uses one sensitive equality chain and one insensitive equality triangle and checks 1/2 = 0.5, plus the empty-source case.

No full-mode corpus report was produced in the preserved study, so F1 has no historical metric to revise. The actual feasibility sensitivity result remains 734/1,239 = 0.5924132365.

## F2: generation CPU boundary

Generation CPU timing now starts at CLI entry and is passed through configuration loading, provenance capture, optional preflight, output setup, and initial manifest creation. Direct API callers default to an origin captured at function entry; a supplied origin must be a finite, nonnegative `process_time` value from the current process. The persisted CPU value is refreshed at the same charged terminal snapshot as elapsed time, after artifact hashing and through the first terminal manifest write/resource check. The existing bounded metadata flush tail remains excluded and is named in the existing accounting note. No manifest fields were added.

The old generation CPU value, 47.125 seconds, is genuine and preserved, but it excludes setup before the original timer and artifact hashing/terminal finalization after it. The recorded combined value, 125.578125 seconds, adds that narrower generation interval to audit CPU measured across a broader boundary. Whole-command CPU was unavailable and cannot be reconstructed. The wall-time value (125.985 seconds), cap-stop NO-GO, peak memory (183,894,016 bytes), and G2 `not_assessed` status are unchanged. No corpus, audit, replay, or model run was repeated, and no old artifact or SHA-256 value was rewritten.

The historical experiment source remains `9f08efddb60537c61a89a90325bc9c832c32ca1c`. The code correction and focused regressions are commit `ae9a90775104f776e029a9e63afbf23a5379c2c2`; this identifies corrected future behavior and does not regenerate old-source artifacts. The documentation/evidence annotation is committed separately after the code commit.
