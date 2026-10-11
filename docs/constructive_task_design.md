# Constructive task design: preserved ambiguity certificates

Date: 2026-10-11. Status: prospective bounded design spike, not a production generator registration. The owner delegated creative design choices to the controller. Historical studies and floors are immutable; the CPU follow-up remains NO-GO. This note authorizes only the fixture mechanism check below. No historical raw rows, protected split statistics, final profiles, corpus sampling, model selection or training.

## Problem and alternatives

The [CPU follow-up](task_suitability_cpu_followup.md) established diversity but failed source sensitivity (893/4,499) and accepted-envelope yield (795/4,499), against unchanged 0.25 floors. The current generator samples constraints satisfied by a planted assignment, then tests necessity. Satisfying the assignment does not imply that any selected constraint matters. The aggregate result does not identify a per-profile cause or prove that all rejected tasks are easy.

| Approach | Benefit | Main limitation | Decision |
| --- | --- | --- | --- |
| Generate densely, then prune to an irredundant basis | Directly removes redundancy | Changes constraint counts and support cells; padding can undo the benefit; adaptive internal trials need full accounting | Do not use as the next main proposal |
| Enumerate a small motif library and instantiate values | Transparent correctness | Isomorphic/value variants may provide little family diversity and teach a template shortcut | Use only for correctness fixtures |
| Construct with explicit alternative assignments, preserving ambiguity until an essential constraint is included | Necessity has a checkable mathematical certificate; padding cannot silently destroy it | Conditional pools may be too small, or concentrate on easy graph motifs | Preferred mechanism, conditional on capacity/difficulty/support evidence |

## Construction rule

For a source task, choose a satisfying assignment `w`, fixed givens, a distinct alternative `a` agreeing with the givens, and an essential constraint `e` that is true under `w` and false under `a`. Every other constraint must be true under both. Removing `e` leaves at least `w` and `a` as solutions, regardless of added compatible padding. Independently check that the full task is unique; the certificate does not establish uniqueness.

For a mixed profile, choose one distinct alternative `a_k` per present relation kind `k`. All alternatives agree with givens. Every selected constraint of kind `r` must be true under `w` and under every `a_k` with `k != r`. At least one constraint of each kind `k` must reject `a_k`. Removing kind `k` then leaves at least two solutions. Independently check full uniqueness, connected unknown-variable factor graph and every present kind's participation in that graph. These are separate requirements.

An especially conservative padding rule makes filler constraints true under `w` and every alternative. It preserves already established certificates, but may sharply limit the available pool. Do not fill a count quota by relaxing certificate checks. Do not equate source single-constraint sensitivity with every constraint being necessary, or mixed kind necessity with individual constraint necessity.

Witnesses, alternatives, construction order and graph annotations remain private generation/audit metadata; model input remains the unchanged task-only serialization. This does not authorize conditioning the learned workspace on an answer.

## Single bounded fixture check

Scope is mathematical mechanism validation, not a sampled population or performance comparison. Use exactly six fixed four-variable fixtures, givens on `v0,v1`, no RNG, no train/validation/test allocation. Only source single-kind and development pair profiles are present; no final three-kind task is constructed. Assignment tuples below follow `v0,v1,v2,v3`. EQ/NEQ operands are unordered; ADD input operands are symmetric but its output is not.

| Profile | `w` | Alternative(s) | Base constraints |
| --- | --- | --- | --- |
| EQ | (0,0,0,0) | EQ: (0,0,0,1) | EQ(2,3), EQ(0,2), EQ(1,2) |
| NEQ | (0,1,2,0) | NEQ: (0,1,2,1) | NEQ(0,2), NEQ(1,2), NEQ(1,3), NEQ(2,3) |
| ADD | (0,1,1,2) | ADD: (0,1,1,0) | ADD(0,1;2), ADD(1,2;3) |
| EQ/NEQ | (0,1,0,2) | EQ: (0,1,1,2); NEQ: (0,1,0,0) | EQ(0,2), NEQ(1,3), NEQ(2,3) |
| EQ/ADD | (0,1,0,1) | EQ: (0,1,2,0); ADD: (0,1,0,2) | EQ(0,2), ADD(1,2;3) |
| NEQ/ADD | (0,1,2,0) | NEQ: (0,1,0,1); ADD: (0,1,2,1) | NEQ(0,2), NEQ(1,2), ADD(1,2;3) |

For source fixtures, designate EQ(2,3), NEQ(1,3), ADD(1,2;3) respectively as the single protected constraint; all other base constraints must satisfy the source alternative. For mixed fixtures the base may contain several constraints rejecting its kind's alternative, exactly as the per-kind rule permits.

Enumerate the complete normalized witness-compatible filler pool whose constraints are true under every alternative and are not already in the base. For pure NEQ exclude constraints involving only given variables, preserving its existing convention. List the base first, then subsets of filler constraints in increasing subset size and lexicographic canonical constraint order, with total constraint count <=8. Evaluate at most 64 variants per profile, including the base, with no adaptive refill or retry. Record the untruncated possible-variant count separately; every evaluated variant is logged, including failures. Do not treat variants, value permutations or isomorphic copies as independent research families.

An independent scalar evaluator over all 81 assignments must count full solutions and single-constraint/type deletion solutions. Cross-check existing exhaustive oracle, reference solver, source/mixed eligibility and independent audit interaction logic. Check the unknown-variable graph independently or against the audit, not by the construction label. Capture canonical content/structure counts and exact prompt bytes/solver nodes descriptively; these n=4 fixtures do not establish the n=6 registered support envelope.

Success means six valid base certificates, unique base tasks and agreement of independent counts/eligibility, with preserved certificates for every bounded variant. Any disagreement is a correctness blocker; stop without modifying the fixture list to obtain success. Also report limited filler pools and collapsed structure diversity even if correctness passes. No corpus floor or G2 is passed by this check.

Implementation uses an ignored, explicitly throwaway helper and unittest file under `.superpowers/sdd/constructive-design/`; no production module changes. Bind helper/tests/configured fixtures, producer hashes, clean source, command, Python/environment and no-RNG identity in an initial manifest before the sole invocation. One worker, 120-second combined script wall cap, 1-GiB peak RSS; checks before/after work and finalization, no reset. Tests have a separate 60-second cap. Use unique ignored `runs/constructive-design-*` output, stable UTF-8 JSON and SHA-256 artifact hashes. Preserve incomplete/failing outputs. Run neither the historical development CLI nor its audit CLI.

## Production design prerequisites, not waived by this spike

Before a new population run, freeze a new generator/RNG identity, witness/alternative/topology distribution and candidate accounting. A candidate receives one declared construction path; no hidden search until eligibility, no result-dependent seed selection, and no selecting based on protected split membership. Log construction failures, pool exhaustion, duplicates, oracle outcomes and envelope rejects. Keep existing scientific denominators/floors and separately report usable unique groups per raw construction attempt and internal solver work so a guaranteed sensitivity certificate cannot conceal low yield.

Demonstrate capacity at the intended six-variable cells, not just these fixtures. Vary where ambiguity lives and how constraints couple unknowns; graph isomorphisms and arithmetic-preserving transformations stay grouped. Avoid relying on redundant padding to make a two-step core appear difficult. Register a difficulty diagnostic before inspecting its result (for example fixed propagation baselines with canonical variable order and a documented treatment of unresolved cases); do not claim solver node bins alone establish learned difficulty.

Retain canonicalization, grouping, prompt/answer interfaces, held-out profile definitions, family splits, 0.25 scientific floors, 0.8 common-support requirement and independent verification. A new distribution is a new study, not an exact-equivalent continuation; any count/cell change requires an explicit prospective amendment and a matched new corpus across future controls. A complete source gate is followed by development support and then the separately specified full five-cohort G2 gate. No fixed-N EX/IDX implementation before G2; no training without its own complete registration.

Creative input is optional here. The controller can choose and test construction mechanisms within this scope. Owner input becomes useful if adequate variety/difficulty cannot coexist with the current grammar and a change to the research task or transfer claim is proposed.

## Prospective startup correction — 2026-10-11, before replacement launch

The first standalone invocation exited 1 at `from dwr.tasks.run_support import ResourceGuard` with `ModuleNotFoundError: No module named 'dwr'`. The unit runner placed the repository on Python's import path, while executing the ignored script directly placed its own directory there. The controller omitted a declared repository import path. Failure receipts/stdout are preserved as `.superpowers/sdd/constructive-design/controller-fixture-command.initial.json`, `controller-fixture-command.json` and `controller-fixture.txt`. The failure precedes guard construction, output-directory creation, initial manifest and every fixture calculation; it yields no mechanism observation. Do not label it a correctness NO-GO or completed check.

This is an explicit operational departure from the original single-invocation allowance, not an unrecorded retry. Authorize one replacement launch only, under distinct receipt identity `fixture-startup-fixed`, after a standalone import-only regression check and scoped launcher review. Set the child's `PYTHONPATH` explicitly to this repository root and record that setting in wrapper and initial manifest. Keep helper fixture definitions, algorithms, caps, enumeration order, scientific floors, protected-data restrictions and no-RNG identity unchanged. The diagnosed Python launch issue supplies no task outcomes for selection. Controller owns this routine environment decision under the owner's delegation; no new research task or transfer claim is chosen.

Limit launch diagnostics to the baseline absent-path import check and corrected-path import check (no fixture construction), then one replacement invocation with the original 120-second/1-GiB/one-worker core limits and separate 124-second external watchdog. If it fails, preserve the original and replacement outcomes and stop this design check without further launch allocation. Historical studies remain unchanged.

## Corrected fixture result — 2026-10-11

The root-owned replacement invocation completed with `MECHANISM_CHECK_PASS` on clean source commit `f1893b952077fb785904a3ce84e1940d8b2a776d`. The exact command was `python .superpowers/sdd/constructive-design/check_design.py --run`, launched with `PYTHONPATH` set to the worktree root, no RNG, one worker, a 120-second/1-GiB core cap and 124-second wrapper watchdog. The initial standalone invocation failed before fixture work with `ModuleNotFoundError: No module named dwr`; the two import-only diagnostics then showed the expected absent-path failure and corrected-path success. Exactly one replacement fixture invocation followed. All receipts and the initial failure are retained in ignored scratch evidence.

All twelve evaluated variants, including the six base fixtures, had one scalar and oracle solution, valid certificates and eligible outcomes. Filler-pool sizes for `EQ`, `NEQ`, `ADD`, `EQ/NEQ`, `EQ/ADD`, `NEQ/ADD` were `[1, 0, 1, 1, 0, 2]`; evaluated variants were `[2, 1, 2, 2, 1, 4]`. The twelve variants had twelve content IDs and twelve structure signatures. Every first-solution solver count was 3 nodes. The mixed-profile connectivity and present-kind participation checks passed. These are observations for this fixed n=4 fixture set only.

At the recorded post-terminal-write resource boundary, wall time was 0.125 seconds, CPU time 0.0625 seconds and sampled peak RSS 24,313,856 bytes. The small bounded metadata flush and stdout are outside that boundary; the wrapper observed 0.172 seconds. The root verified metadata and artifact hashes for the six run files, 17 producer identities, helper/fixture identities and receipts. Unit discovery on the same source commit passed 15 tests in 0.058 seconds.

The compact tracked record is [constructive-fixture-mechanism-result.json](../configs/splits/constructive-fixture-mechanism-result.json); it binds the source, exact command/environment/caps, helpers, all 17 producer hashes, startup attempts and all six run-artifact hashes. Full generated artifacts remain in the ignored run directory. No corpus rows were sampled, and no production code changed.

Interpretation remains narrow: twelve syntactic structures do not establish family variety, all first-solution counts being 3 does not establish difficulty, and the small conditional pools expose a capacity question rather than answering it. This pass does not establish production yield, n=6 capacity, any scientific floor or G2. The certificate principle may inform a broader six-variable constructor; do not promote this fixed motif library. The next step is prospective design and registration of an n=6 capacity and difficulty evaluation before any population run, retaining the original 0.25 source and accepted-envelope floors, 0.8 common-support floor, canonical family grouping and failed-study record.
