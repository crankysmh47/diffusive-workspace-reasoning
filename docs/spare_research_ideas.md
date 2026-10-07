# Spare research directions

Date: 2026-10-07. Status: parked ideas, not execution tasks. The active objective is [transferable computation with exchangeable workspaces](transfer_research_direction.md). Do not expand the active experiment matrix merely because an idea appears here. Activate an alternative through a dated decision with its question, prerequisites, budget and stop rule.

## Task structure as a predictor of useful latent computation

Question: can dependency depth and interacting-constraint structure predict whether joint, sequential or feedforward computation is the better use of a budget?

Candidate evidence: controlled interventions varying dependency structure while matching length/size; predictions evaluated on held-out families; causal perturbations of intermediate states. Ordinary attention plots or linear probes alone are insufficient. Graph-message-passing and recurrent-relational prior work must be reviewed; do not claim a depth theorem for unrestricted attention. Small diagnostics needed to interpret the main transfer study remain allowed; a separate benchmark/theory program is parked.

## Persistent workspaces for problem revision

Question: after a constraint changes, can the old workspace be repaired more cheaply than solving again?

Candidate evidence: end-to-end cost and correctness across edits, including changes that invalidate old conclusions. Compare restart, ordinary prompt/KV caching, and warm-started recurrent alternatives. Partition whole problem-edit lineages together. Count prompt re-encoding and repair policy costs. Warm-starting alone is not novel; the useful contribution would be reliable learned reuse/revision across task structures.

## Learned slot allocation

Question: can input-conditioned slot counts, or slots created/pruned during recurrence, improve the accuracy/cost frontier?

Prerequisites: fixed-N transfer and the separately registered variable-N study in the main direction. Compare a fixed large workspace, simple input-size rules and sampled-N training before adding a learned controller. Preserve permutation symmetry, log allocation overhead and train the selector without test answers. If a selector always chooses the maximum, there is no demonstrated resource-allocation benefit. See [AdaSlot](https://arxiv.org/abs/2406.09196) for close prior work in object-centric learning.

## Symmetry-aware flow distillation

Question: can a set-respecting student reproduce a validated recurrent teacher at lower end-to-end cost?

Prerequisites: interpretable recurrent transfer, a frozen teacher/interface, explicit endpoint correspondence and a direct one-shot distillation control. Investigate permutation handling only if it causes an observed problem; teacher-induced pairing already preserves correspondence. Neither flow matching nor permutation matching is novel on its own. Show benefit beyond simpler distillation, count target-generation/training cost, and do not infer diverse correct solutions from unique-answer tasks.

## Adaptive stopping

Question: can a halt policy spend fewer updates on easy inputs without damaging transfer?

Prerequisites: understood fixed-K behavior and a measurable cost opportunity. Compare fixed budgets and simple confidence/stability policies; charge all policy/verification costs. This is already an active research area, including [RecurTrace](https://arxiv.org/abs/2609.03379); adding a halting head alone is not a novelty argument.
