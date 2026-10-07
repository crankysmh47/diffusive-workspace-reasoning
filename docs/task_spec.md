# Finite-domain composition task, v1

Status: frozen CPU task contract for implementation and a bounded structural-suitability audit. This is a task-design specification, not a validated benchmark, a confirmatory study, or authorization to train. The audit thresholds below are prospective suitability floors; they do not establish power or a practical effect size.

## Question and scope

The task tests whether a model can complete small, uniquely solvable finite-domain constraint instances using a fixed set of known rules. The intended later transfer contrast is between source compositions and unseen combinations of the same primitives. It does not test unseen rule semantics, larger task size, cross-family transfer, or variable slot count. Every instance has 3–6 named variables over the fixed domain `{0,1,2}`. This is task-size variation for data diversity, not a registered size-transfer experiment.

The grammar uses exactly three relation kinds:

| `kind` | Arity and meaning | Canonical operand symmetry |
|---|---|---|
| `eq` | `value(a) == value(b)` | `(a,b)` is unordered |
| `neq` | `value(a) != value(b)` | `(a,b)` is unordered |
| `add_mod` | `(value(a) + value(b)) % 3 == value(c)` | first two operands are unordered; result `c` is distinguished |

Every constraint refers to distinct variables. In particular, `add_mod` has three distinct variables. There is no implicit all-different rule, inequality beyond explicit `neq`, arithmetic over integers outside modulo 3, or arbitrary value-label symmetry. Given assignments pin individual variables to values in `{0,1,2}`.

## Immutable Python contract

The first implementation uses only the standard library. Public immutable records are frozen, slotted dataclasses; collections are tuples and enums are string enums:

```python
class RelationKind(str, Enum): EQ = "eq"; NEQ = "neq"; ADD_MOD = "add_mod"
@dataclass(frozen=True, slots=True)
class Constraint: kind: RelationKind; variables: tuple[str, ...]
@dataclass(frozen=True, slots=True)
class GivenAssignment: variable: str; value: int
@dataclass(frozen=True, slots=True)
class ConstraintTask:
    schema_version: str; modulus: int; variables: tuple[str, ...]
    constraints: tuple[Constraint, ...]; givens: tuple[GivenAssignment, ...]
@dataclass(frozen=True, slots=True)
class Solution: assignments: tuple[tuple[str, int], ...]
@dataclass(frozen=True, slots=True)
class SolverStats:
    first_solution_nodes: int; uniqueness_nodes: int
    first_solution_count: int; uniqueness_solution_count_capped: int
@dataclass(frozen=True, slots=True)
class SolveResult: solution: Solution | None; stats: SolverStats
@dataclass(frozen=True, slots=True)
class OracleResult:
    solutions: tuple[Solution, ...]; assignments_examined: int
@dataclass(frozen=True, slots=True)
class GenerationRecord:
    task: ConstraintTask; content_id: str; structure_signature: str
    split: str; lineage_id: str; candidate_index: int
@dataclass(frozen=True, slots=True)
class VerificationResult: valid: bool; reason: str
```

`schema_version` is exactly `constraint-task-v1`, `modulus` is exactly integer `3` (Python booleans are rejected despite being integers in Python), and variables are exactly the ordered sequence `v0, …, v(n-1)` for a supported `n` from 3 through 6. A task has 2–8 constraints and 1–`n-2` givens, leaving at least two unassigned variables. Duplicate variable names, repeated givens, duplicate constraints after the operand symmetries above, unknown variables, wrong arity, repeated operands, unsupported sizes, out-of-domain values, and all unknown object keys are hard errors. JSON numbers must be integers, never booleans or floats. `parse_task_json` raises `TaskFormatError(code)` for invalid tasks; stable codes are `malformed_json`, `duplicate_json_key`, `unknown_key`, `unsupported_schema_version`, `unsupported_modulus`, `unsupported_size`, `invalid_variable_sequence`, `invalid_kind`, `invalid_arity`, `unknown_variable`, `repeated_operand`, `duplicate_constraint`, `duplicate_given`, `non_integer_value`, `value_out_of_domain`, `invalid_constraint_count`, and `invalid_given_count`.

## Strict serialization and model boundary

Task JSON has exactly these top-level keys and shapes:

```json
{"schema_version":"constraint-task-v1","modulus":3,"variables":["v0","v1","v2"],"constraints":[{"kind":"add_mod","variables":["v0","v1","v2"]},{"kind":"neq","variables":["v0","v1"]}],"givens":[{"variable":"v0","value":1}]}
```

The order of task constraints and givens has no semantic meaning. Parsing accepts either order, rejects duplicate JSON object keys (including duplicate assignment keys), rejects non-finite JSON constants, and rejects unknown keys at every level. `serialize_task(task)` emits UTF-8 JSON with no BOM, compact separators, lexicographically sorted object keys, constraints sorted by their normalized `(kind, operands)` encoding, and givens sorted by variable. The variable list stays in required `v0…v(n-1)` order. `parse_task_json(text) -> ConstraintTask` validates rather than repairing input. The parser and serializer must round-trip to one byte-stable canonical serialization.

`serialize_model_input(task) -> str` emits only the task rules, task variables, constraints, givens, and fixed output instruction/schema. It excludes content/structure IDs, split, lineage, candidate index, hidden witness, solution, solver traces/statistics, and audit-only fields. The instruction is: `Return one complete satisfying assignment as JSON matching the required answer schema.` The answer schema is exactly `{"assignment":{"v0":0,...}}`; all declared variables appear once and only once, in any JSON object key order, with integer values. The model input never includes a target answer or solver annotation.

The verifier API is `verify_answer(task: ConstraintTask, answer_text: str) -> VerificationResult`. Stable reason strings are `valid`, `malformed_json`, `duplicate_json_key`, `unknown_key`, `missing_assignment`, `assignment_not_object`, `unknown_variable`, `missing_variable`, `non_integer_value`, `value_out_of_domain`, `given_mismatch`, and `constraint_violation`. Check in this order: JSON syntax/constants/duplicate keys; top-level key set and presence of `assignment`; assignment object type; unknown then missing variables; value type; domain; given preservation; each relation. The first failing condition determines the reason. A syntactically valid answer with an unknown assignment key yields `unknown_variable`; any extra top-level or nested non-assignment key yields `unknown_key`.

## Solver and independent oracle

The later deterministic reference solver is `solve(task) -> SolveResult`. It searches in variable order `v0…v(n-1)`, values ascending `0,1,2`, and constraints in canonical serialized order. It uses direct partial-consistency checks for each relation and depth-first backtracking. It first records the first solution and its node count, then continues until exhaustion or a second solution, recording uniqueness-search nodes and a solution count capped at 2. First-solution work and uniqueness-proof work remain separate; difficulty comparisons use the first-solution count, not the uniqueness tail.

The independent oracle is `enumerate_solutions(task) -> OracleResult`. It visits the full Cartesian product in lexicographic assignment order (at most `3^6 = 729` assignments), filters givens, then evaluates each primitive formula inline from its mathematical definition. It must not call the solver's partial-consistency helpers, verifier predicates, or generator acceptance predicates. It returns every satisfying solution, not a count capped at two. Solver/verifier/oracle agreement is required before generation is audited.

## Canonical identities and split assignment

`canonical_content(task) -> str` considers all `n!` variable renamings, rewrites variables to the canonical ordered names, applies only the actual operand symmetries listed above, sorts constraints/givens, serializes canonically, and returns the lexicographically smallest representation. The content ID is `sha256(UTF8("dwr-content-v1\\0" + canonical_content))`. Thus variable renaming and constraint/given order do not change the ID. No value relabeling is performed.

`canonical_structure(task) -> str` uses the same exhaustive variable-renaming and relation normalization, but omits all given values while retaining which variables are anchors (the given-variable set after renaming), variable count, and every relation kind and operand position. It retains multiplicity of constraints. The structural signature is `sha256(UTF8("dwr-structure-v1\\0" + canonical_structure))`. This intentionally groups tasks that share a constraint structure and anchor locations despite differing clue values. `n <= 6` makes exhaustive canonicalization bounded and avoids a general graph-canonicalization framework.

The composition partition is fixed before generation and any accuracy inspection, using the set of relation kinds present in a task:

| Partition | Allowed relation-kind set | Use |
|---|---|---|
| Source | exactly one of `{eq}`, `{neq}`, `{add_mod}` | training plus source ID train/validation/test |
| Development composition targets | exactly one of `{eq,neq}`, `{eq,add_mod}`, `{neq,add_mod}` | development-only OOD validation and diagnostic test |
| Final composition targets | exactly `{eq,neq,add_mod}` | untouched final OOD test only |

All instances in a partition, all structural signatures, and all lineage descendants remain in that partition. Source ID assignment is by canonical content/lineage group, with a stable hash bucket of 80% train, 10% validation, and 10% test; this intentionally permits familiar source structures across ID splits while preventing exact content and lineage overlap. Development composition signatures are all reserved from model training and used only for choices; a separately designated development diagnostic test is also non-confirmatory. Final composition signatures are generated only for structural audit and final test after the spec/implementation and analysis are frozen. They cannot be used for model, prompt, threshold, or hyperparameter choices. A final target inspected during the suitability audit remains an untouched target for model choices; no accuracy/model comparison is run at this gate.

This split tests composition reuse at familiar `n` values. It is not random-text splitting, and a source ID split is not a composition holdout. No claim of size/depth, input-symmetry, slot-permutation, or cross-family transfer follows from it.

## Seeded generation and audit contract

The config fixes root seed `20261007`. Each partition/candidate gets an independent stream seed: `int.from_bytes(sha256(b"dwr-generator-v1\\0" + root_seed_as_8_byte_big_endian + b"\\0" + partition_ascii + b"\\0" + candidate_index_as_8_byte_big_endian).digest()[:8], "big")`. Initialize a local `random.Random(stream_seed, version=2)`; never use module-global RNG. Python runtime version and config hash are recorded because seeded generation is not asserted cross-version bitwise invariant. Candidate indices are monotonic from zero independently within each partition.

For each candidate, call the local RNG in this order: choose `n` uniformly from 3–6; choose one allowed relation-kind profile uniformly from that partition; build each kind's eligible normalized operand tuples in lexicographic variable-index order; choose uniformly a feasible total constraint count from 2 through 8; choose uniformly one positive per-kind count vector with that total and no count above its eligible tuple capacity; use `random.sample` independently per kind to choose distinct constraints, then sort them canonically; draw each witness variable uniformly from `{0,1,2}` in variable order; reject if any relation fails on the witness; choose a clue count uniformly from 1 through `n-2`, then `random.sample` that many variables from variable order and sort them before assigning witness values. Parse/validate the candidate, enumerate all assignments with the independent oracle, and accept only exactly-one-solution tasks. The witness, solution and annotations are persisted only in generation/audit records outside model input. Rejections consume that candidate index; accepted output order is candidate-index order. A duplicate canonical content ID is recorded as a duplicate and does not add an output group; its content group and any descendants inherit the same split. The base lineage ID is `sha256(UTF8("dwr-lineage-v1\\0" + content_id))`. For each source content group, assign the ID bucket by `int.from_bytes(sha256(b"dwr-id-split-v1\\0" + content_id_ascii).digest()[:8], "big") % 100`: 0–79 train, 80–89 validation, 90–99 test. Stop each partition at 300 accepted unique-content groups or the 30,000-attempt cap; never silently lower a floor or change the grammar.

The audit writes a small resolved manifest before work: exact command, config SHA-256, source commit/dirty state, Python version, seed derivation, output paths, per-partition attempt counts, wall/CPU time and resource limits. Each accepted task record stores canonical IDs, partition, lineage ID, candidate index, exact task serialization, oracle solution, and first-solution/uniqueness stats. The model-input projection is separately serialized and hashed. Generated datasets and per-example records stay outside Git; only config and aggregate audit report/hash manifests are eligible for version control.

## Prospective suitability gate (not a power calculation)

The configured first audit is bounded to at most 30,000 candidates per partition (90,000 total), 600 seconds wall time, one CPU worker, and 1 GiB resident memory. Stop immediately at the first cap; preserve partial output and mark the audit incomplete. Do not retry under altered rules within this task. Per partition, the structural gate requires at least 100 accepted unique-content groups and 12 distinct structural signatures; source additionally needs at least 100/20/20 train/validation/test groups and at least 12 source signatures in training. Development and final composition targets each need at least 100 groups and 12 signatures. Every primitive must occur in source and every target relation-kind set must match its registered partition.

All accepted tasks must have exactly one oracle solution, at least two unassigned variables, and pass deterministic regeneration on a replay audit. Across each partition, each value `0,1,2` must account for at least 10% of all solution variable-values; at least two different clue counts must occur. Every canonical content ID and every lineage ID must occur in exactly one split, model-input bytes must be invariant to changes in audit-only labels/answers, and model-input fields must contain none of those labels/answers. These are explicit leakage and useful-variation checks, not claims about pretrained-model contamination.

Genuine constraint interaction is measured by a counterfactual: remove each constraint in turn while holding givens fixed and count satisfying assignments with the independent oracle. An instance is interaction-sensitive if removing at least one constraint changes the count from exactly one to at least two. Within source, development, and final partitions, at least 25% of accepted groups must be interaction-sensitive; each held-out structural signature must include at least one such instance. Report the full removal-count distribution. This blocks a dataset made only of clues that already determine the answer. The thresholds are floors for structural suitability and may fail; preserve a no-go outcome instead of relaxing them.

Passing this audit is at most a structural G2 suitability result. It does not set a meaningful EX–IDX effect size, establish prospective precision/power, authorize comparative training, or validate a model. A small structural audit may pass while confirmatory precision remains unresolved. Comparative training remains blocked until development-only data, feasibility probes, and a separate study registration provide the missing effect, sample, seed, compute, checkpoint, and stop settings required by v2.2.

## Module map and next checks

Implement modules only when needed, under `dwr/tasks/`: `schema.py` owns records, parsing and canonical task serialization; `verifier.py` owns answer parsing and reason codes; `oracle.py` owns independent exhaustive enumeration; `solver.py` owns deterministic partial-check DFS and statistics; `canonical.py` owns content/structure canonicalization; `generator.py` owns seeded candidate generation and record output; `audit.py` owns manifests, replay/leakage/interaction/variation checks and aggregate suitability reporting. Do not scaffold empty modules. The intended commands are `python -m dwr.tasks.generator --config configs/tasks/constraint_v1.json --out data/constraint-v1` and `python -m dwr.tasks.audit --config configs/tasks/constraint_v1.json --data data/constraint-v1`; both must fail closed on unknown config keys and invalid records.

Before model components, add `unittest` checks for strict unknown/duplicate-key rejection, bool/float/domain/shape errors, each primitive truth table, given preservation, every verifier reason, zero/one/multiple solution counts, solver/oracle agreement, clue preservation, independent oracle implementation, variable-renaming and constraint-order ID invariance, structure identity with values omitted but anchors retained, no value-relabel assumption, deterministic regeneration, lineage/content split closure, model-input annotation exclusion, interaction counterfactuals, and fail-at-cap behavior. Test G2 reporting with both a passing fixture and a fixture that remains a no-go. These correctness checks do not require GPU training; none is implemented or claimed by this specification.
