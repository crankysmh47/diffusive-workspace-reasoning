"""Closed CPU-only mechanism fixture for certificate sampler v2."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .canonical import canonical_content, canonical_structure, content_id, development_split, structure_signature
from .certificate_generator_v2 import construct
from .certificate_generator import CertificateOutcome, WorkCounters
from .local_propagation import propagate
from .oracle import enumerate_solutions
from .run_support import ResourceGuard, resident_bytes_windows, stable_json, strict_json_loads
from .schema import Constraint, ConstraintTask, RelationKind, Solution, serialize_model_input, serialize_task
from .solver import solve
from .verifier import verify_answer


CONFIG_SHA256 = "0543c10c6028758c8c36cd4f24f295fb9a11b3801c0bb53444e74151f450418b"
ARTIFACTS = ("manifest.initial.json", "outcomes.jsonl", "replay.json", "report.json", "manifest.json")
PROFILES = ("eq/neq", "eq/add_mod", "neq/add_mod")
CELLS = ("A", "B", "C", "D")
SHAPES = {"A": (2, 8), "B": (3, 6), "C": (3, 7), "D": (3, 8)}


class FixtureError(RuntimeError):
    pass


def _json_bytes(value: Any) -> bytes:
    return (stable_json(value) + "\n").encode("utf-8")


def _atomic(path: Path, value: Any, check: Callable[[], None]) -> None:
    check()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(_json_bytes(value))
    tmp.replace(path)
    check()


def _hash(path: Path, check: Callable[[], None]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            check()
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    check()
    return digest.hexdigest()


def _source_identity(check: Callable[[], None]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    check()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True, encoding="utf-8").stdout.strip()
    check()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
                            check=True, capture_output=True, text=True, encoding="utf-8").stdout
    check()
    if status.strip():
        raise FixtureError("production source must be clean and committed")
    files = sorted((root / "dwr" / "tasks").glob("*.py")) + [root / "dwr" / "__init__.py"]
    producers = {}
    for path in files:
        check()
        producers[path.relative_to(root).as_posix()] = _hash(path, check)
    return {"git_commit": commit, "git_clean": True, "producer_sha256": producers}


def _load_config(path: str | Path) -> tuple[dict[str, Any], str, Path]:
    config_path = Path(path).resolve(strict=True)
    raw = config_path.read_bytes()
    expected_path = Path(__file__).resolve().parents[2] / "configs" / "tasks" / "certificate_sampler_v2_fixture.json"
    expected = expected_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if hashlib.sha256(expected).hexdigest() != CONFIG_SHA256 or digest != CONFIG_SHA256:
        raise FixtureError("configuration bytes differ from the closed fixture registration")
    value = strict_json_loads(raw.decode("utf-8"))
    baseline = strict_json_loads(expected.decode("utf-8"))
    if type(value) is not dict or stable_json(value) != stable_json(baseline):
        raise FixtureError("configuration differs from the exact registered schema")
    return value, digest, config_path


def _revoke_terminal(output: Path, hashes: dict[str, str], reason: str,
                     metadata: dict[str, Any] | None = None) -> None:
    """Revoke any previously saved PASS using only bounded small-file writes."""
    preserved: dict[str, Any] = {}
    try:
        old = strict_json_loads((output / "report.json").read_text(encoding="utf-8"))
        preserved = {key: old[key] for key in
                     ("attempts", "attempts_persisted", "elapsed_seconds", "cpu_seconds", "peak_rss_bytes")
                     if key in old and type(old[key]) in (int, float)}
    except Exception:
        pass
    hashes = dict(hashes)
    hashes.pop("report.json", None)
    report = preserved | (metadata or {}) | {"status": "NOT_ASSESSED", "terminal_reason": reason,
              "g2_status": "not_assessed", "preflight_emitted": False}
    (output / "report.json").write_bytes(_json_bytes(report))
    manifest = {"status": "NOT_ASSESSED", "artifacts": hashes,
                "g2_status": "not_assessed", "preflight_emitted": False,
                "terminal_reason": reason}
    (output / "manifest.json").write_bytes(_json_bytes(manifest))


def _scalar_count(task: ConstraintTask, constraints: tuple[Constraint, ...] | None = None,
                  check: Callable[[], None] = lambda: None) -> int:
    fixed = {given.variable: given.value for given in task.givens}
    unknown = tuple(name for name in task.variables if name not in fixed)
    selected = task.constraints if constraints is None else constraints
    count = 0
    for values in itertools.product((0, 1, 2), repeat=len(unknown)):
        check()
        assignment = fixed | dict(zip(unknown, values))
        ok = True
        for rule in selected:
            a = tuple(assignment[name] for name in rule.variables)
            if rule.kind is RelationKind.EQ:
                holds = a[0] == a[1]
            elif rule.kind is RelationKind.NEQ:
                holds = a[0] != a[1]
            else:
                holds = (a[0] + a[1]) % 3 == a[2]
            if not holds:
                ok = False
                break
        count += int(ok)
    return count


def _graph(task: ConstraintTask) -> dict[str, Any]:
    fixed = {given.variable for given in task.givens}
    unknown = {name for name in task.variables if name not in fixed}
    links = {name: set() for name in unknown}
    participating: set[str] = set()
    for rule in task.constraints:
        scope = [name for name in rule.variables if name in unknown]
        if scope:
            participating.add(rule.kind.value)
            for a in scope:
                links[a].update(b for b in scope if b != a)
    reached: set[str] = set()
    stack = list(unknown)[:1]
    while stack:
        item = stack.pop()
        if item not in reached:
            reached.add(item)
            stack.extend(links[item] - reached)
    kinds = {rule.kind.value for rule in task.constraints}
    return {"connected": bool(unknown) and reached == unknown,
            "participating_kinds": sorted(participating),
            "all_kinds_participate": kinds <= participating}


def _eligible(task: ConstraintTask, check: Callable[[], None]) -> tuple[bool, list[int], dict[str, Any]]:
    counts = []
    for kind in sorted({rule.kind.value for rule in task.constraints}):
        retained = tuple(rule for rule in task.constraints if rule.kind.value != kind)
        counts.append(_scalar_count(task, retained, check))
    graph = _graph(task)
    return all(count >= 2 for count in counts) and graph["connected"] and graph["all_kinds_participate"], counts, graph


def _assert_no_unary_gac(task: ConstraintTask, check: Callable[[], None]) -> dict[str, Any]:
    fixed = {g.variable: g.value for g in task.givens}
    unknown = set(task.variables) - set(fixed)
    if any(sum(name in unknown for name in rule.variables) == 1 for rule in task.constraints):
        raise FixtureError("no-unary branch contains an exactly-one-unknown constraint")
    propagation = propagate(task, check_budget=check)
    expected = tuple(frozenset((0, 1, 2)) if name in unknown else frozenset((fixed[name],))
                     for name in task.variables)
    if (propagation.domains != expected or propagation.contradiction or not propagation.stalled or
            propagation.unresolved_unknowns != len(unknown)):
        raise FixtureError("no-unary task fails actual production GAC full-domain stall")
    return {"domains": [sorted(d) for d in propagation.domains], "stalled": propagation.stalled,
            "contradiction": propagation.contradiction, "unresolved_unknowns": propagation.unresolved_unknowns}


def _audit_certificate(task: ConstraintTask, witness: Solution,
                       alternative_items: tuple[tuple[str, Solution], ...],
                       reserved_items: tuple[Constraint, ...], expected_profile: tuple[str, ...]) -> None:
    values = dict(witness.assignments)
    givens = {g.variable: g.value for g in task.givens}
    if witness.assignments != tuple((name, values[name]) for name in task.variables):
        raise FixtureError("witness variables are not in canonical task order")
    if any(values.get(name) != value for name, value in givens.items()):
        raise FixtureError("witness changes a given")
    if not all(_holds(rule, values) for rule in task.constraints):
        raise FixtureError("constructor witness violates a selected constraint")
    alternatives = dict(alternative_items)
    reserved = {rule.kind.value: rule for rule in reserved_items}
    if (tuple(kind for kind, _ in alternative_items) != expected_profile or
            set(alternatives) != set(expected_profile) or set(reserved) != set(expected_profile)):
        raise FixtureError("reserved certificate does not cover the profile")
    for kind, alternative in alternative_items:
        candidate = dict(alternative.assignments)
        if (alternative == witness or alternative.assignments != tuple((name, candidate[name]) for name in task.variables) or
                any(candidate.get(name) != value for name, value in givens.items())):
            raise FixtureError("certificate alternative is malformed or changes a given")
        protected = reserved[kind]
        if protected not in task.constraints or not _holds(protected, values) or _holds(protected, candidate):
            raise FixtureError("reserved certificate does not reject its own alternative")
        for other_kind, other_alt in alternative_items:
            other_values = dict(other_alt.assignments)
            if other_kind != kind and not all(_holds(rule, other_values)
                                              for rule in task.constraints if rule.kind.value == kind):
                raise FixtureError("mixed certificate alternative fails the other kind's constraints")


def _core(task: ConstraintTask, check: Callable[[], None]) -> dict[str, Any]:
    # Parse the canonical full valued task before the single deletion pass.
    from .schema import parse_task_json
    current = parse_task_json(canonical_content(task))
    removed_padding = 0
    for rule in tuple(current.constraints):
        check()
        candidate = tuple(item for item in current.constraints if item != rule)
        if _scalar_count(current, candidate, check) == 1:
            current = ConstraintTask(current.schema_version, current.modulus, current.variables,
                                     candidate, current.givens)
            removed_padding += 1
    if _scalar_count(current, check=check) != 1:
        raise FixtureError("greedy core lost uniqueness")
    for index, rule in enumerate(current.constraints):
        check()
        if _scalar_count(current, current.constraints[:index] + current.constraints[index + 1:], check) == 1:
            raise FixtureError("core contains a removable constraint")
    if {rule.kind for rule in current.constraints} != {rule.kind for rule in task.constraints}:
        raise FixtureError("core removed a required constraint kind")
    graph = _graph(current)
    expected_size = len(current.variables) - len(current.givens)
    eqadd_basis = {r.kind.value for r in current.constraints} == {"eq", "add_mod"}
    if eqadd_basis and len(current.constraints) != expected_size:
        raise FixtureError("EQ/ADD minimal core violates the F3 basis-size invariant")
    return {"task": serialize_task(current), "signature": structure_signature(current),
            "size": len(current.constraints), "given_count": len(current.givens),
            "removed_constraints": removed_padding, "connected": graph["connected"],
            "structure_claim": "syntactic_signature_not_reasoning_mechanism"}


def _verify_candidate(outcome: Any, index: int, check: Callable[[], None]) -> dict[str, Any]:
    if type(outcome) is not CertificateOutcome:
        raise FixtureError("constructor returned an incompatible outcome type")
    task = outcome.task
    profile = "/".join(outcome.profile)
    base = {"candidate_index": index, "profile": profile, "cell": outcome.cell,
            "branch": ("ordinary" if "add_mod" in outcome.profile and (index // 12) % 2 == 0 else
                       "no_unary" if "add_mod" in outcome.profile else None),
            "rejection_reason": outcome.rejection_reason, "task": None,
            "content_id": None, "structure_signature": None, "split": None,
            "oracle_unique": False, "eligible": False, "first_seen_content": False,
            "choice_permitted": False, "core": None}
    expected_profile = PROFILES[(index // 4) % len(PROFILES)]
    expected_cell = CELLS[index % 4]
    if (outcome.partition != "development" or outcome.candidate_index != index or
            profile != expected_profile or outcome.cell != expected_cell or outcome.root_seed != 20261012):
        raise FixtureError("constructor changed the registered attempt identity/schedule")
    work_fields = WorkCounters.__dataclass_fields__
    if (type(outcome.work) is not WorkCounters or
            any(type(getattr(outcome.work, key)) is not int or getattr(outcome.work, key) < 0
                for key in work_fields)):
        raise FixtureError("constructor work counters have an invalid schema or value")
    if task is None:
        if outcome.rejection_reason not in ("empty_witness_pool", "no_feasible_mixed_second_alternative"):
            raise FixtureError("null-task outcome has an unregistered failure reason")
        if outcome.reserved_constraints:
            raise FixtureError("null-task construction rejection contains reserved constraints")
        if outcome.rejection_reason == "empty_witness_pool":
            if (outcome.partition != "source" or outcome.witness is not None or outcome.givens or
                    outcome.alternatives or outcome.work.rng_calls != 1 or
                    outcome.work.given_position_scans != 6 or outcome.work.witness_pool_scans != 3**6):
                raise FixtureError("empty-pool rejection metadata is inconsistent")
            return base | {"rejection_reason": outcome.rejection_reason, "state": "construction_failed",
                           "work": {key: getattr(outcome.work, key) for key in work_fields}}
        if (outcome.partition != "development" or outcome.witness is None or
                len(outcome.alternatives) != 1 or outcome.alternatives[0][0] != outcome.profile[0]):
            raise FixtureError("mixed-pool rejection metadata is inconsistent")
        values = dict(outcome.witness.assignments)
        given_map = {given.variable: given.value for given in outcome.givens}
        first = outcome.alternatives[0][1]
        first_values = dict(first.assignments)
        if (outcome.witness.assignments != tuple((f"v{i}", values[f"v{i}"]) for i in range(6)) or
                len(outcome.givens) != SHAPES[outcome.cell][0] or
                any(values.get(name) != value for name, value in given_map.items()) or
                first == outcome.witness or
                any(first_values.get(name) != value for name, value in given_map.items()) or
                outcome.work.feasible_second_alternatives != 0):
            raise FixtureError("mixed-pool rejection partial certificate is malformed")
        base["certificate"] = {"givens": [[g.variable, g.value] for g in outcome.givens],
                               "witness": [[name, value] for name, value in outcome.witness.assignments],
                               "alternatives": {outcome.alternatives[0][0]:
                                                [[name, value] for name, value in first.assignments],
                                                },
                               "reserved_constraints": [],
                               "work": {key: getattr(outcome.work, key) for key in work_fields}}
        return base | {"rejection_reason": outcome.rejection_reason, "state": "construction_failed"}
    if (len(task.variables) != 6 or len(task.givens) != SHAPES[outcome.cell][0] or
            len(task.constraints) != SHAPES[outcome.cell][1] or
            {rule.kind.value for rule in task.constraints} != set(expected_profile.split("/")) or
            outcome.rejection_reason is not None):
        raise FixtureError("constructed task changed its registered shape")
    base["task"] = serialize_task(task)
    cid, sig = content_id(task), structure_signature(task)
    split = development_split(sig)
    base.update(content_id=cid, structure_signature=sig, split=split)
    base["certificate"] = {
        "givens": [[g.variable, g.value] for g in outcome.givens],
        "witness": [[name, value] for name, value in outcome.witness.assignments],
        "alternatives": {kind: [[name, value] for name, value in solution.assignments]
                         for kind, solution in outcome.alternatives},
        "reserved_constraints": [{"kind": rule.kind.value, "variables": list(rule.variables)}
                                 for rule in outcome.reserved_constraints],
        "work": {name: getattr(outcome.work, name) for name in outcome.work.__dataclass_fields__},
    }
    check()
    # Independent inline scalar equations over just the unknown assignment product.
    fixed = {given.variable: given.value for given in task.givens}
    unknown = tuple(name for name in task.variables if name not in fixed)
    scalar_solutions = []
    for assignment_values in itertools.product((0, 1, 2), repeat=len(unknown)):
        check()
        assignment = fixed | dict(zip(unknown, assignment_values))
        if all(_holds(rule, assignment) for rule in task.constraints):
            scalar_solutions.append(Solution(tuple((name, assignment[name]) for name in task.variables)))
    count = len(scalar_solutions)
    oracle = enumerate_solutions(task)
    if tuple(oracle.solutions) != tuple(scalar_solutions):
        raise FixtureError("independent scalar oracle disagrees with production exhaustive oracle")
    solved = solve(task)
    expected_first = oracle.solutions[0] if oracle.solutions else None
    if (solved.solution != expected_first or
            solved.stats.uniqueness_solution_count_capped != min(2, count)):
        raise FixtureError("solver first solution disagrees with oracle ordering")
    for solution in oracle.solutions:
        assignment = dict(solution.assignments)
        if any(assignment.get(g.variable) != g.value for g in task.givens):
            raise FixtureError("production oracle returned an assignment that changes a given")
        if any(not _holds(rule, assignment) for rule in task.constraints):
            raise FixtureError("production oracle returned a nonsatisfying assignment")
        payload = {"assignment": assignment}
        if not verify_answer(task, json.dumps(payload, separators=(",", ":"))).valid:
            raise FixtureError("production verifier rejected a production oracle solution")
    if outcome.witness not in oracle.solutions:
        raise FixtureError("constructor witness absent from independently checked solutions")
    alternatives = dict(outcome.alternatives)
    _audit_certificate(task, outcome.witness, outcome.alternatives,
                       outcome.reserved_constraints, tuple(expected_profile.split("/")))
    values = dict(outcome.witness.assignments)
    givens = {g.variable: g.value for g in task.givens}
    if not verify_answer(task, json.dumps({"assignment": values}, separators=(",", ":"))).valid:
        raise FixtureError("production verifier rejected constructor witness")
    eligible, removals, graph = _eligible(task, check)
    unique = count == 1
    base.update(oracle_unique=unique, eligible=unique and eligible,
                solution_count=count, kind_removal_counts=removals, graph=graph,
                state="unique_eligible" if unique and eligible else "non_unique" if not unique else "ineligible")
    if profile == "eq/neq":
        witness = values
        first = dict(alternatives["eq"].assignments)
        has_all_colors = set(witness.values()) == {0, 1, 2}
        distinct_givens = len({witness[k] for k in givens}) >= 2
        rejects_eq = any(_holds(Constraint(RelationKind.EQ, pair), witness) and
                         not _holds(Constraint(RelationKind.EQ, pair), first)
                         for pair in itertools.combinations(task.variables, 2))
        if not (has_all_colors and distinct_givens and rejects_eq):
            raise FixtureError("EQ/NEQ necessary witness or first-alternative filter failed")
    if base["branch"] == "no_unary":
        base["gac"] = _assert_no_unary_gac(task, check)
    base["prompt_bytes"] = len(serialize_model_input(task).encode("utf-8"))
    base["first_solution_nodes"] = solved.stats.first_solution_nodes
    base["envelope_descriptive"] = (base["prompt_bytes"] > 1023 and
                                    4 <= solved.stats.first_solution_nodes <= 7)
    return base


def _holds(rule: Constraint, values: dict[str, int]) -> bool:
    args = tuple(values[name] for name in rule.variables)
    if rule.kind is RelationKind.EQ:
        return args[0] == args[1]
    if rule.kind is RelationKind.NEQ:
        return args[0] != args[1]
    return (args[0] + args[1]) % 3 == args[2]


def _own_row(row: dict[str, Any], seen: dict[str, str], check: Callable[[], None]) -> None:
    """Assign content ownership before split-gated counts, examples, or core choices."""
    check()
    if row.get("task") is None:
        return
    cid = row["content_id"]
    split = row.get("split")
    if cid in seen and seen[cid] != split:
        raise FixtureError("one canonical content identity maps to multiple structure splits")
    first = cid not in seen
    seen[cid] = split
    row["first_seen_content"] = first
    row["choice_permitted"] = first and row.get("split") == "development_validation"
    if (row["choice_permitted"] and row.get("oracle_unique") and row.get("eligible") and
            row.get("branch") == "no_unary" and row.get("gac", {}).get("stalled") is True):
        from .schema import parse_task_json
        row["core"] = _core(parse_task_json(row["task"]), check)


def _choices(rows: list[dict[str, Any]], check: Callable[[], None] = lambda: None) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    for row in rows:
        check()
        if (row.get("choice_permitted") is True and row.get("split") == "development_validation" and
                row.get("oracle_unique") is True and row.get("eligible") is True):
            eligible.append(row)
    eqneq = Counter(r["cell"] for r in eligible if r["profile"] == "eq/neq")
    hard = Counter((r["profile"], r["cell"]) for r in eligible
                   if r["branch"] == "no_unary" and r.get("gac", {}).get("stalled") is True)
    signatures: dict[str, dict[str, set[str]]] = {p: {"2": set(), "3": set()} for p in PROFILES[1:]}
    for row in eligible:
        core = row.get("core")
        if core:
            signatures[row["profile"]][str(core["given_count"])].add(core["signature"])
    gate = {"eqneq_each_cell": all(eqneq[cell] >= 1 for cell in CELLS),
            "no_unary_each_add_profile_cell": all(hard[(p, cell)] >= 1 for p in PROFILES[1:] for cell in CELLS),
            "core_signatures_each_add_profile_given_count": all(
                len(signatures[p][g]) >= 2 for p in PROFILES[1:] for g in ("2", "3"))}
    envelope_count = sum(r.get("envelope_descriptive") is True for r in eligible)
    return {"permitted_unique_eligible_count": len(eligible), "descriptive_envelope_count": envelope_count,
            "eqneq_eligible_by_cell": dict(eqneq),
            "no_unary_eligible_by_profile_cell": {f"{p}/{c}": hard[(p, c)] for p in PROFILES[1:] for c in CELLS},
            "core_signatures_by_profile_given_count": {p: {g: len(signatures[p][g]) for g in ("2", "3")} for p in PROFILES[1:]},
            "floors": gate, "passed": all(gate.values())}


def _replay(output: Path, expected_rows: list[dict[str, Any]], check: Callable[[], None],
            outcome_factory: Callable[..., Any], root_seed: int) -> dict[str, Any]:
    saved = []
    with (output / "outcomes.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            check()
            saved.append(json.loads(line))
    if len(saved) != 288 or len(expected_rows) != 288:
        raise FixtureError("saved outcome prefix is not the fixed 288-attempt prefix")
    regenerated = []
    seen: dict[str, str] = {}
    for index in range(288):
        check()
        try:
            outcome = outcome_factory(root_seed, "development", index, check_budget=check)
            row = _verify_candidate(outcome, index, check)
        except Exception as exc:
            raise FixtureError(f"replay constructor/audit failure at index {index}: {type(exc).__name__}") from exc
        _own_row(row, seen, check)
        regenerated.append(row)
    for index, (actual, expected) in enumerate(zip(saved, regenerated)):
        if actual != expected_rows[index] or actual != expected or actual.get("candidate_index") != index:
            raise FixtureError("persisted attempt differs from deterministic constructor replay")
    choice = _choices(saved, check)
    return {"status": "PASS_REPLAY", "attempts_rederived": len(saved),
            "allowed_aggregate_recomputed": True, "allowed_core_linkage_recomputed": True,
            "permitted_choice_digest": hashlib.sha256(_json_bytes(choice)).hexdigest()}


def run_fixture(config_path: str | Path, output_path: str | Path, *, argv: list[str] | None = None,
                guard: Any | None = None, outcome_factory: Callable[..., Any] = construct) -> dict[str, Any]:
    invocation_start = time.monotonic()
    cpu_start = time.process_time()
    if guard is None:
        guard = ResourceGuard(180, 512, memory_reader=resident_bytes_windows)
    guard.check()
    config, cfg_hash, cfg_path = _load_config(config_path)
    limits = config["limits"]
    if limits["active_wall_seconds"] != 180 or limits["max_rss_bytes"] != 536870912 or limits["workers"] != 1:
        raise FixtureError("closed runtime budgets differ from the code-known limits")
    check = guard.check
    requested = Path(output_path).absolute()
    output = requested.resolve()
    root = Path(__file__).resolve().parents[2]
    try:
        output.relative_to((root / "runs").resolve())
    except ValueError as exc:
        raise FixtureError("output must be under repository runs/") from exc
    if not output.name.startswith("certificate-sampler-v2-fixture-"):
        raise FixtureError("output directory name must use certificate-sampler-v2-fixture-* prefix")
    for parent in (requested, *requested.parents):
        try:
            info = parent.lstat()
        except FileNotFoundError:
            continue
        attributes = getattr(info, "st_file_attributes", 0)
        if parent.is_symlink() or attributes & 0x400:
            raise FixtureError("output path contains a symlink or reparse point")
    if output.exists():
        raise FileExistsError(f"refusing existing output path: {output}")
    # Source identity and guard admission precede creation; once created, preserve state on every failure.
    source = _source_identity(check)
    check()
    initial = {"manifest_version": "dwr-certificate-sampler-v2-initial-v1", "status": "running",
               "created_utc": datetime.now(timezone.utc).isoformat(), "command": list(argv or sys.orig_argv),
               "cwd": str(Path.cwd()), "executable": sys.executable, "python": sys.version,
               "implementation": platform.python_implementation(), "python_version": list(sys.version_info[:3]),
               "platform": platform.platform(), "environment": {k: os.environ.get(k) for k in
               ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}, "output_dir": str(output),
               "config_path": str(cfg_path), "config_sha256": cfg_hash, "resolved_config": config,
               "source": source, "rng": config["rng"], "independent_rng_streams": ["development/index"],
               "precision": "standard-library CPU", "workers": 1, "budgets": limits,
               "artifact_names": list(ARTIFACTS)}
    rows: list[dict[str, Any]] = []
    terminal_reason = None
    start = invocation_start
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "manifest.initial.json").write_bytes(_json_bytes(initial))
        check()
        if tuple(sys.version_info[:2]) not in ((3, 11), (3, 12)) or platform.python_implementation() != "CPython":
            raise FixtureError("registered runtime admission failed")
        seen_content: dict[str, str] = {}
        with (output / "outcomes.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for index in range(288):
                check()
                outcome = None
                row = None
                phase = "construction"
                try:
                    outcome = outcome_factory(config["root_seed"], "development", index, check_budget=check)
                    phase = "independent_integrity_checks"
                    row = _verify_candidate(outcome, index, check)
                    phase = "content_ownership_and_core"
                    _own_row(row, seen_content, check)
                except Exception as exc:
                    if row is None:
                        row = {"candidate_index": index, "profile": PROFILES[(index // 4) % 3],
                               "cell": CELLS[index % 4], "branch": None, "task": None,
                               "split": None, "content_id": None, "structure_signature": None,
                               "first_seen_content": False, "choice_permitted": False}
                    row.update(state="fatal_error", failure=type(exc).__name__, failure_phase=phase)
                    if outcome is not None and row.get("task") is None:
                        outcome_profile = getattr(outcome, "profile", PROFILES[(index // 4) % len(PROFILES)])
                        has_add = type(outcome_profile) is tuple and "add_mod" in outcome_profile
                        row["branch"] = ("ordinary" if has_add and (index // 12) % 2 == 0 else
                                         "no_unary" if has_add else None)
                        row["rejection_reason"] = getattr(outcome, "rejection_reason", None)
                        failed_task = getattr(outcome, "task", None)
                        if failed_task is not None:
                            try:
                                row["task"] = serialize_task(failed_task)
                                row["certificate"] = {"givens": [[g.variable, g.value] for g in outcome.givens],
                                                       "witness": ([[n, v] for n, v in outcome.witness.assignments]
                                                                   if outcome.witness is not None else None),
                                                       "alternatives": {kind: [[n, v] for n, v in solution.assignments]
                                                                        for kind, solution in outcome.alternatives}}
                            except Exception:
                                row["task"] = None
                                row["certificate_integrity"] = "unavailable_after_fatal_check"
                    rows.append(row)
                    stream.write(stable_json(row) + "\n"); stream.flush()
                    raise
                # Content ownership, permitted routing, and any permitted core are settled before persistence.
                rows.append(row)
                stream.write(stable_json(row) + "\n"); stream.flush()
        choice = _choices(rows, check)
        replay = _replay(output, rows, check, outcome_factory, config["root_seed"])
        _atomic(output / "replay.json", replay, check)
        status = "PASS_MECHANISM" if choice["passed"] else "NO_GO_MECHANISM"
        report = {"status": status, "attempts": 288, "development_only": True,
                  "choice_aggregates": choice, "g2_status": "not_assessed", "preflight_emitted": False,
                  "envelope": "descriptive_only", "source_attempts": 0, "final_attempts": 0,
                  "elapsed_seconds": time.monotonic() - start, "cpu_seconds": time.process_time() - cpu_start,
                  "peak_rss_bytes": guard.peak_bytes, "accounting_boundary": "generation_checks_replay_hashes_terminal_writes"}
        _atomic(output / "report.json", report, check)
    except Exception as exc:
        terminal_reason = type(exc).__name__
        report = {"status": "NOT_ASSESSED", "terminal_reason": terminal_reason,
                  "attempts_persisted": len(rows), "g2_status": "not_assessed", "preflight_emitted": False,
                  "elapsed_seconds": time.monotonic() - start, "cpu_seconds": time.process_time() - cpu_start,
                  "peak_rss_bytes": guard.peak_bytes,
                  "accounting_boundary": "exception_capture_before_small_failure_flush",
                  "accounting_excludes": "bounded failure report and manifest writes"}
        # A cap-expiry path does only bounded small-file flushes; it never rehashes outcomes.
        try:
            _atomic(output / "report.json", report, lambda: None)
        except Exception:
            (output / "report.json").write_bytes(_json_bytes(report))
        if not (output / "outcomes.jsonl").exists():
            (output / "outcomes.jsonl").write_bytes(b"")
        if not (output / "replay.json").exists():
            (output / "replay.json").write_bytes(_json_bytes({"status": "NOT_ASSESSED", "reason": terminal_reason}))
    hashes: dict[str, str] = {}
    try:
        for name in ARTIFACTS[:-1]:
            if (output / name).is_file():
                hashes[name] = _hash(output / name, check)
        report_doc = strict_json_loads((output / "report.json").read_text(encoding="utf-8"))
        manifest = {"manifest_version": "dwr-certificate-sampler-v2-run-v1", "status": report_doc["status"],
                    "source": source, "config_sha256": cfg_hash, "root_seed": config["root_seed"],
                    "rng": config["rng"], "resources": {k: report_doc.get(k) for k in
                    ("elapsed_seconds", "cpu_seconds", "peak_rss_bytes")}, "artifacts": hashes,
                    "g2_status": "not_assessed", "preflight_emitted": False}
        (output / "manifest.json").write_bytes(_json_bytes(manifest))
        check()
        report_doc.update({"elapsed_seconds": time.monotonic() - start,
                           "cpu_seconds": time.process_time() - cpu_start,
                           "peak_rss_bytes": guard.peak_bytes,
                           "accounting_boundary": "after_first_terminal_manifest_write_and_guard_check",
                           "accounting_excludes": "final report hash and bounded report_manifest accounting flush"})
        _atomic(output / "report.json", report_doc, check)
        hashes["report.json"] = _hash(output / "report.json", check)
        manifest["artifacts"] = hashes
        manifest["resources"] = {k: report_doc.get(k) for k in
                                  ("elapsed_seconds", "cpu_seconds", "peak_rss_bytes")}
        (output / "manifest.json").write_bytes(_json_bytes(manifest))
        check()
    except Exception as exc:
        # Terminal cap revokes any saved PASS without touching large artifacts again.
        _revoke_terminal(output, hashes, type(exc).__name__, {
            "attempts_persisted": len(rows), "elapsed_seconds": time.monotonic() - start,
            "cpu_seconds": time.process_time() - cpu_start, "peak_rss_bytes": guard.peak_bytes,
            "accounting_boundary": "exception_capture_before_small_failure_flush",
            "accounting_excludes": "bounded failure report and manifest writes"})
    return strict_json_loads((output / "report.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_fixture(args.config, args.output, argv=sys.orig_argv)
    except Exception as exc:
        print(f"NOT_ASSESSED: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(stable_json(result))
    return 0 if result["status"] == "PASS_MECHANISM" else 1


if __name__ == "__main__":
    raise SystemExit(main())
