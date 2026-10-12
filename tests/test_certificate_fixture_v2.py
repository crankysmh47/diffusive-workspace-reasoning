import hashlib
import itertools
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dwr.tasks.canonical import canonical_content, content_id, structure_signature
from dwr.tasks.certificate_generator import CertificateOutcome, WorkCounters
from dwr.tasks.certificate_fixture_v2 import (
    FixtureError, _assert_no_unary_gac, _choices, _core, _eligible, _load_config, _own_row, _revoke_terminal,
    _audit_certificate, _replay, _scalar_count, _verify_candidate, run_fixture,
)
import dwr.tasks.certificate_fixture_v2 as fixture_module
from dwr.tasks.schema import (Constraint, ConstraintTask, GivenAssignment, RelationKind, Solution,
                              canonical_constraint_key, serialize_task)


def _task(*, padding=False):
    variables = tuple(f"v{i}" for i in range(6))
    constraints = [Constraint(RelationKind.EQ, ("v0", "v1")),
                   Constraint(RelationKind.EQ, ("v1", "v2")),
                   Constraint(RelationKind.EQ, ("v2", "v3")),
                   Constraint(RelationKind.EQ, ("v3", "v4")),
                   Constraint(RelationKind.EQ, ("v4", "v5"))]
    if padding:
        constraints.append(Constraint(RelationKind.EQ, ("v0", "v5")))
    constraints.sort(key=lambda c: (c.kind.value, tuple(sorted(c.variables))))
    return ConstraintTask("constraint-task-v1", 3, variables, tuple(constraints),
                          (GivenAssignment("v0", 1), GivenAssignment("v5", 1)))


def _rename(task, permutation):
    names = {f"v{i}": f"v{permutation[i]}" for i in range(6)}
    constraints = tuple(sorted((Constraint(c.kind, tuple(names[x] for x in c.variables))
                                for c in task.constraints),
                               key=lambda c: (c.kind.value, tuple(sorted(c.variables)))))
    givens = tuple(sorted((GivenAssignment(names[g.variable], g.value) for g in task.givens),
                          key=lambda g: g.variable))
    return ConstraintTask("constraint-task-v1", 3, task.variables, constraints, givens)


def _eq_add_core():
    return ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2"),
                          (Constraint(RelationKind.EQ, ("v0", "v1")),
                           Constraint(RelationKind.ADD_MOD, ("v0", "v1", "v2"))),
                          (GivenAssignment("v0", 0),))


def _neq_add_core():
    variables = tuple(f"v{i}" for i in range(6))
    rules = (Constraint(RelationKind.ADD_MOD, ("v0", "v3", "v4")),
             Constraint(RelationKind.ADD_MOD, ("v0", "v3", "v5")),
             Constraint(RelationKind.NEQ, ("v0", "v3")),
             Constraint(RelationKind.NEQ, ("v2", "v3")))
    return ConstraintTask("constraint-task-v1", 3, variables, rules,
                          (GivenAssignment("v0", 0), GivenAssignment("v1", 0), GivenAssignment("v2", 1)))


def _canned_eqadd_outcome():
    names = tuple(f"v{i}" for i in range(6))
    witness = Solution(tuple(zip(names, (0, 1, 2, 2, 2, 0))))
    eq_alt = Solution(tuple(zip(names, (0, 1, 1, 2, 2, 0))))
    add_alt = Solution(tuple(zip(names, (0, 1, 1, 1, 1, 0))))
    rules = [Constraint(RelationKind.EQ, pair) for pair in
             (("v0", "v5"), ("v2", "v3"), ("v2", "v4"), ("v3", "v4"))]
    add_rules = [Constraint(RelationKind.ADD_MOD, rule) for rule in
                 (("v1", "v3", "v0"), ("v1", "v3", "v5"),
                  ("v1", "v4", "v0"), ("v1", "v4", "v5"))]
    task = ConstraintTask("constraint-task-v1", 3, names,
                          tuple(sorted((*rules, *add_rules), key=canonical_constraint_key)),
                          (GivenAssignment("v0", 0), GivenAssignment("v1", 1)))
    return CertificateOutcome(20261012, "development", 4, ("eq", "add_mod"), "A",
                              task.givens, witness, (("eq", eq_alt), ("add_mod", add_alt)),
                              (rules[1], add_rules[0]), WorkCounters(), task, None)


class CertificateFixtureV2Tests(unittest.TestCase):
    def test_independent_scalar_oracle_and_given_only_padding(self):
        task = _task(padding=True)
        self.assertEqual(_scalar_count(task), 1)
        core = _core(task, lambda: None)
        self.assertGreaterEqual(core["removed_constraints"], 1)
        self.assertEqual(len(json.loads(core["task"])["constraints"]), 4)

    def test_core_and_identity_are_renaming_invariant(self):
        task = _task()
        renamed = _rename(task, (2, 5, 1, 4, 0, 3))
        self.assertEqual(canonical_content(task), canonical_content(renamed))
        self.assertEqual(content_id(task), content_id(renamed))
        self.assertEqual(structure_signature(task), structure_signature(renamed))
        self.assertEqual(_core(task, lambda: None)["signature"],
                         _core(renamed, lambda: None)["signature"])

    def test_diagnostic_rows_cannot_contribute_to_choice_aggregates(self):
        rows = [
            {"task": "diagnostic-secret", "content_id": "poison", "split": "development_diagnostic_test",
             "oracle_unique": True, "eligible": True, "profile": "eq/neq", "cell": "A",
             "branch": None, "core": {"signature": "poison", "given_count": 2}, "choice_permitted": False},
            {"task": "allowed", "content_id": "allowed", "split": "development_validation",
             "oracle_unique": True, "eligible": True, "profile": "eq/neq", "cell": "A",
             "branch": None, "core": None, "choice_permitted": True},
        ]
        report = _choices(rows)
        self.assertEqual(report["permitted_unique_eligible_count"], 1)
        self.assertFalse(rows[0]["choice_permitted"])
        self.assertNotIn("poison", json.dumps(report))
        self.assertNotIn("diagnostic-secret", json.dumps(report))
        private = {"task": serialize_task(_task()), "content_id": "diag-poison",
                   "split": "development_diagnostic_test", "oracle_unique": True, "eligible": True,
                   "branch": "no_unary", "gac": {"stalled": True}}
        with patch.dict(_own_row.__globals__, {"_core": Mock(side_effect=AssertionError("diagnostic core call"))}):
            _own_row(private, {}, lambda: None)
        self.assertFalse(private["choice_permitted"])
        self.assertEqual(_choices([private])["permitted_unique_eligible_count"], 0)

    def test_duplicate_and_nonunique_content_cannot_repeat_core_choice(self):
        rows = [
            {"task": serialize_task(_task()), "content_id": "dup", "split": "development_validation",
             "oracle_unique": True, "eligible": True, "branch": "no_unary", "gac": {"stalled": True}},
            {"task": serialize_task(_task()), "content_id": "dup", "split": "development_validation",
             "oracle_unique": True, "eligible": True, "branch": "no_unary", "gac": {"stalled": True}},
            {"task": serialize_task(_task(padding=True)), "content_id": "n", "split": "development_validation",
             "oracle_unique": False, "eligible": False, "branch": "no_unary", "gac": {"stalled": True}},
            {"task": serialize_task(_task()), "content_id": "n", "split": "development_validation",
             "oracle_unique": True, "eligible": True, "branch": "no_unary", "gac": {"stalled": True}},
        ]
        seen = {}
        core = Mock(return_value={"signature": "synthetic"})
        with patch.dict(_own_row.__globals__, {"_core": core}):
            for row in rows:
                _own_row(row, seen, lambda: None)
        self.assertTrue(rows[0]["choice_permitted"])
        self.assertFalse(rows[1]["choice_permitted"])
        self.assertTrue(rows[2]["choice_permitted"])
        self.assertFalse(rows[3]["choice_permitted"])
        self.assertEqual(_choices([rows[2]])["permitted_unique_eligible_count"], 0)
        self.assertEqual(core.call_count, 1)

    def test_config_rejects_unknown_and_nonfinite_values_by_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"extra":NaN}', encoding="utf-8")
            with self.assertRaises(Exception):
                _load_config(path)

    def test_candidate_shape_and_schedule_are_checked_before_use(self):
        valid = _canned_eqadd_outcome()
        altered = ConstraintTask("constraint-task-v1", 3, valid.task.variables,
                                 valid.task.constraints[:-1], valid.task.givens)
        object.__setattr__(valid, "task", altered)
        with self.assertRaises(FixtureError):
            _verify_candidate(valid, 4, lambda: None)

    def test_scalar_oracle_duplicate_with_same_cardinality_is_rejected(self):
        names = tuple(f"v{i}" for i in range(6))
        witness = Solution(tuple(zip(names, (1, 2, 1, 1, 0, 0))))
        eq_alt = Solution(tuple(zip(names, (1, 2, 0, 0, 1, 1))))
        neq_alt = Solution(tuple(zip(names, (1, 2, 1, 1, 1, 1))))
        eq_rules = tuple(Constraint(RelationKind.EQ, pair) for pair in
                         (("v0", "v2"), ("v0", "v3"), ("v2", "v3"), ("v4", "v5")))
        neq_rules = tuple(Constraint(RelationKind.NEQ, pair) for pair in
                          (("v1", "v2"), ("v1", "v3"), ("v2", "v4"), ("v3", "v5")))
        givens = (GivenAssignment("v0", 1), GivenAssignment("v1", 2))
        task = ConstraintTask("constraint-task-v1", 3, names,
                              tuple(sorted((*eq_rules, *neq_rules), key=canonical_constraint_key)), givens)
        outcome = CertificateOutcome(20261012, "development", 0, ("eq", "neq"), "A", givens,
                                     witness, (("eq", eq_alt), ("neq", neq_alt)),
                                     (eq_rules[0], neq_rules[2]), WorkCounters(), task, None)
        with patch.object(fixture_module, "enumerate_solutions",
                          return_value=SimpleNamespace(solutions=(witness, witness))):
            with self.assertRaisesRegex(FixtureError, "independent scalar oracle"):
                _verify_candidate(outcome, 0, lambda: None)

    def test_linked_output_ancestor_is_refused(self):
        from unittest.mock import Mock
        root = Path(fixture_module.__file__).resolve().parents[2]
        config = root / "configs" / "tasks" / "certificate_sampler_v2_fixture.json"
        with tempfile.TemporaryDirectory(prefix="fixture-v2-junction-test-", dir=root / "runs") as directory:
            target = Path(directory) / "certificate-sampler-v2-fixture-linked"
            with patch.object(Path, "lstat", return_value=Mock(st_file_attributes=0x400)), \
                 patch.object(Path, "exists", return_value=True), patch.object(Path, "is_symlink", return_value=False):
                with self.assertRaisesRegex(FixtureError, "reparse point"):
                    run_fixture(config, target, guard=SimpleNamespace(check=lambda: None))

    def test_production_outcome_contract_passes_canned_eqadd_integrity_path(self):
        outcome = _canned_eqadd_outcome()
        row = _verify_candidate(outcome, 4, lambda: None)
        self.assertTrue(row["oracle_unique"])
        self.assertTrue(row["eligible"])
        self.assertEqual(row["solution_count"], 1)

    def test_malformed_core_padding_does_not_hide_constraint_essentiality(self):
        task = _task()
        core = _core(task, lambda: None)
        decoded = json.loads(core["task"])
        for index in range(len(decoded["constraints"])):
            remaining = decoded["constraints"][:index] + decoded["constraints"][index + 1:]
            from dwr.tasks.schema import parse_task_json
            candidate = parse_task_json(json.dumps(decoded | {"constraints": remaining}))
            self.assertGreater(_scalar_count(candidate), 1)

    def test_eqadd_and_neqadd_analytic_cores(self):
        eqadd = _eq_add_core()
        neqadd = _neq_add_core()
        self.assertEqual(_scalar_count(eqadd), 1)
        self.assertEqual(_scalar_count(neqadd), 1)
        self.assertEqual(_core(eqadd, lambda: None)["size"], 2)
        self.assertEqual(_core(neqadd, lambda: None)["size"], 4)

    def test_certificate_audit_rejects_tampered_reserved_constraint(self):
        task = _eq_add_core()
        witness = Solution((("v0", 0), ("v1", 0), ("v2", 0)))
        eq_alternative = Solution((("v0", 0), ("v1", 1), ("v2", 1)))
        add_alternative = Solution((("v0", 0), ("v1", 0), ("v2", 1)))
        alternatives = (("eq", eq_alternative), ("add_mod", add_alternative))
        _audit_certificate(task, witness, alternatives, task.constraints, ("eq", "add_mod"))
        tampered = (Constraint(RelationKind.EQ, ("v0", "v1")),
                    Constraint(RelationKind.ADD_MOD, ("v0", "v1", "v2")))
        with self.assertRaises(FixtureError):
            _audit_certificate(task, witness, alternatives, tampered[:1], ("eq", "add_mod"))

    def test_uniqueness_core_can_be_disconnected(self):
        task = ConstraintTask("constraint-task-v1", 3, ("v0", "v1", "v2", "v3"),
                              (Constraint(RelationKind.EQ, ("v0", "v1")),
                               Constraint(RelationKind.ADD_MOD, ("v0", "v2", "v3"))),
                              (GivenAssignment("v0", 0), GivenAssignment("v3", 1)))
        core = _core(task, lambda: None)
        self.assertFalse(core["connected"])
        self.assertEqual(core["size"], 2)

    def test_connected_eqadd_task_can_have_disconnected_uniqueness_core(self):
        variables = tuple(f"v{i}" for i in range(6))
        rules = (Constraint(RelationKind.EQ, ("v0", "v2")),
                 Constraint(RelationKind.EQ, ("v0", "v3")),
                 Constraint(RelationKind.ADD_MOD, ("v0", "v4", "v1")),
                 Constraint(RelationKind.ADD_MOD, ("v1", "v5", "v0")),
                 Constraint(RelationKind.ADD_MOD, ("v2", "v4", "v3")),
                 Constraint(RelationKind.ADD_MOD, ("v4", "v5", "v0")))
        task = ConstraintTask("constraint-task-v1", 3, variables,
                              tuple(sorted(rules, key=canonical_constraint_key)),
                              (GivenAssignment("v0", 0), GivenAssignment("v1", 0)))
        self.assertEqual(_scalar_count(task), 1)
        from dwr.tasks.certificate_fixture_v2 import _graph
        self.assertTrue(_graph(task)["connected"])
        self.assertTrue(_eligible(task, lambda: None)[0])
        core = _core(task, lambda: None)
        self.assertFalse(core["connected"])

    def test_no_unary_eqadd_domains_stay_full_under_actual_gac(self):
        variables = tuple(f"v{i}" for i in range(6))
        rules = [Constraint(RelationKind.EQ, pair)
                  for pair in itertools.combinations(("v3", "v4", "v5"), 2)]
        rules += [Constraint(RelationKind.ADD_MOD, operands)
                  for target in ("v3", "v4", "v5")
                  for pair in itertools.combinations(("v3", "v4", "v5"), 2)
                  if target not in pair for operands in ((*pair, target),)]
        task = ConstraintTask("constraint-task-v1", 3, variables,
                              tuple(sorted(rules, key=lambda r: (r.kind.value, r.variables))),
                              (GivenAssignment("v0", 0), GivenAssignment("v1", 1), GivenAssignment("v2", 2)))
        result = _assert_no_unary_gac(task, lambda: None)
        self.assertTrue(result["stalled"])
        self.assertEqual(result["unresolved_unknowns"], 3)

    def test_no_unary_add_with_given_target_and_addend_stays_full_under_gac(self):
        variables = tuple(f"v{i}" for i in range(6))
        task = ConstraintTask("constraint-task-v1", 3, variables,
                              (Constraint(RelationKind.ADD_MOD, ("v0", "v3", "v4")),
                               Constraint(RelationKind.ADD_MOD, ("v3", "v4", "v0")),
                               Constraint(RelationKind.NEQ, ("v0", "v1"))),
                              (GivenAssignment("v0", 0), GivenAssignment("v1", 1),
                               GivenAssignment("v2", 2)))
        result = _assert_no_unary_gac(task, lambda: None)
        self.assertTrue(result["stalled"])
        self.assertEqual(result["domains"][3:], [[0, 1, 2]] * 3)

    def test_mixed_null_rejection_preserves_and_checks_partial_certificate(self):
        names = tuple(f"v{i}" for i in range(6))
        witness = Solution(tuple(zip(names, (0, 1, 2, 0, 1, 2))))
        first = Solution(tuple(zip(names, (0, 1, 2, 1, 1, 2))))
        givens = (GivenAssignment("v0", 0), GivenAssignment("v1", 1))
        outcome = CertificateOutcome(20261012, "development", 4, ("eq", "add_mod"), "A",
                                     givens, witness, (("eq", first),), (), WorkCounters(), None,
                                     "no_feasible_mixed_second_alternative")
        row = _verify_candidate(outcome, 4, lambda: None)
        self.assertEqual(row["state"], "construction_failed")
        self.assertEqual(row["certificate"]["witness"], [list(pair) for pair in witness.assignments])

    def test_persisted_replay_rederives_exact_fixed_prefix(self):
        rows = [{"candidate_index": i, "task": None, "state": "construction_failed"}
                for i in range(288)]
        seen_indices = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (output / "outcomes.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

            def factory(seed, partition, index, *, check_budget):
                seen_indices.append((seed, partition, index))
                return object()

            with (patch.object(fixture_module, "_verify_candidate", side_effect=lambda outcome, index, check: rows[index]),
                  patch.object(fixture_module, "_own_row", return_value=None),
                  patch.object(fixture_module, "_choices", return_value={"passed": False})):
                result = _replay(output, rows, lambda: None, factory, 20261012)
        self.assertEqual(result["attempts_rederived"], 288)
        self.assertEqual([item[2] for item in seen_indices], list(range(288)))
        self.assertTrue(all(item[0] == 20261012 and item[1] == "development" for item in seen_indices))

    def test_late_failure_revokes_a_saved_pass_with_small_terminal_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "report.json").write_text('{"status":"PASS_MECHANISM"}', encoding="utf-8")
            _revoke_terminal(output, {"outcomes.jsonl": "already-checked"}, "TimeoutError")
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["artifacts"], {"outcomes.jsonl": "already-checked"})

    def test_driver_late_terminal_cap_revokes_pass_and_keeps_only_checked_hashes(self):
        class Guard:
            peak_bytes = 1024
            expired = False
            manifest_path = None

            @classmethod
            def check(cls):
                if cls.expired:
                    raise TimeoutError("synthetic expiry after final terminal manifest write")

        root = Path(fixture_module.__file__).resolve().parents[2]
        config = root / "configs" / "tasks" / "certificate_sampler_v2_fixture.json"
        runs = root / "runs"
        with tempfile.TemporaryDirectory(prefix="fixture-v2-test-", dir=runs) as scratch:
            output = Path(scratch) / "certificate-sampler-v2-fixture-late"
            Guard.manifest_path = output / "manifest.json"
            real_write = Path.write_bytes
            manifest_writes = 0

            def expire_after_second_manifest_write(path, data):
                nonlocal manifest_writes
                result = real_write(path, data)
                if path == Guard.manifest_path:
                    manifest_writes += 1
                    if manifest_writes == 2:
                        Guard.peak_bytes = 2048
                        Guard.expired = True
                return result

            real_hash = fixture_module._hash
            hashed_after_expiry = []
            def observed_hash(path, check):
                if Guard.expired:
                    hashed_after_expiry.append(path.name)
                return real_hash(path, check)

            def fake_row(outcome, index, check):
                return {"candidate_index": index, "profile": "eq/neq", "cell": "A", "task": "opaque",
                        "content_id": f"id-{index}", "structure_signature": f"sig-{index}",
                        "split": "development_validation", "oracle_unique": True, "eligible": True,
                        "first_seen_content": False, "choice_permitted": False, "branch": None, "core": None}

            def fake_owner(row, seen, check):
                row["first_seen_content"] = True
                row["choice_permitted"] = True
                seen[row["content_id"]] = row["split"]

            with (patch.object(fixture_module, "_source_identity", return_value={"git_clean": True}),
                  patch.object(fixture_module, "_verify_candidate", side_effect=fake_row),
                  patch.object(fixture_module, "_own_row", side_effect=fake_owner),
                  patch.object(fixture_module, "_choices", return_value={"passed": True}),
                  patch.object(fixture_module, "_replay", return_value={"status": "PASS_REPLAY"}),
                  patch.object(fixture_module, "_hash", side_effect=observed_hash),
                  patch.object(Path, "write_bytes", autospec=True, side_effect=expire_after_second_manifest_write)):
                report = run_fixture(config, output, guard=Guard(), outcome_factory=lambda *a, **k: object())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(report["attempts"], 288)
            self.assertGreaterEqual(report["elapsed_seconds"], 0)
            self.assertEqual(report["peak_rss_bytes"], 2048)
            self.assertEqual(report["accounting_boundary"], "exception_capture_before_small_failure_flush")
            self.assertEqual(hashed_after_expiry, [])
            self.assertEqual(manifest["status"], "NOT_ASSESSED")
            self.assertEqual(set(manifest["artifacts"]), {"manifest.initial.json", "outcomes.jsonl", "replay.json"})
            self.assertEqual(len((output / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()), 288)

    def test_driver_persists_constructor_and_core_failures_as_fatal_attempts(self):
        class Guard:
            peak_bytes = 32

            @staticmethod
            def check():
                return None

        root = Path(fixture_module.__file__).resolve().parents[2]
        config = root / "configs" / "tasks" / "certificate_sampler_v2_fixture.json"
        with tempfile.TemporaryDirectory(prefix="fixture-v2-failures-test-", dir=root / "runs") as scratch:
            for suffix, fail_phase in (("constructor", "construction"),
                                        ("invalid_outcome", "independent_integrity_checks"),
                                        ("core", "content_ownership_and_core")):
                output = Path(scratch) / f"certificate-sampler-v2-fixture-{suffix}"
                def factory(seed, partition, index, *, check_budget):
                    if fail_phase == "construction":
                        raise RuntimeError("synthetic factory failure")
                    if fail_phase == "independent_integrity_checks":
                        return object()
                    return object()
                def verify(outcome, index, check):
                    return {"candidate_index": index, "profile": "eq/neq", "cell": "A", "task": "opaque",
                            "content_id": f"id-{index}", "structure_signature": f"sig-{index}",
                            "split": "development_validation", "choice_permitted": False}
                def owner(row, seen, check):
                    raise RuntimeError("synthetic core failure")
                with ExitStack() as stack:
                    stack.enter_context(patch.object(fixture_module, "_source_identity", return_value={"git_clean": True}))
                    if fail_phase == "content_ownership_and_core":
                        stack.enter_context(patch.object(fixture_module, "_verify_candidate", side_effect=verify))
                        stack.enter_context(patch.object(fixture_module, "_own_row", side_effect=owner))
                    report = run_fixture(config, output, guard=Guard(), outcome_factory=factory)
                lines = (output / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertTrue(lines, (report, [p.name for p in output.iterdir()]))
                row = json.loads(lines[0])
                self.assertEqual(report["status"], "NOT_ASSESSED")
                self.assertEqual(row["state"], "fatal_error")
                self.assertEqual(row["failure_phase"], fail_phase)

    def test_driver_success_binds_all_four_nonterminal_artifacts(self):
        class Guard:
            peak_bytes = 1024

            @staticmethod
            def check():
                return None

        root = Path(fixture_module.__file__).resolve().parents[2]
        config = root / "configs" / "tasks" / "certificate_sampler_v2_fixture.json"
        with tempfile.TemporaryDirectory(prefix="fixture-v2-success-test-", dir=root / "runs") as scratch:
            output = Path(scratch) / "certificate-sampler-v2-fixture-ok"

            def fake_row(outcome, index, check):
                return {"candidate_index": index, "profile": "eq/neq", "cell": "A", "task": "opaque",
                        "content_id": f"id-{index}", "structure_signature": f"sig-{index}",
                        "split": "development_validation", "oracle_unique": True, "eligible": True,
                        "first_seen_content": False, "choice_permitted": False, "branch": None, "core": None}

            def fake_owner(row, seen, check):
                row["first_seen_content"] = True
                row["choice_permitted"] = True
                seen[row["content_id"]] = row["split"]

            with (patch.object(fixture_module, "_source_identity", return_value={"git_clean": True}),
                  patch.object(fixture_module, "_verify_candidate", side_effect=fake_row),
                  patch.object(fixture_module, "_own_row", side_effect=fake_owner),
                  patch.object(fixture_module, "_choices", return_value={"passed": True}),
                  patch.object(fixture_module, "_replay", return_value={"status": "PASS_REPLAY"})):
                report = run_fixture(config, output, guard=Guard(), outcome_factory=lambda *a, **k: object())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS_MECHANISM")
            self.assertEqual(set(manifest["artifacts"]),
                             {"manifest.initial.json", "outcomes.jsonl", "replay.json", "report.json"})
            for name, digest in manifest["artifacts"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
