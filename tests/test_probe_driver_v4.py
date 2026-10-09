"""Tiny v4 orchestration checks; no quota-sized sampling or corpus access."""
from __future__ import annotations

import unittest
import hashlib
import itertools
import tempfile
from unittest.mock import patch
from pathlib import Path

from dwr.tasks.config_probe_v4 import load_probe_config
from dwr.tasks.probe_v4 import (ProbeState, consume_proposal, project_public_metrics,
                                replay_prefix, run_source_only_fixture, run_probe, ARTIFACTS,
                                run_fixture_probe,
                                _profile_maps, _verify_initial_binding, _producer_identity,
                                _ensure_failure_artifacts,
                                _verify_replay_document, _verify_source_closure,
                                _verify_audit_document, _verify_terminal_report,
                                _FixturePolicy)
from dwr.tasks.audit_probe_v4 import audit_records, audit_run
from dwr.tasks.run_support import ResourceGuard
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind
from dwr.tasks.solver import solve
from dwr.tasks.generator_probe_v4 import ProposalFatalError, ProposalFatalContext
from dwr.tasks.oracle import enumerate_solutions
from dwr.tasks.generator_probe_v4 import ProposalResult
from dwr.tasks.audit_probe_v4 import _independent_interaction_projection
from dwr.tasks.canonical import content_id, development_split, source_split, structure_signature
from dwr.tasks.schema import serialize_model_input, serialize_task, parse_task_json


def _fixture_constraint(kind, a, b, c=None):
    relation = {"eq": RelationKind.EQ, "neq": RelationKind.NEQ, "add_mod": RelationKind.ADD_MOD}[kind]
    return Constraint(relation, (a, b) if c is None else (a, b, c))


def _fixture_task(partition, profile, cell, *, eq_neq_alternative=None):
    given_count, constraint_count = {"A": (2, 8), "B": (3, 6), "C": (3, 7), "D": (3, 8)}[cell]
    if partition == "source" and profile == ("eq",):
        givens = [GivenAssignment("v0", 0), GivenAssignment("v1", 0)]
        values = {f"v{i}": 0 for i in range(6)}
        if given_count == 3:
            givens.append(GivenAssignment("v2", 0))
            edges = ((0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5))
            if constraint_count >= 7:
                edges += ((0, 2),)
            if constraint_count >= 8:
                edges = tuple(itertools.combinations(range(4), 2)) + ((3, 4), (4, 5))
            edges = edges[:constraint_count]
        else:
            edges = tuple(itertools.combinations(range(4), 2)) + ((3, 4), (4, 5))
        extra_edges = ()
        constraints = [_fixture_constraint("eq", f"v{a}", f"v{b}") for a, b in edges]
        eq_extras = [_fixture_constraint("eq", f"v{a}", f"v{b}") for a, b in extra_edges]
        kind = "eq"
    elif partition == "source" and profile == ("neq",):
        if given_count == 2:
            givens = [GivenAssignment("v0", 0), GivenAssignment("v1", 1)]
            values = {"v0": 0, "v1": 1, **{f"v{i}": 2 for i in range(2, 6)}}
            edges = tuple((i, anchor) for i in range(2, 6) for anchor in (0, 1))
        else:
            givens = [GivenAssignment("v0", 0), GivenAssignment("v1", 1), GivenAssignment("v2", 2)]
            values = {"v0": 0, "v1": 1, "v2": 2, "v3": 0, "v4": 1, "v5": 2}
            edges = ((3, 1), (3, 2), (4, 0), (4, 2), (5, 0), (5, 1))
        constraints = [_fixture_constraint("neq", f"v{a}", f"v{b}") for a, b in edges]
        kind = "neq"
    elif partition == "source" and profile == ("add_mod",):
        if given_count == 2:
            givens = [GivenAssignment("v0", 0), GivenAssignment("v1", 0)]
            values = {f"v{i}": 0 for i in range(6)}
            equations = (("v0", "v2", "v3"), ("v1", "v2", "v3"),
                ("v0", "v3", "v4"), ("v1", "v3", "v4"),
                ("v0", "v4", "v5"), ("v1", "v4", "v5"),
                ("v2", "v3", "v4"), ("v0", "v2", "v4"))
        else:
            givens = [GivenAssignment(f"v{i}", 0) for i in range(3)]
            values = {f"v{i}": 0 for i in range(6)}
            equations = (("v0", "v3", "v4"), ("v1", "v3", "v4"), ("v2", "v3", "v4"),
                ("v0", "v4", "v5"), ("v1", "v4", "v5"), ("v3", "v4", "v5"),
                ("v2", "v4", "v5"), ("v0", "v3", "v5"))
        constraints = [_fixture_constraint("add_mod", *equation)
                       for equation in equations[:constraint_count]]
        kind = "add_mod"
    elif partition == "development" and profile == ("eq", "neq"):
        if given_count == 2:
            givens = [GivenAssignment("v0", 0), GivenAssignment("v1", 1)]
            values = {"v0": 0, "v1": 1, "v2": 0, "v3": 0, "v4": 2, "v5": 2}
            constraints = [_fixture_constraint("eq", "v0", "v2"),
                _fixture_constraint("eq", "v2", "v3"), _fixture_constraint("eq", "v4", "v5"),
                _fixture_constraint("neq", "v4", "v0"), _fixture_constraint("neq", "v4", "v1"),
                _fixture_constraint("neq", "v5", "v0"), _fixture_constraint("neq", "v5", "v1"),
                _fixture_constraint("neq", "v3" if cell == "A" else "v2", "v4")]
            if cell == "A":
                alternative = 1 if eq_neq_alternative is None else eq_neq_alternative
                if alternative == 1:
                    constraints[4] = _fixture_constraint("neq", "v2", "v4")
                elif alternative == 2:
                    constraints[6] = _fixture_constraint("neq", "v3", "v5")
                elif alternative == 3:
                    constraints[4] = _fixture_constraint("eq", "v0", "v3")
                elif alternative == 4:
                    constraints[5] = _fixture_constraint("eq", "v0", "v3")
        else:
            givens = [GivenAssignment("v0", 0), GivenAssignment("v1", 1), GivenAssignment("v2", 2)]
            values = {"v0": 0, "v1": 1, "v2": 2, "v3": 0, "v4": 0, "v5": 2}
            constraints = [_fixture_constraint("eq", "v0", "v3"), _fixture_constraint("eq", "v3", "v4"),
                _fixture_constraint("neq", "v0", "v5"), _fixture_constraint("neq", "v1", "v5"),
                _fixture_constraint("neq", "v4", "v5"), _fixture_constraint("neq", "v0", "v1")]
        kind = "eq"
    elif partition == "development" and profile == ("eq", "add_mod"):
        if given_count == 2:
            givens = [GivenAssignment("v0", 0), GivenAssignment("v1", 1)]
            values = {"v0": 0, "v1": 1, "v2": 1, "v3": 1, "v4": 2, "v5": 1}
            constraints = [_fixture_constraint("eq", a, b) for a, b in
                (("v1", "v2"), ("v2", "v3"), ("v1", "v3"), ("v1", "v5"), ("v2", "v5"), ("v3", "v5"))]
            constraints += [_fixture_constraint("add_mod", "v2", "v3", "v4"),
                            _fixture_constraint("add_mod", "v3", "v0", "v5")]
        else:
            givens = [GivenAssignment("v0", 0), GivenAssignment("v1", 1), GivenAssignment("v2", 1)]
            values = {"v0": 0, "v1": 1, "v2": 1, "v3": 1, "v4": 1, "v5": 2}
            constraints = [_fixture_constraint("eq", "v1", "v3"), _fixture_constraint("eq", "v3", "v4"),
                _fixture_constraint("add_mod", "v3", "v4", "v5"),
                _fixture_constraint("add_mod", "v0", "v3", "v4"),
                _fixture_constraint("eq", "v1", "v4"),
                _fixture_constraint("eq", "v2", "v4" if cell == "B" else "v3")]
        kind = "eq"
    else:
        if given_count == 2:
            givens = [GivenAssignment("v0", 0), GivenAssignment("v1", 1)]
            values = {"v0": 0, "v1": 1, "v2": 2, "v3": 0, "v4": 2, "v5": 2}
            constraints = [_fixture_constraint("neq", "v2", "v0"), _fixture_constraint("neq", "v2", "v1"),
                _fixture_constraint("neq", "v3", "v2"), _fixture_constraint("neq", "v3", "v1"),
                _fixture_constraint("add_mod", "v2", "v3", "v4"),
                _fixture_constraint("add_mod", "v0", "v2", "v5"),
                _fixture_constraint("neq", "v3", "v4"), _fixture_constraint("neq", "v3", "v5")]
        else:
            givens = [GivenAssignment("v0", 0), GivenAssignment("v1", 1), GivenAssignment("v2", 2)]
            values = {"v0": 0, "v1": 1, "v2": 2, "v3": 0, "v4": 1, "v5": 1}
            constraints = [_fixture_constraint("neq", *edge) for edge in
                (("v3", "v1"), ("v3", "v2"), ("v4", "v0"), ("v4", "v2"), ("v3", "v4"))]
            constraints.append(_fixture_constraint("add_mod", "v3", "v4", "v5"))
        kind = "neq"

    if (partition == "development" and profile == ("eq", "neq") and
            given_count == 3 and constraint_count >= 7):
        constraints.append(_fixture_constraint("neq", "v3", "v5"))
        if constraint_count == 8:
            constraints.append(_fixture_constraint("eq", "v0", "v4"))
    elif (partition == "development" and profile == ("neq", "add_mod") and
          given_count == 3 and constraint_count >= 7):
        constraints.append(_fixture_constraint("neq", "v0", "v5") if cell == "C" else
                           _fixture_constraint("neq", "v3", "v5"))
        if constraint_count == 8:
            constraints.append(_fixture_constraint("add_mod", "v1", "v2", "v0"))

    # Explicitly selected fixed alternative; never generated or searched.
    if (partition == "development" and profile == ("eq", "neq") and
            given_count == 3 and cell == "B"):
        alternative = 1 if eq_neq_alternative is None else eq_neq_alternative
        if alternative == 1:
            constraints[5] = _fixture_constraint("eq", "v0", "v4")
        elif alternative in (2, 3):
            constraints[5] = _fixture_constraint("neq", "v2",
                                                  "v3" if alternative == 2 else "v4")
        elif alternative == 4:
            constraints[4] = _fixture_constraint("neq", "v3", "v5")

    # Add only satisfied redundant relations to reach the registered fixed cell size.
    while len(constraints) < constraint_count:
        if kind == "eq":
            candidates = (eq_extras if partition == "source" and profile == ("eq",) else
                (_fixture_constraint("eq", f"v{a}", f"v{b}")
                 for a, b in itertools.combinations(range(6), 2) if values[f"v{a}"] == values[f"v{b}"]))
        elif kind == "neq":
            candidates = (_fixture_constraint("neq", f"v{a}", f"v{b}")
                          for a, b in itertools.combinations(range(6), 2) if values[f"v{a}"] != values[f"v{b}"])
        else:
            candidates = (_fixture_constraint("add_mod", f"v{a}", f"v{b}", f"v{c}")
                          for a, b in itertools.combinations(range(6), 2) for c in range(6)
                          if c not in (a, b) and (values[f"v{a}"] + values[f"v{b}"]) % 3 == values[f"v{c}"])
        existing = {item for item in constraints}
        extra = next((item for item in candidates if item not in existing), None)
        if extra is None:
            raise AssertionError("fixture relation supply cannot fill the fixed cell")
        constraints.append(extra)
    return ConstraintTask("constraint-task-v1", 3, tuple(f"v{i}" for i in range(6)),
                          tuple(constraints), tuple(givens))


def _fixed_semantic_fixture_provider(config, partition, index, active_cell):
    profile, cell = active_cell
    # The artifact contract serializes constraints in canonical order; derive
    # fixture oracle and deletion evidence from that exact parsed task.
    task = parse_task_json(serialize_task(_fixture_task(partition, profile, cell)))
    oracle = enumerate_solutions(task)
    solved = solve(task)
    if len(oracle.solutions) != 1 or solved.solution != oracle.solutions[0]:
        raise AssertionError(f"handcrafted fixture is not uniquely solved: {partition}/{profile}/{cell}")
    evidence = _independent_interaction_projection(task, partition)
    if not evidence["eligible"]:
        raise AssertionError(f"handcrafted fixture lacks required interaction: {partition}/{profile}/{cell}")
    return ProposalResult(partition, index, profile, cell, 6, len(task.givens), len(task.constraints),
                          None, task, solved.solution, 1, solved.solution, solved.stats, evidence)


CONFIG = Path(__file__).parents[1] / "configs" / "tasks" / "constraint_probe_v4.json"


class ProbeDriverV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_probe_config(CONFIG)

    def test_denominator_is_first_seen_and_precedes_eligibility(self):
        state = ProbeState(self.config)
        outcome, record = consume_proposal(
            self.config, state, "source", 2, ("add_mod",), "C")
        self.assertTrue(outcome["denominator_counted"])
        self.assertEqual(outcome["state"], "scientifically_ineligible")
        self.assertEqual(state.denominators["source_train"]["unique_oracle_groups"], 1)
        self.assertIsNone(record)

    def test_fixed_semantic_fixture_matrix_has_real_oracle_and_registered_envelope(self):
        for partition, profiles in (("source", self.config.source_profiles),
                                    ("development", self.config.development_profiles)):
            for profile in profiles:
                for cell in "ABCD":
                    task = _fixture_task(partition, profile, cell)
                    oracle = enumerate_solutions(task)
                    solved = solve(task)
                    self.assertEqual(len(oracle.solutions), 1, (partition, profile, cell))
                    self.assertEqual(solved.solution, oracle.solutions[0])
                    self.assertEqual(_independent_interaction_projection(task, partition)["eligible"], True)
                    self.assertGreaterEqual(len(serialize_model_input(task).encode("utf-8")), 1024)
                    self.assertIn(solved.stats.first_solution_nodes, range(4, 8))
                    if partition == "source":
                        self.assertIn(source_split(content_id(task)),
                                      ("source_train", "source_validation", "source_test"))
                    else:
                        self.assertIn(development_split(structure_signature(task)),
                                      ("development_validation", "development_diagnostic_test"))

    def test_eq_neq_alternative_set_covers_validation_and_diagnostic_without_search(self):
        expected = {"A": {1: "development_validation", 2: "development_validation",
                           3: "development_validation", 4: "development_validation"},
                    "B": {1: "development_validation", 2: "development_validation",
                           3: "development_validation", 4: "development_validation"}}
        for cell in "AB":
            original = _fixture_task("development", ("eq", "neq"), cell,
                                     eq_neq_alternative=0)
            self.assertEqual(development_split(structure_signature(original)),
                             "development_diagnostic_test")
            self.assertEqual(len(enumerate_solutions(original).solutions), 1)
            self.assertTrue(_independent_interaction_projection(original, "development")["eligible"])
            for alternative in range(1, 5):
                task = _fixture_task("development", ("eq", "neq"), cell,
                                     eq_neq_alternative=alternative)
                self.assertEqual(development_split(structure_signature(task)),
                                 expected[cell][alternative], (cell, alternative))
                self.assertEqual(len(enumerate_solutions(task).solutions), 1,
                                 (cell, alternative))
                self.assertTrue(_independent_interaction_projection(task, "development")["eligible"])

    def test_independent_mixed_projection_matches_closed_proposal_evidence_order(self):
        task = parse_task_json(serialize_task(
            _fixture_task("development", ("eq", "add_mod"), "A")))
        evidence = _independent_interaction_projection(task, "development")
        self.assertEqual(evidence["kinds"], sorted(evidence["kinds"]))
        oracle = enumerate_solutions(task)
        solved = solve(task)
        proposal = ProposalResult("development", 0, ("eq", "add_mod"), "A", 6,
            len(task.givens), len(task.constraints), None, task, solved.solution, 1,
            oracle.solutions[0], solved.stats, evidence)
        self.assertEqual(proposal.private_eligibility["kinds"], ["add_mod", "eq"])
        state = ProbeState(self.config, _fixture_policy=_FixturePolicy("mixed-order"))
        outcome, record = consume_proposal(self.config, state, "development", 0,
            ("eq", "add_mod"), "A", _proposal_provider=
            lambda config, partition, index, active: _fixed_semantic_fixture_provider(
                config, partition, index, active))
        self.assertEqual(outcome["state"], "accepted_for_quota")
        self.assertIsNotNone(record)
        projection = audit_records([record], [outcome], self.config,
                                   _fixture_targets=_FixturePolicy("mixed-order").quotas)
        self.assertEqual(projection["development_validation_by_profile"]["eq/add_mod"]["status"],
                         "not_assessed")

    def test_unpatched_full_fixed_fixture_runs_all_profiles_cells_proxy_replay_and_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "fixture-full"
            label = run_fixture_probe(CONFIG, output, fixture_id="task3-fixed-full-v1",
                                      _proposal_provider=_fixed_semantic_fixture_provider)
            self.assertEqual(label, "FIXTURE_PASS")
            manifest = __import__("json").loads(
                (output / "probe_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["partitions"]["development"]["attempts"], 12)
            self.assertEqual(manifest["status"], "fixture_complete")
            for artifact in ("source_closure.json", "candidate_replay.json",
                             "independent_audit.json", "probe_report.json"):
                self.assertTrue((output / artifact).is_file(), artifact)
            audit = __import__("json").loads(
                (output / "independent_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["public_projection"]["four_cohort_proxy"]["pass"])
            with self.assertRaisesRegex(ValueError, "fixture"):
                audit_run(output, self.config)
            with self.assertRaises(FileExistsError):
                run_fixture_probe(CONFIG, output, fixture_id="task3-fixed-full-v1",
                    _proposal_provider=_fixed_semantic_fixture_provider)

    def test_closed_artifact_identity_hash_and_nested_schema_mutations_fail_independently(self):
        import copy
        import json
        from dwr.tasks.run_support import sha256_file

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "fixture-artifacts"
            run_fixture_probe(CONFIG, output, fixture_id="task3-identity-matrix",
                _proposal_provider=_fixed_semantic_fixture_provider)
            read = lambda name: json.loads((output / name).read_text(encoding="utf-8"))
            initial = read("probe_manifest.initial.json")
            terminal = read("probe_manifest.json")
            source = initial["source"]
            producer = initial["producer"]
            policy = _FixturePolicy("task3-identity-matrix")

            _verify_initial_binding(initial, self.config, source, producer, CONFIG, policy)
            initial_mutations = []
            for key, value in (("manifest_version", "wrong-version"),
                               ("config_sha256", "0" * 64),
                               ("source", {**source, "commit": "0" * 40}),
                               ("producer", [{**producer[0], "sha256": "0" * 64}, *producer[1:]]),
                               ("limits", {**initial["limits"], "workers": 2})):
                altered = copy.deepcopy(initial); altered[key] = value
                initial_mutations.append((key, altered))
            altered = copy.deepcopy(initial)
            altered["partitions"]["development"]["quota_targets"]["eq/neq"]["A"] += 1
            initial_mutations.append(("nested-development-target", altered))
            altered = copy.deepcopy(initial); altered["unknown"] = None
            initial_mutations.append(("unknown-top-level", altered))
            altered = copy.deepcopy(initial); altered["limits"]["active_wall_seconds"] = float("inf")
            initial_mutations.append(("nonfinite-resource-cap", altered))
            for name, altered in initial_mutations:
                with self.subTest(document="initial", mutation=name), self.assertRaises(ValueError):
                    _verify_initial_binding(altered, self.config, source, producer, CONFIG, policy)

            closure = read("source_closure.json")
            _verify_source_closure(closure, output, self.config, source, producer, policy)
            closure_mutations = []
            for key, value in (("closure_version", "wrong-version"),
                               ("config_sha256", "1" * 64),
                               ("source", {**source, "commit": "1" * 40}),
                               ("producer", [{**producer[0], "sha256": "1" * 64}, *producer[1:]]),
                               ("outcome_prefix_sha256", "2" * 64)):
                altered = copy.deepcopy(closure); altered[key] = value
                closure_mutations.append((key, altered))
            altered = copy.deepcopy(closure)
            altered["source_quota_targets"]["eq"]["A"] += 1
            closure_mutations.append(("nested-source-target", altered))
            altered = copy.deepcopy(closure); altered["unexpected"] = None
            closure_mutations.append(("unknown-top-level", altered))
            altered = copy.deepcopy(closure)
            altered["train_projection"]["unexpected"] = None
            closure_mutations.append(("unknown-projection-field", altered))
            for name, altered in closure_mutations:
                with self.subTest(document="source_closure", mutation=name), self.assertRaises(ValueError):
                    _verify_source_closure(altered, output, self.config, source, producer, policy)

            replay = read("candidate_replay.json")
            _verify_replay_document(replay, output, self.config, source, producer, policy)
            replay_mutations = []
            for key, value in (("replay_version", "wrong-version"),
                               ("config_sha256", "3" * 64),
                               ("source", {**source, "commit": "3" * 40}),
                               ("producer", [{**producer[0], "sha256": "3" * 64}, *producer[1:]]),
                               ("outcome_prefix_sha256", "4" * 64)):
                altered = copy.deepcopy(replay); altered[key] = value
                replay_mutations.append((key, altered))
            altered = copy.deepcopy(replay)
            altered["quota_counts"]["development"]["eq/neq"]["A"] = 2
            replay_mutations.append(("nested-quota-count-out-of-range", altered))
            altered = copy.deepcopy(replay)
            altered["quota_targets"]["development"]["unexpected"] = {}
            replay_mutations.append(("unknown-nested-profile", altered))
            for name, altered in replay_mutations:
                with self.subTest(document="replay", mutation=name), self.assertRaises(ValueError):
                    _verify_replay_document(altered, output, self.config, source, producer, policy)

            audit = read("independent_audit.json")
            _verify_audit_document(audit, output, self.config, source, producer)
            audit_mutations = []
            for key, value in (("audit_version", "wrong-version"),
                               ("config_sha256", "5" * 64),
                               ("source", {**source, "commit": "5" * 40}),
                               ("producer", [{**producer[0], "sha256": "5" * 64}, *producer[1:]]),
                               ("replay_sha256", "6" * 64),
                               ("source_closure_sha256", "7" * 64)):
                altered = copy.deepcopy(audit); altered[key] = value
                audit_mutations.append((key, altered))
            altered = copy.deepcopy(audit)
            altered["public_projection"]["unexpected"] = None
            audit_mutations.append(("nested-projection-fields", altered))
            for name, altered in audit_mutations:
                with self.subTest(document="audit", mutation=name), self.assertRaises(ValueError):
                    _verify_audit_document(altered, output, self.config, source, producer)

            report = read("probe_report.json")
            closure_sha = sha256_file(output / "source_closure.json")
            replay_for_report = {**replay, "artifact_sha256": sha256_file(output / "candidate_replay.json")}
            audit_for_report = {**audit, "artifact_sha256": sha256_file(output / "independent_audit.json")}
            _verify_terminal_report(report, self.config, source,
                sha256_file(output / "probe_manifest.initial.json"), closure_sha,
                closure, replay_for_report, audit_for_report, policy)
            report_mutations = []
            report_source = {**source, "commit": "8" * 40}
            for key, value in (("report_version", "wrong-version"),
                               ("config_sha256", "8" * 64), ("source", report_source),
                               ("initial_manifest_sha256", "8" * 64),
                               ("source_closure_sha256", "8" * 64),
                               ("replay_sha256", "8" * 64), ("audit_sha256", "8" * 64)):
                altered = copy.deepcopy(report); altered[key] = value
                report_mutations.append((key, altered))
            altered = copy.deepcopy(report)
            altered["quotas"]["development_validation_by_profile_cell"]["eq/neq"]["A"] = 0
            report_mutations.append(("nested-quota-count", altered))
            altered = copy.deepcopy(report)
            altered["public_floors"]["development_validation_by_profile"]["eq/neq"]["status"] = "pass"
            altered["public_floors"]["development_validation_by_profile"]["eq/neq"]["values"] = None
            report_mutations.append(("nested-floor-values", altered))
            altered = copy.deepcopy(report); altered["unexpected"] = None
            report_mutations.append(("unknown-top-level", altered))
            altered = copy.deepcopy(report); altered["resource_snapshot"]["wall_seconds"] = float("nan")
            report_mutations.append(("nonfinite-resource-snapshot", altered))
            for name, altered in report_mutations:
                with self.subTest(document="report", mutation=name), self.assertRaises(ValueError):
                    _verify_terminal_report(altered, self.config, source,
                        sha256_file(output / "probe_manifest.initial.json"), closure_sha,
                        closure, replay_for_report, audit_for_report, policy)

            for artifact_name, validator in (("outcomes.jsonl", _verify_source_closure),
                                             ("accepted_records.jsonl", _verify_replay_document)):
                path = output / artifact_name
                original_bytes = path.read_bytes()
                corrupted = bytearray(original_bytes)
                corrupted[0] ^= 1
                try:
                    path.write_bytes(corrupted)
                    with self.subTest(artifact_hash=artifact_name), self.assertRaises(ValueError):
                        if artifact_name == "outcomes.jsonl":
                            validator(closure, output, self.config, source, producer, policy)
                        else:
                            validator(replay, output, self.config, source, producer, policy)
                finally:
                    path.write_bytes(original_bytes)

            for name, reference in terminal["artifacts"].items():
                path = output / name
                self.assertEqual(reference["bytes"], path.stat().st_size)
                self.assertEqual(reference["sha256"], sha256_file(path))
            (output / "fixture_identity.json").unlink()
            with self.assertRaisesRegex(ValueError, "fixture"):
                audit_run(output, self.config)

    def _assert_not_assessed_fixture_bundle(self, output):
        import json
        self.assertEqual({path.name for path in output.iterdir()}, set(ARTIFACTS))
        manifest = json.loads((output / "probe_manifest.json").read_text(encoding="utf-8"))
        report = json.loads((output / "probe_report.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["exit_code"], 1)
        self.assertNotEqual(manifest["status"], "fixture_complete")
        self.assertNotEqual(report["status"], "FIXTURE_PASS")
        self.assertNotEqual(report["status"], "PASS_DEVELOPMENT_PROPOSAL")
        self.assertEqual(manifest["source"]["fixture_id"], "task3-fault-case")
        return manifest, report

    def test_partial_resource_io_fatal_replay_and_audit_failures_preserve_bundle(self):
        import json
        from dwr.tasks import probe_v4
        from dwr.tasks.run_support import ResourceGuard

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            class StopAfterOne(ResourceGuard):
                def __init__(self):
                    super().__init__(900, 1024)
                    self.calls = 0

                def check(self):
                    self.calls += 1
                    if self.calls == 6:
                        raise TimeoutError("injected early resource cap")
                    super().check()

            early = root / "early-resource"
            with self.assertRaises(ValueError):
                run_fixture_probe(CONFIG, early, fixture_id="task3-fault-case",
                    _proposal_provider=_fixed_semantic_fixture_provider, _guard=StopAfterOne())
            manifest, report = self._assert_not_assessed_fixture_bundle(early)
            self.assertEqual(report["status"], "NOT_ASSESSED")
            self.assertEqual(manifest["partitions"]["development"]["attempts"], 0)
            self.assertTrue((early / "outcomes.jsonl").read_text(encoding="utf-8").strip())

            io = root / "io-failure"
            with patch("dwr.tasks.probe_v4._dump_jsonl", side_effect=OSError("injected write error")):
                with self.assertRaises(ValueError):
                    run_fixture_probe(CONFIG, io, fixture_id="task3-fault-case",
                        _proposal_provider=_fixed_semantic_fixture_provider)
            _, report = self._assert_not_assessed_fixture_bundle(io)
            self.assertEqual(report["status"], "NOT_ASSESSED")

            task = _fixture_task("source", ("eq",), "A")
            solved = solve(task)
            def fatal_provider(config, partition, index, active_cell):
                profile, cell = active_cell
                raise ProposalFatalError("oracle_witness_disagreement", ProposalFatalContext(
                    partition, index, profile, cell, 6, len(task.givens), len(task.constraints),
                    task, solved.solution, 0))
            fatal = root / "fatal-oracle"
            with self.assertRaises(ValueError):
                run_fixture_probe(CONFIG, fatal, fixture_id="task3-fault-case",
                                  _proposal_provider=fatal_provider)
            manifest, report = self._assert_not_assessed_fixture_bundle(fatal)
            first = json.loads((fatal / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first["state"], "fatal_error")
            self.assertEqual(first["candidate_index"], 0)
            self.assertEqual(manifest["partitions"]["source"]["attempts"], 1)
            self.assertEqual(manifest["partitions"]["development"]["attempts"], 0)
            self.assertEqual(manifest["error"]["candidate_index"], 0)
            self.assertEqual(report["status"], "NOT_ASSESSED")

            class SourceFloorFailurePolicy(_FixturePolicy):
                def source_floor_passes(self, projection):
                    return False

            source_floor = root / "source-floor-failure"
            policy = SourceFloorFailurePolicy("task3-source-floor-fail")
            with self.assertRaisesRegex(ValueError, "full fixture core"):
                run_fixture_probe(CONFIG, source_floor, fixture_id=policy.fixture_id,
                    _proposal_provider=_fixed_semantic_fixture_provider, _policy=policy)
            floor_manifest = json.loads((source_floor / "probe_manifest.json").read_text(encoding="utf-8"))
            floor_report = json.loads((source_floor / "probe_report.json").read_text(encoding="utf-8"))
            self.assertEqual(floor_manifest["status"], "scientific_no_go")
            self.assertEqual(floor_manifest["partitions"]["source"]["attempts"], 12)
            self.assertEqual(floor_manifest["partitions"]["development"]["attempts"], 0)
            self.assertEqual(floor_report["status"], "NO_GO_PROPOSAL")
            self.assertEqual(floor_report["reason"], "SOURCE_TRAIN_FLOOR_FAILED")
            self.assertEqual(floor_report["public_floors"]["source_train"]["status"], "fail")
            self.assertTrue(all(all(value == 0 for value in profile.values())
                                for profile in floor_report["quotas"][
                                    "development_validation_by_profile_cell"].values()))
            self.assertTrue(all(value is None for value in floor_report["quotas"][
                "development_validation_closed"].values()))
            self.assertTrue(all(value["status"] == "not_attempted" and value["values"] is None
                                for value in floor_report["public_floors"][
                                    "development_validation_by_profile"].values()))

            for phase, target in (("replay", "_verify_replay_document"),
                                  ("audit", "_verify_audit_document")):
                output = root / f"{phase}-failure"
                with patch(f"dwr.tasks.probe_v4.{target}", side_effect=ValueError(f"injected {phase} failure")):
                    with self.assertRaises(ValueError):
                        run_fixture_probe(CONFIG, output, fixture_id="task3-fault-case",
                            _proposal_provider=_fixed_semantic_fixture_provider)
                _, report = self._assert_not_assessed_fixture_bundle(output)
                self.assertEqual(report["status"], "NOT_ASSESSED")

    def test_post_terminal_wall_and_rss_caps_revoke_pass(self):
        import json
        from dwr.tasks import probe_v4
        from dwr.tasks.run_support import ResourceGuard, atomic_json as real_atomic_json

        with tempfile.TemporaryDirectory() as temp:
            for cap_kind in ("wall", "rss"):
                with self.subTest(cap_kind=cap_kind):
                    output = Path(temp) / f"post-terminal-{cap_kind}"
                    over = {"value": False}
                    clock = {"value": 0.0}
                    guard = ResourceGuard(1 if cap_kind == "wall" else 900, 1,
                        clock=lambda: clock["value"],
                        memory_reader=lambda: (2 * 1024 * 1024 if over["value"] else 1024))
                    def after_terminal_write(path, document):
                        real_atomic_json(path, document)
                        if Path(path).name == ARTIFACTS[-1] and not over["value"]:
                            over["value"] = True
                            clock["value"] = 2.0
                    with patch("dwr.tasks.probe_v4.atomic_json", side_effect=after_terminal_write):
                        with self.assertRaises(ValueError):
                            run_fixture_probe(CONFIG, output, fixture_id="task3-fault-case",
                                _proposal_provider=_fixed_semantic_fixture_provider, _guard=guard)
                    manifest = json.loads((output / "probe_manifest.json").read_text(encoding="utf-8"))
                    report = json.loads((output / "probe_report.json").read_text(encoding="utf-8"))
                    self.assertEqual(manifest["status"], "not_assessed")
                    self.assertEqual(manifest["error"]["phase"], "post_terminal_write")
                    self.assertEqual(manifest["error"]["reason"], "NOT_ASSESSED_RESOURCE_CAP")
                    self.assertEqual(report["status"], "NOT_ASSESSED")
                    self.assertEqual(report["reason"], "NOT_ASSESSED_RESOURCE_CAP")
                    self.assertIsNotNone(report["four_cohort_proxy"])
                    self.assertEqual(report["public_floors"]["source_train"]["status"], "pass")
                    self.assertTrue(all(value["status"] == "pass" for value in report["public_floors"][
                                        "development_validation_by_profile"].values()))
                    report_ref = manifest["artifacts"]["probe_report.json"]
                    report_bytes = (output / "probe_report.json").read_bytes()
                    self.assertEqual(report_ref["bytes"], len(report_bytes))
                    self.assertEqual(report_ref["sha256"], hashlib.sha256(report_bytes).hexdigest())

    def test_duplicate_does_not_inflate_denominator_or_lineage(self):
        state = ProbeState(self.config)
        first, _ = consume_proposal(self.config, state, "source", 1, ("neq",), "B")
        repeated, _ = consume_proposal(self.config, state, "source", 1, ("neq",), "B")
        self.assertEqual(first["state"], "accepted_for_quota")
        self.assertEqual(repeated["state"], "duplicate_content")
        self.assertFalse(repeated["denominator_counted"])
        self.assertEqual(state.denominators["source_train"]["unique_oracle_groups"], 1)

    def test_public_projection_never_contains_opaque_split_aggregates(self):
        state = ProbeState(self.config)
        consume_proposal(self.config, state, "source", 0, ("eq",), "A")
        projected = project_public_metrics(state)
        encoded = repr(projected)
        self.assertNotIn("source_validation", encoded)
        self.assertNotIn("source_test", encoded)
        self.assertNotIn("development_diagnostic_test", encoded)
        self.assertEqual(set(state.denominators), {"source_train", "development_validation"})

    def test_source_failure_transition_blocks_development(self):
        from dwr.tasks.probe_v4 import source_gate_transition
        self.assertFalse(source_gate_transition("NO_GO_SOURCE_CLOSURE"))
        self.assertFalse(source_gate_transition("NOT_ASSESSED"))
        self.assertTrue(source_gate_transition("PASS_SOURCE_CLOSURE"))

    def test_quota_increment_closes_only_registered_profile_cell(self):
        state = ProbeState(self.config)
        profile, cell = ("eq",), "A"
        target = self.config.quota_for("source", profile, cell)
        state.quota_counts["source"][profile][cell] = target - 1
        closed = state.add_quota("source", profile, cell)
        self.assertTrue(closed)
        self.assertEqual(state.quota_counts["source"][profile][cell], target)
        self.assertFalse(state.add_quota("source", profile, cell))
        self.assertEqual(state.quota_counts["source"][profile][cell], target)

    def test_fixed_candidate_stream_replay_and_independent_audit_integration(self):
        state = ProbeState(self.config)
        outcomes = []
        for index in range(4):
            from dwr.tasks.generator_probe_v4 import next_active_cell
            profile, cell = next_active_cell(self.config, "source", index, state.remaining("source"))
            outcome, _ = consume_proposal(self.config, state, "source", index, profile, cell)
            outcomes.append(outcome)
        replayed, denominator_hash = replay_prefix(
            self.config, outcomes, state.records, ResourceGuard(30, 1024))
        self.assertEqual(replayed.outcomes, outcomes)
        self.assertEqual(replayed.records, state.records)
        self.assertEqual(len(denominator_hash), 64)
        audit = audit_records(state.records, outcomes, self.config)
        self.assertEqual(audit["source_train"]["unique_oracle_groups"], 2)
        self.assertEqual(audit["source_train"]["accepted_groups"], 1)
        self.assertIsNone(audit["four_cohort_proxy"])

    def test_replay_and_semantic_audit_reject_corrupted_claims(self):
        state = ProbeState(self.config)
        from dwr.tasks.generator_probe_v4 import next_active_cell
        outcomes = []
        for index in range(4):
            profile, cell = next_active_cell(self.config, "source", index, state.remaining("source"))
            row, _ = consume_proposal(self.config, state, "source", index, profile, cell)
            outcomes.append(row)
        corrupted = [dict(row) for row in outcomes]
        corrupted[2]["denominator_counted"] = False
        with self.assertRaises(ValueError):
            audit_records(state.records, corrupted, self.config)

    def test_independent_audit_rejects_mutated_interaction_evidence(self):
        state = ProbeState(self.config)
        outcome, _ = consume_proposal(self.config, state, "source", 2, ("eq",), "C")
        corrupted = dict(outcome)
        corrupted["private_eligibility_or_null"] = dict(outcome["private_eligibility_or_null"])
        corrupted["private_eligibility_or_null"]["removal_solution_counts"] = [1] * 7
        with self.assertRaisesRegex(ValueError, "eligibility evidence"):
            audit_records(state.records, [corrupted], self.config)

    def test_independent_audit_rejects_misbound_source_deletion_count_positions(self):
        state = ProbeState(self.config)
        from dwr.tasks.probe_v4 import consume_proposal
        provider = lambda config, partition, index, active: _fixed_semantic_fixture_provider(
            config, partition, index, active)
        outcome, record = consume_proposal(self.config, state, "source", 0,
                                           ("eq",), "C", _proposal_provider=provider)
        self.assertIsNotNone(record)
        counts = outcome["private_eligibility_or_null"]["removal_solution_counts"]
        left, right = next((i, j) for i in range(len(counts)) for j in range(i + 1, len(counts))
                           if counts[i] != counts[j])
        corrupted = dict(outcome)
        corrupted["private_eligibility_or_null"] = dict(outcome["private_eligibility_or_null"])
        corrupted_counts = list(counts)
        corrupted_counts[left], corrupted_counts[right] = corrupted_counts[right], corrupted_counts[left]
        corrupted["private_eligibility_or_null"]["removal_solution_counts"] = corrupted_counts
        with self.assertRaisesRegex(ValueError, "eligibility evidence"):
            audit_records([record], [corrupted], self.config)

    def test_incomplete_failure_still_writes_typed_replay_and_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            manifest = {"source": {"commit": "a" * 40, "dirty": False,
                "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(), "checkout_root": "fixture"},
                "producer": [], "partitions": {"source": {"attempts": 0,
                    "quota_targets": _profile_maps(self.config, "source", ProbeState(self.config))[0],
                    "quota_counts": _profile_maps(self.config, "source", ProbeState(self.config))[1]},
                    "development": {"attempts": 0}}}
            guard = ResourceGuard(30, 1024)
            _ensure_failure_artifacts(output, self.config, manifest["source"], [],
                                      ProbeState(self.config), manifest, guard)
            replay = __import__("json").loads((output / "candidate_replay.json").read_text(encoding="utf-8"))
            audit = __import__("json").loads((output / "independent_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(replay["status"], "NOT_ASSESSED")
            self.assertEqual(audit["status"], "NOT_ASSESSED")

    def test_replay_identity_dependencies_are_checked_independently(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "outcomes.jsonl").write_bytes(b"")
            (output / "accepted_records.jsonl").write_bytes(b"")
            source = {"commit": "a" * 40, "dirty": False,
                      "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(),
                      "checkout_root": "fixture-root"}
            producer = [{"path": path, "sha256": "b" * 64}
                        for path in self.config.raw_config["producer_hash_paths"]]
            state = ProbeState(self.config)
            targets = {p: _profile_maps(self.config, p, state)[0] for p in ("source", "development")}
            counts = {p: _profile_maps(self.config, p, state)[1] for p in ("source", "development")}
            replay = {"replay_version": "dwr-v4-candidate-replay-v1", "status": "PASS_REPLAY",
                "config_sha256": self.config.config_sha256, "source": source, "producer": producer,
                "partition_attempts": {"source": 0, "development": 0},
                "outcome_prefix_bytes": 0, "outcome_prefix_sha256": hashlib.sha256(b"").hexdigest(),
                "record_prefix_bytes": 0, "record_prefix_sha256": hashlib.sha256(b"").hexdigest(),
                "quota_targets": targets, "quota_counts": counts,
                "quota_closed": {"source": False,
                    "development": {p: False for p in targets["development"]}},
                "denominator_projection_sha256": "c" * 64}
            _verify_replay_document(replay, output, self.config, source, producer)
            for dependency in ("config", "source", "producer"):
                altered = dict(replay)
                if dependency == "config":
                    altered["config_sha256"] = "d" * 64
                elif dependency == "source":
                    altered["source"] = {**source, "commit": "e" * 40}
                else:
                    altered["producer"] = [*producer[:-1], {**producer[-1], "sha256": "f" * 64}]
                with self.subTest(dependency=dependency), self.assertRaises(ValueError):
                    _verify_replay_document(altered, output, self.config, source, producer)
            malformed = dict(replay)
            malformed_targets = {part: dict(profiles) for part, profiles in targets.items()}
            malformed_targets["source"] = dict(malformed_targets["source"])
            malformed_targets["source"]["eq"] = dict(malformed_targets["source"]["eq"])
            malformed_targets["source"]["eq"]["A"] += 1
            malformed["quota_targets"] = malformed_targets
            with self.assertRaises(ValueError):
                _verify_replay_document(malformed, output, self.config, source, producer)

    def test_unpatched_source_only_driver_replay_and_independent_audit_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "fixture-run"
            label = run_source_only_fixture(CONFIG, output, fixture_id="task3-fixed-source-v1")
            self.assertEqual(label, "FIXTURE_SOURCE_ONLY_PASS")
            self.assertEqual({path.name for path in output.iterdir()}, {*ARTIFACTS, "fixture_identity.json"})
            self.assertEqual(state := __import__("json").loads(
                (output / "source_closure.json").read_text(encoding="utf-8"))["status"],
                "PASS_SOURCE_CLOSURE")
            manifest = __import__("json").loads((output / "probe_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source"]["fixture_id"], "task3-fixed-source-v1")
            self.assertEqual(manifest["partitions"]["development"]["attempts"], 0)
            report = __import__("json").loads((output / "probe_report.json").read_text(encoding="utf-8"))
            self.assertTrue(all(value["status"] == "not_attempted" and value["values"] is None
                                for value in report["public_floors"][
                                    "development_validation_by_profile"].values()))
            self.assertEqual(report["public_floors"]["four_cohort_proxy"],
                             {"status": "not_assessed", "values": None})
            self.assertIsNone(report["four_cohort_proxy"])
            self.assertEqual(report["g2_status"], "not_assessed")
            with self.assertRaises(ValueError):
                _verify_initial_binding(__import__("json").loads(
                                            (output / "probe_manifest.initial.json").read_text(encoding="utf-8")),
                                        self.config, manifest["source"],
                                        _producer_identity(self.config, Path(__file__).parents[1]), CONFIG)
            with self.assertRaises(ValueError):
                run_probe(CONFIG, output.parent / "production-run")
            with self.assertRaisesRegex(ValueError, "fixture"):
                audit_run(output, self.config)

    def test_existing_output_directory_is_rejected_without_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "already-used"
            output.mkdir()
            sentinel = output / "owner-file.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "resume is forbidden"):
                run_source_only_fixture(CONFIG, output, fixture_id="existing-dir-rejection")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertEqual({path.name for path in output.iterdir()}, {"owner-file.txt"})

    def test_original_diagnostic_outcome_stays_outcome_only_and_public_denominator(self):
        policy = _FixturePolicy("diagnostic-privacy")
        def diagnostic_provider(config, partition, index, active):
            profile, cell = active
            task = parse_task_json(serialize_task(_fixture_task(
                partition, profile, cell, eq_neq_alternative=0)))
            oracle = enumerate_solutions(task)
            solved = solve(task)
            evidence = _independent_interaction_projection(task, partition)
            return ProposalResult(partition, index, profile, cell, 6, len(task.givens),
                len(task.constraints), None, task, solved.solution, 1, oracle.solutions[0],
                solved.stats, evidence)
        state = ProbeState(self.config, _fixture_policy=policy)
        outcome, record = consume_proposal(self.config, state, "development", 0,
            ("eq", "neq"), "A", _proposal_provider=diagnostic_provider)
        self.assertEqual(outcome["state"], "retained_diagnostic")
        self.assertEqual(outcome["split_or_null"], "development_diagnostic_test")
        self.assertFalse(outcome["denominator_counted"])
        self.assertIsNone(record)
        projected = project_public_metrics(state)
        self.assertNotIn("development_diagnostic_test", repr(projected))
        self.assertEqual(projected["development_validation_by_profile"]["eq/neq"][
            "unique_oracle_groups"], 0)


if __name__ == "__main__":
    unittest.main()
