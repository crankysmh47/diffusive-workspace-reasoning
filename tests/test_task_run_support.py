import json
import hashlib
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from dwr.tasks.dataset import GenerationLimits, run_dataset, source_identity
from dwr.tasks.generator import CandidateOutcome, generate_candidate
from dwr.tasks.run_support import ConfigError, ResourceGuard, atomic_json, load_config, remaining_global_seconds, strict_json_loads
from dwr.tasks.schema import Constraint, ConstraintTask, GivenAssignment, RelationKind, serialize_model_input
from dwr.tasks.solver import solve


class RunSupportTests(unittest.TestCase):
    def test_atomic_json_retries_three_windows_replace_conflicts(self):
        import dwr.tasks.run_support as support
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'manifest.json'
            real_replace = Path.replace
            attempts = []
            retryable_codes = iter((5, 32, 33))

            def replace_with_conflicts(source, destination):
                attempts.append((source, destination))
                if len(attempts) < 4:
                    error = PermissionError('replacement conflict')
                    error.winerror = next(retryable_codes)
                    raise error
                return real_replace(source, destination)

            with patch.object(support.os, 'name', 'nt'), \
                 patch.object(Path, 'replace', replace_with_conflicts), \
                 patch.object(support.time, 'sleep') as sleep:
                atomic_json(target, {'b': 2, 'a': '✓'})

            self.assertEqual(len(attempts), 4)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.01, 0.02, 0.04])
            self.assertEqual(target.read_bytes(), '{"a":"✓","b":2}\n'.encode('utf-8'))
            self.assertFalse(target.with_name(target.name + '.tmp').exists())

    def test_atomic_json_stops_after_four_windows_conflicts_and_preserves_error(self):
        import dwr.tasks.run_support as support
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'manifest.json'
            attempts = []
            errors = []

            def always_conflict(source, destination):
                error = PermissionError('permanent denial or sharing conflict')
                error.winerror = 5
                attempts.append((source, destination))
                errors.append(error)
                raise error

            with patch.object(support.os, 'name', 'nt'), \
                 patch.object(Path, 'replace', always_conflict), \
                 patch.object(support.time, 'sleep') as sleep:
                with self.assertRaises(PermissionError) as caught:
                    atomic_json(target, {'status': 'incomplete'})

            self.assertIs(caught.exception, errors[-1])
            self.assertEqual(len(attempts), 4)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.01, 0.02, 0.04])
            self.assertTrue(target.with_name(target.name + '.tmp').exists())
            self.assertFalse(target.exists())

    def test_atomic_json_does_not_retry_non_windows_or_unrelated_errors(self):
        import dwr.tasks.run_support as support
        cases = (('posix', 32), ('nt', 13), ('nt', None))
        for os_name, winerror in cases:
            with self.subTest(os_name=os_name, winerror=winerror), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'manifest.json'
                attempts = []
                error = PermissionError(f'winerror={winerror}')
                if winerror is not None:
                    error.winerror = winerror

                def fail_replace(source, destination):
                    attempts.append((source, destination))
                    raise error

                with patch.object(support.os, 'name', os_name), \
                     patch.object(Path, 'replace', fail_replace), \
                     patch.object(support.time, 'sleep') as sleep:
                    with self.assertRaises(PermissionError) as caught:
                        atomic_json(target, {'status': 'incomplete'})

                self.assertIs(caught.exception, error)
                self.assertEqual(len(attempts), 1)
                sleep.assert_not_called()

    def test_atomic_json_does_not_retry_temporary_write_failure(self):
        import dwr.tasks.run_support as support
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'manifest.json'
            with patch.object(support.os, 'name', 'nt'), \
                 patch.object(Path, 'write_text', side_effect=OSError('disk write failed')) as write_text, \
                 patch.object(Path, 'replace') as replace_path, \
                 patch.object(support.time, 'sleep') as sleep:
                with self.assertRaisesRegex(OSError, 'disk write failed'):
                    atomic_json(target, {'status': 'incomplete'})
            write_text.assert_called_once()
            replace_path.assert_not_called()
            sleep.assert_not_called()

    def test_atomic_json_retry_time_remains_inside_generation_guard_accounting(self):
        import dwr.tasks.dataset as dataset
        import dwr.tasks.run_support as support
        config = load_config('configs/tasks/constraint_v1.json')
        limits = GenerationLimits({'source': 0, 'development': 0, 'final': 0}, 1, 1, 1, 8)
        elapsed = {'value': 0.0}
        guard = ResourceGuard(1, 8, clock=lambda: elapsed['value'], memory_reader=lambda: 1)
        original_replace = Path.replace
        replace_attempts = {'value': 0}

        def one_terminal_conflict(source, destination):
            replace_attempts['value'] += 1
            if replace_attempts['value'] == 2:
                error = PermissionError('replacement conflict')
                error.winerror = 32
                raise error
            return original_replace(source, destination)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            with patch.object(support.os, 'name', 'nt'), \
                 patch.object(Path, 'replace', one_terminal_conflict), \
                 patch.object(support.time, 'sleep', side_effect=lambda delay: elapsed.__setitem__('value', elapsed['value'] + delay)):
                status = dataset.run_dataset(config, out, ['fixture'], limits=limits, guard=guard,
                                             source={'commit': 'fixture'})
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(status, 0)
            self.assertAlmostEqual(manifest['generation_elapsed_seconds'], guard.elapsed)
            self.assertAlmostEqual(manifest['generation_elapsed_seconds'], 0.01)
            self.assertEqual(manifest['accounting_end_boundary'],
                             'post_first_terminal_manifest_write_resource_check')

    @unittest.skipUnless(__import__('os').name == 'nt', 'Windows ctypes cache regression')
    def test_windows_memory_monitor_types_and_pointer_cache_stabilize_after_warmup(self):
        import ctypes
        import gc
        import dwr.tasks.run_support as support

        support.resident_bytes_windows()
        gc.collect()

        def relevant_pointer_cache():
            return {key: value for key, value in ctypes._pointer_type_cache.items()
                    if isinstance(key, tuple) and key and
                    getattr(key[0], '__name__', None) == 'PROCESS_MEMORY_COUNTERS'}

        before = relevant_pointer_cache()
        before_type = support._WINDOWS_PROCESS_MEMORY_COUNTERS
        before_pointer = support._WINDOWS_PROCESS_MEMORY_COUNTERS_POINTER
        for index in range(252):
            support.resident_bytes_windows()
            if index == 125:
                gc.collect()
        after = relevant_pointer_cache()
        self.assertIs(support._WINDOWS_PROCESS_MEMORY_COUNTERS, before_type)
        self.assertIs(support._WINDOWS_PROCESS_MEMORY_COUNTERS_POINTER, before_pointer)
        self.assertEqual(after, before)

    def test_duplicate_and_unknown_config_keys_fail(self):
        with self.assertRaises(ConfigError):
            strict_json_loads('{"x":1,"x":2}')
        with self.assertRaises(ConfigError):
            strict_json_loads('{"x":NaN}')
        for text in ('{"outer":{"elapsed":1e999}}', '{"values":[-1e999]}'):
            with self.assertRaises(ConfigError):
                strict_json_loads(text)
        config = json.loads(Path('configs/tasks/constraint_v1.json').read_text(encoding='utf-8'))
        config['extra'] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'c.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            with self.assertRaises(ConfigError):
                load_config(path)
        config = json.loads(Path('configs/tasks/constraint_v1.json').read_text(encoding='utf-8'))
        config['generation']['unknown'] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'c.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_seed_keeps_full_unsigned_64_bit_range(self):
        config = json.loads(Path('configs/tasks/constraint_v1.json').read_text(encoding='utf-8'))
        config['seed'] = 2**64 - 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'c.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            self.assertEqual(load_config(path).resolved['seed'], 2**64 - 1)

    def test_seed_rejects_bool_and_float(self):
        for seed in (True, 1.0):
            config = json.loads(Path('configs/tasks/constraint_v1.json').read_text(encoding='utf-8'))
            config['seed'] = seed
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'c.json'
                path.write_text(json.dumps(config), encoding='utf-8')
                with self.assertRaises(ConfigError):
                    load_config(path)

    def test_registered_baseline_cannot_self_authorize_mutation(self):
        import dwr.tasks.run_support as support
        original_file = support.__file__
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dwr' / 'tasks').mkdir(parents=True)
            registered = root / 'configs' / 'tasks'
            registered.mkdir(parents=True)
            source = Path('configs/tasks/constraint_v1.json').read_bytes()
            (registered / 'constraint_v1.json').write_bytes(source + b' ')
            candidate = root / 'candidate.json'
            candidate.write_bytes(source)
            support.__file__ = str(root / 'dwr' / 'tasks' / 'run_support.py')
            try:
                with self.assertRaisesRegex(ConfigError, 'reviewed baseline'):
                    load_config(candidate)
            finally:
                support.__file__ = original_file

    def test_guard_fails_closed_and_caps(self):
        ticks = iter((0.0, 0.25, 0.5, 1.1))
        guard = ResourceGuard(1.0, 8, clock=lambda: next(ticks), memory_reader=lambda: 4)
        guard.check()
        guard.check()
        with self.assertRaises(TimeoutError):
            guard.check()
        with self.assertRaises(RuntimeError):
            ResourceGuard(1, 8, clock=lambda: 0, memory_reader=lambda: (_ for _ in ()).throw(OSError())).check()

    def test_source_identity_uses_implementation_checkout(self):
        import os
        old = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                identity = source_identity()
            finally:
                os.chdir(old)
        self.assertEqual(Path(identity['checkout_root']), Path(__file__).resolve().parents[1])

    def test_generation_partition_stops_at_exact_quota_and_replays_identically(self):
        config = load_config('configs/tasks/constraint_v1.json')
        task = ConstraintTask('constraint-task-v1', 3, ('v0', 'v1', 'v2'),
                              (Constraint(RelationKind.EQ, ('v0', 'v1')),
                               Constraint(RelationKind.EQ, ('v1', 'v2'))),
                              (GivenAssignment('v0', 1),))
        solution = solve(task).solution
        outcome = CandidateOutcome('source', 0, 3, ('eq',), task, solution, solution, solve(task), None)
        limits = GenerationLimits({'source': 1, 'development': 0, 'final': 0}, 9, 1, 10, 8)
        outputs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / 'run'
                seen = []
                status = run_dataset(config, out, ['fixture'], limits=limits,
                                     candidate_fn=lambda *a, **k: (seen.append(a[2]) or outcome),
                                     guard=ResourceGuard(10, 8, clock=lambda: 0, memory_reader=lambda: 1),
                                     source={'commit': 'fixture'})
                self.assertEqual(status, 0)
                self.assertEqual(seen, [0])
                record = json.loads((out / 'records.jsonl').read_text(encoding='utf-8'))
                self.assertEqual(record['model_input'], serialize_model_input(task))
                self.assertEqual(record['candidate_index'], 0)
                outputs.append(((out / 'outcomes.jsonl').read_bytes(), (out / 'records.jsonl').read_bytes()))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(remaining_global_seconds({'generation_elapsed_seconds': 123.5}), 476.5)
        self.assertEqual(remaining_global_seconds({'generation_elapsed_seconds': 600}), 0)

    def test_partial_cap_records_are_retained_and_manifest_precedes_candidate(self):
        config = load_config('configs/tasks/constraint_v1.json')
        task = ConstraintTask('constraint-task-v1', 3, ('v0', 'v1', 'v2'),
                              (Constraint(RelationKind.EQ, ('v0', 'v1')),
                               Constraint(RelationKind.EQ, ('v1', 'v2'))),
                              (GivenAssignment('v0', 1),))
        sol = solve(task).solution
        accepted = CandidateOutcome('source', 0, 3, ('eq',), task, sol, sol, solve(task), None)
        seen = []
        readings = iter((0, 0, 9 * 1024 * 1024))
        guard = ResourceGuard(10, 8, clock=lambda: 0, memory_reader=lambda: next(readings))
        limits = GenerationLimits({'source': 1, 'development': 0, 'final': 0}, 1, 1, 10, 8)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            def candidate_fn(*args, **kwargs):
                manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(manifest['status'], 'running')
                seen.append(True)
                return accepted
            status = run_dataset(config, out, ['fixture'], limits=limits, candidate_fn=candidate_fn,
                                 guard=guard, source={'commit': 'fixture'})
            self.assertEqual(status, 1)
            self.assertTrue(seen)
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'incomplete')
            self.assertEqual(manifest['partitions']['development']['status'], 'not_attempted')
            self.assertIsNotNone(manifest['finished_utc'])
            self.assertEqual(manifest['terminal_status_utc'], manifest['finished_utc'])
            rows = (out / 'records.jsonl').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(rows), 1)
            record = json.loads(rows[0])
            self.assertEqual(record['model_input_sha256'], hashlib.sha256(record['model_input'].encode()).hexdigest())
            self.assertEqual(manifest['artifacts']['records.jsonl']['sha256'], hashlib.sha256((out / 'records.jsonl').read_bytes()).hexdigest())
            self.assertNotIn('solution', json.loads(record['model_input'])['task'])

    def test_unavailable_memory_metric_is_null_in_partial_manifest(self):
        config = load_config('configs/tasks/constraint_v1.json')
        limits = GenerationLimits({'source': 1, 'development': 0, 'final': 0}, 1, 1, 10, 8)
        guard = ResourceGuard(10, 8, clock=lambda: 0,
                              memory_reader=lambda: (_ for _ in ()).throw(OSError('unavailable')))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            status = run_dataset(config, out, ['fixture'], limits=limits, guard=guard,
                                 source={'commit': 'fixture'})
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(status, 1)
            self.assertIsNone(manifest['peak_resident_bytes'])

    def test_setup_consuming_global_cap_blocks_candidates_and_completion(self):
        config = load_config('configs/tasks/constraint_v1.json')
        limits = GenerationLimits({'source': 0, 'development': 0, 'final': 0}, 1, 1, 0.01, 1024)
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            def delayed_source():
                time.sleep(0.03)
                return {'commit': 'fixture'}
            with patch('dwr.tasks.dataset.source_identity', delayed_source):
                status = run_dataset(config, out, ['fixture'], limits=limits,
                                     candidate_fn=lambda *a, **k: calls.append(a))
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(calls, [])
            self.assertEqual(status, 1)
            self.assertEqual(manifest['status'], 'incomplete')
            self.assertGreaterEqual(manifest['generation_elapsed_seconds'], 0.01)
            self.assertTrue(all(manifest['partitions'][part]['status'] == 'not_attempted'
                                and manifest['partitions'][part]['attempts'] == 0
                                for part in ('source', 'development', 'final')))
            self.assertEqual(set(manifest['artifacts']), {'outcomes.jsonl', 'records.jsonl'})

    def test_finalization_failure_preserves_primary_resource_cap(self):
        config = load_config('configs/tasks/constraint_v1.json')
        limits = GenerationLimits({'source': 1, 'development': 0, 'final': 0}, 1, 1, 10, 8)
        readings = iter((9 * 1024 * 1024, None))
        def reader():
            value = next(readings)
            if value is None:
                raise OSError('final measurement unavailable')
            return value
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            status = run_dataset(config, out, ['fixture'], limits=limits,
                                 candidate_fn=lambda *a, **k: self.fail('cap must block candidate work'),
                                 guard=ResourceGuard(10, 8, clock=lambda: 0, memory_reader=reader),
                                 source={'commit': 'fixture'})
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(status, 1)
            self.assertEqual(manifest['error']['message'], 'resident memory cap exceeded')
            self.assertEqual(manifest['error']['phase'], 'generation')
            self.assertEqual(manifest['finalization_errors'][0]['message'], 'resource measurement failed closed')
            self.assertEqual(manifest['finalization_errors'][0]['phase'], 'finalization')
            self.assertTrue(all(manifest['partitions'][part]['status'] == 'not_attempted'
                                for part in ('development', 'final')))

    def test_successful_terminal_write_refreshes_budget_and_finalization_elapsed(self):
        import dwr.tasks.dataset as dataset
        config = load_config('configs/tasks/constraint_v1.json')
        limits = GenerationLimits({'source': 0, 'development': 0, 'final': 0}, 1, 1, 1, 8)
        elapsed = {'value': 0.0, 'delayed': False}
        guard = ResourceGuard(1, 8, clock=lambda: elapsed['value'], memory_reader=lambda: 1)
        real_write = dataset.atomic_json

        def injected_write(path, value):
            real_write(path, value)
            if 'artifacts' in value and not elapsed['delayed']:
                elapsed['value'] += 0.05
                elapsed['delayed'] = True

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            with patch('dwr.tasks.dataset.atomic_json', injected_write):
                status = dataset.run_dataset(config, out, ['fixture'], limits=limits,
                                             guard=guard, source={'commit': 'fixture'})
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(status, 0)
            self.assertAlmostEqual(manifest['generation_elapsed_seconds'], guard.elapsed)
            self.assertAlmostEqual(manifest['finalization_elapsed_seconds'], 0.05)
            self.assertAlmostEqual(remaining_global_seconds(manifest, 1), 0.95)
            self.assertEqual(manifest['accounting_end_boundary'], 'post_first_terminal_manifest_write_resource_check')
            self.assertIn('ultimate metadata flush tail', manifest['accounting_note'])

    def test_real_nonunique_rejection_serializes_through_orchestrator(self):
        config = load_config('configs/tasks/constraint_v1.json')
        rejection = replace(generate_candidate(config.resolved['seed'], 'source', 7), candidate_index=0)
        accepted_task = ConstraintTask('constraint-task-v1', 3, ('v0', 'v1', 'v2'),
                                       (Constraint(RelationKind.EQ, ('v0', 'v1')),
                                        Constraint(RelationKind.EQ, ('v1', 'v2'))),
                                       (GivenAssignment('v0', 1),))
        solution = solve(accepted_task).solution
        accepted = CandidateOutcome('source', 1, 3, ('eq',), accepted_task, solution, solution,
                                    solve(accepted_task), None)
        queue = iter((rejection, accepted))
        limits = GenerationLimits({'source': 1, 'development': 0, 'final': 0}, 2, 1, 10, 8)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            status = run_dataset(config, out, ['fixture'], limits=limits,
                                 candidate_fn=lambda *a, **k: next(queue),
                                 guard=ResourceGuard(10, 8, clock=lambda: 0, memory_reader=lambda: 1),
                                 source={'commit': 'fixture'})
            self.assertEqual(status, 0)
            outcomes = [json.loads(line) for line in (out / 'outcomes.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(outcomes[0]['rejection_reason'], 'oracle_non_unique')
            self.assertIsNone(outcomes[0]['solver_stats'])
            self.assertIsNotNone(outcomes[0]['task'])
            self.assertEqual(len((out / 'records.jsonl').read_text(encoding='utf-8').splitlines()), 1)

    def test_existing_output_is_refused(self):
        config = load_config('configs/tasks/constraint_v1.json')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'existing'
            out.mkdir()
            with self.assertRaises(FileExistsError):
                run_dataset(config, out, ['fixture'])


if __name__ == '__main__':
    unittest.main()
