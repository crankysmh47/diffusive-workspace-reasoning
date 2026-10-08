import unittest

from dwr.tasks.run_support import ResourceGuard, remaining_global_seconds


class BudgetV3Tests(unittest.TestCase):
    def test_generation_elapsed_is_removed_once_from_combined_budget(self):
        remaining = remaining_global_seconds({"generation_elapsed_seconds": 247.25}, 600.0)
        self.assertEqual(remaining, 352.75)
        guard = ResourceGuard(600.0, 1024, clock=lambda: 5.0, memory_reader=lambda: 0)
        guard.started = 0.0
        guard.wall_seconds = min(guard.wall_seconds, remaining)
        self.assertEqual(guard.wall_seconds, 352.75)
        self.assertEqual(guard.elapsed, 5.0)
        # The generation time is not deducted again during a prefix replay/second check.
        guard.wall_seconds = min(guard.wall_seconds, remaining)
        self.assertEqual(guard.wall_seconds, 352.75)

    def test_terminal_budget_check_fails_after_guard_limit(self):
        clock_values = iter((0.0, 0.0, 1.0))
        guard = ResourceGuard(0.5, 1024, clock=lambda: next(clock_values), memory_reader=lambda: 0)
        guard.started = 0.0
        guard.check()
        with self.assertRaises(TimeoutError):
            guard.check()
