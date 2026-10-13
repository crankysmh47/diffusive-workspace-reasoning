import itertools
import unittest

from dwr.tasks.f3 import nullspace, rank


class F3Tests(unittest.TestCase):
    def test_small_matrices_match_independent_enumeration(self):
        for width in range(4):
            rows_pool = tuple(itertools.product(range(3), repeat=width))
            matrices = [()]
            matrices.extend((row,) for row in rows_pool)
            matrices.extend(itertools.product(rows_pool, repeat=2))
            for rows in matrices:
                basis = nullspace(rows, width)
                vectors = tuple(v for v in itertools.product(range(3), repeat=width)
                                if all(sum(a*b for a, b in zip(row, v)) % 3 == 0 for row in rows))
                self.assertEqual(len(vectors), 3 ** (width - rank(rows, width)))
                span = {tuple(sum(coefficient * vector[col] for coefficient, vector in zip(coefficients, basis)) % 3
                              for col in range(width))
                        for coefficients in itertools.product(range(3), repeat=len(basis))}
                self.assertEqual(set(vectors), span)

    def test_strict_shape_and_integer_validation(self):
        for rows, width in [([[1, 0]], 1), ([[1.0]], 1), ([[True]], 1), ([[3]], 1)]:
            with self.subTest(rows=rows):
                with self.assertRaises((TypeError, ValueError)):
                    rank(rows, width)

    def test_budget_and_counters_are_live(self):
        calls, counters = [], {}
        self.assertEqual(rank(((1, 1), (2, 2)), 2,
                              check_budget=lambda: calls.append(1), counters=counters), 1)
        self.assertGreater(len(calls), 0)
        self.assertEqual(counters["rank_calls"], 1)
        self.assertGreaterEqual(counters["row_operations"], 1)

    def test_nullspace_budget_can_interrupt_during_rref(self):
        class BudgetExpired(RuntimeError):
            pass
        calls = 0
        def check_budget():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise BudgetExpired
        with self.assertRaises(BudgetExpired):
            nullspace(((2, 1), (1, 0)), 2, check_budget=check_budget)
        self.assertEqual(calls, 2)

    def test_nullspace_row_operations_are_instrumented(self):
        from dwr.tasks.certificate_rank_generator import _Work
        work = _Work(None)
        self.assertEqual(work.nullspace(((1, 1), (2, 2)), 2), ((2, 1),))
        self.assertEqual(work.rank_calls, 1)
        self.assertGreater(work.row_operations, 0)


if __name__ == "__main__":
    unittest.main()
