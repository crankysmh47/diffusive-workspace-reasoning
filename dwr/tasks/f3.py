"""Small, strict linear algebra over the finite field F3."""
from __future__ import annotations

from typing import Callable, MutableMapping


def _matrix(rows, width: int) -> list[list[int]]:
    if type(width) is not int or width < 0:
        raise ValueError("width must be a nonnegative integer")
    if type(rows) not in (tuple, list):
        raise TypeError("rows must be a tuple or list")
    matrix = []
    for row in rows:
        if type(row) not in (tuple, list) or len(row) != width:
            raise ValueError("rows must be rectangular with the declared width")
        if any(type(value) is not int or value not in (0, 1, 2) for value in row):
            raise ValueError("matrix entries must be integers in F3")
        matrix.append(list(row))
    return matrix


def _rref(rows, width: int, check_budget: Callable[[], None] | None,
          counters: MutableMapping[str, int] | None) -> tuple[list[list[int]], tuple[int, ...]]:
    matrix = _matrix(rows, width)
    if check_budget is not None and not callable(check_budget):
        raise TypeError("check_budget must be callable or None")
    if counters is not None and not hasattr(counters, "__setitem__"):
        raise TypeError("counters must be a mutable mapping or None")
    def tick(n=1):
        if counters is not None:
            counters["row_operations"] = counters.get("row_operations", 0) + n
        if check_budget is not None:
            check_budget()
    pivots = []
    pivot_row = 0
    for col in range(width):
        if check_budget is not None:
            check_budget()
        found = next((r for r in range(pivot_row, len(matrix)) if matrix[r][col]), None)
        if found is None:
            continue
        if found != pivot_row:
            matrix[pivot_row], matrix[found] = matrix[found], matrix[pivot_row]
            tick()
        pivot = matrix[pivot_row][col]
        if pivot == 2:
            matrix[pivot_row] = [(2 * x) % 3 for x in matrix[pivot_row]]
            tick()
        for r in range(len(matrix)):
            if r == pivot_row or matrix[r][col] == 0:
                continue
            factor = matrix[r][col]
            matrix[r] = [(x - factor * y) % 3 for x, y in zip(matrix[r], matrix[pivot_row])]
            tick()
        pivots.append(col)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    if counters is not None:
        counters["rank_calls"] = counters.get("rank_calls", 0) + 1
    return matrix, tuple(pivots)


def rank(rows, width, *, check_budget=None, counters=None) -> int:
    """Return the row rank, using leftmost pivots and canonical RREF."""
    return len(_rref(rows, width, check_budget, counters)[1])


def nullspace(rows, width, *, check_budget=None, counters=None):
    """Return a canonical basis of the right nullspace, ordered by free column."""
    reduced, pivots = _rref(rows, width, check_budget, counters)
    free = tuple(col for col in range(width) if col not in pivots)
    result = []
    for free_col in free:
        vector = [0] * width
        vector[free_col] = 1
        for row_index, pivot_col in enumerate(pivots):
            vector[pivot_col] = (-reduced[row_index][free_col]) % 3
        result.append(tuple(vector))
    return tuple(result)
