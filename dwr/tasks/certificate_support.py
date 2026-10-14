"""Frozen support policies for rank-certificate audits.

This module consumes audit metadata only. It never creates tasks or changes the
task constructor, prompt, oracle, or scientific eligibility decision.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

LEGACY_CELLS = ("A", "B", "C", "D")
FIVE_STRATA = ("A2", "A3", "B2", "C2", "D2")
FIVE_SCHEME = "five-fixed-strata-v1"
_FIVE_DESCRIPTORS = (
    ("A2", "A", 2, 4), ("A3", "A", 3, 4), ("B2", "B", 2, 4),
    ("C2", "C", 2, 4), ("D2", "D", 2, 4),
)


def _policy(config: dict) -> tuple[tuple[str, ...], bool, int, float]:
    support = config.get("support")
    if not isinstance(support, dict):
        raise ValueError("config.support must be an object")
    if "scheme" not in support:
        if "strata" in support:
            raise ValueError("legacy support cannot declare strata without a scheme")
        names, five, min_count, min_mass = LEGACY_CELLS, False, support.get("cell_min"), support.get("cohort_mass_min")
    elif support.get("scheme") == FIVE_SCHEME:
        descriptors = support.get("strata")
        expected = [dict(id=i, cell=c, first_solution_node_bin=n, prompt_byte_bin=b)
                    for i, c, n, b in _FIVE_DESCRIPTORS]
        if descriptors != expected:
            raise ValueError("five-stratum descriptors do not match the frozen policy")
        names, five, min_count, min_mass = FIVE_STRATA, True, support.get("cell_min"), support.get("cohort_mass_min")
    else:
        raise ValueError("unknown support scheme")
    if type(min_count) is not int or min_count < 0:
        raise ValueError("support.cell_min must be a nonnegative integer")
    if type(min_mass) not in (int, float) or not 0 <= min_mass <= 1:
        raise ValueError("support.cohort_mass_min must be between zero and one")
    return names, five, min_count, float(min_mass)


def support_stratum(row: dict, config: dict) -> str | None:
    names, five, _, _ = _policy(config)
    cell, nodes, byte_count = row.get("cell"), row.get("first_solution_nodes"), row.get("prompt_bytes")
    if type(nodes) is not int or type(byte_count) is not int or byte_count <= 1023:
        return None
    if five:
        if cell == "A" and 4 <= nodes <= 7:
            return "A2"
        if cell == "A" and 8 <= nodes <= 15:
            return "A3"
        if cell in ("B", "C", "D") and 4 <= nodes <= 7:
            return f"{cell}2"
        return None
    if cell in LEGACY_CELLS and 4 <= nodes <= 7:
        return cell
    return None


def is_envelope(row: dict, config: dict) -> bool:
    return row.get("envelope") is True and support_stratum(row, config) is not None


def support_counts(rows: Iterable[dict], config: dict) -> dict:
    names, _, minimum, mass_floor = _policy(config)
    eligible = [row for row in rows if row.get("eligible") is True]
    counts = Counter(support_stratum(row, config) for row in eligible if row.get("envelope") is True)
    mass = sum(counts[name] for name in names) / len(eligible) if eligible else 0.0
    passed = bool(eligible) and mass >= mass_floor and all(counts[name] >= minimum for name in names)
    if names == LEGACY_CELLS:
        return {"eligible": len(eligible), "cell_counts": {name: counts[name] for name in names},
                "mass": mass, "pass": passed}
    return {"eligible": len(eligible), "stratum_counts": {name: counts[name] for name in names},
            "mass": mass, "pass": passed}
