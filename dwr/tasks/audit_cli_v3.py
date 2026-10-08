"""Closed v3 independent audit entry point; implementation is shared with v2 CLI."""
from __future__ import annotations

from .audit_cli_v2 import run_closed_versioned_v3


def main(argv: list[str] | None = None) -> int:
    return run_closed_versioned_v3(argv)


if __name__ == "__main__":
    raise SystemExit(main())
