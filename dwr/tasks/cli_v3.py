"""Command-line entry point for the registered bounded v3 generation stream."""
from __future__ import annotations

from .cli_v2 import run_cli_for_stream


def main(argv: list[str] | None = None) -> int:
    return run_cli_for_stream("dwr-generator-v3", argv, entry_module="dwr.tasks.cli_v3")


if __name__ == "__main__":
    raise SystemExit(main())
