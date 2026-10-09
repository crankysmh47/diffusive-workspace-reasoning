"""Single invocation entry point for the reviewed v4 development proposal."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .probe_v4 import run_probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one bounded DWR v4 development proposal")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("development-probe",))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        return run_probe(args.config, args.out,
                         [sys.executable, "-m", "dwr.tasks.probe_cli_v4", *sys.argv[1:]])
    except Exception as exc:
        print(f"probe refused before candidate generation ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
