"""Single invocation entry point for the reviewed v4 development proposal."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .probe_v4 import run_probe


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Run one bounded DWR development proposal with an admitted config")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("development-probe",))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(effective_argv)
    try:
        return run_probe(args.config, args.out,
                         [sys.executable, "-m", "dwr.tasks.probe_cli_v4", *effective_argv])
    except Exception as exc:
        print(f"probe invocation failed ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
