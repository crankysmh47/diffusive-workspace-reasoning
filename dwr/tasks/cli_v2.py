"""Command-line entry point for registered v2 feasibility/full generation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config_v2 import load_config_v2
from .dataset_v2 import run_dataset_v2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the registered bounded v2 task dataset")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("feasibility", "full"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preflight-report", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "feasibility" and args.preflight_report is not None:
        parser.error("--preflight-report is only valid in full mode")
    command = [sys.executable, "-m", "dwr.tasks.cli_v2", "--config", str(args.config),
               "--mode", args.mode, "--out", str(args.out)]
    if args.preflight_report is not None:
        command.extend(["--preflight-report", str(args.preflight_report)])
    try:
        config = load_config_v2(args.config)
        return run_dataset_v2(config, args.out, args.mode, command,
                              preflight_report=args.preflight_report)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"v2 generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
