"""Command-line entry point for registered v2 feasibility/full generation."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config_v2 import load_config_v2
from .dataset_v2 import run_dataset_v2


def run_cli_for_stream(generator_id: str, argv: list[str] | None = None, *,
                       entry_module: str = "dwr.tasks.cli_v2") -> int:
    cpu_origin = time.process_time()
    from .stream_contract import resolve_stream
    contract = resolve_stream(generator_id)
    parser = argparse.ArgumentParser(description=f"Generate the registered bounded {generator_id[-2:]} task dataset")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("feasibility", "full"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preflight-report", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "feasibility" and args.preflight_report is not None:
        parser.error("--preflight-report is only valid in full mode")
    command = [sys.executable, "-m", entry_module, "--config", str(args.config),
               "--mode", args.mode, "--out", str(args.out)]
    if args.preflight_report is not None:
        command.extend(["--preflight-report", str(args.preflight_report)])
    try:
        if generator_id == "dwr-generator-v2":
            from .config_v2 import load_config_v2
            from .dataset_v2 import run_dataset_v2
            config = load_config_v2(args.config)
            return run_dataset_v2(config, args.out, args.mode, command,
                                  preflight_report=args.preflight_report, cpu_origin=cpu_origin)
        from .config_v3 import load_config_v3
        from .dataset_v2 import run_dataset_v3
        config = load_config_v3(args.config)
        return run_dataset_v3(config, args.out, args.mode, command,
                              preflight_report=args.preflight_report, cpu_origin=cpu_origin)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"{generator_id[-2:]} generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    return run_cli_for_stream("dwr-generator-v2", argv)


if __name__ == "__main__":
    raise SystemExit(main())
