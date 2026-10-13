"""Command-line entry point for bounded task dataset generation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset import run_dataset
from .run_support import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the registered bounded constraint-task dataset")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", "dwr.tasks.generator", "--config", str(args.config), "--out", str(args.out)]
    try:
        config = load_config(args.config)
        return run_dataset(config, args.out, command)
    except (ConfigError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
