"""CLI entrypoint for the default dataset build."""
from __future__ import annotations

import argparse
from typing import List

from builder.utils import ensure_env_loaded
from cli.main import main as cli_main


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gogooku5 dataset")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--refresh-listed",
        action="store_true",
        help="Refresh listed securities metadata before building",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_env_loaded()
    # Delegate to unified CLI (automatic chunking + auto-merge to avoid OOM on long ranges)
    cli_argv: List[str] = [
        "build",
        "--start",
        args.start,
        "--end",
        args.end,
        "--merge",
    ]
    if args.refresh_listed:
        cli_argv.append("--refresh-listed")
    return cli_main(cli_argv)


if __name__ == "__main__":
    raise SystemExit(main())
