"""CLI entrypoint for the default dataset build."""
from __future__ import annotations

import argparse
from typing import List

from builder.pipelines.full_pipeline import run_full_pipeline
from builder.utils import ensure_env_loaded


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
    run_full_pipeline(start=args.start, end=args.end, refresh_listed=args.refresh_listed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
