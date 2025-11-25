"""CLI entrypoint for the optimized dataset build."""
from __future__ import annotations

import argparse
from typing import List

from builder.pipelines.optimized_pipeline import run_optimized_pipeline
from builder.utils import ensure_env_loaded


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gogooku5 dataset (optimized mode)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Warm caches without generating final parquet output",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_env_loaded()
    run_optimized_pipeline(start=args.start, end=args.end, cache_only=args.cache_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
