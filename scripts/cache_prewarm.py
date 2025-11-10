"""Prewarm J-Quants/yfinance caches by running targeted fetches."""

from __future__ import annotations

import argparse
import os
import sys

# Path setup must happen before other imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
g5_data = os.path.join(repo_root, "gogooku5", "data")
if g5_data not in sys.path:
    sys.path.insert(0, g5_data)

from scripts import build_chunks  # type: ignore  # noqa: E402

build_chunks._extend_system_site_packages()

from builder.api.advanced_fetcher import (  # type: ignore  # noqa: E402
    AdvancedJQuantsFetcher,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prewarm cache for a date range")
    parser.add_argument("start", help="Start date YYYY-MM-DD")
    parser.add_argument("end", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["futures", "options", "trades_spec"],
        help="Which feeds to prewarm",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fetcher = AdvancedJQuantsFetcher()
    targets = set(args.targets)

    if "futures" in targets:
        print(f"[prewarm] futures {args.start}→{args.end}")
        fetcher.fetch_futures(start=args.start, end=args.end)

    if "options" in targets:
        print(f"[prewarm] options_daily {args.start}→{args.end}")
        fetcher.fetch_options(start=args.start, end=args.end)

    if "trades_spec" in targets:
        print(f"[prewarm] trades_spec {args.start}→{args.end}")
        fetcher.fetch_trades_spec(start=args.start, end=args.end)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
