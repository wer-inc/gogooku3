#!/usr/bin/env python3
"""Dataset quality checker for gogooku5 outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from builder.validation.quality import parse_asof_specs, run_quality_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset quality checks for gogooku5 outputs.")
    parser.add_argument("--dataset", required=True, help="Path to dataset parquet file")
    parser.add_argument("--date-col", default="date", help="Date column name (default: date)")
    parser.add_argument("--code-col", default="code", help="Code column name (default: code)")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=[],
        help="Target columns that must be non-null (space separated)",
    )
    parser.add_argument(
        "--asof-check",
        action="append",
        default=[],
        help="Constraint of the form 'col<=reference_col' to enforce non-leak ordering",
    )
    parser.add_argument(
        "--allow-future-days",
        type=int,
        default=0,
        help="How many days beyond today to tolerate (default: 0)",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="Number of sample rows to show for violations (default: 5)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path to write JSON report",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with code 1 if any warning is produced",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(2)

    try:
        asof_pairs = parse_asof_specs(args.asof_check)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(2)
    summary, errors, warnings = run_quality_checks(
        dataset_path,
        date_col=args.date_col,
        code_col=args.code_col,
        targets=args.targets,
        asof_pairs=asof_pairs,
        allow_future_days=args.allow_future_days,
        sample_rows=args.sample_rows,
    )

    print(json.dumps(summary, indent=2, default=str))

    if args.report:
        args.report.write_text(json.dumps(summary, indent=2, default=str))

    if errors:
        sys.exit(1)
    if warnings and args.fail_on_warning:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
