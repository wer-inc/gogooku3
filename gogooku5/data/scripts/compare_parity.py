"""Compare gogooku datasets against a reference parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

from builder.validation.parity import ParityResult, compare_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two dataset parquet files and emit parity report.")
    parser.add_argument("reference", type=Path, help="Reference dataset parquet")
    parser.add_argument("candidate", type=Path, help="Candidate dataset parquet")
    parser.add_argument(
        "--key-columns",
        nargs="+",
        default=["Code", "Date"],
        help="Key columns used to align rows (default: Code Date)",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row limit for quick comparisons.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write detailed JSON report to the given path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result: ParityResult = compare_datasets(
        reference=args.reference,
        candidate=args.candidate,
        key_columns=args.key_columns,
        sample_rows=args.sample_rows,
    )

    print("=== Dataset Parity Report ===")
    print(f"Reference : {result.reference_path}")
    print(f"Candidate : {result.candidate_path}")
    print(f"Rows      : {result.rows_reference} (ref) vs {result.rows_candidate} (cand)")
    if result.rows_only_reference or result.rows_only_candidate:
        print(f"Rows only in reference: {result.rows_only_reference}")
        print(f"Rows only in candidate: {result.rows_only_candidate}")

    if result.schema_mismatch:
        print("Schema mismatches detected.")
    else:
        print("Schemas aligned.")

    if result.column_diffs:
        print("\nColumn differences:")
        for diff in result.column_diffs:
            flags = []
            if diff.reference_only:
                flags.append("reference-only")
            if diff.candidate_only:
                flags.append("candidate-only")
            line = f"- {diff.column}: ref={diff.dtype_reference} cand={diff.dtype_candidate}"
            if flags:
                line += f" ({', '.join(flags)})"
            elif diff.max_abs_diff is not None:
                line += f" max_abs_diff={diff.max_abs_diff:.6g} mean_abs_diff={diff.mean_abs_diff:.6g}"
            print(line)
    else:
        print("\nNo column differences detected.")

    if args.output_json:
        args.output_json.write_text(result.to_json(), encoding="utf-8")
        print(f"\nDetailed JSON report written to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
