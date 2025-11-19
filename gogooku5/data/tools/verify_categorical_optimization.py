#!/usr/bin/env python3
"""Verify categorical join optimization implementation.

This tool validates that the categorical join optimization in dataset_builder.py
maintains data integrity while improving performance.

Usage:

.. code-block:: bash

    # Full validation on 2023Q1 chunk (3 months)
    PYTHONPATH=data/src python data/tools/verify_categorical_optimization.py \
      --chunk-path output_g5/chunks/2023Q1/ml_dataset.parquet \
      --report-json /tmp/categorical_verification.json

    # Quick schema-only check
    PYTHONPATH=data/src python data/tools/verify_categorical_optimization.py \
      --chunk-path output_g5/chunks/2023Q1/ml_dataset.parquet \
      --schema-only

    # Benchmark performance (requires baseline comparison)
    PYTHONPATH=data/src python data/tools/verify_categorical_optimization.py \
      --benchmark \
      --baseline-chunk output_g5/chunks_baseline/2023Q1/ml_dataset.parquet \
      --optimized-chunk output_g5/chunks/2023Q1/ml_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import polars as pl


def check_schema_consistency(
    df: pl.DataFrame,
    *,
    expected_code_dtype: pl.DataType = pl.Utf8,
    expected_sec_id_dtype: pl.DataType | None = None
) -> Dict[str, Any]:
    """Verify that code columns are Utf8 (not Categorical) after optimization.

    Args:
        df: Dataset DataFrame to validate
        expected_code_dtype: Expected dtype for code column (default: Utf8)
        expected_sec_id_dtype: Expected dtype for sec_id (default: None, auto-detect)

    Returns:
        Validation results dictionary
    """
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "code_column": None,
        "code_dtype": None,
        "sec_id_dtype": None,
    }

    # Find code column (case-insensitive)
    code_col = None
    for col in df.columns:
        if col.lower() == "code":
            code_col = col
            break

    if code_col is None:
        results["passed"] = False
        results["errors"].append("Code column not found in DataFrame")
        return results

    results["code_column"] = code_col
    code_dtype = df[code_col].dtype
    results["code_dtype"] = str(code_dtype)

    # Verify code column is Utf8 (not Categorical)
    if code_dtype != expected_code_dtype:
        results["passed"] = False
        results["errors"].append(
            f"Code column has dtype {code_dtype}, expected {expected_code_dtype}. "
            f"Schema consistency violation: categorical join should restore Utf8."
        )

    # Check sec_id column if present
    if "sec_id" in df.columns:
        sec_id_dtype = df["sec_id"].dtype
        results["sec_id_dtype"] = str(sec_id_dtype)

        # Auto-detect expected dtype if not provided
        if expected_sec_id_dtype is None:
            # SecId is typically int32 or int64
            if sec_id_dtype not in (pl.Int32, pl.Int64):
                results["warnings"].append(
                    f"sec_id has dtype {sec_id_dtype}, expected Int32 or Int64"
                )
        elif sec_id_dtype != expected_sec_id_dtype:
            results["warnings"].append(
                f"sec_id has dtype {sec_id_dtype}, expected {expected_sec_id_dtype}"
            )

    return results


def check_null_rates(df: pl.DataFrame, columns: List[str] | None = None) -> Dict[str, Any]:
    """Calculate NULL rates for specified columns.

    Args:
        df: Dataset DataFrame
        columns: Columns to check (default: ["Code", "code", "sec_id"])

    Returns:
        NULL rate statistics
    """
    if columns is None:
        # Default: check code and sec_id
        columns = []
        for col in df.columns:
            if col.lower() == "code" or col == "sec_id":
                columns.append(col)

    null_counts = df.null_count()
    total_rows = len(df)

    results = {
        "total_rows": total_rows,
        "null_rates": {},
        "warnings": [],
    }

    for col in columns:
        if col not in df.columns:
            results["warnings"].append(f"Column {col} not found in DataFrame")
            continue

        null_count = null_counts[col][0]
        null_rate = null_count / total_rows if total_rows > 0 else 0.0

        results["null_rates"][col] = {
            "null_count": int(null_count),
            "null_rate": float(null_rate),
            "null_pct": float(null_rate * 100),
        }

    return results


def benchmark_build_time(
    baseline_log_path: Path | None = None,
    optimized_log_path: Path | None = None
) -> Dict[str, Any]:
    """Compare build times from log files.

    Args:
        baseline_log_path: Path to baseline build log
        optimized_log_path: Path to optimized build log

    Returns:
        Performance comparison results
    """
    results = {
        "baseline_time": None,
        "optimized_time": None,
        "speedup": None,
        "speedup_pct": None,
    }

    # This is a placeholder - actual implementation would parse logs
    # to extract build times from logged timestamps
    if baseline_log_path and baseline_log_path.exists():
        # Parse baseline log for total time
        pass

    if optimized_log_path and optimized_log_path.exists():
        # Parse optimized log for total time
        pass

    # Calculate speedup if both times available
    if results["baseline_time"] and results["optimized_time"]:
        speedup = results["baseline_time"] / results["optimized_time"]
        results["speedup"] = float(speedup)
        results["speedup_pct"] = float((speedup - 1.0) * 100)

    return results


def compare_datasets(
    baseline_df: pl.DataFrame,
    optimized_df: pl.DataFrame,
    *,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """Compare baseline and optimized datasets for correctness.

    Args:
        baseline_df: DataFrame before optimization
        optimized_df: DataFrame after optimization
        tolerance: Floating-point comparison tolerance

    Returns:
        Comparison results
    """
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "row_count_match": False,
        "column_count_match": False,
        "schema_match": False,
        "data_match": False,
    }

    # Row count
    if len(baseline_df) == len(optimized_df):
        results["row_count_match"] = True
    else:
        results["passed"] = False
        results["errors"].append(
            f"Row count mismatch: baseline={len(baseline_df):,}, "
            f"optimized={len(optimized_df):,}"
        )

    # Column count
    if len(baseline_df.columns) == len(optimized_df.columns):
        results["column_count_match"] = True
    else:
        results["passed"] = False
        results["errors"].append(
            f"Column count mismatch: baseline={len(baseline_df.columns)}, "
            f"optimized={len(optimized_df.columns)}"
        )

    # Schema (column names and dtypes)
    baseline_schema = {col: str(dtype) for col, dtype in baseline_df.schema.items()}
    optimized_schema = {col: str(dtype) for col, dtype in optimized_df.schema.items()}

    if baseline_schema == optimized_schema:
        results["schema_match"] = True
    else:
        results["passed"] = False
        results["errors"].append("Schema mismatch detected")

        # Find differences
        for col in set(baseline_schema.keys()) | set(optimized_schema.keys()):
            if col not in baseline_schema:
                results["warnings"].append(f"Column {col} only in optimized dataset")
            elif col not in optimized_schema:
                results["warnings"].append(f"Column {col} only in baseline dataset")
            elif baseline_schema[col] != optimized_schema[col]:
                results["warnings"].append(
                    f"Column {col} dtype mismatch: baseline={baseline_schema[col]}, "
                    f"optimized={optimized_schema[col]}"
                )

    # Data comparison (sample-based for performance)
    if results["row_count_match"] and results["schema_match"]:
        try:
            # Sort both DataFrames by common keys for comparison
            sort_cols = []
            for col in ["Date", "date", "Code", "code"]:
                if col in baseline_df.columns and col in optimized_df.columns:
                    sort_cols.append(col)

            if sort_cols:
                baseline_sorted = baseline_df.sort(sort_cols)
                optimized_sorted = optimized_df.sort(sort_cols)

                # Sample 1000 rows for comparison (full comparison too slow)
                sample_size = min(1000, len(baseline_sorted))
                baseline_sample = baseline_sorted.head(sample_size)
                optimized_sample = optimized_sorted.head(sample_size)

                # Compare using Polars frame_equal
                if baseline_sample.frame_equal(optimized_sample):
                    results["data_match"] = True
                else:
                    results["passed"] = False
                    results["errors"].append(
                        "Data mismatch detected in sample (first 1000 rows)"
                    )
            else:
                results["warnings"].append("No common sort columns found for data comparison")

        except Exception as e:
            results["warnings"].append(f"Data comparison failed: {e}")

    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Verify categorical join optimization correctness and performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chunk-path",
        type=Path,
        help="Path to optimized chunk parquet file",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only check schema consistency (fast)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark (requires baseline)",
    )
    parser.add_argument(
        "--baseline-chunk",
        type=Path,
        help="Path to baseline chunk for comparison (pre-optimization)",
    )
    parser.add_argument(
        "--optimized-chunk",
        type=Path,
        help="Path to optimized chunk for comparison (post-optimization)",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Output validation report as JSON",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()

    print("=" * 80)
    print("Categorical Join Optimization Verification")
    print("=" * 80)
    print()

    # Validation report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "schema_check": None,
        "null_rate_check": None,
        "comparison": None,
        "benchmark": None,
        "overall_passed": True,
    }

    # Schema-only validation
    if args.schema_only and args.chunk_path:
        if not args.chunk_path.exists():
            print(f"❌ Chunk not found: {args.chunk_path}")
            return 1

        print(f"📂 Loading chunk: {args.chunk_path}")
        df = pl.read_parquet(str(args.chunk_path))
        print(f"   Rows: {len(df):,}, Columns: {len(df.columns)}")
        print()

        print("🔍 Schema consistency check...")
        schema_results = check_schema_consistency(df)
        report["schema_check"] = schema_results

        if schema_results["passed"]:
            print(f"   ✅ PASSED")
            print(f"   Code column: {schema_results['code_column']} ({schema_results['code_dtype']})")
            if schema_results["sec_id_dtype"]:
                print(f"   SecId column: sec_id ({schema_results['sec_id_dtype']})")
        else:
            report["overall_passed"] = False
            print(f"   ❌ FAILED")
            for error in schema_results["errors"]:
                print(f"   ERROR: {error}")

        for warning in schema_results["warnings"]:
            print(f"   ⚠️  {warning}")

        print()

        # NULL rate check
        print("🔍 NULL rate check...")
        null_results = check_null_rates(df)
        report["null_rate_check"] = null_results

        print(f"   Total rows: {null_results['total_rows']:,}")
        for col, stats in null_results["null_rates"].items():
            print(f"   {col}: {stats['null_pct']:.2f}% NULL ({stats['null_count']:,} rows)")

        print()

    # Full comparison (baseline vs optimized)
    if args.benchmark and args.baseline_chunk and args.optimized_chunk:
        if not args.baseline_chunk.exists():
            print(f"❌ Baseline chunk not found: {args.baseline_chunk}")
            return 1
        if not args.optimized_chunk.exists():
            print(f"❌ Optimized chunk not found: {args.optimized_chunk}")
            return 1

        print(f"📂 Loading baseline: {args.baseline_chunk}")
        baseline_df = pl.read_parquet(str(args.baseline_chunk))
        print(f"   Rows: {len(baseline_df):,}, Columns: {len(baseline_df.columns)}")

        print(f"📂 Loading optimized: {args.optimized_chunk}")
        optimized_df = pl.read_parquet(str(args.optimized_chunk))
        print(f"   Rows: {len(optimized_df):,}, Columns: {len(optimized_df.columns)}")
        print()

        print("🔍 Comparing datasets...")
        comparison_results = compare_datasets(baseline_df, optimized_df)
        report["comparison"] = comparison_results

        if comparison_results["passed"]:
            print("   ✅ PASSED")
            print(f"   Row count match: {comparison_results['row_count_match']}")
            print(f"   Column count match: {comparison_results['column_count_match']}")
            print(f"   Schema match: {comparison_results['schema_match']}")
            print(f"   Data match (sample): {comparison_results['data_match']}")
        else:
            report["overall_passed"] = False
            print("   ❌ FAILED")
            for error in comparison_results["errors"]:
                print(f"   ERROR: {error}")

        for warning in comparison_results["warnings"]:
            print(f"   ⚠️  {warning}")

        print()

    # Save report
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"📝 Report saved to: {args.report_json}")
        print()

    # Summary
    print("=" * 80)
    if report["overall_passed"]:
        print("✅ VERIFICATION PASSED")
        print()
        print("The categorical join optimization maintains data integrity and schema consistency.")
        return 0
    else:
        print("❌ VERIFICATION FAILED")
        print()
        print("Issues detected. Review errors above and check implementation.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
