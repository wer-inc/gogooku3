#!/usr/bin/env python3
"""Add graph features to merged dataset as post-processing step.

This script takes a merged dataset (with NULL graph columns) and computes
actual graph features using GraphFeatureEngineer. Designed for use after
merging chunks built with ENABLE_GRAPH_FEATURES=0.

Usage:
    python scripts/add_graph_features_to_dataset.py \
        --input output/ml_dataset_2025_merged.parquet \
        --output output/ml_dataset_2025_with_graph.parquet \
        --window-days 60 \
        --min-observations 20 \
        --correlation-threshold 0.3 \
        --report output/graph_addition_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from builder.features.core.graph.features import (
    GraphFeatureConfig,
    GraphFeatureEngineer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add graph features to dataset with NULL graph columns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input parquet file (must have NULL graph columns)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output parquet file with computed graph features",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=60,
        help="Correlation window in days (default: 60)",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=20,
        help="Minimum observations required (default: 20)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.3,
        help="Peer correlation threshold (default: 0.3)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Output path for quality report JSON (optional)",
    )
    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load dataset
    print(f"📂 Loading dataset from {args.input}")
    start_time = time.time()
    df = pl.read_parquet(args.input)
    load_time = time.time() - start_time
    print(f"   ✅ Loaded {len(df):,} rows × {len(df.columns)} columns in {load_time:.1f}s")

    # Check for required columns
    required_cols = {"code", "date", "ret_prev_1d"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ Error: Missing required columns: {missing}")
        sys.exit(1)

    # Check for existing graph columns (should be all NULL)
    graph_cols = [c for c in df.columns if c.startswith("graph_")]
    if graph_cols:
        print(f"   ⚠️  Found {len(graph_cols)} existing graph columns (will be replaced)")
        # Drop existing graph columns to avoid conflicts
        df = df.drop(graph_cols)
    else:
        print("   ℹ️  No existing graph columns found")

    # Configure GraphFeatureEngineer
    config = GraphFeatureConfig(
        code_column="code",
        date_column="date",
        return_column="ret_prev_1d",
        window_days=args.window_days,
        min_observations=args.min_observations,
        correlation_threshold=args.correlation_threshold,
        shift_to_next_day=True,
        block_size=512,
    )

    print("\n📊 Graph Feature Configuration:")
    print(f"   - Window: {config.window_days} days")
    print(f"   - Min observations: {config.min_observations}")
    print(f"   - Correlation threshold: {config.correlation_threshold}")
    print(f"   - Return column: {config.return_column}")

    # Add graph features
    print("\n🔄 Computing graph features...")
    engineer = GraphFeatureEngineer(config=config)
    graph_start = time.time()
    df_with_graph = engineer.add_features(df)
    graph_time = time.time() - graph_start

    # Identify newly added graph columns
    new_graph_cols = [c for c in df_with_graph.columns if c.startswith("graph_")]
    print(f"   ✅ Added {len(new_graph_cols)} graph columns in {graph_time:.1f}s")

    # Generate quality report
    report = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "total_rows": len(df_with_graph),
        "total_columns": len(df_with_graph.columns),
        "graph_columns_added": len(new_graph_cols),
        "graph_column_names": sorted(new_graph_cols),
        "configuration": {
            "window_days": config.window_days,
            "min_observations": config.min_observations,
            "correlation_threshold": config.correlation_threshold,
            "return_column": config.return_column,
        },
        "timing": {
            "load_time_seconds": round(load_time, 2),
            "graph_computation_seconds": round(graph_time, 2),
        },
        "null_rates": {},
    }

    # Calculate NULL rates for graph columns
    print("\n📈 Graph Feature Quality:")
    for col in sorted(new_graph_cols):
        null_count = df_with_graph[col].null_count()
        null_rate = null_count / len(df_with_graph) * 100
        report["null_rates"][col] = round(null_rate, 2)
        print(f"   - {col}: {null_rate:.2f}% NULL")

    # Save report if requested
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Quality report saved to {args.report}")

    # Save output dataset
    print(f"\n💾 Saving dataset to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_start = time.time()
    df_with_graph.write_parquet(args.output, compression="zstd")
    save_time = time.time() - save_start

    total_time = time.time() - start_time
    print(f"   ✅ Saved {len(df_with_graph):,} rows × {len(df_with_graph.columns)} columns in {save_time:.1f}s")
    print(f"\n✅ Total processing time: {total_time:.1f}s")
    print("\n🎉 Graph features successfully added!")
    print(f"   Input:  {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Graph columns: {len(new_graph_cols)}")


if __name__ == "__main__":
    main()
