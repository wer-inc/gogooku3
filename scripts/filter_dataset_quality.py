#!/usr/bin/env python3
"""
Dataset Quality Filter Script

Applies quality checks and creates a filtered dataset suitable for training/backtesting.
Based on forensic analysis findings from 2025-11-02.

Usage:
    python scripts/filter_dataset_quality.py \\
        --input output/ml_dataset_latest_full_filled.parquet \\
        --output output/ml_dataset_latest_clean.parquet \\
        --min-price 100 \\
        --max-ret-1d 0.15 \\
        --min-adv 50000000
"""

import argparse
from pathlib import Path

import polars as pl


def apply_quality_filters(
    df: pl.DataFrame,
    min_price: float = 100.0,
    max_ret_1d: float = 0.15,
    min_adv: float = 50_000_000,  # Not used (Volume is percentile rank)
    min_volume_days: int = 5,
) -> pl.DataFrame:
    """
    Apply quality filters to dataset.

    NOTE: ML dataset has pre-processed features:
    - Close: Actual prices ✅ (used for filtering)
    - Volume: Percentile rank [0,1] ❌ (cannot calculate actual ADV)
    - returns_1d: Percentile rank [0,1] ❌ (must recalculate from Close)

    Args:
        df: Input dataframe
        min_price: Minimum Close price (JPY)
        max_ret_1d: Maximum absolute daily return (cap)
        min_adv: Minimum median ADV (JPY) - NOT USED (Volume is percentile)
        min_volume_days: Minimum days with non-zero volume (freeze detection)

    Returns:
        Filtered dataframe with quality_bad flag
    """
    # Sort by code and date
    df = df.sort(["Code", "Date"])

    # Calculate derived features
    df = df.with_columns(
        [
            # Actual returns (recalculated from Close, NOT using returns_1d percentile)
            (pl.col("Close") / pl.col("Close").shift(1).over("Code") - 1).alias(
                "actual_ret_1d"
            ),
            # Price unchanged flag (for freeze detection)
            (pl.col("Close") == pl.col("Close").shift(1).over("Code")).alias(
                "price_unchanged"
            ),
        ]
    )

    # Quality flags (per observation)
    df = df.with_columns(
        [
            (pl.col("Close") < min_price).alias("flag_penny"),
            (pl.col("actual_ret_1d").abs() > max_ret_1d).alias("flag_extreme_ret"),
            # Volume is percentile [0,1], so check for extreme low liquidity
            (pl.col("Volume") < 0.01).alias("flag_zero_vol"),  # Bottom 1%
        ]
    )

    # Stock-level flags (computed per Code)
    stock_flags = (
        df.group_by("Code")
        .agg(
            [
                pl.col("price_unchanged").sum().alias("freeze_days"),
            ]
        )
        .with_columns(
            [
                (pl.col("freeze_days") >= min_volume_days).alias("flag_freeze"),
            ]
        )
        .select(["Code", "flag_freeze"])
    )

    # Join stock-level flags back
    df = df.join(stock_flags, on="Code", how="left")

    # Combine all flags into quality_bad (NO ADV filter)
    df = df.with_columns(
        [
            (
                pl.col("flag_penny")
                | pl.col("flag_extreme_ret")
                | pl.col("flag_zero_vol")
                | pl.col("flag_freeze")
            ).alias("quality_bad")
        ]
    )

    return df


def main():
    parser = argparse.ArgumentParser(description="Filter dataset for quality issues")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file (clean dataset)",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=100.0,
        help="Minimum Close price (JPY, default: 100)",
    )
    parser.add_argument(
        "--max-ret-1d",
        type=float,
        default=0.15,
        help="Maximum absolute daily return (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--min-adv",
        type=float,
        default=50_000_000,
        help="Minimum median ADV (JPY, default: 50M)",
    )
    parser.add_argument(
        "--min-volume-days",
        type=int,
        default=5,
        help="Minimum days for freeze detection (default: 5)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Save quality report to file (optional)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("DATASET QUALITY FILTER")
    print("=" * 80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()
    print("Filter thresholds:")
    print(f"  Minimum price: ¥{args.min_price:,.0f}")
    print(f"  Maximum daily return: ±{args.max_ret_1d * 100:.1f}%")
    print(f"  Minimum ADV: ¥{args.min_adv:,.0f}")
    print(f"  Freeze detection: {args.min_volume_days}+ days")
    print("=" * 80)
    print()

    # Load dataset
    print("Loading dataset...")
    df = pl.read_parquet(args.input)
    print(f"  Loaded: {len(df):,} rows, {df['Code'].n_unique():,} unique codes")
    print()

    # Apply filters
    print("Applying quality filters...")
    df_filtered = apply_quality_filters(
        df,
        min_price=args.min_price,
        max_ret_1d=args.max_ret_1d,
        min_adv=args.min_adv,
        min_volume_days=args.min_volume_days,
    )
    print("  ✓ Quality flags computed")
    print()

    # Generate report
    bad_count = df_filtered.filter(pl.col("quality_bad")).height
    bad_pct = bad_count / len(df_filtered) * 100
    good_count = len(df_filtered) - bad_count

    print("Quality Report:")
    print(f"  Total observations: {len(df_filtered):,}")
    print(f"  Quality BAD: {bad_count:,} ({bad_pct:.2f}%)")
    print(f"  Quality GOOD: {good_count:,} ({100-bad_pct:.2f}%)")
    print()

    # Flag breakdown
    flag_counts = {
        "Penny stocks": df_filtered.filter(pl.col("flag_penny")).height,
        "Extreme returns": df_filtered.filter(pl.col("flag_extreme_ret")).height,
        "Zero volume": df_filtered.filter(pl.col("flag_zero_vol")).height,
        "Price freezes": df_filtered.filter(pl.col("flag_freeze")).height,
    }

    print("Flag breakdown:")
    for flag_name, count in flag_counts.items():
        pct = count / len(df_filtered) * 100
        print(f"  {flag_name}: {count:,} ({pct:.2f}%)")
    print()

    # Code-level summary
    bad_codes = df_filtered.filter(pl.col("quality_bad")).select("Code").unique()
    print(f"Unique codes flagged: {len(bad_codes):,}")
    print()

    # Save quality report
    if args.report:
        report_data = {
            "total_observations": len(df_filtered),
            "bad_observations": bad_count,
            "good_observations": good_count,
            "bad_percentage": bad_pct,
            "flag_breakdown": flag_counts,
            "bad_codes_count": len(bad_codes),
        }
        import json

        with open(args.report, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"✓ Quality report saved to: {args.report}")
        print()

    # Save clean dataset (all data with quality_bad flag)
    print("Saving filtered dataset...")
    df_filtered.write_parquet(args.output)
    print(f"✓ Saved to: {args.output}")
    print()

    # Save clean-only version (quality_bad == False)
    clean_only_path = str(Path(args.output).with_stem(Path(args.output).stem + "_clean_only"))
    df_clean = df_filtered.filter(~pl.col("quality_bad"))
    df_clean.write_parquet(clean_only_path)
    print(f"✓ Clean-only saved to: {clean_only_path}")
    print(f"  (Excluded {bad_count:,} bad observations)")
    print()

    print("=" * 80)
    print("COMPLETED")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review quality report")
    print("2. Re-train models with clean dataset")
    print("3. Re-run backtest to verify realistic returns")


if __name__ == "__main__":
    main()
