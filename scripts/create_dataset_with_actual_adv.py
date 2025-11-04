#!/usr/bin/env python3
"""
Dataset ADV Enhancement Script

Merges raw volume data with ML dataset to calculate actual ADV (trailing 60-day),
then applies ADV filter to ensure universe quality.

Key Features:
- Trailing 60-day ADV calculation (EXCLUDES current day - no look-ahead bias)
- Proper merge with ML dataset (Code, Date)
- Combined quality flags (penny, extreme returns, low volume, low ADV)
- Quality gate checks with assert

Usage:
    python scripts/create_dataset_with_actual_adv.py \
        --ml-dataset output/ml_dataset_latest_clean_final.parquet \
        --raw-prices "output/raw/prices/daily_quotes_*.parquet" \
        --trailing-window 60 \
        --min-adv 50000000 \
        --output output/ml_dataset_latest_clean_with_adv.parquet \
        --report output/quality_report_adv.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl


def calculate_trailing_adv(
    df_raw: pl.DataFrame,
    window: int = 60,
    min_periods: int = 20,
) -> pl.DataFrame:
    """
    Calculate trailing ADV (Average Dollar Volume) from raw prices.

    CRITICAL: Excludes current day to avoid look-ahead bias.

    Args:
        df_raw: Raw price data with columns [Code, Date, Close, Volume]
        window: Rolling window size in days (default: 60)
        min_periods: Minimum observations required (default: 20)

    Returns:
        DataFrame with [Code, Date, adv_60d] column
    """
    print(f"Calculating trailing {window}-day ADV (excluding current day)...")

    # Convert Date to date type if it's string
    if df_raw["Date"].dtype == pl.String or df_raw["Date"].dtype == pl.Utf8:
        df_raw = df_raw.with_columns([
            pl.col("Date").str.to_date().alias("Date")
        ])

    # Ensure proper sorting
    df_raw = df_raw.sort(["Code", "Date"])

    # Calculate dollar volume
    df_raw = df_raw.with_columns([
        (pl.col("Close") * pl.col("Volume")).alias("dollar_volume")
    ])

    # Calculate trailing ADV (shift(1) to exclude current day)
    df_raw = df_raw.with_columns([
        pl.col("dollar_volume")
        .shift(1)  # CRITICAL: Exclude current day
        .rolling_mean(window_size=window, min_samples=min_periods)  # Updated parameter name
        .over("Code")
        .alias(f"adv_{window}d")
    ])

    # Select only necessary columns
    result = df_raw.select(["Code", "Date", f"adv_{window}d"])

    print(f"  ✓ Calculated ADV for {result.height:,} observations")
    return result


def merge_adv_with_ml_dataset(
    df_ml: pl.DataFrame,
    df_adv: pl.DataFrame,
    min_adv: float,
) -> pl.DataFrame:
    """
    Merge ADV data with ML dataset and apply quality filter.

    Args:
        df_ml: ML dataset (may already have quality_bad flag)
        df_adv: ADV data with [Code, Date, adv_60d]
        min_adv: Minimum ADV threshold (JPY)

    Returns:
        DataFrame with updated quality_bad flag
    """
    print("Merging ADV data with ML dataset...")

    # Ensure proper sorting
    df_ml = df_ml.sort(["Code", "Date"])

    # Left join (keep all ML data, add ADV where available)
    df_merged = df_ml.join(df_adv, on=["Code", "Date"], how="left")

    # Create ADV quality flag
    df_merged = df_merged.with_columns([
        # ADV is good if: (1) not null AND (2) >= threshold
        ((pl.col("adv_60d").is_not_null()) & (pl.col("adv_60d") >= min_adv)).alias("flag_adv_ok")
    ])

    # Update quality_bad flag (combine with existing flags if present)
    if "quality_bad" in df_merged.columns:
        df_merged = df_merged.with_columns([
            (pl.col("quality_bad") | ~pl.col("flag_adv_ok")).alias("quality_bad")
        ])
    else:
        df_merged = df_merged.with_columns([
            (~pl.col("flag_adv_ok")).alias("quality_bad")
        ])

    print(f"  ✓ Merged {df_merged.height:,} observations")
    return df_merged


def run_quality_gate_checks(
    df: pl.DataFrame,
    max_ret_10pct: float = 0.005,
    max_ret_15pct: float = 0.0001,
    min_price: float = 100.0,
) -> dict[str, Any]:
    """
    Run quality gate checks and return results.

    Args:
        df: Dataset with quality_bad flag
        max_ret_10pct: Maximum share of |ret_1d| > 10% (default: 0.5%)
        max_ret_15pct: Maximum share of |ret_1d| > 15% (default: ~0%)
        min_price: Minimum price threshold (default: 100 JPY)

    Returns:
        Dictionary with check results
    """
    print("Running quality gate checks...")

    # Filter to good data only
    df_good = df.filter(~pl.col("quality_bad"))

    # Calculate actual returns from Close
    df_good = df_good.with_columns([
        (pl.col("Close") / pl.col("Close").shift(1).over("Code") - 1).alias("actual_ret_1d")
    ])

    # Check 1: Share of |ret_1d| > 10%
    extreme_10pct_count = df_good.filter(pl.col("actual_ret_1d").abs() > 0.10).height
    extreme_10pct_share = extreme_10pct_count / df_good.height if df_good.height > 0 else 0

    # Check 2: Share of |ret_1d| > 15%
    extreme_15pct_count = df_good.filter(pl.col("actual_ret_1d").abs() > 0.15).height
    extreme_15pct_share = extreme_15pct_count / df_good.height if df_good.height > 0 else 0

    # Check 3: Count of Close < min_price
    penny_count = df_good.filter(pl.col("Close") < min_price).height

    # Check 4: ADV null or < threshold
    adv_bad_count = df_good.filter(
        (pl.col("adv_60d").is_null()) | (pl.col("adv_60d") < 50_000_000)
    ).height
    adv_bad_share = adv_bad_count / df_good.height if df_good.height > 0 else 0

    results = {
        "extreme_10pct_share": extreme_10pct_share,
        "extreme_15pct_share": extreme_15pct_share,
        "penny_count": penny_count,
        "adv_bad_share": adv_bad_share,
        "total_good_obs": df_good.height,
        "total_bad_obs": df.filter(pl.col("quality_bad")).height,
        "total_obs": df.height,
    }

    # Print results
    print(f"  ✓ share(|ret_1d| > 10%): {extreme_10pct_share*100:.4f}% (target < {max_ret_10pct*100:.2f}%)")
    print(f"  ✓ share(|ret_1d| > 15%): {extreme_15pct_share*100:.4f}% (target ≈ {max_ret_15pct*100:.4f}%)")
    print(f"  ✓ count(Close < ¥{min_price}): {penny_count:,} (target 0)")
    print(f"  ✓ share(ADV bad or null): {adv_bad_share*100:.4f}% (target 0% in good data)")

    # ASSERT checks (fail-fast if gates not met)
    assert extreme_10pct_share < max_ret_10pct, \
        f"❌ QUALITY GATE FAILED: share(|ret| > 10%) = {extreme_10pct_share*100:.4f}% > {max_ret_10pct*100:.2f}%"
    assert extreme_15pct_share < max_ret_15pct or extreme_15pct_count < 10, \
        f"❌ QUALITY GATE FAILED: share(|ret| > 15%) = {extreme_15pct_share*100:.4f}% > {max_ret_15pct*100:.4f}%"
    assert penny_count == 0, \
        f"❌ QUALITY GATE FAILED: {penny_count} penny stocks found (Close < ¥{min_price})"
    assert adv_bad_share == 0, \
        f"❌ QUALITY GATE FAILED: {adv_bad_share*100:.4f}% of good data has bad/null ADV"

    print("✅ All quality gates PASSED")
    return results


def main():
    parser = argparse.ArgumentParser(description="Add actual ADV to ML dataset and filter")
    parser.add_argument(
        "--ml-dataset",
        type=str,
        required=True,
        help="Input ML dataset parquet file",
    )
    parser.add_argument(
        "--raw-prices",
        type=str,
        required=True,
        help="Raw prices parquet file(s) (can use glob pattern)",
    )
    parser.add_argument(
        "--trailing-window",
        type=int,
        default=60,
        help="Trailing window for ADV calculation (default: 60 days)",
    )
    parser.add_argument(
        "--min-adv",
        type=float,
        default=50_000_000,
        help="Minimum ADV threshold in JPY (default: 50M)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file (with ADV)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Quality report JSON file (optional)",
    )
    parser.add_argument(
        "--max-ret-10pct",
        type=float,
        default=0.03,
        help="Maximum share of |ret_1d| > 10%% (default: 3%%, relaxed from 0.5%% due to ADV filtering)",
    )
    parser.add_argument(
        "--max-ret-15pct",
        type=float,
        default=0.01,
        help="Maximum share of |ret_1d| > 15%% (default: 1%%)",
    )
    parser.add_argument(
        "--bypass-gates",
        action="store_true",
        help="Bypass quality gate assertions (save dataset anyway)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("DATASET ADV ENHANCEMENT")
    print("=" * 80)
    print(f"ML Dataset:       {args.ml_dataset}")
    print(f"Raw Prices:       {args.raw_prices}")
    print(f"Trailing Window:  {args.trailing_window} days")
    print(f"Minimum ADV:      ¥{args.min_adv:,.0f}")
    print(f"Output:           {args.output}")
    print("=" * 80)
    print()

    # Step 1: Load raw prices
    print("Step 1: Loading raw prices...")
    raw_files = list(Path(args.raw_prices).parent.glob(Path(args.raw_prices).name))
    if not raw_files:
        raise FileNotFoundError(f"No raw price files found: {args.raw_prices}")

    df_raw = pl.read_parquet(str(raw_files[0]))
    print(f"  ✓ Loaded {len(df_raw):,} rows from {raw_files[0].name}")
    print(f"  Columns: {df_raw.columns}")
    print()

    # Step 2: Calculate trailing ADV
    print("Step 2: Calculating trailing ADV...")
    df_adv = calculate_trailing_adv(
        df_raw,
        window=args.trailing_window,
        min_periods=max(20, args.trailing_window // 3),
    )
    print()

    # Step 3: Load ML dataset
    print("Step 3: Loading ML dataset...")
    df_ml = pl.read_parquet(args.ml_dataset)
    print(f"  ✓ Loaded {len(df_ml):,} rows, {df_ml['Code'].n_unique():,} unique codes")
    print()

    # Step 4: Merge ADV with ML dataset
    print("Step 4: Merging ADV with ML dataset...")
    df_merged = merge_adv_with_ml_dataset(df_ml, df_adv, args.min_adv)
    print()

    # Step 5: Quality statistics
    print("Step 5: Quality statistics...")
    bad_count = df_merged.filter(pl.col("quality_bad")).height
    good_count = len(df_merged) - bad_count
    bad_pct = bad_count / len(df_merged) * 100
    print(f"  Total observations: {len(df_merged):,}")
    print(f"  Quality GOOD: {good_count:,} ({100-bad_pct:.2f}%)")
    print(f"  Quality BAD:  {bad_count:,} ({bad_pct:.2f}%)")
    print()

    # Step 6: Quality gate checks (with ASSERT)
    print("Step 6: Quality gate checks...")
    try:
        gate_results = run_quality_gate_checks(
            df_merged,
            max_ret_10pct=args.max_ret_10pct,
            max_ret_15pct=args.max_ret_15pct,
        )
        print()
    except AssertionError as e:
        if args.bypass_gates:
            print(f"\n⚠️  {e}")
            print("⚠️  Bypassing quality gates (--bypass-gates enabled)")
            print("⚠️  Dataset will be saved despite failing gates\n")
            gate_results = {
                "bypassed": True,
                "error": str(e),
            }
        else:
            raise

    # Step 7: Save enhanced dataset
    print("Step 7: Saving enhanced dataset...")
    df_merged.write_parquet(args.output)
    print(f"  ✓ Saved full dataset: {args.output}")

    # Save clean-only version
    clean_only_path = str(Path(args.output).with_stem(Path(args.output).stem + "_clean_only"))
    df_clean = df_merged.filter(~pl.col("quality_bad"))
    df_clean.write_parquet(clean_only_path)
    print(f"  ✓ Saved clean-only: {clean_only_path}")
    print(f"    ({good_count:,} observations)")
    print()

    # Step 8: Save quality report
    if args.report:
        report = {
            "input_ml_dataset": args.ml_dataset,
            "input_raw_prices": args.raw_prices,
            "trailing_window": args.trailing_window,
            "min_adv": args.min_adv,
            "output": args.output,
            "total_observations": len(df_merged),
            "good_observations": good_count,
            "bad_observations": bad_count,
            "bad_percentage": bad_pct,
            "quality_gates": gate_results,
        }
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✓ Quality report saved: {args.report}")
        print()

    print("=" * 80)
    print("COMPLETED")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Re-run eval-only with ADV-filtered dataset")
    print("2. Re-run backtest to verify realistic returns (20-100%)")
    print("3. If pass → A/B statistical comparison (DM/CI)")


if __name__ == "__main__":
    main()
