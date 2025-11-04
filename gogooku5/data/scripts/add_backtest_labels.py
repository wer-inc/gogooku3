#!/usr/bin/env python3
"""
Phase 2 Patch G (Backtest-Only): Add forward returns for backtest evaluation.

⚠️  IMPORTANT: These labels are ONLY for backtest evaluation, NEVER for training.
⚠️  Phase 2 Patch B correctly removes these from training datasets.

Usage:
    python scripts/add_backtest_labels.py \
        --input output/ml_dataset_latest_full.parquet \
        --output output/ml_dataset_latest_full_with_labels.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("add_backtest_labels")


def add_forward_returns(
    df: pl.DataFrame,
    price_col: str = "Close",
    code_col: str = "Code",
    date_col: str = "Date",
    horizons: list[int] = [1, 5, 10, 20],
) -> pl.DataFrame:
    """
    Add forward returns for backtest evaluation.

    Computes: returns_Nd = (Close[t+N] - Close[t]) / Close[t]

    Args:
        df: Input dataframe with price data
        price_col: Price column name (default: "Close")
        code_col: Stock code column (default: "Code")
        date_col: Date column (default: "Date")
        horizons: Forward horizons in days (default: [1, 5, 10, 20])

    Returns:
        DataFrame with added returns_Nd columns
    """
    LOGGER.info(
        "Computing forward returns for horizons: %s (price=%s, code=%s, date=%s)",
        horizons,
        price_col,
        code_col,
        date_col,
    )

    # Ensure sorted by code, date
    df = df.sort([code_col, date_col])

    # Compute forward returns using shift(-N)
    for horizon in horizons:
        col_name = f"returns_{horizon}d"
        LOGGER.info(f"  Computing {col_name}...")

        df = df.with_columns(
            [
                (
                    (pl.col(price_col).shift(-horizon) - pl.col(price_col))
                    / pl.col(price_col)
                )
                .over(code_col)
                .alias(col_name)
            ]
        )

    LOGGER.info(
        "✅ Forward returns added: %s",
        [f"returns_{h}d" for h in horizons],
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add forward returns to Phase 2 dataset for backtest evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input parquet file (Phase 2 dataset without labels)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file (with forward returns for backtest)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="Forward horizons in days (default: 1 5 10 20)",
    )
    parser.add_argument(
        "--price-col",
        type=str,
        default="Close",
        help="Price column name (default: Close)",
    )
    parser.add_argument(
        "--code-col",
        type=str,
        default="Code",
        help="Stock code column (default: Code)",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="Date",
        help="Date column (default: Date)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    LOGGER.info("="*80)
    LOGGER.info("Phase 2 Patch G: Add Backtest Labels")
    LOGGER.info("="*80)
    LOGGER.info("Input: %s", input_path)
    LOGGER.info("Output: %s", output_path)
    LOGGER.info("")

    # Load dataset
    LOGGER.info("Loading dataset...")
    df = pl.read_parquet(input_path)
    LOGGER.info("  Loaded: %d rows × %d columns", len(df), len(df.columns))

    # Check for required columns
    required = [args.price_col, args.code_col, args.date_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check if labels already exist
    existing_labels = [
        col for col in df.columns if col.startswith("returns_") and col.endswith("d")
    ]
    if existing_labels:
        LOGGER.warning(
            "⚠️  Dataset already has label columns: %s",
            existing_labels,
        )
        LOGGER.warning("   These will be overwritten.")

    # Add forward returns
    df_with_labels = add_forward_returns(
        df=df,
        price_col=args.price_col,
        code_col=args.code_col,
        date_col=args.date_col,
        horizons=args.horizons,
    )

    # Save result
    LOGGER.info("")
    LOGGER.info("Saving result to: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_labels.write_parquet(output_path)

    # Report statistics
    LOGGER.info("")
    LOGGER.info("="*80)
    LOGGER.info("SUMMARY")
    LOGGER.info("="*80)
    LOGGER.info("Input rows: %d", len(df))
    LOGGER.info("Output rows: %d", len(df_with_labels))
    LOGGER.info("Output columns: %d", len(df_with_labels.columns))
    LOGGER.info("")
    LOGGER.info("Added columns:")
    for horizon in args.horizons:
        col_name = f"returns_{horizon}d"
        non_null = df_with_labels[col_name].is_not_null().sum()
        non_null_pct = 100.0 * non_null / len(df_with_labels)
        LOGGER.info(f"  {col_name}: {non_null:,} non-null ({non_null_pct:.1f}%)")

    LOGGER.info("")
    LOGGER.info("✅ Backtest dataset ready: %s", output_path)
    LOGGER.info("")
    LOGGER.info("⚠️  REMINDER: Use this dataset ONLY for backtest, NOT for training")


if __name__ == "__main__":
    main()
