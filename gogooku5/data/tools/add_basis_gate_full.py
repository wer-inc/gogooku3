#!/usr/bin/env python3
"""Post-process merged datasets to add basis_gate family features.

This tool operates on an already-merged yearly (or multi-year) ML dataset and
adds/repairs the following feature family:

* gap_predictor
* basis_gate
* basis_gate_roll_mean_20d
* basis_gate_roll_std_20d
* basis_gate_zscore_20d
* basis_gate_outlier_flag
* basis_gate_sector_mean
* basis_gate_sector_rel
* is_gap_basis_valid

The implementation mirrors DatasetBuilder._add_gap_basis_features and
_add_basis_gate_derivatives, but works at the dataset (parquet) level.

Typical usage (after beta60_topix has been populated with
add_beta_alpha_bd_features_full.py):

.. code-block:: bash

    PYTHONPATH=gogooku5/data/src \\
      python gogooku5/data/tools/add_basis_gate_full.py \\
        --input  data/output/datasets/ml_dataset_2025_with_graph33.parquet \\
        --output data/output/datasets/ml_dataset_2025_with_graph33_basis.parquet \\
        --date-col Date --code-col Code
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import polars as pl
from builder.features.utils.rolling import roll_mean_safe, roll_std_safe

SECTOR_CANDIDATES = ("sector_code", "SectorCode", "sector33_code", "Sector33Code")


def _has_any_non_null(df: pl.DataFrame, column: str) -> bool:
    """Return True if the given column exists and has at least one non-null value."""

    if column not in df.columns:
        return False
    return df.select(pl.col(column).is_not_null().any()).item()


def _resolve_column(df: pl.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    """Return the first column name from candidates that exists in the DataFrame."""

    return next((c for c in candidates if c in df.columns), None)


def _add_gap_basis_features(
    df: pl.DataFrame,
    *,
    date_col: str,
    code_col: str,
    sector_col: Optional[str],
    window: int = 20,
    sigma: float = 2.0,
) -> pl.DataFrame:
    """Add gap_predictor / basis_gate and their derivatives.

    This closely follows DatasetBuilder._add_gap_basis_features and
    _add_basis_gate_derivatives to maintain semantic consistency.
    """

    if df.is_empty():
        return df

    # If basis_gate already has any non-null values, respect existing data.
    if _has_any_non_null(df, "basis_gate"):
        return df

    fut_overnight_col = _resolve_column(df, ("fut_topix_overnight",))
    basis_col = _resolve_column(df, ("fut_topix_basis",))
    beta_col = _resolve_column(df, ("beta60_topix",))

    if fut_overnight_col is None or basis_col is None or beta_col is None:
        # Missing inputs; skip without modifying dataset.
        print(
            f"⚠️  Skipping basis_gate: missing columns "
            f"(overnight={fut_overnight_col}, basis={basis_col}, beta={beta_col})"
        )
        return df

    # Core gap/basis features
    df = df.with_columns(
        (pl.col(fut_overnight_col) * pl.col(beta_col)).alias("gap_predictor"),
        (pl.col(basis_col) * pl.col(beta_col)).alias("basis_gate"),
        pl.when(
            pl.col(fut_overnight_col).is_not_null() & pl.col(basis_col).is_not_null() & pl.col(beta_col).is_not_null()
        )
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("is_gap_basis_valid"),
    )

    # Rolling / zscore / outlier features
    mean_col = "basis_gate_roll_mean_20d"
    std_col = "basis_gate_roll_std_20d"
    z_col = "basis_gate_zscore_20d"
    flag_col = "basis_gate_outlier_flag"

    min_periods = max(window // 2, 5)

    df = df.with_columns(
        roll_mean_safe(pl.col("basis_gate"), window, min_periods=min_periods, by=code_col).alias(mean_col),
        roll_std_safe(pl.col("basis_gate"), window, min_periods=min_periods, by=code_col).alias(std_col),
    )
    df = df.with_columns(((pl.col("basis_gate") - pl.col(mean_col)) / (pl.col(std_col) + 1e-8)).alias(z_col))
    df = df.with_columns((pl.col(z_col).abs() > sigma).cast(pl.Int8).alias(flag_col))

    # Sector-relative features
    if sector_col:
        df = df.with_columns(pl.col("basis_gate").mean().over([date_col, sector_col]).alias("basis_gate_sector_mean"))
        df = df.with_columns((pl.col("basis_gate") - pl.col("basis_gate_sector_mean")).alias("basis_gate_sector_rel"))
    else:
        if "basis_gate_sector_mean" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("basis_gate_sector_mean"))
        if "basis_gate_sector_rel" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("basis_gate_sector_rel"))

    return df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Add basis_gate (and related) features to a merged ML dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input merged dataset parquet file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output parquet file with basis_gate features attached.",
    )
    parser.add_argument(
        "--date-col",
        default="Date",
        help="Date column name.",
    )
    parser.add_argument(
        "--code-col",
        default="Code",
        help="Code column name.",
    )
    parser.add_argument(
        "--sector-col",
        default=None,
        help="Explicit sector column (auto-detected from standard names if omitted).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window size (in days) for basis_gate derivatives.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Sigma threshold for basis_gate_outlier_flag.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        print(f"❌ Input dataset not found: {input_path}")
        return 1

    print(f"📂 Loading dataset from {input_path}")
    df = pl.read_parquet(str(input_path))
    print(f"   rows={df.height:,}, cols={df.width}")

    # Determine sector column if not provided explicitly.
    sector_col = args.sector_col or _resolve_column(df, SECTOR_CANDIDATES)

    df = _add_gap_basis_features(
        df,
        date_col=args.date_col,
        code_col=args.code_col,
        sector_col=sector_col,
        window=args.window,
        sigma=args.sigma,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Writing updated dataset to {output_path}")
    df.write_parquet(str(output_path), compression="zstd")
    print("✅ Completed basis_gate feature augmentation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
