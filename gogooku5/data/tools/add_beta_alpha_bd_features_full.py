#!/usr/bin/env python3
"""Post-process merged datasets to add beta/alpha (and derivatives) plus bd_net_adv60.

This tool operates on an already-merged yearly (or multi-year) ML dataset and
adds/repairs the following feature families:

* beta60_topix / alpha60_topix
    - 60-day rolling beta/alpha vs TOPIX returns
    - Computed from:
        - per-security returns (ret_prev_1d)
        - TOPIX close prices (topix_close) within the same dataset
    - Rolling/sector derivatives:
        - {beta,alpha}_roll_mean_20d / roll_std_20d / zscore_20d
        - {beta,alpha}_outlier_flag and sector_{mean,rel}
* bd_net_adv60
    - Breakdown net value scaled by 60-day ADV proxy
    - Computed from:
        - bd_net_value
        - dollar_volume (used to derive a 60-day rolling ADV in JPY)

The script is designed to be idempotent and conservative:
* If a target column already exists and contains non-NULL values, it is left
  unchanged.
* If required source columns are missing, the corresponding feature family
  is skipped.

Example usage:

    python gogooku5/data/tools/add_beta_alpha_bd_features_full.py \\
        --input output_g5/datasets/ml_dataset_2024_full.parquet \\
        --output output_g5/datasets/ml_dataset_2024_full_with_beta_bd.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import polars as pl


BETA_EPS: float = 1e-12


def _has_any_non_null(df: pl.DataFrame, column: str) -> bool:
    """Return True if the given column exists and has at least one non-null value."""

    if column not in df.columns:
        return False
    return df.select(pl.col(column).is_not_null().any()).item()


def _needs_population(df: pl.DataFrame, column: str) -> bool:
    """Return True if the column is missing or entirely NULL."""

    if column not in df.columns:
        return True
    return not _has_any_non_null(df, column)


def _add_beta_alpha_from_dataset(
    df: pl.DataFrame,
    *,
    date_col: str = "Date",
    code_col: str = "Code",
    ret_col: str = "ret_prev_1d",
    window: int = 60,
    min_periods: int = 30,
) -> pl.DataFrame:
    """Add beta60_topix / alpha60_topix using dataset-level TOPIX close prices.

    This implementation mirrors the logic in DatasetBuilder._add_beta_alpha_features
    but uses the merged dataset itself as the TOPIX source (topix_close).
    """

    if df.is_empty():
        return df

    required = {date_col, code_col, ret_col, "topix_close"}
    if not required.issubset(df.columns):
        return df

    # If beta/alpha already have any non-null values, do not overwrite.
    if _has_any_non_null(df, "beta60_topix") or _has_any_non_null(df, "alpha60_topix"):
        return df

    # Derive TOPIX returns from topix_close (one value per date).
    topix_ret = (
        df.select([pl.col(date_col), pl.col("topix_close")])
        .drop_nulls([date_col, "topix_close"])
        .unique(subset=[date_col], keep="first")
        .sort(date_col)
        .with_columns(
            (
                (pl.col("topix_close") / (pl.col("topix_close").shift(1) + BETA_EPS)) - 1.0
            ).alias("topix_ret_1d")
        )
        .select([date_col, "topix_ret_1d"])
    )
    if topix_ret.is_empty():
        return df

    df_with_topix = df.join(topix_ret, on=date_col, how="left")
    df_sorted = df_with_topix.sort([code_col, date_col])

    # Work at group level to avoid nested window expressions.
    def _add_beta_alpha_group(group: pl.DataFrame) -> pl.DataFrame:
        """Compute rolling beta/alpha within a single Code group."""

        if group.is_empty():
            return group

        group = group.sort(date_col)
        stock_ret = group[ret_col]
        topix_ret_series = group["topix_ret_1d"]

        # Shift by 1 for left-closed window
        stock_shifted = stock_ret.shift(1)
        topix_shifted = topix_ret_series.shift(1)

        mean_stock = stock_shifted.rolling_mean(window_size=window, min_periods=min_periods)
        mean_topix = topix_shifted.rolling_mean(window_size=window, min_periods=min_periods)

        stock_centered = stock_shifted - mean_stock
        topix_centered = topix_shifted - mean_topix

        var_topix = (topix_centered ** 2).rolling_mean(window_size=window, min_periods=min_periods)
        cov_stock_topix = (stock_centered * topix_centered).rolling_mean(
            window_size=window, min_periods=min_periods
        )

        beta = cov_stock_topix / (var_topix + BETA_EPS)
        alpha = mean_stock - beta * mean_topix

        group = group.with_columns(
            [
                pl.Series(name="beta60_topix", values=beta).cast(pl.Float64, strict=False),
                pl.Series(name="alpha60_topix", values=alpha).cast(pl.Float64, strict=False),
            ]
        )
        return group

    df_beta = (
        df_sorted.group_by(code_col, maintain_order=True)
        .map_groups(_add_beta_alpha_group)
        .sort([code_col, date_col])
    )

    # Clean up temporary TOPIX returns if they are not part of the schema contract.
    if "topix_ret_1d" in df_beta.columns:
        df_beta = df_beta.drop("topix_ret_1d")

    return df_beta


def _add_bd_net_adv60_from_dollar_volume(
    df: pl.DataFrame,
    *,
    date_col: str = "Date",
    code_col: str = "Code",
    window: int = 60,
    min_periods: int = 10,
) -> pl.DataFrame:
    """Add bd_net_adv60 using a 60-day ADV proxy derived from dollar_volume.

    This mirrors the fallback logic in DatasetBuilder._attach_breakdown_features:
    - If adv60_yen/_adv60_yen are not present but dollar_volume is, derive
      _adv60_bd as a rolling 60-day mean of dollar_volume (shifted by 1 day),
      and compute:
          bd_net_adv60 = bd_net_value / (_adv60_bd + eps)
    """

    if df.is_empty():
        return df

    required = {date_col, code_col, "bd_net_value", "dollar_volume"}
    if not required.issubset(df.columns):
        return df

    # If bd_net_adv60 already has any non-null values, do not overwrite.
    if _has_any_non_null(df, "bd_net_adv60"):
        return df

    df_sorted = df.sort([code_col, date_col])
    df_adv = df_sorted.with_columns(
        pl.col("dollar_volume")
        .shift(1)
        .rolling_mean(window_size=window, min_periods=min_periods)
        .over(code_col)
        .alias("_adv60_bd")
    )

    df_adv = df_adv.with_columns(
        (pl.col("bd_net_value") / (pl.col("_adv60_bd") + 1e-9)).alias("bd_net_adv60"),
    )
    df_adv = df_adv.drop("_adv60_bd")

    df_adv = df_adv.with_columns(pl.col("bd_net_adv60").cast(pl.Float64, strict=False))
    return df_adv


def _add_beta_alpha_derivatives(
    df: pl.DataFrame,
    *,
    date_col: str = "Date",
    code_col: str = "Code",
    sector_col: str | None = None,
    rolling_window: int = 20,
    min_periods: int | None = None,
    sigma: float = 2.0,
    sector_candidates: Iterable[str] = ("sector_code", "SectorCode", "sector33_code", "Sector33Code"),
) -> pl.DataFrame:
    """Attach rolling / sector features for beta60_topix and alpha60_topix."""

    features = [col for col in ("beta60_topix", "alpha60_topix") if _has_any_non_null(df, col)]
    if not features:
        return df

    if min_periods is None:
        min_periods = max(rolling_window // 2, 5)

    df_sorted = df.sort([code_col, date_col])

    for feature in features:
        mean_col = f"{feature}_roll_mean_{rolling_window}d"
        std_col = f"{feature}_roll_std_{rolling_window}d"
        z_col = f"{feature}_zscore_{rolling_window}d"
        flag_col = f"{feature}_outlier_flag"

        needs_mean = _needs_population(df_sorted, mean_col)
        needs_std = _needs_population(df_sorted, std_col)
        needs_z = _needs_population(df_sorted, z_col)
        needs_flag = _needs_population(df_sorted, flag_col)

        if any([needs_mean, needs_std, needs_z, needs_flag]):
            window_mean = rolling_window
            window_std = rolling_window

            def _add_group(group: pl.DataFrame) -> pl.DataFrame:
                """Compute rolling stats within a single security group."""

                if group.is_empty():
                    return group

                group = group.sort(date_col)
                shifted = group[feature].shift(1)
                new_cols: list[pl.Series] = []

                mean_vals = None
                std_vals = None

                if needs_mean or needs_z:
                    mean_vals = shifted.rolling_mean(window_size=window_mean, min_periods=min_periods)
                    if needs_mean:
                        new_cols.append(pl.Series(name=mean_col, values=mean_vals).cast(pl.Float64, strict=False))

                if needs_std or needs_z:
                    std_vals = shifted.rolling_std(window_size=window_std, min_periods=min_periods)
                    if needs_std:
                        new_cols.append(pl.Series(name=std_col, values=std_vals).cast(pl.Float64, strict=False))

                if needs_z or needs_flag:
                    if mean_vals is None:
                        mean_vals = shifted.rolling_mean(window_size=window_mean, min_periods=min_periods)
                    if std_vals is None:
                        std_vals = shifted.rolling_std(window_size=window_std, min_periods=min_periods)
                    z_vals = (group[feature] - mean_vals) / (std_vals + 1e-8)
                    if needs_z:
                        new_cols.append(pl.Series(name=z_col, values=z_vals).cast(pl.Float64, strict=False))
                    if needs_flag:
                        flags = (abs(z_vals) > sigma).cast(pl.Int8, strict=False)
                        new_cols.append(pl.Series(name=flag_col, values=flags).cast(pl.Int8, strict=False))

                return group.with_columns(new_cols) if new_cols else group

            df_sorted = (
                df_sorted.group_by(code_col, maintain_order=True)
                .map_groups(_add_group)
                .sort([code_col, date_col])
            )

    if sector_col is None:
        sector_col = next((candidate for candidate in sector_candidates if candidate in df_sorted.columns), None)

    if sector_col:
        for feature in features:
            mean_col = f"{feature}_sector_mean"
            rel_col = f"{feature}_sector_rel"
            if _needs_population(df_sorted, mean_col):
                sector_means = (
                    df_sorted.group_by([date_col, sector_col], maintain_order=False)
                    .agg(pl.col(feature).mean().alias(mean_col))
                )
                df_sorted = df_sorted.join(sector_means, on=[date_col, sector_col], how="left")
            if _needs_population(df_sorted, rel_col):
                df_sorted = df_sorted.with_columns((pl.col(feature) - pl.col(mean_col)).alias(rel_col))
    else:
        # Ensure schema columns exist even if sector info is unavailable.
        for feature in features:
            mean_col = f"{feature}_sector_mean"
            rel_col = f"{feature}_sector_rel"
            if mean_col not in df_sorted.columns:
                df_sorted = df_sorted.with_columns(pl.lit(None).cast(pl.Float64).alias(mean_col))
            if rel_col not in df_sorted.columns:
                df_sorted = df_sorted.with_columns(pl.lit(None).cast(pl.Float64).alias(rel_col))

    return df_sorted


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Add beta60_topix/alpha60_topix and bd_net_adv60 to a merged ML dataset."
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
        help="Output parquet file with beta/alpha and bd_net_adv60 attached.",
    )
    parser.add_argument(
        "--date-col",
        default="Date",
        help="Date column name (default: Date).",
    )
    parser.add_argument(
        "--code-col",
        default="Code",
        help="Code column name (default: Code).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Rolling window size for beta/alpha and ADV (default: 60).",
    )
    parser.add_argument(
        "--min-periods-beta",
        type=int,
        default=30,
        help="Minimum periods for beta/alpha rolling stats (default: 30).",
    )
    parser.add_argument(
        "--beta-deriv-window",
        type=int,
        default=20,
        help="Rolling window for beta/alpha derivative stats (default: 20).",
    )
    parser.add_argument(
        "--beta-deriv-min-periods",
        type=int,
        default=10,
        help="Minimum periods for beta/alpha derivative stats (default: 10).",
    )
    parser.add_argument(
        "--beta-deriv-sigma",
        type=float,
        default=2.0,
        help="Sigma threshold for beta/alpha outlier flags (default: 2.0).",
    )
    parser.add_argument(
        "--sector-col",
        default=None,
        help="Explicit sector column for sector_mean/rel (auto-detected if omitted).",
    )
    parser.add_argument(
        "--min-periods-adv",
        type=int,
        default=10,
        help="Minimum periods for ADV rolling stats (default: 10).",
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

    # Add beta/alpha (if possible)
    df = _add_beta_alpha_from_dataset(
        df,
        date_col=args.date_col,
        code_col=args.code_col,
        ret_col="ret_prev_1d",
        window=args.window,
        min_periods=args.min_periods_beta,
    )
    df = _add_beta_alpha_derivatives(
        df,
        date_col=args.date_col,
        code_col=args.code_col,
        sector_col=args.sector_col,
        rolling_window=args.beta_deriv_window,
        min_periods=args.beta_deriv_min_periods,
        sigma=args.beta_deriv_sigma,
    )

    # Add bd_net_adv60 (if possible)
    df = _add_bd_net_adv60_from_dollar_volume(
        df,
        date_col=args.date_col,
        code_col=args.code_col,
        window=args.window,
        min_periods=args.min_periods_adv,
    )

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Writing updated dataset to {output_path}")
    df.write_parquet(str(output_path), compression="zstd")
    print("✅ Completed beta/alpha and bd_net_adv60 augmentation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
