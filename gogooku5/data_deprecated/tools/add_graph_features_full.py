#!/usr/bin/env python3
"""Add full graph feature suite to a merged ML dataset.

This tool is designed to run **after** chunk-level builds and yearly merging.
It applies `GraphFeatureEngineer` on the merged parquet to create:

* Base graph features (3 columns)
    - ``graph_degree``
    - ``graph_peer_corr_mean``
    - ``graph_peer_corr_max``
* Optional derived features (10 per base feature = 30 columns total)
    - Cross-sectional (per Date):
        - ``*_cs_rank``, ``*_cs_pct``,
          ``*_cs_top20_flag``, ``*_cs_bottom20_flag``
    - Rolling statistics (per Code, 20d window):
        - ``*_roll_mean_20d``, ``*_roll_std_20d``, ``*_zscore_20d``
    - Outlier flag:
        - ``*_outlier_flag`` (|z| > 2.0)
    - Sector relative (per Date × sector):
        - ``*_sector_mean``, ``*_sector_rel``

Intended usage (example for 2025 dataset):

.. code-block:: bash

    PYTHONPATH=gogooku5/data/src \\
      python gogooku5/data/tools/add_graph_features_full.py \\
        --input  data/output/datasets/ml_dataset_2025_base.parquet \\
        --output data/output/datasets/ml_dataset_2025_with_graph33.parquet \\
        --date-col Date --code-col Code \\
        --window-days 60 --min-observations 20 --correlation-threshold 0.3 \\
        --with-derived
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import polars as pl
from builder.features.core.graph.features import (
    GraphFeatureConfig,
    GraphFeatureEngineer,
)

GRAPH_BASE_FEATURES: tuple[str, ...] = ("graph_degree", "graph_peer_corr_mean", "graph_peer_corr_max")
DERIVED_ROLL_WINDOW_DAYS: int = 20
DERIVED_MIN_PERIODS: int = 10
DERIVED_OUTLIER_SIGMA: float = 2.0
SECTOR_CANDIDATES: tuple[str, ...] = ("sector_code", "SectorCode", "sector33_code", "Sector33Code")


def _has_any_non_null(df: pl.DataFrame, column: str) -> bool:
    """Return True if the given column exists and has at least one non-null value."""

    if column not in df.columns:
        return False
    return df.select(pl.col(column).is_not_null().any()).item()


def _drop_existing_graph_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Remove any pre-existing graph_* columns to avoid schema conflicts."""

    graph_cols = [c for c in df.columns if c.startswith("graph_")]
    if not graph_cols:
        return df
    return df.drop(graph_cols)


def _add_base_graph_features(
    df: pl.DataFrame,
    *,
    date_col: str,
    code_col: str,
    window_days: int,
    min_observations: int,
    correlation_threshold: float,
) -> pl.DataFrame:
    """Apply GraphFeatureEngineer and attach base graph features."""

    if df.is_empty():
        return df

    if not {date_col, code_col, "ret_prev_1d"}.issubset(df.columns):
        return df

    # Ensure join keys are string-typed to avoid Categorical/Utf8 mismatches during graph joins
    df = df.with_columns(pl.col(code_col).cast(pl.Utf8))

    config = GraphFeatureConfig(
        code_column=code_col,
        date_column=date_col,
        return_column="ret_prev_1d",
        window_days=window_days,
        min_observations=min_observations,
        correlation_threshold=correlation_threshold,
        shift_to_next_day=True,
        block_size=512,
    )
    engineer = GraphFeatureEngineer(config=config)
    return engineer.add_features(df)


def _add_cross_sectional_quantiles(
    df: pl.DataFrame,
    *,
    date_col: str,
    features: Iterable[str],
) -> pl.DataFrame:
    """Add *_cs_rank / *_cs_pct / *_cs_top20_flag / *_cs_bottom20_flag."""

    for feature in features:
        if feature not in df.columns or not _has_any_non_null(df, feature):
            continue

        rank_col = f"{feature}_cs_rank"
        count_col = f"_{feature}_cs_count"
        pct_col = f"{feature}_cs_pct"
        top_flag = f"{feature}_cs_top20_flag"
        bottom_flag = f"{feature}_cs_bottom20_flag"

        df = df.with_columns(
            [
                pl.col(feature).rank(method="ordinal").over(date_col).alias(rank_col),
                pl.count().over(date_col).alias(count_col),
            ]
        )

        df = df.with_columns(
            pl.when(pl.col(count_col) > 1)
            .then((pl.col(rank_col) - 1.0) / (pl.col(count_col) - 1.0))
            .otherwise(0.5)
            .alias(pct_col)
        )

        df = df.with_columns(
            [
                (pl.col(pct_col) >= 0.8).cast(pl.Int8).alias(top_flag),
                (pl.col(pct_col) <= 0.2).cast(pl.Int8).alias(bottom_flag),
            ]
        )

        df = df.drop(count_col)

    return df


def _add_rolling_and_outlier_features(
    df: pl.DataFrame,
    *,
    date_col: str,
    code_col: str,
    features: Iterable[str],
    window: int,
    min_periods: int,
    sigma: float,
) -> pl.DataFrame:
    """Attach *_roll_mean_20d / *_roll_std_20d / *_zscore_20d / *_outlier_flag."""

    if df.is_empty():
        return df

    df_sorted = df.sort([code_col, date_col])

    for feature in features:
        if feature not in df_sorted.columns or not _has_any_non_null(df_sorted, feature):
            continue

        mean_col = f"{feature}_roll_mean_{window}d"
        std_col = f"{feature}_roll_std_{window}d"
        z_col = f"{feature}_zscore_{window}d"
        flag_col = f"{feature}_outlier_flag"

        def _add_group(group: pl.DataFrame) -> pl.DataFrame:
            if group.is_empty():
                return group

            group = group.sort(date_col)
            series = group[feature]
            shifted = series.shift(1)
            mean_vals = shifted.rolling_mean(window_size=window, min_periods=min_periods)
            std_vals = shifted.rolling_std(window_size=window, min_periods=min_periods)

            z_vals = (series - mean_vals) / (std_vals + 1e-8)
            flags = (abs(z_vals) > sigma).cast(pl.Int8)

            new_cols: List[pl.Series] = [
                pl.Series(name=mean_col, values=mean_vals).cast(pl.Float64, strict=False),
                pl.Series(name=std_col, values=std_vals).cast(pl.Float64, strict=False),
                pl.Series(name=z_col, values=z_vals).cast(pl.Float64, strict=False),
                pl.Series(name=flag_col, values=flags).cast(pl.Int8, strict=False),
            ]
            return group.with_columns(new_cols)

        df_sorted = df_sorted.group_by(code_col, maintain_order=True).map_groups(_add_group).sort([code_col, date_col])

    return df_sorted


def _add_sector_relative_features(
    df: pl.DataFrame,
    *,
    date_col: str,
    sector_col: str | None,
    features: Iterable[str],
    sector_candidates: Iterable[str] = SECTOR_CANDIDATES,
) -> pl.DataFrame:
    """Attach *_sector_mean / *_sector_rel for each feature."""

    if df.is_empty():
        return df

    if sector_col is None:
        sector_col = next((c for c in sector_candidates if c in df.columns), None)

    if sector_col is None:
        # No sector information available; synthesise NULL columns to keep schema stable.
        for feature in features:
            if feature not in df.columns:
                continue
            mean_col = f"{feature}_sector_mean"
            rel_col = f"{feature}_sector_rel"
            if mean_col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(mean_col))
            if rel_col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(rel_col))
        return df

    for feature in features:
        if feature not in df.columns or not _has_any_non_null(df, feature):
            continue

        mean_col = f"{feature}_sector_mean"
        rel_col = f"{feature}_sector_rel"

        sector_means = df.group_by([date_col, sector_col], maintain_order=False).agg(
            pl.col(feature).mean().alias(mean_col)
        )
        df = df.join(sector_means, on=[date_col, sector_col], how="left")
        df = df.with_columns((pl.col(feature) - pl.col(mean_col)).alias(rel_col))

    return df


def _add_derived_graph_features(
    df: pl.DataFrame,
    *,
    date_col: str,
    code_col: str,
    sector_col: str | None = None,
) -> pl.DataFrame:
    """Attach cross-sectional, rolling, outlier, and sector-relative graph features."""

    base_features = [f for f in GRAPH_BASE_FEATURES if _has_any_non_null(df, f)]
    if not base_features:
        return df

    df = _add_cross_sectional_quantiles(df, date_col=date_col, features=base_features)
    df = _add_rolling_and_outlier_features(
        df,
        date_col=date_col,
        code_col=code_col,
        features=base_features,
        window=DERIVED_ROLL_WINDOW_DAYS,
        min_periods=DERIVED_MIN_PERIODS,
        sigma=DERIVED_OUTLIER_SIGMA,
    )
    df = _add_sector_relative_features(df, date_col=date_col, sector_col=sector_col, features=base_features)
    return df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for graph feature augmentation."""

    parser = argparse.ArgumentParser(
        description="Add base and derived graph features to a merged ML dataset.",
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
        help="Output parquet file with graph features attached.",
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
        "--window-days",
        type=int,
        default=60,
        help="Correlation window in days for GraphFeatureEngineer.",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=20,
        help="Minimum observations required for correlation statistics.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.3,
        help="Peer correlation threshold for graph edge construction.",
    )
    parser.add_argument(
        "--sector-col",
        default=None,
        help="Explicit sector column for *_sector_mean/_sector_rel (auto-detected if omitted).",
    )
    parser.add_argument(
        "--with-derived",
        action="store_true",
        help="Also compute derived graph features (cross-sectional, rolling, sector-relative).",
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

    # Drop any pre-existing graph_* columns to avoid mixing schemas.
    df = _drop_existing_graph_columns(df)

    # Base graph features (3 columns)
    df = _add_base_graph_features(
        df,
        date_col=args.date_col,
        code_col=args.code_col,
        window_days=args.window_days,
        min_observations=args.min_observations,
        correlation_threshold=args.correlation_threshold,
    )

    # Derived graph features (optional)
    if args.with_derived:
        df = _add_derived_graph_features(
            df,
            date_col=args.date_col,
            code_col=args.code_col,
            sector_col=args.sector_col,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Writing updated dataset to {output_path}")
    df.write_parquet(str(output_path), compression="zstd")
    print("✅ Completed graph feature augmentation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
