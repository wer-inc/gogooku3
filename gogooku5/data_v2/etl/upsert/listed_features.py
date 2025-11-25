"""Upsert listed_info-derived features into listed_meta_features."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import (
    LISTED_META_FEATURES_COLUMNS,
    LISTED_META_FEATURES_PK,
    LISTED_META_FEATURES_TABLE,
    create_table_sql,
)

# Columns computed by compute_listed_features() that need to be updated
LISTED_FEATURE_COLUMNS = [
    # Raw meta copied from listed_info
    "sector17_code",
    "sector33_code",
    "market_code",
    "scale_category",
    "margin_code",
    # Market / margin flags
    "is_prime",
    "is_standard",
    "is_growth",
    "is_margin_eligible",
    "is_shortable",
    "has_sector_meta",
    # Scale flags
    "is_small_cap",
    "is_mid_cap",
    "is_large_cap",
    # Sector returns (equal-weight per date x sector33)
    "sector_ret_1d",
    "sector_ret_5d",
    "sector_ret_20d",
    # Sector-neutral alpha
    "sector_alpha_ret_1d",
    "sector_alpha_ret_5d",
    "sector_alpha_ret_20d",
    # Sector-within percentile ranks
    "sector_pct_ret_1d",
    "sector_pct_ret_5d",
    "sector_pct_ret_20d",
    # Explicit rank column for 5d returns
    "sector_rank_ret_5d",
]


def upsert_listed_features(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    """
    Upsert listed_info-derived features into listed_meta_features.

    This writes a separate meta table keyed by (date, code) so that
    price_flow_features remains focused on price / flow features.

    Args:
        con: DuckDB connection
        df: DataFrame with columns [date, code, ...LISTED_FEATURE_COLUMNS]

    Returns:
        Number of rows upserted
    """
    if df.is_empty():
        return 0

    # Ensure destination table exists and is migrated.
    con.execute(create_table_sql(LISTED_META_FEATURES_TABLE, LISTED_META_FEATURES_COLUMNS, LISTED_META_FEATURES_PK))

    con.register("tmp_listed_features", df.to_arrow())
    cols = ["date", "code"] + LISTED_FEATURE_COLUMNS
    col_list = ", ".join(cols)
    col_values = ", ".join(f"tmp.{c}" for c in cols)

    con.execute(
        f"""
        INSERT OR REPLACE INTO {LISTED_META_FEATURES_TABLE} (
            {col_list}
        )
        SELECT
            {col_values}
        FROM tmp_listed_features AS tmp
        """
    )

    con.unregister("tmp_listed_features")
    return df.height
