"""Sector/market/scale features from listed_info."""

from __future__ import annotations

import duckdb
import polars as pl


def compute_listed_features(
    con: duckdb.DuckDBPyConnection,
    *,
    start: str,
    end: str,
) -> pl.DataFrame:
    """
    Compute sector/market/scale/margin features by joining
    price_flow_features with listed_info.

    New columns added to features_daily (via upsert_listed_features):
    - sector17_code, sector33_code, market_code, scale_category, margin_code
      (copied from listed_info; point-in-time as of each date)
    - is_prime, is_standard, is_growth
      (TSE Prime/Standard/Growth flags from MarketCode)
    - is_margin_eligible
      (MarginCode == "1" → 1 else 0)
    - has_sector_meta
      (sector33_code + market_code が有効なら 1)
    - is_small_cap, is_mid_cap, is_large_cap
      (TOPIX Core30/Large70/Mid400/Small1/Small2 ベースの規模フラグ)
    - sector_ret_1d/5d/20d
      (equal-weight sector mean returns per date×sector33)
    - sector_alpha_ret_1d/5d/20d
      (stock return − sector mean)
    - sector_pct_ret_1d/5d/20d
      (sector-within percentile rank, 0–1)

    This function joins price_flow_features with listed_info to get sector / market /
    scale / margin metadata, then computes sector-level aggregations and
    stock-level metrics for a given date range.
    """
    # Load features with returns, join with listed_info for sector metadata.
    # Build a simple SCD2-style dimension from listed_info:
    # - For each code, each Date defines a start_date
    # - end_date is the day before the next Date for that code (or 9999-12-31)
    # We then as-of join features_daily on date BETWEEN start_date AND end_date.
    features_arrow = con.execute(
        """
        WITH li_raw AS (
            SELECT
                code,
                date,
                sector17_code,
                sector17_code,
                sector33_code,
                scale_category,
                market_code,
                margin_code
            FROM listed_info
            WHERE date BETWEEN ? AND ?
        ),
        li_dim AS (
            SELECT
                code,
                date AS start_date,
                LEAD(date) OVER (PARTITION BY code ORDER BY date) - INTERVAL 1 DAY AS end_date,
                sector17_code,
                sector33_code,
                scale_category,
                market_code,
                margin_code
            FROM li_raw
        ),
        li_final AS (
            SELECT
                code,
                start_date,
                COALESCE(end_date, DATE '9999-12-31') AS end_date,
                sector17_code,
                sector33_code,
                scale_category,
                market_code,
                margin_code
            FROM li_dim
        )
        SELECT
            f.date,
            f.code,
            f.ret_1d,
            f.ret_5d,
            f.ret_20d,
            li.sector17_code,
            li.sector33_code,
            li.scale_category,
            li.market_code,
            li.margin_code
        FROM price_flow_features AS f
        LEFT JOIN li_final AS li
            ON f.code = li.code
           AND f.date BETWEEN li.start_date AND li.end_date
        WHERE f.date BETWEEN ? AND ?
        ORDER BY f.date, f.code
        """,
        [start, end, start, end],
    ).fetch_arrow_table()

    df = pl.from_arrow(features_arrow)
    if df.is_empty():
        return pl.DataFrame()

    # Add scale category boolean flags
    # Note: scale_category values include:
    # - "TOPIX Small 1", "TOPIX Small 2" (small cap)
    # - "TOPIX Mid400" (mid cap)
    # - "TOPIX Core30", "TOPIX Large70" (large cap)
    # - "-" (non-TOPIX stocks)
    df = df.with_columns(
        [
            pl.col("scale_category").str.starts_with("TOPIX Small").cast(pl.Int32).alias("is_small_cap"),
            (pl.col("scale_category") == "TOPIX Mid400").cast(pl.Int32).alias("is_mid_cap"),
            ((pl.col("scale_category") == "TOPIX Core30") | (pl.col("scale_category") == "TOPIX Large70"))
            .cast(pl.Int32)
            .alias("is_large_cap"),
        ]
    )

    # Market segment flags (TSE Prime/Standard/Growth from MarketCode)
    df = df.with_columns(
        [
            pl.when(pl.col("market_code") == "0111").then(1).otherwise(0).cast(pl.Int32).alias("is_prime"),
            pl.when(pl.col("market_code") == "0112").then(1).otherwise(0).cast(pl.Int32).alias("is_standard"),
            pl.when(pl.col("market_code") == "0113").then(1).otherwise(0).cast(pl.Int32).alias("is_growth"),
        ]
    )

    # Margin eligibility (MarginCode == "1" → eligible)
    df = df.with_columns(
        pl.when(pl.col("margin_code") == "1").then(1).otherwise(0).cast(pl.Int32).alias("is_margin_eligible")
    )

    # Shortability flag: current implementation treats margin-eligible names as shortable.
    df = df.with_columns(pl.col("is_margin_eligible").alias("is_shortable"))

    # has_sector_meta: sector & market metadata present and non-placeholder
    df = df.with_columns(
        (
            pl.col("sector33_code").is_not_null()
            & (pl.col("sector33_code") != "UNKNOWN")
            & (pl.col("sector33_code") != "-")
            & (pl.col("sector33_code") != "")
            & pl.col("market_code").is_not_null()
        )
        .cast(pl.Int32)
        .alias("has_sector_meta")
    )

    # Compute sector mean returns (equal-weight, per date x sector33).
    # Reuse has_sector_meta as the single source of truth for "valid sector/meta".
    valid_sector_mask = pl.col("has_sector_meta") == 1

    sector_agg = (
        df.lazy()
        .filter(valid_sector_mask)
        .group_by(["date", "sector33_code"])
        .agg(
            [
                pl.col("ret_1d").mean().alias("sector_ret_1d"),
                pl.col("ret_5d").mean().alias("sector_ret_5d"),
                pl.col("ret_20d").mean().alias("sector_ret_20d"),
            ]
        )
        .collect()
    )

    # Join sector aggregates back to main DataFrame
    df = df.join(sector_agg, on=["date", "sector33_code"], how="left")

    # Compute sector-neutral alpha (stock return - sector mean)
    df = df.with_columns(
        [
            (pl.col("ret_1d") - pl.col("sector_ret_1d")).alias("sector_alpha_ret_1d"),
            (pl.col("ret_5d") - pl.col("sector_ret_5d")).alias("sector_alpha_ret_5d"),
            (pl.col("ret_20d") - pl.col("sector_ret_20d")).alias("sector_alpha_ret_20d"),
        ]
    )

    # Compute sector-within percentile rank (0-1)
    # Using rank().over() with proper null handling
    # Only compute for rows with valid sector
    df = (
        df.lazy()
        .with_columns(
            [
                # Percentile rank within sector (0 = worst, 1 = best)
                pl.when(valid_sector_mask)
                .then(
                    pl.col("ret_1d").rank(method="average").over(["date", "sector33_code"])
                    / pl.col("ret_1d").count().over(["date", "sector33_code"])
                )
                .otherwise(None)
                .alias("sector_pct_ret_1d"),
                pl.when(valid_sector_mask)
                .then(
                    pl.col("ret_5d").rank(method="average").over(["date", "sector33_code"])
                    / pl.col("ret_5d").count().over(["date", "sector33_code"])
                )
                .otherwise(None)
                .alias("sector_pct_ret_5d"),
                pl.when(valid_sector_mask)
                .then(
                    pl.col("ret_20d").rank(method="average").over(["date", "sector33_code"])
                    / pl.col("ret_20d").count().over(["date", "sector33_code"])
                )
                .otherwise(None)
                .alias("sector_pct_ret_20d"),
                # rank版（0-1）は 5日リターンだけ明示的に列を持つ
                pl.when(valid_sector_mask)
                .then(
                    pl.col("ret_5d").rank(method="average").over(["date", "sector33_code"])
                    / pl.col("ret_5d").count().over(["date", "sector33_code"])
                )
                .otherwise(None)
                .alias("sector_rank_ret_5d"),
            ]
        )
        .collect()
    )

    # Select only the columns we're adding (to be used by upsert)
    result_cols = [
        "date",
        "code",
        # raw meta
        "sector17_code",
        "sector33_code",
        "market_code",
        "scale_category",
        "margin_code",
        # flags
        "is_prime",
        "is_standard",
        "is_growth",
        "is_margin_eligible",
        "is_shortable",
        "has_sector_meta",
        "is_small_cap",
        "is_mid_cap",
        "is_large_cap",
        "sector_ret_1d",
        "sector_ret_5d",
        "sector_ret_20d",
        "sector_alpha_ret_1d",
        "sector_alpha_ret_5d",
        "sector_alpha_ret_20d",
        "sector_pct_ret_1d",
        "sector_pct_ret_5d",
        "sector_pct_ret_20d",
        "sector_rank_ret_5d",
    ]

    return df.select(result_cols)
