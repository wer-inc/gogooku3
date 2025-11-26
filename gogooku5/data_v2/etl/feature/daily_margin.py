"""Daily margin interest-derived features."""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import polars as pl


def compute_daily_margin_features(
    con: duckdb.DuckDBPyConnection,
    *,
    start: str,
    end: str,
    warmup_days: int = 7,
) -> pl.DataFrame:
    """
    Project daily_margin_interest onto trading days with leak-safe effective dates.

    Effective date = next trading day after application_date (T+1) to avoid look-ahead.
    Returned columns are aligned to the price_flow grid for join_asof in compute_price_features.
    """
    # trading calendar for forward lookup of next trading day
    cal_arrow = con.execute(
        """
        SELECT date AS cal_date
        FROM trading_calendar
        WHERE holiday_division IN ('1','2')
        ORDER BY date
        """
    ).fetch_arrow_table()
    cal_df = pl.from_arrow(cal_arrow)
    if cal_df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "code": pl.Utf8})

    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    window_start = (start_dt - timedelta(days=warmup_days)).isoformat()

    dmi_arrow = con.execute(
        """
        SELECT
            application_date,
            published_date,
            code,
            short_margin_outstanding,
            long_margin_outstanding,
            short_margin_outstanding_listed_share_ratio,
            tse_margin_regulation_classification,
            precaution_by_jsf,
            restricted_by_jsf
        FROM daily_margin_interest
        WHERE application_date BETWEEN ? AND ?
        """,
        [window_start, end],
    ).fetch_arrow_table()
    dmi = pl.from_arrow(dmi_arrow)
    if dmi.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "code": pl.Utf8})

    dmi = dmi.with_columns(
        [
            pl.col("application_date").cast(pl.Date, strict=False),
            pl.col("published_date").cast(pl.Date, strict=False),
            pl.col("code").cast(pl.Utf8, strict=False),
            pl.col("short_margin_outstanding").cast(pl.Float64, strict=False),
            pl.col("long_margin_outstanding").cast(pl.Float64, strict=False),
            pl.col("short_margin_outstanding_listed_share_ratio").cast(pl.Float64, strict=False),
            pl.col("tse_margin_regulation_classification").cast(pl.Utf8, strict=False),
            pl.col("precaution_by_jsf").cast(pl.Utf8, strict=False),
            pl.col("restricted_by_jsf").cast(pl.Utf8, strict=False),
        ]
    )

    # effective_date = next trading day after application_date
    dmi = dmi.with_columns((pl.col("application_date") + pl.duration(days=1)).alias("_app_plus1"))
    dmi = (
        dmi.join_asof(cal_df, left_on="_app_plus1", right_on="cal_date", strategy="forward")
        .rename({"cal_date": "effective_date"})
        .drop("_app_plus1")
    )
    dmi = dmi.filter(pl.col("effective_date").is_not_null()).sort(["code", "effective_date"])

    # Build daily grid from daily_quotes; restrict to codes with margin data for efficiency
    grid_arrow = con.execute(
        """
        SELECT date, code
        FROM daily_quotes
        WHERE date BETWEEN ? AND ?
        ORDER BY code, date
        """,
        [start, end],
    ).fetch_arrow_table()
    grid = pl.from_arrow(grid_arrow)
    if grid.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "code": pl.Utf8})

    codes_with_dmi = dmi.select(pl.col("code").unique()).to_series()
    if codes_with_dmi.len() > 0 and codes_with_dmi.len() < grid["code"].n_unique():
        grid = grid.filter(pl.col("code").is_in(codes_with_dmi))

    # Sort both frames and set sorted flags for join_asof
    grid = grid.sort(["code", "date"])
    dmi = dmi.sort(["code", "effective_date"])
    grid = grid.with_columns(pl.col("date").set_sorted())
    dmi = dmi.with_columns(pl.col("effective_date").set_sorted())
    daily = grid.join_asof(
        dmi,
        left_on="date",
        right_on="effective_date",
        by="code",
        strategy="backward",
    )

    return daily.select(
        [
            "date",
            "code",
            pl.col("short_margin_outstanding").alias("dmi_short_balance"),
            pl.col("long_margin_outstanding").alias("dmi_long_balance"),
            pl.col("short_margin_outstanding_listed_share_ratio").alias("dmi_short_balance_listed_ratio"),
            pl.col("tse_margin_regulation_classification").alias("dmi_regulation_code"),
            pl.col("precaution_by_jsf"),
            pl.col("restricted_by_jsf"),
        ]
    )
