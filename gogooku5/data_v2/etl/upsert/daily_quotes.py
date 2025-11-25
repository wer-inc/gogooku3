"""Upsert daily_quotes into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import DAILY_QUOTES_COLUMNS, DAILY_QUOTES_PK, create_table_sql


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(create_table_sql("daily_quotes", DAILY_QUOTES_COLUMNS, DAILY_QUOTES_PK))


def upsert_daily_quotes(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    ensure_table(con)
    con.register("tmp_daily_quotes", df.to_arrow())
    con.execute(
        """
        INSERT OR REPLACE INTO daily_quotes (
            date,
            code,
            open,
            high,
            low,
            close,
            upper_limit,
            lower_limit,
            volume,
            turnover_value,
            adjustment_factor,
            adjustment_open,
            adjustment_high,
            adjustment_low,
            adjustment_close,
            adjustment_volume,
            morning_open,
            morning_high,
            morning_low,
            morning_close,
            morning_upper_limit,
            morning_lower_limit,
            morning_volume,
            morning_turnover_value,
            morning_adjustment_open,
            morning_adjustment_high,
            morning_adjustment_low,
            morning_adjustment_close,
            morning_adjustment_volume,
            afternoon_open,
            afternoon_high,
            afternoon_low,
            afternoon_close,
            afternoon_upper_limit,
            afternoon_lower_limit,
            afternoon_volume,
            afternoon_turnover_value,
            afternoon_adjustment_open,
            afternoon_adjustment_high,
            afternoon_adjustment_low,
            afternoon_adjustment_close,
            afternoon_adjustment_volume
        )
        SELECT
            date,
            code,
            open,
            high,
            low,
            close,
            upper_limit,
            lower_limit,
            volume,
            turnover_value,
            adjustment_factor,
            adjustment_open,
            adjustment_high,
            adjustment_low,
            adjustment_close,
            adjustment_volume,
            morning_open,
            morning_high,
            morning_low,
            morning_close,
            morning_upper_limit,
            morning_lower_limit,
            morning_volume,
            morning_turnover_value,
            morning_adjustment_open,
            morning_adjustment_high,
            morning_adjustment_low,
            morning_adjustment_close,
            morning_adjustment_volume,
            afternoon_open,
            afternoon_high,
            afternoon_low,
            afternoon_close,
            afternoon_upper_limit,
            afternoon_lower_limit,
            afternoon_volume,
            afternoon_turnover_value,
            afternoon_adjustment_open,
            afternoon_adjustment_high,
            afternoon_adjustment_low,
            afternoon_adjustment_close,
            afternoon_adjustment_volume
        FROM tmp_daily_quotes
        """
    )
    con.unregister("tmp_daily_quotes")
    return df.height
