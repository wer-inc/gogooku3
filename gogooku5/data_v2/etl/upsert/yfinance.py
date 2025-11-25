"""Upsert yfinance history into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import YF_PRICE_COLUMNS, YF_PRICE_PK, create_table_sql


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(create_table_sql("yf_prices", YF_PRICE_COLUMNS, YF_PRICE_PK))


def upsert_yfinance_prices(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    ensure_table(con)
    con.register("tmp_yf_prices", df.to_arrow())
    con.execute(
        """
        INSERT OR REPLACE INTO yf_prices (
            date,
            ticker,
            open,
            high,
            low,
            close,
            adj_close,
            volume
        )
        SELECT
            date,
            ticker,
            open,
            high,
            low,
            close,
            adj_close,
            volume
        FROM tmp_yf_prices
        """
    )
    con.unregister("tmp_yf_prices")
    return df.height
