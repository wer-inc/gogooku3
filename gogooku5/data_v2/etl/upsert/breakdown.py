"""Upsert markets breakdown into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import BREAKDOWN_COLUMNS, BREAKDOWN_PK, create_table_sql


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(create_table_sql("breakdown", BREAKDOWN_COLUMNS, BREAKDOWN_PK))


def upsert_breakdown(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    ensure_table(con)
    con.register("tmp_breakdown", df.to_arrow())
    con.execute(
        """
        INSERT OR REPLACE INTO breakdown (
            date,
            code,
            long_sell_value,
            short_sell_without_margin_value,
            margin_sell_new_value,
            margin_sell_close_value,
            long_buy_value,
            margin_buy_new_value,
            margin_buy_close_value,
            long_sell_volume,
            short_sell_without_margin_volume,
            margin_sell_new_volume,
            margin_sell_close_volume,
            long_buy_volume,
            margin_buy_new_volume,
            margin_buy_close_volume
        )
        SELECT
            date,
            code,
            long_sell_value,
            short_sell_without_margin_value,
            margin_sell_new_value,
            margin_sell_close_value,
            long_buy_value,
            margin_buy_new_value,
            margin_buy_close_value,
            long_sell_volume,
            short_sell_without_margin_volume,
            margin_sell_new_volume,
            margin_sell_close_volume,
            long_buy_volume,
            margin_buy_new_volume,
            margin_buy_close_volume
        FROM tmp_breakdown
        """
    )
    con.unregister("tmp_breakdown")
    return df.height
