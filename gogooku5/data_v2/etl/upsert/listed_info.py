"""Upsert listed_info into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import LISTED_INFO_COLUMNS, LISTED_INFO_PK, create_table_sql


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(create_table_sql("listed_info", LISTED_INFO_COLUMNS, LISTED_INFO_PK))


def upsert_listed_info(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    ensure_table(con)
    con.register("tmp_listed", df.to_arrow())
    con.execute(
        """
        INSERT OR REPLACE INTO listed_info (
            date,
            available_ts,
            code,
            company_name,
            company_name_english,
            sector17_code,
            sector33_code,
            sector17_name,
            sector33_name,
            scale_category,
            scale_category_name,
            market_code,
            market_code_name,
            margin_code,
            margin_code_name
        )
        SELECT
            date,
            available_ts,
            code,
            company_name,
            company_name_english,
            sector17_code,
            sector33_code,
            sector17_name,
            sector33_name,
            scale_category,
            scale_category_name,
            market_code,
            market_code_name,
            margin_code,
            margin_code_name
        FROM tmp_listed
        """
    )
    con.unregister("tmp_listed")
    return df.height
