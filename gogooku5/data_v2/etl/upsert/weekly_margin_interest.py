"""Upsert weekly_margin_interest into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import (
    WEEKLY_MARGIN_INTEREST_COLUMNS,
    WEEKLY_MARGIN_INTEREST_PK,
    create_table_sql,
)


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(create_table_sql("weekly_margin_interest", WEEKLY_MARGIN_INTEREST_COLUMNS, WEEKLY_MARGIN_INTEREST_PK))
    cols = {row[1] for row in con.execute("PRAGMA table_info('weekly_margin_interest')").fetchall()}
    for name, dtype in WEEKLY_MARGIN_INTEREST_COLUMNS:
        if name not in cols:
            con.execute(f"ALTER TABLE weekly_margin_interest ADD COLUMN {name} {dtype}")


def upsert_weekly_margin_interest(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    ensure_table(con)
    con.register("tmp_weekly_margin_interest", df.to_arrow())
    col_list = ", ".join([c for c, _ in WEEKLY_MARGIN_INTEREST_COLUMNS])
    con.execute(
        f"""
        INSERT OR REPLACE INTO weekly_margin_interest ({col_list})
        SELECT {col_list}
        FROM tmp_weekly_margin_interest
        """
    )
    con.unregister("tmp_weekly_margin_interest")
    return df.height
