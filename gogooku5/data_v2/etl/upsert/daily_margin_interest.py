"""Upsert daily_margin_interest into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import (
    DAILY_MARGIN_INTEREST_COLUMNS,
    DAILY_MARGIN_INTEREST_PK,
    create_table_sql,
)


def upsert_daily_margin_interest(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    """Create table if needed and upsert records."""
    con.execute(create_table_sql("daily_margin_interest", DAILY_MARGIN_INTEREST_COLUMNS, DAILY_MARGIN_INTEREST_PK))
    cols = {row[1] for row in con.execute("PRAGMA table_info('daily_margin_interest')").fetchall()}
    for name, dtype in DAILY_MARGIN_INTEREST_COLUMNS:
        if name not in cols:
            con.execute(f"ALTER TABLE daily_margin_interest ADD COLUMN {name} {dtype}")

    if df.is_empty():
        return 0

    con.register("tmp_daily_margin_interest", df.to_arrow())
    col_list = ", ".join([c for c, _ in DAILY_MARGIN_INTEREST_COLUMNS])
    inserted = con.execute(
        f"""
        INSERT OR REPLACE INTO daily_margin_interest ({col_list})
        SELECT {col_list}
        FROM tmp_daily_margin_interest
        """
    ).rowcount
    con.unregister("tmp_daily_margin_interest")
    return inserted
