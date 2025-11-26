"""Upsert short_selling into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import SHORT_SELLING_COLUMNS, SHORT_SELLING_PK, create_table_sql


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(create_table_sql("short_selling", SHORT_SELLING_COLUMNS, SHORT_SELLING_PK))
    cols = {row[1] for row in con.execute("PRAGMA table_info('short_selling')").fetchall()}
    for name, dtype in SHORT_SELLING_COLUMNS:
        if name not in cols:
            con.execute(f"ALTER TABLE short_selling ADD COLUMN {name} {dtype}")


def upsert_short_selling(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    ensure_table(con)
    con.register("tmp_short_selling", df.to_arrow())
    col_list = ", ".join([c for c, _ in SHORT_SELLING_COLUMNS])
    con.execute(
        f"""
        INSERT OR REPLACE INTO short_selling ({col_list})
        SELECT {col_list}
        FROM tmp_short_selling
        """
    )
    con.unregister("tmp_short_selling")
    return df.height
