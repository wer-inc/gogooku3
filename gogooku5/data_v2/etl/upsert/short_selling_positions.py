"""Upsert short_selling_positions into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import (
    SHORT_SELLING_POSITIONS_COLUMNS,
    SHORT_SELLING_POSITIONS_PK,
    create_table_sql,
)


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        create_table_sql("short_selling_positions", SHORT_SELLING_POSITIONS_COLUMNS, SHORT_SELLING_POSITIONS_PK)
    )
    cols = {row[1] for row in con.execute("PRAGMA table_info('short_selling_positions')").fetchall()}
    for name, dtype in SHORT_SELLING_POSITIONS_COLUMNS:
        if name not in cols:
            con.execute(f"ALTER TABLE short_selling_positions ADD COLUMN {name} {dtype}")


def upsert_short_selling_positions(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    ensure_table(con)
    con.register("tmp_short_selling_positions", df.to_arrow())
    col_list = ", ".join([c for c, _ in SHORT_SELLING_POSITIONS_COLUMNS])
    con.execute(
        f"""
        INSERT OR REPLACE INTO short_selling_positions ({col_list})
        SELECT {col_list}
        FROM tmp_short_selling_positions
        """
    )
    con.unregister("tmp_short_selling_positions")
    return df.height
