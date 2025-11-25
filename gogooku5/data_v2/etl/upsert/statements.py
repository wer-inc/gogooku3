"""Upsert statements (fins/statements) into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import STATEMENTS_COLUMNS, STATEMENTS_PK, create_table_sql


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(create_table_sql("statements", STATEMENTS_COLUMNS, STATEMENTS_PK))
    info = con.execute("PRAGMA table_info('statements')").fetchall()
    existing = {row[1] for row in info}
    for name, dtype in STATEMENTS_COLUMNS:
        if name not in existing:
            con.execute(f"ALTER TABLE statements ADD COLUMN {name} {dtype}")


def upsert_statements(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    ensure_table(con)
    con.register("tmp_statements", df.to_arrow())
    cols = [name for name, _ in STATEMENTS_COLUMNS]
    col_list = ", ".join(cols)
    con.execute(
        f"""
        INSERT OR REPLACE INTO statements ({col_list})
        SELECT {col_list} FROM tmp_statements
        """
    )
    con.unregister("tmp_statements")
    return df.height
