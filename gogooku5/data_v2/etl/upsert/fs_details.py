"""Upsert fs_details (fins/fs_details) into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import FS_DETAILS_COLUMNS, FS_DETAILS_PK, create_table_sql


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(create_table_sql("fs_details", FS_DETAILS_COLUMNS, FS_DETAILS_PK))
    info = con.execute("PRAGMA table_info('fs_details')").fetchall()
    existing = {row[1] for row in info}
    for name, dtype in FS_DETAILS_COLUMNS:
        if name not in existing:
            con.execute(f"ALTER TABLE fs_details ADD COLUMN {name} {dtype}")


def upsert_fs_details(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    ensure_table(con)
    # Ensure any new flattened FinancialStatement columns are present in the table.
    info = con.execute("PRAGMA table_info('fs_details')").fetchall()
    existing = {row[1] for row in info}
    for col in df.columns:
        if col not in existing:
            # Default to VARCHAR for new dynamic columns; raw values are strings from JSON.
            con.execute(f"ALTER TABLE fs_details ADD COLUMN {col} VARCHAR")
            existing.add(col)

    con.register("tmp_fs_details", df.to_arrow())
    # Insert all columns that are common between the temp table and fs_details.
    cols = [c for c in df.columns if c in existing]
    col_list = ", ".join(cols)
    con.execute(
        f"""
        INSERT OR REPLACE INTO fs_details ({col_list})
        SELECT {col_list} FROM tmp_fs_details
        """
    )
    con.unregister("tmp_fs_details")
    return df.height
