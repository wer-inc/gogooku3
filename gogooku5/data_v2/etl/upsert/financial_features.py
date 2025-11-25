"""Upsert financial statement-derived features into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import (
    FINANCIAL_FEATURES_COLUMNS,
    FINANCIAL_FEATURES_PK,
    FINANCIAL_FEATURES_TABLE,
    create_table_sql,
)


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(create_table_sql(FINANCIAL_FEATURES_TABLE, FINANCIAL_FEATURES_COLUMNS, FINANCIAL_FEATURES_PK))
    info = con.execute(f"PRAGMA table_info('{FINANCIAL_FEATURES_TABLE}')").fetchall()
    existing = {row[1] for row in info}
    for name, dtype in FINANCIAL_FEATURES_COLUMNS:
        if name not in existing:
            con.execute(f"ALTER TABLE {FINANCIAL_FEATURES_TABLE} ADD COLUMN {name} {dtype}")


def upsert_financial_features(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    """Upsert daily financial features keyed by (date, code)."""
    if df.is_empty():
        return 0
    ensure_table(con)
    con.register("tmp_financial_features", df.to_arrow())
    cols = [name for name, _ in FINANCIAL_FEATURES_COLUMNS]
    col_list = ", ".join(cols)
    con.execute(
        f"""
        INSERT OR REPLACE INTO {FINANCIAL_FEATURES_TABLE} (
            {col_list}
        )
        SELECT
            {col_list}
        FROM tmp_financial_features
        """
    )
    con.unregister("tmp_financial_features")
    return df.height
