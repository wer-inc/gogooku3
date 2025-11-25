"""Upsert computed features into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import (
    FEATURES_DAILY_COLUMNS,
    FEATURES_DAILY_PK,
    PRICE_FLOW_FEATURES_TABLE,
    create_table_sql,
)
from .listed_features import LISTED_FEATURE_COLUMNS


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(create_table_sql(PRICE_FLOW_FEATURES_TABLE, FEATURES_DAILY_COLUMNS, FEATURES_DAILY_PK))
    info = con.execute(f"PRAGMA table_info('{PRICE_FLOW_FEATURES_TABLE}')").fetchall()
    existing = {row[1] for row in info}
    for name, dtype in FEATURES_DAILY_COLUMNS:
        if name not in existing:
            con.execute(f"ALTER TABLE {PRICE_FLOW_FEATURES_TABLE} ADD COLUMN {name} {dtype}")


def upsert_features_daily(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    ensure_table(con)
    con.register("tmp_features_daily", df.to_arrow())
    # MERGE to avoid clobbering listed_info-derived columns.
    all_cols = [name for name, _ in FEATURES_DAILY_COLUMNS]
    listed_set = set(LISTED_FEATURE_COLUMNS)
    # Columns to update from price/TA/breakdown step (exclude PK and listed/meta columns)
    update_cols = [c for c in all_cols if c not in ("date", "code") and c not in listed_set]
    set_clause = ", ".join(f"{c} = tmp.{c}" for c in update_cols)
    insert_cols = ", ".join(all_cols)
    insert_values = ", ".join(f"tmp.{c}" for c in all_cols)

    con.execute(
        f"""
        MERGE INTO {PRICE_FLOW_FEATURES_TABLE} AS t
        USING tmp_features_daily AS tmp
        ON t.date = tmp.date AND t.code = tmp.code
        WHEN MATCHED THEN UPDATE SET
            {set_clause}
        WHEN NOT MATCHED THEN
            INSERT ({insert_cols})
            VALUES ({insert_values})
        """
    )
    con.unregister("tmp_features_daily")
    return df.height
