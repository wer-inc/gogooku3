"""Upsert section-level flow features into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import (
    SECTION_FLOW_FEATURES_COLUMNS,
    SECTION_FLOW_FEATURES_PK,
    create_table_sql,
)


def upsert_section_flow_features(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    """Upsert section_flow_features keyed by (published_date, section)."""
    con.execute(create_table_sql("section_flow_features", SECTION_FLOW_FEATURES_COLUMNS, SECTION_FLOW_FEATURES_PK))
    if df.is_empty():
        return 0
    con.register("tmp_section_flow_features", df.to_arrow())
    cols = [name for name, _ in SECTION_FLOW_FEATURES_COLUMNS]
    col_list = ", ".join(cols)
    con.execute(
        f"""
        INSERT OR REPLACE INTO section_flow_features ({col_list})
        SELECT {col_list}
        FROM tmp_section_flow_features
        """
    )
    con.unregister("tmp_section_flow_features")
    return df.height
