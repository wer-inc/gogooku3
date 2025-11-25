"""Upsert markets trades_spec into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import TRADES_SPEC_COLUMNS, TRADES_SPEC_PK, create_table_sql


def upsert_trades_spec(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    """Upsert trades_spec rows keyed by (published_date, section, start_date, end_date)."""

    con.execute(create_table_sql("trades_spec", TRADES_SPEC_COLUMNS, TRADES_SPEC_PK))
    if df.is_empty():
        return 0
    con.register("tmp_trades_spec", df.to_arrow())
    cols = [name for name, _ in TRADES_SPEC_COLUMNS]
    col_list = ", ".join(cols)
    con.execute(
        f"""
        INSERT OR REPLACE INTO trades_spec ({col_list})
        SELECT {col_list}
        FROM tmp_trades_spec
        """
    )
    con.unregister("tmp_trades_spec")
    return df.height
