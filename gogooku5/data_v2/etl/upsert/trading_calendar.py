"""Upsert trading_calendar into DuckDB."""

from __future__ import annotations

import duckdb
import polars as pl

from ..schemas import TRADING_CALENDAR_COLUMNS, TRADING_CALENDAR_PK, create_table_sql


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(create_table_sql("trading_calendar", TRADING_CALENDAR_COLUMNS, TRADING_CALENDAR_PK))


def upsert_trading_calendar(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    ensure_table(con)
    con.register("tmp_cal", df.to_arrow())
    con.execute(
        """
        INSERT OR REPLACE INTO trading_calendar
        SELECT date, holiday_division FROM tmp_cal
        """
    )
    con.unregister("tmp_cal")
    return df.height


def trading_days_from_duckdb(
    con: duckdb.DuckDBPyConnection,
    *,
    start: str,
    end: str,
    include_divs: list[str],
) -> list[str]:
    if not include_divs:
        return []
    placeholders = ", ".join(["?"] * len(include_divs))
    sql = (
        f"SELECT date FROM trading_calendar "
        f"WHERE date BETWEEN ? AND ? AND holiday_division IN ({placeholders}) ORDER BY date"
    )
    rows = con.execute(sql, [start, end, *include_divs]).fetchall()
    return [r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]) for r in rows]
