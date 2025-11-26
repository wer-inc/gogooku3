"""DuckDB helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb

from .schemas import (
    BREAKDOWN_COLUMNS,
    BREAKDOWN_PK,
    DAILY_QUOTES_COLUMNS,
    DAILY_QUOTES_PK,
    FEATURES_DAILY_COLUMNS,
    FEATURES_DAILY_PK,
    FINANCIAL_FEATURES_COLUMNS,
    FINANCIAL_FEATURES_PK,
    FINANCIAL_FEATURES_TABLE,
    FS_DETAILS_COLUMNS,
    FS_DETAILS_PK,
    FS_DETAILS_TABLE,
    LISTED_INFO_COLUMNS,
    LISTED_INFO_PK,
    LISTED_META_FEATURES_COLUMNS,
    LISTED_META_FEATURES_PK,
    LISTED_META_FEATURES_TABLE,
    PRICE_FLOW_FEATURES_TABLE,
    SECTION_FLOW_FEATURES_COLUMNS,
    SECTION_FLOW_FEATURES_PK,
    SECTION_FLOW_FEATURES_TABLE,
    SHORT_SELLING_COLUMNS,
    SHORT_SELLING_PK,
    SHORT_SELLING_POSITIONS_COLUMNS,
    SHORT_SELLING_POSITIONS_PK,
    STATEMENTS_COLUMNS,
    STATEMENTS_PK,
    STATEMENTS_TABLE,
    TRADES_SPEC_COLUMNS,
    TRADES_SPEC_PK,
    TRADING_CALENDAR_COLUMNS,
    TRADING_CALENDAR_PK,
    WEEKLY_MARGIN_INTEREST_COLUMNS,
    WEEKLY_MARGIN_INTEREST_PK,
    YF_PRICE_COLUMNS,
    YF_PRICE_PK,
    create_table_sql,
)


def connect_db(db_path: Path, threads: int | None = None) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=str(db_path))
    if threads is not None and threads > 0:
        con.execute(f"PRAGMA threads={threads}")
    return con


def ensure_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(create_table_sql("listed_info", LISTED_INFO_COLUMNS, LISTED_INFO_PK))
    con.execute(create_table_sql("trading_calendar", TRADING_CALENDAR_COLUMNS, TRADING_CALENDAR_PK))
    con.execute(create_table_sql("yf_prices", YF_PRICE_COLUMNS, YF_PRICE_PK))
    con.execute(create_table_sql("daily_quotes", DAILY_QUOTES_COLUMNS, DAILY_QUOTES_PK))
    con.execute(create_table_sql("breakdown", BREAKDOWN_COLUMNS, BREAKDOWN_PK))
    con.execute(create_table_sql("weekly_margin_interest", WEEKLY_MARGIN_INTEREST_COLUMNS, WEEKLY_MARGIN_INTEREST_PK))
    con.execute(create_table_sql("short_selling", SHORT_SELLING_COLUMNS, SHORT_SELLING_PK))
    con.execute(
        create_table_sql("short_selling_positions", SHORT_SELLING_POSITIONS_COLUMNS, SHORT_SELLING_POSITIONS_PK)
    )
    con.execute(create_table_sql(PRICE_FLOW_FEATURES_TABLE, FEATURES_DAILY_COLUMNS, FEATURES_DAILY_PK))
    con.execute(create_table_sql(LISTED_META_FEATURES_TABLE, LISTED_META_FEATURES_COLUMNS, LISTED_META_FEATURES_PK))
    con.execute(create_table_sql(FINANCIAL_FEATURES_TABLE, FINANCIAL_FEATURES_COLUMNS, FINANCIAL_FEATURES_PK))
    con.execute(create_table_sql(SECTION_FLOW_FEATURES_TABLE, SECTION_FLOW_FEATURES_COLUMNS, SECTION_FLOW_FEATURES_PK))
    con.execute(create_table_sql(STATEMENTS_TABLE, STATEMENTS_COLUMNS, STATEMENTS_PK))
    con.execute(create_table_sql(FS_DETAILS_TABLE, FS_DETAILS_COLUMNS, FS_DETAILS_PK))
    con.execute(create_table_sql("trades_spec", TRADES_SPEC_COLUMNS, TRADES_SPEC_PK))
    _migrate_listed_info_schema(con)
    _migrate_price_flow_schema(con)
    _migrate_financial_schema(con)
    _migrate_section_flow_schema(con)
    ensure_views(con)


def _migrate_listed_info_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Add missing columns to listed_info for backward compatibility."""

    columns = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'listed_info'"
        ).fetchall()
    }

    additions = {
        "available_ts": "TIMESTAMP WITH TIME ZONE",
        "sector17_name": "VARCHAR",
        "sector33_name": "VARCHAR",
        "scale_category_name": "VARCHAR",
        "market_code_name": "VARCHAR",
        "margin_code_name": "VARCHAR",
    }
    for name, dtype in additions.items():
        if name not in columns:
            con.execute(f"ALTER TABLE listed_info ADD COLUMN {name} {dtype}")


def _migrate_price_flow_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Add missing columns to price_flow_features for backward compatibility."""

    columns = {
        row[0]
        for row in con.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{PRICE_FLOW_FEATURES_TABLE}'"
        ).fetchall()
    }
    for name, dtype in FEATURES_DAILY_COLUMNS:
        if name not in columns:
            con.execute(f"ALTER TABLE {PRICE_FLOW_FEATURES_TABLE} ADD COLUMN {name} {dtype}")


def _migrate_financial_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Add missing columns to financial_features for backward compatibility."""

    columns = {
        row[0]
        for row in con.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{FINANCIAL_FEATURES_TABLE}'"
        ).fetchall()
    }
    for name, dtype in FINANCIAL_FEATURES_COLUMNS:
        if name not in columns:
            con.execute(f"ALTER TABLE {FINANCIAL_FEATURES_TABLE} ADD COLUMN {name} {dtype}")


def _migrate_section_flow_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Add missing columns to section_flow_features for backward compatibility."""

    columns = {
        row[0]
        for row in con.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{SECTION_FLOW_FEATURES_TABLE}'"
        ).fetchall()
    }
    for name, dtype in SECTION_FLOW_FEATURES_COLUMNS:
        if name not in columns:
            con.execute(f"ALTER TABLE {SECTION_FLOW_FEATURES_TABLE} ADD COLUMN {name} {dtype}")


def ensure_views(con: duckdb.DuckDBPyConnection) -> None:
    """Create ML-facing view ml_dataset as price_flow + listed_meta + financial + section_flow join."""
    # Build SELECT list that includes:
    # - All price/flow/session/listed-meta core features from price_flow_features
    #   (FEATURES_DAILY_COLUMNS)
    # - All financial features (fin_*/fund_*) from financial_features that are not
    #   already present in FEATURES_DAILY_COLUMNS.
    # Sourcing rules:
    # - date/code always from price_flow_features (pf)
    # - listed meta columns from listed_meta_features (lm) when present
    # - financial columns from financial_features (ff)
    base_cols = [name for name, _ in FEATURES_DAILY_COLUMNS]
    fin_only_cols = [
        name for name, _ in FINANCIAL_FEATURES_COLUMNS if name not in ("date", "code") and name not in base_cols
    ]
    section_flow_cols = [
        name
        for name, _ in SECTION_FLOW_FEATURES_COLUMNS
        if name not in ("published_date", "start_date", "end_date", "section")
    ]
    ml_cols = base_cols + fin_only_cols + section_flow_cols

    cols_expr: list[str] = []
    listed_meta_cols = {c for c, _ in LISTED_META_FEATURES_COLUMNS if c not in ("date", "code")}
    financial_cols = {c for c, _ in FINANCIAL_FEATURES_COLUMNS if c not in ("date", "code")}
    section_flow_set = {c for c in section_flow_cols}
    for name in ml_cols:
        if name in ("date", "code"):
            cols_expr.append(f"pf.{name} AS {name}")
        elif name in listed_meta_cols:
            cols_expr.append(f"lm.{name} AS {name}")
        elif name in financial_cols:
            cols_expr.append(f"ff.{name} AS {name}")
        elif name in section_flow_set:
            cols_expr.append(f"sf.{name} AS {name}")
        else:
            cols_expr.append(f"pf.{name} AS {name}")
    select_clause = ",\n        ".join(cols_expr)

    section_expr = """
        CASE
            WHEN lm.market_code = '0111' THEN 'TSEPrime'
            WHEN lm.market_code = '0112' THEN 'TSEStandard'
            WHEN lm.market_code = '0113' THEN 'TSEGrowth'
            WHEN lm.market_code = '0101' THEN 'TSE1st'
            WHEN lm.market_code = '0102' THEN 'TSE2nd'
            WHEN lm.market_code IN ('0106','0107') THEN 'TSEJASDAQ'
            WHEN lm.market_code = '0104' THEN 'TSEMothers'
            ELSE 'TokyoNagoya'
        END
    """

    con.execute(
        f"""
        CREATE OR REPLACE VIEW ml_dataset AS
        SELECT
        {select_clause}
        FROM {PRICE_FLOW_FEATURES_TABLE} AS pf
        LEFT JOIN {LISTED_META_FEATURES_TABLE} AS lm
          ON pf.date = lm.date
         AND pf.code = lm.code
        LEFT JOIN {FINANCIAL_FEATURES_TABLE} AS ff
          ON pf.date = ff.date
         AND pf.code = ff.code
        LEFT JOIN LATERAL (
            SELECT *
            FROM {SECTION_FLOW_FEATURES_TABLE} AS sf
            WHERE sf.section = {section_expr}
              AND sf.effective_date <= pf.date
            ORDER BY sf.effective_date DESC
            LIMIT 1
        ) AS sf ON TRUE
        """
    )
