#!/usr/bin/env python3
"""
DuckDB-first ingestion utilities for J-Quants data (listed_info, trading_calendar).

This script keeps DuckDB as the system of record:
- Init tables once (PRIMARY KEY on natural keys for upserts)
- Ingest historic Parquet into DuckDB
- Fetch from J-Quants and upsert directly into DuckDB
- Export DuckDB tables back to Parquet when needed
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from data_v2.etl.auth import get_id_token
from data_v2.etl.client import (
    SECTOR33_CODES,
    fetch_breakdown_for_date,
    fetch_calendar_records,
    fetch_daily_margin_interest,
    fetch_daily_quotes_for_date,
    fetch_fs_details_for_date,
    fetch_listed_info_for_date,
    fetch_short_selling,
    fetch_short_selling_positions,
    fetch_statements_for_date,
    fetch_trades_spec,
    fetch_weekly_margin_interest,
)
from data_v2.etl.config import DEFAULT_DB_PATH
from data_v2.etl.duckdb import connect_db, ensure_tables
from data_v2.etl.feature.financial import compute_financial_features
from data_v2.etl.feature.listed_info import compute_listed_features
from data_v2.etl.feature.prices import compute_price_features
from data_v2.etl.feature.trades_spec import compute_trades_spec_features
from data_v2.etl.normalize.breakdown import normalize_breakdown
from data_v2.etl.normalize.daily_margin_interest import normalize_daily_margin_interest
from data_v2.etl.normalize.daily_quotes import normalize_daily_quotes
from data_v2.etl.normalize.fs_details import normalize_fs_details
from data_v2.etl.normalize.listed_info import normalize_listed_info
from data_v2.etl.normalize.short_selling import normalize_short_selling
from data_v2.etl.normalize.short_selling_positions import (
    normalize_short_selling_positions,
)
from data_v2.etl.normalize.statements import normalize_statements
from data_v2.etl.normalize.trades_spec import normalize_trades_spec
from data_v2.etl.normalize.trading_calendar import normalize_trading_calendar
from data_v2.etl.normalize.weekly_margin_interest import (
    normalize_weekly_margin_interest,
)
from data_v2.etl.normalize.yfinance import normalize_yfinance_multi
from data_v2.etl.upsert.breakdown import upsert_breakdown
from data_v2.etl.upsert.daily_margin_interest import upsert_daily_margin_interest
from data_v2.etl.upsert.daily_quotes import upsert_daily_quotes
from data_v2.etl.upsert.features import upsert_features_daily
from data_v2.etl.upsert.fs_details import upsert_fs_details
from data_v2.etl.upsert.listed_features import upsert_listed_features
from data_v2.etl.upsert.listed_info import upsert_listed_info
from data_v2.etl.upsert.section_flow_features import upsert_section_flow_features
from data_v2.etl.upsert.short_selling import upsert_short_selling
from data_v2.etl.upsert.short_selling_positions import upsert_short_selling_positions
from data_v2.etl.upsert.statements import upsert_statements
from data_v2.etl.upsert.trades_spec import upsert_trades_spec
from data_v2.etl.upsert.trading_calendar import (
    trading_days_from_duckdb,
    upsert_trading_calendar,
)
from data_v2.etl.upsert.weekly_margin_interest import upsert_weekly_margin_interest
from data_v2.etl.upsert.yfinance import upsert_yfinance_prices
from data_v2.etl.yfinance_tickers import MACRO_FALLBACKS, MACRO_TICKERS

DEFAULT_CALENDAR_PATH = Path()  # deprecated; kept for CLI compatibility


def _fetch_with_retries(fn, day: str, *, retries: int = 3, delay: float = 1.0) -> tuple[str, list[dict[str, Any]]]:
    """Run a day-based fetch with simple retry/backoff."""

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            recs = fn(day)
            return day, recs
        except Exception as exc:  # pragma: no cover - network variability
            last_exc = exc
            if attempt < retries:
                time.sleep(delay)
            else:
                print(f"❌ {day}: fetch failed after {retries} attempts ({exc})")
    return day, []


def _summarize_listed_batch(df: pl.DataFrame) -> str:
    """Return a short quality summary for a listed_info batch."""

    if df.is_empty():
        return "rows=0"
    total = df.height
    null_market = df["market_code"].null_count() if "market_code" in df.columns else total
    null_sector33 = df["sector33_code"].null_count() if "sector33_code" in df.columns else total
    null_sector17 = df["sector17_code"].null_count() if "sector17_code" in df.columns else total
    null_available_ts = df["available_ts"].null_count() if "available_ts" in df.columns else total
    return (
        f"rows={total:,} null_market={null_market:,} "
        f"null_sector33={null_sector33:,} null_sector17={null_sector17:,} "
        f"null_available_ts={null_available_ts:,}"
    )


def _summarize_yf_batch(df: pl.DataFrame) -> str:
    """Return a short quality summary for yfinance batch."""

    if df.is_empty():
        return "rows=0"
    total = df.height
    null_close = df["close"].null_count() if "close" in df.columns else total
    null_volume = df["volume"].null_count() if "volume" in df.columns else total
    return f"rows={total:,} null_close={null_close:,} null_volume={null_volume:,}"


def _summarize_quotes_batch(df: pl.DataFrame) -> str:
    """Return a short quality summary for daily_quotes batch."""

    if df.is_empty():
        return "rows=0"
    total = df.height
    null_close = df["close"].null_count() if "close" in df.columns else total
    null_volume = df["volume"].null_count() if "volume" in df.columns else total
    return f"rows={total:,} null_close={null_close:,} null_volume={null_volume:,}"


def _summarize_breakdown_batch(df: pl.DataFrame) -> str:
    """Return a short quality summary for breakdown batch."""

    if df.is_empty():
        return "rows=0"
    total = df.height
    null_val = df["long_sell_value"].null_count() if "long_sell_value" in df.columns else total
    null_vol = df["long_sell_volume"].null_count() if "long_sell_volume" in df.columns else total
    return f"rows={total:,} null_value={null_val:,} null_volume={null_vol:,}"


def import_parquet_listed(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> int:
    """
    Load historic listed_info Parquet into DuckDB.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    df = pl.read_parquet(parquet_path)
    normalized = normalize_listed_info(df.to_dicts())
    summary = _summarize_listed_batch(normalized)
    print(f"[listed] import summary: {summary}")
    return upsert_listed_info(con, normalized)


def import_parquet_calendar(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> int:
    """
    Load trading_calendar Parquet into DuckDB.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    df = pl.read_parquet(parquet_path)
    return upsert_trading_calendar(con, df)


def upsert_calendar_direct(
    con: duckdb.DuckDBPyConnection,
    *,
    from_date: str,
    to_date: str,
    include_divs: Iterable[str] | None = None,
) -> int:
    """
    Fetch trading calendar via API and upsert directly into DuckDB (no Parquet).
    """
    records = fetch_calendar_records(id_token=get_id_token(), from_date=from_date, to_date=to_date)
    df = normalize_trading_calendar(records)
    if include_divs:
        df = df.filter(pl.col("holiday_division").cast(pl.Utf8).is_in(list(include_divs)))
    return upsert_trading_calendar(con, df)


def fetch_yfinance_into_duckdb(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str],
    start: str,
    end: str,
    sleep_sec: float = 0.0,
    fallbacks: dict[str, str] | None = None,
) -> None:
    """Fetch yfinance OHLCV and upsert into DuckDB."""

    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"yfinance is not installed: {exc}") from exc

    from datetime import datetime, timedelta

    def _end_inclusive(end_str: str) -> str:
        dt = datetime.strptime(end_str, "%Y-%m-%d").date() + timedelta(days=1)
        return dt.isoformat()

    end_exc = _end_inclusive(end)
    for idx, ticker in enumerate(tickers, 1):
        used_fallback = False
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end_exc,
                auto_adjust=False,
                progress=False,
                interval="1d",
            )
        except Exception as exc:
            fallback = (fallbacks or {}).get(ticker)
            if fallback:
                used_fallback = True
                try:
                    raw = yf.download(
                        fallback,
                        start=start,
                        end=end_exc,
                        auto_adjust=False,
                        progress=False,
                        interval="1d",
                    )
                except Exception as exc_fb:
                    print(
                        f"❌ {ticker}: failed to download via yfinance ({exc}); fallback {fallback} also failed ({exc_fb})"
                    )
                    continue
            else:
                print(f"❌ {ticker}: failed to download via yfinance ({exc})")
                continue

        df = normalize_yfinance_multi(ticker, raw)
        inserted = upsert_yfinance_prices(con, df)
        summary = _summarize_yf_batch(df)
        fb_note = f" (fallback {fallbacks[ticker]})" if used_fallback else ""
        print(f"✅ {ticker}: upserted {inserted} rows into DuckDB ({summary}){fb_note}")

        if sleep_sec > 0 and idx % 10 == 0:
            time.sleep(sleep_sec)


def fetch_listed_into_duckdb(
    con: duckdb.DuckDBPyConnection,
    *,
    start: str,
    end: str,
    calendar_path: Path,
    include_divs: Iterable[str],
    sleep_sec: float,
    workers: int = 10,
) -> None:
    """
    Fetch /listed/info day by day and upsert into DuckDB.
    """
    if not calendar_path.exists():
        raise FileNotFoundError(
            f"Calendar parquet not found at {calendar_path}. Provide a valid path or use duckdb_fetch_listed_op with auto_fetch_calendar."
        )

    days = load_trading_days(calendar_path, start, end, list(include_divs))
    if not days:
        print(f"No trading days between {start} and {end}")
        return

    id_token = get_id_token()

    # Parallel fetch, sequential upsert (single connection)
    def _job(day: str) -> tuple[str, list[dict[str, Any]]]:
        try:
            recs = fetch_listed_info_for_date(id_token=id_token, day=day)
            return day, recs
        except Exception:
            return day, []

    results: list[tuple[str, list[dict[str, Any]]]] = []
    max_workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_job, d): d for d in days}
        for idx, fut in enumerate(as_completed(future_map), 1):
            day = future_map[fut]
            day, records = fut.result()
            if records:
                df = normalize_listed_info(records)
                inserted = upsert_listed_info(con, df)
                summary = _summarize_listed_batch(df)
                print(f"✅ {day}: upserted {inserted} rows into DuckDB ({summary})")
            else:
                print(f"⏩ {day}: no records returned")
            if sleep_sec > 0 and idx % 10 == 0:
                time.sleep(sleep_sec)


def export_table_to_parquet(
    con: duckdb.DuckDBPyConnection,
    *,
    table: str,
    out_path: Path,
    split_by_year: bool = False,
) -> None:
    """
    Export a DuckDB table to Parquet (optionally year-split).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not split_by_year:
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM {table}
                ORDER BY 1, 2
            ) TO ?
            (FORMAT 'parquet', COMPRESSION 'zstd')
            """,
            [str(out_path)],
        )
        return

    years = con.execute(f"SELECT DISTINCT EXTRACT(year FROM date)::INT FROM {table} ORDER BY 1").fetchall()
    for (year,) in years:
        yearly_path = out_path.with_name(f"{out_path.stem}_{year}{out_path.suffix}")
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM {table}
                WHERE date BETWEEN DATE '{year}-01-01' AND DATE '{year}-12-31'
                ORDER BY 1, 2
            ) TO ?
            (FORMAT 'parquet', COMPRESSION 'zstd')
            """,
            [str(yearly_path)],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DuckDB-first loader for J-Quants data")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to DuckDB file")
    parser.add_argument("--threads", type=int, default=None, help="DuckDB threads (optional)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create DuckDB and tables if missing")

    load_parquet = sub.add_parser("load-parquet", help="Load existing Parquet into DuckDB")
    load_parquet.add_argument("--table", choices=["listed_info", "trading_calendar"], required=True)
    load_parquet.add_argument("--path", type=Path, required=True, help="Path to source Parquet")

    export = sub.add_parser("export-parquet", help="Export a DuckDB table to Parquet")
    export.add_argument("--table", choices=["listed_info", "trading_calendar"], required=True)
    export.add_argument("--out", type=Path, required=True, help="Destination Parquet path")
    export.add_argument("--split-yearly", action="store_true", help="Emit one Parquet per year")

    cal_direct = sub.add_parser("fetch-calendar-direct", help="Fetch calendar from API and upsert directly into DuckDB")
    cal_direct.add_argument("--from", dest="from_date", required=True, help="Start date (YYYY-MM-DD)")
    cal_direct.add_argument("--to", dest="to_date", required=True, help="End date (YYYY-MM-DD)")

    listed_direct = sub.add_parser(
        "fetch-listed-direct", help="Fetch listed_info via API and upsert directly into DuckDB"
    )
    listed_direct.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    listed_direct.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    listed_direct.add_argument(
        "--holiday-division",
        default="1,2",
        help="Comma-separated HolidayDivision values to treat as trading days (default: 1,2)",
    )
    # Allow env overrides for defaults
    default_workers = int(os.environ.get("WORKERS", "10"))
    default_sleep = float(os.environ.get("SLEEP_SEC", "0.0"))

    listed_direct.add_argument(
        "--sleep-sec",
        type=float,
        default=default_sleep,
        help="Sleep every 10 days to avoid rate limits",
    )
    listed_direct.add_argument(
        "--auto-fetch-calendar",
        action="store_true",
        help="If trading_calendar table is empty for the range, fetch and upsert it first",
    )
    listed_direct.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Parallel fetch workers for listed_info (default from WORKERS env or 10)",
    )

    yf_direct = sub.add_parser("fetch-yf-history", help="Fetch OHLCV via yfinance and upsert into DuckDB")
    yf_direct.add_argument(
        "--start",
        default=os.environ.get("START"),
        required=os.environ.get("START") is None,
        help="Start date (YYYY-MM-DD) [env: START]",
    )
    yf_direct.add_argument(
        "--end",
        default=os.environ.get("END"),
        required=os.environ.get("END") is None,
        help="End date (YYYY-MM-DD) [env: END]",
    )
    yf_direct.add_argument(
        "--sleep-sec",
        type=float,
        default=float(os.environ.get("SLEEP_SEC", "0.0")),
        help="Sleep every 10 tickers to avoid rate limits",
    )

    quotes_direct = sub.add_parser("fetch-daily-quotes", help="Fetch daily_quotes via API and upsert into DuckDB")
    quotes_direct.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    quotes_direct.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    quotes_direct.add_argument(
        "--holiday-division",
        default="1,2",
        help="Comma-separated HolidayDivision values to treat as trading days (default: 1,2)",
    )
    quotes_direct.add_argument(
        "--sleep-sec",
        type=float,
        default=float(os.environ.get("SLEEP_SEC", "0.0")),
        help="Sleep every 10 days to avoid rate limits",
    )
    quotes_direct.add_argument(
        "--auto-fetch-calendar",
        action="store_true",
        help="If trading_calendar table is empty for the range, fetch and upsert it first",
    )
    quotes_direct.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Parallel fetch workers for daily_quotes (default from WORKERS env or 10)",
    )

    breakdown_direct = sub.add_parser("fetch-breakdown", help="Fetch markets/breakdown via API and upsert into DuckDB")
    breakdown_direct.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    breakdown_direct.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    breakdown_direct.add_argument(
        "--holiday-division",
        default="1,2",
        help="Comma-separated HolidayDivision values to treat as trading days (default: 1,2)",
    )
    breakdown_direct.add_argument(
        "--sleep-sec",
        type=float,
        default=float(os.environ.get("SLEEP_SEC", "0.0")),
        help="Sleep every 10 days to avoid rate limits",
    )
    breakdown_direct.add_argument(
        "--auto-fetch-calendar",
        action="store_true",
        help="If trading_calendar table is empty for the range, fetch and upsert it first",
    )
    breakdown_direct.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Parallel fetch workers for breakdown (default from WORKERS env or 10)",
    )

    trades_spec_direct = sub.add_parser(
        "fetch-trades-spec", help="Fetch markets/trades_spec via API and upsert into DuckDB"
    )
    trades_spec_direct.add_argument("--start", required=False, help="Start date (YYYY-MM-DD) for PublishedDate/from")
    trades_spec_direct.add_argument("--end", required=False, help="End date (YYYY-MM-DD) for PublishedDate/to")
    trades_spec_direct.add_argument("--section", required=False, help="Section filter (e.g., TSEPrime)")

    statements_direct = sub.add_parser(
        "fetch-statements",
        help="Fetch /fins/statements by date range and upsert into DuckDB",
    )
    statements_direct.add_argument(
        "--start",
        default=os.environ.get("START"),
        required=False,
        help="Start date (YYYY-MM-DD) [env: START]",
    )
    statements_direct.add_argument(
        "--end",
        default=os.environ.get("END"),
        required=False,
        help="End date (YYYY-MM-DD) [env: END]",
    )
    statements_direct.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="DuckDB path",
    )
    statements_direct.add_argument(
        "--threads",
        type=int,
        default=0,
        help="DuckDB threads (0=default)",
    )
    statements_direct.add_argument(
        "--holiday-division",
        default="1,2",
        help="HolidayDivision CSV (e.g., '1,2')",
    )
    statements_direct.add_argument(
        "--sleep-sec",
        type=float,
        default=float(os.environ.get("SLEEP_SEC", "0.0")),
        help="Sleep seconds every 10 days (rate limit guard)",
    )
    statements_direct.add_argument(
        "--auto-fetch-calendar",
        action="store_true",
        help="If trading_calendar is empty for the range, fetch and upsert it first",
    )
    statements_direct.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Parallel fetch workers for statements (default from WORKERS env or 10)",
    )

    fs_direct = sub.add_parser(
        "fetch-fs-details",
        help="Fetch /fins/fs_details by date range and upsert into DuckDB",
    )
    fs_direct.add_argument(
        "--start",
        default=os.environ.get("START"),
        required=False,
        help="Start date (YYYY-MM-DD) [env: START]",
    )
    fs_direct.add_argument(
        "--end",
        default=os.environ.get("END"),
        required=False,
        help="End date (YYYY-MM-DD) [env: END]",
    )
    fs_direct.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="DuckDB path",
    )
    fs_direct.add_argument(
        "--threads",
        type=int,
        default=0,
        help="DuckDB threads (0=default)",
    )
    fs_direct.add_argument(
        "--holiday-division",
        default="1,2",
        help="HolidayDivision CSV (e.g., '1,2')",
    )
    fs_direct.add_argument(
        "--sleep-sec",
        type=float,
        default=float(os.environ.get("SLEEP_SEC", "0.0")),
        help="Sleep seconds every 10 days (rate limit guard)",
    )
    fs_direct.add_argument(
        "--auto-fetch-calendar",
        action="store_true",
        help="If trading_calendar is empty for the range, fetch and upsert it first",
    )
    fs_direct.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Parallel fetch workers for fs_details (default from WORKERS env or 10)",
    )

    feat_direct = sub.add_parser(
        "build-price-flow-features",
        help="Compute price/flow features and upsert into price_flow_features",
    )
    feat_direct.add_argument(
        "--start",
        default=os.environ.get("START"),
        required=False,
        help="Start date (YYYY-MM-DD) [env: START]",
    )
    feat_direct.add_argument(
        "--end",
        default=os.environ.get("END"),
        required=False,
        help="End date (YYYY-MM-DD) [env: END]",
    )
    feat_direct.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="DuckDB path",
    )
    feat_direct.add_argument(
        "--threads",
        type=int,
        default=0,
        help="DuckDB threads (0=default)",
    )
    feat_direct.add_argument(
        "--warmup-days",
        type=int,
        default=int(os.environ.get("WARMUP_DAYS", "260")),
        help="Calendar days to look back when computing features (for rolling/talib warmup)",
    )

    listed_feat = sub.add_parser(
        "build-listed-meta-features",
        help="Compute sector/market/scale features from listed_info (runs AFTER build-price-flow-features)",
    )
    listed_feat.add_argument(
        "--start",
        default=os.environ.get("START"),
        required=False,
        help="Start date (YYYY-MM-DD) [env: START]",
    )
    listed_feat.add_argument(
        "--end",
        default=os.environ.get("END"),
        required=False,
        help="End date (YYYY-MM-DD) [env: END]",
    )

    fin_feat = sub.add_parser(
        "build-financial-features",
        help="Compute financial statement-based features from statements (runs AFTER build-price-flow-features)",
    )
    fin_feat.add_argument(
        "--start",
        default=os.environ.get("START"),
        required=False,
        help="Start date (YYYY-MM-DD) [env: START]",
    )
    fin_feat.add_argument(
        "--end",
        default=os.environ.get("END"),
        required=False,
        help="End date (YYYY-MM-DD) [env: END]",
    )

    trades_feat = sub.add_parser(
        "build-trades-spec-features",
        help="Compute section-level flow features from trades_spec and upsert into section_flow_features",
    )
    trades_feat.add_argument(
        "--start",
        default=os.environ.get("START"),
        required=False,
        help="Start published_date (YYYY-MM-DD) [env: START]",
    )
    trades_feat.add_argument(
        "--end",
        default=os.environ.get("END"),
        required=False,
        help="End published_date (YYYY-MM-DD) [env: END]",
    )

    wmi_fetch = sub.add_parser(
        "fetch-weekly-margin-interest",
        help="Fetch /markets/weekly_margin_interest by date range and upsert into DuckDB",
    )
    wmi_fetch.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    wmi_fetch.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    wmi_fetch.add_argument("--code", required=False, help="Optional code filter (uses from/to when set)")
    wmi_fetch.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "1")),
        help="Parallel workers for date-based fetch (code=None only)",
    )
    wmi_fetch.add_argument(
        "--holiday-division",
        default=os.environ.get("HOLIDAY_DIVISION", "1,2"),
        help="Comma-separated HolidayDivision values to treat as trading days (default: 1,2)",
    )

    dmi_fetch = sub.add_parser(
        "fetch-daily-margin-interest",
        help="Fetch /markets/daily_margin_interest by date range and upsert into DuckDB",
    )
    dmi_fetch.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    dmi_fetch.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    dmi_fetch.add_argument("--code", required=False, help="Optional code filter (uses from/to when set)")
    dmi_fetch.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "1")),
        help="Parallel workers for date-based fetch (code=None only)",
    )
    dmi_fetch.add_argument(
        "--holiday-division",
        default=os.environ.get("HOLIDAY_DIVISION", "1,2"),
        help="Comma-separated HolidayDivision values to treat as trading days (default: 1,2)",
    )

    ss_fetch = sub.add_parser(
        "fetch-short-selling",
        help="Fetch /markets/short_selling by date range and upsert into DuckDB",
    )
    ss_fetch.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ss_fetch.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    ss_fetch.add_argument("--sector33code", required=False, help="Optional sector33code filter (e.g., 0050)")
    ss_fetch.add_argument(
        "--holiday-division",
        default=os.environ.get("HOLIDAY_DIVISION", "1,2"),
        help="Comma-separated HolidayDivision values to treat as trading days (default: 1,2)",
    )
    ss_fetch.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "1")),
        help="Parallel workers for sector-based fetch (only used when sector33code is not set)",
    )

    ssp_fetch = sub.add_parser(
        "fetch-short-selling-positions",
        help="Fetch /markets/short_selling_positions by disclosed_date range and upsert into DuckDB",
    )
    ssp_fetch.add_argument("--start", required=True, help="disclosed_date start (YYYY-MM-DD)")
    ssp_fetch.add_argument("--end", required=True, help="disclosed_date end (YYYY-MM-DD)")
    ssp_fetch.add_argument("--code", required=False, help="Optional code filter")
    ssp_fetch.add_argument(
        "--holiday-division",
        default=os.environ.get("HOLIDAY_DIVISION", "1,2"),
        help="Comma-separated HolidayDivision values to treat as trading days (default: 1,2)",
    )
    ssp_fetch.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "1")),
        help="Parallel workers for date-based fetch (code=None only)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    con = connect_db(args.db, threads=args.threads)
    ensure_tables(con)

    if args.command == "init":
        print(f"Initialized DuckDB at {args.db}")
        return 0

    if args.command == "load-parquet":
        if args.table == "listed_info":
            count = import_parquet_listed(con, args.path)
        else:
            count = import_parquet_calendar(con, args.path)
        print(f"Loaded {count} rows from {args.path} into {args.table}")
        return 0

    if args.command == "export-parquet":
        export_table_to_parquet(
            con,
            table=args.table,
            out_path=args.out,
            split_by_year=args.split_yearly,
        )
        print(f"Exported {args.table} to {args.out}")
        return 0

    if args.command == "fetch-calendar-direct":
        inserted = upsert_calendar_direct(
            con,
            from_date=args.from_date,
            to_date=args.to_date,
            include_divs=["0", "1", "2", "3"],
        )
        print(f"Upserted {inserted} calendar rows into {args.db}")
        return 0

    if args.command == "fetch-listed-direct":
        include_divs = [v.strip() for v in args.holiday_division.split(",") if v.strip()]
        days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)
        if not days and args.auto_fetch_calendar:
            inserted = upsert_calendar_direct(
                con,
                from_date=args.start,
                to_date=args.end,
                include_divs=include_divs or None,
            )
            print(f"[calendar] Upserted {inserted} rows into {args.db} for {args.start}->{args.end}")
            days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)

        if not days:
            raise SystemExit(
                f"No trading days in trading_calendar for {args.start}->{args.end}. "
                f"Run fetch-calendar-direct or pass --auto-fetch-calendar."
            )

        id_token = get_id_token()
        max_workers = max(1, args.workers)

        def _job(day: str) -> tuple[str, list[dict[str, Any]]]:
            return _fetch_with_retries(lambda d: fetch_listed_info_for_date(id_token=id_token, day=d), day)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_job, d): d for d in days}
            for idx, fut in enumerate(as_completed(futures), 1):
                day = futures[fut]
                day, records = fut.result()
                if records:
                    df = normalize_listed_info(records)
                    inserted = upsert_listed_info(con, df)
                    summary = _summarize_listed_batch(df)
                    print(f"✅ {day}: upserted {inserted} rows into DuckDB ({summary})")
                else:
                    print(f"⏩ {day}: no records returned")
                if args.sleep_sec > 0 and idx % 10 == 0:
                    time.sleep(args.sleep_sec)
        print(f"Upserted listed_info into {args.db} for {args.start}->{args.end}")
        return 0

    if args.command == "fetch-yf-history":
        tickers = MACRO_TICKERS
        fb = MACRO_FALLBACKS
        fetch_yfinance_into_duckdb(
            con,
            tickers=tickers,
            start=args.start,
            end=args.end,
            sleep_sec=args.sleep_sec,
            fallbacks=fb,
        )
        print(f"Upserted yfinance history into {args.db} for tickers={','.join(tickers)} {args.start}->{args.end}")
        return 0

    if args.command == "fetch-daily-quotes":
        include_divs = [v.strip() for v in args.holiday_division.split(",") if v.strip()]
        days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)
        if not days and args.auto_fetch_calendar:
            inserted = upsert_calendar_direct(
                con,
                from_date=args.start,
                to_date=args.end,
                include_divs=include_divs or None,
            )
            print(f"[calendar] Upserted {inserted} rows into {args.db} for {args.start}->{args.end}")
            days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)

        if not days:
            raise SystemExit(
                f"No trading days in trading_calendar for {args.start}->{args.end}. "
                f"Run fetch-calendar-direct or pass --auto-fetch-calendar."
            )

        id_token = get_id_token()
        max_workers = max(1, args.workers)

        def _job(day: str) -> tuple[str, list[dict[str, Any]]]:
            return _fetch_with_retries(lambda d: fetch_daily_quotes_for_date(id_token=id_token, day=d), day)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_job, d): d for d in days}
            for idx, fut in enumerate(as_completed(futures), 1):
                day = futures[fut]
                day, records = fut.result()
                if records:
                    df = normalize_daily_quotes(records)
                    inserted = upsert_daily_quotes(con, df)
                    summary = _summarize_quotes_batch(df)
                    print(f"✅ {day}: upserted {inserted} rows into DuckDB ({summary})")
                else:
                    print(f"⏩ {day}: no records returned")
                if args.sleep_sec > 0 and idx % 10 == 0:
                    time.sleep(args.sleep_sec)
        print(f"Upserted daily_quotes into {args.db} for {args.start}->{args.end}")
        return 0

    if args.command == "fetch-breakdown":
        include_divs = [v.strip() for v in args.holiday_division.split(",") if v.strip()]
        days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)
        if not days and args.auto_fetch_calendar:
            inserted = upsert_calendar_direct(
                con,
                from_date=args.start,
                to_date=args.end,
                include_divs=include_divs or None,
            )
            print(f"[calendar] Upserted {inserted} rows into {args.db} for {args.start}->{args.end}")
            days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)

        if not days:
            raise SystemExit(
                f"No trading days in trading_calendar for {args.start}->{args.end}. "
                f"Run fetch-calendar-direct or pass --auto-fetch-calendar."
            )

        id_token = get_id_token()
        max_workers = max(1, args.workers)

        def _job(day: str) -> tuple[str, list[dict[str, Any]]]:
            return _fetch_with_retries(lambda d: fetch_breakdown_for_date(id_token=id_token, day=d), day)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_job, d): d for d in days}
            for idx, fut in enumerate(as_completed(futures), 1):
                day = futures[fut]
                day, records = fut.result()
                if records:
                    df = normalize_breakdown(records)
                    inserted = upsert_breakdown(con, df)
                    summary = _summarize_breakdown_batch(df)
                    print(f"✅ {day}: upserted {inserted} rows into DuckDB ({summary})")
                else:
                    print(f"⏩ {day}: no records returned")
                if args.sleep_sec > 0 and idx % 10 == 0:
                    time.sleep(args.sleep_sec)
        print(f"Upserted breakdown into {args.db} for {args.start}->{args.end}")
        return 0

    if args.command == "fetch-trades-spec":
        id_token = get_id_token()
        records = fetch_trades_spec(
            id_token=id_token,
            start=args.start,
            end=args.end,
            section=args.section,
        )
        df = normalize_trades_spec(records)
        inserted = upsert_trades_spec(con, df)
        print(
            f"Upserted trades_spec into {args.db} for section={args.section or 'ALL'} "
            f"range {args.start or 'ALL'}->{args.end or 'ALL'} rows={inserted}"
        )
        return 0

    if args.command == "fetch-statements":
        include_divs = [v.strip() for v in args.holiday_division.split(",") if v.strip()]
        days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)
        if not days and args.auto_fetch_calendar:
            inserted = upsert_calendar_direct(
                con,
                from_date=args.start,
                to_date=args.end,
                include_divs=include_divs or None,
            )
            print(f"[calendar] Upserted {inserted} rows into {args.db} for {args.start}->{args.end}")
            days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)

        if not days:
            raise SystemExit(
                f"No trading days in trading_calendar for {args.start}->{args.end}. "
                f"Run fetch-calendar-direct or pass --auto-fetch-calendar."
            )

        id_token = get_id_token()
        max_workers = max(1, args.workers)

        def _job(day: str) -> tuple[str, list[dict[str, Any]]]:
            return _fetch_with_retries(lambda d: fetch_statements_for_date(id_token=id_token, day=d), day)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_job, d): d for d in days}
            for idx, fut in enumerate(as_completed(futures), 1):
                day = futures[fut]
                day, records = fut.result()
                if records:
                    df = normalize_statements(records)
                    inserted = upsert_statements(con, df)
                    print(f"✅ {day}: upserted {inserted} statements rows into DuckDB")
                else:
                    print(f"⏩ {day}: no statements returned")
                if args.sleep_sec > 0 and idx % 10 == 0:
                    time.sleep(args.sleep_sec)
        print(f"Upserted statements into {args.db} for {args.start}->{args.end}")
        return 0

    if args.command == "fetch-fs-details":
        include_divs = [v.strip() for v in args.holiday_division.split(",") if v.strip()]
        days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)
        if not days and args.auto_fetch_calendar:
            inserted = upsert_calendar_direct(
                con,
                from_date=args.start,
                to_date=args.end,
                include_divs=include_divs or None,
            )
            print(f"[calendar] Upserted {inserted} rows into {args.db} for {args.start}->{args.end}")
            days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)

        if not days:
            raise SystemExit(
                f"No trading days in trading_calendar for {args.start}->{args.end}. "
                f"Run fetch-calendar-direct or pass --auto-fetch-calendar."
            )

        id_token = get_id_token()
        max_workers = max(1, args.workers)

        def _job(day: str) -> tuple[str, list[dict[str, Any]]]:
            return _fetch_with_retries(lambda d: fetch_fs_details_for_date(id_token=id_token, day=d), day)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_job, d): d for d in days}
            for idx, fut in enumerate(as_completed(futures), 1):
                day = futures[fut]
                day, records = fut.result()
                if records:
                    df = normalize_fs_details(records)
                    inserted = upsert_fs_details(con, df)
                    print(f"✅ {day}: upserted {inserted} fs_details rows into DuckDB")
                else:
                    print(f"⏩ {day}: no fs_details returned")
                if args.sleep_sec > 0 and idx % 10 == 0:
                    time.sleep(args.sleep_sec)
        print(f"Upserted fs_details into {args.db} for {args.start}->{args.end}")
        return 0

    if args.command == "build-price-flow-features":
        if not args.start or not args.end:
            raise SystemExit("build-price-flow-features: specify --start/--end or set START/END in env/.env")
        features_df = compute_price_features(
            con,
            start=args.start,
            end=args.end,
            warmup_days=args.warmup_days,
        )
        inserted = upsert_features_daily(con, features_df)
        print(f"Upserted price_flow_features: {inserted} rows into {args.db} for {args.start}->{args.end}")
        return 0

    if args.command == "build-listed-meta-features":
        if not args.start or not args.end:
            raise SystemExit("build-listed-meta-features: specify --start/--end or set START/END in env/.env")
        listed_df = compute_listed_features(
            con,
            start=args.start,
            end=args.end,
        )
        updated = upsert_listed_features(con, listed_df)
        print(f"Upserted listed_meta_features: {updated} rows into {args.db} for {args.start}->{args.end}")
        return 0

    if args.command == "build-financial-features":
        if not args.start or not args.end:
            raise SystemExit("build-financial-features: specify --start/--end or set START/END in env/.env")
        fin_df = compute_financial_features(
            con,
            start=args.start,
            end=args.end,
        )
        from data_v2.etl.upsert.financial_features import upsert_financial_features

        updated = upsert_financial_features(con, fin_df)
        print(f"Upserted financial_features: {updated} rows into {args.db} for {args.start}->{args.end}")
        return 0

    if args.command == "build-trades-spec-features":
        df = compute_trades_spec_features(
            con,
            start=args.start or "1900-01-01",
            end=args.end or "2100-01-01",
        )
        inserted = upsert_section_flow_features(con, df)
        print(
            f"Upserted section_flow_features: {inserted} rows into {args.db} "
            f"for {args.start or 'min'}->{args.end or 'max'}"
        )
        return 0

    if args.command == "fetch-weekly-margin-interest":
        id_token = get_id_token()
        recs: list[dict[str, object]] = []
        include_divs = [v.strip() for v in args.holiday_division.split(",") if v.strip()]
        days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)
        if not days:
            # Fallback: fetch calendar directly if DuckDB calendar is empty for range
            cal_recs = fetch_calendar_records(id_token=id_token, from_date=args.start, to_date=args.end)
            cal_df = normalize_trading_calendar(cal_recs)
            _ = upsert_trading_calendar(con, cal_df)
            days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)
        if not days:
            raise SystemExit("No trading days found for weekly_margin_interest fetch range.")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = max(1, args.workers)

        def _job(day: str) -> list[dict[str, object]]:
            try:
                return fetch_weekly_margin_interest(
                    id_token=id_token,
                    start=day,
                    end=day,
                    code=args.code,  # optional; API allows code+date
                )
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_job, d): d for d in days}
            for fut in as_completed(futures):
                recs.extend(fut.result())

        df = normalize_weekly_margin_interest(recs)
        inserted = upsert_weekly_margin_interest(con, df)
        print(
            f"Upserted weekly_margin_interest into {args.db} rows={inserted} for {args.start}->{args.end} "
            f"code={args.code or 'ALL'}"
        )
        return 0

    if args.command == "fetch-short-selling":
        id_token = get_id_token()
        include_divs = [v.strip() for v in args.holiday_division.split(",") if v.strip()]
        if args.sector33code:
            recs = fetch_short_selling(
                id_token=id_token,
                start=args.start,
                end=args.end,
                sector33code=args.sector33code,
            )
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            max_workers = max(1, args.workers)
            sector_codes = SECTOR33_CODES

            def _job(sec: str) -> list[dict[str, object]]:
                try:
                    return fetch_short_selling(
                        id_token=id_token,
                        start=args.start,
                        end=args.end,
                        sector33code=sec,
                    )
                except Exception:
                    return []

            recs = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_job, sec): sec for sec in sector_codes}
                for fut in as_completed(futures):
                    recs.extend(fut.result())

        df = normalize_short_selling(recs)
        inserted = upsert_short_selling(con, df)
        print(
            f"Upserted short_selling into {args.db} rows={inserted} "
            f"for {args.start}->{args.end} sector33={args.sector33code or 'ALL'}"
        )
        return 0

    if args.command == "fetch-daily-margin-interest":
        id_token = get_id_token()
        include_divs = [v.strip() for v in args.holiday_division.split(",") if v.strip()]
        if args.code:
            recs = fetch_daily_margin_interest(id_token=id_token, start=args.start, end=args.end, code=args.code)
        else:
            days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)
            if not days:
                cal_recs = fetch_calendar_records(id_token=id_token, from_date=args.start, to_date=args.end)
                cal_df = normalize_trading_calendar(cal_recs)
                _ = upsert_trading_calendar(con, cal_df)
                days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)
            if not days:
                raise SystemExit("No trading days found for daily_margin_interest fetch range.")

            from concurrent.futures import ThreadPoolExecutor, as_completed

            max_workers = max(1, args.workers)

            def _job(day: str) -> list[dict[str, object]]:
                try:
                    return fetch_daily_margin_interest(id_token=id_token, start=day, end=day, code=None)
                except Exception:
                    return []

            recs: list[dict[str, object]] = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_job, d): d for d in days}
                for fut in as_completed(futures):
                    recs.extend(fut.result())

        df = normalize_daily_margin_interest(recs)
        inserted = upsert_daily_margin_interest(con, df)
        print(
            f"Upserted daily_margin_interest into {args.db} rows={inserted} "
            f"for {args.start}->{args.end} code={args.code or 'ALL'}"
        )
        return 0

    if args.command == "fetch-short-selling-positions":
        id_token = get_id_token()
        include_divs = [v.strip() for v in args.holiday_division.split(",") if v.strip()]
        # If code is specified, use disclosed_date_from/to + code directly.
        # If code is not specified, loop over trading days similar to listed/daily_quotes, using disclosed_date=<day>.
        if args.code:
            recs = fetch_short_selling_positions(
                id_token=id_token,
                start=args.start,
                end=args.end,
                code=args.code,
            )
        else:
            days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)
            if not days:
                cal_recs = fetch_calendar_records(id_token=id_token, from_date=args.start, to_date=args.end)
                cal_df = normalize_trading_calendar(cal_recs)
                _ = upsert_trading_calendar(con, cal_df)
                days = trading_days_from_duckdb(con, start=args.start, end=args.end, include_divs=include_divs)
            if not days:
                raise SystemExit("No trading days found for short_selling_positions fetch range.")

            from concurrent.futures import ThreadPoolExecutor, as_completed

            max_workers = max(1, args.workers)

            def _job(day: str) -> list[dict[str, object]]:
                try:
                    return fetch_short_selling_positions(
                        id_token=id_token,
                        start=day,
                        end=day,
                        code=None,
                    )
                except Exception:
                    return []

            recs: list[dict[str, object]] = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_job, d): d for d in days}
                for fut in as_completed(futures):
                    recs.extend(fut.result())

        df = normalize_short_selling_positions(recs)
        inserted = upsert_short_selling_positions(con, df)
        print(
            f"Upserted short_selling_positions into {args.db} rows={inserted} "
            f"for disclosed_date {args.start}->{args.end} code={args.code or 'ALL'}"
        )
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
