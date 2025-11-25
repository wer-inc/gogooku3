from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dagster import In, Out, job, op
from data_v2.etl.auth import get_id_token
from data_v2.etl.client import (
    fetch_breakdown_for_date,
    fetch_calendar_records,
    fetch_daily_quotes_for_date,
    fetch_fs_details_for_date,
    fetch_listed_info_for_date,
    fetch_statements_for_date,
)
from data_v2.etl.config import DEFAULT_DB_PATH
from data_v2.etl.duckdb import connect_db, ensure_tables
from data_v2.etl.feature.listed_info import compute_listed_features
from data_v2.etl.feature.prices import compute_price_features
from data_v2.etl.normalize.breakdown import normalize_breakdown
from data_v2.etl.normalize.daily_quotes import normalize_daily_quotes
from data_v2.etl.normalize.listed_info import normalize_listed_info
from data_v2.etl.normalize.statements import normalize_statements
from data_v2.etl.normalize.trading_calendar import normalize_trading_calendar
from data_v2.etl.upsert.breakdown import upsert_breakdown
from data_v2.etl.upsert.daily_quotes import upsert_daily_quotes
from data_v2.etl.upsert.features import upsert_features_daily
from data_v2.etl.upsert.listed_features import upsert_listed_features
from data_v2.etl.upsert.listed_info import upsert_listed_info
from data_v2.etl.upsert.statements import upsert_statements
from data_v2.etl.upsert.trading_calendar import (
    trading_days_from_duckdb,
    upsert_trading_calendar,
)
from data_v2.etl.yfinance_tickers import MACRO_FALLBACKS, MACRO_TICKERS
from data_v2.scripts.duckdb_loader import fetch_yfinance_into_duckdb


@op(
    ins={
        "from_date": In(str, description="Start date YYYY-MM-DD", default_value="2015-01-01"),
        "to_date": In(str, description="End date YYYY-MM-DD", default_value="2025-12-31"),
        "force_refresh": In(bool, description="Force re-fetch calendar", default_value=False),
        "db_path": In(str, description="DuckDB path", default_value=str(DEFAULT_DB_PATH)),
        "threads": In(int, description="DuckDB threads (0=default)", default_value=0),
    },
    out=Out(str, description="Rows upserted into DuckDB"),
)
def duckdb_upsert_calendar_op(
    context,
    from_date: str,
    to_date: str,
    force_refresh: bool,
    db_path: str,
    threads: int,
) -> str:
    con = connect_db(Path(db_path), threads=threads or None)
    ensure_tables(con)

    # Fetch directly from API and upsert (no Parquet dependency)
    id_token = get_id_token()
    records = fetch_calendar_records(id_token=id_token, from_date=from_date, to_date=to_date)
    df = normalize_trading_calendar(records)
    inserted = upsert_trading_calendar(con, df)
    msg = f"calendar upserted: {inserted} rows into {db_path} for {from_date}->{to_date}"
    context.log.info(msg)
    return msg


@job
def duckdb_calendar_job():
    duckdb_upsert_calendar_op()


@op(
    ins={
        "start": In(str, description="Start date YYYY-MM-DD"),
        "end": In(str, description="End date YYYY-MM-DD"),
        "db_path": In(str, description="DuckDB path", default_value=str(DEFAULT_DB_PATH)),
        "holiday_division": In(str, description="HolidayDivision CSV (e.g., '1,2')", default_value="1,2"),
        "sleep_sec": In(float, description="Sleep seconds every 10 days", default_value=0.0),
        "threads": In(int, description="DuckDB threads (0=default)", default_value=0),
        "auto_fetch_calendar": In(bool, description="Fetch calendar into DuckDB if missing", default_value=True),
        "workers": In(int, description="Parallel fetch workers for listed_info", default_value=10),
    },
    out=Out(str, description="Result string"),
)
def duckdb_fetch_listed_op(
    context,
    start: str,
    end: str,
    db_path: str,
    holiday_division: str,
    sleep_sec: float,
    threads: int,
    auto_fetch_calendar: bool,
    workers: int,
) -> str:
    include_divs = [v.strip() for v in holiday_division.split(",") if v.strip()]
    con = connect_db(Path(db_path), threads=threads or None)
    ensure_tables(con)

    days = trading_days_from_duckdb(con, start=start, end=end, include_divs=include_divs)
    if not days and auto_fetch_calendar:
        id_token = get_id_token()
        cal_records = fetch_calendar_records(id_token=id_token, from_date=start, to_date=end)
        cal_df = normalize_trading_calendar(cal_records)
        inserted = upsert_trading_calendar(con, cal_df)
        context.log.info(f"[calendar] upserted {inserted} rows into {db_path} for {start}->{end}")
        days = trading_days_from_duckdb(con, start=start, end=end, include_divs=include_divs)

    if not days:
        raise RuntimeError(
            f"No trading days in trading_calendar for {start}->{end}. "
            f"Fetch calendar first or set auto_fetch_calendar=True."
        )

    id_token = get_id_token()
    max_workers = max(1, workers)

    def _job(day: str) -> tuple[str, list[dict]]:
        try:
            recs = fetch_listed_info_for_date(id_token=id_token, day=day)
            return day, recs
        except Exception:
            return day, []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_job, d): d for d in days}
        for idx, fut in enumerate(as_completed(futures), 1):
            day = futures[fut]
            day, records = fut.result()
            if records:
                df = normalize_listed_info(records)
                inserted = upsert_listed_info(con, df)
                context.log.info(f"✅ {day}: upserted {inserted} rows")
            else:
                context.log.info(f"⏩ {day}: no records returned")
            if sleep_sec > 0 and idx % 10 == 0:
                import time

                time.sleep(sleep_sec)

    msg = f"listed_info upserted into {db_path} for {start}->{end}"
    context.log.info(msg)
    return msg


@job
def duckdb_listed_job():
    duckdb_fetch_listed_op()


ENV_START = os.environ.get("START", "2015-01-01")
ENV_END = os.environ.get("END", "2025-12-31")
ENV_SLEEP = float(os.environ.get("SLEEP_SEC", "0.0"))


@op(
    ins={
        "start": In(str, description="Start date YYYY-MM-DD", default_value=ENV_START),
        "end": In(str, description="End date YYYY-MM-DD", default_value=ENV_END),
        "db_path": In(str, description="DuckDB path", default_value=str(DEFAULT_DB_PATH)),
        "sleep_sec": In(float, description="Sleep seconds every 10 tickers", default_value=ENV_SLEEP),
        "threads": In(int, description="DuckDB threads (0=default)", default_value=0),
    },
    out=Out(str, description="Result string"),
)
def duckdb_fetch_yfinance_op(
    context,
    start: str,
    end: str,
    db_path: str,
    sleep_sec: float,
    threads: int,
) -> str:
    con = connect_db(Path(db_path), threads=threads or None)
    ensure_tables(con)

    fetch_yfinance_into_duckdb(
        con,
        tickers=MACRO_TICKERS,
        start=start,
        end=end,
        sleep_sec=sleep_sec,
        fallbacks=MACRO_FALLBACKS,
    )
    msg = f"yfinance history upserted into {db_path} for tickers={','.join(MACRO_TICKERS)} {start}->{end}"
    context.log.info(msg)
    return msg


@job
def duckdb_yfinance_job():
    duckdb_fetch_yfinance_op()


@op(
    ins={
        "start": In(str, description="Start date YYYY-MM-DD", default_value=ENV_START),
        "end": In(str, description="End date YYYY-MM-DD", default_value=ENV_END),
        "db_path": In(str, description="DuckDB path", default_value=str(DEFAULT_DB_PATH)),
        "holiday_division": In(str, description="HolidayDivision CSV (e.g., '1,2')", default_value="1,2"),
        "sleep_sec": In(float, description="Sleep seconds every 10 days", default_value=ENV_SLEEP),
        "threads": In(int, description="DuckDB threads (0=default)", default_value=0),
        "auto_fetch_calendar": In(bool, description="Fetch calendar into DuckDB if missing", default_value=True),
        "workers": In(int, description="Parallel fetch workers", default_value=int(os.environ.get("WORKERS", "10"))),
    },
    out=Out(str, description="Result string"),
)
def duckdb_fetch_daily_quotes_op(
    context,
    start: str,
    end: str,
    db_path: str,
    holiday_division: str,
    sleep_sec: float,
    threads: int,
    auto_fetch_calendar: bool,
    workers: int,
) -> str:
    include_divs = [v.strip() for v in holiday_division.split(",") if v.strip()]
    con = connect_db(Path(db_path), threads=threads or None)
    ensure_tables(con)

    days = trading_days_from_duckdb(con, start=start, end=end, include_divs=include_divs)
    if not days and auto_fetch_calendar:
        id_token = get_id_token()
        cal_records = fetch_calendar_records(id_token=id_token, from_date=start, to_date=end)
        cal_df = normalize_trading_calendar(cal_records)
        inserted = upsert_trading_calendar(con, cal_df)
        context.log.info(f"[calendar] upserted {inserted} rows into {db_path} for {start}->{end}")
        days = trading_days_from_duckdb(con, start=start, end=end, include_divs=include_divs)

    if not days:
        raise RuntimeError(
            f"No trading days in trading_calendar for {start}->{end}. "
            f"Fetch calendar first or set auto_fetch_calendar=True."
        )

    id_token = get_id_token()
    max_workers = max(1, workers)

    def _job(day: str):
        try:
            recs = fetch_daily_quotes_for_date(id_token=id_token, day=day)
            return day, recs
        except Exception:
            return day, []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_job, d): d for d in days}
        for idx, fut in enumerate(as_completed(futures), 1):
            day, records = fut.result()
            if records:
                df = normalize_daily_quotes(records)
                inserted = upsert_daily_quotes(con, df)
                context.log.info(f"✅ {day}: upserted {inserted} rows")
            else:
                context.log.info(f"⏩ {day}: no records returned")
            if sleep_sec > 0 and idx % 10 == 0:
                import time

                time.sleep(sleep_sec)

    msg = f"daily_quotes upserted into {db_path} for {start}->{end}"
    context.log.info(msg)
    return msg


@job
def duckdb_daily_quotes_job():
    duckdb_fetch_daily_quotes_op()


@op(
    ins={
        "start": In(str, description="Start date YYYY-MM-DD", default_value=ENV_START),
        "end": In(str, description="End date YYYY-MM-DD", default_value=ENV_END),
        "db_path": In(str, description="DuckDB path", default_value=str(DEFAULT_DB_PATH)),
        "threads": In(int, description="DuckDB threads (0=default)", default_value=0),
        "warmup_days": In(int, description="Calendar days to look back for rolling/talib warmup", default_value=260),
    },
    out=Out(str, description="Result string"),
)
def duckdb_compute_features_op(
    context,
    start: str,
    end: str,
    db_path: str,
    threads: int,
    warmup_days: int,
) -> str:
    con = connect_db(Path(db_path), threads=threads or None)
    ensure_tables(con)
    # Guard: warn if warmup extends before available data
    min_q_date = con.execute("SELECT min(date) FROM daily_quotes").fetchone()[0]
    if min_q_date:
        warmup_start = (
            date.fromisoformat(start) - timedelta(days=warmup_days) if warmup_days > 0 else date.fromisoformat(start)
        )
        if warmup_start < min_q_date:
            context.log.warning(
                f"warmup truncated: requested start-warmup={warmup_start} but daily_quotes starts at {min_q_date}"
            )
    df = compute_price_features(con, start=start, end=end, warmup_days=warmup_days)
    inserted = upsert_features_daily(con, df)
    msg = f"features_daily upserted: {inserted} rows into {db_path} for {start}->{end}"
    context.log.info(msg)
    return msg


@job
def duckdb_features_job():
    duckdb_compute_features_op()


# New-style ops/jobs aligned to CLI naming
@op(
    ins={
        "start": In(str, description="Start date YYYY-MM-DD", default_value=ENV_START),
        "end": In(str, description="End date YYYY-MM-DD", default_value=ENV_END),
        "db_path": In(str, description="DuckDB path", default_value=str(DEFAULT_DB_PATH)),
        "threads": In(int, description="DuckDB threads (0=default)", default_value=0),
        "warmup_days": In(int, description="Calendar days to look back for rolling/talib warmup", default_value=260),
    },
    out=Out(str, description="Result string"),
)
def duckdb_build_price_flow_features_op(
    context,
    start: str,
    end: str,
    db_path: str,
    threads: int,
    warmup_days: int,
) -> str:
    con = connect_db(Path(db_path), threads=threads or None)
    ensure_tables(con)
    min_q_date = con.execute("SELECT min(date) FROM daily_quotes").fetchone()[0]
    if min_q_date:
        warmup_start = (
            date.fromisoformat(start) - timedelta(days=warmup_days) if warmup_days > 0 else date.fromisoformat(start)
        )
        if warmup_start < min_q_date:
            context.log.warning(
                f"warmup truncated: requested start-warmup={warmup_start} but daily_quotes starts at {min_q_date}"
            )
    df = compute_price_features(con, start=start, end=end, warmup_days=warmup_days)
    inserted = upsert_features_daily(con, df)
    msg = f"price_flow_features upserted: {inserted} rows into {db_path} for {start}->{end}"
    context.log.info(msg)
    return msg


@job
def duckdb_price_flow_features_job():
    duckdb_build_price_flow_features_op()


@op(
    ins={
        "start": In(str, description="Start date YYYY-MM-DD", default_value=ENV_START),
        "end": In(str, description="End date YYYY-MM-DD", default_value=ENV_END),
        "db_path": In(str, description="DuckDB path", default_value=str(DEFAULT_DB_PATH)),
        "threads": In(int, description="DuckDB threads (0=default)", default_value=0),
    },
    out=Out(str, description="Result string"),
)
def duckdb_build_listed_meta_features_op(
    context,
    start: str,
    end: str,
    db_path: str,
    threads: int,
) -> str:
    con = connect_db(Path(db_path), threads=threads or None)
    ensure_tables(con)
    df = compute_listed_features(con, start=start, end=end)
    updated = upsert_listed_features(con, df)
    msg = f"listed_meta_features upserted: {updated} rows into {db_path} for {start}->{end}"
    context.log.info(msg)
    return msg


@job
def duckdb_listed_meta_features_job():
    duckdb_build_listed_meta_features_op()


@op(
    ins={
        "start": In(str, description="Start date YYYY-MM-DD", default_value=ENV_START),
        "end": In(str, description="End date YYYY-MM-DD", default_value=ENV_END),
        "db_path": In(str, description="DuckDB path", default_value=str(DEFAULT_DB_PATH)),
        "holiday_division": In(str, description="HolidayDivision CSV (e.g., '1,2')", default_value="1,2"),
        "sleep_sec": In(float, description="Sleep seconds every 10 days", default_value=ENV_SLEEP),
        "threads": In(int, description="DuckDB threads (0=default)", default_value=0),
        "auto_fetch_calendar": In(bool, description="Fetch calendar into DuckDB if missing", default_value=True),
        "workers": In(int, description="Parallel fetch workers", default_value=int(os.environ.get("WORKERS", "10"))),
    },
    out=Out(str, description="Result string"),
)
def duckdb_fetch_breakdown_op(
    context,
    start: str,
    end: str,
    db_path: str,
    holiday_division: str,
    sleep_sec: float,
    threads: int,
    auto_fetch_calendar: bool,
    workers: int,
) -> str:
    include_divs = [v.strip() for v in holiday_division.split(",") if v.strip()]
    con = connect_db(Path(db_path), threads=threads or None)
    ensure_tables(con)

    days = trading_days_from_duckdb(con, start=start, end=end, include_divs=include_divs)
    if not days and auto_fetch_calendar:
        id_token = get_id_token()
        cal_records = fetch_calendar_records(id_token=id_token, from_date=start, to_date=end)
        cal_df = normalize_trading_calendar(cal_records)
        inserted = upsert_trading_calendar(con, cal_df)
        context.log.info(f"[calendar] upserted {inserted} rows into {db_path} for {start}->{end}")
        days = trading_days_from_duckdb(con, start=start, end=end, include_divs=include_divs)

    if not days:
        raise RuntimeError(
            f"No trading days in trading_calendar for {start}->{end}. "
            f"Fetch calendar first or set auto_fetch_calendar=True."
        )

    id_token = get_id_token()
    max_workers = max(1, workers)

    def _job(day: str):
        try:
            recs = fetch_breakdown_for_date(id_token=id_token, day=day)
            return day, recs
        except Exception:
            return day, []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_job, d): d for d in days}
        for idx, fut in enumerate(as_completed(futures), 1):
            day, records = fut.result()
            if records:
                df = normalize_breakdown(records)
                inserted = upsert_breakdown(con, df)
                context.log.info(f"✅ {day}: upserted {inserted} rows")
            else:
                context.log.info(f"⏩ {day}: no records returned")
            if sleep_sec > 0 and idx % 10 == 0:
                import time

                time.sleep(sleep_sec)

    msg = f"breakdown upserted into {db_path} for {start}->{end}"
    context.log.info(msg)
    return msg


@op(
    ins={
        "start": In(str, description="Start date YYYY-MM-DD", default_value=ENV_START),
        "end": In(str, description="End date YYYY-MM-DD", default_value=ENV_END),
        "db_path": In(str, description="DuckDB path", default_value=str(DEFAULT_DB_PATH)),
        "holiday_division": In(str, description="HolidayDivision CSV (e.g., '1,2')", default_value="1,2"),
        "sleep_sec": In(float, description="Sleep seconds every 10 days", default_value=ENV_SLEEP),
        "threads": In(int, description="DuckDB threads (0=default)", default_value=0),
        "auto_fetch_calendar": In(bool, description="Fetch calendar into DuckDB if missing", default_value=True),
        "workers": In(int, description="Parallel fetch workers", default_value=int(os.environ.get("WORKERS", "10"))),
    },
    out=Out(str, description="Result string"),
)
def duckdb_fetch_statements_op(
    context,
    start: str,
    end: str,
    db_path: str,
    holiday_division: str,
    sleep_sec: float,
    threads: int,
    auto_fetch_calendar: bool,
    workers: int,
) -> str:
    include_divs = [v.strip() for v in holiday_division.split(",") if v.strip()]
    con = connect_db(Path(db_path), threads=threads or None)
    ensure_tables(con)

    days = trading_days_from_duckdb(con, start=start, end=end, include_divs=include_divs)
    if not days and auto_fetch_calendar:
        id_token = get_id_token()
        cal_records = fetch_calendar_records(id_token=id_token, from_date=start, to_date=end)
        cal_df = normalize_trading_calendar(cal_records)
        inserted = upsert_trading_calendar(con, cal_df)
        context.log.info(f"[calendar] upserted {inserted} rows into {db_path} for {start}->{end}")
        days = trading_days_from_duckdb(con, start=start, end=end, include_divs=include_divs)

    if not days:
        raise RuntimeError(
            f"No trading days in trading_calendar for {start}->{end}. "
            f"Fetch calendar first or set auto_fetch_calendar=True."
        )

    id_token = get_id_token()
    max_workers = max(1, workers)

    def _job(day: str) -> tuple[str, list[dict]]:
        try:
            recs = fetch_statements_for_date(id_token=id_token, day=day)
            return day, recs
        except Exception:
            return day, []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_job, d): d for d in days}
        for idx, fut in enumerate(as_completed(futures), 1):
            day = futures[fut]
            day, records = fut.result()
            if records:
                df = normalize_statements(records)
                inserted = upsert_statements(con, df)
                context.log.info(f"✅ {day}: upserted {inserted} statements rows")
            else:
                context.log.info(f"⏩ {day}: no statements returned")
            if sleep_sec > 0 and idx % 10 == 0:
                import time

                time.sleep(sleep_sec)

    msg = f"statements upserted into {db_path} for {start}->{end}"
    context.log.info(msg)
    return msg


@op(
    ins={
        "start": In(str, description="Start date YYYY-MM-DD", default_value=ENV_START),
        "end": In(str, description="End date YYYY-MM-DD", default_value=ENV_END),
        "db_path": In(str, description="DuckDB path", default_value=str(DEFAULT_DB_PATH)),
        "holiday_division": In(str, description="HolidayDivision CSV (e.g., '1,2')", default_value="1,2"),
        "sleep_sec": In(float, description="Sleep seconds every 10 days", default_value=ENV_SLEEP),
        "threads": In(int, description="DuckDB threads (0=default)", default_value=0),
        "auto_fetch_calendar": In(bool, description="Fetch calendar into DuckDB if missing", default_value=True),
        "workers": In(int, description="Parallel fetch workers", default_value=int(os.environ.get("WORKERS", "10"))),
    },
    out=Out(str, description="Result string"),
)
def duckdb_fetch_fs_details_op(
    context,
    start: str,
    end: str,
    db_path: str,
    holiday_division: str,
    sleep_sec: float,
    threads: int,
    auto_fetch_calendar: bool,
    workers: int,
) -> str:
    include_divs = [v.strip() for v in holiday_division.split(",") if v.strip()]
    con = connect_db(Path(db_path), threads=threads or None)
    ensure_tables(con)

    days = trading_days_from_duckdb(con, start=start, end=end, include_divs=include_divs)
    if not days and auto_fetch_calendar:
        id_token = get_id_token()
        cal_records = fetch_calendar_records(id_token=id_token, from_date=start, to_date=end)
        cal_df = normalize_trading_calendar(cal_records)
        inserted = upsert_trading_calendar(con, cal_df)
        context.log.info(f"[calendar] upserted {inserted} rows into {db_path} for {start}->{end}")
        days = trading_days_from_duckdb(con, start=start, end=end, include_divs=include_divs)

    if not days:
        raise RuntimeError(
            f"No trading days in trading_calendar for {start}->{end}. "
            f"Fetch calendar first or set auto_fetch_calendar=True."
        )

    id_token = get_id_token()
    max_workers = max(1, workers)

    def _job(day: str) -> tuple[str, list[dict]]:
        try:
            recs = fetch_fs_details_for_date(id_token=id_token, day=day)
            return day, recs
        except Exception:
            return day, []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_job, d): d for d in days}
        for idx, fut in enumerate(as_completed(futures), 1):
            day = futures[fut]
            day, records = fut.result()
            if records:
                df = normalize_fs_details(records)
                inserted = upsert_fs_details(con, df)
                context.log.info(f"✅ {day}: upserted {inserted} fs_details rows")
            else:
                context.log.info(f"⏩ {day}: no fs_details returned")
            if sleep_sec > 0 and idx % 10 == 0:
                import time

                time.sleep(sleep_sec)

    msg = f"fs_details upserted into {db_path} for {start}->{end}"
    context.log.info(msg)
    return msg


@job
def duckdb_breakdown_job():
    duckdb_fetch_breakdown_op()


@job
def duckdb_statements_job():
    duckdb_fetch_statements_op()


@job
def duckdb_fs_details_job():
    duckdb_fetch_fs_details_op()
