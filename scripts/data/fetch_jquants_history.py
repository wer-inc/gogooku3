#!/usr/bin/env python3
"""
Fetch 5-year history for prices, TOPIX, trade-spec (markets/trades_spec), and financial statements
from J-Quants and save them as Parquet files under an output directory.

Usage examples:
  - All (default 5 years):
      python scripts/data/fetch_jquants_history.py --jquants --all

  - Specific range:
      python scripts/data/fetch_jquants_history.py --jquants \
        --start-date 2020-09-03 --end-date 2025-09-03 --all

  - Only TOPIX:
      python scripts/data/fetch_jquants_history.py --jquants --topix \
        --start-date 2020-09-03 --end-date 2025-09-03

  - Only PRICES (all stocks by date-axis aggregation):
      python scripts/data/fetch_jquants_history.py --jquants --prices \
        --start-date 2020-09-03 --end-date 2025-09-03

  - Only trade-spec and statements to default 5-year window:
      python scripts/data/fetch_jquants_history.py --jquants --trades-spec --statements

Environment:
  - Requires JQuants credentials in .env or environment:
      JQUANTS_AUTH_EMAIL, JQUANTS_AUTH_PASSWORD
"""

from __future__ import annotations

import os
import sys
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import logging
from typing import List, Optional

import aiohttp
import polars as pl
from dotenv import load_dotenv


# Load .env if present
ROOT = Path(__file__).resolve().parents[2]
env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Ensure we can import the existing JQuants fetcher
sys.path.insert(0, str((ROOT / "scripts" / "_archive").resolve()))
try:
    from run_pipeline import JQuantsAsyncFetcher  # type: ignore
except Exception as e:  # pragma: no cover
    print(f"Failed to import JQuantsAsyncFetcher: {e}")
    raise

# Trading calendar for business-day iteration
sys.path.insert(0, str((ROOT / "scripts" / "components").resolve()))
from trading_calendar_fetcher import TradingCalendarFetcher  # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fetch_jquants_history")


def _daterange_inclusive(start: datetime, end: datetime) -> List[datetime]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _write_parquet_with_date_meta(df: pl.DataFrame, path: Path, start_date: str, end_date: str) -> None:
    """Write Parquet with start/end date embedded as file metadata when possible.

    Falls back to plain polars write if pyarrow is unavailable.
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        # Convert to Arrow table
        table = df.to_arrow()
        # Attach/merge metadata
        schema = table.schema
        existing = schema.metadata or {}
        # Ensure bytes keys/values
        meta = dict(existing)
        meta.update({
            b"start_date": start_date.encode("utf-8"),
            b"end_date": end_date.encode("utf-8"),
            b"generator": b"fetch_jquants_history.py",
        })
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, str(path))
    except Exception:
        # Fallback: polars write
        df.write_parquet(path)


async def fetch_statements_by_date(
    fetcher: JQuantsAsyncFetcher,
    session: aiohttp.ClientSession,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Fetch financial statements by date (date-axis API) with simple pagination.

    J-Quants endpoint: /fins/statements supports `date=YYYYMMDD`.
    We iterate business/calendar days; empty days return 404/empty.
    """
    base_url = f"{fetcher.base_url}/fins/statements"
    headers = {"Authorization": f"Bearer {fetcher.id_token}"}

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days = _daterange_inclusive(start_dt, end_dt)

    all_rows: List[dict] = []

    async def fetch_one(day: datetime):
        date_api = day.strftime("%Y%m%d")
        pagination_key: Optional[str] = None
        first_try = True
        while True:
            params = {"date": date_api}
            if pagination_key:
                params["pagination_key"] = pagination_key
            try:
                async with fetcher.semaphore:
                    async with session.get(base_url, headers=headers, params=params) as resp:
                        if resp.status == 404:
                            # No disclosures that day
                            return
                        if resp.status != 200:
                            if first_try:
                                logger.debug(f"statements {date_api}: HTTP {resp.status}")
                            return
                        data = await resp.json()
                        items = data.get("statements", [])
                        if items:
                            all_rows.extend(items)
                        pagination_key = data.get("pagination_key")
                        if not pagination_key:
                            return
            except Exception as e:
                logger.debug(f"statements {date_api} error: {e}")
                return
            finally:
                first_try = False

    # Limit concurrency to avoid overwhelming API (re-use internal semaphore)
    batch_size = 32
    for i in range(0, len(days), batch_size):
        batch = days[i : i + batch_size]
        await asyncio.gather(*(fetch_one(d) for d in batch))

    return pl.DataFrame(all_rows) if all_rows else pl.DataFrame()


async def fetch_prices_by_business_days(
    fetcher: JQuantsAsyncFetcher,
    session: aiohttp.ClientSession,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Fetch price daily_quotes for all stocks by iterating business days and aggregate.

    Endpoint supports `date=YYYYMMDD` returning all codes for that day with pagination.
    """
    url = f"{fetcher.base_url}/prices/daily_quotes"
    headers = {"Authorization": f"Bearer {fetcher.id_token}"}

    # Get business days
    cal_fetcher = TradingCalendarFetcher(fetcher)
    calendar = await cal_fetcher.get_trading_calendar(start_date, end_date, session)
    days = calendar.get("business_days", [])
    logger.info(f"Fetching prices for {len(days)} business days")

    async def fetch_one(day: str) -> pl.DataFrame:
        day_api = day.replace("-", "")
        all_rows: list[dict] = []
        pagination_key: str | None = None
        while True:
            params = {"date": day_api}
            if pagination_key:
                params["pagination_key"] = pagination_key
            try:
                async with fetcher.semaphore:
                    async with session.get(url, headers=headers, params=params) as resp:
                        if resp.status != 200:
                            return pl.DataFrame()
                        data = await resp.json()
                        rows = data.get("daily_quotes", [])
                        if rows:
                            all_rows.extend(rows)
                        pagination_key = data.get("pagination_key")
                        if not pagination_key:
                            break
            except Exception as e:
                logger.debug(f"prices {day} error: {e}")
                return pl.DataFrame()
        return pl.DataFrame(all_rows) if all_rows else pl.DataFrame()

    # Moderate concurrency
    sem = asyncio.Semaphore(8)

    async def guarded(day: str) -> pl.DataFrame:
        async with sem:
            return await fetch_one(day)

    aggregated: list[pl.DataFrame] = []
    for i in range(0, len(days), 32):
        batch = days[i : i + 32]
        results = await asyncio.gather(*(guarded(d) for d in batch))
        aggregated.extend([df for df in results if df is not None and not df.is_empty()])
        logger.info(f"Fetched {sum(len(df) for df in results if df is not None)} rows in batch {i//32+1}")

    return pl.concat(aggregated) if aggregated else pl.DataFrame()


async def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch 5-year TOPIX, trade-spec, and statements history from J-Quants")
    parser.add_argument("--jquants", action="store_true", help="Use J-Quants API (requires creds)")
    parser.add_argument("--prices", action="store_true", help="Fetch prices history (aggregated across business days)")
    parser.add_argument("--topix", action="store_true", help="Fetch TOPIX history")
    parser.add_argument("--trades-spec", action="store_true", help="Fetch trade-spec history")
    parser.add_argument("--statements", action="store_true", help="Fetch statements history")
    parser.add_argument("--all", action="store_true", help="Fetch all of TOPIX + trade-spec + statements")
    parser.add_argument("--start-date", type=str, help="Start date YYYY-MM-DD (default: today-5y)")
    parser.add_argument("--end-date", type=str, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output", help="Output directory")
    args = parser.parse_args()

    if not args.jquants:
        logger.error("--jquants is required to access the API")
        return 1

    # Determine which to fetch
    do_prices = args.all or args.prices
    do_topix = args.all or args.topix
    do_trades = args.all or args.trades_spec
    do_statements = args.all or args.statements
    if not any([do_prices, do_topix, do_trades, do_statements]):
        logger.info("Nothing selected; add --all or a specific flag (e.g., --topix)")
        return 0

    end_dt = datetime.now() if not args.end_date else datetime.strptime(args.end_date, "%Y-%m-%d")
    # Approximate 5 years as 1826 days to include leap years
    start_dt = end_dt - timedelta(days=1826)
    if args.start_date:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")

    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    email = os.getenv("JQUANTS_AUTH_EMAIL", "")
    password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
    if not email or not password:
        logger.error("Missing JQuants credentials (JQUANTS_AUTH_EMAIL / JQUANTS_AUTH_PASSWORD)")
        return 1

    async with aiohttp.ClientSession() as session:
        fetcher = JQuantsAsyncFetcher(email, password)
        await fetcher.authenticate(session)

        # PRICES (aggregate across business days)
        if do_prices:
            logger.info(f"Fetching PRICES (all stocks by date-axis) {start_date} → {end_date}")
            prices_df = await fetch_prices_by_business_days(fetcher, session, start_date, end_date)
            if not prices_df.is_empty():
                pq = out_dir / f"prices_history_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                _write_parquet_with_date_meta(prices_df, pq, start_date, end_date)
                logger.info(f"Saved PRICES: {pq}")
            else:
                logger.warning("PRICES returned no data")

        # TOPIX
        if do_topix:
            logger.info(f"Fetching TOPIX {start_date} → {end_date}")
            topix_df = await fetcher.fetch_topix_data(session, start_date, end_date)
            if not topix_df.is_empty():
                pq = out_dir / f"topix_history_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                _write_parquet_with_date_meta(topix_df, pq, start_date, end_date)
                logger.info(f"Saved TOPIX: {pq}")
            else:
                logger.warning("TOPIX returned no data")

        # trade-spec
        if do_trades:
            logger.info(f"Fetching trade-spec {start_date} → {end_date}")
            trades_df = await fetcher.get_trades_spec(session, start_date, end_date)
            if not trades_df.is_empty():
                pq = out_dir / f"trades_spec_history_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                _write_parquet_with_date_meta(trades_df, pq, start_date, end_date)
                logger.info(f"Saved trade-spec: {pq}")
            else:
                logger.warning("trade-spec returned no data")

        # statements
        if do_statements:
            logger.info(f"Fetching statements (by date) {start_date} → {end_date}")
            stm_df = await fetch_statements_by_date(fetcher, session, start_date, end_date)
            if not stm_df.is_empty():
                pq = out_dir / f"event_raw_statements_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                _write_parquet_with_date_meta(stm_df, pq, start_date, end_date)
                logger.info(f"Saved statements: {pq}")
                # Update convenient symlink for enrichment auto-detection
                latest = out_dir / "event_raw_statements.parquet"
                try:
                    if latest.exists() or latest.is_symlink():
                        latest.unlink()
                except Exception:
                    pass
                latest.symlink_to(pq.name)
            else:
                logger.warning("statements returned no data")

        logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
