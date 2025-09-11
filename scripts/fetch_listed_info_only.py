#!/usr/bin/env python3
"""
Fetch full listed_info (all companies) from J-Quants and save to output/ as Parquet.

Requires .env with JQUANTS_AUTH_EMAIL and JQUANTS_AUTH_PASSWORD (or env set).

Usage:
  python scripts/fetch_listed_info_only.py [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

import aiohttp
import polars as pl

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._archive.run_pipeline import JQuantsAsyncFetcher  # type: ignore


async def fetch_listed_info(date: str | None) -> pl.DataFrame:
    email = os.getenv("JQUANTS_AUTH_EMAIL", "")
    password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
    if not email or not password:
        raise RuntimeError("Set JQUANTS_AUTH_EMAIL/JQUANTS_AUTH_PASSWORD in env/.env")
    fetcher = JQuantsAsyncFetcher(email, password)
    async with aiohttp.ClientSession() as session:
        await fetcher.authenticate(session)
        # _archive.fetcher has get_listed_info(session, date=None)
        df = await fetcher.get_listed_info(session, date=date)  # type: ignore
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch listed_info only")
    ap.add_argument("--date", type=str, default=None, help="Optional date YYYY-MM-DD for snapshot")
    args = ap.parse_args()

    df = asyncio.run(fetch_listed_info(args.date))
    if df is None or df.is_empty():
        print("listed_info fetch returned empty dataset")
        return 1

    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)
    tag = (args.date or datetime.now().strftime("%Y%m%d"))
    path = outdir / f"listed_info_history_{tag}_full.parquet"
    df.write_parquet(path)
    print(f"saved: {path} rows={len(df)}")
    print(f"columns: {df.columns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

