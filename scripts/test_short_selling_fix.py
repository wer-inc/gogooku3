#!/usr/bin/env python3
"""
Quick test to verify short_selling normalization fix.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os

import aiohttp
from dotenv import load_dotenv

from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher

load_dotenv()


async def main():
    email = os.getenv("JQUANTS_AUTH_EMAIL")
    password = os.getenv("JQUANTS_AUTH_PASSWORD")

    if not email or not password:
        print("❌ Missing credentials")
        return

    fetcher = JQuantsAsyncFetcher(email=email, password=password)

    # Test with small date range
    start_date = "2024-10-15"
    end_date = "2024-10-18"

    async with aiohttp.ClientSession() as session:
        print("🔑 Authenticating...")
        await fetcher.authenticate(session)
        print("✅ Authenticated\n")

        print(f"📅 Fetching short_selling data: {start_date} → {end_date}")
        result = await fetcher.get_short_selling(session, start_date, end_date)

        print("\n📊 Results:")
        print(f"   Total records: {len(result)}")

        if not result.is_empty():
            print(f"   Columns: {result.columns}")
            print(f"   Date range: {result['Date'].min()} → {result['Date'].max()}")
            print(f"   Unique codes: {result['Code'].n_unique()}")
            print("\n   First 3 rows:")
            print(result.head(3))
            print("\n✅ Normalization SUCCESS!")
        else:
            print("   ❌ Empty result after normalization")


if __name__ == "__main__":
    asyncio.run(main())
