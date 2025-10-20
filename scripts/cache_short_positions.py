#!/usr/bin/env python3
"""
Fetch and cache short selling positions data (raw, before normalize).
This helps debug the column structure and ensure data is cached.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


import aiohttp

from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher


async def main():
    print("🔍 Fetching short selling positions data (RAW, before normalize)...")

    # Load credentials from environment
    import os

    from dotenv import load_dotenv

    load_dotenv()

    email = os.getenv("JQUANTS_AUTH_EMAIL")
    password = os.getenv("JQUANTS_AUTH_PASSWORD")

    if not email or not password:
        print("❌ JQUANTS_AUTH_EMAIL or JQUANTS_AUTH_PASSWORD not set in .env")
        return

    # Initialize fetcher
    fetcher = JQuantsAsyncFetcher(email=email, password=password)

    # Date range
    start_date = "2020-10-20"
    end_date = "2025-10-19"

    # Fetch data using aiohttp session
    print(f"📅 Date range: {start_date} → {end_date}")

    async with aiohttp.ClientSession() as session:
        # Authenticate first
        print("🔑 Authenticating...")
        await fetcher.authenticate(session)
        print("✅ Authenticated")

        # Now fetch data
        raw_result = await fetcher.get_short_selling_positions(
            session, start_date, end_date
        )

    if raw_result is None or raw_result.is_empty():
        print("❌ No data retrieved from API")
        return

    print(f"✅ Retrieved {len(raw_result)} records from API (BEFORE normalize)")
    print(f"\n📊 RAW Columns: {raw_result.columns}")
    print("\n📋 RAW Schema:")
    for col, dtype in raw_result.schema.items():
        print(f"  {col}: {dtype}")

    print("\n🔍 First 5 rows (RAW):")
    print(raw_result.head(5))

    # Save RAW data (before normalize)
    output_dir = Path("output/raw/short_selling")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = (
        output_dir
        / f"short_positions_RAW_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
    )
    raw_result.write_parquet(raw_path)
    print(
        f"\n💾 Saved RAW data: {raw_path} ({raw_path.stat().st_size / 1024 / 1024:.2f} MB)"
    )

    # Now check what normalize does
    print("\n🔧 Testing normalize...")
    fetcher_instance = fetcher

    # Call the normalize method directly
    normalized = fetcher_instance._normalize_short_selling_positions_data(raw_result)

    print(f"📊 After normalize: {len(normalized)} records")
    if not normalized.is_empty():
        print(f"   Columns: {normalized.columns}")
        print("   First 3 rows:")
        print(normalized.head(3))
    else:
        print("   ❌ Empty after normalize!")

        # Debug: Check if Code/Date columns exist in raw data
        print("\n🔍 Debug: Checking for Code/Date columns in raw data...")
        has_code = "Code" in raw_result.columns
        has_date = "Date" in raw_result.columns
        print(f"   Has 'Code': {has_code}")
        print(f"   Has 'Date': {has_date}")

        if not has_code:
            print("\n   💡 'Code' column not found. Checking alternatives:")
            code_like = [
                c
                for c in raw_result.columns
                if "code" in c.lower() or "stock" in c.lower()
            ]
            print(f"      Candidates: {code_like}")

        if not has_date:
            print("\n   💡 'Date' column not found. Checking alternatives:")
            date_like = [c for c in raw_result.columns if "date" in c.lower()]
            print(f"      Candidates: {date_like}")


if __name__ == "__main__":
    asyncio.run(main())
