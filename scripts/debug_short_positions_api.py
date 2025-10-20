#!/usr/bin/env python3
"""
Debug script to inspect raw J-Quants short_selling_positions API response.
"""
import asyncio
import json
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

import os

from dotenv import load_dotenv

from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher

load_dotenv()


async def main():
    email = os.getenv("JQUANTS_AUTH_EMAIL")
    password = os.getenv("JQUANTS_AUTH_PASSWORD")

    if not email or not password:
        print("❌ Missing credentials in .env")
        return

    fetcher = JQuantsAsyncFetcher(email=email, password=password)

    # Test with single recent date
    test_date = "2024-10-18"

    async with aiohttp.ClientSession() as session:
        # Authenticate
        await fetcher.authenticate(session)
        print("✅ Authenticated")

        # Make direct API request
        url = f"{fetcher.base_url}/markets/short_selling_positions"
        headers = {"Authorization": f"Bearer {fetcher.id_token}"}
        params = {"disclosed_date": test_date}

        print(f"\n🔍 Fetching: {url}")
        print(f"   Params: {params}")

        async with session.get(url, headers=headers, params=params) as response:
            status = response.status
            print(f"\n📊 Status: {status}")

            if status == 200:
                data = await response.json()

                # Extract the array
                records = data.get("short_selling_positions", [])
                print(f"   Records: {len(records)}")

                if records:
                    # Show first record as formatted JSON
                    first_record = records[0]
                    print("\n📋 First record structure:")
                    print(json.dumps(first_record, indent=2, ensure_ascii=False))

                    # List all field names
                    print(f"\n🔑 All field names ({len(first_record)} fields):")
                    for key in sorted(first_record.keys()):
                        value = first_record[key]
                        value_str = str(value)[:50]  # Truncate long values
                        print(f"   - {key}: {type(value).__name__} = {value_str}")
                else:
                    print("   ⚠️  No records found for this date")
            else:
                text = await response.text()
                print(f"   Error: {text}")


if __name__ == "__main__":
    asyncio.run(main())
