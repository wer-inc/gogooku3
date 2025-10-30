#!/usr/bin/env python3
"""
Test J-Quants API endpoints for Standard plan availability.

This script tests questionable API endpoints to determine if they are available
under the Standard plan subscription.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import aiohttp
from dotenv import load_dotenv

from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher

# Load environment variables
load_dotenv()


async def test_api_endpoints():
    """Test each questionable API endpoint for Standard plan availability."""

    # Get credentials from environment
    email = os.getenv("JQUANTS_AUTH_EMAIL")
    password = os.getenv("JQUANTS_AUTH_PASSWORD")

    if not email or not password:
        print("❌ Error: JQUANTS_AUTH_EMAIL and JQUANTS_AUTH_PASSWORD must be set in .env")
        return

    fetcher = JQuantsAsyncFetcher(email=email, password=password)

    # Test configuration
    test_start_date = "2024-01-01"
    test_end_date = "2024-01-10"

    print("=" * 80)
    print("J-Quants API Standard Plan - Endpoint Availability Test")
    print("=" * 80)
    print(f"Test period: {test_start_date} to {test_end_date}\n")

    async with aiohttp.ClientSession() as session:
        # Authenticate first
        print("🔐 Authenticating...")
        try:
            await fetcher.authenticate(session)
            print("✅ Authentication successful\n")
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return

        # Test endpoints
        tests = [
            {
                "name": "Daily Margin Interest",
                "endpoint": "/markets/daily_margin_interest",
                "method": lambda: fetcher.get_daily_margin_interest(
                    session, test_start_date, test_end_date
                ),
            },
            {
                "name": "Index Option (NK225)",
                "endpoint": "/option/index_option",
                "method": lambda: fetcher.get_index_option(
                    session, test_start_date, test_end_date
                ),
            },
            {
                "name": "Earnings Announcements",
                "endpoint": "/fins/announcement",
                "method": lambda: fetcher.get_earnings_announcements(
                    session, test_start_date, test_end_date
                ),
            },
            {
                "name": "Short Selling Positions",
                "endpoint": "/markets/short_selling_positions",
                "method": lambda: fetcher.get_short_selling_positions(
                    session, test_start_date, test_end_date
                ),
            },
        ]

        results = []

        for test in tests:
            print(f"Testing: {test['name']}")
            print(f"Endpoint: {test['endpoint']}")

            try:
                result = await test["method"]()

                if result is not None and not result.is_empty():
                    rows = len(result)
                    cols = len(result.columns)
                    status = "✅ AVAILABLE"
                    detail = f"{rows} rows, {cols} columns"
                    available = True
                elif result is not None and result.is_empty():
                    status = "⚠️  AVAILABLE (but empty for test period)"
                    detail = "0 rows (may be normal for test dates)"
                    available = True
                else:
                    status = "⚠️  RETURNED NULL"
                    detail = "Method returned None"
                    available = False

            except Exception as e:
                error_str = str(e)

                # Check for permission/subscription errors
                if any(keyword in error_str.lower() for keyword in
                       ['403', 'forbidden', 'unauthorized', 'subscription', 'premium']):
                    status = "❌ PREMIUM REQUIRED"
                    detail = f"Access denied: {error_str[:100]}"
                    available = False
                elif '404' in error_str or 'not found' in error_str.lower():
                    status = "❌ ENDPOINT NOT FOUND"
                    detail = f"API endpoint may not exist: {error_str[:100]}"
                    available = False
                elif '400' in error_str or 'bad request' in error_str.lower():
                    status = "⚠️  BAD REQUEST"
                    detail = f"May need different parameters: {error_str[:100]}"
                    available = False
                else:
                    status = "⚠️  ERROR"
                    detail = f"{type(e).__name__}: {error_str[:100]}"
                    available = False

            print(f"Status: {status}")
            print(f"Detail: {detail}")
            print("-" * 80)

            results.append({
                "name": test["name"],
                "endpoint": test["endpoint"],
                "status": status,
                "detail": detail,
                "available": available,
            })

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        available_count = sum(1 for r in results if r["available"])
        unavailable_count = len(results) - available_count

        print(f"\n✅ Available APIs: {available_count}/{len(results)}")
        for r in results:
            if r["available"]:
                print(f"   • {r['name']} ({r['endpoint']})")

        print(f"\n❌ Unavailable APIs: {unavailable_count}/{len(results)}")
        for r in results:
            if not r["available"]:
                print(f"   • {r['name']} ({r['endpoint']})")
                print(f"     Reason: {r['detail']}")

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        for r in results:
            if not r["available"] and "PREMIUM REQUIRED" in r["status"]:
                print(f"\n❌ Disable: {r['name']}")
                print("   This API requires Premium plan subscription.")
                print("   Update run_full_dataset.py to disable by default.")
            elif not r["available"] and "NOT FOUND" in r["status"]:
                print(f"\n❌ Remove: {r['name']}")
                print("   This endpoint may not exist in J-Quants API.")
                print("   Consider removing this feature entirely.")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_api_endpoints())
