#!/usr/bin/env python3
"""Test script for Phase 2 partial match cache functionality.

This script tests:
1. Partial match detection in _find_latest_with_date_range()
2. Differential fetching logic in fetch methods
3. Cache merging with pl.concat()
4. Extended cache saving

Usage:
    python scripts/cache/test_partial_match.py
"""

import os
import sys
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parents[2] / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_helper_function():
    """Test _find_latest_with_date_range() partial match detection."""
    from scripts.pipelines.run_pipeline_v4_optimized import _find_latest_with_date_range

    logger.info("=" * 80)
    logger.info("TEST 1: Helper Function - _find_latest_with_date_range()")
    logger.info("=" * 80)

    # Test scenarios
    test_cases = [
        {
            "name": "Complete match",
            "req_start": "2020-01-01",
            "req_end": "2020-12-31",
            "expected_type": "complete"
        },
        {
            "name": "Partial match - extend forward",
            "req_start": "2020-01-01",
            "req_end": "2021-01-15",  # 15 days beyond cache
            "expected_type": "partial"
        },
        {
            "name": "Partial match - extend backward",
            "req_start": "2019-12-15",  # 17 days before cache
            "req_end": "2020-12-31",
            "expected_type": "partial"
        },
        {
            "name": "Partial match - extend both",
            "req_start": "2019-12-20",
            "req_end": "2021-01-10",
            "expected_type": "partial"
        },
        {
            "name": "No match",
            "req_start": "2018-01-01",
            "req_end": "2018-12-31",
            "expected_type": None
        }
    ]

    # Find actual cache file
    cache_files = sorted(Path("output/raw/prices").glob("daily_quotes_*.parquet"))
    if not cache_files:
        logger.warning("⚠️ No cache files found - creating dummy test scenarios")
        return False

    latest_cache = cache_files[-1]
    logger.info(f"Using cache file: {latest_cache.name}")

    # Run tests
    passed = 0
    failed = 0

    for case in test_cases:
        logger.info(f"\nTest: {case['name']}")
        logger.info(f"  Request: {case['req_start']} to {case['req_end']}")

        result = _find_latest_with_date_range(
            "daily_quotes_*.parquet",
            case["req_start"],
            case["req_end"]
        )

        if case["expected_type"] is None:
            if result is None:
                logger.info("  ✅ PASS - No match found as expected")
                passed += 1
            else:
                logger.error(f"  ❌ FAIL - Expected no match, got {result['match_type']}")
                failed += 1
        else:
            if result and result["match_type"] == case["expected_type"]:
                logger.info(f"  ✅ PASS - {result['match_type']} match detected")
                if result["match_type"] == "partial":
                    logger.info(f"     Coverage: {result.get('coverage', 0)*100:.1f}%")
                    missing_ranges = result.get("missing_ranges") or []
                    if missing_ranges:
                        for start, end in missing_ranges:
                            logger.info(f"     Missing range: {start} to {end}")
                    else:
                        logger.info(f"     Missing range: {result.get('missing_start')} to {result.get('missing_end')}")
                passed += 1
            else:
                logger.error(f"  ❌ FAIL - Expected {case['expected_type']}, got {result['match_type'] if result else 'None'}")
                failed += 1

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Helper Function Tests: {passed} passed, {failed} failed")
    logger.info(f"{'=' * 80}\n")

    return failed == 0


async def test_daily_quotes_partial_fetch(session, id_token):
    """Test partial fetch for daily quotes."""
    from scripts.pipelines.run_pipeline_v4_optimized import JQuantsAsyncFetcher
    from components.trading_calendar_fetcher import TradingCalendarFetcher

    logger.info("=" * 80)
    logger.info("TEST 2: Daily Quotes - Partial Fetch and Merge")
    logger.info("=" * 80)

    # Find existing cache to determine test range
    cache_files = sorted(Path("output/raw/prices").glob("daily_quotes_*.parquet"))
    if not cache_files:
        logger.warning("⚠️ No cache files found - skipping integration test")
        return False

    import re
    latest_cache = cache_files[-1]
    match = re.search(r"_(\d{8})_(\d{8})\.parquet$", latest_cache.name)
    if not match:
        logger.error("❌ Could not parse cache filename")
        return False

    cache_end_dt = datetime.strptime(match.group(2), "%Y%m%d")

    # Request range: from 10 days before cache end to 5 days after
    test_start = (cache_end_dt - timedelta(days=10)).strftime("%Y-%m-%d")
    test_end = (cache_end_dt + timedelta(days=5)).strftime("%Y-%m-%d")

    logger.info(f"Cache ends: {cache_end_dt.strftime('%Y-%m-%d')}")
    logger.info(f"Test range: {test_start} to {test_end}")
    logger.info(f"Expected: Use cache for first part, fetch only last 5 days\n")

    try:
        # Setup fetcher
        api_client = type('obj', (object,), {'id_token': id_token})()
        fetcher = JQuantsAsyncFetcher(api_client)

        # Get trading calendar
        calendar_fetcher = TradingCalendarFetcher(api_client)
        calendar_data = await calendar_fetcher.get_trading_calendar(test_start, test_end, session)
        business_days = calendar_data.get("business_days", [])

        if not business_days:
            logger.error("❌ No business days found")
            return False

        logger.info(f"Business days in test range: {len(business_days)}")

        # Fetch with partial cache
        logger.info("\nFetching daily quotes...")
        result_df = await fetcher.fetch_daily_quotes_optimized(
            business_days=business_days,
            target_codes=None,
            use_cache=True
        )

        if result_df.is_empty():
            logger.error("❌ No data returned")
            return False

        logger.info(f"\n✅ Fetch completed: {len(result_df):,} records")

        # Verify data coverage
        import polars as pl
        date_range = result_df.select([
            pl.col("Date").min().alias("min_date"),
            pl.col("Date").max().alias("max_date"),
            pl.col("Date").n_unique().alias("unique_dates")
        ]).to_dicts()[0]

        logger.info(f"Date coverage: {date_range['min_date']} to {date_range['max_date']}")
        logger.info(f"Unique dates: {date_range['unique_dates']}")

        # Check tracker stats
        tracker_stats = fetcher.tracker.get_stats()
        logger.info(f"\nTracker stats:")
        logger.info(f"  API calls: {tracker_stats.get('api_calls', 0)}")
        logger.info(f"  Cache hits: {tracker_stats.get('cache_hits', 0)}")
        logger.info(f"  Cache misses: {tracker_stats.get('cache_misses', 0)}")

        # Verify extended cache was saved
        extended_cache = Path("output/raw/prices") / f"daily_quotes_{test_start.replace('-', '')}_{test_end.replace('-', '')}.parquet"
        if extended_cache.exists():
            logger.info(f"\n✅ Extended cache saved: {extended_cache.name}")
            cache_size_mb = extended_cache.stat().st_size / (1024 * 1024)
            logger.info(f"   Size: {cache_size_mb:.1f} MB")
        else:
            logger.warning(f"\n⚠️ Extended cache not found: {extended_cache.name}")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_statements_partial_fetch(session, id_token):
    """Test partial fetch for statements."""
    from scripts.pipelines.run_pipeline_v4_optimized import JQuantsAsyncFetcher
    from components.trading_calendar_fetcher import TradingCalendarFetcher

    logger.info("=" * 80)
    logger.info("TEST 3: Statements - Partial Fetch and Merge")
    logger.info("=" * 80)

    # Find existing cache
    cache_files = sorted(Path("output/raw/statements").glob("event_raw_statements_*.parquet"))
    if not cache_files:
        logger.warning("⚠️ No statements cache found - skipping test")
        return False

    import re
    latest_cache = cache_files[-1]
    match = re.search(r"_(\d{8})_(\d{8})\.parquet$", latest_cache.name)
    if not match:
        logger.error("❌ Could not parse cache filename")
        return False

    cache_end_dt = datetime.strptime(match.group(2), "%Y%m%d")

    # Request smaller range for statements (they're slower to fetch)
    test_start = (cache_end_dt - timedelta(days=5)).strftime("%Y-%m-%d")
    test_end = (cache_end_dt + timedelta(days=3)).strftime("%Y-%m-%d")

    logger.info(f"Cache ends: {cache_end_dt.strftime('%Y-%m-%d')}")
    logger.info(f"Test range: {test_start} to {test_end}")

    try:
        api_client = type('obj', (object,), {'id_token': id_token})()
        fetcher = JQuantsAsyncFetcher(api_client)

        # Get business days
        calendar_fetcher = TradingCalendarFetcher(api_client)
        calendar_data = await calendar_fetcher.get_trading_calendar(test_start, test_end, session)
        business_days = calendar_data.get("business_days", [])

        logger.info(f"Business days: {len(business_days)}")

        # Fetch statements
        logger.info("\nFetching statements...")
        result_df = await fetcher.fetch_statements_by_date(
            business_days=business_days,
            use_cache=True
        )

        logger.info(f"✅ Fetch completed: {len(result_df):,} records")

        # Check extended cache
        extended_cache = Path("output/raw/statements") / f"event_raw_statements_{test_start.replace('-', '')}_{test_end.replace('-', '')}.parquet"
        if extended_cache.exists():
            logger.info(f"✅ Extended cache saved: {extended_cache.name}")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def test_topix_partial_fetch(session, id_token):
    """Test partial fetch for TOPIX."""
    from scripts.pipelines.run_pipeline_v4_optimized import JQuantsAsyncFetcher

    logger.info("=" * 80)
    logger.info("TEST 4: TOPIX - Partial Fetch and Merge")
    logger.info("=" * 80)

    # Find existing cache
    cache_files = sorted(Path("output/raw/indices").glob("topix_history_*.parquet"))
    if not cache_files:
        logger.warning("⚠️ No TOPIX cache found - skipping test")
        return False

    import re
    latest_cache = cache_files[-1]
    match = re.search(r"_(\d{8})_(\d{8})\.parquet$", latest_cache.name)
    if not match:
        logger.error("❌ Could not parse cache filename")
        return False

    cache_end_dt = datetime.strptime(match.group(2), "%Y%m%d")

    # Request range extending 3 days past cache
    test_start = (cache_end_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    test_end = (cache_end_dt + timedelta(days=3)).strftime("%Y-%m-%d")

    logger.info(f"Cache ends: {cache_end_dt.strftime('%Y-%m-%d')}")
    logger.info(f"Test range: {test_start} to {test_end}")

    try:
        api_client = type('obj', (object,), {'id_token': id_token})()
        fetcher = JQuantsAsyncFetcher(api_client)

        # Fetch TOPIX
        logger.info("\nFetching TOPIX...")
        result_df = await fetcher.fetch_topix_data(
            from_date=test_start,
            to_date=test_end,
            use_cache=True
        )

        if result_df.is_empty():
            logger.error("❌ No data returned")
            return False

        logger.info(f"✅ Fetch completed: {len(result_df):,} records")

        # Check extended cache
        extended_cache = Path("output/raw/indices") / f"topix_history_{test_start.replace('-', '')}_{test_end.replace('-', '')}.parquet"
        if extended_cache.exists():
            logger.info(f"✅ Extended cache saved: {extended_cache.name}")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("Phase 2 Partial Match Cache - Test Suite")
    logger.info("=" * 80)
    logger.info("")

    # Get credentials
    email = os.getenv("JQUANTS_AUTH_EMAIL", "")
    password = os.getenv("JQUANTS_AUTH_PASSWORD", "")

    if not email or not password:
        logger.error("❌ JQuants credentials not found in .env")
        return 1

    results = {}

    # Test 1: Helper function (no API needed)
    results["helper"] = await test_helper_function()

    # Authenticate for integration tests
    try:
        async with aiohttp.ClientSession() as session:
            # Auth step 1: Get refresh token
            base_url = "https://api.jquants.com/v1"
            auth_url = f"{base_url}/token/auth_user"
            auth_payload = {"mailaddress": email, "password": password}

            async with session.post(auth_url, json=auth_payload) as response:
                if response.status != 200:
                    logger.error(f"❌ Authentication failed: {response.status}")
                    return 1
                data = await response.json()
                refresh_token = data["refreshToken"]

            # Auth step 2: Get ID token
            refresh_url = f"{base_url}/token/auth_refresh"
            params = {"refreshtoken": refresh_token}

            async with session.post(refresh_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"❌ Failed to get ID token: {response.status}")
                    return 1
                data = await response.json()
                id_token = data["idToken"]

            logger.info("✅ Authentication successful\n")

            # Run integration tests
            results["daily_quotes"] = await test_daily_quotes_partial_fetch(session, id_token)
            results["statements"] = await test_statements_partial_fetch(session, id_token)
            results["topix"] = await test_topix_partial_fetch(session, id_token)

    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for r in results.values() if r)
    failed = sum(1 for r in results.values() if not r)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")

    logger.info("=" * 80)
    logger.info(f"Total: {passed} passed, {failed} failed")
    logger.info("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
