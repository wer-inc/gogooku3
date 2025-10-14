#!/usr/bin/env python3
"""Daily Cache Update Script

Pre-populates data caches every morning to ensure 100% cache hit rate
during daytime hours. Designed for cron execution (silent mode).

Usage:
    python scripts/cache/update_daily_cache.py                    # Verbose mode
    python scripts/cache/update_daily_cache.py --silent           # Silent mode (cron)
    make update-cache-silent                                      # Via Makefile

Cron setup:
    0 8 * * * cd /root/gogooku3 && make update-cache-silent >> /var/log/gogooku3_cache.log 2>&1

Updates three data sources:
    1. Daily Quotes (prices)       - 2015-10-16 to today
    2. Statements (financial data) - 2015-10-13 to today
    3. TOPIX (index data)          - 2018-11-07 to today
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

# Import required components
sys.path.append(str(Path(__file__).parents[1]))
from components.trading_calendar_fetcher import TradingCalendarFetcher
from components.market_code_filter import MarketCodeFilter

# Performance tracker (simplified for cache updates)
class SimpleCacheTracker:
    """Simplified performance tracker for cache updates"""

    def __init__(self):
        self.stats = {
            "daily_quotes": {"status": "pending", "time": 0, "records": 0},
            "statements": {"status": "pending", "time": 0, "records": 0},
            "topix": {"status": "pending", "time": 0, "records": 0}
        }
        self.start_time = None

    def start(self, source: str):
        self.start_time = datetime.now()

    def end(self, source: str, records: int = 0, error: str = None):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.stats[source]["time"] = elapsed
        self.stats[source]["records"] = records
        self.stats[source]["status"] = "error" if error else "success"
        if error:
            self.stats[source]["error"] = error

    def get_summary(self) -> str:
        """Get summary report"""
        lines = ["Cache Update Summary:"]
        for source, stats in self.stats.items():
            status_icon = "✅" if stats["status"] == "success" else "❌" if stats["status"] == "error" else "⏳"
            lines.append(f"  {status_icon} {source}: {stats['status']} ({stats['time']:.1f}s, {stats['records']:,} records)")
            if "error" in stats:
                lines.append(f"      Error: {stats['error']}")
        return "\n".join(lines)


async def fetch_and_cache_daily_quotes(session, id_token, business_days, target_codes, semaphore, tracker, logger):
    """Fetch daily quotes and save to cache"""
    import polars as pl

    tracker.start("daily_quotes")

    try:
        # Fetch data using date-axis (most reliable for full range)
        all_quotes = []
        batch_size = 30
        base_url = "https://api.jquants.com/v1"

        for i in range(0, len(business_days), batch_size):
            batch_days = business_days[i:i+batch_size]

            tasks = []
            for date in batch_days:
                date_api = date.replace("-", "")
                task = _fetch_daily_quotes_for_date(session, base_url, id_token, date, date_api, semaphore)
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            for df in results:
                if not df.is_empty():
                    all_quotes.append(df)

            if not logger.silent:
                logger.info(f"  Daily Quotes: {min(i+batch_size, len(business_days))}/{len(business_days)} days")

        if not all_quotes:
            tracker.end("daily_quotes", 0, "No data fetched")
            return

        # Combine and filter
        price_df = pl.concat(all_quotes)

        # Apply market filter if target_codes provided
        if target_codes and "Code" in price_df.columns:
            price_df = price_df.filter(pl.col("Code").is_in(target_codes))

        # Normalize column names
        if "code" in price_df.columns and "Code" not in price_df.columns:
            price_df = price_df.rename({"code": "Code"})

        # Save to cache
        from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
        start_date = business_days[0]
        end_date = business_days[-1]
        cache_dir = Path("output/raw/prices")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"daily_quotes_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"

        save_parquet_with_gcs(price_df, cache_path, auto_sync=True)

        tracker.end("daily_quotes", len(price_df))

        if not logger.silent:
            logger.info(f"✅ Daily Quotes cached: {len(price_df):,} records → {cache_path.name}")

    except Exception as e:
        tracker.end("daily_quotes", 0, str(e))
        logger.error(f"❌ Daily Quotes failed: {e}")


async def _fetch_daily_quotes_for_date(session, base_url, id_token, date, date_api, semaphore):
    """Fetch daily quotes for a specific date"""
    import polars as pl

    url = f"{base_url}/prices/daily_quotes"
    headers = {"Authorization": f"Bearer {id_token}"}

    all_quotes = []
    pagination_key = None

    while True:
        params = {"date": date_api}
        if pagination_key:
            params["pagination_key"] = pagination_key

        try:
            async with semaphore:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        break

                    data = await response.json()
                    quotes = data.get("daily_quotes", [])

                    if quotes:
                        all_quotes.extend(quotes)

                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break

        except Exception:
            break

    if all_quotes:
        return pl.DataFrame(all_quotes)

    return pl.DataFrame()


async def fetch_and_cache_statements(session, id_token, business_days, semaphore, tracker, logger):
    """Fetch statements and save to cache"""
    import polars as pl

    tracker.start("statements")

    try:
        base_url = "https://api.jquants.com/v1"
        url = f"{base_url}/fins/statements"
        headers = {"Authorization": f"Bearer {id_token}"}

        all_statements = []

        for i, date in enumerate(business_days):
            date_api = date.replace("-", "")
            params = {"date": date_api}
            pagination_key = None
            statements_for_date = []

            while True:
                if pagination_key:
                    params["pagination_key"] = pagination_key

                try:
                    async with semaphore:
                        async with session.get(url, headers=headers, params=params) as response:
                            if response.status == 404:
                                break
                            elif response.status != 200:
                                break

                            data = await response.json()
                            statements = data.get("statements", [])

                            if statements:
                                statements_for_date.extend(statements)

                            pagination_key = data.get("pagination_key")
                            if not pagination_key:
                                break

                except Exception:
                    break

            if statements_for_date:
                all_statements.extend(statements_for_date)

            if not logger.silent and (i + 1) % 50 == 0:
                logger.info(f"  Statements: {i+1}/{len(business_days)} days")

        if not all_statements:
            tracker.end("statements", 0, "No data fetched")
            return

        statements_df = pl.DataFrame(all_statements)

        # Save to cache
        from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
        start_date = business_days[0]
        end_date = business_days[-1]
        cache_dir = Path("output/raw/statements")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"event_raw_statements_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"

        save_parquet_with_gcs(statements_df, cache_path, auto_sync=True)

        tracker.end("statements", len(statements_df))

        if not logger.silent:
            logger.info(f"✅ Statements cached: {len(statements_df):,} records → {cache_path.name}")

    except Exception as e:
        tracker.end("statements", 0, str(e))
        logger.error(f"❌ Statements failed: {e}")


async def fetch_and_cache_topix(session, id_token, from_date, to_date, tracker, logger):
    """Fetch TOPIX data and save to cache"""
    import polars as pl

    tracker.start("topix")

    try:
        base_url = "https://api.jquants.com/v1"
        url = f"{base_url}/indices/topix"
        headers = {"Authorization": f"Bearer {id_token}"}

        from_api = from_date.replace("-", "")
        to_api = to_date.replace("-", "")

        all_data = []
        pagination_key = None

        while True:
            params = {"from": from_api, "to": to_api}
            if pagination_key:
                params["pagination_key"] = pagination_key

            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        break

                    data = await response.json()
                    topix_data = data.get("topix", [])

                    if topix_data:
                        all_data.extend(topix_data)

                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break

            except Exception:
                break

        if not all_data:
            tracker.end("topix", 0, "No data fetched")
            return

        topix_df = pl.DataFrame(all_data)

        # Normalize dtypes
        if "Date" in topix_df.columns:
            topix_df = topix_df.with_columns(
                pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            )
        for c in ("Open", "High", "Low", "Close", "Volume"):
            if c in topix_df.columns:
                topix_df = topix_df.with_columns(pl.col(c).cast(pl.Float64, strict=False))
        topix_df = topix_df.sort("Date")

        # Save to cache
        from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
        cache_dir = Path("output/raw/indices")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"topix_history_{from_date.replace('-', '')}_{to_date.replace('-', '')}.parquet"

        save_parquet_with_gcs(topix_df, cache_path, auto_sync=True)

        tracker.end("topix", len(topix_df))

        if not logger.silent:
            logger.info(f"✅ TOPIX cached: {len(topix_df):,} records → {cache_path.name}")

    except Exception as e:
        tracker.end("topix", 0, str(e))
        logger.error(f"❌ TOPIX failed: {e}")


async def main():
    """Main cache update function"""
    import argparse

    parser = argparse.ArgumentParser(description="Update daily data caches")
    parser.add_argument("--silent", action="store_true", help="Silent mode (for cron)")
    args = parser.parse_args()

    # Setup logging
    class SilentLogger:
        def __init__(self, silent=False):
            self.silent = silent
            self.logger = logging.getLogger(__name__)
            if not silent:
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s"
                )

        def info(self, msg):
            if not self.silent:
                self.logger.info(msg)

        def error(self, msg):
            self.logger.error(msg)  # Always log errors

        def warning(self, msg):
            if not self.silent:
                self.logger.warning(msg)

    logger = SilentLogger(silent=args.silent)
    tracker = SimpleCacheTracker()

    # Get credentials
    email = os.getenv("JQUANTS_AUTH_EMAIL", "")
    password = os.getenv("JQUANTS_AUTH_PASSWORD", "")

    if not email or not password:
        logger.error("❌ JQuants credentials not found in .env")
        return 1

    logger.info("=" * 60)
    logger.info("Daily Cache Update - Starting")
    logger.info("=" * 60)

    try:
        async with aiohttp.ClientSession() as session:
            # Authenticate
            logger.info("Authenticating with JQuants API...")
            base_url = "https://api.jquants.com/v1"

            # Auth step 1: Get refresh token
            auth_url = f"{base_url}/token/auth_user"
            auth_payload = {"mailaddress": email, "password": password}

            async with session.post(auth_url, json=auth_payload) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"❌ Authentication failed: {response.status} - {text}")
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

            logger.info("✅ Authentication successful")

            # Setup semaphore for concurrent requests
            max_concurrent = int(os.getenv("MAX_CONCURRENT_FETCH", "75"))
            semaphore = asyncio.Semaphore(max_concurrent)

            # Calculate date ranges
            today = datetime.now().strftime("%Y-%m-%d")

            # Determine subscription start date
            subscription_start_str = os.getenv("JQUANTS_SUBSCRIPTION_START")
            if subscription_start_str:
                subscription_start = subscription_start_str
                logger.info(f"Using explicit subscription start: {subscription_start}")
            else:
                # Rolling contract
                contract_years = int(os.getenv("JQUANTS_CONTRACT_YEARS", "10"))
                subscription_start_dt = datetime.now() - timedelta(days=365 * contract_years + 2)
                subscription_start = subscription_start_dt.strftime("%Y-%m-%d")
                logger.info(f"Using rolling {contract_years}-year contract: {subscription_start}")

            # Get trading calendar
            logger.info(f"Fetching trading calendar ({subscription_start} to {today})...")
            api_client = type('obj', (object,), {'id_token': id_token})()
            calendar_fetcher = TradingCalendarFetcher(api_client)
            calendar_data = await calendar_fetcher.get_trading_calendar(subscription_start, today, session)
            business_days = calendar_data.get("business_days", [])

            if not business_days:
                logger.error("❌ No business days found")
                return 1

            logger.info(f"✅ Trading calendar: {len(business_days)} business days")

            # Get target market codes (for filtering)
            target_codes = set(MarketCodeFilter.TARGET_MARKET_CODES)
            logger.info(f"Target market codes: {len(target_codes)} codes")

            # Update caches in parallel
            logger.info("\nUpdating caches...")
            await asyncio.gather(
                fetch_and_cache_daily_quotes(session, id_token, business_days, target_codes, semaphore, tracker, logger),
                fetch_and_cache_statements(session, id_token, business_days, semaphore, tracker, logger),
                fetch_and_cache_topix(session, id_token, subscription_start, today, tracker, logger)
            )

    except Exception as e:
        logger.error(f"❌ Cache update failed: {e}")
        return 1

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info(tracker.get_summary())
    logger.info("=" * 60)

    # Check if all succeeded
    all_success = all(s["status"] == "success" for s in tracker.stats.values())
    if all_success:
        logger.info("✅ Daily cache update completed successfully")
        return 0
    else:
        logger.error("❌ Some cache updates failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
