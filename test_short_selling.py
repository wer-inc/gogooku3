#!/usr/bin/env python3
"""Test short selling data fetch and save"""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_short_selling():
    """Test short selling data fetch"""
    import os

    from dotenv import load_dotenv

    from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher
    from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs

    # Load environment variables
    load_dotenv()
    email = os.getenv("JQUANTS_AUTH_EMAIL")
    password = os.getenv("JQUANTS_AUTH_PASSWORD")

    if not email or not password:
        raise ValueError(
            "JQUANTS_AUTH_EMAIL and JQUANTS_AUTH_PASSWORD must be set in .env"
        )

    # Initialize fetcher
    fetcher = JQuantsAsyncFetcher(email=email, password=password)

    # Test date range (last 30 days for quick test)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    logger.info(f"Testing short selling fetch from {start_str} to {end_str}")

    try:
        # Create session
        connector = aiohttp.TCPConnector(limit=75, limit_per_host=75)
        timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_read=60)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            # Authenticate first
            logger.info("Authenticating with JQuants API...")
            await fetcher.authenticate(session)
            logger.info("✅ Authenticated successfully")

            # Fetch short selling data
            logger.info("Fetching short selling ratio data...")
            short_selling_df = await fetcher.get_short_selling(
                session, start_str, end_str
            )

            logger.info(f"Retrieved {len(short_selling_df)} short selling records")
            logger.info(f"Columns: {short_selling_df.columns}")

            if not short_selling_df.is_empty():
                # Try to save
                outdir = Path("output/raw/short_selling")
                outdir.mkdir(parents=True, exist_ok=True)

                output_path = (
                    outdir
                    / f"short_selling_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
                )
                logger.info(f"Saving to {output_path}...")

                save_parquet_with_gcs(short_selling_df, output_path, auto_sync=False)
                logger.info(f"✅ Successfully saved to {output_path}")
            else:
                logger.warning("⚠️  No data retrieved")

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(test_short_selling())
