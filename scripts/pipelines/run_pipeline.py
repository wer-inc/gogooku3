#!/usr/bin/env python3
"""
Complete ML Dataset Pipeline with JQuants Integration
Fetches real data and applies all bug fixes from P0-P2
"""

import os
import sys
import asyncio
import aiohttp
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    # Don't log before logging is configured

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from core.ml_dataset_builder import MLDatasetBuilder  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JQuantsAsyncFetcher:
    """Asynchronous JQuants API fetcher with high concurrency."""

    def __init__(self, email: str, password: str, max_concurrent: int = None):
        self.email = email
        self.password = password
        self.base_url = "https://api.jquants.com/v1"
        self.id_token = None
        # Use environment variable or default
        self.max_concurrent = max_concurrent or int(
            os.getenv("MAX_CONCURRENT_FETCH", 150)
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    async def authenticate(self, session: aiohttp.ClientSession):
        """Authenticate with JQuants API."""
        # Get refresh token
        auth_url = f"{self.base_url}/token/auth_user"
        auth_payload = {"mailaddress": self.email, "password": self.password}

        async with session.post(auth_url, json=auth_payload) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Auth failed: {response.status} - {text}")
            data = await response.json()
            refresh_token = data["refreshToken"]

        # Get ID token
        refresh_url = f"{self.base_url}/token/auth_refresh"
        params = {"refreshtoken": refresh_token}

        async with session.post(refresh_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Failed to get ID token: {response.status}")
            data = await response.json()
            self.id_token = data["idToken"]

        logger.info("✅ JQuants authentication successful")

    async def get_listed_info(self, session: aiohttp.ClientSession) -> pl.DataFrame:
        """Get listed company information."""
        url = f"{self.base_url}/listed/info"
        headers = {"Authorization": f"Bearer {self.id_token}"}

        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return pl.DataFrame(data.get("info", []))
            return pl.DataFrame()

    async def fetch_price_batch(
        self, session: aiohttp.ClientSession, code: str, from_date: str, to_date: str
    ) -> Optional[pl.DataFrame]:
        """Fetch price data for a single stock."""
        async with self.semaphore:
            url = f"{self.base_url}/prices/daily_quotes"
            headers = {"Authorization": f"Bearer {self.id_token}"}
            params = {"code": code, "from": from_date, "to": to_date}

            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        quotes = data.get("daily_quotes", [])
                        if quotes:
                            df = pl.DataFrame(quotes)
                            df = df.with_columns(pl.lit(code).alias("Code"))
                            return df
            except Exception as e:
                logger.warning(f"Failed to fetch {code}: {e}")

            return None

    async def fetch_all_prices(
        self,
        session: aiohttp.ClientSession,
        codes: List[str],
        from_date: str,
        to_date: str,
    ) -> pl.DataFrame:
        """Fetch price data for all stocks concurrently."""
        logger.info(f"Fetching price data for {len(codes)} stocks...")

        tasks = [
            self.fetch_price_batch(session, code, from_date, to_date) for code in codes
        ]

        results = await asyncio.gather(*tasks)

        # Filter out None results and concatenate
        valid_results = [df for df in results if df is not None and not df.is_empty()]

        if valid_results:
            combined = pl.concat(valid_results, how="vertical")
            logger.info(
                f"✅ Fetched {len(combined)} price records for {combined['Code'].n_unique()} stocks"
            )
            return combined

        return pl.DataFrame()

    async def get_trades_spec(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Get trades specification data."""
        url = f"{self.base_url}/markets/trades_spec"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        params = {"from": from_date, "to": to_date}

        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                trades = data.get("trades_spec", [])
                if trades:
                    return pl.DataFrame(trades)
            return pl.DataFrame()

    async def fetch_topix_data(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch TOPIX index data."""
        url = f"{self.base_url}/indices/topix"
        headers = {"Authorization": f"Bearer {self.id_token}"}

        all_data = []
        pagination_key = None

        while True:
            params = {"from": from_date, "to": to_date}
            if pagination_key:
                params["pagination_key"] = pagination_key

            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch TOPIX: {response.status}")
                        break

                    data = await response.json()
                    topix_data = data.get("topix", [])

                    if topix_data:
                        all_data.extend(topix_data)

                    # Check for pagination
                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break

            except Exception as e:
                logger.error(f"Error fetching TOPIX: {e}")
                break

        if all_data:
            df = pl.DataFrame(all_data)
            # Ensure proper column names and types
            if "Date" in df.columns:
                df = df.with_columns(
                    pl.col("Date").str.strptime(
                        pl.Date, format="%Y-%m-%d", strict=False
                    )
                )
            if "Close" in df.columns:
                df = df.with_columns(pl.col("Close").cast(pl.Float64))

            logger.info(f"✅ Fetched {len(df)} TOPIX records")
            return df.sort("Date")

        return pl.DataFrame()


class JQuantsPipeline:
    """Complete pipeline from JQuants to ML dataset."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(
            "/home/ubuntu/gogooku2/apps/gogooku3/output"
        )
        self.output_dir.mkdir(exist_ok=True)
        self.fetcher = None
        self.builder = MLDatasetBuilder(output_dir=self.output_dir)

    async def fetch_jquants_data(
        self, n_stocks: int = 100, n_days: int = 300
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Fetch data from JQuants API."""
        # Get credentials
        email = os.getenv("JQUANTS_AUTH_EMAIL", "")
        password = os.getenv("JQUANTS_AUTH_PASSWORD", "")

        if not email or not password:
            logger.error("JQuants credentials not found in environment")
            logger.info("Please set JQUANTS_AUTH_EMAIL and JQUANTS_AUTH_PASSWORD")
            return pl.DataFrame(), pl.DataFrame()

        # Initialize fetcher
        self.fetcher = JQuantsAsyncFetcher(email, password)

        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days)

        async with aiohttp.ClientSession() as session:
            # Authenticate
            await self.fetcher.authenticate(session)

            # Get listed companies
            listed_df = await self.fetcher.get_listed_info(session)

            if listed_df.is_empty():
                logger.error("Failed to fetch listed companies")
                return pl.DataFrame(), pl.DataFrame()

            # Filter TSE stocks (Prime, Standard, Growth)
            tse_codes = ["111", "112", "113"]
            tse_stocks = listed_df.filter(pl.col("MarketCode").is_in(tse_codes))

            if tse_stocks.is_empty():
                tse_stocks = listed_df.head(n_stocks)
            else:
                tse_stocks = tse_stocks.head(n_stocks)

            codes = tse_stocks["Code"].to_list()
            logger.info(f"Selected {len(codes)} stocks: {codes[:5]}...")

            # Fetch price data
            price_df = await self.fetcher.fetch_all_prices(
                session,
                codes,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )

            if price_df.is_empty():
                logger.error("Failed to fetch price data")
                return pl.DataFrame(), pl.DataFrame()

            # Use Adjustment columns preferentially (they are split/dividend adjusted)
            columns_to_rename = {}

            # Always prefer AdjustmentXXX columns if they exist
            if "AdjustmentClose" in price_df.columns:
                columns_to_rename["AdjustmentClose"] = "Close"
                if "Close" in price_df.columns:
                    price_df = price_df.drop("Close")

            if "AdjustmentOpen" in price_df.columns:
                columns_to_rename["AdjustmentOpen"] = "Open"
                if "Open" in price_df.columns:
                    price_df = price_df.drop("Open")

            if "AdjustmentHigh" in price_df.columns:
                columns_to_rename["AdjustmentHigh"] = "High"
                if "High" in price_df.columns:
                    price_df = price_df.drop("High")

            if "AdjustmentLow" in price_df.columns:
                columns_to_rename["AdjustmentLow"] = "Low"
                if "Low" in price_df.columns:
                    price_df = price_df.drop("Low")

            if "AdjustmentVolume" in price_df.columns:
                columns_to_rename["AdjustmentVolume"] = "Volume"
                if "Volume" in price_df.columns:
                    price_df = price_df.drop("Volume")

            if columns_to_rename:
                price_df = price_df.rename(columns_to_rename)
                logger.info(f"Using adjusted columns: {list(columns_to_rename.keys())}")

            # Select required columns
            required_cols = ["Code", "Date", "Open", "High", "Low", "Close", "Volume"]
            available_cols = [col for col in required_cols if col in price_df.columns]
            price_df = price_df.select(available_cols)

            # Ensure numeric types
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in price_df.columns:
                    price_df = price_df.with_columns(pl.col(col).cast(pl.Float64))

            # Sort by Code and Date
            price_df = price_df.sort(["Code", "Date"])

            # Fetch TOPIX data
            topix_df = await self.fetcher.fetch_topix_data(
                session, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )

            return price_df, topix_df

    def process_pipeline(
        self, price_df: pl.DataFrame, topix_df: Optional[pl.DataFrame] = None
    ) -> tuple:
        """Process the complete pipeline with bug fixes."""
        logger.info("=" * 60)
        logger.info("Processing ML Dataset Pipeline")
        logger.info("=" * 60)

        # Apply technical features with all fixes
        df = self.builder.create_technical_features(price_df, topix_df)

        # Add pandas-ta features
        df = self.builder.add_pandas_ta_features(df)

        # Create metadata
        metadata = self.builder.create_metadata(df)

        # Display summary
        logger.info("\nDataset Summary:")
        logger.info(f"  Shape: {len(df)} rows × {len(df.columns)} columns")
        logger.info(f"  Features: {metadata['features']['count']}")
        logger.info(f"  Stocks: {metadata['stocks']}")
        logger.info(
            f"  Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}"
        )

        # Save dataset
        parquet_path, csv_path, meta_path = self.builder.save_dataset(df, metadata)

        return df, metadata, (parquet_path, csv_path, meta_path)

    async def run(
        self, use_jquants: bool = True, n_stocks: int = 100, n_days: int = 300
    ):
        """Run the complete pipeline."""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("COMPLETE ML DATASET PIPELINE")
        logger.info("=" * 60)

        # Step 1: Get data
        topix_df = None
        if use_jquants:
            logger.info("Step 1: Fetching data from JQuants API...")
            price_df, topix_df = await self.fetch_jquants_data(n_stocks, n_days)

            if price_df.is_empty():
                logger.warning("JQuants fetch failed, using sample data instead")
                from core.ml_dataset_builder import create_sample_data

                price_df = create_sample_data(n_stocks, n_days)
                topix_df = None
        else:
            logger.info("Step 1: Creating sample data...")
            from core.ml_dataset_builder import create_sample_data

            price_df = create_sample_data(n_stocks, n_days)

        logger.info(
            f"Data loaded: {len(price_df)} rows, {price_df['Code'].n_unique()} stocks"
        )

        # Step 2: Process pipeline
        logger.info("\nStep 2: Processing ML features...")
        if topix_df is not None and not topix_df.is_empty():
            logger.info(f"  Including TOPIX data: {len(topix_df)} records")
        df, metadata, file_paths = self.process_pipeline(price_df, topix_df)

        # Step 3: Validate results
        logger.info("\nStep 3: Validating results...")
        self.validate_dataset(df, metadata)

        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed:.2f} seconds")
        logger.info(f"Processing speed: {len(df)/elapsed:.0f} rows/second")
        logger.info("\nOutput files:")
        logger.info(f"  Parquet: {file_paths[0]}")
        logger.info(f"  CSV: {file_paths[1]}")
        logger.info(f"  Metadata: {file_paths[2]}")

        return df, metadata

    def validate_dataset(self, df: pl.DataFrame, metadata: dict):
        """Validate the processed dataset."""
        issues = []

        # Check for NaN columns
        nan_cols = []
        for col in df.columns:
            if df[col].is_null().sum() > 0:
                nan_ratio = df[col].is_null().sum() / len(df)
                if nan_ratio > 0.5:  # More than 50% NaN
                    issues.append(f"Column {col} has {nan_ratio:.1%} NaN values")
                nan_cols.append(col)

        # Check feature count
        if metadata["features"]["count"] < 50:
            issues.append(
                f"Only {metadata['features']['count']} features (expected 50+)"
            )

        # Check date range
        if metadata["stocks"] < 10:
            issues.append(f"Only {metadata['stocks']} stocks (expected 10+)")

        if issues:
            logger.warning("Validation issues found:")
            for issue in issues:
                logger.warning(f"  ⚠️  {issue}")
        else:
            logger.info("✅ All validation checks passed")

        # Display fixes applied
        logger.info("\nFixes Applied:")
        for fix in metadata.get("fixes_applied", []):
            logger.info(f"  ✓ {fix}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ML Dataset Pipeline")
    parser.add_argument(
        "--jquants",
        action="store_true",
        help="Use JQuants API (requires credentials in .env)",
    )
    parser.add_argument(
        "--stocks",
        type=int,
        default=None,
        help="Number of stocks to process (default from .env or 100)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days of history (default from .env or 300)",
    )

    args = parser.parse_args()

    # Use environment defaults if not specified
    if args.stocks is None:
        args.stocks = int(os.getenv("DEFAULT_STOCKS", 100))
    if args.days is None:
        args.days = int(os.getenv("DEFAULT_DAYS", 300))

    # Check environment
    if args.jquants:
        if not os.getenv("JQUANTS_AUTH_EMAIL"):
            logger.error("JQuants credentials not found in .env file")
            logger.error("Please check /home/ubuntu/gogooku2/apps/gogooku3/.env")
            logger.info("\nUsing sample data instead...")
            args.jquants = False
        else:
            logger.info(
                f"Using JQuants API with account: {os.getenv('JQUANTS_AUTH_EMAIL')[:10]}..."
            )

    # Run pipeline
    pipeline = JQuantsPipeline()
    df, metadata = await pipeline.run(
        use_jquants=args.jquants, n_stocks=args.stocks, n_days=args.days
    )

    return df, metadata


if __name__ == "__main__":
    # Run the async main function
    df, metadata = asyncio.run(main())
