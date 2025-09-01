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
from data.ml_dataset_builder import MLDatasetBuilder  # noqa: E402
from components.trading_calendar_fetcher import TradingCalendarFetcher  # noqa: E402
from components.market_code_filter import MarketCodeFilter  # noqa: E402
from components.daily_stock_fetcher import DailyStockFetcher  # noqa: E402

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
        # 有料プラン向け設定
        self.max_concurrent = max_concurrent or int(
            os.getenv("MAX_CONCURRENT_FETCH", 75)  # 有料プラン向け
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

    async def get_listed_info(self, session: aiohttp.ClientSession, date: Optional[str] = None) -> pl.DataFrame:
        """Get listed company information with Market Code filtering."""
        url = f"{self.base_url}/listed/info"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        params = {}
        if date:
            params["date"] = date

        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                df = pl.DataFrame(data.get("info", []))
                # Market Codeフィルタリング（8市場のみ）
                if not df.is_empty():
                    df = MarketCodeFilter.filter_stocks(df)
                return df
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
    """Complete pipeline from JQuants to ML dataset with Trading Calendar and Market Code filtering."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(
            "/home/ubuntu/gogooku3-standalone/output"
        )
        self.output_dir.mkdir(exist_ok=True)
        self.fetcher = None
        self.calendar_fetcher = None
        self.daily_stock_fetcher = None
        self.builder = MLDatasetBuilder(output_dir=self.output_dir)

    async def fetch_jquants_data(
        self, start_date: str = None, end_date: str = None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Fetch data from JQuants API using Trading Calendar."""
        # Get credentials
        email = os.getenv("JQUANTS_AUTH_EMAIL", "")
        password = os.getenv("JQUANTS_AUTH_PASSWORD", "")

        if not email or not password:
            logger.error("JQuants credentials not found in environment")
            logger.info("Please set JQUANTS_AUTH_EMAIL and JQUANTS_AUTH_PASSWORD")
            return pl.DataFrame(), pl.DataFrame()

        # Initialize fetchers
        self.fetcher = JQuantsAsyncFetcher(email, password)
        self.calendar_fetcher = TradingCalendarFetcher(self.fetcher)
        self.daily_stock_fetcher = DailyStockFetcher(self.fetcher, MarketCodeFilter())

        # Date range (default: 4 years)
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = "2021-01-01"

        async with aiohttp.ClientSession() as session:
            # Authenticate
            await self.fetcher.authenticate(session)
            
            # Step 1: 営業日カレンダー取得
            logger.info(f"Step 1: 営業日カレンダー取得中 ({start_date} - {end_date})...")
            calendar_data = await self.calendar_fetcher.get_trading_calendar(
                start_date, end_date, session
            )
            business_days = calendar_data.get("business_days", [])
            logger.info(f"✅ 営業日数: {len(business_days)}日")
            
            if not business_days:
                logger.error("営業日データが取得できませんでした")
                return pl.DataFrame(), pl.DataFrame()

            # Step 2: 営業日ごとの銘柄リスト取得
            logger.info("Step 2: 営業日ごとの銘柄リスト取得中...")
            daily_listings = await self.daily_stock_fetcher.get_stocks_for_period(
                business_days,  # 全営業日分を取得
                session
            )
            
            if not daily_listings:
                logger.error("銘柄情報が取得できませんでした")
                return pl.DataFrame(), pl.DataFrame()
            
            # 統計情報表示
            stats = self.daily_stock_fetcher.get_statistics(daily_listings)
            logger.info(f"✅ 期間中のユニーク銘柄数: {stats['total_unique_stocks']}")
            logger.info(f"✅ 日次平均銘柄数: {stats['avg_daily_stocks']:.0f}")
            
            # Step 3: 営業日ごとに価格データ取得
            logger.info(f"Step 3: 価格データ取得中...")
            all_price_data = []
            total_stocks = sum(len(df) for df in daily_listings.values())
            processed_stocks = 0
            
            for idx, (date, stocks_df) in enumerate(daily_listings.items(), 1):
                logger.info(f"  [{idx}/{len(daily_listings)}] {date}: {len(stocks_df)}銘柄の価格データ取得中...")
                codes = stocks_df["Code"].to_list()
                
                # その日の価格データ取得
                daily_prices = await self.fetcher.fetch_all_prices(
                    session,
                    codes,  # 全銘柄を取得
                    date,
                    date,
                )
                
                if not daily_prices.is_empty():
                    all_price_data.append(daily_prices)
                    processed_stocks += len(stocks_df)
                    logger.info(f"    ✓ {len(daily_prices)}レコード取得 (進捗: {processed_stocks}/{total_stocks}銘柄)")
                
                # メモリ管理: 定期的にデータを保存
                if len(all_price_data) >= 30:  # 30日分ごとに中間保存
                    temp_df = pl.concat(all_price_data)
                    logger.info(f"    中間データ: {len(temp_df)}レコード")
            
            # 全データ結合
            if all_price_data:
                price_df = pl.concat(all_price_data)
                logger.info(f"✅ 価格データ取得完了: {len(price_df)}レコード")
            else:
                price_df = pl.DataFrame()

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
            
            # Step 4: 営業日でフィルタリング
            logger.info("Step 4: 営業日フィルタリング中...")
            business_days_set = set(business_days)
            original_count = len(price_df)
            
            # Date列を文字列に変換してフィルタリング
            if "Date" in price_df.columns:
                price_df = price_df.with_columns(
                    pl.col("Date").cast(pl.Utf8)
                )
                price_df = price_df.filter(
                    pl.col("Date").is_in(business_days)
                )
            
            filtered_count = len(price_df)
            logger.info(f"✅ 営業日フィルタリング: {original_count} → {filtered_count} レコード")

            # Step 5: TOPIXデータ取得
            logger.info("Step 5: TOPIXデータ取得中...")
            topix_df = await self.fetcher.fetch_topix_data(
                session, start_date, end_date
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
        self, use_jquants: bool = True, start_date: str = None, end_date: str = None
    ):
        """Run the complete pipeline with Trading Calendar."""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("COMPLETE ML DATASET PIPELINE with Trading Calendar")
        logger.info("=" * 60)

        # Step 1: Get data
        topix_df = None
        if use_jquants:
            logger.info("Fetching data from JQuants API...")
            price_df, topix_df = await self.fetch_jquants_data(start_date, end_date)

            if price_df.is_empty():
                logger.warning("JQuants fetch failed, using sample data instead")
                from data.ml_dataset_builder import create_sample_data

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
        "--start-date",
        type=str,
        default="2021-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: today)",
    )

    args = parser.parse_args()

    # Set end date to today if not specified
    if args.end_date is None:
        args.end_date = datetime.now().strftime("%Y-%m-%d")

    # Check environment
    if args.jquants:
        if not os.getenv("JQUANTS_AUTH_EMAIL"):
            logger.error("JQuants credentials not found in .env file")
            logger.error("Please check /home/ubuntu/gogooku3-standalone/.env")
            logger.info("\nUsing sample data instead...")
            args.jquants = False
        else:
            logger.info(
                f"Using JQuants API with account: {os.getenv('JQUANTS_AUTH_EMAIL')[:10]}..."
            )

    # Run pipeline
    pipeline = JQuantsPipeline()
    df, metadata = await pipeline.run(
        use_jquants=args.jquants, start_date=args.start_date, end_date=args.end_date
    )

    return df, metadata


if __name__ == "__main__":
    # Run the async main function
    df, metadata = asyncio.run(main())
