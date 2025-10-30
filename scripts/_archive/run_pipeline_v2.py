#!/usr/bin/env python3
"""
Complete ML Dataset Pipeline with JQuants Integration V2
正しいデータフロー: daily_quotes と listed_info の内部結合版
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import polars as pl
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from components.market_code_filter import MarketCodeFilter
from components.trading_calendar_fetcher import TradingCalendarFetcher
from data.ml_dataset_builder import MLDatasetBuilder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JQuantsAsyncFetcherV2:
    """Asynchronous JQuants API fetcher with proper data flow."""

    def __init__(self, email: str, password: str, max_concurrent: int = None):
        self.email = email
        self.password = password
        self.base_url = "https://api.jquants.com/v1"
        self.id_token = None
        # 有料プラン向け設定
        self.max_concurrent = max_concurrent or int(
            os.getenv("MAX_CONCURRENT_FETCH", 75)
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

    async def get_listed_info_for_date(
        self, session: aiohttp.ClientSession, date: str
    ) -> pl.DataFrame:
        """Get listed company information for a specific date."""
        url = f"{self.base_url}/listed/info"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        # 日付形式をAPIの要求形式に変換 (YYYY-MM-DD -> YYYYMMDD)
        date_api = date.replace("-", "")
        params = {"date": date_api}

        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    df = pl.DataFrame(data.get("info", []))

                    if not df.is_empty():
                        # Market Codeフィルタリング
                        df = MarketCodeFilter.filter_stocks(df)
                        # 必要な列のみ選択
                        keep_cols = ["Code", "MarketCode", "CompanyName", "Sector17Code", "Sector33Code"]
                        available_cols = [col for col in keep_cols if col in df.columns]
                        df = df.select(available_cols)

                    return df

        except Exception as e:
            logger.warning(f"Failed to fetch listed info for {date}: {e}")

        return pl.DataFrame()

    async def get_daily_quotes_for_date(
        self, session: aiohttp.ClientSession, date: str
    ) -> pl.DataFrame:
        """Get all daily quotes for a specific date."""
        url = f"{self.base_url}/prices/daily_quotes"
        headers = {"Authorization": f"Bearer {self.id_token}"}

        all_quotes = []
        pagination_key = None

        # 日付形式をAPIの要求形式に変換 (YYYY-MM-DD -> YYYYMMDD)
        date_api = date.replace("-", "")

        logger.debug(f"Fetching daily quotes for {date} (API format: {date_api})")

        while True:
            params = {"date": date_api}
            if pagination_key:
                params["pagination_key"] = pagination_key

            try:
                async with self.semaphore:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status != 200:
                            text = await response.text()
                            logger.warning(f"Failed to fetch quotes for {date}: {response.status} - {text}")
                            break

                        data = await response.json()
                        quotes = data.get("daily_quotes", [])

                        logger.debug(f"Got {len(quotes)} quotes for {date}")

                        if quotes:
                            all_quotes.extend(quotes)

                        # Check for pagination
                        pagination_key = data.get("pagination_key")
                        if not pagination_key:
                            break

            except Exception as e:
                logger.error(f"Error fetching quotes for {date}: {e}")
                break

        if all_quotes:
            df = pl.DataFrame(all_quotes)
            # Ensure Date column (keep in YYYY-MM-DD format)
            df = df.with_columns(pl.lit(date).alias("Date"))
            return df

        return pl.DataFrame()

    async def process_single_day(
        self, session: aiohttp.ClientSession, date: str
    ) -> pl.DataFrame:
        """
        1日分のデータ処理: daily_quotes と listed_info を内部結合
        """
        logger.info(f"Processing {date}...")

        # 並列で取得
        quotes_task = self.get_daily_quotes_for_date(session, date)
        listed_task = self.get_listed_info_for_date(session, date)

        quotes_df, listed_df = await asyncio.gather(quotes_task, listed_task)

        if quotes_df.is_empty() or listed_df.is_empty():
            logger.warning(f"No data for {date}")
            return pl.DataFrame()

        logger.info(f"  Raw: {len(quotes_df)} quotes, {len(listed_df)} listed stocks")

        # 内部結合（Code列で結合）
        merged_df = quotes_df.join(listed_df, on="Code", how="inner")

        logger.info(f"  After join: {len(merged_df)} records")

        # 重複除去（もしあれば）
        if "Code" in merged_df.columns and "Date" in merged_df.columns:
            merged_df = merged_df.unique(subset=["Code", "Date"])

        # Adjustment列を優先的に使用
        columns_to_rename = {}

        if "AdjustmentClose" in merged_df.columns:
            columns_to_rename["AdjustmentClose"] = "Close"
            if "Close" in merged_df.columns:
                merged_df = merged_df.drop("Close")

        if "AdjustmentOpen" in merged_df.columns:
            columns_to_rename["AdjustmentOpen"] = "Open"
            if "Open" in merged_df.columns:
                merged_df = merged_df.drop("Open")

        if "AdjustmentHigh" in merged_df.columns:
            columns_to_rename["AdjustmentHigh"] = "High"
            if "High" in merged_df.columns:
                merged_df = merged_df.drop("High")

        if "AdjustmentLow" in merged_df.columns:
            columns_to_rename["AdjustmentLow"] = "Low"
            if "Low" in merged_df.columns:
                merged_df = merged_df.drop("Low")

        if "AdjustmentVolume" in merged_df.columns:
            columns_to_rename["AdjustmentVolume"] = "Volume"
            if "Volume" in merged_df.columns:
                merged_df = merged_df.drop("Volume")

        if columns_to_rename:
            merged_df = merged_df.rename(columns_to_rename)

        return merged_df

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


class JQuantsPipelineV2:
    """Complete pipeline with proper data flow: inner join of daily_quotes and listed_info."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(
            "/home/ubuntu/gogooku3-standalone/output"
        )
        self.output_dir.mkdir(exist_ok=True)
        self.fetcher = None
        self.calendar_fetcher = None
        self.builder = MLDatasetBuilder(output_dir=self.output_dir)

    async def fetch_jquants_data(
        self, start_date: str = None, end_date: str = None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Fetch data from JQuants API with proper data flow."""
        # Get credentials
        email = os.getenv("JQUANTS_AUTH_EMAIL", "")
        password = os.getenv("JQUANTS_AUTH_PASSWORD", "")

        if not email or not password:
            logger.error("JQuants credentials not found in environment")
            return pl.DataFrame(), pl.DataFrame()

        # Initialize fetchers
        self.fetcher = JQuantsAsyncFetcherV2(email, password)
        self.calendar_fetcher = TradingCalendarFetcher(self.fetcher)

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

            # Step 2: 営業日ごとにdaily_quotesとlisted_infoを取得して内部結合
            logger.info("Step 2: 営業日ごとのデータ処理中...")
            all_data = []
            batch_size = 10  # 10日分ずつ並列処理

            for i in range(0, len(business_days), batch_size):
                batch_days = business_days[i:i+batch_size]
                logger.info(f"  Batch {i//batch_size + 1}/{(len(business_days)-1)//batch_size + 1}: {batch_days[0]} - {batch_days[-1]}")

                # バッチ内の日付を並列処理
                tasks = [
                    self.fetcher.process_single_day(session, date)
                    for date in batch_days
                ]

                results = await asyncio.gather(*tasks)

                # 結果を追加（空でないもののみ）
                for df in results:
                    if not df.is_empty():
                        all_data.append(df)

                logger.info(f"    累積レコード数: {sum(len(df) for df in all_data)}")

            # 全データ結合
            if all_data:
                price_df = pl.concat(all_data)
                logger.info(f"✅ データ取得完了: {len(price_df)}レコード")

                # 銘柄・日付でソート
                price_df = price_df.sort(["Code", "Date"])

                # 統計情報
                logger.info(f"  ユニーク銘柄数: {price_df['Code'].n_unique()}")
                logger.info(f"  期間: {price_df['Date'].min()} - {price_df['Date'].max()}")

            else:
                price_df = pl.DataFrame()
                logger.error("データが取得できませんでした")

            # Step 3: TOPIXデータ取得
            logger.info("Step 3: TOPIXデータ取得中...")
            topix_df = await self.fetcher.fetch_topix_data(
                session, start_date, end_date
            )

            return price_df, topix_df

    def process_pipeline(
        self, price_df: pl.DataFrame, topix_df: pl.DataFrame | None = None
    ) -> tuple:
        """Process the complete pipeline with technical indicators."""
        logger.info("=" * 60)
        logger.info("Processing ML Dataset Pipeline")
        logger.info("=" * 60)

        # Apply technical features (銘柄別に計算)
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
        """Run the complete pipeline with proper data flow."""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("COMPLETE ML DATASET PIPELINE V2")
        logger.info("With proper daily_quotes × listed_info inner join")
        logger.info("=" * 60)

        # Step 1: Get data
        topix_df = None
        if use_jquants:
            logger.info("Fetching data from JQuants API...")
            price_df, topix_df = await self.fetch_jquants_data(start_date, end_date)

            if price_df.is_empty():
                logger.error("Failed to fetch data")
                return None, None
        else:
            logger.info("Creating sample data...")
            from data.ml_dataset_builder import create_sample_data
            price_df = create_sample_data(100, 300)

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

    parser = argparse.ArgumentParser(description="Run ML Dataset Pipeline V2")
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
    pipeline = JQuantsPipelineV2()
    df, metadata = await pipeline.run(
        use_jquants=args.jquants, start_date=args.start_date, end_date=args.end_date
    )

    return df, metadata


if __name__ == "__main__":
    # Run the async main function
    df, metadata = asyncio.run(main())
