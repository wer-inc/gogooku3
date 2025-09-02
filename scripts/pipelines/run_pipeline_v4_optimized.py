#!/usr/bin/env python3
"""
Optimized ML Dataset Pipeline V4 - 最適化版
軸自動選択、差分検知、イベント追跡を含む完全最適化版
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
import json
from typing import List, Optional, Dict, Set, Tuple
from dotenv import load_dotenv
from dataclasses import dataclass, asdict

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.ml_dataset_builder import MLDatasetBuilder
from components.trading_calendar_fetcher import TradingCalendarFetcher
from components.market_code_filter import MarketCodeFilter
from components.axis_decider import AxisDecider
from components.daily_quotes_by_code import DailyQuotesByCodeFetcher
from components.listed_info_manager import ListedInfoManager
from components.event_detector import EventDetector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    component: str
    start_time: float
    end_time: float
    api_calls: int = 0
    records_processed: int = 0
    memory_usage_mb: float = 0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "duration_seconds": self.duration,
            "api_calls": self.api_calls,
            "records_processed": self.records_processed,
            "memory_usage_mb": self.memory_usage_mb
        }


class PerformanceTracker:
    """パフォーマンストラッキング"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.api_call_count = 0
        self.start_time = time.time()
        
    def start_component(self, component_name: str) -> PerformanceMetrics:
        """コンポーネントの計測開始"""
        metric = PerformanceMetrics(
            component=component_name,
            start_time=time.time(),
            end_time=0
        )
        return metric
    
    def end_component(self, metric: PerformanceMetrics, 
                     api_calls: int = 0, records: int = 0):
        """コンポーネントの計測終了"""
        metric.end_time = time.time()
        metric.api_calls = api_calls
        metric.records_processed = records
        
        # メモリ使用量を取得
        try:
            import psutil
            process = psutil.Process()
            metric.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except:
            pass
        
        self.metrics.append(metric)
        self.api_call_count += api_calls
    
    def get_summary(self) -> Dict:
        """サマリーを取得"""
        total_duration = time.time() - self.start_time
        
        return {
            "total_duration_seconds": total_duration,
            "total_api_calls": self.api_call_count,
            "total_records": sum(m.records_processed for m in self.metrics),
            "components": [m.to_dict() for m in self.metrics],
            "average_memory_mb": sum(m.memory_usage_mb for m in self.metrics) / len(self.metrics) if self.metrics else 0
        }
    
    def save_report(self, filepath: Path):
        """レポートを保存"""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Performance report saved to {filepath}")


class JQuantsOptimizedFetcherV4:
    """最適化されたJQuants API fetcher V4"""

    def __init__(self, email: str, password: str, tracker: PerformanceTracker):
        self.email = email
        self.password = password
        self.base_url = "https://api.jquants.com/v1"
        self.id_token = None
        self.tracker = tracker
        
        # 有料プラン向け設定
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT_FETCH", 75))
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # 最適化コンポーネント
        self.axis_decider = None
        self.code_fetcher = None
        self.listed_manager = None
        self.event_detector = EventDetector()
        self.calendar_fetcher = None

    async def authenticate(self, session: aiohttp.ClientSession):
        """Authenticate with JQuants API."""
        metric = self.tracker.start_component("authentication")
        
        auth_url = f"{self.base_url}/token/auth_user"
        auth_payload = {"mailaddress": self.email, "password": self.password}

        async with session.post(auth_url, json=auth_payload) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Auth failed: {response.status} - {text}")
            data = await response.json()
            refresh_token = data["refreshToken"]

        refresh_url = f"{self.base_url}/token/auth_refresh"
        params = {"refreshtoken": refresh_token}

        async with session.post(refresh_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Failed to get ID token: {response.status}")
            data = await response.json()
            self.id_token = data["idToken"]

        logger.info("✅ JQuants authentication successful")
        
        # APIクライアントを模擬
        api_client = type('obj', (object,), {'id_token': self.id_token})()
        
        # コンポーネント初期化
        self.axis_decider = AxisDecider(api_client)
        self.code_fetcher = DailyQuotesByCodeFetcher(api_client)
        self.listed_manager = ListedInfoManager(api_client)
        self.calendar_fetcher = TradingCalendarFetcher(api_client)
        
        self.tracker.end_component(metric, api_calls=2)

    async def fetch_daily_quotes_optimized(
        self,
        session: aiohttp.ClientSession,
        business_days: List[str],
        target_codes: Optional[Set[str]] = None
    ) -> pl.DataFrame:
        """
        最適化された日次株価取得
        軸自動選択により最適な方法で取得
        """
        metric = self.tracker.start_component("daily_quotes_optimized")
        
        # 軸の自動選択
        axis, axis_metrics = await self.axis_decider.get_optimal_axis(
            session,
            sample_days=business_days[:3] if len(business_days) > 3 else business_days,
            market_filter=target_codes is not None
        )
        
        logger.info(f"Selected axis: {axis} (reason: {axis_metrics.get('decision_reason')})")
        
        if axis == "by_code" and target_codes:
            # 銘柄軸で取得
            logger.info(f"Fetching by code axis for {len(target_codes)} stocks...")
            
            all_dfs = []
            api_calls = 0
            
            # バッチ処理
            codes_list = list(target_codes)
            batch_size = 50
            
            for i in range(0, len(codes_list), batch_size):
                batch_codes = codes_list[i:i+batch_size]
                
                tasks = []
                for code in batch_codes:
                    task = self.code_fetcher.fetch_by_code(
                        session, code, 
                        business_days[0], business_days[-1]
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                api_calls += len(tasks)
                
                for df in results:
                    if not df.is_empty():
                        all_dfs.append(df)
                
                logger.info(f"  Progress: {min(i+batch_size, len(codes_list))}/{len(codes_list)} stocks")
            
            if all_dfs:
                combined_df = pl.concat(all_dfs)
                self.tracker.end_component(metric, api_calls=api_calls, records=len(combined_df))
                return combined_df
        else:
            # 日付軸で取得（既存の実装を使用）
            logger.info(f"Fetching by date axis for {len(business_days)} days...")
            df = await self.fetch_daily_quotes_bulk(session, business_days)
            
            # 市場フィルタリング
            if target_codes:
                original_count = len(df)
                df = df.filter(pl.col("Code").is_in(target_codes))
                logger.info(f"Filtered: {original_count} → {len(df)} records")
            
            self.tracker.end_component(metric, api_calls=len(business_days), records=len(df))
            return df
        
        self.tracker.end_component(metric)
        return pl.DataFrame()

    async def fetch_daily_quotes_bulk(
        self,
        session: aiohttp.ClientSession,
        business_days: List[str],
        batch_size: int = 30
    ) -> pl.DataFrame:
        """既存のdate軸実装（run_pipeline_v3から流用）"""
        all_quotes = []
        
        for i in range(0, len(business_days), batch_size):
            batch_days = business_days[i:i+batch_size]
            
            tasks = []
            for date in batch_days:
                date_api = date.replace("-", "")
                task = self._fetch_daily_quotes_for_date(session, date, date_api)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            for df in results:
                if not df.is_empty():
                    all_quotes.append(df)
        
        if all_quotes:
            return pl.concat(all_quotes)
        
        return pl.DataFrame()

    async def _fetch_daily_quotes_for_date(
        self, session: aiohttp.ClientSession, date: str, date_api: str
    ) -> pl.DataFrame:
        """特定日の全銘柄のdaily_quotesを取得（ページネーション対応）"""
        url = f"{self.base_url}/prices/daily_quotes"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        
        all_quotes = []
        pagination_key = None
        
        while True:
            params = {"date": date_api}
            if pagination_key:
                params["pagination_key"] = pagination_key
            
            try:
                async with self.semaphore:
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
                            
            except Exception as e:
                logger.error(f"Error fetching quotes for {date}: {e}")
                break
        
        if all_quotes:
            return pl.DataFrame(all_quotes)
        
        return pl.DataFrame()

    async def fetch_listed_info_optimized(
        self,
        session: aiohttp.ClientSession,
        business_days: List[str],
        daily_quotes_df: Optional[pl.DataFrame] = None
    ) -> Tuple[Dict[str, pl.DataFrame], List[Dict]]:
        """
        最適化されたlisted_info取得
        月初＋差分日のみ取得
        """
        metric = self.tracker.start_component("listed_info_optimized")
        
        # 月初スナップショット取得
        snapshots = await self.listed_manager.get_monthly_snapshots(session, business_days)
        
        events = []
        api_calls = len(snapshots)
        
        # 差分検知（daily_quotesがあれば）
        if daily_quotes_df and not daily_quotes_df.is_empty():
            # 日次のCode集合変化を検知
            dates = sorted(daily_quotes_df["Date"].unique().to_list())
            prev_codes = set()
            
            for date in dates:
                curr_codes = set(
                    daily_quotes_df.filter(pl.col("Date") == date)["Code"].unique().to_list()
                )
                
                if prev_codes and curr_codes != prev_codes:
                    # 変化があった日はlisted_infoを追加取得
                    logger.info(f"Code set changed on {date}, fetching listed_info...")
                    snapshot = await self.listed_manager.get_snapshot_at(session, str(date))
                    snapshots[str(date)] = snapshot
                    api_calls += 1
                    
                    # イベント検知
                    if len(snapshots) >= 2:
                        sorted_dates = sorted(snapshots.keys())
                        prev_date = sorted_dates[-2]
                        changes = self.listed_manager.detect_changes(
                            snapshots[prev_date],
                            snapshot,
                            str(date)
                        )
                        
                        # イベント作成
                        for event_type, items in changes.items():
                            for item in items:
                                item["event_type"] = event_type.rstrip("s")
                                events.append(item)
                
                prev_codes = curr_codes
        
        logger.info(f"✅ Listed info: {len(snapshots)} snapshots, {len(events)} events detected")
        self.tracker.end_component(metric, api_calls=api_calls, records=len(events))
        
        return snapshots, events

    async def fetch_statements_by_date(
        self, session: aiohttp.ClientSession, business_days: List[str]
    ) -> pl.DataFrame:
        """date軸での財務諸表取得（run_pipeline_v3から流用）"""
        metric = self.tracker.start_component("statements_by_date")
        
        url = f"{self.base_url}/fins/statements"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        
        all_statements = []
        valid_days = 0
        
        for date in business_days:
            date_api = date.replace("-", "")
            params = {"date": date_api}
            pagination_key = None
            statements_for_date = []
            
            while True:
                if pagination_key:
                    params["pagination_key"] = pagination_key
                
                try:
                    async with self.semaphore:
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
                valid_days += 1
        
        result_df = pl.DataFrame() if not all_statements else pl.DataFrame(all_statements)
        
        self.tracker.end_component(metric, api_calls=len(business_days), records=len(result_df))
        return result_df


class JQuantsPipelineV4Optimized:
    """最適化されたパイプライン V4"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("/home/ubuntu/gogooku3-standalone/output")
        self.output_dir.mkdir(exist_ok=True)
        
        # パフォーマンストラッカー
        self.tracker = PerformanceTracker()
        
        # コンポーネント
        self.fetcher = None
        self.builder = MLDatasetBuilder(output_dir=self.output_dir)
        self.event_detector = EventDetector()

    async def fetch_jquants_data_optimized(
        self, start_date: str = None, end_date: str = None
    ) -> tuple:
        """最適化されたデータ取得フロー"""
        metric = self.tracker.start_component("total_fetch")
        
        # Get credentials
        email = os.getenv("JQUANTS_AUTH_EMAIL", "")
        password = os.getenv("JQUANTS_AUTH_PASSWORD", "")

        if not email or not password:
            logger.error("JQuants credentials not found in environment")
            return pl.DataFrame(), pl.DataFrame(), [], {}

        # Initialize fetcher
        self.fetcher = JQuantsOptimizedFetcherV4(email, password, self.tracker)

        # Date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = "2021-01-01"

        async with aiohttp.ClientSession() as session:
            # Authenticate
            await self.fetcher.authenticate(session)
            
            # Step 1: 営業日カレンダー取得
            logger.info(f"Step 1: Fetching trading calendar ({start_date} - {end_date})...")
            cal_metric = self.tracker.start_component("trading_calendar")
            
            calendar_data = await self.fetcher.calendar_fetcher.get_trading_calendar(
                start_date, end_date, session
            )
            business_days = calendar_data.get("business_days", [])
            
            self.tracker.end_component(cal_metric, api_calls=1, records=len(business_days))
            logger.info(f"✅ Business days: {len(business_days)}")
            
            if not business_days:
                logger.error("No business days found")
                return pl.DataFrame(), pl.DataFrame(), [], {}

            # Step 2: Listed info（月初＋差分）で市場銘柄を特定
            logger.info("Step 2: Fetching listed info (monthly + diff)...")
            snapshots, events = await self.fetcher.fetch_listed_info_optimized(
                session, business_days
            )
            
            # ターゲット銘柄を特定（市場フィルタリング）
            target_codes = set()
            for snapshot_df in snapshots.values():
                if not snapshot_df.is_empty() and "MarketCode" in snapshot_df.columns:
                    # Market Codeフィルタリング
                    filtered = snapshot_df.filter(
                        pl.col("MarketCode").is_in(MarketCodeFilter.TARGET_MARKET_CODES)
                    )
                    target_codes.update(filtered["Code"].to_list())
            
            logger.info(f"✅ Target stocks: {len(target_codes)} (filtered by market)")

            # Step 3: Daily quotes（最適軸で取得）
            logger.info("Step 3: Fetching daily quotes (optimized axis)...")
            price_df = await self.fetcher.fetch_daily_quotes_optimized(
                session, business_days, target_codes
            )
            
            logger.info(f"✅ Price data: {len(price_df)} records, {price_df['Code'].n_unique()} stocks")

            # Step 4: 財務諸表（date軸）
            logger.info("Step 4: Fetching statements (date axis)...")
            statements_df = await self.fetcher.fetch_statements_by_date(
                session, business_days
            )
            
            logger.info(f"✅ Statements: {len(statements_df)} records")

            # Adjustment列の処理
            if not price_df.is_empty():
                columns_to_rename = {}
                
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
                
                # 必要な列を選択
                required_cols = ["Code", "Date", "Open", "High", "Low", "Close", "Volume"]
                available_cols = [col for col in required_cols if col in price_df.columns]
                price_df = price_df.select(available_cols)
                
                # 数値型に変換
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in price_df.columns:
                        price_df = price_df.with_columns(pl.col(col).cast(pl.Float64))
                
                # ソート
                price_df = price_df.sort(["Code", "Date"])
            
            self.tracker.end_component(metric, records=len(price_df))
            
            return price_df, statements_df, events, snapshots

    def process_pipeline(
        self, 
        price_df: pl.DataFrame,
        statements_df: Optional[pl.DataFrame] = None,
        events: Optional[List[Dict]] = None
    ) -> tuple:
        """Process the pipeline with technical indicators."""
        metric = self.tracker.start_component("process_pipeline")
        
        logger.info("=" * 60)
        logger.info("Processing ML Dataset Pipeline")
        logger.info("=" * 60)

        # Apply technical features
        df = self.builder.create_technical_features(price_df)

        # Add pandas-ta features
        df = self.builder.add_pandas_ta_features(df)

        # 財務諸表データを結合
        if statements_df is not None and not statements_df.is_empty():
            logger.info(f"  Adding statement features: {len(statements_df)} records")
            df = self.builder.add_statements_features(df, statements_df)

        # Create metadata
        metadata = self.builder.create_metadata(df)
        
        # Add events to metadata
        if events:
            metadata["events"] = {
                "total": len(events),
                "by_type": {}
            }
            for event in events:
                event_type = event.get("event_type", "unknown")
                metadata["events"]["by_type"][event_type] = \
                    metadata["events"]["by_type"].get(event_type, 0) + 1

        # Display summary
        logger.info("\nDataset Summary:")
        logger.info(f"  Shape: {len(df)} rows × {len(df.columns)} columns")
        logger.info(f"  Features: {metadata['features']['count']}")
        logger.info(f"  Stocks: {metadata['stocks']}")
        logger.info(f"  Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
        
        if events:
            logger.info(f"  Events detected: {len(events)}")

        # Save dataset
        parquet_path, meta_path = self.builder.save_dataset(df, metadata)
        
        self.tracker.end_component(metric, records=len(df))
        
        return df, metadata, (parquet_path, None, meta_path)

    async def run(
        self, use_jquants: bool = True, start_date: str = None, end_date: str = None
    ):
        """Run the optimized pipeline."""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("OPTIMIZED ML DATASET PIPELINE V4")
        logger.info("With axis selection, diff detection, and event tracking")
        logger.info("=" * 60)

        # Step 1: Get data
        if use_jquants:
            logger.info("Fetching data from JQuants API (optimized)...")
            price_df, statements_df, events, snapshots = await self.fetch_jquants_data_optimized(
                start_date, end_date
            )

            if price_df.is_empty():
                logger.error("Failed to fetch data")
                return None, None
                
            # イベント処理
            if events:
                self.event_detector.events = events
                
                # market_membership生成
                membership_df = self.event_detector.generate_market_membership()
                if not membership_df.is_empty():
                    logger.info(f"Generated market membership: {len(membership_df)} records")
                    
                    # 保存
                    membership_path = self.output_dir / "market_membership.parquet"
                    membership_df.write_parquet(membership_path)
                    logger.info(f"Saved to {membership_path}")
                
                # securities_events生成
                events_df = self.event_detector.generate_securities_events_table()
                if not events_df.is_empty():
                    logger.info(f"Generated securities events: {len(events_df)} records")
                    
                    # 保存
                    events_path = self.output_dir / "securities_events.parquet"
                    events_df.write_parquet(events_path)
                    logger.info(f"Saved to {events_path}")
        else:
            logger.info("Creating sample data...")
            from data.ml_dataset_builder import create_sample_data
            price_df = create_sample_data(100, 300)
            statements_df = None
            events = []

        logger.info(f"Data loaded: {len(price_df)} rows, {price_df['Code'].n_unique()} stocks")

        # Step 2: Process pipeline
        logger.info("\nStep 2: Processing ML features...")
        df, metadata, file_paths = self.process_pipeline(price_df, statements_df, events)

        # Step 3: Performance report
        logger.info("\nStep 3: Generating performance report...")
        report_path = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.tracker.save_report(report_path)
        
        # Display performance summary
        summary = self.tracker.get_summary()
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total time: {summary['total_duration_seconds']:.2f} seconds")
        logger.info(f"Total API calls: {summary['total_api_calls']}")
        logger.info(f"Total records: {summary['total_records']:,}")
        logger.info(f"Average memory: {summary['average_memory_mb']:.0f} MB")
        
        logger.info("\nComponent breakdown:")
        for comp in summary['components']:
            logger.info(f"  {comp['component']}: {comp['duration_seconds']:.2f}s, "
                       f"{comp['api_calls']} calls, {comp['records_processed']:,} records")

        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed:.2f} seconds")
        logger.info(f"Processing speed: {len(df)/elapsed:.0f} rows/second")
        logger.info("\nOutput files:")
        logger.info(f"  Dataset: {file_paths[0]}")
        logger.info(f"  Metadata: {file_paths[2]}")
        logger.info(f"  Performance: {report_path}")
        
        if events:
            logger.info(f"  Events: {self.output_dir}/securities_events.parquet")
            logger.info(f"  Membership: {self.output_dir}/market_membership.parquet")

        return df, metadata


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Optimized ML Dataset Pipeline V4")
    parser.add_argument(
        "--jquants",
        action="store_true",
        help="Use JQuants API (requires credentials in .env)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-01-01",
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
            logger.info(f"Using JQuants API with account: {os.getenv('JQUANTS_AUTH_EMAIL')[:10]}...")

    # Run pipeline
    pipeline = JQuantsPipelineV4Optimized()
    df, metadata = await pipeline.run(
        use_jquants=args.jquants, start_date=args.start_date, end_date=args.end_date
    )

    return df, metadata


if __name__ == "__main__":
    # Run the async main function
    df, metadata = asyncio.run(main())