#!/usr/bin/env python3
# DEPRECATED: Moved to scripts/_archive on 2025-09-04
"""
Optimized Pipeline Test - 最適化されたJ-Quants APIパイプラインの統合テスト
軸自動選択、差分検知、イベント追跡を含む完全なテスト
"""

import asyncio
import aiohttp
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import logging
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from components.axis_decider import AxisDecider
from components.daily_quotes_by_code import DailyQuotesByCodeFetcher, MarketMembership
from components.listed_info_manager import ListedInfoManager
from components.event_detector import EventDetector
from components.trading_calendar_fetcher import TradingCalendarFetcher
from components.market_code_filter import MarketCodeFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OptimizedPipelineTest:
    """最適化パイプラインの統合テスト"""
    
    def __init__(self):
        self.output_dir = Path("output/optimized_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 環境変数読み込み
        load_dotenv()
        
        self.email = os.getenv("JQUANTS_AUTH_EMAIL")
        self.password = os.getenv("JQUANTS_AUTH_PASSWORD")
        
        if not self.email or not self.password:
            raise Exception("JQuants credentials not found in .env")
        
        # コンポーネント初期化
        self.api_client = None
        self.axis_decider = None
        self.code_fetcher = None
        self.listed_manager = None
        self.event_detector = EventDetector()
        self.calendar_fetcher = None
        self.market_filter = MarketCodeFilter()
    
    async def authenticate(self, session: aiohttp.ClientSession):
        """JQuants API認証"""
        auth_url = "https://api.jquants.com/v1/token/auth_user"
        auth_payload = {"mailaddress": self.email, "password": self.password}
        
        async with session.post(auth_url, json=auth_payload) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Auth failed: {response.status} - {text}")
            data = await response.json()
            refresh_token = data["refreshToken"]
        
        refresh_url = "https://api.jquants.com/v1/token/auth_refresh"
        params = {"refreshtoken": refresh_token}
        
        async with session.post(refresh_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Failed to get ID token: {response.status}")
            data = await response.json()
            self.id_token = data["idToken"]
        
        logger.info("✅ Authentication successful")
        
        # APIクライアントを模擬
        self.api_client = type('obj', (object,), {'id_token': self.id_token})()
        
        # コンポーネント初期化
        self.axis_decider = AxisDecider(self.api_client)
        self.code_fetcher = DailyQuotesByCodeFetcher(self.api_client)
        self.listed_manager = ListedInfoManager(self.api_client)
        self.calendar_fetcher = TradingCalendarFetcher(self.api_client)
    
    async def test_axis_decision(self, session: aiohttp.ClientSession):
        """軸自動選択のテスト"""
        logger.info("\n=== Testing Axis Decision ===")
        
        # サンプル日付と銘柄
        sample_days = ["2025-01-06", "2025-01-07", "2025-01-08"]
        sample_codes = ["7203", "6758", "9984", "8306", "9432"]  # 主要銘柄
        
        # 最適軸の決定
        axis, metrics = await self.axis_decider.get_optimal_axis(
            session,
            sample_days=sample_days,
            sample_codes=sample_codes,
            market_filter=True
        )
        
        logger.info(f"✅ Optimal axis: {axis}")
        logger.info(f"  Date axis pages: {metrics.get('date_metrics', {}).get('total_pages', 'N/A')}")
        logger.info(f"  Code axis pages: {metrics.get('code_metrics', {}).get('total_pages', 'N/A')}")
        logger.info(f"  Decision: {metrics.get('decision_reason', 'N/A')}")
        
        return axis
    
    async def test_code_axis_fetching(self, session: aiohttp.ClientSession):
        """銘柄軸取得のテスト"""
        logger.info("\n=== Testing Code Axis Fetching ===")
        
        # テスト用のmembership設定
        self.code_fetcher.market_membership.add_membership(
            "7203", "0111", "2025-01-01", None  # トヨタ、プライム市場
        )
        
        # 銘柄軸で取得
        df = await self.code_fetcher.fetch_by_code(
            session,
            "7203",
            "2025-01-06",
            "2025-01-10"
        )
        
        if not df.is_empty():
            logger.info(f"✅ Fetched {len(df)} records for code 7203")
            logger.info(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
            logger.info(f"  Columns: {df.columns[:5]}...")  # 最初の5列表示
        else:
            logger.warning("⚠️ No data fetched for code 7203")
        
        return df
    
    async def test_listed_info_manager(self, session: aiohttp.ClientSession):
        """月初＋差分検知のテスト"""
        logger.info("\n=== Testing Listed Info Manager ===")
        
        # テスト期間の営業日
        business_days = [
            "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"
        ]
        
        # 月初スナップショット取得
        snapshots = await self.listed_manager.get_monthly_snapshots(
            session, business_days
        )
        
        for date, df in snapshots.items():
            logger.info(f"✅ Snapshot {date}: {len(df)} stocks")
            
            # Market Code分布を確認
            if "MarketCode" in df.columns:
                market_counts = df.group_by("MarketCode").agg(
                    pl.count().alias("count")
                ).sort("MarketCode")
                
                logger.info("  Market distribution:")
                for row in market_counts.iter_rows(named=True):
                    market_name = MarketCodeFilter.MARKET_NAMES.get(
                        row["MarketCode"], row["MarketCode"]
                    )
                    logger.info(f"    {market_name}: {row['count']} stocks")
        
        return snapshots
    
    async def test_event_detection(self, session: aiohttp.ClientSession):
        """イベント検知のテスト"""
        logger.info("\n=== Testing Event Detection ===")
        
        # テスト用のlisted_info変化を模擬
        prev_info = pl.DataFrame({
            "Code": ["1234", "5678", "9012"],
            "MarketCode": ["0111", "0112", "0113"],
            "CompanyName": ["TestA", "TestB", "TestC"]
        })
        
        curr_info = pl.DataFrame({
            "Code": ["1234", "5678", "3456"],  # 9012廃止、3456新規
            "MarketCode": ["0111", "0111", "0113"],  # 5678市場変更
            "CompanyName": ["NewTestA", "TestB", "TestD"]  # 1234社名変更
        })
        
        # イベント検知
        events = self.event_detector.process_listed_info_changes(
            prev_info, curr_info, "2025-01-10"
        )
        
        logger.info(f"✅ Detected {len(events)} events:")
        for event in events:
            logger.info(f"  {event['event_type']}: {event['code']} - "
                       f"{event.get('description', '')}")
        
        # market_membership生成
        if events:
            membership_df = self.event_detector.generate_market_membership()
            if not membership_df.is_empty():
                logger.info(f"✅ Generated market membership: {len(membership_df)} records")
        
        return events
    
    async def test_full_pipeline(self, session: aiohttp.ClientSession):
        """完全なパイプラインテスト"""
        logger.info("\n=== Testing Full Optimized Pipeline ===")
        
        # Step 1: 営業日取得
        logger.info("\nStep 1: Fetching trading calendar...")
        calendar_data = await self.calendar_fetcher.get_trading_calendar(
            "2025-01-01", "2025-01-31", session
        )
        business_days = calendar_data.get("business_days", [])[:5]  # 最初の5日でテスト
        logger.info(f"✅ Business days: {len(business_days)}")
        
        # Step 2: 軸選択
        logger.info("\nStep 2: Determining optimal axis...")
        axis, _ = await self.axis_decider.get_optimal_axis(
            session,
            sample_days=business_days[:3],
            market_filter=True
        )
        logger.info(f"✅ Selected axis: {axis}")
        
        # Step 3: 月初スナップショット
        logger.info("\nStep 3: Fetching monthly snapshots...")
        snapshots = await self.listed_manager.get_monthly_snapshots(
            session, business_days
        )
        logger.info(f"✅ Snapshots: {len(snapshots)}")
        
        # Step 4: イベント検知（仮）
        logger.info("\nStep 4: Detecting events...")
        if len(snapshots) >= 2:
            dates = sorted(snapshots.keys())
            changes = self.listed_manager.detect_changes(
                snapshots[dates[0]], 
                snapshots[dates[-1]],
                dates[-1]
            )
            
            total_changes = sum(len(v) for v in changes.values())
            logger.info(f"✅ Changes detected: {total_changes}")
            for change_type, items in changes.items():
                if items:
                    logger.info(f"  {change_type}: {len(items)}")
        
        # Step 5: サマリー
        logger.info("\n=== Pipeline Test Summary ===")
        logger.info(f"✅ All components tested successfully")
        logger.info(f"  - Axis decision: {axis}")
        logger.info(f"  - Business days: {len(business_days)}")
        logger.info(f"  - Snapshots: {len(snapshots)}")
        logger.info(f"  - Events detected: {len(self.event_detector.events)}")
        
        # 統計情報
        stats = self.event_detector.get_statistics()
        if stats["total_events"] > 0:
            logger.info(f"\nEvent Statistics:")
            logger.info(f"  Total: {stats['total_events']}")
            for event_type, count in stats["by_type"].items():
                logger.info(f"  {event_type}: {count}")
    
    async def run(self):
        """テスト実行"""
        start_time = datetime.now()
        
        logger.info("=" * 60)
        logger.info("OPTIMIZED J-QUANTS PIPELINE TEST")
        logger.info("=" * 60)
        
        async with aiohttp.ClientSession() as session:
            # 認証
            await self.authenticate(session)
            
            # 各コンポーネントのテスト
            await self.test_axis_decision(session)
            await self.test_code_axis_fetching(session)
            await self.test_listed_info_manager(session)
            await self.test_event_detection(session)
            
            # 統合テスト
            await self.test_full_pipeline(session)
        
        elapsed = datetime.now() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info(f"Total time: {elapsed.total_seconds():.2f} seconds")
        logger.info("=" * 60)


async def main():
    """メインエントリポイント"""
    test = OptimizedPipelineTest()
    await test.run()


if __name__ == "__main__":
    asyncio.run(main())
