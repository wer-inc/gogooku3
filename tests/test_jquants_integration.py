#!/usr/bin/env python3
"""
J-Quants API統合テスト
Trading Calendar、Market Codeフィルタリング、データ取得の統合テスト
"""

import asyncio
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.components.trading_calendar_fetcher import TradingCalendarFetcher
from scripts.components.market_code_filter import MarketCodeFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_trading_calendar():
    """Trading Calendar APIのテスト"""
    logger.info("=" * 60)
    logger.info("Trading Calendar APIテスト")
    logger.info("=" * 60)
    
    # 簡易認証クラス
    class SimpleAPIClient:
        def __init__(self):
            self.id_token = None
            
        async def authenticate(self, session):
            import os
            import aiohttp
            
            email = os.getenv("JQUANTS_AUTH_EMAIL")
            password = os.getenv("JQUANTS_AUTH_PASSWORD")
            
            if not email or not password:
                raise Exception("認証情報が設定されていません")
            
            # リフレッシュトークン取得
            auth_url = "https://api.jquants.com/v1/token/auth_user"
            auth_payload = {"mailaddress": email, "password": password}
            
            async with session.post(auth_url, json=auth_payload) as response:
                if response.status != 200:
                    raise Exception(f"Auth failed: {response.status}")
                data = await response.json()
                refresh_token = data["refreshToken"]
            
            # IDトークン取得
            refresh_url = "https://api.jquants.com/v1/token/auth_refresh"
            params = {"refreshtoken": refresh_token}
            
            async with session.post(refresh_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get ID token: {response.status}")
                data = await response.json()
                self.id_token = data["idToken"]
            
            logger.info("✅ 認証成功")
    
    try:
        import aiohttp
        
        api_client = SimpleAPIClient()
        fetcher = TradingCalendarFetcher(api_client)
        
        async with aiohttp.ClientSession() as session:
            # 認証
            await api_client.authenticate(session)
            
            # 1週間分のカレンダー取得テスト
            calendar_data = await fetcher.get_trading_calendar(
                "2025-01-01",
                "2025-01-07",
                session,
                use_cache=False  # テスト時はキャッシュ無効
            )
            
            logger.info(f"✅ 営業日数: {len(calendar_data.get('business_days', []))}")
            logger.info(f"✅ 休日数: {len(calendar_data.get('holidays', []))}")
            logger.info(f"✅ 半休日数: {len(calendar_data.get('half_days', []))}")
            
            if calendar_data.get('business_days'):
                logger.info(f"  最初の営業日: {calendar_data['business_days'][0]}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Trading Calendarテスト失敗: {e}")
        return False


def test_market_code_filter():
    """Market Codeフィルタリングのテスト"""
    logger.info("=" * 60)
    logger.info("Market Codeフィルタリングテスト")
    logger.info("=" * 60)
    
    try:
        import polars as pl
        
        # テストデータ作成
        test_data = {
            "Code": ["1234", "5678", "9012", "3456", "7890", "1111", "2222", "3333"],
            "MarketCode": ["0111", "0112", "0113", "0105", "0109", "0101", "0106", "9999"],
            "Name": ["プライム銘柄", "スタンダード銘柄", "グロース銘柄", 
                    "PRO銘柄", "その他銘柄", "東証一部銘柄", "JASDAQ銘柄", "未知銘柄"]
        }
        
        df = pl.DataFrame(test_data)
        logger.info(f"テストデータ: {len(df)}銘柄")
        
        # フィルタリング実行
        filtered_df = MarketCodeFilter.filter_stocks(df)
        logger.info(f"フィルタリング後: {len(filtered_df)}銘柄")
        
        # 検証
        expected_codes = ["1234", "5678", "9012", "1111", "2222"]  # 8市場の銘柄のみ
        actual_codes = filtered_df["Code"].to_list()
        
        if set(actual_codes) == set(expected_codes):
            logger.info("✅ Market Codeフィルタリング成功")
            return True
        else:
            logger.error(f"❌ フィルタリング結果が期待値と異なる")
            logger.error(f"  期待: {expected_codes}")
            logger.error(f"  実際: {actual_codes}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Market Codeフィルタリングテスト失敗: {e}")
        return False


async def test_integrated_pipeline():
    """統合パイプラインの簡易テスト"""
    logger.info("=" * 60)
    logger.info("統合パイプラインテスト")
    logger.info("=" * 60)
    
    try:
        from scripts.pipelines.run_pipeline import JQuantsPipeline
        
        # パイプライン初期化
        pipeline = JQuantsPipeline()
        
        # 短期間でテスト（1日分）
        start_date = "2025-01-06"
        end_date = "2025-01-06"
        
        logger.info(f"テスト期間: {start_date} - {end_date}")
        
        # データ取得テスト（実際のAPIは呼ばない）
        logger.info("✅ パイプライン初期化成功")
        
        # 各コンポーネントの存在確認
        assert hasattr(pipeline, 'calendar_fetcher'), "calendar_fetcherが存在しない"
        assert hasattr(pipeline, 'fetcher'), "fetcherが存在しない"
        assert hasattr(pipeline, 'builder'), "builderが存在しない"
        
        logger.info("✅ 必要なコンポーネントが存在")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 統合パイプラインテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """メインテスト実行"""
    logger.info("🚀 J-Quants API統合テスト開始")
    logger.info(f"実行時刻: {datetime.now()}")
    
    results = []
    
    # 1. Market Codeフィルタリングテスト
    result = test_market_code_filter()
    results.append(("Market Codeフィルタリング", result))
    
    # 2. Trading Calendar APIテスト（実際のAPI使用）
    import os
    if os.getenv("JQUANTS_AUTH_EMAIL"):
        result = await test_trading_calendar()
        results.append(("Trading Calendar API", result))
    else:
        logger.warning("⚠️ J-Quants認証情報なし、Trading Calendarテストをスキップ")
    
    # 3. 統合パイプラインテスト
    result = await test_integrated_pipeline()
    results.append(("統合パイプライン", result))
    
    # 結果サマリー
    logger.info("")
    logger.info("=" * 60)
    logger.info("テスト結果サマリー")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("")
        logger.info("🎉 全テスト成功！")
    else:
        logger.info("")
        logger.info("⚠️ 一部のテストが失敗しました")
    
    return all_passed


if __name__ == "__main__":
    # .envファイル読み込み
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    # テスト実行
    success = asyncio.run(main())
    sys.exit(0 if success else 1)