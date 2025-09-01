#!/usr/bin/env python3
"""
Daily Stock Fetcher - 営業日ごとの銘柄リスト取得
上場・廃止を考慮して、各営業日の正確な銘柄リストを管理
"""

import asyncio
import aiohttp
import polars as pl
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)


class DailyStockFetcher:
    """営業日ごとの銘柄リストを管理"""
    
    def __init__(self, api_client, market_filter=None):
        self.api_client = api_client
        self.base_url = "https://api.jquants.com/v1"
        self.market_filter = market_filter
        self.cache_dir = Path("cache/daily_listings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 営業日ごとの銘柄キャッシュ
        self.daily_listings_cache = {}
        
    async def get_stocks_for_date(
        self, 
        date: str, 
        session: aiohttp.ClientSession,
        use_cache: bool = True
    ) -> pl.DataFrame:
        """
        特定日の上場銘柄リストを取得
        
        Args:
            date: 対象日 (YYYY-MM-DD)
            session: aiohttp ClientSession
            use_cache: キャッシュを使用するか
            
        Returns:
            その日に取引可能だった銘柄のDataFrame
        """
        # キャッシュチェック
        if use_cache and date in self.daily_listings_cache:
            return self.daily_listings_cache[date]
        
        cache_file = self.cache_dir / f"listings_{date}.parquet"
        if use_cache and cache_file.exists():
            try:
                df = pl.read_parquet(cache_file)
                self.daily_listings_cache[date] = df
                return df
            except:
                pass
        
        # APIから取得
        logger.debug(f"取得中: {date}の銘柄リスト")
        
        url = f"{self.base_url}/listed/info"
        headers = {"Authorization": f"Bearer {self.api_client.id_token}"}
        params = {"date": date}
        
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"Listed info取得失敗 ({date}): {response.status}")
                    return pl.DataFrame()
                
                data = await response.json()
                info_data = data.get("info", [])
                
                if not info_data:
                    logger.warning(f"{date}: 銘柄情報なし")
                    return pl.DataFrame()
                
                # DataFrameに変換
                df = pl.DataFrame(info_data)
                
                # 上場日・廃止日でフィルタリング
                df = self._filter_active_stocks(df, date)
                
                # Market Codeフィルタリング（8市場のみ）
                if self.market_filter:
                    df = self.market_filter.filter_stocks(df)
                
                # キャッシュ保存
                if use_cache and not df.is_empty():
                    df.write_parquet(cache_file)
                    self.daily_listings_cache[date] = df
                
                return df
                
        except Exception as e:
            logger.error(f"銘柄リスト取得エラー ({date}): {e}")
            return pl.DataFrame()
    
    def _filter_active_stocks(self, df: pl.DataFrame, target_date: str) -> pl.DataFrame:
        """対象日にアクティブな銘柄のみフィルタリング"""
        if df.is_empty():
            return df
        
        # Date列とDelistingDate列の存在確認
        if "Date" not in df.columns:
            logger.warning("Date列が存在しません")
            return df
        
        # 上場日が対象日以前の銘柄
        df = df.filter(pl.col("Date") <= target_date)
        
        # 廃止日が対象日より後、またはNULLの銘柄
        if "DelistingDate" in df.columns:
            df = df.filter(
                pl.col("DelistingDate").is_null() | 
                (pl.col("DelistingDate") > target_date)
            )
        
        return df
    
    async def get_stocks_for_period(
        self,
        business_days: List[str],
        session: aiohttp.ClientSession,
        progress_callback=None,
        batch_size: int = 10
    ) -> Dict[str, pl.DataFrame]:
        """
        期間中の各営業日の銘柄リストを取得（バッチ処理対応）
        
        Args:
            business_days: 営業日のリスト
            session: aiohttp ClientSession
            progress_callback: 進捗コールバック
            batch_size: 並列で処理する日数
            
        Returns:
            {date: DataFrame} の辞書
        """
        daily_listings = {}
        total_days = len(business_days)
        
        # バッチごとに処理
        for i in range(0, total_days, batch_size):
            batch_days = business_days[i:i+batch_size]
            
            # バッチ内の日付を並列処理
            tasks = [
                self.get_stocks_for_date(date, session)
                for date in batch_days
            ]
            
            results = await asyncio.gather(*tasks)
            
            # 結果を辞書に格納
            for date, df in zip(batch_days, results):
                if not df.is_empty():
                    daily_listings[date] = df
                    logger.info(f"{date}: {len(df)}銘柄")
                else:
                    logger.warning(f"{date}: 銘柄データなし")
                
                if progress_callback:
                    current_progress = min(i + batch_days.index(date) + 1, total_days)
                    progress_callback(current_progress, total_days)
        
        return daily_listings
    
    def get_statistics(self, daily_listings: Dict[str, pl.DataFrame]) -> Dict:
        """統計情報を計算"""
        if not daily_listings:
            return {}
        
        dates = sorted(daily_listings.keys())
        counts = [len(df) for df in daily_listings.values()]
        
        # 新規上場・廃止の検出
        all_codes = set()
        new_listings = []
        delistings = []
        
        prev_codes = set()
        for date in dates:
            current_codes = set(daily_listings[date]["Code"].to_list())
            
            # 新規上場
            new = current_codes - prev_codes
            if new and prev_codes:  # 初日は除外
                new_listings.append((date, len(new)))
            
            # 廃止
            delisted = prev_codes - current_codes
            if delisted and prev_codes:
                delistings.append((date, len(delisted)))
            
            all_codes.update(current_codes)
            prev_codes = current_codes
        
        return {
            "total_unique_stocks": len(all_codes),
            "min_daily_stocks": min(counts),
            "max_daily_stocks": max(counts),
            "avg_daily_stocks": sum(counts) / len(counts),
            "new_listings": new_listings,
            "delistings": delistings,
            "first_date": dates[0],
            "last_date": dates[-1],
            "total_days": len(dates)
        }


async def test_daily_fetcher():
    """テスト関数"""
    import os
    from dotenv import load_dotenv
    
    # 環境変数読み込み
    load_dotenv()
    
    # 簡易認証クラス
    class SimpleAPIClient:
        def __init__(self):
            self.id_token = None
            
        async def authenticate(self, session):
            email = os.getenv("JQUANTS_AUTH_EMAIL")
            password = os.getenv("JQUANTS_AUTH_PASSWORD")
            
            if not email or not password:
                raise Exception("認証情報が設定されていません")
            
            # 認証処理（省略）
            # ...
            self.id_token = "test_token"
    
    # テスト実行
    api_client = SimpleAPIClient()
    fetcher = DailyStockFetcher(api_client)
    
    # 1週間分のテスト
    business_days = [
        "2025-01-06",
        "2025-01-07", 
        "2025-01-08",
        "2025-01-09",
        "2025-01-10"
    ]
    
    async with aiohttp.ClientSession() as session:
        await api_client.authenticate(session)
        
        daily_listings = await fetcher.get_stocks_for_period(
            business_days,
            session
        )
        
        # 統計表示
        stats = fetcher.get_statistics(daily_listings)
        print(f"期間中のユニーク銘柄数: {stats['total_unique_stocks']}")
        print(f"日次平均銘柄数: {stats['avg_daily_stocks']:.0f}")
        
        if stats['new_listings']:
            print(f"新規上場: {stats['new_listings']}")
        if stats['delistings']:
            print(f"上場廃止: {stats['delistings']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_daily_fetcher())