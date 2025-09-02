#!/usr/bin/env python3
"""
DailyQuotesByCodeFetcher - 銘柄軸でのdaily_quotes取得
市場membership期間を考慮して、効率的に株価データを取得
"""

import asyncio
import aiohttp
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)


class MarketMembership:
    """銘柄の市場所属期間を管理"""
    
    def __init__(self):
        # {code: [(market_code, from_date, to_date), ...]}
        self.memberships: Dict[str, List[Tuple[str, str, Optional[str]]]] = {}
        
    def add_membership(
        self,
        code: str,
        market_code: str,
        from_date: str,
        to_date: Optional[str] = None
    ):
        """市場所属期間を追加"""
        if code not in self.memberships:
            self.memberships[code] = []
        
        # 既存の期間と重複チェック
        for i, (mc, fd, td) in enumerate(self.memberships[code]):
            if mc == market_code and fd == from_date:
                # 同じ市場・開始日なら更新
                self.memberships[code][i] = (market_code, from_date, to_date)
                return
        
        self.memberships[code].append((market_code, from_date, to_date))
        
    def get_ranges_for_code(
        self,
        code: str,
        target_markets: Optional[List[str]] = None
    ) -> List[Tuple[str, str]]:
        """
        銘柄の取得すべき期間レンジを返す
        
        Args:
            code: 銘柄コード
            target_markets: 対象市場コードリスト（Noneなら全市場）
            
        Returns:
            [(from_date, to_date), ...] のリスト
        """
        if code not in self.memberships:
            return []
        
        ranges = []
        for market_code, from_date, to_date in self.memberships[code]:
            # 市場フィルタリング
            if target_markets and market_code not in target_markets:
                continue
            
            # to_dateがNoneなら今日まで
            if to_date is None:
                to_date = datetime.now().strftime("%Y-%m-%d")
            
            ranges.append((from_date, to_date))
        
        # 重複期間をマージ
        return self._merge_overlapping_ranges(ranges)
    
    def _merge_overlapping_ranges(
        self,
        ranges: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """重複する期間をマージ"""
        if not ranges:
            return []
        
        # ソート
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        
        merged = [sorted_ranges[0]]
        for current_from, current_to in sorted_ranges[1:]:
            last_from, last_to = merged[-1]
            
            # 重複または連続する場合はマージ
            if current_from <= last_to:
                # より長い終了日を採用
                merged[-1] = (last_from, max(last_to, current_to))
            else:
                merged.append((current_from, current_to))
        
        return merged


class DailyQuotesByCodeFetcher:
    """銘柄軸でdaily_quotesを取得"""
    
    def __init__(self, api_client, cache_dir: Path = None):
        self.api_client = api_client
        self.base_url = "https://api.jquants.com/v1"
        
        # キャッシュ設定
        self.cache_dir = cache_dir or Path("cache/daily_quotes_by_code")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 市場membership管理
        self.market_membership = MarketMembership()
        
    def normalize_code(self, code: str) -> str:
        """コードを5桁LocalCodeに正規化"""
        # 4桁コードを5桁に変換（末尾0追加）
        if len(code) == 4:
            return code + "0"
        return code
    
    async def fetch_by_code(
        self,
        session: aiohttp.ClientSession,
        code: str,
        from_date: str,
        to_date: str,
        use_cache: bool = True
    ) -> pl.DataFrame:
        """
        特定銘柄の期間データを取得
        
        Args:
            session: aiohttp ClientSession
            code: 銘柄コード（4桁）
            from_date: 開始日 (YYYY-MM-DD)
            to_date: 終了日 (YYYY-MM-DD)
            use_cache: キャッシュを使用するか
            
        Returns:
            株価データのDataFrame
        """
        # キャッシュチェック
        if use_cache:
            cached_df = self._load_cache(code, from_date, to_date)
            if cached_df is not None:
                return cached_df
        
        logger.debug(f"Fetching {code}: {from_date} to {to_date}")
        
        url = f"{self.base_url}/prices/daily_quotes"
        headers = {"Authorization": f"Bearer {self.api_client.id_token}"}
        
        from_api = from_date.replace("-", "")
        to_api = to_date.replace("-", "")
        
        all_quotes = []
        pagination_key = None
        page_count = 0
        
        while True:
            params = {
                "code": code,
                "from": from_api,
                "to": to_api
            }
            if pagination_key:
                params["pagination_key"] = pagination_key
            
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.debug(f"No quotes for {code}: {response.status}")
                        break
                    
                    data = await response.json()
                    quotes = data.get("daily_quotes", [])
                    
                    if quotes:
                        all_quotes.extend(quotes)
                        page_count += 1
                    
                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break
                        
            except Exception as e:
                logger.error(f"Error fetching quotes for {code}: {e}")
                break
        
        if all_quotes:
            df = pl.DataFrame(all_quotes)
            
            # LocalCodeカラムを追加（5桁）
            df = df.with_columns([
                pl.lit(self.normalize_code(code)).alias("LocalCode")
            ])
            
            # Adjustment列があれば優先使用
            df = self._use_adjustment_columns(df)
            
            # 数値型に変換
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
            
            # Date列の処理
            if "Date" in df.columns:
                df = df.with_columns(
                    pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
                )
            
            # ソート
            df = df.sort("Date")
            
            logger.info(f"  {code}: {len(df)} records fetched ({page_count} pages)")
            
            # キャッシュ保存
            if use_cache and not df.is_empty():
                self._save_cache(df, code, from_date, to_date)
            
            return df
        
        return pl.DataFrame()
    
    async def fetch_by_membership(
        self,
        session: aiohttp.ClientSession,
        code: str,
        target_markets: Optional[List[str]] = None,
        batch_size: int = 5
    ) -> pl.DataFrame:
        """
        市場membership期間に基づいて銘柄データを取得
        
        Args:
            session: aiohttp ClientSession
            code: 銘柄コード
            target_markets: 対象市場コードリスト
            batch_size: 並列処理数
            
        Returns:
            全期間の統合DataFrame
        """
        ranges = self.market_membership.get_ranges_for_code(code, target_markets)
        
        if not ranges:
            logger.debug(f"No membership ranges for {code}")
            return pl.DataFrame()
        
        all_dfs = []
        
        # 期間ごとに取得
        for from_date, to_date in ranges:
            df = await self.fetch_by_code(session, code, from_date, to_date)
            if not df.is_empty():
                all_dfs.append(df)
        
        if all_dfs:
            # 全期間を結合
            combined_df = pl.concat(all_dfs)
            
            # 重複除去（Date, LocalCodeで一意）
            combined_df = combined_df.unique(subset=["Date", "LocalCode"])
            
            # ソート
            combined_df = combined_df.sort(["LocalCode", "Date"])
            
            return combined_df
        
        return pl.DataFrame()
    
    async def fetch_multiple_codes(
        self,
        session: aiohttp.ClientSession,
        codes: List[str],
        from_date: str,
        to_date: str,
        batch_size: int = 10,
        progress_callback=None
    ) -> pl.DataFrame:
        """
        複数銘柄を並列で取得
        
        Args:
            session: aiohttp ClientSession
            codes: 銘柄コードリスト
            from_date: 開始日
            to_date: 終了日
            batch_size: 並列処理数
            progress_callback: 進捗コールバック
            
        Returns:
            全銘柄の統合DataFrame
        """
        all_dfs = []
        total_codes = len(codes)
        
        for i in range(0, total_codes, batch_size):
            batch_codes = codes[i:i+batch_size]
            
            # バッチ内並列処理
            tasks = [
                self.fetch_by_code(session, code, from_date, to_date)
                for code in batch_codes
            ]
            
            results = await asyncio.gather(*tasks)
            
            for code, df in zip(batch_codes, results):
                if not df.is_empty():
                    all_dfs.append(df)
                
                if progress_callback:
                    current_progress = min(i + batch_codes.index(code) + 1, total_codes)
                    progress_callback(current_progress, total_codes)
        
        if all_dfs:
            # 全銘柄を結合
            combined_df = pl.concat(all_dfs)
            return combined_df.sort(["Code", "Date"])
        
        return pl.DataFrame()
    
    def _use_adjustment_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Adjustment列があれば優先使用"""
        columns_to_rename = {}
        
        if "AdjustmentClose" in df.columns:
            columns_to_rename["AdjustmentClose"] = "Close"
            if "Close" in df.columns:
                df = df.drop("Close")
        
        if "AdjustmentOpen" in df.columns:
            columns_to_rename["AdjustmentOpen"] = "Open"
            if "Open" in df.columns:
                df = df.drop("Open")
        
        if "AdjustmentHigh" in df.columns:
            columns_to_rename["AdjustmentHigh"] = "High"
            if "High" in df.columns:
                df = df.drop("High")
        
        if "AdjustmentLow" in df.columns:
            columns_to_rename["AdjustmentLow"] = "Low"
            if "Low" in df.columns:
                df = df.drop("Low")
        
        if "AdjustmentVolume" in df.columns:
            columns_to_rename["AdjustmentVolume"] = "Volume"
            if "Volume" in df.columns:
                df = df.drop("Volume")
        
        if columns_to_rename:
            df = df.rename(columns_to_rename)
        
        return df
    
    def _load_cache(
        self,
        code: str,
        from_date: str,
        to_date: str
    ) -> Optional[pl.DataFrame]:
        """キャッシュから読み込み"""
        cache_file = self.cache_dir / f"{code}_{from_date}_{to_date}.parquet"
        
        if cache_file.exists():
            try:
                df = pl.read_parquet(cache_file)
                logger.debug(f"Loaded from cache: {code} ({from_date} to {to_date})")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {code}: {e}")
        
        return None
    
    def _save_cache(
        self,
        df: pl.DataFrame,
        code: str,
        from_date: str,
        to_date: str
    ):
        """キャッシュに保存"""
        cache_file = self.cache_dir / f"{code}_{from_date}_{to_date}.parquet"
        
        try:
            df.write_parquet(cache_file)
            logger.debug(f"Saved to cache: {code} ({from_date} to {to_date})")
        except Exception as e:
            logger.warning(f"Failed to save cache for {code}: {e}")
    
    def update_membership_from_events(
        self,
        events: List[Dict]
    ):
        """
        イベントデータから市場membershipを更新
        
        Args:
            events: イベントリスト
                [{"code": "1234", "event_type": "listing", 
                  "market_code": "0111", "effective_from": "2021-01-04"}, ...]
        """
        for event in events:
            code = event["code"]
            event_type = event["event_type"]
            
            if event_type == "listing":
                # 新規上場
                self.market_membership.add_membership(
                    code,
                    event["market_code"],
                    event["effective_from"],
                    None  # 現在も上場中
                )
            
            elif event_type == "delisting":
                # 上場廃止
                # 最後のmembershipを終了
                if code in self.market_membership.memberships:
                    for i, (mc, fd, td) in enumerate(self.market_membership.memberships[code]):
                        if td is None:  # 現在有効なmembership
                            self.market_membership.memberships[code][i] = (
                                mc, fd, event["effective_from"]
                            )
            
            elif event_type == "market_change":
                # 市場変更
                old_market = event.get("old_market")
                new_market = event["new_market"]
                change_date = event["effective_from"]
                
                # 旧市場の期間を終了
                if code in self.market_membership.memberships:
                    for i, (mc, fd, td) in enumerate(self.market_membership.memberships[code]):
                        if mc == old_market and td is None:
                            self.market_membership.memberships[code][i] = (
                                mc, fd, change_date
                            )
                
                # 新市場の期間を開始
                self.market_membership.add_membership(
                    code,
                    new_market,
                    change_date,
                    None
                )


async def test_fetcher():
    """テスト関数"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # 簡易APIクライアント
    class SimpleAPIClient:
        def __init__(self):
            self.id_token = None
            
        async def authenticate(self, session):
            email = os.getenv("JQUANTS_AUTH_EMAIL")
            password = os.getenv("JQUANTS_AUTH_PASSWORD")
            
            if not email or not password:
                raise Exception("認証情報が設定されていません")
            
            auth_url = "https://api.jquants.com/v1/token/auth_user"
            auth_payload = {"mailaddress": email, "password": password}
            
            async with session.post(auth_url, json=auth_payload) as response:
                if response.status != 200:
                    raise Exception(f"Auth failed: {response.status}")
                data = await response.json()
                refresh_token = data["refreshToken"]
            
            refresh_url = "https://api.jquants.com/v1/token/auth_refresh"
            params = {"refreshtoken": refresh_token}
            
            async with session.post(refresh_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get ID token: {response.status}")
                data = await response.json()
                self.id_token = data["idToken"]
    
    # テスト実行
    api_client = SimpleAPIClient()
    fetcher = DailyQuotesByCodeFetcher(api_client)
    
    # テスト用のmembership設定
    fetcher.market_membership.add_membership("7203", "0111", "2022-04-04", None)  # トヨタ、プライム
    
    async with aiohttp.ClientSession() as session:
        await api_client.authenticate(session)
        
        # 単一銘柄取得テスト
        df = await fetcher.fetch_by_code(
            session,
            "7203",
            "2025-01-01",
            "2025-01-10"
        )
        
        if not df.is_empty():
            print(f"Fetched {len(df)} records for 7203")
            print(df.head())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_fetcher())