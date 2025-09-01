#!/usr/bin/env python3
"""
Trading Calendar Fetcher - J-Quants営業日カレンダー取得
営業日、休日、半休日を管理し、正確な取引日のみでデータ取得を行う
"""

import asyncio
import aiohttp
import polars as pl
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


class TradingCalendarFetcher:
    """営業日カレンダー取得・管理クラス"""
    
    def __init__(self, api_client=None):
        self.base_url = "https://api.jquants.com/v1"
        self.api_client = api_client
        self.cache_dir = Path("cache/trading_calendar")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    async def get_trading_calendar(
        self, 
        from_date: str, 
        to_date: str,
        session: aiohttp.ClientSession,
        use_cache: bool = True
    ) -> Dict[str, List[str]]:
        """
        営業日カレンダーを取得
        
        Args:
            from_date: 開始日 (YYYY-MM-DD)
            to_date: 終了日 (YYYY-MM-DD)
            session: aiohttp ClientSession
            use_cache: キャッシュを使用するか
            
        Returns:
            {
                "business_days": ["2021-01-04", "2021-01-05", ...],
                "holidays": ["2021-01-01", "2021-01-02", ...],
                "half_days": ["2021-12-30", ...]
            }
        """
        # キャッシュチェック
        if use_cache:
            cached_data = self._load_cache(from_date, to_date)
            if cached_data:
                logger.info(f"キャッシュから営業日カレンダーを読み込み: {from_date} - {to_date}")
                return cached_data
        
        logger.info(f"営業日カレンダーを取得中: {from_date} - {to_date}")
        
        # APIから取得
        url = f"{self.base_url}/markets/trading_calendar"
        headers = {"Authorization": f"Bearer {self.api_client.id_token}"}
        
        all_calendar_data = []
        
        # 日付形式をAPIの要求形式に変換 (YYYY-MM-DD -> YYYYMMDD)
        from_date_api = from_date.replace("-", "")
        to_date_api = to_date.replace("-", "")
        
        # HolidayDivision: 0=営業日, 1=休日, 2=祝日, 3=年末年始, 4=半休日
        # holidaydivisionを指定しないことで、指定期間の全データを取得
        params = {
            "from": from_date_api,
            "to": to_date_api
        }
        
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"Trading calendar API failed: {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
                    raise Exception(f"Trading calendar API failed: {response.status}")
                
                data = await response.json()
                calendar_data = data.get("trading_calendar", [])
                
                if calendar_data:
                    all_calendar_data.extend(calendar_data)
                    
        except Exception as e:
            logger.error(f"Trading calendar取得エラー: {e}")
            raise
        
        # カレンダーデータを分類
        # HolidayDivision: 0=非営業日, 1=営業日, 2=東証半日立会日, 3=非営業日(祝日取引あり)
        business_days = []
        holidays = []
        half_days = []
        special_trading_days = []
        
        for day in all_calendar_data:
            date = day.get("Date")
            holiday_division = str(day.get("HolidayDivision", "0"))  # 文字列に統一
            
            if holiday_division == "1":  # 営業日
                business_days.append(date)
            elif holiday_division == "2":  # 東証半日立会日
                half_days.append(date)
                business_days.append(date)  # 半日でも取引があるので営業日に含める
            elif holiday_division == "3":  # 非営業日(祝日取引あり)
                special_trading_days.append(date)
                business_days.append(date)  # 祝日取引も営業日に含める
            elif holiday_division == "0":  # 非営業日
                holidays.append(date)
        
        result = {
            "business_days": sorted(business_days),
            "holidays": sorted(holidays),
            "half_days": sorted(half_days),
            "from_date": from_date,
            "to_date": to_date,
            "fetched_at": datetime.now().isoformat()
        }
        
        logger.info(f"営業日: {len(business_days)}日, 休日: {len(holidays)}日, 半休日: {len(half_days)}日")
        
        # キャッシュ保存
        if use_cache:
            self._save_cache(result, from_date, to_date)
        
        return result
    
    def _load_cache(self, from_date: str, to_date: str) -> Optional[Dict]:
        """キャッシュから読み込み"""
        cache_file = self.cache_dir / f"calendar_{from_date}_{to_date}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    
                # キャッシュの有効期限チェック（24時間）
                cached_time = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
                if (datetime.now() - cached_time).total_seconds() < 86400:  # 24時間
                    return data
                    
            except Exception as e:
                logger.warning(f"キャッシュ読み込みエラー: {e}")
        
        return None
    
    def _save_cache(self, data: Dict, from_date: str, to_date: str):
        """キャッシュに保存"""
        cache_file = self.cache_dir / f"calendar_{from_date}_{to_date}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"キャッシュ保存: {cache_file}")
        except Exception as e:
            logger.warning(f"キャッシュ保存エラー: {e}")
    
    @staticmethod
    def filter_business_days_only(dates: List[str], business_days: List[str]) -> List[str]:
        """営業日のみフィルタリング"""
        business_set = set(business_days)
        return [date for date in dates if date in business_set]
    
    @staticmethod
    def get_business_days_df(calendar_data: Dict) -> pl.DataFrame:
        """営業日のDataFrameを作成"""
        business_days = calendar_data.get("business_days", [])
        
        if not business_days:
            return pl.DataFrame()
        
        return pl.DataFrame({
            "date": business_days,
            "is_business_day": [True] * len(business_days)
        }).with_columns([
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
        ])


async def test_trading_calendar():
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
            """認証処理（既存のコードから流用）"""
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
            
            logger.info("認証成功")
    
    # テスト実行
    api_client = SimpleAPIClient()
    fetcher = TradingCalendarFetcher(api_client)
    
    async with aiohttp.ClientSession() as session:
        # 認証
        await api_client.authenticate(session)
        
        # 営業日カレンダー取得（1ヶ月分のテスト）
        calendar_data = await fetcher.get_trading_calendar(
            "2025-01-01",
            "2025-01-31",
            session
        )
        
        print(f"営業日数: {len(calendar_data['business_days'])}")
        print(f"最初の5営業日: {calendar_data['business_days'][:5]}")
        print(f"休日数: {len(calendar_data['holidays'])}")
        print(f"半休日数: {len(calendar_data['half_days'])}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_trading_calendar())