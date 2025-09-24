#!/usr/bin/env python3
"""
Trading Calendar Fetcher - J-Quants営業日カレンダー取得
営業日、休日、半休日を管理し、正確な取引日のみでデータ取得を行う
"""

import asyncio
import aiohttp
import json
import logging
import os
import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

logger = logging.getLogger(__name__)


class TradingCalendarFetcher:
    """営業日カレンダー取得・管理クラス"""
    
    def __init__(self, api_client=None):
        self.base_url = "https://api.jquants.com/v1"
        self.api_client = api_client
        self.cache_dir = Path("cache/trading_calendar")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.subscription_start, self.subscription_end = self._load_subscription_bounds()

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
        original_range = (from_date, to_date)
        effective_from, effective_to, env_adjusted = self._clamp_requested_range(
            from_date, to_date
        )

        # キャッシュチェック（要求区間と補正区間の両方）
        cache_candidates = [original_range]
        effective_range = (effective_from, effective_to)
        if effective_range not in cache_candidates:
            cache_candidates.append(effective_range)

        if use_cache:
            for cache_from, cache_to in cache_candidates:
                cached_data = self._load_cache(cache_from, cache_to)
                if cached_data:
                    logger.info(
                        "キャッシュから営業日カレンダーを読み込み: %s - %s (requested %s - %s)",
                        cache_from,
                        cache_to,
                        from_date,
                        to_date,
                    )
                    return cached_data

        if env_adjusted:
            logger.warning(
                "J-Quantsの契約範囲に合わせて営業日カレンダーの開始/終了日を補正しました: %s - %s (指定: %s - %s)",
                effective_from,
                effective_to,
                from_date,
                to_date,
            )

        logger.info(
            "営業日カレンダーを取得中: %s - %s (requested %s - %s)",
            effective_from,
            effective_to,
            from_date,
            to_date,
        )

        url = f"{self.base_url}/markets/trading_calendar"
        headers = {"Authorization": f"Bearer {self.api_client.id_token}"}

        all_calendar_data: list[dict] = []
        attempt_from = effective_from
        attempt_to = effective_to
        max_retries = 2
        retries = 0

        while True:
            params = {"from": attempt_from.replace("-", ""), "to": attempt_to.replace("-", "")}
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        calendar_data = data.get("trading_calendar", [])
                        if calendar_data:
                            all_calendar_data.extend(calendar_data)
                        break

                    text = await response.text()
                    logger.error(f"Trading calendar API failed: {response.status}")
                    logger.error(f"Response: {text}")

                    if response.status == 400 and retries < max_retries:
                        bounds_updated = self._update_subscription_bounds_from_error(text)
                        new_from, new_to, adjusted = self._clamp_requested_range(
                            attempt_from, attempt_to
                        )
                        if bounds_updated or adjusted:
                            retries += 1
                            attempt_from, attempt_to = new_from, new_to
                            logger.warning(
                                "契約範囲に合わせて再試行します: %s - %s", attempt_from, attempt_to
                            )
                            continue

                    raise Exception(f"Trading calendar API failed: {response.status}")

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
            "from_date": attempt_from,
            "to_date": attempt_to,
            "requested_from_date": from_date,
            "requested_to_date": to_date,
            "subscription_start": self.subscription_start.isoformat()
            if self.subscription_start
            else None,
            "subscription_end": self.subscription_end.isoformat()
            if self.subscription_end
            else None,
            "fetched_at": datetime.now().isoformat(),
        }

        logger.info(f"営業日: {len(business_days)}日, 休日: {len(holidays)}日, 半休日: {len(half_days)}日")

        # キャッシュ保存
        if use_cache:
            # 要求区間と実際に取得した区間の両方でキャッシュを保存
            self._save_cache(result, from_date, to_date)
            if (from_date, to_date) != (attempt_from, attempt_to):
                self._save_cache(result, attempt_from, attempt_to)

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

    def _load_subscription_bounds(self) -> Tuple[Optional[date], Optional[date]]:
        """環境変数から契約範囲を読み込む。"""
        start_candidates = [
            os.getenv("JQUANTS_SUBSCRIPTION_START"),
            os.getenv("JQUANTS_MIN_API_DATE"),
            os.getenv("JQUANTS_PLAN_START_DATE"),
        ]
        end_candidates = [
            os.getenv("JQUANTS_SUBSCRIPTION_END"),
            os.getenv("JQUANTS_MAX_API_DATE"),
        ]

        start_dates = [d for d in (self._parse_date(v) for v in start_candidates) if d]
        end_dates = [d for d in (self._parse_date(v) for v in end_candidates) if d]

        start = max(start_dates) if start_dates else None
        end = min(end_dates) if end_dates else None
        return start, end

    def _clamp_requested_range(
        self, from_date: str, to_date: str
    ) -> Tuple[str, str, bool]:
        """契約範囲に合わせて日付レンジを補正する。"""
        start_dt = self._parse_date(from_date)
        end_dt = self._parse_date(to_date)
        if not start_dt or not end_dt:
            raise ValueError("from_date/to_date must be YYYY-MM-DD")

        adjusted = False
        if self.subscription_start and start_dt < self.subscription_start:
            start_dt = self.subscription_start
            adjusted = True
        if self.subscription_end and end_dt > self.subscription_end:
            end_dt = self.subscription_end
            adjusted = True

        if start_dt > end_dt:
            raise ValueError(
                "Requested trading calendar range is outside subscription coverage"
            )

        return start_dt.isoformat(), end_dt.isoformat(), adjusted

    def _update_subscription_bounds_from_error(self, message: str) -> bool:
        """エラーメッセージから契約範囲を更新する。"""
        start, end = self._extract_subscription_bounds(message)
        updated = False
        if start:
            if not self.subscription_start or start > self.subscription_start:
                self.subscription_start = start
                updated = True
        if end:
            if not self.subscription_end or end < self.subscription_end:
                self.subscription_end = end
                updated = True
        return updated

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        try:
            cleaned = value.strip()
            if not cleaned:
                return None
            if len(cleaned) == 8 and cleaned.isdigit():
                cleaned = f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:]}"
            return datetime.strptime(cleaned, "%Y-%m-%d").date()
        except ValueError:
            return None

    @classmethod
    def _extract_subscription_bounds(
        cls, text: str
    ) -> Tuple[Optional[date], Optional[date]]:
        """エラーテキストから契約開始・終了日を抽出する。"""
        if not text:
            return None, None

        try:
            payload = json.loads(text)
            message = payload.get("message", "") if isinstance(payload, dict) else ""
        except json.JSONDecodeError:
            message = text

        matches = re.findall(r"20\d{2}-\d{2}-\d{2}", message)
        if not matches:
            return None, None

        start = cls._parse_date(matches[0])
        end = cls._parse_date(matches[1]) if len(matches) > 1 else None
        return start, end


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
