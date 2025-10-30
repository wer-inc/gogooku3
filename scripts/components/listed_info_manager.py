#!/usr/bin/env python3
"""
ListedInfoManager - 月初＋差分日の効率的な銘柄リスト管理
上場/廃止/市場変更を日単位で正確に検知し、APIコール数を最小化
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import aiohttp
import polars as pl

logger = logging.getLogger(__name__)


class ListedInfoManager:
    """月初＋差分日による効率的な銘柄リスト管理"""

    def __init__(self, api_client, cache_dir: Path = None):
        self.api_client = api_client
        self.base_url = "https://api.jquants.com/v1"

        # キャッシュ設定
        self.cache_dir = cache_dir or Path("cache/listed_info")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # スナップショット保存用
        self.snapshots_dir = self.cache_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)

        # 差分保存用
        self.diffs_dir = self.cache_dir / "diffs"
        self.diffs_dir.mkdir(exist_ok=True)

        # イベント保存用
        self.events_dir = self.cache_dir / "events"
        self.events_dir.mkdir(exist_ok=True)

    async def get_snapshot_at(
        self,
        session: aiohttp.ClientSession,
        date: str,
        use_cache: bool = True
    ) -> pl.DataFrame:
        """
        特定日の銘柄スナップショットを取得

        Args:
            session: aiohttp ClientSession
            date: 対象日 (YYYY-MM-DD)
            use_cache: キャッシュを使用するか

        Returns:
            銘柄リストのDataFrame
        """
        # キャッシュチェック
        if use_cache:
            cached_df = self._load_snapshot_cache(date)
            if cached_df is not None:
                return cached_df

        logger.info(f"Fetching listed info snapshot for {date}")

        url = f"{self.base_url}/listed/info"
        headers = {"Authorization": f"Bearer {self.api_client.id_token}"}
        date_api = date.replace("-", "")
        params = {"date": date_api}

        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch listed info for {date}: {response.status}")
                    return pl.DataFrame()

                data = await response.json()
                info_list = data.get("info", [])

                if not info_list:
                    logger.warning(f"No listed info for {date}")
                    return pl.DataFrame()

                df = pl.DataFrame(info_list)

                # 5桁LocalCodeを追加
                if "Code" in df.columns:
                    df = df.with_columns([
                        (pl.col("Code").cast(pl.Utf8).str.zfill(4) + "0").alias("LocalCode")
                    ])

                logger.info(f"  {date}: {len(df)} stocks in snapshot")

                # キャッシュ保存
                if use_cache and not df.is_empty():
                    self._save_snapshot_cache(df, date)

                return df

        except Exception as e:
            logger.error(f"Error fetching listed info for {date}: {e}")
            return pl.DataFrame()

    async def get_monthly_snapshots(
        self,
        session: aiohttp.ClientSession,
        business_days: list[str]
    ) -> dict[str, pl.DataFrame]:
        """
        月初営業日のスナップショットを取得

        Args:
            session: aiohttp ClientSession
            business_days: 営業日リスト

        Returns:
            {date: DataFrame} の辞書
        """
        month_starts = self._get_month_start_days(business_days)
        snapshots = {}

        logger.info(f"Fetching {len(month_starts)} monthly snapshots...")

        for date in month_starts:
            df = await self.get_snapshot_at(session, date)
            if not df.is_empty():
                snapshots[date] = df

        return snapshots

    def detect_changes(
        self,
        prev_snapshot: pl.DataFrame,
        curr_snapshot: pl.DataFrame,
        date: str
    ) -> dict[str, list]:
        """
        2つのスナップショット間の変化を検知

        Args:
            prev_snapshot: 前回のスナップショット
            curr_snapshot: 今回のスナップショット
            date: 変化検知日

        Returns:
            {"new_listings": [...], "delistings": [...],
             "market_changes": [...], "name_changes": [...]}
        """
        changes = {
            "new_listings": [],
            "delistings": [],
            "market_changes": [],
            "name_changes": []
        }

        if prev_snapshot.is_empty() or curr_snapshot.is_empty():
            return changes

        # コード集合の比較
        prev_codes = set(prev_snapshot["Code"].to_list())
        curr_codes = set(curr_snapshot["Code"].to_list())

        # 新規上場
        new_codes = curr_codes - prev_codes
        for code in new_codes:
            row = curr_snapshot.filter(pl.col("Code") == code).row(0, named=True)
            changes["new_listings"].append({
                "code": code,
                "market_code": row.get("MarketCode"),
                "company_name": row.get("CompanyName"),
                "effective_from": date
            })

        # 上場廃止
        delisted_codes = prev_codes - curr_codes
        for code in delisted_codes:
            row = prev_snapshot.filter(pl.col("Code") == code).row(0, named=True)
            changes["delistings"].append({
                "code": code,
                "last_market_code": row.get("MarketCode"),
                "company_name": row.get("CompanyName"),
                "effective_from": date
            })

        # 継続銘柄の属性変化をチェック
        common_codes = prev_codes & curr_codes
        for code in common_codes:
            prev_row = prev_snapshot.filter(pl.col("Code") == code).row(0, named=True)
            curr_row = curr_snapshot.filter(pl.col("Code") == code).row(0, named=True)

            # 市場変更
            if prev_row.get("MarketCode") != curr_row.get("MarketCode"):
                changes["market_changes"].append({
                    "code": code,
                    "old_market": prev_row.get("MarketCode"),
                    "new_market": curr_row.get("MarketCode"),
                    "effective_from": date
                })

            # 社名変更
            if prev_row.get("CompanyName") != curr_row.get("CompanyName"):
                changes["name_changes"].append({
                    "code": code,
                    "old_name": prev_row.get("CompanyName"),
                    "new_name": curr_row.get("CompanyName"),
                    "effective_from": date
                })

        return changes

    async def binary_search_event_date(
        self,
        session: aiohttp.ClientSession,
        start_date: str,
        end_date: str,
        predicate_func,
        business_days: list[str]
    ) -> str | None:
        """
        二分探索でイベント発生日を特定

        Args:
            session: aiohttp ClientSession
            start_date: 探索開始日
            end_date: 探索終了日
            predicate_func: 判定関数（DataFrameを受け取りboolを返す）
            business_days: 営業日リスト

        Returns:
            イベント発生日（見つからない場合はNone）
        """
        # 期間内の営業日を抽出
        days_in_range = [d for d in business_days if start_date <= d <= end_date]

        if not days_in_range:
            return None

        logger.info(f"Binary searching event date in {len(days_in_range)} days...")

        left, right = 0, len(days_in_range) - 1
        result_date = None

        while left <= right:
            mid = (left + right) // 2
            mid_date = days_in_range[mid]

            # キャッシュまたはAPIから取得
            df = await self.get_snapshot_at(session, mid_date)

            if df.is_empty():
                # データなしの場合は次へ
                left = mid + 1
                continue

            if predicate_func(df):
                # 条件を満たす最初の日を探す
                result_date = mid_date
                right = mid - 1
            else:
                left = mid + 1

        if result_date:
            logger.info(f"  Event date found: {result_date}")
        else:
            logger.info("  Event date not found in range")

        return result_date

    async def detect_all_events(
        self,
        session: aiohttp.ClientSession,
        business_days: list[str],
        use_binary_search: bool = True
    ) -> list[dict]:
        """
        全期間のイベントを検知

        Args:
            session: aiohttp ClientSession
            business_days: 営業日リスト
            use_binary_search: 二分探索を使用するか

        Returns:
            イベントリスト
        """
        all_events = []

        # 月初スナップショットを取得
        snapshots = await self.get_monthly_snapshots(session, business_days)
        sorted_dates = sorted(snapshots.keys())

        logger.info(f"Detecting events across {len(sorted_dates)} monthly snapshots...")

        for i in range(1, len(sorted_dates)):
            prev_date = sorted_dates[i-1]
            curr_date = sorted_dates[i]

            prev_snapshot = snapshots[prev_date]
            curr_snapshot = snapshots[curr_date]

            # 月次差分を検出
            changes = self.detect_changes(prev_snapshot, curr_snapshot, curr_date)

            # 二分探索で正確な日付を特定
            if use_binary_search:
                # 新規上場の正確な日付を特定
                for listing in changes["new_listings"]:
                    code = listing["code"]

                    def has_code(df):
                        return code in df["Code"].to_list()

                    exact_date = await self.binary_search_event_date(
                        session, prev_date, curr_date, has_code, business_days
                    )

                    if exact_date:
                        listing["effective_from"] = exact_date
                        listing["event_type"] = "listing"
                        all_events.append(listing)

                # 上場廃止の正確な日付を特定
                for delisting in changes["delistings"]:
                    code = delisting["code"]

                    def has_no_code(df):
                        return code not in df["Code"].to_list()

                    exact_date = await self.binary_search_event_date(
                        session, prev_date, curr_date, has_no_code, business_days
                    )

                    if exact_date:
                        delisting["effective_from"] = exact_date
                        delisting["event_type"] = "delisting"
                        all_events.append(delisting)

                # 市場変更・社名変更も同様に処理
                for change in changes["market_changes"]:
                    change["event_type"] = "market_change"
                    all_events.append(change)

                for change in changes["name_changes"]:
                    change["event_type"] = "name_change"
                    all_events.append(change)
            else:
                # 二分探索なしの場合は月初日をそのまま使用
                for event_type, events in changes.items():
                    for event in events:
                        event["event_type"] = event_type.rstrip("s")  # 複数形を単数形に
                        all_events.append(event)

        logger.info(f"✅ Total {len(all_events)} events detected")

        return all_events

    def save_events(self, events: list[dict], filename: str = None):
        """イベントを保存"""
        if not filename:
            filename = f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.events_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(events, f, indent=2, ensure_ascii=False)
            logger.info(f"Events saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save events: {e}")

    def _get_month_start_days(self, business_days: list[str]) -> list[str]:
        """月初営業日を抽出"""
        month_starts = []
        current_month = None

        for date in sorted(business_days):
            month = date[:7]  # YYYY-MM
            if month != current_month:
                month_starts.append(date)
                current_month = month

        return month_starts

    def _load_snapshot_cache(self, date: str) -> pl.DataFrame | None:
        """スナップショットキャッシュを読み込み"""
        cache_file = self.snapshots_dir / f"snapshot_{date}.parquet"

        if cache_file.exists():
            try:
                df = pl.read_parquet(cache_file)
                logger.debug(f"Loaded snapshot from cache: {date}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load snapshot cache for {date}: {e}")

        return None

    def _save_snapshot_cache(self, df: pl.DataFrame, date: str):
        """スナップショットキャッシュを保存"""
        cache_file = self.snapshots_dir / f"snapshot_{date}.parquet"

        try:
            df.write_parquet(cache_file)
            logger.debug(f"Saved snapshot to cache: {date}")
        except Exception as e:
            logger.warning(f"Failed to save snapshot cache for {date}: {e}")


async def test_manager():
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
    manager = ListedInfoManager(api_client)

    # テスト用の営業日（2025年1月）
    business_days = [
        "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10",
        "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17",
        "2025-01-20", "2025-01-21", "2025-01-22", "2025-01-23", "2025-01-24",
        "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31"
    ]

    async with aiohttp.ClientSession() as session:
        await api_client.authenticate(session)

        # 月初スナップショット取得テスト
        snapshots = await manager.get_monthly_snapshots(session, business_days)

        for date, df in snapshots.items():
            print(f"{date}: {len(df)} stocks")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_manager())
