#!/usr/bin/env python3
"""
AxisDecider - J-Quants API daily_quotes取得軸の自動選択
date軸とcode軸のAPIコール数・転送量を実測して最適な軸を決定
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import aiohttp

logger = logging.getLogger(__name__)


class AxisDecider:
    """daily_quotes取得軸の自動選択器"""

    AxisChoice = Literal["by_date", "by_code"]

    def __init__(self, api_client=None, cache_dir: Path = None):
        self.api_client = api_client
        self.base_url = "https://api.jquants.com/v1"

        # キャッシュ設定
        self.cache_dir = cache_dir or Path("cache/axis_decision")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 計測結果のキャッシュ
        self.last_measurement = None
        self.measurement_ttl = 3600  # 1時間
        self.measurement_timestamp = None

        # 閾値設定（環境変数で調整可能）
        self.threshold = float(os.getenv("AXIS_SELECTION_THRESHOLD", "0.9"))

    async def get_optimal_axis(
        self,
        session: aiohttp.ClientSession,
        sample_days: list[str] | None = None,
        sample_codes: list[str] | None = None,
        market_filter: bool = True,
        force_refresh: bool = False
    ) -> tuple[AxisChoice, dict]:
        """
        最適な取得軸を決定

        Args:
            session: aiohttp ClientSession
            sample_days: サンプル営業日リスト（省略時は自動選択）
            sample_codes: サンプル銘柄コードリスト（省略時は自動選択）
            market_filter: 市場フィルタリングを行うか
            force_refresh: キャッシュを無視して再計測

        Returns:
            (選択された軸, 計測結果の詳細)
        """
        # キャッシュチェック
        if not force_refresh and self._is_cache_valid():
            logger.info(f"Using cached axis decision: {self.last_measurement['axis']}")
            return self.last_measurement['axis'], self.last_measurement

        logger.info("Measuring API efficiency for axis selection...")

        try:
            # サンプルデータの準備
            if not sample_days:
                sample_days = self._get_default_sample_days()
            if not sample_codes:
                sample_codes = await self._get_default_sample_codes(session)

            # 両軸での計測
            date_metrics = await self._measure_date_axis(
                session, sample_days
            )

            code_metrics = await self._measure_code_axis(
                session, sample_codes, sample_days[0], sample_days[-1]
            )

            # 最適軸の決定
            axis, decision_reason = self._decide_optimal_axis(
                date_metrics, code_metrics, market_filter
            )

            # 結果をキャッシュ
            result = {
                "axis": axis,
                "date_metrics": date_metrics,
                "code_metrics": code_metrics,
                "decision_reason": decision_reason,
                "threshold": self.threshold,
                "market_filter": market_filter,
                "timestamp": datetime.now().isoformat()
            }

            self._save_measurement(result)

            # ログ出力
            logger.info(f"Axis decision: {axis}")
            logger.info(f"  Date axis: {date_metrics['total_pages']} pages, "
                       f"{date_metrics['total_calls']} calls")
            logger.info(f"  Code axis: {code_metrics['total_pages']} pages, "
                       f"{code_metrics['total_calls']} calls")
            logger.info(f"  Decision reason: {decision_reason}")

            return axis, result

        except Exception as e:
            logger.error(f"Failed to measure API efficiency: {e}")
            # 計測失敗時はデフォルト戦略
            return self._get_default_axis(market_filter), {
                "axis": self._get_default_axis(market_filter),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _measure_date_axis(
        self,
        session: aiohttp.ClientSession,
        sample_days: list[str]
    ) -> dict:
        """date軸での計測"""
        url = f"{self.base_url}/prices/daily_quotes"
        headers = {"Authorization": f"Bearer {self.api_client.id_token}"}

        total_pages = 0
        total_calls = 0
        total_records = 0
        start_time = time.time()

        for date in sample_days[:3]:  # 最大3日分で計測
            date_api = date.replace("-", "")
            params = {"date": date_api}
            pagination_key = None
            pages_for_date = 0

            while True:
                if pagination_key:
                    params["pagination_key"] = pagination_key

                try:
                    async with session.get(url, headers=headers, params=params) as response:
                        total_calls += 1

                        if response.status != 200:
                            break

                        data = await response.json()
                        quotes = data.get("daily_quotes", [])
                        total_records += len(quotes)
                        pages_for_date += 1

                        pagination_key = data.get("pagination_key")
                        if not pagination_key:
                            break

                except Exception as e:
                    logger.debug(f"Error measuring date axis for {date}: {e}")
                    break

            total_pages += pages_for_date
            logger.debug(f"  Date {date}: {pages_for_date} pages")

        elapsed = time.time() - start_time

        # 全期間への推定
        estimated_total_pages = total_pages * (len(sample_days) / min(3, len(sample_days)))

        return {
            "total_pages": int(estimated_total_pages),
            "total_calls": total_calls,
            "total_records": total_records,
            "elapsed_seconds": elapsed,
            "sample_days": len(sample_days[:3])
        }

    async def _measure_code_axis(
        self,
        session: aiohttp.ClientSession,
        sample_codes: list[str],
        from_date: str,
        to_date: str
    ) -> dict:
        """code軸での計測"""
        url = f"{self.base_url}/prices/daily_quotes"
        headers = {"Authorization": f"Bearer {self.api_client.id_token}"}

        from_api = from_date.replace("-", "")
        to_api = to_date.replace("-", "")

        total_pages = 0
        total_calls = 0
        total_records = 0
        start_time = time.time()

        for code in sample_codes[:50]:  # 最大50銘柄で計測
            params = {
                "code": code,
                "from": from_api,
                "to": to_api
            }
            pagination_key = None
            pages_for_code = 0

            while True:
                if pagination_key:
                    params["pagination_key"] = pagination_key

                try:
                    async with session.get(url, headers=headers, params=params) as response:
                        total_calls += 1

                        if response.status != 200:
                            break

                        data = await response.json()
                        quotes = data.get("daily_quotes", [])
                        total_records += len(quotes)
                        pages_for_code += 1

                        pagination_key = data.get("pagination_key")
                        if not pagination_key:
                            break

                except Exception as e:
                    logger.debug(f"Error measuring code axis for {code}: {e}")
                    break

            total_pages += pages_for_code

        elapsed = time.time() - start_time

        # 全銘柄への推定（市場フィルタ後を想定して約3000銘柄）
        estimated_total_pages = total_pages * (3000 / min(50, len(sample_codes)))

        return {
            "total_pages": int(estimated_total_pages),
            "total_calls": total_calls,
            "total_records": total_records,
            "elapsed_seconds": elapsed,
            "sample_codes": len(sample_codes[:50])
        }

    def _decide_optimal_axis(
        self,
        date_metrics: dict,
        code_metrics: dict,
        market_filter: bool
    ) -> tuple[AxisChoice, str]:
        """最適軸の決定"""
        date_pages = date_metrics["total_pages"]
        code_pages = code_metrics["total_pages"]

        # 市場フィルタリングする場合はcode軸が有利になりやすい
        if market_filter:
            # code軸にボーナス（市場フィルタで約40%削減を想定）
            effective_code_pages = code_pages * 0.6
        else:
            effective_code_pages = code_pages

        # 閾値ベースの判定
        if date_pages <= self.threshold * effective_code_pages:
            return "by_date", f"Date axis is more efficient ({date_pages} <= {self.threshold} * {effective_code_pages:.0f})"
        else:
            return "by_code", f"Code axis is more efficient ({effective_code_pages:.0f} < {date_pages} / {self.threshold})"

    def _get_default_axis(self, market_filter: bool) -> AxisChoice:
        """デフォルト戦略"""
        # 市場フィルタありならcode軸、なしならdate軸
        return "by_code" if market_filter else "by_date"

    def _is_cache_valid(self) -> bool:
        """キャッシュの有効性チェック"""
        if not self.last_measurement or not self.measurement_timestamp:
            # キャッシュファイルから読み込み試行
            cache_file = self.cache_dir / "last_measurement.json"
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        cached = json.load(f)
                        cached_time = datetime.fromisoformat(cached["timestamp"])
                        if (datetime.now() - cached_time).total_seconds() < self.measurement_ttl:
                            self.last_measurement = cached
                            self.measurement_timestamp = cached_time
                            return True
                except Exception:
                    pass
            return False

        elapsed = (datetime.now() - self.measurement_timestamp).total_seconds()
        return elapsed < self.measurement_ttl

    def _save_measurement(self, result: dict):
        """計測結果の保存"""
        self.last_measurement = result
        self.measurement_timestamp = datetime.now()

        # ファイルにも保存
        cache_file = self.cache_dir / "last_measurement.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save measurement cache: {e}")

    def _get_default_sample_days(self) -> list[str]:
        """デフォルトのサンプル営業日を生成"""
        # 直近の営業日から3日分
        days = []
        current = datetime.now()

        while len(days) < 3:
            if current.weekday() < 5:  # 月-金
                days.append(current.strftime("%Y-%m-%d"))
            current -= timedelta(days=1)

        return days

    async def _get_default_sample_codes(
        self,
        session: aiohttp.ClientSession
    ) -> list[str]:
        """デフォルトのサンプル銘柄を取得"""
        # 主要な銘柄コード（日経225採用銘柄の一部）
        return [
            "7203",  # トヨタ
            "6758",  # ソニー
            "6861",  # キーエンス
            "9984",  # ソフトバンクG
            "8306",  # 三菱UFJ
            "9432",  # NTT
            "4063",  # 信越化学
            "6098",  # リクルート
            "7267",  # ホンダ
            "8035",  # 東エレク
            "6501",  # 日立
            "4661",  # OLC
            "6902",  # デンソー
            "9433",  # KDDI
            "7974",  # 任天堂
            "6954",  # ファナック
            "8058",  # 三菱商事
            "8001",  # 伊藤忠
            "6367",  # ダイキン
            "8031",  # 三井物産
            # ... 実際は50銘柄程度
        ][:50]


async def test_axis_decider():
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
    decider = AxisDecider(api_client)

    async with aiohttp.ClientSession() as session:
        await api_client.authenticate(session)

        # 最適軸の決定
        axis, metrics = await decider.get_optimal_axis(
            session,
            market_filter=True
        )

        print(f"Optimal axis: {axis}")
        print(f"Decision reason: {metrics.get('decision_reason', 'N/A')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_axis_decider())
