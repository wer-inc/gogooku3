#!/usr/bin/env python3
"""
EventDetector - 上場/廃止/市場変更/社名変更イベントの検知と管理
日次のCode集合変化から効率的にイベントを特定
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


class EventDetector:
    """株式市場イベントの検知と管理"""

    def __init__(self, cache_dir: Path = None):
        # キャッシュ設定
        self.cache_dir = cache_dir or Path("cache/events")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # イベントタイプ定義
        self.EVENT_TYPES = {
            "listing": "新規上場",
            "delisting": "上場廃止",
            "market_change": "市場変更",
            "name_change": "社名変更",
            "merger": "合併",
            "split": "分割"
        }

        # イベント履歴
        self.events = []

        # 日次Code集合のキャッシュ
        self.daily_code_sets = {}

    def detect_from_daily_quotes(
        self,
        quotes_df: pl.DataFrame,
        date: str
    ) -> dict[str, set[str]]:
        """
        daily_quotesからCode集合を抽出し、前日との差分を検知

        Args:
            quotes_df: daily_quotesのDataFrame
            date: 対象日

        Returns:
            {"codes": set, "new": set, "removed": set}
        """
        # 当日のCode集合
        if quotes_df.is_empty():
            current_codes = set()
        else:
            current_codes = set(quotes_df["Code"].unique().to_list())

        # 前日のCode集合を取得
        prev_date = self._get_previous_date(date)
        prev_codes = self.daily_code_sets.get(prev_date, set())

        # 差分計算
        new_codes = current_codes - prev_codes
        removed_codes = prev_codes - current_codes

        # キャッシュ更新
        self.daily_code_sets[date] = current_codes

        result = {
            "codes": current_codes,
            "new": new_codes,
            "removed": removed_codes
        }

        if new_codes or removed_codes:
            logger.info(f"{date}: +{len(new_codes)} new, -{len(removed_codes)} removed")

        return result

    def create_listing_event(
        self,
        code: str,
        listing_date: str,
        market_code: str | None = None,
        company_name: str | None = None,
        additional_info: dict | None = None
    ) -> dict:
        """新規上場イベントを作成"""
        event = {
            "event_type": "listing",
            "code": code,
            "effective_from": listing_date,
            "effective_to": None,
            "market_code": market_code,
            "company_name": company_name,
            "description": f"{code} listed on {listing_date}",
            "timestamp": datetime.now().isoformat()
        }

        if additional_info:
            event["additional_info"] = additional_info

        self.events.append(event)
        return event

    def create_delisting_event(
        self,
        code: str,
        delisting_date: str,
        last_trading_date: str | None = None,
        reason: str | None = None,
        additional_info: dict | None = None
    ) -> dict:
        """上場廃止イベントを作成"""
        event = {
            "event_type": "delisting",
            "code": code,
            "effective_from": delisting_date,
            "effective_to": None,
            "last_trading_date": last_trading_date or self._get_previous_date(delisting_date),
            "reason": reason,
            "description": f"{code} delisted on {delisting_date}",
            "timestamp": datetime.now().isoformat()
        }

        if additional_info:
            event["additional_info"] = additional_info

        self.events.append(event)
        return event

    def create_market_change_event(
        self,
        code: str,
        change_date: str,
        old_market: str,
        new_market: str,
        additional_info: dict | None = None
    ) -> dict:
        """市場変更イベントを作成"""
        event = {
            "event_type": "market_change",
            "code": code,
            "effective_from": change_date,
            "effective_to": None,
            "old_market": old_market,
            "new_market": new_market,
            "description": f"{code} moved from {old_market} to {new_market}",
            "timestamp": datetime.now().isoformat()
        }

        if additional_info:
            event["additional_info"] = additional_info

        self.events.append(event)
        return event

    def create_name_change_event(
        self,
        code: str,
        change_date: str,
        old_name: str,
        new_name: str,
        additional_info: dict | None = None
    ) -> dict:
        """社名変更イベントを作成"""
        event = {
            "event_type": "name_change",
            "code": code,
            "effective_from": change_date,
            "effective_to": None,
            "old_name": old_name,
            "new_name": new_name,
            "description": f"{code} renamed from {old_name} to {new_name}",
            "timestamp": datetime.now().isoformat()
        }

        if additional_info:
            event["additional_info"] = additional_info

        self.events.append(event)
        return event

    def process_listed_info_changes(
        self,
        prev_info: pl.DataFrame,
        curr_info: pl.DataFrame,
        date: str
    ) -> list[dict]:
        """
        listed_infoの変化からイベントを生成

        Args:
            prev_info: 前回のlisted_info
            curr_info: 今回のlisted_info
            date: 変化検知日

        Returns:
            生成されたイベントリスト
        """
        events = []

        if prev_info.is_empty() or curr_info.is_empty():
            return events

        # コード集合の比較
        prev_codes = set(prev_info["Code"].to_list())
        curr_codes = set(curr_info["Code"].to_list())

        # 新規上場
        for code in (curr_codes - prev_codes):
            row = curr_info.filter(pl.col("Code") == code).row(0, named=True)
            event = self.create_listing_event(
                code=code,
                listing_date=date,
                market_code=row.get("MarketCode"),
                company_name=row.get("CompanyName")
            )
            events.append(event)

        # 上場廃止
        for code in (prev_codes - curr_codes):
            row = prev_info.filter(pl.col("Code") == code).row(0, named=True)
            event = self.create_delisting_event(
                code=code,
                delisting_date=date,
                last_trading_date=self._get_previous_date(date)
            )
            events.append(event)

        # 継続銘柄の変更チェック
        for code in (prev_codes & curr_codes):
            prev_row = prev_info.filter(pl.col("Code") == code).row(0, named=True)
            curr_row = curr_info.filter(pl.col("Code") == code).row(0, named=True)

            # 市場変更
            if prev_row.get("MarketCode") != curr_row.get("MarketCode"):
                event = self.create_market_change_event(
                    code=code,
                    change_date=date,
                    old_market=prev_row.get("MarketCode"),
                    new_market=curr_row.get("MarketCode")
                )
                events.append(event)

            # 社名変更
            if prev_row.get("CompanyName") != curr_row.get("CompanyName"):
                event = self.create_name_change_event(
                    code=code,
                    change_date=date,
                    old_name=prev_row.get("CompanyName"),
                    new_name=curr_row.get("CompanyName")
                )
                events.append(event)

        return events

    def generate_market_membership(self) -> pl.DataFrame:
        """
        イベントからmarket_membershipテーブルを生成

        Returns:
            market_membershipのDataFrame
        """
        membership_records = []
        code_market_map = {}  # {code: [(market, from, to), ...]}

        # イベントを時系列順にソート
        sorted_events = sorted(self.events, key=lambda x: x["effective_from"])

        for event in sorted_events:
            code = event["code"]
            date = event["effective_from"]

            if code not in code_market_map:
                code_market_map[code] = []

            if event["event_type"] == "listing":
                # 新規上場
                code_market_map[code].append({
                    "code": code,
                    "market_code": event.get("market_code"),
                    "from_date": date,
                    "to_date": None
                })

            elif event["event_type"] == "delisting":
                # 上場廃止 - 最後のmembershipを終了
                if code_market_map[code]:
                    code_market_map[code][-1]["to_date"] = date

            elif event["event_type"] == "market_change":
                # 市場変更
                # 現在のmembershipを終了
                if code_market_map[code]:
                    code_market_map[code][-1]["to_date"] = date

                # 新しいmembershipを開始
                code_market_map[code].append({
                    "code": code,
                    "market_code": event["new_market"],
                    "from_date": date,
                    "to_date": None
                })

        # フラット化してDataFrameに変換
        for code, memberships in code_market_map.items():
            membership_records.extend(memberships)

        if membership_records:
            df = pl.DataFrame(membership_records)

            # 5桁LocalCodeを追加
            df = df.with_columns([
                (pl.col("code").cast(pl.Utf8).str.zfill(4) + "0").alias("local_code")
            ])

            return df.sort(["code", "from_date"])

        return pl.DataFrame()

    def generate_securities_events_table(self) -> pl.DataFrame:
        """
        securities_eventsテーブル用のDataFrameを生成

        Returns:
            securities_eventsのDataFrame
        """
        if not self.events:
            return pl.DataFrame()

        # 必要なカラムのみ抽出
        records = []
        for event in self.events:
            record = {
                "local_code": self._to_local_code(event["code"]),
                "event_type": event["event_type"],
                "effective_from": event["effective_from"],
                "effective_to": event.get("effective_to"),
                "details_text": event.get("description", "")
            }

            # イベントタイプ別の追加情報
            if event["event_type"] == "market_change":
                record["details_text"] += f" ({event.get('old_market')} -> {event.get('new_market')})"
            elif event["event_type"] == "name_change":
                record["details_text"] += f" ({event.get('old_name')} -> {event.get('new_name')})"

            records.append(record)

        df = pl.DataFrame(records)
        return df.sort(["local_code", "effective_from"])

    def save_events(self, filename: str = None):
        """イベントをJSONファイルに保存"""
        if not filename:
            filename = f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.cache_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.events, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Saved {len(self.events)} events to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save events: {e}")

    def load_events(self, filename: str) -> list[dict]:
        """保存されたイベントを読み込み"""
        filepath = self.cache_dir / filename

        if not filepath.exists():
            logger.warning(f"Event file not found: {filepath}")
            return []

        try:
            with open(filepath, encoding='utf-8') as f:
                events = json.load(f)
            self.events = events
            logger.info(f"Loaded {len(events)} events from {filepath}")
            return events
        except Exception as e:
            logger.error(f"Failed to load events: {e}")
            return []

    def get_statistics(self) -> dict:
        """イベント統計を取得"""
        stats = {
            "total_events": len(self.events),
            "by_type": {},
            "by_year": {},
            "by_month": {}
        }

        for event in self.events:
            # タイプ別集計
            event_type = event["event_type"]
            stats["by_type"][event_type] = stats["by_type"].get(event_type, 0) + 1

            # 年別集計
            year = event["effective_from"][:4]
            stats["by_year"][year] = stats["by_year"].get(year, 0) + 1

            # 月別集計
            month = event["effective_from"][:7]
            stats["by_month"][month] = stats["by_month"].get(month, 0) + 1

        return stats

    def _get_previous_date(self, date: str) -> str:
        """前営業日を取得（簡易版）"""
        dt = datetime.strptime(date, "%Y-%m-%d")
        dt -= timedelta(days=1)

        # 土日をスキップ（簡易版）
        while dt.weekday() >= 5:
            dt -= timedelta(days=1)

        return dt.strftime("%Y-%m-%d")

    def _to_local_code(self, code: str) -> str:
        """4桁コードを5桁LocalCodeに変換"""
        if len(code) == 4:
            return code + "0"
        return code


def test_detector():
    """テスト関数"""
    detector = EventDetector()

    # テストデータ作成
    prev_info = pl.DataFrame({
        "Code": ["1234", "5678", "9012"],
        "MarketCode": ["0111", "0112", "0113"],
        "CompanyName": ["会社A", "会社B", "会社C"]
    })

    curr_info = pl.DataFrame({
        "Code": ["1234", "5678", "3456"],  # 9012が廃止、3456が新規
        "MarketCode": ["0111", "0111", "0113"],  # 5678が市場変更
        "CompanyName": ["新会社A", "会社B", "会社D"]  # 1234が社名変更
    })

    # イベント検知
    events = detector.process_listed_info_changes(
        prev_info, curr_info, "2025-01-10"
    )

    print(f"Detected {len(events)} events:")
    for event in events:
        print(f"  {event['event_type']}: {event['code']} - {event.get('description', '')}")

    # market_membership生成
    membership_df = detector.generate_market_membership()
    if not membership_df.is_empty():
        print("\nMarket Membership:")
        print(membership_df)

    # 統計表示
    stats = detector.get_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_detector()
