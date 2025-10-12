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
    # override=True so that .env values replace empty/existing envs
    load_dotenv(env_path, override=True)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
# Ensure project root is importable (so we can import src.* modules)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gogooku3.pipeline.builder import MLDatasetBuilder
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


SUPPORT_LOOKBACK_DAYS = 420  # approx 20 months to cover YoY/Z-score lookbacks
# Minimum collection span enforced regardless of requested start date
MIN_COLLECTION_DAYS = int(os.getenv("MIN_COLLECTION_DAYS", "3650"))  # ~10 years by default


# ========================================
# Cache Utility Functions
# ========================================

def _find_latest_with_date_range(
    glob: str, req_start: str, req_end: str
) -> Path | None:
    """Find cached file that covers the requested date range.

    Args:
        glob: Glob pattern to match files (e.g., "daily_quotes_*.parquet")
        req_start: Requested start date (YYYY-MM-DD)
        req_end: Requested end date (YYYY-MM-DD)

    Returns:
        Path to the cached file if found and valid, None otherwise
    """
    import re

    req_start_dt = datetime.strptime(req_start, "%Y-%m-%d")
    req_end_dt = datetime.strptime(req_end, "%Y-%m-%d")

    # Find all matching files under output/
    candidates = sorted(Path("output").rglob(glob))

    for cand in reversed(candidates):  # Try newest first
        # Extract date range from filename: *_YYYYMMDD_YYYYMMDD.parquet
        match = re.search(r"_(\d{8})_(\d{8})\.parquet$", cand.name)
        if match:
            file_start = datetime.strptime(match.group(1), "%Y%m%d")
            file_end = datetime.strptime(match.group(2), "%Y%m%d")

            # Check if file covers the requested range
            if file_start <= req_start_dt and file_end >= req_end_dt:
                logger.debug(
                    f"✅ Cache found: {cand.name} covers {req_start} to {req_end}"
                )
                return cand
            else:
                logger.debug(
                    f"⏩ Cache skip: {cand.name} "
                    f"(file: {file_start.strftime('%Y-%m-%d')} to {file_end.strftime('%Y-%m-%d')}, "
                    f"need: {req_start} to {req_end})"
                )

    return None


def _is_cache_valid(file_path: Path | None, max_age_days: int) -> bool:
    """Check if cached file exists and is not stale.

    Args:
        file_path: Path to cached file
        max_age_days: Maximum age in days before considering stale

    Returns:
        True if file exists and is fresh enough, False otherwise
    """
    if file_path is None or not file_path.exists():
        return False

    try:
        file_age_seconds = time.time() - file_path.stat().st_mtime
        file_age_days = file_age_seconds / (24 * 3600)
        is_valid = file_age_days <= max_age_days

        if is_valid:
            logger.info(
                f"✅ Cache valid: {file_path.name} "
                f"(age: {file_age_days:.1f} days, limit: {max_age_days} days)"
            )
        else:
            logger.info(
                f"⏰ Cache stale: {file_path.name} "
                f"(age: {file_age_days:.1f} days, limit: {max_age_days} days)"
            )

        return is_valid
    except Exception as e:
        logger.warning(f"Failed to check cache validity for {file_path}: {e}")
        return False


# ========================================
# End Cache Utility Functions
# ========================================


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
    """パフォーマンストラッキング（キャッシュ統計付き）"""

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.api_call_count = 0
        self.start_time = time.time()

        # キャッシュ統計
        self.cache_stats = {
            "total_sources": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "time_saved_sec": 0.0,
            "details": []
        }

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

    def record_cache_hit(self, data_source: str, time_saved: float):
        """キャッシュヒットを記録"""
        self.cache_stats["total_sources"] += 1
        self.cache_stats["cache_hits"] += 1
        self.cache_stats["time_saved_sec"] += time_saved
        self.cache_stats["details"].append({
            "source": data_source,
            "status": "HIT",
            "time_saved": time_saved
        })
        logger.info(f"📦 CACHE HIT: {data_source} (saved ~{time_saved:.0f}s)")

    def record_cache_miss(self, data_source: str):
        """キャッシュミスを記録"""
        self.cache_stats["total_sources"] += 1
        self.cache_stats["cache_misses"] += 1
        self.cache_stats["details"].append({
            "source": data_source,
            "status": "MISS",
            "time_saved": 0
        })
        logger.info(f"🌐 CACHE MISS: {data_source} (fetching from API)")

    def get_cache_summary(self) -> str:
        """キャッシュサマリーを取得"""
        stats = self.cache_stats
        if stats["total_sources"] == 0:
            return "No cache operations recorded"

        hit_rate = (stats["cache_hits"] / stats["total_sources"]) * 100

        summary = f"""
🎯 Cache Performance Summary:
   Total Sources: {stats['total_sources']}
   Cache Hits: {stats['cache_hits']} ({hit_rate:.1f}%)
   Cache Misses: {stats['cache_misses']} ({100-hit_rate:.1f}%)
   Time Saved: ~{stats['time_saved_sec']:.0f}s
   Speedup: {stats['time_saved_sec'] / (time.time() - self.start_time) * 100:.0f}% faster

📊 Details:
"""
        for detail in stats["details"]:
            status_icon = "✅" if detail["status"] == "HIT" else "❌"
            saved = f" (saved {detail['time_saved']:.0f}s)" if detail["time_saved"] > 0 else ""
            summary += f"   {status_icon} {detail['source']}: {detail['status']}{saved}\n"

        return summary.strip()

    def get_summary(self) -> Dict:
        """サマリーを取得"""
        total_duration = time.time() - self.start_time

        return {
            "total_duration_seconds": total_duration,
            "total_api_calls": self.api_call_count,
            "total_records": sum(m.records_processed for m in self.metrics),
            "components": [m.to_dict() for m in self.metrics],
            "average_memory_mb": sum(m.memory_usage_mb for m in self.metrics) / len(self.metrics) if self.metrics else 0,
            "cache_stats": self.cache_stats
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
        キャッシュを優先的に使用（最大60%高速化）
        """
        metric = self.tracker.start_component("daily_quotes_optimized")

        # Check environment variables for cache control
        use_cache = os.getenv("USE_CACHE", "1") == "1"
        max_cache_age_days = int(os.getenv("CACHE_MAX_AGE_DAYS", "7"))

        # Step 1: キャッシュをチェック
        if use_cache and business_days:
            start_date = business_days[0]
            end_date = business_days[-1]

            cached_file = _find_latest_with_date_range(
                "daily_quotes_*.parquet",
                start_date,
                end_date
            )

            if cached_file and _is_cache_valid(cached_file, max_cache_age_days):
                try:
                    price_df = pl.read_parquet(cached_file)

                    # 日付範囲フィルタ
                    if "Date" in price_df.columns:
                        price_df = price_df.filter(
                            (pl.col("Date") >= start_date) & (pl.col("Date") <= end_date)
                        )

                    # 市場フィルタリング
                    if target_codes and "Code" in price_df.columns:
                        original_count = len(price_df)
                        price_df = price_df.filter(pl.col("Code").is_in(target_codes))
                        logger.info(f"  Filtered: {original_count} → {len(price_df)} records (market codes)")

                    # キャッシュヒットを記録
                    self.tracker.record_cache_hit("Daily Quotes", 45.0)  # 平均30-60秒の中央値

                    self.tracker.end_component(metric, api_calls=0, records=len(price_df))
                    return price_df
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load cache: {e}, falling back to API")
                    self.tracker.record_cache_miss("Daily Quotes")
            else:
                self.tracker.record_cache_miss("Daily Quotes")

        # Step 2: キャッシュがなければAPIから取得
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

                # Step 3: キャッシュに保存
                if use_cache and business_days:
                    try:
                        from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
                        start_date = business_days[0]
                        end_date = business_days[-1]
                        cache_dir = Path("output/raw/prices")
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        cache_path = cache_dir / f"daily_quotes_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                        save_parquet_with_gcs(combined_df, cache_path)
                        logger.info(f"💾 Saved to cache: {cache_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to save cache: {e}")

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

            # Step 3: キャッシュに保存
            if use_cache and business_days and not df.is_empty():
                try:
                    from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
                    start_date = business_days[0]
                    end_date = business_days[-1]
                    cache_dir = Path("output/raw/prices")
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path = cache_dir / f"daily_quotes_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                    save_parquet_with_gcs(df, cache_path)
                    logger.info(f"💾 Saved to cache: {cache_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to save cache: {e}")

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
        """date軸での財務諸表取得 - キャッシュ優先（20%高速化）"""
        metric = self.tracker.start_component("statements_by_date")

        # Check environment variables for cache control
        use_cache = os.getenv("USE_CACHE", "1") == "1"
        max_cache_age_days = int(os.getenv("CACHE_MAX_AGE_DAYS", "7"))

        # Step 1: キャッシュをチェック
        if use_cache and business_days:
            start_date = business_days[0]
            end_date = business_days[-1]

            cached_file = _find_latest_with_date_range(
                "event_raw_statements_*.parquet",
                start_date,
                end_date
            )

            if cached_file and _is_cache_valid(cached_file, max_cache_age_days):
                try:
                    statements_df = pl.read_parquet(cached_file)

                    # 日付範囲フィルタ（DisclosedDateまたはDisclosureDate列がある場合）
                    date_col = None
                    if "DisclosedDate" in statements_df.columns:
                        date_col = "DisclosedDate"
                    elif "DisclosureDate" in statements_df.columns:
                        date_col = "DisclosureDate"

                    if date_col:
                        statements_df = statements_df.filter(
                            (pl.col(date_col) >= start_date) & (pl.col(date_col) <= end_date)
                        )

                    # キャッシュヒットを記録
                    self.tracker.record_cache_hit("Statements", 30.0)  # 平均20-40秒の中央値

                    self.tracker.end_component(metric, api_calls=0, records=len(statements_df))
                    return statements_df
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load cache: {e}, falling back to API")
                    self.tracker.record_cache_miss("Statements")
            else:
                self.tracker.record_cache_miss("Statements")

        # Step 2: キャッシュがなければAPIから取得
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

        # Step 3: キャッシュに保存
        if use_cache and business_days and not result_df.is_empty():
            try:
                from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
                start_date = business_days[0]
                end_date = business_days[-1]
                cache_dir = Path("output/raw/statements")
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"event_raw_statements_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                save_parquet_with_gcs(result_df, cache_path)
                logger.info(f"💾 Saved to cache: {cache_path.name}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

        self.tracker.end_component(metric, api_calls=len(business_days), records=len(result_df))
        return result_df

    async def fetch_topix_data(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch TOPIX index data - キャッシュ優先（5%高速化）"""

        # Check environment variables for cache control
        use_cache = os.getenv("USE_CACHE", "1") == "1"
        max_cache_age_days = int(os.getenv("CACHE_MAX_AGE_DAYS", "7"))

        # Step 1: キャッシュをチェック
        if use_cache:
            cached_file = _find_latest_with_date_range(
                "topix_history_*.parquet",
                from_date,
                to_date
            )

            if cached_file and _is_cache_valid(cached_file, max_cache_age_days):
                try:
                    topix_df = pl.read_parquet(cached_file)

                    # 日付範囲フィルタ
                    if "Date" in topix_df.columns:
                        topix_df = topix_df.filter(
                            (pl.col("Date") >= from_date) & (pl.col("Date") <= to_date)
                        )

                    # キャッシュヒットを記録
                    self.tracker.record_cache_hit("TOPIX", 3.5)  # 平均2-5秒の中央値

                    return topix_df
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load cache: {e}, falling back to API")
                    self.tracker.record_cache_miss("TOPIX")
            else:
                self.tracker.record_cache_miss("TOPIX")

        # Step 2: キャッシュがなければAPIから取得
        url = f"{self.base_url}/indices/topix"
        headers = {"Authorization": f"Bearer {self.id_token}"}

        from_api = from_date.replace("-", "")
        to_api = to_date.replace("-", "")

        all_data = []
        pagination_key = None

        while True:
            params = {"from": from_api, "to": to_api}
            if pagination_key:
                params["pagination_key"] = pagination_key

            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch TOPIX: {response.status}")
                        break

                    data = await response.json()
                    topix_data = data.get("topix", [])

                    if topix_data:
                        all_data.extend(topix_data)

                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break

            except Exception as e:
                logger.error(f"Error fetching TOPIX: {e}")
                break

        if all_data:
            df = pl.DataFrame(all_data)
            # Normalize dtypes
            if "Date" in df.columns:
                df = df.with_columns(
                    pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
                )
            for c in ("Open", "High", "Low", "Close", "Volume"):
                if c in df.columns:
                    df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))
            df = df.sort("Date")

            # Step 3: キャッシュに保存
            if use_cache and not df.is_empty():
                try:
                    from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
                    cache_dir = Path("output/raw/indices")
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path = cache_dir / f"topix_history_{from_date.replace('-', '')}_{to_date.replace('-', '')}.parquet"
                    save_parquet_with_gcs(df, cache_path)
                    logger.info(f"💾 Saved to cache: {cache_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to save cache: {e}")

            return df

        return pl.DataFrame()

    async def fetch_trades_spec(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch trades_spec (投資部門別) between [from, to]."""
        url = f"{self.base_url}/markets/trades_spec"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        from_api = from_date.replace("-", "")
        to_api = to_date.replace("-", "")
        all_data = []
        pagination_key = None
        while True:
            params = {"from": from_api, "to": to_api}
            if pagination_key:
                params["pagination_key"] = pagination_key
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch trades_spec: {response.status}")
                        break
                    data = await response.json()
                    items = data.get("trades_spec", [])
                    if items:
                        all_data.extend(items)
                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break
            except Exception as e:
                logger.error(f"Error fetching trades_spec: {e}")
                break
        return pl.DataFrame(all_data) if all_data else pl.DataFrame()


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

    def _ensure_code_utf8(self, df: pl.DataFrame | None) -> pl.DataFrame | None:
        """Ensure Code column dtype is Utf8 for safe joins.
        Returns the same df if None or empty.
        """
        try:
            if df is not None and not df.is_empty() and "Code" in df.columns:
                return df.with_columns([pl.col("Code").cast(pl.Utf8).alias("Code")])
        except Exception:
            pass
        return df

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
            start_date = os.getenv("ML_PIPELINE_START_DATE", "2019-01-01")

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        min_available_dt: datetime | None = None
        min_available_str = os.getenv("JQUANTS_MIN_AVAILABLE_DATE") or os.getenv("ML_PIPELINE_START_DATE")
        if min_available_str:
            try:
                min_available_dt = datetime.strptime(min_available_str, "%Y-%m-%d")
            except ValueError:
                logger.warning(
                    "Invalid subscription lower bound %s; ignoring",
                    min_available_str,
                )

        span_days = (end_dt - start_dt).days
        min_span = max(MIN_COLLECTION_DAYS, SUPPORT_LOOKBACK_DAYS)
        if span_days < min_span:
            adjusted_start_dt = end_dt - timedelta(days=min_span)
            if min_available_dt and adjusted_start_dt < min_available_dt:
                adjusted_start_dt = min_available_dt
            if adjusted_start_dt < start_dt:
                logger.info(
                    "Extending start date from %s to %s to satisfy minimum collection span of %d days",
                    start_dt.strftime("%Y-%m-%d"),
                    adjusted_start_dt.strftime("%Y-%m-%d"),
                    min_span,
                )
            start_dt = adjusted_start_dt
            start_date = start_dt.strftime("%Y-%m-%d")

        if min_available_dt and start_dt < min_available_dt:
            logger.info(
                "Capping start date at subscription lower bound %s",
                min_available_dt.strftime("%Y-%m-%d"),
            )
            start_dt = min_available_dt
            start_date = start_dt.strftime("%Y-%m-%d")

        support_start_dt = start_dt - timedelta(days=SUPPORT_LOOKBACK_DAYS)
        if min_available_dt and support_start_dt < min_available_dt:
            logger.info(
                "Capping support lookback start %s at %s",
                support_start_dt.strftime("%Y-%m-%d"),
                min_available_dt.strftime("%Y-%m-%d"),
            )
            support_start_dt = min_available_dt
        support_start_date = support_start_dt.strftime("%Y-%m-%d")

        async with aiohttp.ClientSession() as session:
            # Authenticate
            await self.fetcher.authenticate(session)
            
            # Step 1: 営業日カレンダー取得
            logger.info(
                "Step 1: Fetching trading calendar (%s - %s) [support lookback=%s days]...",
                support_start_date,
                end_date,
                SUPPORT_LOOKBACK_DAYS,
            )
            cal_metric = self.tracker.start_component("trading_calendar")
            
            calendar_data = await self.fetcher.calendar_fetcher.get_trading_calendar(
                support_start_date, end_date, session
            )
            business_days_full = calendar_data.get("business_days", [])
            # Trim to requested window for price-based resources
            business_days = [d for d in business_days_full if d >= start_date]

            self.tracker.end_component(cal_metric, api_calls=1, records=len(business_days_full))
            logger.info(
                "✅ Business days: %s (support window), %s within target range",
                len(business_days_full),
                len(business_days),
            )
            
            if not business_days:
                logger.error("No business days found")
                return pl.DataFrame(), pl.DataFrame(), [], {}

            # Step 2: Listed info（月初＋差分）で市場銘柄を特定
            logger.info("Step 2: Fetching listed info (monthly + diff)...")
            snapshots, events = await self.fetcher.fetch_listed_info_optimized(
                session, business_days_full
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

            # カラム名の正規化: code -> Code (JQuants APIは小文字を返す場合がある)
            if not price_df.is_empty():
                if "code" in price_df.columns and "Code" not in price_df.columns:
                    price_df = price_df.rename({"code": "Code"})
                code_col = "Code" if "Code" in price_df.columns else price_df.columns[0]
                logger.info(f"✅ Price data: {len(price_df)} records, {price_df[code_col].n_unique()} stocks")
            else:
                logger.warning("⚠️ Price data is empty")

            # Step 4: 財務諸表（date軸）
            logger.info("Step 4: Fetching statements (date axis)...")
            statements_df = await self.fetcher.fetch_statements_by_date(
                session, business_days_full
            )
            
            logger.info(f"✅ Statements: {len(statements_df)} records")

            # Step 5: TOPIX（市場インデックス）
            logger.info("Step 5: Fetching TOPIX index data...")
            topix_df = await self.fetcher.fetch_topix_data(session, support_start_date, end_date)
            if not topix_df.is_empty():
                logger.info(f"✅ TOPIX: {len(topix_df)} records from {start_date} to {end_date}")
            else:
                logger.warning("No TOPIX data fetched via API; will rely on offline fallback if available")

            # Step 6: Fetch trades_spec (flow)
            logger.info("Step 6: Fetching trades_spec (flow data)...")
            trades_spec_df = await self.fetcher.fetch_trades_spec(session, support_start_date, end_date)
            if not trades_spec_df.is_empty():
                logger.info(f"✅ trades_spec: {len(trades_spec_df)} records")
            else:
                logger.warning("No trades_spec fetched via API; will rely on offline fallback if available")

            # Adjustment列の処理とカラム名の正規化
            if not price_df.is_empty():
                columns_to_rename = {}

                # JQuants APIのカラム名正規化 (小文字 -> 大文字)
                if "code" in price_df.columns and "Code" not in price_df.columns:
                    columns_to_rename["code"] = "Code"
                if "date" in price_df.columns and "Date" not in price_df.columns:
                    columns_to_rename["date"] = "Date"

                # 価格カラムの正規化
                price_cols_map = {
                    "open": "Open", "high": "High", "low": "Low",
                    "close": "Close", "volume": "Volume"
                }
                for lower_col, upper_col in price_cols_map.items():
                    if lower_col in price_df.columns and upper_col not in price_df.columns:
                        columns_to_rename[lower_col] = upper_col

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
            
            # Unify Code dtype to Utf8 for all core frames used in joins
            if not price_df.is_empty():
                result = self._ensure_code_utf8(price_df)
                price_df = result if result is not None and not result.is_empty() else price_df
            else:
                price_df = pl.DataFrame()

            if not statements_df.is_empty():
                result = self._ensure_code_utf8(statements_df)
                statements_df = result if result is not None and not result.is_empty() else statements_df
                # Save a reusable parquet for future runs (caching)
                try:
                    from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
                    outdir = self.output_dir / "raw" / "statements"
                    outdir.mkdir(parents=True, exist_ok=True)
                    out_path = outdir / f"event_raw_statements_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                    save_parquet_with_gcs(statements_df, out_path)
                    # Maintain a stable symlink for fallback loaders
                    try:
                        link = outdir / "event_raw_statements.parquet"
                        if link.exists() or link.is_symlink():
                            link.unlink()
                        link.symlink_to(out_path)
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"  Failed to save statements parquet for reuse: {e}")

            if not trades_spec_df.is_empty():
                result = self._ensure_code_utf8(trades_spec_df)
                trades_spec_df = result if result is not None and not result.is_empty() else trades_spec_df

            self.tracker.end_component(metric, records=len(price_df))
            
            return price_df, statements_df, events, snapshots, topix_df, trades_spec_df

    def process_pipeline(
        self, 
        price_df: pl.DataFrame,
        statements_df: Optional[pl.DataFrame] = None,
        events: Optional[List[Dict]] = None,
        topix_df: Optional[pl.DataFrame] = None,
        snapshots: Optional[Dict[str, pl.DataFrame]] = None,
        trades_spec_df: Optional[pl.DataFrame] = None,
        beta_lag: int = 1,
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

        # Section / section_norm を付与（listed snapshotsがある場合）
        try:
            if snapshots:
                # snapshots(dict[str,pl.DataFrame]) → listed_info_df
                listed_frames = []
                for d, snap in snapshots.items():
                    if not snap.is_empty():
                        if "Date" not in snap.columns:
                            snap = snap.with_columns(
                                pl.lit(d)
                                .cast(pl.Utf8)
                                .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
                                .alias("Date")
                            )
                        select_cols = [c for c in ("Code", "MarketCode", "Date") if c in snap.columns]
                        listed_frames.append(snap.select(select_cols))
                if listed_frames:
                    from features.section_mapper import SectionMapper
                    listed_info_df = pl.concat(listed_frames, how="diagonal_relaxed")
                    mapper = SectionMapper()
                    mapping = mapper.create_section_mapping(listed_info_df)
                    df = mapper.attach_section_to_daily(df, mapping)
                    # section_norm を追加
                    def norm_expr():
                        return (
                            pl.when(pl.col("Section").is_in(["TSE1st","Prime","Prime Market","TSE Prime","東証プライム"]))
                              .then(pl.lit("TSEPrime"))
                              .when(pl.col("Section").is_in(["TSE2nd","Standard","Standard Market","TSE Standard","JASDAQ","JASDAQ Standard","Other","東証スタンダード"]))
                              .then(pl.lit("TSEStandard"))
                              .when(pl.col("Section").is_in(["Mothers","Growth","Growth Market","TSE Growth","JASDAQ Growth","東証グロース"]))
                              .then(pl.lit("TSEGrowth"))
                              .otherwise(pl.col("Section"))
                        )
                    df = df.with_columns(norm_expr().alias("section_norm"))
            else:
                # フォールバック: 欠損時は AllMarket / AllMarket を付与
                if "Section" not in df.columns:
                    df = df.with_columns(pl.lit("AllMarket").alias("Section"))
                if "section_norm" not in df.columns:
                    df = df.with_columns(pl.lit("AllMarket").alias("section_norm"))
        except Exception as e:
            logger.warning(f"  Failed to attach Section: {e}")
            if "Section" not in df.columns:
                df = df.with_columns(pl.lit("AllMarket").alias("Section"))
            if "section_norm" not in df.columns:
                df = df.with_columns(pl.lit("AllMarket").alias("section_norm"))

        # 財務諸表のローカルフォールバック（symlink or latest file）
        try:
            if (statements_df is None) or statements_df.is_empty():
                stm_path = self.output_dir / "event_raw_statements.parquet"
                if not stm_path.exists():
                    cands = sorted(self.output_dir.glob("event_raw_statements_*.parquet"))
                    if cands:
                        stm_path = cands[-1]
                if stm_path.exists():
                    logger.info(f"  Loading statements fallback from {stm_path}")
                    statements_df = pl.read_parquet(stm_path)
        except Exception as e:
            logger.warning(f"  Failed to load statements fallback: {e}")

        # 財務諸表データを結合
        if statements_df is not None and not statements_df.is_empty():
            logger.info(f"  Adding statement features: {len(statements_df)} records")
            # スキーマの簡易ログ（型不一致デバッグ用）
            try:
                key_cols = [
                    "Code","LocalCode","DisclosedDate","DisclosedTime","FiscalYear",
                    "NetSales","OperatingProfit","Profit",
                    "ForecastOperatingProfit","ForecastProfit",
                    "Equity","TotalAssets"
                ]
                schema_map = {c: str(statements_df.schema.get(c)) for c in key_cols if c in statements_df.columns}
                logger.info(f"  Statements key dtypes: {schema_map}")
            except Exception as e:
                logger.warning(f"  Failed to log statements schema: {e}")
            df = self.builder.add_statements_features(df, statements_df)

        # 市場（TOPIX）特徴量の統合（mkt_* + cross）
        try:
            df_before = len(df.columns)
            df = self.builder.add_topix_features(df, topix_df=topix_df, beta_lag=beta_lag)
            added = len(df.columns) - df_before
            if added > 0:
                logger.info(f"  Added {added} market features (TOPIX + cross)")
        except Exception as e:
            logger.warning(f"  Skipped TOPIX feature integration due to error: {e}")

        # Flow特徴量の統合（trades_spec）
        try:
            if trades_spec_df is None or trades_spec_df.is_empty():
                # フォールバック: output/ 以下のtrades_spec parquetを探索
                flow_cands = sorted(self.output_dir.rglob("trades_spec_*.parquet"))
                if flow_cands:
                    trades_spec_df = pl.read_parquet(flow_cands[-1])
            if trades_spec_df is not None and not trades_spec_df.is_empty():
                # listed_info_df（Section割当用）を構築
                listed_frames = []
                if snapshots:
                    for d, snap in snapshots.items():
                        if not snap.is_empty():
                            if "Date" not in snap.columns:
                                snap = snap.with_columns(pl.lit(d).cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias("Date"))
                            listed_frames.append(snap)
                listed_info_df = pl.concat(listed_frames, how="diagonal_relaxed") if listed_frames else None
                df = self.builder.add_flow_features(df, trades_spec_df, listed_info_df)
                logger.info("  Flow features integrated from trades_spec")
            else:
                logger.info("  No trades_spec available; skipping flow features")
        except Exception as e:
            logger.warning(f"  Skipped flow integration due to error: {e}")

        # Margin weekly (existing style): auto-discover under output/, skip if missing
        try:
            cands = sorted(self.output_dir.rglob("weekly_margin_interest_*.parquet"))
            wdf = pl.read_parquet(cands[-1]) if cands else None
            if wdf is not None and not wdf.is_empty():
                df = self.builder.add_margin_weekly_block(
                    df, wdf, lag_bdays_weekly=3, adv_window_days=20
                )
                logger.info("  Added weekly margin features")
            else:
                logger.info("  No weekly margin parquet found; skipping margin features")
        except Exception as e:
            logger.warning(f"  Skipped margin weekly integration due to error: {e}")

        # Finalize dataset for spec conformance (column set + names)
        try:
            df = self.builder.finalize_for_spec(df)
        except Exception as e:
            logger.warning(f"  Spec finalization skipped due to error: {e}")

        # Create metadata (after finalization)
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

        # Save dataset (builder returns parquet, csv (optional), metadata)
        parquet_path, csv_path, meta_path = self.builder.save_dataset(df, metadata)
        
        self.tracker.end_component(metric, records=len(df))
        
        return df, metadata, (parquet_path, csv_path, meta_path)

    async def run(
        self,
        use_jquants: bool = True,
        start_date: str = None,
        end_date: str = None,
    ):
        """Run the optimized pipeline."""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("OPTIMIZED ML DATASET PIPELINE V4")
        logger.info("With axis selection, diff detection, and event tracking")
        logger.info("(Note) For full enriched dataset builds, prefer run_full_dataset.py")
        logger.info("=" * 60)

        # Step 1: Get data
        if use_jquants:
            logger.info("Fetching data from JQuants API (optimized)...")
            price_df, statements_df, events, snapshots, topix_df, trades_spec_df = await self.fetch_jquants_data_optimized(
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
                    from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
                    membership_path = self.output_dir / "market_membership.parquet"
                    save_parquet_with_gcs(membership_df, membership_path)
                
                # securities_events生成
                events_df = self.event_detector.generate_securities_events_table()
                if not events_df.is_empty():
                    logger.info(f"Generated securities events: {len(events_df)} records")

                    # 保存
                    from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
                    events_path = self.output_dir / "securities_events.parquet"
                    save_parquet_with_gcs(events_df, events_path)
        else:
            logger.info("Creating sample data...")
            # Use stable facade import (avoids package path issues)
            try:
                from src.gogooku3.pipeline.builder import create_sample_data
            except Exception:
                from scripts.data.ml_dataset_builder import create_sample_data  # fallback
            price_df = create_sample_data(100, 300)
            statements_df = None
            events = []
            snapshots = {}
            # Try to load a local TOPIX parquet for offline market features
            try:
                import re as _re
                best = None
                best_span = -1
                for cand in sorted((self.output_dir).rglob('topix_history_*.parquet')):
                    m = _re.search(r"topix_history_(\d{8})_(\d{8})\.parquet$", cand.name)
                    if not m:
                        continue
                    span = int(m.group(2)) - int(m.group(1))
                    if span > best_span:
                        best = cand
                        best_span = span
                topix_df = pl.read_parquet(best) if best else pl.DataFrame()
            except Exception:
                topix_df = pl.DataFrame()
            # Load local trades_spec if available
            try:
                import re as _re
                flow_best = None
                for cand in sorted((self.output_dir).rglob('trades_spec_*.parquet')):
                    flow_best = cand
                trades_spec_df = pl.read_parquet(flow_best) if flow_best else pl.DataFrame()
            except Exception:
                trades_spec_df = pl.DataFrame()

        # カラム名の正規化 (再確認)
        if not price_df.is_empty():
            if "code" in price_df.columns and "Code" not in price_df.columns:
                price_df = price_df.rename({"code": "Code"})
            code_col = "Code" if "Code" in price_df.columns else price_df.columns[0]
            logger.info(f"Data loaded: {len(price_df)} rows, {price_df[code_col].n_unique()} stocks")
        else:
            logger.warning("Data loaded: empty price_df")

        # Step 2: Process pipeline
        logger.info("\nStep 2: Processing ML features...")
        # Parse beta_lag from args if available (fallback to env or default 1)
        beta_lag = int(os.getenv("BETA_LAG", "1"))
        df, metadata, file_paths = self.process_pipeline(price_df, statements_df, events, topix_df, snapshots, trades_spec_df, beta_lag)

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

        # Display cache summary
        logger.info("\n" + "=" * 60)
        logger.info(self.tracker.get_cache_summary())
        logger.info("=" * 60)

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
        default=None,
        help="Start date (YYYY-MM-DD). If omitted, uses ML_PIPELINE_START_DATE or auto-extends to satisfy minimum span.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--beta-lag",
        type=int,
        default=1,
        help="Lag to use for market returns in beta calc (0=no lag, 1=t-1)"
    )
    # (No enable flags for margin; integration is attempted automatically like other enrichments)

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
    # Propagate beta_lag via env for simplicity
    os.environ["BETA_LAG"] = str(args.beta_lag)
    df, metadata = await pipeline.run(
        use_jquants=args.jquants, start_date=args.start_date, end_date=args.end_date
    )

    return df, metadata


if __name__ == "__main__":
    # Run the async main function
    df, metadata = asyncio.run(main())
