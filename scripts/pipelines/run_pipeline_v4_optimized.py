#!/usr/bin/env python3
"""
Optimized ML Dataset Pipeline V4 - 最適化版
軸自動選択、差分検知、イベント追跡を含む完全最適化版
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import polars as pl
from dotenv import load_dotenv

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

from components.axis_decider import AxisDecider
from components.daily_quotes_by_code import DailyQuotesByCodeFetcher
from components.event_detector import EventDetector
from components.listed_info_manager import ListedInfoManager
from components.market_code_filter import MarketCodeFilter
from components.trading_calendar_fetcher import TradingCalendarFetcher

from src.gogooku3.pipeline.builder import MLDatasetBuilder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


SUPPORT_LOOKBACK_DAYS = 420  # approx 20 months to cover YoY/Z-score lookbacks
# Note: MIN_COLLECTION_DAYS removed - use requested dates + lookback only
# To fetch long-term data, specify explicit --start-date instead of forcing minimum span


# ========================================
# Cache Utility Functions
# ========================================

def _find_latest_with_date_range(
    glob: str, req_start: str, req_end: str
) -> dict | None:
    """Find cached file with complete or partial match for requested date range.

    Args:
        glob: Glob pattern to match files (e.g., "daily_quotes_*.parquet")
        req_start: Requested start date (YYYY-MM-DD)
        req_end: Requested end date (YYYY-MM-DD)

    Returns:
        Dictionary with cache info if found, None otherwise:
        {
            "path": Path to cache file,
            "cache_start": Cache start date (YYYY-MM-DD),
            "cache_end": Cache end date (YYYY-MM-DD),
            "match_type": "complete" | "partial",
            "missing_start": Start date to fetch (YYYY-MM-DD) or None,
            "missing_end": End date to fetch (YYYY-MM-DD) or None,
            "missing_ranges": List of (start, end) tuples for gaps outside the cache
        }
    """
    import re

    req_start_dt = datetime.strptime(req_start, "%Y-%m-%d")
    req_end_dt = datetime.strptime(req_end, "%Y-%m-%d")

    # Find all matching files under output/
    candidates = sorted(Path("output").rglob(glob))

    best_match = None
    best_coverage = 0.0  # Percentage of requested range covered

    for cand in reversed(candidates):  # Try newest first
        # Extract date range from filename: *_YYYYMMDD_YYYYMMDD.parquet
        match = re.search(r"_(\d{8})_(\d{8})\.parquet$", cand.name)
        if not match:
            continue

        file_start = datetime.strptime(match.group(1), "%Y%m%d")
        file_end = datetime.strptime(match.group(2), "%Y%m%d")

        # Check for any overlap
        overlap_start = max(file_start, req_start_dt)
        overlap_end = min(file_end, req_end_dt)

        if overlap_start <= overlap_end:
            # There is overlap
            overlap_days = (overlap_end - overlap_start).days + 1
            total_days = (req_end_dt - req_start_dt).days + 1
            coverage = overlap_days / total_days

            # Complete match (100% coverage)
            if file_start <= req_start_dt and file_end >= req_end_dt:
                logger.info(
                    f"📦 COMPLETE MATCH: {cand.name} covers {req_start} to {req_end}"
                )
                return {
                    "path": cand,
                    "cache_start": file_start.strftime("%Y-%m-%d"),
                    "cache_end": file_end.strftime("%Y-%m-%d"),
                    "match_type": "complete",
                    "missing_start": None,
                    "missing_end": None
                }

            # Partial match - keep track of best coverage
            if coverage > best_coverage:
                best_coverage = coverage

                # Determine missing ranges
                missing_ranges: list[tuple[str, str]] = []

                if req_start_dt < file_start:
                    # Need data before cache starts
                    missing_ranges.append((
                        req_start,
                        (file_start - timedelta(days=1)).strftime("%Y-%m-%d")
                    ))

                if req_end_dt > file_end:
                    # Need data after cache ends
                    trailing_gap_start = (file_end + timedelta(days=1)).strftime("%Y-%m-%d")
                    missing_ranges.append((trailing_gap_start, req_end))

                # Backwards compatibility: only expose single range via legacy keys
                if len(missing_ranges) == 1:
                    legacy_missing_start, legacy_missing_end = missing_ranges[0]
                else:
                    legacy_missing_start = None
                    legacy_missing_end = None

                best_match = {
                    "path": cand,
                    "cache_start": file_start.strftime("%Y-%m-%d"),
                    "cache_end": file_end.strftime("%Y-%m-%d"),
                    "match_type": "partial",
                    "missing_start": legacy_missing_start,
                    "missing_end": legacy_missing_end,
                    "missing_ranges": missing_ranges,
                    "coverage": coverage
                }

    if best_match:
        # Check minimum coverage threshold (Phase 2 optimization #1)
        min_coverage = float(os.getenv("MIN_CACHE_COVERAGE", "0.3"))

        # Try multi-cache if enabled and single file coverage is below threshold (Phase 2 optimization #2)
        enable_multi_cache = os.getenv("ENABLE_MULTI_CACHE", "1") == "1"
        if best_match['coverage'] < min_coverage and enable_multi_cache:
            logger.info(
                f"⚠️  Single file coverage too low ({best_match['coverage']*100:.1f}% < {min_coverage*100:.0f}% threshold)"
            )
            logger.info("   Trying multi-cache file combination...")

            # Try to find better coverage with multiple files
            multi_match = _find_best_cache_combination(glob, req_start, req_end, max_files=3)

            if multi_match and multi_match['coverage'] > best_match['coverage']:
                # Multi-cache provides better coverage
                logger.info(f"✅ Multi-cache improves coverage: {best_match['coverage']*100:.1f}% → {multi_match['coverage']*100:.1f}%")

                # Still check if even multi-cache meets threshold
                if multi_match['coverage'] < min_coverage:
                    logger.info(
                        f"⚠️  Even with multi-cache, coverage too low ({multi_match['coverage']*100:.1f}% < {min_coverage*100:.0f}% threshold), "
                        f"falling back to full API fetch"
                    )
                    return None

                # Return multi-cache result
                missing_ranges = multi_match.get("missing_ranges") or []
                if missing_ranges:
                    for range_start, range_end in missing_ranges:
                        logger.info(f"   Need to fetch: {range_start} to {range_end}")
                else:
                    logger.info("   Need to fetch: (none)")

                return multi_match
            else:
                # Multi-cache doesn't help, fall back to full API fetch
                logger.info("   Multi-cache didn't improve coverage, falling back to full API fetch")
                return None
        elif best_match['coverage'] < min_coverage:
            # Multi-cache disabled and coverage too low
            logger.info(
                f"⚠️  Coverage too low ({best_match['coverage']*100:.1f}% < {min_coverage*100:.0f}% threshold), "
                f"falling back to full API fetch for efficiency"
            )
            return None

        logger.info(
            f"🔄 PARTIAL MATCH: {best_match['path'].name} covers "
            f"{best_match['coverage']*100:.1f}% ({best_match['cache_start']} to {best_match['cache_end']})"
        )
        missing_ranges = best_match.get("missing_ranges") or []
        if missing_ranges:
            for range_start, range_end in missing_ranges:
                logger.info(f"   Need to fetch: {range_start} to {range_end}")
        else:
            logger.info("   Need to fetch: (none)")

    return best_match


def _find_best_cache_combination(
    glob: str, req_start: str, req_end: str, max_files: int = 3
) -> dict | None:
    """Find the best combination of multiple cache files to maximize coverage.

    This function attempts to find multiple cache files that together provide
    better coverage than a single file. It's useful when you have fragmented
    cache files (e.g., from different time periods).

    Args:
        glob: Glob pattern to match files (e.g., "daily_quotes_*.parquet")
        req_start: Requested start date (YYYY-MM-DD)
        req_end: Requested end date (YYYY-MM-DD)
        max_files: Maximum number of cache files to combine (default: 3)

    Returns:
        Dictionary with multi-cache info if found, None otherwise:
        {
            "cache_files": [
                {"path": Path, "start": str, "end": str},
                ...
            ],
            "match_type": "multi-partial",
            "missing_ranges": List of (start, end) tuples for remaining gaps,
            "coverage": float (0.0-1.0),
            "total_cached_days": int
        }
    """
    import re

    req_start_dt = datetime.strptime(req_start, "%Y-%m-%d")
    req_end_dt = datetime.strptime(req_end, "%Y-%m-%d")
    total_days = (req_end_dt - req_start_dt).days + 1

    # Find all matching files with overlap
    candidates = []
    for cand in sorted(Path("output").rglob(glob)):
        match = re.search(r"_(\d{8})_(\d{8})\.parquet$", cand.name)
        if not match:
            continue

        file_start = datetime.strptime(match.group(1), "%Y%m%d")
        file_end = datetime.strptime(match.group(2), "%Y%m%d")

        # Check for any overlap with requested range
        overlap_start = max(file_start, req_start_dt)
        overlap_end = min(file_end, req_end_dt)

        if overlap_start <= overlap_end:
            candidates.append({
                "path": cand,
                "start": file_start,
                "end": file_end,
                "overlap_start": overlap_start,
                "overlap_end": overlap_end
            })

    if len(candidates) == 0:
        return None

    # Sort by start date
    candidates.sort(key=lambda x: x["start"])

    # Greedy algorithm: Select files that fill gaps
    # Start with the file that covers the earliest part of the requested range
    selected = []
    covered_ranges = []  # List of (start_dt, end_dt) tuples

    for cand in candidates:
        if len(selected) >= max_files:
            break

        # Check if this file adds new coverage
        adds_coverage = False

        if len(covered_ranges) == 0:
            # First file - take it if it overlaps
            adds_coverage = True
        else:
            # Check if this file fills a gap or extends coverage
            cand_range = (cand["overlap_start"], cand["overlap_end"])

            for covered_start, covered_end in covered_ranges:
                # Check if this file extends coverage before or after
                if cand_range[1] >= covered_start and cand_range[0] <= covered_end:
                    # Overlaps or extends existing coverage
                    if cand_range[0] < covered_start or cand_range[1] > covered_end:
                        adds_coverage = True
                        break

            # Also check if it fills a gap between covered ranges
            if not adds_coverage and len(covered_ranges) >= 2:
                for i in range(len(covered_ranges) - 1):
                    gap_start = covered_ranges[i][1] + timedelta(days=1)
                    gap_end = covered_ranges[i+1][0] - timedelta(days=1)
                    if cand_range[0] <= gap_end and cand_range[1] >= gap_start:
                        adds_coverage = True
                        break

        if adds_coverage:
            selected.append(cand)
            covered_ranges.append((cand["overlap_start"], cand["overlap_end"]))

    if len(selected) <= 1:
        # No benefit from multi-file approach
        return None

    # Merge overlapping ranges to calculate actual coverage
    covered_ranges.sort()
    merged_ranges = []
    current_start, current_end = covered_ranges[0]

    for start, end in covered_ranges[1:]:
        if start <= current_end + timedelta(days=1):
            # Overlapping or contiguous - merge
            current_end = max(current_end, end)
        else:
            # Gap - save current range and start new one
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end
    merged_ranges.append((current_start, current_end))

    # Calculate total covered days
    covered_days = sum((end - start).days + 1 for start, end in merged_ranges)
    coverage = covered_days / total_days

    # Calculate missing ranges
    missing_ranges: list[tuple[str, str]] = []

    if merged_ranges[0][0] > req_start_dt:
        # Gap before first covered range
        missing_ranges.append((
            req_start,
            (merged_ranges[0][0] - timedelta(days=1)).strftime("%Y-%m-%d")
        ))

    # Gaps between covered ranges
    for i in range(len(merged_ranges) - 1):
        gap_start = merged_ranges[i][1] + timedelta(days=1)
        gap_end = merged_ranges[i+1][0] - timedelta(days=1)
        if gap_start <= gap_end:
            missing_ranges.append((
                gap_start.strftime("%Y-%m-%d"),
                gap_end.strftime("%Y-%m-%d")
            ))

    if merged_ranges[-1][1] < req_end_dt:
        # Gap after last covered range
        missing_ranges.append((
            (merged_ranges[-1][1] + timedelta(days=1)).strftime("%Y-%m-%d"),
            req_end
        ))

    # Format result
    cache_files = [
        {
            "path": c["path"],
            "start": c["start"].strftime("%Y-%m-%d"),
            "end": c["end"].strftime("%Y-%m-%d")
        }
        for c in selected
    ]

    logger.info(
        f"🔗 MULTI-CACHE MATCH: {len(selected)} files provide {coverage*100:.1f}% coverage"
    )
    for cf in cache_files:
        logger.info(f"   - {cf['path'].name} ({cf['start']} to {cf['end']})")

    return {
        "cache_files": cache_files,
        "match_type": "multi-partial",
        "missing_ranges": missing_ranges,
        "coverage": coverage,
        "total_cached_days": covered_days
    }


def _is_cache_valid(cache_info: dict | Path | None, max_age_days: int) -> bool:
    """Check if cached file exists and is not stale.

    Args:
        cache_info: Dictionary from _find_latest_with_date_range() or Path or None
        max_age_days: Maximum age in days before considering stale

    Returns:
        True if file exists and is fresh enough, False otherwise
    """
    # Handle both dict (new format) and Path (legacy format)
    if cache_info is None:
        return False

    if isinstance(cache_info, dict):
        file_path = cache_info.get("path")
    else:
        file_path = cache_info

    if file_path is None or not file_path.exists():
        return False

    try:
        file_age_seconds = time.time() - file_path.stat().st_mtime
        file_age_days = file_age_seconds / (24 * 3600)
        is_valid = file_age_days <= max_age_days

        if is_valid:
            logger.debug(
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


def _merge_contiguous_ranges(ranges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge contiguous date ranges to minimize API calls.

    Args:
        ranges: List of (start_date, end_date) tuples in YYYY-MM-DD format

    Returns:
        Merged list of date ranges where contiguous ranges are combined

    Example:
        Input: [('2025-01-01', '2025-01-05'), ('2025-01-06', '2025-01-10')]
        Output: [('2025-01-01', '2025-01-10')]
    """
    if not ranges:
        return []

    # Convert to datetime for easier comparison
    dt_ranges = []
    for start_str, end_str in ranges:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
        dt_ranges.append((start_dt, end_dt))

    # Sort by start date
    dt_ranges.sort()

    # Merge contiguous ranges
    merged = []
    current_start, current_end = dt_ranges[0]

    for start, end in dt_ranges[1:]:
        # Check if this range is contiguous with the current range
        if start <= current_end + timedelta(days=1):
            # Contiguous or overlapping - extend current range
            current_end = max(current_end, end)
        else:
            # Gap - save current range and start a new one
            merged.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))
            current_start, current_end = start, end

    # Add the last range
    merged.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))

    # Log merging result if ranges were combined
    if len(merged) < len(ranges):
        logger.info(f"   Optimized: Merged {len(ranges)} ranges into {len(merged)} contiguous ranges")

    return merged


def _align_frames_for_concat(
    left: pl.DataFrame | None, right: pl.DataFrame | None
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """
    Align schemas of cached/new frames so they can be concatenated safely.

    Args:
        left: Cached DataFrame (may be None)
        right: Newly fetched DataFrame (may be None)

    Returns:
        Tuple of aligned DataFrames (left_aligned, right_aligned)
    """

    if (
        left is None
        or right is None
        or left.is_empty()
        or right.is_empty()
    ):
        return left, right

    schema_order: list[str] = []
    schema_dtypes: dict[str, pl.DataType] = {}

    for df in (left, right):
        for name, dtype in zip(df.columns, df.dtypes, strict=False):
            if name not in schema_dtypes:
                schema_order.append(name)
                schema_dtypes[name] = dtype

    def _project(df: pl.DataFrame) -> pl.DataFrame:
        exprs: list[pl.Expr] = []
        df_schema = df.schema
        for name in schema_order:
            dtype = schema_dtypes[name]
            if name in df_schema:
                expr = pl.col(name)
                if df_schema[name] != dtype:
                    expr = expr.cast(dtype, strict=False)
                exprs.append(expr.alias(name))
            else:
                exprs.append(pl.lit(None, dtype=dtype).alias(name))
        return df.select(exprs)

    try:
        return _project(left), _project(right)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(f"Failed to align cached/new schema for concat: {exc}")
        return left, right


def _load_cache_data(cache_info: dict, start_date: str, end_date: str, date_column: str = "Date") -> pl.DataFrame | None:
    """Load cached data, handling both single-file and multi-file cases.

    Args:
        cache_info: Cache info dictionary from _find_latest_with_date_range()
        start_date: Start date to filter (YYYY-MM-DD)
        end_date: End date to filter (YYYY-MM-DD)
        date_column: Name of the date column to filter (default: "Date")

    Returns:
        Merged DataFrame from cache files, or None if loading fails
    """
    try:
        match_type = cache_info.get("match_type")

        if match_type == "multi-partial":
            # Load and merge multiple cache files
            cache_files = cache_info.get("cache_files", [])
            if not cache_files:
                return None

            logger.info(f"   Loading {len(cache_files)} cache files...")
            all_frames = []

            for cf in cache_files:
                cache_path = cf["path"]
                try:
                    df = pl.read_parquet(cache_path)
                    logger.info(f"   Loaded: {cache_path.name} ({len(df):,} records)")
                    all_frames.append(df)
                except Exception as e:
                    logger.warning(f"   Failed to load {cache_path.name}: {e}")

            if not all_frames:
                return None

            # Merge all frames
            merged_df = pl.concat(all_frames)
            logger.info(f"   Merged: {len(merged_df):,} total records from {len(all_frames)} files")

            # Filter to requested date range
            if date_column in merged_df.columns:
                # Handle Date type (for TOPIX)
                if merged_df.schema[date_column] == pl.Date:
                    from_date_typed = pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d")
                    to_date_typed = pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")
                    merged_df = merged_df.filter(
                        (pl.col(date_column) >= from_date_typed) & (pl.col(date_column) <= to_date_typed)
                    )
                else:
                    # String type
                    merged_df = merged_df.filter(
                        (pl.col(date_column) >= start_date) & (pl.col(date_column) <= end_date)
                    )

            return merged_df
        else:
            # Single file case (complete or partial match)
            cache_path = cache_info.get("path")
            if not cache_path:
                return None

            df = pl.read_parquet(cache_path)

            # Filter to requested date range
            if date_column in df.columns:
                # Handle Date type (for TOPIX)
                if df.schema[date_column] == pl.Date:
                    from_date_typed = pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d")
                    to_date_typed = pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")
                    df = df.filter(
                        (pl.col(date_column) >= from_date_typed) & (pl.col(date_column) <= to_date_typed)
                    )
                else:
                    # String type
                    df = df.filter(
                        (pl.col(date_column) >= start_date) & (pl.col(date_column) <= end_date)
                    )

            return df

    except Exception as e:
        logger.warning(f"Failed to load cache data: {e}")
        return None


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
        self.metrics: list[PerformanceMetrics] = []
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

    def get_summary(self) -> dict:
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
        business_days: list[str],
        target_codes: set[str] | None = None
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
        cached_data = None  # Will hold cached DataFrame for partial matches
        missing_business_days = []  # Business days to fetch from API

        if use_cache and business_days:
            start_date = business_days[0]
            end_date = business_days[-1]

            cache_info = _find_latest_with_date_range(
                "daily_quotes_*.parquet",
                start_date,
                end_date
            )

            if cache_info and _is_cache_valid(cache_info, max_cache_age_days):
                match_type = cache_info.get("match_type") if isinstance(cache_info, dict) else "complete"

                if match_type == "complete":
                    # Complete match - return directly
                    try:
                        price_df = pl.read_parquet(cache_info["path"])

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

                elif match_type == "partial" or match_type == "multi-partial":
                    # Partial match (single or multi-file) - load cached data and prepare to fetch missing range
                    try:
                        logger.info(f"🔄 Using {match_type.replace('-', ' ')} cache match ({cache_info['coverage']*100:.1f}% coverage)")
                        cached_data = _load_cache_data(cache_info, start_date, end_date, date_column="Date")

                        # Calculate which business days are missing
                        missing_ranges = cache_info.get("missing_ranges") or []
                        if not missing_ranges and cache_info.get("missing_start") and cache_info.get("missing_end"):
                            missing_ranges = [(cache_info["missing_start"], cache_info["missing_end"])]

                        missing_business_days = []
                        range_summaries = []
                        for range_start, range_end in missing_ranges:
                            range_days = [
                                d for d in business_days
                                if d >= range_start and d <= range_end
                            ]
                            missing_business_days.extend(range_days)
                            range_summaries.append((range_start, range_end, len(range_days)))
                        if missing_business_days:
                            missing_business_days = sorted(set(missing_business_days))

                        logger.info(f"   Cached: {len(cached_data):,} records from cache")
                        if range_summaries:
                            for start, end, count in range_summaries:
                                logger.info(f"   Need to fetch: {count} business days ({start} to {end})")
                        else:
                            logger.info("   Need to fetch: 0 business days (range already satisfied?)")

                        # Record partial cache hit (estimate time saved)
                        coverage = cache_info.get("coverage", 0.5)
                        time_saved = 45.0 * coverage  # Proportional savings
                        self.tracker.record_cache_hit(f"Daily Quotes (partial {coverage*100:.0f}%)", time_saved)

                    except Exception as e:
                        logger.warning(f"⚠️  Failed to load partial cache: {e}, fetching all from API")
                        cached_data = None
                        missing_business_days = []
                        self.tracker.record_cache_miss("Daily Quotes")
            else:
                self.tracker.record_cache_miss("Daily Quotes")

        # If we have partial cache, only fetch missing days; otherwise fetch all
        days_to_fetch = missing_business_days if missing_business_days else business_days

        # Step 2: Fetch from API (if needed)
        new_data = None

        if days_to_fetch:
            # 軸の自動選択
            axis, axis_metrics = await self.axis_decider.get_optimal_axis(
                session,
                sample_days=days_to_fetch[:3] if len(days_to_fetch) > 3 else days_to_fetch,
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
                            days_to_fetch[0], days_to_fetch[-1]
                        )
                        tasks.append(task)

                    results = await asyncio.gather(*tasks)
                    api_calls += len(tasks)

                    for df in results:
                        if not df.is_empty():
                            all_dfs.append(df)

                    logger.info(f"  Progress: {min(i+batch_size, len(codes_list))}/{len(codes_list)} stocks")

                if all_dfs:
                    new_data = pl.concat(all_dfs)
            else:
                # 日付軸で取得（既存の実装を使用）
                logger.info(f"Fetching by date axis for {len(days_to_fetch)} days...")
                new_data = await self.fetch_daily_quotes_bulk(session, days_to_fetch)

                # 市場フィルタリング
                if target_codes and not new_data.is_empty():
                    original_count = len(new_data)
                    new_data = new_data.filter(pl.col("Code").is_in(target_codes))
                    logger.info(f"Filtered: {original_count} → {len(new_data)} records")

        # Step 3: Merge cached + new data (if we have partial cache)
        if cached_data is not None and new_data is not None and not new_data.is_empty():
            # Merge cached and new data
            logger.info(f"🔀 Merging cached ({len(cached_data):,}) + new ({len(new_data):,}) data...")
            cached_data, new_data = _align_frames_for_concat(cached_data, new_data)
            final_df = pl.concat([cached_data, new_data])
            logger.info(f"   Total: {len(final_df):,} records after merge")
        elif cached_data is not None:
            # Only cached data (all business days were in cache)
            final_df = cached_data
        elif new_data is not None and not new_data.is_empty():
            # Only new data (no cache or cache miss)
            final_df = new_data
        else:
            # No data at all
            self.tracker.end_component(metric)
            return pl.DataFrame()

        # Apply market filtering to final result
        if target_codes and "Code" in final_df.columns and cached_data is not None:
            # Need to filter if we used cached data (already filtered for new_data above)
            original_count = len(final_df)
            final_df = final_df.filter(pl.col("Code").is_in(target_codes))
            logger.info(f"  Final filter: {original_count} → {len(final_df)} records")

        # Step 4: Save extended cache (with full date range)
        if use_cache and business_days and not final_df.is_empty():
            try:
                from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
                start_date = business_days[0]
                end_date = business_days[-1]
                cache_dir = Path("output/raw/prices")
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"daily_quotes_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"

                # Save merged data with full date range
                save_parquet_with_gcs(final_df, cache_path, auto_sync=False)
                logger.info(f"💾 Saved extended cache: {cache_path.name} ({len(final_df):,} records)")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

        # Calculate API calls for tracking
        api_calls_made = len(days_to_fetch) if days_to_fetch and new_data is not None else 0
        self.tracker.end_component(metric, api_calls=api_calls_made, records=len(final_df))
        return final_df

    async def fetch_daily_quotes_bulk(
        self,
        session: aiohttp.ClientSession,
        business_days: list[str],
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
        business_days: list[str],
        daily_quotes_df: pl.DataFrame | None = None
    ) -> tuple[dict[str, pl.DataFrame], list[dict]]:
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
        self, session: aiohttp.ClientSession, business_days: list[str]
    ) -> pl.DataFrame:
        """date軸での財務諸表取得 - キャッシュ優先（20%高速化）+ 部分マッチ対応"""
        metric = self.tracker.start_component("statements_by_date")

        # Check environment variables for cache control
        use_cache = os.getenv("USE_CACHE", "1") == "1"
        max_cache_age_days = int(os.getenv("CACHE_MAX_AGE_DAYS", "7"))

        # Step 1: キャッシュをチェック
        cached_data = None
        missing_business_days = []

        if use_cache and business_days:
            start_date = business_days[0]
            end_date = business_days[-1]

            cache_info = _find_latest_with_date_range(
                "event_raw_statements_*.parquet",
                start_date,
                end_date
            )

            if cache_info and _is_cache_valid(cache_info, max_cache_age_days):
                match_type = cache_info.get("match_type") if isinstance(cache_info, dict) else "complete"

                if match_type == "complete":
                    # Complete match - return directly
                    try:
                        statements_df = pl.read_parquet(cache_info["path"])

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

                elif match_type == "partial" or match_type == "multi-partial":
                    # Partial match (single or multi-file) - load cached data and prepare to fetch missing range
                    try:
                        logger.info(f"🔄 Using {match_type.replace('-', ' ')} cache match ({cache_info['coverage']*100:.1f}% coverage)")

                        # Determine which date column to use
                        # Note: We'll use DisclosedDate as the primary column
                        date_col = "DisclosedDate"  # Default, will be checked in helper
                        cached_data = _load_cache_data(cache_info, start_date, end_date, date_column=date_col)

                        # Calculate missing business days
                        missing_ranges = cache_info.get("missing_ranges") or []
                        if not missing_ranges and cache_info.get("missing_start") and cache_info.get("missing_end"):
                            missing_ranges = [(cache_info["missing_start"], cache_info["missing_end"])]

                        missing_business_days = []
                        range_summaries = []
                        for range_start, range_end in missing_ranges:
                            range_days = [
                                d for d in business_days
                                if d >= range_start and d <= range_end
                            ]
                            missing_business_days.extend(range_days)
                            range_summaries.append((range_start, range_end, len(range_days)))
                        if missing_business_days:
                            missing_business_days = sorted(set(missing_business_days))

                        logger.info(f"   Cached: {len(cached_data):,} records from cache")
                        if range_summaries:
                            for start, end, count in range_summaries:
                                logger.info(f"   Need to fetch: {count} business days ({start} to {end})")
                        else:
                            logger.info("   Need to fetch: 0 business days (range already satisfied?)")

                        # Record partial cache hit
                        coverage = cache_info.get("coverage", 0.5)
                        time_saved = 30.0 * coverage
                        self.tracker.record_cache_hit(f"Statements (partial {coverage*100:.0f}%)", time_saved)

                    except Exception as e:
                        logger.warning(f"⚠️  Failed to load partial cache: {e}, fetching all from API")
                        cached_data = None
                        missing_business_days = []
                        self.tracker.record_cache_miss("Statements")
            else:
                self.tracker.record_cache_miss("Statements")

        # If we have partial cache, only fetch missing days; otherwise fetch all
        days_to_fetch = missing_business_days if missing_business_days else business_days

        # Step 2: Fetch from API (if needed)
        new_data = None

        if days_to_fetch:
            url = f"{self.base_url}/fins/statements"
            headers = {"Authorization": f"Bearer {self.id_token}"}

            all_statements = []

            for date in days_to_fetch:
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

            if all_statements:
                new_data = pl.DataFrame(all_statements)

        # Step 3: Merge cached + new data
        if cached_data is not None and new_data is not None and not new_data.is_empty():
            logger.info(f"🔀 Merging cached ({len(cached_data):,}) + new ({len(new_data):,}) statements...")
            final_df = pl.concat([cached_data, new_data])
            logger.info(f"   Total: {len(final_df):,} records after merge")
        elif cached_data is not None:
            final_df = cached_data
        elif new_data is not None and not new_data.is_empty():
            final_df = new_data
        else:
            self.tracker.end_component(metric, api_calls=0, records=0)
            return pl.DataFrame()

        # Step 4: Save extended cache
        if use_cache and business_days and not final_df.is_empty():
            try:
                from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
                start_date = business_days[0]
                end_date = business_days[-1]
                cache_dir = Path("output/raw/statements")
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"event_raw_statements_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                save_parquet_with_gcs(final_df, cache_path, auto_sync=False)
                logger.info(f"💾 Saved extended cache: {cache_path.name} ({len(final_df):,} records)")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

        api_calls_made = len(days_to_fetch) if days_to_fetch and new_data is not None else 0
        self.tracker.end_component(metric, api_calls=api_calls_made, records=len(final_df))
        return final_df

    async def fetch_topix_data(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch TOPIX index data - キャッシュ優先（5%高速化）+ 部分マッチ対応"""

        # Check environment variables for cache control
        use_cache = os.getenv("USE_CACHE", "1") == "1"
        max_cache_age_days = int(os.getenv("CACHE_MAX_AGE_DAYS", "7"))

        # Step 1: キャッシュをチェック
        cached_data = None
        missing_ranges: list[tuple[str, str]] = []

        if use_cache:
            cache_info = _find_latest_with_date_range(
                "topix_history_*.parquet",
                from_date,
                to_date
            )

            if cache_info and _is_cache_valid(cache_info, max_cache_age_days):
                match_type = cache_info.get("match_type") if isinstance(cache_info, dict) else "complete"

                if match_type == "complete":
                    # Complete match - return directly
                    try:
                        topix_df = pl.read_parquet(cache_info["path"])

                        # 日付範囲フィルタ (Date型の列と比較するため、文字列をDate型に変換)
                        if "Date" in topix_df.columns:
                            from_date_typed = pl.lit(from_date).str.strptime(pl.Date, "%Y-%m-%d")
                            to_date_typed = pl.lit(to_date).str.strptime(pl.Date, "%Y-%m-%d")
                            topix_df = topix_df.filter(
                                (pl.col("Date") >= from_date_typed) & (pl.col("Date") <= to_date_typed)
                            )

                        # キャッシュヒットを記録
                        self.tracker.record_cache_hit("TOPIX", 3.5)  # 平均2-5秒の中央値

                        return topix_df
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to load cache: {e}, falling back to API")
                        self.tracker.record_cache_miss("TOPIX")

                elif match_type == "partial" or match_type == "multi-partial":
                    # Partial match (single or multi-file) - load cached data and prepare to fetch missing range
                    try:
                        logger.info(f"🔄 Using {match_type.replace('-', ' ')} cache match ({cache_info['coverage']*100:.1f}% coverage)")
                        cached_data = _load_cache_data(cache_info, from_date, to_date, date_column="Date")

                        # Get missing date ranges
                        missing_ranges = cache_info.get("missing_ranges") or []
                        if not missing_ranges and cache_info.get("missing_start") and cache_info.get("missing_end"):
                            missing_ranges = [(cache_info["missing_start"], cache_info["missing_end"])]

                        logger.info(f"   Cached: {len(cached_data):,} records from cache")
                        if missing_ranges:
                            for start, end in missing_ranges:
                                logger.info(f"   Need to fetch: {start} to {end}")
                        else:
                            logger.info("   Need to fetch: (no gaps detected)")

                        # Record partial cache hit
                        coverage = cache_info.get("coverage", 0.5)
                        time_saved = 3.5 * coverage
                        self.tracker.record_cache_hit(f"TOPIX (partial {coverage*100:.0f}%)", time_saved)

                    except Exception as e:
                        logger.warning(f"⚠️  Failed to load partial cache: {e}, fetching all from API")
                        cached_data = None
                        missing_ranges = []
                        self.tracker.record_cache_miss("TOPIX")
            else:
                self.tracker.record_cache_miss("TOPIX")

        # Determine what to fetch
        if cached_data is not None:
            ranges_to_fetch = missing_ranges
        else:
            ranges_to_fetch = [(from_date, to_date)]

        # Phase 2 optimization #3: Merge contiguous ranges to reduce API calls
        if len(ranges_to_fetch) > 1:
            ranges_to_fetch = _merge_contiguous_ranges(ranges_to_fetch)

        # Step 2: Fetch from API (if needed)
        new_data = None

        if ranges_to_fetch:
            url = f"{self.base_url}/indices/topix"
            headers = {"Authorization": f"Bearer {self.id_token}"}

            fetched_frames: list[pl.DataFrame] = []

            for fetch_from, fetch_to in ranges_to_fetch:
                from_api = fetch_from.replace("-", "")
                to_api = fetch_to.replace("-", "")

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
                    frame = pl.DataFrame(all_data)
                    if "Date" in frame.columns:
                        frame = frame.with_columns(
                            pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
                        )
                    for c in ("Open", "High", "Low", "Close", "Volume"):
                        if c in frame.columns:
                            frame = frame.with_columns(pl.col(c).cast(pl.Float64, strict=False))
                    frame = frame.sort("Date")
                    fetched_frames.append(frame)

            if fetched_frames:
                new_data = pl.concat(fetched_frames).sort("Date")

        # Step 3: Merge cached + new data
        if cached_data is not None and new_data is not None and not new_data.is_empty():
            logger.info(f"🔀 Merging cached ({len(cached_data):,}) + new ({len(new_data):,}) TOPIX data...")
            final_df = pl.concat([cached_data, new_data]).sort("Date")
            logger.info(f"   Total: {len(final_df):,} records after merge")
        elif cached_data is not None:
            final_df = cached_data
        elif new_data is not None and not new_data.is_empty():
            final_df = new_data
        else:
            return pl.DataFrame()

        # Step 4: Save extended cache
        if use_cache and not final_df.is_empty():
            try:
                from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs
                cache_dir = Path("output/raw/indices")
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"topix_history_{from_date.replace('-', '')}_{to_date.replace('-', '')}.parquet"
                save_parquet_with_gcs(final_df, cache_path, auto_sync=False)
                logger.info(f"💾 Saved extended cache: {cache_path.name} ({len(final_df):,} records)")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

        return final_df

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
        if output_dir is None:
            output_dir = Path(os.getenv("OUTPUT_DIR", "/home/ubuntu/gogooku3/output"))
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            # Default to 1 year ago if not specified (instead of forcing 10 years)
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        # Support rolling contracts (e.g., "last 10 years")
        # Priority: JQUANTS_SUBSCRIPTION_START > dynamic calculation from JQUANTS_CONTRACT_YEARS
        subscription_start_str = os.getenv("JQUANTS_SUBSCRIPTION_START")

        if subscription_start_str:
            # Explicit start date provided
            try:
                subscription_start_dt = datetime.strptime(subscription_start_str, "%Y-%m-%d")
            except ValueError:
                logger.warning(
                    "Invalid JQUANTS_SUBSCRIPTION_START=%s; falling back to rolling contract",
                    subscription_start_str,
                )
                subscription_start_str = None

        if not subscription_start_str:
            # Dynamic rolling contract (e.g., last 10 years from today)
            contract_years = int(os.getenv("JQUANTS_CONTRACT_YEARS", "10"))
            subscription_start_dt = datetime.now() - timedelta(days=365 * contract_years + 2)  # +2 for leap years
            logger.info(
                "Using rolling %d-year contract: subscription starts from %s (dynamic)",
                contract_years,
                subscription_start_dt.strftime("%Y-%m-%d"),
            )

        # Note: No longer enforce MIN_COLLECTION_DAYS
        # Just use requested dates + lookback for technical indicators
        # If you want long-term data, specify explicit --start-date

        support_start_dt = start_dt - timedelta(days=SUPPORT_LOOKBACK_DAYS)
        if support_start_dt < subscription_start_dt:
            logger.warning(
                "⚠️  Lookback start %s is before subscription coverage %s; clamping to %s",
                support_start_dt.strftime("%Y-%m-%d"),
                subscription_start_dt.strftime("%Y-%m-%d"),
                subscription_start_dt.strftime("%Y-%m-%d"),
            )
            logger.warning("   (This is normal for early dates - technical indicators may have less history)")
            support_start_dt = subscription_start_dt
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
                    save_parquet_with_gcs(statements_df, out_path, auto_sync=False)
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
        statements_df: pl.DataFrame | None = None,
        events: list[dict] | None = None,
        topix_df: pl.DataFrame | None = None,
        snapshots: dict[str, pl.DataFrame] | None = None,
        trades_spec_df: pl.DataFrame | None = None,
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
                from scripts.data.ml_dataset_builder import (
                    create_sample_data,  # fallback
                )
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
