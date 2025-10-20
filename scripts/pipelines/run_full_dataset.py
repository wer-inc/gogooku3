#!/usr/bin/env python3
"""
One-shot: Build a 5-year, fully enriched ML dataset (prices + TA + statements + TOPIX + flow)

This wrapper hides intermediate steps so you only run a single command.

Flow:
  1) Fetch 5y trade-spec (markets/trades_spec) and save as Parquet
  2) Run optimized base pipeline (prices + technical + pandas-ta + statements)
  3) Enrich with TOPIX (mkt_* + cross features) and flow (flow_*)
  4) Save as ml_dataset_latest_full.parquet (+ metadata)

Usage:
  python scripts/pipelines/run_full_dataset.py --jquants \
    --start-date 2020-09-03 --end-date 2025-09-03 \
    [--topix-parquet output/topix_history_YYYYMMDD_YYYYMMDD.parquet] \
    [--statements-parquet output/event_raw_statements_YYYYMMDD_YYYYMMDD.parquet] \
    [--enable-sector-short-selling] \
    [--sector-short-selling-parquet output/sector_short_selling_YYYYMMDD_YYYYMMDD.parquet]

If dates are omitted, it defaults to last ~5 years.
Requires .env with JQUANTS_AUTH_EMAIL, JQUANTS_AUTH_PASSWORD.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Awaitable
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import polars as pl
from dotenv import load_dotenv

# Ensure project root is on sys.path so that `scripts` is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env from project root (for JQuants credentials etc.)
env_path = ROOT / ".env"
try:
    if env_path.exists():
        # override=True so that .env values win over empty/placeholder envs
        load_dotenv(env_path, override=True)
except Exception:
    pass

# Import JQuants fetcher to get trade-spec directly (moved out of _archive)
from scripts.components.trading_calendar_fetcher import (
    TradingCalendarFetcher,  # type: ignore
)
from scripts.pipelines.run_pipeline_v4_optimized import JQuantsPipelineV4Optimized
from src.gogooku3.components.jquants_async_fetcher import (
    JQuantsAsyncFetcher,  # type: ignore
)
from src.pipeline.full_dataset import enrich_and_save

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_full_dataset")

# Defaults
DEFAULT_LOOKBACK_DAYS = 1826  # ~5 years
FLOW_SUPPORT_LOOKBACK_DAYS = 420  # ensure >52w history for flow z-scores


def _get_jquants_plan_tier() -> str:
    """
    Get J-Quants plan tier from environment variable.

    Returns:
        str: Plan tier ('standard' or 'premium'), defaults to 'standard'
    """
    return os.getenv("JQUANTS_PLAN_TIER", "standard").lower()


def _is_futures_available() -> bool:
    """
    Check if futures API is available based on J-Quants plan tier.

    Futures data (/derivatives/futures) is only available on Premium plan.

    Returns:
        bool: True if Premium plan (futures available), False otherwise
    """
    plan_tier = _get_jquants_plan_tier()
    is_available = plan_tier == "premium"

    if is_available:
        logger.debug("Futures API enabled (Premium plan detected)")
    else:
        logger.debug(
            f"Futures API disabled (plan tier: {plan_tier}, requires: premium)"
        )

    return is_available


def _mask_email(value: str) -> str:
    """Return a masked representation of the J-Quants account email."""
    if "@" not in value:
        return value[:2] + "***" if value else ""
    local, domain = value.split("@", 1)
    masked_local = local[:2] + "***" if len(local) > 2 else "***"
    return f"{masked_local}@{domain}"


def _check_jquants_credentials(*, strict: bool = False) -> bool:
    """Verify that J-Quants credentials are present for API-backed runs."""
    email = os.getenv("JQUANTS_AUTH_EMAIL", "").strip()
    password = os.getenv("JQUANTS_AUTH_PASSWORD", "").strip()
    if email and password:
        logger.info("J-Quants credentials detected (email=%s)", _mask_email(email))
        return True

    message = (
        "J-Quants credentials missing. Set JQUANTS_AUTH_EMAIL and "
        "JQUANTS_AUTH_PASSWORD in your environment or .env file."
    )
    if strict:
        logger.error(message)
    else:
        logger.warning(message)
    return False


def _check_gpu_graph_support(*, strict: bool = False) -> bool:
    """Inspect whether CuPy GPU dependencies are ready (cuGraph not required)."""
    try:
        import cupy as cp  # type: ignore

        # cuGraph は不要（graph_builder_gpu.py が CuPy のみで動作）

        device_count = cp.cuda.runtime.getDeviceCount()  # type: ignore[attr-defined]
        if device_count <= 0:
            raise RuntimeError("CuPy reports no CUDA devices")
        logger.info(
            "GPU graph dependencies detected (CuPy %s, CUDA devices=%d)",
            getattr(cp, "__version__", "?"),
            device_count,
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive logging only
        level = logger.error if strict else logger.warning
        level(
            "GPU graph dependencies unavailable (CuPy). "
            "Falling back to CPU graph features will significantly slow the build.%s",
            f" Details: {exc}" if strict else "",
        )
        logger.debug("GPU graph dependency check failed", exc_info=exc)
        return False


def _find_latest(glob: str) -> Path | None:
    """Return the latest matching file anywhere under `output/`.

    Searches recursively to support the refactored folder layout under output/.
    Files are expected to include sortable date tokens in their names.
    """
    cands = sorted(Path("output").rglob(glob))
    return cands[-1] if cands else None


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

    Example:
        File: daily_quotes_20200906_20250906.parquet
        Request: 2020-10-01 to 2025-08-01
        Result: Returns the file (covers the range)
    """
    import re

    req_start_dt = datetime.strptime(req_start, "%Y-%m-%d")
    req_end_dt = datetime.strptime(req_end, "%Y-%m-%d")

    # Find all matching files
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
        import time

        file_age_seconds = time.time() - file_path.stat().st_mtime
        file_age_days = file_age_seconds / (24 * 3600)
        is_valid = file_age_days <= max_age_days

        if is_valid:
            logger.info(
                f"✅ Cache valid: {file_path.name} (age: {file_age_days:.1f} days, limit: {max_age_days} days)"
            )
        else:
            logger.info(
                f"⏰ Cache stale: {file_path.name} (age: {file_age_days:.1f} days, limit: {max_age_days} days)"
            )

        return is_valid
    except Exception as e:
        logger.warning(f"Failed to check cache validity for {file_path}: {e}")
        return False


def _validate_enriched_dataset(parquet_path: Path, metadata_path: Path) -> None:
    """Ensure that the enriched dataset keeps required feature families."""

    try:
        schema = pl.scan_parquet(parquet_path).schema
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to inspect dataset schema: {exc}") from exc

    columns = set(schema.keys())

    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to read metadata {metadata_path}: {exc}") from exc

    feature_count = int(metadata.get("features", {}).get("count", 0))
    try:
        min_features = int(os.getenv("MIN_ENRICHED_FEATURES", "300"))
    except Exception:
        min_features = 300

    required_groups = {
        "flow": {"flow_foreign_net_z", "flow_activity_ratio", "flow_smart_idx"},
        "margin": {"margin_long_tot", "margin_short_tot", "margin_credit_ratio"},
        "sector": {"ret_1d_vs_sec", "ret_1d_in_sec_z", "volume_in_sec_z"},
    }

    missing_groups = [
        group
        for group, candidates in required_groups.items()
        if not any(col in columns for col in candidates)
    ]

    issues: list[str] = []
    if feature_count < min_features:
        issues.append(f"feature_count={feature_count} < required {min_features}")
    if missing_groups:
        issues.append("missing feature groups: " + ", ".join(sorted(missing_groups)))

    if issues:
        raise RuntimeError(
            "Dataset enrichment incomplete: "
            + "; ".join(issues)
            + ". Re-run with --force-refresh or verify upstream fetchers."
        )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for building the full enriched dataset."""
    parser = argparse.ArgumentParser(
        description="Build fully enriched 5y dataset in one command"
    )
    parser.add_argument(
        "--jquants", action="store_true", help="Use JQuants API (requires .env)"
    )
    # Trading calendar API usage (enable to enumerate business days accurately)
    parser.add_argument(
        "--use-calendar-api",
        action="store_true",
        default=True,
        help="Use Trading Calendar API to enumerate business days (default: enabled)",
    )
    parser.add_argument(
        "--start-date", type=str, default=None, help="YYYY-MM-DD (default: today-5y)"
    )
    parser.add_argument(
        "--end-date", type=str, default=None, help="YYYY-MM-DD (default: today)"
    )
    # GPU-ETL (デフォルトで有効、--no-gpu-etlで無効化可能)
    parser.add_argument(
        "--gpu-etl",
        action="store_true",
        default=True,  # デフォルトで有効
        help="Enable GPU ETL path (cuDF/RAPIDS). Enabled by default. Use --no-gpu-etl to disable.",
    )
    parser.add_argument(
        "--no-gpu-etl",
        dest="gpu_etl",
        action="store_false",
        help="Disable GPU ETL and use CPU only",
    )
    parser.add_argument(
        "--topix-parquet",
        type=Path,
        default=None,
        help="Optional TOPIX parquet for offline enrichment",
    )
    # Indices OHLC integration (spreads/breadth)
    parser.add_argument(
        "--enable-indices",
        action="store_true",
        default=True,  # Default enabled
        help="Enable indices OHLC features (spreads, breadth) (default: enabled)",
    )
    parser.add_argument(
        "--indices-parquet",
        type=Path,
        default=None,
        help="Optional indices OHLC parquet (Date, Code, Open, High, Low, Close)",
    )
    parser.add_argument(
        "--indices-codes",
        type=str,
        default=None,
        help="Comma-separated index codes to fetch via API (e.g., 0000,0040,0500,0501,0502,0075,8100,8200,0028,002D,8501,8502,8503)",
    )
    parser.add_argument(
        "--disable-halt-mask",
        action="store_true",
        help="Disable special halt-day masking (2020-10-01) for range-derived features",
    )
    parser.add_argument(
        "--statements-parquet",
        type=Path,
        default=None,
        help="Optional statements parquet for offline enrichment",
    )
    parser.add_argument(
        "--listed-info-parquet",
        type=Path,
        default=None,
        help="Optional listed_info parquet (for sector/market enrichment)",
    )
    parser.add_argument(
        "--weekly-margin-parquet",
        type=Path,
        default=None,
        help="Path to weekly_margin_interest parquet (Code/Date/PublishedDate + volumes)",
    )
    parser.add_argument(
        "--margin-weekly-lag",
        type=int,
        default=3,
        help="Publish lag (business days) if PublishedDate is missing (default=3)",
    )
    parser.add_argument(
        "--adv-window-days",
        type=int,
        default=20,
        help="ADV window (days) for scaling margin stocks (default=20)",
    )
    parser.add_argument(
        "--daily-margin-parquet",
        type=Path,
        default=None,
        help="Path to daily_margin_interest parquet (Code/Date/PublishedDate + credit data)",
    )
    parser.add_argument(
        "--enable-daily-margin",
        action="store_true",
        default=True,  # Default enabled (395 theoretical max; ~307 active with futures disabled)
        help="Enable daily margin interest features (default: enabled)",
    )
    # Futures options
    parser.add_argument(
        "--futures-parquet",
        type=Path,
        default=None,
        help="Optional derivatives/futures daily parquet for enrichment",
    )
    parser.add_argument(
        "--futures-categories",
        type=str,
        default="TOPIXF,NK225F,JN400F,REITF",
        help="Comma-separated futures categories to include",
    )
    parser.add_argument(
        "--disable-futures",
        action="store_true",
        help="Disable futures features (default: enabled)",
    )
    parser.add_argument(
        "--futures-continuous",
        action="store_true",
        help="Enable continuous futures series (fut_whole_ret_cont_*) (default: off)",
    )
    # Advanced volatility
    parser.add_argument(
        "--enable-advanced-vol",
        action="store_true",
        default=True,  # Default enabled (395 theoretical max; ~307 active with futures disabled)
        help="Enable Yang–Zhang volatility and VoV features (default: enabled)",
    )
    parser.add_argument(
        "--adv-vol-windows",
        type=str,
        default="20,60",
        help="Comma-separated windows for advanced volatility (e.g., 20,60)",
    )
    # Optional spot index parquets for basis mapping
    parser.add_argument(
        "--nk225-parquet",
        type=Path,
        default=None,
        help="Optional Nikkei225 spot parquet for basis mapping (Date, Close)",
    )
    parser.add_argument(
        "--reit-parquet",
        type=Path,
        default=None,
        help="Optional REIT index spot parquet for basis mapping (Date, Close)",
    )
    parser.add_argument(
        "--jpx400-parquet",
        type=Path,
        default=None,
        help="Optional JPX400 spot parquet for basis mapping (Date, Close)",
    )
    # Advanced equity features (interactions, cross-section, calendar)
    parser.add_argument(
        "--enable-advanced-features",
        action="store_true",
        default=True,  # Default enabled (395 theoretical max; ~307 active with futures disabled)
        help="Enable advanced T+0 features (RSI×Vol, momentum×volume, MACD slope, cross-sectional ranks, calendar) (default: enabled)",
    )
    # Graph-structured features (correlation graph)
    parser.add_argument(
        "--enable-graph-features",
        action="store_true",
        default=True,  # Default enabled (395 theoretical max; ~307 active with futures disabled)
        help="Enable graph-structured features (degree, peer corr mean, peer count) (default: enabled)",
    )
    parser.add_argument(
        "--graph-window",
        type=int,
        default=60,
        help="Correlation window (days) for graph features (default: 60)",
    )
    parser.add_argument(
        "--graph-threshold",
        type=float,
        default=0.3,
        help="Absolute correlation threshold for edges (default: 0.3)",
    )
    parser.add_argument(
        "--graph-max-k",
        type=int,
        default=4,
        help="Max edges per node (default: 4)",
    )
    parser.add_argument(
        "--graph-cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for graph artifacts (e.g., output/graph_cache)",
    )
    # Nikkei225 index option features (optional)
    parser.add_argument(
        "--enable-nk225-option-features",
        action="store_true",
        default=True,  # Default enabled
        help="Enable Nikkei225 index option features build and save (default: enabled)",
    )
    parser.add_argument(
        "--index-option-parquet",
        type=Path,
        default=None,
        help="Path to pre-saved /option/index_option raw parquet (optional)",
    )
    parser.add_argument(
        "--enable-sector-cs",
        action="store_true",
        default=True,  # Default enabled (395 theoretical max; ~307 active with futures disabled)
        help="Enable sector cross-sectional features (sector-relative deviations, ranks, z-scores) (default: enabled)",
    )
    parser.add_argument(
        "--sector-cs-cols",
        type=str,
        default=None,
        help="Comma-separated extra columns to compute sector-relative stats for (e.g., rsi_14,returns_10d)",
    )
    parser.add_argument(
        "--attach-nk225-option-market",
        action="store_true",
        help="Attach Nikkei225 option market aggregates (CMAT IV, term slope, flows) to equity panel (default: off)",
    )
    # Optional YAML config to override defaults for sector CS / graph / options
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML file to set sector CS include_cols and graph params (CLI overrides config)",
    )
    # Sector feature toggles
    parser.add_argument(
        "--sector-onehot33",
        action="store_true",
        help="Enable 33-sector one-hot encodings (default: off)",
    )
    parser.add_argument(
        "--sector-series-mcap",
        type=str,
        choices=["auto", "always", "never"],
        default="auto",
        help="Compute market-cap weighted sector series (auto when shares_outstanding present)",
    )
    parser.add_argument(
        "--sector-te-targets",
        type=str,
        default="target_5d",
        help="Comma-separated target cols for sector TE (e.g., target_5d,target_1d)",
    )
    parser.add_argument(
        "--sector-series-levels",
        type=str,
        default="33",
        help="Comma-separated sector levels for series (choices: 33,17)",
    )
    parser.add_argument(
        "--sector-te-levels",
        type=str,
        default="33",
        help="Comma-separated sector levels for TE (choices: 33,17)",
    )
    # Futures integration arguments (already defined above)
    # Short selling integration arguments
    parser.add_argument(
        "--enable-short-selling",
        action="store_true",
        default=True,  # Default enabled
        help="Enable short selling data integration (default: enabled)",
    )
    parser.add_argument(
        "--short-selling-parquet",
        type=Path,
        default=None,
        help="Path to pre-saved short selling parquet file",
    )
    parser.add_argument(
        "--short-positions-parquet",
        type=Path,
        default=None,
        help="Path to pre-saved short selling positions parquet file",
    )
    # Earnings events integration arguments
    parser.add_argument(
        "--enable-earnings-events",
        action="store_true",
        default=True,  # Default enabled
        help="Enable earnings announcement events data integration (default: enabled)",
    )
    parser.add_argument(
        "--earnings-announcements-parquet",
        type=Path,
        default=None,
        help="Path to pre-saved earnings announcements parquet file",
    )
    parser.add_argument(
        "--enable-pead-features",
        action="store_true",
        default=True,
        help="Enable Post-Earnings Announcement Drift (PEAD) features (default: on)",
    )
    # Sector-wise short selling integration arguments
    parser.add_argument(
        "--enable-sector-short-selling",
        action="store_true",
        default=True,  # Default enabled
        help="Enable sector-wise short selling features (default: enabled)",
    )
    parser.add_argument(
        "--sector-short-selling-parquet",
        type=Path,
        default=None,
        help="Path to pre-saved sector short selling parquet file",
    )
    parser.add_argument(
        "--disable-sector-short-z-scores",
        action="store_true",
        help="Disable Z-score features for sector short selling (default: enabled)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned steps and exit",
    )
    parser.add_argument(
        "--check-env-only",
        action="store_true",
        help="Run preflight checks (J-Quants credentials, GPU graph dependencies) and exit",
    )
    parser.add_argument(
        "--require-gpu-graph",
        action="store_true",
        help="Abort if cuGraph/CuPy GPU graph path is unavailable instead of falling back to CPU",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh from API even when local cached data exists (default: use cache when available)",
    )
    parser.add_argument(
        "--max-cache-age-days",
        type=int,
        default=7,
        help="Maximum age of cached data in days before forcing refresh (default: 7)",
    )
    return parser.parse_args()


## Saving is delegated to src/pipeline/full_dataset.save_with_symlinks via enrich_and_save


async def main() -> int:
    """Entry point to orchestrate base pipeline and enrichment steps.

    Follows the documented 3-step flow and persists artifacts under `output/`.
    """
    args = _parse_args()

    creds_ok = True
    if args.jquants:
        creds_ok = _check_jquants_credentials(strict=args.check_env_only)

    # Display J-Quants plan tier information
    plan_tier = _get_jquants_plan_tier()
    futures_plan_available = _is_futures_available()
    logger.info("=" * 80)
    logger.info(f"📋 J-Quants Plan Tier: {plan_tier.upper()}")
    if futures_plan_available:
        logger.info("✅ Futures API enabled (Premium plan)")
        logger.info("   → Full feature set available (~395 features)")
    else:
        logger.info("⚠️  Futures API disabled (Standard plan)")
        logger.info(
            "   → ~303-307 features available (88-92 futures features excluded)"
        )
        logger.info("   → To enable: Set JQUANTS_PLAN_TIER=premium in .env")
    logger.info("=" * 80)

    # GPU graph check: only strict if --require-gpu-graph is explicitly set
    gpu_ready = _check_gpu_graph_support(strict=args.require_gpu_graph)

    if args.check_env_only:
        # Relaxed mode: only fail if credentials are missing
        # GPU fallback is allowed unless --require-gpu-graph is set
        if args.require_gpu_graph:
            # Strict mode: both credentials and GPU required
            if creds_ok and gpu_ready:
                logger.info("✅ Preflight checks passed (strict mode)")
                return 0
            logger.error("❌ Preflight checks failed (strict mode)")
            return 1
        else:
            # Relaxed mode: only credentials required
            if creds_ok:
                if gpu_ready:
                    logger.info("✅ Preflight checks passed (GPU accelerated)")
                else:
                    logger.info("✅ Preflight checks passed (CPU fallback)")
                return 0
            logger.error("❌ Preflight checks failed (credentials missing)")
            return 1

    if args.jquants and not creds_ok:
        return 1
    if args.require_gpu_graph and not gpu_ready:
        return 1

    # GPU-ETLのデフォルト設定：環境変数またはコマンドライン引数から判定
    # 優先順位: コマンドライン引数 > 環境変数 > デフォルト(True)
    gpu_etl_enabled = getattr(args, "gpu_etl", os.getenv("USE_GPU_ETL", "1") == "1")

    # Propagate GPU-ETL flag to env for downstream modules
    if gpu_etl_enabled:
        os.environ["USE_GPU_ETL"] = "1"
        logger.info("GPU-ETL: enabled (will use RAPIDS/cuDF if available)")
        # Best-effort RMM init with large pool if provided
        try:
            from src.utils.gpu_etl import init_rmm  # type: ignore

            pool = os.getenv("RMM_POOL_SIZE", "0")
            ok = init_rmm(pool)
            if ok:
                allocator = os.getenv("RMM_ALLOCATOR", "pool")
                logger.info(
                    f"RMM initialized (allocator={allocator}, pool_size={pool})"
                )
            else:
                logger.info("RMM initialization skipped or failed (continuing)")
        except Exception:
            pass
    else:
        os.environ.pop("USE_GPU_ETL", None)

    # Load YAML config if provided (CLI takes precedence)
    if (
        getattr(args, "config", None) is not None
        and args.config
        and args.config.exists()
    ):
        try:
            import yaml  # type: ignore

            with open(args.config, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            # Sector CS
            sec = cfg.get("sector_cs") or {}
            if isinstance(sec, dict):
                if not args.enable_sector_cs and isinstance(sec.get("enable"), bool):
                    args.enable_sector_cs = bool(sec.get("enable"))
                if not args.sector_cs_cols and isinstance(
                    sec.get("include_cols"), (list, tuple)
                ):
                    args.sector_cs_cols = ",".join(
                        str(s) for s in sec.get("include_cols") if s
                    )
            # Graph
            g = cfg.get("graph") or {}
            if isinstance(g, dict):
                if not args.enable_graph_features and isinstance(g.get("enable"), bool):
                    args.enable_graph_features = bool(g.get("enable"))
                if getattr(args, "graph_window", None) in (None, 60) and isinstance(
                    g.get("window"), int
                ):
                    args.graph_window = int(g.get("window"))
                if getattr(args, "graph_threshold", None) in (None, 0.3) and isinstance(
                    g.get("threshold"), (float, int)
                ):
                    args.graph_threshold = float(g.get("threshold"))
                if getattr(args, "graph_max_k", None) in (None, 4) and isinstance(
                    g.get("max_k"), int
                ):
                    args.graph_max_k = int(g.get("max_k"))
                if getattr(args, "graph_cache_dir", None) in (None,) and g.get(
                    "cache_dir"
                ):
                    args.graph_cache_dir = Path(str(g.get("cache_dir")))
            # Option market attach
            om = cfg.get("option_market") or {}
            if (
                isinstance(om, dict)
                and not args.attach_nk225_option_market
                and isinstance(om.get("attach"), bool)
            ):
                args.attach_nk225_option_market = bool(om.get("attach"))
            # Indices (market/sector) features
            ind = cfg.get("indices") or {}
            if isinstance(ind, dict):
                if not args.enable_indices and isinstance(ind.get("enable"), bool):
                    args.enable_indices = bool(ind.get("enable"))
                # codes can be list or comma-separated string
                if not getattr(args, "indices_codes", None) and ind.get("codes"):
                    codes = ind.get("codes")
                    if isinstance(codes, (list, tuple)):
                        args.indices_codes = ",".join(
                            str(c).strip() for c in codes if str(c).strip()
                        )
                    elif isinstance(codes, str):
                        args.indices_codes = codes
                # parquet path
                if not getattr(args, "indices_parquet", None) and ind.get("parquet"):
                    p = Path(str(ind.get("parquet")))
                    args.indices_parquet = p
                # special day mask
                if not getattr(args, "disable_halt_mask", False) and isinstance(
                    ind.get("disable_halt_mask"), bool
                ):
                    args.disable_halt_mask = bool(ind.get("disable_halt_mask"))
        except Exception as e:
            logger.warning(f"Failed to load config YAML {args.config}: {e}")

    # Dry-run: print planned steps and exit successfully
    if getattr(args, "dry_run", False):
        print("=" * 60)
        print("[DRY-RUN] Build fully enriched 5y dataset")
        print("Steps:")
        print(" 0) Prepare trade-spec (JQuants optional or local fallback)")
        print(" 0.5) Fetch futures data (if not --disable-futures)")
        print(" 0.6) Fetch short selling data (if --enable-short-selling)")
        print(
            " 0.7) Fetch sector short selling data (if --enable-sector-short-selling)"
        )
        print(" 0.8) Fetch Nikkei225 index options (if --enable-nk225-option-features)")
        print(" 1) Run base optimized pipeline (prices + TA + statements)")
        print(
            " 2) Enrich with TOPIX, flow, sector, futures, short selling, sector short selling"
        )
        print(" 2.5) Build and save Nikkei225 option features (separate parquet)")
        print(" 3) Save ml_dataset_latest_full.parquet (+ metadata)")
        print("=" * 60)
        return 0

    # Determine date range (default last ~5 years)
    end_dt = (
        datetime.now()
        if not args.end_date
        else datetime.strptime(args.end_date, "%Y-%m-%d")
    )
    start_dt = (
        end_dt - timedelta(days=DEFAULT_LOOKBACK_DAYS)
        if not args.start_date
        else datetime.strptime(args.start_date, "%Y-%m-%d")
    )
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")
    requested_start_date = start_date

    # Separate contract validation from data fetch range
    # Contract validation: Use JQUANTS_SUBSCRIPTION_START for API 400 prevention
    # Data fetch range: Use user-requested dates (no forced historical data)

    # Support rolling contracts (e.g., "last 10 years")
    # Priority: JQUANTS_SUBSCRIPTION_START > dynamic calculation from JQUANTS_CONTRACT_YEARS
    subscription_start_str = os.getenv("JQUANTS_SUBSCRIPTION_START")

    if subscription_start_str:
        # Explicit start date provided
        try:
            subscription_start_dt = datetime.strptime(
                subscription_start_str, "%Y-%m-%d"
            )
        except ValueError:
            logger.warning(
                "Invalid JQUANTS_SUBSCRIPTION_START=%s; falling back to rolling contract",
                subscription_start_str,
            )
            subscription_start_str = None

    if not subscription_start_str:
        # Dynamic rolling contract (e.g., last 10 years from today)
        contract_years = int(os.getenv("JQUANTS_CONTRACT_YEARS", "10"))
        subscription_start_dt = datetime.now() - timedelta(
            days=365 * contract_years + 2
        )  # +2 for leap years
        logger.info(
            "Using rolling %d-year contract: subscription starts from %s (dynamic)",
            contract_years,
            subscription_start_dt.strftime("%Y-%m-%d"),
        )

    # Validate user-requested range is within subscription coverage
    # If out of range, stop immediately with clear error message
    if args.jquants:
        if start_dt < subscription_start_dt:
            logger.error(
                "❌ Requested start date %s is before subscription coverage %s",
                start_date,
                subscription_start_dt.strftime("%Y-%m-%d"),
            )
            logger.error("   Please adjust --start-date or check your J-Quants plan")
            logger.error(
                "   Your subscription covers: %s ~ now",
                subscription_start_dt.strftime("%Y-%m-%d"),
            )
            return 1

        if end_dt < subscription_start_dt:
            logger.error(
                "❌ Requested end date %s is before subscription coverage %s",
                end_date,
                subscription_start_dt.strftime("%Y-%m-%d"),
            )
            logger.error(
                "   Your subscription covers: %s ~ now",
                subscription_start_dt.strftime("%Y-%m-%d"),
            )
            return 1

    # Calculate lookback start for technical indicators (420 days)
    # This is a system requirement, not a user request
    flow_start_dt = start_dt - timedelta(days=FLOW_SUPPORT_LOOKBACK_DAYS)

    # Clamp lookback to subscription start if needed (this is acceptable)
    # User didn't request this far back, it's just for indicator calculation
    if flow_start_dt < subscription_start_dt:
        logger.warning(
            "⚠️  Lookback start %s is before subscription coverage %s; clamping to %s",
            flow_start_dt.strftime("%Y-%m-%d"),
            subscription_start_dt.strftime("%Y-%m-%d"),
            subscription_start_dt.strftime("%Y-%m-%d"),
        )
        logger.warning(
            "   (This is normal for early dates - technical indicators may have less history)"
        )
        flow_start_dt = subscription_start_dt

    flow_start_date = flow_start_dt.strftime("%Y-%m-%d")

    # If we already know the earliest business day for the requested window,
    # align the flow start to it to avoid subscription-boundary 400 errors.
    try:
        _bd_env = os.getenv("__BDAYS_HINT__")
        if _bd_env:
            # Expect comma-separated YYYY-MM-DD list if provided by upstream steps
            _b = [b for b in _bd_env.split(",") if b]
            if _b:
                first_b = min(_b)
                if first_b > flow_start_date:
                    logger.info(
                        "Flow start pre-aligned to first business day %s (was %s)",
                        first_b,
                        flow_start_date,
                    )
                    flow_start_date = first_b
    except Exception:
        pass

    logger.info("=== STEP 0: Prepare trade-spec for flow features ===")
    trades_spec_path: Path | None = None
    listed_info_path: Path | None = args.listed_info_parquet
    futures_path: Path | None = None
    short_selling_path: Path | None = None
    short_positions_path: Path | None = None

    # Respect user-provided futures parquet regardless of plan tier
    if args.futures_parquet is not None:
        if args.futures_parquet.exists():
            futures_path = args.futures_parquet
            logger.info(f"Using provided futures parquet: {futures_path}")
        else:
            logger.warning(
                "Specified futures parquet not found: %s", args.futures_parquet
            )

    # Smart reuse: Check for cached data first (unless --force-refresh)
    use_cached = False
    if args.jquants and not args.force_refresh:
        logger.info("🔍 Checking for cached data (use --force-refresh to skip cache)")
        max_age = args.max_cache_age_days

        # Check for cached data files
        cached_trades = _find_latest("trades_spec_history_*.parquet")
        cached_listed_info = _find_latest("listed_info_history_*.parquet")
        cached_weekly_margin = _find_latest("weekly_margin_interest_*.parquet")
        cached_daily_margin = _find_latest("daily_margin_interest_*.parquet")
        cached_short_selling = (
            _find_latest("short_selling_*.parquet")
            if args.enable_short_selling
            else None
        )
        cached_sector_short = (
            _find_latest("sector_short_selling_*.parquet")
            if args.enable_sector_short_selling
            else None
        )

        # Verify cache validity
        trades_valid = _is_cache_valid(cached_trades, max_age)
        listed_valid = _is_cache_valid(cached_listed_info, max_age)
        weekly_margin_valid = _is_cache_valid(cached_weekly_margin, max_age)
        daily_margin_valid = _is_cache_valid(cached_daily_margin, max_age)
        short_selling_valid = (
            _is_cache_valid(cached_short_selling, max_age)
            if args.enable_short_selling
            else True
        )
        sector_short_valid = (
            _is_cache_valid(cached_sector_short, max_age)
            if args.enable_sector_short_selling
            else True
        )

        # Use cache if all required files are valid
        if trades_valid and listed_valid and short_selling_valid and sector_short_valid:
            logger.info(
                "✅ All cached data is valid, using cached files (skip API fetch)"
            )
            use_cached = True
            trades_spec_path = cached_trades
            listed_info_path = cached_listed_info
            if args.enable_short_selling and cached_short_selling:
                short_selling_path = cached_short_selling
            if args.enable_sector_short_selling and cached_sector_short:
                # Note: sector_short_path will be set from cached_sector_short
                pass  # Sector short uses different variable names in offline fallback
        else:
            logger.info("⚠️  Some cached data is missing or stale, will fetch from API")

    if args.jquants and not use_cached:
        # Fetch trade-spec and save to Parquet
        email = os.getenv("JQUANTS_AUTH_EMAIL", "")
        password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
        if not email or not password:
            logger.error("JQuants credentials not found in environment/.env")
            return 1

        fetcher = JQuantsAsyncFetcher(email, password)
        tcp_limit = int(os.getenv("JQUANTS_TCP_LIMIT", "30"))
        tcp_limit_per_host = int(os.getenv("JQUANTS_TCP_LIMIT_PER_HOST", "15"))
        sock_connect_timeout = float(os.getenv("JQUANTS_SOCK_CONNECT_TIMEOUT", "10"))
        sock_read_timeout = float(os.getenv("JQUANTS_SOCK_READ_TIMEOUT", "60"))

        trades_df: pl.DataFrame | None = pl.DataFrame()
        wmi_df: pl.DataFrame | None = pl.DataFrame()
        dmi_df: pl.DataFrame | None = pl.DataFrame()
        info_df: pl.DataFrame | None = pl.DataFrame()

        connector = aiohttp.TCPConnector(
            limit=tcp_limit,
            limit_per_host=tcp_limit_per_host,
            ttl_dns_cache=300,
        )
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=sock_connect_timeout,
            sock_read=sock_read_timeout,
        )

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            await fetcher.authenticate(session)

            # Optional: use Trading Calendar API to enumerate business days
            business_days: list[str] | None = None
            if args.use_calendar_api:
                try:
                    cal_fetcher = TradingCalendarFetcher(api_client=fetcher)
                    cal = await cal_fetcher.get_trading_calendar(
                        start_date, end_date, session
                    )
                    all_bdays = cal.get("business_days", [])
                    business_days = [d for d in all_bdays if d >= start_date]
                    logger.info(
                        "Trading calendar fetched: %s business days", len(all_bdays)
                    )
                except Exception as e:
                    logger.warning(
                        f"Trading calendar fetch failed; fallback to weekday-only: {e}"
                    )
            # Fallback to naive weekday-only calendar when API is disabled or failed
            if not business_days:
                _bd = []
                _cur = datetime.strptime(start_date, "%Y-%m-%d")
                _end = datetime.strptime(end_date, "%Y-%m-%d")
                while _cur <= _end:
                    if _cur.weekday() < 5:  # Mon-Fri
                        _bd.append(_cur.strftime("%Y-%m-%d"))
                    _cur += timedelta(days=1)
                business_days = _bd
                logger.info("Using weekday-only calendar: %d days", len(business_days))

            # Align flow start to first available business day
            if business_days:
                _first = next(
                    (d for d in business_days if d >= flow_start_date), business_days[0]
                )
                if _first > flow_start_date:
                    logger.info(
                        "Flow start adjusted to first business day: %s -> %s",
                        flow_start_date,
                        _first,
                    )
                    flow_start_date = _first

            fetch_coroutines: list[tuple[str, str, Awaitable[pl.DataFrame | None]]] = []

            logger.info(
                "Fetching trade-spec from %s to %s (lookback %s days)",
                flow_start_date,
                end_date,
                FLOW_SUPPORT_LOOKBACK_DAYS,
            )
            fetch_coroutines.append(
                (
                    "trades",
                    "trade-spec",
                    fetcher.get_trades_spec(session, flow_start_date, end_date),
                )
            )

            logger.info("Fetching weekly margin interest for margin features")
            fetch_coroutines.append(
                (
                    "weekly_margin",
                    "weekly margin interest",
                    fetcher.get_weekly_margin_interest(session, start_date, end_date),
                )
            )

            logger.info("Fetching daily margin interest for daily credit features")
            fetch_coroutines.append(
                (
                    "daily_margin",
                    "daily margin interest",
                    fetcher.get_daily_margin_interest(
                        session, start_date, end_date, business_days=business_days
                    ),
                )
            )

            logger.info("Fetching listed_info for sector/market enrichment")
            fetch_coroutines.append(
                (
                    "listed_info",
                    "listed_info",
                    fetcher.get_listed_info(session),
                )
            )

            results = await asyncio.gather(
                *(coro for _, _, coro in fetch_coroutines), return_exceptions=True
            )

            for (key, label, _), result in zip(fetch_coroutines, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to fetch {label}: {result}")
                    continue
                if key == "trades":
                    trades_df = result
                elif key == "weekly_margin":
                    wmi_df = result
                elif key == "daily_margin":
                    dmi_df = result
                elif key == "listed_info":
                    info_df = result

            metrics_fn = getattr(fetcher, "throttle_metrics", None)
            if callable(metrics_fn):
                tmetrics = metrics_fn()
                logger.info(
                    "Throttle metrics → hits=%s, recoveries=%s, current_concurrency=%s",
                    tmetrics.get("hits"),
                    tmetrics.get("recoveries"),
                    tmetrics.get("current_concurrency"),
                )

        if wmi_df is not None and not wmi_df.is_empty() and "Code" in wmi_df.columns:
            wmi_df = wmi_df.with_columns([pl.col("Code").cast(pl.Utf8).alias("Code")])
            logger.info("Weekly margin: unified Code dtype to Utf8")
        else:
            wmi_df = pl.DataFrame()

        if dmi_df is not None and not dmi_df.is_empty() and "Code" in dmi_df.columns:
            dmi_df = dmi_df.with_columns([pl.col("Code").cast(pl.Utf8).alias("Code")])
            logger.info("Daily margin: unified Code dtype to Utf8")
        else:
            dmi_df = pl.DataFrame()

        if trades_df is None:
            trades_df = pl.DataFrame()
        if info_df is None:
            info_df = pl.DataFrame()

        if trades_df is None or trades_df.is_empty():
            logger.warning(
                "No trade-spec data fetched; will try local fallback for flow features"
            )
        else:
            from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs

            output_dir = Path("output/raw/flow")
            output_dir.mkdir(parents=True, exist_ok=True)
            trades_spec_path = (
                output_dir
                / f"trades_spec_history_{flow_start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
            )
            save_parquet_with_gcs(trades_df, trades_spec_path, auto_sync=False)
        # Save listed_info if fetched (even if trade-spec failed)
        if listed_info_path is None:
            # Name by end date for reproducibility
            listed_info_path = (
                Path("output/raw/jquants")
                / f"listed_info_history_{end_dt.strftime('%Y%m%d')}.parquet"
            )
        if info_df is not None and not info_df.is_empty():
            try:
                from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs

                save_parquet_with_gcs(info_df, listed_info_path, auto_sync=False)
            except Exception as e:
                logger.warning(f"Failed to save listed_info parquet: {e}")
        # Save weekly margin interest if fetched
        wmi_path: Path | None = None
        if wmi_df is not None and not wmi_df.is_empty():
            try:
                from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs

                outdir = Path("output/raw/margin")
                outdir.mkdir(parents=True, exist_ok=True)
                wmi_path = (
                    outdir
                    / f"weekly_margin_interest_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                )
                save_parquet_with_gcs(wmi_df, wmi_path, auto_sync=False)
            except Exception as e:
                logger.warning(f"Failed to save weekly margin parquet: {e}")
        # Save daily margin interest if fetched
        dmi_path: Path | None = None
        if dmi_df is not None and not dmi_df.is_empty():
            try:
                from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs

                outdir = Path("output/raw/margin")
                outdir.mkdir(parents=True, exist_ok=True)
                dmi_path = (
                    outdir
                    / f"daily_margin_interest_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                )
                save_parquet_with_gcs(dmi_df, dmi_path, auto_sync=False)
            except Exception as e:
                logger.warning(f"Failed to save daily margin parquet: {e}")

        # Fetch futures/short-selling within a fresh session (previous session closed)
        try:
            connector_aux = aiohttp.TCPConnector(
                limit=tcp_limit,
                limit_per_host=tcp_limit_per_host,
                ttl_dns_cache=300,
            )
            timeout_aux = aiohttp.ClientTimeout(
                total=None,
                sock_connect=sock_connect_timeout,
                sock_read=sock_read_timeout,
            )
            async with aiohttp.ClientSession(
                connector=connector_aux, timeout=timeout_aux
            ) as _session_aux:
                # Authenticate if needed (reuse token when available)
                try:
                    await fetcher.authenticate(_session_aux)
                    logger.info(
                        "✅ Authenticated successfully for short selling/futures fetch"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Authentication failed for short selling/futures fetch: {e}"
                    )
                    raise  # Re-raise to skip the rest of the block
                # Futures data (Premium plan only)
                # NOTE: /derivatives/futures API is only available on Premium plan
                # Automatically enabled when JQUANTS_PLAN_TIER=premium
                if futures_plan_available:
                    try:
                        logger.info(
                            "Fetching futures data for derivatives features (Premium plan)"
                        )
                        futures_df = await fetcher.get_futures_daily(
                            _session_aux, start_date, end_date
                        )
                        if futures_df is not None and not futures_df.is_empty():
                            outdir = Path("output/raw/futures")
                            outdir.mkdir(parents=True, exist_ok=True)
                            futures_path = (
                                outdir
                                / f"futures_daily_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                            )
                            futures_df.write_parquet(futures_path)
                            logger.info(f"Saved futures data: {futures_path}")
                        else:
                            logger.warning("No futures data retrieved from API")
                    except Exception as e:
                        logger.warning(f"Failed to fetch futures data: {e}")

                # Short selling data (optional)
                fetch_aux: list[tuple[str, str, Awaitable[pl.DataFrame | None]]] = []

                if args.enable_short_selling:
                    logger.info(
                        f"Fetching short selling ratio data from {start_date} to {end_date}"
                    )
                    if start_date and end_date:
                        fetch_aux.append(
                            (
                                "short_selling",
                                "short selling data",
                                fetcher.get_short_selling(
                                    _session_aux,
                                    start_date,
                                    end_date,
                                    business_days=business_days,
                                ),
                            )
                        )
                    else:
                        logger.warning(
                            f"Invalid date range for short selling: start={start_date}, end={end_date}"
                        )

                    logger.info(
                        f"Fetching short selling positions data from {start_date} to {end_date}"
                    )
                    if start_date and end_date:
                        fetch_aux.append(
                            (
                                "short_positions",
                                "short selling positions data",
                                fetcher.get_short_selling_positions(
                                    _session_aux,
                                    start_date,
                                    end_date,
                                    business_days=business_days,
                                ),
                            )
                        )
                    else:
                        logger.warning(
                            f"Invalid date range for short positions: start={start_date}, end={end_date}"
                        )

                if args.enable_sector_short_selling:
                    logger.info(
                        f"Fetching sector-wise short selling data from {start_date} to {end_date}"
                    )
                    if start_date and end_date:
                        fetch_aux.append(
                            (
                                "sector_short",
                                "sector short selling data",
                                fetcher.get_sector_short_selling(
                                    _session_aux,
                                    start_date,
                                    end_date,
                                    business_days=business_days,
                                ),
                            )
                        )
                    else:
                        logger.warning(
                            f"Invalid date range for sector short: start={start_date}, end={end_date}"
                        )

                if fetch_aux:
                    aux_results = await asyncio.gather(
                        *(coro for _, _, coro in fetch_aux), return_exceptions=True
                    )

                    for (key, label, _), result in zip(fetch_aux, aux_results):
                        if isinstance(result, Exception):
                            logger.warning(f"Failed to fetch {label}: {result}")
                            continue

                        if key == "short_selling":
                            if result is None or result.is_empty():
                                logger.warning(
                                    "No short selling data retrieved from API"
                                )
                            else:
                                from src.gogooku3.utils.gcs_storage import (
                                    save_parquet_with_gcs,
                                )

                                outdir = Path("output/raw/short_selling")
                                outdir.mkdir(parents=True, exist_ok=True)
                                short_selling_path = (
                                    outdir
                                    / f"short_selling_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                                )
                                save_parquet_with_gcs(
                                    result, short_selling_path, auto_sync=False
                                )
                                logger.info(
                                    f"✅ Saved short selling data: {short_selling_path} ({len(result)} records)"
                                )

                        elif key == "short_positions":
                            if result is None or result.is_empty():
                                logger.warning(
                                    "No short selling positions data retrieved from API"
                                )
                            else:
                                from src.gogooku3.utils.gcs_storage import (
                                    save_parquet_with_gcs,
                                )

                                outdir = Path("output/raw/short_selling")
                                outdir.mkdir(parents=True, exist_ok=True)
                                short_positions_path = (
                                    outdir
                                    / f"short_positions_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                                )
                                save_parquet_with_gcs(
                                    result, short_positions_path, auto_sync=False
                                )
                                logger.info(
                                    f"✅ Saved short positions data: {short_positions_path} ({len(result)} records)"
                                )

                        elif key == "sector_short":
                            if result is None or result.is_empty():
                                logger.warning(
                                    "No sector short selling data retrieved from API"
                                )
                            else:
                                from src.gogooku3.utils.gcs_storage import (
                                    save_parquet_with_gcs,
                                )

                                outdir = Path("output/raw/short_selling")
                                outdir.mkdir(parents=True, exist_ok=True)
                                sector_short_path = (
                                    outdir
                                    / f"sector_short_selling_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                                )
                                save_parquet_with_gcs(
                                    result, sector_short_path, auto_sync=False
                                )
                                logger.info(
                                    f"✅ Saved sector short selling data: {sector_short_path} ({len(result)} records)"
                                )
        except Exception as e:
            logger.warning(f"Aux session for futures/short features failed: {e}")
    else:
        # Offline fallback: look for a local trades_spec parquet
        trades_spec_path = _find_latest("trades_spec_history_*.parquet")
        if trades_spec_path:
            logger.info(f"Using local trade-spec parquet: {trades_spec_path}")
        else:
            logger.warning(
                "No local trades_spec parquet found; flow features may be skipped"
            )
        # Offline listed_info fallback
        if listed_info_path is None:
            listed_info_path = _find_latest("listed_info_history_*.parquet")
        if listed_info_path:
            logger.info(f"Using listed_info parquet: {listed_info_path}")
        else:
            logger.warning(
                "No listed_info parquet provided/found; sector enrichment will be skipped"
            )

        # Offline futures fallback (Premium plan only)
        # NOTE: Futures features are only available on Premium plan
        # Automatically enabled when JQUANTS_PLAN_TIER=premium
        futures_path: Path | None = None
        if futures_path is None:
            futures_path = _find_latest("futures_daily_*.parquet")
            if futures_path:
                logger.info(f"Using local futures parquet: {futures_path}")
            else:
                if futures_plan_available:
                    logger.warning(
                        "No futures parquet found; futures features will be skipped"
                    )

        # Offline short selling fallback
        if args.enable_short_selling:
            # Short selling ratio data
            if (
                args.short_selling_parquet is not None
                and args.short_selling_parquet.exists()
            ):
                short_selling_path = args.short_selling_parquet
                logger.info(
                    f"Using provided short selling parquet: {short_selling_path}"
                )
            else:
                short_selling_path = _find_latest("short_selling_*.parquet")
                if short_selling_path:
                    logger.info(
                        f"Using local short selling parquet: {short_selling_path}"
                    )
                else:
                    logger.warning(
                        "No short selling parquet found; short selling features will be skipped"
                    )

            # Short selling positions data
            if (
                args.short_positions_parquet is not None
                and args.short_positions_parquet.exists()
            ):
                short_positions_path = args.short_positions_parquet
                logger.info(
                    f"Using provided short positions parquet: {short_positions_path}"
                )
            else:
                short_positions_path = _find_latest("short_positions_*.parquet")
                if short_positions_path:
                    logger.info(
                        f"Using local short positions parquet: {short_positions_path}"
                    )
                else:
                    logger.warning(
                        "No short positions parquet found; positions features will be skipped"
                    )

    logger.info(
        "=== STEP 1: Run base optimized pipeline (prices + TA + statements) ==="
    )
    pipeline = JQuantsPipelineV4Optimized()
    df_base, metadata = await pipeline.run(
        use_jquants=args.jquants, start_date=start_date, end_date=end_date
    )
    if df_base is None or metadata is None:
        logger.error("Base pipeline failed")
        return 1
    # Unify Code dtype to Utf8 early to avoid join type mismatches downstream
    try:
        if "Code" in df_base.columns:
            df_base = df_base.with_columns([pl.col("Code").cast(pl.Utf8).alias("Code")])
            logger.info("Normalized base frame Code dtype to Utf8")
    except Exception as e:
        logger.warning(f"Failed to normalize Code dtype in base frame: {e}")

    # Use freshly built base frame; avoid overriding with older ml_dataset_latest.parquet
    # Save datasets under refactored folder
    output_dir = Path("output/datasets")

    logger.info(
        "=== STEP 2: Enrich with TOPIX + statements + flow (trade-spec) + margin weekly ==="
    )
    # Resolve weekly margin parquet (existing style: auto-discover if not provided; skip gracefully if missing)
    margin_weekly_parquet: Path | None = None
    if args.weekly_margin_parquet is not None and args.weekly_margin_parquet.exists():
        margin_weekly_parquet = args.weekly_margin_parquet
    else:
        # prefer the one we just saved (if any)
        if "wmi_path" in locals() and wmi_path and wmi_path.exists():
            margin_weekly_parquet = wmi_path
        else:
            margin_weekly_parquet = _find_latest("weekly_margin_interest_*.parquet")

    # Resolve daily margin parquet (similar to weekly margin handling)
    daily_margin_parquet: Path | None = None
    if args.daily_margin_parquet is not None and args.daily_margin_parquet.exists():
        daily_margin_parquet = args.daily_margin_parquet
    else:
        # prefer the one we just saved (if any)
        if "dmi_path" in locals() and dmi_path and dmi_path.exists():
            daily_margin_parquet = dmi_path
        else:
            daily_margin_parquet = _find_latest("daily_margin_interest_*.parquet")

    # Resolve sector short selling parquet (similar to other data sources)
    sector_short_selling_parquet: Path | None = None
    if (
        args.sector_short_selling_parquet is not None
        and args.sector_short_selling_parquet.exists()
    ):
        sector_short_selling_parquet = args.sector_short_selling_parquet
    else:
        # prefer the one we just saved (if any)
        if (
            "sector_short_path" in locals()
            and sector_short_path
            and sector_short_path.exists()
        ):
            sector_short_selling_parquet = sector_short_path
        else:
            sector_short_selling_parquet = _find_latest(
                "sector_short_selling_*.parquet"
            )

    te_targets = [
        s.strip() for s in (args.sector_te_targets or "").split(",") if s.strip()
    ]
    series_levels = [
        s.strip()
        for s in (getattr(args, "sector_series_levels", "33") or "").split(",")
        if s.strip()
    ]
    te_levels = [
        s.strip()
        for s in (getattr(args, "sector_te_levels", "33") or "").split(",")
        if s.strip()
    ]
    futures_categories_list = [
        s.strip()
        for s in (getattr(args, "futures_categories", "") or "").split(",")
        if s.strip()
    ] or ["TOPIXF", "NK225F", "JN400F", "REITF"]

    futures_enabled: bool
    if args.disable_futures:
        futures_enabled = False
        logger.info("ℹ️ Futures features disabled via --disable-futures")
    elif futures_plan_available:
        futures_enabled = True
    elif futures_path is not None:
        futures_enabled = True
        logger.info(
            "🧩 Futures features enabled via offline parquet (plan tier: %s)",
            plan_tier,
        )
    else:
        futures_enabled = False
        logger.info(
            "ℹ️ Futures features unavailable (plan tier: %s, no offline parquet detected)",
            plan_tier,
        )

    if futures_enabled and not futures_plan_available and futures_path is None:
        logger.warning(
            "Futures features requested but no parquet available; disabling to continue."
        )
        futures_enabled = False

    pq_path, meta_path = await enrich_and_save(
        df_base,
        output_dir=output_dir,
        jquants=args.jquants,
        start_date=start_date,
        end_date=end_date,
        business_days=locals().get("business_days", None),
        trades_spec_path=trades_spec_path,
        topix_parquet=args.topix_parquet,
        enable_indices=args.enable_indices,
        indices_parquet=args.indices_parquet,
        indices_codes=[
            s.strip()
            for s in (getattr(args, "indices_codes", None) or "").split(",")
            if s.strip()
        ]
        or None,
        statements_parquet=args.statements_parquet,
        listed_info_parquet=listed_info_path,
        # Futures features (Premium plan only - auto-enabled via JQUANTS_PLAN_TIER)
        enable_futures=futures_enabled,
        futures_parquet=futures_path if futures_enabled else None,
        futures_categories=futures_categories_list if futures_enabled else [],
        futures_continuous=(
            getattr(args, "futures_continuous", False) if futures_enabled else False
        ),
        nk225_parquet=getattr(args, "nk225_parquet", None),
        reit_parquet=getattr(args, "reit_parquet", None),
        jpx400_parquet=getattr(args, "jpx400_parquet", None),
        enable_advanced_vol=args.enable_advanced_vol,
        adv_vol_windows=[
            int(s.strip()) for s in (args.adv_vol_windows or "").split(",") if s.strip()
        ],
        enable_margin_weekly=bool(
            margin_weekly_parquet is not None and margin_weekly_parquet.exists()
        ),
        margin_weekly_parquet=margin_weekly_parquet,
        margin_weekly_lag=getattr(args, "margin_weekly_lag", 3),
        adv_window_days=getattr(args, "adv_window_days", 20),
        # Default behavior: auto-enable daily margin when JQuants is used or a parquet exists
        enable_daily_margin=(
            args.jquants
            or args.enable_daily_margin
            or bool(daily_margin_parquet is not None and daily_margin_parquet.exists())
        ),
        daily_margin_parquet=daily_margin_parquet,
        # Short selling parameters
        enable_short_selling=args.enable_short_selling,
        short_selling_parquet=short_selling_path,
        short_positions_parquet=short_positions_path,
        # Earnings events parameters
        enable_earnings_events=args.enable_earnings_events,
        earnings_announcements_parquet=args.earnings_announcements_parquet,
        enable_pead_features=args.enable_pead_features,
        # Sector short selling parameters
        enable_sector_short_selling=args.enable_sector_short_selling,
        sector_short_selling_parquet=sector_short_selling_parquet,
        enable_sector_short_z_scores=not args.disable_sector_short_z_scores,
        sector_onehot33=args.sector_onehot33,
        sector_series_mcap=args.sector_series_mcap,
        sector_te_targets=te_targets,
        sector_series_levels=series_levels,
        sector_te_levels=te_levels,
        # Option market aggregates
        enable_option_market_features=args.attach_nk225_option_market,
        index_option_features_parquet=None,
        index_option_raw_parquet=args.index_option_parquet,
        # Advanced features
        enable_advanced_features=args.enable_advanced_features,
        # Sector cross-sectional features
        enable_sector_cs=args.enable_sector_cs,
        sector_cs_cols=[
            s.strip() for s in (args.sector_cs_cols or "").split(",") if s.strip()
        ],
        # Graph features
        enable_graph_features=args.enable_graph_features,
        graph_window=getattr(args, "graph_window", 60),
        graph_threshold=getattr(args, "graph_threshold", 0.3),
        graph_max_k=getattr(args, "graph_max_k", 4),
        graph_cache_dir=str(args.graph_cache_dir) if args.graph_cache_dir else None,
        disable_halt_mask=getattr(args, "disable_halt_mask", False),
    )
    logger.info("Full enriched dataset saved")
    if requested_start_date != start_date:
        logger.info(
            "Dataset start date adjusted to %s (requested %s) due to subscription coverage",
            start_date,
            requested_start_date,
        )
    logger.info(f"  Dataset : {pq_path}")
    logger.info(f"  Metadata: {meta_path}")
    logger.info(f"  Symlink : {Path('output') / 'ml_dataset_latest_full.parquet'}")

    try:
        _validate_enriched_dataset(pq_path, meta_path)
        meta_info = json.loads(meta_path.read_text())
        feature_total = meta_info.get("features", {}).get("count")
        logger.info("✅ Enriched dataset schema validated (features=%s)", feature_total)
    except RuntimeError as e:
        logger.error(str(e))
        return 2
    except Exception as e:
        logger.warning(f"Dataset validation skipped: {e}")

    # Also save a day-level TOPIX market-features artifact for auditing/consumers
    try:
        df_saved = pl.read_parquet(pq_path)
        mkt_cols = [c for c in df_saved.columns if c.startswith("mkt_")]
        if mkt_cols:
            topix_daily = (
                df_saved.select(["Date"] + mkt_cols)
                .group_by("Date")
                .agg([pl.all().first()])
                .sort("Date")
            )
            from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs

            out_topix = (
                output_dir
                / f"topix_market_features_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
            )
            save_parquet_with_gcs(topix_daily, out_topix)
        else:
            logger.warning(
                "No mkt_* columns found in saved dataset; TOPIX market artifact not written"
            )
    except Exception as e:
        logger.warning(f"TOPIX market features save skipped: {e}")

    # Optionally build Nikkei225 index option features and save as a separate artifact
    try:
        if args.enable_nk225_option_features:
            opt_raw: pl.DataFrame | None = None
            # Prefer provided parquet
            if args.index_option_parquet and args.index_option_parquet.exists():
                try:
                    opt_raw = pl.read_parquet(args.index_option_parquet)
                    logger.info(
                        f"Loaded index_option parquet: {args.index_option_parquet}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load index_option parquet: {e}")

            # If not provided, try to find cached index option features
            if opt_raw is None or opt_raw.is_empty():
                cached_option = _find_latest("nk225_index_option_features_*.parquet")
                if cached_option:
                    try:
                        logger.info(f"📦 Found cached index options: {cached_option}")
                        # Check if cache covers requested date range
                        cache_name = cached_option.stem
                        # Extract dates from filename: nk225_index_option_features_YYYYMMDD_YYYYMMDD
                        parts = cache_name.split("_")
                        if len(parts) >= 5:
                            cache_start = parts[-2]
                            cache_end = parts[-1]
                            cache_start_dt = datetime.strptime(cache_start, "%Y%m%d")
                            cache_end_dt = datetime.strptime(cache_end, "%Y%m%d")

                            # Check if cache covers our range (allow 1-day tolerance)
                            if cache_start_dt <= start_dt and cache_end_dt >= (
                                end_dt - timedelta(days=1)
                            ):
                                opt_raw = pl.read_parquet(cached_option)
                                logger.info(
                                    "✅ CACHE HIT: Index Options (saved 15-30 min)"
                                )
                            else:
                                logger.info(
                                    f"⚠️  Cache date range mismatch: cache={cache_start}→{cache_end}, requested={start_dt.strftime('%Y%m%d')}→{end_dt.strftime('%Y%m%d')}"
                                )
                    except Exception as e:
                        logger.warning(f"Failed to load cached index options: {e}")

            # If not provided, and JQuants is enabled, fetch from API per date range
            if (opt_raw is None or opt_raw.is_empty()) and args.jquants:
                try:
                    email = os.getenv("JQUANTS_AUTH_EMAIL", "")
                    password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
                    if email and password:
                        fetcher = JQuantsAsyncFetcher(email, password)
                        async with aiohttp.ClientSession() as session:
                            await fetcher.authenticate(session)
                            start_date = start_dt.strftime("%Y-%m-%d")
                            end_date = end_dt.strftime("%Y-%m-%d")
                            logger.info(
                                f"Fetching index options {start_date} → {end_date}"
                            )
                            opt_raw = await fetcher.get_index_option(
                                session, start_date, end_date
                            )
                except Exception as e:
                    logger.warning(f"Index option fetch failed: {e}")

            if opt_raw is not None and not opt_raw.is_empty():
                try:
                    from src.gogooku3.features.index_option import (
                        build_index_option_features,
                    )
                    from src.gogooku3.utils.gcs_storage import save_parquet_with_gcs

                    opt_feats = build_index_option_features(opt_raw)
                    out = (
                        output_dir
                        / f"nk225_index_option_features_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                    )
                    save_parquet_with_gcs(opt_feats, out)
                except Exception as e:
                    logger.warning(f"Failed to build/save option features: {e}")
            else:
                logger.info(
                    "No index_option data available; skipping option features build"
                )
    except Exception as e:
        logger.warning(f"Index option features step skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
