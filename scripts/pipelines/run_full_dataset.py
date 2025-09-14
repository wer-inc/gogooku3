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
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import polars as pl

# Ensure project root is on sys.path so that `scripts` is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import JQuants fetcher to get trade-spec directly (moved out of _archive)
from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher  # type: ignore
from scripts.pipelines.run_pipeline_v4_optimized import JQuantsPipelineV4Optimized
from src.pipeline.full_dataset import enrich_and_save

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_full_dataset")

# Defaults
DEFAULT_LOOKBACK_DAYS = 1826  # ~5 years


def _find_latest(glob: str) -> Path | None:
    """Return the latest matching file in `output/` by lexicographic order.

    Files are expected to include sortable date tokens in their names.
    """
    cands = sorted(Path("output").glob(glob))
    return cands[-1] if cands else None


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for building the full enriched dataset."""
    parser = argparse.ArgumentParser(
        description="Build fully enriched 5y dataset in one command"
    )
    parser.add_argument(
        "--jquants", action="store_true", help="Use JQuants API (requires .env)"
    )
    parser.add_argument(
        "--start-date", type=str, default=None, help="YYYY-MM-DD (default: today-5y)"
    )
    parser.add_argument(
        "--end-date", type=str, default=None, help="YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--topix-parquet",
        type=Path,
        default=None,
        help="Optional TOPIX parquet for offline enrichment",
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
        help="Enable daily margin interest features (default: off)",
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
        help="Enable Yang–Zhang volatility and VoV features",
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
        help="Enable advanced T+0 features (RSI×Vol, momentum×volume, MACD slope, cross-sectional ranks, calendar)",
    )
    # Graph-structured features (correlation graph)
    parser.add_argument(
        "--enable-graph-features",
        action="store_true",
        help="Enable graph-structured features (degree, peer corr mean, peer count)",
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
        default=10,
        help="Max edges per node (default: 10)",
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
        help="Enable Nikkei225 index option features build and save (default: off)",
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
        help="Enable sector cross-sectional features (sector-relative deviations, ranks, z-scores)",
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
        help="Enable short selling data integration (default: off)",
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
        help="Enable earnings announcement events data integration (default: off)",
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
        help="Enable sector-wise short selling features (default: off)",
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
    return parser.parse_args()


## Saving is delegated to src/pipeline/full_dataset.save_with_symlinks via enrich_and_save


async def main() -> int:
    """Entry point to orchestrate base pipeline and enrichment steps.

    Follows the documented 3-step flow and persists artifacts under `output/`.
    """
    args = _parse_args()

    # Load YAML config if provided (CLI takes precedence)
    if getattr(args, "config", None) is not None and args.config and args.config.exists():
        try:
            import yaml  # type: ignore
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            # Sector CS
            sec = cfg.get("sector_cs") or {}
            if isinstance(sec, dict):
                if not args.enable_sector_cs and isinstance(sec.get("enable"), bool):
                    args.enable_sector_cs = bool(sec.get("enable"))
                if not args.sector_cs_cols and isinstance(sec.get("include_cols"), (list, tuple)):
                    args.sector_cs_cols = ",".join(str(s) for s in sec.get("include_cols") if s)
            # Graph
            g = cfg.get("graph") or {}
            if isinstance(g, dict):
                if not args.enable_graph_features and isinstance(g.get("enable"), bool):
                    args.enable_graph_features = bool(g.get("enable"))
                if getattr(args, "graph_window", None) in (None, 60) and isinstance(g.get("window"), int):
                    args.graph_window = int(g.get("window"))
                if getattr(args, "graph_threshold", None) in (None, 0.3) and isinstance(g.get("threshold"), (float, int)):
                    args.graph_threshold = float(g.get("threshold"))
                if getattr(args, "graph_max_k", None) in (None, 10) and isinstance(g.get("max_k"), int):
                    args.graph_max_k = int(g.get("max_k"))
                if getattr(args, "graph_cache_dir", None) in (None,) and g.get("cache_dir"):
                    args.graph_cache_dir = Path(str(g.get("cache_dir")))
            # Option market attach
            om = cfg.get("option_market") or {}
            if isinstance(om, dict) and not args.attach_nk225_option_market and isinstance(om.get("attach"), bool):
                args.attach_nk225_option_market = bool(om.get("attach"))
        except Exception as e:
            logger.warning(f"Failed to load config YAML {args.config}: {e}")

    # Dry-run: print planned steps and exit successfully
    if getattr(args, "dry_run", False):
        print("=" * 60)
        print("[DRY-RUN] Build fully enriched 5y dataset")
        print("Steps:")
        print(" 0) Prepare trade-spec (JQuants optional or local fallback)")
        print(" 0.5) Fetch futures data (if --enable-futures)")
        print(" 0.6) Fetch short selling data (if --enable-short-selling)")
        print(" 0.7) Fetch sector short selling data (if --enable-sector-short-selling)")
        print(" 0.8) Fetch Nikkei225 index options (if --enable-nk225-option-features)")
        print(" 1) Run base optimized pipeline (prices + TA + statements)")
        print(" 2) Enrich with TOPIX, flow, sector, futures, short selling, sector short selling")
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

    logger.info("=== STEP 0: Prepare trade-spec for flow features ===")
    trades_spec_path: Path | None = None
    listed_info_path: Path | None = args.listed_info_parquet
    futures_path: Path | None = None
    short_selling_path: Path | None = None
    short_positions_path: Path | None = None
    if args.jquants:
        # Fetch trade-spec and save to Parquet
        email = os.getenv("JQUANTS_AUTH_EMAIL", "")
        password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
        if not email or not password:
            logger.error("JQuants credentials not found in environment/.env")
            return 1

        fetcher = JQuantsAsyncFetcher(email, password)
        async with aiohttp.ClientSession() as session:
            await fetcher.authenticate(session)
            logger.info(f"Fetching trade-spec from {start_date} to {end_date}")
            trades_df = await fetcher.get_trades_spec(session, start_date, end_date)
            # Fetch weekly margin interest for auto-attach (optional)
            try:
                logger.info("Fetching weekly margin interest for margin features")
                wmi_df = await fetcher.get_weekly_margin_interest(session, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to fetch weekly margin interest: {e}")
                wmi_df = pl.DataFrame()
            # Fetch daily margin interest for high-frequency credit data (optional)
            try:
                logger.info("Fetching daily margin interest for daily credit features")
                dmi_df = await fetcher.get_daily_margin_interest(session, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to fetch daily margin interest: {e}")
                dmi_df = pl.DataFrame()
            # Also fetch listed_info for sector/market enrichment
            try:
                logger.info("Fetching listed_info for sector/market enrichment")
                info_df = await fetcher.get_listed_info(session)
            except Exception as e:
                logger.warning(f"Failed to fetch listed_info: {e}")
                info_df = pl.DataFrame()
        if trades_df is None or trades_df.is_empty():
            logger.warning("No trade-spec data fetched; will try local fallback for flow features")
        else:
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            trades_spec_path = output_dir / f"trades_spec_history_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
            trades_df.write_parquet(trades_spec_path)
            logger.info(f"Saved trade-spec: {trades_spec_path}")
        # Save listed_info if fetched (even if trade-spec failed)
        if listed_info_path is None:
            # Name by end date for reproducibility
            listed_info_path = (Path("output") / f"listed_info_history_{end_dt.strftime('%Y%m%d')}.parquet")
        if 'info_df' in locals() and info_df is not None and not info_df.is_empty():
            try:
                listed_info_path.parent.mkdir(parents=True, exist_ok=True)
                info_df.write_parquet(listed_info_path)
                logger.info(f"Saved listed_info: {listed_info_path}")
            except Exception as e:
                logger.warning(f"Failed to save listed_info parquet: {e}")
        # Save weekly margin interest if fetched
        wmi_path: Path | None = None
        if 'wmi_df' in locals() and wmi_df is not None and not wmi_df.is_empty():
            try:
                outdir = Path("output"); outdir.mkdir(parents=True, exist_ok=True)
                wmi_path = outdir / f"weekly_margin_interest_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                wmi_df.write_parquet(wmi_path)
                logger.info(f"Saved weekly margin interest: {wmi_path}")
            except Exception as e:
                logger.warning(f"Failed to save weekly margin parquet: {e}")
        # Save daily margin interest if fetched
        dmi_path: Path | None = None
        if 'dmi_df' in locals() and dmi_df is not None and not dmi_df.is_empty():
            try:
                outdir = Path("output"); outdir.mkdir(parents=True, exist_ok=True)
                dmi_path = outdir / f"daily_margin_interest_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                dmi_df.write_parquet(dmi_path)
                logger.info(f"Saved daily margin interest: {dmi_path}")
            except Exception as e:
                logger.warning(f"Failed to save daily margin parquet: {e}")

        # Fetch futures data for derivatives features (optional)
        if not args.disable_futures or (args.futures_parquet is not None):
            try:
                logger.info("Fetching futures data for derivatives features")
                futures_df = await fetcher.get_futures_daily(session, start_date, end_date)
                if futures_df is not None and not futures_df.is_empty():
                    outdir = Path("output"); outdir.mkdir(parents=True, exist_ok=True)
                    futures_path = outdir / f"futures_daily_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                    futures_df.write_parquet(futures_path)
                    logger.info(f"Saved futures data: {futures_path}")
                else:
                    logger.warning("No futures data retrieved from API")
            except Exception as e:
                logger.warning(f"Failed to fetch futures data: {e}")

        # Fetch short selling data for short selling features (optional)
        if args.enable_short_selling:
            # Fetch short selling ratio data
            try:
                logger.info("Fetching short selling ratio data")
                short_df = await fetcher.get_short_selling(session, start_date, end_date)
                if short_df is not None and not short_df.is_empty():
                    outdir = Path("output"); outdir.mkdir(parents=True, exist_ok=True)
                    short_selling_path = outdir / f"short_selling_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                    short_df.write_parquet(short_selling_path)
                    logger.info(f"Saved short selling data: {short_selling_path}")
                else:
                    logger.warning("No short selling data retrieved from API")
            except Exception as e:
                logger.warning(f"Failed to fetch short selling data: {e}")

            # Fetch short selling positions data
            try:
                logger.info("Fetching short selling positions data")
                positions_df = await fetcher.get_short_selling_positions(session, start_date, end_date)
                if positions_df is not None and not positions_df.is_empty():
                    outdir = Path("output"); outdir.mkdir(parents=True, exist_ok=True)
                    short_positions_path = outdir / f"short_positions_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                    positions_df.write_parquet(short_positions_path)
                    logger.info(f"Saved short selling positions data: {short_positions_path}")
                else:
                    logger.warning("No short selling positions data retrieved from API")
            except Exception as e:
                logger.warning(f"Failed to fetch short selling positions data: {e}")

        # Fetch sector-wise short selling data (optional)
        if args.enable_sector_short_selling:
            try:
                logger.info("Fetching sector-wise short selling data")
                sector_short_df = await fetcher.get_sector_short_selling(session, start_date, end_date)
                if sector_short_df is not None and not sector_short_df.is_empty():
                    outdir = Path("output"); outdir.mkdir(parents=True, exist_ok=True)
                    sector_short_path = outdir / f"sector_short_selling_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                    sector_short_df.write_parquet(sector_short_path)
                    logger.info(f"Saved sector short selling data: {sector_short_path}")
                else:
                    logger.warning("No sector short selling data retrieved from API")
            except Exception as e:
                logger.warning(f"Failed to fetch sector short selling data: {e}")
    else:
        # Offline fallback: look for a local trades_spec parquet
        trades_spec_path = _find_latest("trades_spec_history_*.parquet")
        if trades_spec_path:
            logger.info(f"Using local trade-spec parquet: {trades_spec_path}")
        else:
            logger.warning("No local trades_spec parquet found; flow features may be skipped")
        # Offline listed_info fallback
        if listed_info_path is None:
            listed_info_path = _find_latest("listed_info_history_*.parquet")
        if listed_info_path:
            logger.info(f"Using listed_info parquet: {listed_info_path}")
        else:
            logger.warning("No listed_info parquet provided/found; sector enrichment will be skipped")

        # Offline futures fallback
        if args.enable_futures or args.futures_parquet is not None:
            if args.futures_parquet is not None and args.futures_parquet.exists():
                futures_path = args.futures_parquet
                logger.info(f"Using provided futures parquet: {futures_path}")
            else:
                futures_path = _find_latest("futures_daily_*.parquet")
                if futures_path:
                    logger.info(f"Using local futures parquet: {futures_path}")
                else:
                    logger.warning("No futures parquet found; futures features will be skipped")

        # Offline short selling fallback
        if args.enable_short_selling:
            # Short selling ratio data
            if args.short_selling_parquet is not None and args.short_selling_parquet.exists():
                short_selling_path = args.short_selling_parquet
                logger.info(f"Using provided short selling parquet: {short_selling_path}")
            else:
                short_selling_path = _find_latest("short_selling_*.parquet")
                if short_selling_path:
                    logger.info(f"Using local short selling parquet: {short_selling_path}")
                else:
                    logger.warning("No short selling parquet found; short selling features will be skipped")

            # Short selling positions data
            if args.short_positions_parquet is not None and args.short_positions_parquet.exists():
                short_positions_path = args.short_positions_parquet
                logger.info(f"Using provided short positions parquet: {short_positions_path}")
            else:
                short_positions_path = _find_latest("short_positions_*.parquet")
                if short_positions_path:
                    logger.info(f"Using local short positions parquet: {short_positions_path}")
                else:
                    logger.warning("No short positions parquet found; positions features will be skipped")

    logger.info("=== STEP 1: Run base optimized pipeline (prices + TA + statements) ===")
    pipeline = JQuantsPipelineV4Optimized()
    df_base, metadata = await pipeline.run(use_jquants=args.jquants, start_date=start_date, end_date=end_date)
    if df_base is None or metadata is None:
        logger.error("Base pipeline failed")
        return 1

    # Use freshly built base frame; avoid overriding with older ml_dataset_latest.parquet
    output_dir = pipeline.output_dir if hasattr(pipeline, "output_dir") else Path("output")

    logger.info("=== STEP 2: Enrich with TOPIX + statements + flow (trade-spec) + margin weekly ===")
    # Resolve weekly margin parquet (existing style: auto-discover if not provided; skip gracefully if missing)
    margin_weekly_parquet: Path | None = None
    if args.weekly_margin_parquet is not None and args.weekly_margin_parquet.exists():
        margin_weekly_parquet = args.weekly_margin_parquet
    else:
        # prefer the one we just saved (if any)
        if 'wmi_path' in locals() and wmi_path and wmi_path.exists():
            margin_weekly_parquet = wmi_path
        else:
            margin_weekly_parquet = _find_latest("weekly_margin_interest_*.parquet")

    # Resolve daily margin parquet (similar to weekly margin handling)
    daily_margin_parquet: Path | None = None
    if args.daily_margin_parquet is not None and args.daily_margin_parquet.exists():
        daily_margin_parquet = args.daily_margin_parquet
    else:
        # prefer the one we just saved (if any)
        if 'dmi_path' in locals() and dmi_path and dmi_path.exists():
            daily_margin_parquet = dmi_path
        else:
            daily_margin_parquet = _find_latest("daily_margin_interest_*.parquet")

    # Resolve sector short selling parquet (similar to other data sources)
    sector_short_selling_parquet: Path | None = None
    if args.sector_short_selling_parquet is not None and args.sector_short_selling_parquet.exists():
        sector_short_selling_parquet = args.sector_short_selling_parquet
    else:
        # prefer the one we just saved (if any)
        if 'sector_short_path' in locals() and sector_short_path and sector_short_path.exists():
            sector_short_selling_parquet = sector_short_path
        else:
            sector_short_selling_parquet = _find_latest("sector_short_selling_*.parquet")

    te_targets = [s.strip() for s in (args.sector_te_targets or "").split(",") if s.strip()]
    series_levels = [s.strip() for s in (getattr(args, 'sector_series_levels', '33') or "").split(",") if s.strip()]
    te_levels = [s.strip() for s in (getattr(args, 'sector_te_levels', '33') or "").split(",") if s.strip()]
    pq_path, meta_path = await enrich_and_save(
        df_base,
        output_dir=output_dir,
        jquants=args.jquants,
        start_date=start_date,
        end_date=end_date,
        trades_spec_path=trades_spec_path,
        topix_parquet=args.topix_parquet,
        statements_parquet=args.statements_parquet,
        listed_info_parquet=listed_info_path,
        enable_futures=not getattr(args, "disable_futures", False),
        futures_parquet=futures_path,
        futures_categories=[s.strip() for s in (getattr(args, "futures_categories", "") or "").split(",") if s.strip()],
        futures_continuous=args.futures_continuous,
        nk225_parquet=getattr(args, "nk225_parquet", None),
        reit_parquet=getattr(args, "reit_parquet", None),
        jpx400_parquet=getattr(args, "jpx400_parquet", None),
        enable_advanced_vol=args.enable_advanced_vol,
        adv_vol_windows=[int(s.strip()) for s in (args.adv_vol_windows or '').split(',') if s.strip()],
        enable_margin_weekly=bool(margin_weekly_parquet is not None and margin_weekly_parquet.exists()),
        margin_weekly_parquet=margin_weekly_parquet,
        margin_weekly_lag=getattr(args, "margin_weekly_lag", 3),
        adv_window_days=getattr(args, "adv_window_days", 20),
        enable_daily_margin=args.enable_daily_margin or bool(daily_margin_parquet is not None and daily_margin_parquet.exists()),
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
        sector_cs_cols=[s.strip() for s in (args.sector_cs_cols or '').split(',') if s.strip()],
        # Graph features
        enable_graph_features=args.enable_graph_features,
        graph_window=getattr(args, "graph_window", 60),
        graph_threshold=getattr(args, "graph_threshold", 0.3),
        graph_max_k=getattr(args, "graph_max_k", 10),
        graph_cache_dir=str(args.graph_cache_dir) if args.graph_cache_dir else None,
    )
    logger.info("Full enriched dataset saved")
    logger.info(f"  Dataset : {pq_path}")
    logger.info(f"  Metadata: {meta_path}")
    logger.info(f"  Symlink : {output_dir / 'ml_dataset_latest_full.parquet'}")

    # Optionally build Nikkei225 index option features and save as a separate artifact
    try:
        if args.enable_nk225_option_features:
            opt_raw: pl.DataFrame | None = None
            # Prefer provided parquet
            if args.index_option_parquet and args.index_option_parquet.exists():
                try:
                    opt_raw = pl.read_parquet(args.index_option_parquet)
                    logger.info(f"Loaded index_option parquet: {args.index_option_parquet}")
                except Exception as e:
                    logger.warning(f"Failed to load index_option parquet: {e}")

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
                            logger.info(f"Fetching index options {start_date} → {end_date}")
                            opt_raw = await fetcher.get_index_option(session, start_date, end_date)
                except Exception as e:
                    logger.warning(f"Index option fetch failed: {e}")

            if opt_raw is not None and not opt_raw.is_empty():
                try:
                    from src.gogooku3.features.index_option import build_index_option_features

                    opt_feats = build_index_option_features(opt_raw)
                    out = output_dir / f"nk225_index_option_features_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
                    opt_feats.write_parquet(out)
                    logger.info(f"Saved Nikkei225 option features: {out}")
                except Exception as e:
                    logger.warning(f"Failed to build/save option features: {e}")
            else:
                logger.info("No index_option data available; skipping option features build")
    except Exception as e:
        logger.warning(f"Index option features step skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
