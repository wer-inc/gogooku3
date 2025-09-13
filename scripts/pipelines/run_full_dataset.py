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
    [--statements-parquet output/event_raw_statements_YYYYMMDD_YYYYMMDD.parquet]

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

# Import JQuants fetcher to get trade-spec directly
from scripts._archive.run_pipeline import JQuantsAsyncFetcher  # type: ignore
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

    # Dry-run: print planned steps and exit successfully
    if getattr(args, "dry_run", False):
        print("=" * 60)
        print("[DRY-RUN] Build fully enriched 5y dataset")
        print("Steps:")
        print(" 0) Prepare trade-spec (JQuants optional or local fallback)")
        print(" 1) Run base optimized pipeline (prices + TA + statements)")
        print(" 2) Enrich with TOPIX, flow, and sector features")
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

    logger.info("=== STEP 1: Run base optimized pipeline (prices + TA + statements) ===")
    pipeline = JQuantsPipelineV4Optimized()
    df_base, metadata = await pipeline.run(use_jquants=args.jquants, start_date=start_date, end_date=end_date)
    if df_base is None or metadata is None:
        logger.error("Base pipeline failed")
        return 1

    # Load base latest parquet for consistent schema before enrichment
    output_dir = pipeline.output_dir if hasattr(pipeline, "output_dir") else Path("output")
    base_latest = output_dir / "ml_dataset_latest.parquet"
    if base_latest.exists():
        try:
            df_base = pl.read_parquet(base_latest)
        except Exception:
            pass

    logger.info("=== STEP 2: Enrich with TOPIX + statements + flow (trade-spec) ===")
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
        sector_onehot33=args.sector_onehot33,
        sector_series_mcap=args.sector_series_mcap,
        sector_te_targets=te_targets,
        sector_series_levels=series_levels,
        sector_te_levels=te_levels,
    )
    logger.info("Full enriched dataset saved")
    logger.info(f"  Dataset : {pq_path}")
    logger.info(f"  Metadata: {meta_path}")
    logger.info(f"  Symlink : {output_dir / 'ml_dataset_latest_full.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
