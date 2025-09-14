#!/usr/bin/env python3
"""
Full dataset pipeline orchestrator (enrichment + save).

Provides reusable functions so that CLI wrappers under scripts/ remain thin.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import aiohttp
import polars as pl

logger = logging.getLogger(__name__)


def save_with_symlinks(
    df: pl.DataFrame,
    output_dir: Path,
    tag: str = "full",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize column names to align with docs/DATASET.md (non-breaking rename)
    try:
        rename_map = {
            "is_ema_5_valid": "is_ema5_valid",
            "is_ema_10_valid": "is_ema10_valid",
            "is_ema_20_valid": "is_ema20_valid",
            "is_ema_60_valid": "is_ema60_valid",
            "is_ema_200_valid": "is_ema200_valid",
            "is_rsi_2_valid": "is_rsi2_valid",
        }
        present_map = {k: v for k, v in rename_map.items() if k in df.columns}
        if present_map:
            df = df.rename(present_map)
            logger.info(f"Column names normalized: {present_map}")
    except Exception as e:
        logger.warning(f"Column rename normalization skipped: {e}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if start_date and end_date:
        dr = f"{start_date.replace('-', '')}_{end_date.replace('-', '')}"
        parquet_path = output_dir / f"ml_dataset_{dr}_{ts}_{tag}.parquet"
        meta_path = output_dir / f"ml_dataset_{dr}_{ts}_{tag}_metadata.json"
    else:
        parquet_path = output_dir / f"ml_dataset_{ts}_{tag}.parquet"
        meta_path = output_dir / f"ml_dataset_{ts}_{tag}_metadata.json"

    # Write parquet with metadata where possible
    try:
        import pyarrow.parquet as pq  # type: ignore
        table = df.to_arrow()
        schema = table.schema
        existing = schema.metadata or {}
        meta = dict(existing)
        if start_date and end_date:
            meta.update({
                b"start_date": start_date.encode("utf-8"),
                b"end_date": end_date.encode("utf-8"),
                b"generator": b"full_dataset.py",
            })
        else:
            meta.update({b"generator": b"full_dataset.py"})
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, str(parquet_path))
    except Exception:
        df.write_parquet(parquet_path)

    # Build metadata json via builder (to keep consistent shape)
    from src.gogooku3.pipeline.builder import MLDatasetBuilder
    builder = MLDatasetBuilder(output_dir=output_dir)
    metadata = builder.create_metadata(df)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)

    latest_pq = output_dir / f"ml_dataset_latest_{tag}.parquet"
    latest_meta = output_dir / f"ml_dataset_latest_{tag}_metadata.json"
    for link, target in [(latest_pq, parquet_path.name), (latest_meta, meta_path.name)]:
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
        except Exception:
            pass
        link.symlink_to(target)

    return parquet_path, meta_path


def _find_latest(glob: str) -> Path | None:
    cands = sorted((Path('output')).glob(glob))
    return cands[-1] if cands else None


async def enrich_and_save(
    df_base: pl.DataFrame,
    *,
    output_dir: Path,
    jquants: bool,
    start_date: str,
    end_date: str,
    trades_spec_path: Path | None = None,
    topix_parquet: Path | None = None,
    statements_parquet: Path | None = None,
    listed_info_parquet: Path | None = None,
    # Futures integration (optional)
    enable_futures: bool = True,
    futures_parquet: Path | None = None,
    futures_categories: list[str] | None = None,
    futures_continuous: bool = False,
    # Optional spot index parquets for basis mapping
    nk225_parquet: Path | None = None,
    reit_parquet: Path | None = None,
    jpx400_parquet: Path | None = None,
    # Margin weekly integration (optional)
    enable_margin_weekly: bool = False,
    margin_weekly_parquet: Path | None = None,
    margin_weekly_lag: int = 3,
    adv_window_days: int = 20,
    # Margin daily integration (optional)
    enable_daily_margin: bool = False,
    daily_margin_parquet: Path | None = None,
    # Short selling integration (optional)
    enable_short_selling: bool = False,
    short_selling_parquet: Path | None = None,
    short_positions_parquet: Path | None = None,
    short_selling_z_window: int = 252,
    # Earnings events integration (optional)
    enable_earnings_events: bool = False,
    earnings_announcements_parquet: Path | None = None,
    enable_pead_features: bool = True,
    # Sector short selling integration (optional)
    enable_sector_short_selling: bool = False,
    sector_short_selling_parquet: Path | None = None,
    enable_sector_short_z_scores: bool = True,
    sector_onehot33: bool = False,
    sector_series_mcap: str = "auto",
    sector_te_targets: list[str] | None = None,
    sector_series_levels: list[str] | None = None,
    sector_te_levels: list[str] | None = None,
    # Advanced volatility
    enable_advanced_vol: bool = False,
    adv_vol_windows: list[int] | None = None,
    # Nikkei225 index option market aggregates (optional)
    enable_option_market_features: bool = False,
    index_option_features_parquet: Path | None = None,
    index_option_raw_parquet: Path | None = None,
    # Advanced features (Phase 1)
    enable_advanced_features: bool = False,
    # Sector cross-sectional features (Phase 2)
    enable_sector_cs: bool = False,
    sector_cs_cols: list[str] | None = None,
    # Graph features (Phase 3)
    enable_graph_features: bool = False,
    graph_window: int = 60,
    graph_threshold: float = 0.3,
    graph_max_k: int = 10,
    graph_cache_dir: str | None = None,
) -> tuple[Path, Path]:
    """Attach TOPIX + statements + flow then save with symlinks.

    Includes an assurance step to guarantee mkt_* presence by discovering
    or fetching TOPIX parquet when needed.
    """

    from src.gogooku3.pipeline.builder import MLDatasetBuilder

    builder = MLDatasetBuilder(output_dir=output_dir)

    # TOPIX: prefer provided parquet, else API, else local, else fallback in builder
    topix_df = None
    if topix_parquet and Path(topix_parquet).exists():
        try:
            topix_df = pl.read_parquet(topix_parquet)
            logger.info(f"Loaded TOPIX from parquet: {topix_parquet}")
        except Exception as e:
            logger.warning(f"Failed to read TOPIX parquet: {e}")
    if topix_df is None and jquants:
        try:
            import os

            from src.gogooku3.components.jquants_async_fetcher import (
                JQuantsAsyncFetcher,  # type: ignore
            )
            email = os.getenv("JQUANTS_AUTH_EMAIL", "")
            password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
            if email and password:
                fetcher = JQuantsAsyncFetcher(email, password)
                async with aiohttp.ClientSession() as session:
                    await fetcher.authenticate(session)
                    logger.info(f"Fetching TOPIX {start_date} → {end_date}")
                    topix_df = await fetcher.fetch_topix_data(session, start_date, end_date)
        except Exception as e:
            logger.warning(f"TOPIX API fetch failed: {e}")
    if topix_df is None:
        # Choose the most suitable local TOPIX parquet by coverage
        try:
            import re
            best_path = None
            best_score = -1
            start_int = int(start_date.replace('-', ''))
            end_int = int(end_date.replace('-', ''))
            for cand in sorted((Path('output')).glob('topix_history_*.parquet')):
                m = re.search(r"topix_history_(\d{8})_(\d{8})\.parquet$", cand.name)
                if not m:
                    continue
                s, e = int(m.group(1)), int(m.group(2))
                # Score by coverage and whether it encloses [start, end]
                encloses = (s <= start_int and e >= end_int)
                coverage = e - s
                score = (1_000_000_000 if encloses else 0) + coverage
                if score > best_score:
                    best_score = score
                    best_path = cand
            if best_path and best_path.exists():
                try:
                    topix_df = pl.read_parquet(best_path)
                    logger.info(f"Loaded TOPIX from local: {best_path}")
                except Exception as e:
                    logger.warning(f"Failed to read local TOPIX parquet: {e}")
        except Exception as e:
            logger.warning(f"TOPIX local selection failed: {e}")

    df = builder.add_topix_features(df_base, topix_df=topix_df)

    # CRITICAL FIX: Add forward return labels (feat_ret_1d, feat_ret_5d, feat_ret_10d, feat_ret_20d)
    # This fixes the missing supervised learning targets identified in the PDF diagnosis
    try:
        df = builder.create_technical_features(df)
        logger.info("Forward return labels (feat_ret_1d, feat_ret_5d, feat_ret_10d, feat_ret_20d) added successfully")

        # QUALITY CHECK: Validate forward return labels per PDF requirements
        validation_results = builder.validate_forward_return_labels(df)
        if validation_results["passed"]:
            logger.info("✅ Forward return labels validation PASSED")
            if validation_results["statistics"]:
                logger.info(f"📊 Validation stats: {validation_results['statistics']}")
        else:
            logger.warning("⚠️  Forward return labels validation FAILED")
            for warning in validation_results["warnings"]:
                logger.warning(f"   - {warning}")
            for error in validation_results["critical_errors"]:
                logger.error(f"   - CRITICAL: {error}")
            # Don't fail the pipeline on validation warnings, but log them prominently

    except Exception as e:
        logger.error(f"CRITICAL: Failed to add forward return labels: {e}")
        # This is a critical failure - forward returns are required for supervised learning
        raise

    # Advanced equity features (optional, Phase 1)
    try:
        if enable_advanced_features:
            from src.gogooku3.features.advanced_features import add_advanced_features
            df = add_advanced_features(df)
            logger.info("Advanced features attached (interactions, CS ranks, calendar)")
        else:
            logger.info("Advanced features disabled; skipping")
    except Exception as e:
        logger.warning(f"Advanced features attach skipped: {e}")

    # Finalize to match DATASET.md
    try:
        df = builder.finalize_for_spec(df)
    except Exception:
        pass

    # Statements attach if not present
    if not any(c.startswith("stmt_") for c in df.columns):
        stm_path: Path | None = None
        if statements_parquet and Path(statements_parquet).exists():
            stm_path = statements_parquet
        else:
            symlink = output_dir / "event_raw_statements.parquet"
            if symlink.exists():
                stm_path = symlink
            else:
                stm_path = _find_latest("event_raw_statements_*.parquet")
        if stm_path and stm_path.exists():
            try:
                stm_df = pl.read_parquet(stm_path)
                df = builder.add_statements_features(df, stm_df)
                logger.info("Statements features attached from parquet")
            except Exception as e:
                logger.warning(f"Failed to attach statements: {e}")

    # Optional sector enrichment (listed_info)
    listed_info_df = None
    if listed_info_parquet and Path(listed_info_parquet).exists():
        try:
            listed_info_df = pl.read_parquet(listed_info_parquet)
            df = builder.add_sector_features(df, listed_info_df)
            # Sector series (eq-median by level list)
            levels = sector_series_levels or ["33"]
            for lvl in levels:
                try:
                    df = builder.add_sector_series(df, level=lvl, windows=(1, 5, 20), series_mcap=sector_series_mcap)
                except Exception as e:
                    logger.warning(f"Sector series attach failed for level {lvl}: {e}")
            # Sector encodings (one-hot for 17, daily freqs)
            try:
                df = builder.add_sector_encodings(df, onehot_17=True, onehot_33=sector_onehot33, freq_daily=True)
            except Exception as e:
                logger.warning(f"Sector encodings failed: {e}")
            # Relative-to-sector features (rel/alpha/z-in-sec)
            try:
                df = builder.add_relative_to_sector(df, level="33", x_cols=("returns_5d", "ma_gap_5_20"))
            except Exception as e:
                logger.warning(f"Relative-to-sector features failed: {e}")
            # Sector target encoding with cross-fit + lag
            te_targets = sector_te_targets or ["target_5d"]
            te_levels = sector_te_levels or ["33"]
            for lvl in te_levels:
                for tcol in te_targets:
                    try:
                        df = builder.add_sector_target_encoding(df, target_col=tcol, level=lvl, k_folds=5, lag_days=1, m=100.0)
                    except Exception as e:
                        logger.warning(f"Sector target encoding failed for level {lvl}, target {tcol}: {e}")
            # Back-compat: if only 33-level TE requested, alias te33_* → te_* for consumers
            if te_levels == ["33"]:
                for tcol in te_targets:
                    src = f"te33_sec_{tcol}"
                    dst = f"te_sec_{tcol}"
                    if src in df.columns and dst not in df.columns:
                        df = df.with_columns(pl.col(src).alias(dst))
            logger.info("Sector enrichment completed (sector33/MarketCode/CompanyName)")
        except Exception as e:
            logger.warning(f"Failed sector enrichment: {e}")

    # Sector cross-sectional features (Phase 2): relies on sector column if present
    try:
        if enable_sector_cs:
            from src.gogooku3.features.sector_cross_sectional import add_sector_cross_sectional_features
            df = add_sector_cross_sectional_features(df, include_cols=sector_cs_cols)
            logger.info("Sector cross-sectional features attached (ret_vs_sec, rank_in_sec, volume/rv20 z in sector)")
        else:
            logger.info("Sector cross-sectional features disabled; skipping")
    except Exception as e:
        logger.warning(f"Sector cross-sectional features attach skipped: {e}")

    # Graph-structured features (Phase 3): degree, peer corr mean, peer count
    try:
        if enable_graph_features:
            from src.gogooku3.features.graph_features import add_graph_features
            df = add_graph_features(
                df,
                return_col="returns_1d" if "returns_1d" in df.columns else "feat_ret_1d",
                window=graph_window,
                min_obs=max(20, min(graph_window // 2, graph_window - 5)),
                threshold=graph_threshold,
                max_k=graph_max_k,
                method="pearson",
                cache_dir=graph_cache_dir,
            )
            logger.info("Graph features attached (graph_degree, peer_corr_mean, peer_count)")
        else:
            logger.info("Graph features disabled; skipping")
    except Exception as e:
        logger.warning(f"Graph features attach skipped: {e}")

    # Futures features (ON/EOD, T+0/T+1 leak-safe)
    futures_df = None
    if enable_futures:
        # Resolve parquet first
        try:
            if futures_parquet and Path(futures_parquet).exists():
                futures_df = pl.read_parquet(futures_parquet)
            else:
                # Auto-discover under output/
                cands = sorted(output_dir.glob("futures_daily_*.parquet"))
                if cands:
                    futures_df = pl.read_parquet(cands[-1])
        except Exception as e:
            logger.warning(f"Failed to read futures parquet: {e}")

        # If missing and jquants available, fetch from API for the requested range
        if futures_df is None and jquants:
            try:
                import os

                from src.gogooku3.components.jquants_async_fetcher import (
                    JQuantsAsyncFetcher,  # type: ignore
                )

                email = os.getenv("JQUANTS_AUTH_EMAIL", "")
                password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
                if email and password:
                    fetcher2 = JQuantsAsyncFetcher(email, password)
                    async with aiohttp.ClientSession() as session:
                        await fetcher2.authenticate(session)
                        logger.info(
                            f"Fetching futures daily {start_date} → {end_date} for enrichment"
                        )
                        fut_df = await fetcher2.get_futures_daily(
                            session, start_date, end_date
                        )
                        if fut_df is not None and not fut_df.is_empty():
                            futures_df = fut_df
                            # Save for reuse
                            try:
                                out = (
                                    output_dir
                                    / f"futures_daily_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                                )
                                fut_df.write_parquet(out)
                                logger.info(f"Saved futures parquet: {out}")
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"Futures API fetch failed: {e}")

    if enable_futures and futures_df is not None and not futures_df.is_empty():
        try:
            # Use v3.2 leak-safe attachment via builder
            spot_map: dict[str, pl.DataFrame] = {}
            if topix_df is not None and not topix_df.is_empty():
                spot_map["TOPIXF"] = topix_df
            # Additional optional spot parquets
            def _load_spot(path: Path | None) -> pl.DataFrame | None:
                if path and Path(path).exists():
                    try:
                        return pl.read_parquet(path)
                    except Exception:
                        return None
                return None
            _nk = _load_spot(nk225_parquet)
            _rt = _load_spot(reit_parquet)
            _j4 = _load_spot(jpx400_parquet)
            # Auto-discover spot parquets under output/ if not provided
            def _auto_find_spot(keywords: list[str]) -> Path | None:
                cands = []
                for p in (output_dir or Path("output")).glob("*.parquet"):
                    name = p.name.lower()
                    if all(k in name for k in keywords):
                        cands.append(p)
                return cands[-1] if cands else None
            if _nk is None:
                _nk_path = _auto_find_spot(["nikkei"]) or _auto_find_spot(["nk225"]) or _auto_find_spot(["nikkei225"])  # type: ignore[assignment]
                _nk = _load_spot(_nk_path)
            if _rt is None:
                _rt_path = _auto_find_spot(["reit"])  # type: ignore[assignment]
                _rt = _load_spot(_rt_path)
            if _j4 is None:
                _j4_path = _auto_find_spot(["jpx400"]) or _auto_find_spot(["jp", "400"])  # type: ignore[assignment]
                _j4 = _load_spot(_j4_path)
            if _nk is not None and not _nk.is_empty():
                spot_map["NK225F"] = _nk
            if _rt is not None and not _rt.is_empty():
                spot_map["REITF"] = _rt
            if _j4 is not None and not _j4.is_empty():
                spot_map["JN400F"] = _j4
            df = builder.add_futures_block(
                df,
                futures_df=futures_df,
                categories=(futures_categories or ["TOPIXF", "NK225F", "JN400F", "REITF"]),
                topix_df=topix_df,
                spot_map=spot_map or None,
                make_continuous_series=bool(futures_continuous),
            )

            # Log basis coverage per category
            categories = futures_categories or ["TOPIXF", "NK225F", "JN400F", "REITF"]
            logger.info("🚀 Futures features attached (ON/EOD)")
            logger.info("📊 Basis coverage by category:")
            for category in categories:
                spot_available = category in (spot_map or {})
                futures_cols = [c for c in df.columns if f"fut_{category.lower()}_" in c.lower()]
                basis_cols = [c for c in futures_cols if "basis" in c.lower()]

                logger.info(f"  - {category}: spot={'✅' if spot_available else '❌'}, "
                          f"futures_features={len(futures_cols)}, basis_features={len(basis_cols)}")

            continuous_enabled = "✅ ON" if futures_continuous else "❌ OFF"
            logger.info(f"📈 Continuous series (fut_whole_ret_cont_*): {continuous_enabled}")
        except Exception as e:
            logger.warning(f"Futures enrichment skipped: {e}")

    # Flow attach
    if trades_spec_path and Path(trades_spec_path).exists():
        try:
            trades_spec_df = pl.read_parquet(trades_spec_path)
            # Pass listed_info_df (if available) for accurate Section mapping
            df = builder.add_flow_features(df, trades_spec_df, listed_info_df=listed_info_df)
        except Exception as e:
            logger.warning(f"Skipping flow enrichment due to error: {e}")

    # Margin weekly attach (leak-safe as-of)
    # Be tolerant: if a weekly margin parquet exists, attach it even if the flag wasn't set.
    try:
        w_path: Path | None = None
        if margin_weekly_parquet and Path(margin_weekly_parquet).exists():
            w_path = margin_weekly_parquet
        else:
            # Auto-discover under output/
            cands = sorted(output_dir.glob("weekly_margin_interest_*.parquet"))
            if cands:
                w_path = cands[-1]
        if enable_margin_weekly or (w_path and w_path.exists()):
            if w_path and w_path.exists():
                wdf = pl.read_parquet(w_path)
                df = builder.add_margin_weekly_block(
                    df,
                    wdf,
                    lag_bdays_weekly=margin_weekly_lag,
                    adv_window_days=adv_window_days,
                )
                logger.info(f"Margin weekly features attached from: {w_path}")
            else:
                logger.info("Margin weekly requested but no parquet provided/found; skipping")
        else:
            logger.info("Margin weekly not enabled and no parquet found; skipping")
    except Exception as e:
        logger.warning(f"Margin weekly attach skipped: {e}")

    # Daily margin attach (leak-safe as-of with T+1 rule)
    try:
        d_path: Path | None = None
        if daily_margin_parquet and Path(daily_margin_parquet).exists():
            d_path = daily_margin_parquet
        else:
            # Auto-discover under output/
            cands = sorted(output_dir.glob("daily_margin_interest_*.parquet"))
            if cands:
                d_path = cands[-1]
        if enable_daily_margin or (d_path and d_path.exists()):
            if d_path and d_path.exists():
                ddf = pl.read_parquet(d_path)
                df = builder.add_daily_margin_block(
                    df,
                    ddf,
                    adv_window_days=adv_window_days,
                    enable_z_scores=True,
                )
                logger.info(f"Daily margin features attached from: {d_path}")
            else:
                logger.info("Daily margin requested but no parquet provided/found; skipping")
        else:
            logger.info("Daily margin not enabled and no parquet found; skipping")
    except Exception as e:
        logger.warning(f"Daily margin attach skipped: {e}")

    # Short selling attach (leak-safe as-of with T+1 rule)
    try:
        ss_path: Path | None = None
        pos_path: Path | None = None

        if short_selling_parquet and Path(short_selling_parquet).exists():
            ss_path = short_selling_parquet
        else:
            # Auto-discover under output/
            cands = sorted(output_dir.glob("short_selling_*.parquet"))
            if cands:
                ss_path = cands[-1]

        if short_positions_parquet and Path(short_positions_parquet).exists():
            pos_path = short_positions_parquet
        else:
            # Auto-discover under output/
            cands = sorted(output_dir.glob("short_positions_*.parquet"))
            if cands:
                pos_path = cands[-1]

        if enable_short_selling or (ss_path and ss_path.exists()):
            # Load short selling data
            short_df = None
            positions_df = None

            if ss_path and ss_path.exists():
                try:
                    short_df = pl.read_parquet(ss_path)
                    logger.info(f"Loaded short selling data from: {ss_path}")
                except Exception as e:
                    logger.warning(f"Failed to load short selling data: {e}")

            if pos_path and pos_path.exists():
                try:
                    positions_df = pl.read_parquet(pos_path)
                    logger.info(f"Loaded short positions data from: {pos_path}")
                except Exception as e:
                    logger.warning(f"Failed to load short positions data: {e}")

            # If missing and jquants available, fetch from API for the requested range
            if (short_df is None or short_df.is_empty()) and jquants:
                try:
                    import os

                    from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher  # type: ignore

                    email = os.getenv("JQUANTS_AUTH_EMAIL", "")
                    password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
                    if email and password:
                        fetcher3 = JQuantsAsyncFetcher(email, password)
                        async with aiohttp.ClientSession() as session:
                            await fetcher3.authenticate(session)
                            logger.info(f"Fetching short selling data {start_date} → {end_date} for enrichment")

                            # Fetch both ratio and positions data
                            ss_df = await fetcher3.get_short_selling(session, start_date, end_date)
                            if ss_df is not None and not ss_df.is_empty():
                                short_df = ss_df
                                # Save for reuse
                                try:
                                    out = output_dir / f"short_selling_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                                    ss_df.write_parquet(out)
                                    logger.info(f"Saved short selling parquet: {out}")
                                except Exception:
                                    pass

                            if positions_df is None:
                                pos_df = await fetcher3.get_short_selling_positions(session, start_date, end_date)
                                if pos_df is not None and not pos_df.is_empty():
                                    positions_df = pos_df
                                    # Save for reuse
                                    try:
                                        out = output_dir / f"short_positions_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                                        pos_df.write_parquet(out)
                                        logger.info(f"Saved short positions parquet: {out}")
                                    except Exception:
                                        pass
                except Exception as e:
                    logger.warning(f"Short selling API fetch failed: {e}")

            # Apply short selling features if we have data
            if short_df is not None and not short_df.is_empty():
                # Compute ADV20 data if needed for scaling
                adv20_df = None
                if "AdjustmentVolume" in df.columns:
                    try:
                        adv20_df = (
                            df.select(["Code", "Date", "AdjustmentVolume"])
                            .with_columns([
                                pl.col("AdjustmentVolume")
                                .rolling_mean(adv_window_days)
                                .over("Code")
                                .alias("ADV20_shares")
                            ])
                            .drop_nulls(subset=["ADV20_shares"])
                        )
                        logger.info(f"Computed ADV{adv_window_days} for short selling scaling")
                    except Exception as e:
                        logger.warning(f"ADV computation for short selling failed: {e}")

                # Import and apply short selling features
                from src.gogooku3.features.short_selling import add_short_selling_block

                df = add_short_selling_block(
                    quotes=df,
                    short_df=short_df,
                    positions_df=positions_df,
                    adv20_df=adv20_df,
                    enable_z_scores=True,
                    z_window=short_selling_z_window,
                )

                logger.info("Short selling features attached successfully")
            else:
                logger.info("Short selling requested but no parquet provided/found; skipping")
        else:
            logger.info("Short selling not enabled and no parquet found; skipping")
    except Exception as e:
        logger.warning(f"Short selling attach skipped: {e}")

    # ========================================================================
    # Earnings Events Integration (NEW)
    # ========================================================================
    try:
        earnings_path = None

        if earnings_announcements_parquet and Path(earnings_announcements_parquet).exists():
            earnings_path = earnings_announcements_parquet
        else:
            # Auto-discover under output/
            cands = sorted(output_dir.glob("earnings_announcements_*.parquet"))
            if cands:
                earnings_path = cands[-1]

        if enable_earnings_events or (earnings_path and earnings_path.exists()):
            # Load earnings announcement data
            announcements_df = None

            if earnings_path and earnings_path.exists():
                try:
                    announcements_df = pl.read_parquet(earnings_path)
                    logger.info(f"Loaded earnings announcements from: {earnings_path}")
                except Exception as e:
                    logger.warning(f"Failed to load earnings announcements: {e}")

            # If missing and jquants available, fetch from API for the requested range
            if (announcements_df is None or announcements_df.is_empty()) and jquants:
                try:
                    import os

                    from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher  # type: ignore

                    email = os.getenv("JQUANTS_AUTH_EMAIL", "")
                    password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
                    if email and password:
                        fetcher4 = JQuantsAsyncFetcher(email, password)
                        async with aiohttp.ClientSession() as session:
                            await fetcher4.authenticate(session)
                            logger.info(f"Fetching earnings announcements {start_date} → {end_date} for enrichment")

                            # Fetch earnings announcement data
                            ea_df = await fetcher4.get_earnings_announcements(session, start_date, end_date)
                            if ea_df is not None and not ea_df.is_empty():
                                announcements_df = ea_df
                                # Save for reuse
                                try:
                                    out = (
                                        output_dir
                                        / f"earnings_announcements_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                                    )
                                    ea_df.write_parquet(out)
                                    logger.info(f"Saved earnings announcements parquet: {out}")
                                except Exception:
                                    pass
                except Exception as e:
                    logger.warning(f"Earnings announcements API fetch failed: {e}")

            # Apply earnings event features if we have data
            if announcements_df is not None and not announcements_df.is_empty():
                # Import and apply earnings event features
                from src.gogooku3.features.earnings_events import add_earnings_event_block

                # Use statements data if available for EPS growth
                statements_df = None
                if statements_parquet and Path(statements_parquet).exists():
                    try:
                        statements_df = pl.read_parquet(statements_parquet)
                        logger.info("Using statements data for EPS growth features")
                    except Exception as e:
                        logger.warning(f"Failed to load statements for earnings features: {e}")

                df = add_earnings_event_block(
                    quotes=df,
                    announcement_df=announcements_df,
                    statements_df=statements_df,
                    enable_pead=enable_pead_features,
                    enable_volatility=True,
                )

                logger.info("Earnings event features attached successfully")
            else:
                logger.info("Earnings events requested but no data available; adding null features")
                # Add null earnings features for consistency
                from src.gogooku3.features.earnings_events import add_earnings_event_block
                df = add_earnings_event_block(
                    quotes=df,
                    announcement_df=None,
                    statements_df=None,
                    enable_pead=enable_pead_features,
                    enable_volatility=True,
                )
        else:
            logger.info("Earnings events not enabled; skipping")
    except Exception as e:
        logger.warning(f"Earnings events attach skipped: {e}")

    # ========================================================================
    # Sector-wise Short Selling Integration (NEW)
    # ========================================================================
    try:
        sector_short_path = None

        if sector_short_selling_parquet and Path(sector_short_selling_parquet).exists():
            sector_short_path = sector_short_selling_parquet
        else:
            # Auto-discover under output/
            cands = sorted(output_dir.glob("sector_short_selling_*.parquet"))
            if cands:
                sector_short_path = cands[-1]

        if enable_sector_short_selling or (sector_short_path and sector_short_path.exists()):
            # Load sector short selling data
            sector_short_df = None

            if sector_short_path and sector_short_path.exists():
                try:
                    sector_short_df = pl.read_parquet(sector_short_path)
                    logger.info(f"Loaded sector short selling data from: {sector_short_path}")
                except Exception as e:
                    logger.warning(f"Failed to load sector short selling data: {e}")

            # If missing and jquants available, fetch from API for the requested range
            if (sector_short_df is None or sector_short_df.is_empty()) and jquants:
                try:
                    import os

                    from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher  # type: ignore

                    email = os.getenv("JQUANTS_AUTH_EMAIL", "")
                    password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
                    if email and password:
                        fetcher5 = JQuantsAsyncFetcher(email, password)
                        async with aiohttp.ClientSession() as session:
                            await fetcher5.authenticate(session)
                            logger.info(f"Fetching sector short selling data {start_date} → {end_date} for enrichment")

                            # Fetch sector short selling data
                            ss_df = await fetcher5.get_sector_short_selling(session, start_date, end_date)
                            if ss_df is not None and not ss_df.is_empty():
                                sector_short_df = ss_df
                                # Save for reuse
                                try:
                                    out = (
                                        output_dir
                                        / f"sector_short_selling_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                                    )
                                    ss_df.write_parquet(out)
                                    logger.info(f"Saved sector short selling parquet: {out}")
                                except Exception:
                                    pass
                except Exception as e:
                    logger.warning(f"Sector short selling API fetch failed: {e}")

            # Apply sector short selling features if we have data
            if sector_short_df is not None and not sector_short_df.is_empty():
                # Import and apply sector short selling features
                from src.gogooku3.features.short_selling_sector import add_sector_short_selling_block

                df = add_sector_short_selling_block(
                    quotes=df,
                    ss_df=sector_short_df,
                    listed_info_df=listed_info_df,
                    enable_z_scores=enable_sector_short_z_scores,
                    enable_relative_features=True,
                )

                logger.info("Sector short selling features attached successfully")
            else:
                logger.info("Sector short selling requested but no data available; adding null features")
                # Add null sector short selling features for consistency
                from src.gogooku3.features.short_selling_sector import add_sector_short_selling_block
                df = add_sector_short_selling_block(
                    quotes=df,
                    ss_df=None,
                    listed_info_df=listed_info_df,
                    enable_z_scores=enable_sector_short_z_scores,
                    enable_relative_features=True,
                )
        else:
            logger.info("Sector short selling not enabled; skipping")
    except Exception as e:
        logger.warning(f"Sector short selling attach skipped: {e}")

    # ========================================================================
    # Nikkei225 Index Option Market Aggregates (T+1 attach)
    # ========================================================================
    try:
        if enable_option_market_features:
            opt_feats_df = None
            # 1) Prefer provided features parquet
            if index_option_features_parquet and Path(index_option_features_parquet).exists():
                try:
                    opt_feats_df = pl.read_parquet(index_option_features_parquet)
                    logger.info(f"Loaded index option features from: {index_option_features_parquet}")
                except Exception as e:
                    logger.warning(f"Failed to load option features parquet: {e}")

            # 2) Else, if raw parquet provided, build features
            if (opt_feats_df is None or opt_feats_df.is_empty()) and index_option_raw_parquet and Path(index_option_raw_parquet).exists():
                try:
                    raw = pl.read_parquet(index_option_raw_parquet)
                    from src.gogooku3.features.index_option import build_index_option_features

                    opt_feats_df = build_index_option_features(raw)
                    logger.info("Built option features from raw parquet")
                except Exception as e:
                    logger.warning(f"Failed to build option features from raw: {e}")

            # 3) Else, try API fetch and build features
            if (opt_feats_df is None or opt_feats_df.is_empty()) and jquants:
                try:
                    import os
                    from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher  # type: ignore

                    email = os.getenv("JQUANTS_AUTH_EMAIL", "")
                    password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
                    if email and password:
                        fetcher = JQuantsAsyncFetcher(email, password)
                        async with aiohttp.ClientSession() as session:
                            await fetcher.authenticate(session)
                            logger.info(f"Fetching index options {start_date} → {end_date} for market aggregates")
                            raw = await fetcher.get_index_option(session, start_date, end_date)
                            if raw is not None and not raw.is_empty():
                                from src.gogooku3.features.index_option import build_index_option_features

                                opt_feats_df = build_index_option_features(raw)
                except Exception as e:
                    logger.warning(f"Index option API fetch failed: {e}")

            if opt_feats_df is not None and not opt_feats_df.is_empty():
                try:
                    from src.gogooku3.features.futures_features import build_next_bday_expr_from_quotes
                    from src.gogooku3.features.index_option import (
                        build_option_market_aggregates,
                        attach_option_market_to_equity,
                    )

                    nb = build_next_bday_expr_from_quotes(df)
                    mkt = build_option_market_aggregates(opt_feats_df, next_bday_expr=nb)
                    df = attach_option_market_to_equity(df, mkt)
                    logger.info("Option market aggregates attached (opt_iv_cmat_*, opt_term_slope_30_60, flows)")
                except Exception as e:
                    logger.warning(f"Failed to attach option market aggregates: {e}")
            else:
                logger.info("Option market features requested but no data available; skipping attach")
        else:
            logger.info("Option market features not enabled; skipping")
    except Exception as e:
        logger.warning(f"Index option market aggregates step skipped: {e}")

    # Assurance: guarantee mkt_*
    if not any(c.startswith("mkt_") for c in df.columns):
        logger.warning("mkt_* missing; attempting offline attach (assurance)...")
        topo = topix_parquet if topix_parquet else _find_latest("topix_history_*.parquet")
        # Fetch and save a TOPIX parquet if still missing and jquants is enabled
        if (topo is None or not Path(topo).exists()) and jquants:
            try:
                import os

                import aiohttp

                from src.gogooku3.components.jquants_async_fetcher import (
                    JQuantsAsyncFetcher,  # type: ignore
                )
                email = os.getenv("JQUANTS_AUTH_EMAIL", "")
                password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
                if email and password:
                    fetcher = JQuantsAsyncFetcher(email, password)
                    async with aiohttp.ClientSession() as session:
                        await fetcher.authenticate(session)
                        topo_df = await fetcher.fetch_topix_data(session, start_date, end_date)
                        if topo_df is not None and not topo_df.is_empty():
                            topo = output_dir / f"topix_history_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                            try:
                                import pyarrow.parquet as pq
                                table = topo_df.to_arrow()
                                meta = (table.schema.metadata or {}) | {
                                    b"start_date": start_date.encode(),
                                    b"end_date": end_date.encode(),
                                    b"generator": b"full_dataset.py",
                                }
                                table = table.replace_schema_metadata(meta)
                                pq.write_table(table, str(topo))
                            except Exception:
                                topo_df.write_parquet(topo)
                            logger.info(f"Saved TOPIX parquet for reuse: {topo}")
            except Exception as e:
                logger.warning(f"Assurance TOPIX fetch failed: {e}")
        if topo and Path(topo).exists():
            try:
                topo_df2 = pl.read_parquet(topo)
                df = builder.add_topix_features(df, topix_df=topo_df2)
                try:
                    df = builder.finalize_for_spec(df)
                except Exception:
                    pass
                ok = any(c.startswith("mkt_") for c in df.columns)
                logger.info(f"Offline TOPIX attach {'succeeded' if ok else 'failed'}")
            except Exception as e:
                logger.warning(f"Offline TOPIX attach failed: {e}")

    # Advanced volatility (Yang–Zhang + VoV)
    try:
        if enable_advanced_vol:
            from src.gogooku3.features.advanced_volatility import add_advanced_vol_block
            wins = tuple(int(w) for w in (adv_vol_windows or [20, 60]) if int(w) > 1)
            if wins:
                df = add_advanced_vol_block(df, windows=wins, shift_to_next_day=True)
                logger.info(f"Advanced volatility attached (windows={wins})")
            else:
                logger.info("Advanced volatility requested but windows empty; skipping")
    except Exception as e:
        logger.warning(f"Advanced volatility attach skipped: {e}")

    # Align to DATASET.md (strict schema) just before saving
    try:

        eps = 1e-12

        # Canonical, ordered schema from docs/DATASET.md
        DOC_COLUMNS: list[str] = [
            # 0) Identifiers/Meta (6)
            "Code", "Date", "Section", "section_norm", "row_idx", "shares_outstanding",
            # 1.1 OHLCV (6)
            "Open", "High", "Low", "Close", "Volume", "TurnoverValue",
            # 1.2 Returns (6) + log returns (4)
            "returns_1d", "returns_5d", "returns_10d", "returns_20d", "returns_60d", "returns_120d",
            "log_returns_1d", "log_returns_5d", "log_returns_10d", "log_returns_20d",
            # 1.3 Volatility (5)
            "volatility_5d", "volatility_10d", "volatility_20d", "volatility_60d", "realized_volatility",
            # 1.4 SMA/EMA (10)
            "sma_5", "sma_10", "sma_20", "sma_60", "sma_120",
            "ema_5", "ema_10", "ema_20", "ema_60", "ema_200",
            # 1.5 Price position/gaps (8)
            "price_to_sma5", "price_to_sma20", "price_to_sma60",
            "ma_gap_5_20", "ma_gap_20_60", "high_low_ratio", "close_to_high", "close_to_low",
            # 1.6 Volume/turnover (6)
            "volume_ma_5", "volume_ma_20", "volume_ratio_5", "volume_ratio_20", "turnover_rate", "dollar_volume",
            # 1.7 Technical (approx 10)
            "rsi_2", "rsi_14", "macd", "macd_signal", "macd_histogram", "atr_14", "adx_14", "stoch_k", "bb_width", "bb_position",
            # 2) TOPIX (26)
            "mkt_ret_1d","mkt_ret_5d","mkt_ret_10d","mkt_ret_20d",
            "mkt_ema_5","mkt_ema_20","mkt_ema_60","mkt_ema_200",
            "mkt_dev_20","mkt_gap_5_20","mkt_ema20_slope_3",
            "mkt_vol_20d","mkt_atr_14","mkt_natr_14",
            "mkt_bb_pct_b","mkt_bb_bw",
            "mkt_dd_from_peak","mkt_big_move_flag",
            "mkt_ret_1d_z","mkt_vol_20d_z","mkt_bb_bw_z","mkt_dd_from_peak_z",
            "mkt_bull_200","mkt_trend_up","mkt_high_vol","mkt_squeeze",
            # 3) Cross (8)
            "beta_60d","alpha_1d","alpha_5d","rel_strength_5d","trend_align_mkt","alpha_vs_regime","idio_vol_ratio","beta_stability_60d",
            # 4) Flow (13 enumerated in docs)
            "flow_foreign_net_ratio","flow_individual_net_ratio","flow_activity_ratio","foreign_share_activity","breadth_pos",
            "flow_foreign_net_z","flow_individual_net_z","flow_activity_z",
            "flow_smart_idx","flow_smart_mom4","flow_shock_flag",
            "flow_impulse","flow_days_since",
            # 5) Statements (17)
            "stmt_yoy_sales","stmt_yoy_op","stmt_yoy_np","stmt_opm","stmt_npm","stmt_progress_op","stmt_progress_np",
            "stmt_rev_fore_op","stmt_rev_fore_np","stmt_rev_fore_eps","stmt_rev_div_fore","stmt_roe","stmt_roa",
            "stmt_change_in_est","stmt_nc_flag","stmt_imp_statement","stmt_days_since_statement",
            # 6) Flags (8)
            "is_rsi2_valid","is_ema5_valid","is_ema10_valid","is_ema20_valid","is_ema200_valid","is_valid_ma","is_flow_valid","is_stmt_valid",
            # 7) Optional: Margin weekly block (kept if present; not added as Nulls)
            "margin_long_tot","margin_short_tot","margin_total_gross",
            "margin_credit_ratio","margin_imbalance",
            "margin_d_long_wow","margin_d_short_wow","margin_d_net_wow","margin_d_ratio_wow",
            "long_z52","short_z52","margin_gross_z52","ratio_z52",
            "margin_long_to_adv20","margin_short_to_adv20","margin_d_long_to_adv20","margin_d_short_to_adv20",
            "margin_impulse","margin_days_since","is_margin_valid","margin_issue_type","is_borrowable",
            # 7) Targets (7)
            "target_1d","target_5d","target_10d","target_20d","target_1d_binary","target_5d_binary","target_10d_binary",
            # 8) Forward Return Labels (4) - CRITICAL for supervised learning
            "feat_ret_1d","feat_ret_5d","feat_ret_10d","feat_ret_20d",
        ]

        # Rename any alternate technical names to docs naming
        rename_map = {
            # Bollinger naming
            "bb_bandwidth": "bb_width",
            "bb_pct_b": "bb_position",
            # Realized volatility (Parkinson 20d)
            "realized_vol_20": "realized_volatility",
            # Volume MAs / ratios common aliases
            "vol_ma_5": "volume_ma_5",
            "vol_ma_20": "volume_ma_20",
            "vol_ratio_5d": "volume_ratio_5",
            "vol_ratio_20d": "volume_ratio_20",
            # Valid flags legacy naming
            "is_ema_5_valid": "is_ema5_valid",
            "is_ema_10_valid": "is_ema10_valid",
            "is_ema_20_valid": "is_ema20_valid",
            "is_ema_60_valid": "is_ema60_valid",
            "is_ema_200_valid": "is_ema200_valid",
            "is_rsi_2_valid": "is_rsi2_valid",
        }
        to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        if to_rename:
            df = df.rename(to_rename)

        # Compute absolutely-required missing features (so they are not null-only)
        # Returns (1d/5d) often dropped accidentally by earlier selection
        need_returns = [c for c in ["returns_1d","returns_5d"] if c not in df.columns]
        if need_returns:
            df = df.with_columns([
                pl.col("Close").pct_change().over("Code").alias("returns_1d") if "returns_1d" in need_returns else pl.lit(0),
                pl.col("Close").pct_change(5).over("Code").alias("returns_5d") if "returns_5d" in need_returns else pl.lit(0),
            ])
            # Drop filler 0 columns that are not needed (the lit(0) placeholders)
            drop_tmp = [c for c in ["literal"] if c in df.columns]
            if drop_tmp:
                df = df.drop(drop_tmp)

        # Extend returns if missing
        add_more_returns: list[pl.Expr] = []
        if "returns_10d" not in df.columns:
            add_more_returns.append(pl.col("Close").pct_change(10).over("Code").alias("returns_10d"))
        if "returns_20d" not in df.columns:
            add_more_returns.append(pl.col("Close").pct_change(20).over("Code").alias("returns_20d"))
        if "returns_60d" not in df.columns:
            add_more_returns.append(pl.col("Close").pct_change(60).over("Code").alias("returns_60d"))
        if "returns_120d" not in df.columns:
            add_more_returns.append(pl.col("Close").pct_change(120).over("Code").alias("returns_120d"))
        if "log_returns_1d" not in df.columns:
            add_more_returns.append((pl.col("Close").log() - pl.col("Close").log().shift(1).over("Code")).alias("log_returns_1d"))
        if "log_returns_5d" not in df.columns:
            add_more_returns.append((pl.col("Close").log() - pl.col("Close").log().shift(5).over("Code")).alias("log_returns_5d"))
        if "log_returns_10d" not in df.columns:
            add_more_returns.append((pl.col("Close").log() - pl.col("Close").log().shift(10).over("Code")).alias("log_returns_10d"))
        if "log_returns_20d" not in df.columns:
            add_more_returns.append((pl.col("Close").log() - pl.col("Close").log().shift(20).over("Code")).alias("log_returns_20d"))
        if add_more_returns:
            df = df.with_columns(add_more_returns)

        # EMAs and MA gaps
        ema_needed = [c for c in ["ema_5","ema_10","ema_20","ema_60","ema_200"] if c not in df.columns]
        if ema_needed:
            df = df.with_columns([
                pl.col("Close").ewm_mean(span=5, adjust=False, ignore_nulls=True).over("Code").alias("ema_5") if "ema_5" in ema_needed else pl.lit(0),
                pl.col("Close").ewm_mean(span=10, adjust=False, ignore_nulls=True).over("Code").alias("ema_10") if "ema_10" in ema_needed else pl.lit(0),
                pl.col("Close").ewm_mean(span=20, adjust=False, ignore_nulls=True).over("Code").alias("ema_20") if "ema_20" in ema_needed else pl.lit(0),
                pl.col("Close").ewm_mean(span=60, adjust=False, ignore_nulls=True).over("Code").alias("ema_60") if "ema_60" in ema_needed else pl.lit(0),
                pl.col("Close").ewm_mean(span=200, adjust=False, ignore_nulls=True).over("Code").alias("ema_200") if "ema_200" in ema_needed else pl.lit(0),
            ])
            # Remove any lit(0) placeholders that might have been added
            for col in df.columns:
                if col.startswith("literal"):
                    df = df.drop(col)

        if "ma_gap_5_20" not in df.columns and all(c in df.columns for c in ["ema_5","ema_20"]):
            df = df.with_columns(((pl.col("ema_5") - pl.col("ema_20")) / (pl.col("ema_20") + eps)).alias("ma_gap_5_20"))
        if "ma_gap_20_60" not in df.columns and all(c in df.columns for c in ["ema_20","ema_60"]):
            df = df.with_columns(((pl.col("ema_20") - pl.col("ema_60")) / (pl.col("ema_60") + eps)).alias("ma_gap_20_60"))

        # SMAs and price positions
        sma_spans = [5, 10, 20, 60, 120]
        sma_exprs: list[pl.Expr] = []
        for w in sma_spans:
            name = f"sma_{w}"
            if name not in df.columns:
                sma_exprs.append(pl.col("Close").rolling_mean(window_size=w, min_periods=w).over("Code").alias(name))
        if sma_exprs:
            df = df.with_columns(sma_exprs)
        if "price_to_sma5" not in df.columns and "sma_5" in df.columns:
            df = df.with_columns((pl.col("Close") / (pl.col("sma_5") + eps)).alias("price_to_sma5"))
        if "price_to_sma20" not in df.columns and "sma_20" in df.columns:
            df = df.with_columns((pl.col("Close") / (pl.col("sma_20") + eps)).alias("price_to_sma20"))
        if "price_to_sma60" not in df.columns and "sma_60" in df.columns:
            df = df.with_columns((pl.col("Close") / (pl.col("sma_60") + eps)).alias("price_to_sma60"))

        # Range/position metrics
        if "high_low_ratio" not in df.columns and all(c in df.columns for c in ["High","Low"]):
            df = df.with_columns((pl.col("High") / (pl.col("Low") + eps)).alias("high_low_ratio"))
        if "close_to_high" not in df.columns and all(c in df.columns for c in ["High","Low","Close"]):
            df = df.with_columns(((pl.col("High") - pl.col("Close")) / ((pl.col("High") - pl.col("Low")) + eps)).alias("close_to_high"))
        if "close_to_low" not in df.columns and all(c in df.columns for c in ["High","Low","Close"]):
            df = df.with_columns(((pl.col("Close") - pl.col("Low")) / ((pl.col("High") - pl.col("Low")) + eps)).alias("close_to_low"))

        # Volume moving averages and ratios
        if "volume_ma_5" not in df.columns:
            df = df.with_columns(pl.col("Volume").rolling_mean(window_size=5, min_periods=5).over("Code").alias("volume_ma_5"))
        if "volume_ma_20" not in df.columns:
            df = df.with_columns(pl.col("Volume").rolling_mean(window_size=20, min_periods=20).over("Code").alias("volume_ma_20"))
        if "volume_ratio_5" not in df.columns and "volume_ma_5" in df.columns:
            df = df.with_columns((pl.col("Volume") / (pl.col("volume_ma_5") + eps)).alias("volume_ratio_5"))
        if "volume_ratio_20" not in df.columns and "volume_ma_20" in df.columns:
            df = df.with_columns((pl.col("Volume") / (pl.col("volume_ma_20") + eps)).alias("volume_ratio_20"))

        # Dollar volume
        if "dollar_volume" not in df.columns:
            df = df.with_columns((pl.col("Close") * pl.col("Volume")).alias("dollar_volume"))

        # Volatilities
        if "volatility_5d" not in df.columns and "returns_1d" in df.columns:
            df = df.with_columns(pl.col("returns_1d").rolling_std(window_size=5, min_periods=5).over("Code").map_elements(lambda x: x * (252 ** 0.5) if x is not None else None).alias("volatility_5d"))
        if "volatility_10d" not in df.columns and "returns_1d" in df.columns:
            df = df.with_columns(pl.col("returns_1d").rolling_std(window_size=10, min_periods=10).over("Code").map_elements(lambda x: x * (252 ** 0.5) if x is not None else None).alias("volatility_10d"))
        if "volatility_60d" not in df.columns and "returns_1d" in df.columns:
            df = df.with_columns(pl.col("returns_1d").rolling_std(window_size=60, min_periods=60).over("Code").map_elements(lambda x: x * (252 ** 0.5) if x is not None else None).alias("volatility_60d"))

        # Reconstruct MACD line if signal+histogram exist
        if "macd" not in df.columns and all(c in df.columns for c in ["macd_signal","macd_histogram"]):
            df = df.with_columns((pl.col("macd_signal") + pl.col("macd_histogram")).alias("macd"))

        # ATR(14) and Stochastic %K (14)
        if "atr_14" not in df.columns and all(c in df.columns for c in ["High","Low","Close"]):
            tr = pl.max_horizontal([
                pl.col("High") - pl.col("Low"),
                (pl.col("High") - pl.col("Close").shift(1).over("Code")).abs(),
                (pl.col("Low") - pl.col("Close").shift(1).over("Code")).abs(),
            ])
            df = df.with_columns(tr.alias("_tr"))
            df = df.with_columns(pl.col("_tr").ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code").alias("atr_14")).drop("_tr")
        if "stoch_k" not in df.columns and all(c in df.columns for c in ["High","Low","Close"]):
            ll = pl.col("Low").rolling_min(window_size=14, min_periods=14).over("Code")
            hh = pl.col("High").rolling_max(window_size=14, min_periods=14).over("Code")
            df = df.with_columns(((pl.col("Close") - ll) / ((hh - ll) + eps) * 100.0).alias("stoch_k"))

        # ADX(14) (simplified Wilder smoothing using EWM span)
        if "adx_14" not in df.columns and all(c in df.columns for c in ["High","Low","Close"]):
            up_move = (pl.col("High") - pl.col("High").shift(1).over("Code")).clip_min(0)
            down_move = (pl.col("Low").shift(1).over("Code") - pl.col("Low")).clip_min(0)
            plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
            minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)
            tr2 = pl.max_horizontal([
                pl.col("High") - pl.col("Low"),
                (pl.col("High") - pl.col("Close").shift(1).over("Code")).abs(),
                (pl.col("Low") - pl.col("Close").shift(1).over("Code")).abs(),
            ])
            atr14 = tr2.ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code")
            plus_di = (plus_dm.ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code") / (atr14 + eps)) * 100.0
            minus_di = (minus_dm.ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code") / (atr14 + eps)) * 100.0
            dx = ((plus_di - minus_di).abs() / ((plus_di + minus_di) + eps)) * 100.0
            adx14 = dx.ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code")
            df = df.with_columns(adx14.alias("adx_14"))

        # Section normalization fallback
        if "section_norm" not in df.columns and "Section" in df.columns:
            df = df.with_columns(pl.col("Section").alias("section_norm"))

        # Validity flags fallback
        if "row_idx" in df.columns:
            if "is_rsi2_valid" not in df.columns:
                df = df.with_columns((pl.col("row_idx") >= 5).cast(pl.Int8).alias("is_rsi2_valid"))
            if "is_valid_ma" not in df.columns:
                df = df.with_columns((pl.col("row_idx") >= 60).cast(pl.Int8).alias("is_valid_ma"))
        # Statement validity fallback
        if "is_stmt_valid" not in df.columns and "stmt_days_since_statement" in df.columns:
            df = df.with_columns((pl.col("stmt_days_since_statement") >= 0).cast(pl.Int8).alias("is_stmt_valid"))

        # Add any missing spec columns as nulls (safe defaults)
        existing = set(df.columns)
        # Keep optional blocks (e.g., margin_*) truly optional: don't add Nulls for them
        to_add_nulls = [
            c for c in DOC_COLUMNS if c not in existing and not c.startswith("margin_")
        ]
        if to_add_nulls:
            df = df.with_columns([pl.lit(None).alias(c) for c in to_add_nulls])

        # Fill conservative defaults for statement flags when statements are absent
        fill_zero_flags = []
        for c in ["stmt_change_in_est", "stmt_nc_flag", "is_stmt_valid"]:
            if c in df.columns:
                fill_zero_flags.append(pl.col(c).fill_null(0).cast(pl.Int8).alias(c))
        if fill_zero_flags:
            df = df.with_columns(fill_zero_flags)

        # Cast common boolean/int flags to Int8 for stable schema
        flag_like = [
            c for c in df.columns if c.startswith("is_") or c.endswith("_impulse")
        ]
        if flag_like:
            # Handle mixed bool/int types by first converting booleans to integers
            cast_exprs = []
            for c in flag_like:
                if c in df.columns:
                    # Convert boolean True/False to 1/0, then cast to Int8
                    cast_exprs.append(
                        pl.when(pl.col(c).is_null()).then(None)
                        .when(pl.col(c).is_boolean()).then(pl.col(c).cast(pl.Int32))
                        .otherwise(pl.col(c))
                        .cast(pl.Int8).alias(c)
                    )
            if cast_exprs:
                df = df.with_columns(cast_exprs)

        # Finally, project to the exact schema (drops all non-spec columns)
        keep_cols = [c for c in DOC_COLUMNS if c in df.columns]
        df = df.select(keep_cols)
        logger.info(f"Aligned dataset to DATASET.md exact schema (n={len(keep_cols)})")
    except Exception as _e:
        logger.warning(f"DATASET.md strict alignment skipped: {_e}")

    # Ensure (Code, Date) uniqueness (keep last occurrence)
    try:
        df = df.unique(subset=["Code", "Date"], keep="last")
        logger.info("De-duplicated (Code, Date) pairs with keep=last")
    except Exception:
        pass

    # Stable ordering: sort by (Code, Date) before saving so downstream
    # tools that rely on windowed operations (e.g., shift over Code) are
    # evaluated against a canonical ordering.
    try:
        if {"Code", "Date"}.issubset(df.columns):
            df = df.sort(["Code", "Date"])  # type: ignore[arg-type]
            logger.info("Sorted dataset by (Code, Date) prior to save")
    except Exception:
        pass

    pq_path, meta_path = save_with_symlinks(df, output_dir, tag="full", start_date=start_date, end_date=end_date)
    return pq_path, meta_path
