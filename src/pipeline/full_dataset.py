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
    from scripts.data.ml_dataset_builder import MLDatasetBuilder
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
    sector_onehot33: bool = False,
    sector_series_mcap: str = "auto",
    sector_te_targets: list[str] | None = None,
    sector_series_levels: list[str] | None = None,
    sector_te_levels: list[str] | None = None,
) -> tuple[Path, Path]:
    """Attach TOPIX + statements + flow then save with symlinks.

    Includes an assurance step to guarantee mkt_* presence by discovering
    or fetching TOPIX parquet when needed.
    """
    import aiohttp

    from scripts.data.ml_dataset_builder import MLDatasetBuilder

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

            from scripts._archive.run_pipeline import (
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
                import polars as _pl
                for tcol in te_targets:
                    src = f"te33_sec_{tcol}"
                    dst = f"te_sec_{tcol}"
                    if src in df.columns and dst not in df.columns:
                        df = df.with_columns(_pl.col(src).alias(dst))
            logger.info("Sector enrichment completed (sector33/MarketCode/CompanyName)")
        except Exception as e:
            logger.warning(f"Failed sector enrichment: {e}")

    # Flow attach
    if trades_spec_path and Path(trades_spec_path).exists():
        try:
            trades_spec_df = pl.read_parquet(trades_spec_path)
            # Pass listed_info_df (if available) for accurate Section mapping
            df = builder.add_flow_features(df, trades_spec_df, listed_info_df=listed_info_df)
        except Exception as e:
            logger.warning(f"Skipping flow enrichment due to error: {e}")

    # Assurance: guarantee mkt_*
    if not any(c.startswith("mkt_") for c in df.columns):
        logger.warning("mkt_* missing; attempting offline attach (assurance)...")
        topo = topix_parquet if topix_parquet else _find_latest("topix_history_*.parquet")
        # Fetch and save a TOPIX parquet if still missing and jquants is enabled
        if (topo is None or not Path(topo).exists()) and jquants:
            try:
                import os

                import aiohttp

                from scripts._archive.run_pipeline import (
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

    # Align to DATASET.md (strict schema) just before saving
    try:

        import polars as _pl
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
            # 7) Targets (7)
            "target_1d","target_5d","target_10d","target_20d","target_1d_binary","target_5d_binary","target_10d_binary",
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
                _pl.col("Close").pct_change().over("Code").alias("returns_1d") if "returns_1d" in need_returns else _pl.lit(0),
                _pl.col("Close").pct_change(5).over("Code").alias("returns_5d") if "returns_5d" in need_returns else _pl.lit(0),
            ])
            # Drop filler 0 columns that are not needed (the lit(0) placeholders)
            drop_tmp = [c for c in ["literal"] if c in df.columns]
            if drop_tmp:
                df = df.drop(drop_tmp)

        # Extend returns if missing
        add_more_returns: list[_pl.Expr] = []
        if "returns_10d" not in df.columns:
            add_more_returns.append(_pl.col("Close").pct_change(10).over("Code").alias("returns_10d"))
        if "returns_20d" not in df.columns:
            add_more_returns.append(_pl.col("Close").pct_change(20).over("Code").alias("returns_20d"))
        if "returns_60d" not in df.columns:
            add_more_returns.append(_pl.col("Close").pct_change(60).over("Code").alias("returns_60d"))
        if "returns_120d" not in df.columns:
            add_more_returns.append(_pl.col("Close").pct_change(120).over("Code").alias("returns_120d"))
        if "log_returns_1d" not in df.columns:
            add_more_returns.append((_pl.col("Close").log() - _pl.col("Close").log().shift(1).over("Code")).alias("log_returns_1d"))
        if "log_returns_5d" not in df.columns:
            add_more_returns.append((_pl.col("Close").log() - _pl.col("Close").log().shift(5).over("Code")).alias("log_returns_5d"))
        if "log_returns_10d" not in df.columns:
            add_more_returns.append((_pl.col("Close").log() - _pl.col("Close").log().shift(10).over("Code")).alias("log_returns_10d"))
        if "log_returns_20d" not in df.columns:
            add_more_returns.append((_pl.col("Close").log() - _pl.col("Close").log().shift(20).over("Code")).alias("log_returns_20d"))
        if add_more_returns:
            df = df.with_columns(add_more_returns)

        # EMAs and MA gaps
        ema_needed = [c for c in ["ema_5","ema_10","ema_20","ema_60","ema_200"] if c not in df.columns]
        if ema_needed:
            df = df.with_columns([
                _pl.col("Close").ewm_mean(span=5, adjust=False, ignore_nulls=True).over("Code").alias("ema_5") if "ema_5" in ema_needed else _pl.lit(0),
                _pl.col("Close").ewm_mean(span=10, adjust=False, ignore_nulls=True).over("Code").alias("ema_10") if "ema_10" in ema_needed else _pl.lit(0),
                _pl.col("Close").ewm_mean(span=20, adjust=False, ignore_nulls=True).over("Code").alias("ema_20") if "ema_20" in ema_needed else _pl.lit(0),
                _pl.col("Close").ewm_mean(span=60, adjust=False, ignore_nulls=True).over("Code").alias("ema_60") if "ema_60" in ema_needed else _pl.lit(0),
                _pl.col("Close").ewm_mean(span=200, adjust=False, ignore_nulls=True).over("Code").alias("ema_200") if "ema_200" in ema_needed else _pl.lit(0),
            ])
            # Remove any lit(0) placeholders that might have been added
            for col in df.columns:
                if col.startswith("literal"):
                    df = df.drop(col)

        if "ma_gap_5_20" not in df.columns and all(c in df.columns for c in ["ema_5","ema_20"]):
            df = df.with_columns(((_pl.col("ema_5") - _pl.col("ema_20")) / (_pl.col("ema_20") + eps)).alias("ma_gap_5_20"))
        if "ma_gap_20_60" not in df.columns and all(c in df.columns for c in ["ema_20","ema_60"]):
            df = df.with_columns(((_pl.col("ema_20") - _pl.col("ema_60")) / (_pl.col("ema_60") + eps)).alias("ma_gap_20_60"))

        # SMAs and price positions
        sma_spans = [5, 10, 20, 60, 120]
        sma_exprs: list[_pl.Expr] = []
        for w in sma_spans:
            name = f"sma_{w}"
            if name not in df.columns:
                sma_exprs.append(_pl.col("Close").rolling_mean(window_size=w, min_periods=w).over("Code").alias(name))
        if sma_exprs:
            df = df.with_columns(sma_exprs)
        if "price_to_sma5" not in df.columns and "sma_5" in df.columns:
            df = df.with_columns((_pl.col("Close") / (_pl.col("sma_5") + eps)).alias("price_to_sma5"))
        if "price_to_sma20" not in df.columns and "sma_20" in df.columns:
            df = df.with_columns((_pl.col("Close") / (_pl.col("sma_20") + eps)).alias("price_to_sma20"))
        if "price_to_sma60" not in df.columns and "sma_60" in df.columns:
            df = df.with_columns((_pl.col("Close") / (_pl.col("sma_60") + eps)).alias("price_to_sma60"))

        # Range/position metrics
        if "high_low_ratio" not in df.columns and all(c in df.columns for c in ["High","Low"]):
            df = df.with_columns((_pl.col("High") / (_pl.col("Low") + eps)).alias("high_low_ratio"))
        if "close_to_high" not in df.columns and all(c in df.columns for c in ["High","Low","Close"]):
            df = df.with_columns(((_pl.col("High") - _pl.col("Close")) / ((_pl.col("High") - _pl.col("Low")) + eps)).alias("close_to_high"))
        if "close_to_low" not in df.columns and all(c in df.columns for c in ["High","Low","Close"]):
            df = df.with_columns(((_pl.col("Close") - _pl.col("Low")) / ((_pl.col("High") - _pl.col("Low")) + eps)).alias("close_to_low"))

        # Volume moving averages and ratios
        if "volume_ma_5" not in df.columns:
            df = df.with_columns(_pl.col("Volume").rolling_mean(window_size=5, min_periods=5).over("Code").alias("volume_ma_5"))
        if "volume_ma_20" not in df.columns:
            df = df.with_columns(_pl.col("Volume").rolling_mean(window_size=20, min_periods=20).over("Code").alias("volume_ma_20"))
        if "volume_ratio_5" not in df.columns and "volume_ma_5" in df.columns:
            df = df.with_columns((_pl.col("Volume") / (_pl.col("volume_ma_5") + eps)).alias("volume_ratio_5"))
        if "volume_ratio_20" not in df.columns and "volume_ma_20" in df.columns:
            df = df.with_columns((_pl.col("Volume") / (_pl.col("volume_ma_20") + eps)).alias("volume_ratio_20"))

        # Dollar volume
        if "dollar_volume" not in df.columns:
            df = df.with_columns((_pl.col("Close") * _pl.col("Volume")).alias("dollar_volume"))

        # Volatilities
        if "volatility_5d" not in df.columns and "returns_1d" in df.columns:
            df = df.with_columns(_pl.col("returns_1d").rolling_std(window_size=5, min_periods=5).over("Code").map_elements(lambda x: x * (252 ** 0.5) if x is not None else None).alias("volatility_5d"))
        if "volatility_10d" not in df.columns and "returns_1d" in df.columns:
            df = df.with_columns(_pl.col("returns_1d").rolling_std(window_size=10, min_periods=10).over("Code").map_elements(lambda x: x * (252 ** 0.5) if x is not None else None).alias("volatility_10d"))
        if "volatility_60d" not in df.columns and "returns_1d" in df.columns:
            df = df.with_columns(_pl.col("returns_1d").rolling_std(window_size=60, min_periods=60).over("Code").map_elements(lambda x: x * (252 ** 0.5) if x is not None else None).alias("volatility_60d"))

        # Reconstruct MACD line if signal+histogram exist
        if "macd" not in df.columns and all(c in df.columns for c in ["macd_signal","macd_histogram"]):
            df = df.with_columns((_pl.col("macd_signal") + _pl.col("macd_histogram")).alias("macd"))

        # ATR(14) and Stochastic %K (14)
        if "atr_14" not in df.columns and all(c in df.columns for c in ["High","Low","Close"]):
            tr = _pl.max_horizontal([
                _pl.col("High") - _pl.col("Low"),
                (_pl.col("High") - _pl.col("Close").shift(1).over("Code")).abs(),
                (_pl.col("Low") - _pl.col("Close").shift(1).over("Code")).abs(),
            ])
            df = df.with_columns(tr.alias("_tr"))
            df = df.with_columns(_pl.col("_tr").ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code").alias("atr_14")).drop("_tr")
        if "stoch_k" not in df.columns and all(c in df.columns for c in ["High","Low","Close"]):
            ll = _pl.col("Low").rolling_min(window_size=14, min_periods=14).over("Code")
            hh = _pl.col("High").rolling_max(window_size=14, min_periods=14).over("Code")
            df = df.with_columns(((_pl.col("Close") - ll) / ((hh - ll) + eps) * 100.0).alias("stoch_k"))

        # ADX(14) (simplified Wilder smoothing using EWM span)
        if "adx_14" not in df.columns and all(c in df.columns for c in ["High","Low","Close"]):
            up_move = (_pl.col("High") - _pl.col("High").shift(1).over("Code")).clip_min(0)
            down_move = (_pl.col("Low").shift(1).over("Code") - _pl.col("Low")).clip_min(0)
            plus_dm = _pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
            minus_dm = _pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)
            tr2 = _pl.max_horizontal([
                _pl.col("High") - _pl.col("Low"),
                (_pl.col("High") - _pl.col("Close").shift(1).over("Code")).abs(),
                (_pl.col("Low") - _pl.col("Close").shift(1).over("Code")).abs(),
            ])
            atr14 = tr2.ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code")
            plus_di = (plus_dm.ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code") / (atr14 + eps)) * 100.0
            minus_di = (minus_dm.ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code") / (atr14 + eps)) * 100.0
            dx = ((plus_di - minus_di).abs() / ((plus_di + minus_di) + eps)) * 100.0
            adx14 = dx.ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code")
            df = df.with_columns(adx14.alias("adx_14"))

        # Section normalization fallback
        if "section_norm" not in df.columns and "Section" in df.columns:
            df = df.with_columns(_pl.col("Section").alias("section_norm"))

        # Validity flags fallback
        if "row_idx" in df.columns:
            if "is_rsi2_valid" not in df.columns:
                df = df.with_columns((_pl.col("row_idx") >= 5).cast(_pl.Int8).alias("is_rsi2_valid"))
            if "is_valid_ma" not in df.columns:
                df = df.with_columns((_pl.col("row_idx") >= 60).cast(_pl.Int8).alias("is_valid_ma"))
        # Statement validity fallback
        if "is_stmt_valid" not in df.columns and "stmt_days_since_statement" in df.columns:
            df = df.with_columns((_pl.col("stmt_days_since_statement") >= 0).cast(_pl.Int8).alias("is_stmt_valid"))

        # Add any missing spec columns as nulls (safe defaults)
        existing = set(df.columns)
        to_add_nulls = [c for c in DOC_COLUMNS if c not in existing]
        if to_add_nulls:
            df = df.with_columns([_pl.lit(None).alias(c) for c in to_add_nulls])

        # Fill conservative defaults for statement flags when statements are absent
        fill_zero_flags = []
        for c in ["stmt_change_in_est", "stmt_nc_flag", "is_stmt_valid"]:
            if c in df.columns:
                fill_zero_flags.append(_pl.col(c).fill_null(0).cast(_pl.Int8).alias(c))
        if fill_zero_flags:
            df = df.with_columns(fill_zero_flags)

        # Finally, project to the exact schema (drops all non-spec columns)
        keep_cols = [c for c in DOC_COLUMNS if c in df.columns]
        df = df.select(keep_cols)
        logger.info(f"Aligned dataset to DATASET.md exact schema (n={len(keep_cols)})")
    except Exception as _e:
        logger.warning(f"DATASET.md strict alignment skipped: {_e}")

    pq_path, meta_path = save_with_symlinks(df, output_dir, tag="full", start_date=start_date, end_date=end_date)
    return pq_path, meta_path
