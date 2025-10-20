#!/usr/bin/env python3
"""
Full dataset pipeline orchestrator (enrichment + save).

Provides reusable functions so that CLI wrappers under scripts/ remain thin.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import aiohttp
import polars as pl

from src.features.calendar_utils import build_next_bday_expr_from_dates

logger = logging.getLogger(__name__)


def _ensure_code_utf8(df: pl.DataFrame | None, source: str = "") -> pl.DataFrame | None:
    """Cast Code column to Utf8 if present; log where applied."""
    try:
        if df is not None and not df.is_empty() and "Code" in df.columns:
            # Polars 1.x: Use df.schema instead of df["Code"].dtype
            code_dtype = df.schema.get("Code")
            if code_dtype is not None and code_dtype != pl.Utf8:
                df = df.with_columns(pl.col("Code").cast(pl.Utf8).alias("Code"))
                if source:
                    logger.info(f"Normalized Code dtype to Utf8: {source}")
    except Exception as e:
        logger.warning(f"Failed to normalize Code dtype for {source or 'frame'}: {e}")
    return df


def _validate_code_type_consistency(df: pl.DataFrame, data_source: str) -> bool:
    """Validate Code dtype is Utf8; warn if not."""
    try:
        if "Code" in df.columns:
            # Polars 1.x: Use df.schema instead of df["Code"].dtype
            code_type = df.schema.get("Code")
            if code_type is not None and code_type != pl.Utf8:
                logger.warning(
                    f"{data_source}: Code dtype is {code_type}, expected Utf8"
                )
                return False
    except Exception:
        pass
    return True


def save_with_symlinks(
    df: pl.DataFrame,
    output_dir: Path,
    tag: str = "full",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize column names to align with docs/ml/dataset_new.md (non-breaking rename)
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
            meta.update(
                {
                    b"start_date": start_date.encode("utf-8"),
                    b"end_date": end_date.encode("utf-8"),
                    b"generator": b"full_dataset.py",
                }
            )
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
    with open(meta_path, "w", encoding="utf-8") as f:
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

    # Back-compat: also refresh symlinks at the parent (output/) level so
    # legacy tooling that expects output/ml_dataset_latest_full.parquet keeps working.
    root_dir = output_dir.parent
    if root_dir != output_dir:
        root_links = [
            (
                root_dir / f"ml_dataset_latest_{tag}.parquet",
                parquet_path.relative_to(root_dir),
            ),
            (
                root_dir / f"ml_dataset_latest_{tag}_metadata.json",
                meta_path.relative_to(root_dir),
            ),
        ]
        for link, target in root_links:
            try:
                if link.exists() or link.is_symlink():
                    link.unlink()
            except Exception:
                pass
            link.symlink_to(target)

    # GCS sync: automatically upload to cloud storage if enabled
    if os.getenv("GCS_SYNC_AFTER_SAVE") == "1":
        try:
            from src.gogooku3.utils.gcs_storage import upload_to_gcs

            logger.info("GCS sync enabled, uploading dataset and metadata...")
            upload_to_gcs(parquet_path)
            upload_to_gcs(meta_path)
        except Exception as e:
            logger.warning(f"GCS sync failed (non-blocking): {e}")

    return parquet_path, meta_path


def _find_latest(glob: str) -> Path | None:
    """Find the latest matching file anywhere under `output/`.

    The original implementation only searched the top-level `output/` directory,
    which caused auto-discovery to miss files saved under subfolders like
    `output/raw/margin/`. This recursive variant searches the entire tree and
    preserves the existing naming convention where file names embed sortable
    date tokens.
    """
    base = Path("output")
    if not base.exists():
        return None
    # Use rglob to search recursively. Sorted() will order lexicographically;
    # our filenames contain YYYYMMDD tokens so lexical order matches recency.
    cands = sorted(base.rglob(glob))
    return cands[-1] if cands else None


async def enrich_and_save(
    df_base: pl.DataFrame,
    *,
    output_dir: Path,
    jquants: bool,
    start_date: str,
    end_date: str,
    business_days: list[str] | None = None,
    trades_spec_path: Path | None = None,
    topix_parquet: Path | None = None,
    # Multi-index OHLC integration (optional)
    enable_indices: bool = False,
    indices_parquet: Path | None = None,
    indices_codes: list[str] | None = None,
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
    graph_max_k: int = 4,
    graph_cache_dir: str | None = None,
    # Special days handling
    disable_halt_mask: bool = False,
) -> tuple[Path, Path]:
    """Attach TOPIX + statements + flow then save with symlinks.

    Includes an assurance step to guarantee mkt_* presence by discovering
    or fetching TOPIX parquet when needed.
    """

    from src.gogooku3.pipeline.builder import MLDatasetBuilder

    builder = MLDatasetBuilder(output_dir=output_dir)

    def _fill_from_source(
        frame: pl.DataFrame, target: str, source: str
    ) -> pl.DataFrame:
        """Fill nulls in ``target`` using ``source`` (creating the column if missing)."""
        if source not in frame.columns:
            return frame
        if target in frame.columns:
            return frame.with_columns(
                pl.col(target).fill_null(pl.col(source)).alias(target)
            )
        return frame.with_columns(pl.col(source).alias(target))

    # Ensure base quotes have Code as Utf8 before any joins
    result = _ensure_code_utf8(df_base, source="base_quotes")
    df_base = result if result is not None and not result.is_empty() else df_base

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
                    topix_df = await fetcher.fetch_topix_data(
                        session, start_date, end_date
                    )
        except Exception as e:
            logger.warning(f"TOPIX API fetch failed: {e}")
    if topix_df is None:
        # Choose the most suitable local TOPIX parquet by coverage
        try:
            import re

            best_path = None
            best_score = -1
            start_int = int(start_date.replace("-", ""))
            end_int = int(end_date.replace("-", ""))
            for cand in sorted((Path("output")).rglob("topix_history_*.parquet")):
                m = re.search(r"topix_history_(\d{8})_(\d{8})\.parquet$", cand.name)
                if not m:
                    continue
                s, e = int(m.group(1)), int(m.group(2))
                # Score by coverage and whether it encloses [start, end]
                encloses = s <= start_int and e >= end_int
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
        logger.info(
            "Forward return labels (feat_ret_1d, feat_ret_5d, feat_ret_10d, feat_ret_20d) added successfully"
        )

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

    # Finalize to match dataset_new.md spec
    try:
        df = builder.finalize_for_spec(df)
    except Exception:
        pass

    # ----- Optional: indices OHLC fetch/load and attach day-level features -----
    indices_df = None
    if enable_indices:
        # Prefer provided parquet
        if indices_parquet and Path(indices_parquet).exists():
            try:
                indices_df = pl.read_parquet(indices_parquet)
                logger.info(f"Loaded indices from parquet: {indices_parquet}")
            except Exception as e:
                logger.warning(f"Failed to read indices parquet: {e}")
        # Try API if not present and jquants enabled
        if (
            indices_df is None
            and jquants
            and (indices_codes is not None and len(indices_codes) > 0)
        ):
            try:
                import os

                from src.gogooku3.components.jquants_async_fetcher import (
                    JQuantsAsyncFetcher,  # type: ignore
                )

                email = os.getenv("JQUANTS_AUTH_EMAIL", "")
                password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
                if email and password:
                    fetcher_idx = JQuantsAsyncFetcher(email, password)
                    async with aiohttp.ClientSession() as session:
                        await fetcher_idx.authenticate(session)
                        logger.info(
                            f"Fetching indices OHLC {start_date} → {end_date} for {len(indices_codes)} codes"
                        )
                        indices_df = await fetcher_idx.fetch_indices_ohlc(
                            session, start_date, end_date, codes=indices_codes
                        )
                        if indices_df is not None and not indices_df.is_empty():
                            try:
                                out = (
                                    output_dir
                                    / f"indices_history_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                                )
                                indices_df.write_parquet(out)
                                logger.info(f"Saved indices parquet: {out}")
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"Indices API fetch failed: {e}")
        # Auto-discover local indices parquet by coverage
        if indices_df is None:
            try:
                import re

                best_path = None
                best_score = -1
                start_int = int(start_date.replace("-", ""))
                end_int = int(end_date.replace("-", ""))
                for cand in sorted((Path("output")).glob("indices_history_*.parquet")):
                    m = re.search(
                        r"indices_history_(\d{8})_(\d{8})\.parquet$", cand.name
                    )
                    if not m:
                        continue
                    s, e = int(m.group(1)), int(m.group(2))
                    encloses = s <= start_int and e >= end_int
                    coverage = e - s
                    score = coverage + (10000000 if encloses else 0)
                    if score > best_score:
                        best_score = score
                        best_path = cand
                if best_path is not None:
                    indices_df = pl.read_parquet(best_path)
                    logger.info(f"Loaded indices from local: {best_path}")
            except Exception as e:
                logger.warning(f"Indices local selection failed: {e}")

    # Statements attach (refresh when coverage missing)
    stmt_cols = [c for c in df.columns if c.startswith("stmt_")]
    stmt_signal_cols = [
        c
        for c in stmt_cols
        if c in {"stmt_revenue_growth", "stmt_profit_margin", "stmt_roe"}
    ]
    needs_stmt_refresh = True
    if stmt_signal_cols and len(df) > 0:
        try:
            non_null_counts = df.select(
                [pl.col(col).is_not_null().sum().alias(col) for col in stmt_signal_cols]
            ).row(0)
            needs_stmt_refresh = not any(count > 0 for count in non_null_counts)
        except Exception:
            needs_stmt_refresh = True
    if needs_stmt_refresh:
        if stmt_cols:
            df = df.drop(stmt_cols)
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
                    df = builder.add_sector_series(
                        df,
                        level=lvl,
                        windows=(1, 5, 20),
                        series_mcap=sector_series_mcap,
                    )
                except Exception as e:
                    logger.warning(f"Sector series attach failed for level {lvl}: {e}")
            # Sector encodings (one-hot for 17, daily freqs)
            try:
                df = builder.add_sector_encodings(
                    df, onehot_17=True, onehot_33=sector_onehot33, freq_daily=True
                )
            except Exception as e:
                logger.warning(f"Sector encodings failed: {e}")
            # Relative-to-sector features (rel/alpha/z-in-sec)
            try:
                df = builder.add_relative_to_sector(
                    df, level="33", x_cols=("returns_5d", "ma_gap_5_20")
                )
            except Exception as e:
                logger.warning(f"Relative-to-sector features failed: {e}")
            # Sector target encoding with cross-fit + lag
            te_targets = sector_te_targets or ["target_5d"]
            te_levels = sector_te_levels or ["33"]
            for lvl in te_levels:
                for tcol in te_targets:
                    try:
                        df = builder.add_sector_target_encoding(
                            df,
                            target_col=tcol,
                            level=lvl,
                            k_folds=5,
                            lag_days=1,
                            m=100.0,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Sector target encoding failed for level {lvl}, target {tcol}: {e}"
                        )
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

    # Attach indices aggregates if available
    try:
        if enable_indices and indices_df is not None and not indices_df.is_empty():
            df = builder.add_index_features(
                df, indices_df, mask_halt_day=(not disable_halt_mask)
            )
            logger.info("Index day-level features attached (spreads, breadth)")
        elif enable_indices:
            logger.info("Indices enabled but no data found; skipping")
    except Exception as e:
        logger.warning(f"Indices attach skipped: {e}")

    # Sector index join (via listed_info mapping to sector index code)
    try:
        if (
            enable_indices
            and indices_df is not None
            and not indices_df.is_empty()
            and listed_info_df is not None
            and not listed_info_df.is_empty()
        ):
            df = builder.add_sector_index_features(
                df,
                indices_df,
                listed_info_df,
                prefix="sect_",
                mask_halt_day=(not disable_halt_mask),
            )
            logger.info("Sector index features attached (via Sector33 → index mapping)")
    except Exception as e:
        logger.warning(f"Sector index attach skipped: {e}")

    # Sector cross-sectional features (Phase 2): relies on sector column if present
    try:
        if enable_sector_cs:
            from src.gogooku3.features.sector_cross_sectional import (
                add_sector_cross_sectional_features,
            )

            df = add_sector_cross_sectional_features(df, include_cols=sector_cs_cols)
            logger.info(
                "Sector cross-sectional features attached (ret_vs_sec, rank_in_sec, volume/rv20 z in sector)"
            )
        else:
            logger.info("Sector cross-sectional features disabled; skipping")
    except Exception as e:
        logger.warning(f"Sector cross-sectional features attach skipped: {e}")

    # Sector aggregation features (Phase 2.5): sector equal-weighted returns, momentum, volatility
    # CRITICAL: Error stops pipeline to ensure root cause resolution
    from src.gogooku3.features.sector_aggregation import add_sector_aggregation_features

    df = add_sector_aggregation_features(df, min_members=3)
    logger.info(
        "✅ Sector aggregation features attached (sec_ret_eq, sec_mom, sec_vol, rel_to_sec, beta_to_sec) +30 columns"
    )

    # Sector One-Hot encoding (17 industry sectors)
    # CRITICAL: Error stops pipeline to ensure root cause resolution
    if "sector17_id" in df.columns:
        # Generate dummy variables for sector17_id
        onehot_df = (
            df.select(["Code", "Date", "sector17_id"])
            .with_columns([pl.col("sector17_id").cast(pl.Utf8)])
            .to_dummies(columns=["sector17_id"], separator="_")
        )

        # Rename columns to have sec17_onehot prefix
        rename_map = {
            c: c.replace("sector17_id_", "sec17_onehot_")
            for c in onehot_df.columns
            if c.startswith("sector17_id_")
        }
        if rename_map:
            onehot_df = onehot_df.rename(rename_map)

        # Join back to main dataframe
        onehot_cols = [c for c in onehot_df.columns if c.startswith("sec17_onehot_")]
        df = df.join(
            onehot_df.select(["Code", "Date"] + onehot_cols),
            on=["Code", "Date"],
            how="left",
        )
        logger.info(
            f"✅ Sector One-Hot encoding attached: +{len(onehot_cols)} columns (sec17_onehot_*)"
        )
    else:
        logger.warning("sector17_id not found, skipping One-Hot encoding")

    # Window maturity validity flags (Phase 2.6): dataset_new.md Section 10
    # Indicates when rolling window features have enough historical data to be valid
    # CRITICAL: Error stops pipeline to ensure root cause resolution
    # Add row index within each stock's time series
    df = df.with_columns([pl.col("Date").cum_count().over("Code").alias("_row_idx")])

    # Window maturity flags based on minimum required periods
    validity_flags = []

    # RSI2: requires at least 5 periods
    if "rsi_2" in df.columns or "rsi2" in df.columns:
        validity_flags.append(
            (pl.col("_row_idx") >= 5).cast(pl.Int8).alias("is_rsi2_valid")
        )

    # EMA validity flags
    if any(c in df.columns for c in ["ema_5", "ma_5", "ema5"]):
        validity_flags.append(
            (pl.col("_row_idx") >= 15).cast(pl.Int8).alias("is_ema5_valid")
        )

    if any(c in df.columns for c in ["ema_10", "ma_10", "ema10"]):
        validity_flags.append(
            (pl.col("_row_idx") >= 30).cast(pl.Int8).alias("is_ema10_valid")
        )

    if any(c in df.columns for c in ["ema_20", "ma_20", "ema20"]):
        validity_flags.append(
            (pl.col("_row_idx") >= 60).cast(pl.Int8).alias("is_ema20_valid")
        )

    if any(c in df.columns for c in ["ema_200", "ma_200", "ema200"]):
        validity_flags.append(
            (pl.col("_row_idx") >= 200).cast(pl.Int8).alias("is_ema200_valid")
        )

    # Market Z-score: requires 252 trading days (1 year)
    if any(c in df.columns for c in ["mkt_z_20d", "mkt_vol_20d_z"]):
        validity_flags.append(
            (pl.col("_row_idx") >= 252).cast(pl.Int8).alias("is_mkt_z_valid")
        )

    # Sector validity: based on member count
    if "sec_member_cnt" in df.columns:
        validity_flags.append(
            (pl.col("sec_member_cnt") >= 3).cast(pl.Int8).alias("is_sec_valid")
        )

    # General MA validity: at least 60 periods for stable moving averages
    validity_flags.append((pl.col("_row_idx") >= 60).cast(pl.Int8).alias("is_valid_ma"))

    # Add all validity flags at once
    if validity_flags:
        df = df.with_columns(validity_flags)
        logger.info(
            f"✅ Window maturity validity flags attached: +{len(validity_flags)} columns (is_*_valid)"
        )

    # Clean up temporary row index column
    df = df.drop("_row_idx")

    # Additional interaction and derived features (Phase 2.7)
    # Capture important relationships between existing features for enhanced predictive power
    # CRITICAL: Error stops pipeline to ensure root cause resolution
    try:
        interaction_features = []

        # Volume-price interactions
        if "volume_ratio_20d" in df.columns and "returns_1d" in df.columns:
            interaction_features.append(
                (pl.col("volume_ratio_20d") * pl.col("returns_1d")).alias(
                    "vol_ret_interaction"
                )
            )

        if "turnover_ratio_20d" in df.columns and "volatility_20d" in df.columns:
            interaction_features.append(
                (pl.col("turnover_ratio_20d") * pl.col("volatility_20d")).alias(
                    "turnover_vol_interaction"
                )
            )

        # Momentum-volatility interactions
        if "returns_20d" in df.columns and "volatility_20d" in df.columns:
            interaction_features.append(
                (pl.col("returns_20d") / (pl.col("volatility_20d") + 1e-12)).alias(
                    "risk_adjusted_momentum_20d"
                )
            )

        if "returns_5d" in df.columns and "volatility_20d" in df.columns:
            interaction_features.append(
                (pl.col("returns_5d") / (pl.col("volatility_20d") + 1e-12)).alias(
                    "risk_adjusted_momentum_5d"
                )
            )

        # Market-relative interactions
        if "beta_60d" in df.columns and "volatility_20d" in df.columns:
            interaction_features.append(
                (pl.col("beta_60d") * pl.col("volatility_20d")).alias("systematic_risk")
            )

        if "alpha_1d" in df.columns and "volume_ratio_20d" in df.columns:
            interaction_features.append(
                (pl.col("alpha_1d") * pl.col("volume_ratio_20d")).alias(
                    "alpha_volume_signal"
                )
            )

        # Sector-relative interactions
        if "rel_to_sec_5d" in df.columns and "sec_vol_20" in df.columns:
            interaction_features.append(
                (pl.col("rel_to_sec_5d") / (pl.col("sec_vol_20") + 1e-12)).alias(
                    "sec_risk_adjusted_rel_5d"
                )
            )

        if "beta_to_sec_60" in df.columns and "sec_mom_20" in df.columns:
            interaction_features.append(
                (pl.col("beta_to_sec_60") * pl.col("sec_mom_20")).alias(
                    "sec_beta_momentum"
                )
            )

        # Technical indicator combinations
        if "rsi_14" in df.columns and "returns_5d" in df.columns:
            interaction_features.append(
                (pl.col("rsi_14") / 50.0 - 1.0)
                * pl.col("returns_5d").alias("rsi_momentum_signal")
            )

        if "ma_gap_5_20" in df.columns and "volatility_20d" in df.columns:
            interaction_features.append(
                (pl.col("ma_gap_5_20") / (pl.col("volatility_20d") + 1e-12)).alias(
                    "ma_gap_volatility_ratio"
                )
            )

        # Liquidity-momentum interactions
        if "adv_ratio_1d_20d" in df.columns and "returns_5d" in df.columns:
            interaction_features.append(
                (pl.col("adv_ratio_1d_20d") * pl.col("returns_5d")).alias(
                    "liquidity_momentum_signal"
                )
            )

        # Cross-sectional rank-based features
        if "returns_20d" in df.columns:
            interaction_features.append(
                pl.col("returns_20d")
                .rank(method="average")
                .over("Date")
                .alias("cs_rank_returns_20d")
            )

        if "volume_ratio_20d" in df.columns:
            interaction_features.append(
                pl.col("volume_ratio_20d")
                .rank(method="average")
                .over("Date")
                .alias("cs_rank_volume_ratio")
            )

        if "volatility_20d" in df.columns:
            interaction_features.append(
                pl.col("volatility_20d")
                .rank(method="average")
                .over("Date")
                .alias("cs_rank_volatility")
            )

        # Price level indicators
        if "Close" in df.columns:
            # Distance from 52-week high/low
            interaction_features.append(
                pl.col("Close")
                .rolling_max(window_size=252)
                .over("Code")
                .alias("high_252d")
            )
            interaction_features.append(
                pl.col("Close")
                .rolling_min(window_size=252)
                .over("Code")
                .alias("low_252d")
            )

        if "Close" in df.columns and "high_252d" in [
            f.meta.output_name() for f in interaction_features if hasattr(f, "meta")
        ]:
            # Will be computed in second pass after high_252d exists
            pass

        # Add all interaction features
        if interaction_features:
            df = df.with_columns(interaction_features)

            # Second pass: features that depend on first pass
            second_pass = []
            if (
                "high_252d" in df.columns
                and "low_252d" in df.columns
                and "Close" in df.columns
            ):
                second_pass.append(
                    (
                        (pl.col("Close") - pl.col("low_252d"))
                        / (pl.col("high_252d") - pl.col("low_252d") + 1e-12)
                    ).alias("pct_from_52w_range")
                )

            if second_pass:
                df = df.with_columns(second_pass)
                interaction_features.extend(second_pass)

            logger.info(
                f"✅ Interaction and derived features attached: +{len(interaction_features)} columns"
            )
    except Exception as e:
        logger.error(f"CRITICAL ERROR in interaction features: {e}")
        raise  # Stop pipeline to ensure root cause resolution

    # Additional rolling statistics and quantile features (Phase 2.8)
    # Extended time-series features for pattern recognition
    # CRITICAL: Error stops pipeline to ensure root cause resolution
    try:
        rolling_features = []

        # Extended momentum features
        if "returns_1d" in df.columns:
            # 3-day and 10-day momentum
            rolling_features.append(
                pl.col("returns_1d")
                .rolling_sum(window_size=3)
                .over("Code")
                .alias("returns_3d")
            )
            rolling_features.append(
                pl.col("returns_1d")
                .rolling_sum(window_size=10)
                .over("Code")
                .alias("returns_10d")
            )
            rolling_features.append(
                pl.col("returns_1d")
                .rolling_sum(window_size=60)
                .over("Code")
                .alias("returns_60d")
            )

        # Extended volatility features
        if "returns_1d" in df.columns:
            rolling_features.append(
                pl.col("returns_1d")
                .rolling_std(window_size=5)
                .over("Code")
                .mul(252**0.5)
                .alias("volatility_5d")
            )
            rolling_features.append(
                pl.col("returns_1d")
                .rolling_std(window_size=60)
                .over("Code")
                .mul(252**0.5)
                .alias("volatility_60d")
            )

        # Rolling skewness and kurtosis indicators (simplified)
        if "returns_1d" in df.columns:
            # Positive/negative return ratios as proxy for skewness
            rolling_features.append(
                (pl.col("returns_1d") > 0)
                .cast(pl.Float64)
                .rolling_mean(window_size=20)
                .over("Code")
                .alias("positive_return_ratio_20d")
            )
            rolling_features.append(
                (pl.col("returns_1d") > 0)
                .cast(pl.Float64)
                .rolling_mean(window_size=60)
                .over("Code")
                .alias("positive_return_ratio_60d")
            )

        # Volume trend features
        if "Volume" in df.columns:
            rolling_features.append(
                pl.col("Volume")
                .rolling_mean(window_size=5)
                .over("Code")
                .alias("volume_ma_5d")
            )
            rolling_features.append(
                pl.col("Volume")
                .rolling_mean(window_size=10)
                .over("Code")
                .alias("volume_ma_10d")
            )
            rolling_features.append(
                pl.col("Volume")
                .rolling_mean(window_size=60)
                .over("Code")
                .alias("volume_ma_60d")
            )

        # Volume acceleration (change in volume trend)
        if "Volume" in df.columns:
            # Will compute after volume_ma_5d exists
            pass

        # Price momentum consistency
        if "returns_5d" in df.columns and "returns_20d" in df.columns:
            rolling_features.append(
                (pl.col("returns_5d").sign() == pl.col("returns_20d").sign())
                .cast(pl.Int8)
                .alias("momentum_consistency_5_20")
            )

        # Cross-sectional quantile features
        if "returns_1d" in df.columns:
            rolling_features.append(
                pl.col("returns_1d").rank(method="average").over("Date")
                / pl.count().over("Date").alias("cs_quantile_returns_1d")
            )

        if "returns_5d" in df.columns:
            rolling_features.append(
                pl.col("returns_5d").rank(method="average").over("Date")
                / pl.count().over("Date").alias("cs_quantile_returns_5d")
            )

        if "Volume" in df.columns:
            rolling_features.append(
                pl.col("Volume").rank(method="average").over("Date")
                / pl.count().over("Date").alias("cs_quantile_volume")
            )

        # Add first batch of rolling features
        if rolling_features:
            df = df.with_columns(rolling_features)

            # Second pass: features that depend on first pass
            second_pass_rolling = []

            # Volume acceleration
            if "volume_ma_5d" in df.columns and "volume_ma_20d" in df.columns:
                second_pass_rolling.append(
                    (
                        (pl.col("volume_ma_5d") - pl.col("volume_ma_20d"))
                        / (pl.col("volume_ma_20d") + 1e-12)
                    ).alias("volume_ma_acceleration")
                )

            # Volatility ratio (recent vs longer-term)
            if "volatility_5d" in df.columns and "volatility_20d" in df.columns:
                second_pass_rolling.append(
                    (
                        pl.col("volatility_5d") / (pl.col("volatility_20d") + 1e-12)
                    ).alias("volatility_ratio_5_20")
                )

            if second_pass_rolling:
                df = df.with_columns(second_pass_rolling)
                rolling_features.extend(second_pass_rolling)

            logger.info(
                f"✅ Extended rolling statistics and quantile features attached: +{len(rolling_features)} columns"
            )
    except Exception as e:
        logger.error(f"CRITICAL ERROR in rolling statistics features: {e}")
        raise  # Stop pipeline to ensure root cause resolution

    # Calendar and regime features (Phase 2.9)
    # Time-based patterns and market regime indicators
    # CRITICAL: Error stops pipeline to ensure root cause resolution
    try:
        calendar_features = []

        # Calendar features from Date column
        if "Date" in df.columns:
            calendar_features.extend(
                [
                    pl.col("Date").dt.month().alias("month"),
                    pl.col("Date").dt.quarter().alias("quarter"),
                    pl.col("Date").dt.weekday().alias("day_of_week"),
                    pl.col("Date").dt.day().alias("day_of_month"),
                ]
            )

            # Month-end effect (last 3 trading days approximation)
            calendar_features.append(
                (pl.col("Date").dt.day() >= 27).cast(pl.Int8).alias("is_month_end")
            )

            # Quarter-end effect
            calendar_features.append(
                (
                    pl.col("Date").dt.month().is_in([3, 6, 9, 12])
                    & (pl.col("Date").dt.day() >= 27)
                )
                .cast(pl.Int8)
                .alias("is_quarter_end")
            )

            # Year-end effect (December)
            calendar_features.append(
                (pl.col("Date").dt.month() == 12)
                .cast(pl.Int8)
                .alias("is_year_end_month")
            )

        # Gap features (open vs previous close)
        if "Open" in df.columns and "Close" in df.columns:
            calendar_features.append(
                (
                    (pl.col("Open") - pl.col("Close").shift(1).over("Code"))
                    / (pl.col("Close").shift(1).over("Code") + 1e-12)
                ).alias("overnight_gap")
            )

            # Gap up/down flags
            calendar_features.append(
                (pl.col("Open") > pl.col("Close").shift(1).over("Code"))
                .cast(pl.Int8)
                .alias("gap_up")
            )

        # Intraday range features
        if "High" in df.columns and "Low" in df.columns and "Close" in df.columns:
            calendar_features.append(
                ((pl.col("High") - pl.col("Low")) / (pl.col("Close") + 1e-12)).alias(
                    "intraday_range_pct"
                )
            )

            # Close position within daily range
            calendar_features.append(
                (
                    (pl.col("Close") - pl.col("Low"))
                    / (pl.col("High") - pl.col("Low") + 1e-12)
                ).alias("close_position_in_range")
            )

        # Regime indicators based on market environment
        if "mkt_ret_1d" in df.columns:
            # Bull/bear regime based on recent market performance
            calendar_features.append(
                (pl.col("mkt_ret_1d").rolling_mean(window_size=20).over("Date") > 0)
                .cast(pl.Int8)
                .alias("mkt_bull_20d")
            )

            # High/low volatility regime
            if "mkt_vol_20d" in df.columns:
                calendar_features.append(
                    (
                        pl.col("mkt_vol_20d")
                        > pl.col("mkt_vol_20d")
                        .rolling_mean(window_size=60)
                        .over("Date")
                    )
                    .cast(pl.Int8)
                    .alias("mkt_high_vol_regime")
                )

        # Consecutive up/down days
        if "returns_1d" in df.columns:
            calendar_features.append(
                (pl.col("returns_1d") > 0).cast(pl.Int8).alias("up_day")
            )

            # Streak of positive/negative days (simplified)
            calendar_features.append(
                (pl.col("returns_1d") > 0)
                .cast(pl.Int8)
                .rolling_sum(window_size=5)
                .over("Code")
                .alias("up_days_in_5d")
            )

        # Price reversal indicators
        if "returns_1d" in df.columns and "returns_5d" in df.columns:
            # Recent reversal (1d vs 5d direction mismatch)
            calendar_features.append(
                (pl.col("returns_1d").sign() != pl.col("returns_5d").sign())
                .cast(pl.Int8)
                .alias("potential_reversal_1d_5d")
            )

        # Distance from moving averages (additional windows)
        if "Close" in df.columns:
            for window in [10, 50, 100]:
                calendar_features.append(
                    pl.col("Close")
                    .rolling_mean(window_size=window)
                    .over("Code")
                    .alias(f"ma_{window}d")
                )

        # Add calendar features
        if calendar_features:
            df = df.with_columns(calendar_features)

            # Second pass: MA gap features
            second_pass_calendar = []
            if "ma_10d" in df.columns and "Close" in df.columns:
                second_pass_calendar.append(
                    (
                        (pl.col("Close") - pl.col("ma_10d"))
                        / (pl.col("ma_10d") + 1e-12)
                    ).alias("ma_gap_10d")
                )

            if "ma_50d" in df.columns and "Close" in df.columns:
                second_pass_calendar.append(
                    (
                        (pl.col("Close") - pl.col("ma_50d"))
                        / (pl.col("ma_50d") + 1e-12)
                    ).alias("ma_gap_50d")
                )

            if "ma_100d" in df.columns and "Close" in df.columns:
                second_pass_calendar.append(
                    (
                        (pl.col("Close") - pl.col("ma_100d"))
                        / (pl.col("ma_100d") + 1e-12)
                    ).alias("ma_gap_100d")
                )

            # Golden cross / death cross indicators
            if "ma_10d" in df.columns and "ma_50d" in df.columns:
                second_pass_calendar.append(
                    (pl.col("ma_10d") > pl.col("ma_50d"))
                    .cast(pl.Int8)
                    .alias("golden_cross_10_50")
                )

            if second_pass_calendar:
                df = df.with_columns(second_pass_calendar)
                calendar_features.extend(second_pass_calendar)

            logger.info(
                f"✅ Calendar and regime features attached: +{len(calendar_features)} columns"
            )
    except Exception as e:
        logger.error(f"CRITICAL ERROR in calendar and regime features: {e}")
        raise  # Stop pipeline to ensure root cause resolution

    # Graph-structured features (Phase 3): degree, peer corr mean, peer count
    try:
        if enable_graph_features:
            # Try to use GPU-accelerated version first
            use_gpu_graph = False
            try:
                import cupy as cp  # CuPy のみで GPU 動作可能

                if cp.cuda.runtime.getDeviceCount() > 0:
                    from src.gogooku3.features.graph_features_gpu import (
                        add_graph_features,
                    )

                    use_gpu_graph = True
                    logger.info(
                        "✅ Using GPU-accelerated graph computation (CuPy detected)"
                    )
            except ImportError:
                pass

            # Fallback to CPU version if GPU not available
            if not use_gpu_graph:
                from src.gogooku3.features.graph_features import add_graph_features

                logger.info("📊 Using CPU graph computation (CuPy not available)")

            df = add_graph_features(
                df,
                return_col="returns_1d"
                if "returns_1d" in df.columns
                else "feat_ret_1d",
                window=graph_window,
                min_obs=max(20, min(graph_window // 2, graph_window - 5)),
                threshold=graph_threshold,
                max_k=graph_max_k,
                method="pearson",
                cache_dir=graph_cache_dir,
            )
            logger.info(
                "Graph features attached (graph_degree, peer_corr_mean, peer_count)"
            )
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
                # Auto-discover anywhere under output/
                path = _find_latest("futures_daily_*.parquet")
                if path:
                    futures_df = pl.read_parquet(path)
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
                                out = Path("output/raw/futures")
                                out.mkdir(parents=True, exist_ok=True)
                                out = (
                                    out
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
                _nk_path = (
                    _auto_find_spot(["nikkei"])
                    or _auto_find_spot(["nk225"])
                    or _auto_find_spot(["nikkei225"])
                )  # type: ignore[assignment]
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
                categories=(
                    futures_categories or ["TOPIXF", "NK225F", "JN400F", "REITF"]
                ),
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
                futures_cols = [
                    c for c in df.columns if f"fut_{category.lower()}_" in c.lower()
                ]
                basis_cols = [c for c in futures_cols if "basis" in c.lower()]

                logger.info(
                    f"  - {category}: spot={'✅' if spot_available else '❌'}, "
                    f"futures_features={len(futures_cols)}, basis_features={len(basis_cols)}"
                )

            continuous_enabled = "✅ ON" if futures_continuous else "❌ OFF"
            logger.info(
                f"📈 Continuous series (fut_whole_ret_cont_*): {continuous_enabled}"
            )
        except Exception as e:
            logger.warning(f"Futures enrichment skipped: {e}")

    # Flow attach
    if trades_spec_path and Path(trades_spec_path).exists():
        try:
            trades_spec_df = pl.read_parquet(trades_spec_path)
            # Pass listed_info_df (if available) for accurate Section mapping
            df = builder.add_flow_features(
                df, trades_spec_df, listed_info_df=listed_info_df
            )
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
            alt = _find_latest("weekly_margin_interest_*.parquet")
            if alt:
                w_path = alt
        if enable_margin_weekly or (w_path and w_path.exists()):
            if w_path and w_path.exists():
                wdf = pl.read_parquet(w_path)
                tmp_w = _ensure_code_utf8(wdf, source="weekly_margin")
                if tmp_w is not None and not tmp_w.is_empty():
                    wdf = tmp_w
                tmp_q = _ensure_code_utf8(df, source="quotes_for_weekly_margin")
                if tmp_q is not None and not tmp_q.is_empty():
                    df = tmp_q
                # Validate and attach
                _validate_code_type_consistency(df, "quotes (weekly margin)")
                _validate_code_type_consistency(wdf, "weekly_margin")
                df = builder.add_margin_weekly_block(
                    df,
                    wdf,
                    lag_bdays_weekly=margin_weekly_lag,
                    adv_window_days=adv_window_days,
                )
                logger.info(f"Margin weekly features attached from: {w_path}")
            else:
                logger.info(
                    "Margin weekly requested but no parquet provided/found; skipping"
                )
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
            alt = _find_latest("daily_margin_interest_*.parquet")
            if alt:
                d_path = alt
        if enable_daily_margin or (d_path and d_path.exists()):
            if d_path and d_path.exists():
                ddf = pl.read_parquet(d_path)
                tmp_d = _ensure_code_utf8(ddf, source="daily_margin")
                if tmp_d is not None and not tmp_d.is_empty():
                    ddf = tmp_d
                tmp_q = _ensure_code_utf8(df, source="quotes_for_daily_margin")
                if tmp_q is not None and not tmp_q.is_empty():
                    df = tmp_q
                # Validate and attach
                _validate_code_type_consistency(df, "quotes (daily margin)")
                _validate_code_type_consistency(ddf, "daily_margin")
                df = builder.add_daily_margin_block(
                    df,
                    ddf,
                    adv_window_days=adv_window_days,
                    enable_z_scores=True,
                )
                logger.info(f"Daily margin features attached from: {d_path}")

                # Rename raw identifiers to avoid collisions with other enrichments
                rename_daily: dict[str, str] = {}
                if "PublishedDate" in df.columns:
                    rename_daily["PublishedDate"] = "dmi_published_date"
                if "ApplicationDate" in df.columns:
                    rename_daily["ApplicationDate"] = "dmi_application_date"
                if "PublishReason" in df.columns:
                    rename_daily["PublishReason"] = "dmi_publish_reason"
                if rename_daily:
                    df = df.rename(rename_daily)
                if "effective_start" in df.columns:
                    df = df.drop("effective_start")

                # Backfill canonical margin columns with daily data where possible
                df = _fill_from_source(df, "margin_d_long_wow", "dmi_d_long_1d")
                df = _fill_from_source(df, "margin_d_short_wow", "dmi_d_short_1d")
                df = _fill_from_source(df, "margin_d_net_wow", "dmi_d_net_1d")
                df = _fill_from_source(df, "margin_d_ratio_wow", "dmi_d_ratio_1d")
                df = _fill_from_source(
                    df, "margin_d_long_to_adv20", "dmi_d_long_to_adv1d"
                )
                df = _fill_from_source(
                    df, "margin_d_short_to_adv20", "dmi_d_short_to_adv1d"
                )
                df = _fill_from_source(df, "margin_long_to_adv20", "dmi_long_to_adv20")
                df = _fill_from_source(
                    df, "margin_short_to_adv20", "dmi_short_to_adv20"
                )
                df = _fill_from_source(df, "margin_gross_z52", "dmi_z26_total")
                df = _fill_from_source(df, "ratio_z52", "dmi_z26_d_short_1d")
                df = _fill_from_source(df, "long_z52", "dmi_z26_long")
                df = _fill_from_source(df, "short_z52", "dmi_z26_short")
                df = _fill_from_source(df, "margin_impulse", "dmi_impulse")
                df = _fill_from_source(df, "margin_days_since", "dmi_days_since_pub")
                df = _fill_from_source(df, "is_margin_valid", "is_dmi_valid")
            else:
                logger.info(
                    "Daily margin requested but no parquet provided/found; skipping"
                )
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
            # Auto-discover recursively under output/
            ss_path = _find_latest("short_selling_*.parquet")

        if short_positions_parquet and Path(short_positions_parquet).exists():
            pos_path = short_positions_parquet
        else:
            # Auto-discover recursively under output/
            pos_path = _find_latest("short_positions_*.parquet")

        if enable_short_selling or (ss_path and ss_path.exists()):
            # Load short selling data
            short_df = None
            positions_df = None

            if ss_path and ss_path.exists():
                try:
                    short_df = pl.read_parquet(ss_path)
                    result = _ensure_code_utf8(short_df, source="short_selling")
                    if result is not None:
                        short_df = result
                    logger.info(f"Loaded short selling data from: {ss_path}")
                except Exception as e:
                    logger.warning(f"Failed to load short selling data: {e}")

            if pos_path and pos_path.exists():
                try:
                    positions_df = pl.read_parquet(pos_path)
                    result = _ensure_code_utf8(positions_df, source="short_positions")
                    if result is not None:
                        positions_df = result
                    # LendingBalance and LendingBalanceRatio are NOT provided by J-Quants API
                    # No patching needed - removed as of 2025-10-20
                    logger.info(f"Loaded short positions data from: {pos_path}")
                except Exception as e:
                    logger.warning(f"Failed to load short positions data: {e}")

            # If missing and jquants available, fetch from API for the requested range
            if (short_df is None or short_df.is_empty()) and jquants:
                try:
                    import os

                    from src.gogooku3.components.jquants_async_fetcher import (
                        JQuantsAsyncFetcher,  # type: ignore
                    )

                    email = os.getenv("JQUANTS_AUTH_EMAIL", "")
                    password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
                    if email and password:
                        fetcher3 = JQuantsAsyncFetcher(email, password)
                        async with aiohttp.ClientSession() as session:
                            await fetcher3.authenticate(session)
                            logger.info(
                                f"Fetching short selling data {start_date} → {end_date} for enrichment"
                            )

                            # Fetch both ratio and positions data
                            ss_df = await fetcher3.get_short_selling(
                                session,
                                start_date,
                                end_date,
                                business_days=business_days,
                            )
                            if ss_df is not None and not ss_df.is_empty():
                                short_df = ss_df
                                # Save for reuse
                                try:
                                    out = (
                                        output_dir
                                        / f"short_selling_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                                    )
                                    ss_df.write_parquet(out)
                                    logger.info(f"Saved short selling parquet: {out}")
                                except Exception:
                                    pass

                            if positions_df is None:
                                pos_df = await fetcher3.get_short_selling_positions(
                                    session,
                                    start_date,
                                    end_date,
                                    business_days=business_days,
                                )
                                if pos_df is not None and not pos_df.is_empty():
                                    positions_df = pos_df
                                    # Save for reuse
                                    try:
                                        out = (
                                            output_dir
                                            / f"short_positions_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                                        )
                                        pos_df.write_parquet(out)
                                        logger.info(
                                            f"Saved short positions parquet: {out}"
                                        )
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
                            .with_columns(
                                [
                                    pl.col("AdjustmentVolume")
                                    .rolling_mean(adv_window_days)
                                    .over("Code")
                                    .alias("ADV20_shares")
                                ]
                            )
                            .drop_nulls(subset=["ADV20_shares"])
                        )
                        logger.info(
                            f"Computed ADV{adv_window_days} for short selling scaling"
                        )
                    except Exception as e:
                        logger.warning(f"ADV computation for short selling failed: {e}")

                # Import and apply short selling features
                from src.gogooku3.features.short_selling import add_short_selling_block

                result = _ensure_code_utf8(df, source="quotes_for_short_selling")
                if result is not None:
                    df = result
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
                logger.info(
                    "Short selling requested but no parquet provided/found; skipping"
                )
        else:
            logger.info("Short selling not enabled and no parquet found; skipping")
    except Exception as e:
        logger.warning(f"Short selling attach skipped: {e}")

    # ========================================================================
    # Earnings Events Integration (NEW)
    # ========================================================================
    try:
        earnings_path = None

        if (
            earnings_announcements_parquet
            and Path(earnings_announcements_parquet).exists()
        ):
            earnings_path = earnings_announcements_parquet
        else:
            # Auto-discover recursively under output/
            earnings_path = _find_latest("earnings_announcements_*.parquet")

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

                    from src.gogooku3.components.jquants_async_fetcher import (
                        JQuantsAsyncFetcher,  # type: ignore
                    )

                    email = os.getenv("JQUANTS_AUTH_EMAIL", "")
                    password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
                    if email and password:
                        fetcher4 = JQuantsAsyncFetcher(email, password)
                        async with aiohttp.ClientSession() as session:
                            await fetcher4.authenticate(session)
                            logger.info(
                                f"Fetching earnings announcements {start_date} → {end_date} for enrichment"
                            )

                            # Fetch earnings announcement data
                            ea_df = await fetcher4.get_earnings_announcements(
                                session, start_date, end_date
                            )
                            if ea_df is not None and not ea_df.is_empty():
                                announcements_df = ea_df
                                # Save for reuse
                                try:
                                    out = (
                                        output_dir
                                        / f"earnings_announcements_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                                    )
                                    ea_df.write_parquet(out)
                                    logger.info(
                                        f"Saved earnings announcements parquet: {out}"
                                    )
                                except Exception:
                                    pass
                except Exception as e:
                    logger.warning(f"Earnings announcements API fetch failed: {e}")

            # Apply earnings event features if we have data
            if announcements_df is not None and not announcements_df.is_empty():
                # Import and apply earnings event features
                from src.gogooku3.features.earnings_events import (
                    add_earnings_event_block,
                )

                # Use statements data if available for EPS growth
                statements_df = None
                if statements_parquet and Path(statements_parquet).exists():
                    try:
                        statements_df = pl.read_parquet(statements_parquet)
                        logger.info("Using statements data for EPS growth features")
                    except Exception as e:
                        logger.warning(
                            f"Failed to load statements for earnings features: {e}"
                        )

                df = add_earnings_event_block(
                    quotes=df,
                    announcement_df=announcements_df,
                    statements_df=statements_df,
                    enable_pead=enable_pead_features,
                    enable_volatility=True,
                )

                logger.info("Earnings event features attached successfully")
            else:
                logger.info(
                    "Earnings events requested but no data available; adding null features"
                )
                # Add null earnings features for consistency
                from src.gogooku3.features.earnings_events import (
                    add_earnings_event_block,
                )

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
            # Auto-discover recursively under output/
            sector_short_path = _find_latest("sector_short_selling_*.parquet")

        if enable_sector_short_selling or (
            sector_short_path and sector_short_path.exists()
        ):
            # Load sector short selling data
            sector_short_df = None

            if sector_short_path and sector_short_path.exists():
                try:
                    sector_short_df = pl.read_parquet(sector_short_path)
                    logger.info(
                        f"Loaded sector short selling data from: {sector_short_path}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load sector short selling data: {e}")

            # If missing and jquants available, fetch from API for the requested range
            if (sector_short_df is None or sector_short_df.is_empty()) and jquants:
                try:
                    import os

                    from src.gogooku3.components.jquants_async_fetcher import (
                        JQuantsAsyncFetcher,  # type: ignore
                    )

                    email = os.getenv("JQUANTS_AUTH_EMAIL", "")
                    password = os.getenv("JQUANTS_AUTH_PASSWORD", "")
                    if email and password:
                        fetcher5 = JQuantsAsyncFetcher(email, password)
                        async with aiohttp.ClientSession() as session:
                            await fetcher5.authenticate(session)
                            logger.info(
                                f"Fetching sector short selling data {start_date} → {end_date} for enrichment"
                            )

                            # Fetch sector short selling data
                            ss_df = await fetcher5.get_sector_short_selling(
                                session,
                                start_date,
                                end_date,
                                business_days=business_days,
                            )
                            if ss_df is not None and not ss_df.is_empty():
                                sector_short_df = ss_df
                                # Save for reuse
                                try:
                                    out = (
                                        output_dir
                                        / f"sector_short_selling_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                                    )
                                    ss_df.write_parquet(out)
                                    logger.info(
                                        f"Saved sector short selling parquet: {out}"
                                    )
                                except Exception:
                                    pass
                except Exception as e:
                    logger.warning(f"Sector short selling API fetch failed: {e}")

            # Apply sector short selling features if we have data
            if sector_short_df is not None and not sector_short_df.is_empty():
                # Import and apply sector short selling features
                from src.gogooku3.features.short_selling_sector import (
                    add_sector_short_selling_block,
                )

                # Build calendar-aware next business day callable
                if business_days:
                    _next_bd_expr = build_next_bday_expr_from_dates(business_days)
                else:
                    _dates = (
                        df.select("Date").unique().sort("Date")["Date"].to_list()
                        if "Date" in df.columns
                        else []
                    )
                    _next_bd_expr = build_next_bday_expr_from_dates(_dates)

                df = add_sector_short_selling_block(
                    quotes=df,
                    ss_df=sector_short_df,
                    listed_info_df=listed_info_df,
                    enable_z_scores=enable_sector_short_z_scores,
                    enable_relative_features=True,
                    calendar_next_bday=_next_bd_expr,
                )

                logger.info("Sector short selling features attached successfully")
            else:
                logger.info(
                    "Sector short selling requested but no data available; adding null features"
                )
                # Add null sector short selling features for consistency
                from src.gogooku3.features.short_selling_sector import (
                    add_sector_short_selling_block,
                )

                if business_days:
                    _next_bd_expr = build_next_bday_expr_from_dates(business_days)
                else:
                    _dates = (
                        df.select("Date").unique().sort("Date")["Date"].to_list()
                        if "Date" in df.columns
                        else []
                    )
                    _next_bd_expr = build_next_bday_expr_from_dates(_dates)

                df = add_sector_short_selling_block(
                    quotes=df,
                    ss_df=None,
                    listed_info_df=listed_info_df,
                    enable_z_scores=enable_sector_short_z_scores,
                    enable_relative_features=True,
                    calendar_next_bday=_next_bd_expr,
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
            if (
                index_option_features_parquet
                and Path(index_option_features_parquet).exists()
            ):
                try:
                    opt_feats_df = pl.read_parquet(index_option_features_parquet)
                    logger.info(
                        f"Loaded index option features from: {index_option_features_parquet}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load option features parquet: {e}")

            # 2) Else, if raw parquet provided, build features
            if (
                (opt_feats_df is None or opt_feats_df.is_empty())
                and index_option_raw_parquet
                and Path(index_option_raw_parquet).exists()
            ):
                try:
                    raw = pl.read_parquet(index_option_raw_parquet)
                    from src.gogooku3.features.index_option import (
                        build_index_option_features,
                    )

                    opt_feats_df = build_index_option_features(raw)
                    logger.info("Built option features from raw parquet")
                except Exception as e:
                    logger.warning(f"Failed to build option features from raw: {e}")

            # 2.5) Before API fetch, check for cached Index Options features
            if opt_feats_df is None or opt_feats_df.is_empty():
                import glob
                from datetime import timedelta

                # Look for cached nk225_index_option_features files in multiple locations
                search_dirs = [
                    Path("output/datasets"),
                    Path("output/raw"),
                    Path("output"),
                ]
                cache_files = []
                for output_dir in search_dirs:
                    if output_dir.exists():
                        cache_pattern = str(
                            output_dir / "nk225_index_option_features_*.parquet"
                        )
                        cache_files.extend(glob.glob(cache_pattern))

                if cache_files:
                    cache_files = sorted(cache_files, reverse=True)
                    try:
                        cached_option = Path(cache_files[0])
                        logger.info(
                            f"📦 Found cached index options: {cached_option.name}"
                        )

                        # Check if cache covers requested date range
                        cache_name = cached_option.stem
                        parts = cache_name.split("_")
                        if len(parts) >= 5:
                            cache_start = parts[-2]
                            cache_end = parts[-1]
                            cache_start_dt = datetime.strptime(cache_start, "%Y%m%d")
                            cache_end_dt = datetime.strptime(cache_end, "%Y%m%d")

                            # Parse requested dates
                            start_dt = (
                                datetime.strptime(start_date, "%Y-%m-%d")
                                if isinstance(start_date, str)
                                else start_date
                            )
                            end_dt = (
                                datetime.strptime(end_date, "%Y-%m-%d")
                                if isinstance(end_date, str)
                                else end_date
                            )

                            # Check if cache covers our range (allow 1-day tolerance)
                            if cache_start_dt <= start_dt and cache_end_dt >= (
                                end_dt - timedelta(days=1)
                            ):
                                # Load cached raw data and build features
                                raw = pl.read_parquet(cached_option)
                                from src.gogooku3.features.index_option import (
                                    build_index_option_features,
                                )

                                opt_feats_df = build_index_option_features(raw)
                                logger.info(
                                    "✅ CACHE HIT: Index Options (saved 15-30 min)"
                                )
                            else:
                                logger.info(
                                    f"⚠️  Cache date range mismatch: cache={cache_start}→{cache_end}, requested={start_dt.strftime('%Y%m%d')}→{end_dt.strftime('%Y%m%d')}"
                                )
                    except Exception as e:
                        logger.warning(f"Failed to load cached index options: {e}")

            # 3) Else, try API fetch and build features
            if (opt_feats_df is None or opt_feats_df.is_empty()) and jquants:
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
                            logger.info(
                                f"Fetching index options {start_date} → {end_date} for market aggregates"
                            )
                            raw = await fetcher.get_index_option(
                                session, start_date, end_date
                            )
                            if raw is not None and not raw.is_empty():
                                from src.gogooku3.features.index_option import (
                                    build_index_option_features,
                                )

                                opt_feats_df = build_index_option_features(raw)
                except Exception as e:
                    logger.warning(f"Index option API fetch failed: {e}")

            if opt_feats_df is not None and not opt_feats_df.is_empty():
                try:
                    from src.gogooku3.features.index_option import (
                        attach_option_market_to_equity,
                        build_option_market_aggregates,
                    )

                    if business_days:
                        nb = build_next_bday_expr_from_dates(business_days)
                    else:
                        _dates = (
                            df.select("Date").unique().sort("Date")["Date"].to_list()
                            if "Date" in df.columns
                            else []
                        )
                        nb = build_next_bday_expr_from_dates(_dates)
                    mkt = build_option_market_aggregates(
                        opt_feats_df, next_bday_expr=nb
                    )
                    df = attach_option_market_to_equity(df, mkt)
                    logger.info(
                        "Option market aggregates attached (opt_iv_cmat_*, opt_term_slope_30_60, flows)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to attach option market aggregates: {e}")
            else:
                logger.info(
                    "Option market features requested but no data available; skipping attach"
                )
        else:
            logger.info("Option market features not enabled; skipping")
    except Exception as e:
        logger.warning(f"Index option market aggregates step skipped: {e}")

    # Assurance: guarantee mkt_*
    if not any(c.startswith("mkt_") for c in df.columns):
        logger.warning("mkt_* missing; attempting offline attach (assurance)...")
        topo = (
            topix_parquet if topix_parquet else _find_latest("topix_history_*.parquet")
        )
        # Fetch and save a TOPIX parquet if still missing and jquants is enabled
        if (topo is None or not Path(topo).exists()) and jquants:
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
                        topo_df = await fetcher.fetch_topix_data(
                            session, start_date, end_date
                        )
                        if topo_df is not None and not topo_df.is_empty():
                            raw_dir = Path("output/raw/market")
                            raw_dir.mkdir(parents=True, exist_ok=True)
                            topo = (
                                raw_dir
                                / f"topix_history_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
                            )
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

    # Align to dataset_new.md (strict schema) just before saving
    try:
        eps = 1e-12

        try:
            df = builder.add_interaction_features(df)
            logger.info("Interaction features attached (x_*)")
        except Exception as e:
            logger.warning(f"Interaction features attach skipped: {e}")

        # Canonical, ordered schema from docs/ml/dataset_new.md
        DOC_COLUMNS: list[str] = [
            # 0) Identifiers/Meta (expanded per docs/ml/dataset_new.md)
            "Code",
            "Date",
            "Section",
            "MarketCode",
            "section_norm",
            "row_idx",
            # Sector metadata (ids + names)
            "sector17_code",
            "sector17_name",
            "sector17_id",
            "sector33_code",
            "sector33_name",
            "sector33_id",
            # Shares outstanding (for turnover/mcap uses)
            "shares_outstanding",
            # 1.1 OHLCV (6)
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "TurnoverValue",
            # 1.2 Returns (6) + log returns (4)
            "returns_1d",
            "returns_5d",
            "returns_10d",
            "returns_20d",
            "returns_60d",
            "returns_120d",
            "log_returns_1d",
            "log_returns_5d",
            "log_returns_10d",
            "log_returns_20d",
            # 1.3 Volatility (5)
            "volatility_5d",
            "volatility_10d",
            "volatility_20d",
            "volatility_60d",
            "realized_volatility",
            # 1.4 SMA/EMA (10)
            "sma_5",
            "sma_10",
            "sma_20",
            "sma_60",
            "sma_120",
            "ema_5",
            "ema_10",
            "ema_20",
            "ema_60",
            "ema_200",
            # 1.5 Price position/gaps (8)
            "price_to_sma5",
            "price_to_sma20",
            "price_to_sma60",
            "ma_gap_5_20",
            "ma_gap_20_60",
            "high_low_ratio",
            "close_to_high",
            "close_to_low",
            # 1.6 Volume/turnover (6)
            "volume_ma_5",
            "volume_ma_20",
            "volume_ratio_5",
            "volume_ratio_20",
            "turnover_rate",
            "dollar_volume",
            # 1.7 Technical (approx 10)
            "rsi_2",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_histogram",
            "atr_14",
            "adx_14",
            "stoch_k",
            "bb_width",
            "bb_position",
            # 2) TOPIX (26)
            "mkt_ret_1d",
            "mkt_ret_5d",
            "mkt_ret_10d",
            "mkt_ret_20d",
            "mkt_ema_5",
            "mkt_ema_20",
            "mkt_ema_60",
            "mkt_ema_200",
            "mkt_dev_20",
            "mkt_gap_5_20",
            "mkt_ema20_slope_3",
            "mkt_vol_20d",
            "mkt_atr_14",
            "mkt_natr_14",
            "mkt_bb_pct_b",
            "mkt_bb_bw",
            "mkt_dd_from_peak",
            "mkt_big_move_flag",
            "mkt_ret_1d_z",
            "mkt_vol_20d_z",
            "mkt_bb_bw_z",
            "mkt_dd_from_peak_z",
            "mkt_bull_200",
            "mkt_trend_up",
            "mkt_high_vol",
            "mkt_squeeze",
            # 3) Cross (8)
            "beta_60d",
            "alpha_1d",
            "alpha_5d",
            "rel_strength_5d",
            "trend_align_mkt",
            "alpha_vs_regime",
            "idio_vol_ratio",
            "beta_stability_60d",
            # 3.1) Interaction (24)
            "x_trend_intensity",
            "x_trend_intensity_g",
            "x_rel_sec_mom",
            "x_z_sec_gap_mom",
            "x_mom_sh_5",
            "x_mom_sh_10",
            "x_mom_sh_5_mktneu",
            "x_rvol5_dir",
            "x_rvol5_bb",
            "x_squeeze_pressure",
            "x_credit_rev_bias",
            "x_pead_effect",
            "x_pead_times_mkt",
            "x_rev_gate",
            "x_bo_gate",
            "x_alpha_meanrev_stable",
            "x_flow_smart_rel",
            "x_foreign_relsec",
            "x_tri_align",
            "x_bbpos_rvol5",
            "x_bbneg_rvol5",
            "x_liquidityshock_mom",
            "x_dmi_impulse_dir",
            "x_breadth_rel",
            # 4) Flow (13 enumerated in docs)
            "flow_foreign_net_ratio",
            "flow_individual_net_ratio",
            "flow_activity_ratio",
            "foreign_share_activity",
            "breadth_pos",
            "flow_foreign_net_z",
            "flow_individual_net_z",
            "flow_activity_z",
            "flow_smart_idx",
            "flow_smart_mom4",
            "flow_shock_flag",
            "flow_impulse",
            "flow_days_since",
            # 5) Statements (17)
            "stmt_yoy_sales",
            "stmt_yoy_op",
            "stmt_yoy_np",
            "stmt_opm",
            "stmt_npm",
            "stmt_progress_op",
            "stmt_progress_np",
            "stmt_rev_fore_op",
            "stmt_rev_fore_np",
            "stmt_rev_fore_eps",
            "stmt_rev_div_fore",
            "stmt_roe",
            "stmt_roa",
            "stmt_change_in_est",
            "stmt_nc_flag",
            "stmt_imp_statement",
            "stmt_days_since_statement",
            # 6) Flags (+ special market day)
            "is_rsi2_valid",
            "is_ema5_valid",
            "is_ema10_valid",
            "is_ema20_valid",
            "is_ema200_valid",
            "is_valid_ma",
            "is_flow_valid",
            "is_stmt_valid",
            "is_halt_20201001",
            # 7) Optional: Margin weekly block (kept if present; not added as Nulls)
            "margin_long_tot",
            "margin_short_tot",
            "margin_total_gross",
            "margin_credit_ratio",
            "margin_imbalance",
            "margin_d_long_wow",
            "margin_d_short_wow",
            "margin_d_net_wow",
            "margin_d_ratio_wow",
            "long_z52",
            "short_z52",
            "margin_gross_z52",
            "ratio_z52",
            "margin_long_to_adv20",
            "margin_short_to_adv20",
            "margin_d_long_to_adv20",
            "margin_d_short_to_adv20",
            "margin_impulse",
            "margin_days_since",
            "is_margin_valid",
            "margin_issue_type",
            "is_borrowable",
            # 7) Targets (7)
            "target_1d",
            "target_5d",
            "target_10d",
            "target_20d",
            "target_1d_binary",
            "target_5d_binary",
            "target_10d_binary",
            # 8) Forward Return Labels (4) - CRITICAL for supervised learning
            "feat_ret_1d",
            "feat_ret_5d",
            "feat_ret_10d",
            "feat_ret_20d",
        ]

        # Ensure mandatory identifier/meta columns exist even when upstream
        # enrichment did not attach them (e.g., offline runs without listed_info).
        meta_specs: list[tuple[str, pl.DataType, object | None]] = [
            ("MarketCode", pl.Utf8, None),
            ("sector17_code", pl.Utf8, None),
            ("sector17_name", pl.Utf8, None),
            ("sector17_id", pl.Int32, -1),
            ("sector33_code", pl.Utf8, None),
            ("sector33_name", pl.Utf8, None),
            ("sector33_id", pl.Int32, -1),
            ("shares_outstanding", pl.Float64, None),
        ]

        null_dtype_overrides: dict[str, pl.DataType] = {
            "MarketCode": pl.Utf8,
            "sector17_code": pl.Utf8,
            "sector17_name": pl.Utf8,
            "sector17_id": pl.Int32,
            "sector33_code": pl.Utf8,
            "sector33_name": pl.Utf8,
            "sector33_id": pl.Int32,
            "shares_outstanding": pl.Float64,
            "turnover_rate": pl.Float64,
            "stmt_yoy_sales": pl.Float64,
            "stmt_yoy_op": pl.Float64,
            "stmt_yoy_np": pl.Float64,
            "stmt_opm": pl.Float64,
            "stmt_npm": pl.Float64,
            "stmt_progress_op": pl.Float64,
            "stmt_progress_np": pl.Float64,
            "stmt_rev_fore_op": pl.Float64,
            "stmt_rev_fore_np": pl.Float64,
            "stmt_rev_fore_eps": pl.Float64,
            "stmt_rev_div_fore": pl.Float64,
            "stmt_roe": pl.Float64,
            "stmt_roa": pl.Float64,
            "stmt_change_in_est": pl.Int8,
            "stmt_nc_flag": pl.Int8,
            "stmt_imp_statement": pl.Int8,
            "stmt_days_since_statement": pl.Int32,
            "is_stmt_valid": pl.Int8,
        }

        meta_exprs: list[pl.Expr] = []
        for col_name, dtype, default in meta_specs:
            if col_name not in df.columns:
                lit_expr = pl.lit(default) if default is not None else pl.lit(None)
                meta_exprs.append(lit_expr.cast(dtype).alias(col_name))
        if meta_exprs:
            df = df.with_columns(meta_exprs)

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
        # If both source and target exist, drop the source to avoid duplicate target names
        for _src, _tgt in list(rename_map.items()):
            if _src in df.columns and _tgt in df.columns:
                try:
                    df = df.drop(_src)
                    logger.info(
                        f"Dropped duplicate alias column '{_src}' (target '{_tgt}' already present)"
                    )
                except Exception:
                    pass
        # Apply renames only where target is not already present
        to_rename = {
            k: v
            for k, v in rename_map.items()
            if (k in df.columns) and (v not in df.columns)
        }
        if to_rename:
            df = df.rename(to_rename)
            logger.info(f"Renamed columns to docs naming: {to_rename}")

        # Compute absolutely-required missing features (so they are not null-only)
        # Returns (1d/5d) often dropped accidentally by earlier selection
        need_returns = [c for c in ["returns_1d", "returns_5d"] if c not in df.columns]
        if need_returns:
            df = df.with_columns(
                [
                    pl.col("Close").pct_change().over("Code").alias("returns_1d")
                    if "returns_1d" in need_returns
                    else pl.lit(0),
                    pl.col("Close").pct_change(5).over("Code").alias("returns_5d")
                    if "returns_5d" in need_returns
                    else pl.lit(0),
                ]
            )
            # Drop filler 0 columns that are not needed (the lit(0) placeholders)
            drop_tmp = [c for c in ["literal"] if c in df.columns]
            if drop_tmp:
                df = df.drop(drop_tmp)

        # Extend returns if missing
        add_more_returns: list[pl.Expr] = []
        if "returns_10d" not in df.columns:
            add_more_returns.append(
                pl.col("Close").pct_change(10).over("Code").alias("returns_10d")
            )
        if "returns_20d" not in df.columns:
            add_more_returns.append(
                pl.col("Close").pct_change(20).over("Code").alias("returns_20d")
            )
        if "returns_60d" not in df.columns:
            add_more_returns.append(
                pl.col("Close").pct_change(60).over("Code").alias("returns_60d")
            )
        if "returns_120d" not in df.columns:
            add_more_returns.append(
                pl.col("Close").pct_change(120).over("Code").alias("returns_120d")
            )
        if "log_returns_1d" not in df.columns:
            add_more_returns.append(
                (
                    pl.col("Close").log() - pl.col("Close").log().shift(1).over("Code")
                ).alias("log_returns_1d")
            )
        if "log_returns_5d" not in df.columns:
            add_more_returns.append(
                (
                    pl.col("Close").log() - pl.col("Close").log().shift(5).over("Code")
                ).alias("log_returns_5d")
            )
        if "log_returns_10d" not in df.columns:
            add_more_returns.append(
                (
                    pl.col("Close").log() - pl.col("Close").log().shift(10).over("Code")
                ).alias("log_returns_10d")
            )
        if "log_returns_20d" not in df.columns:
            add_more_returns.append(
                (
                    pl.col("Close").log() - pl.col("Close").log().shift(20).over("Code")
                ).alias("log_returns_20d")
            )
        if add_more_returns:
            df = df.with_columns(add_more_returns)

        # EMAs and MA gaps
        ema_needed = [
            c
            for c in ["ema_5", "ema_10", "ema_20", "ema_60", "ema_200"]
            if c not in df.columns
        ]
        if ema_needed:
            df = df.with_columns(
                [
                    pl.col("Close")
                    .ewm_mean(span=5, adjust=False, ignore_nulls=True)
                    .over("Code")
                    .alias("ema_5")
                    if "ema_5" in ema_needed
                    else pl.lit(0),
                    pl.col("Close")
                    .ewm_mean(span=10, adjust=False, ignore_nulls=True)
                    .over("Code")
                    .alias("ema_10")
                    if "ema_10" in ema_needed
                    else pl.lit(0),
                    pl.col("Close")
                    .ewm_mean(span=20, adjust=False, ignore_nulls=True)
                    .over("Code")
                    .alias("ema_20")
                    if "ema_20" in ema_needed
                    else pl.lit(0),
                    pl.col("Close")
                    .ewm_mean(span=60, adjust=False, ignore_nulls=True)
                    .over("Code")
                    .alias("ema_60")
                    if "ema_60" in ema_needed
                    else pl.lit(0),
                    pl.col("Close")
                    .ewm_mean(span=200, adjust=False, ignore_nulls=True)
                    .over("Code")
                    .alias("ema_200")
                    if "ema_200" in ema_needed
                    else pl.lit(0),
                ]
            )
            # Remove any lit(0) placeholders that might have been added
            for col in df.columns:
                if col.startswith("literal"):
                    df = df.drop(col)

        if "ma_gap_5_20" not in df.columns and all(
            c in df.columns for c in ["ema_5", "ema_20"]
        ):
            df = df.with_columns(
                ((pl.col("ema_5") - pl.col("ema_20")) / (pl.col("ema_20") + eps)).alias(
                    "ma_gap_5_20"
                )
            )
        if "ma_gap_20_60" not in df.columns and all(
            c in df.columns for c in ["ema_20", "ema_60"]
        ):
            df = df.with_columns(
                (
                    (pl.col("ema_20") - pl.col("ema_60")) / (pl.col("ema_60") + eps)
                ).alias("ma_gap_20_60")
            )

        # SMAs and price positions
        sma_spans = [5, 10, 20, 60, 120]
        sma_exprs: list[pl.Expr] = []
        for w in sma_spans:
            name = f"sma_{w}"
            if name not in df.columns:
                sma_exprs.append(
                    pl.col("Close")
                    .rolling_mean(window_size=w, min_periods=w)
                    .over("Code")
                    .alias(name)
                )
        if sma_exprs:
            df = df.with_columns(sma_exprs)
        if "price_to_sma5" not in df.columns and "sma_5" in df.columns:
            df = df.with_columns(
                (pl.col("Close") / (pl.col("sma_5") + eps)).alias("price_to_sma5")
            )
        if "price_to_sma20" not in df.columns and "sma_20" in df.columns:
            df = df.with_columns(
                (pl.col("Close") / (pl.col("sma_20") + eps)).alias("price_to_sma20")
            )
        if "price_to_sma60" not in df.columns and "sma_60" in df.columns:
            df = df.with_columns(
                (pl.col("Close") / (pl.col("sma_60") + eps)).alias("price_to_sma60")
            )

        # Range/position metrics
        if "high_low_ratio" not in df.columns and all(
            c in df.columns for c in ["High", "Low"]
        ):
            df = df.with_columns(
                (pl.col("High") / (pl.col("Low") + eps)).alias("high_low_ratio")
            )
        if "close_to_high" not in df.columns and all(
            c in df.columns for c in ["High", "Low", "Close"]
        ):
            df = df.with_columns(
                (
                    (pl.col("High") - pl.col("Close"))
                    / ((pl.col("High") - pl.col("Low")) + eps)
                ).alias("close_to_high")
            )
        if "close_to_low" not in df.columns and all(
            c in df.columns for c in ["High", "Low", "Close"]
        ):
            df = df.with_columns(
                (
                    (pl.col("Close") - pl.col("Low"))
                    / ((pl.col("High") - pl.col("Low")) + eps)
                ).alias("close_to_low")
            )

        # Volume moving averages and ratios
        if "volume_ma_5" not in df.columns:
            df = df.with_columns(
                pl.col("Volume")
                .rolling_mean(window_size=5, min_periods=5)
                .over("Code")
                .alias("volume_ma_5")
            )
        if "volume_ma_20" not in df.columns:
            df = df.with_columns(
                pl.col("Volume")
                .rolling_mean(window_size=20, min_periods=20)
                .over("Code")
                .alias("volume_ma_20")
            )
        if "volume_ratio_5" not in df.columns and "volume_ma_5" in df.columns:
            df = df.with_columns(
                (pl.col("Volume") / (pl.col("volume_ma_5") + eps)).alias(
                    "volume_ratio_5"
                )
            )
        if "volume_ratio_20" not in df.columns and "volume_ma_20" in df.columns:
            df = df.with_columns(
                (pl.col("Volume") / (pl.col("volume_ma_20") + eps)).alias(
                    "volume_ratio_20"
                )
            )

        # Turnover rate requires shares outstanding; ensure column exists
        if "turnover_rate" not in df.columns:
            if {"Volume", "shares_outstanding"}.issubset(df.columns):
                df = df.with_columns(
                    pl.when(pl.col("shares_outstanding") > 0)
                    .then(pl.col("Volume") / (pl.col("shares_outstanding") + eps))
                    .otherwise(None)
                    .alias("turnover_rate")
                )
            else:
                df = df.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("turnover_rate")
                )

        # Validity flags for technical indicators (align with dataset_new spec)
        flag_sources: dict[str, str] = {
            "is_ema5_valid": "ema_5",
            "is_ema10_valid": "ema_10",
            "is_ema20_valid": "ema_20",
            "is_ema60_valid": "ema_60",
            "is_ema200_valid": "ema_200",
            "is_rsi2_valid": "rsi_2",
        }
        flag_exprs: list[pl.Expr] = []
        for flag_name, source_col in flag_sources.items():
            if flag_name not in df.columns:
                if source_col in df.columns:
                    flag_exprs.append(
                        pl.col(source_col).is_not_null().cast(pl.Int8).alias(flag_name)
                    )
                else:
                    flag_exprs.append(pl.lit(None).cast(pl.Int8).alias(flag_name))
        if flag_exprs:
            df = df.with_columns(flag_exprs)

        if "is_valid_ma" not in df.columns:
            if all(col in df.columns for col in ("sma_5", "sma_20", "sma_60")):
                df = df.with_columns(
                    (
                        pl.col("sma_5").is_not_null().cast(pl.Int8)
                        * pl.col("sma_20").is_not_null().cast(pl.Int8)
                        * pl.col("sma_60").is_not_null().cast(pl.Int8)
                    )
                    .cast(pl.Int8)
                    .alias("is_valid_ma")
                )
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Int8).alias("is_valid_ma"))

        # Ensure statement block columns exist (fill with nulls when upstream data absent)
        stmt_columns = [
            "stmt_yoy_sales",
            "stmt_yoy_op",
            "stmt_yoy_np",
            "stmt_opm",
            "stmt_npm",
            "stmt_progress_op",
            "stmt_progress_np",
            "stmt_rev_fore_op",
            "stmt_rev_fore_np",
            "stmt_rev_fore_eps",
            "stmt_rev_div_fore",
            "stmt_roe",
            "stmt_roa",
            "stmt_change_in_est",
            "stmt_nc_flag",
            "stmt_imp_statement",
            "stmt_days_since_statement",
        ]
        stmt_exprs: list[pl.Expr] = []
        for col_name in stmt_columns:
            if col_name not in df.columns:
                stmt_exprs.append(pl.lit(None).cast(pl.Float64).alias(col_name))
        if stmt_exprs:
            df = df.with_columns(stmt_exprs)

        # Dollar volume
        if "dollar_volume" not in df.columns:
            df = df.with_columns(
                (pl.col("Close") * pl.col("Volume")).alias("dollar_volume")
            )

        # Volatilities
        if "volatility_5d" not in df.columns and "returns_1d" in df.columns:
            df = df.with_columns(
                pl.col("returns_1d")
                .rolling_std(window_size=5, min_periods=5)
                .over("Code")
                .map_elements(lambda x: x * (252**0.5) if x is not None else None)
                .alias("volatility_5d")
            )
        if "volatility_10d" not in df.columns and "returns_1d" in df.columns:
            df = df.with_columns(
                pl.col("returns_1d")
                .rolling_std(window_size=10, min_periods=10)
                .over("Code")
                .map_elements(lambda x: x * (252**0.5) if x is not None else None)
                .alias("volatility_10d")
            )
        if "volatility_60d" not in df.columns and "returns_1d" in df.columns:
            df = df.with_columns(
                pl.col("returns_1d")
                .rolling_std(window_size=60, min_periods=60)
                .over("Code")
                .map_elements(lambda x: x * (252**0.5) if x is not None else None)
                .alias("volatility_60d")
            )

        # Reconstruct MACD line if signal+histogram exist
        if "macd" not in df.columns and all(
            c in df.columns for c in ["macd_signal", "macd_histogram"]
        ):
            df = df.with_columns(
                (pl.col("macd_signal") + pl.col("macd_histogram")).alias("macd")
            )

        # ATR(14) and Stochastic %K (14)
        if "atr_14" not in df.columns and all(
            c in df.columns for c in ["High", "Low", "Close"]
        ):
            tr = pl.max_horizontal(
                [
                    pl.col("High") - pl.col("Low"),
                    (pl.col("High") - pl.col("Close").shift(1).over("Code")).abs(),
                    (pl.col("Low") - pl.col("Close").shift(1).over("Code")).abs(),
                ]
            )
            df = df.with_columns(tr.alias("_tr"))
            df = df.with_columns(
                pl.col("_tr")
                .ewm_mean(span=14, adjust=False, ignore_nulls=True)
                .over("Code")
                .alias("atr_14")
            ).drop("_tr")
        if "stoch_k" not in df.columns and all(
            c in df.columns for c in ["High", "Low", "Close"]
        ):
            ll = pl.col("Low").rolling_min(window_size=14, min_periods=14).over("Code")
            hh = pl.col("High").rolling_max(window_size=14, min_periods=14).over("Code")
            df = df.with_columns(
                ((pl.col("Close") - ll) / ((hh - ll) + eps) * 100.0).alias("stoch_k")
            )

        # ADX(14) (simplified Wilder smoothing using EWM span)
        if "adx_14" not in df.columns and all(
            c in df.columns for c in ["High", "Low", "Close"]
        ):
            up_move = (pl.col("High") - pl.col("High").shift(1).over("Code")).clip(
                lower_bound=0
            )
            down_move = (pl.col("Low").shift(1).over("Code") - pl.col("Low")).clip(
                lower_bound=0
            )
            plus_dm = (
                pl.when((up_move > down_move) & (up_move > 0))
                .then(up_move)
                .otherwise(0.0)
            )
            minus_dm = (
                pl.when((down_move > up_move) & (down_move > 0))
                .then(down_move)
                .otherwise(0.0)
            )
            tr2 = pl.max_horizontal(
                [
                    pl.col("High") - pl.col("Low"),
                    (pl.col("High") - pl.col("Close").shift(1).over("Code")).abs(),
                    (pl.col("Low") - pl.col("Close").shift(1).over("Code")).abs(),
                ]
            )
            atr14 = tr2.ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code")
            plus_di = (
                plus_dm.ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code")
                / (atr14 + eps)
            ) * 100.0
            minus_di = (
                minus_dm.ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code")
                / (atr14 + eps)
            ) * 100.0
            dx = ((plus_di - minus_di).abs() / ((plus_di + minus_di) + eps)) * 100.0
            adx14 = dx.ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code")
            df = df.with_columns(adx14.alias("adx_14"))

        # Mask range-derived features on 2020-10-01 (TSE halt day)
        if not disable_halt_mask:
            try:
                mask_date = pl.date(2020, 10, 1)
                range_cols = [
                    c
                    for c in [
                        "atr_14",
                        "high_low_ratio",
                        "close_to_high",
                        "close_to_low",
                    ]
                    if c in df.columns
                ]
                if range_cols:
                    df = df.with_columns(
                        [
                            pl.when(pl.col("Date") == mask_date)
                            .then(None)
                            .otherwise(pl.col(c))
                            .alias(c)
                            for c in range_cols
                        ]
                    )
            except Exception:
                pass

        # Section normalization fallback
        if "section_norm" not in df.columns and "Section" in df.columns:
            df = df.with_columns(pl.col("Section").alias("section_norm"))

        # Validity flags fallback
        if "row_idx" in df.columns:
            if "is_rsi2_valid" not in df.columns:
                df = df.with_columns(
                    (pl.col("row_idx") >= 5).cast(pl.Int8).alias("is_rsi2_valid")
                )
            if "is_valid_ma" not in df.columns:
                df = df.with_columns(
                    (pl.col("row_idx") >= 60).cast(pl.Int8).alias("is_valid_ma")
                )
        # Statement validity fallback
        if (
            "is_stmt_valid" not in df.columns
            and "stmt_days_since_statement" in df.columns
        ):
            df = df.with_columns(
                (pl.col("stmt_days_since_statement") >= 0)
                .cast(pl.Int8)
                .alias("is_stmt_valid")
            )

        # Add any missing spec columns as nulls (safe defaults)
        # docs/ml/dataset_new.md の完全仕様に合わせ、margin_ も含めて
        # すべての文書化済み列を最終スキーマに揃える
        existing = set(df.columns)
        to_add_nulls = [c for c in DOC_COLUMNS if c not in existing]
        if to_add_nulls:
            logger.info(
                "Adding null fillers for missing spec columns: %s", to_add_nulls[:10]
            )
            df = df.with_columns(
                [
                    pl.lit(None).cast(null_dtype_overrides.get(c, pl.Float64)).alias(c)
                    for c in to_add_nulls
                ]
            )

        # Fill conservative defaults for statement flags when statements are absent
        zero_literal = pl.lit(0, dtype=pl.Int8)
        fill_zero_flags = []
        for c in ["stmt_change_in_est", "stmt_nc_flag", "is_stmt_valid"]:
            if c in df.columns:
                fill_zero_flags.append(
                    pl.coalesce(
                        [
                            pl.col(c).cast(pl.Int8, strict=False),
                            zero_literal,
                        ]
                    ).alias(c)
                )
        if fill_zero_flags:
            df = df.with_columns(fill_zero_flags)

        # Cast common boolean/int flags to Int8 for stable schema
        flag_like = [
            c for c in df.columns if c.startswith("is_") or c.endswith("_impulse")
        ]
        if flag_like:
            cast_exprs = [
                pl.col(c).cast(pl.Int8, strict=False).alias(c)
                for c in flag_like
                if c in df.columns
            ]
            if cast_exprs:
                df = df.with_columns(cast_exprs)

        # Finally, project to the exact schema and preserve documented feature groups
        keep_cols = [c for c in DOC_COLUMNS if c in df.columns]

        # Allow-listed prefixes that must be kept if present (documented groups)
        allowed_prefixes = (
            "graph_",
            "peer_",
            "flow_",
            "margin_",
            "dmi_",
            "x_",
            "mkt_",
            "stmt_",
            "sec17_onehot_",
            "sect_",
        )
        allowed_exact = {
            # Flow compatibility names without flow_ prefix
            "foreign_share_activity",
            "breadth_pos",
            # Sector frequency helper
            "sec33_daily_freq",
        }
        deny_prefixes = ("_",)  # internal temporaries
        deny_suffixes = ("_right",)  # join helpers

        def _is_allowed_extra(name: str) -> bool:
            if any(name.startswith(p) for p in deny_prefixes):
                return False
            if any(name.endswith(s) for s in deny_suffixes):
                return False
            if name in allowed_exact:
                return True
            return any(name.startswith(p) for p in allowed_prefixes)

        opt_extra = [c for c in df.columns if _is_allowed_extra(c)]
        for c in opt_extra:
            if c not in keep_cols:
                keep_cols.append(c)

        # Enforce minimum column count (docs minimum=395)
        MIN_COLS = 395
        if len(keep_cols) < MIN_COLS:
            candidates = [
                c
                for c in df.columns
                if c not in keep_cols
                and not any(c.startswith(p) for p in deny_prefixes)
                and not any(c.endswith(s) for s in deny_suffixes)
            ]
            for c in candidates:
                keep_cols.append(c)
                if len(keep_cols) >= MIN_COLS:
                    break
            logger.info(
                "Minimum column enforcement applied: final=%d (target=%d)",
                len(keep_cols),
                MIN_COLS,
            )
        logger.info(
            "Post-alignment column check: MarketCode=%s, sector33_code=%s, shares_outstanding=%s, stmt_yoy_sales=%s",
            "MarketCode" in df.columns,
            "sector33_code" in df.columns,
            "shares_outstanding" in df.columns,
            "stmt_yoy_sales" in df.columns,
        )
        df = df.select(keep_cols)
        logger.info(f"Aligned dataset to docs schema (n={len(keep_cols)})")
    except Exception as _e:
        logger.exception("dataset_new.md strict alignment skipped: %s", _e)

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

    pq_path, meta_path = save_with_symlinks(
        df, output_dir, tag="full", start_date=start_date, end_date=end_date
    )
    return pq_path, meta_path
