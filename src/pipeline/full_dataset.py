#!/usr/bin/env python3
"""
Full dataset pipeline orchestrator (enrichment + save).

Provides reusable functions so that CLI wrappers under scripts/ remain thin.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Optional

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
        import pyarrow as pa  # type: ignore
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
    trades_spec_path: Optional[Path] = None,
    topix_parquet: Optional[Path] = None,
    statements_parquet: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Attach TOPIX + statements + flow then save with symlinks.

    Includes an assurance step to guarantee mkt_* presence by discovering
    or fetching TOPIX parquet when needed.
    """
    from scripts.data.ml_dataset_builder import MLDatasetBuilder
    import aiohttp

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
            from scripts._archive.run_pipeline import JQuantsAsyncFetcher  # type: ignore
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
        cand = _find_latest("topix_history_*.parquet")
        if cand and cand.exists():
            try:
                topix_df = pl.read_parquet(cand)
                logger.info(f"Loaded TOPIX from local: {cand}")
            except Exception as e:
                logger.warning(f"Failed to read local TOPIX parquet: {e}")

    df = builder.add_topix_features(df_base, topix_df=topix_df)

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

    # Flow attach
    if trades_spec_path and Path(trades_spec_path).exists():
        try:
            trades_spec_df = pl.read_parquet(trades_spec_path)
            df = builder.add_flow_features(df, trades_spec_df, listed_info_df=None)
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
                from scripts._archive.run_pipeline import JQuantsAsyncFetcher  # type: ignore
                import aiohttp
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
                                import pyarrow as pa
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
                ok = any(c.startswith("mkt_") for c in df.columns)
                logger.info(f"Offline TOPIX attach {'succeeded' if ok else 'failed'}")
            except Exception as e:
                logger.warning(f"Offline TOPIX attach failed: {e}")

    pq_path, meta_path = save_with_symlinks(df, output_dir, tag="full", start_date=start_date, end_date=end_date)
    return pq_path, meta_path

