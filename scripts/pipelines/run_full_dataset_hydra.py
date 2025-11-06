#!/usr/bin/env python3
from __future__ import annotations

import asyncio

# Ensure imports
import sys
from datetime import datetime, timedelta
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipelines.run_pipeline_v4_optimized import JQuantsPipelineV4Optimized
from src.pipeline.data_quality_checker import DataQualityChecker, QualityConfig
from src.pipeline.full_dataset import enrich_and_save, save_with_symlinks
from src.pipeline.incremental_updater import (
    IncrementalConfig,
    IncrementalDatasetUpdater,
)
from src.pipeline.lineage_tracker import DataLineageTracker
from src.pipeline.performance_monitor import PipelineProfiler
from src.pipeline.resilience import ResilienceConfig, ResilientPipeline


def _resolve_dates(start: str | None, end: str | None) -> tuple[str, str]:
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        dt_end = datetime.strptime(end, "%Y-%m-%d")
        start = (dt_end - timedelta(days=1826)).strftime("%Y-%m-%d")
    return start, end


@hydra.main(version_base=None, config_path="../../configs/pipeline/full_dataset", config_name="config")
def main(cfg: DictConfig) -> None:
    p = cfg.pipeline
    output_dir = Path(p.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    async def _run() -> None:
        # Resolve dates inside async function
        start, end = _resolve_dates(p.start_date, p.end_date)

        pipe = JQuantsPipelineV4Optimized(output_dir=output_dir)
        profiler = PipelineProfiler() if bool(p.profiling.enabled) else None
        # Incremental mode: adjust start date based on last metadata (if not provided)
        inc = IncrementalDatasetUpdater(
            output_dir, IncrementalConfig(update_mode=str(p.update_mode), since_date=p.since_date)
        )
        if str(p.update_mode) == "incremental":
            guess_since = inc.compute_since_date(fallback_end=end)
            if guess_since:
                start = guess_since
        lineage = DataLineageTracker(output_dir)
        # Run base pipeline (profiled)
        if profiler:
            with profiler.timer("base_pipeline_run"):
                base_df, metadata = await pipe.run(use_jquants=bool(p.jquants), start_date=start, end_date=end)
        else:
            base_df, metadata = await pipe.run(use_jquants=bool(p.jquants), start_date=start, end_date=end)
        # Lineage for base build
        lineage.track_transformation(
            inputs=[f"JQuants:{start}->{end}" if bool(p.jquants) else "offline"],
            output=str(output_dir / "base_df_in_memory"),
            transformation="base_pipeline_run",
            metadata={"start": start, "end": end},
        )
        if base_df is None or base_df.is_empty():
            print("❌ Base dataset build failed")
            return
        # Call enrichment/saver
        resil = ResilientPipeline(
            output_dir,
            ResilienceConfig(
                enabled=bool(p.resilience.enabled),
                max_retries=int(p.resilience.max_retries),
                checkpoint_enabled=bool(p.resilience.checkpoint_enabled),
            ),
        )

        async def _enrich():
            return await enrich_and_save(
                base_df,
                output_dir=output_dir,
                jquants=bool(p.jquants),
                start_date=start,
                end_date=end,
                topix_parquet=Path(p.topix_parquet) if p.topix_parquet else None,
                enable_indices=bool(p.enable_indices),
                indices_parquet=Path(p.indices_parquet) if p.indices_parquet else None,
                indices_codes=(str(p.indices_codes).split(",") if p.indices_codes else None),
                statements_parquet=Path(p.statements_parquet) if p.statements_parquet else None,
                listed_info_parquet=Path(p.listed_info_parquet) if p.listed_info_parquet else None,
                enable_futures=bool(p.enable_futures),
                futures_parquet=Path(p.futures_parquet) if p.futures_parquet else None,
                futures_categories=list(p.futures_categories) if p.futures_categories else None,
                futures_continuous=bool(p.futures_continuous),
                nk225_parquet=Path(p.nk225_parquet) if p.nk225_parquet else None,
                reit_parquet=Path(p.reit_parquet) if p.reit_parquet else None,
                jpx400_parquet=Path(p.jpx400_parquet) if p.jpx400_parquet else None,
                enable_margin_weekly=bool(p.enable_margin_weekly),
                margin_weekly_parquet=Path(p.margin_weekly_parquet) if p.margin_weekly_parquet else None,
                margin_weekly_lag=int(p.margin_weekly_lag),
                adv_window_days=int(p.adv_window_days),
                enable_daily_margin=bool(p.enable_daily_margin),
                daily_margin_parquet=Path(p.daily_margin_parquet) if p.daily_margin_parquet else None,
                enable_macro_vix=bool(getattr(p, "enable_macro_vix", True)),
                vix_parquet=Path(p.vix_parquet) if getattr(p, "vix_parquet", None) else None,
                vix_force_refresh=bool(getattr(p, "force_refresh_vix", False)),
                enable_macro_fx_usdjpy=bool(getattr(p, "enable_fx_usdjpy", True)),
                fx_parquet=Path(p.fx_parquet) if getattr(p, "fx_parquet", None) else None,
                fx_force_refresh=bool(getattr(p, "force_refresh_fx", False)),
                enable_macro_btc=bool(getattr(p, "enable_btc", True)),
                btc_parquet=Path(p.btc_parquet) if getattr(p, "btc_parquet", None) else None,
                btc_force_refresh=bool(getattr(p, "force_refresh_btc", False)),
                am_quotes_parquet=Path(p.am_quotes_parquet) if getattr(p, "am_quotes_parquet", None) else None,
                enable_am_features=bool(getattr(p, "enable_am_features", True)),
                am_asof_policy=str(getattr(p, "am_asof_policy", "T+1")),
                enable_short_selling=bool(p.enable_short_selling),
                short_selling_parquet=Path(p.short_selling_parquet) if p.short_selling_parquet else None,
                short_positions_parquet=Path(p.short_positions_parquet) if p.short_positions_parquet else None,
                short_selling_z_window=int(p.short_selling_z_window),
                enable_earnings_events=bool(p.enable_earnings_events),
                earnings_announcements_parquet=Path(p.earnings_announcements_parquet)
                if p.earnings_announcements_parquet
                else None,
                enable_pead_features=bool(p.enable_pead_features),
                earnings_windows=list(getattr(p, "earnings_windows", [])) or None,
                earnings_asof_hour=int(getattr(p, "earnings_asof_hour", 15)),
                enable_sector_short_selling=bool(p.enable_sector_short_selling),
                sector_short_selling_parquet=Path(p.sector_short_selling_parquet)
                if p.sector_short_selling_parquet
                else None,
                enable_sector_short_z_scores=bool(p.enable_sector_short_z_scores),
                sector_onehot33=bool(p.sector_onehot33),
                sector_series_mcap=str(p.sector_series_mcap),
                sector_te_targets=list(p.sector_te_targets) if p.sector_te_targets else None,
                sector_series_levels=list(p.sector_series_levels) if p.sector_series_levels else None,
                sector_te_levels=list(p.sector_te_levels) if p.sector_te_levels else None,
                enable_advanced_vol=bool(p.enable_advanced_vol),
                adv_vol_windows=list(p.adv_vol_windows) if p.adv_vol_windows else None,
                enable_option_market_features=False,
                index_option_features_parquet=None,
                index_option_raw_parquet=None,
                enable_advanced_features=bool(p.enable_advanced_features),
                enable_sector_cs=bool(p.enable_sector_cs),
                sector_cs_cols=list(p.sector_cs_cols) if p.sector_cs_cols else None,
                enable_graph_features=bool(p.enable_graph_features),
                graph_window=int(p.graph_window),
                graph_threshold=float(p.graph_threshold),
                graph_max_k=int(p.graph_max_k),
                graph_cache_dir=str(p.graph_cache_dir) if p.graph_cache_dir else None,
                disable_halt_mask=bool(p.disable_halt_mask),
            )

        if profiler:
            with profiler.timer("enrich_and_save"):
                if p.resilience.enabled:
                    parquet_meta = await resil.execute_with_retry_async("enrich_and_save", _enrich)
                else:
                    parquet_meta = await _enrich()
        else:
            if p.resilience.enabled:
                parquet_meta = await resil.execute_with_retry_async("enrich_and_save", _enrich)
            else:
                parquet_meta = await _enrich()
        # Lineage for enrichment
        if parquet_meta and isinstance(parquet_meta, tuple):
            lineage.track_transformation(
                inputs=["base_df_in_memory"],
                output=str(parquet_meta[0]),
                transformation="enrich_and_save",
                metadata={"start": start, "end": end},
            )

        # Incremental post-merge: combine previous full with new enriched, write final
        if str(p.update_mode) == "incremental":
            old_pq, old_meta = inc.latest_artifacts()
            if old_pq and parquet_meta and isinstance(parquet_meta, tuple):
                new_pq = parquet_meta[0]
                # Merge enriched datasets
                if profiler:
                    with profiler.timer("incremental_merge"):
                        merged = inc.merge_enriched(old_pq, new_pq)
                else:
                    merged = inc.merge_enriched(old_pq, new_pq)
                # Determine overall date range
                last_start, _ = inc.read_last_date_range()
                s_for_meta = last_start or start
                e_for_meta = end
                pq, _ = save_with_symlinks(merged, output_dir, tag="full", start_date=s_for_meta, end_date=e_for_meta)
                lineage.track_transformation(
                    inputs=[str(old_pq), str(new_pq)],
                    output=str(pq),
                    transformation="incremental_merge",
                    metadata={"start": s_for_meta, "end": e_for_meta},
                )

        # Data quality checks on saved dataset (optional)
        if bool(p.quality_checks.enabled):
            qc = DataQualityChecker(
                QualityConfig(
                    outlier_threshold=float(p.quality_checks.outlier_threshold),
                    missing_threshold=float(p.quality_checks.missing_threshold),
                    save_report=bool(p.quality_checks.save_report),
                )
            )
            # Run on base_df（軽量）。必要なら最終パケットを再ロードする実装に拡張可
            res = qc.validate_dataset(base_df)
            report = qc.generate_quality_report(res)
            if qc.cfg.save_report:
                out = output_dir / "data_quality_report.md"
                out.write_text(report, encoding="utf-8")
                (output_dir / "data_quality_summary.json").write_text(
                    __import__("json").dumps(res, indent=2), encoding="utf-8"
                )
        # Save profiling report
        if profiler and bool(p.profiling.save_report):
            (output_dir / "profiling_report.json").write_text(
                __import__("json").dumps(profiler.generate_report(), indent=2), encoding="utf-8"
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
