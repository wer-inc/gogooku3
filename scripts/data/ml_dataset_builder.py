from __future__ import annotations

"""
Lightweight ML Dataset Builder used by scripts/_archive pipelines.

Provides minimal implementations required by run_pipeline_v3 while keeping
compatibility with downstream consumers. Heavy enrichments (TOPIX, flows,
statements) are treated as no-ops unless data and utilities are available.
Includes integration for margin weekly block.
"""

from pathlib import Path
from typing import Optional, Tuple
import json
import logging

import polars as pl

from src.gogooku3.features.margin_weekly import (
    add_margin_weekly_block as _add_margin_weekly_block,
)


logger = logging.getLogger(__name__)


def create_sample_data(n_stocks: int = 50, n_days: int = 120) -> pl.DataFrame:
    import numpy as np
    from datetime import datetime, timedelta

    rows = []
    base_date = datetime(2024, 1, 1)
    for i in range(n_stocks):
        code = f"{1000 + i}"
        price = 1000.0
        for d in range(n_days):
            dt = base_date + timedelta(days=d)
            if dt.weekday() >= 5:
                continue
            ret = np.random.randn() * 0.01
            price *= (1 + ret)
            rows.append(
                {
                    "Code": code,
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Open": price * (1 - 0.002),
                    "High": price * (1 + 0.003),
                    "Low": price * (1 - 0.004),
                    "Close": price,
                    "Volume": float(np.random.randint(5_000, 50_000)),
                }
            )
    df = pl.DataFrame(rows).with_columns(pl.col("Date").str.strptime(pl.Date))
    return df


class MLDatasetBuilder:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ========== Core feature stages (minimal) ==========
    def create_technical_features(self, df: pl.DataFrame) -> pl.DataFrame:
        eps = 1e-12
        df = df.with_columns(pl.col("Date").cast(pl.Date)).sort(["Code", "Date"])  # type: ignore
        # Returns
        if "Close" in df.columns:
            df = df.with_columns(
                [
                    pl.col("Close").pct_change().over("Code").alias("returns_1d"),
                    pl.col("Close").pct_change(5).over("Code").alias("returns_5d"),
                    pl.col("Close").pct_change(10).over("Code").alias("returns_10d"),
                    pl.col("Close").pct_change(20).over("Code").alias("returns_20d"),
                ]
            )
        # Row maturity index
        df = df.with_columns(pl.col("Date").cum_count().over("Code").alias("row_idx"))
        # Simple liquidity proxy
        if all(c in df.columns for c in ("Close", "Volume")):
            df = df.with_columns((pl.col("Close") * pl.col("Volume")).alias("dollar_volume"))
        return df

    def add_pandas_ta_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # Optional; keep pipeline robust without pandas_ta
        try:
            import pandas as pd
            import pandas_ta as ta  # noqa: F401
            pdf = df.to_pandas()
            # Example: short RSI if price is present
            if {"High", "Low", "Close"}.issubset(pdf.columns):
                rsi = pd.Series(pdf["Close"]).ta.rsi(length=14)
                pdf["rsi_14"] = rsi
            return pl.from_pandas(pdf)
        except Exception:
            return df

    def add_topix_features(self, df: pl.DataFrame, topix_df: Optional[pl.DataFrame], *, beta_lag: int | None = 1) -> pl.DataFrame:
        if topix_df is None or topix_df.is_empty():
            logger.info("[builder] TOPIX enrichment skipped (no data)")
            return df
        try:
            from src.features.market_features import (
                MarketFeaturesGenerator,
                CrossMarketFeaturesGenerator,
            )
            mfg = MarketFeaturesGenerator()
            market_feats = mfg.build_topix_features(topix_df)
            out = df.join(market_feats, on="Date", how="left")
            xfg = CrossMarketFeaturesGenerator(beta_lag=beta_lag or 1)
            out = xfg.attach_market_and_cross(out, market_feats)
            return out
        except Exception as e:
            logger.warning(f"[builder] TOPIX integration failed: {e}")
            return df

    # ---- Sector-related (stubs to keep callers safe) ----
    def add_sector_features(self, df: pl.DataFrame, listed_info_df: pl.DataFrame) -> pl.DataFrame:  # pragma: no cover - compatibility stub
        return df

    def add_sector_series(
        self, df: pl.DataFrame, *, level: str = "33", windows: tuple[int, int, int] = (1, 5, 20), series_mcap: str = "auto"
    ) -> pl.DataFrame:  # pragma: no cover - compatibility stub
        return df

    def add_sector_encodings(
        self, df: pl.DataFrame, *, onehot_17: bool = True, onehot_33: bool = False, freq_daily: bool = True
    ) -> pl.DataFrame:  # pragma: no cover - compatibility stub
        return df

    def add_relative_to_sector(self, df: pl.DataFrame, *, level: str = "33", x_cols: tuple[str, str] = ("returns_5d", "ma_gap_5_20")) -> pl.DataFrame:  # pragma: no cover - stub
        return df

    def add_sector_target_encoding(
        self, df: pl.DataFrame, *, target_col: str, level: str = "33", k_folds: int = 5, lag_days: int = 1, m: float = 100.0
    ) -> pl.DataFrame:  # pragma: no cover - compatibility stub
        return df

    def finalize_for_spec(self, df: pl.DataFrame) -> pl.DataFrame:  # pragma: no cover - optional normalization
        try:
            rename_map = {
                "bb_bandwidth": "bb_width",
                "bb_pct_b": "bb_position",
                "realized_vol_20": "realized_volatility",
                "vol_ma_5": "volume_ma_5",
                "vol_ma_20": "volume_ma_20",
                "vol_ratio_5d": "volume_ratio_5",
                "vol_ratio_20d": "volume_ratio_20",
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
        except Exception:
            pass
        return df

    def add_flow_features(
        self, df: pl.DataFrame, trades_df: Optional[pl.DataFrame], listed_info_df: Optional[pl.DataFrame]
    ) -> pl.DataFrame:
        if trades_df is None or trades_df.is_empty():
            logger.info("[builder] flow enrichment skipped (no trades_spec)")
            return df
        try:
            import datetime as _dt
            from src.features.flow_joiner import (
                build_flow_intervals,
                add_flow_features as _build_flow_features,
                expand_flow_daily,
                attach_flow_with_fallback,
            )

            def _next_bd(d: _dt.datetime) -> _dt.date:
                dt = d.date() if isinstance(d, _dt.datetime) else d
                wd = dt.weekday()
                if wd <= 3:
                    return (dt + _dt.timedelta(days=1))
                elif wd == 4:
                    return (dt + _dt.timedelta(days=3))
                elif wd == 5:
                    return (dt + _dt.timedelta(days=2))
                else:
                    return (dt + _dt.timedelta(days=1))

            intervals = build_flow_intervals(trades_df, _next_bd)
            flow_feat = _build_flow_features(intervals)
            bdays = df.select("Date").unique().sort("Date")["Date"].to_list()
            flow_daily = expand_flow_daily(flow_feat, bdays)
            out = attach_flow_with_fallback(df, flow_daily, section_mapper=None)
            return out
        except Exception as e:
            logger.warning(f"[builder] flow integration failed: {e}")
            return df

    def add_statements_features(self, df: pl.DataFrame, stm_df: Optional[pl.DataFrame]) -> pl.DataFrame:
        if stm_df is None or stm_df.is_empty():
            logger.info("[builder] statements enrichment skipped (no stm_df)")
            return df
        try:
            from src.features.safe_joiner_v2 import SafeJoinerV2
            sj = SafeJoinerV2()
            out = sj.join_statements_with_dedup(df, stm_df, use_time_cutoff=True, calendar_df=None)
            return out
        except Exception as e:
            logger.warning(f"[builder] statements integration failed: {e}")
            return df

    # ========== Margin weekly integration ==========
    def add_margin_weekly_block(
        self,
        df: pl.DataFrame,
        weekly_df: Optional[pl.DataFrame],
        *,
        lag_bdays_weekly: int = 3,
        adv_window_days: int = 20,
    ) -> pl.DataFrame:
        if weekly_df is None or weekly_df.is_empty():
            logger.info("[builder] margin weekly skipped (no data)")
            return df
        out = _add_margin_weekly_block(
            quotes=df,
            weekly_df=weekly_df,
            lag_bdays_weekly=lag_bdays_weekly,
            adv_window_days=adv_window_days,
        )
        # Coverage log
        try:
            cov = float(out.select(pl.col("is_margin_valid").mean()).item())
            logger.info(f"Margin feature coverage: {cov:.1%}")
        except Exception:
            pass
        return out

    # ========== IO / metadata ==========
    def create_metadata(self, df: pl.DataFrame) -> dict:
        n_rows = len(df)
        n_cols = len(df.columns)
        n_codes = df["Code"].n_unique() if "Code" in df.columns else 0
        start_date = str(df["Date"].min()) if "Date" in df.columns and n_rows else None
        end_date = str(df["Date"].max()) if "Date" in df.columns and n_rows else None
        numeric_cols = [c for c, t in zip(df.columns, df.dtypes) if t.is_numeric()]
        return {
            "rows": n_rows,
            "cols": n_cols,
            "stocks": int(n_codes),
            "date_range": {"start": start_date, "end": end_date},
            "features": {"count": len(numeric_cols)},
            "fixes_applied": [],
        }

    def save_dataset(self, df: pl.DataFrame, metadata: dict) -> Tuple[Path, Path, Path]:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pq = self.output_dir / f"ml_dataset_{ts}_full.parquet"
        csv = self.output_dir / f"ml_dataset_{ts}_full.csv"
        meta = self.output_dir / f"ml_dataset_{ts}_full_metadata.json"

        df.write_parquet(pq)
        try:
            # For small samples only; guard large frame
            if len(df) <= 2_000_000:
                df.write_csv(csv)
        except Exception:
            pass
        with open(meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        # latest symlinks
        latest_pq = self.output_dir / "ml_dataset_latest_full.parquet"
        latest_meta = self.output_dir / "ml_dataset_latest_full_metadata.json"
        for link, target in [(latest_pq, pq.name), (latest_meta, meta.name)]:
            try:
                if link.exists() or link.is_symlink():
                    link.unlink()
            except Exception:
                pass
            link.symlink_to(target)

        return pq, csv, meta
