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
import math

import polars as pl

from src.gogooku3.features.margin_weekly import (
    add_margin_weekly_block as _add_margin_weekly_block,
)
from src.gogooku3.features.margin_daily import (
    add_daily_margin_block as _add_daily_margin_block,
)
from src.gogooku3.features.short_selling import (
    add_short_selling_block as _add_short_selling_block,
)
from src.features.calendar_utils import build_next_bday_expr_from_quotes


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

        # Ensure stable dtypes/order before applying rolling operations
        cast_exprs: list[pl.Expr] = []
        if "Date" in df.columns:
            cast_exprs.append(pl.col("Date").cast(pl.Date))
        if "Code" in df.columns:
            cast_exprs.append(pl.col("Code").cast(pl.Utf8))
        if "Volume" in df.columns:
            cast_exprs.append(pl.col("Volume").cast(pl.Float64))
        if cast_exprs:
            df = df.with_columns(cast_exprs)

        if {"Code", "Date"}.issubset(df.columns):
            df = df.sort(["Code", "Date"])  # type: ignore[arg-type]

        if "row_idx" not in df.columns and "Code" in df.columns:
            df = df.with_columns(pl.cum_count().over("Code").alias("row_idx"))

        if "Close" not in df.columns or "Code" not in df.columns:
            return df

        # Backward-looking returns
        ret_exprs: list[pl.Expr] = []
        for horizon in (1, 5, 10, 20, 60, 120):
            name = f"returns_{horizon}d"
            if name not in df.columns:
                ret_exprs.append(
                    ((pl.col("Close") / pl.col("Close").shift(horizon).over("Code")) - 1.0).alias(name)
                )
        if ret_exprs:
            df = df.with_columns(ret_exprs)

        # Log returns
        log_exprs: list[pl.Expr] = []
        log_close = pl.col("Close").log()
        for horizon in (1, 5, 10, 20):
            name = f"log_returns_{horizon}d"
            if name not in df.columns:
                log_exprs.append((log_close - log_close.shift(horizon).over("Code")).alias(name))
        if log_exprs:
            df = df.with_columns(log_exprs)

        # Forward-looking labels
        fwd_exprs: list[pl.Expr] = []
        for horizon in (1, 5, 10, 20):
            name = f"feat_ret_{horizon}d"
            if name not in df.columns:
                fwd_exprs.append(
                    ((pl.col("Close").shift(-horizon).over("Code") / (pl.col("Close") + eps)) - 1.0).alias(name)
                )
        if fwd_exprs:
            df = df.with_columns(fwd_exprs)

        # Rolling simple moving averages (SMA)
        sma_exprs: list[pl.Expr] = []
        for window in (5, 10, 20, 60, 120):
            name = f"sma_{window}"
            if name not in df.columns:
                sma_exprs.append(
                    pl.col("Close")
                    .rolling_mean(window_size=window, min_periods=window)
                    .over("Code")
                    .alias(name)
                )
        if sma_exprs:
            df = df.with_columns(sma_exprs)

        # Exponential moving averages (EMA) including MACD spans
        ema_spans = {
            5: "ema_5",
            10: "ema_10",
            20: "ema_20",
            60: "ema_60",
            200: "ema_200",
            12: "_ema_fast",
            26: "_ema_slow",
        }
        ema_exprs: list[pl.Expr] = []
        for span, name in ema_spans.items():
            if name not in df.columns:
                ema_exprs.append(
                    pl.col("Close").ewm_mean(span=span, adjust=False, ignore_nulls=True).over("Code").alias(name)
                )
        if ema_exprs:
            df = df.with_columns(ema_exprs)

        # Price position relative to short SMAs
        ratio_exprs: list[pl.Expr] = []
        for window in (5, 20, 60):
            src = f"sma_{window}"
            dst = f"price_to_sma{window}"
            if src in df.columns and dst not in df.columns:
                ratio_exprs.append((pl.col("Close") / (pl.col(src) + eps)).alias(dst))
        if ratio_exprs:
            df = df.with_columns(ratio_exprs)

        # EMA gap metrics
        gap_exprs: list[pl.Expr] = []
        if all(c in df.columns for c in ("ema_5", "ema_20")) and "ma_gap_5_20" not in df.columns:
            gap_exprs.append(((pl.col("ema_5") - pl.col("ema_20")) / (pl.col("ema_20") + eps)).alias("ma_gap_5_20"))
        if all(c in df.columns for c in ("ema_20", "ema_60")) and "ma_gap_20_60" not in df.columns:
            gap_exprs.append(((pl.col("ema_20") - pl.col("ema_60")) / (pl.col("ema_60") + eps)).alias("ma_gap_20_60"))
        if gap_exprs:
            df = df.with_columns(gap_exprs)

        # Range-based intraday ratios
        range_exprs: list[pl.Expr] = []
        if "High" in df.columns and "Low" in df.columns:
            if "high_low_ratio" not in df.columns:
                range_exprs.append((pl.col("High") / (pl.col("Low") + eps)).alias("high_low_ratio"))
        if {"High", "Low", "Close"}.issubset(df.columns):
            if "close_to_high" not in df.columns:
                range_exprs.append(
                    ((pl.col("High") - pl.col("Close")) / ((pl.col("High") - pl.col("Low")) + eps)).alias("close_to_high")
                )
            if "close_to_low" not in df.columns:
                range_exprs.append(
                    ((pl.col("Close") - pl.col("Low")) / ((pl.col("High") - pl.col("Low")) + eps)).alias("close_to_low")
                )
        if range_exprs:
            df = df.with_columns(range_exprs)

        # Volume-derived features
        vol_exprs: list[pl.Expr] = []
        if "Volume" in df.columns:
            if "volume_ma_5" not in df.columns:
                vol_exprs.append(
                    pl.col("Volume")
                    .rolling_mean(window_size=5, min_periods=5)
                    .over("Code")
                    .alias("volume_ma_5")
                )
            if "volume_ma_20" not in df.columns:
                vol_exprs.append(
                    pl.col("Volume")
                    .rolling_mean(window_size=20, min_periods=20)
                    .over("Code")
                    .alias("volume_ma_20")
                )
        if vol_exprs:
            df = df.with_columns(vol_exprs)

        ratio_exprs = []
        if "volume_ma_5" in df.columns and "volume_ratio_5" not in df.columns:
            ratio_exprs.append((pl.col("Volume") / (pl.col("volume_ma_5") + eps)).alias("volume_ratio_5"))
        if "volume_ma_20" in df.columns and "volume_ratio_20" not in df.columns:
            ratio_exprs.append((pl.col("Volume") / (pl.col("volume_ma_20") + eps)).alias("volume_ratio_20"))
        if ratio_exprs:
            df = df.with_columns(ratio_exprs)

        # Turnover proxies
        if "dollar_volume" not in df.columns:
            df = df.with_columns((pl.col("Close") * pl.col("Volume")).alias("dollar_volume"))
        if "TurnoverValue" not in df.columns:
            df = df.with_columns((pl.col("Close") * pl.col("Volume")).alias("TurnoverValue"))

        # Volatility estimates based on returns
        vol_stats: list[pl.Expr] = []
        if "returns_1d" in df.columns:
            for window, name in ((5, "volatility_5d"), (10, "volatility_10d"), (20, "volatility_20d"), (60, "volatility_60d")):
                if name not in df.columns:
                    vol_stats.append(
                        (
                            pl.col("returns_1d")
                            .rolling_std(window_size=window, min_periods=window)
                            .over("Code")
                            * (252**0.5)
                        ).alias(name)
                    )
        if vol_stats:
            df = df.with_columns(vol_stats)

        # Realized volatility via Parkinson estimator (20-day window)
        if {"High", "Low"}.issubset(df.columns) and "realized_volatility" not in df.columns:
            log_hl = (pl.col("High") / pl.col("Low")).log().pow(2)
            rv = (
                log_hl
                .rolling_sum(window_size=20, min_periods=20)
                .over("Code")
                / (4.0 * math.log(2))
            ).sqrt().alias("realized_volatility")
            df = df.with_columns(rv)

        # RSI indicators
        if "row_idx" in df.columns:
            if "_delta" not in df.columns:
                df = df.with_columns(pl.col("Close").diff().over("Code").alias("_delta"))
            helper_exprs = []
            if "_gain" not in df.columns:
                helper_exprs.append(
                    pl.when(pl.col("_delta") > 0).then(pl.col("_delta")).otherwise(0.0).alias("_gain")
                )
            if "_loss" not in df.columns:
                helper_exprs.append(
                    pl.when(pl.col("_delta") < 0).then(-pl.col("_delta")).otherwise(0.0).alias("_loss")
                )
            if helper_exprs:
                df = df.with_columns(helper_exprs)

            def _rsi_expr(length: int, name: str) -> pl.Expr:
                avg_gain = pl.col("_gain").ewm_mean(span=length, adjust=False, ignore_nulls=True).over("Code")
                avg_loss = pl.col("_loss").ewm_mean(span=length, adjust=False, ignore_nulls=True).over("Code")
                raw = 100.0 - (100.0 / (1.0 + (avg_gain / (avg_loss + eps))))
                return (
                    pl.when(pl.col("row_idx") >= (length - 1)).then(raw).otherwise(None).alias(name)
                )

            if "rsi_2" not in df.columns:
                df = df.with_columns(_rsi_expr(2, "rsi_2"))
            if "rsi_14" not in df.columns:
                df = df.with_columns(_rsi_expr(14, "rsi_14"))

            drop_helpers = [c for c in ("_delta", "_gain", "_loss") if c in df.columns]
            if drop_helpers:
                df = df.drop(drop_helpers)

        if "rsi_14" in df.columns and "rsi_delta" not in df.columns:
            df = df.with_columns(pl.col("rsi_14").diff().over("Code").alias("rsi_delta"))

        # MACD (12-26-9)
        if all(c in df.columns for c in ("_ema_fast", "_ema_slow")):
            if "macd" not in df.columns:
                df = df.with_columns((pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd"))
            if "macd_signal" not in df.columns:
                df = df.with_columns(pl.col("macd").ewm_mean(span=9, adjust=False, ignore_nulls=True).over("Code").alias("macd_signal"))
            if "macd_histogram" not in df.columns:
                df = df.with_columns((pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram"))
            df = df.drop([c for c in ("_ema_fast", "_ema_slow") if c in df.columns])
        else:
            df = df.drop([c for c in ("_ema_fast", "_ema_slow") if c in df.columns])

        # Bollinger Bands (20, 2σ)
        if {"Close", "Code"}.issubset(df.columns) and "bb_pct_b" not in df.columns:
            m = pl.col("Close").rolling_mean(window_size=20, min_periods=20).over("Code")
            s = pl.col("Close").rolling_std(window_size=20, min_periods=20).over("Code")
            upper = (m + 2.0 * s).alias("_bb_upper")
            lower = (m - 2.0 * s).alias("_bb_lower")
            mid = m.alias("_bb_mid")
            df = df.with_columns([upper, lower, mid])
            df = df.with_columns(
                [
                    ((pl.col("Close") - pl.col("_bb_lower")) / ((pl.col("_bb_upper") - pl.col("_bb_lower")) + eps))
                    .clip(0.0, 1.0)
                    .alias("bb_pct_b"),
                    ((pl.col("_bb_upper") - pl.col("_bb_lower")) / (pl.col("_bb_mid") + eps)).alias("bb_bw"),
                ]
            )
            df = df.drop([c for c in ("_bb_upper", "_bb_lower", "_bb_mid") if c in df.columns])

        # Average True Range (ATR-14)
        if {"High", "Low", "Close"}.issubset(df.columns) and "atr_14" not in df.columns:
            tr = pl.max_horizontal(
                [
                    pl.col("High") - pl.col("Low"),
                    (pl.col("High") - pl.col("Close").shift(1).over("Code")).abs(),
                    (pl.col("Low") - pl.col("Close").shift(1).over("Code")).abs(),
                ]
            )
            df = df.with_columns(tr.alias("_tr"))
            df = df.with_columns(
                pl.col("_tr").ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code").alias("atr_14")
            )
            df = df.drop("_tr")

        # Stochastic %K (14)
        if {"High", "Low", "Close"}.issubset(df.columns) and "stoch_k" not in df.columns:
            lowest = pl.col("Low").rolling_min(window_size=14, min_periods=14).over("Code")
            highest = pl.col("High").rolling_max(window_size=14, min_periods=14).over("Code")
            df = df.with_columns(((pl.col("Close") - lowest) / ((highest - lowest) + eps) * 100.0).alias("stoch_k"))

        # ADX(14)
        if {"High", "Low", "Close"}.issubset(df.columns) and "adx_14" not in df.columns:
            df = df.with_columns([
                (pl.col("High") - pl.col("High").shift(1).over("Code")).alias("_up_move_raw"),
                (pl.col("Low").shift(1).over("Code") - pl.col("Low")).alias("_down_move_raw"),
            ])
            df = df.with_columns([
                pl.col("_up_move_raw").clip_min(0.0).alias("_up_move"),
                pl.col("_down_move_raw").clip_min(0.0).alias("_down_move"),
            ])
            df = df.with_columns([
                pl.when((pl.col("_up_move") > pl.col("_down_move")) & (pl.col("_up_move") > 0)).then(pl.col("_up_move")).otherwise(0.0).alias("_plus_dm"),
                pl.when((pl.col("_down_move") > pl.col("_up_move")) & (pl.col("_down_move") > 0)).then(pl.col("_down_move")).otherwise(0.0).alias("_minus_dm"),
            ])
            df = df.with_columns(
                pl.max_horizontal([
                    pl.col("High") - pl.col("Low"),
                    (pl.col("High") - pl.col("Close").shift(1).over("Code")).abs(),
                    (pl.col("Low") - pl.col("Close").shift(1).over("Code")).abs(),
                ]).alias("_true_range")
            )
            df = df.with_columns([
                pl.col("_true_range").ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code").alias("_tr14"),
                pl.col("_plus_dm").ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code").alias("_plus_dm_ewm"),
                pl.col("_minus_dm").ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code").alias("_minus_dm_ewm"),
            ])
            df = df.with_columns([
                (pl.col("_plus_dm_ewm") / (pl.col("_tr14") + eps) * 100.0).alias("_plus_di"),
                (pl.col("_minus_dm_ewm") / (pl.col("_tr14") + eps) * 100.0).alias("_minus_di"),
            ])
            df = df.with_columns(
                ((pl.col("_plus_di") - pl.col("_minus_di")).abs() / ((pl.col("_plus_di") + pl.col("_minus_di")) + eps) * 100.0).alias("_dx")
            )
            df = df.with_columns(
                pl.col("_dx").ewm_mean(span=14, adjust=False, ignore_nulls=True).over("Code").alias("adx_14")
            )
            drop_cols = [
                "_up_move_raw",
                "_down_move_raw",
                "_up_move",
                "_down_move",
                "_plus_dm",
                "_minus_dm",
                "_true_range",
                "_tr14",
                "_plus_dm_ewm",
                "_minus_dm_ewm",
                "_plus_di",
                "_minus_di",
                "_dx",
            ]
            df = df.drop([c for c in drop_cols if c in df.columns])

        return df

    def validate_forward_return_labels(self, df: pl.DataFrame) -> dict[str, any]:
        """
        Validate forward return labels quality as per PDF diagnosis requirements.

        Checks:
        1. feat_ret_1d(Code,t) == returns_1d(Code,t+1) for consistency
        2. Proper masking of trailing windows (last h rows should be NULL)
        3. No data leakage in forward-looking computation

        Returns validation results with warnings and statistics.
        """
        validation_results = {
            "passed": True,
            "warnings": [],
            "statistics": {},
            "critical_errors": []
        }

        try:
            # Check if forward return labels exist
            feat_ret_cols = [col for col in df.columns if col.startswith("feat_ret_")]
            if not feat_ret_cols:
                validation_results["critical_errors"].append("No forward return labels found (feat_ret_* columns missing)")
                validation_results["passed"] = False
                return validation_results

            # Check 1: Consistency validation - feat_ret_1d(Code,t) should equal returns_1d(Code,t+1)
            if all(col in df.columns for col in ["feat_ret_1d", "returns_1d", "Code"]):
                # Calculate shifted returns_1d for comparison
                comparison_df = df.with_columns([
                    pl.col("returns_1d").shift(-1).over("Code").alias("returns_1d_shifted_future")
                ])

                # Compare feat_ret_1d with shifted returns_1d (should be identical)
                diff_check = comparison_df.select([
                    pl.col("Code"),
                    pl.col("feat_ret_1d"),
                    pl.col("returns_1d_shifted_future"),
                    (pl.col("feat_ret_1d") - pl.col("returns_1d_shifted_future")).abs().alias("abs_diff")
                ]).filter(
                    pl.col("feat_ret_1d").is_not_null() & pl.col("returns_1d_shifted_future").is_not_null()
                )

                if diff_check.height > 0:
                    max_diff = diff_check.select(pl.col("abs_diff").max()).item()
                    mean_diff = diff_check.select(pl.col("abs_diff").mean()).item()

                    validation_results["statistics"]["feat_ret_1d_consistency_max_diff"] = max_diff
                    validation_results["statistics"]["feat_ret_1d_consistency_mean_diff"] = mean_diff

                    if max_diff > 1e-10:  # Allow for floating point precision
                        validation_results["warnings"].append(
                            f"feat_ret_1d consistency check failed: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
                        )
                        validation_results["passed"] = False
                    else:
                        validation_results["statistics"]["feat_ret_1d_consistency"] = "PASSED"

            # Check 2: Proper masking of trailing windows
            horizons = [1, 5, 10, 20]
            for h in horizons:
                col_name = f"feat_ret_{h}d"
                if col_name in df.columns:
                    # Count trailing nulls per Code (should be exactly h nulls at the end)
                    trailing_nulls = df.group_by("Code").agg([
                        pl.col(col_name).tail(h).null_count().alias(f"trailing_nulls_{h}d")
                    ])

                    # Check if all codes have exactly h trailing nulls
                    perfect_masking = trailing_nulls.filter(pl.col(f"trailing_nulls_{h}d") == h).height
                    total_codes = trailing_nulls.height

                    validation_results["statistics"][f"trailing_mask_{h}d_compliance"] = perfect_masking / total_codes if total_codes > 0 else 0

                    if perfect_masking < total_codes * 0.95:  # 95% compliance threshold
                        validation_results["warnings"].append(
                            f"Trailing window masking for {col_name}: only {perfect_masking}/{total_codes} codes properly masked"
                        )

            # Check 3: Non-null data availability
            for col_name in feat_ret_cols:
                null_pct = df.select((pl.col(col_name).null_count() / pl.len() * 100).alias("null_pct")).item()
                validation_results["statistics"][f"{col_name}_null_percentage"] = null_pct

                if null_pct > 80:  # More than 80% null is suspicious
                    validation_results["warnings"].append(f"{col_name} has {null_pct:.1f}% null values")

            # Summary statistics
            validation_results["statistics"]["forward_return_columns_found"] = len(feat_ret_cols)
            validation_results["statistics"]["total_rows"] = df.height
            validation_results["statistics"]["unique_codes"] = df.select(pl.col("Code").n_unique()).item() if "Code" in df.columns else 0

        except Exception as e:
            validation_results["critical_errors"].append(f"Validation failed with error: {str(e)}")
            validation_results["passed"] = False

        return validation_results

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

    def add_topix_features(self, df: pl.DataFrame, topix_df: Optional[pl.DataFrame] = None, *, beta_lag: int | None = 1) -> pl.DataFrame:
        """Attach TOPIX market features and cross features safely.

        This implementation prevents duplicate mkt_* columns when upstream stages
        already attached market features. If mkt_* columns are detected, we skip
        re-attaching to avoid suffix collisions (e.g., mkt_ret_1d_right) that can
        trigger hstack errors inside Polars joins.
        """
        # Early exit if market features already present (assume cross features too)
        try:
            if any(c.startswith("mkt_") for c in df.columns):
                logger.info("[builder] TOPIX features already present; skipping re-attach")
                return df
        except Exception:
            # fall through to normal flow if inspection fails
            pass

        # If TOPIX series is missing, build a robust market proxy from equities to avoid all-null mkt_*.
        if topix_df is None or topix_df.is_empty():
            try:
                if {"Date", "Code", "Close"}.issubset(df.columns):
                    # Use median 1d return across stocks per day as market return, then cumprod into synthetic Close
                    if "returns_1d" not in df.columns:
                        eq = df.select([
                            pl.col("Date"),
                            ((pl.col("Close") / pl.col("Close").shift(1).over("Code")) - 1.0).alias("returns_1d"),
                        ])
                    else:
                        eq = df.select(["Date", "returns_1d"])  # type: ignore[list-item]
                    mkt = (
                        eq.group_by("Date")
                        .agg(pl.col("returns_1d").median().alias("mkt_ret_1d"))
                        .sort("Date")
                        .with_columns(((1.0 + pl.col("mkt_ret_1d")).cum_prod() * 100.0).alias("Close"))
                        .select(["Date", "Close"])  # Provide Close-only; ATR branch will fall back
                    )
                    topix_df = mkt
                    logger.info("[builder] Built synthetic TOPIX proxy from equities (median returns)")
                else:
                    logger.info("[builder] TOPIX enrichment skipped (no data and cannot build proxy)")
                    return df
            except Exception as e:
                logger.warning(f"[builder] TOPIX proxy build failed: {e}")
                return df
        try:
            from src.features.market_features import (
                MarketFeaturesGenerator,
                CrossMarketFeaturesGenerator,
            )

            mfg = MarketFeaturesGenerator()
            market_feats = mfg.build_topix_features(topix_df)

            # Drop any columns from right that already exist on the left to prevent
            # suffix duplication (and downstream hstack errors).
            try:
                dup_cols = [c for c in market_feats.columns if c in df.columns and c != "Date"]
                if dup_cols:
                    market_feats = market_feats.drop(dup_cols)
                    logger.info(f"[builder] Dropped duplicate market cols prior to join: {dup_cols[:4]}{'...' if len(dup_cols)>4 else ''}")
            except Exception:
                pass

            out = df.join(market_feats, on="Date", how="left")

            # Compute cross features (beta/alpha/relative) using the market frame
            xfg = CrossMarketFeaturesGenerator(beta_lag=beta_lag or 1)
            out = xfg.attach_market_and_cross(out, market_feats)
            return out
        except Exception as e:
            logger.warning(f"[builder] TOPIX integration failed: {e}")
            return df

    def add_index_features(
        self,
        df: pl.DataFrame,
        indices_df: Optional[pl.DataFrame],
        *,
        mask_halt_day: bool = True,
    ) -> pl.DataFrame:
        """Attach cross-index day-level features (spreads, breadth) to equities.

        This function computes per-index features and daily aggregates and joins
        only the day-level aggregates (shared across equities) to avoid the need
        for per-stock mapping. Sector/style/size-specific joins can be added by
        callers if mapping data is available.
        """
        if indices_df is None or indices_df.is_empty():
            logger.info("[builder] indices enrichment skipped (no indices_df)")
            return df
        try:
            from src.gogooku3.features.index_features import (
                build_all_index_features,
                attach_index_features_to_equity,
            )
            per_index, daily = build_all_index_features(indices_df, mask_halt_day=mask_halt_day)
            if daily is None or daily.is_empty():
                logger.info("[builder] indices: no daily aggregates; skipping attach")
                return df
            out = attach_index_features_to_equity(df, per_index, daily)
            return out
        except Exception as e:
            logger.warning(f"[builder] indices integration failed: {e}")
            return df

    def add_sector_index_features(
        self,
        df: pl.DataFrame,
        indices_df: Optional[pl.DataFrame],
        listed_info_df: Optional[pl.DataFrame],
        *,
        prefix: str = "sect_",
        mask_halt_day: bool = True,
    ) -> pl.DataFrame:
        """Attach sector index features using listed_info mapping.

        Joins a subset of per-index SECTOR-family features onto equities via
        (Date, SectorIndexCode), with columns prefixed by `prefix`.
        """
        if indices_df is None or indices_df.is_empty() or listed_info_df is None or listed_info_df.is_empty():
            logger.info("[builder] sector index enrichment skipped (missing indices or listed_info)")
            return df
        try:
            from src.gogooku3.features.index_features import attach_sector_index_features
            out = attach_sector_index_features(df, indices_df, listed_info_df, prefix=prefix, mask_halt_day=mask_halt_day)
            return out
        except Exception as e:
            logger.warning(f"[builder] sector index integration failed: {e}")
            return df

    # ---- Sector-related (now with actual implementation) ----
    def add_sector_features(self, df: pl.DataFrame, listed_info_df: pl.DataFrame) -> pl.DataFrame:
        """Attach sector metadata (sections, sector codes, shares) from listed_info."""

        if listed_info_df is None or listed_info_df.is_empty():
            logger.warning("[builder] sector enrichment skipped (no listed_info)")
            return df

        result = df
        eps = 1e-12

        # Step 1: attach Section using SectionMapper (interval-aware)
        try:
            from src.features.section_mapper import SectionMapper

            mapper = SectionMapper()
            mapping_df = mapper.create_section_mapping(listed_info_df)
            result = mapper.attach_section_to_daily(df, mapping_df)

            coverage_stats = mapper.validate_section_coverage(result)
            logger.info(f"[builder] Section coverage: {coverage_stats['section_coverage']:.1%}")
        except Exception as exc:
            logger.warning(f"[builder] section mapping failed ({exc}); continuing without Section enrichment")
            result = df

        # Step 2: as-of join sector codes/names/shares
        try:
            cols = set(listed_info_df.columns)
            code_col = next((c for c in ("Code", "LocalCode", "code") if c in cols), None)
            date_col = next((c for c in ("Date", "date", "EffectiveDate", "EffectiveFrom", "ListedDate", "valid_from") if c in cols), None)
            if code_col is None or date_col is None:
                logger.warning("[builder] listed_info missing Code/Date column; skipping sector metadata join")
                return result

            exprs: list[pl.Expr] = [
                pl.col(code_col).cast(pl.Utf8).alias("Code"),
                pl.col(date_col).cast(pl.Date).alias("valid_from"),
            ]
            used_alias = {"Code", "valid_from"}

            alias_candidates = [
                ("MarketCode", "MarketCode", pl.Utf8),
                ("MarketName", "MarketName", pl.Utf8),
                ("Sector33CodeName", "sector33_name", pl.Utf8),
                ("Sector33Code", "sector33_code", pl.Utf8),
                ("Sector33Name", "sector33_name", pl.Utf8),
                ("Sector33NameEnglish", "sector33_name", pl.Utf8),
                ("Sector17CodeName", "sector17_name", pl.Utf8),
                ("Sector17Code", "sector17_code", pl.Utf8),
                ("Sector17Name", "sector17_name", pl.Utf8),
                ("Sector17NameEnglish", "sector17_name", pl.Utf8),
                ("CompanyName", "CompanyName", pl.Utf8),
            ]

            for src, alias, dtype in alias_candidates:
                if src in cols and alias not in used_alias:
                    exprs.append(pl.col(src).cast(dtype).alias(alias))
                    used_alias.add(alias)

            shares_col = next(
                (c for c in (
                    "SharesOutstanding",
                    "shares_outstanding",
                    "NumberOfIssuedShares",
                    "NumberOfListedShares",
                    "IssuedShareNumber",
                    "IssuedShareNumberOfListing",
                ) if c in cols),
                None,
            )
            if shares_col and "shares_outstanding" not in used_alias:
                exprs.append(pl.col(shares_col).cast(pl.Float64).alias("shares_outstanding"))
                used_alias.add("shares_outstanding")

            info = listed_info_df.select(exprs).drop_nulls(subset=["Code", "valid_from"])
            if info.is_empty():
                return result

            try:
                unique_valid_from = info.select(pl.col("valid_from").n_unique()).item()
            except Exception:
                unique_valid_from = None
            if unique_valid_from == 1 and "Date" in df.columns:
                try:
                    dataset_start = df.select(pl.col("Date").cast(pl.Date).min()).item()
                except Exception:
                    dataset_start = None
                if dataset_start is not None:
                    info = info.with_columns(
                        pl.lit(dataset_start).cast(pl.Date).alias("valid_from")
                    )
                    logger.info(
                        "[builder] Single listed_info snapshot detected; backfilled valid_from to dataset start %s",
                        dataset_start,
                    )

            info = info.sort(["Code", "valid_from"]).with_columns(
                pl.col("valid_from").shift(-1).over("Code").alias("next_change")
            )
            info = info.with_columns(
                pl.when(pl.col("next_change").is_not_null())
                .then(pl.col("next_change") - pl.duration(days=1))
                .otherwise(pl.date(2999, 12, 31))
                .alias("valid_to")
            ).drop("next_change")

            joined = result.with_columns([
                pl.col("Code").cast(pl.Utf8),
                pl.col("Date").cast(pl.Date),
            ]).sort(["Code", "Date"])  # type: ignore[arg-type]

            joined = joined.join_asof(
                info,
                by="Code",
                left_on="Date",
                right_on="valid_from",
                strategy="backward",
                suffix="_info",
            )

            if "valid_to" in joined.columns:
                joined = joined.filter(
                    (pl.col("valid_to").is_null()) | (pl.col("Date") <= pl.col("valid_to"))
                )
                joined = joined.drop([c for c in ("valid_from", "valid_to") if c in joined.columns])

            # Prefer info columns when available
            for col in [
                "MarketCode",
                "MarketName",
                "CompanyName",
                "sector33_code",
                "sector33_name",
                "sector17_code",
                "sector17_name",
                "shares_outstanding",
            ]:
                info_col = f"{col}_info"
                if info_col in joined.columns:
                    if col in joined.columns:
                        joined = joined.with_columns(pl.coalesce([pl.col(info_col), pl.col(col)]).alias(col))
                    else:
                        joined = joined.rename({info_col: col})
                    joined = joined.drop(info_col)

            # Stable categorical IDs
            for src, dst in (("sector33_code", "sector33_id"), ("sector17_code", "sector17_id")):
                if src in joined.columns and dst not in joined.columns:
                    codes = joined.select(pl.col(src)).to_series().drop_nulls().unique().to_list()
                    codes = [c for c in codes if c is not None]
                    if codes:
                        mapping = {code: idx for idx, code in enumerate(sorted(codes))}
                        joined = joined.with_columns(
                            pl.when(pl.col(src).is_null())
                            .then(-1)
                            .otherwise(pl.col(src).replace(mapping, default=-1))
                            .cast(pl.Int32)
                            .alias(dst)
                        )
                    else:
                        joined = joined.with_columns(pl.lit(-1).cast(pl.Int32).alias(dst))

            # Compute turnover rate when shares outstanding available
            if {"Volume", "shares_outstanding"}.issubset(joined.columns) and "turnover_rate" not in joined.columns:
                joined = joined.with_columns(
                    pl.when(pl.col("shares_outstanding") > 0)
                    .then(pl.col("Volume") / (pl.col("shares_outstanding") + eps))
                    .otherwise(None)
                    .alias("turnover_rate")
                )

            attached = (
                joined.select(pl.col("sector33_code").is_not_null().sum().alias("cnt"))
                .item()
                if "sector33_code" in joined.columns
                else 0
            )
            if attached:
                logger.info(f"[builder] Sector metadata attached for {attached} rows")

            result = joined

        except Exception as exc:
            logger.warning(f"[builder] sector metadata join failed: {exc}")

        return result

    def add_interaction_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add cross-feature interactions (market/sector/flow gates and composites)."""

        eps = 1e-12
        features: list[pl.Expr] = []

        def has(*cols: str) -> bool:
            return all(c in df.columns for c in cols)

        def hinge_pos(col: str) -> pl.Expr:
            return pl.col(col).clip_min(0.0)

        def hinge_neg(col: str) -> pl.Expr:
            return (-pl.col(col)).clip_min(0.0)

        # 1) Market × individual trend alignment
        if has("ma_gap_5_20", "mkt_gap_5_20") and "x_trend_intensity" not in df.columns:
            features.append((pl.col("ma_gap_5_20") * pl.col("mkt_gap_5_20")).alias("x_trend_intensity"))

        if has("ma_gap_5_20", "mkt_trend_up") and "x_trend_intensity_g" not in df.columns:
            features.append(
                (
                    pl.col("ma_gap_5_20")
                    * (pl.col("mkt_trend_up").cast(pl.Float64).fill_null(0.0) * 2.0 - 1.0)
                ).alias("x_trend_intensity_g")
            )

        # 2) Sector relative momentum alignment
        if has("rel_to_sec_5d", "sec_mom_20") and "x_rel_sec_mom" not in df.columns:
            features.append((pl.col("rel_to_sec_5d") * pl.col("sec_mom_20")).alias("x_rel_sec_mom"))

        if has("z_in_sec_ma_gap_5_20", "sec_mom_20") and "x_z_sec_gap_mom" not in df.columns:
            features.append((pl.col("z_in_sec_ma_gap_5_20") * pl.col("sec_mom_20")).alias("x_z_sec_gap_mom"))

        # 3) Risk-adjusted momentum (local Sharpe)
        if has("returns_5d", "volatility_20d") and "x_mom_sh_5" not in df.columns:
            features.append((pl.col("returns_5d") / (pl.col("volatility_20d") + eps)).alias("x_mom_sh_5"))

        if has("returns_10d", "volatility_20d") and "x_mom_sh_10" not in df.columns:
            features.append((pl.col("returns_10d") / (pl.col("volatility_20d") + eps)).alias("x_mom_sh_10"))

        if has("returns_5d", "beta_60d", "mkt_ret_5d", "volatility_20d") and "x_mom_sh_5_mktneu" not in df.columns:
            features.append(
                (
                    (pl.col("returns_5d") - pl.col("beta_60d") * pl.col("mkt_ret_5d"))
                    / (pl.col("volatility_20d") + eps)
                ).alias("x_mom_sh_5_mktneu")
            )

        # 4) Volume shock × price direction/bollinger position
        if has("volume_ratio_5", "returns_1d") and "x_rvol5_dir" not in df.columns:
            returns_sign = (
                pl.when(pl.col("returns_1d").is_null())
                .then(None)
                .when(pl.col("returns_1d") > 0)
                .then(1.0)
                .when(pl.col("returns_1d") < 0)
                .then(-1.0)
                .otherwise(0.0)
            )
            features.append((pl.col("volume_ratio_5") * returns_sign).alias("x_rvol5_dir"))

        if has("volume_ratio_5", "bb_pct_b") and "x_rvol5_bb" not in df.columns:
            features.append((pl.col("volume_ratio_5") * pl.col("bb_pct_b")).alias("x_rvol5_bb"))

        # 5) Short squeeze pressure
        if has("dmi_short_to_adv20", "rel_strength_5d") and "x_squeeze_pressure" not in df.columns:
            features.append((pl.col("dmi_short_to_adv20") * hinge_pos("rel_strength_5d")).alias("x_squeeze_pressure"))

        # 6) Credit flow bias × reversal gate
        if has("dmi_credit_ratio", "z_close_20", "Code") and "x_credit_rev_bias" not in df.columns:
            credit_bias = (pl.col("dmi_credit_ratio") - 1.0).rolling_mean(window_size=26, min_periods=1).over("Code")
            features.append((credit_bias.fill_null(0.0) * hinge_neg("z_close_20")).alias("x_credit_rev_bias"))

        # 7) PEAD decay
        if has("stmt_rev_fore_op", "stmt_progress_op", "stmt_days_since_statement") and "x_pead_effect" not in df.columns:
            pead_base = pl.col("stmt_rev_fore_op").fill_null(0.0) + pl.col("stmt_progress_op").fill_null(0.0)
            pead_decay = (-pl.col("stmt_days_since_statement") / 5.0).exp()
            features.append((pead_base * pead_decay).alias("x_pead_effect"))

        if has("stmt_rev_fore_op", "stmt_progress_op", "stmt_days_since_statement", "mkt_trend_up") and "x_pead_times_mkt" not in df.columns:
            pead_base = pl.col("stmt_rev_fore_op").fill_null(0.0) + pl.col("stmt_progress_op").fill_null(0.0)
            pead_decay = (-pl.col("stmt_days_since_statement") / 5.0).exp()
            market_gate = pl.col("mkt_trend_up").cast(pl.Float64).fill_null(0.0) * 2.0 - 1.0
            features.append((pead_base * pead_decay * market_gate).alias("x_pead_times_mkt"))

        # 8) Regime gates (volatility vs breakout)
        if has("mkt_high_vol", "z_close_20") and "x_rev_gate" not in df.columns:
            features.append(
                (pl.col("mkt_high_vol").cast(pl.Float64).fill_null(0.0) * hinge_neg("z_close_20")).alias("x_rev_gate")
            )

        if has("mkt_high_vol", "ma_gap_5_20") and "x_bo_gate" not in df.columns:
            bull_gate = 1.0 - pl.col("mkt_high_vol").cast(pl.Float64).fill_null(0.0)
            features.append((bull_gate * pl.col("ma_gap_5_20").gt(0).cast(pl.Float64)).alias("x_bo_gate"))

        # 9) Alpha mean reversion weighted by beta stability
        if has("alpha_1d", "beta_stability_60d") and "x_alpha_meanrev_stable" not in df.columns:
            features.append(((-pl.col("alpha_1d")) * pl.col("beta_stability_60d")).alias("x_alpha_meanrev_stable"))

        # 10) Weekly flow × relative strength
        if has("flow_smart_idx", "rel_strength_5d") and "x_flow_smart_rel" not in df.columns:
            features.append((pl.col("flow_smart_idx") * pl.col("rel_strength_5d")).alias("x_flow_smart_rel"))

        if has("flow_foreign_net_z", "rel_to_sec_5d") and "x_foreign_relsec" not in df.columns:
            features.append((pl.col("flow_foreign_net_z") * pl.col("rel_to_sec_5d")).alias("x_foreign_relsec"))

        # 11) Three-layer alignment (market, sector, individual)
        if has("mkt_gap_5_20", "sec_mom_20", "ma_gap_5_20") and "x_tri_align" not in df.columns:
            features.append(
                (
                    pl.col("mkt_gap_5_20").gt(0).cast(pl.Float64)
                    * pl.col("sec_mom_20").gt(0).cast(pl.Float64)
                    * pl.col("ma_gap_5_20")
                ).alias("x_tri_align")
            )

        # 12) Bollinger hinge × relative volume
        if has("bb_pct_b", "volume_ratio_5") and "x_bbpos_rvol5" not in df.columns:
            features.append((hinge_pos("bb_pct_b") * pl.col("volume_ratio_5")).alias("x_bbpos_rvol5"))

        if has("bb_pct_b", "volume_ratio_5") and "x_bbneg_rvol5" not in df.columns:
            features.append((hinge_neg("bb_pct_b") * pl.col("volume_ratio_5")).alias("x_bbneg_rvol5"))

        # 13) Liquidity shock × momentum
        if has("turnover_rate", "returns_5d", "Code") and "x_liquidityshock_mom" not in df.columns:
            shock = (
                (pl.col("turnover_rate") / (pl.col("turnover_rate").shift(1).over("Code") + eps) - 1.0)
                .clip(-0.5, 0.5)
            )
            features.append((shock * pl.col("returns_5d")).alias("x_liquidityshock_mom"))

        # 16) DMI impulse × return direction
        if has("dmi_impulse", "returns_1d") and "x_dmi_impulse_dir" not in df.columns:
            features.append(
                (pl.col("dmi_impulse").cast(pl.Float64).fill_null(0.0) * pl.col("returns_1d")).alias("x_dmi_impulse_dir")
            )

        # 17) Breadth × relative strength
        if has("flow_breadth_pos", "rel_strength_5d") and "x_breadth_rel" not in df.columns:
            features.append((pl.col("flow_breadth_pos") * pl.col("rel_strength_5d")).alias("x_breadth_rel"))

        if features:
            df = df.with_columns(features)

        return df

    def add_sector_series(
        self,
        df: pl.DataFrame,
        *,
        level: str = "33",
        windows: tuple[int, int, int] = (1, 5, 20),
        series_mcap: str = "auto",
    ) -> pl.DataFrame:
        """Compute sector-level aggregate series (momentum/volatility) and attach to rows."""

        id_col = "sector33_id" if level == "33" else "sector17_id"
        if id_col not in df.columns:
            logger.warning(f"[{level}] sector id column '{id_col}' missing; skipping sector series")
            return df

        if "Date" not in df.columns:
            logger.warning("Date column missing; cannot compute sector series")
            return df

        eps = 1e-12

        try:
            # Base equal-weighted sector returns
            agg_cols = ["returns_1d"]
            optional_cols = ["returns_5d", "returns_20d"]
            available_optional = [c for c in optional_cols if c in df.columns]
            select_cols = ["Date", id_col] + agg_cols + available_optional
            sec_daily = (
                df.select(select_cols)
                .group_by(["Date", id_col])
                .agg(
                    [pl.col("returns_1d").median().alias("sec_ret_1d_eq")] +
                    [pl.col(c).median().alias(f"sec_ret_{c.split('_')[1]}") for c in available_optional]
                    + [pl.len().alias("sec_member_cnt")]
                )
                .sort([id_col, "Date"])
            )

            # Standardize column names for optional aggregates
            if "sec_ret_5d" in sec_daily.columns:
                sec_daily = sec_daily.rename({"sec_ret_5d": "sec_ret_5d_eq"})
            if "sec_ret_20d" in sec_daily.columns:
                sec_daily = sec_daily.rename({"sec_ret_20d": "sec_ret_20d_eq"})

            # Derive member-based flags
            sec_daily = sec_daily.with_columns(
                (pl.col("sec_member_cnt") < 5).cast(pl.Int8).alias("sec_small_flag")
            )

            # Multi-day compounded returns (equal weight)
            def _add_window(pdf: pl.DataFrame, horizon: int) -> pl.DataFrame:
                if horizon <= 1:
                    return pdf
                return pdf.with_columns(
                    (
                        (1.0 + pl.col("sec_ret_1d_eq"))
                        .log()
                        .rolling_sum(window_size=horizon, min_periods=horizon)
                        .over(id_col)
                        .exp()
                        - 1.0
                    ).alias(f"sec_ret_{horizon}d_eq")
                )

            for w in windows:
                sec_daily = _add_window(sec_daily, w)

            # Optional market-cap weighted sector returns
            def _compute_mcap_series() -> pl.DataFrame:
                if "shares_outstanding" not in df.columns:
                    return pl.DataFrame()
                tmp = df
                if "returns_1d" not in tmp.columns and "Close" in tmp.columns:
                    tmp = tmp.with_columns(pl.col("Close").pct_change().over("Code").alias("returns_1d"))
                if "returns_1d" not in tmp.columns:
                    return pl.DataFrame()
                tmp = tmp.with_columns((pl.col("Close") * pl.col("shares_outstanding")).alias("mcap"))
                return (
                    tmp.select(["Date", id_col, "returns_1d", "mcap"])
                    .group_by(["Date", id_col])
                    .agg([
                        (pl.col("returns_1d") * pl.col("mcap")).sum().alias("num"),
                        pl.col("mcap").sum().alias("den"),
                    ])
                    .with_columns((pl.col("num") / (pl.col("den") + eps)).alias("sec_ret_1d_mcap"))
                    .select(["Date", id_col, "sec_ret_1d_mcap"])
                    .sort([id_col, "Date"])
                )

            use_mcap = series_mcap in ("auto", "always")
            sec_mcap = pl.DataFrame()
            if use_mcap:
                sec_mcap = _compute_mcap_series()
                if not sec_mcap.is_empty():
                    sec_daily = sec_daily.join(sec_mcap, on=["Date", id_col], how="left")

                    def _add_mcap_window(pdf: pl.DataFrame, horizon: int) -> pl.DataFrame:
                        if horizon <= 1 or "sec_ret_1d_mcap" not in pdf.columns:
                            return pdf
                        return pdf.with_columns(
                            (
                                (1.0 + pl.col("sec_ret_1d_mcap"))
                                .log()
                                .rolling_sum(window_size=horizon, min_periods=horizon)
                                .over(id_col)
                                .exp()
                                - 1.0
                            ).alias(f"sec_ret_{horizon}d_mcap")
                        )

                    for w in windows:
                        sec_daily = _add_mcap_window(sec_daily, w)
                elif series_mcap == "always":
                    logger.warning("series_mcap='always' requested but shares_outstanding data unavailable")

            # Momentum / EMA / volatility features on sector returns
            sec_daily = sec_daily.with_columns([
                pl.col("sec_ret_1d_eq").rolling_sum(window_size=20, min_periods=20).over(id_col).alias("sec_mom_20"),
                pl.col("sec_ret_1d_eq").ewm_mean(span=5, adjust=False).over(id_col).alias("sec_ema_5"),
                pl.col("sec_ret_1d_eq").ewm_mean(span=20, adjust=False).over(id_col).alias("sec_ema_20"),
                (
                    pl.col("sec_ret_1d_eq").rolling_std(window_size=20, min_periods=20).over(id_col)
                    * math.sqrt(252.0)
                ).alias("sec_vol_20"),
            ]).with_columns(
                ((pl.col("sec_ema_5") - pl.col("sec_ema_20")) / (pl.col("sec_ema_20") + eps)).alias("sec_gap_5_20")
            )

            mu252 = pl.col("sec_vol_20").rolling_mean(window_size=252, min_periods=252).over(id_col)
            sd252 = pl.col("sec_vol_20").rolling_std(window_size=252, min_periods=252).over(id_col) + eps
            sec_daily = sec_daily.with_columns(((pl.col("sec_vol_20") - mu252) / sd252).alias("sec_vol_20_z"))

            # Rename when working on 17-level series to avoid clashes
            if level == "17":
                rename_map = {c: f"sec17_{c[4:]}" for c in sec_daily.columns if c.startswith("sec_")}
                if rename_map:
                    sec_daily = sec_daily.rename(rename_map)

            # Join back to main frame
            join_cols = [col for col in sec_daily.columns if col != id_col and col != "Date"]
            out = df.join(sec_daily, on=["Date", id_col], how="left")

            # Ensure sector flags persist (rename for level 33 only)
            if level == "33":
                if "sec_member_cnt" not in out.columns and "sec_member_cnt" in join_cols:
                    pass  # already attached via join
            return out
        except Exception as exc:
            logger.error(f"Error computing sector series ({exc})")
            return df

    def add_sector_encodings(
        self,
        df: pl.DataFrame,
        *,
        onehot_17: bool = True,
        onehot_33: bool = False,
        freq_daily: bool = True,
        rare_threshold: float = 0.005,
    ) -> pl.DataFrame:
        """Add sector categorical encodings (one-hot and daily frequencies)."""

        out = df
        n_rows = len(out)
        if n_rows == 0:
            return out

        try:
            eps = 1e-12
            # 17-sector one-hot (rare categories folded into "other")
            if onehot_17 and "sector17_id" in out.columns:
                counts = out.group_by("sector17_id").count().rename({"count": "cnt"})
                counts = counts.with_columns((pl.col("cnt") / float(n_rows)).alias("ratio"))
                keep_ids = set(counts.filter(pl.col("ratio") >= rare_threshold)["sector17_id"].to_list())
                for k in sorted(keep_ids):
                    col_name = f"sec17_onehot_{k}"
                    if col_name not in out.columns:
                        out = out.with_columns((pl.col("sector17_id") == pl.lit(k)).cast(pl.Int8).alias(col_name))
                out = out.with_columns(
                    (~pl.col("sector17_id").is_in(list(keep_ids))).cast(pl.Int8).alias("sec17_onehot_other")
                )

            # 33-sector one-hot (optional)
            if onehot_33 and "sector33_id" in out.columns:
                counts33 = out.group_by("sector33_id").count().rename({"count": "cnt"})
                counts33 = counts33.with_columns((pl.col("cnt") / float(n_rows)).alias("ratio"))
                keep33 = set(counts33.filter(pl.col("ratio") >= rare_threshold)["sector33_id"].to_list())
                for k in sorted(keep33):
                    col_name = f"sec33_onehot_{k}"
                    if col_name not in out.columns:
                        out = out.with_columns((pl.col("sector33_id") == pl.lit(k)).cast(pl.Int8).alias(col_name))
                out = out.with_columns(
                    (~pl.col("sector33_id").is_in(list(keep33))).cast(pl.Int8).alias("sec33_onehot_other")
                )

            # Daily frequency encodings (share of listings per day)
            if freq_daily and "Date" in out.columns:
                if "sector17_id" in out.columns and "sec17_daily_freq" not in out.columns:
                    out = out.with_columns([
                        pl.len().over(["Date", "sector17_id"]).alias("_cnt17"),
                        pl.len().over("Date").alias("_tot17"),
                    ])
                    out = out.with_columns(
                        (pl.col("_cnt17") / (pl.col("_tot17") + eps)).alias("sec17_daily_freq")
                    ).drop(["_cnt17", "_tot17"])

                if "sector33_id" in out.columns and "sec33_daily_freq" not in out.columns:
                    out = out.with_columns([
                        pl.len().over(["Date", "sector33_id"]).alias("_cnt33"),
                        pl.len().over("Date").alias("_tot33"),
                    ])
                    out = out.with_columns(
                        (pl.col("_cnt33") / (pl.col("_tot33") + eps)).alias("sec33_daily_freq")
                    ).drop(["_cnt33", "_tot33"])

            return out
        except Exception as exc:
            logger.warning(f"Failed to add sector encodings: {exc}")
            return df

    def add_relative_to_sector(
        self,
        df: pl.DataFrame,
        *,
        level: str = "33",
        x_cols: tuple[str, ...] = ("returns_5d", "ma_gap_5_20"),
    ) -> pl.DataFrame:
        """Add per-sector relative features (demeaned, z-scored, sector betas)."""

        id_col = "sector33_id" if level == "33" else "sector17_id"
        if id_col not in df.columns:
            logger.warning(f"[{level}] sector id column '{id_col}' missing; skipping relative-to-sector features")
            return df

        try:
            eps = 1e-12
            out = df

            # Ensure required columns exist
            if "returns_5d" not in out.columns and "Close" in out.columns:
                out = out.with_columns(
                    ((pl.col("Close").shift(-5).over("Code") / pl.col("Close")) - 1.0).alias("returns_5d")
                )

            # Relative performance vs sector
            if {"returns_5d", "sec_ret_5d_eq", id_col}.issubset(out.columns) and "rel_to_sec_5d" not in out.columns:
                out = out.with_columns(
                    (pl.col("returns_5d") - pl.col("sec_ret_5d_eq")).alias("rel_to_sec_5d")
                )

            # Sector beta (60d) and alpha vs sector
            if {"returns_1d", "sec_ret_1d_eq"}.issubset(out.columns) and f"beta_to_sec_60" not in out.columns:
                out = out.sort(["Code", "Date"]).with_columns([
                    pl.col("returns_1d").rolling_mean(window_size=60, min_periods=1).over("Code").alias("x_mean"),
                    pl.col("sec_ret_1d_eq").rolling_mean(window_size=60, min_periods=1).over("Code").alias("y_mean"),
                    (pl.col("returns_1d") * pl.col("sec_ret_1d_eq")).rolling_mean(window_size=60, min_periods=1).over("Code").alias("xy_mean"),
                    pl.col("sec_ret_1d_eq").pow(2).rolling_mean(window_size=60, min_periods=1).over("Code").alias("y2_mean"),
                ]).with_columns([
                    (pl.col("xy_mean") - pl.col("x_mean") * pl.col("y_mean")).alias("cov_xy"),
                    (pl.col("y2_mean") - pl.col("y_mean") ** 2).alias("var_y"),
                ]).with_columns([
                    (pl.col("cov_xy") / (pl.col("var_y") + eps)).alias("beta_to_sec_60"),
                ]).with_columns([
                    (pl.col("returns_1d") - pl.col("beta_to_sec_60") * pl.col("sec_ret_1d_eq")).alias("alpha_vs_sec_1d"),
                ]).drop(["x_mean", "y_mean", "xy_mean", "y2_mean", "cov_xy", "var_y"])

            # Demeaned returns within sector per day
            if "returns_1d" in out.columns and "ret_1d_demeaned" not in out.columns:
                out = out.with_columns(
                    pl.col("returns_1d").mean().over(["Date", id_col]).alias("ret_1d_mean_sec")
                ).with_columns(
                    (pl.col("returns_1d") - pl.col("ret_1d_mean_sec")).alias("ret_1d_demeaned")
                ).drop("ret_1d_mean_sec")

            # Sector-wise z-scores for selected columns (defaults include ma_gap_5_20)
            for col_name in x_cols:
                if col_name in out.columns:
                    mu = pl.col(col_name).mean().over(["Date", id_col])
                    sd = pl.col(col_name).std(ddof=0).over(["Date", id_col]) + eps
                    z_col = f"z_in_sec_{col_name}"
                    if z_col not in out.columns:
                        out = out.with_columns(((pl.col(col_name) - mu) / sd).alias(z_col))

            return out
        except Exception as exc:
            logger.error(f"Error adding relative-to-sector features ({exc})")
            return df

    def add_sector_target_encoding(
        self,
        df: pl.DataFrame,
        *,
        target_col: str = "target_5d",
        level: str = "33",
        k_folds: int = 5,
        lag_days: int = 1,
        m: float = 100.0,
    ) -> pl.DataFrame:
        """Add sector-level target encoding with cross-fit, lag, and Bayesian smoothing.

        This aligns with docs/ml/dataset_new.md (v1.1) which requires
        `te{level}_sec_{target}` columns (and alias te_sec_* when only one
        level is requested).
        """

        try:
            if k_folds <= 1:
                k_folds = 2

            id_col = "sector33_id" if level == "33" else "sector17_id"
            if id_col not in df.columns:
                logger.warning(f"{id_col} not present; skipping sector target encoding")
                return df

            computed_target = False
            if target_col not in df.columns:
                if "Close" in df.columns and "Code" in df.columns:
                    try:
                        if target_col == "target_5d":
                            df = df.with_columns(
                                (pl.col("Close").shift(-5).over("Code") / pl.col("Close") - 1).alias("target_5d")
                            )
                            computed_target = True
                        elif target_col == "target_1d":
                            df = df.with_columns(
                                (pl.col("Close").shift(-1).over("Code") / pl.col("Close") - 1).alias("target_1d")
                            )
                            computed_target = True
                        else:
                            logger.warning(f"{target_col} missing and cannot be derived; skipping sector target encoding")
                            return df
                    except Exception as exc:
                        logger.warning(f"Failed to derive {target_col}: {exc}")
                        return df
                else:
                    logger.warning(f"{target_col} missing; skipping sector target encoding")
                    return df

            # Guard against null-heavy targets (if everything is null we bail out early)
            try:
                non_null = df.select(pl.col(target_col).is_not_null().sum()).item()
                if non_null == 0:
                    logger.warning(f"{target_col} all null; skipping sector target encoding")
                    return df
            except Exception:
                pass

            x = df.with_columns([
                pl.col("Date").cast(pl.Date),
                pl.col("Code").cast(pl.Utf8) if "Code" in df.columns else pl.lit("").alias("Code"),
            ])

            # Deterministic fold assignment by Code hash keeps time ordering intact
            x = x.with_columns(
                (pl.col("Code").hash().cast(pl.UInt64) % pl.lit(k_folds)).cast(pl.Int32).alias("fold")
            )

            te_col = f"te{level}_sec_{target_col}"
            eps = 1e-12

            # Filter rows with valid targets for statistics
            xt = x.filter(pl.col(target_col).is_not_null())
            if xt.is_empty():
                logger.warning("No valid targets after filtering; skipping sector target encoding")
                return df

            # Daily aggregates per sector/fold for excl-fold adjustment
            fold_daily = (
                xt.group_by(["Date", id_col, "fold"]).agg([
                    pl.col(target_col).sum().alias("sum_f"),
                    pl.len().alias("cnt_f"),
                ])
                .sort([id_col, "fold", "Date"])
                .with_columns([
                    pl.col("sum_f").cumsum().over([id_col, "fold"]).alias("cum_sum_f"),
                    pl.col("cnt_f").cumsum().over([id_col, "fold"]).alias("cum_cnt_f"),
                    (pl.col(id_col).cast(pl.Utf8) + pl.lit("_") + pl.col("fold").cast(pl.Utf8)).alias("pair"),
                ])
            )

            # Daily aggregates per sector across folds
            all_daily = (
                xt.group_by(["Date", id_col]).agg([
                    pl.col(target_col).sum().alias("sum_all"),
                    pl.len().alias("cnt_all"),
                ])
                .sort([id_col, "Date"]).with_columns([
                    pl.col("sum_all").cumsum().over([id_col]).alias("cum_sum_all"),
                    pl.col("cnt_all").cumsum().over([id_col]).alias("cum_cnt_all"),
                ])
            )

            # Global per-date aggregates (all folds)
            glob_daily = (
                xt.group_by(["Date"]).agg([
                    pl.col(target_col).sum().alias("sum_glob"),
                    pl.len().alias("cnt_glob"),
                ])
                .sort(["Date"]).with_columns([
                    pl.col("sum_glob").cumsum().alias("cum_sum_glob"),
                    pl.col("cnt_glob").cumsum().alias("cum_cnt_glob"),
                ])
            )

            # Global per fold (needed to exclude fold from global mean)
            glob_fold_daily = (
                xt.group_by(["Date", "fold"]).agg([
                    pl.col(target_col).sum().alias("sum_gf"),
                    pl.len().alias("cnt_gf"),
                ])
                .sort(["fold", "Date"])
                .with_columns([
                    pl.col("sum_gf").cumsum().over(["fold"]).alias("cum_sum_gf"),
                    pl.col("cnt_gf").cumsum().over(["fold"]).alias("cum_cnt_gf"),
                ])
            )

            # Keys: unique Date×sector×fold with lookback shift
            keys = (
                x.select(["Date", id_col, "fold"]).unique().with_columns([
                    (pl.col("Date") - pl.duration(days=lag_days)).alias("lookback_date"),
                    (pl.col(id_col).cast(pl.Utf8) + pl.lit("_") + pl.col("fold").cast(pl.Utf8)).alias("pair"),
                ])
            )

            # Join-asof sector cumulative stats at lookback (all folds)
            keys = keys.sort([id_col, "lookback_date"])
            keys = keys.join_asof(
                all_daily.select([id_col, "Date", "cum_sum_all", "cum_cnt_all"]).sort([id_col, "Date"]),
                by=id_col,
                left_on="lookback_date",
                right_on="Date",
                strategy="backward",
            ).rename({"cum_sum_all": "all_sum_lag", "cum_cnt_all": "all_cnt_lag"})
            if "Date_right" in keys.columns:
                keys = keys.drop("Date_right")

            # Join-asof sector cumulative stats for current fold
            keys = keys.sort(["pair", "lookback_date"])
            keys = keys.join_asof(
                fold_daily.select(["pair", "Date", "cum_sum_f", "cum_cnt_f"]).sort(["pair", "Date"]),
                by="pair",
                left_on="lookback_date",
                right_on="Date",
                strategy="backward",
            ).rename({"cum_sum_f": "f_sum_lag", "cum_cnt_f": "f_cnt_lag"})
            if "Date_right" in keys.columns:
                keys = keys.drop("Date_right")

            # Join-asof global cumulative stats (all folds)
            keys = keys.sort(["lookback_date"])
            keys = keys.join_asof(
                glob_daily.select(["Date", "cum_sum_glob", "cum_cnt_glob"]).sort(["Date"]),
                left_on="lookback_date",
                right_on="Date",
                strategy="backward",
            ).rename({"cum_sum_glob": "glob_sum_lag", "cum_cnt_glob": "glob_cnt_lag"})
            if "Date_right" in keys.columns:
                keys = keys.drop("Date_right")

            # Join-asof global per fold cumulative stats
            keys = keys.sort(["fold", "lookback_date"])
            keys = keys.join_asof(
                glob_fold_daily.select(["fold", "Date", "cum_sum_gf", "cum_cnt_gf"]).sort(["fold", "Date"]),
                by="fold",
                left_on="lookback_date",
                right_on="Date",
                strategy="backward",
            ).rename({"cum_sum_gf": "gf_sum_lag", "cum_cnt_gf": "gf_cnt_lag"})
            if "Date_right" in keys.columns:
                keys = keys.drop("Date_right")

            # Compute exclusion stats and Bayesian-smoothed TE
            m_lit = pl.lit(float(m))
            keys = keys.with_columns([
                (pl.col("all_sum_lag").fill_null(0.0) - pl.col("f_sum_lag").fill_null(0.0)).alias("sec_sum_excl"),
                (pl.col("all_cnt_lag").fill_null(0) - pl.col("f_cnt_lag").fill_null(0)).alias("sec_cnt_excl"),
                (pl.col("glob_sum_lag").fill_null(0.0) - pl.col("gf_sum_lag").fill_null(0.0)).alias("glob_sum_excl"),
                (pl.col("glob_cnt_lag").fill_null(0) - pl.col("gf_cnt_lag").fill_null(0)).alias("glob_cnt_excl"),
            ]).with_columns([
                (pl.when(pl.col("sec_cnt_excl") > 0)
                 .then(pl.col("sec_sum_excl") / (pl.col("sec_cnt_excl") + eps))
                 .otherwise(None)
                ).alias("mu_sec_excl"),
                (pl.when(pl.col("glob_cnt_excl") > 0)
                 .then(pl.col("glob_sum_excl") / (pl.col("glob_cnt_excl") + eps))
                 .otherwise(0.0)
                ).alias("mu_glob_excl"),
            ]).with_columns([
                (
                    (pl.col("sec_cnt_excl").cast(pl.Float64) * pl.col("mu_sec_excl").fill_null(pl.col("mu_glob_excl"))
                     + m_lit * pl.col("mu_glob_excl"))
                    / (pl.col("sec_cnt_excl").cast(pl.Float64) + m_lit + eps)
                ).alias(te_col)
            ])

            out = x.join(keys.select(["Date", id_col, "fold", te_col]), on=["Date", id_col, "fold"], how="left")

            # Detect if TE failed to populate; fall back to simpler cumulative encoding
            try:
                populated = out.select(pl.col(te_col).is_not_null().sum()).item() > 0
            except Exception:
                populated = False

            if populated:
                if "fold" in out.columns:
                    out = out.drop("fold")
                if computed_target and target_col in out.columns:
                    out = out.drop(target_col)
                return out

            logger.warning("Cross-fit TE empty; applying no-fold fallback")

            # Fallback: no fold exclusion, still lagged and smoothed
            keys_nf = x.select(["Date", id_col]).unique().with_columns(
                (pl.col("Date") - pl.duration(days=lag_days)).alias("lookback_date")
            )

            sec_daily = (
                xt.group_by(["Date", id_col]).agg([
                    pl.col(target_col).sum().alias("sum_s"),
                    pl.len().alias("cnt_s"),
                ])
                .sort([id_col, "Date"]).with_columns([
                    pl.col("sum_s").cumsum().over([id_col]).alias("cum_sum_s"),
                    pl.col("cnt_s").cumsum().over([id_col]).alias("cum_cnt_s"),
                ])
            )

            glob_daily_nf = (
                xt.group_by(["Date"]).agg([
                    pl.col(target_col).sum().alias("sum_g"),
                    pl.len().alias("cnt_g"),
                ])
                .sort(["Date"]).with_columns([
                    pl.col("sum_g").cumsum().alias("cum_sum_g"),
                    pl.col("cnt_g").cumsum().alias("cum_cnt_g"),
                ])
            )

            keys_nf = keys_nf.sort([id_col, "lookback_date"]).join_asof(
                sec_daily.select([id_col, "Date", "cum_sum_s", "cum_cnt_s"]).sort([id_col, "Date"]),
                by=id_col,
                left_on="lookback_date",
                right_on="Date",
                strategy="backward",
            ).rename({"cum_sum_s": "sec_sum_lag", "cum_cnt_s": "sec_cnt_lag"})
            if "Date_right" in keys_nf.columns:
                keys_nf = keys_nf.drop("Date_right")

            keys_nf = keys_nf.sort(["lookback_date"]).join_asof(
                glob_daily_nf.select(["Date", "cum_sum_g", "cum_cnt_g"]).sort(["Date"]),
                left_on="lookback_date",
                right_on="Date",
                strategy="backward",
            ).rename({"cum_sum_g": "glob_sum_lag", "cum_cnt_g": "glob_cnt_lag"})
            if "Date_right" in keys_nf.columns:
                keys_nf = keys_nf.drop("Date_right")

            keys_nf = keys_nf.with_columns([
                (pl.when(pl.col("sec_cnt_lag") > 0)
                 .then(pl.col("sec_sum_lag") / (pl.col("sec_cnt_lag") + eps))
                 .otherwise(None)
                ).alias("mu_sec_lag"),
                (pl.when(pl.col("glob_cnt_lag") > 0)
                 .then(pl.col("glob_sum_lag") / (pl.col("glob_cnt_lag") + eps))
                 .otherwise(0.0)
                ).alias("mu_glob_lag"),
            ])

            keys_nf = keys_nf.with_columns(
                (
                    (pl.col("sec_cnt_lag").cast(pl.Float64).fill_null(0.0) * pl.col("mu_sec_lag").fill_null(pl.col("mu_glob_lag"))
                     + m_lit * pl.col("mu_glob_lag"))
                    / (pl.col("sec_cnt_lag").cast(pl.Float64).fill_null(0.0) + m_lit + eps)
                ).alias(te_col)
            )

            result = x.join(keys_nf.select(["Date", id_col, te_col]), on=["Date", id_col], how="left")
            if "fold" in result.columns:
                result = result.drop("fold")
            if computed_target and target_col in result.columns:
                result = result.drop(target_col)
            return result
        except Exception as exc:  # pragma: no cover - defensive guard for production runs
            logger.warning(f"Sector target encoding failed: {exc}")
            return df

    def finalize_for_spec(self, df: pl.DataFrame) -> pl.DataFrame:  # pragma: no cover - optional normalization
        try:
            rename_map = {
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

            # Ensure spec aliases (docs/ml/dataset_new.md) are present
            alias_map = {
                "bb_width": "bb_bw",
                "bb_position": "bb_pct_b",
            }
            alias_exprs = [pl.col(src).alias(dst) for dst, src in alias_map.items() if src in df.columns and dst not in df.columns]

            # Dynamic aliases for te33 → te (docs require both names)
            for col in df.columns:
                if col.startswith("te33_sec_"):
                    alias_name = "te_" + col[len("te33_"):]
                    if alias_name not in df.columns:
                        alias_exprs.append(pl.col(col).alias(alias_name))
            if alias_exprs:
                df = df.with_columns(alias_exprs)

            # Generate supervised learning targets if missing (copy from forward returns or derive from Close)
            tgt_defs = {
                "target_1d": (1,),
                "target_5d": (5,),
                "target_10d": (10,),
                "target_20d": (20,),
            }
            make_tgts: list[pl.Expr] = []
            for name, (h,) in tgt_defs.items():
                if name not in df.columns:
                    if f"feat_ret_{h}d" in df.columns:
                        make_tgts.append(pl.col(f"feat_ret_{h}d").alias(name))
                    elif {"Close", "Code"}.issubset(df.columns):
                        make_tgts.append(((pl.col("Close").shift(-h).over("Code") / (pl.col("Close") + 1e-12)) - 1.0).alias(name))
            if make_tgts:
                df = df.with_columns(make_tgts)

            # Binary targets (sign)
            bin_exprs = []
            for h in (1, 5, 10, 20):
                t = f"target_{h}d"
                tb = f"target_{h}d_binary"
                if t in df.columns and tb not in df.columns:
                    bin_exprs.append((pl.col(t) > 0).cast(pl.Int8).alias(tb))
            if bin_exprs:
                df = df.with_columns(bin_exprs)

            # Clip bb_position to [0,1] when present
            if "bb_position" in df.columns:
                df = df.with_columns(pl.col("bb_position").clip(0.0, 1.0).alias("bb_position"))

            # Drop join artefacts like *_right/_left to prevent schema pollution
            drop_join = [c for c in df.columns if c.endswith("_right") or c.endswith("_left")]
            if drop_join:
                df = df.drop(drop_join)

            # Clip extreme statement ratios and derived interaction terms to guard against
            # division-by-near-zero explosions that destabilize downstream scalers.
            clip_map = {
                # Statement progress / revision ratios (expect O(±5), clip at ±100 for safety)
                "stmt_progress_op": (-100.0, 100.0),
                "stmt_progress_np": (-100.0, 100.0),
                "stmt_rev_fore_op": (-100.0, 100.0),
                "stmt_rev_fore_np": (-100.0, 100.0),
                "stmt_rev_fore_eps": (-100.0, 100.0),
                "stmt_rev_div_fore": (-100.0, 100.0),
                "stmt_yoy_sales": (-100.0, 100.0),
                "stmt_yoy_op": (-100.0, 100.0),
                "stmt_yoy_np": (-100.0, 100.0),
                "stmt_opm": (-100.0, 100.0),
                "stmt_npm": (-100.0, 100.0),
                "stmt_profit_margin": (-100.0, 100.0),
                "stmt_revenue_growth": (-100.0, 100.0),
                "stmt_roe": (-100.0, 100.0),
                "stmt_roa": (-100.0, 100.0),
                # Derived PEAD features (linear combos of the above)
                "x_pead_effect": (-100.0, 100.0),
                "x_pead_times_mkt": (-100.0, 100.0),
                "x_credit_rev_bias": (-100.0, 100.0),
                # Margin ratios occasionally explode on thin denominators
                "dmi_credit_ratio": (-1_000.0, 1_000.0),
                "dmi_d_ratio_1d": (-1_000.0, 1_000.0),
                "margin_credit_ratio": (-1_000.0, 1_000.0),
            }
            clip_exprs = [pl.col(col).clip(min_val, max_val).alias(col) for col, (min_val, max_val) in clip_map.items() if col in df.columns]
            if clip_exprs:
                df = df.with_columns(clip_exprs)

            # Remove helper columns that should not leak into the shipped parquet
            drop_cols = [c for c in ("fold",) if c in df.columns]
            if drop_cols:
                df = df.drop(drop_cols)
        except Exception:
            pass

        if "LocalCode" not in df.columns and "Code" in df.columns:
            df = df.with_columns(pl.col("Code").cast(pl.Utf8).alias("LocalCode"))

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
            # Create section mapper if listed_info is available
            section_mapper = None
            if listed_info_df is not None and not listed_info_df.is_empty():
                try:
                    from src.features.section_mapper import SectionMapper
                    section_mapper = SectionMapper()
                    logger.info("[builder] Using section mapper for flow features")
                except Exception as e:
                    logger.warning(f"[builder] Failed to create section mapper: {e}")

            out = attach_flow_with_fallback(df, flow_daily, section_mapper=section_mapper)
            return out
        except Exception as e:
            logger.warning(f"[builder] flow integration failed: {e}")
            return df

    def add_earnings_features(
        self,
        df: pl.DataFrame,
        fetcher=None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: int = 90,
        lookahead_days: int = 90
    ) -> pl.DataFrame:
        """Add earnings event features from J-Quants API.

        Args:
            df: Base dataset with [code, date] columns
            fetcher: JQuantsAsyncFetcher instance
            start_date: Start date for earnings data (defaults to min date in df)
            end_date: End date for earnings data (defaults to max date in df)
            lookback_days: Days to look back for past earnings
            lookahead_days: Days to look ahead for future earnings

        Returns:
            DataFrame with earnings features added
        """
        if fetcher is None:
            logger.info("[builder] Earnings features skipped (no fetcher provided)")
            return df

        try:
            from src.gogooku3.features.earnings_features import add_earnings_features as _add_earnings

            # Use df date range if not provided
            if start_date is None:
                start_date = str(df['Date'].min())
            if end_date is None:
                end_date = str(df['Date'].max())

            # Rename columns for compatibility
            df_renamed = df.rename({'Date': 'date', 'Code': 'code'})

            # Add features
            df_with_earnings = _add_earnings(
                df_renamed,
                fetcher,
                start_date,
                end_date,
                lookback_days,
                lookahead_days
            )

            # Rename back
            df_with_earnings = df_with_earnings.rename({'date': 'Date', 'code': 'Code'})

            logger.info("[builder] Added earnings features: days_to_earnings, days_since_earnings, is_earnings_week, etc.")
            return df_with_earnings

        except Exception as e:
            logger.warning(f"[builder] Earnings features failed: {e}")
            return df

    def add_short_position_features(
        self,
        df: pl.DataFrame,
        fetcher=None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ma_windows: list[int] = [5, 20]
    ) -> pl.DataFrame:
        """Add short position features from J-Quants API.

        Args:
            df: Base dataset with [code, date, volume] columns
            fetcher: JQuantsAsyncFetcher instance
            start_date: Start date for short data
            end_date: End date for short data
            ma_windows: Moving average windows for trends

        Returns:
            DataFrame with short position features added
        """
        if fetcher is None:
            logger.info("[builder] Short position features skipped (no fetcher provided)")
            return df

        try:
            from src.gogooku3.features.short_features import add_short_position_features as _add_shorts

            # Use df date range if not provided
            if start_date is None:
                start_date = str(df['Date'].min())
            if end_date is None:
                end_date = str(df['Date'].max())

            # Rename columns for compatibility
            df_renamed = df.rename({'Date': 'date', 'Code': 'code', 'Volume': 'volume'})

            # Add features
            df_with_shorts = _add_shorts(
                df_renamed,
                fetcher,
                start_date,
                end_date,
                ma_windows
            )

            # Rename back
            rename_dict = {'date': 'Date', 'code': 'Code', 'volume': 'Volume'}
            for col in df_with_shorts.columns:
                if col in rename_dict:
                    df_with_shorts = df_with_shorts.rename({col: rename_dict[col]})

            logger.info("[builder] Added short position features: short_ratio, days_to_cover, short_squeeze_risk, etc.")
            return df_with_shorts

        except Exception as e:
            logger.warning(f"[builder] Short position features failed: {e}")
            return df

    def add_enhanced_listed_features(
        self,
        df: pl.DataFrame,
        fetcher=None,
        df_prices: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Add enhanced listed info features from J-Quants API.

        Args:
            df: Base dataset with [code, date] columns
            fetcher: JQuantsAsyncFetcher instance
            df_prices: Price data for momentum calculations (optional)

        Returns:
            DataFrame with enhanced listed info features added
        """
        if fetcher is None:
            logger.info("[builder] Enhanced listed features skipped (no fetcher provided)")
            return df

        try:
            from src.gogooku3.features.listed_features import add_listed_info_features as _add_listed

            # Rename columns for compatibility
            df_renamed = df.rename({'Date': 'date', 'Code': 'code'})

            # Prepare price data if available
            if df_prices is not None:
                df_prices = df_prices.rename({'Date': 'date', 'Code': 'code', 'Close': 'close'})

            # Add features
            df_with_listed = _add_listed(
                df_renamed,
                fetcher,
                df_prices
            )

            # Rename back
            df_with_listed = df_with_listed.rename({'date': 'Date', 'code': 'Code'})

            logger.info("[builder] Added enhanced listed features: market_cap_log, liquidity_score, sector_momentum, etc.")
            return df_with_listed

        except Exception as e:
            logger.warning(f"[builder] Enhanced listed features failed: {e}")
            return df

    def add_enhanced_margin_features(
        self,
        df: pl.DataFrame,
        fetcher=None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_weekly: bool = True
    ) -> pl.DataFrame:
        """Add enhanced margin trading features from J-Quants API.

        Args:
            df: Base dataset with [code, date] columns
            fetcher: JQuantsAsyncFetcher instance
            start_date: Start date for margin data
            end_date: End date for margin data
            use_weekly: Whether to include weekly margin data

        Returns:
            DataFrame with enhanced margin features added
        """
        if fetcher is None:
            logger.info("[builder] Enhanced margin features skipped (no fetcher provided)")
            return df

        try:
            from src.gogooku3.features.margin_trading_features import add_enhanced_margin_features as _add_margin

            # Use df date range if not provided
            if start_date is None:
                start_date = str(df['Date'].min())
            if end_date is None:
                end_date = str(df['Date'].max())

            # Rename columns for compatibility
            df_renamed = df.rename({'Date': 'date', 'Code': 'code'})

            # Add features
            df_with_margin = _add_margin(
                df_renamed,
                fetcher,
                start_date,
                end_date,
                use_weekly
            )

            # Rename back
            df_with_margin = df_with_margin.rename({'date': 'Date', 'code': 'Code'})

            logger.info("[builder] Added enhanced margin features: margin_balance_ratio, margin_velocity, margin_stress_indicator, etc.")
            return df_with_margin

        except Exception as e:
            logger.warning(f"[builder] Enhanced margin features failed: {e}")
            return df

    def add_option_sentiment_features(
        self,
        df: pl.DataFrame,
        fetcher=None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pl.DataFrame:
        """Add option sentiment features from J-Quants index options.

        Args:
            df: Base dataset with [date] columns
            fetcher: JQuantsAsyncFetcher instance
            start_date: Start date for option data
            end_date: End date for option data

        Returns:
            DataFrame with option sentiment features added
        """
        if fetcher is None:
            logger.info("[builder] Option sentiment features skipped (no fetcher provided)")
            return df

        try:
            from src.gogooku3.features.option_sentiment_features import add_option_sentiment_features as _add_options

            # Use df date range if not provided
            if start_date is None:
                start_date = str(df['Date'].min())
            if end_date is None:
                end_date = str(df['Date'].max())

            # Rename columns for compatibility
            df_renamed = df.rename({'Date': 'date', 'Code': 'code'})

            # Add features
            df_with_options = _add_options(
                df_renamed,
                fetcher,
                start_date,
                end_date
            )

            # Rename back
            df_with_options = df_with_options.rename({'date': 'Date', 'code': 'Code'})

            logger.info("[builder] Added option sentiment features: put_call_ratio, iv_skew, smart_money_indicator, etc.")
            return df_with_options

        except Exception as e:
            logger.warning(f"[builder] Option sentiment features failed: {e}")
            return df

    def add_enhanced_flow_features(
        self,
        df: pl.DataFrame,
        fetcher=None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pl.DataFrame:
        """Add enhanced flow analysis features from J-Quants trades_spec.

        Args:
            df: Base dataset with [code, date] columns
            fetcher: JQuantsAsyncFetcher instance
            start_date: Start date for flow data
            end_date: End date for flow data

        Returns:
            DataFrame with enhanced flow features added
        """
        if fetcher is None:
            logger.info("[builder] Enhanced flow features skipped (no fetcher provided)")
            return df

        try:
            from src.gogooku3.features.enhanced_flow_features import add_enhanced_flow_features as _add_flow

            # Use df date range if not provided
            if start_date is None:
                start_date = str(df['Date'].min())
            if end_date is None:
                end_date = str(df['Date'].max())

            # Rename columns for compatibility
            df_renamed = df.rename({'Date': 'date', 'Code': 'code'})

            # Add features
            df_with_flow = _add_flow(
                df_renamed,
                fetcher,
                start_date,
                end_date
            )

            # Rename back
            df_with_flow = df_with_flow.rename({'date': 'Date', 'code': 'Code'})

            logger.info("[builder] Added enhanced flow features: institutional_accumulation, foreign_sentiment, smart_flow_indicator, etc.")
            return df_with_flow

        except Exception as e:
            logger.warning(f"[builder] Enhanced flow features failed: {e}")
            return df

    def add_futures_block(
        self,
        df: pl.DataFrame,
        futures_df: Optional[pl.DataFrame],
        *,
        categories: Optional[list[str]] = None,
        topix_df: Optional[pl.DataFrame] = None,
        spot_map: Optional[dict[str, pl.DataFrame]] = None,
        on_z_window: int = 60,
        z_window_eod: int = 252,
        make_continuous_series: bool = False,
    ) -> pl.DataFrame:
        """Attach futures ON/EOD features to the equity panel.

        Args:
            df: Equity panel (Code, Date, ...)
            futures_df: Raw /derivatives/futures daily dataset
            categories: Target futures categories (e.g., ["TOPIXF","NK225F"]) 
            topix_df: Spot TOPIX series for basis (if available)
            on_z_window: ON z-score window (default=60)
            z_window_eod: EOD z-score window (default=252)

        Returns:
            DataFrame with fut_* features joined on Date (ON) and T+1 (EOD)
        """
        if futures_df is None or futures_df.is_empty():
            logger.info("[builder] futures block skipped (no futures_df)")
            return df

        cats = categories or ["TOPIXF", "NK225F", "JN400F", "REITF"]
        try:
            from src.gogooku3.features.futures_features import (
                attach_to_equity_panel,
                build_eod_features,
                build_next_bday_expr_from_quotes,
                build_on_features,
                prep_futures,
            )

            # Prepare raw futures
            fprep = prep_futures(futures_df, cats)
            if fprep.is_empty():
                logger.info("[builder] futures block: no data after filtering; skipping")
                return df

            # ON features
            on_df = build_on_features(fprep, on_z_window=on_z_window)

            # Spot mapping (basis)
            _spot_map: dict[str, pl.DataFrame] = {}
            if topix_df is not None and not topix_df.is_empty():
                _spot_map["TOPIXF"] = topix_df.select([
                    pl.col("Date").cast(pl.Date),
                    pl.col("Close").cast(pl.Float64).alias("S"),
                ])
            # Merge external spot map if provided
            if spot_map:
                for k, v in spot_map.items():
                    if v is None or v.is_empty():
                        continue
                    vv = v
                    if "Close" in vv.columns and "S" not in vv.columns:
                        vv = vv.rename({"Close": "S"})
                    if vv["Date"].dtype == pl.Utf8:
                        vv = vv.with_columns(pl.col("Date").str.strptime(pl.Date, strict=False))
                    _spot_map[k] = vv.select(["Date", "S"]).with_columns(pl.col("S").cast(pl.Float64))

            # Next business day mapping (from equity dates)
            next_bd_expr = build_next_bday_expr_from_quotes(df)

            eod_df = build_eod_features(
                fprep,
                spot_map=_spot_map,
                next_bday_expr=next_bd_expr,
                z_window=z_window_eod,
                make_continuous_series=make_continuous_series,
            )

            out = attach_to_equity_panel(df, on_df, eod_df, cats)

            # Coverage logs (best effort)
            try:
                for cat in cats:
                    on_col = f"is_fut_on_valid_{cat.lower()}"
                    eod_col = f"is_fut_eod_valid_{cat.lower()}"
                    if on_col in out.columns:
                        on_cov = float(out.select(pl.col(on_col).mean()).item())
                        logger.info(f"Futures ON coverage {cat}: {on_cov:.1%}")
                    if eod_col in out.columns:
                        eod_cov = float(out.select(pl.col(eod_col).mean()).item())
                        logger.info(f"Futures EOD coverage {cat}: {eod_cov:.1%}")
            except Exception:
                pass
            return out
        except Exception as e:
            logger.warning(f"[builder] futures block failed: {e}")
            return df

    def add_statements_features(self, df: pl.DataFrame, stm_df: Optional[pl.DataFrame]) -> pl.DataFrame:
        if stm_df is None or stm_df.is_empty():
            logger.info("[builder] statements enrichment skipped (no stm_df)")
            return df
        try:
            from src.features.safe_joiner_v2 import SafeJoinerV2
            sj = SafeJoinerV2()
            out = sj.join_statements_with_dedup(df, stm_df, use_time_cutoff=True, calendar_df=None)
            if "shares_outstanding" in out.columns:
                base_shares = pl.col("shares_outstanding")
            else:
                base_shares = None
            if "stmt_shares_outstanding" in out.columns:
                if "shares_outstanding" in out.columns:
                    out = out.with_columns(
                        pl.coalesce([
                            pl.col("shares_outstanding"),
                            pl.col("stmt_shares_outstanding")
                        ]).alias("shares_outstanding")
                    )
                else:
                    out = out.with_columns(
                        pl.col("stmt_shares_outstanding").alias("shares_outstanding")
                    )
                drop_stmt_share_cols = [c for c in (
                    "stmt_shares_outstanding",
                    "stmt_issued_shares",
                    "stmt_treasury_shares",
                    "stmt_average_shares",
                ) if c in out.columns]
                if drop_stmt_share_cols:
                    out = out.drop(drop_stmt_share_cols)
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

    def add_daily_margin_block(
        self,
        df: pl.DataFrame,
        daily_df: Optional[pl.DataFrame],
        *,
        adv_window_days: int = 20,
        enable_z_scores: bool = True,
    ) -> pl.DataFrame:
        """Add daily margin interest features with leak-safe as-of join.

        Args:
            df: Base daily quotes DataFrame
            daily_df: Raw daily margin interest data from API
            adv_window_days: Window for ADV calculation
            enable_z_scores: Whether to compute z-score features

        Returns:
            DataFrame with dmi_* features attached
        """
        if daily_df is None or daily_df.is_empty():
            logger.info("[builder] daily margin skipped (no data)")
            return df

        # Compute ADV data if not present
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
                logger.info(f"[builder] computed ADV{adv_window_days} for daily margin scaling (AdjustmentVolume)")
            except Exception as e:
                logger.warning(f"[builder] ADV computation (AdjustmentVolume) failed: {e}")

        # Fallback: derive ADV from raw volume if adjustment series absent
        if adv20_df is None and "Volume" in df.columns:
            try:
                adv20_df = (
                    df.select(["Code", "Date", "Volume"])
                    .with_columns([
                        pl.col("Volume")
                        .rolling_mean(adv_window_days, min_periods=5)
                        .over("Code")
                        .alias("ADV20_shares")
                    ])
                    .drop_nulls(subset=["ADV20_shares"])
                )
                logger.info(f"[builder] computed ADV{adv_window_days} fallback from Volume")
            except Exception as e:
                logger.warning(f"[builder] ADV fallback (Volume) failed: {e}")

        # Build a next-business-day mapper (quotes由来のグリッドから共通ヘルパーで生成)
        _next_bd = build_next_bday_expr_from_quotes(df)

        out = _add_daily_margin_block(
            quotes=df,
            daily_df=daily_df,
            adv20_df=adv20_df,
            enable_z_scores=enable_z_scores,
            next_business_day=_next_bd,
        )

        # Coverage log
        try:
            cov = float(out.select(pl.col("is_dmi_valid").mean()).item())
            logger.info(f"Daily margin feature coverage: {cov:.1%}")
        except Exception:
            pass

        # Ensure turnover_rate available (fallback to volume-based rate)
        if "turnover_rate" not in out.columns and "Volume" in out.columns:
            try:
                out = out.sort(["Code", "Date"]).with_columns(
                    (
                        pl.col("Volume")
                        / (pl.col("Volume").rolling_mean(adv_window_days, min_periods=5).over("Code") + 1e-12)
                    ).alias("turnover_rate")
                )
            except Exception as exc:
                logger.warning(f"[builder] turnover_rate fallback failed: {exc}")

        return out

    def add_short_selling_block(
        self,
        df: pl.DataFrame,
        short_df: Optional[pl.DataFrame],
        positions_df: Optional[pl.DataFrame] = None,
        adv20_df: Optional[pl.DataFrame] = None,
        *,
        enable_z_scores: bool = True,
        z_window: int = 252,
    ) -> pl.DataFrame:
        """Add short selling features with leak-safe as-of join.

        Args:
            df: Base daily quotes DataFrame
            short_df: Short selling ratio data from API
            positions_df: Short selling positions data (optional)
            adv20_df: ADV20 data for liquidity scaling (optional)
            enable_z_scores: Whether to compute z-score features
            z_window: Window for Z-score calculation

        Returns:
            DataFrame with ss_* features attached
        """
        if short_df is None or short_df.is_empty():
            logger.info("[builder] short selling enrichment skipped (no short_df)")
            return df

        # Compute ADV data if not provided
        if adv20_df is None and "AdjustmentVolume" in df.columns:
            try:
                adv20_df = (
                    df.select(["Code", "Date", "AdjustmentVolume"])
                    .with_columns([
                        pl.col("AdjustmentVolume")
                        .rolling_mean(20)
                        .over("Code")
                        .alias("ADV20_shares")
                    ])
                    .drop_nulls(subset=["ADV20_shares"])
                )
                logger.info("[builder] computed ADV20 for short selling scaling")
            except Exception as e:
                logger.warning(f"[builder] ADV computation failed: {e}")

        out = _add_short_selling_block(
            quotes=df,
            short_df=short_df,
            positions_df=positions_df,
            adv20_df=adv20_df,
            enable_z_scores=enable_z_scores,
            z_window=z_window,
        )

        # Coverage log
        try:
            cov = float(out.select(pl.col("is_ss_valid").mean()).item())
            logger.info(f"Short selling feature coverage: {cov:.1%}")
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
