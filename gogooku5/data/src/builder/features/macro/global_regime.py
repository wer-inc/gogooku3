"""VVMD (Volatility × Volume × Momentum × Demand) global macro features.

Phase 1: 14 core features from US and global markets (SPY, QQQ, VIX, DXY, BTC)
to capture market regime dynamics for Japanese stock prediction.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from .yfinance_utils import (
    flatten_yfinance_columns,
    get_yfinance_module,
    resolve_cached_parquet,
)

LOGGER = logging.getLogger(__name__)
_DATE_FMT = "%Y-%m-%d"

# Data source tickers
_TICKERS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",  # US Dollar Index
    "UUP": "UUP",  # Fallback for DXY
    "BTC": "BTC-USD",
}


def load_global_regime_data(
    start: str,
    end: str,
    *,
    parquet_path: Path | None = None,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Load global market data from cache or yfinance.

    Fetches OHLCV data for:
    - SPY: S&P 500 ETF
    - QQQ: Nasdaq-100 ETF
    - ^VIX: CBOE Volatility Index
    - DX-Y.NYB: US Dollar Index (with UUP fallback)
    - BTC-USD: Bitcoin

    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        parquet_path: Optional cache file path
        force_refresh: Skip cache if True

    Returns:
        DataFrame with columns: Date, spy_close, spy_volume, qqq_close, qqq_volume,
                                vix_close, dxy_close, btc_close
    """
    resolved_cache = None
    if not force_refresh:
        resolved_cache = resolve_cached_parquet(
            parquet_path, prefix="global_regime", start=start, end=end
        )

    if resolved_cache and resolved_cache.exists() and not force_refresh:
        try:
            df = pl.read_parquet(resolved_cache)
            LOGGER.info("Loaded global regime data from cache: %s", resolved_cache)
            return df
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to read cached global regime parquet (%s): %s", resolved_cache, exc)

    yf = get_yfinance_module(raise_on_missing=False)
    if yf is None:
        LOGGER.warning("yfinance not available; global regime data unavailable")
        return pl.DataFrame()

    start_dt = datetime.strptime(start, _DATE_FMT)
    end_dt = datetime.strptime(end, _DATE_FMT) + timedelta(days=1)

    # Fetch all tickers
    combined_data = {}
    for name, ticker in _TICKERS.items():
        if name == "UUP":
            # Skip UUP unless DXY fails
            continue

        try:
            data = yf.download(  # type: ignore[attr-defined]
                ticker,
                start=start_dt.strftime(_DATE_FMT),
                end=end_dt.strftime(_DATE_FMT),
                auto_adjust=False,
                progress=False,
                interval="1d",
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to download %s (%s) via yfinance: %s", name, ticker, exc)

            # Fallback: DXY → UUP
            if name == "DXY":
                LOGGER.info("Attempting DXY fallback to UUP...")
                try:
                    data = yf.download(  # type: ignore[attr-defined]
                        _TICKERS["UUP"],
                        start=start_dt.strftime(_DATE_FMT),
                        end=end_dt.strftime(_DATE_FMT),
                        auto_adjust=False,
                        progress=False,
                        interval="1d",
                    )
                    LOGGER.info("Successfully fetched UUP as DXY fallback")
                except Exception as fallback_exc:  # pragma: no cover
                    LOGGER.warning("DXY fallback to UUP also failed: %s", fallback_exc)
                    continue
            else:
                continue

        if getattr(data, "empty", True):
            LOGGER.warning("%s download returned no rows for %s → %s", name, start, end)
            continue

        # Flatten columns and process
        data = flatten_yfinance_columns(data, ticker=ticker)
        if data.index.name is None:
            data.index.name = "Date"
        data = data.reset_index()

        if "Date" not in data.columns:
            LOGGER.warning("Flattened %s DataFrame missing Date column; skipping", name)
            continue

        # Timezone localization
        data["Date"] = data["Date"].dt.tz_localize(None)

        # Store with lowercase prefix
        prefix = name.lower()
        combined_data[prefix] = data

    if not combined_data:
        LOGGER.warning("No global regime data fetched successfully")
        return pl.DataFrame()

    # Merge all data sources on Date
    base_df = None
    for prefix, data in combined_data.items():
        # Select relevant columns: Date, Close, Volume (if available)
        cols_to_keep = ["Date"]
        rename_map = {}

        if "Close" in data.columns:
            cols_to_keep.append("Close")
            rename_map["Close"] = f"{prefix}_close"

        if "Volume" in data.columns:
            cols_to_keep.append("Volume")
            rename_map["Volume"] = f"{prefix}_volume"

        # Subset and rename
        data_subset = data[cols_to_keep].copy()
        if rename_map:
            data_subset = data_subset.rename(columns=rename_map)

        # Convert to polars
        df_pl = pl.from_pandas(data_subset, include_index=False)
        df_pl = df_pl.with_columns(pl.col("Date").cast(pl.Date))

        if base_df is None:
            base_df = df_pl
        else:
            base_df = base_df.join(df_pl, on="Date", how="full", coalesce=True)

    if base_df is None or base_df.is_empty():
        LOGGER.warning("Failed to merge global regime data")
        return pl.DataFrame()

    base_df = base_df.sort("Date")

    # Cache the raw data
    target_cache = parquet_path or resolved_cache
    if target_cache:
        try:
            target_cache.parent.mkdir(parents=True, exist_ok=True)
            base_df.write_parquet(target_cache)
            LOGGER.info("Cached global regime data to %s", target_cache)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to cache global regime parquet (%s): %s", target_cache, exc)

    return base_df


def _robust_z(x: pl.Expr, window: int = 252, min_periods: int = 60) -> pl.Expr:
    """Calculate robust z-score: (x - rolling_median) / rolling_MAD.

    More resistant to outliers than standard z-score.
    """
    eps = 1e-9
    median = x.rolling_median(window_size=window, min_periods=min_periods)
    mad = (x - median).abs().rolling_median(window_size=window, min_periods=min_periods)
    return (x - median) / (mad + eps)


def _realized_vol(returns: pl.Expr, window: int, min_periods: int | None = None) -> pl.Expr:
    """Calculate annualized realized volatility: rolling_std(returns) * sqrt(252)."""
    if min_periods is None:
        min_periods = max(1, window // 2)
    return returns.rolling_std(window_size=window, min_periods=min_periods) * (252.0 ** 0.5)


def prepare_vvmd_features(regime_df: pl.DataFrame) -> pl.DataFrame:
    """Transform global regime data into 14 VVMD features.

    Feature Categories:
    - V (Volatility): 4 features
    - Vlm (Volume): 2 features
    - M (Momentum/Trend): 5 features
    - D (Demand): 3 features

    Args:
        regime_df: Raw data from load_global_regime_data()

    Returns:
        DataFrame with Date and 14 macro_vvmd_* features
    """
    if regime_df.is_empty() or "Date" not in regime_df.columns:
        return pl.DataFrame(schema={"Date": pl.Date})

    # Ensure required columns exist
    required = {
        "spy_close": pl.Float64,
        "spy_volume": pl.Float64,
        "qqq_close": pl.Float64,
        "qqq_volume": pl.Float64,
        "vix_close": pl.Float64,
        "dxy_close": pl.Float64,
        "btc_close": pl.Float64,
    }

    missing_cols = set(required.keys()) - set(regime_df.columns)
    if missing_cols:
        # Add missing columns as null
        fill_exprs = [pl.lit(None, dtype=required[col]).alias(col) for col in missing_cols]
        regime_df = regime_df.with_columns(fill_exprs)

    df = regime_df.sort("Date").with_columns(
        [
            pl.col("Date").cast(pl.Date),
            # Calculate returns
            (pl.col("spy_close") / pl.col("spy_close").shift(1) - 1.0).alias("_spy_ret"),
            (pl.col("qqq_close") / pl.col("qqq_close").shift(1) - 1.0).alias("_qqq_ret"),
            (pl.col("btc_close") / pl.col("btc_close").shift(1) - 1.0).alias("_btc_ret"),
        ]
    )

    eps = 1e-9

    # === V (Volatility): 4 features ===

    # 1. SPY realized volatility 20d
    df = df.with_columns(
        _realized_vol(pl.col("_spy_ret"), window=20, min_periods=10)
        .alias("macro_vvmd_vol_spy_rv20")
    )

    # 2. SPY volatility differential: RV20 - RV63
    df = df.with_columns(
        _realized_vol(pl.col("_spy_ret"), window=63, min_periods=30)
        .alias("_spy_rv63")
    )
    df = df.with_columns(
        (pl.col("macro_vvmd_vol_spy_rv20") - pl.col("_spy_rv63"))
        .alias("macro_vvmd_vol_spy_drv_20_63")
    )

    # 3. QQQ realized volatility 20d
    df = df.with_columns(
        _realized_vol(pl.col("_qqq_ret"), window=20, min_periods=10)
        .alias("macro_vvmd_vol_qqq_rv20")
    )

    # 4. VIX robust z-score 252d
    df = df.with_columns(
        _robust_z(pl.col("vix_close"), window=252, min_periods=60)
        .alias("macro_vvmd_vol_vix_z_252d")
    )

    # === Vlm (Volume): 2 features ===

    # 5. SPY volume surge: vol / rolling_median(vol, 20) - 1
    df = df.with_columns(
        (
            pl.col("spy_volume")
            / (pl.col("spy_volume").rolling_median(window_size=20, min_periods=10) + eps)
            - 1.0
        ).alias("macro_vvmd_vlm_spy_surge20")
    )

    # 6. QQQ volume surge
    df = df.with_columns(
        (
            pl.col("qqq_volume")
            / (pl.col("qqq_volume").rolling_median(window_size=20, min_periods=10) + eps)
            - 1.0
        ).alias("macro_vvmd_vlm_qqq_surge20")
    )

    # === M (Momentum/Trend): 5 features ===

    # 7. SPY momentum 63d (simple cumulative return)
    df = df.with_columns(
        (pl.col("spy_close") / pl.col("spy_close").shift(63) - 1.0)
        .alias("macro_vvmd_mom_spy_63d")
    )

    # 8. QQQ momentum 63d
    df = df.with_columns(
        (pl.col("qqq_close") / pl.col("qqq_close").shift(63) - 1.0)
        .alias("macro_vvmd_mom_qqq_63d")
    )

    # 9. SPY MA gap: (MA20 - MA100) / MA100
    df = df.with_columns(
        [
            pl.col("spy_close").rolling_mean(window_size=20, min_periods=20).alias("_spy_ma20"),
            pl.col("spy_close").rolling_mean(window_size=100, min_periods=50).alias("_spy_ma100"),
        ]
    )
    df = df.with_columns(
        ((pl.col("_spy_ma20") - pl.col("_spy_ma100")) / (pl.col("_spy_ma100") + eps))
        .alias("macro_vvmd_trend_spy_ma_gap_20_100")
    )

    # 10. QQQ MA gap
    df = df.with_columns(
        [
            pl.col("qqq_close").rolling_mean(window_size=20, min_periods=20).alias("_qqq_ma20"),
            pl.col("qqq_close").rolling_mean(window_size=100, min_periods=50).alias("_qqq_ma100"),
        ]
    )
    df = df.with_columns(
        ((pl.col("_qqq_ma20") - pl.col("_qqq_ma100")) / (pl.col("_qqq_ma100") + eps))
        .alias("macro_vvmd_trend_qqq_ma_gap_20_100")
    )

    # 11. SPY 52-week breakout position: (close - low52) / (high52 - low52)
    df = df.with_columns(
        [
            pl.col("spy_close").rolling_max(window_size=252, min_periods=126).alias("_spy_high52"),
            pl.col("spy_close").rolling_min(window_size=252, min_periods=126).alias("_spy_low52"),
        ]
    )
    df = df.with_columns(
        (
            (pl.col("spy_close") - pl.col("_spy_low52"))
            / (pl.col("_spy_high52") - pl.col("_spy_low52") + eps)
        ).alias("macro_vvmd_breakout_spy_bo52")
    )

    # === D (Demand): 3 features ===

    # 12. DXY robust z-score 252d
    df = df.with_columns(
        _robust_z(pl.col("dxy_close"), window=252, min_periods=60)
        .alias("macro_vvmd_demand_dxy_z_252d")
    )

    # 13. BTC relative momentum 63d: BTC_mom / SPY_mom - 1
    df = df.with_columns(
        (pl.col("btc_close") / pl.col("btc_close").shift(63) - 1.0).alias("_btc_mom_63d")
    )
    df = df.with_columns(
        (
            pl.col("_btc_mom_63d") / (pl.col("macro_vvmd_mom_spy_63d") + eps) - 1.0
        ).alias("macro_vvmd_demand_btc_rel_63d")
    )

    # 14. BTC realized volatility 20d
    df = df.with_columns(
        _realized_vol(pl.col("_btc_ret"), window=20, min_periods=10)
        .alias("macro_vvmd_vol_btc_rv20")
    )

    # Drop temporary columns
    drop_cols = [c for c in df.columns if c.startswith("_")]
    if drop_cols:
        df = df.drop(drop_cols)

    # Select only Date + 14 VVMD features
    vvmd_cols = [c for c in df.columns if c.startswith("macro_vvmd_")]
    final_cols = ["Date"] + vvmd_cols

    return df.select(final_cols)
