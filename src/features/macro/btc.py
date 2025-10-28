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

logger = logging.getLogger(__name__)

_DATE_FMT = "%Y-%m-%d"
_DEFAULT_TICKER = "BTC-USD"


def load_btc_history(
    start: str,
    end: str,
    *,
    ticker: str = _DEFAULT_TICKER,
    parquet_path: Path | None = None,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """
    Load (or fetch) daily BTC/USD history covering the requested range.

    Args:
        start: Inclusive start date (YYYY-MM-DD)
        end: Inclusive end date (YYYY-MM-DD)
        ticker: Yahoo Finance ticker symbol (default: BTC-USD)
        parquet_path: Optional cache parquet to read/write
        force_refresh: If True, bypass cache and refetch

    Returns:
        Polars DataFrame with columns including Date, Open, High, Low, Close, Volume.
    """

    resolved_cache: Path | None = None
    if not force_refresh:
        prefix = "btc"
        if parquet_path:
            stem = parquet_path.stem
            if "_history_" in stem:
                prefix = stem.split("_history_")[0]
        else:
            prefix = f"btc_{ticker.replace('-', '').lower()}"
        resolved_cache = resolve_cached_parquet(
            parquet_path, prefix=prefix, start=start, end=end
        )

    if resolved_cache and resolved_cache.exists() and not force_refresh:
        try:
            df = pl.read_parquet(resolved_cache)
            logger.info("Loaded BTC history from cache: %s", resolved_cache)
            return df
        except Exception as exc:  # pragma: no cover - IO guard
            logger.warning("Failed to read cached BTC parquet (%s): %s", resolved_cache, exc)

    yf = get_yfinance_module(raise_on_missing=False)
    if yf is None:
        logger.warning(
            "yfinance is not available; cannot fetch BTC history. "
            "Install yfinance or provide --btc-parquet."
        )
        return pl.DataFrame()

    start_dt = datetime.strptime(start, _DATE_FMT)
    end_dt = datetime.strptime(end, _DATE_FMT) + timedelta(days=1)  # yfinance end is exclusive

    try:
        data = yf.download(
            ticker,
            start=start_dt.strftime(_DATE_FMT),
            end=end_dt.strftime(_DATE_FMT),
            auto_adjust=False,
            progress=False,
            interval="1d",
        )
    except Exception as exc:  # pragma: no cover - network guard
        logger.warning("Failed to download BTC data via yfinance (%s): %s", ticker, exc)
        return pl.DataFrame()

    if data.empty:
        logger.warning("BTC download returned no rows for %s → %s (ticker=%s)", start, end, ticker)
        return pl.DataFrame()

    data = flatten_yfinance_columns(data, ticker=ticker)
    if data.index.name is None:
        data.index.name = "Date"
    data = data.reset_index()
    data = flatten_yfinance_columns(data, ticker=ticker)

    if "Date" not in data.columns:
        logger.warning("Flattened BTC DataFrame missing Date column; skipping")
        return pl.DataFrame()

    data["Date"] = data["Date"].dt.tz_localize(None)
    df = pl.from_pandas(data, include_index=False)
    df = df.with_columns(pl.col("Date").cast(pl.Date))

    target_cache = parquet_path or resolved_cache

    if target_cache:
        try:
            target_cache.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(target_cache)
            logger.info("Cached BTC history to %s", target_cache)
        except Exception as exc:  # pragma: no cover - IO guard
            logger.warning("Failed to cache BTC parquet (%s): %s", target_cache, exc)

    return df


def prepare_btc_features(
    btc_df: pl.DataFrame,
    *,
    feature_prefix: str = "macro_btc",
    spike_threshold: float = 2.0,
) -> pl.DataFrame:
    """
    Transform raw BTC/USD levels into engineered sentiment and flow features.

    Args:
        btc_df: DataFrame with at least Date and Close columns
        feature_prefix: Prefix for generated feature names
        spike_threshold: Z-score threshold used for spike flag detection

    Returns:
        Polars DataFrame with Date and engineered BTC features
    """

    if btc_df.is_empty() or "Date" not in btc_df.columns or "Close" not in btc_df.columns:
        logger.warning("BTC DataFrame empty or missing required columns; skipping features")
        return pl.DataFrame(schema={"Date": pl.Date})

    ensure_exprs: list[pl.Expr] = []
    for col_name in ("High", "Low", "Volume"):
        if col_name not in btc_df.columns:
            ensure_exprs.append(pl.lit(None, dtype=pl.Float64).alias(col_name))
    if ensure_exprs:
        btc_df = btc_df.with_columns(ensure_exprs)

    eps = 1e-9
    close = pl.col("Close")
    volume = pl.col("Volume")

    df = (
        btc_df.sort("Date")
        .with_columns(
            [
                pl.col("Date").cast(pl.Date),
                close.alias(f"{feature_prefix}_close"),
                (close + eps).log().alias(f"{feature_prefix}_log_close"),
                ((close / close.shift(1)) - 1.0).alias(f"{feature_prefix}_ret_1d"),
                ((close / close.shift(5)) - 1.0).alias(f"{feature_prefix}_ret_5d"),
                ((close / close.shift(10)) - 1.0).alias(f"{feature_prefix}_ret_10d"),
                ((close / close.shift(20)) - 1.0).alias(f"{feature_prefix}_ret_20d"),
                close.rolling_mean(window_size=7, min_periods=5).alias("_btc_sma7"),
                close.rolling_mean(window_size=30, min_periods=10).alias("_btc_sma30"),
                close.rolling_mean(window_size=60, min_periods=20).alias("_btc_sma60"),
                close.rolling_mean(window_size=120, min_periods=40).alias("_btc_mean_120"),
                close.rolling_std(window_size=60, min_periods=20).alias("_btc_std_60"),
                close.rolling_std(window_size=120, min_periods=40).alias("_btc_std_120"),
                ((close / close.shift(1)) - 1.0)
                .rolling_std(window_size=20, min_periods=10)
                .mul((365.0) ** 0.5)
                .alias(f"{feature_prefix}_vol_20"),
                ((close / close.shift(1)) - 1.0)
                .rolling_std(window_size=60, min_periods=20)
                .mul((365.0) ** 0.5)
                .alias(f"{feature_prefix}_vol_60"),
                pl.col("High").cast(pl.Float64, strict=False).alias("_btc_high"),
                pl.col("Low").cast(pl.Float64, strict=False).alias("_btc_low"),
                volume.cast(pl.Float64, strict=False).alias("_btc_volume"),
                volume.rolling_mean(window_size=20, min_periods=10).alias("_btc_vol_mean20"),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("_btc_sma7") - pl.col("_btc_sma30"))
                    / (pl.col("_btc_sma30") + eps)
                ).alias(f"{feature_prefix}_sma7_over_sma30"),
                (
                    (close - pl.col("_btc_mean_120"))
                    / (pl.col("_btc_std_120") + eps)
                ).alias(f"{feature_prefix}_zscore_120"),
                (
                    (close - close.rolling_mean(window_size=60, min_periods=20))
                    / (pl.col("_btc_std_60") + eps)
                ).alias(f"{feature_prefix}_zscore_60"),
                (
                    (pl.col("_btc_sma7") / (pl.col("_btc_sma7").shift(3) + eps)) - 1.0
                ).alias(f"{feature_prefix}_sma7_slope_3"),
                ((close / pl.col("_btc_sma60")) - 1.0).alias(f"{feature_prefix}_ma_gap_60"),
                ((pl.col("_btc_high") - pl.col("_btc_low")) / (close + eps)).alias(
                    f"{feature_prefix}_range_pct"
                ),
                ((close / pl.col("_btc_sma30")) - 1.0).alias(f"{feature_prefix}_ma_gap_30"),
                ((close / close.shift(1)) - 1.0).abs().alias(f"{feature_prefix}_abs_ret_1d"),
                (
                    (volume + eps) / (pl.col("_btc_vol_mean20") + eps)
                ).alias(f"{feature_prefix}_volume_ratio_20"),
            ]
        )
    )

    df = df.with_columns(
        [
            pl.when(pl.col(f"{feature_prefix}_zscore_60").abs() > spike_threshold)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"{feature_prefix}_spike_flag"),
            pl.when(pl.col(f"{feature_prefix}_vol_20") > 0.9)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"{feature_prefix}_high_vol_regime"),
            pl.when(pl.col("_btc_sma7") > pl.col("_btc_sma30"))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"{feature_prefix}_trend_up"),
        ]
    )

    df = df.drop(
        [
            "_btc_sma7",
            "_btc_sma30",
            "_btc_sma60",
            "_btc_mean_120",
            "_btc_std_60",
            "_btc_std_120",
            "_btc_high",
            "_btc_low",
            "_btc_volume",
            "_btc_vol_mean20",
        ]
    )

    return df
