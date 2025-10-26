from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import polars as pl

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yf = None  # type: ignore[assignment]

from .vix import shift_to_next_business_day

logger = logging.getLogger(__name__)

_DATE_FMT = "%Y-%m-%d"
_DEFAULT_FX_TICKER = "JPY=X"  # USD/JPY


def load_fx_history(
    start: str,
    end: str,
    *,
    ticker: str = _DEFAULT_FX_TICKER,
    parquet_path: Path | None = None,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """
    Load FX history for the given ticker using yfinance (daily frequency).

    Args:
        start: Inclusive start date (YYYY-MM-DD)
        end: Inclusive end date (YYYY-MM-DD)
        ticker: Yahoo Finance ticker (default: JPY=X for USD/JPY)
        parquet_path: Optional cache file
        force_refresh: Force re-download even if cache exists
    """
    if parquet_path and parquet_path.exists() and not force_refresh:
        try:
            df = pl.read_parquet(parquet_path)
            logger.info("Loaded FX history from cache: %s", parquet_path)
            return df
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to read cached FX parquet (%s): %s", parquet_path, exc)

    if yf is None:
        logger.warning(
            "yfinance missing; cannot fetch FX history. Install yfinance or provide cached parquet."
        )
        return pl.DataFrame()

    start_dt = datetime.strptime(start, _DATE_FMT)
    end_dt = datetime.strptime(end, _DATE_FMT) + timedelta(days=1)

    try:
        data = yf.download(
            ticker,
            start=start_dt.strftime(_DATE_FMT),
            end=end_dt.strftime(_DATE_FMT),
            auto_adjust=False,
            progress=False,
            interval="1d",
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to download FX data (%s): %s", ticker, exc)
        return pl.DataFrame()

    if data.empty:
        logger.warning("FX download returned no rows for %s (%s→%s)", ticker, start, end)
        return pl.DataFrame()

    data = data.reset_index().rename(columns={"Date": "Date"})
    data["Date"] = data["Date"].dt.tz_localize(None)
    df = pl.from_pandas(data, include_index=False).with_columns(pl.col("Date").cast(pl.Date))

    if parquet_path:
        try:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(parquet_path)
            logger.info("Cached FX history to %s", parquet_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to cache FX parquet (%s): %s", parquet_path, exc)

    return df


def prepare_fx_features(
    fx_df: pl.DataFrame,
    *,
    feature_prefix: str = "macro_fx_usdjpy",
    spike_threshold: float = 1.5,
) -> pl.DataFrame:
    """
    Prepare engineered FX features from raw levels.

    Args:
        fx_df: DataFrame with Date and Close columns
        feature_prefix: Prefix for generated columns
        spike_threshold: Z-score threshold for spike flag
    """
    if fx_df.is_empty() or "Date" not in fx_df.columns or "Close" not in fx_df.columns:
        logger.warning("FX DataFrame empty or missing required columns; skipping")
        return pl.DataFrame(schema={"Date": pl.Date})

    eps = 1e-9
    close = pl.col("Close")

    ensure_exprs: list[pl.Expr] = []
    for col_name in ("High", "Low"):
        if col_name not in fx_df.columns:
            ensure_exprs.append(pl.lit(None, dtype=pl.Float64).alias(col_name))
    if ensure_exprs:
        fx_df = fx_df.with_columns(ensure_exprs)

    df = (
        fx_df.sort("Date")
        .with_columns(
            [
                pl.col("Date").cast(pl.Date),
                close.alias(f"{feature_prefix}_close"),
                (close + eps).log().alias(f"{feature_prefix}_log_close"),
                ((close / close.shift(1)) - 1.0).alias(f"{feature_prefix}_ret_1d"),
                ((close / close.shift(5)) - 1.0).alias(f"{feature_prefix}_ret_5d"),
                ((close / close.shift(10)) - 1.0).alias(f"{feature_prefix}_ret_10d"),
                ((close / close.shift(20)) - 1.0).alias(f"{feature_prefix}_ret_20d"),
                close.rolling_mean(window_size=5, min_periods=5).alias("_fx_sma5"),
                close.rolling_mean(window_size=20, min_periods=20).alias("_fx_sma20"),
                close.rolling_mean(window_size=60, min_periods=30).alias("_fx_mean_60"),
                ((close / close.shift(1)) - 1.0)
                .rolling_std(window_size=20, min_periods=10)
                .mul((252.0) ** 0.5)
                .alias(f"{feature_prefix}_vol_20"),
                close.rolling_mean(window_size=252, min_periods=126).alias("_fx_mean_252"),
                close.rolling_std(window_size=252, min_periods=126).alias("_fx_std_252"),
                close.rolling_std(window_size=60, min_periods=30).alias("_fx_std_60"),
                pl.col("High").cast(pl.Float64, strict=False).alias("_fx_high"),
                pl.col("Low").cast(pl.Float64, strict=False).alias("_fx_low"),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("_fx_sma5") - pl.col("_fx_sma20"))
                    / (pl.col("_fx_sma20") + eps)
                ).alias(f"{feature_prefix}_sma5_over_sma20"),
                (
                    (close - pl.col("_fx_mean_252"))
                    / (pl.col("_fx_std_252") + eps)
                ).alias(f"{feature_prefix}_zscore_252"),
                (
                    (close - pl.col("_fx_mean_60"))
                    / (pl.col("_fx_std_60") + eps)
                ).alias(f"{feature_prefix}_zscore_60"),
                ((close / pl.col("_fx_sma20")) - 1.0).alias(f"{feature_prefix}_ma_gap_20"),
                ((pl.col("_fx_high") - pl.col("_fx_low")) / (close + eps)).alias(
                    f"{feature_prefix}_range_pct"
                ),
                ((close / close.shift(1)) - 1.0).abs().alias(f"{feature_prefix}_abs_ret_1d"),
            ]
        )
    )

    df = df.with_columns(
        [
            pl.when(pl.col(f"{feature_prefix}_zscore_252").abs() > spike_threshold)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"{feature_prefix}_spike_flag"),
            (
                (pl.col("_fx_sma5") / (pl.col("_fx_sma5").shift(3) + eps)) - 1.0
            ).alias(f"{feature_prefix}_sma5_slope_3"),
            pl.when(pl.col("_fx_sma5") > pl.col("_fx_sma20"))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"{feature_prefix}_trend_up"),
            pl.when(pl.col(f"{feature_prefix}_abs_ret_1d") > 0.01)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"{feature_prefix}_shock_flag"),
        ]
    )

    df = df.drop(
        [
            "_fx_sma5",
            "_fx_sma20",
            "_fx_mean_60",
            "_fx_mean_252",
            "_fx_std_252",
            "_fx_std_60",
            "_fx_high",
            "_fx_low",
        ]
    )

    return df
