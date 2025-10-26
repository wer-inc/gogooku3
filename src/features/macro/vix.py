from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import polars as pl

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yf = None  # type: ignore[assignment]

import logging

logger = logging.getLogger(__name__)

_VIX_TICKER = "^VIX"
_DATE_FMT = "%Y-%m-%d"


def load_vix_history(
    start: str,
    end: str,
    *,
    parquet_path: Path | None = None,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """
    Load (or fetch) daily VIX levels covering the requested range.

    Args:
        start: Inclusive start date (YYYY-MM-DD)
        end: Inclusive end date (YYYY-MM-DD)
        parquet_path: Optional cache parquet to read/write
        force_refresh: If True, bypass cache and refetch

    Returns:
        Polars DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
    """

    if parquet_path and parquet_path.exists() and not force_refresh:
        try:
            df = pl.read_parquet(parquet_path)
            logger.info("Loaded VIX history from cache: %s", parquet_path)
            return df
        except Exception as exc:  # pragma: no cover - IO guard
            logger.warning("Failed to read cached VIX parquet (%s): %s", parquet_path, exc)

    if yf is None:
        logger.warning(
            "yfinance is not available; cannot fetch VIX history. "
            "Install yfinance or provide --vix-parquet."
        )
        return pl.DataFrame()

    start_dt = datetime.strptime(start, _DATE_FMT)
    end_dt = datetime.strptime(end, _DATE_FMT) + timedelta(days=1)  # yfinance end is exclusive

    try:
        data = yf.download(
            _VIX_TICKER,
            start=start_dt.strftime(_DATE_FMT),
            end=end_dt.strftime(_DATE_FMT),
            auto_adjust=False,
            progress=False,
            interval="1d",
        )
    except Exception as exc:  # pragma: no cover - network guard
        logger.warning("Failed to download VIX data via yfinance: %s", exc)
        return pl.DataFrame()

    if data.empty:
        logger.warning("VIX download returned no rows for %s → %s", start, end)
        return pl.DataFrame()

    data = data.reset_index().rename(columns={"Date": "Date"})
    data["Date"] = data["Date"].dt.tz_localize(None)
    df = pl.from_pandas(data, include_index=False)
    df = df.with_columns(pl.col("Date").cast(pl.Date))

    if parquet_path:
        try:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(parquet_path)
            logger.info("Cached VIX history to %s", parquet_path)
        except Exception as exc:  # pragma: no cover - IO guard
            logger.warning("Failed to cache VIX parquet (%s): %s", parquet_path, exc)

    return df


def prepare_vix_features(
    vix_df: pl.DataFrame,
    *,
    feature_prefix: str = "macro_vix",
    spike_threshold: float = 1.5,
) -> pl.DataFrame:
    """
    Transform raw VIX levels into engineered macro features.

    Args:
        vix_df: DataFrame with at least Date and Close columns
        feature_prefix: Prefix for generated feature names
        spike_threshold: Z-score threshold used for spike flag

    Returns:
        Polars DataFrame with Date and engineered VIX features
    """

    if vix_df.is_empty() or "Date" not in vix_df.columns or "Close" not in vix_df.columns:
        logger.warning("VIX DataFrame empty or missing required columns; skipping features")
        return pl.DataFrame(schema={"Date": pl.Date})

    eps = 1e-9
    close = pl.col("Close")

    ensure_exprs: list[pl.Expr] = []
    for col_name in ("High", "Low"):
        if col_name not in vix_df.columns:
            ensure_exprs.append(pl.lit(None, dtype=pl.Float64).alias(col_name))
    if ensure_exprs:
        vix_df = vix_df.with_columns(ensure_exprs)

    df = (
        vix_df.sort("Date")
        .with_columns(
            [
                pl.col("Date").cast(pl.Date),
                close.alias(f"{feature_prefix}_close"),
                (close + eps).log().alias(f"{feature_prefix}_log_close"),
                ((close / close.shift(1)) - 1.0).alias(f"{feature_prefix}_ret_1d"),
                ((close / close.shift(5)) - 1.0).alias(f"{feature_prefix}_ret_5d"),
                ((close / close.shift(10)) - 1.0).alias(f"{feature_prefix}_ret_10d"),
                ((close / close.shift(20)) - 1.0).alias(f"{feature_prefix}_ret_20d"),
                close.rolling_mean(window_size=5, min_periods=5).alias("_vix_sma5"),
                close.rolling_mean(window_size=20, min_periods=20).alias("_vix_sma20"),
                close.rolling_mean(window_size=60, min_periods=30).alias("_vix_mean_60"),
                ((close / close.shift(1)) - 1.0)
                .rolling_std(window_size=20, min_periods=10)
                .mul((252.0) ** 0.5)
                .alias(f"{feature_prefix}_vol_20"),
                close.rolling_mean(window_size=252, min_periods=126).alias("_vix_mean_252"),
                close.rolling_std(window_size=252, min_periods=126).alias("_vix_std_252"),
                close.rolling_std(window_size=60, min_periods=30).alias("_vix_std_60"),
                pl.col("High").cast(pl.Float64, strict=False).alias("_vix_high"),
                pl.col("Low").cast(pl.Float64, strict=False).alias("_vix_low"),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("_vix_sma5") - pl.col("_vix_sma20"))
                    / (pl.col("_vix_sma20") + eps)
                ).alias(f"{feature_prefix}_sma5_over_sma20"),
                (
                    (close - pl.col("_vix_mean_252"))
                    / (pl.col("_vix_std_252") + eps)
                ).alias(f"{feature_prefix}_zscore_252"),
                (
                    (close - pl.col("_vix_mean_60"))
                    / (pl.col("_vix_std_60") + eps)
                ).alias(f"{feature_prefix}_zscore_60"),
                ((close / pl.col("_vix_sma20")) - 1.0).alias(f"{feature_prefix}_ma_gap_20"),
                ((pl.col("_vix_high") - pl.col("_vix_low")) / (close + eps)).alias(
                    f"{feature_prefix}_range_pct"
                ),
                ((close / close.shift(1)) - 1.0).abs().alias(f"{feature_prefix}_abs_ret_1d"),
            ]
        )
    )

    df = df.with_columns(
        [
            pl.when(pl.col(f"{feature_prefix}_zscore_252") > spike_threshold)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"{feature_prefix}_spike_flag"),
            (
                (pl.col("_vix_sma5") / (pl.col("_vix_sma5").shift(3) + eps)) - 1.0
            ).alias(f"{feature_prefix}_sma5_slope_3"),
            pl.when(pl.col(f"{feature_prefix}_close") >= 30.0)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"{feature_prefix}_high_regime"),
            pl.when(pl.col(f"{feature_prefix}_close") <= 15.0)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"{feature_prefix}_low_regime"),
        ]
    )

    # Clean up helper columns
    df = df.drop(
        [
            "_vix_sma5",
            "_vix_sma20",
            "_vix_mean_60",
            "_vix_mean_252",
            "_vix_std_252",
            "_vix_std_60",
            "_vix_high",
            "_vix_low",
        ]
    )

    return df


def shift_to_next_business_day(
    macro_df: pl.DataFrame,
    *,
    business_days: Iterable[str] | None = None,
) -> pl.DataFrame:
    """
    Shift macro series forward to align with next Japanese business day (T+1).

    Args:
        macro_df: DataFrame with Date column
        business_days: Optional iterable of YYYY-MM-DD strings defining valid business days

    Returns:
        DataFrame where Date represents the effective date for equity join
    """

    if macro_df.is_empty() or "Date" not in macro_df.columns:
        return macro_df

    try:
        from src.features.calendar_utils import build_next_bday_expr_from_dates
    except Exception:  # pragma: no cover - defensive
        return macro_df

    if business_days:
        expr = build_next_bday_expr_from_dates(list(business_days))
    else:
        expr = None

    if expr is None:
        return macro_df

    return macro_df.with_columns(expr.alias("effective_date"))
