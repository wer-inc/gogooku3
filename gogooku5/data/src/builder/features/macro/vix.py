"""VIX macro feature generation."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ..utils.lazy_io import lazy_load
from .yfinance_utils import (
    flatten_yfinance_columns,
    get_yfinance_module,
    resolve_cached_parquet,
)

LOGGER = logging.getLogger(__name__)
_VIX_TICKER = "^VIX"
_DATE_FMT = "%Y-%m-%d"


def load_vix_history(
    start: str,
    end: str,
    *,
    parquet_path: Path | None = None,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Load VIX price history from cache or yfinance."""

    resolved_cache = None
    if not force_refresh:
        resolved_cache = resolve_cached_parquet(parquet_path, prefix="vix", start=start, end=end)

    if resolved_cache and resolved_cache.exists() and not force_refresh:
        try:
            df = lazy_load(resolved_cache, prefer_ipc=True)
            LOGGER.info("Loaded VIX history from cache: %s (IPC-optimized)", resolved_cache)
            return df
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to read cached VIX parquet (%s): %s", resolved_cache, exc)

    yf = get_yfinance_module(raise_on_missing=False)
    if yf is None:
        LOGGER.warning("yfinance not available; VIX history unavailable")
        return pl.DataFrame()

    start_dt = datetime.strptime(start, _DATE_FMT)
    end_dt = datetime.strptime(end, _DATE_FMT) + timedelta(days=1)

    try:
        data = yf.download(  # type: ignore[attr-defined]
            _VIX_TICKER,
            start=start_dt.strftime(_DATE_FMT),
            end=end_dt.strftime(_DATE_FMT),
            auto_adjust=False,
            progress=False,
            interval="1d",
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to download VIX data via yfinance: %s", exc)
        return pl.DataFrame()

    if getattr(data, "empty", True):
        LOGGER.warning("VIX download returned no rows for %s → %s", start, end)
        return pl.DataFrame()

    data = flatten_yfinance_columns(data, ticker=_VIX_TICKER)
    if data.index.name is None:
        data.index.name = "Date"
    data = data.reset_index()
    data = flatten_yfinance_columns(data, ticker=_VIX_TICKER)

    if "Date" not in data.columns:
        LOGGER.warning("Flattened VIX DataFrame missing Date column; skipping")
        return pl.DataFrame()

    data["Date"] = data["Date"].dt.tz_localize(None)
    df = pl.from_pandas(data, include_index=False)
    df = df.with_columns(pl.col("Date").cast(pl.Date))

    target_cache = parquet_path or resolved_cache
    if target_cache:
        try:
            target_cache.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(target_cache)
            LOGGER.info("Cached VIX history to %s", target_cache)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to cache VIX parquet (%s): %s", target_cache, exc)

    return df


def prepare_vix_features(
    vix_df: pl.DataFrame,
    *,
    feature_prefix: str = "macro_vix",
    spike_threshold: float = 1.5,
) -> pl.DataFrame:
    """Transform VIX history into model features."""

    if vix_df.is_empty() or "Date" not in vix_df.columns or "Close" not in vix_df.columns:
        return pl.DataFrame(schema={"Date": pl.Date})

    close = pl.col("Close")
    eps = 1e-9

    ensure_exprs: list[pl.Expr] = []
    for name in ("High", "Low"):
        if name not in vix_df.columns:
            ensure_exprs.append(pl.lit(None, dtype=pl.Float64).alias(name))
    if ensure_exprs:
        vix_df = vix_df.with_columns(ensure_exprs)

    df = vix_df.sort("Date").with_columns(
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
        ]
    )

    df = df.with_columns(
        (pl.col("_vix_sma5") / (pl.col("_vix_sma20") + eps) - 1.0).alias(f"{feature_prefix}_sma_ratio_5_20")
    )

    df = df.with_columns(
        pl.when(pl.col(f"{feature_prefix}_vol_20").is_not_null())
        .then(
            (
                pl.col(f"{feature_prefix}_vol_20")
                - pl.col(f"{feature_prefix}_vol_20").rolling_mean(window_size=252, min_periods=60)
            )
            / (pl.col(f"{feature_prefix}_vol_20").rolling_std(window_size=252, min_periods=60) + eps)
        )
        .otherwise(None)
        .alias(f"{feature_prefix}_vol_z")
    )

    df = df.with_columns(
        pl.when(pl.col(f"{feature_prefix}_vol_z") > spike_threshold)
        .then(1)
        .otherwise(0)
        .alias(f"{feature_prefix}_spike")
    )

    drop_cols = [c for c in df.columns if c.startswith("_vix_")]
    if drop_cols:
        df = df.drop(drop_cols)
    return df
