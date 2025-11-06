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

# Yahoo Finance tickers for cross-market risk proxies
_TICKERS: dict[str, str] = {
    "SPY": "SPY",  # S&P 500 ETF
    "VIX": "^VIX",  # Implied volatility (front month)
    "VIX9D": "^VIX9D",  # 9-day VIX
    "VIX3M": "^VIX3M",  # 3-month VIX
    "HYG": "HYG",  # High-yield corporate bond ETF
    "LQD": "LQD",  # Investment-grade corporate bond ETF
    "TLT": "TLT",  # 20+ year Treasury ETF
    "IEF": "IEF",  # 7-10 year Treasury ETF
}


def _download_single_ticker(
    yf_module,
    ticker: str,
    start: datetime,
    end: datetime,
) -> pl.DataFrame | None:
    """Fetch a single ticker from yfinance and normalize columns."""
    try:
        data = yf_module.download(  # type: ignore[attr-defined]
            ticker,
            start=start.strftime(_DATE_FMT),
            end=end.strftime(_DATE_FMT),
            auto_adjust=False,
            progress=False,
            interval="1d",
        )
    except Exception as exc:  # pragma: no cover - defensive network guard
        logger.warning("Failed to download %s via yfinance: %s", ticker, exc)
        return None

    if getattr(data, "empty", True):
        logger.warning("yfinance download returned no rows for %s", ticker)
        return None

    data = flatten_yfinance_columns(data, ticker=ticker)
    if data.index.name is None:
        data.index.name = "Date"
    data = data.reset_index()
    data = flatten_yfinance_columns(data, ticker=ticker)

    if "Date" not in data.columns:
        logger.warning("Flattened DataFrame for %s missing Date column; skipping", ticker)
        return None

    data["Date"] = data["Date"].dt.tz_localize(None)
    df = pl.from_pandas(data, include_index=False)
    df = df.with_columns(pl.col("Date").cast(pl.Date))
    return df


def load_cross_market_history(
    start: str,
    end: str,
    *,
    parquet_path: Path | None = None,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """
    Load (or fetch) cross-market risk proxy history used for macro features.

    Includes: SPY, VIX (spot), VIX9D, VIX3M, HYG, LQD, TLT, IEF.
    """
    resolved_cache: Path | None = None
    if not force_refresh:
        resolved_cache = resolve_cached_parquet(
            parquet_path,
            prefix="cross_market",
            start=start,
            end=end,
        )

    if resolved_cache and resolved_cache.exists() and not force_refresh:
        try:
            df = pl.read_parquet(resolved_cache)
            logger.info("Loaded cross-market history from cache: %s", resolved_cache)
            return df
        except Exception as exc:  # pragma: no cover - IO guard
            logger.warning("Failed to read cached cross-market parquet (%s): %s", resolved_cache, exc)

    yf = get_yfinance_module(raise_on_missing=False)
    if yf is None:
        logger.warning(
            "yfinance is not available; cannot fetch cross-market history. "
            "Install yfinance or provide cached parquet."
        )
        return pl.DataFrame()

    start_dt = datetime.strptime(start, _DATE_FMT)
    # yfinance end is exclusive, so add one day
    end_dt = datetime.strptime(end, _DATE_FMT) + timedelta(days=1)

    frames: dict[str, pl.DataFrame] = {}
    for name, ticker in _TICKERS.items():
        df = _download_single_ticker(yf, ticker, start_dt, end_dt)
        if df is None or df.is_empty():
            continue
        frames[name] = df

    if not frames:
        logger.warning("No cross-market tickers fetched successfully")
        return pl.DataFrame()

    result: pl.DataFrame | None = None
    for name, frame in frames.items():
        columns = ["Date"]
        rename_map: dict[str, str] = {}

        if "Close" in frame.columns:
            columns.append("Close")
            rename_map["Close"] = f"{name.lower()}_close"

        if name == "SPY" and "Open" in frame.columns:
            # SPY open is required for overnight gap calculations
            columns.append("Open")
            rename_map["Open"] = f"{name.lower()}_open"

        if not rename_map:
            logger.debug("No usable columns found for %s; skipping join", name)
            continue

        subset = frame.select(columns).rename(rename_map)
        if result is None:
            result = subset
        else:
            result = result.join(subset, on="Date", how="outer")

    if result is None or result.is_empty():
        logger.warning("Cross-market history construction produced no rows")
        return pl.DataFrame()

    result = result.sort("Date")

    cache_target = parquet_path or resolved_cache
    if cache_target:
        try:
            cache_target.parent.mkdir(parents=True, exist_ok=True)
            result.write_parquet(cache_target)
            logger.info("Cached cross-market history to %s", cache_target)
        except Exception as exc:  # pragma: no cover - IO guard
            logger.warning("Failed to cache cross-market parquet (%s): %s", cache_target, exc)

    return result


def prepare_cross_market_features(history: pl.DataFrame) -> pl.DataFrame:
    """
    Generate engineered cross-market macro features from raw history.

    Returns a DataFrame with Date and the following feature groups (when data permits):
      - VRP (variance risk premium): macro_vrp_spy, macro_vrp_spy_zscore_252, macro_vrp_spy_high_flag
      - Credit spread proxy (HYG vs LQD): macro_credit_spread_ratio, macro_credit_spread_zscore_63
      - Rates term slope (TLT vs IEF): macro_rates_term_ratio, macro_rates_term_zscore_63
      - VIX term structure: macro_vix_term_slope, macro_vix_term_ratio, macro_vix_term_zscore_126
      - SPY overnight/intraday returns: macro_us_spy_overnight_ret, macro_us_spy_intraday_ret,
        macro_us_spy_ret_1d, macro_us_spy_rv_20
    """
    if history.is_empty() or "Date" not in history.columns:
        return pl.DataFrame(schema={"Date": pl.Date})

    df = history.sort("Date").with_columns(pl.col("Date").cast(pl.Date))
    eps = 1e-12

    # Ensure numeric columns are float64 for stable operations
    numeric_cols = [c for c in df.columns if c != "Date"]
    if numeric_cols:
        df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in numeric_cols])

    working_cols: list[str] = ["Date"]

    # === SPY returns and volatility ===
    if {"spy_close"}.issubset(df.columns):
        df = df.with_columns(
            [
                (pl.col("spy_close") / pl.col("spy_close").shift(1) - 1.0).alias("_spy_ret_1d"),
                (pl.col("spy_close") / pl.col("spy_close").shift(5) - 1.0).alias("_spy_ret_5d"),
            ]
        )
        df = df.with_columns(
            (pl.col("_spy_ret_1d").rolling_std(window_size=20, min_periods=10) * (252.0**0.5)).alias(
                "macro_us_spy_rv_20"
            )
        )
        df = df.with_columns(
            (pl.col("_spy_ret_1d")).alias("macro_us_spy_ret_1d"),
            (pl.col("_spy_ret_5d")).alias("macro_us_spy_ret_5d"),
        )
        working_cols.extend(
            [
                "macro_us_spy_ret_1d",
                "macro_us_spy_ret_5d",
                "macro_us_spy_rv_20",
            ]
        )

        if "spy_open" in df.columns:
            df = df.with_columns(
                [
                    (pl.col("spy_open") / pl.col("spy_close").shift(1) - 1.0).alias("macro_us_spy_overnight_ret"),
                    (pl.col("spy_close") / pl.col("spy_open") - 1.0).alias("macro_us_spy_intraday_ret"),
                ]
            )
            working_cols.extend(
                [
                    "macro_us_spy_overnight_ret",
                    "macro_us_spy_intraday_ret",
                ]
            )

    # === Variance Risk Premium (VRP) ===
    if {"vix_close", "macro_us_spy_rv_20"}.issubset(df.columns):
        df = df.with_columns(
            ((pl.col("vix_close") / 100.0) ** 2 - (pl.col("macro_us_spy_rv_20") ** 2)).alias("macro_vrp_spy")
        )
        df = df.with_columns(
            [
                pl.col("macro_vrp_spy").rolling_mean(window_size=252, min_periods=126).alias("_vrp_mean_252"),
                pl.col("macro_vrp_spy").rolling_std(window_size=252, min_periods=126).alias("_vrp_std_252"),
            ]
        )
        df = df.with_columns(
            ((pl.col("macro_vrp_spy") - pl.col("_vrp_mean_252")) / (pl.col("_vrp_std_252") + eps)).alias(
                "macro_vrp_spy_zscore_252"
            )
        )
        df = df.with_columns(
            pl.when(pl.col("macro_vrp_spy_zscore_252") > 1.0)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("macro_vrp_spy_high_flag")
        )
        working_cols.extend(
            [
                "macro_vrp_spy",
                "macro_vrp_spy_zscore_252",
                "macro_vrp_spy_high_flag",
            ]
        )
        df = df.drop(["_vrp_mean_252", "_vrp_std_252"], strict=False)

    # === Credit spread proxy (HYG vs LQD) ===
    if {"hyg_close", "lqd_close"}.issubset(df.columns):
        df = df.with_columns((pl.col("hyg_close") / (pl.col("lqd_close") + eps)).alias("macro_credit_spread_ratio"))
        df = df.with_columns(
            [
                pl.col("macro_credit_spread_ratio")
                .rolling_mean(window_size=63, min_periods=30)
                .alias("_credit_mean_63"),
                pl.col("macro_credit_spread_ratio").rolling_std(window_size=63, min_periods=30).alias("_credit_std_63"),
            ]
        )
        df = df.with_columns(
            (
                (pl.col("macro_credit_spread_ratio") - pl.col("_credit_mean_63")) / (pl.col("_credit_std_63") + eps)
            ).alias("macro_credit_spread_zscore_63")
        )
        working_cols.extend(
            [
                "macro_credit_spread_ratio",
                "macro_credit_spread_zscore_63",
            ]
        )
        df = df.drop(["_credit_mean_63", "_credit_std_63"], strict=False)

    # === Rates term slope (TLT vs IEF) ===
    if {"tlt_close", "ief_close"}.issubset(df.columns):
        df = df.with_columns((pl.col("tlt_close") / (pl.col("ief_close") + eps)).alias("macro_rates_term_ratio"))
        df = df.with_columns(
            [
                pl.col("macro_rates_term_ratio").rolling_mean(window_size=63, min_periods=30).alias("_term_mean_63"),
                pl.col("macro_rates_term_ratio").rolling_std(window_size=63, min_periods=30).alias("_term_std_63"),
            ]
        )
        df = df.with_columns(
            ((pl.col("macro_rates_term_ratio") - pl.col("_term_mean_63")) / (pl.col("_term_std_63") + eps)).alias(
                "macro_rates_term_zscore_63"
            )
        )
        working_cols.extend(
            [
                "macro_rates_term_ratio",
                "macro_rates_term_zscore_63",
            ]
        )
        df = df.drop(["_term_mean_63", "_term_std_63"], strict=False)

    # === VIX term structure metrics ===
    if {"vix_close", "vix3m_close"}.issubset(df.columns):
        df = df.with_columns((pl.col("vix3m_close") - pl.col("vix_close")).alias("macro_vix_term_slope"))
        working_cols.append("macro_vix_term_slope")

    if {"vix9d_close", "vix3m_close"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("vix9d_close") / (pl.col("vix3m_close") + eps) - 1.0).alias("macro_vix_term_ratio")
        )
        working_cols.append("macro_vix_term_ratio")

    if {"macro_vix_term_slope"}.issubset(df.columns):
        df = df.with_columns(
            [
                pl.col("macro_vix_term_slope")
                .rolling_mean(window_size=126, min_periods=60)
                .alias("_vix_term_mean_126"),
                pl.col("macro_vix_term_slope").rolling_std(window_size=126, min_periods=60).alias("_vix_term_std_126"),
            ]
        )
        df = df.with_columns(
            (
                (pl.col("macro_vix_term_slope") - pl.col("_vix_term_mean_126")) / (pl.col("_vix_term_std_126") + eps)
            ).alias("macro_vix_term_zscore_126")
        )
        working_cols.append("macro_vix_term_zscore_126")
        df = df.drop(["_vix_term_mean_126", "_vix_term_std_126"], strict=False)

    final_cols = [col for col in working_cols if col in df.columns]
    if not final_cols:
        logger.warning("No cross-market features generated; returning Date column only")
        return df.select(["Date"])

    return df.select(final_cols)
