"""Technical indicator features (KAMA, VIDYA, fractional diff, rolling quantiles)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl


def _kama(series: pd.Series, window: int, fast: int, slow: int) -> pd.Series:
    x = series.astype(float).to_numpy()
    n = x.size
    out = np.full(n, np.nan, dtype=float)
    if n == 0 or window < 1:
        return pd.Series(out, index=series.index)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    kama_prev = x[0]
    out[0] = kama_prev
    for t in range(1, n):
        if t < window:
            out[t] = np.nan
            kama_prev = x[t]
            continue
        change = abs(x[t] - x[t - window])
        volatility = np.sum(np.abs(np.diff(x[t - window : t + 1]))) + 1e-12
        er = change / volatility
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_prev = kama_prev + sc * (x[t] - kama_prev)
        out[t] = kama_prev
    return pd.Series(out, index=series.index)


def _vidya(series: pd.Series, window: int) -> pd.Series:
    x = series.astype(float).to_numpy()
    n = x.size
    out = np.full(n, np.nan, dtype=float)
    if n == 0 or window < 2:
        return pd.Series(out, index=series.index)

    def cmo(arr: np.ndarray) -> float:
        diff = np.diff(arr)
        up = np.sum(diff[diff > 0])
        down = np.sum(-diff[diff < 0])
        denom = up + down + 1e-12
        return 100.0 * (up - down) / denom

    v_prev = x[0]
    out[0] = v_prev
    k = 2.0 / (window + 1.0)
    for t in range(1, n):
        if t < window:
            out[t] = np.nan
            v_prev = x[t]
            continue
        c = abs(cmo(x[t - window : t + 1])) / 100.0
        alpha = c * k
        v_prev = v_prev + alpha * (x[t] - v_prev)
        out[t] = v_prev
    return pd.Series(out, index=series.index)


def _fractional_diff(series: pd.Series, d: float, window: int) -> pd.Series:
    x = series.astype(float).to_numpy()
    n = x.size
    w = np.zeros(window, dtype=float)
    w[0] = 1.0
    for k in range(1, window):
        w[k] = -w[k - 1] * (d - (k - 1)) / k
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        kmax = min(t + 1, window)
        if kmax <= 0:
            out[t] = np.nan
            continue
        start = max(0, t - kmax + 1)
        segment = x[start : t + 1][::-1]
        weights = w[: segment.shape[0]]
        out[t] = float(np.dot(weights, segment))
    return pd.Series(out, index=series.index)


def _rolling_quantiles(series: pd.Series, window: int, quants: Sequence[float]) -> pd.DataFrame:
    r = series.rolling(window=window, min_periods=window)
    data = {f"rq_{window}_{int(q*100):02d}": r.quantile(q) for q in quants}
    return pd.DataFrame(data, index=series.index)


@dataclass
class TechnicalFeatureConfig:
    code_column: str = "code"
    date_column: str = "date"
    value_column: str = "close"
    open_column: str = "open"
    high_column: str = "high"
    low_column: str = "low"
    volume_column: str = "volume"
    kama_set: Sequence[tuple[int, int, int]] = ((10, 2, 30),)
    vidya_windows: Sequence[int] = (14,)
    fractional_diff_d: float = 0.4
    fractional_diff_window: int = 100
    rq_windows: Sequence[int] = (63,)
    rq_quantiles: Sequence[float] = (0.1, 0.5, 0.9)
    roll_std_windows: Sequence[int] = (20,)


class TechnicalFeatureEngineer:
    def __init__(self, config: TechnicalFeatureConfig | None = None) -> None:
        self.config = config or TechnicalFeatureConfig()

    def add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        if df.is_empty() or cfg.value_column not in df.columns:
            return df

        base_cols = [cfg.code_column, cfg.date_column, cfg.value_column]
        optional_cols = [
            cfg.open_column,
            cfg.high_column,
            cfg.low_column,
            cfg.volume_column,
            "returns_1d",
            "returns_5d",
            "returns_10d",
            "returns_20d",
        ]
        available = [col for col in optional_cols if col in df.columns]
        pdf = df.select(base_cols + available).to_pandas()
        pdf[cfg.date_column] = pd.to_datetime(pdf[cfg.date_column])
        pdf.sort_values([cfg.code_column, cfg.date_column], inplace=True)

        frames = []
        for code, group in pdf.groupby(cfg.code_column, sort=False):
            g = group.copy()
            values = g[cfg.value_column]
            eps = 1e-12

            for win, fast, slow in cfg.kama_set:
                g[f"kama_{win}_{fast}_{slow}"] = _kama(values, int(win), int(fast), int(slow))

            for vw in cfg.vidya_windows:
                g[f"vidya_{int(vw)}"] = _vidya(values, int(vw))

            g[f"fdiff_{str(cfg.fractional_diff_d).replace('.', 'p')}_{cfg.fractional_diff_window}"] = _fractional_diff(
                values, cfg.fractional_diff_d, cfg.fractional_diff_window
            )

            log_close = np.log(values.replace(0, np.nan))
            for horizon in (1, 5, 10, 20):
                g[f"log_returns_{horizon}d"] = log_close.diff(horizon)
                # REMOVED: g[f"feat_ret_{horizon}d"] = values.shift(-horizon) / (values + eps) - 1.0
                # Reason: shift(-horizon) introduces forward-looking data leak (uses future prices)

            for window in (5, 10, 20, 60, 120):
                g[f"sma_{window}"] = values.rolling(window=window, min_periods=window).mean()

            ema_map = {5: "ema_5", 10: "ema_10", 20: "ema_20", 60: "ema_60", 120: "ema_120", 200: "ema_200"}
            for span, name in ema_map.items():
                g[name] = values.ewm(span=span, adjust=False).mean()

            if "ema_5" in g and "ema_20" in g:
                g["ma_gap_5_20"] = (g["ema_5"] - g["ema_20"]) / (g["ema_20"] + eps)
            if "ema_20" in g and "ema_60" in g:
                g["ma_gap_20_60"] = (g["ema_20"] - g["ema_60"]) / (g["ema_60"] + eps)

            for window in (5, 20, 60):
                sma_col = f"sma_{window}"
                if sma_col in g:
                    g[f"price_to_sma{window}"] = values / (g[sma_col] + eps)

            volume = g[cfg.volume_column] if cfg.volume_column in g else None
            if volume is not None:
                vol_ma_5 = volume.rolling(window=5, min_periods=5).mean()
                vol_ma_20 = volume.rolling(window=20, min_periods=20).mean()
                if "volume_ma_5" not in g:
                    g["volume_ma_5"] = vol_ma_5
                if "volume_ma_20" not in g:
                    g["volume_ma_20"] = vol_ma_20
                g["volume_ratio_5"] = volume / (vol_ma_5 + eps)
                g["volume_ratio_20"] = volume / (vol_ma_20 + eps)

            macd_fast = values.ewm(span=12, adjust=False).mean()
            macd_slow = values.ewm(span=26, adjust=False).mean()
            macd = macd_fast - macd_slow
            signal = macd.ewm(span=9, adjust=False).mean()
            g["macd"] = macd
            g["macd_signal"] = signal
            g["macd_histogram"] = macd - signal

            high = g[cfg.high_column] if cfg.high_column in g else None
            low = g[cfg.low_column] if cfg.low_column in g else None
            if high is not None and low is not None:
                g["high_low_ratio"] = high / (low + eps)
                g["close_to_high"] = (high - values) / ((high - low) + eps)
                g["close_to_low"] = (values - low) / ((high - low) + eps)

                true_range = pd.concat(
                    [
                        high - low,
                        (high - values.shift(1)).abs(),
                        (low - values.shift(1)).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                g["atr_14"] = true_range.ewm(span=14, adjust=False).mean()

                log_hl = np.log(high / low) ** 2
                g["realized_volatility"] = (log_hl.rolling(window=20, min_periods=20).sum() / (4.0 * np.log(2))).pow(
                    0.5
                )

            returns_1d = g.get("returns_1d")
            if returns_1d is not None:
                for window, name in (
                    (5, "volatility_5d"),
                    (10, "volatility_10d"),
                    (20, "volatility_20d"),
                    (60, "volatility_60d"),
                ):
                    g[name] = returns_1d.rolling(window=window, min_periods=window).std() * np.sqrt(252.0)

            rolling_mean_20 = values.rolling(window=20, min_periods=20).mean()
            rolling_std_20 = values.rolling(window=20, min_periods=20).std()
            upper = rolling_mean_20 + 2 * rolling_std_20
            lower = rolling_mean_20 - 2 * rolling_std_20
            g["bb_pct_b"] = ((values - lower) / ((upper - lower) + eps)).clip(0.0, 1.0)
            g["bb_bw"] = (upper - lower) / (rolling_mean_20 + eps)
            g["z_close_20"] = (values - rolling_mean_20) / (rolling_std_20 + eps)

            for window in cfg.rq_windows:
                rq = _rolling_quantiles(values, int(window), cfg.rq_quantiles)
                g = pd.concat([g, rq.reset_index(drop=True)], axis=1)

            for win in cfg.roll_std_windows:
                g[f"roll_std_{int(win)}"] = values.rolling(window=int(win), min_periods=int(win)).std()

            frames.append(g)

        merged = pd.concat(frames, ignore_index=True)
        merged.sort_values([cfg.date_column, cfg.code_column], inplace=True)
        merged[cfg.date_column] = pd.to_datetime(merged[cfg.date_column]).dt.date
        right = pl.from_pandas(merged.drop(columns=[cfg.value_column])).with_columns(
            pl.col(cfg.date_column).cast(pl.Date)
        )
        return df.join(right, on=[cfg.code_column, cfg.date_column], how="left")
