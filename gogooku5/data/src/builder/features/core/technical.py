"""Technical indicator features (KAMA, VIDYA, fractional diff, rolling quantiles)."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

# ============================================================================
# Polars-native indicator implementations (Phase A: Quick Wins optimization)
# ============================================================================


def _compute_rsi_polars(
    df: pl.DataFrame,
    column: str,
    period: int,
    group_col: str = "code",
    output_name: str | None = None,
) -> pl.DataFrame:
    """Compute RSI using pure Polars expressions (30-40% faster than pandas).

    Args:
        df: Input dataframe
        column: Column to compute RSI on (typically 'close')
        period: RSI period (e.g., 3, 14)
        group_col: Grouping column (typically 'code' for per-stock)
        output_name: Output column name (default: 'rsi_{period}')

    Returns:
        DataFrame with RSI column added
    """
    eps = 1e-12
    out_name = output_name or f"rsi_{period}"

    # Step 1: Compute delta (price change) per group
    df = df.with_columns(pl.col(column).diff().over(group_col).alias("_delta"))

    # Step 2: Separate gains and losses
    df = df.with_columns(
        [pl.col("_delta").clip(lower_bound=0).alias("_gain"), (-pl.col("_delta")).clip(lower_bound=0).alias("_loss")]
    )

    # Step 3: Compute rolling average of gains and losses per group
    df = df.with_columns(
        [
            pl.col("_gain").rolling_mean(window_size=period, min_periods=period).over(group_col).alias("_avg_gain"),
            pl.col("_loss").rolling_mean(window_size=period, min_periods=period).over(group_col).alias("_avg_loss"),
        ]
    )

    # Step 4: Compute RS and RSI
    df = df.with_columns((100.0 - (100.0 / (1.0 + pl.col("_avg_gain") / (pl.col("_avg_loss") + eps)))).alias(out_name))

    # Cleanup temporary columns
    return df.drop(["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss"])


def _compute_macd_polars(
    df: pl.DataFrame,
    column: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    group_col: str = "code",
) -> pl.DataFrame:
    """Compute MACD using pure Polars expressions (30-40% faster than pandas).

    Args:
        df: Input dataframe
        column: Column to compute MACD on (typically 'close')
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
        group_col: Grouping column (typically 'code' for per-stock)

    Returns:
        DataFrame with macd, macd_signal, macd_histogram columns added
    """
    # Step 1: Compute fast and slow EMAs per group
    df = df.with_columns(
        [
            pl.col(column).ewm_mean(span=fast, ignore_nulls=True).over(group_col).alias("_ema_fast"),
            pl.col(column).ewm_mean(span=slow, ignore_nulls=True).over(group_col).alias("_ema_slow"),
        ]
    )

    # Step 2: MACD line = EMA(fast) - EMA(slow)
    df = df.with_columns((pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd"))

    # Step 3: Signal line = EMA(macd, signal)
    df = df.with_columns(pl.col("macd").ewm_mean(span=signal, ignore_nulls=True).over(group_col).alias("macd_signal"))

    # Step 4: Histogram = MACD - Signal
    df = df.with_columns((pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram"))

    # Cleanup temporary columns
    return df.drop(["_ema_fast", "_ema_slow"])


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
    # Multi-period indicator support
    atr_periods: Sequence[int] = (14, 2)
    enable_parallel: bool = True
    max_parallel_workers: int | None = None
    parallel_row_threshold: int = 250_000


class TechnicalFeatureEngineer:
    def __init__(self, config: TechnicalFeatureConfig | None = None) -> None:
        self.config = config or TechnicalFeatureConfig()

    def add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        if df.is_empty() or cfg.value_column not in df.columns:
            return df

        # ===================================================================
        # Phase A: Quick Wins - Compute RSI/MACD using Polars BEFORE pandas
        # conversion (30-40% faster, avoids pandas overhead)
        # ===================================================================

        # Sort by code and date for proper time-series operations
        df = df.sort([cfg.code_column, cfg.date_column])

        # Compute MACD (12, 26, 9) in Polars
        df = _compute_macd_polars(df, column=cfg.value_column, fast=12, slow=26, signal=9, group_col=cfg.code_column)

        # Compute RSI(3) for CRSI indicator (used later in pandas section)
        df = _compute_rsi_polars(
            df, column=cfg.value_column, period=3, group_col=cfg.code_column, output_name="rsi_3_polars"
        )

        # ===================================================================
        # Pandas section: Complex indicators that need custom loops
        # (KAMA, VIDYA, Aroon, OBV, etc.)
        # ===================================================================

        base_cols = [cfg.code_column, cfg.date_column, cfg.value_column]
        optional_cols = [
            cfg.open_column,
            cfg.high_column,
            cfg.low_column,
            cfg.volume_column,
            # P0 FIX: Use only ret_prev_* (backward-looking) to avoid forward-looking bias
            "ret_prev_1d",
            "ret_prev_5d",
            "ret_prev_10d",
            "ret_prev_20d",
            # Include Polars-computed indicators
            "macd",
            "macd_signal",
            "macd_histogram",
            "rsi_3_polars",
        ]
        available = [col for col in optional_cols if col in df.columns]
        pdf = df.select(base_cols + available).to_pandas()
        pdf[cfg.date_column] = pd.to_datetime(pdf[cfg.date_column])
        pdf.sort_values([cfg.code_column, cfg.date_column], inplace=True)

        groups = [group.copy() for _, group in pdf.groupby(cfg.code_column, sort=False)]
        frames = _compute_group_frames(groups, cfg, len(pdf))
        if not frames:
            return df
        merged = pd.concat(frames, ignore_index=True)
        merged.sort_values([cfg.date_column, cfg.code_column], inplace=True)
        merged[cfg.date_column] = pd.to_datetime(merged[cfg.date_column]).dt.date
        right = pl.from_pandas(merged.drop(columns=[cfg.value_column])).with_columns(
            pl.col(cfg.date_column).cast(pl.Date)
        )
        return df.join(right, on=[cfg.code_column, cfg.date_column], how="left")


def _should_parallelize_groups(num_groups: int, total_rows: int, cfg: TechnicalFeatureConfig) -> bool:
    if not cfg.enable_parallel:
        return False
    if num_groups < 2:
        return False
    if total_rows < cfg.parallel_row_threshold:
        return False
    max_workers = cfg.max_parallel_workers or (os.cpu_count() or 1)
    return max_workers > 1


def _compute_group_frames(
    groups: list[pd.DataFrame],
    cfg: TechnicalFeatureConfig,
    total_rows: int,
) -> list[pd.DataFrame]:
    if not groups:
        return []
    if _should_parallelize_groups(len(groups), total_rows, cfg):
        max_workers = cfg.max_parallel_workers or (os.cpu_count() or 1)
        max_workers = min(max_workers, len(groups))
        payloads = [(group, cfg) for group in groups]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_process_group_payload, payloads))
    return [_compute_indicators_for_group(group, cfg) for group in groups]


def _process_group_payload(payload: tuple[pd.DataFrame, TechnicalFeatureConfig]) -> pd.DataFrame:
    group, cfg = payload
    return _compute_indicators_for_group(group, cfg)


def _compute_indicators_for_group(group: pd.DataFrame, cfg: TechnicalFeatureConfig) -> pd.DataFrame:
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

    # Phase A: MACD already computed in Polars (30-40% faster)
    # Columns macd, macd_signal, macd_histogram are already in dataframe
    # (pandas MACD computation REMOVED for performance)
    # Legacy pandas code (DISABLED):
    # macd_fast = values.ewm(span=12, adjust=False).mean()
    # macd_slow = values.ewm(span=26, adjust=False).mean()
    # macd = macd_fast - macd_slow
    # signal = macd.ewm(span=9, adjust=False).mean()
    # g["macd"] = macd
    # g["macd_signal"] = signal
    # g["macd_histogram"] = macd - signal

    high = g[cfg.high_column] if cfg.high_column in g else None
    low = g[cfg.low_column] if cfg.low_column in g else None
    if high is not None and low is not None:
        g["high_low_ratio"] = high / (low + eps)
        g["close_to_high"] = (high - values) / ((high - low) + eps)
        g["close_to_low"] = (values - low) / ((high - low) + eps)

        # Use reindex to ensure index alignment for shift operations
        shifted_close_tr = values.shift(1).reindex(g.index)
        true_range = pd.concat(
            [
                high - low,
                (high - shifted_close_tr).abs(),
                (low - shifted_close_tr).abs(),
            ],
            axis=1,
        ).max(axis=1)
        # Compute ATR for all configured periods
        for period in cfg.atr_periods:
            g[f"atr_{period}"] = true_range.ewm(span=period, adjust=False).mean()

        log_hl = np.log(high / low) ** 2
        g["realized_volatility"] = (log_hl.rolling(window=20, min_periods=20).sum() / (4.0 * np.log(2))).pow(0.5)

    # Phase 2: try multiple column names for returns
    returns_1d_col = None
    for candidate in ("ret_prev_1d", "returns_1d", "log_returns_1d"):
        if candidate in g.columns:
            returns_1d_col = candidate
            break

    if returns_1d_col is not None:
        returns_1d = g[returns_1d_col]
        for window, name in (
            (5, "volatility_5d"),
            (10, "volatility_10d"),
            (20, "volatility_20d"),
            (60, "volatility_60d"),
        ):
            g[name] = returns_1d.rolling(window=window, min_periods=window).std() * np.sqrt(252.0)

        # EWMA-based realized volatility (RV_EWMA)
        # Half-life approximately 16.7 days when alpha=0.06 (lambda ≈ 0.94).
        # This follows sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_t^2.
        alpha = 0.06
        rv_ewm2 = returns_1d.pow(2).ewm(alpha=alpha, adjust=False).mean()
        g["rv_ewm2"] = rv_ewm2
        g["rv_ewm"] = (rv_ewm2 * 252.0) ** 0.5

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

    # ========================================================================
    # P0: Additional Technical Indicators (情報の重なりを最小化、短期〜中期検出力向上)
    # ========================================================================

    # 1. ADX / DMI (14) - トレンド強度の独立センサー
    if high is not None and low is not None:
        # True Range (already computed above)
        tr = true_range

        # +DM and -DM
        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)

        # When both +DM and -DM are positive, set the larger to 0
        both_positive = (plus_dm > 0) & (minus_dm > 0)
        larger_is_plus = plus_dm >= minus_dm
        plus_dm = plus_dm.where(~both_positive | larger_is_plus, 0)
        minus_dm = minus_dm.where(~both_positive | ~larger_is_plus, 0)

        # EMA(14) smoothing
        period = 14
        alpha = 2.0 / (period + 1.0)
        tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

        # PDI and MDI
        eps_dmi = 1e-12
        pdi = 100.0 * (plus_dm_smooth / (tr_smooth + eps_dmi))
        mdi = 100.0 * (minus_dm_smooth / (tr_smooth + eps_dmi))

        # ADX = EMA(14) of |PDI - MDI| / (PDI + MDI)
        dx = 100.0 * (pdi - mdi).abs() / ((pdi + mdi) + eps_dmi)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        # Shift(1) for left-closed
        g["dmi_pos_14"] = pdi.shift(1)
        g["dmi_neg_14"] = mdi.shift(1)
        g["adx_14"] = adx.shift(1)

        # ADX Z-score (20-day)
        adx_ma20 = g["adx_14"].rolling(window=20, min_periods=10).mean()
        adx_std20 = g["adx_14"].rolling(window=20, min_periods=10).std()
        g["adx_14_z20"] = ((g["adx_14"] - adx_ma20) / (adx_std20 + eps)).shift(1)

        # 2. Donchian Channels (20)
        don_period = 20
        don_high = high.rolling(window=don_period, min_periods=don_period).max()
        don_low = low.rolling(window=don_period, min_periods=don_period).min()
        don_width = (don_high - don_low) / (values + eps)

        # Shift(1) for left-closed
        g["don_high_20"] = don_high.shift(1)
        g["don_low_20"] = don_low.shift(1)
        g["don_width_20"] = don_width.shift(1)

        # Break flags (current close vs shifted donchian)
        shifted_close = values.shift(1).reindex(g.index)
        g["don_break_20_up"] = ((shifted_close > g["don_high_20"]) & g["don_high_20"].notna()).astype(int)
        g["don_break_20_down"] = ((shifted_close < g["don_low_20"]) & g["don_low_20"].notna()).astype(int)

        # 3. Keltner Channels (EMA20, ATR×1.5) + TTM Squeeze
        kc_period = 20
        kc_mult = 1.5
        kc_mid = values.ewm(span=kc_period, adjust=False).mean()
        kc_atr = tr.ewm(span=kc_period, adjust=False).mean()
        kc_up = kc_mid + (kc_mult * kc_atr)
        kc_dn = kc_mid - (kc_mult * kc_atr)

        # Shift(1) for left-closed
        g["kc_mid_20"] = kc_mid.shift(1)
        g["kc_up_20"] = kc_up.shift(1)
        g["kc_dn_20"] = kc_dn.shift(1)

        # TTM Squeeze: bb_bw < kc_bw
        kc_bw = (kc_up - kc_dn) / (kc_mid + eps)
        if "bb_bw" in g.columns:
            g["ttm_squeeze_on"] = (g["bb_bw"].shift(1) < kc_bw.shift(1).reindex(g.index)).astype(int)
            # TTM Squeeze Fire: squeeze_on→OFF & price breaks BB
            squeeze_was_on = g["ttm_squeeze_on"].shift(1) == 1
            squeeze_now_off = g["ttm_squeeze_on"] == 0
            bb_upper = rolling_mean_20 + 2 * rolling_std_20
            bb_lower = rolling_mean_20 - 2 * rolling_std_20
            price_breaks_bb = (shifted_close > bb_upper.shift(1).reindex(g.index)) | (
                shifted_close < bb_lower.shift(1).reindex(g.index)
            )
            g["ttm_squeeze_fire"] = (squeeze_was_on & squeeze_now_off & price_breaks_bb).astype(int)
        else:
            g["ttm_squeeze_on"] = 0
            g["ttm_squeeze_fire"] = 0

        # 4. Aroon (25) - トレンドの"新鮮さ"検出
        aroon_period = 25
        aroon_up = []
        aroon_dn = []
        high_arr = high.values
        low_arr = low.values
        for i in range(len(high_arr)):
            if i < aroon_period:
                aroon_up.append(np.nan)
                aroon_dn.append(np.nan)
            else:
                window_start = i - aroon_period + 1
                window_high = high_arr[window_start : i + 1]
                window_low = low_arr[window_start : i + 1]
                # 最高値/最低値の位置（窓内の相対位置）
                highest_pos = np.argmax(window_high)
                lowest_pos = np.argmin(window_low)
                days_since_high = aroon_period - 1 - highest_pos
                days_since_low = aroon_period - 1 - lowest_pos
                aroon_up.append(100.0 * (aroon_period - days_since_high) / aroon_period)
                aroon_dn.append(100.0 * (aroon_period - days_since_low) / aroon_period)

        # Use high.index to match the actual data length
        g["aroon_up_25"] = pd.Series(aroon_up, index=high.index).reindex(g.index).shift(1)
        g["aroon_dn_25"] = pd.Series(aroon_dn, index=high.index).reindex(g.index).shift(1)
        g["aroon_osc_25"] = g["aroon_up_25"] - g["aroon_dn_25"]

        # 7. ATR正規化ギャップ/レンジ
        if "ret_overnight" in g.columns:
            gap_atr = g["ret_overnight"].abs() / (g[f"atr_{cfg.atr_periods[0]}"] / values + eps)
            g["gap_atr"] = gap_atr.shift(1)
        else:
            g["gap_atr"] = np.nan

        idr_atr = (high - low) / (g[f"atr_{cfg.atr_periods[0]}"] + eps)
        g["idr_atr"] = idr_atr.shift(1)

    # 5. Connors RSI (CRSI: RSI(3), StreakRSI(2), PercentRank(100))
    # Define pandas RSI helper (used for StreakRSI below)
    def rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        rs = gain / (loss + eps)
        return 100.0 - (100.0 / (1.0 + rs))

    # Phase A: RSI(3) already computed in Polars (30-40% faster)
    # Use pre-computed Polars RSI column
    if "rsi_3_polars" in g.columns:
        rsi_3 = g["rsi_3_polars"]
    else:
        # Fallback if Polars version not available
        rsi_3 = rsi(values, 3)

    # StreakRSI(2): 連続上昇/下降日数の符号付きRSI
    streak = []
    streak_val = 0
    for i in range(len(values)):
        if i == 0:
            streak.append(0)
        else:
            if values.iloc[i] > values.iloc[i - 1]:
                streak_val = max(1, streak_val + 1) if streak_val > 0 else 1
            elif values.iloc[i] < values.iloc[i - 1]:
                streak_val = min(-1, streak_val - 1) if streak_val < 0 else -1
            else:
                streak_val = 0
            streak.append(streak_val)

    # Use values.index to match the actual data length
    streak_series = pd.Series(streak, index=values.index).reindex(g.index)
    streak_rsi = rsi(streak_series, 2)

    # PercentRank(100): 現在の価格が過去100日間の何パーセンタイルか
    def percent_rank(series, window):
        ranks = []
        for i in range(len(series)):
            if i < window:
                ranks.append(np.nan)
            else:
                window_data = series.iloc[i - window : i + 1]
                current = series.iloc[i]
                rank = (window_data < current).sum() / window
                ranks.append(rank * 100.0)
        return pd.Series(ranks, index=series.index)

    pr_100 = percent_rank(values, 100)

    # CRSI = (RSI(3) + StreakRSI(2) + PercentRank(100)) / 3
    crsi = (rsi_3 + streak_rsi + pr_100) / 3.0
    g["crsi_3_2_100"] = crsi.shift(1)

    # 6. OBV / Chaikin Money Flow (20)
    if volume is not None:
        # OBV: On Balance Volume
        obv = []
        obv_val = 0.0
        for i in range(len(values)):
            if i == 0:
                obv.append(0.0)
            else:
                if values.iloc[i] > values.iloc[i - 1]:
                    obv_val += volume.iloc[i]
                elif values.iloc[i] < values.iloc[i - 1]:
                    obv_val -= volume.iloc[i]
                # If price unchanged, OBV stays the same
                obv.append(obv_val)

        # Use values.index to match the actual data length
        obv_series = pd.Series(obv, index=values.index).reindex(g.index)
        g["obv"] = obv_series.shift(1)

        # OBV Z-score (20-day)
        obv_ma20 = obv_series.rolling(window=20, min_periods=10).mean()
        obv_std20 = obv_series.rolling(window=20, min_periods=10).std()
        g["obv_z20"] = ((obv_series - obv_ma20) / (obv_std20 + eps)).shift(1)

        # Chaikin Money Flow (20)
        if high is not None and low is not None:
            clv = ((values - low) - (high - values)) / ((high - low) + eps)  # Close Location Value
            cmf_raw = (clv * volume).rolling(window=20, min_periods=20).sum() / (
                volume.rolling(window=20, min_periods=20).sum() + eps
            )
            g["cmf_20"] = cmf_raw.shift(1)
        else:
            g["cmf_20"] = np.nan
    else:
        g["obv"] = np.nan
        g["obv_z20"] = np.nan
        g["cmf_20"] = np.nan

    # 8. Amihud Illiquidity (20, median) - 流動性摩擦の定量化
    if volume is not None and "ret_prev_1d" in g.columns:
        dollar_volume = values * volume
        amihud_raw = g["ret_prev_1d"].abs() / (dollar_volume + eps)
        amihud_20 = amihud_raw.rolling(window=20, min_periods=10).median()
        g["amihud_20"] = amihud_20.shift(1)

        # Amihud Z-score (20-day)
        amihud_ma20 = amihud_20.rolling(window=20, min_periods=10).mean()
        amihud_std20 = amihud_20.rolling(window=20, min_periods=10).std()
        g["amihud_z20"] = ((amihud_20 - amihud_ma20) / (amihud_std20 + eps)).shift(1)
    elif "ret_prev_1d" in g.columns:
        g["amihud_20"] = np.nan
        g["amihud_z20"] = np.nan
    else:
        g["amihud_20"] = np.nan
        g["amihud_z20"] = np.nan

    return g
