"""
Polarsベースのテクニカル指標計算モジュール
高速かつメモリ効率的な実装
"""

import polars as pl
import numpy as np
from typing import Optional, List


class TechnicalIndicators:
    """Polarsベースの高速テクニカル指標計算"""
    
    @staticmethod
    def add_returns(df: pl.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pl.DataFrame:
        """リターン計算"""
        exprs = []
        for period in periods:
            exprs.extend([
                (pl.col("Close") / pl.col("Close").shift(period) - 1).alias(f"px_returns_{period}d"),
                ((pl.col("Close") / pl.col("Close").shift(period)).log()).alias(f"px_log_returns_{period}d")
            ])
        return df.with_columns(exprs)
    
    @staticmethod
    def add_volatility(df: pl.DataFrame, windows: List[int] = [20, 60]) -> pl.DataFrame:
        """ボラティリティ（年率化）"""
        for window in windows:
            # ボラティリティ計算（min_periods明示）
            df = df.with_columns(
                (pl.col("px_returns_1d").rolling_std(window, min_periods=window) * np.sqrt(252)).alias(f"px_volatility_{window}d")
            )
            # 有効フラグ（十分なデータがある場合のみTrue）
            df = df.with_columns(
                pl.col(f"px_volatility_{window}d").is_not_null().alias(f"is_vol{window}_valid")
            )
        return df
    
    @staticmethod
    def add_realized_volatility(df: pl.DataFrame, window: int = 20, method: str = "parkinson") -> pl.DataFrame:
        """実現ボラティリティ（Parkinson/Garman-Klass）"""
        if method == "parkinson":
            # Parkinson volatility: sqrt(1/(4*ln(2)) * mean(ln(H/L)^2))
            park_var = (pl.col("High") / pl.col("Low")).log().pow(2) / (4 * np.log(2))
            df = df.with_columns((park_var.rolling_mean(window, min_periods=window).sqrt() * np.sqrt(252)).alias(f"px_park_vol_{window}d"))
            df = df.with_columns(pl.col(f"px_park_vol_{window}d").is_not_null().alias(f"is_park_vol{window}_valid"))
        elif method == "garman_klass":
            # Garman-Klass: more complex formula using OHLC
            term1 = 0.5 * (pl.col("High") / pl.col("Low")).log().pow(2)
            term2 = (2 * np.log(2) - 1) * (pl.col("Close") / pl.col("Open")).log().pow(2)
            gk_var = term1 - term2
            df = df.with_columns((gk_var.rolling_mean(window, min_periods=window).sqrt() * np.sqrt(252)).alias(f"px_gk_vol_{window}d"))
            df = df.with_columns(pl.col(f"px_gk_vol_{window}d").is_not_null().alias(f"is_gk_vol{window}_valid"))
        elif method == "rogers_satchell":
            # Rogers-Satchell: handles drift
            rs_var = ((pl.col("High") / pl.col("Close")).log() * 
                     (pl.col("High") / pl.col("Open")).log() +
                     (pl.col("Low") / pl.col("Close")).log() * 
                     (pl.col("Low") / pl.col("Open")).log())
            df = df.with_columns((rs_var.rolling_mean(window, min_periods=window).sqrt() * np.sqrt(252)).alias(f"px_rs_vol_{window}d"))
            df = df.with_columns(pl.col(f"px_rs_vol_{window}d").is_not_null().alias(f"is_rs_vol{window}_valid"))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df
    
    @staticmethod
    def add_moving_averages(df: pl.DataFrame, windows: List[int] = [5, 20, 60, 200]) -> pl.DataFrame:
        """移動平均（SMA/EMA）"""
        for window in windows:
            df = df.with_columns([
                pl.col("Close").rolling_mean(window, min_periods=window).alias(f"px_sma_{window}"),
                pl.col("Close").ewm_mean(span=window, adjust=False).alias(f"px_ema_{window}")
            ])
            df = df.with_columns(
                pl.col(f"px_sma_{window}").is_not_null().alias(f"is_sma{window}_valid")
            )
        return df
    
    @staticmethod
    def add_price_ratios(df: pl.DataFrame, windows: List[int] = [5, 20, 60]) -> pl.DataFrame:
        """価格比率指標（仕様準拠 + 既存互換）"""
        exprs = []
        for window in windows:
            exprs.extend([
                (pl.col("Close") / (pl.col(f"px_sma_{window}") + 1e-12)).alias(f"px_price_to_sma_{window}"),
                (pl.col("Close") / (pl.col(f"px_ema_{window}") + 1e-12)).alias(f"px_price_to_ema_{window}"),
                ((pl.col(f"px_sma_5") - pl.col(f"px_sma_{window}")) / (pl.col(f"px_sma_{window}") + 1e-12)).alias(f"px_ma_gap_5_{window}"),
                # 仕様: 乖離率 (Close - ema_k) / ema_k
                ((pl.col("Close") - pl.col(f"px_ema_{window}")) / (pl.col(f"px_ema_{window}") + 1e-12)).alias(f"px_price_ema{window}_dev"),
            ])
        
        # 仕様: EMAギャップとクロス（Binary）
        exprs.extend([
            ((pl.col("px_ema_5") - pl.col("px_ema_20")) / (pl.col("px_ema_20") + 1e-12)).alias("px_ema_gap_5_20"),
            ((pl.col("px_ema_20") - pl.col("px_ema_60")) / (pl.col("px_ema_60") + 1e-12)).alias("px_ema_gap_20_60"),
            ((pl.col("px_ema_60") - pl.col("px_ema_200")) / (pl.col("px_ema_200") + 1e-12)).alias("px_ema_gap_60_200"),
            (pl.col("px_ema_5") > pl.col("px_ema_20")).cast(pl.Int8).alias("px_ema_cross_5_20"),
        ])
        
        # High/Low range and positions（仕様の命名列も合わせて出力）
        exprs.extend([
            (pl.col("High") / (pl.col("Low") + 1e-12)).alias("px_high_low_ratio"),
            ((pl.col("Close") - pl.col("Low")) / (pl.col("High") - pl.col("Low") + 1e-12)).alias("px_close_position"),
            (pl.col("Close") / (pl.col("High") + 1e-12)).alias("px_close_to_high"),
            (pl.col("Close") / (pl.col("Low") + 1e-12)).alias("px_close_to_low"),
            # 仕様そのままの名前も追加（重複列名は避けるため存在チェックは別レイヤで行う想定）
            ((pl.col("High") - pl.col("Close")) / (pl.col("High") - pl.col("Low") + 1e-12)).alias("close_to_high"),
            ((pl.col("Close") - pl.col("Low")) / (pl.col("High") - pl.col("Low") + 1e-12)).alias("close_to_low"),
        ])
        
        return df.with_columns(exprs)

    @staticmethod
    def add_trend_indicators(df: pl.DataFrame, slope_windows: List[int] = [10, 20, 60]) -> pl.DataFrame:
        """EMAスロープ（pct_change(n=3)）を追加"""
        exprs = []
        for k in slope_windows:
            exprs.append(pl.col(f"px_ema_{k}").pct_change(3).alias(f"px_ema{k}_slope_3"))
        return df.with_columns(exprs)
    
    @staticmethod
    def add_volume_indicators(df: pl.DataFrame, windows: List[int] = [5, 20]) -> pl.DataFrame:
        """出来高指標"""
        # Volume moving averages
        for window in windows:
            df = df.with_columns([
                pl.col("Volume").rolling_mean(window, min_periods=window).alias(f"px_volume_ma_{window}"),
                (pl.col("Volume") / (pl.col("Volume").rolling_mean(window, min_periods=window) + 1e-12)).alias(f"px_volume_ratio_{window}")
            ])
            df = df.with_columns(
                pl.col(f"px_volume_ma_{window}").is_not_null().alias(f"is_volume{window}_valid")
            )
        
        # Turnover and dollar volume
        exprs = []
        if "SharesOutstanding" in df.columns:
            exprs.append((pl.col("Volume") / (pl.col("SharesOutstanding") + 1e-12)).alias("px_turnover_rate"))
        
        exprs.append((pl.col("Close") * pl.col("Volume")).alias("px_dollar_volume"))
        
        return df.with_columns(exprs)
    
    @staticmethod
    def add_rsi(df: pl.DataFrame, periods: List[int] = [2, 14]) -> pl.DataFrame:
        """RSI（相対力指数）"""
        for period in periods:
            # Price changes
            delta = pl.col("Close").diff()
            gain = pl.when(delta > 0).then(delta).otherwise(0)
            loss = pl.when(delta < 0).then(-delta).otherwise(0)
            
            # Average gain/loss (EMA)
            avg_gain = gain.ewm_mean(span=period, adjust=False)
            avg_loss = loss.ewm_mean(span=period, adjust=False)
            
            # RSI
            rs = avg_gain / (avg_loss + 1e-12)
            rsi = 100 - (100 / (1 + rs))
            
            df = df.with_columns(rsi.alias(f"px_rsi_{period}"))
        
        # RSI delta
        if 14 in periods and "px_rsi_14" in df.columns:
            df = df.with_columns((pl.col("px_rsi_14").diff()).alias("px_rsi_delta"))
        
        return df
    
    @staticmethod
    def add_macd(df: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pl.DataFrame:
        """MACD指標"""
        ema_fast = pl.col("Close").ewm_mean(span=fast, adjust=False)
        ema_slow = pl.col("Close").ewm_mean(span=slow, adjust=False)
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm_mean(span=signal, adjust=False)
        macd_histogram = macd - macd_signal
        
        return df.with_columns([
            macd.alias("px_macd"),
            macd_signal.alias("px_macd_signal"),
            macd_histogram.alias("px_macd_histogram")
        ])
    
    @staticmethod
    def add_bollinger_bands(df: pl.DataFrame, window: int = 20, num_std: float = 2.0) -> pl.DataFrame:
        """ボリンジャーバンド"""
        middle = pl.col("Close").rolling_mean(window, min_periods=window)
        std = pl.col("Close").rolling_std(window, min_periods=window)
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        width = upper - lower
        
        # Position within bands (0=lower, 1=upper)
        position = (pl.col("Close") - lower) / (width + 1e-12)
        
        df = df.with_columns([
            upper.alias("px_bb_upper"),
            lower.alias("px_bb_lower"),
            middle.alias("px_bb_middle"),
            width.alias("px_bb_width"),
            position.alias("px_bb_position")
        ])
        return df.with_columns(
            pl.col("px_bb_middle").is_not_null().alias(f"is_bb{window}_valid")
        )
    
    @staticmethod
    def add_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Average True Range"""
        high_low = pl.col("High") - pl.col("Low")
        high_close = (pl.col("High") - pl.col("Close").shift(1)).abs()
        low_close = (pl.col("Low") - pl.col("Close").shift(1)).abs()
        
        true_range = pl.max_horizontal([high_low, high_close, low_close])
        atr = true_range.ewm_mean(span=period, adjust=False)
        
        # Normalized ATR
        natr = (atr / pl.col("Close")) * 100
        
        return df.with_columns([
            atr.alias(f"px_atr_{period}"),
            natr.alias(f"px_natr_{period}")
        ])
    
    @staticmethod
    def add_adx(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Average Directional Index"""
        high_diff = pl.col("High").diff()
        low_diff = -pl.col("Low").diff()
        
        # Directional movements
        dm_plus = pl.when((high_diff > low_diff) & (high_diff > 0)).then(high_diff).otherwise(0)
        dm_minus = pl.when((low_diff > high_diff) & (low_diff > 0)).then(low_diff).otherwise(0)
        
        # Smoothed TR and DMs
        atr = TechnicalIndicators.add_atr(df, period)[f"px_atr_{period}"]
        dm_plus_smooth = dm_plus.ewm_mean(span=period, adjust=False)
        dm_minus_smooth = dm_minus.ewm_mean(span=period, adjust=False)
        
        # Directional indicators
        di_plus = (dm_plus_smooth / (atr + 1e-12)) * 100
        di_minus = (dm_minus_smooth / (atr + 1e-12)) * 100
        
        # DX and ADX
        dx = ((di_plus - di_minus).abs() / (di_plus + di_minus + 1e-12)) * 100
        adx = dx.ewm_mean(span=period, adjust=False)
        
        return df.with_columns(adx.alias("px_adx"))
    
    @staticmethod
    def add_stochastic(df: pl.DataFrame, period: int = 14, smooth: int = 3) -> pl.DataFrame:
        """Stochastic Oscillator"""
        low_min = pl.col("Low").rolling_min(period, min_periods=period)
        high_max = pl.col("High").rolling_max(period, min_periods=period)
        
        # %K
        k = ((pl.col("Close") - low_min) / (high_max - low_min + 1e-12)) * 100
        
        # %D (smoothed %K)
        d = k.rolling_mean(smooth, min_periods=smooth)
        
        return df.with_columns([
            k.alias("px_stoch_k"),
            d.alias("px_stoch_d"),
            k.is_not_null().alias(f"is_stoch{period}_valid")
        ])
    
    @staticmethod
    def add_targets(df: pl.DataFrame, horizons: List[int] = [1, 5, 10, 20]) -> pl.DataFrame:
        """予測ターゲット生成"""
        exprs = []
        
        for horizon in horizons:
            # Forward returns
            fwd_return = (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"y_{horizon}d")
            exprs.append(fwd_return)
            
            # Binary targets (positive return)
            exprs.append(
                (fwd_return > 0).cast(pl.Int8).alias(f"y_{horizon}d_bin")
            )
        
        return df.with_columns(exprs)
    
    @staticmethod
    def add_all_indicators(
        df: pl.DataFrame,
        return_periods: List[int] = [1, 5, 10, 20, 60, 120],
        ma_windows: List[int] = [5, 20, 60, 200],
        vol_windows: List[int] = [5, 10, 20, 60],
        realized_vol_method: str = "parkinson",
        target_horizons: List[int] = [1, 5, 10, 20]
    ) -> pl.DataFrame:
        """全テクニカル指標を追加（仕様強化版）"""
        
        # Sequential processing to ensure dependencies
        df = TechnicalIndicators.add_returns(df, return_periods)
        df = TechnicalIndicators.add_volatility(df, vol_windows)
        df = TechnicalIndicators.add_realized_volatility(df, window=20, method=realized_vol_method)
        df = TechnicalIndicators.add_moving_averages(df, ma_windows)
        df = TechnicalIndicators.add_price_ratios(df, [w for w in ma_windows if w <= 60])
        df = TechnicalIndicators.add_trend_indicators(df, [10, 20, 60])
        df = TechnicalIndicators.add_volume_indicators(df, [5, 20])
        df = TechnicalIndicators.add_rsi(df, [2, 14])
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_atr(df)
        df = TechnicalIndicators.add_adx(df)
        df = TechnicalIndicators.add_stochastic(df)
        df = TechnicalIndicators.add_targets(df, target_horizons)
        
        return df


class CrossSectionalNormalizer:
    """断面正規化（日次、無状態変換）"""
    
    @staticmethod
    def normalize_daily(
        df: pl.DataFrame,
        feature_cols: List[str],
        method: str = "zscore",
        robust: bool = True,
        winsorize_pct: Optional[float] = 0.01
    ) -> pl.DataFrame:
        """
        日次断面での正規化（リーク防止）
        
        Args:
            df: 入力データフレーム
            feature_cols: 正規化対象列
            method: 正規化方法 ("zscore", "minmax", "rank")
            robust: Robust統計量（median/MAD）を使用
            winsorize_pct: Winsorize閾値（0.01 = 1%/99%）
        """
        
        # Group by date for cross-sectional normalization
        normalized_exprs = []
        
        for col in feature_cols:
            if method == "zscore":
                if robust:
                    # Robust Z-score using median and MAD
                    median = pl.col(col).median().over("meta_date")
                    mad = (pl.col(col) - median).abs().median().over("meta_date")
                    z = (pl.col(col) - median) / (mad * 1.4826 + 1e-12)  # 1.4826 converts MAD to std
                else:
                    # Standard Z-score
                    mean = pl.col(col).mean().over("meta_date")
                    std = pl.col(col).std().over("meta_date")
                    z = (pl.col(col) - mean) / (std + 1e-12)
                
                # Winsorize if specified
                if winsorize_pct:
                    lower = z.quantile(winsorize_pct).over("meta_date")
                    upper = z.quantile(1 - winsorize_pct).over("meta_date")
                    z = z.clip(lower, upper)
                
                # Apply final clip to [-10, 10] as per specification
                z = z.clip(-10, 10)
                
                normalized_exprs.append(z.alias(f"{col}_z"))
                
            elif method == "rank":
                # Rank transformation (0 to 1)
                rank = pl.col(col).rank(method="average").over("meta_date")
                count = pl.col(col).count().over("meta_date")
                normalized = (rank - 1) / (count - 1)
                normalized_exprs.append(normalized.alias(f"{col}_rank"))
                
            elif method == "minmax":
                # Min-Max scaling
                min_val = pl.col(col).min().over("meta_date")
                max_val = pl.col(col).max().over("meta_date")
                normalized = (pl.col(col) - min_val) / (max_val - min_val + 1e-12)
                normalized_exprs.append(normalized.alias(f"{col}_minmax"))
        
        return df.with_columns(normalized_exprs)