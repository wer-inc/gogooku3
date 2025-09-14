"""
Regime Feature Engineering for MoE Gate Enhancement

レジーム特徴量生成:
- J-UVX (Japan Uncertainty & Volatility Index): VIX風の不確実性・ボラティリティ指標
- KAMA (Kaufman's Adaptive Moving Average): トレンド効率性
- VIDYA (Variable Index Dynamic Average): 価格効率性
- Market Regime Classification: トレンド・レンジ・高/低ボラティリティ分類

These features enhance MoE gate decisions by providing market context.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class JUVXCalculator:
    """J-UVX (Japan Uncertainty & Volatility Index) Calculator

    VIXの日本版として、以下を統合:
    1. Realized Volatility (短期・中期・長期)
    2. Price Efficiency (効率的フロンティアからの距離)
    3. Cross-sectional Dispersion (銘柄間散らばり)
    4. Momentum Uncertainty (モメンタムの不安定性)
    """

    def __init__(self,
                 short_window: int = 5,
                 medium_window: int = 21,
                 long_window: int = 63,
                 percentile_window: int = 252):
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.percentile_window = percentile_window

    def calculate_realized_volatility_structure(self, returns: pl.DataFrame) -> pl.DataFrame:
        """多期間のRealized Volatility構造を計算"""

        # 各銘柄の各期間でのボラティリティ
        df = returns.with_columns([
            # Short-term RV
            pl.col("feat_ret_1d").rolling_std(self.short_window).alias("rv_short"),
            # Medium-term RV
            pl.col("feat_ret_1d").rolling_std(self.medium_window).alias("rv_medium"),
            # Long-term RV
            pl.col("feat_ret_1d").rolling_std(self.long_window).alias("rv_long"),
        ])

        # Term Structure (ボラティリティのターム構造)
        df = df.with_columns([
            # Short vs Medium slope
            ((pl.col("rv_medium") - pl.col("rv_short")) / pl.col("rv_short")).alias("rv_slope_sm"),
            # Medium vs Long slope
            ((pl.col("rv_long") - pl.col("rv_medium")) / pl.col("rv_medium")).alias("rv_slope_ml"),
            # Overall slope
            ((pl.col("rv_long") - pl.col("rv_short")) / pl.col("rv_short")).alias("rv_slope_total"),
        ])

        return df

    def calculate_cross_sectional_dispersion(self, returns: pl.DataFrame) -> pl.DataFrame:
        """Cross-sectional Dispersion (市場全体の銘柄間散らばり)"""

        # 各日付での銘柄間リターン分散
        cross_dispersion = (returns
            .group_by("date")
            .agg([
                pl.col("feat_ret_1d").std().alias("cross_dispersion_std"),
                pl.col("feat_ret_1d").var().alias("cross_dispersion_var"),
                # 90-10 percentile range
                (pl.col("feat_ret_1d").quantile(0.9) - pl.col("feat_ret_1d").quantile(0.1)).alias("cross_dispersion_range"),
                # Interquartile range
                (pl.col("feat_ret_1d").quantile(0.75) - pl.col("feat_ret_1d").quantile(0.25)).alias("cross_dispersion_iqr"),
            ])
        )

        # 元データと結合
        result = returns.join(cross_dispersion, on="date", how="left")

        # 分散の時系列volatility (uncertainty of uncertainty)
        result = result.with_columns([
            pl.col("cross_dispersion_std").rolling_std(self.medium_window).alias("dispersion_volatility"),
            pl.col("cross_dispersion_range").rolling_std(self.medium_window).alias("range_volatility"),
        ])

        return result

    def calculate_momentum_uncertainty(self, returns: pl.DataFrame) -> pl.DataFrame:
        """Momentum Uncertainty (モメンタムの不安定性)"""

        # 複数期間のモメンタム
        df = returns.with_columns([
            pl.col("feat_ret_1d").rolling_sum(5).alias("momentum_5d"),
            pl.col("feat_ret_1d").rolling_sum(10).alias("momentum_10d"),
            pl.col("feat_ret_1d").rolling_sum(21).alias("momentum_21d"),
        ])

        # モメンタムの方向一致性
        df = df.with_columns([
            # Sign consistency across periods
            ((pl.col("momentum_5d") > 0).cast(pl.Int32) +
             (pl.col("momentum_10d") > 0).cast(pl.Int32) +
             (pl.col("momentum_21d") > 0).cast(pl.Int32)).alias("momentum_consensus"),

            # Magnitude consistency (correlation proxy)
            (pl.col("momentum_5d") * pl.col("momentum_10d") > 0).cast(pl.Float32).alias("momentum_5_10_agree"),
            (pl.col("momentum_10d") * pl.col("momentum_21d") > 0).cast(pl.Float32).alias("momentum_10_21_agree"),
        ])

        # Uncertainty = 1 - consistency
        df = df.with_columns([
            (3 - pl.col("momentum_consensus")).alias("momentum_uncertainty_discrete"),
            (1.0 - pl.col("momentum_5_10_agree")).alias("momentum_uncertainty_5_10"),
            (1.0 - pl.col("momentum_10_21_agree")).alias("momentum_uncertainty_10_21"),
        ])

        return df

    def calculate_price_efficiency(self, prices: pl.DataFrame) -> pl.DataFrame:
        """Price Efficiency (価格効率性の測定)"""

        # Random walkからの逸脱度 (Variance Ratio)
        df = prices.with_columns([
            pl.col("close").pct_change().alias("returns_1d"),
        ])

        # Multi-period returns for variance ratio
        df = df.with_columns([
            pl.col("returns_1d").rolling_sum(2).alias("returns_2d"),
            pl.col("returns_1d").rolling_sum(5).alias("returns_5d"),
            pl.col("returns_1d").rolling_sum(10).alias("returns_10d"),
        ])

        # Variance ratios (efficiency indicators)
        df = df.with_columns([
            # VR(2) = Var(R_2d) / (2 * Var(R_1d))
            (pl.col("returns_2d").rolling_var(21) / (2 * pl.col("returns_1d").rolling_var(21))).alias("variance_ratio_2"),
            (pl.col("returns_5d").rolling_var(21) / (5 * pl.col("returns_1d").rolling_var(21))).alias("variance_ratio_5"),
            (pl.col("returns_10d").rolling_var(21) / (10 * pl.col("returns_1d").rolling_var(21))).alias("variance_ratio_10"),
        ])

        # Efficiency = |VR - 1| (perfect random walk has VR=1)
        df = df.with_columns([
            (pl.col("variance_ratio_2") - 1.0).abs().alias("inefficiency_2d"),
            (pl.col("variance_ratio_5") - 1.0).abs().alias("inefficiency_5d"),
            (pl.col("variance_ratio_10") - 1.0).abs().alias("inefficiency_10d"),
        ])

        return df

    def calculate_juvx(self, market_data: pl.DataFrame) -> pl.DataFrame:
        """Calculate complete J-UVX index"""

        # 1. Realized Volatility Structure
        df = self.calculate_realized_volatility_structure(market_data)

        # 2. Cross-sectional Dispersion
        df = self.calculate_cross_sectional_dispersion(df)

        # 3. Momentum Uncertainty
        df = self.calculate_momentum_uncertainty(df)

        # 4. Price Efficiency (need price data)
        if "close" in df.columns:
            df = self.calculate_price_efficiency(df)
            has_price_efficiency = True
        else:
            has_price_efficiency = False

        # 5. Combine into J-UVX components
        juvx_components = []
        weights = []

        # Volatility component (40% weight)
        juvx_components.extend(["rv_short", "rv_medium", "rv_long", "rv_slope_total"])
        weights.extend([0.1, 0.15, 0.1, 0.05])  # 40% total

        # Cross-sectional component (30% weight)
        juvx_components.extend(["cross_dispersion_std", "dispersion_volatility"])
        weights.extend([0.2, 0.1])  # 30% total

        # Momentum uncertainty component (20% weight)
        juvx_components.extend(["momentum_uncertainty_discrete", "momentum_uncertainty_5_10"])
        weights.extend([0.1, 0.1])  # 20% total

        if has_price_efficiency:
            # Price efficiency component (10% weight)
            juvx_components.extend(["inefficiency_2d", "inefficiency_5d"])
            weights.extend([0.05, 0.05])  # 10% total

        # Normalize each component to [0,1] using rolling percentile
        normalized_components = []
        for component in juvx_components:
            if component in df.columns:
                normalized = (df.select([
                    pl.col(component).rank(method="average").over(pl.col("code")) /
                    pl.col(component).count().over(pl.col("code"))
                ]).to_series())
                normalized_components.append(normalized.alias(f"{component}_norm"))

        # Weight and combine
        if normalized_components:
            # Create J-UVX as weighted average
            juvx_expr = sum(
                pl.col(f"{comp}_norm") * w
                for comp, w in zip(juvx_components, weights)
                if f"{comp}_norm" in [c.meta.output_name() for c in normalized_components]
            )

            df = df.with_columns([
                juvx_expr.alias("juvx_raw")
            ])

            # Scale to VIX-like range (10-50)
            df = df.with_columns([
                (10.0 + 40.0 * pl.col("juvx_raw")).alias("juvx")
            ])

        return df


class AdaptiveMovingAverageCalculator:
    """KAMA & VIDYA Calculator for trend/momentum efficiency"""

    def __init__(self,
                 kama_period: int = 21,
                 kama_fast_sc: float = 2.0,
                 kama_slow_sc: float = 30.0,
                 vidya_period: int = 21):
        self.kama_period = kama_period
        self.kama_fast_sc = kama_fast_sc
        self.kama_slow_sc = kama_slow_sc
        self.vidya_period = vidya_period

    def calculate_kama(self, prices: pl.Series) -> pl.Series:
        """Kaufman's Adaptive Moving Average"""

        # Convert to numpy for calculation
        price_array = prices.to_numpy()
        n = len(price_array)

        # Initialize arrays
        change = np.abs(np.diff(price_array))
        volatility = np.zeros(n)
        efficiency_ratio = np.zeros(n)
        kama = np.zeros(n)

        # Smoothing constants
        fastest = 2.0 / (self.kama_fast_sc + 1)
        slowest = 2.0 / (self.kama_slow_sc + 1)

        # Calculate efficiency ratio and KAMA
        for i in range(self.kama_period, n):
            # Direction (net change over period)
            direction = abs(price_array[i] - price_array[i - self.kama_period])

            # Volatility (sum of absolute changes)
            volatility[i] = np.sum(change[i - self.kama_period:i])

            # Efficiency ratio
            if volatility[i] > 0:
                efficiency_ratio[i] = direction / volatility[i]
            else:
                efficiency_ratio[i] = 0

            # Smoothing constant
            sc = (efficiency_ratio[i] * (fastest - slowest) + slowest) ** 2

            # KAMA
            if i == self.kama_period:
                kama[i] = price_array[i]
            else:
                kama[i] = kama[i-1] + sc * (price_array[i] - kama[i-1])

        return pl.Series(kama, name="kama")

    def calculate_vidya(self, prices: pl.Series, volumes: Optional[pl.Series] = None) -> pl.Series:
        """Variable Index Dynamic Average"""

        price_array = prices.to_numpy()
        n = len(price_array)

        # Use volume if available, otherwise use price volatility as proxy
        if volumes is not None:
            vol_array = volumes.to_numpy()
        else:
            # Use price change volatility as volume proxy
            returns = np.abs(np.diff(price_array, prepend=price_array[0]))
            vol_array = returns

        # Calculate Chande Momentum Oscillator (CMO) as volatility index
        cmo = np.zeros(n)
        vidya = np.zeros(n)

        for i in range(self.vidya_period, n):
            # Price changes over period
            changes = np.diff(price_array[i - self.vidya_period:i + 1])

            # Separate gains and losses
            gains = changes[changes > 0].sum()
            losses = -changes[changes < 0].sum()

            # CMO calculation
            if gains + losses > 0:
                cmo[i] = (gains - losses) / (gains + losses)
            else:
                cmo[i] = 0

            # Variable Index (VI) = abs(CMO)
            vi = abs(cmo[i])

            # VIDYA
            if i == self.vidya_period:
                vidya[i] = price_array[i]
            else:
                alpha = 2.0 / (self.vidya_period + 1) * vi  # Adaptive alpha
                vidya[i] = alpha * price_array[i] + (1 - alpha) * vidya[i-1]

        return pl.Series(vidya, name="vidya")


class MarketRegimeClassifier:
    """Market Regime Classification for MoE gating"""

    def __init__(self, volatility_window: int = 21, trend_window: int = 21):
        self.volatility_window = volatility_window
        self.trend_window = trend_window

    def classify_regime(self, market_data: pl.DataFrame) -> pl.DataFrame:
        """Classify market regime into 4 states:
        1. Low Vol + Trend
        2. High Vol + Trend
        3. Low Vol + Range
        4. High Vol + Range
        """

        df = market_data.with_columns([
            # Volatility regime
            pl.col("feat_ret_1d").rolling_std(self.volatility_window).alias("rolling_vol"),
            # Trend strength (using KAMA efficiency proxy)
            (pl.col("close").rolling_corr(pl.arange(self.trend_window), self.trend_window) ** 2).alias("trend_strength"),
        ])

        # Percentile-based thresholds
        df = df.with_columns([
            # High volatility = above 70th percentile
            (pl.col("rolling_vol").rank().over(pl.col("code")) / pl.col("rolling_vol").count().over(pl.col("code")) > 0.7).alias("high_vol_regime"),
            # Strong trend = above 60th percentile
            (pl.col("trend_strength").rank().over(pl.col("code")) / pl.col("trend_strength").count().over(pl.col("code")) > 0.6).alias("trend_regime"),
        ])

        # Regime encoding (0-3)
        df = df.with_columns([
            (pl.col("high_vol_regime").cast(pl.Int32) * 2 + pl.col("trend_regime").cast(pl.Int32)).alias("regime_class"),
        ])

        # One-hot encoding for neural network
        df = df.with_columns([
            (pl.col("regime_class") == 0).cast(pl.Float32).alias("regime_low_vol_range"),    # Low Vol + Range
            (pl.col("regime_class") == 1).cast(pl.Float32).alias("regime_low_vol_trend"),    # Low Vol + Trend
            (pl.col("regime_class") == 2).cast(pl.Float32).alias("regime_high_vol_range"),   # High Vol + Range
            (pl.col("regime_class") == 3).cast(pl.Float32).alias("regime_high_vol_trend"),   # High Vol + Trend
        ])

        return df


class RegimeFeatureExtractor(nn.Module):
    """Complete Regime Feature Extractor for MoE Gate Enhancement"""

    def __init__(self,
                 juvx_config: Optional[Dict] = None,
                 kama_config: Optional[Dict] = None,
                 regime_config: Optional[Dict] = None):
        super().__init__()

        # Feature calculators
        self.juvx_calc = JUVXCalculator(**(juvx_config or {}))
        self.ama_calc = AdaptiveMovingAverageCalculator(**(kama_config or {}))
        self.regime_clf = MarketRegimeClassifier(**(regime_config or {}))

        # Feature names for consistency
        self.juvx_features = ["juvx", "rv_short", "rv_medium", "rv_long", "cross_dispersion_std", "momentum_uncertainty_discrete"]
        self.ama_features = ["kama", "vidya"]
        self.regime_features = ["regime_low_vol_range", "regime_low_vol_trend", "regime_high_vol_range", "regime_high_vol_trend"]

        self.all_regime_features = self.juvx_features + self.ama_features + self.regime_features

    def extract_regime_features(self, market_data: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
        """Extract all regime features from market data"""

        # 1. Calculate J-UVX features
        df_with_juvx = self.juvx_calc.calculate_juvx(market_data)

        # 2. Calculate adaptive moving averages for each stock
        if "close" in df_with_juvx.columns:
            # Group by stock and calculate KAMA/VIDYA
            df_with_ama = df_with_juvx.group_by("code").map_groups(lambda group:
                group.with_columns([
                    self.ama_calc.calculate_kama(group.select("close").to_series()),
                    self.ama_calc.calculate_vidya(
                        group.select("close").to_series(),
                        group.select("volume").to_series() if "volume" in group.columns else None
                    ),
                ])
            ).sort(["date", "code"])
        else:
            df_with_ama = df_with_juvx

        # 3. Classify market regimes
        df_complete = self.regime_clf.classify_regime(df_with_ama)

        # 4. Select only the regime features we need
        available_features = [f for f in self.all_regime_features if f in df_complete.columns]

        return df_complete, available_features

    def forward(self, regime_features: torch.Tensor) -> torch.Tensor:
        """Neural network preprocessing of regime features (optional)"""
        # For now, pass through - can add normalization/transformation layers later
        return regime_features


def create_regime_features_from_data(market_data: pl.DataFrame) -> Tuple[torch.Tensor, List[str]]:
    """Convenience function to create regime features for MoE gating"""

    extractor = RegimeFeatureExtractor()
    df_with_features, feature_names = extractor.extract_regime_features(market_data)

    # Convert to tensor (assume last row for each stock, or use specific date)
    if len(feature_names) > 0:
        feature_matrix = df_with_features.select(feature_names).fill_null(0.0).to_numpy()
        regime_tensor = torch.from_numpy(feature_matrix).float()
    else:
        regime_tensor = torch.zeros(1, len(extractor.all_regime_features))

    return regime_tensor, feature_names