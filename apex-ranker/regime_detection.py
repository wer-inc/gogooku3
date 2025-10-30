"""
Market Regime Detection and Defensive Logic for APEX-Ranker Phase 4.3

Key Findings from Walk-Forward Analysis:
- Weak Folds: 2021-11 to 2023-09 (8 folds, Sharpe < 0)
- Average Return: -6.35%, Max DD: 17.02%
- Strong Folds: 2024-2025 (10 folds, Sharpe > 4.6)
- Key Differentiator: Volatility (weak: 7-12%, strong: <5%)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import polars as pl


class MarketRegime(Enum):
    """Market regime classification."""

    BULL = "bull"  # Low vol, positive momentum
    BEAR = "bear"  # High vol, negative returns
    SIDEWAYS = "sideways"  # Low vol, low momentum
    CRISIS = "crisis"  # Extreme vol, large drawdown


@dataclass
class RegimeSignals:
    """Regime detection signals."""

    realized_vol: float  # 20-day realized volatility
    momentum_20d: float  # 20-day price momentum
    max_dd_20d: float  # Maximum drawdown in 20 days
    avg_correlation: float  # Average stock correlation (market stress)
    regime: MarketRegime  # Detected regime
    confidence: float  # Detection confidence [0-1]


class RegimeDetector:
    """
    Market regime detector based on volatility, momentum, and correlation.

    Calibrated thresholds from walk-forward analysis:
    - Crisis: realized_vol > 25% annualized, max_dd > 10%
    - Bear: realized_vol > 18%, negative momentum
    - Bull: realized_vol < 15%, positive momentum
    - Sideways: Low vol + low momentum
    """

    def __init__(
        self,
        vol_window: int = 20,
        crisis_vol_threshold: float = 0.25,
        crisis_dd_threshold: float = 0.10,
        bear_vol_threshold: float = 0.18,
        bull_vol_threshold: float = 0.15,
    ):
        self.vol_window = vol_window
        self.crisis_vol_threshold = crisis_vol_threshold
        self.crisis_dd_threshold = crisis_dd_threshold
        self.bear_vol_threshold = bear_vol_threshold
        self.bull_vol_threshold = bull_vol_threshold

    def detect_regime(
        self,
        prices: pl.DataFrame,
        date: str,
        lookback_days: int = 20,
    ) -> RegimeSignals:
        """
        Detect market regime at a specific date.

        Args:
            prices: DataFrame with columns [Date, Code, Close]
            date: Target date for regime detection
            lookback_days: Number of days to analyze

        Returns:
            RegimeSignals with detected regime and confidence
        """
        # Filter data for lookback period
        end_date = pl.datetime(date)
        start_date = end_date - pl.duration(days=lookback_days)

        recent = prices.filter(
            (pl.col("Date") >= start_date) & (pl.col("Date") <= end_date)
        )

        if len(recent) == 0:
            return RegimeSignals(
                realized_vol=0.0,
                momentum_20d=0.0,
                max_dd_20d=0.0,
                avg_correlation=0.0,
                regime=MarketRegime.SIDEWAYS,
                confidence=0.0,
            )

        # Calculate realized volatility (annualized)
        returns = recent.group_by("Code").agg(
            pl.col("Close").pct_change().std().alias("vol")
        )
        avg_vol = returns["vol"].mean() * np.sqrt(252)

        # Calculate momentum (20-day return)
        momentum = recent.group_by("Code").agg(
            ((pl.col("Close").last() / pl.col("Close").first()) - 1).alias("ret_20d")
        )
        avg_momentum = momentum["ret_20d"].mean()

        # Calculate maximum drawdown
        cummax = recent.group_by("Code").agg(pl.col("Close").cum_max().alias("cummax"))
        drawdowns = recent.join(cummax, on="Code", how="left").with_columns(
            ((pl.col("Close") / pl.col("cummax")) - 1).alias("dd")
        )
        max_dd = abs(drawdowns["dd"].min())

        # Calculate average correlation (market stress indicator)
        pivot = recent.pivot(
            index="Date",
            columns="Code",
            values="Close",
        ).fill_null(strategy="forward")

        if pivot.shape[1] > 2:
            corr_matrix = pivot.select(pl.all().exclude("Date")).corr()
            # Average absolute correlation (excluding diagonal)
            mask = np.ones_like(corr_matrix, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_corr = np.abs(corr_matrix[mask]).mean()
        else:
            avg_corr = 0.5

        # Regime classification
        regime, confidence = self._classify_regime(
            avg_vol, avg_momentum, max_dd, avg_corr
        )

        return RegimeSignals(
            realized_vol=avg_vol,
            momentum_20d=avg_momentum,
            max_dd_20d=max_dd,
            avg_correlation=avg_corr,
            regime=regime,
            confidence=confidence,
        )

    def _classify_regime(
        self,
        vol: float,
        momentum: float,
        max_dd: float,
        correlation: float,
    ) -> tuple[MarketRegime, float]:
        """Classify regime based on signals with confidence score."""

        # Crisis: Extreme volatility + large drawdown
        if vol > self.crisis_vol_threshold and max_dd > self.crisis_dd_threshold:
            confidence = min(
                1.0,
                (vol / self.crisis_vol_threshold) * (max_dd / self.crisis_dd_threshold),
            )
            return MarketRegime.CRISIS, confidence

        # Bear: High volatility + negative momentum
        if vol > self.bear_vol_threshold and momentum < -0.05:
            confidence = min(1.0, (vol / self.bear_vol_threshold) * abs(momentum) * 10)
            return MarketRegime.BEAR, confidence

        # Bull: Low volatility + positive momentum
        if vol < self.bull_vol_threshold and momentum > 0.05:
            confidence = min(1.0, (1 - vol / self.bull_vol_threshold) * momentum * 10)
            return MarketRegime.BULL, confidence

        # Sideways: Everything else
        confidence = 1.0 - abs(momentum) * 5  # Low momentum = high sideways confidence
        return MarketRegime.SIDEWAYS, max(0.3, min(0.8, confidence))


class DefensiveRiskManager:
    """
    Defensive risk management for weak market regimes.

    Based on weak fold analysis:
    - Crisis folds (Sharpe < -3): Reduce exposure 70-80%
    - Bear folds (Sharpe < -1): Reduce exposure 40-50%
    - Volatility targeting: Scale positions inversely to realized vol
    - Drawdown control: Reduce on 10%+ drawdown
    """

    def __init__(
        self,
        target_vol: float = 0.15,
        max_drawdown: float = 0.10,
        crisis_exposure: float = 0.20,
        bear_exposure: float = 0.50,
        normal_exposure: float = 1.00,
    ):
        self.target_vol = target_vol
        self.max_drawdown = max_drawdown
        self.crisis_exposure = crisis_exposure
        self.bear_exposure = bear_exposure
        self.normal_exposure = normal_exposure

    def calculate_exposure(
        self,
        signals: RegimeSignals,
        current_dd: Optional[float] = None,
    ) -> float:
        """
        Calculate recommended portfolio exposure based on regime and drawdown.

        Args:
            signals: Regime detection signals
            current_dd: Current portfolio drawdown (optional)

        Returns:
            Exposure multiplier [0-1]
        """
        # Base exposure from regime
        if signals.regime == MarketRegime.CRISIS:
            base_exposure = self.crisis_exposure
        elif signals.regime == MarketRegime.BEAR:
            base_exposure = self.bear_exposure
        else:
            base_exposure = self.normal_exposure

        # Volatility targeting adjustment
        if signals.realized_vol > 0:
            vol_adj = min(1.0, self.target_vol / signals.realized_vol)
        else:
            vol_adj = 1.0

        # Drawdown control
        dd_adj = 1.0
        if current_dd is not None and current_dd < -self.max_drawdown:
            # Reduce exposure by 50% if exceeding max drawdown
            dd_adj = 0.5

        # Combined exposure (minimum of all constraints)
        final_exposure = base_exposure * min(vol_adj, dd_adj)

        # Apply confidence weighting (low confidence = more conservative)
        if signals.confidence < 0.5:
            final_exposure *= 0.7

        return max(0.1, min(1.0, final_exposure))  # Clamp to [0.1, 1.0]

    def adjust_portfolio_size(
        self,
        base_capital: float,
        signals: RegimeSignals,
        current_dd: Optional[float] = None,
    ) -> float:
        """
        Calculate adjusted capital allocation based on risk conditions.

        Args:
            base_capital: Base capital available
            signals: Regime signals
            current_dd: Current drawdown

        Returns:
            Adjusted capital to deploy
        """
        exposure = self.calculate_exposure(signals, current_dd)
        return base_capital * exposure


if __name__ == "__main__":
    # Example usage

    detector = RegimeDetector()
    risk_mgr = DefensiveRiskManager()

    print("=" * 80)
    print("APEX-Ranker Phase 4.3: Regime Detection & Defensive Risk Management")
    print("=" * 80)

    # Test scenarios based on weak fold analysis
    scenarios = [
        {
            "name": "2021-11 Crisis (Fold 01)",
            "vol": 0.30,
            "momentum": -0.14,
            "max_dd": 0.16,
            "corr": 0.85,
        },
        {
            "name": "2022-07 Bear (Fold 09)",
            "vol": 0.20,
            "momentum": -0.04,
            "max_dd": 0.07,
            "corr": 0.70,
        },
        {
            "name": "2024-11 Bull (Fold 36)",
            "vol": 0.12,
            "momentum": 0.37,
            "max_dd": 0.05,
            "corr": 0.45,
        },
    ]

    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print("-" * 80)

        regime, confidence = detector._classify_regime(
            scenario["vol"],
            scenario["momentum"],
            scenario["max_dd"],
            scenario["corr"],
        )

        signals = RegimeSignals(
            realized_vol=scenario["vol"],
            momentum_20d=scenario["momentum"],
            max_dd_20d=scenario["max_dd"],
            avg_correlation=scenario["corr"],
            regime=regime,
            confidence=confidence,
        )

        print(f"Detected Regime: {regime.value.upper()} (confidence: {confidence:.2%})")
        print(
            f"Realized Vol: {scenario['vol']:.1%} | Momentum: {scenario['momentum']:.1%}"
        )

        exposure = risk_mgr.calculate_exposure(signals)
        print(f"\n💰 Recommended Exposure: {exposure:.1%}")
        print(f"   Capital Allocation: ¥{100_000_000 * exposure:,.0f} / ¥100,000,000")

        if exposure < 0.5:
            print(
                f"   ⚠️  DEFENSIVE MODE: Reduced to {exposure:.0%} due to {regime.value} regime"
            )

    print("\n" + "=" * 80)
