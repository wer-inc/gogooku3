#!/usr/bin/env python3
"""
Real-Time Regime Detection for Backtest Integration (Phase 4.3.2)

Calculates market regime from actual portfolio price data during backtest execution.
Provides precise volatility estimates and dynamic exposure control.

Key Improvements over Post-Hoc Estimation:
1. Access to daily returns → True realized volatility
2. Rolling window calculation → Captures recent market dynamics
3. Portfolio-level metrics → Better risk assessment
4. Real-time updates → Precise exposure adjustments

Usage:
    from realtime_regime import RealtimeRegimeCalculator

    calculator = RealtimeRegimeCalculator(lookback_days=20)
    signals = calculator.calculate_regime(
        portfolio_history=daily_results,
        current_date="2022-01-15"
    )
    exposure = risk_manager.calculate_exposure(signals, current_dd)
"""
from __future__ import annotations

from datetime import date as Date
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from regime_detection import MarketRegime, RegimeDetector, RegimeSignals


class RealtimeRegimeCalculator:
    """
    Calculate regime signals from actual backtest portfolio history.

    Designed for integration into backtest_smoke_test.py rebalance loop.
    """

    def __init__(
        self,
        lookback_days: int = 20,
        min_observations: int = 10,
        detector: RegimeDetector | None = None,
    ):
        """
        Initialize regime calculator.

        Args:
            lookback_days: Number of days for rolling calculations
            min_observations: Minimum observations required for regime detection
            detector: Optional custom RegimeDetector (uses default if None)
        """
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        self.detector = detector or RegimeDetector()

    def calculate_regime(
        self,
        portfolio_history: list[dict],
        current_date: Date | str,
    ) -> RegimeSignals:
        """
        Calculate regime signals from portfolio history.

        Args:
            portfolio_history: List of daily portfolio states from backtest
                Expected fields: date, portfolio_value, daily_return, drawdown
            current_date: Current rebalance date

        Returns:
            RegimeSignals with precise regime classification and confidence
        """
        current_dt = self._ensure_date(current_date)

        # Convert history to DataFrame for efficient filtering
        if not portfolio_history:
            return self._default_signals()

        df = pl.DataFrame(portfolio_history)

        # Ensure date column is proper type
        if "date" not in df.columns:
            return self._default_signals()

        df = df.with_columns(pl.col("date").cast(pl.Date).alias("date_parsed"))

        # Filter to lookback window
        cutoff_date = current_dt - timedelta(days=self.lookback_days)
        recent_data = df.filter(
            (pl.col("date_parsed") >= cutoff_date)
            & (pl.col("date_parsed") <= current_dt)
        ).sort("date_parsed")

        if len(recent_data) < self.min_observations:
            return self._default_signals()

        # Extract daily returns
        returns = recent_data["daily_return"].cast(pl.Float64).to_numpy()
        returns = np.asarray(returns, dtype=np.float64)

        # Debug: Inspect actual return values
        print(f"\n[DEBUG-REGIME] {current_dt}")
        print(f"  Returns sample (first 5): {returns[:5]}")
        print(f"  Returns sample (last 5): {returns[-5:]}")
        print(
            f"  Returns stats: mean={np.mean(returns):.6f}, std={np.std(returns):.6f}, min={np.min(returns):.6f}, max={np.max(returns):.6f}"
        )

        # Detect scale: if returns are expressed in percentages (e.g., 0.03% = 0.03) convert to decimal
        # Typical daily returns are -10% to +10% in decimal (0.1), or -10 to +10 in percentage format
        # Threshold of 1.0 catches percentage format (e.g., -3.0% stored as -3.0)
        max_abs_return = np.nanmax(np.abs(returns)) if returns.size > 0 else 0.0
        print(f"  max_abs_return: {max_abs_return:.6f}")
        print(f"  Scale detection triggered (>1.0): {max_abs_return > 1.0}")

        if max_abs_return > 1.0:
            returns_decimal = returns / 100.0
            print("  → Converting from percentage to decimal")
        else:
            returns_decimal = returns.copy()
            print("  → Using returns as-is (already decimal)")

        print(f"  Returns_decimal sample (first 5): {returns_decimal[:5]}")
        print(
            f"  Returns_decimal stats: mean={np.mean(returns_decimal):.6f}, std={np.std(returns_decimal):.6f}"
        )

        # Calculate realized volatility (annualized) using decimal returns
        raw_std = np.std(returns_decimal, ddof=0)
        realized_vol = float(raw_std * np.sqrt(252))
        print(
            f"  Volatility calc: std(daily)={raw_std:.6f}, annualized={realized_vol:.4f} ({realized_vol*100:.1f}%)"
        )

        # Calculate momentum (20-day cumulative return) using decimal returns
        momentum_20d = float(np.prod(1 + returns_decimal) - 1)
        print(f"  Momentum 20d: {momentum_20d:.4f} ({momentum_20d*100:.1f}%)")

        # Calculate max drawdown in window
        portfolio_values = recent_data["portfolio_value"].to_numpy()
        drawdowns = self._calculate_drawdown_series(portfolio_values)
        max_dd_20d = abs(np.min(drawdowns))

        # Calculate average correlation (use market proxy if available)
        # For now, estimate from consistency of returns
        avg_correlation = self._estimate_correlation(returns_decimal)

        # Classify regime using detector
        regime, confidence = self.detector._classify_regime(
            vol=realized_vol,
            momentum=momentum_20d,
            max_dd=max_dd_20d,
            correlation=avg_correlation,
        )

        return RegimeSignals(
            realized_vol=realized_vol,
            momentum_20d=momentum_20d,
            max_dd_20d=max_dd_20d,
            avg_correlation=avg_correlation,
            regime=regime,
            confidence=confidence,
        )

    def calculate_regime_from_prices(
        self,
        daily_frames: dict[Date, pl.DataFrame],
        current_date: Date,
        held_codes: set[str],
    ) -> RegimeSignals:
        """
        Calculate regime directly from price data (alternative method).

        Useful when portfolio history is not available but daily price frames are.

        Args:
            daily_frames: Dictionary mapping Date → DataFrame with columns [Code, Close]
            current_date: Current rebalance date
            held_codes: Set of stock codes currently held in portfolio

        Returns:
            RegimeSignals based on price movements of held stocks
        """
        cutoff_date = current_date - timedelta(days=self.lookback_days)

        # Collect prices for held stocks in lookback window
        trading_dates = sorted(
            [d for d in daily_frames.keys() if cutoff_date <= d <= current_date]
        )

        if len(trading_dates) < self.min_observations:
            return self._default_signals()

        # Build price matrix: [dates × stocks]
        price_matrix = []
        valid_codes = []

        for code in held_codes:
            code_prices = []
            valid = True

            for trade_date in trading_dates:
                day_frame = daily_frames.get(trade_date)
                if day_frame is None:
                    valid = False
                    break

                code_data = day_frame.filter(pl.col("Code") == code)
                if code_data.is_empty():
                    valid = False
                    break

                price = code_data[0, "Close"]
                if price is None or price <= 0:
                    valid = False
                    break

                code_prices.append(float(price))

            if valid and len(code_prices) == len(trading_dates):
                price_matrix.append(code_prices)
                valid_codes.append(code)

        if not price_matrix or len(price_matrix) < 3:
            return self._default_signals()

        # Convert to numpy for vectorized operations
        prices = np.array(price_matrix)  # shape: [stocks, dates]

        # Calculate returns for each stock
        returns = np.diff(prices, axis=1) / prices[:, :-1]  # shape: [stocks, dates-1]

        # Portfolio-level metrics (equal-weighted)
        portfolio_returns = np.mean(returns, axis=0)

        # Realized volatility (annualized)
        realized_vol = np.std(portfolio_returns) * np.sqrt(252)

        # Momentum (cumulative return)
        momentum_20d = np.prod(1 + portfolio_returns) - 1

        # Max drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        drawdowns = self._calculate_drawdown_series(cumulative_returns)
        max_dd_20d = abs(np.min(drawdowns))

        # Average correlation (pairwise correlation of stock returns)
        avg_correlation = self._calculate_correlation_matrix(returns)

        # Classify regime
        regime, confidence = self.detector._classify_regime(
            vol=realized_vol,
            momentum=momentum_20d,
            max_dd=max_dd_20d,
            correlation=avg_correlation,
        )

        return RegimeSignals(
            realized_vol=realized_vol,
            momentum_20d=momentum_20d,
            max_dd_20d=max_dd_20d,
            avg_correlation=avg_correlation,
            regime=regime,
            confidence=confidence,
        )

    @staticmethod
    def _ensure_date(value: Date | datetime | str) -> Date:
        """Convert various date types to datetime.date."""
        if isinstance(value, Date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d").date()
        raise TypeError(f"Unsupported date type: {type(value)}")

    @staticmethod
    def _calculate_drawdown_series(values: np.ndarray) -> np.ndarray:
        """Calculate drawdown series from portfolio values."""
        cummax = np.maximum.accumulate(values)
        drawdowns = (values - cummax) / cummax
        return drawdowns

    @staticmethod
    def _estimate_correlation(returns: np.ndarray) -> float:
        """
        Estimate average correlation from return consistency.

        High autocorrelation → Higher cross-stock correlation (trending market)
        Low autocorrelation → Lower correlation (dispersed market)
        """
        if len(returns) < 3:
            return 0.5

        # Calculate autocorrelation at lag 1
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]

        # Map autocorrelation to correlation estimate
        # Positive autocorr → Higher correlation (trending)
        # Negative autocorr → Lower correlation (mean-reverting)
        correlation_estimate = 0.5 + 0.3 * autocorr

        return float(np.clip(correlation_estimate, 0.0, 1.0))

    @staticmethod
    def _calculate_correlation_matrix(returns: np.ndarray) -> float:
        """
        Calculate average pairwise correlation from stock returns.

        Args:
            returns: Shape [stocks, dates-1]

        Returns:
            Average correlation (excluding diagonal)
        """
        if returns.shape[0] < 2:
            return 0.5

        try:
            corr_matrix = np.corrcoef(returns)

            # Extract upper triangle (excluding diagonal)
            n = corr_matrix.shape[0]
            upper_tri_indices = np.triu_indices(n, k=1)
            correlations = corr_matrix[upper_tri_indices]

            # Remove NaN values
            valid_corrs = correlations[~np.isnan(correlations)]

            if len(valid_corrs) == 0:
                return 0.5

            avg_corr = float(np.mean(valid_corrs))
            return np.clip(avg_corr, 0.0, 1.0)

        except (ValueError, FloatingPointError):
            return 0.5

    def _default_signals(self) -> RegimeSignals:
        """Return default signals when insufficient data available."""
        return RegimeSignals(
            realized_vol=0.20,  # Neutral volatility
            momentum_20d=0.0,  # No momentum
            max_dd_20d=0.05,  # Small drawdown
            avg_correlation=0.5,  # Neutral correlation
            regime=MarketRegime.SIDEWAYS,
            confidence=0.5,
        )


def integrate_regime_into_backtest(
    portfolio_history: list[dict],
    current_date: Date,
    current_dd: float,
    lookback_days: int = 20,
) -> tuple[RegimeSignals, float]:
    """
    Convenience function for backtest integration.

    Args:
        portfolio_history: List of daily portfolio states
        current_date: Current rebalance date
        current_dd: Current drawdown (negative value)
        lookback_days: Lookback window for regime calculation

    Returns:
        (RegimeSignals, recommended_exposure)

    Example:
        # In backtest_smoke_test.py rebalance loop:
        from realtime_regime import integrate_regime_into_backtest
        from regime_detection import DefensiveRiskManager

        risk_manager = DefensiveRiskManager()
        signals, exposure = integrate_regime_into_backtest(
            portfolio_history=daily_results,
            current_date=current_date,
            current_dd=current_drawdown,
        )

        # Adjust target weights by exposure
        adjusted_weights = {
            code: weight * exposure
            for code, weight in target_weights.items()
        }
    """
    from regime_detection import DefensiveRiskManager

    calculator = RealtimeRegimeCalculator(lookback_days=lookback_days)
    signals = calculator.calculate_regime(portfolio_history, current_date)

    risk_manager = DefensiveRiskManager()
    exposure = risk_manager.calculate_exposure(signals, current_dd)

    return signals, exposure
