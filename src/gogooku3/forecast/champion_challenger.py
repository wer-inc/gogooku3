from __future__ import annotations

"""Champion/Challenger framework for model deployment and A/B testing.

This module provides infrastructure for:
- Maintaining champion (production) and challenger (experimental) models
- Running comparative evaluations
- Automated promotion/demotion based on performance metrics
- Safe rollback mechanisms

Key Components:
- ModelRegistry: Central model management
- PerformanceTracker: Metric collection and comparison
- PromotionEngine: Automated decision making
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd


class ForecastModel(Protocol):
    """Protocol for forecast models in champion/challenger system."""

    def predict(self, df_obs: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate forecasts for input observations."""
        ...

    def fit(self, df_obs: pd.DataFrame, **kwargs) -> Any:
        """Train the model on observations."""
        ...


@dataclass
class ModelMetrics:
    """Performance metrics for a model."""
    model_name: str
    timestamp: datetime
    mae: float
    rmse: float
    mape: float
    rank_ic: float = 0.0  # Rank Information Coefficient for financial models
    sharpe_ratio: float = 0.0  # For financial model evaluation
    backtest_results: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRegistration:
    """Model registration in the champion/challenger system."""
    model: ForecastModel
    name: str
    version: str
    registered_at: datetime
    performance_history: list[ModelMetrics] = field(default_factory=list)
    status: str = "challenger"  # "champion", "challenger", "retired"


class PerformanceTracker:
    """Tracks and compares model performance over time."""

    def __init__(self, evaluation_window_days: int = 30):
        self.evaluation_window_days = evaluation_window_days

    def evaluate_model(
        self,
        model: ForecastModel,
        model_name: str,
        test_data: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> ModelMetrics:
        """Evaluate model performance on test data.

        Args:
            model: The model to evaluate
            model_name: Model identifier
            test_data: Input data for predictions
            ground_truth: Actual values for comparison

        Returns:
            Performance metrics
        """
        # Generate predictions
        predictions = model.predict(test_data)

        # Calculate metrics (simplified implementation)
        mae = self._calculate_mae(predictions, ground_truth)
        rmse = self._calculate_rmse(predictions, ground_truth)
        mape = self._calculate_mape(predictions, ground_truth)

        # Financial-specific metrics would go here
        rank_ic = self._calculate_rank_ic(predictions, ground_truth)
        sharpe_ratio = self._calculate_sharpe(predictions, ground_truth)

        return ModelMetrics(
            model_name=model_name,
            timestamp=datetime.now(),
            mae=mae,
            rmse=rmse,
            mape=mape,
            rank_ic=rank_ic,
            sharpe_ratio=sharpe_ratio
        )

    def _calculate_mae(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
        """Calculate Mean Absolute Error."""
        # Mock implementation - replace with actual calculation
        return 0.15

    def _calculate_rmse(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
        """Calculate Root Mean Square Error."""
        # Mock implementation - replace with actual calculation
        return 0.25

    def _calculate_mape(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Mock implementation - replace with actual calculation
        return 0.18

    def _calculate_rank_ic(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
        """Calculate Rank Information Coefficient for financial models."""
        # Mock implementation - replace with actual calculation
        return 0.12

    def _calculate_sharpe(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
        """Calculate Sharpe ratio for financial model evaluation."""
        # Mock implementation - replace with actual calculation
        return 0.85


class PromotionEngine:
    """Handles automated champion/challenger promotion decisions."""

    def __init__(
        self,
        min_evaluation_period_days: int = 14,
        min_performance_improvement: float = 0.05,  # 5% minimum improvement
        confidence_threshold: float = 0.95
    ):
        self.min_evaluation_period_days = min_evaluation_period_days
        self.min_performance_improvement = min_performance_improvement
        self.confidence_threshold = confidence_threshold

    def should_promote(
        self,
        champion_metrics: list[ModelMetrics],
        challenger_metrics: list[ModelMetrics]
    ) -> bool:
        """Determine if challenger should be promoted to champion.

        Args:
            champion_metrics: Historical performance of current champion
            challenger_metrics: Historical performance of challenger

        Returns:
            True if challenger should be promoted
        """
        if not challenger_metrics:
            return False

        # Require minimum evaluation period
        latest_challenger = challenger_metrics[-1]
        evaluation_start = datetime.now() - timedelta(days=self.min_evaluation_period_days)
        if latest_challenger.timestamp < evaluation_start:
            return False

        # Compare recent performance
        recent_champion = [m for m in champion_metrics if m.timestamp >= evaluation_start]
        recent_challenger = [m for m in challenger_metrics if m.timestamp >= evaluation_start]

        if not recent_champion or not recent_challenger:
            return False

        # Use Sharpe ratio as primary financial metric for promotion
        champion_sharpe = np.mean([m.sharpe_ratio for m in recent_champion])
        challenger_sharpe = np.mean([m.sharpe_ratio for m in recent_challenger])

        improvement = (challenger_sharpe - champion_sharpe) / abs(champion_sharpe) if champion_sharpe != 0 else 0

        return improvement >= self.min_performance_improvement


class ChampionChallengerFramework:
    """Main interface for champion/challenger model management."""

    def __init__(self):
        self.models: dict[str, ModelRegistration] = {}
        self.performance_tracker = PerformanceTracker()
        self.promotion_engine = PromotionEngine()
        self._champion_model: str | None = None

    def register_model(
        self,
        model: ForecastModel,
        name: str,
        version: str,
        as_champion: bool = False
    ) -> None:
        """Register a new model in the framework.

        Args:
            model: The model instance
            name: Model identifier
            version: Model version
            as_champion: Whether to set as champion immediately
        """
        registration = ModelRegistration(
            model=model,
            name=name,
            version=version,
            registered_at=datetime.now(),
            status="champion" if as_champion else "challenger"
        )

        model_key = f"{name}_{version}"
        self.models[model_key] = registration

        if as_champion or self._champion_model is None:
            self._champion_model = model_key

    def get_champion(self) -> ForecastModel | None:
        """Get the current champion model."""
        if self._champion_model and self._champion_model in self.models:
            return self.models[self._champion_model].model
        return None

    def get_challenger(self, name: str) -> ForecastModel | None:
        """Get a specific challenger model."""
        for key, registration in self.models.items():
            if registration.name == name and registration.status == "challenger":
                return registration.model
        return None

    def evaluate_and_promote(
        self,
        test_data: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> dict[str, Any]:
        """Run evaluation and promotion logic.

        Args:
            test_data: Test dataset for evaluation
            ground_truth: Ground truth values

        Returns:
            Evaluation results and promotion decisions
        """
        results = {
            "evaluations": {},
            "promotions": [],
            "champion": self._champion_model
        }

        # Evaluate all models
        for key, registration in self.models.items():
            if registration.status in ["champion", "challenger"]:
                metrics = self.performance_tracker.evaluate_model(
                    registration.model,
                    registration.name,
                    test_data,
                    ground_truth
                )
                registration.performance_history.append(metrics)
                results["evaluations"][key] = metrics

        # Check for promotions
        if self._champion_model:
            champion_reg = self.models[self._champion_model]

            for key, registration in self.models.items():
                if registration.status == "challenger":
                    should_promote = self.promotion_engine.should_promote(
                        champion_reg.performance_history,
                        registration.performance_history
                    )

                    if should_promote:
                        # Promote challenger to champion
                        old_champion = self._champion_model
                        self.models[old_champion].status = "retired"
                        registration.status = "champion"
                        self._champion_model = key

                        results["promotions"].append({
                            "new_champion": key,
                            "old_champion": old_champion,
                            "timestamp": datetime.now()
                        })

        results["champion"] = self._champion_model
        return results

    def get_model_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all registered models."""
        status = {}
        for key, registration in self.models.items():
            latest_metrics = registration.performance_history[-1] if registration.performance_history else None
            status[key] = {
                "name": registration.name,
                "version": registration.version,
                "status": registration.status,
                "registered_at": registration.registered_at,
                "latest_metrics": latest_metrics,
                "evaluation_count": len(registration.performance_history)
            }
        return status
