"""
Multi-horizon objective functions for HPO optimization
Optimizes 1d, 5d, 10d, 20d predictions jointly with weighted RankIC/Sharpe
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger(__name__)


class MultiHorizonObjective:
    """Multi-horizon objective function for ATFT-GAT-FAN optimization"""

    def __init__(self,
                 horizons: List[str] = None,
                 rank_ic_weight: float = 0.7,
                 sharpe_weight: float = 0.3,
                 horizon_weights: Dict[str, float] = None):
        """
        Initialize multi-horizon objective

        Args:
            horizons: Prediction horizons (default: ['1d', '5d', '10d', '20d'])
            rank_ic_weight: Weight for RankIC in combined score
            sharpe_weight: Weight for Sharpe in combined score
            horizon_weights: Per-horizon weights (default: balanced with 5d emphasis)
        """
        self.horizons = horizons or ['1d', '5d', '10d', '20d']
        self.rank_ic_weight = rank_ic_weight
        self.sharpe_weight = sharpe_weight

        # Default weights emphasize medium-term (5d, 10d) performance
        self.horizon_weights = horizon_weights or {
            '1d': 0.2,   # Short-term, more noise
            '5d': 0.35,  # Primary target
            '10d': 0.35, # Secondary target
            '20d': 0.1   # Long-term, harder to predict
        }

        # Validate weights sum to 1.0
        total_weight = sum(self.horizon_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Horizon weights sum to {total_weight:.3f}, normalizing to 1.0")
            self.horizon_weights = {k: v/total_weight for k, v in self.horizon_weights.items()}

    def compute_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute weighted multi-horizon score

        Args:
            metrics: Dictionary with 'rank_ic' and 'sharpe' subdictionaries

        Returns:
            Combined score (higher is better)
        """
        try:
            rank_ic_metrics = metrics.get('rank_ic', {})
            sharpe_metrics = metrics.get('sharpe', {})

            # Compute weighted RankIC score
            rank_ic_score = 0.0
            for horizon in self.horizons:
                horizon_ic = rank_ic_metrics.get(horizon, -0.1)  # Penalty for missing
                horizon_weight = self.horizon_weights.get(horizon, 0.0)
                rank_ic_score += horizon_weight * horizon_ic

            # Compute weighted Sharpe score (normalized)
            sharpe_score = 0.0
            for horizon in self.horizons:
                horizon_sharpe = sharpe_metrics.get(horizon, -1.0)  # Penalty for missing
                # Normalize Sharpe to [0,1] range (assume typical range [-2, 3])
                normalized_sharpe = max(0.0, min(1.0, (horizon_sharpe + 2.0) / 5.0))
                horizon_weight = self.horizon_weights.get(horizon, 0.0)
                sharpe_score += horizon_weight * normalized_sharpe

            # Combine RankIC and Sharpe
            combined_score = (
                self.rank_ic_weight * rank_ic_score +
                self.sharpe_weight * sharpe_score
            )

            # Add stability bonus for consistent performance
            stability_bonus = self._compute_stability_bonus(rank_ic_metrics, sharpe_metrics)
            combined_score += stability_bonus

            logger.debug(f"Multi-horizon score: RankIC={rank_ic_score:.4f}, "
                        f"Sharpe={sharpe_score:.4f}, Combined={combined_score:.4f}")

            return combined_score

        except Exception as e:
            logger.error(f"Error computing multi-horizon score: {e}")
            return -1.0  # Large penalty for errors

    def _compute_stability_bonus(self,
                               rank_ic_metrics: Dict[str, float],
                               sharpe_metrics: Dict[str, float]) -> float:
        """
        Compute stability bonus for consistent cross-horizon performance

        Args:
            rank_ic_metrics: RankIC by horizon
            sharpe_metrics: Sharpe by horizon

        Returns:
            Stability bonus (0 to 0.05)
        """
        try:
            # Get valid RankIC values
            ic_values = [rank_ic_metrics.get(h, 0.0) for h in self.horizons]
            ic_values = [v for v in ic_values if v > -0.05]  # Filter out poor performance

            if len(ic_values) < 2:
                return 0.0

            # Bonus for low standard deviation (consistent performance)
            ic_std = np.std(ic_values)
            consistency_bonus = max(0.0, 0.02 - ic_std)  # Up to 0.02 bonus

            # Bonus for all horizons being positive
            all_positive_ic = all(v > 0.0 for v in ic_values)
            positive_bonus = 0.01 if all_positive_ic else 0.0

            # Small bonus for improving with horizon (trend following)
            if len(ic_values) >= 3:
                # Check if medium-term (5d, 10d) outperforms short-term (1d)
                ic_1d = rank_ic_metrics.get('1d', 0.0)
                ic_5d = rank_ic_metrics.get('5d', 0.0)
                ic_10d = rank_ic_metrics.get('10d', 0.0)

                if ic_5d > ic_1d and ic_10d > ic_1d:
                    trend_bonus = 0.005
                else:
                    trend_bonus = 0.0
            else:
                trend_bonus = 0.0

            total_bonus = consistency_bonus + positive_bonus + trend_bonus

            logger.debug(f"Stability bonus: consistency={consistency_bonus:.4f}, "
                        f"positive={positive_bonus:.4f}, trend={trend_bonus:.4f}")

            return min(0.05, total_bonus)  # Cap at 0.05

        except Exception as e:
            logger.warning(f"Error computing stability bonus: {e}")
            return 0.0

    def suggest_horizon_weights(self, trial) -> Dict[str, float]:
        """
        Suggest horizon weights for optimization

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of normalized horizon weights
        """
        # Suggest raw weights
        raw_weights = {}
        for horizon in self.horizons:
            raw_weights[horizon] = trial.suggest_float(
                f"weight_{horizon}",
                0.05,
                2.0
            )

        # Normalize weights to sum to 1.0
        total = sum(raw_weights.values())
        normalized_weights = {k: v/total for k, v in raw_weights.items()}

        return normalized_weights

    def format_best_params(self, best_params: Dict[str, Any]) -> str:
        """Format best parameters for logging"""
        horizon_weights = {}
        other_params = {}

        for k, v in best_params.items():
            if k.startswith('weight_'):
                horizon = k.replace('weight_', '')
                horizon_weights[horizon] = v
            else:
                other_params[k] = v

        # Normalize horizon weights for display
        if horizon_weights:
            total = sum(horizon_weights.values())
            horizon_weights = {k: v/total for k, v in horizon_weights.items()}

        formatted = []
        formatted.append("🎯 Best Multi-Horizon Configuration:")
        formatted.append(f"   Horizon weights: {horizon_weights}")
        formatted.append(f"   Other parameters: {other_params}")

        return '\n'.join(formatted)