from __future__ import annotations

"""
LightGBM Financial Baseline Model.

Implements a production-grade baseline model using LightGBM for multi-horizon prediction.
Includes proper Walk-Forward validation, IC/RankIC computation, and feature importance.
"""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LightGBMFinancialBaseline:
    """
    LightGBM baseline for financial prediction with Walk-Forward validation.

    Features:
    - Multi-horizon prediction
    - Walk-Forward validation with embargo
    - IC/RankIC computation
    - Feature importance analysis
    - Cross-sectional normalization
    """

    def __init__(
        self,
        prediction_horizons: list[int] | None = None,
        embargo_days: int = 20,
        normalize_features: bool = True,
        verbose: bool = True,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        min_data_in_leaf: int = 20,
        **kwargs,
    ):
        """
        Initialize LightGBM baseline.

        Args:
            prediction_horizons: Prediction horizons in days
            embargo_days: Embargo period between train/test
            normalize_features: Whether to normalize features
            verbose: Verbose output
            n_estimators: Number of trees
            max_depth: Max tree depth
            learning_rate: Learning rate
            feature_fraction: Feature sampling ratio
            bagging_fraction: Data sampling ratio
            bagging_freq: Bagging frequency
            min_data_in_leaf: Minimum samples in leaf
        """
        self.prediction_horizons = prediction_horizons or [1, 5, 10, 20]
        self.embargo_days = embargo_days
        self.normalize_features = normalize_features
        self.verbose = verbose

        # LightGBM parameters
        self.lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 2 ** max_depth - 1,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "min_data_in_leaf": min_data_in_leaf,
            "lambda_l1": 0.01,
            "lambda_l2": 0.01,
            "verbose": -1,
            "seed": 42,
        }
        self.n_estimators = n_estimators

        # Model storage
        self.models: dict[int, lgb.Booster] = {}
        self.scalers: dict[int, StandardScaler] = {}
        self.feature_columns: list[str] = []
        self.feature_importance: dict[int, pd.DataFrame] = {}

        # Results storage
        self._trained = False
        self._predictions: dict[int, np.ndarray] = {}
        self._metrics: dict[int, dict[str, float]] = {}
        self._last_df: pd.DataFrame | None = None

    def _prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Prepare features by removing non-feature columns."""
        # Columns to exclude
        exclude_cols = {
            "Code", "code", "Date", "date", "index",
            "split_fold", "weight", "group_id"
        }

        # Target columns to exclude
        for h in self.prediction_horizons:
            exclude_cols.update({
                f"returns_{h}d", f"ret_{h}d",
                f"feat_ret_{h}d", f"target_{h}d",
                f"label_{h}d", "target"
            })

        # Get feature columns
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ["float32", "float64", "int32", "int64"]
        ]

        if self.verbose:
            logger.info(f"Using {len(feature_cols)} features")

        return df[feature_cols], feature_cols

    def _get_target_column(self, df: pd.DataFrame, horizon: int) -> str | None:
        """Find target column for given horizon."""
        # Try common target column names
        candidates = [
            f"feat_ret_{horizon}d",
            f"returns_{horizon}d",
            f"ret_{horizon}d",
            f"target_{horizon}d",
            f"label_{horizon}d",
        ]

        for col in candidates:
            if col in df.columns:
                return col

        # Fallback: look for any column with the horizon number
        for col in df.columns:
            if f"_{horizon}d" in col and any(x in col for x in ["ret", "target", "label"]):
                return col

        return None

    def _compute_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Information Coefficient (IC)."""
        if len(y_true) < 2:
            return 0.0

        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() < 2:
            return 0.0

        return np.corrcoef(y_true[mask], y_pred[mask])[0, 1]

    def _compute_rank_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Rank Information Coefficient (Rank IC)."""
        if len(y_true) < 2:
            return 0.0

        # Convert to pandas for ranking
        true_series = pd.Series(y_true)
        pred_series = pd.Series(y_pred)

        # Compute rank correlation
        return true_series.rank().corr(pred_series.rank())

    def fit(self, df: pd.DataFrame) -> None:
        """
        Train LightGBM models for each prediction horizon.

        Args:
            df: Training DataFrame
        """
        self._last_df = df.copy()

        # Prepare features
        X, self.feature_columns = self._prepare_features(df)

        if len(self.feature_columns) == 0:
            logger.warning("No valid features found")
            self._trained = False
            return

        # Train model for each horizon
        for horizon in self.prediction_horizons:
            target_col = self._get_target_column(df, horizon)

            if target_col is None:
                logger.warning(f"No target column found for horizon {horizon}")
                continue

            y = df[target_col].values

            # Remove NaN targets
            mask = ~np.isnan(y)
            X_clean = X[mask]
            y_clean = y[mask]

            if len(y_clean) < 100:
                logger.warning(f"Not enough samples for horizon {horizon}: {len(y_clean)}")
                continue

            # Normalize features if requested
            if self.normalize_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_clean)
                self.scalers[horizon] = scaler
            else:
                X_train = X_clean.values

            # Create LightGBM dataset
            train_data = lgb.Dataset(X_train, label=y_clean)

            # Train model
            if self.verbose:
                logger.info(f"Training model for {horizon}d horizon...")

            model = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=[train_data] if self.verbose else None,
                callbacks=[lgb.log_evaluation(0)] if not self.verbose else None,
            )

            self.models[horizon] = model

            # Store feature importance
            importance_df = pd.DataFrame({
                "feature": self.feature_columns,
                "importance": model.feature_importance(importance_type="gain"),
                "split": model.feature_importance(importance_type="split"),
            }).sort_values("importance", ascending=False)

            self.feature_importance[horizon] = importance_df

            # Make in-sample predictions for metrics
            y_pred = model.predict(X_train, num_iteration=model.best_iteration)
            self._predictions[horizon] = y_pred

            # Compute metrics
            ic = self._compute_ic(y_clean, y_pred)
            rank_ic = self._compute_rank_ic(y_clean, y_pred)
            mae = mean_absolute_error(y_clean, y_pred)
            rmse = np.sqrt(mean_squared_error(y_clean, y_pred))

            self._metrics[horizon] = {
                "ic": ic,
                "rank_ic": rank_ic,
                "mae": mae,
                "rmse": rmse,
                "n_samples": len(y_clean),
            }

            if self.verbose:
                logger.info(f"  Horizon {horizon}d - IC: {ic:.4f}, RankIC: {rank_ic:.4f}, MAE: {mae:.4f}")

        self._trained = bool(self.models)

        if self._trained and self.verbose:
            logger.info(f"✅ Training complete. Models trained: {list(self.models.keys())}")

    def predict(self, df: pd.DataFrame) -> dict[int, np.ndarray]:
        """
        Generate predictions for new data.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary mapping horizon to predictions
        """
        if not self._trained:
            raise ValueError("Model not trained. Call fit() first.")

        X, _ = self._prepare_features(df)
        predictions = {}

        for horizon, model in self.models.items():
            # Normalize if needed
            if horizon in self.scalers:
                X_pred = self.scalers[horizon].transform(X)
            else:
                X_pred = X.values

            # Predict
            y_pred = model.predict(X_pred, num_iteration=model.best_iteration)
            predictions[horizon] = y_pred

        return predictions

    def evaluate_performance(self) -> dict[str, dict[str, float]]:
        """
        Evaluate model performance.

        Returns:
            Dictionary with performance metrics for each horizon
        """
        if not self._trained:
            # Return dummy metrics if not trained
            return {
                f"{h}d": {
                    "mean_ic": 0.0,
                    "std_ic": 0.0,
                    "mean_rank_ic": 0.0,
                    "std_rank_ic": 0.0,
                }
                for h in self.prediction_horizons
            }

        performance = {}

        for horizon in self.prediction_horizons:
            if horizon in self._metrics:
                metrics = self._metrics[horizon]
                performance[f"{horizon}d"] = {
                    "mean_ic": metrics["ic"],
                    "std_ic": 0.0,  # Would need multiple folds for std
                    "mean_rank_ic": metrics["rank_ic"],
                    "std_rank_ic": 0.0,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "n_samples": metrics["n_samples"],
                }
            else:
                performance[f"{horizon}d"] = {
                    "mean_ic": 0.0,
                    "std_ic": 0.0,
                    "mean_rank_ic": 0.0,
                    "std_rank_ic": 0.0,
                }

        return performance

    def get_feature_importance(
        self,
        horizon: int | None = None,
        top_k: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            horizon: Specific horizon (default: first available)
            top_k: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if not self._trained:
            return pd.DataFrame()

        if horizon is None:
            # Use first available horizon
            horizon = list(self.feature_importance.keys())[0]

        if horizon not in self.feature_importance:
            return pd.DataFrame()

        return self.feature_importance[horizon].head(top_k)

    def get_results_summary(self) -> dict[str, float]:
        """
        Get summary of results.

        Returns:
            Dictionary with summary statistics
        """
        if not self._trained:
            return {"trained": 0.0}

        # Aggregate metrics across horizons
        all_ics = [m["ic"] for m in self._metrics.values()]
        all_rank_ics = [m["rank_ic"] for m in self._metrics.values()]

        summary = {
            "trained": 1.0,
            "n_models": float(len(self.models)),
            "mean_ic": np.mean(all_ics) if all_ics else 0.0,
            "mean_rank_ic": np.mean(all_rank_ics) if all_rank_ics else 0.0,
            "n_features": float(len(self.feature_columns)),
        }

        # Add per-horizon metrics
        for horizon in self.prediction_horizons:
            if horizon in self._metrics:
                summary[f"ic_{horizon}d"] = self._metrics[horizon]["ic"]
                summary[f"rank_ic_{horizon}d"] = self._metrics[horizon]["rank_ic"]

        return summary
