"""LightGBM baseline model for financial prediction."""

import logging
import numpy as np
from typing import Dict, Any, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available, using mock implementation")


class LightGBMFinancialBaseline:
    """
    LightGBM baseline model for financial time series prediction.
    
    A simple gradient boosting baseline for comparison with neural network models.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        num_leaves: int = 31,
        feature_fraction: float = 0.9,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize LightGBM baseline model.
        
        Args:
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            num_leaves: Maximum number of leaves
            feature_fraction: Feature sampling ratio
            bagging_fraction: Data sampling ratio
            bagging_freq: Bagging frequency
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
        logger.info(f"Initialized LightGBM baseline with n_estimators={n_estimators}")
    
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False
    ) -> 'LightGBMFinancialBaseline':
        """
        Fit the LightGBM model.
        
        Args:
            X: Training features
            y: Training targets
            eval_set: Validation set for early stopping
            early_stopping_rounds: Early stopping rounds
            verbose: Verbose training
            
        Returns:
            Self for method chaining
        """
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, using mock fit")
            self.is_fitted = True
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            return self
        
        try:
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            
            self.feature_names = list(X.columns)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': self.num_leaves,
                'learning_rate': self.learning_rate,
                'feature_fraction': self.feature_fraction,
                'bagging_fraction': self.bagging_fraction,
                'bagging_freq': self.bagging_freq,
                'max_depth': self.max_depth,
                'random_state': self.random_state,
                'verbosity': 1 if verbose else -1,
                **self.kwargs
            }
            
            train_data = lgb.Dataset(X, label=y)
            valid_sets = [train_data]
            valid_names = ['train']
            
            if eval_set is not None:
                X_val, y_val = eval_set
                if isinstance(X_val, np.ndarray):
                    X_val = pd.DataFrame(X_val, columns=self.feature_names)
                valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                valid_sets.append(valid_data)
                valid_names.append('valid')
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds) if early_stopping_rounds else None,
                    lgb.log_evaluation(period=100 if verbose else 0)
                ]
            )
            
            self.is_fitted = True
            logger.info("LightGBM model training completed")
            
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            raise
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the fitted model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, returning mock predictions")
            return np.random.normal(0, 1, size=len(X))
        
        try:
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.feature_names)
            
            if self.model is not None:
                predictions = self.model.predict(X)
                return predictions
            else:
                return np.random.normal(0, 1, size=len(X))
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'split')
            
        Returns:
            Dictionary of feature importances
        """
        if not self.is_fitted or not LIGHTGBM_AVAILABLE:
            logger.warning("Model not fitted or LightGBM not available")
            return {}
        
        try:
            if self.model is not None:
                importance = self.model.feature_importance(importance_type=importance_type)
                if self.feature_names:
                    return dict(zip(self.feature_names, importance))
            return {}
            
        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            return {}
    
    def summary(self) -> Dict[str, Any]:
        """Get model summary."""
        return {
            "model_name": "LightGBM Financial Baseline",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "is_fitted": self.is_fitted,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "lightgbm_available": LIGHTGBM_AVAILABLE
        }


def create_lightgbm_baseline(**kwargs) -> LightGBMFinancialBaseline:
    """Factory function to create LightGBM baseline model."""
    return LightGBMFinancialBaseline(**kwargs)
