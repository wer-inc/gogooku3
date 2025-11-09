"""Enhanced inference post-processing (A.1-A.4 improvements).

This module implements four inference-time improvements that boost Sharpe ratio
without requiring model retraining:

A.1: Rank Ensemble - Combine multiple model checkpoints via rank averaging
A.2: Uncertainty Filter - Exclude stocks with high rank variance across folds
A.3: Exit Hysteresis - Use different thresholds for entry vs exit to reduce turnover
A.4: Risk Neutralization - Regress out systematic risk factors (beta, size, sector)

References:
    User specification: Message 3 (2025-11-02)
    Motivation: Improve Sharpe from 0.478 → target ≥ 0.430 (90% gate)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.linear_model import Ridge


def rank_ensemble(
    fold_scores_dict: dict[str, dict[str, NDArray[np.float32]]],
    horizon_key: str = "5d",
) -> NDArray[np.float32]:
    """A.1: Rank Ensemble - Average normalized ranks across model checkpoints.

    Combines predictions from multiple CV folds and/or EMA snapshots by:
    1. Converting scores to ranks (descending, high score = low rank number)
    2. Normalizing ranks to zero mean, unit variance
    3. Averaging normalized ranks across all checkpoints

    This reduces variance and improves robustness compared to single-model scores.

    Args:
        fold_scores_dict: Dict mapping fold/checkpoint IDs to score dictionaries.
            Format: {"fold_0": {"5d": array([...]), "10d": array([...])}, ...}
        horizon_key: Which horizon to ensemble (default: "5d")

    Returns:
        Mean normalized rank for each stock (shape: n_codes)
        Lower values = higher consensus ranking

    Examples:
        >>> fold_scores = {
        ...     "fold_0": {"5d": np.array([0.5, 0.8, 0.3])},
        ...     "fold_1": {"5d": np.array([0.6, 0.7, 0.4])}
        ... }
        >>> mean_rank = rank_ensemble(fold_scores, "5d")
        >>> mean_rank.shape
        (3,)
        >>> # Stock 1 (index 1) should have lowest rank (highest consensus)
    """
    if not fold_scores_dict:
        raise ValueError("fold_scores_dict is empty")

    all_ranks = []
    for fold_id, fold_data in fold_scores_dict.items():
        if horizon_key not in fold_data:
            raise KeyError(f"Horizon '{horizon_key}' not found in fold '{fold_id}'")

        scores = fold_data[horizon_key]
        if scores.ndim != 1:
            raise ValueError(f"Expected 1D scores array, got shape {scores.shape}")

        # Rank descending (high score → low rank number → good)
        # rankdata returns 1-indexed ranks, ties handled by default (average)
        ranks = stats.rankdata(-scores, method="average")

        # Normalize to zero mean, unit variance
        ranks_norm = (ranks - ranks.mean()) / (ranks.std() + 1e-9)
        all_ranks.append(ranks_norm)

    # Average across all folds
    mean_rank = np.mean(all_ranks, axis=0).astype(np.float32)
    return mean_rank


def filter_uncertain(
    fold_scores_dict: dict[str, dict[str, NDArray[np.float32]]],
    horizon_key: str = "5d",
    top_pct: float = 0.2,
) -> NDArray[np.bool_]:
    """A.2: Uncertainty Filter - Exclude stocks with high rank variance.

    Stocks with high rank variance across folds are uncertain predictions.
    This filter excludes the top_pct most uncertain stocks to improve quality.

    Implementation:
    1. Compute rank standard deviation across folds for each stock
    2. Set threshold at (1 - top_pct) percentile of rank std
    3. Keep only stocks with rank_std <= threshold

    Args:
        fold_scores_dict: Dict mapping fold IDs to score dictionaries
        horizon_key: Which horizon to use (default: "5d")
        top_pct: Fraction of most uncertain stocks to exclude (default: 0.2 = 20%)

    Returns:
        Boolean mask (shape: n_codes) where True = keep, False = exclude

    Examples:
        >>> fold_scores = {
        ...     "fold_0": {"5d": np.array([0.5, 0.8, 0.3])},
        ...     "fold_1": {"5d": np.array([0.6, 0.2, 0.4])}
        ... }
        >>> mask = filter_uncertain(fold_scores, "5d", top_pct=0.3)
        >>> # Stock 1 (high variance: 0.8→0.2) should be excluded
        >>> mask[1]
        False
    """
    if not fold_scores_dict:
        raise ValueError("fold_scores_dict is empty")

    if not 0.0 <= top_pct <= 1.0:
        raise ValueError(f"top_pct must be in [0, 1], got {top_pct}")

    # Collect ranks across all folds
    all_ranks = []
    for fold_id, fold_data in fold_scores_dict.items():
        if horizon_key not in fold_data:
            raise KeyError(f"Horizon '{horizon_key}' not found in fold '{fold_id}'")

        scores = fold_data[horizon_key]
        ranks = stats.rankdata(-scores, method="average")
        all_ranks.append(ranks)

    # Compute rank standard deviation
    num_folds = len(all_ranks)
    if num_folds == 1:
        # With a single fold there is no dispersion; keep all stocks
        rank_std = np.zeros_like(all_ranks[0], dtype=np.float64)
    else:
        rank_std = np.std(all_ranks, axis=0, ddof=1)

    # Threshold: exclude top_pct most uncertain (high std)
    uncertain_threshold = np.percentile(rank_std, 100 * (1 - top_pct))

    # Keep stocks with rank_std <= threshold
    mask_keep = rank_std <= uncertain_threshold
    return mask_keep


def hysteresis_selection(
    scores: NDArray[np.float32],
    current_holdings: list[int] | None,
    entry_k: int = 35,
    exit_k: int = 60,
) -> list[int]:
    """A.3: Exit Hysteresis - Different thresholds for entry vs exit.

    Reduces turnover by using asymmetric selection thresholds:
    - Entry: Only add stocks ranked within top entry_k
    - Exit: Keep existing holdings if ranked within top exit_k

    This creates a "hysteresis band" (entry_k, exit_k) where holdings are
    maintained but new entries are not allowed.

    Args:
        scores: Model scores for all stocks (shape: n_codes)
        current_holdings: List of stock indices currently held (None for initial)
        entry_k: Threshold for adding new stocks (default: 35)
        exit_k: Threshold for keeping existing stocks (default: 60)

    Returns:
        List of stock indices to hold in the next period (length ≤ entry_k)

    Examples:
        >>> scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        >>> current = [1, 2]  # Currently hold stocks 1, 2
        >>> holdings = hysteresis_selection(scores, current, entry_k=2, exit_k=3)
        >>> # Stock 1 (rank 2) stays (within exit_k=3)
        >>> # Stock 2 (rank 3) stays (within exit_k=3)
        >>> # Stock 0 (rank 1) added (within entry_k=2)
        >>> sorted(holdings)
        [0, 1]  # Stock 2 dropped since len > entry_k
    """
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores array, got shape {scores.shape}")

    if entry_k <= 0 or exit_k <= 0:
        raise ValueError(f"entry_k and exit_k must be > 0, got {entry_k}, {exit_k}")

    if exit_k < entry_k:
        raise ValueError(f"exit_k must be >= entry_k, got exit_k={exit_k}, entry_k={entry_k}")

    n_codes = len(scores)

    # Rank stocks descending (high score → low rank index)
    rank_list = np.argsort(-scores).tolist()

    entry_set = set(rank_list[:entry_k])
    exit_set = set(rank_list[:exit_k])

    # If no current holdings, just return top entry_k
    if current_holdings is None or len(current_holdings) == 0:
        return rank_list[:entry_k]

    # Keep existing holdings that are still within exit_k threshold
    candidate_holdings = [code_idx for code_idx in current_holdings if code_idx in exit_set]
    candidate_set = set(candidate_holdings)

    # Consider new entries up to entry_k threshold
    for code_idx in rank_list:
        if code_idx in entry_set and code_idx not in candidate_set:
            candidate_holdings.append(code_idx)
            candidate_set.add(code_idx)

    # Always cap to entry_k by prioritizing higher scores
    candidate_holdings.sort(key=lambda idx: scores[idx], reverse=True)
    return candidate_holdings[:entry_k]


def risk_neutralize(
    scores: NDArray[np.float32],
    df_features: pd.DataFrame,
    factors: list[str] | None = None,
    alpha: float = 10.0,
    gamma: float = 0.3,
) -> NDArray[np.float32]:
    """A.4: Risk Neutralization - Partial removal of factor exposures (SAFE VERSION).

    Uses partial neutralization with re-scaling and safety guards to avoid
    over-correction that can degrade performance. Removes only γ (gamma) fraction
    of factor exposures instead of 100%, preserving score distribution.

    Implementation (6-step safe algorithm):
    1. Z-score normalize X (factors) and y (scores)
    2. Ridge regression: β = argmin ||y - Xβ||² + α||β||²
    3. Partial neutralization: y_resid = y - γ·(Xβ)
    4. Re-center: y_resid ← y_resid - mean(y_resid)
    5. Re-scale: std(y_resid) ← std(y)
    6. Safety guards:
       - Clip: ||y - y_resid||_∞ ≤ 0.25·std(y)
       - Skip if R² < 0.05 or |t(β)| < 2

    Args:
        scores: Model scores (shape: n_codes)
        df_features: DataFrame with risk factors (rows = stocks)
        factors: List of factor column names (default: ["Sector33Code", "volatility_60d"])
        alpha: Ridge regression regularization (default: 10.0, higher = more conservative)
        gamma: Partial neutralization factor (default: 0.3, range: [0.2, 0.5])
                0.0 = no neutralization, 1.0 = full neutralization

    Returns:
        Residual scores after partial factor exposure removal (shape: n_codes)

    Examples:
        >>> scores = np.array([0.5, 0.8, 0.3])
        >>> df = pd.DataFrame({
        ...     "Sector33Code": ["17", "50", "17"],
        ...     "volatility_60d": [0.25, 0.30, 0.20]
        ... })
        >>> # Partial neutralization (30% factor removal)
        >>> residuals = risk_neutralize(scores, df, gamma=0.3)
        >>> # Full neutralization (100% factor removal, not recommended)
        >>> residuals_full = risk_neutralize(scores, df, gamma=1.0)
    """
    # Default: Start with sector and volatility only (safer than beta/size)
    if factors is None:
        factors = ["sector33_code", "volatility_60d"]

    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores array, got shape {scores.shape}")

    if len(scores) != len(df_features):
        raise ValueError(
            f"Length mismatch: scores ({len(scores)}) vs df_features ({len(df_features)})"
        )

    # Extract risk factors
    try:
        X = df_features[factors].copy()
    except KeyError as e:
        missing = set(factors) - set(df_features.columns)
        raise KeyError(f"Missing factors in df_features: {missing}") from e

    # Handle categorical factors (one-hot encoding)
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Step 1: Z-score normalize X (factors)
    X_mean = X.mean()
    X_std = X.std()
    X_std = X_std.replace(0, 1)  # Avoid division by zero
    X_normalized = (X - X_mean) / X_std

    # Handle NaN (fill with 0 after normalization)
    X_normalized = X_normalized.fillna(0).values

    # Step 1 (cont): Z-score normalize y (scores)
    y = scores.copy()
    y_mean = np.mean(y)
    y_std = np.std(y)

    if y_std < 1e-6:
        # Scores have zero variance - skip neutralization
        return scores

    y_normalized = (y - y_mean) / y_std

    # Step 2: Ridge regression with regularization
    reg = Ridge(alpha=alpha, fit_intercept=False)  # Intercept absorbed in normalization
    reg.fit(X_normalized, y_normalized)

    # Step 6a: Safety guard - Check R² (model fit quality)
    y_pred = reg.predict(X_normalized)
    ss_res = np.sum((y_normalized - y_pred) ** 2)
    ss_tot = np.sum((y_normalized - np.mean(y_normalized)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    if r2 < 0.05:
        # Model explains <5% variance - skip neutralization
        return scores

    # Step 6b: Safety guard - Check coefficient significance (t-statistic)
    # Approximate: |β| / se(β), skip if all t-stats < 2
    n_samples = len(y_normalized)
    n_features = X_normalized.shape[1]

    if n_samples > n_features + 1:
        # Estimate standard errors
        mse = ss_res / (n_samples - n_features - 1)
        X_var = np.sum(X_normalized ** 2, axis=0)
        se_beta = np.sqrt(mse / (X_var + 1e-10))  # Avoid division by zero
        t_stats = np.abs(reg.coef_ / (se_beta + 1e-10))

        if np.max(t_stats) < 2.0:
            # No significant coefficients - skip neutralization
            return scores

    # Step 3: Partial neutralization (remove only γ fraction of factor exposure)
    correction = gamma * y_pred  # Partial correction (not full: y_pred)
    y_resid = y_normalized - correction

    # Step 4: Re-center (zero mean)
    y_resid = y_resid - np.mean(y_resid)

    # Step 5: Re-scale (restore original std)
    resid_std = np.std(y_resid)
    if resid_std > 1e-6:
        y_resid = y_resid * (y_std / resid_std)

    # Restore original mean
    y_resid = y_resid + y_mean

    # Step 6c: Safety guard - Clip max correction to 0.25σ
    max_correction = 0.25 * y_std
    correction_magnitude = np.abs(y_resid - scores)

    if np.max(correction_magnitude) > max_correction:
        # Clip corrections that are too large
        correction_direction = np.sign(y_resid - scores)
        y_resid = np.where(
            correction_magnitude > max_correction,
            scores + correction_direction * max_correction,
            y_resid
        )

    return y_resid.astype(np.float32)


__all__ = [
    "rank_ensemble",
    "filter_uncertain",
    "hysteresis_selection",
    "risk_neutralize",
]
