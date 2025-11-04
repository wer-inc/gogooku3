from __future__ import annotations

"""
Statistical utilities for evaluating model performance.

The helpers below favour self-contained numerical routines so they can be used
inside the training/backtest loops without external dependencies.
"""

from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


def _ensure_array(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected 1-D array-like input.")
    if array.size == 0:
        raise ValueError("Input array cannot be empty.")
    return array


def _newey_west_covariance(
    residuals: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """
    Compute Newey–West HAC covariance estimate for zero-mean residual series.

    Args:
        residuals: Array of shape (T, K) with zero-mean series in columns.
        max_lag: Maximum lag to include (Bartlett kernel).

    Returns:
        HAC covariance matrix of shape (K, K).
    """
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")
    t, k = residuals.shape
    gamma0 = residuals.T @ residuals / t
    cov = gamma0.copy()
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = (residuals[lag:].T @ residuals[:-lag]) / t
        cov += weight * (gamma + gamma.T)
    return cov / t


def moving_block_bootstrap(
    series: np.ndarray | list[float] | tuple[float, ...],
    *,
    block_size: int | None = None,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Moving block bootstrap samples preserving short-term autocorrelation.

    Args:
        series: 1-D array-like samples.
        block_size: Length of contiguous blocks (defaults to sqrt(T)).
        n_bootstrap: Number of bootstrap replicates.
        rng: Optional RNG instance.

    Returns:
        Array with shape (n_bootstrap, T) containing sampled series.
    """
    values = _ensure_array(series)
    n = values.size
    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))
    block_size = max(1, int(block_size))

    rng = rng or np.random.default_rng()

    blocks = np.lib.stride_tricks.sliding_window_view(values, block_size)
    n_blocks = blocks.shape[0]
    reps = np.empty((n_bootstrap, n), dtype=float)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_blocks, size=int(np.ceil(n / block_size)))
        sampled = blocks[idx].reshape(-1)[:n]
        reps[i] = sampled

    return reps


def block_bootstrap_ci(
    series: np.ndarray | list[float] | tuple[float, ...],
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    block_size: int | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Compute bootstrap confidence interval for a statistic using moving blocks.

    Args:
        series: 1-D array of observations.
        statistic: Function applied to each bootstrap replicate (defaults to mean).
        block_size: Optional block length (defaults to sqrt(T)).
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level (two-sided).

    Returns:
        Dict containing estimate, lower/upper bounds, and bootstrap std.
    """
    values = _ensure_array(series)
    statistic = statistic or np.mean

    reps = moving_block_bootstrap(
        values,
        block_size=block_size,
        n_bootstrap=n_bootstrap,
    )
    boot_stats = np.apply_along_axis(statistic, 1, reps)
    estimate = float(statistic(values))
    lower = float(np.quantile(boot_stats, alpha / 2))
    upper = float(np.quantile(boot_stats, 1 - alpha / 2))
    std = float(np.std(boot_stats, ddof=1))
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "std": std,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
    }


def diebold_mariano(
    series_a: np.ndarray | list[float] | tuple[float, ...],
    series_b: np.ndarray | list[float] | tuple[float, ...],
    *,
    horizon: int = 1,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """
    Diebold–Mariano test comparing predictive accuracy of two series.

    Args:
        series_a: Loss series from model A (e.g., -NDCG).
        series_b: Loss series from model B.
        horizon: Forecast horizon (controls Newey–West lag).
        alternative: ``two-sided``, ``greater`` (A worse than B), or ``less``.

    Returns:
        Dict with statistic and p-value.
    """
    loss_a = _ensure_array(series_a)
    loss_b = _ensure_array(series_b)
    if loss_a.shape != loss_b.shape:
        raise ValueError("Input series must share the same shape.")

    diff = loss_a - loss_b
    n = diff.size
    mean_diff = float(np.mean(diff))
    max_lag = max(0, int(horizon) - 1)
    var = _newey_west_covariance(diff.reshape(-1, 1), max_lag)[0, 0]
    if var <= 0:
        return {"statistic": 0.0, "pvalue": 1.0}
    dm_stat = mean_diff / np.sqrt(var)
    dm_stat *= np.sqrt(n)

    dist = NormalDist()
    if alternative == "two-sided":
        pvalue = 2 * (1 - dist.cdf(abs(dm_stat)))
    elif alternative == "greater":
        pvalue = 1 - dist.cdf(dm_stat)
    elif alternative == "less":
        pvalue = dist.cdf(dm_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return {"statistic": float(dm_stat), "pvalue": float(pvalue)}


def ledoit_wolf_sharpe_diff(
    returns_a: np.ndarray | list[float] | tuple[float, ...],
    returns_b: np.ndarray | list[float] | tuple[float, ...],
    *,
    periods_per_year: int = 252,
    max_lag: int | None = None,
) -> dict[str, float]:
    """
    Ledoit–Wolf style Sharpe difference test using HAC covariance estimates.

    Args:
        returns_a: Return series for strategy A (in decimal form).
        returns_b: Return series for strategy B (decimal form).
        periods_per_year: Annualisation factor for Sharpe ratios.
        max_lag: Optional HAC lag (defaults to cube-root of T).

    Returns:
        Dict with sharpe ratios, difference, standard error, and p-value.
    """
    x = _ensure_array(returns_a)
    y = _ensure_array(returns_b)
    if x.shape != y.shape:
        raise ValueError("Return series must have identical shape.")

    mu_a, mu_b = np.mean(x), np.mean(y)
    var_a, var_b = np.var(x, ddof=1), np.var(y, ddof=1)
    sigma_a, sigma_b = np.sqrt(var_a), np.sqrt(var_b)
    if sigma_a == 0 or sigma_b == 0:
        return {"statistic": 0.0, "pvalue": 1.0, "sharpe_diff": 0.0}

    sr_a = mu_a / sigma_a * np.sqrt(periods_per_year)
    sr_b = mu_b / sigma_b * np.sqrt(periods_per_year)
    sr_diff = sr_a - sr_b

    max_lag = int(max_lag) if max_lag is not None else int(np.cbrt(x.size))
    max_lag = max(0, max_lag)

    residuals = np.column_stack(
        [
            x - mu_a,
            (x - mu_a) ** 2 - var_a,
            y - mu_b,
            (y - mu_b) ** 2 - var_b,
        ]
    )
    cov_hat = _newey_west_covariance(residuals, max_lag)

    grad = np.array(
        [
            1.0 / sigma_a * np.sqrt(periods_per_year),
            -0.5 * mu_a / (sigma_a**3) * np.sqrt(periods_per_year),
            -1.0 / sigma_b * np.sqrt(periods_per_year),
            0.5 * mu_b / (sigma_b**3) * np.sqrt(periods_per_year),
        ]
    )

    var_diff = float(grad @ cov_hat @ grad.T)
    if var_diff <= 0:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "sharpe_a": float(sr_a),
            "sharpe_b": float(sr_b),
            "sharpe_diff": float(sr_diff),
            "stderr": 0.0,
        }

    stderr = np.sqrt(var_diff)
    stat = sr_diff / stderr
    dist = NormalDist()
    pvalue = 2 * (1 - dist.cdf(abs(stat)))
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "sharpe_a": float(sr_a),
        "sharpe_b": float(sr_b),
        "sharpe_diff": float(sr_diff),
        "stderr": float(stderr),
    }


@dataclass
class DeflatedSharpeResult:
    sharpe: float
    dsr: float
    sr_threshold: float
    sr_std: float
    n_obs: int
    n_trials: int


def deflated_sharpe_ratio(
    returns: np.ndarray | list[float] | tuple[float, ...],
    *,
    periods_per_year: int = 252,
    n_trials: int = 1,
) -> DeflatedSharpeResult | None:
    """
    Compute a simplified Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Args:
        returns: Decimal return series.
        periods_per_year: Annualisation factor.
        n_trials: Effective number of strategy variations tested.

    Returns:
        Dataclass with Sharpe statistics, or ``None`` when undefined.
    """
    values = _ensure_array(returns)
    n_obs = values.size
    if n_obs < 2:
        return None

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if std == 0:
        return None

    sharpe = mean / std * np.sqrt(periods_per_year)
    sr_std = np.sqrt((1 + 0.5 * sharpe**2) / n_obs)

    if n_trials <= 1:
        threshold = 0.0
    else:
        dist = NormalDist()
        z = dist.inv_cdf(1 - 1 / n_trials)
        threshold = sr_std * z

    if sr_std == 0:
        dsr = 0.0
    else:
        dsr = (sharpe - threshold) / sr_std

    return DeflatedSharpeResult(
        sharpe=float(sharpe),
        dsr=float(dsr),
        sr_threshold=float(threshold),
        sr_std=float(sr_std),
        n_obs=int(n_obs),
        n_trials=int(max(1, n_trials)),
    )


def probability_of_backtest_overfitting(
    logits: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """
    Estimate Probability of Backtest Overfitting (PBO) from CSCV logits.

    Args:
        logits: Array of logit values (log(OS / IS) ratios) from CSCV folds.

    Returns:
        Estimated probability that out-of-sample performance is below in-sample.
    """
    values = _ensure_array(logits)
    return float(np.mean(values < 0))
