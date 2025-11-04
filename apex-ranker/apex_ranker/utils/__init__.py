"""Utility helpers for APEX-Ranker."""

from .config import load_config
from .metrics import (
    k_from_ratio,
    ndcg_at_k,
    ndcg_random_baseline,
    precision_at_k,
    precision_at_k_pos,
    spearman_ic,
    top_bottom_spread,
    topk_overlap,
    wil_at_k,
)
from .statistics import (
    DeflatedSharpeResult,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    diebold_mariano,
    ledoit_wolf_sharpe_diff,
    moving_block_bootstrap,
    probability_of_backtest_overfitting,
)

__all__ = [
    "load_config",
    "precision_at_k",
    "precision_at_k_pos",
    "spearman_ic",
    "ndcg_at_k",
    "ndcg_random_baseline",
    "topk_overlap",
    "top_bottom_spread",
    "k_from_ratio",
    "wil_at_k",
    "moving_block_bootstrap",
    "block_bootstrap_ci",
    "diebold_mariano",
    "ledoit_wolf_sharpe_diff",
    "deflated_sharpe_ratio",
    "DeflatedSharpeResult",
    "probability_of_backtest_overfitting",
]
