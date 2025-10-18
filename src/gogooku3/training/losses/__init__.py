"""
Loss functions for ATFT-GAT-FAN training.

Available loss functions:
- RankPreservingLoss: Spearman correlation penalty for rank preservation
- MultiHorizonRankPreservingLoss: Multi-horizon version for ATFT-GAT-FAN
"""

from .rank_preserving_loss import MultiHorizonRankPreservingLoss, RankPreservingLoss

__all__ = [
    "RankPreservingLoss",
    "MultiHorizonRankPreservingLoss",
]
