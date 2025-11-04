"""
Loss functions for ATFT-GAT-FAN training.

Available loss functions:
- RankPreservingLoss: Spearman correlation penalty for rank preservation
- MultiHorizonRankPreservingLoss: Multi-horizon version for ATFT-GAT-FAN
- HuberMultiHorizon: Smooth L1 (Huber) loss per horizon with optional volatility scaling (legacy)
"""

# Import legacy HuberMultiHorizon from parent losses.py for backward compatibility
import importlib.util
from pathlib import Path

from .rank_preserving_loss import MultiHorizonRankPreservingLoss, RankPreservingLoss

_legacy_losses_path = Path(__file__).parent.parent / "losses.py"
_spec = importlib.util.spec_from_file_location("gogooku3.training.losses_legacy", _legacy_losses_path)
_legacy_module = importlib.util.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(_legacy_module)  # type: ignore
HuberMultiHorizon = _legacy_module.HuberMultiHorizon

__all__ = [
    "RankPreservingLoss",
    "MultiHorizonRankPreservingLoss",
    "HuberMultiHorizon",
]
