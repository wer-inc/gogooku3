"""Selection helpers for gating model outputs before optimisation."""
from __future__ import annotations

import math

import torch


def _sanitize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Ensure scores are a 1D float tensor on CPU."""
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    if scores.dtype not in (torch.float32, torch.float64):
        scores = scores.to(torch.float32)
    return scores.cpu()


def select_by_percentile(
    scores: torch.Tensor,
    *,
    k_ratio: float,
    k_min: int = 1,
    sign: int = 1,
) -> tuple[torch.Tensor, float, bool]:
    """
    Select indices whose (signed) scores fall within the top ``k_ratio`` percentile.

    Args:
        scores: 1D tensor of model scores (higher is better by default).
        k_ratio: Fraction of the universe to keep (0–1). Values outside this range
            are clipped.
        k_min: Minimum number of securities to keep after gating. Acts as a
            safeguard when the percentile filter is too selective.
        sign: Orientation of the signal. ``+1`` keeps the largest scores (long bias),
            ``-1`` keeps the smallest scores (short bias or long-bottom).

    Returns:
        indices: 1D tensor of selected indices sorted in ascending order.
        threshold: Score threshold used for the percentile cut (after applying
            ``sign``). When the fallback Top-K path is triggered this reflects
            the lowest score retained.
        used_fallback: ``True`` when the percentile filter produced fewer than
            ``max(k_min, round(len(scores) * k_ratio))`` candidates and the
            deterministic Top-K fallback was applied.
    """
    scores = _sanitize_scores(scores)
    n = scores.numel()
    if n == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, float("nan"), False

    sign = 1 if sign >= 0 else -1
    ranked_scores = scores * float(sign)

    k_ratio = float(max(0.0, min(1.0, k_ratio)))
    desired_from_ratio = int(round(k_ratio * n))
    k_min = max(1, int(k_min))
    desired = max(k_min, desired_from_ratio)

    if desired >= n or k_ratio >= 1.0:
        indices = torch.arange(n, dtype=torch.long)
        threshold = float(ranked_scores.min().item())
        return indices, threshold, False

    if k_ratio <= 0.0:
        topk = torch.topk(ranked_scores, min(desired, n), sorted=True)
        indices = topk.indices
        threshold = float(ranked_scores[indices[-1]].item())
        return indices, threshold, True

    threshold = float(torch.quantile(ranked_scores, 1.0 - k_ratio))
    selection = ranked_scores >= threshold
    indices = selection.nonzero(as_tuple=True)[0]

    used_fallback = False
    if indices.numel() < desired:
        used_fallback = True
        topk = torch.topk(ranked_scores, min(desired, n), sorted=True)
        indices = topk.indices
        threshold = float(ranked_scores[indices[-1]].item())

    if indices.numel() == 0:
        indices = torch.arange(min(desired, n), dtype=torch.long)
        threshold = float(ranked_scores[indices[-1]].item())
        used_fallback = True

    indices = torch.unique(indices, sorted=True)
    return indices, threshold, used_fallback


def compute_autosupply_k_ratio(
    candidate_count: int,
    target_top_k: int,
    alpha: float = 1.5,
    floor: float = 0.15,
) -> float:
    """
    Auto-calculate k_ratio to ensure sufficient candidate supply for optimization.

    The 1.5-2.0× multiplier rule ensures the optimizer has enough candidates
    to work with after turnover limits and cost penalties are applied.

    Args:
        candidate_count: Number of candidates available (universe size after filters).
        target_top_k: Target number of holdings in optimized portfolio.
        alpha: Multiplier for autosupply (typically 1.5-2.0).
               1.5× ensures optimization has room to maneuver.
        floor: Minimum k_ratio to return (prevents too few candidates).

    Returns:
        k_ratio: Fraction of candidates to select (0-1), ensuring at least
                ceil(alpha × target_top_k) candidates are available.

    Examples:
        >>> compute_autosupply_k_ratio(70, 35, alpha=1.5)
        0.7571  # 53 candidates = ceil(1.5 × 35)

        >>> compute_autosupply_k_ratio(50, 35, alpha=1.5)
        1.0     # Need 53 but only 50 available → use all

        >>> compute_autosupply_k_ratio(200, 35, alpha=1.5)
        0.265   # 53 candidates = ceil(1.5 × 35), ratio = 53/200
    """
    if candidate_count <= 0 or target_top_k <= 0:
        return floor

    # Calculate required candidates (with multiplier)
    k_required = math.ceil(alpha * target_top_k)

    # Calculate k_ratio
    k_ratio = k_required / max(candidate_count, 1)

    # Apply bounds
    k_ratio = max(floor, min(1.0, k_ratio))

    return k_ratio


__all__ = ["select_by_percentile", "compute_autosupply_k_ratio"]
