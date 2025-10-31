from __future__ import annotations

import torch


def spearman_ic(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute Spearman rank correlation for a single cross-section."""
    if scores.numel() <= 1 or torch.std(labels) < 1e-8:
        return 0.0

    pred_rank = torch.argsort(torch.argsort(scores, descending=True))
    label_rank = torch.argsort(torch.argsort(labels, descending=True))

    n = scores.numel()
    diff = (pred_rank - label_rank).float()
    rho = 1.0 - (6.0 * (diff.pow(2)).sum()) / (n * (n * n - 1))
    return rho.item()


def precision_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Precision@K based on positive label threshold (>0)."""
    if scores.numel() == 0:
        return 0.0
    k = max(1, min(k, scores.numel()))

    _, top_idx = torch.topk(scores, k=k, largest=True)
    positives = (labels[top_idx] > 0).float()
    return positives.mean().item()
