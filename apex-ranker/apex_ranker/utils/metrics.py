from __future__ import annotations

import math

import torch


def _ensure_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D (got {x.ndim}-D)")
    return x


def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return _ensure_1d(x, "tensor").to(dtype=torch.float32)


def spearman_ic(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute Spearman rank correlation for a single cross-section."""
    scores = _to_float_tensor(scores)
    labels = _to_float_tensor(labels)

    if scores.numel() <= 1 or torch.std(labels) < 1e-8:
        return 0.0

    pred_rank = torch.argsort(torch.argsort(scores, descending=True))
    label_rank = torch.argsort(torch.argsort(labels, descending=True))

    n = scores.numel()
    diff = (pred_rank - label_rank).float()
    rho = 1.0 - (6.0 * (diff.pow(2)).sum()) / (n * (n * n - 1))
    return rho.item()


def k_from_ratio(n: int, *, k: int | None = None, ratio: float | None = None) -> int:
    """Resolve evaluation K from explicit value or ratio (defaults to 10% of universe)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if k is not None:
        return max(1, min(int(k), n))
    if ratio is None:
        ratio = 0.1
    return max(1, min(int(round(n * float(ratio))), n))


def precision_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Legacy Precision@K based on positive label threshold (>0)."""
    value, _ = precision_at_k_pos(scores, labels, k)
    return value


def precision_at_k_pos(
    scores: torch.Tensor, labels: torch.Tensor, k: int
) -> tuple[float, float]:
    """
    Precision@K for positive labels (labels > 0).

    Returns:
        (p_at_k, p_rand) where p_rand is the random baseline = positives / N
    """
    scores = _to_float_tensor(scores)
    labels = _to_float_tensor(labels)
    n = scores.numel()
    if n == 0:
        return 0.0, 0.0

    k = max(1, min(int(k), n))
    pos_mask = (labels > 0).float()
    p_rand = pos_mask.mean()
    _, top_idx = torch.topk(scores, k=k, largest=True)
    p_at_k = pos_mask[top_idx].mean()
    return p_at_k.item(), p_rand.item()


def _softplus_gain(labels: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return torch.nn.functional.softplus(labels * beta)


def ndcg_at_k(
    scores: torch.Tensor, labels: torch.Tensor, k: int, *, beta: float = 1.0
) -> float:
    """
    Compute NDCG@K with non-negative gains (softplus).
    """
    scores = _to_float_tensor(scores)
    labels = _to_float_tensor(labels)
    n = scores.numel()
    if n == 0:
        return 0.0

    k = max(1, min(int(k), n))
    gain = _softplus_gain(labels, beta=beta)
    order = torch.argsort(scores, descending=True)
    top_order = order[:k]
    ranks = torch.arange(1, k + 1, device=scores.device, dtype=torch.float32)
    discounts = torch.log2(ranks + 1.0)
    dcg = (gain[top_order] / discounts).sum()

    ideal_indices = torch.argsort(gain, descending=True)[:k]
    ideal_dcg = (gain[ideal_indices] / discounts).sum().clamp_min(1e-6)
    return (dcg / ideal_dcg).item()


def ndcg_random_baseline(
    labels: torch.Tensor, k: int, *, beta: float = 1.0
) -> float:
    """
    Expected NDCG@K under random rankings (analytical approximation).
    """
    labels = _to_float_tensor(labels)
    n = labels.numel()
    if n == 0:
        return 0.0

    k = max(1, min(int(k), n))
    gain = _softplus_gain(labels, beta=beta)
    mean_gain = gain.mean()
    ranks = torch.arange(1, k + 1, device=labels.device, dtype=torch.float32)
    edcg = mean_gain * (1.0 / torch.log2(ranks + 1.0)).sum()
    ideal_top = torch.topk(gain, k).values
    idcg = (ideal_top / torch.log2(ranks + 1.0)).sum().clamp_min(1e-6)
    value = (edcg / idcg).clamp_max(1.0)
    return value.item()


def topk_overlap(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Fraction of overlap between predicted Top-K and true Top-K."""
    scores = _to_float_tensor(scores)
    labels = _to_float_tensor(labels)
    n = scores.numel()
    if n == 0:
        return 0.0
    k = max(1, min(int(k), n))

    pred_idx = torch.topk(scores, k).indices.cpu().tolist()
    true_idx = torch.topk(labels, k).indices.cpu().tolist()
    inter = len(set(pred_idx).intersection(true_idx))
    return inter / k


def top_bottom_spread(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Mean difference between top-K and bottom-K labels."""
    scores = _to_float_tensor(scores)
    labels = _to_float_tensor(labels)
    n = scores.numel()
    if n == 0:
        return 0.0
    k = max(1, min(int(k), n))

    order = torch.argsort(scores, descending=True)
    top = order[:k]
    bottom = order[-k:]
    spread = labels[top].mean() - labels[bottom].mean()
    return spread.item()


def wil_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    *,
    decay: float = 0.85,
) -> float:
    """
    Weighted information@K style score emphasising higher ranks.
    """
    scores = _to_float_tensor(scores)
    labels = _to_float_tensor(labels)
    if scores.numel() == 0:
        return 0.0
    k = max(1, min(k, scores.numel()))

    _, top_idx = torch.topk(scores, k=k, largest=True)
    weights = torch.tensor(
        [math.pow(decay, i) for i in range(k)],
        dtype=scores.dtype,
        device=scores.device,
    )
    weights = weights / weights.sum()
    positives = (labels[top_idx] > 0).float()
    return (weights * positives).sum().item()
