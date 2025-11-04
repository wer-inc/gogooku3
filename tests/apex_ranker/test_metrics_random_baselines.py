from __future__ import annotations

import torch

from apex_ranker.utils.metrics import (
    k_from_ratio,
    ndcg_at_k,
    ndcg_random_baseline,
    precision_at_k_pos,
    top_bottom_spread,
    topk_overlap,
)


def test_precision_at_k_pos_random_matches_pos_rate() -> None:
    labels = torch.tensor([1.0, -0.5, 0.3, -0.2, 0.0])
    scores = torch.rand_like(labels)
    k = 2
    p_at_k, p_rand = precision_at_k_pos(scores, labels, k)
    expected = (labels > 0).float().mean().item()
    assert abs(p_rand - expected) < 1e-6
    assert 0.0 <= p_at_k <= 1.0


def test_ndcg_random_baseline_between_zero_and_perfect() -> None:
    labels = torch.randn(128)
    k = 10
    ndcg_rand = ndcg_random_baseline(labels, k)
    ndcg_perfect = ndcg_at_k(labels, labels, k)
    assert 0.0 <= ndcg_rand <= 1.0
    assert ndcg_rand <= ndcg_perfect <= 1.0


def test_topk_overlap_and_spread_basic_properties() -> None:
    labels = torch.tensor([0.5, 0.4, 0.3, -0.1, -0.2])
    # perfect alignment when scores == labels
    scores = labels.clone()
    k = 2
    assert abs(topk_overlap(scores, labels, k) - 1.0) < 1e-6
    assert top_bottom_spread(scores, labels, k) > 0.0


def test_k_from_ratio_behaviour() -> None:
    assert k_from_ratio(100, k=15) == 15
    assert k_from_ratio(10, ratio=0.2) == 2
    assert k_from_ratio(5, ratio=0.5) == 2  # banker's rounding behaviour
