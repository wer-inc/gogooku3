"""Unit tests for selection utilities."""
from __future__ import annotations

import pytest
import torch

from apex_ranker.backtest.selection import select_by_percentile


def test_select_by_percentile_basic_top_half() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    indices, threshold, fallback = select_by_percentile(
        scores, k_ratio=0.5, k_min=1, sign=1
    )

    assert fallback is False
    assert threshold == pytest.approx(0.25, rel=1e-6)
    assert indices.tolist() == [2, 3]


def test_select_by_percentile_respects_k_min() -> None:
    scores = torch.tensor([0.05, 0.06, 0.07])
    indices, threshold, fallback = select_by_percentile(
        scores, k_ratio=0.01, k_min=2, sign=1
    )

    assert fallback is True
    assert len(indices) == 2
    # Expect the two largest scores (indices 1 and 2)
    assert indices.tolist() == [1, 2]
    assert threshold == pytest.approx(0.06, rel=1e-6)


def test_select_by_percentile_supports_sign_flip() -> None:
    scores = torch.tensor([0.4, -0.2, -0.3, 0.1])
    indices, threshold, fallback = select_by_percentile(
        scores, k_ratio=0.5, k_min=1, sign=-1
    )

    assert fallback is False
    # Expect indices pointing to the two most negative scores
    assert indices.tolist() == [1, 2]
    # Threshold reported in original orientation
    assert threshold == pytest.approx(0.05, rel=1e-6)
