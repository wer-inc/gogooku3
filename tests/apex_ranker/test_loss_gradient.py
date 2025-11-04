"""Test loss functions maintain gradient graph correctly."""

from __future__ import annotations

import torch

from apex_ranker.losses.ranking import CompositeLoss, ListNetLoss, RankNetLoss


def test_listnet_loss_requires_grad() -> None:
    """ListNetLoss should maintain gradient graph."""
    loss_fn = ListNetLoss(tau=0.5)

    # Valid input
    scores = torch.randn(128, requires_grad=True)
    labels = torch.randn(128)

    loss = loss_fn(scores, labels)

    assert loss is not None, "Loss should not be None for valid input"
    assert loss.requires_grad, "Loss should have requires_grad=True"

    # Backward should work
    loss.backward()
    assert scores.grad is not None, "Scores should have gradients"


def test_listnet_loss_returns_none_for_invalid() -> None:
    """ListNetLoss should return None for invalid inputs (not zero tensor)."""
    loss_fn = ListNetLoss(tau=0.5)

    # Single element
    scores = torch.randn(1, requires_grad=True)
    labels = torch.randn(1)
    assert loss_fn(scores, labels) is None

    # Zero variance
    scores = torch.randn(128, requires_grad=True)
    labels = torch.zeros(128)  # std=0
    assert loss_fn(scores, labels) is None


def test_ranknet_loss_requires_grad() -> None:
    """RankNetLoss should maintain gradient graph."""
    loss_fn = RankNetLoss()

    # Valid input
    scores = torch.randn(128, requires_grad=True)
    labels = torch.randn(128)

    loss = loss_fn(scores, labels)

    assert loss is not None, "Loss should not be None for valid input"
    assert loss.requires_grad, "Loss should have requires_grad=True"

    # Backward should work
    loss.backward()
    assert scores.grad is not None, "Scores should have gradients"


def test_ranknet_loss_returns_none_for_invalid() -> None:
    """RankNetLoss should return None for invalid inputs."""
    loss_fn = RankNetLoss()

    # Single element
    scores = torch.randn(1, requires_grad=True)
    labels = torch.randn(1)
    assert loss_fn(scores, labels) is None

    # Zero variance
    scores = torch.randn(128, requires_grad=True)
    labels = torch.zeros(128)
    assert loss_fn(scores, labels) is None


def test_composite_loss_requires_grad() -> None:
    """CompositeLoss should maintain gradient graph."""
    loss_fn = CompositeLoss(
        listnet_weight=1.0,
        ranknet_weight=0.8,
        mse_weight=0.05,
    )

    # Valid input
    scores = torch.randn(128, requires_grad=True)
    labels = torch.randn(128)

    loss = loss_fn(scores, labels)

    assert loss is not None, "Loss should not be None for valid input"
    assert loss.requires_grad, "Loss should have requires_grad=True"

    # Backward should work
    loss.backward()
    assert scores.grad is not None, "Scores should have gradients"


def test_composite_loss_returns_none_when_all_invalid() -> None:
    """CompositeLoss should return None if all sub-losses are invalid."""
    loss_fn = CompositeLoss(
        listnet_weight=1.0,
        ranknet_weight=0.8,
        mse_weight=0.0,  # Disable MSE (always valid)
    )

    # Zero variance (ListNet and RankNet both invalid)
    scores = torch.randn(128, requires_grad=True)
    labels = torch.zeros(128)

    loss = loss_fn(scores, labels)
    assert loss is None, "Loss should be None when all sub-losses invalid"


def test_composite_loss_partial_aggregation() -> None:
    """CompositeLoss should work when some sub-losses are invalid."""
    loss_fn = CompositeLoss(
        listnet_weight=1.0,
        ranknet_weight=0.8,
        mse_weight=0.05,  # MSE always valid
    )

    # Zero variance (ListNet/RankNet invalid, but MSE valid)
    scores = torch.randn(128, requires_grad=True)
    labels = torch.zeros(128)

    loss = loss_fn(scores, labels)

    assert loss is not None, "Loss should not be None (MSE still valid)"
    assert loss.requires_grad, "Loss should have requires_grad=True"

    # Backward should work
    loss.backward()
    assert scores.grad is not None, "Scores should have gradients"


def test_loss_no_item_float_detach_in_aggregation() -> None:
    """Verify loss aggregation doesn't use .item()/.float()/detach()."""
    loss_fn = CompositeLoss()

    scores = torch.randn(128, requires_grad=True)
    labels = torch.randn(128)

    loss = loss_fn(scores, labels)

    # Check loss is a proper tensor with gradient
    assert isinstance(loss, torch.Tensor), "Loss should be Tensor"
    assert loss.requires_grad, "Loss should have requires_grad=True"
    assert loss.grad_fn is not None, "Loss should have grad_fn (not leaf)"

    # Check backward doesn't raise
    try:
        loss.backward()
    except RuntimeError as e:
        raise AssertionError(f"Backward failed: {e}") from e
