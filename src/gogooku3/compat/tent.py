"""
Test-Time Adaptation (TENT) utilities.

Conservative implementation that updates only normalization statistics/parameters
(e.g., LayerNorm/BatchNorm) to adapt to distribution shift at inference.

This module is optional and does not affect training unless explicitly used.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


def _iter_norm_modules(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            yield m


def freeze_except_norm(model: nn.Module) -> None:
    """Freeze all parameters except those in normalization layers."""
    for p in model.parameters():
        p.requires_grad = False
    for m in _iter_norm_modules(model):
        for p in m.parameters(recurse=True):
            p.requires_grad = True


def configure_tent_optimizer(model: nn.Module, lr: float = 1e-4) -> torch.optim.Optimizer:
    """Create an optimizer over normalization parameters only."""
    norm_params = []
    for m in _iter_norm_modules(model):
        norm_params += [p for p in m.parameters(recurse=True) if p.requires_grad]
    return torch.optim.Adam(norm_params, lr=lr)


def quantile_width_loss(predictions: dict[str, torch.Tensor]) -> torch.Tensor:
    """A TENT-style unsupervised objective for quantile regressors.

    Minimizes the average predictive interval width across horizons.
    Assumes each tensor is shaped [B, n_quantiles] and sorted ascending.
    """
    if not predictions:
        return torch.tensor(0.0)
    widths = []
    for tensor in predictions.values():
        if tensor.dim() != 2 or tensor.size(1) < 2:
            continue
        w = (tensor[:, -1] - tensor[:, 0]).mean()
        widths.append(w)
    if widths:
        return torch.stack(widths).mean()
    return torch.tensor(0.0)


@torch.no_grad()
def tent_adapt_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    forward_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    steps: int = 1,
    lr: float = 1e-4,
) -> None:
    """Perform a minimal TENT adaptation step on normalization layers.

    Args:
        model: The model to adapt.
        batch: A single inference batch dict. Must be consumable by forward_fn.
        forward_fn: Callable producing dict with key 'predictions' -> per-horizon tensors.
        steps: Number of tiny adaptation steps.
        lr: Learning rate for norm params.
    """
    was_training = model.training
    model.train()  # enable norm stat updates
    freeze_except_norm(model)
    opt = configure_tent_optimizer(model, lr=lr)

    for _ in range(max(1, steps)):
        opt.zero_grad(set_to_none=True)
        # Forward
        outputs = forward_fn(batch)
        preds = outputs.get("predictions", {})
        # Unsupervised objective: shrink predictive width
        loss = quantile_width_loss(preds)
        loss.backward()
        opt.step()

    # Restore eval/train state
    model.train(was_training)

