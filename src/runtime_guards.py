"""
Runtime guard helpers for ATFT-GAT-FAN training.

Installs lightweight sanity checks to catch NaN/Inf activations early and
validates parameter tensors before the training loop starts.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

import torch
from torch import nn

logger = logging.getLogger(__name__)

__all__ = ["apply_all_guards"]


def _iter_tensors(obj: torch.Tensor | Iterable | dict) -> Iterator[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item)


def _validate_parameter_finiteness(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param is None:
            continue
        tensor = param.detach()
        if tensor.numel() == 0:
            continue
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"[runtime-guards] Non-finite parameter detected in '{name}'"
            )
        max_abs = torch.max(torch.abs(tensor)).item()
        if max_abs > 1e3:
            logger.warning(
                "[runtime-guards] Large parameter magnitude detected in '%s' (|w|max=%.3e)",
                name,
                max_abs,
            )


def _register_activation_guard(
    module: nn.Module, name: str, state: set[str]
) -> nn.modules.module.RegisterForwardHook:
    def _hook(_: nn.Module, __: tuple, output: torch.Tensor | Iterable | dict) -> None:
        for tensor in _iter_tensors(output):
            if tensor is None or tensor.numel() == 0:
                continue
            data = tensor.detach()
            if not torch.isfinite(data).all():
                raise RuntimeError(
                    f"[runtime-guards] Detected non-finite activation in '{name}'"
                )
            max_abs = torch.max(torch.abs(data)).item()
            if max_abs > 5.0 and name not in state:
                state.add(name)
                logger.warning(
                    "[runtime-guards] High activation detected in '%s' (|x|max=%.3e)",
                    name,
                    max_abs,
                )

    return module.register_forward_hook(_hook)


def apply_all_guards(model: nn.Module) -> None:
    """
    Attach runtime guards to the model.

    Guards installed:
        * Parameter finiteness check (raises on NaN/Inf)
        * Activation monitor on prediction_head and GAT modules
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    _validate_parameter_finiteness(model)

    guard_state: set[str] = set()
    handles: list[nn.modules.module.RegisterForwardHook] = []
    hook_targets: list[str] = []

    if hasattr(model, "prediction_head") and isinstance(
        model.prediction_head, nn.Module
    ):
        handles.append(
            _register_activation_guard(
                model.prediction_head, "prediction_head", guard_state
            )
        )
        hook_targets.append("prediction_head")

    if hasattr(model, "gat") and isinstance(model.gat, nn.Module):
        handles.append(_register_activation_guard(model.gat, "gat", guard_state))
        hook_targets.append("gat")

    if handles:
        model._runtime_guard_handles = handles
        logger.info(
            "[runtime-guards] Installed %d activation guard(s) on: %s",
            len(handles),
            ", ".join(hook_targets),
        )
    else:
        logger.info("[runtime-guards] No activation guards installed (modules missing)")
