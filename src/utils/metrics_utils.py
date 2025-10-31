"""Unified metrics computation utilities."""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def gather_point_preds(
    outputs: list[dict[str, torch.Tensor]], horizon: int
) -> torch.Tensor:
    """
    Gather point predictions from outputs for a specific horizon.

    Args:
        outputs: List of output dictionaries from model forward passes
        horizon: The prediction horizon to gather

    Returns:
        Concatenated predictions as a 1D tensor
    """
    preds = []
    for o in outputs:
        # Try point_horizon_X first, then horizon_X
        pred_key = f"point_horizon_{horizon}"
        if pred_key not in o:
            pred_key = f"horizon_{horizon}"

        if pred_key in o:
            pred = o[pred_key].detach().float().cpu()
            if pred.dim() > 1:
                pred = pred.squeeze(-1)
            preds.append(pred)

    if not preds:
        return torch.tensor([])

    return torch.cat(preds, dim=0)


def compute_pred_std(outputs: list[dict[str, torch.Tensor]], horizon: int) -> float:
    """
    Compute standard deviation of predictions for a specific horizon.
    Always uses point predictions, never t-distribution parameters.

    Args:
        outputs: List of output dictionaries from model forward passes
        horizon: The prediction horizon

    Returns:
        Standard deviation of predictions (always >= 0)
    """
    yhat = gather_point_preds(outputs, horizon)

    if yhat.numel() == 0:
        return 0.0

    std_val = float(yhat.std(unbiased=False).item())

    # Ensure finite and non-negative
    if not np.isfinite(std_val) or std_val < 0:
        return 0.0

    return std_val


def compute_pred_std_batch(output: dict[str, torch.Tensor], horizon: int) -> float:
    """
    Compute standard deviation of predictions for a single batch.

    Args:
        output: Single output dictionary from model forward pass
        horizon: The prediction horizon

    Returns:
        Standard deviation of predictions (always >= 0)
    """
    # Try point_horizon_X first, then horizon_X
    pred_key = f"point_horizon_{horizon}"
    if pred_key not in output:
        pred_key = f"horizon_{horizon}"

    if pred_key not in output:
        return 0.0

    yhat = output[pred_key].detach().float()
    if yhat.dim() > 1:
        yhat = yhat.view(-1)

    if yhat.numel() == 0:
        return 0.0

    std_val = float(yhat.std(unbiased=False).item())

    # Ensure finite and non-negative
    if not np.isfinite(std_val) or std_val < 0:
        return 0.0

    return std_val


def collect_metrics_from_outputs(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    horizons: list[int],
    batch_metadata: dict | None = None,
) -> dict[int, dict[str, list]]:
    """
    Collect metrics from model outputs and targets.

    Args:
        outputs: Model outputs dictionary
        targets: Target values dictionary
        horizons: List of prediction horizons

    Returns:
        Dictionary mapping horizons to collected metrics
    """
    metrics = {
        h: {"y": [], "yhat": [], "t_params": [], "quantiles": [], "groups": []}
        for h in horizons
    }

    group_array = None
    if isinstance(batch_metadata, dict):
        group_val = batch_metadata.get("group_day")
        if isinstance(group_val, torch.Tensor):
            group_array = group_val.detach().view(-1).cpu().numpy()
        elif group_val is not None:
            try:
                group_array = np.asarray(group_val).reshape(-1)
            except Exception:
                group_array = None

    for h in horizons:
        # Collect predictions
        pred_key = f"point_horizon_{h}"
        if pred_key not in outputs:
            pred_key = f"horizon_{h}"

        targ_key = f"horizon_{h}"

        if pred_key in outputs and targ_key in targets:
            yhat = outputs[pred_key].detach().float().view(-1).cpu().numpy()
            y = targets[targ_key].detach().float().view(-1).cpu().numpy()
            metrics[h]["yhat"].append(yhat)
            metrics[h]["y"].append(y)
            if group_array is not None and group_array.shape[0] == yhat.shape[0]:
                metrics[h]["groups"].append(group_array)

        # Collect t-params if available
        t_key = f"t_params_horizon_{h}"
        if t_key in outputs:
            metrics[h]["t_params"].append(outputs[t_key].detach().float().cpu().numpy())

        # Collect quantiles if available
        q_key = f"quantile_horizon_{h}"
        if q_key in outputs:
            metrics[h]["quantiles"].append(
                outputs[q_key].detach().float().cpu().numpy()
            )

    return metrics
