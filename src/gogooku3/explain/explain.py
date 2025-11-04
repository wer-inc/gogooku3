from __future__ import annotations

"""
Explainability utilities for ATFT-GAT-FAN.

Provides two low-dependency paths:
  - Variable Selection gates snapshot (built-in)
  - Gradient-based attributions (Integrated Gradients-like)

Optional:
  - If 'shap' is available, expose a KernelExplainer entry point.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ExplainConfig:
    horizon_key: str = "horizon_1d"
    top_k: int = 20


def export_vsn_gates(model: nn.Module, output_path: Path) -> Path | None:
    """Export last VSN gates if present on the model instance."""
    gates = getattr(model, "_last_variable_gates", None)
    if gates is None or not torch.is_tensor(gates):
        logger.warning("No VSN gates available on model; run a forward pass first.")
        return None
    vals = gates.detach().float().cpu().numpy().tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"vsn_gates": vals}, indent=2), encoding="utf-8")
    logger.info("Saved VSN gates to %s", output_path)
    return output_path


def integrated_gradients(model: nn.Module, inputs: torch.Tensor, target_index: int | None = None, steps: int = 32) -> torch.Tensor:
    """Simple Integrated Gradients attribution for last-step score.

    Assumes inputs shape [B, T, F]. Returns attributions with same shape.
    """
    model.eval()
    baseline = torch.zeros_like(inputs)
    attributions = torch.zeros_like(inputs)
    for alpha in torch.linspace(0, 1, steps, device=inputs.device):
        x = baseline + alpha * (inputs - baseline)
        x.requires_grad_(True)
        out = model.forward_features_only(x) if hasattr(model, "forward_features_only") else model(x)
        # Fallback: if model returns dict, pick primary horizon median score
        if isinstance(out, dict):
            # choose first tensor in dict
            y = next((v for v in out.values() if torch.is_tensor(v)), None)
        else:
            y = out
        if y is None:
            raise RuntimeError("Model forward produced no tensor output")
        # Reduce to scalar per-sample
        if y.ndim > 2:
            y = y[..., 0]
        loss = y.sum()
        grads = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
        attributions = attributions + grads
    attributions = (inputs - baseline) * attributions / steps
    return attributions.detach()


def try_shap_kernel(model: nn.Module, sample: torch.Tensor, nsamples: int = 100) -> Any | None:
    """Best-effort SHAP KernelExplainer (optional dependency)."""
    try:
        import shap  # type: ignore
        model.eval()

        def f(x):
            xt = torch.tensor(x, dtype=sample.dtype, device=sample.device).view(sample.shape)
            out = model(xt)
            if isinstance(out, dict):
                out = next((v for v in out.values() if torch.is_tensor(v)), None)
            return out.detach().cpu().numpy()

        explainer = shap.KernelExplainer(f, sample.detach().cpu().numpy())
        shap_values = explainer.shap_values(sample.detach().cpu().numpy(), nsamples=nsamples)
        return shap_values
    except Exception as e:
        logger.warning("SHAP not available or failed: %s", e)
        return None

