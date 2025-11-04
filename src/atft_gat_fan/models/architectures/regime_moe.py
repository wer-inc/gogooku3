"""
Regime-aware Mixture-of-Experts (MoE) prediction heads for multi-horizon quantile outputs.

Design goals:
- Drop-in alternative to MultiHorizonPredictionHeads in atft_gat_fan.
- Keep output dict shape: {"horizon_{h}d": [B, n_quantiles]} for each horizon.
- Enhanced gate using regime features (J-UVX, KAMA/VIDYA, market regimes) for better expert selection.

Features:
- J-UVX: Japan Uncertainty & Volatility Index for market stress detection
- KAMA/VIDYA: Adaptive moving averages for trend/momentum efficiency
- Market Regime Classification: Trend vs Range, High vs Low Volatility
- Expert specialization based on market conditions
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class _ExpertHead(nn.Module):
    """One expert that produces per-horizon quantile outputs.

    Each expert shares a small encoder then has horizon-specific linear heads to n_quantiles.
    """

    def __init__(self, hidden_size: int, horizons: list[int], n_quantiles: int, dropout: float = 0.1):
        super().__init__()
        bottleneck = max(16, hidden_size // 2)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(bottleneck),
        )
        self.horizon_heads = nn.ModuleDict({})
        for h in horizons:
            self.horizon_heads[f"horizon_{h}d"] = nn.Linear(bottleneck, n_quantiles)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        out: dict[str, torch.Tensor] = {}
        for key, head in self.horizon_heads.items():
            out[key] = head(z)
        return out


class EnhancedRegimeGate(nn.Module):
    """Enhanced gate that combines backbone features with regime features"""

    def __init__(self,
                 hidden_size: int,
                 num_experts: int,
                 regime_feature_dim: int = 12,  # J-UVX(6) + AMA(2) + Regime(4)
                 use_regime_features: bool = True,
                 dropout: float = 0.1):
        super().__init__()

        self.use_regime_features = use_regime_features
        self.regime_feature_dim = regime_feature_dim

        # Backbone feature encoder
        backbone_hidden = max(32, hidden_size // 2)
        self.backbone_encoder = nn.Sequential(
            nn.Linear(hidden_size, backbone_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if use_regime_features:
            # Regime feature encoder
            regime_hidden = max(16, regime_feature_dim)
            self.regime_encoder = nn.Sequential(
                nn.Linear(regime_feature_dim, regime_hidden),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),  # Less dropout for regime features
            )

            # Attention mechanism to combine backbone and regime
            combined_dim = backbone_hidden + regime_hidden
            self.attention = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.Tanh(),
                nn.Linear(combined_dim // 2, 2),  # attention weights for [backbone, regime]
                nn.Softmax(dim=-1)
            )

            gate_input_dim = combined_dim
        else:
            gate_input_dim = backbone_hidden

        # Final gate prediction
        self.gate_predictor = nn.Sequential(
            nn.Linear(gate_input_dim, max(16, gate_input_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, gate_input_dim // 2), num_experts)
        )

    def forward(self, backbone_features: torch.Tensor, regime_features: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            backbone_features: [B, H] - features from ATFT-GAT-FAN backbone
            regime_features: [B, R] - J-UVX + KAMA/VIDYA + Market regimes (optional)
        Returns:
            gate_logits: [B, E] - unnormalized expert selection logits
        """
        # Encode backbone features
        backbone_encoded = self.backbone_encoder(backbone_features)  # [B, H']

        if self.use_regime_features and regime_features is not None:
            # Encode regime features
            regime_encoded = self.regime_encoder(regime_features)  # [B, R']

            # Combine with attention
            combined = torch.cat([backbone_encoded, regime_encoded], dim=-1)  # [B, H' + R']

            # Attention weights
            attn_weights = self.attention(combined)  # [B, 2]

            # Apply attention (weighted combination)
            weighted_backbone = backbone_encoded * attn_weights[:, 0:1]
            weighted_regime = regime_encoded * attn_weights[:, 1:2]

            gate_input = torch.cat([weighted_backbone, weighted_regime], dim=-1)
        else:
            # Use only backbone features
            gate_input = backbone_encoded

        # Predict gate logits
        gate_logits = self.gate_predictor(gate_input)
        return gate_logits


class RegimeMoEPredictionHeads(nn.Module):
    """Enhanced Regime-aware Mixture-of-Experts for multi-horizon quantile prediction.

    Args:
        hidden_size: Input feature size from backbone.
        config: Model config; expects fields:
            - training.prediction.horizons: list[int]
            - prediction_head.output.quantile_prediction.quantiles: list[float]
            - prediction_head.moe.experts: int (default 3)
            - prediction_head.moe.temperature: float (default 1.0)
            - prediction_head.moe.dropout: float (default 0.1)
            - prediction_head.moe.use_regime_features: bool (default True)
            - prediction_head.moe.regime_feature_dim: int (default 12)
            - prediction_head.moe.balance_lambda: float (optional, for external use)

    Forward:
        x: Tensor of shape [B, H] or [B, T, H]; if 3D, uses last timestep.
        regime_features: Optional[Tensor] of shape [B, R] - regime features for enhanced gating

    Returns:
        Dict with keys per horizon to quantile predictions.
        Additionally stores `last_gate_probs` and `last_regime_features` for analysis.
    """

    def __init__(self, hidden_size: int, config: DictConfig):
        super().__init__()
        self.config = config

        # Horizons
        if hasattr(config.training, "prediction") and hasattr(config.training.prediction, "horizons"):
            self.horizons: list[int] = list(config.training.prediction.horizons)
        else:
            self.horizons = [1, 5, 10, 20]

        # Quantiles
        q_cfg = config.prediction_head.output.quantile_prediction
        self.quantiles: list[float] = list(q_cfg.quantiles)
        n_quantiles = len(self.quantiles)

        # MoE params
        moe_cfg = getattr(config.prediction_head, "moe", None)
        n_experts = int(getattr(moe_cfg, "experts", 3) if moe_cfg is not None else 3)
        temperature = float(getattr(moe_cfg, "temperature", 1.0) if moe_cfg is not None else 1.0)
        dropout = float(getattr(moe_cfg, "dropout", 0.1) if moe_cfg is not None else 0.1)

        # Enhanced regime features settings
        use_regime_features = bool(getattr(moe_cfg, "use_regime_features", True) if moe_cfg is not None else True)
        regime_feature_dim = int(getattr(moe_cfg, "regime_feature_dim", 12) if moe_cfg is not None else 12)

        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)
        self.num_experts = n_experts
        self.use_regime_features = use_regime_features

        # Enhanced gate with regime features
        self.gate = EnhancedRegimeGate(
            hidden_size=hidden_size,
            num_experts=n_experts,
            regime_feature_dim=regime_feature_dim,
            use_regime_features=use_regime_features,
            dropout=dropout
        )

        # Experts (specialized for different market conditions)
        self.experts = nn.ModuleList([
            _ExpertHead(hidden_size, self.horizons, n_quantiles, dropout=dropout)
            for _ in range(n_experts)
        ])

        # Initialize expert specialization (optional)
        self._initialize_expert_specialization()

        # Buffers for analysis
        self.register_buffer("last_gate_probs", torch.zeros(1, n_experts))
        self.register_buffer("last_regime_features", torch.zeros(1, regime_feature_dim))

    def _initialize_expert_specialization(self):
        """Initialize experts with different specializations for market regimes"""
        # Expert 0: Low volatility specialist (smaller weights)
        # Expert 1: High volatility specialist (larger weights)
        # Expert 2: Trend specialist (balanced weights)

        for expert_idx, expert in enumerate(self.experts):
            for name, param in expert.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    if expert_idx == 0:  # Low vol expert
                        nn.init.xavier_uniform_(param, gain=0.5)
                    elif expert_idx == 1:  # High vol expert
                        nn.init.xavier_uniform_(param, gain=1.5)
                    else:  # Balanced expert
                        nn.init.xavier_uniform_(param, gain=1.0)

    def forward(self, x: torch.Tensor, regime_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Enhanced forward pass with regime features support

        Args:
            x: [B, H] or [B, T, H] - backbone features from ATFT-GAT-FAN
            regime_features: [B, R] - regime features (J-UVX + KAMA/VIDYA + market regimes)
        """
        if x.dim() == 3:
            # [B, T, H] -> use last step
            x = x[:, -1, :]

        # Store regime features for analysis
        if regime_features is not None:
            self.last_regime_features = regime_features.detach()

        # Enhanced gate with regime features
        gate_logits = self.gate(x, regime_features)  # [B, E]
        probs = torch.softmax(gate_logits / self.temperature.clamp(min=1e-6), dim=-1)  # [B, E]
        self.last_gate_probs = probs.detach()

        # Expert outputs
        expert_outs = [exp(x) for exp in self.experts]  # list of dicts per expert

        # Initialize mixture outputs with zeros
        mixed: dict[str, torch.Tensor] = {}
        for h in self.horizons:
            key = f"horizon_{h}d"
            mixed[key] = torch.zeros(x.size(0), len(self.quantiles), device=x.device, dtype=x.dtype)

        # Weighted sum across experts: Σ_k p_k * E_k(h)
        for e_idx, out_dict in enumerate(expert_outs):
            p_e = probs[:, e_idx].unsqueeze(-1)  # [B, 1]
            for key, val in out_dict.items():
                mixed[key] = mixed[key] + p_e * val

        # Enforce non-crossing quantiles by sorting (stable, differentiable w.r.t. values ordering)
        for key in mixed.keys():
            mixed[key], _ = torch.sort(mixed[key], dim=-1)

        return mixed

    def get_gate_analysis(self) -> dict[str, torch.Tensor]:
        """Get gate analysis for interpretability"""
        return {
            "gate_probs": self.last_gate_probs,
            "regime_features": self.last_regime_features,
            "expert_specialization": torch.tensor([0.5, 1.5, 1.0])  # gain values used in initialization
        }


# Lightweight utility for load-balance regularization
def moe_load_balance_penalty(gate_probs: torch.Tensor) -> torch.Tensor:
    """Encourage uniform expert utilization.

    Args:
        gate_probs: [B, E]
    Returns:
        Scalar loss tensor.
    """
    if gate_probs.numel() == 0:
        return torch.tensor(0.0, device=gate_probs.device)
    mean_usage = gate_probs.mean(dim=0)  # [E]
    target = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
    return ((mean_usage - target) ** 2).sum()

