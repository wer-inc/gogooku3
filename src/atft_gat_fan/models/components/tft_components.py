"""Temporal Fusion Transformer components."""

import math
import os

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .grn import GatedResidualNetwork


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # keep pe long enough; will be sliced at forward
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (seq_len, batch, d_model)

        Returns:
            Tensor with positional encoding added
        """
        # Ensure positional buffer is long enough; if not, extend on the fly
        if x.size(0) > self.pe.size(0):
            # regenerate pe to the required length (rare; protects against config drift)
            max_len = int(x.size(0))
            d_model = int(self.pe.size(-1))
            device = self.pe.device
            pe = torch.zeros(max_len, d_model, device=device)
            position = torch.arange(
                0, max_len, dtype=torch.float, device=device
            ).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, device=device).float()
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            # replace buffer to avoid future reallocations
            self.register_buffer("pe", pe, persistent=False)
        return x + self.pe[: x.size(0), :]


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature selection."""

    def __init__(
        self,
        input_size: int,
        num_features: int,
        hidden_size: int,
        dropout: float = 0.1,
        use_sigmoid: bool = True,
        sparsity_coefficient: float = 0.01,
    ):
        """Initialize VSN.

        Args:
            input_size: Input dimension per feature
            num_features: Number of features to select from
            hidden_size: Hidden dimension
            dropout: Dropout rate
            use_sigmoid: Whether to use sigmoid for gating
            sparsity_coefficient: L1 penalty for sparsity
        """
        super().__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.use_sigmoid = use_sigmoid
        self.sparsity_coefficient = sparsity_coefficient

        # Flattened GRN for joint feature processing
        self.flattened_grn = GatedResidualNetwork(
            input_size=num_features * input_size,
            hidden_size=hidden_size,
            output_size=num_features,
            dropout=dropout,
        )

        # Individual feature transformations
        self.feature_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(num_features)
            ]
        )

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            features: Input tensor of shape (batch, seq_len, num_features, input_size)

        Returns:
            Tuple of (selected_features, feature_weights, sparsity_loss)
        """
        batch_size, seq_len, num_features, input_size = features.shape

        # Flatten features for selection
        flattened = features.view(batch_size, seq_len, -1)

        # Get feature weights (compute in fp32 for numerical stability)
        orig_dtype = flattened.dtype
        # Optional gradient checkpointing to reduce VRAM peak
        use_ckpt = os.getenv("GRAD_CHECKPOINT_VSN", "0").lower() in ("1", "true", "yes")
        flat_fp32 = flattened.to(torch.float32)
        if use_ckpt and torch.is_grad_enabled():
            try:
                # Non-reentrant path avoids the requires_grad constraint on inputs (PyTorch 2.x)
                feature_weights = cp.checkpoint(self.flattened_grn, flat_fp32, use_reentrant=False)
            except TypeError:  # PyTorch < 2.0 fallback
                feature_weights = cp.checkpoint(self.flattened_grn, flat_fp32)
            except Exception:
                feature_weights = self.flattened_grn(flat_fp32)
        else:
            feature_weights = self.flattened_grn(flat_fp32)
        feature_weights = feature_weights.to(orig_dtype)
        if self.use_sigmoid:
            feature_weights = torch.sigmoid(feature_weights)
        else:
            feature_weights = torch.softmax(feature_weights, dim=-1)

        # Expand weights for broadcasting
        feature_weights_expanded = feature_weights.unsqueeze(-1)

        # Apply feature-specific transformations in a streaming manner to save memory
        # Avoid stacking all transformed features (which can blow up GPU memory).
        selected_features = features.new_zeros((batch_size, seq_len, self.hidden_size))
        for i in range(num_features):
            feat = features[:, :, i, :]
            if use_ckpt and torch.is_grad_enabled():
                try:
                    transformed = cp.checkpoint(self.feature_grns[i], feat, use_reentrant=False)
                except TypeError:
                    transformed = cp.checkpoint(self.feature_grns[i], feat)
                except Exception:
                    transformed = self.feature_grns[i](feat)
            else:
                transformed = self.feature_grns[i](feat)
            weight_i = feature_weights_expanded[:, :, i, :]  # (B, L, 1)
            selected_features = selected_features + transformed * weight_i

        # Calculate sparsity loss
        sparsity_loss = self.sparsity_coefficient * torch.mean(
            torch.abs(feature_weights)
        )

        return selected_features, feature_weights, sparsity_loss


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer module."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        max_sequence_length: int = 100,
    ):
        """Initialize TFT.

        Args:
            input_size: Input feature dimension
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_positional_encoding: Whether to use positional encoding
            max_sequence_length: Maximum sequence length
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding

        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Positional encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                hidden_size, max_sequence_length
            )

        # Output layers
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.position_wise_ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.output_layer_norm = nn.LayerNorm(hidden_size)

        # Gate for combining LSTM and attention
        self.gate = GatedResidualNetwork(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            return_attention_weights: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Self-attention
        if self.use_positional_encoding:
            # Add positional encoding
            x_pos = x.transpose(0, 1)  # (seq_len, batch, hidden)
            x_pos = self.positional_encoding(x_pos)
            x_pos = x_pos.transpose(0, 1)  # (batch, seq_len, hidden)
        else:
            x_pos = x

        # Apply self-attention
        if return_attention_weights:
            attn_out, attn_weights = self.self_attention(
                x_pos, x_pos, x_pos, need_weights=True
            )
        else:
            attn_out, attn_weights = self.self_attention(
                x_pos, x_pos, x_pos, need_weights=False
            )

        # Residual connection and layer norm
        attn_out = self.attention_layer_norm(attn_out + x)

        # Position-wise feed-forward
        ff_out = self.position_wise_ff(attn_out)
        ff_out = self.output_layer_norm(ff_out + attn_out)

        # Gate to combine LSTM and attention outputs
        combined = torch.cat([lstm_out, ff_out], dim=-1)
        output = self.gate(combined)

        return output, attn_weights
