"""Gated Residual Network (GRN) component."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for feature transformation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        return_gate: bool = False,
    ):
        """Initialize GRN.

        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            output_size: Output dimension (if None, same as input_size)
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            return_gate: Whether to return gate values
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.hidden_size = hidden_size
        self.use_layer_norm = use_layer_norm
        self.return_gate = return_gate

        # Linear layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.fc3 = nn.Linear(hidden_size, self.output_size)

        # Gating mechanism
        self.gate_fc1 = nn.Linear(input_size, hidden_size)
        self.gate_fc2 = nn.Linear(hidden_size, self.output_size)

        # Normalization and dropout
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(dropout)

        # Skip connection (if dimensions match)
        if input_size != self.output_size:
            self.skip_proj = nn.Linear(input_size, self.output_size)
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., input_size)

        Returns:
            Output tensor of shape (..., output_size)
            If return_gate is True, returns (output, gate)
        """
        # Skip connection
        if self.skip_proj is not None:
            skip = self.skip_proj(x)
        else:
            skip = x

        # Main transformation path
        hidden = self.fc1(x)
        hidden = F.elu(hidden)
        hidden = self.dropout(hidden)

        hidden = self.fc2(hidden)
        hidden = F.elu(hidden)
        hidden = self.dropout(hidden)

        hidden = self.fc3(hidden)

        # Gating path
        gate = self.gate_fc1(x)
        gate = F.relu(gate)
        gate = self.dropout(gate)
        gate = self.gate_fc2(gate)
        gate = torch.sigmoid(gate)

        # Apply gating
        output = gate * hidden + (1 - gate) * skip

        # Layer normalization
        if self.use_layer_norm:
            output = self.layer_norm(output)

        if self.return_gate:
            return output, gate
        return output
