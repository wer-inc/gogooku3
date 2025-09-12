"""GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training.

Reference: https://arxiv.org/abs/2009.03294
"""

import torch
import torch.nn as nn
from typing import Optional


class GraphNorm(nn.Module):
    """Graph Normalization layer for Graph Neural Networks.

    GraphNorm normalizes node features by considering both the graph structure
    and feature statistics, preventing gradient explosion and NaN issues.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        norm_type: str = "graph",  # "graph", "batch", "layer", "instance"
    ):
        """Initialize GraphNorm layer.

        Args:
            num_features: Number of features to normalize
            eps: Small value to avoid division by zero
            affine: Whether to learn affine parameters (gamma, beta)
            norm_type: Type of normalization to apply
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.norm_type = norm_type

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass of GraphNorm.

        Args:
            x: Node features of shape [num_nodes, num_features]
            batch: Batch assignment vector of shape [num_nodes]
            batch_size: Number of graphs in the batch

        Returns:
            Normalized features of shape [num_nodes, num_features]
        """
        if self.norm_type == "graph":
            return self._graph_norm(x, batch, batch_size)
        elif self.norm_type == "batch":
            return self._batch_norm(x, batch, batch_size)
        elif self.norm_type == "layer":
            return self._layer_norm(x)
        elif self.norm_type == "instance":
            return self._instance_norm(x, batch, batch_size)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")

    def _graph_norm(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor],
        batch_size: Optional[int],
    ) -> torch.Tensor:
        """Graph normalization (default).

        Normalizes across all nodes in each graph separately.
        """
        if batch is None:
            # Single graph case
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        else:
            # Batched graphs case
            if batch_size is None:
                batch_size = int(batch.max().item()) + 1

            # Compute statistics per graph
            x_norm = torch.zeros_like(x)
            for i in range(batch_size):
                mask = batch == i
                if mask.sum() == 0:
                    continue

                x_graph = x[mask]
                mean = x_graph.mean(dim=0, keepdim=True)
                var = x_graph.var(dim=0, keepdim=True, unbiased=False)

                # Prevent NaN by clamping variance
                var = var.clamp(min=self.eps)

                x_norm[mask] = (x_graph - mean) / torch.sqrt(var + self.eps)

        # Apply affine transformation
        if self.gamma is not None and self.beta is not None:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm

    def _batch_norm(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor],
        batch_size: Optional[int],
    ) -> torch.Tensor:
        """Batch normalization across all nodes."""
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        var = var.clamp(min=self.eps)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.gamma is not None and self.beta is not None:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm

    def _layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Layer normalization (normalize across features)."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        var = var.clamp(min=self.eps)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.gamma is not None and self.beta is not None:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm

    def _instance_norm(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor],
        batch_size: Optional[int],
    ) -> torch.Tensor:
        """Instance normalization (normalize each node independently)."""
        # For GNNs, instance norm is equivalent to layer norm
        return self._layer_norm(x)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, eps={self.eps}, "
            f"affine={self.gamma is not None}, norm_type={self.norm_type}"
        )
