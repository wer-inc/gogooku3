"""Graph Attention Network (GAT) layer implementation."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, dropout_edge

from .graph_norm import GraphNorm


class GATLayer(nn.Module):
    """GAT layer with edge features and regularization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.2,
        edge_dropout: float = 0.1,
        negative_slope: float = 0.2,
        add_self_loops_: bool = True,
        bias: bool = True,
        edge_dim: int = None,
        edge_projection: str = "linear",
    ):
        """Initialize GAT layer.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            heads: Number of attention heads
            concat: Whether to concatenate heads
            dropout: Feature dropout rate
            edge_dropout: Edge dropout rate
            negative_slope: LeakyReLU negative slope
            add_self_loops_: Whether to add self loops
            bias: Whether to use bias
            edge_dim: Edge feature dimension
            edge_projection: How to project edge features
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.add_self_loops = add_self_loops_
        self.edge_dim = edge_dim

        # GAT convolution
        self.conv = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=0.0,  # We handle dropout ourselves
            add_self_loops=False,  # We handle self loops ourselves
            bias=bias,
            edge_dim=edge_dim,
        )

        # Edge projection if needed
        # 注意: edge_projは必要ない。GATConvが内部でedge特徴を処理する
        # if edge_dim is not None and edge_projection == "linear":
        #     self.edge_proj = nn.Linear(edge_dim, heads)
        # else:
        #     self.edge_proj = None
        self.edge_proj = None  # 無効化

        # Feature dropout
        self.feat_dropout = nn.Dropout(dropout)

        # Output projection if not concatenating heads
        if not concat and heads > 1:
            self.out_proj = nn.Linear(out_channels, out_channels)
        else:
            self.out_proj = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        return_attention_weights: bool = False,
        temperature: float = 1.0,
        alpha_min: float = 0.05,
        alpha_penalty: float = 2e-3,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape (num_nodes, in_channels)
            edge_index: Edge indices of shape (2, num_edges)
            edge_attr: Edge features of shape (num_edges, edge_dim)
            return_attention_weights: Whether to return attention weights

        Returns:
            Node embeddings of shape (num_nodes, out_channels * heads) if concat
            else (num_nodes, out_channels)
        """
        # Add self loops if needed
        if self.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=x.size(0)
            )

        # Edge dropout
        if self.training and self.edge_dropout > 0:
            edge_index, edge_mask = dropout_edge(
                edge_index, p=self.edge_dropout, training=self.training
            )
            if edge_attr is not None and edge_mask is not None:
                edge_attr = edge_attr[edge_mask]

        # Feature dropout
        x = self.feat_dropout(x)

        # Project edge features if needed
        if self.edge_proj is not None and edge_attr is not None:
            edge_attr = self.edge_proj(edge_attr)

        # Apply GAT convolution with custom attention
        if return_attention_weights:
            out, (edge_index_att, attention_weights) = self._custom_gat_conv(
                x, edge_index, edge_attr, temperature, alpha_min, alpha_penalty, return_attention_weights=True
            )
        else:
            out = self._custom_gat_conv(
                x, edge_index, edge_attr, temperature, alpha_min, alpha_penalty, return_attention_weights=False
            )

        # Output projection if not concatenating
        if self.out_proj is not None:
            out = self.out_proj(out)

        if return_attention_weights:
            return out, (edge_index_att, attention_weights)
        return out

    def _custom_gat_conv(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        temperature: float = 1.0,
        alpha_min: float = 0.05,
        alpha_penalty: float = 2e-3,
        return_attention_weights: bool = False,
    ):
        """カスタムGAT畳み込み（温度・row-norm・α下限緩和対応）"""
        # 直接GATConvを使用（内部属性へのアクセスを避ける）
        if return_attention_weights:
            return self.conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        else:
            return self.conv(x, edge_index, edge_attr=edge_attr)


class MultiLayerGAT(nn.Module):
    """Multi-layer GAT with residual connections."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        hidden_channels: list,
        heads: list,
        concat_list: list = None,
        dropout: float = 0.2,
        edge_dropout: float = 0.1,
        edge_dim: int = None,
        edge_weight_penalty: float = 0.0,
        attention_entropy_penalty: float = 0.0,
        use_graph_norm: bool = True,
        graph_norm_type: str = "graph",
    ):
        """Initialize multi-layer GAT.

        Args:
            num_layers: Number of GAT layers
            in_channels: Input feature dimension
            hidden_channels: Hidden dimensions for each layer
            heads: Number of heads for each layer
            concat_list: Whether to concatenate heads for each layer
            dropout: Feature dropout rate
            edge_dropout: Edge dropout rate
            edge_dim: Edge feature dimension
            edge_weight_penalty: L2 penalty on edge weights
            attention_entropy_penalty: Entropy penalty on attention
            use_graph_norm: Whether to use GraphNorm instead of LayerNorm
            graph_norm_type: Type of GraphNorm ("graph", "batch", "layer", "instance")
        """
        super().__init__()
        self.num_layers = num_layers
        self.edge_weight_penalty = edge_weight_penalty
        self.attention_entropy_penalty = attention_entropy_penalty

        if concat_list is None:
            concat_list = [True] * (num_layers - 1) + [False]

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        prev_channels = in_channels
        for i in range(num_layers):
            # GAT layer
            layer = GATLayer(
                in_channels=prev_channels,
                out_channels=hidden_channels[i],
                heads=heads[i],
                concat=concat_list[i],
                dropout=dropout,
                edge_dropout=edge_dropout,
                edge_dim=edge_dim
                if i == 0
                else None,  # Only first layer uses edge features
            )
            self.layers.append(layer)

            # Normalization layer
            out_channels = (
                hidden_channels[i] * heads[i] if concat_list[i] else hidden_channels[i]
            )
            if use_graph_norm:
                self.norms.append(
                    GraphNorm(
                        num_features=out_channels,
                        norm_type=graph_norm_type,
                        affine=True,
                    )
                )
            else:
                self.norms.append(nn.LayerNorm(out_channels))

            # Residual projection if dimensions don't match
            if prev_channels != out_channels:
                self.residual_projs.append(nn.Linear(prev_channels, out_channels))
            else:
                self.residual_projs.append(nn.Identity())

            prev_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        batch: torch.Tensor = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            batch: Batch assignment vector for GraphNorm
            return_attention_weights: Whether to return attention weights

        Returns:
            Node embeddings
        """
        attention_weights_list = []

        for i, (layer, norm, res_proj) in enumerate(
            zip(self.layers, self.norms, self.residual_projs)
        ):
            # Residual connection
            residual = res_proj(x)

            # GAT layer
            if i == 0 and edge_attr is not None:
                # Only first layer uses edge attributes
                if return_attention_weights:
                    x, att_weights = layer(
                        x, edge_index, edge_attr, return_attention_weights=True
                    )
                    attention_weights_list.append(att_weights)
                else:
                    x = layer(x, edge_index, edge_attr)
            else:
                if return_attention_weights:
                    x, att_weights = layer(x, edge_index, return_attention_weights=True)
                    attention_weights_list.append(att_weights)
                else:
                    x = layer(x, edge_index)

            # Add residual and normalize
            if isinstance(norm, GraphNorm):
                x = norm(x + residual, batch=batch)
            else:
                x = norm(x + residual)
            x = F.relu(x)

        if return_attention_weights:
            return x, attention_weights_list
        return x

    def get_attention_entropy(self, attention_weights: list) -> torch.Tensor:
        """Calculate attention entropy for regularization.

        Args:
            attention_weights: List of attention weights from each layer

        Returns:
            Average attention entropy
        """
        entropies = []
        for _, weights in attention_weights:
            # Normalize weights per node
            weights = weights.view(-1, weights.size(-1))
            weights = F.softmax(weights, dim=-1)
            # Calculate entropy
            entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
            entropies.append(entropy.mean())

        return torch.stack(entropies).mean() if entropies else torch.tensor(0.0)
