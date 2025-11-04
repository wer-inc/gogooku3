"""Graph Attention Network (GAT) layer implementation without torch_geometric."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_norm import GraphNorm


def _add_self_loops(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_attr: torch.Tensor | None = None,
    fill_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Add self-loops to the edge index (and attributes)."""
    device = edge_index.device
    loop_indices = torch.arange(num_nodes, device=device)
    loop_edges = torch.stack([loop_indices, loop_indices], dim=0)

    edge_index = torch.cat([edge_index, loop_edges], dim=1)

    if edge_attr is not None:
        attr_dim = edge_attr.size(-1)
        loop_attr = torch.full(
            (num_nodes, attr_dim),
            fill_value,
            device=edge_attr.device,
            dtype=edge_attr.dtype,
        )
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    return edge_index, edge_attr


def _dropout_edge(
    edge_index: torch.Tensor,
    p: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply dropout to edges."""
    if p <= 0.0 or not training:
        return edge_index, None

    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=edge_index.device) > p

    if mask.sum().item() == 0:
        # Avoid removing all edges; fall back to original graph
        return edge_index, torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)

    return edge_index[:, mask], mask


class SimpleGATConv(nn.Module):
    """Minimal GAT convolution implemented in pure PyTorch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        edge_dim: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim

        self.weight = nn.Parameter(torch.empty(in_channels, heads * out_channels))
        self.att_src = nn.Parameter(torch.empty(heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(heads, out_channels))

        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, heads, bias=False)
        else:
            self.edge_proj = None

        if bias:
            bias_dim = heads * out_channels if concat else out_channels
            self.bias = nn.Parameter(torch.zeros(bias_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.edge_proj is not None:
            nn.init.xavier_uniform_(self.edge_proj.weight)

    def _edge_softmax(
        self, scores: torch.Tensor, dst: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute softmax over incoming edges for each destination node."""
        finfo = torch.finfo(scores.dtype)
        max_init = torch.full(
            (num_nodes,),
            -finfo.max,
            dtype=scores.dtype,
            device=scores.device,
        )
        try:
            max_scores = max_init.clone()
            max_scores.scatter_reduce_(
                0, dst, scores, reduce="amax", include_self=True
            )
        except RuntimeError:
            max_scores = max_init.clone()
            for node in range(num_nodes):
                mask = dst == node
                if mask.any():
                    max_scores[node] = torch.max(scores[mask])
        exp_scores = torch.exp(scores - max_scores[dst])

        denom = torch.zeros(
            num_nodes,
            dtype=exp_scores.dtype,
            device=exp_scores.device,
        )
        denom.index_add_(0, dst, exp_scores)

        return exp_scores / (denom[dst] + 1e-9)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        num_nodes = x.size(0)

        if self.add_self_loops:
            edge_index, edge_attr = _add_self_loops(edge_index, num_nodes, edge_attr)

        src, dst = edge_index

        h = torch.matmul(x, self.weight)  # (N, heads * out_channels)
        h = h.view(num_nodes, self.heads, self.out_channels)

        h_src = h[src]  # (E, heads, out_channels)
        h_dst = h[dst]

        att_src = (h_src * self.att_src.unsqueeze(0)).sum(dim=-1)
        att_dst = (h_dst * self.att_dst.unsqueeze(0)).sum(dim=-1)

        scores = att_src + att_dst

        if edge_attr is not None and self.edge_proj is not None:
            edge_scores = self.edge_proj(edge_attr)  # (E, heads)
            scores = scores + edge_scores

        scores = F.leaky_relu(scores, negative_slope=self.negative_slope)

        outputs: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []

        for head in range(self.heads):
            e_head = scores[:, head]
            alpha = self._edge_softmax(e_head, dst, num_nodes)

            if self.training and self.dropout > 0.0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)

            agg = torch.zeros(
                num_nodes,
                self.out_channels,
                device=x.device,
                dtype=x.dtype,
            )
            messages = alpha.unsqueeze(-1) * h_src[:, head, :]
            agg.index_add_(0, dst, messages)
            outputs.append(agg)

            if return_attention_weights:
                attentions.append(alpha)

        out = torch.stack(outputs, dim=1)  # (N, heads, out_channels)

        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            attn = (
                torch.stack(attentions, dim=0)
                if attentions
                else torch.zeros(
                    self.heads,
                    edge_index.size(1),
                    device=out.device,
                    dtype=out.dtype,
                )
            )
            return out, (edge_index, attn)
        return out


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
        edge_dim: int | None = None,
        edge_projection: str = "linear",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.add_self_loops = add_self_loops_
        self.edge_dim = edge_dim

        self.conv = SimpleGATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops_,
            bias=bias,
            edge_dim=edge_dim,
        )

        self.edge_proj = None  # Kept for compatibility
        self.feat_dropout = nn.Dropout(dropout)

        if not concat and heads > 1:
            self.out_proj = nn.Linear(out_channels, out_channels)
        else:
            self.out_proj = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        return_attention_weights: bool = False,
        temperature: float = 1.0,
        alpha_min: float = 0.05,
        alpha_penalty: float = 2e-3,
    ) -> torch.Tensor:
        if self.add_self_loops:
            edge_index, edge_attr = _add_self_loops(
                edge_index, x.size(0), edge_attr
            )

        edge_index, edge_mask = _dropout_edge(
            edge_index, self.edge_dropout, self.training
        )
        if edge_attr is not None and edge_mask is not None:
            edge_attr = edge_attr[edge_mask]

        x = self.feat_dropout(x)

        if return_attention_weights:
            out, attn = self.conv(
                x,
                edge_index,
                edge_attr=edge_attr,
                return_attention_weights=True,
            )
        else:
            out = self.conv(
                x,
                edge_index,
                edge_attr=edge_attr,
                return_attention_weights=False,
            )
            attn = None

        if self.out_proj is not None:
            out = self.out_proj(out)

        if return_attention_weights:
            return out, attn  # type: ignore[return-value]
        return out


class MultiLayerGAT(nn.Module):
    """Multi-layer GAT with residual connections."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        hidden_channels: list[int],
        heads: list[int],
        concat_list: list[bool] | None = None,
        dropout: float = 0.2,
        edge_dropout: float = 0.1,
        edge_dim: int | None = None,
        edge_weight_penalty: float = 0.0,
        attention_entropy_penalty: float = 0.0,
        use_graph_norm: bool = True,
        graph_norm_type: str = "graph",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.edge_weight_penalty = edge_weight_penalty
        self.attention_entropy_penalty = attention_entropy_penalty
        self.use_graph_norm = use_graph_norm

        if concat_list is None:
            concat_list = [True] * num_layers

        layers: list[GATLayer] = []
        norms: list[nn.Module] = []
        residuals: list[nn.Module] = []

        prev_channels = in_channels
        for i in range(num_layers):
            layer = GATLayer(
                in_channels=prev_channels,
                out_channels=hidden_channels[i],
                heads=heads[i],
                concat=concat_list[i],
                dropout=dropout,
                edge_dropout=edge_dropout,
                edge_dim=edge_dim,
            )
            layers.append(layer)

            output_channels = (
                hidden_channels[i] * heads[i] if concat_list[i] else hidden_channels[i]
            )
            prev_channels = output_channels

            if use_graph_norm:
                norms.append(GraphNorm(output_channels, norm_type=graph_norm_type))
            else:
                norms.append(nn.LayerNorm(output_channels))

            if i == 0:
                residual_in = in_channels
            else:
                residual_in = (
                    hidden_channels[i - 1] * heads[i - 1]
                    if concat_list[i - 1]
                    else hidden_channels[i - 1]
                )

            if residual_in != output_channels:
                residuals.append(nn.Linear(residual_in, output_channels))
            else:
                residuals.append(nn.Identity())

        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.residual_projs = nn.ModuleList(residuals)
        self.activation = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None] | torch.Tensor:
        attentions: list[tuple[torch.Tensor, torch.Tensor]] = []

        for idx, (layer, norm, res_proj) in enumerate(
            zip(self.layers, self.norms, self.residual_projs, strict=False)
        ):
            residual = res_proj(x)
            layer_edge_attr = edge_attr if idx == 0 else None

            if return_attention_weights:
                x_new, attn = layer(
                    x,
                    edge_index,
                    edge_attr=layer_edge_attr,
                    return_attention_weights=True,
                )
                if attn is not None:
                    attentions.append(attn)
            else:
                x_new = layer(
                    x,
                    edge_index,
                    edge_attr=layer_edge_attr,
                    return_attention_weights=False,
                )

            x_new = x_new + residual

            if self.use_graph_norm:
                x_new = norm(x_new, batch=batch)
            else:
                x_new = norm(x_new)

            x = self.activation(x_new)

        if return_attention_weights:
            return x, attentions
        return x

    def get_attention_entropy(
        self, attention_weights: list[tuple[torch.Tensor, torch.Tensor]] | None
    ) -> torch.Tensor:
        if not attention_weights:
            return torch.tensor(0.0)

        entropies = []
        for _, weights in attention_weights:
            if weights is None:
                continue
            probs = torch.clamp(weights, min=1e-9)
            entropy = -(probs * probs.log()).mean(dim=-1)
            entropies.append(entropy.mean())

        if not entropies:
            ref = attention_weights[0][1]
            device = ref.device if ref is not None else torch.device("cpu")
            return torch.tensor(0.0, device=device)

        return torch.stack(entropies).mean()
