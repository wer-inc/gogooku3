"""
Graph Builder for ATFT-GAT-FAN
Minimal implementation for training compatibility
"""

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class GBConfig:
    """Graph Builder Configuration"""
    max_nodes: int = 100
    edge_threshold: float = 0.3
    use_dynamic_graph: bool = False
    update_frequency: int = 10
    cache_dir: str | None = "graph_cache"
    ewm_halflife: int = 20
    k: int = 10
    # New safeguards for training stability
    min_k: int = 5  # ensure at least this many neighbors per node (per direction)
    add_self_loops: bool = True  # add i->i edges; many GAT variants expect this
    min_edges: int = 0  # global lower bound on total edges (0 = disabled)
    log_mktcap_col: str | None = None
    lookback: int = 60
    method: str = "ewm_demean"
    min_obs: int = 40
    use_in_training: bool = False
    return_cols: Sequence[str] | None = None
    sector_col: str | None = None
    shrinkage_gamma: float = 0.05
    size_tau: float = 1.0
    source_glob: str | None = None
    symmetric: bool = True

    def __post_init__(self):
        # Ensure numeric fields remain positive where required
        self.max_nodes = max(1, int(self.max_nodes))
        self.k = max(1, int(self.k))
        self.ewm_halflife = max(1, int(self.ewm_halflife))
        self.lookback = max(1, int(self.lookback))
        self.min_obs = max(1, int(self.min_obs))
        self.update_frequency = max(1, int(self.update_frequency))
        if self.return_cols is not None:
            self.return_cols = tuple(self.return_cols)


class GraphBuilder:
    """
    Graph Builder for constructing stock relationship graphs
    Used by ATFT-GAT-FAN for Graph Attention Network component
    """

    def __init__(self, config: GBConfig | None = None):
        self.config = config or GBConfig()
        self.edge_index = None
        self.edge_attr = None
        logger.info(f"GraphBuilder initialized with max_nodes={self.config.max_nodes}")

    def build_graph(self, features: torch.Tensor, codes: list | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph from features.

        Args:
            features: Node features tensor (N, F) or (N, T, F) for time series
            codes: Optional list of stock codes

        Returns:
            edge_index: Edge indices (2, E)
            edge_attr: Edge attributes (E, A)
        """
        n_nodes = min(features.shape[0], self.config.max_nodes)

        # Check if we have returns data for correlation calculation
        if features.ndim == 3 and features.shape[1] >= 20:
            # Use returns correlation (A+ approach)
            edge_index, edge_attr = self.build_correlation_edges(features)
        else:
            # Fallback to simple similarity
            edge_list = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        # Simple threshold based on feature similarity
                        if features.ndim == 3:
                            feat_i = features[i, -1, :]  # Last timestep
                            feat_j = features[j, -1, :]
                        else:
                            feat_i = features[i:i+1]
                            feat_j = features[j:j+1]

                        similarity = torch.cosine_similarity(
                            feat_i.unsqueeze(0) if feat_i.ndim == 1 else feat_i,
                            feat_j.unsqueeze(0) if feat_j.ndim == 1 else feat_j,
                            dim=-1
                        )
                        if similarity > self.config.edge_threshold:
                            edge_list.append([i, j])

            if len(edge_list) == 0:
                # If no edges, create a minimal connected graph
                edge_list = [[i, (i+1) % n_nodes] for i in range(n_nodes)]
                edge_list += [[(i+1) % n_nodes, i] for i in range(n_nodes)]

            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.ones(edge_index.shape[1], 1)  # Simple unit weights

        self.edge_index = edge_index
        self.edge_attr = edge_attr

        return edge_index, edge_attr

    def build_correlation_edges(
        self, features: torch.Tensor, window: int = 20, k: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges based on returns correlation (A+ approach).

        Args:
            features: Time series features (N, T, F) where first feature is returns
            window: Correlation window size
            k: Number of nearest neighbors

        Returns:
            edge_index: Edge indices
            edge_attr: Edge attributes (correlation values)
        """
        n_nodes = int(features.shape[0])
        k = int(min(k, max(1, n_nodes - 1)))

        # Extract returns (assuming first feature channel is returns)
        if features.shape[1] >= window:
            returns = features[:, -window:, 0]  # Last 'window' timesteps, first feature
        else:
            returns = features[:, :, 0]  # Use all available timesteps

        # Compute correlation matrix
        # Normalize returns for correlation calculation
        returns_mean = returns.mean(dim=1, keepdim=True)
        returns_std = returns.std(dim=1, keepdim=True) + 1e-8
        returns_norm = (returns - returns_mean) / returns_std

        # Correlation = (X @ X.T) / T
        corr_matrix = torch.mm(returns_norm, returns_norm.t()) / returns.shape[1]

        # Set diagonal to -inf to exclude self-loops
        corr_matrix.fill_diagonal_(-float('inf'))

        # Get top-k correlations for each node
        values, indices = torch.topk(corr_matrix, k=k, dim=1)

        # Build directed edge set with threshold, then densify to min_k if needed
        edge_set: set[Tuple[int, int]] = set()
        edge_attr_map: dict[Tuple[int, int], float] = {}

        thr = float(self.config.edge_threshold)

        for i in range(n_nodes):
            keep_for_i: List[Tuple[int, float]] = []
            for j_idx, corr_val in zip(indices[i], values[i], strict=False):
                j = int(j_idx.item())
                c = float(corr_val.item())
                if not torch.isinf(corr_val) and c > thr:
                    keep_for_i.append((j, c))
            # If below min_k, backfill with best neighbors regardless of threshold
            if len(keep_for_i) < int(self.config.min_k):
                needed = int(self.config.min_k) - len(keep_for_i)
                # candidates sorted by raw correlation (already sorted by topk)
                for j_idx, corr_val in zip(indices[i], values[i], strict=False):
                    if needed <= 0:
                        break
                    j = int(j_idx.item())
                    if j == i:
                        continue
                    # avoid duplicates
                    if all(j != jj for jj, _ in keep_for_i):
                        keep_for_i.append((j, float(corr_val.item())))
                        needed -= 1
            # Insert directed edges (i -> j)
            for j, c in keep_for_i:
                edge_set.add((i, j))
                # Normalize correlation to [0, 1]
                edge_attr_map[(i, j)] = (c + 1.0) / 2.0

        # Enforce symmetry if configured
        if bool(self.config.symmetric):
            symm_pairs = list(edge_set)
            for i, j in symm_pairs:
                if (j, i) not in edge_set:
                    edge_set.add((j, i))
                    edge_attr_map[(j, i)] = edge_attr_map[(i, j)]

        # Add self-loops if requested
        if bool(self.config.add_self_loops):
            for i in range(n_nodes):
                if (i, i) not in edge_set:
                    edge_set.add((i, i))
                    edge_attr_map[(i, i)] = 1.0

        # Global minimum edges safeguard
        if int(self.config.min_edges) > 0 and len(edge_set) < int(self.config.min_edges):
            # Densify by adding additional top-k neighbors per node cyclically
            needed_extra = int(self.config.min_edges) - len(edge_set)
            if needed_extra > 0:
                for i in range(n_nodes):
                    if needed_extra <= 0:
                        break
                    for j_idx, corr_val in zip(indices[i], values[i], strict=False):
                        if needed_extra <= 0:
                            break
                        j = int(j_idx.item())
                        if (i, j) not in edge_set:
                            edge_set.add((i, j))
                            edge_attr_map[(i, j)] = (float(corr_val.item()) + 1.0) / 2.0
                            needed_extra -= 1

        if not edge_set:
            # Last-resort minimal connectivity
            logger.warning("No edges after safeguards; creating chain connectivity")
            for i in range(max(1, n_nodes - 1)):
                edge_set.add((i, (i + 1) % n_nodes))
                edge_set.add(((i + 1) % n_nodes, i))
                edge_attr_map[(i, (i + 1) % n_nodes)] = 0.5
                edge_attr_map[((i + 1) % n_nodes, i)] = 0.5

        # Materialize tensors
        edge_list: List[Tuple[int, int]] = list(edge_set)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr_vals: List[float] = [edge_attr_map[e] for e in edge_list]
        edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float32).unsqueeze(-1)

        # Diagnostics
        with torch.no_grad():
            deg = torch.bincount(edge_index[0], minlength=n_nodes).float()
            avg_deg = float(deg.mean().item()) if deg.numel() > 0 else 0.0
            min_deg = float(deg.min().item()) if deg.numel() > 0 else 0.0
        logger.info(
            f"Built correlation graph: nodes={n_nodes}, edges={int(edge_index.shape[1])}, "
            f"avg_deg={avg_deg:.2f}, min_deg={min_deg:.0f}, thr={thr}, k={k}, min_k={self.config.min_k}, self_loops={self.config.add_self_loops}"
        )

        return edge_index, edge_attr

    def update_graph(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update graph with new features"""
        return self.build_graph(features)

    def get_adjacency_matrix(self, n_nodes: int) -> torch.Tensor:
        """Get adjacency matrix from edge index"""
        if self.edge_index is None:
            return torch.eye(n_nodes)

        adj = torch.zeros(n_nodes, n_nodes)
        adj[self.edge_index[0], self.edge_index[1]] = 1.0
        return adj

    def to(self, device):
        """Move graph to device"""
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        return self
