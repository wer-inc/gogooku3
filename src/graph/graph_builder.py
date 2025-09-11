"""
Graph Builder for ATFT-GAT-FAN
Minimal implementation for training compatibility
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class GBConfig:
    """Graph Builder Configuration"""
    max_nodes: int = 100
    edge_threshold: float = 0.3
    use_dynamic_graph: bool = False
    update_frequency: int = 10
    
    def __post_init__(self):
        pass


class GraphBuilder:
    """
    Graph Builder for constructing stock relationship graphs
    Used by ATFT-GAT-FAN for Graph Attention Network component
    """
    
    def __init__(self, config: Optional[GBConfig] = None):
        self.config = config or GBConfig()
        self.edge_index = None
        self.edge_attr = None
        logger.info(f"GraphBuilder initialized with max_nodes={self.config.max_nodes}")
    
    def build_graph(self, features: torch.Tensor, codes: Optional[list] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph from features
        
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
    
    def build_correlation_edges(self, features: torch.Tensor, window: int = 20, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges based on returns correlation (A+ approach)
        
        Args:
            features: Time series features (N, T, F) where first feature is returns
            window: Correlation window size
            k: Number of nearest neighbors
            
        Returns:
            edge_index: Edge indices
            edge_attr: Edge attributes (correlation values)
        """
        n_nodes = features.shape[0]
        k = min(k, n_nodes - 1)
        
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
        
        edge_list = []
        edge_attr_list = []
        
        for i in range(n_nodes):
            for j_idx, corr_val in zip(indices[i], values[i]):
                if corr_val > self.config.edge_threshold and not torch.isinf(corr_val):
                    edge_list.append([i, j_idx.item()])
                    # Normalize correlation to [0, 1] range
                    edge_attr_list.append((corr_val.item() + 1.0) / 2.0)
        
        if len(edge_list) == 0:
            # Fallback: create minimal connectivity
            logger.warning("No edges found with correlation threshold, creating minimal graph")
            for i in range(n_nodes - 1):
                edge_list.append([i, i + 1])
                edge_list.append([i + 1, i])
                edge_attr_list.extend([0.5, 0.5])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).unsqueeze(-1)
        
        logger.info(f"Built correlation graph: {n_nodes} nodes, {edge_index.shape[1]} edges")
        
        return edge_index, edge_attr
    
    def update_graph(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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