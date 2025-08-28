from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_knn_from_embeddings(
    h: torch.Tensor,
    k: int = 15,
    exclude_self: bool = True,
    symmetric: bool = True,
    min_degree: int = 1,
):
    """
    Build KNN graph from node embeddings with improved robustness.

    Args:
        h: [N, d] Node embeddings
        k: Number of nearest neighbors
        exclude_self: Whether to exclude self-loops in KNN search
        symmetric: Whether to make the graph symmetric (bidirectional edges)
        min_degree: Minimum degree for each node (adds self-loops if needed)

    Returns:
        edge_index: [2, E] Edge indices (long dtype, contiguous)
        edge_attr: [E, 1] Edge attributes (cosine similarity in [0,1]) or None
    """
    if h.ndim != 2:
        raise ValueError(f"Expected 2D tensor [N, d], got shape={tuple(h.shape)}")

    N = h.size(0)
    device = h.device

    # Handle edge cases
    if N <= 1:
        if min_degree > 0:
            # Create self-loop for single node
            return torch.zeros((2, 1), dtype=torch.long, device=device), None
        else:
            # No edges
            return torch.zeros((2, 0), dtype=torch.long, device=device), None

    # Compute cosine similarity matrix
    h_norm = F.normalize(h, p=2, dim=1)
    sim = torch.matmul(h_norm, h_norm.t())  # [N, N]

    # Exclude self-loops if requested
    if exclude_self:
        sim.fill_diagonal_(-1e9)  # Large negative value to exclude from top-k

    # Find k nearest neighbors
    k_actual = min(int(k), N - 1 if exclude_self else N)
    if k_actual <= 0:
        k_actual = 1  # At least one neighbor

    vals, idxs = torch.topk(sim, k=k_actual, dim=1)

    # Build edge index
    rows = torch.arange(N, device=device).unsqueeze(1).expand_as(idxs)
    edge_index = torch.stack([rows.reshape(-1), idxs.reshape(-1)], dim=0)  # [2, N*k]

    # Make symmetric if requested
    if symmetric:
        # Add reverse edges
        edge_index_rev = edge_index[[1, 0], :]
        edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
        # Remove duplicates (keep unique edges)
        edge_set = set()
        unique_edges = []
        for i in range(edge_index.size(1)):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            if edge not in edge_set:
                edge_set.add(edge)
                unique_edges.append(i)
        if unique_edges:
            edge_index = edge_index[:, unique_edges]

    # Ensure minimum degree by adding self-loops if needed
    if min_degree > 0:
        # Count degree for each node
        deg = torch.bincount(edge_index[0], minlength=N)
        # Find nodes with degree less than min_degree
        lonely = torch.nonzero(deg < min_degree, as_tuple=False).flatten()
        if lonely.numel() > 0:
            # Add self-loops for lonely nodes
            self_loops = torch.stack([lonely, lonely], dim=0)
            edge_index = torch.cat([edge_index, self_loops], dim=1)

    # Return edge_index as long and contiguous, edge_attr as None for simplicity
    # (GAT doesn't require edge attributes for basic operation)
    return edge_index.long().contiguous(), None
