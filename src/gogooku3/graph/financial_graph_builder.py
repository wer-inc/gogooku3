from __future__ import annotations

"""
Minimal financial graph builder for pipeline compatibility.

Builds a simple correlation-based graph and returns summary stats.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd


class FinancialGraphBuilder:
    def __init__(
        self,
        correlation_window: int = 60,
        include_negative_correlation: bool = True,
        max_edges_per_node: int = 10,
        correlation_method: str = "pearson",
        ewm_halflife: int = 20,
        shrinkage_gamma: float = 0.05,
        symmetric: bool = True,
    ):
        self.max_edges_per_node = max_edges_per_node
        self.correlation_method = correlation_method

    def build_graph(self, df: pd.DataFrame, codes: Iterable[str], date_end: str | None = None) -> dict:
        # Simple implementation: compute correlations between returns columns per code
        try:
            sub = df[df["code"].isin(list(codes))]
            # Pivot a common target if available
            target_col = None
            for c in ("returns_1d", "ret_1d", "feat_ret_1d", "target_1d"):
                if c in sub.columns:
                    target_col = c
                    break
            if target_col is None:
                return {"n_nodes": len(set(sub["code"])), "n_edges": 0}
            wide = sub.pivot_table(index="date", columns="code", values=target_col)
            corr = wide.corr().fillna(0.0)
            # Count edges above threshold (undirected)
            thr = 0.3
            adj = (np.abs(corr.values) > thr).astype(int)
            np.fill_diagonal(adj, 0)
            # Limit edges per node
            if self.max_edges_per_node > 0:
                limited = 0
                for i in range(adj.shape[0]):
                    idx = np.argsort(-adj[i])  # descending
                    keep = idx[: self.max_edges_per_node]
                    row = np.zeros_like(adj[i])
                    row[keep] = adj[i][keep]
                    adj[i] = row
            # Make symmetric
            adj = np.maximum(adj, adj.T)
            n_edges = int(adj.sum() / 2)
            return {"n_nodes": adj.shape[0], "n_edges": n_edges}
        except Exception:
            return {"n_nodes": 0, "n_edges": 0}

