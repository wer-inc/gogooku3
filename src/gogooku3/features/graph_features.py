from __future__ import annotations

"""
Graph-structured features (Phase 3) using FinancialGraphBuilder.

Per Date:
- Builds correlation-based graph over recent returns window
- Attaches per-node features:
  - graph_degree: number of edges incident to the node
  - peer_corr_mean: mean absolute correlation to selected peers
  - peer_count: number of selected peers

Leak-safety: uses only past window up to Date (T day), no future info.
"""

from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl

from src.data.utils.graph_builder import FinancialGraphBuilder


def add_graph_features(
    df: pl.DataFrame,
    *,
    return_col: str = "returns_1d",
    window: int = 60,
    min_obs: int = 40,
    threshold: float = 0.3,
    max_k: int = 4,
    method: str = "pearson",
    cache_dir: str | None = None,
) -> pl.DataFrame:
    if df.is_empty() or return_col not in df.columns:
        return df

    # Prepare pandas view with required columns: date, code, return_1d
    pdf = df.select(["Code", "Date", return_col]).rename({
        "Code": "code",
        "Date": "date",
        return_col: "return_1d",
    }).to_pandas()

    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(pdf["date"]):
        pdf["date"] = pd.to_datetime(pdf["date"])

    builder = FinancialGraphBuilder(
        correlation_window=window,
        min_observations=min_obs,
        correlation_threshold=threshold,
        max_edges_per_node=max_k,
        correlation_method=method,
        include_negative_correlation=True,
        symmetric=True,
        cache_dir=cache_dir,
        verbose=False,
    )

    dates = list(pdf["date"].sort_values().unique())
    feats_rows: list[dict] = []

    for dt in dates:
        day_df = pdf.loc[pdf["date"] == dt]
        codes: Iterable[str] = day_df["code"].unique().tolist()
        if len(codes) < 2:
            continue
        try:
            res = builder.build_graph(pdf, codes, date_end=str(dt.date()))
        except Exception:
            continue
        # Peer-level stats per code (from build_graph result, not instance state)
        peer_map = res.get('peer_features', {})
        # Degree, clustering, avg neighbor degree, core number, centralities, components, local density
        deg_map: dict[str, int] = {}
        clus_map: dict[str, float] = {}
        and_map: dict[str, float] = {}
        core_map: dict[str, int] = {}
        degcen_map: dict[str, float] = {}
        close_map: dict[str, float] = {}
        pr_map: dict[str, float] = {}
        comp_size_map: dict[str, int] = {}
        comp_id_map: dict[str, int] = {}
        deg_in_comp_map: dict[str, float] = {}
        local_density_map: dict[str, float] = {}
        try:
            edge_index = res.get("edge_index")
            node_mapping = res.get("node_mapping", {})
            if edge_index is not None and node_mapping:
                # Reverse map idx->code
                inv_map = {v: k for k, v in node_mapping.items()}
                deg_counts = {}
                # edge_index is shape [2, E]
                ei = edge_index.cpu().numpy()
                # Build networkx graph for clustering
                G = nx.Graph()
                G.add_nodes_from(range(len(inv_map)))
                for u, v in ei.T:
                    deg_counts[u] = deg_counts.get(u, 0) + 1
                    deg_counts[v] = deg_counts.get(v, 0) + 1
                    G.add_edge(int(u), int(v))
                # Degree map
                for idx, d in deg_counts.items():
                    code = inv_map.get(idx)
                    if code is not None:
                        deg_map[code] = int(d)
                # Clustering map
                try:
                    clus = nx.clustering(G)
                    for idx, c in clus.items():
                        code = inv_map.get(idx)
                        if code is not None:
                            clus_map[code] = float(c)
                except Exception:
                    pass
                # Average neighbor degree
                try:
                    andeg = nx.average_neighbor_degree(G)
                    for idx, val in andeg.items():
                        code = inv_map.get(idx)
                        if code is not None:
                            and_map[code] = float(val)
                except Exception:
                    pass
                # Core number
                try:
                    core = nx.core_number(G)
                    for idx, val in core.items():
                        code = inv_map.get(idx)
                        if code is not None:
                            core_map[code] = int(val)
                except Exception:
                    pass
                # Degree centrality (normalized)
                try:
                    dc = nx.degree_centrality(G)
                    for idx, val in dc.items():
                        code = inv_map.get(idx)
                        if code is not None:
                            degcen_map[code] = float(val)
                except Exception:
                    pass
                # Closeness centrality (can be heavy on large components)
                try:
                    cc = nx.closeness_centrality(G)
                    for idx, val in cc.items():
                        code = inv_map.get(idx)
                        if code is not None:
                            close_map[code] = float(val)
                except Exception:
                    pass
                # PageRank (lightweight, alpha=0.85)
                try:
                    pr = nx.pagerank(G, alpha=0.85, max_iter=100)
                    for idx, val in pr.items():
                        code = inv_map.get(idx)
                        if code is not None:
                            pr_map[code] = float(val)
                except Exception:
                    pass
                # Connected components: size and per-node component-normalized degree
                try:
                    comps = list(nx.connected_components(G))
                    # Pre-compute per-component mean/std for degree & pagerank (stability within component)
                    deg_mu_comp: dict[int, float] = {}
                    deg_sd_comp: dict[int, float] = {}
                    pr_mu_comp: dict[int, float] = {}
                    pr_sd_comp: dict[int, float] = {}
                    for cid, comp in enumerate(comps):
                        size = len(comp)
                        if size <= 0:
                            continue
                        # Gather values for this component
                        deg_vals: list[float] = []
                        pr_vals: list[float] = []
                        for idx in comp:
                            code = inv_map.get(idx)
                            if code is None:
                                continue
                            comp_id_map[code] = int(cid)
                            comp_size_map[code] = int(size)
                            d = float(deg_counts.get(idx, 0))
                            deg_in_comp_map[code] = (d / (size - 1)) if size > 1 else 0.0
                            deg_vals.append(d)
                            pr_vals.append(float(pr_map.get(code, 0.0)))
                        # Compute stats
                        if len(deg_vals) > 1:
                            dv = np.array(deg_vals, dtype=float)
                            deg_mu_comp[cid] = float(dv.mean())
                            deg_sd_comp[cid] = float(dv.std())
                        else:
                            deg_mu_comp[cid] = 0.0
                            deg_sd_comp[cid] = 0.0
                        if len(pr_vals) > 1:
                            pv = np.array(pr_vals, dtype=float)
                            pr_mu_comp[cid] = float(pv.mean())
                            pr_sd_comp[cid] = float(pv.std())
                        else:
                            pr_mu_comp[cid] = 0.0
                            pr_sd_comp[cid] = 0.0
                except Exception:
                    pass
                # Local density around each node (edges among neighbors / possible)
                try:
                    for idx in G.nodes:
                        k = G.degree(idx)
                        code = inv_map.get(idx)
                        if code is None:
                            continue
                        if k < 2:
                            local_density_map[code] = 0.0
                            continue
                        ego = nx.ego_graph(G, idx, radius=1)
                        m_total = ego.number_of_edges()
                        # edges among neighbors exclude the k spokes to the center
                        m_nn = max(0, m_total - k)
                        local_density_map[code] = (2.0 * m_nn) / (k * (k - 1))
                except Exception:
                    pass
        except Exception:
            pass

        # Degree z-score within date (if possible)
        deg_vals = np.array(list(deg_map.values())) if deg_map else np.array([])
        if deg_vals.size > 1:
            mu = float(deg_vals.mean())
            sd = float(deg_vals.std())
        else:
            mu, sd = 0.0, 0.0

        for code in codes:
            peer = peer_map.get(code, {})
            d = float(deg_map.get(code, 0))
            dz = (d - mu) / sd if sd > 0 else 0.0
            comp_sz = int(comp_size_map.get(code, 0))
            isolated = 1 if comp_sz <= 1 or d == 0 else 0
            # Per-component normalized centralities
            cid = comp_id_map.get(code, -1)
            if cid >= 0:
                mu_d_c = locals().get('deg_mu_comp', {}).get(cid, 0.0)
                sd_d_c = locals().get('deg_sd_comp', {}).get(cid, 0.0)
                mu_pr_c = locals().get('pr_mu_comp', {}).get(cid, 0.0)
                sd_pr_c = locals().get('pr_sd_comp', {}).get(cid, 0.0)
            else:
                mu_d_c = sd_d_c = mu_pr_c = sd_pr_c = 0.0
            deg_z_in_comp = (d - mu_d_c) / sd_d_c if sd_d_c and sd_d_c > 0 else 0.0
            # Pagerank share within component
            pr_val = float(pr_map.get(code, 0.0))
            if comp_sz > 0:
                # sum pr within comp
                comp_id = comp_id_map.get(code, -1)
                if comp_id >= 0:
                    total_pr = 0.0
                    for c2, cid2 in comp_id_map.items():
                        if cid2 == comp_id:
                            total_pr += float(pr_map.get(c2, 0.0))
                    pr_share = (pr_val / total_pr) if total_pr > 0 else 0.0
                else:
                    pr_share = 0.0
            else:
                pr_share = 0.0
            pr_z_in_comp = (pr_val - mu_pr_c) / sd_pr_c if sd_pr_c and sd_pr_c > 0 else 0.0
            feats_rows.append({
                "Code": code,
                "Date": dt,
                "graph_degree": int(d),
                "graph_degree_z": float(dz),
                "graph_degree_z_in_comp": float(deg_z_in_comp),
                "peer_corr_mean": float(peer.get("peer_correlation_mean", 0.0)),
                "peer_count": int(peer.get("peer_count", 0)),
                "graph_clustering": float(clus_map.get(code, 0.0)),
                "graph_avg_neigh_deg": float(and_map.get(code, 0.0)),
                "graph_core": int(core_map.get(code, 0)),
                "graph_degree_centrality": float(degcen_map.get(code, 0.0)),
                "graph_closeness": float(close_map.get(code, 0.0)),
                "graph_pagerank": float(pr_map.get(code, 0.0)),
                "graph_comp_size": int(comp_size_map.get(code, 0)),
                "graph_degree_in_comp": float(deg_in_comp_map.get(code, 0.0)),
                "graph_local_density": float(local_density_map.get(code, 0.0)),
                "graph_pagerank_share_comp": float(pr_share),
                "graph_pagerank_z_in_comp": float(pr_z_in_comp),
                "graph_isolated": int(isolated),
            })

    if not feats_rows:
        return df

    addf = pl.from_pandas(pd.DataFrame(feats_rows)).with_columns([
        pl.col("Date").cast(pl.Date)
    ])
    out = df.join(addf, on=["Code", "Date"], how="left")
    return out
