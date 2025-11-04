"""
Graph Builder for ATFT-GAT-FAN
Minimal implementation for training compatibility
"""

import glob
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date as _date_cls
from datetime import datetime as _datetime_cls
from datetime import timedelta
from typing import Any

import polars as pl
import torch

try:
    import pandas as pd  # noqa: F401  # optional downstream use
except Exception:  # pragma: no cover - pandas optional
    pd = None  # type: ignore[assignment]

from src.data.utils.graph_builder import FinancialGraphBuilder

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
    k: int = 20
    # New safeguards for training stability
    min_k: int = 10  # ensure at least this many neighbors per node (per direction)
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
    returns_channel_index: int | None = None
    date_column: str = "date"
    code_column: str = "code"
    market_col: str | None = None

    def __post_init__(self) -> None:
        # Ensure numeric fields remain positive where required
        self.max_nodes = max(1, int(self.max_nodes))
        self.k = max(1, int(self.k))
        self.ewm_halflife = max(1, int(self.ewm_halflife))
        self.lookback = max(1, int(self.lookback))
        self.min_obs = max(1, int(self.min_obs))
        self.update_frequency = max(1, int(self.update_frequency))
        self.min_k = max(0, min(int(self.min_k), self.k))
        if self.return_cols is not None:
            self.return_cols = tuple(str(col) for col in self.return_cols)
        if self.returns_channel_index is not None:
            self.returns_channel_index = max(0, int(self.returns_channel_index))


class GraphBuilder:
    """
    Graph Builder for constructing stock relationship graphs
    Used by ATFT-GAT-FAN for Graph Attention Network component
    """

    def __init__(self, config: GBConfig | None = None):
        self.config = config or GBConfig()
        self.edge_index = None
        self.edge_attr = None
        self._warned_missing_returns = False
        self._last_asof_ts: _datetime_cls | None = None
        self._last_result: dict[str, Any] | None = None

        # Allow GRAPH_RET_IDX env override to align with runtime pipelines
        env_idx = os.getenv("GRAPH_RET_IDX", "").strip()
        if env_idx:
            try:
                self.config.returns_channel_index = max(0, int(env_idx))
            except ValueError:
                logger.warning("GRAPH_RET_IDX=%s cannot be interpreted as int", env_idx)

        self._returns_channel_index = self.config.returns_channel_index
        self.edge_index = None
        self.edge_attr = None
        logger.info(
            "GraphBuilder initialized with max_nodes=%d, returns_channel_index=%s",
            self.config.max_nodes,
            "auto"
            if self._returns_channel_index is None
            else self._returns_channel_index,
        )

        self.source_glob = getattr(self.config, "source_glob", None)
        self.date_column = getattr(self.config, "date_column", "date")
        self.code_column = getattr(self.config, "code_column", "code")
        self.return_columns = list(getattr(self.config, "return_cols", ()) or ())
        if not self.return_columns:
            self.return_columns = ["return_1d", "feat_ret_1d", "target_1d"]

        self._financial_builder: FinancialGraphBuilder | None = None
        if self.source_glob:
            try:
                self._financial_builder = FinancialGraphBuilder(
                    correlation_window=int(getattr(self.config, "lookback", 60)),
                    min_observations=int(getattr(self.config, "min_obs", 40)),
                    correlation_threshold=float(
                        getattr(self.config, "edge_threshold", 0.3)
                    ),
                    max_edges_per_node=int(getattr(self.config, "k", 10)),
                    include_negative_correlation=True,
                    correlation_method=str(
                        getattr(self.config, "method", "ewm_demean")
                    ),
                    ewm_halflife=int(getattr(self.config, "ewm_halflife", 20)),
                    shrinkage_gamma=float(
                        getattr(self.config, "shrinkage_gamma", 0.05)
                    ),
                    symmetric=bool(getattr(self.config, "symmetric", True)),
                    cache_dir=str(getattr(self.config, "cache_dir", "graph_cache")),
                    verbose=False,
                    sector_col=getattr(self.config, "sector_col", None),
                    market_col=getattr(self.config, "market_col", None),
                )
                logger.info(
                    "FinancialGraphBuilder enabled (source_glob=%s, window=%d)",
                    self.source_glob,
                    int(getattr(self.config, "lookback", 60)),
                )
            except Exception as exc:
                logger.warning("Failed to initialize FinancialGraphBuilder: %s", exc)
                self._financial_builder = None
        else:
            logger.info(
                "GraphBuilder running without source_glob; using local correlation fallback."
            )

    def build_graph(
        self,
        features: torch.Tensor | dict[str, Any],
        codes: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph from features.

        Args:
            features: Node features tensor (N, F) or (N, T, F) for time series
            codes: Optional list of stock codes

        Returns:
            edge_index: Edge indices (2, E)
            edge_attr: Edge attributes (E, A)
        """
        tensor_features = (
            features
            if torch.is_tensor(features)
            else features.get("features")
            if isinstance(features, dict)
            else None
        )

        if tensor_features is None and torch.is_tensor(features):
            tensor_features = features

        if tensor_features is None or tensor_features.ndim < 2:
            raise ValueError("GraphBuilder expects features with at least 2 dimensions")

        n_nodes = min(tensor_features.shape[0], self.config.max_nodes)

        returns_matrix = self._extract_returns_matrix(features)

        if returns_matrix is not None and returns_matrix.shape[1] >= 2:
            edge_index, edge_attr = self.build_correlation_edges(returns_matrix)
        else:
            # Fallback to simple similarity
            edge_list = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        # Simple threshold based on feature similarity
                        if tensor_features.ndim == 3:
                            feat_i = tensor_features[i, -1, :]  # Last timestep
                            feat_j = tensor_features[j, -1, :]
                        else:
                            feat_i = tensor_features[i : i + 1]
                            feat_j = tensor_features[j : j + 1]

                        similarity = torch.cosine_similarity(
                            feat_i.unsqueeze(0) if feat_i.ndim == 1 else feat_i,
                            feat_j.unsqueeze(0) if feat_j.ndim == 1 else feat_j,
                            dim=-1,
                        )
                        if similarity > self.config.edge_threshold:
                            edge_list.append([i, j])

            if len(edge_list) == 0:
                # If no edges, create a minimal connected graph
                edge_list = [[i, (i + 1) % n_nodes] for i in range(n_nodes)]
                edge_list += [[(i + 1) % n_nodes, i] for i in range(n_nodes)]

            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.ones(edge_index.shape[1], 1)  # Simple unit weights

        self.edge_index = edge_index
        self.edge_attr = edge_attr

        return edge_index, edge_attr

    def _extract_returns_matrix(
        self, features: torch.Tensor | dict[str, Any]
    ) -> torch.Tensor | None:
        """
        Try to extract a (N, T) returns matrix from different feature containers.

        Supports:
        - dict batches containing keys like 'returns', 'past_ret', or 'return_matrix'
        - tensor features with shape (N, T, F) using configured return channel index
        """

        if isinstance(features, dict):
            candidate_keys = ("returns", "return_matrix", "past_ret", "returns_matrix")
            for key in candidate_keys:
                value = features.get(key)
                if torch.is_tensor(value):
                    if value.dim() == 3:
                        # Collapse feature dimension if present
                        return value[..., 0].detach()
                    if value.dim() == 2:
                        return value.detach()
            tensor_features = features.get("features")
        else:
            tensor_features = features

        if tensor_features is None or not torch.is_tensor(tensor_features):
            return None

        if tensor_features.dim() < 3:
            if not self._warned_missing_returns:
                logger.warning(
                    "GraphBuilder received tensor with shape %s; unable to extract "
                    "returns history for correlation graph.",
                    tuple(tensor_features.shape),
                )
                self._warned_missing_returns = True
            return None

        # Resolve channel index to use
        channel_idx = self._returns_channel_index
        if channel_idx is None:
            # Default to channel 0 but warn once so configs can be explicit
            channel_idx = 0
            if not self._warned_missing_returns:
                logger.warning(
                    "GraphBuilder fallback: returns_channel_index not provided; "
                    "defaulting to channel 0. Set GBConfig.returns_channel_index "
                    "or GRAPH_RET_IDX to control this selection."
                )
                self._warned_missing_returns = True

        channel_idx = max(0, min(int(channel_idx), tensor_features.size(-1) - 1))
        return tensor_features[:, :, channel_idx].detach()

    def build_correlation_edges(
        self,
        data: torch.Tensor,
        window: int | None = None,
        k: int | None = None,
        sectors: Sequence[Any] | None = None,
        markets: Sequence[Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges based on returns correlation (A+ approach).

        Args:
            data: Either (N, T) returns matrix or (N, T, F) feature tensor
            window: Correlation window size
            k: Number of nearest neighbors

        Returns:
            edge_index: Edge indices
            edge_attr: Edge attributes (correlation values)
        """
        if data.dim() == 3:
            returns_matrix = self._extract_returns_matrix({"features": data})
            if returns_matrix is None:
                raise ValueError(
                    "Unable to derive returns history from feature tensor for correlation edges."
                )
        else:
            returns_matrix = data

        if returns_matrix.dim() != 2:
            raise ValueError(
                f"Correlation graph expects returns with shape (N, T); got {tuple(returns_matrix.shape)}"
            )

        n_nodes = int(returns_matrix.shape[0])
        k = int(min(k, max(1, n_nodes - 1)))

        # Resolve parameters from config if not provided
        corr_window = int(window if window is not None else self.config.lookback)
        k_neighbors = int(k if k is not None else self.config.k)
        k_neighbors = max(1, min(k_neighbors, n_nodes - 1))

        if returns_matrix.shape[1] >= corr_window:
            returns = returns_matrix[:, -corr_window:]
        else:
            returns = returns_matrix

        # Compute correlation matrix
        # Normalize returns for correlation calculation
        returns_mean = returns.mean(dim=1, keepdim=True)
        returns_std = returns.std(dim=1, keepdim=True) + 1e-8
        returns_norm = (returns - returns_mean) / returns_std

        # Correlation = (X @ X.T) / T
        corr_matrix = torch.mm(returns_norm, returns_norm.t()) / returns.shape[1]

        # Set diagonal to -inf to exclude self-loops
        corr_matrix.fill_diagonal_(-float("inf"))

        # Get top-k correlations for each node
        abs_corr = corr_matrix.abs()
        abs_corr.fill_diagonal_(-float("inf"))
        values, indices = torch.topk(abs_corr, k=k_neighbors, dim=1)

        # Build directed edge set with threshold, then densify to min_k if needed
        edge_set: set[tuple[int, int]] = set()
        edge_attr_map: dict[tuple[int, int], list[float]] = {}

        thr = float(self.config.edge_threshold)

        for i in range(n_nodes):
            keep_for_i: list[tuple[int, float]] = []
            for j_idx, abs_val in zip(indices[i], values[i], strict=False):
                j = int(j_idx.item())
                if j == i:
                    continue
                if not torch.isinf(abs_val):
                    actual_corr = float(corr_matrix[i, j].item())
                    if abs(actual_corr) >= thr:
                        keep_for_i.append((j, actual_corr))
            # If below min_k, backfill with best neighbors regardless of threshold
            if len(keep_for_i) < int(self.config.min_k):
                needed = int(self.config.min_k) - len(keep_for_i)
                # candidates sorted by raw correlation (already sorted by topk)
                for j_idx, abs_val in zip(indices[i], values[i], strict=False):
                    if needed <= 0:
                        break
                    j = int(j_idx.item())
                    if j == i:
                        continue
                    # avoid duplicates
                    if all(j != jj for jj, _ in keep_for_i):
                        keep_for_i.append((j, float(corr_matrix[i, j].item())))
                        needed -= 1
            # Insert directed edges (i -> j)
            for j, c in keep_for_i:
                edge_set.add((i, j))
                same_sector = (
                    1.0
                    if sectors is not None
                    and len(sectors) == n_nodes
                    and sectors[i] == sectors[j]
                    else 0.0
                )
                same_market = (
                    1.0
                    if markets is not None
                    and len(markets) == n_nodes
                    and markets[i] == markets[j]
                    else 0.0
                )
                attr_vals: list[float] = [float(c), same_sector, same_market]
                edge_attr_map[(i, j)] = attr_vals

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
                    same_sector = (
                        1.0
                        if sectors is not None
                        and len(sectors) == n_nodes
                        else 0.0
                    )
                    same_market = (
                        1.0
                        if markets is not None
                        and len(markets) == n_nodes
                        else 0.0
                    )
                    attr_vals = [1.0, same_sector, same_market]
                    edge_attr_map[(i, i)] = attr_vals

        # Global minimum edges safeguard
        if int(self.config.min_edges) > 0 and len(edge_set) < int(
            self.config.min_edges
        ):
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
                            same_sector = (
                                1.0
                                if sectors is not None
                                and len(sectors) == n_nodes
                                and sectors[i] == sectors[j]
                                else 0.0
                            )
                            same_market = (
                                1.0
                                if markets is not None
                                and len(markets) == n_nodes
                                and markets[i] == markets[j]
                                else 0.0
                            )
                            attr_vals = [float(corr_val.item()), same_sector, same_market]
                            edge_attr_map[(i, j)] = attr_vals
                            needed_extra -= 1

        if not edge_set:
            # Last-resort minimal connectivity
            logger.warning("No edges after safeguards; creating chain connectivity")
            for i in range(max(1, n_nodes - 1)):
                edge_set.add((i, (i + 1) % n_nodes))
                edge_set.add(((i + 1) % n_nodes, i))
                default_attr = [0.5, 0.0, 0.0]
                edge_attr_map[(i, (i + 1) % n_nodes)] = list(default_attr)
                edge_attr_map[((i + 1) % n_nodes, i)] = list(default_attr)

        # Materialize tensors
        edge_list: list[tuple[int, int]] = list(edge_set)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        attr_dim = 3
        edge_attr_vals: list[list[float]] = [edge_attr_map[e] for e in edge_list]
        if not edge_attr_vals:
            edge_attr = torch.empty((0, attr_dim), dtype=torch.float32)
        else:
            edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float32)

        # Diagnostics
        with torch.no_grad():
            deg = torch.bincount(edge_index[0], minlength=n_nodes).float()
            avg_deg = float(deg.mean().item()) if deg.numel() > 0 else 0.0
            min_deg = float(deg.min().item()) if deg.numel() > 0 else 0.0
        logger.info(
            f"Built correlation graph: nodes={n_nodes}, edges={int(edge_index.shape[1])}, "
            f"avg_deg={avg_deg:.2f}, min_deg={min_deg:.0f}, thr={thr}, k={k_neighbors}, "
            f"min_k={self.config.min_k}, self_loops={self.config.add_self_loops}"
        )

        return edge_index, edge_attr

    def update_graph(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update graph with new features"""
        return self.build_graph(features)

    # ------------------------------------------------------------------
    # FinancialGraphBuilder integration
    # ------------------------------------------------------------------
    def _parse_date(self, value: str | _date_cls | _datetime_cls) -> _date_cls:
        if isinstance(value, _datetime_cls):
            return value.date()
        if isinstance(value, _date_cls):
            return value
        if isinstance(value, str):
            try:
                return _datetime_cls.strptime(value[:10], "%Y-%m-%d").date()
            except ValueError:
                return _datetime_cls.fromisoformat(value).date()
        raise TypeError(f"Unsupported date value: {value!r}")

    def _load_source_data(
        self, date_end: str | _date_cls | _datetime_cls, codes: Sequence[str]
    ) -> pl.DataFrame | None:
        if not self.source_glob:
            return None
        try:
            target_date = self._parse_date(date_end)
        except Exception as exc:
            logger.warning("Failed to parse date %r: %s", date_end, exc)
            return None

        lookback = int(getattr(self.config, "lookback", 60))
        buffer_days = max(5, lookback // 4)
        start_date = target_date - timedelta(days=lookback + buffer_days)

        codes_str = [str(code) for code in codes]

        frames: list[pl.DataFrame] = []
        for path in glob.iglob(self.source_glob):
            try:
                scan = (
                    pl.scan_parquet(path)
                    .filter(
                        (pl.col(self.code_column).cast(pl.Utf8).is_in(codes_str))
                        & (pl.col(self.date_column) >= pl.lit(start_date))
                        & (pl.col(self.date_column) <= pl.lit(target_date))
                    )
                    .collect(streaming=True)
                )
                if scan.height > 0:
                    frames.append(scan)
            except Exception as exc:
                logger.debug("GraphBuilder source load failed for %s: %s", path, exc)
        if not frames:
            logger.debug(
                "No source data collected for %s in range [%s, %s]",
                target_date,
                start_date,
                target_date,
            )
            return None
        return pl.concat(frames, how="vertical")

    def build_for_day(
        self, date_end: str | _date_cls | _datetime_cls, codes: Sequence[str]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build graph using FinancialGraphBuilder for a specific day."""
        if self._financial_builder is None or not self.source_glob:
            logger.debug("FinancialGraphBuilder unavailable; skipping build_for_day.")
            return None, None

        data = self._load_source_data(date_end, codes)
        if data is None:
            return None, None

        codes_str = [str(code) for code in codes]
        result: dict[str, Any] | None = None
        for return_col in self.return_columns:
            try:
                candidate = self._financial_builder.build_graph(
                    data, codes_str, date_end=date_end, return_column=return_col
                )
                result = candidate
                # Stop early if we obtained edges
                edge_index = candidate.get("edge_index")
                if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
                    break
            except Exception as exc:
                logger.debug(
                    "FinancialGraphBuilder failed for return_col=%s: %s",
                    return_col,
                    exc,
                )
                continue

        if result is None:
            return None, None

        edge_index = result.get("edge_index")
        edge_attr = result.get("edge_attr")
        if not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
            return None, None

        if isinstance(edge_attr, torch.Tensor):
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            if edge_attr.shape[1] < 3:
                pad_cols = 3 - edge_attr.shape[1]
                edge_attr = torch.cat(
                    [edge_attr, torch.zeros(edge_attr.size(0), pad_cols)], dim=1
                )
        else:
            edge_attr = torch.zeros(edge_index.size(1), 3, dtype=torch.float32)

        # Track metadata for staleness checks
        try:
            asof = result.get("date", date_end)
            self._last_asof_ts = _datetime_cls.combine(
                self._parse_date(asof), _datetime_cls.min.time()
            )
        except Exception:
            self._last_asof_ts = None
        self._last_result = result

        return edge_index, edge_attr

    def last_asof_ts(self) -> _datetime_cls | None:
        """Return the timestamp corresponding to the most recent financial graph build."""
        return self._last_asof_ts

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
