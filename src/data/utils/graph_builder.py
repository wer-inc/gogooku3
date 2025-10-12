"""
Graph Builder for Financial Time Series Correlation
系列相関に基づく金融グラフ構築器

主な機能:
- N×T行列からN×N相関行列の計算
- 負相関を含む有意なエッジの抽出
- 週次更新による効率化
- peer_mean/peer_var特徴量の生成
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from datetime import date, datetime, timedelta
import gzip
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import torch

logger = logging.getLogger(__name__)


class FinancialGraphBuilder:
    """
    金融時系列の相関グラフ構築器
    
    正しい系列相関の実装:
    - 同一期間T日のリターンベクトルで銘柄×銘柄の相関行列を計算
    - 負相関も含む有意なエッジを抽出
    - peer特徴量の生成
    """

    def __init__(
        self,
        correlation_window: int = 60,
        min_observations: int = 40,
        correlation_threshold: float = 0.3,
        max_edges_per_node: int = 10,
        include_negative_correlation: bool = True,
        update_frequency: str = 'weekly',
        correlation_method: str = 'pearson',  # 'pearson' | 'spearman' | 'ewm_demean'
        ewm_halflife: int = 20,
        shrinkage_gamma: float = 0.05,
        symmetric: bool = True,
        cache_dir: str | None = None,
        verbose: bool = True,
        sector_col: str | None = None,
        market_col: str | None = None,
        keep_in_memory: bool = False,  # メモリ効率化: デフォルトでFalse
    ):
        """
        Args:
            correlation_window: 相関計算の窓サイズ（日数）
            min_observations: 最小観測数
            correlation_threshold: 相関閾値（絶対値）
            max_edges_per_node: ノードあたり最大エッジ数
            include_negative_correlation: 負相関を含むか
            update_frequency: 更新頻度（'daily', 'weekly', 'monthly'）
            correlation_method: 相関計算方法（'pearson', 'spearman'）
            cache_dir: キャッシュディレクトリ
            verbose: 詳細ログ出力
            keep_in_memory: 相関行列をメモリに保持（デフォルト: False、メモリ節約）
        """
        self.correlation_window = correlation_window
        self.min_observations = min_observations
        self.correlation_threshold = correlation_threshold
        self.max_edges_per_node = max_edges_per_node
        self.include_negative_correlation = include_negative_correlation
        self.update_frequency = update_frequency
        self.correlation_method = correlation_method
        self.ewm_halflife = int(max(1, ewm_halflife))
        self.shrinkage_gamma = float(max(0.0, min(1.0, shrinkage_gamma)))
        self.symmetric = bool(symmetric)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.verbose = verbose
        self.keep_in_memory = keep_in_memory  # メモリ効率化フラグ
        # Column preferences (best-effort; falls back to common aliases)
        self.sector_col = sector_col
        self.market_col = market_col

        # キャッシュ設定
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # グラフデータ
        self.correlation_matrices: dict[str, np.ndarray] = {}
        self.edge_indices: dict[str, torch.Tensor] = {}
        self.edge_attributes: dict[str, torch.Tensor] = {}
        self.node_mappings: dict[str, dict[str, int]] = {}
        self.peer_features: dict[str, dict[str, float]] = {}

        if self.verbose:
            logger.info(
                f"FinancialGraphBuilder: window={correlation_window}, "
                f"threshold={correlation_threshold}, method={correlation_method}, "
                f"ewm_halflife={self.ewm_halflife}, shrinkage_gamma={self.shrinkage_gamma}, symmetric={self.symmetric}"
            )

    def _get_cache_key(self, date: date, codes: list[str]) -> str:
        """キャッシュキーを生成（決定的ハッシュで衝突を回避）"""
        # 決定的なダイジェスト（プロセス間で再現可能）
        codes_str = "|".join(sorted(codes))
        code_digest = hashlib.blake2s(
            codes_str.encode('utf-8'),
            digest_size=8
        ).hexdigest()

        # 閾値は2桁精度で表現（例: 0.3 -> 030）
        try:
            thr100 = int(round(float(self.correlation_threshold) * 100))
        except Exception:
            thr100 = 0

        # 主要パラメータをキーに含める（GPU版と一貫性を保つ）
        parts = [
            "graph",
            str(date),
            code_digest,  # 決定的なハッシュ
            f"w{int(self.correlation_window)}",
            f"t{thr100:03d}",
            f"k{int(self.max_edges_per_node)}",
            f"m{str(self.correlation_method)}",
            f"freq-{getattr(self, 'update_frequency', 'daily')}",
            f"neg-{int(bool(getattr(self, 'include_negative_correlation', True)))}",
            f"sym-{int(bool(getattr(self, 'symmetric', True)))}",
        ]
        return "_".join(parts)

    def _load_from_cache(self, cache_key: str) -> dict[str, Any] | None:
        """キャッシュから読み込み"""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with gzip.open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                if self.verbose:
                    logger.info(f"✅ Cache HIT: {cache_file.name}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, data: dict[str, Any]):
        """キャッシュに保存"""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            # Drop huge correlation matrix to reduce cache size
            data_to_save = dict(data)
            if 'correlation_matrix' in data_to_save:
                data_to_save.pop('correlation_matrix', None)

            # Compressed pickle (~3-10x smaller depending on payload)
            with gzip.open(cache_file, 'wb', compresslevel=3) as f:
                pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            if self.verbose:
                logger.info(f"💾 Cache SAVE: {cache_file.name}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_file}: {e}")

    def _prepare_return_matrix(
        self,
        data: pd.DataFrame | pl.DataFrame,
        codes: list[str],
        date_end: date,
        return_column: str = 'return_1d'
    ) -> tuple[np.ndarray, list[str]]:
        """
        リターン行列を準備（N×T形式）
        
        Returns:
            (return_matrix, valid_codes): リターン行列と有効銘柄コード
        """
        # Polarsに変換
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)

        # 日付範囲を計算
        date_start = date_end - timedelta(days=self.correlation_window + 10)  # バッファ含む

        # 型正規化（Date vs Datetimeの不一致を避ける）
        try:
            if data.schema.get('date', None) is not None and data['date'].dtype != pl.Date:
                data = data.with_columns(pl.col('date').cast(pl.Date, strict=False))
        except Exception:
            pass

        # データフィルタ
        filtered_data = data.filter(
            (pl.col('date') >= date_start) &
            (pl.col('date') <= date_end) &
            (pl.col('code').is_in(codes)) &
            (pl.col(return_column).is_not_null()) &
            (pl.col(return_column).is_finite())
        ).sort(['code', 'date'])

        if filtered_data.is_empty():
            logger.debug(f"No data available for graph building on {date_end}")
            return np.array([]), []

        # Early check: sufficient historical data available?
        unique_dates = filtered_data.select('date').unique().height
        if unique_dates < self.min_observations:
            if self.verbose:
                logger.debug(
                    f"Skipping {date_end}: only {unique_dates} days available "
                    f"(need {self.min_observations})"
                )
            return np.array([]), []

        # 銘柄×日付のリターン行列を構築
        return_matrix = []
        valid_codes = []

        for code in codes:
            code_data = filtered_data.filter(pl.col('code') == code)

            if len(code_data) < self.min_observations:
                continue

            # 最新のcorrelation_window日分を取得
            recent_returns = code_data.tail(self.correlation_window)[return_column].to_numpy()

            # 欠損値チェック
            if len(recent_returns) < self.min_observations or not np.all(np.isfinite(recent_returns)):
                continue

            return_matrix.append(recent_returns)
            valid_codes.append(code)

        if not return_matrix:
            logger.debug(f"No valid codes for graph building on {date_end}")
            return np.array([]), []

        # 行列を統一長に調整（最短長に合わせる）
        min_length = min(len(arr) for arr in return_matrix)
        return_matrix = np.array([arr[-min_length:] for arr in return_matrix])

        if self.verbose:
            logger.debug(
                f"Prepared return matrix: {len(valid_codes)} codes × {min_length} days"
            )

        return return_matrix, valid_codes

    def _compute_correlation_matrix(
        self,
        return_matrix: np.ndarray
    ) -> np.ndarray:
        """相関行列を計算（GPU最適化あり）"""
        n_stocks = return_matrix.shape[0]

        if n_stocks < 2:
            return np.eye(1) if n_stocks == 1 else np.array([])

        try:
            if self.correlation_method == 'pearson':
                # GPU最適化: CuPyが利用可能であればGPUで相関行列を計算
                use_gpu = False
                try:
                    import cupy as cp  # type: ignore
                    if cp.cuda.runtime.getDeviceCount() > 0:
                        use_gpu = True
                except Exception:
                    use_gpu = False

                if use_gpu:
                    Xg = None
                    try:
                        Xg = cp.asarray(return_matrix, dtype=cp.float64)
                        corr_g = cp.corrcoef(Xg)
                        corr_matrix = cp.asnumpy(corr_g)
                    finally:
                        # 明示的にGPUメモリを解放
                        try:
                            del Xg, corr_g  # type: ignore[name-defined]
                        except Exception:
                            pass
                else:
                    # CPUフォールバック
                    corr_matrix = np.corrcoef(return_matrix)

            elif self.correlation_method == 'spearman':
                # Spearman相関（ランク相関）
                from scipy.stats import spearmanr
                corr_matrix, _ = spearmanr(return_matrix, axis=1)
                if np.isscalar(corr_matrix):
                    corr_matrix = np.array([[corr_matrix]])

            elif self.correlation_method == 'ewm_demean':
                # Exponentially-weighted correlation with per-asset de-meaning（CPU）
                X = return_matrix.astype(np.float64, copy=False)  # [N,T]
                N, T = X.shape
                lam = np.log(2.0) / float(self.ewm_halflife)
                t_idx = np.arange(T)
                w = np.exp(-lam * (T - 1 - t_idx))
                w_sum = w.sum() + 1e-12
                w = w / w_sum
                mu = X @ w  # [N]
                Xc = X - mu[:, None]
                WX = Xc * w  # broadcast weights
                cov = WX @ Xc.T  # weighted covariance (already divided by sum via normalized w)
                var = np.diag(cov).copy()
                var[var <= 1e-12] = 1e-12
                std = np.sqrt(var)
                denom = np.outer(std, std)
                corr_matrix = cov / (denom + 1e-12)
                corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
                # Symmetrize + diag=1
                corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
                np.fill_diagonal(corr_matrix, 1.0)
                # Shrinkage toward zero on off-diagonals
                if self.shrinkage_gamma > 0:
                    off = ~np.eye(N, dtype=bool)
                    corr_matrix[off] = (1.0 - self.shrinkage_gamma) * corr_matrix[off]
            else:
                raise ValueError(f"Unsupported correlation method: {self.correlation_method}")

            # NaN/inf処理
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

            # 対角成分を1.0に設定
            np.fill_diagonal(corr_matrix, 1.0)

            return corr_matrix

        except Exception as e:
            logger.warning(f"Correlation computation failed: {e}")
            return np.eye(n_stocks)

    def _extract_edges(
        self,
        corr_matrix: np.ndarray,
        codes: list[str],
        market_map: dict[str, Any] | None = None,
        sector_map: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        相関行列からエッジを抽出
        
        Returns:
            (edge_index, edge_attr): エッジインデックスとエッジ属性
        """
        n_stocks = len(codes)
        if n_stocks < 2:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)

        edges = []
        edge_attrs: list[list[float]] = []

        # 各ノードから上位k個のエッジを抽出
        for i in range(n_stocks):
            # 自分以外の相関を取得
            correlations = corr_matrix[i, :]

            # 閾値でフィルタ
            if self.include_negative_correlation:
                valid_indices = np.where(np.abs(correlations) >= self.correlation_threshold)[0]
                valid_correlations = correlations[valid_indices]
            else:
                valid_indices = np.where(correlations >= self.correlation_threshold)[0]
                valid_correlations = correlations[valid_indices]

            # 自ループを除外
            mask = valid_indices != i
            valid_indices = valid_indices[mask]
            valid_correlations = valid_correlations[mask]

            if len(valid_indices) == 0:
                continue

            # 相関の絶対値で並び替え（強い相関を優先）
            abs_correlations = np.abs(valid_correlations)
            sorted_indices = np.argsort(abs_correlations)[::-1]

            # 上位k個を選択
            top_k = min(self.max_edges_per_node, len(sorted_indices))
            selected_indices = sorted_indices[:top_k]

            for idx in selected_indices:
                j = valid_indices[idx]
                correlation = float(valid_correlations[idx])
                corr_norm = (correlation + 1.0) / 2.0
                # 2) market similarity (1 if same, else 0)
                m_sim = 0.0
                if market_map is not None:
                    mi = market_map.get(codes[i])
                    mj = market_map.get(codes[j])
                    if mi is not None and mj is not None and mi == mj:
                        m_sim = 1.0
                # 3) sector similarity (1 if same, else 0)
                s_sim = 0.0
                if sector_map is not None:
                    si = sector_map.get(codes[i])
                    sj = sector_map.get(codes[j])
                    if si is not None and sj is not None and si == sj:
                        s_sim = 1.0

                edges.append([i, j])
                edge_attrs.append([corr_norm, m_sim, s_sim])
                if self.symmetric:
                    edges.append([j, i])
                    edge_attrs.append([corr_norm, m_sim, s_sim])

        if not edges:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 3), dtype=torch.float),
            )

        # テンソルに変換
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        if self.verbose:
            logger.debug(f"Extracted {len(edges)} edges from {n_stocks} nodes")

        return edge_index, edge_attr

    def _compute_peer_features(
        self,
        return_matrix: np.ndarray,
        corr_matrix: np.ndarray,
        codes: list[str]
    ) -> dict[str, dict[str, float]]:
        """peer特徴量を計算"""
        peer_features = {}

        for i, code in enumerate(codes):
            # 現在の銘柄の相関をベースに近傍を決定
            correlations = corr_matrix[i, :]

            # 閾値を満たすpeerを特定
            if self.include_negative_correlation:
                peer_mask = (np.abs(correlations) >= self.correlation_threshold) & (np.arange(len(correlations)) != i)
            else:
                peer_mask = (correlations >= self.correlation_threshold) & (np.arange(len(correlations)) != i)

            peer_indices = np.where(peer_mask)[0]

            if len(peer_indices) == 0:
                # peer がいない場合
                peer_features[code] = {
                    'peer_mean_return': 0.0,
                    'peer_var_return': 1.0,
                    'peer_count': 0,
                    'peer_correlation_mean': 0.0
                }
                continue

            # peer のリターン統計
            peer_returns = return_matrix[peer_indices, :]  # shape: (n_peers, T)
            peer_mean_return = np.mean(peer_returns.flatten())
            peer_var_return = np.var(peer_returns.flatten())

            # peer との平均相関
            peer_correlations = correlations[peer_indices]
            peer_correlation_mean = np.mean(np.abs(peer_correlations))

            peer_features[code] = {
                'peer_mean_return': float(peer_mean_return),
                'peer_var_return': float(peer_var_return),
                'peer_count': len(peer_indices),
                'peer_correlation_mean': float(peer_correlation_mean)
            }

        return peer_features

    def build_graph(
        self,
        data: pd.DataFrame | pl.DataFrame,
        codes: list[str],
        date_end: str | date | datetime,
        return_column: str = 'return_1d'
    ) -> dict[str, Any]:
        """
        グラフを構築
        
        Args:
            data: 時系列データ
            codes: 銘柄コードリスト
            date_end: 基準日
            return_column: リターン列名
            
        Returns:
            グラフ情報辞書
        """
        # 日付変換
        if isinstance(date_end, str):
            date_end = datetime.strptime(date_end, '%Y-%m-%d').date()
        elif isinstance(date_end, datetime):
            date_end = date_end.date()

        date_key = str(date_end)

        # キャッシュチェック
        cache_key = self._get_cache_key(date_end, codes)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            if self.verbose:
                logger.debug(f"Loaded graph from cache for {date_end}")
            return cached_result

        # リターン行列を準備
        return_matrix, valid_codes = self._prepare_return_matrix(
            data, codes, date_end, return_column
        )

        if len(valid_codes) < 2:
            logger.debug(f"Insufficient codes ({len(valid_codes)}) for graph building")
            empty_result = {
                'edge_index': torch.empty((2, 0), dtype=torch.long),
                'edge_attr': torch.empty(0, dtype=torch.float),
                'node_mapping': {},
                'correlation_matrix': np.array([]),
                'peer_features': {},
                'date': date_end,
                'n_nodes': 0,
                'n_edges': 0
            }
            return empty_result

        # 相関行列を計算
        corr_matrix = self._compute_correlation_matrix(return_matrix)

        # 市場/セクタマップを準備（可能なら）
        market_col = self.market_col
        sector_col = self.sector_col
        try:
            # Determine available columns (lower-case for robustness)
            candidate_market_cols = []
            if isinstance(market_col, str) and market_col:
                candidate_market_cols.append(market_col.lower())
            candidate_market_cols += ["marketcode", "market_code", "market", "section", "meta_section"]
            candidate_sector_cols = []
            if isinstance(sector_col, str) and sector_col:
                candidate_sector_cols.append(sector_col.lower())
            candidate_sector_cols += ["sector33", "sectorcode", "sector", "meta_section", "section"]
            # Convert to pandas only if needed (メモリ最適化)
            if isinstance(data, pd.DataFrame):
                _df = data
            else:
                # Polarsの場合: valid_codesのみを選択してからPandasに変換
                _df = data.filter(pl.col('code').is_in(valid_codes)).to_pandas()
            try:
                _df.columns = [str(c).lower() for c in _df.columns]
            except Exception:
                pass
            # Filter to valid codes only; prefer latest date rows within window buffer
            _subset = _df[_df["code"].isin(valid_codes)].copy()
            # Try to pick latest row per code
            if "date" in _subset.columns:
                _subset = _subset.sort_values(["code", "date"]).drop_duplicates("code", keep="last")
            market_map = None
            sector_map = None
            for col in candidate_market_cols:
                if col in _subset.columns:
                    market_map = {str(r.code): r[col] for r in _subset[["code", col]].itertuples(index=False)}
                    break
            for col in candidate_sector_cols:
                if col in _subset.columns:
                    sector_map = {str(r.code): r[col] for r in _subset[["code", col]].itertuples(index=False)}
                    break
        except Exception as _e:
            market_map = None
            sector_map = None
            if self.verbose:
                logger.warning(f"Failed to prepare market/sector maps: {_e}")

        # エッジを抽出（相関 + 市場/セクタ類似）
        edge_index, edge_attr = self._extract_edges(
            corr_matrix, valid_codes, market_map=market_map, sector_map=sector_map
        )

        # ノードマッピング
        node_mapping = {code: i for i, code in enumerate(valid_codes)}

        # peer特徴量を計算
        peer_features = self._compute_peer_features(return_matrix, corr_matrix, valid_codes)

        # 結果をまとめる
        result = {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'node_mapping': node_mapping,
            'correlation_matrix': corr_matrix,
            'peer_features': peer_features,
            'date': date_end,
            'n_nodes': len(valid_codes),
            'n_edges': edge_index.size(1)
        }

        # キャッシュに保存
        self._save_to_cache(cache_key, result)

        # 内部状態を更新（オプション: メモリ効率化のためデフォルトでスキップ）
        if self.keep_in_memory:
            self.correlation_matrices[date_key] = corr_matrix
            self.edge_indices[date_key] = edge_index
            self.edge_attributes[date_key] = edge_attr
            self.node_mappings[date_key] = node_mapping
            self.peer_features[date_key] = peer_features

        if self.verbose:
            logger.info(
                f"Built graph for {date_end}: {len(valid_codes)} nodes, "
                f"{edge_index.size(1)} edges"
            )

        return result

    def get_peer_features_for_codes(
        self,
        codes: list[str],
        date: str | date | datetime
    ) -> dict[str, dict[str, float]]:
        """指定銘柄のpeer特徴量を取得"""
        date_key = str(date)

        if date_key not in self.peer_features:
            logger.warning(f"No peer features available for {date}")
            return {code: {'peer_mean_return': 0.0, 'peer_var_return': 1.0,
                          'peer_count': 0, 'peer_correlation_mean': 0.0} for code in codes}

        peer_data = self.peer_features[date_key]
        return {code: peer_data.get(code, {'peer_mean_return': 0.0, 'peer_var_return': 1.0,
                                          'peer_count': 0, 'peer_correlation_mean': 0.0})
                for code in codes}

    def analyze_graph_statistics(
        self,
        date: str | date | datetime
    ) -> dict[str, Any]:
        """グラフの統計情報を分析"""
        date_key = str(date)

        if date_key not in self.edge_indices:
            return {}

        edge_index = self.edge_indices[date_key]
        edge_attr = self.edge_attributes[date_key]
        corr_matrix = self.correlation_matrices[date_key]

        # 基本統計
        n_nodes = corr_matrix.shape[0] if corr_matrix.size > 0 else 0
        n_edges = edge_index.size(1)

        # 相関統計
        if corr_matrix.size > 0:
            # 上三角のみ（対角除外）
            triu_indices = np.triu_indices(n_nodes, k=1)
            correlations = corr_matrix[triu_indices]

            correlation_stats = {
                'mean_correlation': float(np.mean(correlations)),
                'std_correlation': float(np.std(correlations)),
                'min_correlation': float(np.min(correlations)),
                'max_correlation': float(np.max(correlations)),
                'median_correlation': float(np.median(correlations))
            }
        else:
            correlation_stats = {}

        # エッジ統計
        if n_edges > 0:
            edge_stats = {
                'mean_edge_weight': float(torch.mean(edge_attr)),
                'std_edge_weight': float(torch.std(edge_attr)),
                'min_edge_weight': float(torch.min(edge_attr)),
                'max_edge_weight': float(torch.max(edge_attr))
            }

            # ネットワーク統計（NetworkX使用）
            try:
                G = nx.Graph()
                G.add_nodes_from(range(n_nodes))
                edges_list = edge_index.t().numpy().tolist()
                weights = edge_attr.numpy().tolist()
                weighted_edges = [(e[0], e[1], {'weight': abs(w)}) for e, w in zip(edges_list, weights, strict=False)]
                G.add_edges_from(weighted_edges)

                network_stats = {
                    'density': nx.density(G),
                    'average_clustering': nx.average_clustering(G),
                    'n_connected_components': nx.number_connected_components(G)
                }
            except Exception as e:
                logger.warning(f"Network analysis failed: {e}")
                network_stats = {}
        else:
            edge_stats = {}
            network_stats = {}

        return {
            'date': date,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'correlation_stats': correlation_stats,
            'edge_stats': edge_stats,
            'network_stats': network_stats
        }
