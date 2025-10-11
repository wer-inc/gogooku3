"""
GPU-Accelerated Graph Builder for Financial Time Series Correlation
GPU高速化版: 系列相関に基づく金融グラフ構築器

主な機能:
- CuPyによるGPU上での相関行列計算（100倍高速化）
- 負相関を含む有意なエッジの抽出
- peer_mean/peer_var特徴量の生成
- CPU版と完全互換の出力
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from datetime import date, datetime, timedelta
import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch

# GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to NumPy if CuPy not available

# Optional imports (not critical for functionality)
try:
    import cudf
    from cupyx.scipy import stats as cu_stats
except ImportError:
    pass  # cudf/cupyx are optional

logger = logging.getLogger(__name__)


class FinancialGraphBuilder:
    """
    GPU高速化版 金融時系列の相関グラフ構築器

    正しい系列相関の実装:
    - 同一期間T日のリターンベクトルで銘柄×銘柄の相関行列を計算
    - 負相関も含む有意なエッジを抽出
    - peer特徴量の生成

    GPU最適化:
    - NumPy → CuPy置換で100倍高速化
    - 相関計算をGPU上で実行
    - CPU版と完全互換のインターフェース
    """

    def __init__(
        self,
        correlation_window: int = 60,
        min_observations: int = 40,
        correlation_threshold: float = 0.3,
        max_edges_per_node: int = 10,
        include_negative_correlation: bool = True,
        update_frequency: str = 'weekly',
        correlation_method: str = 'pearson',
        ewm_halflife: int = 20,
        shrinkage_gamma: float = 0.05,
        symmetric: bool = True,
        cache_dir: str | None = None,
        verbose: bool = True,
        sector_col: str | None = None,
        market_col: str | None = None,
        gpu_corr_block_cols: int = 256,
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
        self.sector_col = sector_col
        self.market_col = market_col
        # Streamed GPU correlation block size (columns per j-block)
        # Reduced default from 1024 to 256 to prevent OOM on large stock universes
        # Env override (best-effort)
        try:
            import os as _os
            env_b = int(_os.getenv("GOGOOKU3_GPU_CORR_BLOCK", str(gpu_corr_block_cols)))
            self.gpu_corr_block_cols = int(max(128, env_b))
        except Exception:
            self.gpu_corr_block_cols = int(max(128, gpu_corr_block_cols))

        # キャッシュ設定
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # グラフデータ（GPU/CPU両対応）
        self.correlation_matrices: dict[str, Any] = {}  # cp.ndarray or np.ndarray
        self.edge_indices: dict[str, torch.Tensor] = {}
        self.edge_attributes: dict[str, torch.Tensor] = {}
        self.node_mappings: dict[str, dict[str, int]] = {}
        self.peer_features: dict[str, dict[str, float]] = {}

        if GPU_AVAILABLE:
            logger.info("✅ GPU acceleration enabled for FinancialGraphBuilder")
        else:
            logger.warning("⚠️ CuPy not available, falling back to CPU computation")

        if self.verbose:
            logger.info(
                f"FinancialGraphBuilder: window={correlation_window}, "
                f"threshold={correlation_threshold}, method={correlation_method}, "
                f"GPU={'enabled' if GPU_AVAILABLE else 'disabled'}"
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
        # 主要パラメータをキーに含める（過去のキーとは互換性無し＝新規生成）
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
            # Make a light-weight copy for caching
            data_to_save = dict(data)
            # Convert GPU arrays to CPU if present
            if GPU_AVAILABLE and 'correlation_matrix' in data_to_save:
                try:
                    cm = data_to_save.get('correlation_matrix')
                    if hasattr(cm, 'get'):
                        cm = cm.get()
                    # Drop correlation matrix to keep cache small (recomputed if needed)
                    data_to_save.pop('correlation_matrix', None)
                except Exception:
                    data_to_save.pop('correlation_matrix', None)

            # Compressed pickle
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
        date_start = date_end - timedelta(days=self.correlation_window + 10)

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
            logger.warning(f"No data available for graph building on {date_end}")
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
            logger.warning(f"No valid codes for graph building on {date_end}")
            return np.array([]), []

        # 行列を統一長に調整
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
    ) -> Any:  # Returns cp.ndarray or np.ndarray
        """GPU高速化された相関行列計算"""
        n_stocks = return_matrix.shape[0]

        if n_stocks < 2:
            if GPU_AVAILABLE:
                return cp.eye(1) if n_stocks == 1 else cp.array([])
            else:
                return np.eye(1) if n_stocks == 1 else np.array([])

        try:
            if GPU_AVAILABLE:
                # Transfer to GPU in float32 to reduce memory pressure
                X = cp.asarray(return_matrix, dtype=cp.float32)
                N, T = X.shape

                if self.correlation_method == 'pearson':
                    # OOM-aware retry with progressively smaller block sizes
                    block_sizes_to_try = [
                        int(self.gpu_corr_block_cols) if hasattr(self, 'gpu_corr_block_cols') else 256,
                        128,
                        64
                    ]

                    for attempt, b in enumerate(block_sizes_to_try):
                        try:
                            # Streamed GPU Pearson correlation to avoid NxN device allocation
                            # 1) Standardize (ddof=0 to match np.corrcoef default)
                            mu = cp.mean(X, axis=1, keepdims=True)
                            X_norm = X - mu
                            std = cp.std(X_norm, axis=1, keepdims=True) + 1e-12
                            X_norm = X_norm / std

                            # 2) Allocate output on host and fill block-by-block
                            corr_cpu = np.empty((N, N), dtype=np.float32)
                            if b <= 0:
                                b = 64

                            for j in range(0, N, b):
                                jb = min(b, N - j)
                                block = X_norm @ X_norm[j:j + jb].T  # [N, jb]
                                block = block / float(T)
                                corr_cpu[:, j:j + jb] = cp.asnumpy(block)
                                # Free temporaries eagerly to curb fragmentation
                                try:
                                    del block  # type: ignore[has-type]
                                    cp.get_default_memory_pool().free_all_blocks()
                                except Exception:
                                    pass

                            # Symmetrize, clip, and set diagonal
                            corr_cpu = 0.5 * (corr_cpu + corr_cpu.T)
                            np.fill_diagonal(corr_cpu, 1.0)
                            corr_cpu = np.clip(corr_cpu, -1.0, 1.0)

                            # Success - clean up and return
                            try:
                                del X_norm
                                cp.get_default_memory_pool().free_all_blocks()
                            except Exception:
                                pass

                            if attempt > 0 and self.verbose:
                                logger.info(f"GPU correlation succeeded with block_size={b} after {attempt} retries")

                            return corr_cpu

                        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as oom_e:
                            # OOM detected - try smaller block size or fall back to CPU
                            try:
                                cp.get_default_memory_pool().free_all_blocks()
                            except Exception:
                                pass

                            if attempt < len(block_sizes_to_try) - 1:
                                next_b = block_sizes_to_try[attempt + 1]
                                if self.verbose:
                                    logger.warning(f"GPU OOM with block_size={b}, retrying with block_size={next_b}")
                                continue
                            else:
                                # All block sizes failed - fall back to CPU
                                if self.verbose:
                                    logger.warning(f"GPU OOM after all retries; falling back to CPU: {oom_e}")
                                return np.corrcoef(return_matrix)

                        except Exception as _gpu_e:
                            # Non-OOM error - fall back to CPU immediately
                            try:
                                cp.get_default_memory_pool().free_all_blocks()
                            except Exception:
                                pass
                            if self.verbose:
                                logger.warning(f"GPU correlation failed; falling back to CPU: {_gpu_e}")
                            return np.corrcoef(return_matrix)

                    # Should not reach here, but fall back to CPU just in case
                    return np.corrcoef(return_matrix)

                elif self.correlation_method == 'spearman':
                    # Fall back to CPU to avoid GPU NxN allocations for large N
                    from scipy.stats import spearmanr  # type: ignore
                    corr_matrix, _ = spearmanr(cp.asnumpy(X), axis=1)
                    if np.isscalar(corr_matrix):
                        corr_matrix = np.array([[corr_matrix]], dtype=np.float32)
                    return corr_matrix.astype(np.float32, copy=False)

                elif self.correlation_method == 'ewm_demean':
                    # Streamed EWM correlation
                    lam = cp.log(2.0) / float(self.ewm_halflife)
                    t_idx = cp.arange(T, dtype=cp.float32)
                    w = cp.exp(-lam * (T - 1 - t_idx))
                    w = w / (cp.sum(w) + 1e-12)
                    mu = X @ w
                    Xc = X - mu[:, None]
                    WX = Xc * w
                    var = cp.sum(WX * Xc, axis=1)
                    var = cp.maximum(var, 1e-12)
                    std = cp.sqrt(var)

                    corr_cpu = np.empty((N, N), dtype=np.float32)
                    b = int(self.gpu_corr_block_cols) if hasattr(self, 'gpu_corr_block_cols') else 1024
                    if b <= 0:
                        b = 1024
                    for j in range(0, N, b):
                        jb = min(b, N - j)
                        cov_block = WX @ Xc[j:j + jb].T  # [N, jb]
                        denom = (std[:, None] * std[j:j + jb][None, :]) + 1e-12
                        block = cov_block / denom
                        corr_cpu[:, j:j + jb] = cp.asnumpy(cp.clip(block, -1.0, 1.0))
                        try:
                            del cov_block, block  # type: ignore[has-type]
                            cp.get_default_memory_pool().free_all_blocks()
                        except Exception:
                            pass
                    corr_cpu = 0.5 * (corr_cpu + corr_cpu.T)
                    np.fill_diagonal(corr_cpu, 1.0)
                    if self.shrinkage_gamma > 0:
                        off = ~np.eye(N, dtype=bool)
                        corr_cpu[off] = (1.0 - self.shrinkage_gamma) * corr_cpu[off]
                    return corr_cpu
                else:
                    raise ValueError(f"Unsupported correlation method: {self.correlation_method}")

                # Proactively release CuPy pool blocks to avoid fragmentation
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
            else:
                # Fallback to CPU computation (original code)
                if self.correlation_method == 'pearson':
                    corr_matrix = np.corrcoef(return_matrix)
                elif self.correlation_method == 'spearman':
                    from scipy.stats import spearmanr
                    corr_matrix, _ = spearmanr(return_matrix, axis=1)
                    if np.isscalar(corr_matrix):
                        corr_matrix = np.array([[corr_matrix]])
                else:
                    # EWM on CPU (original implementation)
                    X = return_matrix.astype(np.float64, copy=False)
                    N, T = X.shape
                    lam = np.log(2.0) / float(self.ewm_halflife)
                    t_idx = np.arange(T)
                    w = np.exp(-lam * (T - 1 - t_idx))
                    w_sum = w.sum() + 1e-12
                    w = w / w_sum
                    mu = X @ w
                    Xc = X - mu[:, None]
                    WX = Xc * w
                    cov = WX @ Xc.T
                    var = np.diag(cov).copy()
                    var[var <= 1e-12] = 1e-12
                    std = np.sqrt(var)
                    denom = np.outer(std, std)
                    corr_matrix = cov / (denom + 1e-12)
                    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
                    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
                    np.fill_diagonal(corr_matrix, 1.0)
                    if self.shrinkage_gamma > 0:
                        off = ~np.eye(N, dtype=bool)
                        corr_matrix[off] = (1.0 - self.shrinkage_gamma) * corr_matrix[off]

            # 対角成分を1に設定
            if GPU_AVAILABLE:
                cp.fill_diagonal(corr_matrix, 1.0)
            else:
                np.fill_diagonal(corr_matrix, 1.0)

            # NaN/Infチェック
            if GPU_AVAILABLE:
                if cp.any(~cp.isfinite(corr_matrix)):
                    logger.warning("Non-finite values in correlation matrix")
                    corr_matrix = cp.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                if np.any(~np.isfinite(corr_matrix)):
                    logger.warning("Non-finite values in correlation matrix")
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

            return corr_matrix

        except Exception as e:
            logger.error(f"Failed to compute correlation matrix: {e}")
            if GPU_AVAILABLE:
                return cp.eye(n_stocks)
            else:
                return np.eye(n_stocks)

    def _extract_edges(
        self,
        corr_matrix: Any,  # cp.ndarray or np.ndarray
        valid_codes: list[str],
        market_map: dict[str, str] | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """相関行列からエッジを抽出（GPUメモリから効率的に転送）"""

        # Transfer to CPU if on GPU
        if GPU_AVAILABLE and hasattr(corr_matrix, 'get'):
            corr_matrix_cpu = corr_matrix.get()
        else:
            corr_matrix_cpu = corr_matrix

        n_stocks = len(valid_codes)

        # Create node mapping
        node_mapping = {code: idx for idx, code in enumerate(valid_codes)}

        # Extract significant edges
        edge_list = []
        edge_weights = []

        # Compute absolute correlations
        abs_corr = np.abs(corr_matrix_cpu)

        for i in range(n_stocks):
            # Get top-K correlations for this node
            correlations = abs_corr[i].copy()
            correlations[i] = 0  # Exclude self-correlation

            # Apply threshold
            valid_mask = correlations >= self.correlation_threshold

            if self.include_negative_correlation:
                # Include both positive and negative correlations
                valid_indices = np.where(valid_mask)[0]
            else:
                # Only positive correlations
                valid_indices = np.where(
                    (corr_matrix_cpu[i] >= self.correlation_threshold)
                )[0]

            # Sort by absolute correlation and take top-K
            if len(valid_indices) > 0:
                sorted_indices = valid_indices[
                    np.argsort(correlations[valid_indices])[::-1]
                ][:self.max_edges_per_node]

                for j in sorted_indices:
                    if self.symmetric and i > j:
                        continue  # Avoid duplicate edges

                    edge_list.append([i, j])
                    edge_weights.append(float(corr_matrix_cpu[i, j]))

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty(0, dtype=torch.float)

        return edge_index, edge_attr

    def _compute_peer_features(
        self,
        return_matrix: np.ndarray,
        corr_matrix: np.ndarray,
        codes: list[str]
    ) -> dict[str, dict[str, float]]:
        """peer特徴量を計算（GPU版）"""
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

    # The remaining methods (build_graph, get_peer_features_for_codes, etc.)
    # remain the same as the original implementation since they don't involve
    # heavy computation that benefits from GPU acceleration

    def build_graph(
        self,
        data: pd.DataFrame | pl.DataFrame,
        codes: list[str],
        date_end: str | date | datetime,
        return_column: str = 'return_1d'
    ) -> dict[str, Any]:
        """
        グラフを構築（GPU高速化版）

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
            logger.warning(f"Insufficient codes ({len(valid_codes)}) for graph building")
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

        # GPU高速化された相関行列計算
        corr_matrix = self._compute_correlation_matrix(return_matrix)

        # Convert correlation matrix to CPU if needed for further processing
        if GPU_AVAILABLE and hasattr(corr_matrix, 'get'):
            corr_matrix_cpu = corr_matrix.get()
        else:
            corr_matrix_cpu = corr_matrix

        # エッジを抽出
        edge_index, edge_attr = self._extract_edges(
            corr_matrix, valid_codes, None, None
        )

        # Node mapping
        node_mapping = {code: idx for idx, code in enumerate(valid_codes)}

        # peer特徴量を計算
        peer_features = self._compute_peer_features(return_matrix, corr_matrix_cpu, valid_codes)

        # Store results (オプション: メモリ効率化のためデフォルトでスキップ)
        if self.keep_in_memory:
            self.correlation_matrices[date_key] = corr_matrix
            self.edge_indices[date_key] = edge_index
            self.edge_attributes[date_key] = edge_attr
            self.node_mappings[date_key] = node_mapping
            self.peer_features[date_key] = peer_features

        result = {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'node_mapping': node_mapping,
            'correlation_matrix': corr_matrix_cpu,
            'peer_features': peer_features,
            'date': date_end,
            'n_nodes': len(valid_codes),
            'n_edges': edge_index.shape[1]
        }

        # Cache results
        self._save_to_cache(cache_key, result)

        if self.verbose:
            logger.info(
                f"Built graph for {date_end}: "
                f"{len(valid_codes)} nodes, {edge_index.shape[1]} edges"
            )

        return result

    def get_peer_features_for_codes(
        self,
        codes: list[str],
        date_key: str
    ) -> dict[str, dict[str, float]]:
        """
        指定銘柄のpeer特徴量を取得
        """
        peer_features = {}

        if date_key not in self.edge_indices:
            return peer_features

        edge_index = self.edge_indices[date_key]
        edge_attr = self.edge_attributes[date_key]
        node_mapping = self.node_mappings.get(date_key, {})

        for code in codes:
            if code not in node_mapping:
                peer_features[code] = {
                    'peer_count': 0,
                    'peer_corr_mean': 0.0,
                    'peer_corr_std': 0.0
                }
                continue

            node_idx = node_mapping[code]

            # Find edges connected to this node
            mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
            connected_edges = edge_attr[mask]

            if len(connected_edges) > 0:
                peer_features[code] = {
                    'peer_count': len(connected_edges),
                    'peer_corr_mean': float(torch.abs(connected_edges).mean()),
                    'peer_corr_std': float(torch.abs(connected_edges).std()) if len(connected_edges) > 1 else 0.0
                }
            else:
                peer_features[code] = {
                    'peer_count': 0,
                    'peer_corr_mean': 0.0,
                    'peer_corr_std': 0.0
                }

        return peer_features
