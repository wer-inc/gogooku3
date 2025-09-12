"""
Enhanced Collate Functions with Graph Snapshot Integration
グラフスナップショット統合による拡張collate関数
"""

from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def enhanced_financial_collate_fn(
    batch: List[Dict[str, Any]], 
    include_graph: bool = True,
    include_peer_features: bool = True,
    graph_snapshot_cache: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    金融時系列用の拡張collate関数
    
    機能:
    1. 基本的な特徴量とターゲットのスタック
    2. グラフスナップショット情報の統合
    3. Peer特徴量（近傍統計）の追加
    4. メタデータの保持（code, date, market）
    
    Args:
        batch: バッチデータ（DayBatchSamplerからの同日サンプル）
        include_graph: グラフ情報を含めるか
        include_peer_features: Peer特徴量を計算するか
        graph_snapshot_cache: グラフスナップショットのキャッシュ
        
    Returns:
        拡張されたバッチ辞書
    """
    if not batch:
        return {}

    # 基本collate処理
    result = _basic_collate(batch)
    
    # 同日データの検証
    first_date = batch[0].get('date')
    if first_date is not None:
        for item in batch[1:]:
            item_date = item.get('date')
            if item_date != first_date:
                logger.warning(f"Mixed dates in batch: {first_date} vs {item_date}")
                break
    
    # グラフ情報の追加
    if include_graph and first_date is not None:
        graph_info = _get_graph_snapshot(first_date, batch, graph_snapshot_cache)
        if graph_info:
            result.update(graph_info)
    
    # Peer特徴量の追加
    if include_peer_features:
        peer_features = _compute_peer_features(batch)
        if peer_features is not None:
            result['peer_features'] = peer_features
    
    # バッチ統計（デバッグ用）
    result['batch_stats'] = {
        'batch_size': len(batch),
        'date': first_date,
        'n_codes': len(set(item.get('code', '') for item in batch)),
        'has_graph': 'edge_index' in result,
        'has_peer': 'peer_features' in result
    }
    
    return result


def _basic_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """基本的なcollate処理"""
    # 特徴量をスタック
    if 'features' in batch[0]:
        features = torch.stack([item['features'] for item in batch], dim=0)
    elif 'dynamic_features' in batch[0]:
        features = torch.stack([item['dynamic_features'] for item in batch], dim=0)
    else:
        raise ValueError("No 'features' or 'dynamic_features' found in batch")
    
    # ターゲットを処理
    targets = {}
    if 'targets' in batch[0]:
        if isinstance(batch[0]['targets'], dict):
            # マルチホライズン辞書形式
            target_keys = batch[0]['targets'].keys()
            for key in target_keys:
                targets[key] = torch.stack([item['targets'][key] for item in batch], dim=0)
        else:
            # 単一ターゲット
            targets = torch.stack([item['targets'] for item in batch], dim=0)
    
    # Valid maskの処理（オプション）
    valid_masks = None
    if 'valid_mask' in batch[0] and batch[0]['valid_mask'] is not None:
        if isinstance(batch[0]['valid_mask'], dict):
            valid_masks = {}
            mask_keys = batch[0]['valid_mask'].keys()
            for key in mask_keys:
                valid_masks[key] = torch.stack([item['valid_mask'][key] for item in batch], dim=0)
        else:
            valid_masks = torch.stack([item['valid_mask'] for item in batch], dim=0)
    
    # メタデータ
    codes = [item.get('code', '') for item in batch]
    dates = [item.get('date') for item in batch]
    
    result = {
        'features': features,
        'targets': targets,
        'codes': codes,
        'dates': dates
    }
    
    if valid_masks is not None:
        result['valid_masks'] = valid_masks
    
    return result


def _get_graph_snapshot(
    date: Any, 
    batch: List[Dict[str, Any]], 
    graph_cache: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    指定日のグラフスナップショットを取得
    
    Args:
        date: 対象日付
        batch: バッチデータ
        graph_cache: グラフキャッシュ
        
    Returns:
        グラフ情報（edge_index, edge_attr等）
    """
    # バッチからグラフ情報を直接取得
    first_item = batch[0]
    if 'edge_index' in first_item:
        # 既にエッジ情報が含まれている場合
        edge_index = first_item['edge_index']
        edge_attr = first_item.get('edge_attr')
        
        result = {'edge_index': edge_index}
        if edge_attr is not None:
            result['edge_attr'] = edge_attr
        return result
    
    # キャッシュからグラフ情報を取得
    if graph_cache is not None:
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d') if date else None
        if date_str and date_str in graph_cache:
            cached_graph = graph_cache[date_str]
            
            result = {}
            if 'edge_index' in cached_graph:
                result['edge_index'] = cached_graph['edge_index']
            if 'edge_attr' in cached_graph:
                result['edge_attr'] = cached_graph['edge_attr']
            if 'node_mapping' in cached_graph:
                result['node_mapping'] = cached_graph['node_mapping']
                
            return result
    
    # グラフ情報がない場合は動的生成
    if len(batch) >= 10:  # 最低限の銘柄数
        return _build_dynamic_graph(batch)
    
    return None


def _build_dynamic_graph(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    バッチから動的グラフを構築
    
    Args:
        batch: バッチデータ
        
    Returns:
        グラフ情報
    """
    try:
        from ..graph.dynamic_knn import build_knn_from_embeddings
    except ImportError:
        # フォールバック: 基本的な相関グラフ
        return _build_correlation_graph(batch)
    
    # 特徴量からグラフを構築
    features = torch.stack([item['features'] for item in batch], dim=0)
    
    # 最新の時系列ポイントを使用（通常は特徴量の最後の時点）
    if features.dim() == 3:  # (B, L, F)
        node_embeddings = features[:, -1, :]  # 最新の時点を使用
    else:  # (B, F)
        node_embeddings = features
    
    # k-NNグラフ構築
    k = min(10, len(batch) - 1)  # バッチサイズに応じて調整
    edge_index, edge_attr = build_knn_from_embeddings(
        node_embeddings, 
        k=k,
        exclude_self=True,
        symmetric=True
    )
    
    return {
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'graph_type': 'dynamic_knn',
        'k': k
    }


def _build_correlation_graph(batch: List[Dict[str, Any]], threshold: float = 0.3) -> Dict[str, Any]:
    """
    フォールバック: 相関ベースの基本グラフ
    
    Args:
        batch: バッチデータ
        threshold: 相関閾値
        
    Returns:
        グラフ情報
    """
    # 特徴量から相関を計算
    features = torch.stack([item['features'] for item in batch], dim=0)
    
    if features.dim() == 3:  # (B, L, F)
        # 時系列の平均を使用
        node_features = features.mean(dim=1)  # (B, F)
    else:
        node_features = features
    
    # 相関行列計算
    corr_matrix = torch.corrcoef(node_features)
    
    # 閾値以上の相関でエッジを作成
    edge_indices = []
    edge_weights = []
    
    n_nodes = corr_matrix.size(0)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            corr_val = corr_matrix[i, j].item()
            if abs(corr_val) > threshold:
                edge_indices.append([i, j])
                edge_indices.append([j, i])  # 双方向
                edge_weights.extend([abs(corr_val), abs(corr_val)])
    
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
    else:
        # エッジがない場合は自己ループを作成
        edge_index = torch.arange(n_nodes).repeat(2, 1)
        edge_attr = torch.ones(n_nodes, 1)
    
    return {
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'graph_type': 'correlation',
        'threshold': threshold
    }


def _compute_peer_features(batch: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
    """
    Peer特徴量（近傍統計）を計算
    
    Args:
        batch: バッチデータ
        
    Returns:
        Peer特徴量テンソル (B, peer_features)
    """
    if len(batch) < 5:  # 最小銘柄数
        return None
    
    try:
        # 各サンプルの特徴量を取得
        features = torch.stack([item['features'] for item in batch], dim=0)
        
        if features.dim() == 3:  # (B, L, F)
            # 最新の時点を使用
            current_features = features[:, -1, :]  # (B, F)
        else:  # (B, F)
            current_features = features
        
        batch_size, n_features = current_features.shape
        
        # マーケット情報がある場合はマーケット別統計
        markets = [item.get('market_code', 'unknown') for item in batch]
        unique_markets = list(set(markets))
        
        peer_features_list = []
        
        for i, market in enumerate(markets):
            # 同じマーケットの他の銘柄を取得
            market_mask = torch.tensor([j != i and markets[j] == market for j in range(batch_size)])
            
            if market_mask.sum() > 0:
                # 同一マーケット統計
                market_features = current_features[market_mask]
                peer_mean = market_features.mean(dim=0)
                peer_std = market_features.std(dim=0) + 1e-8
            else:
                # 全体統計をフォールバック
                others_mask = torch.tensor([j != i for j in range(batch_size)])
                other_features = current_features[others_mask]
                peer_mean = other_features.mean(dim=0)
                peer_std = other_features.std(dim=0) + 1e-8
            
            # 現在の銘柄と近傍統計の差分
            current_feat = current_features[i]
            peer_diff = (current_feat - peer_mean) / peer_std
            
            # Peer特徴量: [mean, std, diff, rank_percentile]
            peer_feat = torch.cat([
                peer_mean,
                peer_std,
                peer_diff,
                torch.tensor([float(i) / batch_size])  # バッチ内順位
            ])
            
            peer_features_list.append(peer_feat)
        
        return torch.stack(peer_features_list, dim=0)
        
    except Exception as e:
        logger.warning(f"Failed to compute peer features: {e}")
        return None


class GraphSnapshotCache:
    """グラフスナップショットのキャッシュ管理"""
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_days: int = 30):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_cache_days = max_cache_days
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_graph(self, date: str) -> Optional[Dict[str, Any]]:
        """グラフスナップショットを取得"""
        # メモリキャッシュから確認
        if date in self.memory_cache:
            return self.memory_cache[date]
        
        # ディスクキャッシュから確認
        if self.cache_dir:
            cache_file = self.cache_dir / f"graph_{date}.pt"
            if cache_file.exists():
                try:
                    graph_data = torch.load(cache_file, map_location='cpu')
                    self.memory_cache[date] = graph_data
                    return graph_data
                except Exception as e:
                    logger.warning(f"Failed to load graph cache for {date}: {e}")
        
        return None
    
    def save_graph(self, date: str, graph_data: Dict[str, Any]):
        """グラフスナップショットを保存"""
        # メモリキャッシュに保存
        self.memory_cache[date] = graph_data
        
        # ディスクキャッシュに保存
        if self.cache_dir:
            cache_file = self.cache_dir / f"graph_{date}.pt"
            try:
                torch.save(graph_data, cache_file)
            except Exception as e:
                logger.warning(f"Failed to save graph cache for {date}: {e}")
        
        # 古いキャッシュをクリーンアップ
        self._cleanup_old_cache()
    
    def _cleanup_old_cache(self):
        """古いキャッシュをクリーンアップ"""
        if len(self.memory_cache) > self.max_cache_days:
            # 古い日付から削除
            sorted_dates = sorted(self.memory_cache.keys())
            to_remove = sorted_dates[:-self.max_cache_days]
            
            for date in to_remove:
                del self.memory_cache[date]


# Factory function
def create_collate_fn(
    include_graph: bool = True,
    include_peer_features: bool = True,
    graph_cache_dir: Optional[str] = None
) -> callable:
    """
    設定に基づいてcollate関数を作成
    
    Args:
        include_graph: グラフ情報を含めるか
        include_peer_features: Peer特徴量を含めるか
        graph_cache_dir: グラフキャッシュディレクトリ
        
    Returns:
        設定されたcollate関数
    """
    graph_cache = GraphSnapshotCache(graph_cache_dir) if graph_cache_dir else None
    
    def collate_fn(batch):
        return enhanced_financial_collate_fn(
            batch,
            include_graph=include_graph,
            include_peer_features=include_peer_features,
            graph_snapshot_cache=graph_cache.memory_cache if graph_cache else None
        )
    
    return collate_fn