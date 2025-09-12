"""
Walk-Forward Date Bucket Sampler with Embargo
安全なバッチ処理のためのWalk-Forward + Embargo統合サンプラー
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
from typing import Iterator, List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WalkForwardDateBucketSampler(Sampler):
    """
    Walk-Forward分割 + Embargo + Date-Bucketed Sampling
    
    データリーク防止の三重保護:
    1. Walk-Forward分割: 時系列順で train → val → test 分割
    2. Embargo: train/val境界に gap_days の空白期間を設定
    3. Date-Bucketed: 同日内のサンプルをまとめてバッチ化
    
    クロスセクション統計と完全に整合:
    - 各バッチは同一日のサンプルのみで構成
    - 日内でのみシャッフル（時系列順は保持）
    - 当日統計による正規化と相性◎
    """

    def __init__(
        self,
        dataset,
        n_splits: int = 5,
        embargo_days: int = 20,
        max_batch_size: int = 2048,
        min_nodes_per_day: int = 20,
        shuffle: bool = True,
        seed: int = 42,
        current_fold: int = 0,
        phase: str = 'train'
    ):
        """
        Args:
            dataset: データセット（日付情報を持つ）
            n_splits: Walk-Forward分割数
            embargo_days: train/val境界の空白期間（日数）
            max_batch_size: 最大バッチサイズ（超過時は分割）
            min_nodes_per_day: 最小日次ノード数（未満はスキップ）
            shuffle: 日内シャッフルの有無
            seed: 乱数シード
            current_fold: 現在のfold番号（0ベース）
            phase: 'train', 'val', 'test'
        """
        self.dataset = dataset
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.max_batch_size = max_batch_size
        self.min_nodes_per_day = min_nodes_per_day
        self.shuffle = shuffle
        self.seed = seed
        self.current_fold = current_fold
        self.phase = phase

        # Walk-Forward分割を構築
        self.fold_ranges = self._build_walk_forward_folds()
        
        # 現在のfold + phaseに対応するデータを取得
        self.valid_indices = self._get_phase_indices()
        
        # 日ごとにグループ化
        self.day_groups = self._group_by_day()

        # 統計情報をログ
        self._log_statistics()

    def _build_walk_forward_folds(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Walk-Forward分割を構築"""
        # データセットから全日付を取得
        all_dates = self._extract_all_dates()
        
        if len(all_dates) < self.n_splits:
            raise ValueError(f"Not enough dates ({len(all_dates)}) for {self.n_splits} splits")
            
        unique_dates = sorted(all_dates.unique())
        n = len(unique_dates)
        
        fold_ranges = []
        fold_size = n // self.n_splits
        
        for i in range(self.n_splits):
            # 各foldの範囲を計算
            if i < n % self.n_splits:
                fold_len = fold_size + 1
            else:
                fold_len = fold_size
                
            val_start_idx = fold_size * i + min(i, n % self.n_splits)
            val_end_idx = val_start_idx + fold_len
            
            if val_start_idx >= n or val_end_idx > n:
                continue
                
            val_start = unique_dates[val_start_idx]
            val_end = unique_dates[val_end_idx - 1]
            
            # Train期間（val開始前まで）
            train_end = val_start - pd.Timedelta(days=self.embargo_days + 1)
            train_start = unique_dates[0] if i == 0 else unique_dates[0]
            
            # Train期間が十分にない場合はスキップ
            train_dates_in_range = [d for d in unique_dates if train_start <= d <= train_end]
            if len(train_dates_in_range) < 30:  # 最低30日の学習期間
                continue
                
            fold_ranges.append((train_start, train_end, val_start, val_end))
            
        logger.info(f"Built {len(fold_ranges)} walk-forward folds with {self.embargo_days}-day embargo")
        return fold_ranges

    def _extract_all_dates(self) -> pd.Series:
        """データセットから全日付を抽出"""
        dates = []
        
        for idx in range(len(self.dataset)):
            if hasattr(self.dataset, 'get_date'):
                date = self.dataset.get_date(idx)
            elif hasattr(self.dataset, 'data') and 'date' in self.dataset.data.columns:
                # DataFrame形式の場合
                date = self.dataset.data.iloc[idx]['date']
            else:
                # フォールバック: ファイル名から推定
                logger.warning("Using fallback date extraction - implement get_date() method")
                date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=idx//1000)
                
            if isinstance(date, str):
                date = pd.to_datetime(date)
            dates.append(date)
            
        return pd.Series(dates)

    def _get_phase_indices(self) -> List[int]:
        """現在のfold+phaseに対応するインデックスを取得"""
        if self.current_fold >= len(self.fold_ranges):
            raise ValueError(f"Fold {self.current_fold} not available (have {len(self.fold_ranges)} folds)")
            
        train_start, train_end, val_start, val_end = self.fold_ranges[self.current_fold]
        
        all_dates = self._extract_all_dates()
        valid_indices = []
        
        for idx, date in enumerate(all_dates):
            if self.phase == 'train':
                if train_start <= date <= train_end:
                    valid_indices.append(idx)
            elif self.phase == 'val':
                if val_start <= date <= val_end:
                    valid_indices.append(idx)
            elif self.phase == 'test':
                # テスト期間は次のfoldのval期間（存在すれば）
                if self.current_fold + 1 < len(self.fold_ranges):
                    _, _, next_val_start, next_val_end = self.fold_ranges[self.current_fold + 1]
                    if next_val_start <= date <= next_val_end:
                        valid_indices.append(idx)
                        
        logger.info(f"Fold {self.current_fold}, phase {self.phase}: {len(valid_indices)} samples")
        logger.info(f"Date range: {train_start.date()} to {train_end.date()}" if self.phase == 'train' 
                   else f"Date range: {val_start.date()} to {val_end.date()}")
        
        return valid_indices

    def _group_by_day(self) -> List[List[int]]:
        """有効インデックスを日ごとにグループ化"""
        day_to_indices = {}
        all_dates = self._extract_all_dates()
        
        for idx in self.valid_indices:
            date = all_dates[idx]
            # 日付を文字列に変換してキーとして使用
            date_key = date.strftime('%Y-%m-%d')
            
            if date_key not in day_to_indices:
                day_to_indices[date_key] = []
            day_to_indices[date_key].append(idx)
        
        # 最小ノード数でフィルタ
        day_groups = [
            indices for indices in day_to_indices.values() 
            if len(indices) >= self.min_nodes_per_day
        ]
        
        # 日付順にソート
        sorted_days = sorted(day_to_indices.keys())
        day_groups = [
            day_to_indices[day] for day in sorted_days 
            if len(day_to_indices[day]) >= self.min_nodes_per_day
        ]
        
        return day_groups

    def _log_statistics(self):
        """統計情報をログ出力"""
        if not self.day_groups:
            logger.warning(f"No valid day groups for fold {self.current_fold}, phase {self.phase}")
            return
            
        sizes = [len(group) for group in self.day_groups]
        total_batches = 0
        
        for size in sizes:
            if size > self.max_batch_size:
                total_batches += (size + self.max_batch_size - 1) // self.max_batch_size
            else:
                total_batches += 1
        
        logger.info(
            f"WalkForwardDateBucketSampler [fold={self.current_fold}, phase={self.phase}]:\n"
            f"  Days: {len(self.day_groups)}, "
            f"  Samples: {sum(sizes)}\n"
            f"  Day sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}\n"
            f"  Batches: {total_batches}, "
            f"  Max batch size: {self.max_batch_size}\n"
            f"  Embargo days: {self.embargo_days}"
        )

    def __iter__(self) -> Iterator[List[int]]:
        """バッチを生成"""
        if not self.day_groups:
            return iter([])
            
        # 日の順序を決定（trainは時系列順、val/testもembargo確保のため順序維持）
        day_order = list(range(len(self.day_groups)))
        
        # 日内のサンプル順序のみシャッフル（日の順序は保持）
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
        else:
            rng = None
            
        for day_idx in day_order:
            day_indices = self.day_groups[day_idx].copy()
            
            # 同日内のサンプルをシャッフル
            if self.shuffle and rng is not None:
                rng.shuffle(day_indices)
                
            # バッチサイズを超える場合は分割
            if len(day_indices) > self.max_batch_size:
                for i in range(0, len(day_indices), self.max_batch_size):
                    chunk = day_indices[i:i + self.max_batch_size]
                    if len(chunk) >= self.min_nodes_per_day:
                        yield chunk
            else:
                yield day_indices

    def __len__(self) -> int:
        """バッチ数を返す"""
        if not self.day_groups:
            return 0
            
        total_batches = 0
        for day_indices in self.day_groups:
            if len(day_indices) > self.max_batch_size:
                n_chunks = (len(day_indices) + self.max_batch_size - 1) // self.max_batch_size
                total_batches += n_chunks
            else:
                total_batches += 1
        return total_batches

    def get_fold_info(self) -> Dict[str, Any]:
        """現在のfold情報を返す"""
        if self.current_fold >= len(self.fold_ranges):
            return {}
            
        train_start, train_end, val_start, val_end = self.fold_ranges[self.current_fold]
        
        return {
            'fold': self.current_fold,
            'phase': self.phase,
            'train_range': (train_start, train_end),
            'val_range': (val_start, val_end),
            'embargo_days': self.embargo_days,
            'total_days': len(self.day_groups),
            'total_samples': sum(len(group) for group in self.day_groups),
            'total_batches': len(self)
        }


def create_walk_forward_samplers(
    dataset, 
    n_splits: int = 5, 
    embargo_days: int = 20,
    current_fold: int = 0,
    **sampler_kwargs
) -> Tuple[WalkForwardDateBucketSampler, WalkForwardDateBucketSampler, Optional[WalkForwardDateBucketSampler]]:
    """
    train/val/testサンプラーを作成
    
    Args:
        dataset: データセット
        n_splits: 分割数
        embargo_days: embargo期間
        current_fold: 現在のfold
        **sampler_kwargs: サンプラーの追加引数
        
    Returns:
        (train_sampler, val_sampler, test_sampler)
    """
    train_sampler = WalkForwardDateBucketSampler(
        dataset, 
        n_splits=n_splits,
        embargo_days=embargo_days,
        current_fold=current_fold,
        phase='train',
        **sampler_kwargs
    )
    
    val_sampler = WalkForwardDateBucketSampler(
        dataset,
        n_splits=n_splits, 
        embargo_days=embargo_days,
        current_fold=current_fold,
        phase='val',
        shuffle=False,  # 評価時は順序固定
        **sampler_kwargs
    )
    
    # テストサンプラー（次のfoldがあれば作成）
    test_sampler = None
    if current_fold + 1 < n_splits:
        test_sampler = WalkForwardDateBucketSampler(
            dataset,
            n_splits=n_splits,
            embargo_days=embargo_days,
            current_fold=current_fold,
            phase='test',
            shuffle=False,
            **sampler_kwargs
        )
    
    return train_sampler, val_sampler, test_sampler