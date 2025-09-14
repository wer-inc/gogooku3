"""
Walk-Forward Cross-Validation with Enhanced Embargo (V2)
強化版ウォークフォワード検証（embargo=最大ホライズン対応）

主な改良点:
- embargo_days=20（最大ホライズン20dに対応）
- 日付ベースの厳密な時系列分割
- データリーク防止の3重保護
- 高速化とメモリ効率の改善
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


class WalkForwardSplitterV2:
    """
    強化版Walk-Forward分割器
    
    データリーク防止の3重保護:
    1. Walk-Forward分割: 厳密な時系列順序
    2. Embargo: train/val境界にgap_days空白期間
    3. Purge: 重複期間の完全排除
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 20,
        purge_days: int = 0,
        min_train_days: int = 252,
        min_test_days: int = 63,
        test_size_days: int | None = None,
        overlap_check: bool = True,
        date_column: str = 'date',
        verbose: bool = True
    ):
        """
        Args:
            n_splits: Walk-Forward分割数
            embargo_days: train/val境界の空白期間（日数）
            purge_days: 追加のpurge期間（embargo後さらに除外）
            min_train_days: 最小訓練期間（営業日）
            min_test_days: 最小テスト期間（営業日）
            test_size_days: テスト期間の固定日数（None=自動計算）
            overlap_check: 重複チェックを実行するか
            date_column: 日付列名
            verbose: 詳細ログ出力
        """
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.purge_days = purge_days
        self.min_train_days = min_train_days
        self.min_test_days = min_test_days
        self.test_size_days = test_size_days
        self.overlap_check = overlap_check
        self.date_column = date_column
        self.verbose = verbose

        # 分割結果のキャッシュ
        self.splits: list[dict[str, Any]] = []
        self.date_info: dict[str, Any] = {}

        if self.verbose:
            logger.info(
                f"WalkForwardSplitterV2: n_splits={n_splits}, "
                f"embargo_days={embargo_days}, purge_days={purge_days}"
            )

    def _extract_dates(self, data: pd.DataFrame | pl.DataFrame) -> np.ndarray:
        """データから日付配列を抽出"""
        if isinstance(data, pd.DataFrame):
            dates = pd.to_datetime(data[self.date_column]).dt.date
        else:  # polars
            dates = data[self.date_column].cast(pl.Date)
            if hasattr(dates, 'to_pandas'):
                dates = dates.to_pandas().dt.date
            else:
                dates = [d.date() if hasattr(d, 'date') else d for d in dates.to_list()]

        unique_dates = np.array(sorted(set(dates)))
        return unique_dates

    def _calculate_split_dates(self, unique_dates: np.ndarray) -> list[dict[str, date]]:
        """分割日付を計算"""
        total_days = len(unique_dates)

        if self.verbose:
            logger.info(f"Total unique dates: {total_days}")
            logger.info(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")

        # テスト期間サイズの決定
        if self.test_size_days:
            test_days = min(self.test_size_days, total_days // (self.n_splits + 1))
        else:
            # 自動計算: 全体をn_splits+1で割った期間
            test_days = max(self.min_test_days, total_days // (self.n_splits + 1))

        if self.verbose:
            logger.info(f"Test period size: {test_days} days")

        splits = []

        for fold in range(self.n_splits):
            # テスト期間の終了日（最新日から逆算）
            test_end_idx = total_days - 1 - fold * (test_days + self.embargo_days + self.purge_days)
            test_start_idx = test_end_idx - test_days + 1

            # Embargo適用後の訓練期間終了日
            embargo_end_idx = test_start_idx - self.embargo_days - 1
            purge_end_idx = embargo_end_idx - self.purge_days

            # 訓練期間の開始日（最小期間を確保）
            train_end_idx = purge_end_idx
            train_start_idx = max(0, train_end_idx - self.min_train_days + 1)

            # インデックス妥当性チェック
            if (train_start_idx < 0 or
                train_end_idx < train_start_idx or
                test_start_idx < 0 or
                test_end_idx < test_start_idx or
                test_start_idx <= train_end_idx + self.embargo_days + self.purge_days):

                if self.verbose:
                    logger.warning(f"Fold {fold}: Invalid date range, skipping")
                continue

            split_info = {
                'fold': fold,
                'train_start': unique_dates[train_start_idx],
                'train_end': unique_dates[train_end_idx],
                'test_start': unique_dates[test_start_idx],
                'test_end': unique_dates[test_end_idx],
                'embargo_days': self.embargo_days,
                'purge_days': self.purge_days,
                'train_days': train_end_idx - train_start_idx + 1,
                'test_days': test_end_idx - test_start_idx + 1,
                'gap_days': test_start_idx - train_end_idx - 1
            }

            splits.append(split_info)

            if self.verbose:
                logger.info(
                    f"Fold {fold}: Train[{split_info['train_start']} - {split_info['train_end']}] "
                    f"Test[{split_info['test_start']} - {split_info['test_end']}] "
                    f"Gap={split_info['gap_days']}d"
                )

        return splits

    def _check_overlaps(self, splits: list[dict[str, Any]]) -> list[str]:
        """分割間のオーバーラップをチェック"""
        warnings = []

        for i, split_i in enumerate(splits):
            for j, split_j in enumerate(splits):
                if i >= j:
                    continue

                # Train-Train overlap
                if (split_i['train_start'] <= split_j['train_end'] and
                    split_j['train_start'] <= split_i['train_end']):
                    warnings.append(
                        f"Fold {i} train overlaps with Fold {j} train"
                    )

                # Test-Test overlap
                if (split_i['test_start'] <= split_j['test_end'] and
                    split_j['test_start'] <= split_i['test_end']):
                    warnings.append(
                        f"Fold {i} test overlaps with Fold {j} test"
                    )

                # Train-Test overlap (critical)
                if (split_i['train_start'] <= split_j['test_end'] and
                    split_j['test_start'] <= split_i['train_end']):
                    warnings.append(
                        f"CRITICAL: Fold {i} train overlaps with Fold {j} test"
                    )

        return warnings

    def fit(self, data: pd.DataFrame | pl.DataFrame) -> WalkForwardSplitterV2:
        """データに基づいて分割を準備"""
        unique_dates = self._extract_dates(data)

        # 分割日付を計算
        self.splits = self._calculate_split_dates(unique_dates)

        # 重複チェック
        if self.overlap_check and self.splits:
            warnings = self._check_overlaps(self.splits)
            if warnings:
                logger.warning(f"Found {len(warnings)} overlaps:")
                for warning in warnings:
                    logger.warning(f"  {warning}")

        # メタ情報を保存
        self.date_info = {
            'total_dates': len(unique_dates),
            'date_min': unique_dates[0],
            'date_max': unique_dates[-1],
            'valid_splits': len(self.splits)
        }

        if self.verbose:
            logger.info(f"Prepared {len(self.splits)} valid splits")

        return self

    def split(
        self,
        data: pd.DataFrame | pl.DataFrame
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Walk-Forward分割を生成
        
        Yields:
            (train_indices, test_indices): インデックス配列のタプル
        """
        if not self.splits:
            self.fit(data)

        # 日付列を取得
        if isinstance(data, pd.DataFrame):
            dates = pd.to_datetime(data[self.date_column]).dt.date
        else:  # polars
            dates = data[self.date_column].cast(pl.Date)
            if hasattr(dates, 'to_pandas'):
                dates = dates.to_pandas().dt.date
            else:
                dates = [d.date() if hasattr(d, 'date') else d for d in dates.to_list()]

        dates = np.array(dates)

        for split_info in self.splits:
            # 訓練インデックス
            train_mask = ((dates >= split_info['train_start']) &
                         (dates <= split_info['train_end']))
            train_indices = np.where(train_mask)[0]

            # テストインデックス
            test_mask = ((dates >= split_info['test_start']) &
                        (dates <= split_info['test_end']))
            test_indices = np.where(test_mask)[0]

            if len(train_indices) == 0 or len(test_indices) == 0:
                logger.warning(f"Fold {split_info['fold']}: Empty train or test set")
                continue

            if self.verbose:
                logger.info(
                    f"Fold {split_info['fold']}: "
                    f"Train={len(train_indices)} samples, "
                    f"Test={len(test_indices)} samples"
                )

            yield train_indices, test_indices

    def get_split_info(self) -> list[dict[str, Any]]:
        """分割情報を取得"""
        return self.splits.copy()

    def get_date_info(self) -> dict[str, Any]:
        """日付情報を取得"""
        return self.date_info.copy()

    def validate_split(self, data: pd.DataFrame | pl.DataFrame) -> dict[str, Any]:
        """分割の妥当性を検証"""
        if not self.splits:
            self.fit(data)

        validation_result = {
            'total_splits': len(self.splits),
            'overlaps': [],
            'gaps': [],
            'coverage': {}
        }

        # オーバーラップチェック
        validation_result['overlaps'] = self._check_overlaps(self.splits)

        # ギャップ情報
        for split in self.splits:
            validation_result['gaps'].append({
                'fold': split['fold'],
                'gap_days': split['gap_days'],
                'embargo_days': split['embargo_days'],
                'purge_days': split['purge_days']
            })

        # カバレッジ情報
        if self.splits:
            total_train_days = sum(s['train_days'] for s in self.splits)
            total_test_days = sum(s['test_days'] for s in self.splits)

            validation_result['coverage'] = {
                'total_train_days': total_train_days,
                'total_test_days': total_test_days,
                'avg_train_days': total_train_days / len(self.splits),
                'avg_test_days': total_test_days / len(self.splits)
            }

        return validation_result


class WalkForwardDateBucketSamplerV2:
    """
    Walk-Forward分割 + Date-Bucketed Sampling (V2)
    
    強化版の統合サンプラー:
    - WalkForwardSplitterV2との統合
    - embargo=20対応
    - 同日バッチングによる統計整合性確保
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
            embargo_days: embargo期間（日数）
            max_batch_size: 最大バッチサイズ
            min_nodes_per_day: 最小ノード数/日
            shuffle: 日をシャッフルするか
            seed: 乱数シード
            current_fold: 現在のfold番号
            phase: 'train' or 'test'
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

        # Walk-Forward分割器を初期化
        self.splitter = WalkForwardSplitterV2(
            n_splits=n_splits,
            embargo_days=embargo_days,
            verbose=False
        )

        # バッチ生成用データを準備
        self._prepare_batches()

    def _prepare_batches(self):
        """バッチ生成用データを準備"""
        # データセットから日付を取得
        if hasattr(self.dataset, 'data') and self.dataset.data is not None:
            data = self.dataset.data
        elif hasattr(self.dataset, 'df'):
            data = self.dataset.df
        else:
            raise ValueError("Dataset must have 'data' or 'df' attribute")

        # Walk-Forward分割を実行
        self.splitter.fit(data)
        splits = list(self.splitter.split(data))

        if self.current_fold >= len(splits):
            raise ValueError(f"current_fold={self.current_fold} >= n_splits={len(splits)}")

        # 現在のfoldのインデックスを取得
        train_indices, test_indices = splits[self.current_fold]

        if self.phase == 'train':
            self.indices = train_indices
        else:
            self.indices = test_indices

        # 日付ごとにインデックスをグループ化
        self.day_groups = self._group_by_day(self.indices, data)

        # 統計情報
        if self.day_groups:
            sizes = [len(g) for g in self.day_groups]
            logger.info(
                f"WalkForwardDateBucketSamplerV2 fold={self.current_fold} phase={self.phase}: "
                f"{len(self.day_groups)} days, sizes: min={min(sizes)}, max={max(sizes)}, "
                f"mean={np.mean(sizes):.1f}"
            )

    def _group_by_day(self, indices: np.ndarray, data) -> list[list[int]]:
        """インデックスを日付ごとにグループ化"""
        if isinstance(data, pd.DataFrame):
            dates = data.iloc[indices]['date'].dt.date if 'date' in data.columns else None
        else:  # polars
            dates = data.slice(indices[0], len(indices))['date'].cast(pl.Date) if 'date' in data.columns else None

        if dates is None:
            # 日付がない場合は仮実装
            logger.warning("No date column found, using index-based grouping")
            groups = []
            group_size = min(self.max_batch_size, 1000)
            for i in range(0, len(indices), group_size):
                group = indices[i:i+group_size].tolist()
                if len(group) >= self.min_nodes_per_day:
                    groups.append(group)
            return groups

        # 日付でグループ化
        day_to_indices = {}

        if isinstance(data, pd.DataFrame):
            dates_list = dates.tolist()
        else:
            dates_list = dates.to_list()

        for i, date_val in enumerate(dates_list):
            if date_val not in day_to_indices:
                day_to_indices[date_val] = []
            day_to_indices[date_val].append(indices[i])

        # リストに変換してフィルタ
        day_groups = [
            group for group in day_to_indices.values()
            if len(group) >= self.min_nodes_per_day
        ]

        return day_groups

    def __iter__(self):
        """バッチのイテレータを返す"""
        # 日の順序を決定
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.current_fold)
            day_order = rng.permutation(len(self.day_groups))
        else:
            day_order = np.arange(len(self.day_groups))

        # 各日のデータを返す
        for day_idx in day_order:
            day_indices = self.day_groups[day_idx]

            # バッチサイズを超える場合は分割
            if len(day_indices) > self.max_batch_size:
                for i in range(0, len(day_indices), self.max_batch_size):
                    chunk = day_indices[i:i + self.max_batch_size]
                    if len(chunk) >= self.min_nodes_per_day:
                        yield chunk
            else:
                yield day_indices

    def __len__(self):
        """バッチ数を返す"""
        total_batches = 0
        for day_indices in self.day_groups:
            if len(day_indices) > self.max_batch_size:
                n_chunks = (len(day_indices) + self.max_batch_size - 1) // self.max_batch_size
                total_batches += n_chunks
            else:
                total_batches += 1
        return total_batches
