"""
Streaming Dataset with PyArrow/Parquet Memory Mapping and Online Normalization
ATFT-GAT-FAN向け最適化データローダー
"""

import os
import logging
from typing import Dict, List, Optional, Any, Iterator, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info
import pyarrow.parquet as pq
import pyarrow as pa
from functools import lru_cache
import gc

from ...utils.settings import get_settings

logger = logging.getLogger(__name__)
config = get_settings()


class StreamingParquetDataset(IterableDataset):
    """
    PyArrowベースのストリーミングデータセット

    特徴:
    - メモリマップ読み込み
    - オンライン正規化
    - ゼロコピーTensor変換
    - マルチプロセス対応
    """

    def __init__(
        self,
        parquet_files: List[str],
        feature_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        data_config: Optional[Any] = None,
        sequence_length: int = 60,
        prediction_horizons: List[int] = [1, 5, 10],
        batch_size: int = 1024,
        use_memory_map: bool = True,
        online_normalization: bool = True,
        cache_stats: bool = True,
        **kwargs
    ):
        """
        Args:
            parquet_files: Parquetファイルパスのリスト
            feature_cols: 特徴量列名リスト（指定がない場合はdata_configから自動生成）
            target_cols: ターゲット列名リスト（指定がない場合は自動生成）
            data_config: データ設定オブジェクト（ML_DATASET_COLUMNS.md準拠）
            sequence_length: シーケンス長
            prediction_horizons: 予測ホライズン
            batch_size: バッチサイズ
            use_memory_map: メモリマップを使用
            online_normalization: オンライン正規化を使用
            cache_stats: 統計量をキャッシュ
        """
        self.parquet_files = [Path(f) for f in parquet_files]
        self.data_config = data_config
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.batch_size = batch_size
        self.use_memory_map = use_memory_map
        self.online_normalization = online_normalization
        self.cache_stats = cache_stats

        # 特徴量カラムの設定（自動生成 or 指定）
        if feature_cols is None and data_config is not None:
            self.feature_cols = self._generate_feature_cols(data_config)
        else:
            self.feature_cols = feature_cols or []

        if target_cols is None:
            self.target_cols = self._generate_target_cols()
        else:
            self.target_cols = target_cols or []

        # 統計量キャッシュ
        self._stats_cache = {}
        self._global_stats = {}

        # メモリ効率のためのバッファ
        self._feature_buffer = None
        self._target_buffer = None

        # 統計量の事前計算
        if self.cache_stats and self.online_normalization:
            self._compute_global_stats()

        logger.info(f"Initialized StreamingParquetDataset with {len(self.parquet_files)} files")
        logger.info(f"Feature columns: {len(self.feature_cols)}, Target columns: {len(self.target_cols)}")
        if self.data_config:
            logger.info("Using ML_DATASET_COLUMNS.md compatible configuration")

    def _generate_feature_cols(self, data_config) -> List[str]:
        """ML_DATASET_COLUMNS.md準拠の特徴量カラムを自動生成"""
        feature_cols = []

        # 基本特徴量
        if hasattr(data_config, 'basic'):
            if hasattr(data_config.basic, 'price_volume'):
                feature_cols.extend(data_config.basic.price_volume)
            if hasattr(data_config.basic, 'flags'):
                feature_cols.extend(data_config.basic.flags)

        # テクニカル指標
        if hasattr(data_config, 'technical'):
            # Momentum (RSI)
            if hasattr(data_config.technical, 'momentum'):
                feature_cols.extend(data_config.technical.momentum)

            # Volatility
            if hasattr(data_config.technical, 'volatility'):
                feature_cols.extend(data_config.technical.volatility)

            # Trend (ADX)
            if hasattr(data_config.technical, 'trend'):
                feature_cols.extend(data_config.technical.trend)

            # Moving Averages
            if hasattr(data_config.technical, 'moving_averages'):
                feature_cols.extend(data_config.technical.moving_averages)

            # MACD
            if hasattr(data_config.technical, 'macd'):
                feature_cols.extend(data_config.technical.macd)

            # Bollinger Bands
            if hasattr(data_config.technical, 'bollinger_bands'):
                feature_cols.extend(data_config.technical.bollinger_bands)

        # MA派生特徴量
        if hasattr(data_config, 'ma_derived'):
            for category in ['price_deviations', 'ma_gaps', 'ma_slopes', 'ma_crosses', 'ma_ribbon']:
                if hasattr(data_config.ma_derived, category):
                    feature_cols.extend(getattr(data_config.ma_derived, category))

        # リターン×MA相互作用特徴量
        if hasattr(data_config, 'returns_ma_interaction'):
            for category in ['momentum', 'interactions']:
                if hasattr(data_config.returns_ma_interaction, category):
                    feature_cols.extend(getattr(data_config.returns_ma_interaction, category))

        # フロー特徴量
        if hasattr(data_config, 'flow'):
            feature_cols.extend(data_config.flow)

        # リターン特徴量
        if hasattr(data_config, 'returns') and hasattr(data_config.returns, 'columns'):
            feature_cols.extend(data_config.returns.columns)

        # 成熟度フラグ
        if hasattr(data_config, 'maturity_flags'):
            feature_cols.extend(data_config.maturity_flags)

        logger.info(f"Auto-generated {len(feature_cols)} feature columns from config")
        return list(set(feature_cols))  # 重複除去

    def _generate_target_cols(self) -> List[str]:
        """ターゲットカラムを自動生成"""
        target_cols = []

        # ML_DATASET_COLUMNS.md準拠のターゲット
        for horizon in self.prediction_horizons:
            target_cols.append(f"target_{horizon}d")
            target_cols.append(f"target_{horizon}d_binary")

        logger.info(f"Auto-generated target columns: {target_cols}")
        return target_cols

    def _compute_global_stats(self):
        """グローバル統計量の事前計算"""
        logger.info("Computing global statistics for online normalization...")

        all_means = []
        all_stds = []
        sample_count = 0

        # サンプリングで統計量を計算（メモリ効率）
        for file_path in self.parquet_files[:min(10, len(self.parquet_files))]:
            try:
                # メモリマップで読み込み
                table = pq.read_table(
                    file_path,
                    columns=self.feature_cols,
                    memory_map=self.use_memory_map
                )

                # サンプリング（10%）
                n_rows = len(table)
                sample_indices = np.random.choice(n_rows, size=min(10000, n_rows//10), replace=False)
                sample_table = table.take(sample_indices)

                df = sample_table.to_pandas()

                # 数値列のみ処理
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    means = df[numeric_cols].mean()
                    stds = df[numeric_cols].std()

                    all_means.append(means)
                    all_stds.append(stds)
                    sample_count += len(df)

            except Exception as e:
                logger.warning(f"Failed to compute stats for {file_path}: {e}")
                continue

        if all_means:
            # 全体の統計量を集約
            self._global_stats = {
                'mean': pd.concat(all_means, axis=1).mean(axis=1),
                'std': pd.concat(all_stds, axis=1).mean(axis=1),
                'sample_count': sample_count
            }

            logger.info(f"Global stats computed from {sample_count} samples")
            logger.info(f"Feature means range: {self._global_stats['mean'].min():.3f} - {self._global_stats['mean'].max():.3f}")
            logger.info(f"Feature stds range: {self._global_stats['std'].min():.3f} - {self._global_stats['std'].max():.3f}")

    @lru_cache(maxsize=32)
    def _get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """ファイルごとの統計量をキャッシュ"""
        if not self.online_normalization or not self._global_stats:
            return {}

        # 簡易的にグローバル統計を使用
        return self._global_stats

    def _normalize_features(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """特徴量の正規化"""
        if not self.online_normalization or not self._global_stats:
            return df

        stats = self._get_file_stats(file_path)

        # pandasの新しいfillna APIを使用（FutureWarning対策）
        df = df.copy()
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # 数値列のみ正規化
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in stats['mean'] and col in stats['std']:
                mean_val = stats['mean'][col]
                std_val = stats['std'][col]
                if std_val > 1e-8:  # ゼロ除算防止
                    df[col] = (df[col] - mean_val) / std_val

        return df

    def _create_sequences_from_table(
        self,
        table: pa.Table,
        file_path: str
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """テーブルからシーケンスを作成"""

        # 必要な列のみ抽出
        try:
            feature_table = table.select(self.feature_cols)
            target_table = table.select(self.target_cols) if self.target_cols else None
        except KeyError as e:
            logger.warning(f"Missing columns in {file_path}: {e}")
            return

        # DataFrameに変換
        df_features = feature_table.to_pandas()
        df_targets = target_table.to_pandas() if target_table is not None else None

        # 正規化
        df_features = self._normalize_features(df_features, file_path)

        # シーケンス作成
        n_samples = len(df_features)
        if n_samples < self.sequence_length + max(self.prediction_horizons):
            return

        # ベクトル化されたシーケンス作成
        seq_indices = np.arange(
            0,
            n_samples - self.sequence_length - max(self.prediction_horizons) + 1,
            1  # ストライド1で全シーケンス
        )

        for start_idx in seq_indices:
            try:
                # 入力シーケンス
                seq_end = start_idx + self.sequence_length
                feature_seq = df_features.iloc[start_idx:seq_end].values.astype(np.float32)

                # ターゲット
                targets = {}
                for horizon in self.prediction_horizons:
                    target_idx = seq_end + horizon - 1
                    if target_idx < n_samples:
                        if df_targets is not None and len(df_targets) > target_idx:
                            targets[f'h{horizon}'] = df_targets.iloc[target_idx].values.astype(np.float32)
                        else:
                            # 特徴量からターゲットを計算（簡易版）
                            targets[f'h{horizon}'] = feature_seq[-1].astype(np.float32)  # 最後の値をターゲットに

                if targets:
                    # Tensor変換（ゼロコピー）
                    feature_tensor = torch.from_numpy(feature_seq)
                    target_tensors = {k: torch.from_numpy(v) for k, v in targets.items()}

                    # メタデータ
                    metadata = {
                        'file_path': str(file_path),
                        'sequence_idx': start_idx,
                        'sequence_length': self.sequence_length,
                        'prediction_horizons': self.prediction_horizons
                    }

                    yield feature_tensor, target_tensors, metadata

            except Exception as e:
                logger.debug(f"Failed to create sequence at index {start_idx}: {e}")
                continue

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]]:
        """イテレータ実装"""
        worker_info = get_worker_info()

        if worker_info is None:
            # シングルプロセス
            file_list = self.parquet_files
        else:
            # マルチプロセス対応
            per_worker = len(self.parquet_files) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.parquet_files)
            file_list = self.parquet_files[start:end]

        # ファイルを順次処理
        for file_path in file_list:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            try:
                # PyArrowでメモリマップ読み込み
                table = pq.read_table(
                    file_path,
                    memory_map=self.use_memory_map,
                    use_threads=True
                )

                # バッチ処理でメモリ効率化
                batch_size = min(50000, len(table))  # 適度なバッチサイズ

                for batch in table.to_batches(max_chunksize=batch_size):
                    # シーケンス作成
                    for feature_seq, target_seq, metadata in self._create_sequences_from_table(
                        pa.Table.from_batches([batch]), str(file_path)
                    ):
                        yield feature_seq, target_seq, metadata

                    # メモリ解放
                    del batch
                    gc.collect()

            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                continue

    def __len__(self) -> int:
        """データセットサイズの推定"""
        # 簡易的な推定（実際のシーケンス数を正確に計算するのはコストがかかる）
        total_sequences = 0
        for file_path in self.parquet_files[:5]:  # サンプリング
            try:
                table = pq.read_table(file_path, memory_map=True)
                n_samples = len(table)
                if n_samples >= self.sequence_length + max(self.prediction_horizons):
                    seq_count = n_samples - self.sequence_length - max(self.prediction_horizons) + 1
                    total_sequences += seq_count
            except Exception:
                continue

        # 全ファイルに外挿
        if len(self.parquet_files) > 5:
            total_sequences = int(total_sequences * len(self.parquet_files) / 5)

        return max(total_sequences, 1)


class OptimizedDataLoader:
    """
    最適化されたDataLoaderクラス

    特徴:
    - pin_memory最適化
    - prefetch_factor調整
    - persistent_workers
    - メモリ使用量最適化
    """

    def __init__(
        self,
        dataset: StreamingParquetDataset,
        batch_size: int = 1024,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs
    ):
        """
        Args:
            dataset: データセット
            batch_size: バッチサイズ
            num_workers: ワーカー数
            prefetch_factor: プリフェッチ係数
            pin_memory: メモリピニング
            persistent_workers: 永続ワーカー
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # システム設定に基づく自動調整
        if config.num_workers > 0:
            self.num_workers = min(config.num_workers, self.num_workers)
        if config.prefetch_factor > 0:
            self.prefetch_factor = config.prefetch_factor
        self.pin_memory = config.pin_memory

        logger.info(f"OptimizedDataLoader configured: batch_size={batch_size}, "
                   f"num_workers={self.num_workers}, prefetch_factor={self.prefetch_factor}, "
                   f"pin_memory={self.pin_memory}")

    def get_dataloader(self):
        """DataLoaderインスタンスを取得"""
        from torch.utils.data import DataLoader

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """カスタムcollate関数"""
        if not batch:
            return None

        features = []
        targets = {f'h{h}': [] for h in self.dataset.prediction_horizons}
        metadata_list = []

        for feature_seq, target_seq, metadata in batch:
            features.append(feature_seq)
            for horizon in self.dataset.prediction_horizons:
                key = f'h{horizon}'
                if key in target_seq:
                    targets[key].append(target_seq[key])

            metadata_list.append(metadata)

        # バッチ化
        feature_batch = torch.stack(features)

        # targetsのバッチ化（存在するもののみ）
        target_batch = {}
        for horizon in self.dataset.prediction_horizons:
            key = f'h{horizon}'
            if targets[key]:
                target_batch[key] = torch.stack(targets[key])
            else:
                # ダミーのターゲット
                target_batch[key] = torch.zeros(len(batch), feature_seq.size(-1))

        return {
            'features': feature_batch,
            'targets': target_batch,
            'metadata': metadata_list
        }


# 後方互換性のためのエイリアス
StreamingDataset = StreamingParquetDataset
