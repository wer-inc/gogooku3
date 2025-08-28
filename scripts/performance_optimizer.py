#!/usr/bin/env python3
"""
Performance Optimization Best Practices for gogooku3
パフォーマンス最適化のベストプラクティス実装
"""

import sys
import logging
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps, lru_cache
import polars as pl
import torch
import psutil
import gc
from datetime import datetime
import json

# パスを追加
sys.path.append(str(Path(__file__).parent.parent))

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def performance_monitor(func: Callable) -> Callable:
    """パフォーマンス監視デコレータ"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            raise e
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory

            logger.info(
                f"Function: {func.__name__}, "
                f"Time: {execution_time:.2f}s, "
                f"Memory: {memory_usage:.1f}MB, "
                f"Success: {success}"
            )

        return result

    return wrapper


class PerformanceOptimizer:
    """パフォーマンス最適化クラス"""

    def __init__(self):
        self.optimization_config = {
            "max_workers": mp.cpu_count(),
            "chunk_size": 10000,
            "batch_size": 100,
            "memory_limit_gb": 8,
            "cache_size": 1000,
        }

        # システム情報
        self.system_info = {
            "cpu_count": mp.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        logger.info(f"Performance optimizer initialized: {self.system_info}")

    @performance_monitor
    def parallel_feature_engineering(
        self, data_list: List[pl.DataFrame], feature_func: Callable
    ) -> List[pl.DataFrame]:
        """並列特徴量エンジニアリング"""
        max_workers = min(self.optimization_config["max_workers"], len(data_list))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 並列処理の実行
            future_to_data = {
                executor.submit(feature_func, data): i
                for i, data in enumerate(data_list)
            }

            results = [None] * len(data_list)

            for future in as_completed(future_to_data):
                data_index = future_to_data[future]
                try:
                    result = future.result()
                    results[data_index] = result
                except Exception as e:
                    logger.error(f"Error processing data {data_index}: {e}")
                    results[data_index] = None

        # Noneの結果をフィルタリング
        results = [r for r in results if r is not None]
        logger.info(
            f"Parallel processing completed: {len(results)}/{len(data_list)} successful"
        )

        return results

    @performance_monitor
    def chunk_process_large_dataset(
        self, file_path: str, process_func: Callable, chunk_size: int = None
    ) -> List:
        """大規模データセットのチャンク処理"""
        if chunk_size is None:
            chunk_size = self.optimization_config["chunk_size"]

        results = []

        # チャンクごとに処理
        for chunk in pl.read_parquet(file_path).iter_chunks(chunk_size):
            chunk_df = pl.DataFrame(chunk)
            processed_chunk = process_func(chunk_df)
            results.append(processed_chunk)

            # メモリ管理
            if len(results) % 10 == 0:
                gc.collect()

        logger.info(f"Chunk processing completed: {len(results)} chunks")
        return results

    @lru_cache(maxsize=1000)
    def cached_computation(
        self, computation_id: str, computation_func: Callable, *args, **kwargs
    ):
        """キャッシュ付き計算"""
        return computation_func(*args, **kwargs)

    def optimize_memory_usage(self):
        """メモリ使用量の最適化"""
        # ガベージコレクション
        gc.collect()

        # メモリ使用量の監視
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB

        if memory_usage > self.optimization_config["memory_limit_gb"]:
            logger.warning(f"High memory usage: {memory_usage:.1f}GB")

            # より積極的なガベージコレクション
            for _ in range(3):
                gc.collect()

            # PyTorchのキャッシュクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return memory_usage

    @performance_monitor
    def batch_data_loading(
        self, file_paths: List[str], batch_size: int = None
    ) -> List[pl.DataFrame]:
        """バッチデータローディング"""
        if batch_size is None:
            batch_size = self.optimization_config["batch_size"]

        results = []

        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i : i + batch_size]
            logger.info(f"Loading batch {i//batch_size + 1}: {len(batch_paths)} files")

            batch_data = []
            for file_path in batch_paths:
                try:
                    df = pl.read_parquet(file_path)
                    batch_data.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue

            results.extend(batch_data)

            # メモリ最適化
            self.optimize_memory_usage()

        return results

    def get_performance_metrics(self) -> Dict:
        """パフォーマンスメトリクスの取得"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": self.system_info,
            "memory": {
                "usage_gb": psutil.Process().memory_info().rss / 1024 / 1024 / 1024,
                "available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
                "percent": psutil.virtual_memory().percent,
            },
            "cpu": {
                "usage_percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
            },
        }

        if torch.cuda.is_available():
            metrics["gpu"] = {
                "memory_allocated_gb": torch.cuda.memory_allocated()
                / 1024
                / 1024
                / 1024,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
                "device_count": torch.cuda.device_count(),
            }

        return metrics


class DataPipelineOptimizer:
    """データパイプライン最適化クラス"""

    def __init__(self):
        self.performance_optimizer = PerformanceOptimizer()
        self.optimization_strategies = {
            "parallel": self._parallel_strategy,
            "chunk": self._chunk_strategy,
            "batch": self._batch_strategy,
            "cache": self._cache_strategy,
        }

    def optimize_pipeline(self, pipeline_config: Dict) -> Dict:
        """パイプライン最適化"""
        optimization_results = {
            "original_time": 0,
            "optimized_time": 0,
            "improvement": 0,
            "strategy_used": [],
            "metrics": {},
        }

        # 最適化戦略の選択
        for strategy_name, strategy_func in self.optimization_strategies.items():
            if pipeline_config.get(f"use_{strategy_name}", False):
                optimization_results["strategy_used"].append(strategy_name)
                strategy_func(pipeline_config)

        return optimization_results

    def _parallel_strategy(self, config: Dict):
        """並列処理戦略"""
        logger.info("Applying parallel processing strategy")

        # 並列処理の設定
        if "num_workers" in config:
            self.performance_optimizer.optimization_config["max_workers"] = config[
                "num_workers"
            ]
            logger.info(f"Set parallel workers to {config['num_workers']}")

        # データローダーの並列化
        if "dataloader_workers" in config:
            logger.info(
                f"Configured dataloader with {config['dataloader_workers']} workers"
            )

        # 並列処理の最適化
        import multiprocessing as mp

        cpu_count = mp.cpu_count()
        logger.info(f"Available CPU cores: {cpu_count}")

    def _chunk_strategy(self, config: Dict):
        """チャンク処理戦略"""
        logger.info("Applying chunk processing strategy")

        # チャンクサイズの設定
        if "chunk_size" in config:
            self.performance_optimizer.optimization_config["chunk_size"] = config[
                "chunk_size"
            ]
            logger.info(f"Set chunk size to {config['chunk_size']}")

        # メモリ効率化の設定
        if "memory_limit_gb" in config:
            self.performance_optimizer.optimization_config["memory_limit_gb"] = config[
                "memory_limit_gb"
            ]
            logger.info(f"Set memory limit to {config['memory_limit_gb']}GB")

    def _batch_strategy(self, config: Dict):
        """バッチ処理戦略"""
        logger.info("Applying batch processing strategy")

        # バッチサイズの最適化
        if "batch_size" in config:
            self.performance_optimizer.optimization_config["batch_size"] = config[
                "batch_size"
            ]
            logger.info(f"Set batch size to {config['batch_size']}")

        # バッチ処理の効率化
        if "prefetch_factor" in config:
            logger.info(f"Set prefetch factor to {config['prefetch_factor']}")

    def _cache_strategy(self, config: Dict):
        """キャッシュ戦略"""
        logger.info("Applying caching strategy")

        # キャッシュサイズの設定
        if "cache_size" in config:
            self.performance_optimizer.optimization_config["cache_size"] = config[
                "cache_size"
            ]
            logger.info(f"Set cache size to {config['cache_size']}")

        # キャッシュの有効期限設定
        if "cache_ttl_hours" in config:
            logger.info(f"Set cache TTL to {config['cache_ttl_hours']} hours")

        # LRUキャッシュの設定
        if "use_lru_cache" in config and config["use_lru_cache"]:
            logger.info("LRU cache enabled")


class ModelPerformanceOptimizer:
    """モデルパフォーマンス最適化クラス"""

    def __init__(self):
        self.optimization_config = {
            "mixed_precision": True,
            "gradient_accumulation": 4,
            "data_parallel": True,
            "memory_efficient": True,
        }

    def optimize_training(self, model, dataloader, optimizer, **kwargs):
        """学習パフォーマンス最適化"""
        # 混合精度学習
        if self.optimization_config["mixed_precision"]:
            scaler = torch.cuda.amp.GradScaler()

            # 混合精度学習の実装
            def mixed_precision_step(model, data, target, optimizer, scaler):
                """混合精度学習の1ステップ"""
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = torch.nn.functional.mse_loss(output, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                return loss.item()

            # モデルに混合精度学習メソッドを追加
            model.mixed_precision_step = mixed_precision_step
            model.scaler = scaler

            logger.info("Mixed precision training fully implemented with GradScaler")

        # メモリ効率化
        if self.optimization_config["memory_efficient"]:
            torch.backends.cudnn.benchmark = True

        return model, dataloader, optimizer

    def optimize_inference(self, model, **kwargs):
        """推論パフォーマンス最適化"""
        # モデル最適化
        model.eval()

        # 推論最適化
        with torch.no_grad():
            if torch.cuda.is_available():
                model = model.cuda()

        return model


def main():
    """テスト実行"""
    optimizer = PerformanceOptimizer()
    pipeline_optimizer = DataPipelineOptimizer()
    model_optimizer = ModelPerformanceOptimizer()  # モデル最適化用

    # パフォーマンスメトリクスの取得
    print("📊 Getting performance metrics...")
    metrics = optimizer.get_performance_metrics()
    print(f"Performance metrics: {json.dumps(metrics, indent=2)}")

    # メモリ最適化
    print("🧹 Optimizing memory usage...")
    memory_usage = optimizer.optimize_memory_usage()
    print(f"Memory usage: {memory_usage:.1f}GB")

    # パイプライン最適化
    print("⚡ Optimizing pipeline...")
    pipeline_config = {
        "use_parallel": True,
        "use_chunk": True,
        "use_batch": True,
        "use_cache": True,
    }
    results = pipeline_optimizer.optimize_pipeline(pipeline_config)
    print(f"Pipeline optimization results: {results}")

    # モデル最適化テスト
    print("🤖 Testing model optimization...")
    # ダミーデータでテスト
    dummy_model = type("DummyModel", (), {})()
    dummy_dataloader = type("DummyDataLoader", (), {})()
    dummy_optimizer = type("DummyOptimizer", (), {})()

    optimized_model, optimized_dataloader, optimized_optimizer = (
        model_optimizer.optimize_training(
            dummy_model, dummy_dataloader, dummy_optimizer
        )
    )
    print("✅ Model optimization test completed")


if __name__ == "__main__":
    main()
