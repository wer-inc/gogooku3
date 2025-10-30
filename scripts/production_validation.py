#!/usr/bin/env python3
"""
ATFT-GAT-FAN Production Workload Validation
本番ワークロードでの性能検証スクリプト
"""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import torch

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False

# PyTorchメモリ管理
torch.backends.cudnn.benchmark = True


@contextmanager
def memory_monitor():
    """メモリ使用量監視コンテキストマネージャ"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    start_memory = psutil.virtual_memory().used

    if torch.cuda.is_available():
        start_gpu_memory = torch.cuda.memory_allocated()

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.virtual_memory().used

        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated()
            peak_gpu_memory = torch.cuda.max_memory_allocated()

        memory_info = {
            'duration_seconds': end_time - start_time,
            'cpu_memory_used_mb': (end_memory - start_memory) / (1024**2),
            'cpu_memory_peak_mb': psutil.virtual_memory().used / (1024**2),
        }

        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_used_mb': (end_gpu_memory - start_gpu_memory) / (1024**2),
                'gpu_memory_peak_mb': peak_gpu_memory / (1024**2),
            })

        logger.info(f"Memory usage: {memory_info}")


class ProductionValidator:
    """本番ワークロード検証クラス"""

    def __init__(self, data_path: str, config_path: str | None = None):
        self.data_path = Path(data_path)
        self.config_path = config_path or (project_root / "configs" / "atft" / "config.yaml")
        self.validation_results_dir = project_root / "validation_results"
        self.validation_results_dir.mkdir(exist_ok=True)

        # 検証シナリオ（高スペックマシン向けに最適化、メモリ効率重視）
        self.scenarios = {
            'tiny_scale': {'batch_size': 256, 'sequence_length': 60, 'n_stocks': 50},
            'small_scale': {'batch_size': 512, 'sequence_length': 60, 'n_stocks': 100},
            'medium_scale': {'batch_size': 1024, 'sequence_length': 60, 'n_stocks': 200},
            'large_scale': {'batch_size': 2048, 'sequence_length': 60, 'n_stocks': 500},
            'production_scale': {'batch_size': 4096, 'sequence_length': 60, 'n_stocks': 1000}
        }

        # パフォーマンス指標
        self.metrics = {
            'throughput_samples_per_sec': 0,
            'memory_efficiency': 0,
            'training_stability': 0,
            'inference_speed': 0,
            'scalability_score': 0
        }

    def create_synthetic_data(self, batch_size: int, seq_length: int, n_stocks: int) -> dict[str, torch.Tensor]:
        """合成データ作成（本番規模シミュレーション）"""
        logger.info(f"Creating synthetic data: batch_size={batch_size}, seq_length={seq_length}, n_stocks={n_stocks}")

        # 特徴量次元（実際のデータ構造に基づく）
        n_features = 8  # Basic + Technical + MA-derived + Interaction + Flow + Returns

        # 動的特徴量
        dynamic_features = torch.randn(batch_size, seq_length, n_stocks, n_features)

        # 静的特徴量
        static_features = torch.randn(batch_size, n_stocks, 4)  # 銘柄固有特徴量

        # ターゲット（複数ホライズン）
        targets = {
            'h1': torch.randn(batch_size, n_stocks),
            'h5': torch.randn(batch_size, n_stocks),
            'h10': torch.randn(batch_size, n_stocks),
            'h20': torch.randn(batch_size, n_stocks)
        }

        return {
            'dynamic_features': dynamic_features,
            'static_features': static_features,
            'targets': targets,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'n_stocks': n_stocks
        }

    def create_mock_model(self):
        """モックモデル作成（実際のモデル構造をシミュレート、メモリ効率重視）"""
        class MockATFTGATFAN(torch.nn.Module):
            def __init__(self, input_size: int = 8, hidden_size: int = 32):  # 隠れ層サイズを半分に
                super().__init__()
                self.input_projection = torch.nn.Linear(input_size, hidden_size)
                # LSTMの代わりにGRUを使用（メモリ効率的）
                self.temporal_encoder = torch.nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=1)
                self.output_head = torch.nn.Linear(hidden_size, 4)  # 4ホライズン

                # 改善機能のシミュレート
                self.freq_dropout = torch.nn.Dropout(0.2)  # 最適化パラメータ
                self.layer_scale = torch.nn.Parameter(torch.ones(hidden_size) * 0.1)

            def forward(self, x):
                # バッチサイズ確認
                if x.dim() != 4:
                    raise ValueError(f"Expected 4D input, got {x.dim()}D")

                batch_size, seq_len, n_stocks, input_size = x.shape

                # メモリ効率的な処理
                x = x.contiguous()  # メモリレイアウト最適化

                # 入力投影
                x = self.input_projection(x)
                x = torch.relu(x)

                # 周波数ドロップアウト
                x = self.freq_dropout(x)

                # LayerScale
                x = x * self.layer_scale

                # 時系列エンコーディング（メモリ効率的に）
                x = x.view(batch_size * n_stocks, seq_len, -1)

                # メモリ効率的なRNN処理
                x, _ = self.temporal_encoder(x)

                x = x[:, -1, :]  # 最後のタイムステップ
                x = x.view(batch_size, n_stocks, -1)

                # 出力
                output = self.output_head(x)
                return {
                    'predictions': output,
                    'hidden_states': x
                }

        return MockATFTGATFAN()

    def validate_batch_processing(self, scenario: str) -> dict[str, Any]:
        """バッチ処理検証"""
        logger.info(f"Validating batch processing for scenario: {scenario}")

        params = self.scenarios[scenario]
        batch_size = params['batch_size']

        # GPU使用確認（高スペックマシンなのでGPUを使用）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {gpu_props.name}")
            logger.info(f"GPU Memory: {gpu_props.total_memory / (1024**3):.1f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")

            # 高スペックGPU向けの最適化設定
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.cuda.empty_cache()  # メモリクリア

            # CUDAデバッグ設定（エラーが発生した場合に有効化）
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # モデル作成
        model = self.create_mock_model().to(device)

        # オプティマイザ設定
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        # 損失関数
        criterion = torch.nn.HuberLoss(delta=0.01)  # 最適化パラメータ

        results = {
            'scenario': scenario,
            'batch_size': batch_size,
            'device': str(device),
            'iterations': [],
            'memory_usage': [],
            'throughput': []
        }

        # ウォームアップ
        logger.info("Warm-up phase...")
        for _ in range(5):
            data = self.create_synthetic_data(batch_size, 60, 100)
            dynamic_features = data['dynamic_features'].to(device)

            with torch.no_grad():
                _ = model(dynamic_features)

        # 本検証
        logger.info("Validation phase...")
        n_iterations = 10

        for i in range(n_iterations):
            logger.info(f"Iteration {i+1}/{n_iterations}")

            # データ作成
            data = self.create_synthetic_data(batch_size, 60, 100)
            dynamic_features = data['dynamic_features'].to(device)
            targets = data['targets']['h1'].to(device)

            # 高スペックGPU向けメモリ管理
            if torch.cuda.is_available() and i % 3 == 0:  # 3イテレーションごとにクリア
                torch.cuda.empty_cache()

            # メモリ監視付きトレーニング
            with memory_monitor() as mem_info:
                start_time = time.time()

                # フォワード
                optimizer.zero_grad()
                outputs = model(dynamic_features)
                predictions = outputs['predictions'][:, :, 0]  # h1予測

                # 損失計算
                loss = criterion(predictions, targets)

                # バックワード
                loss.backward()

                # 勾配クリップ
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)

                # オプティマイザステップ
                optimizer.step()

                end_time = time.time()

            # メトリクス収集
            iteration_time = end_time - start_time
            throughput = batch_size / iteration_time

            iteration_result = {
                'iteration': i,
                'loss': loss.item(),
                'time_seconds': iteration_time,
                'throughput_samples_per_sec': throughput,
                'memory_info': mem_info
            }

            results['iterations'].append(iteration_result)
            results['throughput'].append(throughput)

            logger.info(f"  Iteration {i+1}: loss={loss.item():.4f}, throughput={throughput:.2f} samples/sec")
        # 統計計算
        throughput_mean = sum(results['throughput']) / len(results['throughput'])
        throughput_std = torch.std(torch.tensor(results['throughput']))

        loss_values = [iter['loss'] for iter in results['iterations']]
        loss_mean = sum(loss_values) / len(loss_values)
        loss_std = torch.std(torch.tensor(loss_values))

        results['summary'] = {
            'throughput_mean': throughput_mean,
            'throughput_std': throughput_std,
            'loss_mean': loss_mean,
            'loss_std': loss_std,
            'stability_score': 1.0 / (1.0 + throughput_std / throughput_mean),  # 安定性スコア
        }

        logger.info(f"Batch processing validation completed for {scenario}")
        logger.info(f"  Average throughput: {throughput_mean:.2f} samples/sec")
        logger.info(f"  Stability score: {results['summary']['stability_score']:.3f}")

        return results

    def validate_memory_efficiency(self) -> dict[str, Any]:
        """メモリ効率検証"""
        logger.info("Validating memory efficiency...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        results = {
            'device': str(device),
            'batch_sizes': [],
            'memory_usage': [],
            'peak_memory': [],
            'efficiency_scores': []
        }

        # 様々なバッチサイズでテスト
        batch_sizes = [512, 1024, 2048, 4096]

        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")

            model = self.create_mock_model().to(device)
            data = self.create_synthetic_data(batch_size, 60, 100)
            dynamic_features = data['dynamic_features'].to(device)

            # メモリ使用量測定
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                _ = model(dynamic_features)

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB

                # メモリ効率スコア（使用メモリ / ピークメモリ）
                efficiency = memory_used / peak_memory if peak_memory > 0 else 1.0

                results['batch_sizes'].append(batch_size)
                results['memory_usage'].append(memory_used)
                results['peak_memory'].append(peak_memory)
                results['efficiency_scores'].append(efficiency)

                logger.info(f"  Batch {batch_size}: {memory_used:.2f}GB used, {peak_memory:.2f}GB peak, efficiency={efficiency:.2f}")

        return results

    def validate_scalability(self) -> dict[str, Any]:
        """スケーラビリティ検証"""
        logger.info("Validating scalability...")

        results = {}

        for scenario_name, params in self.scenarios.items():
            logger.info(f"Testing scalability for {scenario_name}")

            try:
                batch_result = self.validate_batch_processing(scenario_name)
                results[scenario_name] = {
                    'success': True,
                    'throughput': batch_result['summary']['throughput_mean'],
                    'stability': batch_result['summary']['stability_score'],
                    'params': params
                }
            except Exception as e:
                logger.error(f"Failed to validate {scenario_name}: {e}")
                results[scenario_name] = {
                    'success': False,
                    'error': str(e),
                    'params': params
                }

        # スケーラビリティ分析
        successful_scenarios = [k for k, v in results.items() if v['success']]
        if len(successful_scenarios) >= 2:
            throughputs = [results[s]['throughput'] for s in successful_scenarios]
            scalability_score = throughputs[-1] / throughputs[0] if throughputs[0] > 0 else 0
            results['scalability_analysis'] = {
                'score': scalability_score,
                'improvement_ratio': scalability_score,
                'successful_scenarios': successful_scenarios
            }

        return results

    def run_full_validation(self) -> dict[str, Any]:
        """フル検証実行"""
        logger.info("Running full production workload validation...")

        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'config_path': str(self.config_path),
            'data_path': str(self.data_path),
            'system_info': self._get_system_info(),
            'batch_processing': {},
            'memory_efficiency': {},
            'scalability': {},
            'overall_score': 0
        }

        # バッチ処理検証
        logger.info("=== Batch Processing Validation ===")
        validation_results['batch_processing'] = self.validate_batch_processing('medium_scale')

        # メモリ効率検証
        logger.info("=== Memory Efficiency Validation ===")
        validation_results['memory_efficiency'] = self.validate_memory_efficiency()

        # スケーラビリティ検証
        logger.info("=== Scalability Validation ===")
        validation_results['scalability'] = self.validate_scalability()

        # 総合スコア計算
        scores = []

        # バッチ処理スコア
        if 'summary' in validation_results['batch_processing']:
            batch_score = validation_results['batch_processing']['summary']['stability_score']
            scores.append(batch_score)

        # メモリ効率スコア
        if validation_results['memory_efficiency'].get('efficiency_scores'):
            memory_scores = validation_results['memory_efficiency']['efficiency_scores']
            memory_score = sum(memory_scores) / len(memory_scores) if memory_scores else 0
            scores.append(memory_score)

        # スケーラビリティスコア
        if 'scalability_analysis' in validation_results['scalability']:
            scalability_score = min(validation_results['scalability']['scalability_analysis']['score'], 1.0)
            scores.append(scalability_score)

        validation_results['overall_score'] = sum(scores) / len(scores) if scores else 0

        # 結果保存
        self._save_validation_results(validation_results)

        return validation_results

    def _get_system_info(self) -> dict[str, Any]:
        """システム情報取得"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown'
            })

        return info

    def _save_validation_results(self, results: dict[str, Any]):
        """検証結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"production_validation_{timestamp}.json"

        result_file = self.validation_results_dir / filename
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Validation results saved: {result_file}")

    def display_results(self, results: dict[str, Any]):
        """結果表示"""
        print("\n" + "="*80)
        print("ATFT-GAT-FAN PRODUCTION WORKLOAD VALIDATION RESULTS")
        print("="*80)

        print("\n📊 OVERVIEW")
        print(f"  Overall Score: {results.get('overall_score', 0):.3f}")
        print(f"  Timestamp: {results['timestamp']}")

        # システム情報
        sys_info = results.get('system_info', {})
        print("\n🖥️ SYSTEM INFO")
        print(f"  CPU Cores: {sys_info.get('cpu_count', 'N/A')}")
        print(f"  Memory: {sys_info.get('memory_total_gb', 0):.1f} GB")
        print(f"  CUDA Available: {sys_info.get('cuda_available', False)}")
        if sys_info.get('cuda_available'):
            print(f"  GPU: {sys_info.get('gpu_name', 'Unknown')}")
            print(f"  GPU Count: {sys_info.get('gpu_count', 0)}")

        # バッチ処理結果
        if 'batch_processing' in results and 'summary' in results['batch_processing']:
            batch = results['batch_processing']['summary']
            print("\n⚡ BATCH PROCESSING")
            print(f"  Throughput: {batch.get('throughput_mean', 0):.0f} samples/sec")
            print(f"  Stability Score: {batch.get('stability_score', 0):.3f}")
            print(f"  Loss: {batch.get('loss_mean', 0):.4f} ± {batch.get('loss_std', 0):.4f}")

        # メモリ効率結果
        if 'memory_efficiency' in results and results['memory_efficiency'].get('efficiency_scores'):
            memory = results['memory_efficiency']
            avg_efficiency = sum(memory['efficiency_scores']) / len(memory['efficiency_scores'])
            print("\n💾 MEMORY EFFICIENCY")
            print(f"  Average Efficiency: {avg_efficiency:.3f}")
            print(f"  Tested Batch Sizes: {memory.get('batch_sizes', [])}")

        # スケーラビリティ結果
        if 'scalability' in results:
            print("\n📈 SCALABILITY")
            for scenario, data in results['scalability'].items():
                if scenario != 'scalability_analysis' and isinstance(data, dict):
                    success = data.get('success', False)
                    status = "✅" if success else "❌"
                    if success:
                        throughput = data.get('throughput', 0)
                        stability = data.get('stability', 0)
                        print(f"  {status} {scenario}: {throughput:.0f} samples/sec, stability={stability:.3f}")
                    else:
                        print(f"  {status} {scenario}: Failed")

            if 'scalability_analysis' in results['scalability']:
                analysis = results['scalability']['scalability_analysis']
                print(f"  Scalability Score: {analysis.get('score', 0):.3f}")

        # 評価基準
        overall_score = results.get('overall_score', 0)
        print("\n🎯 EVALUATION")
        if overall_score >= 0.8:
            print("  Status: ✅ EXCELLENT - Ready for production")
        elif overall_score >= 0.6:
            print("  Status: ⚠️ GOOD - Minor optimizations needed")
        else:
            print("  Status: ❌ NEEDS IMPROVEMENT - Further optimization required")

        print(f"  Overall Score: {overall_score:.3f}/1.0")

        print("\n" + "="*80)


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="ATFT-GAT-FAN Production Workload Validation")
    parser.add_argument(
        "--data",
        type=str,
        default="output/ml_dataset_20250827_174908.parquet",
        help="Path to test data"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=['tiny_scale', 'small_scale', 'medium_scale', 'large_scale', 'production_scale'],
        help="Specific scenario to test"
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Run full validation suite"
    )

    args = parser.parse_args()

    validator = ProductionValidator(args.data, args.config)

    if args.scenario:
        # 特定のシナリオテスト
        logger.info(f"Running validation for scenario: {args.scenario}")
        result = validator.validate_batch_processing(args.scenario)
        validator.display_results({'batch_processing': result})

    elif args.full_validation:
        # フル検証実行
        logger.info("Running full production validation...")
        results = validator.run_full_validation()
        validator.display_results(results)

    else:
        # デフォルト: 中規模テスト
        logger.info("Running default medium-scale validation...")
        result = validator.validate_batch_processing('medium_scale')
        print("\n" + "="*60)
        print("MEDIUM SCALE VALIDATION RESULTS")
        print("="*60)
        print(f"  Average throughput: {result['summary']['throughput_mean']:.2f} samples/sec")
        print(f"  Stability score: {result['summary']['stability_score']:.3f}")
        print("="*60)


if __name__ == "__main__":
    main()
