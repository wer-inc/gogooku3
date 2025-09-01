#!/usr/bin/env python3
"""
ATFT-GAT-FAN Hyperparameter Tuning
ハイパーパラメータ最適化スクリプト
"""

import os
import sys
import json
import logging
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")


class HyperparameterTuner:
    """ハイパーパラメータチューナークラス"""

    def __init__(self, data_path: str, config_path: Optional[str] = None):
        self.data_path = Path(data_path)
        self.config_path = config_path or (project_root / "configs" / "atft" / "config.yaml")
        self.tuning_results_dir = project_root / "tuning_results"
        self.tuning_results_dir.mkdir(exist_ok=True)

        # チューニング対象パラメータ
        self.tuning_params = {
            'freq_dropout_p': [0.0, 0.05, 0.1, 0.15, 0.2],
            'ema_decay': [0.95, 0.97, 0.99, 0.995, 0.999],
            'gat_temperature': [0.7, 0.8, 1.0, 1.2, 1.3],
            'huber_delta': [0.001, 0.01, 0.05, 0.1],
            'lr_scheduler_warmup_steps': [1000, 1500, 2000, 2500],
            'lr_scheduler_gamma': [0.9, 0.95, 0.98, 0.99]
        }

        # 評価指標
        self.metrics = ['rankic_h1', 'rankic_h5', 'loss', 'training_time']

        # ベストパラメータ
        self.best_params = {}
        self.best_score = -float('inf')

    def load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def create_config_for_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータに基づいて設定を作成"""
        config = self.load_config()

        # パラメータ適用
        if 'freq_dropout_p' in params:
            config['improvements']['freq_dropout_p'] = params['freq_dropout_p']
        if 'ema_decay' in params:
            config['improvements']['ema_decay'] = params['ema_decay']
        if 'gat_temperature' in params:
            config['improvements']['gat_temperature'] = params['gat_temperature']
        if 'huber_delta' in params:
            config['improvements']['huber_delta'] = params['huber_delta']

        # 学習率スケジューラパラメータ
        if 'lr_scheduler_warmup_steps' in params:
            if 'train' not in config:
                config['train'] = {}
            if 'optimizer' not in config['train']:
                config['train']['optimizer'] = {}
            config['train']['optimizer']['warmup_steps'] = params['lr_scheduler_warmup_steps']

        if 'lr_scheduler_gamma' in params:
            if 'train' not in config:
                config['train'] = {}
            if 'scheduler' not in config['train']:
                config['train']['scheduler'] = {}
            config['train']['scheduler']['gamma'] = params['lr_scheduler_gamma']

        return config

    def evaluate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータを評価"""
        logger.info(f"Evaluating parameters: {params}")

        try:
            # 設定作成
            config = self.create_config_for_params(params)

            # 簡易評価（実際のトレーニングの代わりに）
            # 本番ではここで実際のモデルトレーニングを実行
            score = self._mock_evaluation(params)

            result = {
                'params': params,
                'score': score,
                'metrics': {
                    'rankic_h1': score * 0.15 + np.random.normal(0, 0.01),
                    'rankic_h5': score * 0.12 + np.random.normal(0, 0.01),
                    'loss': 0.05 - score * 0.02 + np.random.normal(0, 0.005),
                    'training_time': 10.0 - score * 0.5 + np.random.normal(0, 0.2)
                },
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Evaluation result: score={score:.4f}")
            return result

        except Exception as e:
            logger.error(f"Parameter evaluation failed: {e}")
            return {
                'params': params,
                'score': -1.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _mock_evaluation(self, params: Dict[str, Any]) -> float:
        """モック評価関数（実際のトレーニングの代用）"""
        # パラメータに基づいてスコアを計算
        score = 0.0

        # FreqDropout: 0.1付近が最適
        if 'freq_dropout_p' in params:
            dropout = params['freq_dropout_p']
            score += 1.0 - abs(dropout - 0.1) * 2

        # EMA decay: 0.999付近が最適
        if 'ema_decay' in params:
            ema = params['ema_decay']
            score += 1.0 - abs(ema - 0.999) * 5

        # GAT温度: 1.0付近が最適
        if 'gat_temperature' in params:
            temp = params['gat_temperature']
            score += 1.0 - abs(temp - 1.0) * 2

        # Huber delta: 0.01付近が最適
        if 'huber_delta' in params:
            delta = params['huber_delta']
            score += 1.0 - abs(delta - 0.01) * 10

        # 学習率スケジューラ: 1500-2000ステップが最適
        if 'lr_scheduler_warmup_steps' in params:
            warmup = params['lr_scheduler_warmup_steps']
            optimal_warmup = 1750
            score += 1.0 - abs(warmup - optimal_warmup) / 1000

        # 正規化とノイズ
        score = score / 5.0  # 平均化
        score += np.random.normal(0, 0.1)  # ノイズ追加
        score = np.clip(score, 0.0, 1.0)  # 0-1にクリップ

        return score

    def grid_search(self) -> Dict[str, Any]:
        """グリッドサーチ実行"""
        logger.info("Starting grid search...")

        # パラメータの組み合わせを生成
        param_keys = list(self.tuning_params.keys())
        param_values = [self.tuning_params[key] for key in param_keys]

        all_combinations = list(itertools.product(*param_values))
        logger.info(f"Total combinations to evaluate: {len(all_combinations)}")

        results = []

        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_keys, combination))
            logger.info(f"Evaluating combination {i+1}/{len(all_combinations)}")

            result = self.evaluate_params(params)
            results.append(result)

            # ベストパラメータ更新
            if result['score'] > self.best_score:
                self.best_score = result['score']
                self.best_params = result['params'].copy()
                logger.info(f"New best score: {self.best_score:.4f}")

        # 結果を保存
        tuning_result = {
            'method': 'grid_search',
            'total_combinations': len(all_combinations),
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': results,
            'timestamp': datetime.now().isoformat()
        }

        self._save_results(tuning_result)
        return tuning_result

    def random_search(self, n_trials: int = 50) -> Dict[str, Any]:
        """ランダムサーチ実行"""
        logger.info(f"Starting random search with {n_trials} trials...")

        results = []
        np.random.seed(42)

        for i in range(n_trials):
            # ランダムパラメータ生成
            params = {}
            for param_name, param_values in self.tuning_params.items():
                params[param_name] = np.random.choice(param_values)

            logger.info(f"Random trial {i+1}/{n_trials}: {params}")

            result = self.evaluate_params(params)
            results.append(result)

            # ベストパラメータ更新
            if result['score'] > self.best_score:
                self.best_score = result['score']
                self.best_params = result['params'].copy()
                logger.info(f"New best score: {self.best_score:.4f}")

        # 結果を保存
        tuning_result = {
            'method': 'random_search',
            'n_trials': n_trials,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': results,
            'timestamp': datetime.now().isoformat()
        }

        self._save_results(tuning_result)
        return tuning_result

    def optuna_optimization(self, n_trials: int = 50) -> Dict[str, Any]:
        """Optuna最適化実行"""
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not available")
            return {}

        logger.info(f"Starting Optuna optimization with {n_trials} trials...")

        def objective(trial):
            params = {
                'freq_dropout_p': trial.suggest_categorical('freq_dropout_p', self.tuning_params['freq_dropout_p']),
                'ema_decay': trial.suggest_categorical('ema_decay', self.tuning_params['ema_decay']),
                'gat_temperature': trial.suggest_categorical('gat_temperature', self.tuning_params['gat_temperature']),
                'huber_delta': trial.suggest_categorical('huber_delta', self.tuning_params['huber_delta']),
                'lr_scheduler_warmup_steps': trial.suggest_categorical('lr_scheduler_warmup_steps', self.tuning_params['lr_scheduler_warmup_steps']),
                'lr_scheduler_gamma': trial.suggest_categorical('lr_scheduler_gamma', self.tuning_params['lr_scheduler_gamma'])
            }

            result = self.evaluate_params(params)
            return result['score']

        # Optunaスタディ作成
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # 結果整理
        best_trial = study.best_trial
        tuning_result = {
            'method': 'optuna',
            'n_trials': n_trials,
            'best_params': best_trial.params,
            'best_score': best_trial.value,
            'all_trials': [
                {
                    'trial_number': trial.number,
                    'params': trial.params,
                    'value': trial.value
                }
                for trial in study.trials
            ],
            'timestamp': datetime.now().isoformat()
        }

        # ベストパラメータ更新
        self.best_params = best_trial.params
        self.best_score = best_trial.value

        self._save_results(tuning_result)
        return tuning_result

    def _save_results(self, result: Dict[str, Any]):
        """結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tuning_result_{result['method']}_{timestamp}.json"

        # numpyデータ型をPython標準型に変換
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        result_converted = convert_numpy_types(result)

        result_file = self.tuning_results_dir / filename
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_converted, f, indent=2, ensure_ascii=False)

        logger.info(f"Tuning results saved: {result_file}")

    def get_best_config(self) -> Dict[str, Any]:
        """最適パラメータを適用した設定を取得"""
        if not self.best_params:
            logger.warning("No best parameters found. Using default.")
            return self.load_config()

        config = self.create_config_for_params(self.best_params)
        return config

    def save_best_config(self):
        """最適パラメータを適用した設定を保存"""
        config = self.get_best_config()

        best_config_file = self.tuning_results_dir / "best_config.yaml"
        import yaml
        with open(best_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Best configuration saved: {best_config_file}")
        return best_config_file

    def display_results(self, result: Dict[str, Any]):
        """結果を表示"""
        print("\n" + "="*80)
        print(f"ATFT-GAT-FAN HYPERPARAMETER TUNING RESULTS ({result['method'].upper()})")
        print("="*80)

        if 'best_params' in result:
            print(f"\n🏆 BEST PARAMETERS (Score: {result.get('best_score', 'N/A'):.4f})")
            for param, value in result['best_params'].items():
                print(f"  {param}: {value}")

        if 'total_combinations' in result:
            print(f"\n📊 SEARCH STATISTICS")
            print(f"  Total Combinations: {result['total_combinations']}")

        if 'n_trials' in result:
            print(f"  Number of Trials: {result['n_trials']}")

        print(f"  Best Score: {result.get('best_score', 'N/A')}")
        print(f"  Timestamp: {result['timestamp']}")

        # パラメータ重要度分析（簡易）
        if 'all_results' in result and result['all_results']:
            self._analyze_parameter_importance(result['all_results'])

        print("\n" + "="*80)

    def _analyze_parameter_importance(self, results: List[Dict[str, Any]]):
        """パラメータ重要度の簡易分析"""
        print(f"\n🔍 PARAMETER IMPORTANCE ANALYSIS")

        # 各パラメータの相関係数を計算
        param_scores = {}
        for param_name in self.tuning_params.keys():
            param_values = []
            scores = []

            for result in results:
                if param_name in result['params']:
                    param_values.append(result['params'][param_name])
                    scores.append(result['score'])

            if param_values and scores:
                # 数値パラメータの場合は相関係数
                try:
                    if all(isinstance(v, (int, float)) for v in param_values):
                        correlation = np.corrcoef(param_values, scores)[0, 1]
                        param_scores[param_name] = abs(correlation)
                    else:
                        # カテゴリパラメータの場合は平均スコア差
                        unique_values = list(set(param_values))
                        if len(unique_values) > 1:
                            avg_scores = {}
                            for val in unique_values:
                                val_scores = [s for p, s in zip(param_values, scores) if p == val]
                                avg_scores[val] = np.mean(val_scores) if val_scores else 0

                            max_avg = max(avg_scores.values())
                            min_avg = min(avg_scores.values())
                            param_scores[param_name] = (max_avg - min_avg) / (max_avg + min_avg + 1e-10)
                except:
                    param_scores[param_name] = 0

        # 重要度順にソートして表示
        if param_scores:
            sorted_params = sorted(param_scores.items(), key=lambda x: x[1], reverse=True)
            print("  Parameter Importance (higher = more important):")
            for param, importance in sorted_params[:5]:  # Top 5
                print("3.1f")


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="ATFT-GAT-FAN Hyperparameter Tuning")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['grid', 'random', 'optuna'],
        default='grid',
        help="Tuning method"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials for random/optuna search"
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save best configuration"
    )

    args = parser.parse_args()

    # チューナー初期化
    tuner = HyperparameterTuner(args.data, args.config)

    # チューニング実行
    if args.method == 'grid':
        logger.info("Running grid search...")
        result = tuner.grid_search()
    elif args.method == 'random':
        logger.info(f"Running random search with {args.trials} trials...")
        result = tuner.random_search(args.trials)
    elif args.method == 'optuna':
        logger.info(f"Running Optuna optimization with {args.trials} trials...")
        result = tuner.optuna_optimization(args.trials)
    else:
        logger.error(f"Unknown method: {args.method}")
        return 1

    # 結果表示
    tuner.display_results(result)

    # ベスト設定保存
    if args.save_best and tuner.best_params:
        tuner.save_best_config()

    # 推奨設定の表示
    if tuner.best_params:
        print("\n✅ TUNING COMPLETED!")
        print(f"Best Score: {tuner.best_score:.4f}")
        print("Best Parameters:")
        for param, value in tuner.best_params.items():
            print(f"  {param}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
