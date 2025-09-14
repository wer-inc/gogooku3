#!/usr/bin/env python3
"""
ATFT-GAT-FAN Real Hyperparameter Tuning
実際のトレーニングに接続したハイパーパラメータ最適化
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
import yaml

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available for tracking")


class MLflowOptunaCallback:
    """MLflow統合Optunaコールバック"""
    
    def __init__(self, experiment_name: str = "ATFT-GAT-FAN-Optuna"):
        self.experiment_name = experiment_name
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
    
    def __call__(self, study, trial):
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(nested=True) as run:
                # パラメータをログ
                mlflow.log_params(trial.params)
                
                # メトリクスをログ
                mlflow.log_metric("trial_number", trial.number)
                mlflow.log_metric("sharpe_ratio", trial.value if trial.value else -999)
                
                # 中間値もログ
                for step, intermediate_value in trial.intermediate_values.items():
                    mlflow.log_metric("intermediate_value", intermediate_value, step=step)
                
                # 状態をログ
                mlflow.set_tag("trial_state", str(trial.state))
                
                if trial.value is not None:
                    mlflow.log_metric("objective_value", trial.value)


class RealHyperparameterTuner:
    """実学習接続ハイパーパラメータチューナー"""

    def __init__(
        self,
        data_path: str = "output/atft_data/train",
        base_config: str = "configs/atft/config.yaml",
        n_epochs_trial: int = 5,
        max_data_files: int = 100
    ):
        self.data_path = Path(data_path)
        self.base_config = base_config
        self.n_epochs_trial = n_epochs_trial  # トライアル用短縮エポック
        self.max_data_files = max_data_files  # データファイル制限
        self.tuning_results_dir = project_root / "tuning_results"
        self.tuning_results_dir.mkdir(exist_ok=True)
        
        # 最適化結果
        self.best_params = {}
        self.best_score = -float('inf')
        
        # MLflowコールバック
        self.mlflow_callback = MLflowOptunaCallback()

    def get_phase_search_space(self, phase: int) -> Dict[str, Any]:
        """Phase別の探索空間を定義"""
        
        if phase == 1:  # 基本パラメータ
            return {
                'lr': ('log_uniform', 1e-5, 5e-4),
                'batch_size': ('categorical', [256, 512, 1024]),
                'weight_decay': ('log_uniform', 1e-4, 5e-2),
                'dropout': ('uniform', 0.05, 0.2),
                'warmup_steps': ('int', 1000, 2500),
                'scheduler_gamma': ('uniform', 0.9, 0.99),
            }
        elif phase == 2:  # グラフパラメータ
            return {
                'graph_k': ('int', 10, 30),
                'edge_threshold': ('uniform', 0.1, 0.4),
                'ewm_halflife': ('int', 10, 60),
                'shrinkage_gamma': ('uniform', 0.05, 0.3),
                'graph_symmetric': ('categorical', [True, False]),
            }
        elif phase == 3:  # FAN/TFT融合
            return {
                'gat_alpha_init': ('uniform', 0.05, 0.4),
                'freq_dropout_p': ('uniform', 0.0, 0.2),
                'freq_dropout_max_width': ('uniform', 0.1, 0.3),
                'edge_dropout_p': ('uniform', 0.0, 0.2),
                'ema_decay': ('uniform', 0.995, 0.999),
            }
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def suggest_params(self, trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """試行パラメータを生成"""
        params = {}
        
        for param_name, (dist_type, *args) in search_space.items():
            if dist_type == 'uniform':
                params[param_name] = trial.suggest_float(param_name, args[0], args[1])
            elif dist_type == 'log_uniform':
                params[param_name] = trial.suggest_float(param_name, args[0], args[1], log=True)
            elif dist_type == 'int':
                params[param_name] = trial.suggest_int(param_name, args[0], args[1])
            elif dist_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, args[0])
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
        
        return params

    def create_hydra_overrides(self, params: Dict[str, Any]) -> List[str]:
        """Hydraオーバーライドコマンドを生成"""
        overrides = []
        
        # 基本パラメータ
        if 'lr' in params:
            overrides.append(f'train.optimizer.lr={params["lr"]}')
        if 'batch_size' in params:
            overrides.append(f'train.batch.train_batch_size={params["batch_size"]}')
        if 'weight_decay' in params:
            overrides.append(f'train.optimizer.weight_decay={params["weight_decay"]}')
        if 'warmup_steps' in params:
            overrides.append(f'train.scheduler.warmup_steps={params["warmup_steps"]}')
        if 'scheduler_gamma' in params:
            overrides.append(f'train.scheduler.gamma={params["scheduler_gamma"]}')
            
        # グラフパラメータ
        if 'graph_k' in params:
            overrides.append(f'data.graph_builder.k={params["graph_k"]}')
        if 'edge_threshold' in params:
            overrides.append(f'data.graph_builder.edge_threshold={params["edge_threshold"]}')
        if 'ewm_halflife' in params:
            overrides.append(f'data.graph_builder.ewm_halflife={params["ewm_halflife"]}')
        if 'shrinkage_gamma' in params:
            overrides.append(f'data.graph_builder.shrinkage_gamma={params["shrinkage_gamma"]}')
        if 'graph_symmetric' in params:
            overrides.append(f'data.graph_builder.symmetric={params["graph_symmetric"]}')
            
        # FAN/TFT融合
        if 'freq_dropout_p' in params:
            overrides.append(f'improvements.freq_dropout_p={params["freq_dropout_p"]}')
        if 'freq_dropout_max_width' in params:
            overrides.append(f'improvements.freq_dropout_max_width={params["freq_dropout_max_width"]}')
        if 'ema_decay' in params:
            overrides.append(f'improvements.ema_decay={params["ema_decay"]}')
            
        return overrides

    def execute_training(self, params: Dict[str, Any], trial: optuna.Trial = None) -> Dict[str, float]:
        """実際のトレーニングを実行してメトリクスを取得"""
        logger.info(f"Executing training with params: {params}")
        
        try:
            # Hydraオーバーライド生成
            overrides = self.create_hydra_overrides(params)
            
            # 短縮モード設定
            overrides.extend([
                f'train.trainer.max_epochs={self.n_epochs_trial}',
                f'data.source.data_dir={self.data_path}',
                'train.trainer.enable_progress_bar=false',  # 出力を簡潔に
                'train.trainer.enable_model_summary=false',
            ])
            
            # 環境変数設定
            env = os.environ.copy()
            env['SMOKE_DATA_MAX_FILES'] = str(self.max_data_files)
            env['MINIMAL_COLUMNS'] = '1'
            env['OPTUNA_TRIAL'] = '1'  # オプトゥナトライアル指示
            
            # MLflow無効化（ネストランが複雑になるため）
            env['MLFLOW'] = '0'
            
            # コマンド構築
            cmd = [
                'python', str(project_root / 'scripts' / 'train_atft.py'),
                '--config-path', str(project_root / 'configs' / 'atft'),
                '--config-name', 'config'
            ] + overrides
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # 実行
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1時間タイムアウト
                env=env,
                cwd=project_root
            )
            
            if result.returncode != 0:
                logger.error(f"Training failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return {'sharpe': -1.0, 'ic': 0.0, 'rankic': 0.0, 'loss': 999.0}
            
            # ログから結果を抽出
            metrics = self.parse_training_output(result.stdout, result.stderr)
            
            # 中間値報告（プルーニング用）
            if trial is not None and 'sharpe' in metrics:
                trial.report(metrics['sharpe'], step=self.n_epochs_trial)
                
                # プルーニング判定
                if trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned at step {self.n_epochs_trial}")
                    raise optuna.TrialPruned()
            
            return metrics
            
        except subprocess.TimeoutExpired:
            logger.error("Training timeout")
            return {'sharpe': -1.0, 'ic': 0.0, 'rankic': 0.0, 'loss': 999.0}
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            return {'sharpe': -1.0, 'ic': 0.0, 'rankic': 0.0, 'loss': 999.0}

    def parse_training_output(self, stdout: str, stderr: str) -> Dict[str, float]:
        """トレーニング出力からメトリクスを抽出"""
        metrics = {'sharpe': -1.0, 'ic': 0.0, 'rankic': 0.0, 'loss': 999.0, 'hit_rate': 0.0}
        
        try:
            # 最後のエポックのメトリクスを抽出
            lines = stdout.split('\n') + stderr.split('\n')
            
            for line in reversed(lines):  # 後ろから探索
                if 'Sharpe:' in line:
                    # "Sharpe: 0.0084" のような行から値を抽出
                    parts = line.split('Sharpe:')
                    if len(parts) > 1:
                        try:
                            sharpe_val = float(parts[1].strip().split()[0])
                            metrics['sharpe'] = sharpe_val
                            break
                        except (ValueError, IndexError):
                            continue
                            
            # IC/RankICも同様に抽出
            for line in reversed(lines):
                if 'Val Metrics' in line and 'IC:' in line:
                    # "Val Metrics   - Sharpe: 0.0000, IC: 0.0150, RankIC: 0.0113"
                    try:
                        parts = line.split('IC:')
                        if len(parts) > 1:
                            ic_part = parts[1].split(',')[0].strip()
                            metrics['ic'] = float(ic_part)
                            
                        if 'RankIC:' in line:
                            rankic_part = line.split('RankIC:')[1].strip()
                            metrics['rankic'] = float(rankic_part)
                        if 'HitRate' in line:
                            # HitRate(h1): 0.6123
                            try:
                                hr = line.split('HitRate')[1]
                                hr = hr.split(':')[1].strip()
                                # Remove trailing commas if any
                                hr = hr.split(',')[0].strip()
                                metrics['hit_rate'] = float(hr)
                            except Exception:
                                pass
                        break
                    except (ValueError, IndexError):
                        continue
            
            logger.info(f"Extracted metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Failed to parse training output: {e}")
            
        return metrics

    def robust_objective(self, metrics: Dict[str, float]) -> float:
        """ロバストな目的関数"""
        sharpe = metrics.get('sharpe', -1.0)
        ic = metrics.get('ic', 0.0)
        rankic = metrics.get('rankic', 0.0)
        hit_rate = metrics.get('hit_rate', 0.0)
        
        # 基本: Sharpe Ratio重視
        score = sharpe
        
        # IC Information Ratioで補正
        ic_bonus = abs(ic) * 0.1  # ICの絶対値を小さく加点
        rankic_bonus = abs(rankic) * 0.1  # RankICの絶対値を小さく加点
        
        # 複合スコア
        # Hit Rate as a small stabilizer (directional consistency)
        hr_bonus = max(0.0, hit_rate - 0.5) * 0.2  # above-random portion scaled
        total_score = score + ic_bonus + rankic_bonus + hr_bonus
        
        # 異常値チェック
        if sharpe < -0.5 or sharpe > 2.0:  # 異常なSharpe値
            total_score = -1.0
            
        logger.info(f"Objective: sharpe={sharpe:.4f}, ic={ic:.4f}, rankic={rankic:.4f} → score={total_score:.4f}")
        
        return total_score

    def optimize_phase(self, phase: int, n_trials: int = 10) -> optuna.Study:
        """特定Phase用の最適化実行"""
        logger.info(f"Starting Phase {phase} optimization with {n_trials} trials...")
        
        search_space = self.get_phase_search_space(phase)
        
        def objective(trial):
            params = self.suggest_params(trial, search_space)
            metrics = self.execute_training(params, trial)
            return self.robust_objective(metrics)
        
        # Study作成
        study_name = f"atft_gat_fan_phase_{phase}"
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            pruner=HyperbandPruner(
                min_resource=1,
                max_resource=self.n_epochs_trial,
                reduction_factor=3
            ),
            storage=f"sqlite:///{self.tuning_results_dir}/optuna_phase_{phase}.db",
            load_if_exists=True
        )
        
        # 最適化実行
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[self.mlflow_callback]
        )
        
        # 結果保存
        best_params = study.best_trial.params
        best_score = study.best_trial.value
        
        result = {
            'phase': phase,
            'method': 'optuna',
            'n_trials': n_trials,
            'best_params': best_params,
            'best_score': best_score,
            'study_name': study_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # JSON保存
        result_file = self.tuning_results_dir / f"optuna_phase_{phase}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Phase {phase} completed. Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
        
        return study

    def create_optimized_config(self, phase_results: List[Dict]) -> str:
        """最適化結果を統合して新しい設定ファイルを作成"""
        # ベース設定を読み込み
        with open(self.base_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 各Phaseの最適パラメータを統合
        all_best_params = {}
        for result in phase_results:
            all_best_params.update(result['best_params'])
        
        # 設定に反映
        self.apply_params_to_config(config, all_best_params)
        
        # 新しい設定ファイルを保存
        output_file = self.tuning_results_dir / "optimized_config.yaml"
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Optimized config saved: {output_file}")
        return str(output_file)

    def apply_params_to_config(self, config: Dict, params: Dict[str, Any]):
        """パラメータを設定辞書に適用"""
        # Nested dictionary helper
        def set_nested(d, path, value):
            keys = path.split('.')
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = value
        
        # パラメータマッピング
        param_mapping = {
            'lr': 'train.optimizer.lr',
            'batch_size': 'train.batch.train_batch_size',
            'weight_decay': 'train.optimizer.weight_decay',
            'warmup_steps': 'train.scheduler.warmup_steps',
            'scheduler_gamma': 'train.scheduler.gamma',
            'graph_k': 'data.graph_builder.k',
            'edge_threshold': 'data.graph_builder.edge_threshold',
            'ewm_halflife': 'data.graph_builder.ewm_halflife',
            'shrinkage_gamma': 'data.graph_builder.shrinkage_gamma',
            'graph_symmetric': 'data.graph_builder.symmetric',
            'freq_dropout_p': 'improvements.freq_dropout_p',
            'freq_dropout_max_width': 'improvements.freq_dropout_max_width',
            'ema_decay': 'improvements.ema_decay',
        }
        
        for param, value in params.items():
            if param in param_mapping:
                set_nested(config, param_mapping[param], value)


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="ATFT-GAT-FAN Real Hyperparameter Tuning")
    parser.add_argument(
        "--data-path",
        type=str,
        default="output/atft_data/train",
        help="Path to training data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/atft/config.yaml",
        help="Base config file"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="Specific phase to optimize (1=basic, 2=graph, 3=fusion)"
    )
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="Run all phases sequentially"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per phase"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for trial runs"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Maximum number of data files for trials"
    )

    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        logger.error("Optuna is required. Install with: pip install optuna")
        return 1

    # チューナー初期化
    tuner = RealHyperparameterTuner(
        data_path=args.data_path,
        base_config=args.config,
        n_epochs_trial=args.epochs,
        max_data_files=args.max_files
    )

    phase_results = []

    if args.all_phases:
        # 全Phase実行
        for phase in [1, 2, 3]:
            logger.info(f"Starting Phase {phase}")
            study = tuner.optimize_phase(phase, args.trials)
            
            result = {
                'phase': phase,
                'best_params': study.best_trial.params,
                'best_score': study.best_trial.value
            }
            phase_results.append(result)
    
    elif args.phase:
        # 特定Phase実行
        study = tuner.optimize_phase(args.phase, args.trials)
        
        result = {
            'phase': args.phase,
            'best_params': study.best_trial.params,
            'best_score': study.best_trial.value
        }
        phase_results.append(result)
    
    else:
        logger.error("Please specify --phase or --all-phases")
        return 1

    # 結果統合
    if phase_results:
        if len(phase_results) > 1:
            optimized_config = tuner.create_optimized_config(phase_results)
            print(f"\n✅ ALL PHASES COMPLETED!")
            print(f"Optimized config saved: {optimized_config}")
        
        print(f"\n🏆 OPTIMIZATION RESULTS:")
        for result in phase_results:
            print(f"Phase {result['phase']}: Score {result['best_score']:.4f}")
            print(f"  Best params: {result['best_params']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
