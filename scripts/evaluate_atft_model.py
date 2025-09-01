#!/usr/bin/env python3
"""
ATFT-GAT-FAN Model Evaluation Script
632銘柄モデルの性能評価を実施
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN
import wandb

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ATFTModelEvaluator:
    """ATFT-GAT-FANモデルの評価クラス"""

    def __init__(self, model_path: str = None, data_path: str = None):
        self.model_path = model_path or "models/best_atft_model.pth"
        self.data_path = data_path or "output/ml_dataset_production.parquet"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 評価設定
        self.batch_size = 256
        self.sequence_length = 20

        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def load_model(self) -> Tuple[ATFT_GAT_FAN, Dict]:
        """モデルの読み込み"""
        logger.info(f"Loading model from {self.model_path}")

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # モデル設定の作成（トレーニング時と同じ）
        import yaml
        from types import SimpleNamespace

        config_path = project_root / "configs" / "atft" / "config.yaml"
        data_config_path = project_root / "configs" / "atft" / "data" / "jpx_large_scale.yaml"

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        with open(data_config_path, 'r') as f:
            data_config_dict = yaml.safe_load(f)

        config_dict['data'] = config_dict.get('data', {})
        config_dict['data']['features'] = data_config_dict.get('features', {})

        # モデルパラメータ設定
        config_dict['model'] = config_dict.get('model', {})
        config_dict['model']['input_dim'] = 64
        config_dict['model']['hidden_size'] = 64
        config_dict['model']['num_stocks'] = 632
        config_dict['model']['prediction_horizons'] = [1, 5, 10, 20]

        # ATFT_GAT_FAN設定
        config_dict['model']['input_projection'] = {'use_layer_norm': True, 'dropout': 0.1}
        config_dict['model']['tft'] = {'num_heads': 8, 'dropout': 0.1, 'num_layers': 2}
        config_dict['model']['temporal_fusion'] = {'num_heads': 8, 'dropout': 0.1, 'num_layers': 2}
        config_dict['model']['graph_attention'] = {'num_heads': 8, 'dropout': 0.1, 'alpha': 0.8}
        config_dict['model']['gat'] = {
            'enabled': True,
            'num_heads': 8,
            'dropout': 0.1,
            'alpha': 0.8,
            'architecture': {'heads': [8]},
            'layer_config': {'dropout': 0.1}
        }
        config_dict['model']['frequency_adaptive'] = {'use_fan': True, 'freq_dropout_p': 0.2}
        config_dict['model']['adaptive_normalization'] = {
            'enabled': True,
            'type': 'layer_norm',
            'fan': {
                'enabled': True,
                'type': 'frequency_adaptive',
                'window_sizes': [5, 10, 20, 60]
            }
        }
        config_dict['model']['prediction_head'] = {'use_layer_scale': True, 'layer_scale_init': 0.1, 'dropout': 0.1}
        config_dict['model']['ema'] = {'decay': 0.995, 'enabled': True}
        config_dict['model']['dropout'] = 0.1
        config_dict['model']['activation'] = 'relu'

        # LSTM設定（TemporalFusionTransformer用）
        config_dict['lstm'] = {'layers': 2, 'dropout': 0.1, 'hidden_size': 64}

        # TFT設定のLSTM情報を含める
        config_dict['model']['tft']['lstm'] = {'layers': 2, 'dropout': 0.1, 'hidden_size': 64}

        # GAT設定（GraphAttentionNetwork用）
        config_dict['architecture'] = {'heads': [8]}
        config_dict['layer_config'] = {'dropout': 0.1}

        # FAN設定（AdaptiveNormalization用）
        config_dict['fan'] = {
            'enabled': True,
            'type': 'frequency_adaptive',
            'window_sizes': [5, 10, 20, 60]
        }

        # SAN設定（AdaptiveNormalization用）
        config_dict['san'] = {'enabled': False, 'type': 'slice_adaptive'}

        def dict_to_obj(d):
            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(value, dict):
                        d[key] = dict_to_obj(value)
                return SimpleNamespace(**d)
            return d

        config = dict_to_obj(config_dict)

        # モデル初期化
        model = ATFT_GAT_FAN(config).to(self.device)

        # チェックポイント読み込み
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✅ Model loaded from checkpoint")
            return model, checkpoint
        else:
            raise ValueError("Invalid checkpoint format")

    def load_evaluation_data(self) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """評価用データの読み込み"""
        logger.info(f"Loading evaluation data from {self.data_path}")

        df = pd.read_parquet(self.data_path)

        # 最新のデータをテスト用として使用（最後の30日分）
        df_sorted = df.sort_values(['Code', 'Date'])
        test_stocks = df['Code'].unique()[:50]  # 最初の50銘柄をテスト用に

        test_df = df_sorted[df_sorted['Code'].isin(test_stocks)]
        test_df = test_df.groupby('Code').tail(30)  # 各銘柄の最新30日分

        logger.info(f"Test data: {len(test_df)} records, {len(test_stocks)} stocks")

        return test_df

    def create_evaluation_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """評価用のシーケンス作成"""
        sequences = []
        targets = []

        for stock_code in df['Code'].unique():
            stock_data = df[df['Code'] == stock_code].sort_values('Date')

            if len(stock_data) < self.sequence_length + 1:
                continue

            # 特徴量の選択（ATFT変換後の特徴量を使用）
            feature_cols = [col for col in stock_data.columns if col.startswith('feat_') or col in ['open', 'high', 'low', 'close', 'volume']]
            if len(feature_cols) > 64:  # モデルの入力次元に合わせる
                feature_cols = feature_cols[:64]

            target_cols = ['returns_1d', 'returns_5d', 'returns_10d', 'returns_20d']

            if not all(col in stock_data.columns for col in target_cols):
                continue

            features = stock_data[feature_cols].fillna(0).values
            targets_data = stock_data[target_cols].fillna(0).values

            # シーケンス作成
            for i in range(len(stock_data) - self.sequence_length):
                seq_features = features[i:i+self.sequence_length]
                seq_targets = targets_data[i+self.sequence_length]

                sequences.append(seq_features)
                targets.append(seq_targets)

        if not sequences:
            raise ValueError("No valid sequences created")

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        logger.info(f"Created {len(sequences)} evaluation sequences")

        return torch.tensor(sequences), torch.tensor(targets)

    def evaluate_model(self, model: ATFT_GAT_FAN, sequences: torch.Tensor, targets: torch.Tensor) -> Dict:
        """モデルの評価"""
        logger.info("Starting model evaluation...")

        model.eval()
        predictions = []
        actuals = []

        dataset = torch.utils.data.TensorDataset(sequences, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(self.device)

                # モデル推論
                outputs = model({
                    'dynamic_features': batch_features,
                    'static_features': None
                })

                # returns_1dの予測を取得
                preds = outputs['predictions'][:, :, 0].cpu().numpy()  # [batch, seq, horizon] -> [batch]
                actual = batch_targets[:, 0].cpu().numpy()  # returns_1d

                predictions.extend(preds)
                actuals.extend(actual)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # 評価指標計算
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))

        # 相関係数
        correlation = np.corrcoef(predictions, actuals)[0, 1]

        # Rank IC (予測と実績のリターンの順位相関)
        pred_ranks = predictions.argsort().argsort()
        actual_ranks = actuals.argsort().argsort()
        rank_ic = np.corrcoef(pred_ranks, actual_ranks)[0, 1]

        # Sharpe比計算（簡易版）
        if len(predictions) > 1:
            returns = predictions - actuals.mean()  # 超過リターンの簡易計算
            sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)  # 年率化
        else:
            sharpe = 0.0

        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'rank_ic': rank_ic,
            'sharpe_ratio': sharpe,
            'num_predictions': len(predictions),
            'predictions_mean': float(np.mean(predictions)),
            'predictions_std': float(np.std(predictions)),
            'actuals_mean': float(np.mean(actuals)),
            'actuals_std': float(np.std(actuals))
        }

        logger.info("✅ Model evaluation completed")
        return results

    def log_to_wandb(self, results: Dict, model_info: Dict = None):
        """W&Bへの結果ログ"""
        try:
            wandb.init(
                project='ATFT-GAT-FAN',
                name=f'evaluation_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}',
                config={'evaluation_type': '632_stocks_model'}
            )

            # 評価結果をログ
            wandb.log({
                'eval_mse': results['mse'],
                'eval_rmse': results['rmse'],
                'eval_mae': results['mae'],
                'eval_correlation': results['correlation'],
                'eval_rank_ic': results['rank_ic'],
                'eval_sharpe_ratio': results['sharpe_ratio'],
                'eval_num_predictions': results['num_predictions'],
                'eval_predictions_mean': results['predictions_mean'],
                'eval_predictions_std': results['predictions_std'],
                'eval_actuals_mean': results['actuals_mean'],
                'eval_actuals_std': results['actuals_std']
            })

            wandb.finish()
            logger.info("✅ Results logged to W&B")

        except Exception as e:
            logger.warning(f"W&B logging failed: {e}")

    def run_evaluation(self) -> Dict:
        """評価実行"""
        logger.info("🚀 Starting ATFT-GAT-FAN model evaluation")

        try:
            # モデル読み込み
            model, checkpoint = self.load_model()

            # データ読み込み
            test_df = self.load_evaluation_data()

            # シーケンス作成
            sequences, targets = self.create_evaluation_sequences(test_df)

            # 評価実行
            results = self.evaluate_model(model, sequences, targets)

            # 結果表示
            print("\n" + "="*60)
            print("🎯 ATFT-GAT-FAN 632銘柄モデル評価結果")
            print("="*60)
            print(".6f")
            print(".6f")
            print(".6f")
            print(".4f")
            print(".4f")
            print(".4f")
            print(f"   予測平均: {results['predictions_mean']:.6f}")
            print(f"   実績平均: {results['actuals_mean']:.6f}")
            print(f"   予測標準偏差: {results['predictions_std']:.6f}")
            print(f"   実績標準偏差: {results['actuals_std']:.6f}")
            print("="*60)

            # W&Bログ
            self.log_to_wandb(results, checkpoint)

            return results

        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="ATFT-GAT-FAN Model Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_atft_model.pth",
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="output/ml_dataset_production.parquet",
        help="Evaluation data path"
    )

    args = parser.parse_args()

    # 環境変数設定
    os.environ['WANDB_API_KEY'] = "e9d88303ceecaf6037cfc47d3a8fa275211b138d"

    # 評価実行
    evaluator = ATFTModelEvaluator(args.model, args.data)
    results = evaluator.run_evaluation()

    if results:
        print("\n🎉 評価完了！")
        print("   次のステップ: 4000銘柄データ取得と比較実験")
    else:
        print("\n❌ 評価失敗")


if __name__ == "__main__":
    main()
