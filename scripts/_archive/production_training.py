#!/usr/bin/env python3
# DEPRECATED: Moved to scripts/_archive on 2025-09-04
"""
ATFT-GAT-FAN Production Training Script
本番環境での完全トレーニング実行スクリプト
"""

import os
import sys
import torch
import wandb
import logging
from pathlib import Path
from types import SimpleNamespace

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionTrainer:
    """本番トレーニングクラス"""

    def __init__(self, data_path: str, config_path: str = None):
        self.data_path = Path(data_path)
        self.config_path = config_path or (project_root / "configs" / "atft" / "config.yaml")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 最適化されたハイパーパラメータ
        self.hyperparams = {
            'freq_dropout_p': 0.2,
            'ema_decay': 0.995,
            'gat_temperature': 0.8,
            'huber_delta': 0.01,
            'learning_rate': 5e-05,
            'batch_size': 64,  # 小さめのバッチサイズで安定性を確保
            'max_epochs': 50,
            'warmup_steps': 1500,
            'scheduler_gamma': 0.98
        }

        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    def dict_to_obj(self, d):
        """辞書をオブジェクトに変換するヘルパー関数"""
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = self.dict_to_obj(value)
            return SimpleNamespace(**d)
        return d

    def load_and_preprocess_data(self):
        """データ読み込みと前処理"""
        logger.info(f"Loading data from {self.data_path}")

        # データ読み込み
        df = pd.read_parquet(self.data_path)
        logger.info(f"Data shape: {df.shape}")

        # 欠損値処理
        df = df.ffill().bfill().fillna(0)
        logger.info(f"Missing values after imputation: {df.isnull().sum().sum()}")

        # 特徴量とターゲットの分離
        feature_cols = [col for col in df.columns if col not in ['Code', 'Date', 'row_idx']]
        target_cols = ['returns_1d', 'returns_5d', 'returns_10d', 'returns_20d']

        # 数値特徴量のみを使用
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Using {len(numeric_cols)} numeric features")

        # 正規化
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # 銘柄ごとのデータ分割
        stocks = df['Code'].unique()
        train_stocks = stocks[:4]  # 4銘柄をトレーニング用
        val_stocks = stocks[4:] if len(stocks) > 4 else stocks[:1]  # 1銘柄を検証用

        train_data = df[df['Code'].isin(train_stocks)]
        val_data = df[df['Code'].isin(val_stocks)]

        logger.info(f"Train stocks: {len(train_stocks)}, Val stocks: {len(val_stocks)}")
        logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

        return train_data, val_data, numeric_cols, target_cols

    def create_sequences(self, data, feature_cols, target_cols, seq_length=20):
        """時系列シーケンス作成"""
        sequences = []
        targets = []

        # 銘柄ごとにシーケンス作成
        for stock_code in data['Code'].unique():
            stock_data = data[data['Code'] == stock_code].sort_values('Date')

            if len(stock_data) < seq_length + 1:
                continue

            features = stock_data[feature_cols].values
            targets_data = stock_data[target_cols].values

            for i in range(len(stock_data) - seq_length):
                seq_features = features[i:i+seq_length]
                seq_targets = targets_data[i+seq_length]

                sequences.append(seq_features)
                targets.append(seq_targets)

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        logger.info(f"Created {len(sequences)} sequences of length {seq_length}")

        return torch.tensor(sequences), torch.tensor(targets)

    def initialize_wandb(self):
        """W&B初期化"""
        # W&B APIキーの確認
        api_key = os.getenv('WANDB_API_KEY')
        if not api_key:
            logger.warning("WANDB_API_KEY not set, W&B logging disabled")
            return None

        try:
            wandb.login(key=api_key)

            run = wandb.init(
                project='ATFT-GAT-FAN',
                name=f'production_training_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}',
                config={
                    **self.hyperparams,
                    'data_path': str(self.data_path),
                    'device': str(self.device),
                    'pytorch_version': torch.__version__,
                    'cuda_available': torch.cuda.is_available()
                },
                tags=['production', 'atft-gat-fan', 'optimized']
            )

            logger.info(f"✅ W&B initialized: {run.name}")
            logger.info(f"🌐 URL: {run.url}")

            return run

        except Exception as e:
            logger.error(f"W&B initialization failed: {e}")
            return None

    def train(self):
        """トレーニング実行"""
        logger.info("🚀 Starting ATFT-GAT-FAN production training")

        # W&B初期化
        wandb_run = self.initialize_wandb()

        # データ準備
        train_data, val_data, feature_cols, target_cols = self.load_and_preprocess_data()

        # シーケンス作成
        train_sequences, train_targets = self.create_sequences(train_data, feature_cols, target_cols)
        val_sequences, val_targets = self.create_sequences(val_data, feature_cols, target_cols)

        # データセット作成
        train_dataset = torch.utils.data.TensorDataset(train_sequences, train_targets)
        val_dataset = torch.utils.data.TensorDataset(val_sequences, val_targets)

        # データローダー作成
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hyperparams['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.hyperparams['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # モデル用のconfig作成
        import yaml

        # 設定ファイル読み込み
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # data設定ファイルも読み込み
        data_config_path = project_root / "configs" / "atft" / "data" / "jpx_large_scale.yaml"
        with open(data_config_path, 'r') as f:
            data_config_dict = yaml.safe_load(f)

        # data.featuresをメイン設定に統合
        config_dict['data'] = config_dict.get('data', {})
        config_dict['data']['features'] = data_config_dict.get('features', {})

        # 必要な設定を追加・更新
        config_dict['model'] = config_dict.get('model', {})
        config_dict['model']['input_dim'] = len(feature_cols)
        config_dict['model']['hidden_size'] = 64
        config_dict['model']['num_stocks'] = train_data['Code'].nunique()
        config_dict['model']['prediction_horizons'] = [1, 5, 10, 20]

        # ATFT_GAT_FANモデルが必要とする設定項目を追加
        config_dict['model']['input_projection'] = config_dict['model'].get('input_projection', {})
        config_dict['model']['input_projection']['use_layer_norm'] = True
        config_dict['model']['input_projection']['dropout'] = 0.1

        config_dict['model']['temporal_fusion'] = config_dict['model'].get('temporal_fusion', {})
        config_dict['model']['temporal_fusion']['num_heads'] = 8
        config_dict['model']['temporal_fusion']['dropout'] = 0.1
        config_dict['model']['temporal_fusion']['num_layers'] = 2

        config_dict['model']['graph_attention'] = config_dict['model'].get('graph_attention', {})
        config_dict['model']['graph_attention']['num_heads'] = 8
        config_dict['model']['graph_attention']['dropout'] = 0.1
        config_dict['model']['graph_attention']['alpha'] = self.hyperparams['gat_temperature']

        config_dict['model']['frequency_adaptive'] = config_dict['model'].get('frequency_adaptive', {})
        config_dict['model']['frequency_adaptive']['use_fan'] = True
        config_dict['model']['frequency_adaptive']['freq_dropout_p'] = self.hyperparams['freq_dropout_p']
        config_dict['model']['frequency_adaptive']['min_width'] = 0.05
        config_dict['model']['frequency_adaptive']['max_width'] = 0.2

        config_dict['model']['prediction_head'] = config_dict['model'].get('prediction_head', {})
        config_dict['model']['prediction_head']['use_layer_scale'] = True
        config_dict['model']['prediction_head']['layer_scale_init'] = 0.1
        config_dict['model']['prediction_head']['dropout'] = 0.1

        # EMA設定
        config_dict['model']['ema'] = config_dict['model'].get('ema', {})
        config_dict['model']['ema']['decay'] = self.hyperparams['ema_decay']
        config_dict['model']['ema']['enabled'] = True

        # 追加のグローバル設定
        config_dict['model']['dropout'] = 0.1
        config_dict['model']['activation'] = 'relu'

        # dictをobjectに変換
        config = self.dict_to_obj(config_dict)

        # モデル初期化
        model = ATFT_GAT_FAN(config).to(self.device)

        # オプティマイザと損失関数
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.hyperparams['learning_rate'],
            weight_decay=0.01
        )

        # Huber損失（最適化パラメータ使用）
        criterion = torch.nn.HuberLoss(delta=self.hyperparams['huber_delta'])

        # 学習率スケジューラ
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=self.hyperparams['scheduler_gamma']
        )

        # トレーニングループ
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        logger.info("🎯 Starting training loop...")

        for epoch in range(self.hyperparams['max_epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_steps = 0

            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model({
                    'dynamic_features': batch_features,
                    'static_features': None  # 簡易版
                })

                # Loss calculation (最初のホライズンのみ使用)
                predictions = outputs['predictions'][:, :, 0]  # returns_1d
                targets = batch_targets[:, 0]

                loss = criterion(predictions.squeeze(), targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                train_steps += 1

            avg_train_loss = train_loss / train_steps

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    outputs = model({
                        'dynamic_features': batch_features,
                        'static_features': None
                    })

                    predictions = outputs['predictions'][:, :, 0]
                    targets = batch_targets[:, 0]

                    loss = criterion(predictions.squeeze(), targets)
                    val_loss += loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0

            # 学習率更新
            scheduler.step()

            # ログ出力
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(".4f")

            # W&Bログ
            if wandb_run:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'learning_rate': current_lr,
                    'gpu_memory_used': torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
                    'gpu_memory_reserved': torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
                })

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # ベストモデル保存
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'hyperparams': self.hyperparams
                }, 'models/best_atft_model.pth')

                logger.info("💾 Best model saved")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break

        # トレーニング完了
        logger.info("🎉 Training completed!")

        if wandb_run:
            wandb_run.summary.update({
                'final_train_loss': avg_train_loss,
                'final_val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'total_epochs': epoch + 1,
                'early_stopped': patience_counter >= patience
            })
            wandb_run.finish()

        return {
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'epochs_completed': epoch + 1,
            'early_stopped': patience_counter >= patience
        }


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="ATFT-GAT-FAN Production Training")
    parser.add_argument(
        "--data",
        type=str,
        default="output/ml_dataset_20250827_174908.parquet",
        help="Training data file path"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )

    args = parser.parse_args()

    # 環境変数設定
    os.environ['WANDB_API_KEY'] = "e9d88303ceecaf6037cfc47d3a8fa275211b138d"

    # トレーナー初期化
    trainer = ProductionTrainer(args.data, args.config)

    # ハイパーパラメータ更新
    trainer.hyperparams['max_epochs'] = args.epochs
    trainer.hyperparams['batch_size'] = args.batch_size

    # トレーニング実行
    results = trainer.train()

    # 結果表示
    print("\n" + "="*60)
    print("🎉 ATFT-GAT-FAN PRODUCTION TRAINING COMPLETED!")
    print("="*60)
    print(".4f")
    print(".4f")
    print(".4f")
    print(f"📊 Epochs Completed: {results['epochs_completed']}")
    print(f"🛑 Early Stopped: {results['early_stopped']}")
    print("="*60)

    # W&B URL表示
    if os.getenv('WANDB_API_KEY'):
        print("🌐 View results at: https://wandb.ai/wer-inc/ATFT-GAT-FAN")


if __name__ == "__main__":
    main()
