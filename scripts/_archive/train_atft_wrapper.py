#!/usr/bin/env python3
# DEPRECATED: Moved to scripts/_archive on 2025-09-04. Use internal trainer:
#   python scripts/integrated_ml_training_pipeline.py
"""
Complete ATFT-GAT-FAN Training Wrapper for gogooku3
ATFT-GAT-FANの成果（Sharpe 0.849）を完全に再現する学習ラッパー
"""

import os
import subprocess
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_atft_environment():
    """ATFT-GAT-FANの成果を再現するための環境設定"""
    logger.info("🔧 Setting up ATFT-GAT-FAN environment for results reproduction...")

    # W&B APIキーの設定
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if wandb_api_key:
        os.environ['WANDB_API_KEY'] = wandb_api_key
        logger.info("✅ W&B API key configured")
    else:
        logger.warning("⚠️  W&B API key not found")

    # ATFT-GAT-FANの成果設定（Sharpe 0.849を達成した設定）
    atft_settings = {
        "USE_T_NLL": "1",
        "OUTPUT_NOISE_STD": "0.02",
        "HEAD_NOISE_STD": "0.05",
        "HEAD_NOISE_WARMUP_EPOCHS": "5",
        "GAT_ALPHA_INIT": "0.3",
        "GAT_ALPHA_MIN": "0.1",
        "GAT_ALPHA_PENALTY": "1e-3",
        "EDGE_DROPOUT_INPUT_P": "0.1",
        "DEGENERACY_GUARD": "1",
        "DEGENERACY_WARMUP_STEPS": "1000",
        "DEGENERACY_CHECK_EVERY": "200",
        "DEGENERACY_MIN_RATIO": "0.05",
        "USE_AMP": "1",
        "AMP_DTYPE": "bf16",
        "LABEL_CLIP_BPS_MAP": "1:2000,2:2000,3:2000,5:2000,10:5000",
        "NUM_WORKERS": "16",
        "PREFETCH_FACTOR": "4",
        "PIN_MEMORY": "1",
        "PERSISTENT_WORKERS": "1",
    }

    # 環境変数として設定
    for key, value in atft_settings.items():
        os.environ[key] = value
        logger.info(f"  {key}={value}")

    logger.info("✅ ATFT-GAT-FAN environment setup completed")


def train_atft_model(
    data_dir: str,
    batch_size: int = 2048,
    learning_rate: float = 5e-5,
    max_epochs: int = 75,
    precision: str = "bf16-mixed",
    config_profile: str = "profiles/robust",
) -> Dict:
    """
    ATFT-GAT-FANモデルの学習を実行（成果再現）

    Args:
        data_dir: データディレクトリのパス
        batch_size: バッチサイズ
        learning_rate: 学習率
        max_epochs: 最大エポック数
        precision: 精度設定
        config_profile: 設定プロファイル

    Returns:
        学習結果の辞書
    """
    try:
        logger.info("🏋️ Starting ATFT-GAT-FAN training with results reproduction...")
        logger.info("🎯 Target Sharpe Ratio: 0.849")
        logger.info(
            f"📊 Settings: batch_size={batch_size}, lr={learning_rate}, epochs={max_epochs}"
        )

        # ATFT-GAT-FANのパス確認
        atft_path = Path("/home/ubuntu/gogooku2/apps/ATFT-GAT-FAN")
        if not atft_path.exists():
            raise FileNotFoundError(f"ATFT-GAT-FAN path not found: {atft_path}")

        # 環境設定
        setup_atft_environment()

        # W&B設定の準備
        import pandas as pd
        wandb_name = f"atft_training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        train_files_count = len([f for f in os.listdir(data_dir + '/train') if f.endswith('.parquet')])
        wandb_notes = f"ATFT-GAT-FAN training with {train_files_count} training files, batch_size={batch_size}"

        # 学習コマンドの構築（ATFT-GAT-FANの成果設定 + W&B改善）
        cmd = [
            "python",
            "scripts/train.py",
            f"train={config_profile}",
            f"train.batch.train_batch_size={batch_size}",
            f"train.batch.val_batch_size={batch_size * 2}",
            "train.batch.gradient_accumulation_steps=1",
            f"train.optimizer.lr={learning_rate}",
            f"train.trainer.max_epochs={max_epochs}",
            f"train.precision={precision}",
            f"data.source.data_dir={data_dir}",
            "data=jpx_large_scale",
            "model=atft_gat_fan",
            "train.trainer.accelerator=gpu",
            "train.trainer.devices=1",
            "train.trainer.strategy=auto",
            "train.trainer.logger=wandb",  # W&Bロギング有効化
            f"train.trainer.logger.wandb.name={wandb_name}",
            f"train.trainer.logger.wandb.notes={wandb_notes}",
            "train.trainer.logger.wandb.project=ATFT-GAT-FAN",
            "train.trainer.logger.wandb.entity=wer-inc",
            "train.trainer.logger.wandb.log_model=true",
            "train.trainer.logger.wandb.save_code=true",
            "train.trainer.log_every_n_steps=10",  # より頻繁にログ
            "train.trainer.val_check_interval=0.25",
            "train.trainer.check_val_every_n_epoch=1",
            "train.trainer.enable_progress_bar=true",
            "train.trainer.enable_model_summary=true",
            "train.trainer.deterministic=false",
            "train.trainer.benchmark=true",
            "train.trainer.profiler=simple",
            "train.callbacks.early_stopping.patience=10",
            "train.callbacks.model_checkpoint.monitor=val_total_loss",
            "train.callbacks.model_checkpoint.mode=min",
            "train.callbacks.model_checkpoint.save_top_k=3",
            "train.callbacks.model_checkpoint.filename=atft_gat_fan-{epoch:03d}-{val_total_loss:.4f}",
            "train.callbacks.model_checkpoint.save_last=true",
            "train.callbacks.model_checkpoint.save_on_train_epoch_end=false",
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        # 学習実行
        result = subprocess.run(
            cmd,
            cwd=str(atft_path),
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

        # 結果の解析
        training_result = {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": cmd,
            "data_dir": data_dir,
            "settings": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_epochs": max_epochs,
                "precision": precision,
                "config_profile": config_profile,
            },
        }

        if result.returncode == 0:
            logger.info("✅ ATFT-GAT-FAN training completed successfully")

            # 成果の確認
            sharpe_ratio = extract_sharpe_ratio(result.stdout)
            if sharpe_ratio:
                logger.info(f"🎯 Achieved Sharpe Ratio: {sharpe_ratio}")
                training_result["sharpe_ratio"] = sharpe_ratio
                training_result["target_achieved"] = sharpe_ratio >= 0.849
            else:
                logger.warning("⚠️ Sharpe ratio not found in output")

            # W&B run情報のログ
            logger.info("📊 Training completed successfully with W&B logging enabled")
        else:
            logger.error(f"❌ ATFT-GAT-FAN training failed: {result.stderr}")
            logger.error("🔍 Error details:")
            logger.error(f"   Return code: {result.returncode}")
            if result.stderr:
                logger.error(f"   STDERR: {result.stderr[:500]}...")  # 最初の500文字
            if result.stdout:
                logger.error(f"   STDOUT: {result.stdout[-500:]}...")  # 最後の500文字

        return training_result

    except Exception as e:
        logger.error(f"❌ ATFT training failed: {e}")
        return {"success": False, "error": str(e), "data_dir": data_dir}


def extract_sharpe_ratio(output: str) -> Optional[float]:
    """学習出力からSharpe比率を抽出"""
    import re

    # 複数のパターンでSharpe比率を検索
    patterns = [
        r"Sharpe[:\s]*([0-9.]+)",
        r"sharpe[:\s]*([0-9.]+)",
        r"Sharpe Ratio[:\s]*([0-9.]+)",
        r"Test Sharpe[:\s]*([0-9.]+)",
        r"Validation Sharpe[:\s]*([0-9.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def validate_training_results() -> Dict:
    """学習結果の検証"""
    try:
        logger.info("🔍 Validating training results...")

        # チェックポイントの確認
        checkpoint_path = Path(
            "/home/ubuntu/gogooku2/apps/ATFT-GAT-FAN/models/checkpoints"
        )
        if not checkpoint_path.exists():
            return {"error": "Checkpoint directory not found"}

        checkpoints = list(checkpoint_path.glob("*.pt"))
        if not checkpoints:
            return {"error": "No checkpoints found"}

        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

        # モデル情報の確認
        import torch

        checkpoint = torch.load(latest_checkpoint, map_location="cpu")

        # パラメータ数の計算
        param_count = 0
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                param_count = sum(
                    p.numel() for p in checkpoint["model_state_dict"].values()
                )
            else:
                param_count = sum(
                    p.numel()
                    for p in checkpoint.values()
                    if isinstance(p, torch.Tensor)
                )
        else:
            param_count = sum(
                p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor)
            )

        validation_result = {
            "checkpoint_path": str(latest_checkpoint),
            "checkpoint_size_mb": latest_checkpoint.stat().st_size / (1024 * 1024),
            "param_count": param_count,
            "expected_params": 5611803,
            "param_match": param_count == 5611803,
            "checkpoint_count": len(checkpoints),
        }

        logger.info(f"✅ Validation completed: {param_count} parameters")
        return validation_result

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return {"error": str(e)}


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="Complete ATFT-GAT-FAN Training Wrapper"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Data directory path"
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--max_epochs", type=int, default=75, help="Maximum epochs")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision")
    parser.add_argument(
        "--config_profile", type=str, default="profiles/robust", help="Config profile"
    )
    parser.add_argument(
        "--validate_only", action="store_true", help="Validate results only"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Complete ATFT-GAT-FAN Training Wrapper")
    print("Target Sharpe Ratio: 0.849")
    print("=" * 60)

    if args.validate_only:
        # 結果検証のみ
        validation_result = validate_training_results()
        if "error" not in validation_result:
            print("✅ Validation successful")
            print(f"📊 Parameters: {validation_result['param_count']:,}")
            print(f"📁 Checkpoint: {validation_result['checkpoint_path']}")
        else:
            print(f"❌ Validation failed: {validation_result['error']}")
        return

    # 完全な学習実行
    training_result = train_atft_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        precision=args.precision,
        config_profile=args.config_profile,
    )

    if training_result["success"]:
        print("🎉 ATFT-GAT-FAN training succeeded!")

        # 結果検証
        validation_result = validate_training_results()
        if "error" not in validation_result:
            print(f"📊 Parameters: {validation_result['param_count']:,}")
            print(f"📁 Checkpoint: {validation_result['checkpoint_path']}")

        if "sharpe_ratio" in training_result:
            sharpe = training_result["sharpe_ratio"]
            target_achieved = training_result.get("target_achieved", False)
            print(f"🎯 Sharpe Ratio: {sharpe}")
            print(f"🎯 Target Achieved: {'✅ Yes' if target_achieved else '❌ No'}")
    else:
        print(
            f"❌ ATFT-GAT-FAN training failed: {training_result.get('error', 'Unknown error')}"
        )


if __name__ == "__main__":
    main()
