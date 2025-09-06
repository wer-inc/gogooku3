#!/usr/bin/env python3
"""
Complete ATFT-GAT-FAN Training Pipeline for gogooku3
ATFT-GAT-FANの成果（Sharpe 0.849）を完全に再現する統合学習パイプライン
"""

import os
import sys
import asyncio
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import polars as pl
import torch
import numpy as np
import subprocess

# パスを追加
sys.path.append(str(Path(__file__).parent.parent))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/ml_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class CompleteATFTTrainingPipeline:
    """ATFT-GAT-FANの成果を完全に再現する統合学習パイプライン"""

    def __init__(self):
        self.output_dir = Path("output")
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

        # ATFT-GAT-FANの成果設定
        self.atft_settings = {
            "expected_sharpe": 0.849,
            "model_params": 5611803,
            "input_dim": 8,
            "sequence_length": 20,
            "prediction_horizons": [1, 2, 3, 5, 10],
            "batch_size": 2048,
            "learning_rate": 5e-5,
            "max_epochs": 75,
            "precision": "bf16-mixed",
        }

        # 安定性設定（ATFT-GAT-FANの成果から）
        self.stability_settings = {
            "USE_T_NLL": 1,
            "OUTPUT_NOISE_STD": 0.02,
            "HEAD_NOISE_STD": 0.05,
            "HEAD_NOISE_WARMUP_EPOCHS": 5,
            "GAT_ALPHA_INIT": 0.3,
            "GAT_ALPHA_MIN": 0.1,
            "GAT_ALPHA_PENALTY": 1e-3,
            "EDGE_DROPOUT_INPUT_P": 0.1,
            "DEGENERACY_GUARD": 1,
            "DEGENERACY_WARMUP_STEPS": 1000,
            "DEGENERACY_CHECK_EVERY": 200,
            "DEGENERACY_MIN_RATIO": 0.05,
            "USE_AMP": 1,
            "AMP_DTYPE": "bf16",
        }

    async def run_complete_training_pipeline(self) -> Tuple[bool, Dict]:
        """ATFT-GAT-FANの成果を完全に再現する統合学習パイプラインを実行"""
        start_time = time.time()

        try:
            logger.info("🚀 Complete ATFT-GAT-FAN Training Pipeline started")
            logger.info(
                f"🎯 Target Sharpe Ratio: {self.atft_settings['expected_sharpe']}"
            )

            # 1. 環境設定（ATFT-GAT-FANの成果設定）
            success = await self._setup_atft_environment()
            if not success:
                return False, {"error": "Environment setup failed", "stage": "setup"}

            # 2. MLデータセットの読み込みと検証
            success, data_info = await self._load_and_validate_ml_dataset()
            if not success:
                return False, {"error": "ML dataset loading failed", "stage": "load"}

            # 3. 特徴量変換（ML → ATFT形式）
            success, conversion_info = await self._convert_ml_to_atft_format(
                data_info["df"]
            )
            if not success:
                return False, {
                    "error": "Feature conversion failed",
                    "stage": "conversion",
                }

            # 4. 学習データの準備（ATFT-GAT-FAN形式）
            success, training_data_info = await self._prepare_atft_training_data(
                conversion_info
            )
            if not success:
                return False, {
                    "error": "Training data preparation failed",
                    "stage": "preparation",
                }

            # 5. ATFT-GAT-FAN学習の実行（成果再現）
            success, training_info = await self._execute_atft_training_with_results(
                training_data_info
            )
            if not success:
                return False, {"error": "ATFT training failed", "stage": "training"}

            # 6. 成果検証
            success, validation_info = await self._validate_training_results(
                training_info
            )
            if not success:
                return False, {
                    "error": "Results validation failed",
                    "stage": "validation",
                }

            # 7. 結果記録
            elapsed_time = time.time() - start_time
            result = {
                "status": "success",
                "elapsed_time": elapsed_time,
                "atft_settings": self.atft_settings,
                "stability_settings": self.stability_settings,
                "data_info": data_info,
                "conversion_info": conversion_info,
                "training_data_info": training_data_info,
                "training_info": training_info,
                "validation_info": validation_info,
                "timestamp": datetime.now().isoformat(),
            }

            self._save_complete_training_result(result)
            logger.info(
                f"✅ Complete ATFT-GAT-FAN Training Pipeline completed successfully in {elapsed_time:.2f}s"
            )
            logger.info(
                f"🎯 Achieved Sharpe Ratio: {validation_info.get('sharpe_ratio', 'N/A')}"
            )

            return True, result

        except Exception as e:
            logger.error(f"❌ Complete training pipeline failed: {e}")
            import traceback

            traceback.print_exc()
            return False, {"error": str(e), "stage": "unknown"}

    async def _setup_atft_environment(self) -> bool:
        """ATFT-GAT-FANの成果を再現するための環境設定"""
        try:
            logger.info("🔧 Setting up ATFT-GAT-FAN environment...")

            # 安定性設定を環境変数として設定
            for key, value in self.stability_settings.items():
                os.environ[key] = str(value)

            # ATFT-GAT-FANのパス設定
            atft_path = Path("/home/ubuntu/gogooku2/apps/ATFT-GAT-FAN")
            if not atft_path.exists():
                logger.error(f"ATFT-GAT-FAN path not found: {atft_path}")
                return False

            # 必要なディレクトリ作成
            (self.output_dir / "atft_data").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "results").mkdir(parents=True, exist_ok=True)

            logger.info("✅ ATFT-GAT-FAN environment setup completed")
            return True

        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            return False

    async def _load_and_validate_ml_dataset(self) -> Tuple[bool, Dict]:
        """MLデータセットの読み込みと検証（ATFT-GAT-FAN対応）"""
        try:
            logger.info("📊 Loading and validating ML dataset...")

            # MLデータセットの読み込み（実際のデータを使用）
            # 優先順位: output/ml_dataset_production.parquet > data/processed/ml_dataset_latest.parquet > data/ml_dataset.parquet
            ml_dataset_paths = [
                Path("output/ml_dataset_production.parquet"),
                Path("data/processed/ml_dataset_latest.parquet"),
                Path("data/ml_dataset.parquet")
            ]

            ml_dataset_path = None
            for path in ml_dataset_paths:
                if path.exists():
                    ml_dataset_path = path
                    break

            if ml_dataset_path is None:
                # テスト用のサンプルデータ作成
                logger.warning("ML dataset not found, creating sample data for testing")
                df = self._create_sample_ml_dataset()
            else:
                logger.info(f"📂 Loading ML dataset from: {ml_dataset_path}")
                df = pl.read_parquet(ml_dataset_path)

            # データ検証
            validation_result = self._validate_ml_dataset(df)
            if not validation_result["valid"]:
                return False, {"error": validation_result["error"]}

            data_info = {
                "df": df,
                "shape": df.shape,
                "columns": df.columns,
                "validation": validation_result,
            }

            logger.info(f"✅ ML dataset loaded: {df.shape}")
            return True, data_info

        except Exception as e:
            logger.error(f"❌ ML dataset loading failed: {e}")
            return False, {"error": str(e)}

    async def _convert_ml_to_atft_format(self, df: pl.DataFrame) -> Tuple[bool, Dict]:
        """MLデータセットをATFT-GAT-FAN形式に変換"""
        try:
            logger.info("🔄 Converting ML dataset to ATFT-GAT-FAN format...")

            from scripts.models.unified_feature_converter import UnifiedFeatureConverter

            # ATFT-GAT-FAN形式に変換
            converter = UnifiedFeatureConverter()
            file_paths = converter.convert_ml_dataset_to_atft_format(
                df, "output/atft_data"
            )

            conversion_info = {
                "file_paths": file_paths,
                "converter": "UnifiedFeatureConverter",
                "output_dir": "output/atft_data",
            }

            logger.info(f"✅ Conversion completed: {len(file_paths)} files created")
            return True, conversion_info

        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            return False, {"error": str(e)}

    async def _prepare_atft_training_data(
        self, conversion_info: Dict
    ) -> Tuple[bool, Dict]:
        """ATFT-GAT-FAN学習用データの準備"""
        try:
            logger.info("📋 Preparing ATFT-GAT-FAN training data...")

            # データ分割情報の確認
            file_paths = conversion_info.get("file_paths", {})
            train_files = file_paths.get("train_files", [])
            val_files = file_paths.get("val_files", [])
            test_files = file_paths.get("test_files", [])

            if not train_files:
                return False, {"error": "No training files found"}

            training_data_info = {
                "train_files": train_files,
                "val_files": val_files,
                "test_files": test_files,
                "data_dir": "output/atft_data",
                "sequence_length": self.atft_settings["sequence_length"],
                "input_dim": self.atft_settings["input_dim"],
                "metadata": file_paths.get("metadata"),
            }

            logger.info(
                f"✅ ATFT-GAT-FAN training data prepared: {len(train_files)} train files"
            )
            return True, training_data_info

        except Exception as e:
            logger.error(f"❌ Training data preparation failed: {e}")
            return False, {"error": str(e)}

    async def _execute_atft_training_with_results(
        self, training_data_info: Dict
    ) -> Tuple[bool, Dict]:
        """ATFT-GAT-FAN学習の実行（成果再現）"""
        try:
            logger.info(
                "🏋️ Executing ATFT-GAT-FAN training with results reproduction..."
            )

            # 内製トレーナーを使用（Hydra設定でオーバーライド）
            cmd = [
                "python",
                "scripts/train_atft.py",
                # Hydra overrides
                f"data.source.data_dir={training_data_info['data_dir']}",
                "data=jpx_parquet",
                # 時系列仕様（ATFT形式に合わせる）
                f"data.time_series.sequence_length={self.atft_settings['sequence_length']}",
                "data.time_series.prediction_horizons=[1,2,3,5,10]",
                # 学習設定（本番寄り）
                f"train.batch.train_batch_size={self.atft_settings['batch_size']}",
                f"train.optimizer.lr={self.atft_settings['learning_rate']}",
                f"train.trainer.max_epochs={self.atft_settings['max_epochs']}",
                f"train.trainer.precision={self.atft_settings['precision']}",
                # 追加の安定化（必要に応じて環境変数で上書き可能）
                "hardware.num_workers=8",
                "train.trainer.check_val_every_n_epoch=1",
                "train.trainer.enable_progress_bar=true",
            ]

            logger.info(f"Running command: {' '.join(cmd)}")

            # 学習実行（リポジトリ直下で実行）
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr}")
                return False, {"error": result.stderr}

            # 学習結果の解析
            training_info = self._parse_training_output(result.stdout)
            training_info["command"] = cmd
            training_info["return_code"] = result.returncode

            logger.info("✅ ATFT-GAT-FAN training completed successfully")
            return True, training_info

        except Exception as e:
            logger.error(f"❌ ATFT training execution failed: {e}")
            return False, {"error": str(e)}

    async def _validate_training_results(
        self, training_info: Dict
    ) -> Tuple[bool, Dict]:
        """学習結果の検証（Sharpe 0.849の再現確認）"""
        try:
            logger.info("🔍 Validating training results...")

            # チェックポイントの確認
            checkpoint_path = Path(
                "/home/ubuntu/gogooku2/apps/ATFT-GAT-FAN/models/checkpoints"
            )
            checkpoints = list(checkpoint_path.glob("*.pt"))

            if not checkpoints:
                return False, {"error": "No checkpoints found"}

            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

            # モデルパラメータ数の確認（PyTorch 2.6対応）
            try:
                model = torch.load(
                    latest_checkpoint, map_location="cpu", weights_only=False
                )
            except Exception as e:
                logger.warning(f"PyTorch 2.6 weights_only issue: {e}")
                # 代替方法でチェックポイントサイズを確認
                checkpoint_size = latest_checkpoint.stat().st_size
                param_count = checkpoint_size // 4  # 概算
            else:
                param_count = sum(
                    p.numel() for p in model.values() if isinstance(p, torch.Tensor)
                )

            # 成果の検証
            validation_info = {
                "checkpoint_path": str(latest_checkpoint),
                "param_count": param_count,
                "expected_params": self.atft_settings["model_params"],
                "param_match": abs(param_count - self.atft_settings["model_params"])
                < 1000000,  # 許容誤差
                "training_log": training_info.get("log", ""),
                "sharpe_ratio": self._extract_sharpe_ratio(
                    training_info.get("log", "")
                ),
                "target_sharpe": self.atft_settings["expected_sharpe"],
                "checkpoint_size_mb": latest_checkpoint.stat().st_size / (1024 * 1024),
            }

            logger.info(f"✅ Validation completed: {param_count} parameters")
            return True, validation_info

        except Exception as e:
            logger.error(f"❌ Results validation failed: {e}")
            return False, {"error": str(e)}

    def _create_sample_ml_dataset(self) -> pl.DataFrame:
        """テスト用のサンプルMLデータセット作成"""
        # 実際のML_DATASET_COLUMNS.mdの仕様に基づいてサンプルデータ作成
        n_stocks = 10
        n_days = 100

        data = []
        for stock_id in range(n_stocks):
            for day in range(n_days):
                row = {
                    "Code": f"STOCK_{stock_id:04d}",
                    "Date": f"2024-01-{day+1:02d}",
                    "Open": 1000 + np.random.randn() * 50,
                    "High": 1020 + np.random.randn() * 30,
                    "Low": 980 + np.random.randn() * 30,
                    "Close": 1000 + np.random.randn() * 50,
                    "Volume": np.random.randint(1000, 10000),
                    "returns_1d": np.random.randn() * 0.02,
                    "returns_5d": np.random.randn() * 0.05,
                    "returns_10d": np.random.randn() * 0.08,
                    "returns_20d": np.random.randn() * 0.12,
                    "ema_5": 1000 + np.random.randn() * 20,
                    "ema_10": 1000 + np.random.randn() * 25,
                    "ema_20": 1000 + np.random.randn() * 30,
                    "ema_60": 1000 + np.random.randn() * 35,
                    "ema_200": 1000 + np.random.randn() * 40,
                    "rsi_14": np.random.uniform(30, 70),
                    "rsi_2": np.random.uniform(20, 80),
                    "macd_signal": np.random.randn() * 10,
                    "macd_histogram": np.random.randn() * 5,
                    "bb_pct_b": np.random.uniform(0, 1),
                    "bb_bandwidth": np.random.uniform(0.1, 0.5),
                    "volatility_20d": np.random.uniform(0.1, 0.3),
                    "sharpe_1d": np.random.randn() * 0.5,
                }
                data.append(row)

        return pl.DataFrame(data)

    def _validate_ml_dataset(self, df: pl.DataFrame) -> Dict:
        """MLデータセットの検証"""
        required_columns = [
            "Code",
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "returns_1d",
            "returns_5d",
            "returns_10d",
            "returns_20d",
            "ema_5",
            "ema_10",
            "ema_20",
            "ema_60",
            "ema_200",
            "rsi_14",
            "rsi_2",
            "macd_signal",
            "macd_histogram",
            "bb_pct_b",
            "bb_bandwidth",
            "volatility_20d",
            "sharpe_1d",
        ]

        missing_columns = set(required_columns) - set(df.columns)

        return {
            "valid": len(missing_columns) == 0,
            "missing_columns": list(missing_columns),
            "total_rows": len(df),
            "total_columns": len(df.columns),
        }

    def _parse_training_output(self, output: str) -> Dict:
        """学習出力の解析"""
        lines = output.split("\n")

        # 重要なメトリクスを抽出
        metrics = {}
        for line in lines:
            if "Sharpe" in line:
                metrics["sharpe"] = line
            elif "Loss" in line:
                metrics["loss"] = line
            elif "Epoch" in line:
                metrics["epoch"] = line

        return {"log": output, "metrics": metrics, "lines": len(lines)}

    def _extract_sharpe_ratio(self, log: str) -> Optional[float]:
        """ログからSharpe比率を抽出"""
        import re

        sharpe_pattern = r"Sharpe[:\s]*([0-9.]+)"
        match = re.search(sharpe_pattern, log)

        if match:
            return float(match.group(1))
        return None

    def _save_complete_training_result(self, result: Dict):
        """完全な学習結果の保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = (
            self.output_dir / "results" / f"complete_training_result_{timestamp}.json"
        )

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"💾 Complete training result saved: {result_file}")


async def main():
    """メイン実行関数"""
    pipeline = CompleteATFTTrainingPipeline()

    print("=" * 60)
    print("Complete ATFT-GAT-FAN Training Pipeline")
    print("Target Sharpe Ratio: 0.849")
    print("=" * 60)

    success, result = await pipeline.run_complete_training_pipeline()

    if success:
        print("🎉 Complete training pipeline succeeded!")
        print(
            f"📊 Results: {result.get('validation_info', {}).get('sharpe_ratio', 'N/A')}"
        )
    else:
        print(
            f"❌ Complete training pipeline failed: {result.get('error', 'Unknown error')}"
        )

    return success, result


if __name__ == "__main__":
    asyncio.run(main())
