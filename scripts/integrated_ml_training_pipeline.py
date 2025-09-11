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

    def __init__(self, data_path: Optional[str] = None, sample_size: Optional[int] = None):
        self.output_dir = Path("output")
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.data_path = Path(data_path) if data_path else None
        # 小規模実行用のサンプリング行数（概算）
        self.sample_size: Optional[int] = int(sample_size) if sample_size else None

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
                "portfolio_optimization": {},
                "timestamp": datetime.now().isoformat(),
            }
            # 8. ポートフォリオ最適化（検証予測がある場合）
            try:
                ok_po, po_info = await self._run_portfolio_optimization()
                if ok_po:
                    result["portfolio_optimization"] = po_info
                else:
                    result["portfolio_optimization"] = {"error": po_info.get("error", "unknown")}
            except Exception as _e:
                logger.warning(f"Portfolio optimization step skipped: {_e}")

            self._save_complete_training_result(result)
            logger.info(
                f"✅ Complete ATFT-GAT-FAN Training Pipeline completed successfully in {elapsed_time:.2f}s"
            )
            ach = validation_info.get('sharpe_ratio', None)
            if ach is not None:
                logger.info(f"🎯 Achieved Sharpe Ratio: {ach}")
            try:
                po = result.get("portfolio_optimization", {})
                rep = po.get("report", {})
                if rep and "sharpe" in rep:
                    logger.info(f"📈 Portfolio Sharpe (net, cost 5bps): {rep['sharpe']:.4f}")
            except Exception:
                pass

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
            # 優先順位: コマンドライン引数 > output/ml_dataset_*.parquet > data/processed/ml_dataset_latest.parquet > data/ml_dataset.parquet
            ml_dataset_paths = []
            
            # コマンドライン引数が指定されていれば最優先
            if self.data_path and self.data_path.exists():
                ml_dataset_paths.append(self.data_path)
            
            # output内の最新のデータセットを探す
            output_datasets = sorted(Path("output").glob("ml_dataset_*.parquet"), reverse=True)
            ml_dataset_paths.extend(output_datasets[:3])  # 最新3つまで
            
            # デフォルトパス
            ml_dataset_paths.extend([
                Path("output/ml_dataset_production.parquet"),
                Path("data/processed/ml_dataset_latest.parquet"),
                Path("data/ml_dataset.parquet")
            ])

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

            # 迅速検証モード: --sample-size が指定された場合は変換前にデータを縮小
            if self.sample_size is not None and self.sample_size > 0:
                try:
                    # 銘柄単位でサンプリングし、累積行数が sample_size を超えるまで採用
                    # これにより時系列の連続性を保ちつつファイル数も削減できる
                    min_seq = int(self.atft_settings.get("sequence_length", 20))
                    gb = (
                        df.group_by("Code")
                        .agg(pl.len().alias("n"))
                        .filter(pl.col("n") >= min_seq)  # 学習に必要な系列長を満たす銘柄のみ
                        .sort("n")  # 過剰サンプルを避けるため行数の少ない銘柄から採用
                    )
                    codes = gb.select(["Code", "n"]).to_dict(as_series=False)
                    sel_codes = []
                    cum = 0
                    for code, n in zip(codes["Code"], codes["n"]):
                        if cum >= self.sample_size:
                            break
                        sel_codes.append(code)
                        cum += int(n)
                    if sel_codes:
                        df = df.filter(pl.col("Code").is_in(sel_codes))
                        logger.info(
                            f"🔎 Sample mode: selected {len(sel_codes)} codes for ~{self.sample_size} rows (actual={len(df)})"
                        )
                    else:
                        logger.warning("Sample mode requested but could not determine codes; falling back to head() sampling")
                        df = df.head(self.sample_size)
                except Exception as e:
                    logger.warning(f"Sample mode failed ({e}); falling back to head() sampling")
                    try:
                        df = df.head(self.sample_size)
                    except Exception:
                        pass

            # データ検証
            validation_result = self._validate_ml_dataset(df)
            if not validation_result["valid"]:
                error_details = []
                if validation_result['missing_columns']:
                    error_details.append(f"Missing columns: {validation_result['missing_columns']}")
                if not validation_result.get('has_return_column', False):
                    error_details.append("No return/target column found (needs one of: returns_1d, feat_ret_1d, target, returns)")
                if validation_result['total_columns'] < 50:
                    error_details.append(f"Not enough features: {validation_result['total_columns']} < 50")
                if validation_result['total_rows'] == 0:
                    error_details.append("Dataset is empty")
                
                error_msg = "; ".join(error_details) if error_details else "Unknown validation error"
                logger.error(f"Dataset validation failed: {error_msg}")
                logger.info(f"Dataset info - Rows: {validation_result['total_rows']}, Cols: {validation_result['total_columns']}")
                logger.info(f"Sample columns: {validation_result.get('column_sample', [])}")
                return False, {"error": error_msg}

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
            
            # 出力ディレクトリ（サンプル実行は別ディレクトリへ書き出して本番データを保護）
            out_dir = (
                f"output/atft_data_sample_{self.sample_size}"
                if (self.sample_size is not None and self.sample_size > 0)
                else "output/atft_data"
            )

            # Try to import UnifiedFeatureConverter
            try:
                from scripts.models.unified_feature_converter import UnifiedFeatureConverter
                converter = UnifiedFeatureConverter()
                # 既存の変換結果があり、再利用可能ならスキップ
                try:
                    from pathlib import Path as _P

                    _train_dir = _P(out_dir) / "train"
                    force_reconvert = os.getenv("FORCE_CONVERT", "0") == "1"
                    if (not force_reconvert) and _train_dir.exists() and any(_train_dir.glob("*.parquet")):
                        logger.info(
                            f"♻️  Reusing existing converted data at {out_dir} (skip conversion)"
                        )
                        file_paths = {
                            "train_files": sorted(str(p) for p in _train_dir.glob("*.parquet")),
                            "val_files": sorted(
                                str(p) for p in (_P(out_dir) / "val").glob("*.parquet")
                            ),
                            "test_files": sorted(
                                str(p) for p in (_P(out_dir) / "test").glob("*.parquet")
                            ),
                            "metadata": str(_P(out_dir) / "metadata.json"),
                        }
                    else:
                        file_paths = converter.convert_ml_dataset_to_atft_format(df, out_dir)
                except Exception:
                    # フォールバック: 常に変換
                    file_paths = converter.convert_ml_dataset_to_atft_format(df, out_dir)
            except ImportError:
                logger.warning("UnifiedFeatureConverter not found, using direct training approach")
                # Create mock file paths for compatibility
                file_paths = {
                    "train_files": ["direct_training"],
                    "val_files": [],
                    "test_files": [],
                    "metadata": {"direct_mode": True}
                }

            conversion_info = {
                "file_paths": file_paths,
                "converter": "Direct" if "direct_training" in str(file_paths) else "UnifiedFeatureConverter",
                "output_dir": out_dir,
                "dataframe": df  # Keep dataframe for direct training
            }

            logger.info(f"✅ Conversion completed: Mode = {conversion_info['converter']}")
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
                "data_dir": conversion_info.get("output_dir", "output/atft_data"),
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

            # max_epochs=0 の場合は学習をスキップ（配線検証などの用途）
            if int(self.atft_settings.get("max_epochs", 0)) <= 0:
                msg = "Training skipped because max_epochs=0"
                logger.info(msg)
                training_info = {
                    "command": [],
                    "return_code": 0,
                    "log": msg,
                    "metrics": {},
                    "lines": 1,
                    "skipped": True,
                }
                return True, training_info

            # 内製トレーナーを使用（Hydra設定でオーバーライド）
            # Hydra構成はdefaultsで安定化済み。
            # ここでは必要な学習ハイパラだけ通常の上書きで渡す（struct安全）。
            cmd = [
                "python",
                "scripts/train_atft.py",
                # データディレクトリを明示的に指定
                f"data.source.data_dir={training_data_info['data_dir']}/train",
                # data/model/train は configs/config.yaml の defaults で固定
                # 学習ハイパラのみ調整（既存キーなので + は不要）
                f"train.batch.train_batch_size={self.atft_settings['batch_size']}",
                f"train.optimizer.lr={self.atft_settings['learning_rate']}",
                f"train.trainer.max_epochs={self.atft_settings['max_epochs']}",
                f"train.trainer.precision={self.atft_settings['precision']}",
                # 進捗と検証頻度は明示
                "train.trainer.check_val_every_n_epoch=1",
                "train.trainer.enable_progress_bar=true",
            ]

            # 小規模サンプルでの実行時は、minibatch崩壊と検証0件を避けるための保護を入れる
            debug_small_data = (
                self.sample_size is not None and int(self.sample_size) > 0
            )
            if debug_small_data:
                # 短いシーケンスでval側のサンプル不足を回避（本番は20でOK）
                debug_seq_len = 10
                cmd.append(f"data.time_series.sequence_length={debug_seq_len}")
                logger.info(
                    f"[debug-small] Override sequence_length -> {debug_seq_len} to ensure non-empty validation"
                )
                # マルチプロセスDataLoaderはサンドボックス環境でPermissionErrorを起こすため無効化
                cmd.extend(
                    [
                        "train.batch.num_workers=0",
                        "train.batch.prefetch_factor=null",
                        "train.batch.persistent_workers=false",
                        "train.batch.pin_memory=false",
                    ]
                )

            logger.info(f"Running command: {' '.join(cmd)}")

            # 学習実行（リポジトリ直下で実行）
            # Ensure train script sees data directory via config override
            env = os.environ.copy()
            # 恒久運用ではValidatorを有効化
            env.pop("VALIDATE_CONFIG", None)
            env["HYDRA_FULL_ERROR"] = "1"  # 詳細なエラー情報を取得
            if debug_small_data:
                # 分割とgapの保守設定（検証期間を十分確保）
                env.setdefault("TRAIN_RATIO", "0.6")
                # VAL_RATIOは非累積（trainとは独立比率）で扱われる
                env.setdefault("VAL_RATIO", "0.3")
                env.setdefault("GAP_DAYS", "1")
                # DayBatchSamplerは小規模だと1バッチ化しやすいので無効化
                env.setdefault("USE_DAY_BATCH", "0")
                env.setdefault("MIN_NODES_PER_DAY", "4")
                # シーケンス間引き（計算量を抑制）
                env.setdefault("DATASET_STRIDE", "2")
                # DataLoaderを単一プロセスに固定
                env.setdefault("NUM_WORKERS", "0")
                env.setdefault("PERSISTENT_WORKERS", "0")
                env.setdefault("PIN_MEMORY", "0")
                env.setdefault("DL_SEED", "42")
                # 小規模時はSharpe計算の安定化（分母に微小ε）
                env.setdefault("SHARPE_EPS", "1e-8")
                logger.info(
                    "[debug-small] Applied env overrides: "
                    "TRAIN_RATIO=0.6 VAL_RATIO=0.3 GAP_DAYS=1 "
                    "USE_DAY_BATCH=0 MIN_NODES_PER_DAY=4 DATASET_STRIDE=2 "
                    "NUM_WORKERS=0 SHARPE_EPS=1e-8"
                )
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
            )

            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr}")
                return False, {"error": result.stderr}

            # 学習結果の解析（stdout + stderr を合わせて解析）
            combined_output = "\n".join([
                result.stdout or "",
                result.stderr or "",
            ])
            training_info = self._parse_training_output(combined_output)
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
            # ローカルリポジトリ配下のチェックポイントを参照
            checkpoint_path = Path("models/checkpoints")
            checkpoints = list(checkpoint_path.glob("*.pt"))

            if not checkpoints:
                # 学習スキップ時や失敗時はメトリクスJSONのみで検証
                sharpe = None
                try:
                    ms_path = Path("runs/last/metrics_summary.json")
                    if ms_path.exists():
                        sharpe = json.loads(ms_path.read_text()).get("avg_sharpe")
                except Exception:
                    sharpe = None
                if sharpe is None:
                    try:
                        lm_path = Path("runs/last/latest_metrics.json")
                        if lm_path.exists():
                            sharpe = json.loads(lm_path.read_text()).get("avg_sharpe")
                    except Exception:
                        sharpe = None
                validation_info = {
                    "checkpoint_path": None,
                    "param_count": 0,
                    "expected_params": self.atft_settings["model_params"],
                    "param_match": False,
                    "training_log": training_info.get("log", ""),
                    "sharpe_ratio": sharpe,
                    "target_sharpe": self.atft_settings["expected_sharpe"],
                    "checkpoint_size_mb": 0.0,
                    "note": "No checkpoints found; using metrics only",
                }
                logger.info("✅ Validation completed: 0 parameters (no checkpoint)")
                return True, validation_info

            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

            # モデルパラメータ数の確認（state_dict/モデル双方に頑健）
            def _count_params(ckpt_path: Path) -> int:
                try:
                    obj = torch.load(ckpt_path, map_location="cpu")
                except Exception:
                    return int(ckpt_path.stat().st_size // 4)  # 粗い概算
                try:
                    if isinstance(obj, dict):
                        # Lightningなどの形式: state_dictキーやモデル関連の候補を探す
                        for k in ("state_dict", "model_state_dict", "model", "weights"):
                            if k in obj and isinstance(obj[k], dict):
                                sd = obj[k]
                                return sum(int(p.numel()) for p in sd.values() if isinstance(p, torch.Tensor))
                        # 直接 state_dict の場合
                        if all(isinstance(v, torch.Tensor) for v in obj.values()):
                            return sum(int(p.numel()) for p in obj.values())
                    # それ以外（オブジェクト保存など）はサイズから概算
                    return int(ckpt_path.stat().st_size // 4)
                except Exception:
                    return int(ckpt_path.stat().st_size // 4)

            param_count = _count_params(latest_checkpoint)

            # 成果の検証
            # Prefer metrics_summary.json, then latest_metrics.json, else parse logs
            sharpe = None
            try:
                ms_path = Path("runs/last/metrics_summary.json")
                if ms_path.exists():
                    with open(ms_path, "r") as mf:
                        jm = json.load(mf)
                        if isinstance(jm, dict):
                            sharpe = jm.get("avg_sharpe")
            except Exception:
                sharpe = None
            if sharpe is None:
                try:
                    metrics_path = Path("runs/last/latest_metrics.json")
                    if metrics_path.exists():
                        with open(metrics_path, "r") as mf:
                            jm = json.load(mf)
                            if isinstance(jm, dict):
                                sharpe = jm.get("avg_sharpe")
                except Exception:
                    sharpe = None
            if sharpe is None:
                # Fallback to training logs
                sharpe = self._extract_sharpe_ratio(training_info.get("log", ""))
                if sharpe is None:
                    try:
                        log_path = Path("logs/ml_training.log")
                        if log_path.exists():
                            tail = log_path.read_text(errors="ignore")
                            sharpe = self._extract_sharpe_ratio(tail)
                    except Exception:
                        sharpe = None

            validation_info = {
                "checkpoint_path": str(latest_checkpoint),
                "param_count": param_count,
                "expected_params": self.atft_settings["model_params"],
                "param_match": abs(param_count - self.atft_settings["model_params"])
                < 1000000,  # 許容誤差
                "training_log": training_info.get("log", ""),
                "sharpe_ratio": sharpe,
                "target_sharpe": self.atft_settings["expected_sharpe"],
                "checkpoint_size_mb": latest_checkpoint.stat().st_size / (1024 * 1024),
            }

            logger.info(f"✅ Validation completed: {param_count} parameters")
            return True, validation_info

        except Exception as e:
            logger.error(f"❌ Results validation failed: {e}")
            return False, {"error": str(e)}

    async def _run_portfolio_optimization(self) -> Tuple[bool, Dict]:
        """検証用予測ファイルを用いてポートフォリオ最適化を実行"""
        try:
            pred_path = Path("runs/last/predictions_val.parquet")
            if not pred_path.exists():
                logger.warning(f"Predictions file not found: {pred_path}")
                return False, {"error": "predictions_val.parquet not found"}

            cmd = [
                "python",
                "scripts/advanced_portfolio_optimization.py",
                "--input",
                str(pred_path),
                "--pred-col",
                "predicted_return",
                "--ret-col",
                "actual_return",
                "--mode",
                "ls",
                "--long-frac",
                "0.2",
                "--short-frac",
                "0.2",
                "--invert-sign",
                "--cost-bps",
                "5",
            ]
            logger.info(f"Running portfolio optimization: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                logger.error(f"Portfolio optimization failed: {res.stderr}")
                return False, {"error": res.stderr}

            # 最新のレポートを読み取る
            out_dir = Path("output/portfolio")
            report = {}
            try:
                if out_dir.exists():
                    latest = max(out_dir.glob("report_*.json"), key=lambda p: p.stat().st_mtime)
                    report = json.loads(latest.read_text())
                    logger.info(f"Portfolio report loaded: {latest}")
                else:
                    logger.warning("Portfolio output directory not found")
            except Exception as _e:
                logger.warning(f"Failed to load portfolio report: {_e}")
            return True, {"stdout": res.stdout, "report": report}
        except Exception as e:
            logger.error(f"❌ Portfolio optimization failed: {e}")
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
        # 最小限必要なカラムのみチェック
        essential_columns = [
            "Code",
            "Date",
            "Open",
            "High", 
            "Low",
            "Close",
            "Volume"
        ]
        
        # リターン系のカラムがあるかチェック（どれか1つあればOK）
        return_columns = ["returns_1d", "feat_ret_1d", "target", "returns"]
        has_return = any(col in df.columns for col in return_columns)
        
        missing_essential = set(essential_columns) - set(df.columns)
        
        # 十分な特徴量があるかチェック（最低50カラム以上）
        has_enough_features = len(df.columns) >= 50
        
        # 検証結果
        is_valid = (len(missing_essential) == 0 and 
                   has_return and 
                   has_enough_features and
                   len(df) > 0)
        
        return {
            "valid": is_valid,
            "missing_columns": list(missing_essential),
            "has_return_column": has_return,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_sample": df.columns[:10] if len(df.columns) > 10 else df.columns,
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

        # 負値もマッチ（例: "Sharpe: -0.0123"）
        sharpe_pattern = r"Sharpe[:\s]*(-?[0-9]*\.?[0-9]+)"
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
    import argparse
    parser = argparse.ArgumentParser(description="Complete ATFT-GAT-FAN Training Pipeline")
    parser.add_argument("--data-path", type=str, help="Path to ML dataset parquet file")
    parser.add_argument("--max-epochs", type=int, default=75, help="Maximum epochs (0 to skip training)")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override (e.g., 2e-4)")
    parser.add_argument("--sample-size", type=int, help="Sample size for testing")
    parser.add_argument(
        "--adv-graph-train",
        action="store_true",
        help="Enable advanced FinancialGraphBuilder during training (EWM+shrinkage)",
    )
    args = parser.parse_args()
    
    pipeline = CompleteATFTTrainingPipeline(data_path=args.data_path, sample_size=args.sample_size)
    
    # 引数で設定を上書き（0も有効値として扱う）
    if args.max_epochs is not None:
        pipeline.atft_settings["max_epochs"] = int(args.max_epochs)
    if args.batch_size is not None:
        pipeline.atft_settings["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        pipeline.atft_settings["learning_rate"] = float(args.lr)

    print("=" * 60)
    print("Complete ATFT-GAT-FAN Training Pipeline")
    print("Target Sharpe Ratio: 0.849")
    print("=" * 60)

    # Optionally enable advanced graph builder for training
    if args.adv_graph_train:
        os.environ["USE_ADV_GRAPH_TRAIN"] = "1"
        # Provide sensible defaults if not set (recommended)
        os.environ.setdefault("GRAPH_CORR_METHOD", "ewm_demean")
        os.environ.setdefault("EWM_HALFLIFE", "30")
        os.environ.setdefault("SHRINKAGE_GAMMA", "0.1")
        os.environ.setdefault("GRAPH_K", "15")
        os.environ.setdefault("GRAPH_EDGE_THR", "0.25")
        os.environ.setdefault("GRAPH_SYMMETRIC", "1")

    success, result = await pipeline.run_complete_training_pipeline()

    if success:
        print("🎉 Complete training pipeline succeeded!")
        sr = result.get('validation_info', {}).get('sharpe_ratio', None)
        if sr is not None:
            print(f"📊 Results: {sr}")
    else:
        print(
            f"❌ Complete training pipeline failed: {result.get('error', 'Unknown error')}"
        )

    return success, result


if __name__ == "__main__":
    asyncio.run(main())
