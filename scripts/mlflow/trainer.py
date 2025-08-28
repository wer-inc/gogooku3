#!/usr/bin/env python3
"""
MLflow Model Training Integration
MLflowを使用したモデル学習とトラッキング
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import json
from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTrainer:
    """MLflow統合トレーナークラス"""

    def __init__(
        self,
        experiment_name: str = "gogooku3-ml",
        tracking_uri: str = None,
        artifact_location: str = None,
    ):
        """
        Initialize MLflow trainer

        Args:
            experiment_name: 実験名
            tracking_uri: MLflow tracking server URI
            artifact_location: アーティファクト保存先
        """
        self.experiment_name = experiment_name

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to local or Docker MLflow
            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )

        # Set artifact location (MinIO)
        if artifact_location:
            os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_location

        # Create or get experiment
        self.experiment = mlflow.set_experiment(
            experiment_name,
            tags={
                "project": "gogooku3",
                "team": "ml",
                "version": "1.0.0",
            },
        )

        self.client = MlflowClient()

    def start_run(
        self,
        run_name: str = None,
        tags: Dict[str, str] = None,
        nested: bool = False,
    ):
        """
        MLflow run を開始

        Args:
            run_name: 実行名
            tags: タグ辞書
            nested: ネストした実行かどうか
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        default_tags = {
            "framework": "pytorch",
            "dataset": "jquants",
            "model_type": "gat",
        }

        if tags:
            default_tags.update(tags)

        return mlflow.start_run(
            run_name=run_name,
            tags=default_tags,
            nested=nested,
        )

    def log_params(self, params: Dict[str, Any]):
        """パラメータをログ"""
        for key, value in params.items():
            if isinstance(value, (list, dict)):
                # Complex types as JSON
                mlflow.log_param(key, json.dumps(value))
            else:
                mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """メトリクスをログ"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_dataset(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        name: str = "dataset",
        description: str = None,
    ):
        """データセットをログ"""
        # Convert to pandas if necessary
        if isinstance(df, pl.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df

        # Create dataset info
        dataset_info = {
            "name": name,
            "shape": df_pd.shape,
            "columns": list(df_pd.columns),
            "dtypes": {col: str(dtype) for col, dtype in df_pd.dtypes.items()},
            "description": description or f"Dataset {name}",
            "statistics": df_pd.describe().to_dict(),
        }

        # Log as artifact
        info_path = Path(f"/tmp/{name}_info.json")
        with open(info_path, "w") as f:
            json.dump(dataset_info, f, indent=2, default=str)

        mlflow.log_artifact(info_path, artifact_path="datasets")

        # Log sample data
        sample_path = Path(f"/tmp/{name}_sample.parquet")
        df_pd.head(1000).to_parquet(sample_path)
        mlflow.log_artifact(sample_path, artifact_path="datasets")

        # Log dataset stats
        mlflow.log_metrics(
            {
                f"{name}_rows": df_pd.shape[0],
                f"{name}_cols": df_pd.shape[1],
                f"{name}_null_ratio": df_pd.isnull().sum().sum()
                / (df_pd.shape[0] * df_pd.shape[1]),
            }
        )

    def log_model(
        self,
        model: Any,
        model_name: str,
        signature: Any = None,
        input_example: Any = None,
        pip_requirements: list = None,
        code_paths: list = None,
    ):
        """
        モデルをログして登録

        Args:
            model: モデルオブジェクト
            model_name: モデル名
            signature: モデルシグネチャ
            input_example: 入力例
            pip_requirements: pip要件
            code_paths: コードパス
        """
        # Determine model flavor
        if hasattr(model, "predict") and hasattr(model, "fit"):
            # Scikit-learn style model
            mlflow.sklearn.log_model(
                model,
                model_name,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_requirements,
                code_paths=code_paths,
                registered_model_name=model_name,
            )
        elif "torch" in str(type(model)):
            # PyTorch model
            mlflow.pytorch.log_model(
                model,
                model_name,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_requirements or ["torch", "torch_geometric"],
                code_paths=code_paths,
                registered_model_name=model_name,
            )
        else:
            # Generic Python model
            mlflow.pyfunc.log_model(
                model_name,
                python_model=model,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_requirements,
                code_paths=code_paths,
                registered_model_name=model_name,
            )

        logger.info(f"Model {model_name} logged and registered")

    def log_artifacts(self, artifact_paths: Dict[str, str]):
        """
        アーティファクトをログ

        Args:
            artifact_paths: {local_path: artifact_path} の辞書
        """
        for local_path, artifact_path in artifact_paths.items():
            if Path(local_path).is_file():
                mlflow.log_artifact(local_path, artifact_path)
            elif Path(local_path).is_dir():
                mlflow.log_artifacts(local_path, artifact_path)

    def log_figure(self, fig: Any, name: str = "figure"):
        """matplotlib/plotly図をログ"""
        fig_path = Path(f"/tmp/{name}.png")

        if hasattr(fig, "savefig"):
            # Matplotlib
            fig.savefig(fig_path, dpi=100, bbox_inches="tight")
        elif hasattr(fig, "write_image"):
            # Plotly
            fig.write_image(fig_path)
        else:
            logger.warning(f"Unknown figure type: {type(fig)}")
            return

        mlflow.log_artifact(fig_path, artifact_path="figures")

    def track_experiment(
        self,
        train_fn: callable,
        params: Dict[str, Any],
        data: Dict[str, Any],
        model_name: str = None,
    ) -> Any:
        """
        実験全体をトラッキング

        Args:
            train_fn: 学習関数
            params: パラメータ
            data: データ辞書
            model_name: モデル名

        Returns:
            学習済みモデル
        """
        with self.start_run() as run:
            # Log parameters
            self.log_params(params)

            # Log datasets
            if "train" in data:
                self.log_dataset(data["train"], "train_data")
            if "val" in data:
                self.log_dataset(data["val"], "val_data")
            if "test" in data:
                self.log_dataset(data["test"], "test_data")

            # Train model
            logger.info("Starting training...")
            model, metrics = train_fn(params, data)

            # Log metrics
            if isinstance(metrics, dict):
                self.log_metrics(metrics)
            elif isinstance(metrics, list):
                # Assume list of epoch metrics
                for epoch, epoch_metrics in enumerate(metrics):
                    self.log_metrics(epoch_metrics, step=epoch)

            # Log model
            if model and model_name:
                self.log_model(model, model_name)

            # Log run info
            run_info = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
            }

            logger.info(f"Run completed: {run_info['run_id']}")

            return model, run_info

    def compare_models(self, experiment_name: str = None) -> pd.DataFrame:
        """
        実験内のモデルを比較

        Args:
            experiment_name: 実験名（デフォルトは現在の実験）

        Returns:
            比較結果のDataFrame
        """
        if experiment_name is None:
            experiment_name = self.experiment_name

        experiment = self.client.get_experiment_by_name(experiment_name)

        if experiment is None:
            logger.error(f"Experiment {experiment_name} not found")
            return pd.DataFrame()

        # Get all runs
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_loss ASC"],
        )

        # Create comparison DataFrame
        comparison_data = []
        for run in runs:
            row = {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
            }

            # Add metrics
            for key, value in run.data.metrics.items():
                row[f"metric_{key}"] = value

            # Add params
            for key, value in run.data.params.items():
                row[f"param_{key}"] = value

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def get_best_model(
        self,
        metric: str = "val_loss",
        ascending: bool = True,
    ) -> tuple:
        """
        最良のモデルを取得

        Args:
            metric: 評価メトリック
            ascending: 昇順でソート（小さい方が良い）

        Returns:
            (model, run_info) のタプル
        """
        experiment = self.client.get_experiment_by_name(self.experiment_name)

        # Search for best run
        order = "ASC" if ascending else "DESC"
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        if not runs:
            logger.error("No runs found")
            return None, None

        best_run = runs[0]

        # Load model
        model_uri = f"runs:/{best_run.info.run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)

        run_info = {
            "run_id": best_run.info.run_id,
            "metrics": best_run.data.metrics,
            "params": best_run.data.params,
        }

        return model, run_info


def test_mlflow_trainer():
    """MLflowトレーナーのテスト"""
    print("Testing MLflow Trainer")
    print("=" * 50)

    # Initialize trainer
    trainer = MLflowTrainer(
        experiment_name="gogooku3-test",
        tracking_uri="http://localhost:5000",
    )

    # Create dummy data
    data = {
        "train": pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            }
        )
    }

    # Define dummy training function
    def dummy_train(params, data):
        from sklearn.ensemble import RandomForestClassifier

        X = data["train"][["feature1", "feature2"]]
        y = data["train"]["target"]

        model = RandomForestClassifier(**params)
        model.fit(X, y)

        metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.93,
        }

        return model, metrics

    # Track experiment
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    }

    with trainer.start_run(run_name="test_run"):
        trainer.log_params(params)
        trainer.log_dataset(data["train"], "training_data")

        model, metrics = dummy_train(params, data)

        trainer.log_metrics(metrics)

        print("✓ MLflow tracking successful")

    # Compare models
    comparison = trainer.compare_models()
    if not comparison.empty:
        print(f"✓ Found {len(comparison)} runs")
        print(comparison.head())


if __name__ == "__main__":
    test_mlflow_trainer()
