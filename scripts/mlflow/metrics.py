#!/usr/bin/env python3
"""
MLflow Metrics Tracking
メトリクス追跡とモニタリング
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def track_metrics(
    metrics: dict[str, float],
    step: int | None = None,
    run_id: str | None = None,
):
    """
    メトリクスをトラッキング

    Args:
        metrics: メトリクス辞書
        step: ステップ番号（エポックなど）
        run_id: 実行ID（現在の実行を使用する場合はNone）
    """
    if run_id:
        with mlflow.start_run(run_id=run_id):
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
    else:
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)


class MetricsTracker:
    """高度なメトリクストラッキング"""

    def __init__(self, tracking_uri: str = None):
        """
        Initialize metrics tracker

        Args:
            tracking_uri: MLflow tracking server URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )

        self.client = MlflowClient()
        self.metrics_buffer = []
        self.step_counter = 0

    def log_batch_metrics(
        self,
        metrics_list: list[dict[str, float]],
        run_id: str | None = None,
    ):
        """
        バッチでメトリクスをログ

        Args:
            metrics_list: メトリクス辞書のリスト
            run_id: 実行ID
        """
        if run_id is None:
            run = mlflow.active_run()
            if run is None:
                raise ValueError("No active run and no run_id provided")
            run_id = run.info.run_id

        # Flatten metrics
        metrics_batch = []
        for step, metrics in enumerate(metrics_list):
            for key, value in metrics.items():
                metrics_batch.append(
                    mlflow.entities.Metric(
                        key, value, int(datetime.now().timestamp() * 1000), step
                    )
                )

        # Log batch
        self.client.log_batch(run_id, metrics=metrics_batch)
        logger.info(f"Logged {len(metrics_batch)} metrics in batch")

    def track_training_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: dict[str, float] = None,
        val_metrics: dict[str, float] = None,
        learning_rate: float = None,
    ):
        """
        学習メトリクスをトラッキング

        Args:
            epoch: エポック番号
            train_loss: 訓練損失
            val_loss: 検証損失
            train_metrics: 訓練メトリクス
            val_metrics: 検証メトリクス
            learning_rate: 学習率
        """
        # Core metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Additional train metrics
        if train_metrics:
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train_{key}", value, step=epoch)

        # Additional val metrics
        if val_metrics:
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value, step=epoch)

        # Learning rate
        if learning_rate is not None:
            mlflow.log_metric("learning_rate", learning_rate, step=epoch)

        # Overfitting metric
        overfitting_score = val_loss - train_loss
        mlflow.log_metric("overfitting_score", overfitting_score, step=epoch)

    def track_prediction_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "test",
    ):
        """
        予測メトリクスをトラッキング

        Args:
            y_true: 正解ラベル
            y_pred: 予測値
            prefix: メトリック名のプレフィックス
        """
        from sklearn.metrics import (
            accuracy_score,
            mean_absolute_error,
            mean_squared_error,
            precision_recall_fscore_support,
            r2_score,
            roc_auc_score,
        )

        # Check if classification or regression
        is_classification = len(np.unique(y_true)) < 20  # Simple heuristic

        if is_classification:
            # Classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted"
            )

            mlflow.log_metric(f"{prefix}_accuracy", accuracy)
            mlflow.log_metric(f"{prefix}_precision", precision)
            mlflow.log_metric(f"{prefix}_recall", recall)
            mlflow.log_metric(f"{prefix}_f1", f1)

            # ROC AUC if binary
            if len(np.unique(y_true)) == 2:
                try:
                    auc = roc_auc_score(y_true, y_pred)
                    mlflow.log_metric(f"{prefix}_auc", auc)
                except Exception:
                    pass
        else:
            # Regression metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            mlflow.log_metric(f"{prefix}_mse", mse)
            mlflow.log_metric(f"{prefix}_mae", mae)
            mlflow.log_metric(f"{prefix}_rmse", np.sqrt(mse))
            mlflow.log_metric(f"{prefix}_r2", r2)

    def track_resource_metrics(
        self,
        memory_usage_mb: float,
        gpu_usage_percent: float = None,
        cpu_usage_percent: float = None,
        training_time_seconds: float = None,
    ):
        """
        リソース使用状況をトラッキング

        Args:
            memory_usage_mb: メモリ使用量(MB)
            gpu_usage_percent: GPU使用率(%)
            cpu_usage_percent: CPU使用率(%)
            training_time_seconds: 学習時間(秒)
        """
        mlflow.log_metric("memory_usage_mb", memory_usage_mb)

        if gpu_usage_percent is not None:
            mlflow.log_metric("gpu_usage_percent", gpu_usage_percent)

        if cpu_usage_percent is not None:
            mlflow.log_metric("cpu_usage_percent", cpu_usage_percent)

        if training_time_seconds is not None:
            mlflow.log_metric("training_time_seconds", training_time_seconds)
            mlflow.log_metric("training_time_hours", training_time_seconds / 3600)

    def track_data_quality_metrics(
        self,
        df: pd.DataFrame,
        prefix: str = "data",
    ):
        """
        データ品質メトリクスをトラッキング

        Args:
            df: データフレーム
            prefix: メトリック名のプレフィックス
        """
        # Basic stats
        mlflow.log_metric(f"{prefix}_num_rows", len(df))
        mlflow.log_metric(f"{prefix}_num_cols", len(df.columns))

        # Null values
        null_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        mlflow.log_metric(f"{prefix}_null_ratio", null_ratio)

        # Duplicates
        duplicate_ratio = df.duplicated().sum() / len(df)
        mlflow.log_metric(f"{prefix}_duplicate_ratio", duplicate_ratio)

        # Numeric columns stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:10]:  # Limit to first 10
                mlflow.log_metric(f"{prefix}_{col}_mean", df[col].mean())
                mlflow.log_metric(f"{prefix}_{col}_std", df[col].std())
                mlflow.log_metric(f"{prefix}_{col}_skew", df[col].skew())

    def track_feature_importance(
        self,
        feature_names: list[str],
        importance_scores: np.ndarray,
        top_k: int = 20,
    ):
        """
        特徴量重要度をトラッキング

        Args:
            feature_names: 特徴量名リスト
            importance_scores: 重要度スコア
            top_k: トップK個の特徴量をログ
        """
        # Sort by importance
        indices = np.argsort(importance_scores)[::-1][:top_k]

        for i, idx in enumerate(indices):
            mlflow.log_metric(
                f"feature_importance_{i:02d}_{feature_names[idx]}",
                importance_scores[idx],
            )

        # Save full importance as artifact
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance_scores,
            }
        ).sort_values("importance", ascending=False)

        importance_path = Path("/tmp/feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, artifact_path="analysis")

    def get_metrics_history(
        self,
        run_id: str,
        metric_keys: list[str] = None,
    ) -> pd.DataFrame:
        """
        メトリクス履歴を取得

        Args:
            run_id: 実行ID
            metric_keys: 取得するメトリックキー（Noneで全て）

        Returns:
            メトリクス履歴のDataFrame
        """
        run = self.client.get_run(run_id)

        if metric_keys is None:
            metric_keys = list(run.data.metrics.keys())

        metrics_data = []
        for key in metric_keys:
            history = self.client.get_metric_history(run_id, key)
            for metric in history:
                metrics_data.append(
                    {
                        "metric": key,
                        "value": metric.value,
                        "step": metric.step,
                        "timestamp": datetime.fromtimestamp(metric.timestamp / 1000),
                    }
                )

        return pd.DataFrame(metrics_data)

    def compare_runs_metrics(
        self,
        run_ids: list[str],
        metric_keys: list[str],
    ) -> pd.DataFrame:
        """
        複数の実行のメトリクスを比較

        Args:
            run_ids: 実行IDリスト
            metric_keys: 比較するメトリックキー

        Returns:
            比較結果のDataFrame
        """
        comparison_data = []

        for run_id in run_ids:
            run = self.client.get_run(run_id)
            row = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
            }

            for key in metric_keys:
                if key in run.data.metrics:
                    row[key] = run.data.metrics[key]
                else:
                    row[key] = None

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)


def test_metrics_tracking():
    """メトリクストラッキングのテスト"""

    print("Testing MLflow Metrics Tracking")
    print("=" * 50)

    # Initialize tracker
    tracker = MetricsTracker()

    # Create test data
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)

    # Start a test run
    with mlflow.start_run(run_name="test_metrics"):
        # Track training metrics
        for epoch in range(5):
            tracker.track_training_metrics(
                epoch=epoch,
                train_loss=0.5 - epoch * 0.05,
                val_loss=0.6 - epoch * 0.04,
                train_metrics={"accuracy": 0.8 + epoch * 0.02},
                val_metrics={"accuracy": 0.75 + epoch * 0.015},
                learning_rate=0.001 * (0.9**epoch),
            )

        print("✓ Training metrics tracked")

        # Track prediction metrics
        tracker.track_prediction_metrics(y_true, y_pred, prefix="test")
        print("✓ Prediction metrics tracked")

        # Track resource metrics
        tracker.track_resource_metrics(
            memory_usage_mb=1024,
            gpu_usage_percent=75.5,
            cpu_usage_percent=45.2,
            training_time_seconds=3600,
        )
        print("✓ Resource metrics tracked")

        # Track data quality
        test_df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        tracker.track_data_quality_metrics(test_df, prefix="test_data")
        print("✓ Data quality metrics tracked")

        # Track feature importance
        feature_names = [f"feature_{i}" for i in range(10)]
        importance_scores = np.random.rand(10)
        tracker.track_feature_importance(feature_names, importance_scores, top_k=5)
        print("✓ Feature importance tracked")


if __name__ == "__main__":
    test_metrics_tracking()
