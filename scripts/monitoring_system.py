#!/usr/bin/env python3
"""
Monitoring System Best Practices for gogooku3
モニタリングシステムのベストプラクティス実装
"""

import sys
import logging
import time
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime
import threading
import queue
import psutil
import torch
import numpy as np
import polars as pl
from dataclasses import dataclass
import matplotlib.pyplot as plt

# パスを追加
sys.path.append(str(Path(__file__).parent.parent))

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""

    timestamp: datetime
    cpu_usage: float
    memory_usage_gb: float
    gpu_memory_gb: Optional[float] = None
    execution_time: Optional[float] = None
    throughput: Optional[float] = None
    error_count: int = 0
    success_count: int = 0


@dataclass
class ModelMetrics:
    """モデルメトリクス"""

    timestamp: datetime
    model_name: str
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_return: float
    volatility: float
    model_version: str = "1.0.0"


@dataclass
class DataQualityMetrics:
    """データ品質メトリクス"""

    timestamp: datetime
    total_records: int
    null_ratio: float
    duplicate_ratio: float
    data_freshness_hours: float
    validation_errors: int = 0


class MetricsCollector:
    """メトリクス収集クラス"""

    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.metrics_queue = queue.Queue()
        self.is_collecting = False
        self.collection_thread = None

        # データベース初期化
        self._init_database()

    def _init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # パフォーマンスメトリクステーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_usage REAL,
                memory_usage_gb REAL,
                gpu_memory_gb REAL,
                execution_time REAL,
                throughput REAL,
                error_count INTEGER,
                success_count INTEGER
            )
        """)

        # モデルメトリクステーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                total_return REAL,
                volatility REAL,
                model_version TEXT
            )
        """)

        # データ品質メトリクステーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_records INTEGER,
                null_ratio REAL,
                duplicate_ratio REAL,
                data_freshness_hours REAL,
                validation_errors INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def start_collection(self, interval_seconds: int = 60):
        """メトリクス収集開始"""
        if self.is_collecting:
            logger.warning("Metrics collection already running")
            return

        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, args=(interval_seconds,), daemon=True
        )
        self.collection_thread.start()
        logger.info(f"Started metrics collection (interval: {interval_seconds}s)")

    def stop_collection(self):
        """メトリクス収集停止"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Stopped metrics collection")

    def _collection_loop(self, interval_seconds: int):
        """メトリクス収集ループ"""
        while self.is_collecting:
            try:
                # システムメトリクス収集
                metrics = self._collect_system_metrics()
                self._save_performance_metrics(metrics)

                # キューメトリクス処理
                self._process_queued_metrics()

                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(interval_seconds)

    def _collect_system_metrics(self) -> PerformanceMetrics:
        """システムメトリクス収集"""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB

        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB

        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage_gb=memory_usage,
            gpu_memory_gb=gpu_memory,
        )

    def _save_performance_metrics(self, metrics: PerformanceMetrics):
        """パフォーマンスメトリクス保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO performance_metrics
            (timestamp, cpu_usage, memory_usage_gb, gpu_memory_gb,
             execution_time, throughput, error_count, success_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics.timestamp.isoformat(),
                metrics.cpu_usage,
                metrics.memory_usage_gb,
                metrics.gpu_memory_gb,
                metrics.execution_time,
                metrics.throughput,
                metrics.error_count,
                metrics.success_count,
            ),
        )

        conn.commit()
        conn.close()

    def _process_queued_metrics(self):
        """キューされたメトリクス処理"""
        while not self.metrics_queue.empty():
            try:
                metric = self.metrics_queue.get_nowait()
                if isinstance(metric, ModelMetrics):
                    self._save_model_metrics(metric)
                elif isinstance(metric, DataQualityMetrics):
                    self._save_data_quality_metrics(metric)
            except queue.Empty:
                break

    def _save_model_metrics(self, metrics: ModelMetrics):
        """モデルメトリクス保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO model_metrics
            (timestamp, model_name, sharpe_ratio, max_drawdown,
             win_rate, total_return, volatility, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics.timestamp.isoformat(),
                metrics.model_name,
                metrics.sharpe_ratio,
                metrics.max_drawdown,
                metrics.win_rate,
                metrics.total_return,
                metrics.volatility,
                metrics.model_version,
            ),
        )

        conn.commit()
        conn.close()

    def _save_data_quality_metrics(self, metrics: DataQualityMetrics):
        """データ品質メトリクス保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO data_quality_metrics
            (timestamp, total_records, null_ratio, duplicate_ratio,
             data_freshness_hours, validation_errors)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                metrics.timestamp.isoformat(),
                metrics.total_records,
                metrics.null_ratio,
                metrics.duplicate_ratio,
                metrics.data_freshness_hours,
                metrics.validation_errors,
            ),
        )

        conn.commit()
        conn.close()

    def queue_metric(self, metric: Union[ModelMetrics, DataQualityMetrics]):
        """メトリクスをキューに追加"""
        self.metrics_queue.put(metric)


class ModelMonitor:
    """モデル監視クラス"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_thresholds = {
            "sharpe_ratio_min": 0.5,
            "max_drawdown_max": -0.2,
            "win_rate_min": 0.5,
            "data_freshness_max_hours": 24,
        }

    def monitor_model_performance(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        model_name: str = "ATFT-GAT-FAN",
    ) -> ModelMetrics:
        """モデルパフォーマンス監視"""
        # パフォーマンス計算
        sharpe_ratio = self._calculate_sharpe_ratio(predictions, actual_returns)
        max_drawdown = self._calculate_max_drawdown(predictions, actual_returns)
        win_rate = self._calculate_win_rate(predictions, actual_returns)
        total_return = self._calculate_total_return(predictions, actual_returns)
        volatility = self._calculate_volatility(predictions, actual_returns)

        # メトリクス作成
        metrics = ModelMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_return=total_return,
            volatility=volatility,
        )

        # アラートチェック
        self._check_alerts(metrics)

        # メトリクス保存
        self.metrics_collector.queue_metric(metrics)

        return metrics

    def _calculate_sharpe_ratio(
        self, predictions: np.ndarray, actual_returns: np.ndarray
    ) -> float:
        """シャープレシオ計算"""
        returns = predictions - actual_returns
        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        return mean_return / std_return

    def _calculate_max_drawdown(
        self, predictions: np.ndarray, actual_returns: np.ndarray
    ) -> float:
        """最大ドローダウン計算"""
        returns = predictions - actual_returns
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    def _calculate_win_rate(
        self, predictions: np.ndarray, actual_returns: np.ndarray
    ) -> float:
        """勝率計算"""
        returns = predictions - actual_returns
        wins = np.sum(returns > 0)
        total = len(returns)
        return wins / total if total > 0 else 0.0

    def _calculate_total_return(
        self, predictions: np.ndarray, actual_returns: np.ndarray
    ) -> float:
        """総リターン計算"""
        returns = predictions - actual_returns
        return np.prod(1 + returns) - 1

    def _calculate_volatility(
        self, predictions: np.ndarray, actual_returns: np.ndarray
    ) -> float:
        """ボラティリティ計算"""
        returns = predictions - actual_returns
        return np.std(returns)

    def _check_alerts(self, metrics: ModelMetrics):
        """アラートチェック"""
        alerts = []

        if metrics.sharpe_ratio < self.alert_thresholds["sharpe_ratio_min"]:
            alerts.append(f"Low Sharpe ratio: {metrics.sharpe_ratio:.3f}")

        if metrics.max_drawdown < self.alert_thresholds["max_drawdown_max"]:
            alerts.append(f"High drawdown: {metrics.max_drawdown:.3f}")

        if metrics.win_rate < self.alert_thresholds["win_rate_min"]:
            alerts.append(f"Low win rate: {metrics.win_rate:.3f}")

        if alerts:
            logger.warning(f"Model alerts for {metrics.model_name}: {alerts}")


class DataQualityMonitor:
    """データ品質監視クラス"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector

    def monitor_data_quality(
        self, df: pl.DataFrame, data_source: str = "ml_dataset"
    ) -> DataQualityMetrics:
        """データ品質監視"""
        # 品質メトリクス計算
        total_records = len(df)
        null_ratio = self._calculate_null_ratio(df)
        duplicate_ratio = self._calculate_duplicate_ratio(df)
        data_freshness = self._calculate_data_freshness(df)
        validation_errors = self._validate_data(df)

        # メトリクス作成
        metrics = DataQualityMetrics(
            timestamp=datetime.now(),
            total_records=total_records,
            null_ratio=null_ratio,
            duplicate_ratio=duplicate_ratio,
            data_freshness_hours=data_freshness,
            validation_errors=validation_errors,
        )

        # メトリクス保存
        self.metrics_collector.queue_metric(metrics)

        return metrics

    def _calculate_null_ratio(self, df: pl.DataFrame) -> float:
        """欠損値比率計算"""
        null_counts = df.null_count()
        total_cells = len(df) * len(df.columns)
        return (
            sum(null_counts.to_dict().values()) / total_cells
            if total_cells > 0
            else 0.0
        )

    def _calculate_duplicate_ratio(self, df: pl.DataFrame) -> float:
        """重複比率計算"""
        total_rows = len(df)
        unique_rows = len(df.unique())
        return (total_rows - unique_rows) / total_rows if total_rows > 0 else 0.0

    def _calculate_data_freshness(self, df: pl.DataFrame) -> float:
        """データ鮮度計算（時間）"""
        if "date" not in df.columns:
            return 0.0

        try:
            latest_date = df["date"].max()
            if latest_date is None:
                return 0.0

            if isinstance(latest_date, str):
                latest_date = datetime.fromisoformat(latest_date)

            time_diff = datetime.now() - latest_date
            return time_diff.total_seconds() / 3600  # 時間単位
        except Exception:
            return 0.0

    def _validate_data(self, df: pl.DataFrame) -> int:
        """データ検証エラー数"""
        error_count = 0

        # 必須列チェック
        required_columns = ["code", "date"]
        for col in required_columns:
            if col not in df.columns:
                error_count += 1

        # データ型チェック
        if "date" in df.columns:
            try:
                df.select(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d"))
            except Exception:
                error_count += 1

        return error_count


class MonitoringDashboard:
    """モニタリングダッシュボード"""

    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path

    def generate_performance_report(self, hours: int = 24) -> Dict:
        """パフォーマンスレポート生成"""
        conn = sqlite3.connect(self.db_path)

        # 最新のメトリクス取得
        query = """
            SELECT * FROM performance_metrics
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        """.format(hours)

        df = pl.read_database(query, conn)
        conn.close()

        if len(df) == 0:
            return {"error": "No data available"}

        # 統計計算
        report = {
            "period_hours": hours,
            "data_points": len(df),
            "cpu": {
                "mean": df["cpu_usage"].mean(),
                "max": df["cpu_usage"].max(),
                "min": df["cpu_usage"].min(),
            },
            "memory": {
                "mean_gb": df["memory_usage_gb"].mean(),
                "max_gb": df["memory_usage_gb"].max(),
                "min_gb": df["memory_usage_gb"].min(),
            },
        }

        if "gpu_memory_gb" in df.columns:
            report["gpu"] = {
                "mean_gb": df["gpu_memory_gb"].mean(),
                "max_gb": df["gpu_memory_gb"].max(),
                "min_gb": df["gpu_memory_gb"].min(),
            }

        return report

    def generate_model_report(self, hours: int = 24) -> Dict:
        """モデルレポート生成"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT * FROM model_metrics
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        """.format(hours)

        df = pl.read_database(query, conn)
        conn.close()

        if len(df) == 0:
            return {"error": "No model data available"}

        report = {
            "period_hours": hours,
            "data_points": len(df),
            "models": df["model_name"].unique().to_list(),
            "latest_metrics": {},
        }

        # 最新のメトリクス
        for model in report["models"]:
            model_data = df.filter(pl.col("model_name") == model)
            if len(model_data) > 0:
                latest = model_data.head(1)
                report["latest_metrics"][model] = {
                    "sharpe_ratio": latest["sharpe_ratio"][0],
                    "max_drawdown": latest["max_drawdown"][0],
                    "win_rate": latest["win_rate"][0],
                    "total_return": latest["total_return"][0],
                    "volatility": latest["volatility"][0],
                    "timestamp": latest["timestamp"][0],
                }

        return report

    def plot_performance_trends(self, hours: int = 24, save_path: str = None):
        """パフォーマンストレンドの可視化"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT * FROM performance_metrics
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp
        """.format(hours)

        df = pl.read_database(query, conn)
        conn.close()

        if len(df) == 0:
            logger.warning("No performance data available for plotting")
            return

        # プロット作成
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # CPU使用率
        axes[0, 0].plot(df["timestamp"], df["cpu_usage"])
        axes[0, 0].set_title("CPU Usage (%)")
        axes[0, 0].set_ylabel("Usage %")

        # メモリ使用量
        axes[0, 1].plot(df["timestamp"], df["memory_usage_gb"])
        axes[0, 1].set_title("Memory Usage (GB)")
        axes[0, 1].set_ylabel("GB")

        # GPU使用量（利用可能な場合）
        if "gpu_memory_gb" in df.columns:
            axes[1, 0].plot(df["timestamp"], df["gpu_memory_gb"])
            axes[1, 0].set_title("GPU Memory Usage (GB)")
            axes[1, 0].set_ylabel("GB")

        # エラー数
        if "error_count" in df.columns:
            axes[1, 1].plot(df["timestamp"], df["error_count"])
            axes[1, 1].set_title("Error Count")
            axes[1, 1].set_ylabel("Count")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Performance plot saved: {save_path}")
        else:
            plt.show()


def main():
    """テスト実行"""
    # メトリクス収集開始
    collector = MetricsCollector()
    collector.start_collection(interval_seconds=30)

    # モデル監視
    model_monitor = ModelMonitor(collector)

    # データ品質監視
    data_monitor = DataQualityMonitor(collector)

    # サンプルデータでのテスト
    print("📊 Testing monitoring system...")

    # モデルパフォーマンステスト
    predictions = np.random.randn(100) * 0.01
    actual_returns = np.random.randn(100) * 0.01
    model_metrics = model_monitor.monitor_model_performance(predictions, actual_returns)
    print(f"Model metrics: {model_metrics}")

    # データ品質テスト
    sample_data = pl.DataFrame(
        {
            "code": ["1234"] * 50,
            "date": [f"2024-01-{i:02d}" for i in range(1, 51)],
            "close": np.random.randn(50) * 100 + 1000,
        }
    )
    data_metrics = data_monitor.monitor_data_quality(sample_data)
    print(f"Data quality metrics: {data_metrics}")

    # 少し待ってからレポート生成
    time.sleep(5)

    # ダッシュボード
    dashboard = MonitoringDashboard()
    performance_report = dashboard.generate_performance_report(hours=1)
    model_report = dashboard.generate_model_report(hours=1)

    print(f"Performance report: {json.dumps(performance_report, indent=2)}")
    print(f"Model report: {json.dumps(model_report, indent=2)}")

    # 収集停止
    collector.stop_collection()


if __name__ == "__main__":
    main()
