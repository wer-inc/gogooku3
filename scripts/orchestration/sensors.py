#!/usr/bin/env python3
"""
Dagster Sensor Definitions
センサー定義（イベント駆動実行）
"""

from dagster import (
    sensor,
    RunRequest,
    SensorEvaluationContext,
    SkipReason,
    DefaultSensorStatus,
    asset_sensor,
    AssetKey,
    EventLogEntry,
    MultiAssetSensorEvaluationContext,
    multi_asset_sensor,
)
from datetime import datetime, timedelta
from pathlib import Path
import json
import requests

from .jobs import (
    daily_pipeline_job,
    ml_training_job,
    quality_check_job,
    feature_sync_job,
)

# ========== File Sensors ==========


@sensor(
    job=daily_pipeline_job,
    name="data_arrival_sensor",
    description="JQuantsデータ到着検知センサー",
    default_status=DefaultSensorStatus.RUNNING,
    minimum_interval_seconds=300,  # Check every 5 minutes
)
def data_arrival_sensor(context: SensorEvaluationContext):
    """
    データ到着を検知して日次パイプラインを起動

    - JQuants APIのデータ更新を監視
    - 新しいデータが利用可能になったら実行
    """
    # Check for new data file or API status
    data_dir = Path("/home/ubuntu/gogooku2/apps/gogooku3/data/raw")

    # Get the last processed date from cursor
    last_processed_date = context.cursor or "2024-01-01"

    # Check for new files
    new_files = []
    for file in data_dir.glob("*.parquet"):
        file_date = file.stem.split("_")[-1]  # Extract date from filename
        if file_date > last_processed_date:
            new_files.append(file)

    if new_files:
        latest_date = max(f.stem.split("_")[-1] for f in new_files)

        # Update cursor
        context.update_cursor(latest_date)

        return RunRequest(
            run_key=f"data_arrival_{latest_date}",
            partition_key=latest_date,
            tags={
                "trigger": "data_arrival",
                "date": latest_date,
                "num_files": str(len(new_files)),
            },
        )

    return SkipReason("No new data files found")


@sensor(
    job=quality_check_job,
    name="quality_alert_sensor",
    description="品質アラート検知センサー",
    default_status=DefaultSensorStatus.RUNNING,
    minimum_interval_seconds=60,
)
def quality_alert_sensor(context: SensorEvaluationContext):
    """
    品質問題を検知して緊急チェックを実行

    - エラー率の急上昇を検知
    - データ欠損を検知
    """
    # Check quality metrics
    metrics_file = Path(
        "/home/ubuntu/gogooku2/apps/gogooku3/output/quality_metrics.json"
    )

    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

        # Check error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > 0.05:  # 5% threshold
            return RunRequest(
                run_key=f"quality_alert_{datetime.now().strftime('%Y%m%d_%H%M')}",
                tags={
                    "trigger": "quality_alert",
                    "error_rate": str(error_rate),
                    "severity": "high" if error_rate > 0.1 else "medium",
                },
            )

        # Check data completeness
        completeness = metrics.get("completeness", 1.0)
        if completeness < 0.95:  # 95% threshold
            return RunRequest(
                run_key=f"completeness_alert_{datetime.now().strftime('%Y%m%d_%H%M')}",
                tags={
                    "trigger": "completeness_alert",
                    "completeness": str(completeness),
                },
            )

    return SkipReason("Quality metrics within acceptable range")


# ========== Asset Sensors ==========


@asset_sensor(
    asset_key=AssetKey("ml_dataset"),
    name="ml_dataset_ready_sensor",
    description="MLデータセット準備完了センサー",
    job=ml_training_job,
    default_status=DefaultSensorStatus.RUNNING,
)
def ml_dataset_ready_sensor(
    context: SensorEvaluationContext, asset_event: EventLogEntry
):
    """
    MLデータセットが準備できたら学習を開始

    - ml_datasetアセットの成功を検知
    - 一定量のデータが蓄積されたら学習開始
    """
    # Get metadata from the asset materialization
    metadata = asset_event.dagster_event.event_specific_data.materialization.metadata

    num_samples = metadata.get("num_samples", 0)
    num_features = metadata.get("num_features", 0)

    # Check if we have enough data for training
    if num_samples >= 1000 and num_features >= 50:
        return RunRequest(
            run_key=f"ml_training_{asset_event.run_id}",
            tags={
                "trigger": "dataset_ready",
                "num_samples": str(num_samples),
                "num_features": str(num_features),
                "source_run": asset_event.run_id,
            },
        )

    return SkipReason(
        f"Not enough data: {num_samples} samples, {num_features} features"
    )


@multi_asset_sensor(
    monitored_assets=[
        AssetKey("price_data"),
        AssetKey("topix_data"),
    ],
    name="data_sync_sensor",
    description="データ同期完了センサー",
    job=feature_sync_job,
    default_status=DefaultSensorStatus.RUNNING,
)
def data_sync_sensor(context: MultiAssetSensorEvaluationContext):
    """
    複数データソースの同期完了を検知

    - price_dataとtopix_dataの両方が更新されたら実行
    - データの整合性を保証
    """
    # Check if both assets have been materialized for the same partition
    materializations = {}

    for asset_key, events in context.latest_materialization_records_by_key().items():
        if events:
            latest_event = events[0]
            partition = latest_event.dagster_event.partition
            if partition:
                materializations[asset_key.to_string()] = partition

    # Check if both assets have the same partition
    if len(materializations) == 2:
        partitions = list(materializations.values())
        if partitions[0] == partitions[1]:
            return RunRequest(
                run_key=f"sync_{partitions[0]}",
                partition_key=partitions[0],
                tags={
                    "trigger": "data_sync",
                    "partition": partitions[0],
                },
            )

    return SkipReason(
        "Waiting for all assets to be materialized for the same partition"
    )


# ========== API Sensors ==========


@sensor(
    job=daily_pipeline_job,
    name="jquants_api_sensor",
    description="JQuants API状態監視センサー",
    default_status=DefaultSensorStatus.STOPPED,  # Enable manually
    minimum_interval_seconds=600,  # Check every 10 minutes
)
def jquants_api_sensor(context: SensorEvaluationContext):
    """
    JQuants APIの状態を監視

    - API利用可能状態をチェック
    - メンテナンス終了を検知
    """
    try:
        # Check JQuants API health (hypothetical endpoint)
        response = requests.get("https://api.jquants.com/health", timeout=10)

        if response.status_code == 200:
            api_status = response.json()

            # Check if new data is available
            if api_status.get("new_data_available"):
                latest_date = api_status.get("latest_date")

                # Check if we haven't processed this date yet
                if context.cursor != latest_date:
                    context.update_cursor(latest_date)

                    return RunRequest(
                        run_key=f"api_new_data_{latest_date}",
                        partition_key=latest_date,
                        tags={
                            "trigger": "api_sensor",
                            "api_status": "new_data",
                            "date": latest_date,
                        },
                    )
    except Exception as e:
        context.log.error(f"API check failed: {e}")

    return SkipReason("No new data from API")


# ========== System Sensors ==========


@sensor(
    job=quality_check_job,
    name="disk_space_sensor",
    description="ディスク容量監視センサー",
    default_status=DefaultSensorStatus.RUNNING,
    minimum_interval_seconds=3600,  # Check every hour
)
def disk_space_sensor(context: SensorEvaluationContext):
    """
    ディスク容量を監視してクリーンアップを実行

    - 容量が閾値を超えたらアラート
    - 自動クリーンアップを起動
    """
    import shutil

    # Check disk usage
    usage = shutil.disk_usage("/home/ubuntu/gogooku2/apps/gogooku3/output")
    usage_percent = (usage.used / usage.total) * 100

    if usage_percent > 80:  # 80% threshold
        return RunRequest(
            run_key=f"disk_alert_{datetime.now().strftime('%Y%m%d_%H%M')}",
            tags={
                "trigger": "disk_space",
                "usage_percent": f"{usage_percent:.1f}",
                "severity": "high" if usage_percent > 90 else "medium",
                "action": "cleanup_required",
            },
        )

    return SkipReason(f"Disk usage OK: {usage_percent:.1f}%")


@sensor(
    job=daily_pipeline_job,
    name="retry_failed_runs_sensor",
    description="失敗実行のリトライセンサー",
    default_status=DefaultSensorStatus.STOPPED,  # Enable manually
    minimum_interval_seconds=1800,  # Check every 30 minutes
)
def retry_failed_runs_sensor(context: SensorEvaluationContext):
    """
    失敗した実行を検知して自動リトライ

    - 一定回数まで自動リトライ
    - エクスポネンシャルバックオフ
    """
    # This would query the Dagster instance for failed runs
    # For demonstration, using a simple implementation

    failed_runs_file = Path("/tmp/failed_runs.json")
    if failed_runs_file.exists():
        with open(failed_runs_file) as f:
            failed_runs = json.load(f)

        for run_id, run_info in failed_runs.items():
            retry_count = run_info.get("retry_count", 0)

            if retry_count < 3:  # Max 3 retries
                # Exponential backoff
                last_failure = datetime.fromisoformat(run_info["failed_at"])
                wait_time = timedelta(minutes=10 * (2**retry_count))

                if datetime.now() > last_failure + wait_time:
                    return RunRequest(
                        run_key=f"retry_{run_id}_{retry_count + 1}",
                        partition_key=run_info.get("partition_key"),
                        tags={
                            "trigger": "retry",
                            "original_run": run_id,
                            "retry_count": str(retry_count + 1),
                        },
                    )

    return SkipReason("No failed runs to retry")


# ========== All Sensors ==========

all_sensors = [
    data_arrival_sensor,
    quality_alert_sensor,
    ml_dataset_ready_sensor,
    data_sync_sensor,
    jquants_api_sensor,
    disk_space_sensor,
    retry_failed_runs_sensor,
]

# Production sensors (auto-enabled)
production_sensors = [
    data_arrival_sensor,
    quality_alert_sensor,
    ml_dataset_ready_sensor,
    data_sync_sensor,
    disk_space_sensor,
]

# Optional sensors (manually enabled)
optional_sensors = [
    jquants_api_sensor,
    retry_failed_runs_sensor,
]


# ========== Test Function ==========


def test_sensors():
    """センサー定義のテスト"""
    print("Dagster Sensors Defined:")
    print("=" * 50)

    print("\nProduction Sensors (Auto-enabled):")
    sensors_info = [
        ("data_arrival", "File", "Detect new data files"),
        ("quality_alert", "Metrics", "Monitor quality metrics"),
        ("ml_dataset_ready", "Asset", "ML dataset completion"),
        ("data_sync", "Multi-Asset", "Price + TOPIX sync"),
        ("disk_space", "System", "Disk usage monitoring"),
    ]

    for name, type_, desc in sensors_info:
        print(f"  - {name} ({type_}): {desc}")

    print("\nOptional Sensors (Manually enabled):")
    optional_info = [
        ("jquants_api", "API", "JQuants API monitoring"),
        ("retry_failed", "System", "Auto-retry failed runs"),
    ]

    for name, type_, desc in optional_info:
        print(f"  - {name} ({type_}): {desc}")

    print("\nSensor Types:")
    print("  - File Sensors: Monitor file system changes")
    print("  - Asset Sensors: React to asset materializations")
    print("  - API Sensors: Monitor external APIs")
    print("  - System Sensors: Monitor system resources")


if __name__ == "__main__":
    test_sensors()
