#!/usr/bin/env python3
"""
Dagster Schedule Definitions
スケジュール定義
"""

from dagster import (
    schedule,
    RunRequest,
    ScheduleEvaluationContext,
    DefaultScheduleStatus,
)
from datetime import timedelta
import pytz

from .jobs import (
    daily_pipeline_job,
    ml_training_job,
    quality_check_job,
    feature_sync_job,
    cleanup_job,
)

# JST timezone
JST = pytz.timezone("Asia/Tokyo")

# ========== Schedules ==========


@schedule(
    job=daily_pipeline_job,
    cron_schedule="0 19 * * 1-5",  # 19:00 JST on weekdays
    name="daily_pipeline_schedule",
    description="日次データパイプライン（平日19:00実行）",
    default_status=DefaultScheduleStatus.RUNNING,
    execution_timezone="Asia/Tokyo",
)
def daily_schedule(context: ScheduleEvaluationContext):
    """
    日次データパイプラインスケジュール

    - 平日19:00に実行（市場クローズ後）
    - 当日のデータを処理
    """
    scheduled_date = context.scheduled_execution_time.date()

    # Skip weekends and holidays
    if scheduled_date.weekday() >= 5:  # Saturday or Sunday
        context.log.info(f"Skipping weekend: {scheduled_date}")
        return []

    return RunRequest(
        run_key=f"daily_{scheduled_date}",
        partition_key=scheduled_date.strftime("%Y-%m-%d"),
        tags={
            "schedule": "daily",
            "date": scheduled_date.strftime("%Y-%m-%d"),
        },
    )


@schedule(
    job=ml_training_job,
    cron_schedule="0 2 * * 0",  # Sunday 02:00 JST
    name="weekly_ml_training",
    description="週次MLモデル学習（日曜日02:00実行）",
    default_status=DefaultScheduleStatus.RUNNING,
    execution_timezone="Asia/Tokyo",
)
def weekly_schedule(context: ScheduleEvaluationContext):
    """
    週次MLモデル学習スケジュール

    - 日曜日の深夜に実行
    - 過去1週間のデータで学習
    """
    scheduled_date = context.scheduled_execution_time.date()
    start_date = scheduled_date - timedelta(days=7)

    return RunRequest(
        run_key=f"weekly_ml_{scheduled_date}",
        tags={
            "schedule": "weekly",
            "week": scheduled_date.strftime("%Y-W%U"),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": scheduled_date.strftime("%Y-%m-%d"),
        },
    )


@schedule(
    job=quality_check_job,
    cron_schedule="*/30 9-15 * * 1-5",  # Every 30 min during market hours
    name="intraday_quality_check",
    description="日中品質チェック（市場時間中30分毎）",
    default_status=DefaultScheduleStatus.STOPPED,  # Start manually
    execution_timezone="Asia/Tokyo",
)
def intraday_quality_schedule(context: ScheduleEvaluationContext):
    """
    日中品質チェックスケジュール

    - 市場時間中（9:00-15:00）に30分毎実行
    - リアルタイムデータの品質監視
    """
    current_time = context.scheduled_execution_time

    return RunRequest(
        run_key=f"quality_{current_time.strftime('%Y%m%d_%H%M')}",
        tags={
            "schedule": "intraday",
            "time": current_time.strftime("%H:%M"),
        },
    )


@schedule(
    job=feature_sync_job,
    cron_schedule="0 20 * * 1-5",  # 20:00 JST on weekdays
    name="feature_store_sync",
    description="Feature Store同期（平日20:00実行）",
    default_status=DefaultScheduleStatus.RUNNING,
    execution_timezone="Asia/Tokyo",
)
def feature_sync_schedule(context: ScheduleEvaluationContext):
    """
    Feature Store同期スケジュール

    - 日次パイプライン完了後に実行
    - オンラインストアへの同期
    """
    scheduled_date = context.scheduled_execution_time.date()

    return RunRequest(
        run_key=f"feature_sync_{scheduled_date}",
        partition_key=scheduled_date.strftime("%Y-%m-%d"),
        tags={
            "schedule": "daily",
            "type": "sync",
        },
    )


@schedule(
    job=cleanup_job,
    cron_schedule="0 3 * * 0",  # Sunday 03:00 JST
    name="weekly_cleanup",
    description="週次クリーンアップ（日曜日03:00実行）",
    default_status=DefaultScheduleStatus.RUNNING,
    execution_timezone="Asia/Tokyo",
)
def cleanup_schedule(context: ScheduleEvaluationContext):
    """
    週次クリーンアップスケジュール

    - 古いパーティションデータの削除
    - ログファイルのローテーション
    """
    retention_days = 90  # Keep 90 days of data
    cutoff_date = context.scheduled_execution_time.date() - timedelta(
        days=retention_days
    )

    return RunRequest(
        run_key=f"cleanup_{context.scheduled_execution_time.date()}",
        tags={
            "schedule": "weekly",
            "type": "maintenance",
            "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
        },
    )


# ========== Conditional Schedules ==========


@schedule(
    job=daily_pipeline_job,
    cron_schedule="0 6 * * 1-5",  # 06:00 JST on weekdays
    name="morning_catchup",
    description="朝のキャッチアップ実行（失敗時のリトライ）",
    default_status=DefaultScheduleStatus.STOPPED,
    execution_timezone="Asia/Tokyo",
)
def morning_catchup_schedule(context: ScheduleEvaluationContext):
    """
    朝のキャッチアップスケジュール

    - 前日の実行が失敗した場合のリトライ
    - 手動で有効化する
    """
    yesterday = context.scheduled_execution_time.date() - timedelta(days=1)

    # Check if yesterday's run failed (would need to query run status)
    # For now, always create a run request

    return RunRequest(
        run_key=f"catchup_{yesterday}",
        partition_key=yesterday.strftime("%Y-%m-%d"),
        tags={
            "schedule": "catchup",
            "retry": "true",
            "original_date": yesterday.strftime("%Y-%m-%d"),
        },
    )


# ========== Schedule Groups ==========

# All schedules
all_schedules = [
    daily_schedule,
    weekly_schedule,
    intraday_quality_schedule,
    feature_sync_schedule,
    cleanup_schedule,
    morning_catchup_schedule,
]

# Production schedules (auto-enabled)
production_schedules = [
    daily_schedule,
    weekly_schedule,
    feature_sync_schedule,
    cleanup_schedule,
]

# Optional schedules (manually enabled)
optional_schedules = [
    intraday_quality_schedule,
    morning_catchup_schedule,
]


# ========== Test Function ==========


def test_schedules():
    """スケジュール定義のテスト"""
    print("Dagster Schedules Defined:")
    print("=" * 50)

    print("\nProduction Schedules (Auto-enabled):")
    schedules_info = [
        ("daily_pipeline", "19:00 JST weekdays", "Daily data pipeline"),
        ("weekly_ml_training", "Sunday 02:00 JST", "Weekly ML training"),
        ("feature_store_sync", "20:00 JST weekdays", "Feature store sync"),
        ("weekly_cleanup", "Sunday 03:00 JST", "Data cleanup"),
    ]

    for name, time, desc in schedules_info:
        print(f"  - {name}: {time} - {desc}")

    print("\nOptional Schedules (Manually enabled):")
    optional_info = [
        ("intraday_quality", "Every 30min (9:00-15:00)", "Real-time quality checks"),
        ("morning_catchup", "06:00 JST weekdays", "Failed run retry"),
    ]

    for name, time, desc in optional_info:
        print(f"  - {name}: {time} - {desc}")

    print("\nTimezone: Asia/Tokyo (JST)")
    print("Market Hours: 09:00-15:00 JST")
    print("Data Available: After 18:00 JST")


if __name__ == "__main__":
    test_schedules()
