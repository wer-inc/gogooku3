#!/usr/bin/env python3
"""
Dagster Job Definitions
ジョブとスケジュールの定義
"""

from dagster import (
    job,
    define_asset_job,
    AssetSelection,
    RunConfig,
)

from .assets import (
    PipelineConfig,
)

# ========== Jobs ==========

# Daily data pipeline job
daily_pipeline_job = define_asset_job(
    name="daily_pipeline",
    description="日次データパイプライン（価格取得→特徴量生成→品質チェック→Feature Store）",
    selection=AssetSelection.all(),
    config={
        "ops": {
            "price_data_asset": {
                "config": {
                    "from_date": "2024-01-01",
                    "to_date": "2024-12-31",
                    "use_sample": False,
                    "max_workers": 24,
                }
            }
        }
    },
    tags={
        "team": "data",
        "priority": "high",
        "env": "prod",
    },
)

# ML training job (weekly)
ml_training_job = define_asset_job(
    name="ml_training",
    description="ML モデル学習ジョブ（週次実行）",
    selection=AssetSelection.assets(
        "quality_checked_dataset",
        "feature_store",
    ),
    tags={
        "team": "ml",
        "priority": "medium",
        "env": "prod",
    },
)

# Backfill job for historical data
backfill_job = define_asset_job(
    name="backfill_historical",
    description="過去データのバックフィルジョブ",
    selection=AssetSelection.all(),
    tags={
        "team": "data",
        "priority": "low",
        "env": "backfill",
    },
)

# Quality check only job
quality_check_job = define_asset_job(
    name="quality_check_only",
    description="品質チェックのみ実行",
    selection=AssetSelection.assets(
        "quality_checked_dataset",
    ).upstream(),  # Include upstream dependencies
    tags={
        "team": "data",
        "priority": "high",
        "env": "prod",
    },
)

# Feature store sync job
feature_sync_job = define_asset_job(
    name="feature_store_sync",
    description="Feature Store同期ジョブ",
    selection=AssetSelection.assets("feature_store").upstream(),
    tags={
        "team": "data",
        "priority": "medium",
        "env": "prod",
    },
)

# ========== Adhoc Jobs ==========


@job(
    name="cleanup_old_data",
    description="古いデータのクリーンアップ",
    tags={"team": "data", "type": "maintenance"},
)
def cleanup_job():
    """古いパーティションデータを削除"""
    # This would be implemented with ops
    pass


@job(
    name="validate_pipeline",
    description="パイプライン検証ジョブ",
    tags={"team": "data", "type": "validation"},
)
def validation_job():
    """パイプライン全体の検証"""
    # This would be implemented with ops
    pass


# ========== Job Configurations ==========


def get_daily_config(date: str) -> RunConfig:
    """日次ジョブの設定を生成"""
    return RunConfig(
        ops={
            "price_data_asset": PipelineConfig(
                from_date=date,
                to_date=date,
                output_dir="/home/ubuntu/gogooku2/apps/gogooku3/output",
                use_sample=False,
                max_workers=24,
            )
        }
    )


def get_backfill_config(start_date: str, end_date: str) -> RunConfig:
    """バックフィルジョブの設定を生成"""
    return RunConfig(
        ops={
            "price_data_asset": PipelineConfig(
                from_date=start_date,
                to_date=end_date,
                output_dir="/home/ubuntu/gogooku2/apps/gogooku3/output/backfill",
                use_sample=False,
                max_workers=48,  # More workers for backfill
            )
        }
    )


# ========== Test Function ==========


def test_jobs():
    """ジョブ定義のテスト"""
    print("Dagster Jobs Defined:")
    print("=" * 50)

    jobs = [
        ("daily_pipeline", "Daily data pipeline (price → features → quality → store)"),
        ("ml_training", "Weekly ML model training"),
        ("backfill_historical", "Historical data backfill"),
        ("quality_check_only", "Quality checks only"),
        ("feature_store_sync", "Feature store synchronization"),
        ("cleanup_old_data", "Old data cleanup"),
        ("validate_pipeline", "Pipeline validation"),
    ]

    for job_name, description in jobs:
        print(f"  - {job_name}: {description}")

    print("\nJob Configurations:")
    print("  - get_daily_config(): Daily job config")
    print("  - get_backfill_config(): Backfill job config")


if __name__ == "__main__":
    test_jobs()
