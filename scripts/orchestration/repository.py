#!/usr/bin/env python3
"""
Dagster Repository Definition
リポジトリ定義（エントリーポイント）
"""

from dagster import (
    Definitions,
    FilesystemIOManager,
    ConfigurableIOManager,
    io_manager,
)
import polars as pl
from pathlib import Path

from . import assets, jobs, schedules, sensors

# ========== IO Managers ==========


class PolarsParquetIOManager(ConfigurableIOManager):
    """Polars DataFrame用のParquet IOマネージャー"""

    base_path: str = "/home/ubuntu/gogooku2/apps/gogooku3/output/dagster"

    def handle_output(self, context, obj: pl.DataFrame):
        """DataFrameをParquetファイルとして保存"""
        # Create path from asset key and partition
        asset_key = context.asset_key.to_python_identifier()

        if context.has_partition_key:
            partition = context.partition_key
            file_path = Path(self.base_path) / asset_key / f"{partition}.parquet"
        else:
            file_path = Path(self.base_path) / f"{asset_key}.parquet"

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write parquet file
        obj.write_parquet(file_path)
        context.log.info(f"Wrote {len(obj)} rows to {file_path}")

    def load_input(self, context) -> pl.DataFrame:
        """ParquetファイルからDataFrameを読み込み"""
        asset_key = context.asset_key.to_python_identifier()

        if context.has_partition_key:
            partition = context.partition_key
            file_path = Path(self.base_path) / asset_key / f"{partition}.parquet"
        else:
            file_path = Path(self.base_path) / f"{asset_key}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"Asset file not found: {file_path}")

        # Read parquet file
        df = pl.read_parquet(file_path)
        context.log.info(f"Loaded {len(df)} rows from {file_path}")

        return df


@io_manager
def polars_parquet_io_manager():
    """Polars Parquet IOマネージャーのファクトリ"""
    return PolarsParquetIOManager()


# ========== Resource Definitions ==========

resources = {
    "io_manager": FilesystemIOManager(
        base_dir="/home/ubuntu/gogooku2/apps/gogooku3/output/dagster",
    ),
    "polars_parquet_io_manager": PolarsParquetIOManager(
        base_path="/home/ubuntu/gogooku2/apps/gogooku3/output/dagster",
    ),
}


# ========== Repository Definition ==========

defs = Definitions(
    assets=assets.all_assets,
    jobs=[
        jobs.daily_pipeline_job,
        jobs.ml_training_job,
        jobs.backfill_job,
        jobs.quality_check_job,
        jobs.feature_sync_job,
        jobs.cleanup_job,
        jobs.validation_job,
    ],
    schedules=schedules.all_schedules,
    sensors=sensors.all_sensors,
    resources=resources,
)


# ========== Configuration ==========


def get_dagster_config() -> dict:
    """Dagster設定を取得"""
    return {
        "telemetry": {"enabled": False},
        "run_coordinator": {
            "module": "dagster.core.run_coordinator",
            "class": "QueuedRunCoordinator",
            "config": {
                "max_concurrent_runs": 10,
                "tag_concurrency_limits": [
                    {"key": "team", "value": "data", "limit": 5},
                    {"key": "priority", "value": "high", "limit": 3},
                ],
            },
        },
        "run_launcher": {
            "module": "dagster.core.launcher",
            "class": "DefaultRunLauncher",
        },
        "scheduler": {
            "module": "dagster.core.scheduler",
            "class": "DagsterDaemonScheduler",
        },
        "run_storage": {
            "module": "dagster.core.storage.runs",
            "class": "SqliteRunStorage",
            "config": {
                "base_dir": "/home/ubuntu/gogooku2/apps/gogooku3/dagster_home",
            },
        },
        "event_log_storage": {
            "module": "dagster.core.storage.event_log",
            "class": "SqliteEventLogStorage",
            "config": {
                "base_dir": "/home/ubuntu/gogooku2/apps/gogooku3/dagster_home",
            },
        },
        "compute_logs": {
            "module": "dagster.core.storage.local_compute_log_manager",
            "class": "LocalComputeLogManager",
            "config": {
                "base_dir": "/home/ubuntu/gogooku2/apps/gogooku3/dagster_home/compute_logs",
            },
        },
    }


# ========== CLI Commands ==========


def create_dagster_yaml():
    """dagster.yamlファイルを生成"""
    yaml_content = """
# Dagster instance configuration
telemetry:
  enabled: false

run_coordinator:
  module: dagster.core.run_coordinator
  class: QueuedRunCoordinator
  config:
    max_concurrent_runs: 10
    tag_concurrency_limits:
      - key: team
        value: data
        limit: 5
      - key: priority
        value: high
        limit: 3

scheduler:
  module: dagster.core.scheduler
  class: DagsterDaemonScheduler

run_launcher:
  module: dagster.core.launcher
  class: DefaultRunLauncher

run_storage:
  module: dagster.core.storage.runs
  class: SqliteRunStorage
  config:
    base_dir: /home/ubuntu/gogooku2/apps/gogooku3/dagster_home

event_log_storage:
  module: dagster.core.storage.event_log
  class: SqliteEventLogStorage
  config:
    base_dir: /home/ubuntu/gogooku2/apps/gogooku3/dagster_home

compute_logs:
  module: dagster.core.storage.local_compute_log_manager
  class: LocalComputeLogManager
  config:
    base_dir: /home/ubuntu/gogooku2/apps/gogooku3/dagster_home/compute_logs
"""

    dagster_home = Path("/home/ubuntu/gogooku2/apps/gogooku3/dagster_home")
    dagster_home.mkdir(parents=True, exist_ok=True)

    yaml_file = dagster_home / "dagster.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)

    print(f"Created dagster.yaml at {yaml_file}")


def create_workspace_yaml():
    """workspace.yamlファイルを生成"""
    yaml_content = """
# Dagster workspace configuration
load_from:
  - python_file:
      relative_path: scripts/orchestration/repository.py
      attribute: defs
"""

    workspace_file = Path("/home/ubuntu/gogooku2/apps/gogooku3/workspace.yaml")
    with open(workspace_file, "w") as f:
        f.write(yaml_content)

    print(f"Created workspace.yaml at {workspace_file}")


# ========== Test Function ==========


def test_repository():
    """リポジトリ定義のテスト"""
    print("Dagster Repository Configuration:")
    print("=" * 50)

    print("\nAssets:")
    for asset in assets.all_assets:
        print(f"  - {asset.name}")

    print("\nJobs:")
    job_list = [
        "daily_pipeline",
        "ml_training",
        "backfill_historical",
        "quality_check_only",
        "feature_store_sync",
        "cleanup_old_data",
        "validate_pipeline",
    ]
    for job_name in job_list:
        print(f"  - {job_name}")

    print("\nSchedules:")
    for schedule in schedules.all_schedules:
        print(f"  - {schedule.name}")

    print("\nSensors:")
    for sensor in sensors.all_sensors:
        print(f"  - {sensor.name}")

    print("\nResources:")
    for resource_name in resources.keys():
        print(f"  - {resource_name}")

    print("\n" + "=" * 50)
    print("To start Dagster:")
    print("  1. Create config files:")
    print(
        "     python -c 'from scripts.orchestration.repository import create_dagster_yaml, create_workspace_yaml; create_dagster_yaml(); create_workspace_yaml()'"
    )
    print("  2. Start Dagster daemon:")
    print(
        "     DAGSTER_HOME=/home/ubuntu/gogooku2/apps/gogooku3/dagster_home dagster-daemon run"
    )
    print("  3. Start Dagster UI:")
    print(
        "     DAGSTER_HOME=/home/ubuntu/gogooku2/apps/gogooku3/dagster_home dagster-webserver -w /home/ubuntu/gogooku2/apps/gogooku3/workspace.yaml"
    )


if __name__ == "__main__":
    test_repository()

    # Optionally create config files
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        create_dagster_yaml()
        create_workspace_yaml()
        print("\nSetup complete! You can now start Dagster.")
