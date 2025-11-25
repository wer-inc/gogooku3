"""Dagster Definitions for gogooku5 dataset operations."""

from dagster import AssetSelection, Definitions, define_asset_job, multiprocess_executor

from .assets import build_dataset_chunks, merge_latest_dataset, validate_chunk_schemas
from .resources import dataset_builder_resource

# Asset job that materialises the full dataset pipeline (chunks → schema gate → merge)
g5_dataset_rebuild_job = define_asset_job(
    name="g5_dataset_rebuild_job",
    description="Build gogooku5 chunks, validate schema, and merge into the latest dataset.",
    selection=AssetSelection.keys("g5_dataset_chunks", "g5_schema_gate", "g5_dataset_full"),
)

defs = Definitions(
    assets=[build_dataset_chunks, validate_chunk_schemas, merge_latest_dataset],
    jobs=[g5_dataset_rebuild_job],
    resources={"dataset_builder": dataset_builder_resource},
    executor=multiprocess_executor.configured(
        {
            "max_concurrent": 3,  # 3チャンク同時ビルド（CPU 255コア、メモリ1.8TB空きを活用）
            "retries": {
                "enabled": {},  # 失敗時の自動リトライを有効化
            },
        }
    ),
)
