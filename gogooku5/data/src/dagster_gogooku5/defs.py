"""Dagster Definitions for gogooku5 dataset operations."""

from dagster import Definitions, multiprocess_executor

from .assets import build_dataset_chunks, merge_latest_dataset, validate_chunk_schemas
from .resources import dataset_builder_resource

defs = Definitions(
    assets=[build_dataset_chunks, validate_chunk_schemas, merge_latest_dataset],
    resources={"dataset_builder": dataset_builder_resource},
    executor=multiprocess_executor.configured({
        "max_concurrent": 3,  # 3チャンク同時ビルド（CPU 255コア、メモリ1.8TB空きを活用）
        "retries": {
            "enabled": {},  # 失敗時の自動リトライを有効化
        },
    }),
)
