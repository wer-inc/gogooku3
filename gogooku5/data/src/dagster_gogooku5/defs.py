"""Dagster Definitions for gogooku5 dataset operations."""

from dagster import Definitions

from .assets import build_dataset_chunks, merge_latest_dataset
from .resources import dataset_builder_resource

defs = Definitions(
    assets=[build_dataset_chunks, merge_latest_dataset],
    resources={"dataset_builder": dataset_builder_resource},
)
