"""Dagster resources for gogooku5 dataset builds."""

from pathlib import Path

from builder.config.settings import DatasetBuilderSettings
from builder.pipelines.dataset_builder import DatasetBuilder
from dagster import Bool, Field, InitResourceContext, resource


@resource(
    config_schema={
        "refresh_listed": Field(
            Bool,
            default_value=False,
            description="Refresh listed metadata once before the first chunk build.",
        ),
        "data_output_dir": Field(
            str,
            is_required=False,
            description="Override DatasetBuilder output directory.",
        ),
        "dataset_tag": Field(
            str,
            is_required=False,
            description="Override DatasetBuilder dataset tag.",
        ),
    }
)
def dataset_builder_resource(ctx: InitResourceContext) -> DatasetBuilder:
    """Instantiate DatasetBuilder with optional overrides supplied via Dagster config."""

    settings = DatasetBuilderSettings()
    cfg = ctx.resource_config or {}

    output_dir = cfg.get("data_output_dir")
    if output_dir:
        settings.data_output_dir = Path(output_dir)

    dataset_tag = cfg.get("dataset_tag")
    if dataset_tag:
        settings.dataset_tag = dataset_tag

    builder = DatasetBuilder(settings=settings)
    ctx.log.info(
        "DatasetBuilder initialized (output_dir=%s, tag=%s)",
        settings.data_output_dir,
        settings.dataset_tag,
    )
    builder._dagster_refresh_listed = bool(cfg.get("refresh_listed", False))  # type: ignore[attr-defined]
    return builder
