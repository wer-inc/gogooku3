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
        "source_cache_mode": Field(
            str,
            is_required=False,
            description="Override source cache mode (off/read/read_write).",
        ),
        "source_cache_force_refresh": Field(
            Bool,
            is_required=False,
            description="Force refresh of cached API sources regardless of TTL.",
        ),
        "source_cache_asof": Field(
            str,
            is_required=False,
            description="Use a fixed ASOF date for source cache snapshots (YYYY-MM-DD).",
        ),
        "source_cache_tag": Field(
            str,
            is_required=False,
            description="Additional tag appended to source cache keys.",
        ),
        "source_cache_ttl_override_days": Field(
            int,
            is_required=False,
            description="Override TTL (days) for cached API sources.",
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

    cache_mode = cfg.get("source_cache_mode")
    if cache_mode:
        settings.source_cache_mode = cache_mode

    if "source_cache_force_refresh" in cfg:
        settings.source_cache_force_refresh = bool(cfg["source_cache_force_refresh"])

    cache_asof = cfg.get("source_cache_asof")
    if cache_asof:
        settings.source_cache_asof = cache_asof

    cache_tag = cfg.get("source_cache_tag")
    if cache_tag:
        settings.source_cache_tag = cache_tag

    ttl_override = cfg.get("source_cache_ttl_override_days")
    if ttl_override is not None:
        settings.source_cache_ttl_override_days = ttl_override

    builder = DatasetBuilder(settings=settings)
    ctx.log.info(
        "DatasetBuilder initialized (output_dir=%s, tag=%s)",
        settings.data_output_dir,
        settings.dataset_tag,
    )
    builder._dagster_refresh_listed = bool(cfg.get("refresh_listed", False))  # type: ignore[attr-defined]
    return builder
