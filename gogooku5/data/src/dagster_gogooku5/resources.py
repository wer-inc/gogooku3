"""Dagster resources for gogooku5 dataset builds."""

from pathlib import Path

from builder.config.settings import DatasetBuilderSettings
from builder.pipelines.dataset_builder import DatasetBuilder
from builder.utils import ensure_env_loaded
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
        "index_option_parallel_fetch": Field(
            Bool,
            is_required=False,
            description="Enable async parallel fetching for /option/index_option.",
        ),
        "index_option_parallel_concurrency": Field(
            int,
            is_required=False,
            description="Max concurrent index option fetch tasks.",
        ),
        "index_option_cache_ttl_days": Field(
            int,
            is_required=False,
            description="TTL for index option raw cache entries.",
        ),
        "enable_mlflow_logging": Field(
            Bool,
            is_required=False,
            description="Enable MLflow logging for dataset builds.",
        ),
        "mlflow_experiment_name": Field(
            str,
            is_required=False,
            description="Override MLflow experiment name.",
        ),
        "mlflow_tracking_uri": Field(
            str,
            is_required=False,
            description="Override MLflow tracking URI.",
        ),
        "enable_dataset_quality_check": Field(
            Bool,
            is_required=False,
            description="Run dataset quality checker after chunk/full outputs.",
        ),
        "dataset_quality_targets": Field(
            str,
            is_required=False,
            description="Space/comma separated target columns.",
        ),
        "dataset_quality_asof_checks": Field(
            str,
            is_required=False,
            description="Space/comma separated `col<=reference` specs.",
        ),
        "dataset_quality_date_col": Field(
            str,
            is_required=False,
            description="Override date column name for quality checks.",
        ),
        "dataset_quality_code_col": Field(
            str,
            is_required=False,
            description="Override code column name for quality checks.",
        ),
        "dataset_quality_allow_future_days": Field(
            int,
            is_required=False,
            description="Days after today to tolerate in quality checks.",
        ),
        "dataset_quality_sample_rows": Field(
            int,
            is_required=False,
            description="Sample rows recorded in violation reports.",
        ),
        "dataset_quality_fail_on_warning": Field(
            Bool,
            is_required=False,
            description="Treat warnings as failures.",
        ),
    }
)
def dataset_builder_resource(ctx: InitResourceContext) -> DatasetBuilder:
    """Instantiate DatasetBuilder with optional overrides supplied via Dagster config."""

    # Load environment variables from .env file
    ensure_env_loaded()

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

    if "index_option_parallel_fetch" in cfg:
        settings.index_option_parallel_fetch = bool(cfg["index_option_parallel_fetch"])
    if "index_option_parallel_concurrency" in cfg:
        settings.index_option_parallel_concurrency = int(cfg["index_option_parallel_concurrency"])
    if "index_option_cache_ttl_days" in cfg:
        settings.index_option_cache_ttl_days = int(cfg["index_option_cache_ttl_days"])
    if "enable_mlflow_logging" in cfg:
        settings.enable_mlflow_logging = bool(cfg["enable_mlflow_logging"])
    experiment_override = cfg.get("mlflow_experiment_name")
    if experiment_override:
        settings.mlflow_experiment_name = experiment_override
    tracking_uri_override = cfg.get("mlflow_tracking_uri")
    if tracking_uri_override:
        settings.mlflow_tracking_uri = tracking_uri_override
    if "enable_dataset_quality_check" in cfg:
        settings.enable_dataset_quality_check = bool(cfg["enable_dataset_quality_check"])
    if cfg.get("dataset_quality_targets"):
        settings.dataset_quality_targets = cfg["dataset_quality_targets"]
    if cfg.get("dataset_quality_asof_checks"):
        settings.dataset_quality_asof_checks = cfg["dataset_quality_asof_checks"]
    if cfg.get("dataset_quality_date_col"):
        settings.dataset_quality_date_col = cfg["dataset_quality_date_col"]
    if cfg.get("dataset_quality_code_col"):
        settings.dataset_quality_code_col = cfg["dataset_quality_code_col"]
    if cfg.get("dataset_quality_allow_future_days") is not None:
        settings.dataset_quality_allow_future_days = int(cfg["dataset_quality_allow_future_days"])
    if cfg.get("dataset_quality_sample_rows") is not None:
        settings.dataset_quality_sample_rows = int(cfg["dataset_quality_sample_rows"])
    if "dataset_quality_fail_on_warning" in cfg:
        settings.dataset_quality_fail_on_warning = bool(cfg["dataset_quality_fail_on_warning"])

    builder = DatasetBuilder(settings=settings)
    ctx.log.info(
        "DatasetBuilder initialized (output_dir=%s, tag=%s)",
        settings.data_output_dir,
        settings.dataset_tag,
    )
    builder._dagster_refresh_listed = bool(cfg.get("refresh_listed", False))  # type: ignore[attr-defined]
    return builder
