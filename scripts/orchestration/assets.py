#!/usr/bin/env python3
"""
Dagster Asset Definitions
仕様書§6準拠のアセットベースオーケストレーション
"""

from dagster import (
    asset,
    AssetIn,
    Output,
    DailyPartitionsDefinition,
    OpExecutionContext,
    MaterializeResult,
    MetadataValue,
    Config,
    AssetCheckSpec,
    AssetCheckResult,
    FreshnessPolicy,
    AutoMaterializePolicy,
)

# Note: dagster_polars and dagster_pandas not installed yet
# from dagster_polars import PolarsTypeHandler
# from dagster_pandas import PandasTypeHandler
import polars as pl
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from pipelines.run_pipeline import JQuantsPipeline
from core.ml_dataset_builder import MLDatasetBuilder
from quality.price_checks import PolarsValidator
from calendar.tse_calendar import TSECalendar
from corporate_actions.adjust import CorporateActionsAdjuster
from feature_store.defs import get_feature_store_config, materialize_features

# ========== Configurations ==========


class PipelineConfig(Config):
    """パイプライン設定"""

    from_date: str = "2024-01-01"
    to_date: str = "2024-12-31"
    output_dir: str = "/home/ubuntu/gogooku2/apps/gogooku3/output"
    use_sample: bool = False
    max_workers: int = 24


# ========== Partitions ==========

daily_partition = DailyPartitionsDefinition(
    start_date=datetime(2024, 1, 1),
    fmt="%Y-%m-%d",
    timezone="Asia/Tokyo",
)

# ========== Assets ==========


@asset(
    name="price_data",
    description="日次株価データ（JQuants API）",
    compute_kind="python",
    metadata={
        "source": "JQuants API",
        "frequency": "daily",
        "data_type": "OHLCV",
    },
    partitions_def=daily_partition,
    freshness_policy=FreshnessPolicy(
        maximum_lag_minutes=60 * 24,  # 1 day
        cron_schedule="0 18 * * *",  # 18:00 JST
    ),
    auto_materialize_policy=AutoMaterializePolicy.eager(),
    io_manager_key="polars_parquet_io_manager",
)
async def price_data_asset(
    context: OpExecutionContext, config: PipelineConfig
) -> Output[pl.DataFrame]:
    """
    株価データ取得アセット

    Args:
        context: Dagster実行コンテキスト
        config: パイプライン設定

    Returns:
        株価DataFrame
    """
    partition_date = context.partition_key
    context.log.info(f"Fetching price data for {partition_date}")

    # Initialize pipeline
    pipeline = JQuantsPipeline(use_sample=config.use_sample)

    # Fetch data for partition date
    from_date = partition_date
    to_date = partition_date

    # Fetch price data
    df = await pipeline.fetch_daily_quotes(from_date, to_date)

    # Log metadata
    context.add_output_metadata(
        {
            "num_records": len(df),
            "num_tickers": df["Code"].n_unique(),
            "date": partition_date,
            "columns": df.columns,
        }
    )

    return Output(
        value=df,
        metadata={
            "preview": MetadataValue.md(df.head(10).to_pandas().to_markdown()),
            "shape": f"{df.shape[0]} x {df.shape[1]}",
        },
    )


@asset(
    name="topix_data",
    description="TOPIX指数データ",
    compute_kind="python",
    metadata={
        "source": "JQuants API",
        "frequency": "daily",
    },
    partitions_def=daily_partition,
    freshness_policy=FreshnessPolicy(
        maximum_lag_minutes=60 * 24,
        cron_schedule="0 18 * * *",
    ),
    io_manager_key="polars_parquet_io_manager",
)
async def topix_data_asset(
    context: OpExecutionContext, config: PipelineConfig
) -> Output[pl.DataFrame]:
    """
    TOPIX指数データ取得アセット
    """
    partition_date = context.partition_key
    context.log.info(f"Fetching TOPIX data for {partition_date}")

    pipeline = JQuantsPipeline(use_sample=config.use_sample)

    # Fetch TOPIX data
    df = await pipeline.fetch_topix_data(partition_date, partition_date)

    context.add_output_metadata(
        {
            "num_records": len(df),
            "date": partition_date,
        }
    )

    return Output(value=df)


@asset(
    name="trades_spec",
    description="売買高・売買代金データ",
    compute_kind="python",
    metadata={
        "source": "JQuants API",
        "frequency": "daily",
    },
    partitions_def=daily_partition,
    io_manager_key="polars_parquet_io_manager",
)
async def trades_spec_asset(
    context: OpExecutionContext, config: PipelineConfig
) -> Output[pl.DataFrame]:
    """
    売買仕様データ取得アセット
    """
    partition_date = context.partition_key

    pipeline = JQuantsPipeline(use_sample=config.use_sample)

    # Fetch trades spec data
    df = await pipeline.fetch_trades_spec(partition_date, partition_date)

    return Output(value=df)


@asset(
    name="listed_info",
    description="上場銘柄一覧情報",
    compute_kind="python",
    metadata={
        "source": "JQuants API",
        "frequency": "monthly",
    },
    io_manager_key="polars_parquet_io_manager",
)
async def listed_info_asset(
    context: OpExecutionContext, config: PipelineConfig
) -> Output[pl.DataFrame]:
    """
    上場銘柄情報取得アセット
    """
    pipeline = JQuantsPipeline(use_sample=config.use_sample)

    # Fetch listed info (no date needed)
    df = await pipeline.fetch_listed_info()

    context.add_output_metadata(
        {
            "num_tickers": len(df),
            "markets": df["MarketCode"].unique().to_list(),
        }
    )

    return Output(value=df)


@asset(
    name="adjusted_price_data",
    description="コーポレートアクション調整済み価格データ",
    compute_kind="python",
    ins={
        "price_data": AssetIn("price_data"),
    },
    partitions_def=daily_partition,
    io_manager_key="polars_parquet_io_manager",
)
def adjusted_price_data_asset(
    context: OpExecutionContext,
    price_data: pl.DataFrame,
) -> Output[pl.DataFrame]:
    """
    コーポレートアクション調整済み価格データ
    """
    # Apply corporate actions adjustments
    adjuster = CorporateActionsAdjuster()
    df_adjusted = adjuster.adjust_for_actions(price_data)

    # Add adjustment flags
    df_with_flags = adjuster.add_adjustment_flags(df_adjusted)

    context.add_output_metadata(
        {
            "num_adjustments": (df_with_flags["corporate_action"] != "").sum(),
        }
    )

    return Output(value=df_with_flags)


@asset(
    name="ml_dataset",
    description="ML学習用データセット（62+ features）",
    compute_kind="python",
    metadata={
        "features": "62+ technical indicators",
        "framework": "Polars + pandas-ta",
    },
    ins={
        "adjusted_price_data": AssetIn("adjusted_price_data"),
        "topix_data": AssetIn("topix_data"),
    },
    partitions_def=daily_partition,
    io_manager_key="polars_parquet_io_manager",
)
def ml_dataset_asset(
    context: OpExecutionContext,
    adjusted_price_data: pl.DataFrame,
    topix_data: pl.DataFrame,
) -> Output[pl.DataFrame]:
    """
    ML用特徴量データセット生成アセット
    """
    # Initialize ML dataset builder
    builder = MLDatasetBuilder()

    # Build features
    df_features = builder.build_features(adjusted_price_data, topix_data)

    # Add metadata
    context.add_output_metadata(
        {
            "num_features": len(df_features.columns) - 3,  # Exclude Code, Date, target
            "num_samples": len(df_features),
            "feature_names": [
                c for c in df_features.columns if c not in ["Code", "Date", "target"]
            ],
        }
    )

    return Output(value=df_features)


@asset(
    name="quality_checked_dataset",
    description="品質チェック済みデータセット",
    compute_kind="python",
    ins={
        "ml_dataset": AssetIn("ml_dataset"),
    },
    partitions_def=daily_partition,
    io_manager_key="polars_parquet_io_manager",
    check_specs=[
        AssetCheckSpec(
            name="ohlc_consistency",
            asset="quality_checked_dataset",
            description="OHLC整合性チェック",
        ),
        AssetCheckSpec(
            name="null_values",
            asset="quality_checked_dataset",
            description="NULL値チェック",
        ),
        AssetCheckSpec(
            name="business_days",
            asset="quality_checked_dataset",
            description="営業日チェック",
        ),
    ],
)
def quality_check_asset(
    context: OpExecutionContext,
    ml_dataset: pl.DataFrame,
) -> Output[pl.DataFrame]:
    """
    データ品質チェックアセット
    """
    # Validate schema
    validator = PolarsValidator()
    validator.validate_schema(ml_dataset)

    # Check OHLC consistency
    ohlc_valid = validator.check_ohlc_consistency(ml_dataset)

    # Check null values
    null_counts = validator.check_null_values(ml_dataset)

    # Check business days
    calendar = TSECalendar()
    dates = ml_dataset["Date"].unique()
    invalid_dates = []
    for d in dates:
        if isinstance(d, datetime):
            d = d.date()
        if not calendar.is_business_day(d):
            invalid_dates.append(d)

    # Add check results
    context.add_output_metadata(
        {
            "ohlc_valid": ohlc_valid,
            "null_columns": len(null_counts),
            "invalid_dates": len(invalid_dates),
        }
    )

    # Filter out invalid data if needed
    if invalid_dates:
        context.log.warning(f"Removing {len(invalid_dates)} non-business days")
        valid_dates = [d for d in dates if d not in invalid_dates]
        ml_dataset = ml_dataset.filter(pl.col("Date").is_in(valid_dates))

    return Output(value=ml_dataset)


@asset(
    name="feature_store",
    description="Feast特徴量ストアへの登録",
    compute_kind="python",
    ins={
        "quality_checked_dataset": AssetIn("quality_checked_dataset"),
    },
    partitions_def=daily_partition,
)
def feature_store_asset(
    context: OpExecutionContext,
    quality_checked_dataset: pl.DataFrame,
) -> MaterializeResult:
    """
    Feast Feature Store登録アセット
    """
    from feast import FeatureStore

    # Convert to pandas for Feast
    df_pandas = quality_checked_dataset.to_pandas()

    # Initialize Feast
    config = get_feature_store_config(mode="local")

    # Save to parquet for Feast offline store
    partition_date = context.partition_key
    output_path = (
        Path(config["offline_store"]["path"]) / f"features_{partition_date}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_pandas.to_parquet(output_path)

    context.log.info(f"Saved features to {output_path}")

    # Materialize to online store (if configured)
    try:
        store = FeatureStore(config=config)
        materialize_features(
            store=store,
            start_date=partition_date,
            end_date=partition_date,
            feature_views=["price_features", "technical_features"],
        )
        context.log.info("Materialized features to online store")
    except Exception as e:
        context.log.warning(f"Could not materialize to online store: {e}")

    return MaterializeResult(
        metadata={
            "features_path": str(output_path),
            "num_features": len(df_pandas.columns),
            "num_rows": len(df_pandas),
        }
    )


# ========== Asset Checks ==========


def check_data_freshness(
    context: OpExecutionContext,
    quality_checked_dataset: pl.DataFrame,
) -> AssetCheckResult:
    """データが最新かチェック"""
    latest_date = quality_checked_dataset["Date"].max()
    today = datetime.now().date()

    # Allow 2 business days lag
    days_lag = (today - latest_date).days
    is_fresh = days_lag <= 2

    return AssetCheckResult(
        passed=is_fresh,
        metadata={
            "latest_date": str(latest_date),
            "days_lag": days_lag,
        },
    )


def check_feature_completeness(
    context: OpExecutionContext,
    ml_dataset: pl.DataFrame,
) -> AssetCheckResult:
    """必須特徴量が全て含まれているかチェック"""
    required_features = [
        "returns_1d",
        "returns_5d",
        "returns_20d",
        "rsi_14",
        "ema_20",
        "ema_60",
        "bb_pct_b",
        "macd_signal",
        "vol_ratio_20",
        "alpha_1d_topix",
    ]

    missing_features = []
    for feat in required_features:
        if feat not in ml_dataset.columns:
            missing_features.append(feat)

    return AssetCheckResult(
        passed=len(missing_features) == 0,
        metadata={
            "missing_features": missing_features,
            "total_features": len(ml_dataset.columns),
        },
    )


# ========== Asset Groups ==========

from dagster import AssetSelection  # noqa: E402

# Define asset groups
price_assets = AssetSelection.assets(
    "price_data", "topix_data", "trades_spec", "listed_info"
)
ml_assets = AssetSelection.assets(
    "adjusted_price_data", "ml_dataset", "quality_checked_dataset"
)
feature_assets = AssetSelection.assets("feature_store")

# All assets
all_assets = [
    price_data_asset,
    topix_data_asset,
    trades_spec_asset,
    listed_info_asset,
    adjusted_price_data_asset,
    ml_dataset_asset,
    quality_check_asset,
    feature_store_asset,
]


def test_assets():
    """アセット定義のテスト"""
    print("Dagster Assets Defined:")
    print("=" * 50)

    for asset_fn in all_assets:
        print(f"  - {asset_fn.name}: {asset_fn.description}")

    print("\nAsset Checks:")
    print("  - data_freshness")
    print("  - feature_completeness")

    print("\nAsset Groups:")
    print("  - price_assets: Raw price data from JQuants")
    print("  - ml_assets: ML feature engineering")
    print("  - feature_assets: Feature store management")


if __name__ == "__main__":
    test_assets()
