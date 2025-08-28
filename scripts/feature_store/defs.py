#!/usr/bin/env python3
"""
Feast Feature Store Definitions
特徴量ストアの定義（仕様書§5.2準拠）
"""

from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    ValueType,
    FeatureService,
    PushSource,
    RequestSource,
    FeatureStore,
)
from feast.types import Float32, Float64, Int32, String
import pandas as pd


# ========== Entities ==========
ticker_entity = Entity(
    name="ticker",
    description="Stock ticker code (4 digits)",
    join_keys=["ticker"],
    value_type=ValueType.STRING,
)

# ========== Data Sources ==========

# Offline source (Parquet on MinIO/S3)
price_parquet_source = FileSource(
    name="price_parquet",
    path="s3://gogooku/features/price_features.parquet",  # MinIO path
    timestamp_field="date",
    created_timestamp_column="created_at",
    description="Historical price features from Parquet files on MinIO",
)

# Alternative local source for development
price_local_source = FileSource(
    name="price_local",
    path="/home/ubuntu/gogooku2/apps/gogooku3/output/price_features.parquet",
    timestamp_field="date",
    created_timestamp_column="created_at",
    description="Local price features for development",
)

# Push source for real-time updates
price_push_source = PushSource(
    name="price_push",
    batch_source=price_parquet_source,
    description="Push source for real-time price updates",
)

# Request source for on-demand features
request_source = RequestSource(
    name="request_features",
    schema=[
        Field(name="current_price", dtype=Float32),
        Field(name="market_cap", dtype=Float64),
        Field(name="sector", dtype=String),
    ],
    description="On-demand features provided at request time",
)

# ========== Feature Views ==========

# Price features (daily)
price_features = FeatureView(
    name="price_features",
    entities=[ticker_entity],
    ttl=timedelta(days=1),  # Features expire after 1 day
    schema=[
        Field(name="open", dtype=Float32, description="Opening price"),
        Field(name="high", dtype=Float32, description="High price"),
        Field(name="low", dtype=Float32, description="Low price"),
        Field(name="close", dtype=Float32, description="Closing price"),
        Field(name="volume", dtype=Float64, description="Trading volume"),
        Field(name="returns_1d", dtype=Float32, description="1-day return"),
        Field(name="returns_5d", dtype=Float32, description="5-day return"),
        Field(name="returns_20d", dtype=Float32, description="20-day return"),
    ],
    source=price_parquet_source,
    online=True,  # Enable online serving
    tags={"team": "data", "priority": "high"},
    description="Daily price and basic return features",
)

# Technical indicators
technical_features = FeatureView(
    name="technical_features",
    entities=[ticker_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rsi_14", dtype=Float32, description="RSI (14 days)"),
        Field(name="ema_20", dtype=Float32, description="EMA (20 days)"),
        Field(name="ema_60", dtype=Float32, description="EMA (60 days)"),
        Field(name="bb_pct_b", dtype=Float32, description="Bollinger Band %B"),
        Field(name="macd_signal", dtype=Float32, description="MACD signal"),
        Field(name="vol_ratio_20", dtype=Float32, description="Volume ratio (20d)"),
    ],
    source=price_parquet_source,
    online=True,
    tags={"team": "data", "category": "technical"},
    description="Technical analysis indicators",
)

# Market relative features (with TOPIX)
market_features = FeatureView(
    name="market_features",
    entities=[ticker_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="alpha_1d", dtype=Float32, description="1-day alpha vs TOPIX"),
        Field(name="alpha_5d", dtype=Float32, description="5-day alpha vs TOPIX"),
        Field(
            name="relative_strength_1d",
            dtype=Float32,
            description="Relative strength vs market",
        ),
        Field(name="market_regime", dtype=Int32, description="Market regime indicator"),
    ],
    source=price_parquet_source,
    online=True,
    tags={"team": "data", "category": "market"},
    description="Market-relative performance features",
)

# Volatility features
volatility_features = FeatureView(
    name="volatility_features",
    entities=[ticker_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="volatility_20d", dtype=Float32, description="20-day volatility"),
        Field(
            name="volatility_ratio", dtype=Float32, description="Vol ratio (20d/60d)"
        ),
        Field(name="sharpe_20d", dtype=Float32, description="20-day Sharpe ratio"),
        Field(name="high_vol_flag", dtype=Int32, description="High volatility flag"),
    ],
    source=price_parquet_source,
    online=True,
    tags={"team": "data", "category": "risk"},
    description="Volatility and risk metrics",
)

# ========== Feature Services ==========

# Basic trading features
basic_trading_service = FeatureService(
    name="basic_trading",
    features=[
        price_features[["close", "volume", "returns_1d"]],
        technical_features[["rsi_14", "ema_20"]],
    ],
    description="Basic features for simple trading strategies",
    tags={"strategy": "basic"},
)

# Advanced ML features
ml_features_service = FeatureService(
    name="ml_features",
    features=[
        price_features,
        technical_features,
        market_features,
        volatility_features,
    ],
    description="Complete feature set for ML models",
    tags={"strategy": "ml", "version": "v1"},
)

# Real-time features (subset for low latency)
realtime_features_service = FeatureService(
    name="realtime_features",
    features=[
        price_features[["close", "returns_1d"]],
        technical_features[["rsi_14"]],
        market_features[["alpha_1d"]],
    ],
    description="Minimal feature set for real-time inference",
    tags={"strategy": "realtime", "sla": "10ms"},
)

# ========== Feature Store Configuration ==========


def get_feature_store_config(mode: str = "local") -> dict:
    """
    Get Feast configuration based on mode

    Args:
        mode: "local", "minio", or "prod"

    Returns:
        Feast configuration dictionary
    """
    if mode == "local":
        return {
            "project": "gogooku3",
            "provider": "local",
            "registry": "/home/ubuntu/gogooku2/apps/gogooku3/feature_store/registry.db",
            "online_store": {
                "type": "sqlite",
                "path": "/home/ubuntu/gogooku2/apps/gogooku3/feature_store/online.db",
            },
            "offline_store": {
                "type": "file",
            },
        }
    elif mode == "minio":
        return {
            "project": "gogooku3",
            "provider": "local",
            "registry": "s3://gogooku/feast/registry.db",
            "online_store": {
                "type": "redis",
                "connection_string": "redis://localhost:6379",
            },
            "offline_store": {
                "type": "file",  # Works with S3-compatible storage
            },
        }
    elif mode == "prod":
        return {
            "project": "gogooku3",
            "provider": "aws",  # or "gcp" depending on deployment
            "registry": "s3://gogooku-prod/feast/registry.db",
            "online_store": {
                "type": "redis",
                "connection_string": "redis://redis-cluster:6379",
                "key_ttl_seconds": 86400,  # 1 day
            },
            "offline_store": {
                "type": "bigquery",  # or "redshift", "snowflake"
                "project_id": "gogooku-prod",
                "dataset": "feast",
            },
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ========== Helper Functions ==========


def materialize_features(
    store: FeatureStore,
    start_date: str,
    end_date: str,
    feature_views: list = None,
):
    """
    Materialize features to online store

    Args:
        store: Feast FeatureStore instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        feature_views: List of feature view names (None for all)
    """
    from datetime import datetime

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if feature_views is None:
        feature_views = [
            "price_features",
            "technical_features",
            "market_features",
            "volatility_features",
        ]

    store.materialize(
        feature_views=feature_views,
        start_date=start,
        end_date=end,
    )


def get_training_data(
    store: FeatureStore,
    entity_df: pd.DataFrame,
    feature_service: str = "ml_features",
) -> pd.DataFrame:
    """
    Get historical features for training

    Args:
        store: Feast FeatureStore instance
        entity_df: DataFrame with entity values and timestamps
        feature_service: Name of feature service to use

    Returns:
        DataFrame with features
    """

    # Get feature service
    service = store.get_feature_service(feature_service)

    # Retrieve historical features
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=service,
    ).to_df()

    return training_df


def get_online_features(
    store: FeatureStore,
    tickers: list,
    feature_service: str = "realtime_features",
) -> dict:
    """
    Get online features for inference

    Args:
        store: Feast FeatureStore instance
        tickers: List of ticker codes
        feature_service: Name of feature service to use

    Returns:
        Dictionary with features
    """
    # Get feature service
    service = store.get_feature_service(feature_service)

    # Create entity dict
    entity_rows = [{"ticker": ticker} for ticker in tickers]

    # Get online features
    online_response = store.get_online_features(
        features=service,
        entity_rows=entity_rows,
    )

    return online_response.to_dict()


# ========== Test Function ==========


def test_feast_definitions():
    """Test Feast definitions"""
    print("Feast Feature Store Definitions")
    print("=" * 50)

    print("\nEntities:")
    print(f"  - {ticker_entity.name}: {ticker_entity.description}")

    print("\nFeature Views:")
    for fv in [
        price_features,
        technical_features,
        market_features,
        volatility_features,
    ]:
        print(f"  - {fv.name}: {len(fv.schema)} features")
        for field in fv.schema[:3]:  # Show first 3 fields
            print(f"      • {field.name} ({field.dtype})")

    print("\nFeature Services:")
    for fs in [basic_trading_service, ml_features_service, realtime_features_service]:
        print(f"  - {fs.name}: {fs.description}")

    print("\nConfiguration Modes:")
    for mode in ["local", "minio", "prod"]:
        config = get_feature_store_config(mode)
        print(
            f"  - {mode}: {config['provider']} provider, {config['online_store']['type']} online store"
        )


if __name__ == "__main__":
    test_feast_definitions()
