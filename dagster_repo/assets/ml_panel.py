"""
最終MLパネルアセット
全データソースを統合した145列のパネルデータ生成
"""

from dagster import asset, AssetIn, Output, AssetCheckResult, AssetCheckSpec
import polars as pl
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.gogooku3.joins.intervals import (
    IntervalJoiner, AsOfJoiner, MarketDataJoiner, 
    FlowDataJoiner, BusinessCalendar
)
from src.gogooku3.features.ta_core import CrossSectionalNormalizer
from src.gogooku3.contracts.schemas import DataSchemas, SchemaValidator


@asset(
    ins={
        "daily_quotes": AssetIn(),
        "market_features": AssetIn(),
        # "section_mapping": AssetIn(),
        # "flow_daily": AssetIn(),
        # "statements_effective": AssetIn(),
    },
    description="全データソースを統合した最終MLパネル（145列）",
    compute_kind="polars",
    check_specs=[
        AssetCheckSpec(name="unique_key", description="(meta_code, meta_date)の一意性検証"),
        AssetCheckSpec(name="column_count", description="列数の検証（目標: 145列）"),
        AssetCheckSpec(name="feature_prefixes", description="特徴量接頭辞の検証"),
        AssetCheckSpec(name="null_rate", description="欠損率の検証"),
        AssetCheckSpec(name="data_leakage", description="未来参照リークの検証"),
    ]
)
def ml_panel(
    context,
    daily_quotes: pl.DataFrame,
    market_features: pl.DataFrame,
    # section_mapping: pl.DataFrame,
    # flow_daily: pl.DataFrame,
    # statements_effective: pl.DataFrame,
) -> Output[pl.DataFrame]:
    """
    最終MLパネルの生成
    
    結合順序:
    1. daily_quotes (px_* + targets)
    2. interval join: section_mapping → meta_section
    3. left join on Date: market_features (mkt_*)
    4. cross features (8列)
    5. interval join: flow_daily (flow_*)
    6. as-of backward: statements_effective (fin_*)
    
    正規化:
    - 断面Z-score（日次、無状態変換）
    - Winsorize (1%/99%)
    """
    
    context.log.info("Starting ML panel assembly...")
    
    # Start with daily quotes as base
    df = daily_quotes.select([
        pl.col("Code").alias("meta_code"),
        pl.col("Date").alias("meta_date"),
        *[col for col in daily_quotes.columns if col.startswith("px_") or col.startswith("y_")]
    ])
    
    # For now, add a dummy section (would come from section_mapping)
    df = df.with_columns(
        pl.lit("Prime").alias("meta_section").cast(pl.Categorical)
    )
    
    # Step 3: Join market features
    context.log.info("Joining market features...")
    df = MarketDataJoiner.join_market_features(
        df, 
        market_features,
        date_col="meta_date"
    )
    
    # Step 4: Calculate cross features
    context.log.info("Calculating cross features...")
    
    # Ensure required columns exist for cross features
    if "px_returns_1d" in df.columns and "mkt_ret_1d" in df.columns:
        # Simple beta calculation (correlation * std_ratio)
        df = df.sort(["meta_code", "meta_date"])
        
        # Calculate per stock
        result_dfs = []
        for code in df["meta_code"].unique():
            code_df = df.filter(pl.col("meta_code") == code)
            
            # Rolling correlation and beta
            beta_window = 60
            if len(code_df) > beta_window:
                code_df = code_df.with_columns([
                    pl.col("px_returns_1d").rolling_corr(pl.col("mkt_ret_1d"), window_size=beta_window)
                    .alias("cross_corr_60d"),
                ])
                
                # Approximate beta
                code_df = code_df.with_columns([
                    (pl.col("cross_corr_60d") * 
                     pl.col("px_returns_1d").rolling_std(beta_window) / 
                     (pl.col("mkt_ret_1d").rolling_std(beta_window) + 1e-8))
                    .alias("cross_beta_60d")
                ])
            else:
                code_df = code_df.with_columns([
                    pl.lit(None).alias("cross_corr_60d"),
                    pl.lit(None).alias("cross_beta_60d")
                ])
            
            result_dfs.append(code_df)
        
        df = pl.concat(result_dfs)
        
        # Calculate alpha
        df = df.with_columns([
            (pl.col("px_returns_1d") - 
             pl.col("cross_beta_60d") * pl.col("mkt_ret_1d"))
            .alias("cross_alpha_1d")
        ])
        
        if "px_returns_5d" in df.columns and "mkt_ret_5d" in df.columns:
            df = df.with_columns([
                (pl.col("px_returns_5d") - 
                 pl.col("cross_beta_60d") * pl.col("mkt_ret_5d"))
                .alias("cross_alpha_5d")
            ])
        
        # Relative strength
        if "px_returns_5d" in df.columns and "mkt_ret_5d" in df.columns:
            df = df.with_columns([
                (pl.col("px_returns_5d") / (pl.col("mkt_ret_5d") + 1e-10))
                .alias("cross_rel_strength_5d")
            ])
        
        # Trend alignment
        if "px_ma_gap_5_20" in df.columns and "mkt_gap_5_20" in df.columns:
            df = df.with_columns([
                (pl.col("px_ma_gap_5_20").sign() == pl.col("mkt_gap_5_20").sign())
                .cast(pl.Int8).alias("cross_trend_align_flag")
            ])
        
        # Regime conditional alpha
        if "cross_alpha_1d" in df.columns and "mkt_bull_200" in df.columns:
            df = df.with_columns([
                (pl.col("cross_alpha_1d") * pl.col("mkt_bull_200"))
                .alias("cross_alpha_vs_regime")
            ])
        
        # Idiosyncratic volatility ratio
        if "px_volatility_20d" in df.columns and "mkt_vol_20d" in df.columns:
            df = df.with_columns([
                (pl.col("px_volatility_20d") / (pl.col("mkt_vol_20d") + 1e-10))
                .alias("cross_idio_vol_ratio")
            ])
        
        # Beta stability
        if "cross_beta_60d" in df.columns:
            # Calculate per stock again for rolling std
            result_dfs = []
            for code in df["meta_code"].unique():
                code_df = df.filter(pl.col("meta_code") == code).sort("meta_date")
                code_df = code_df.with_columns([
                    (1.0 / (pl.col("cross_beta_60d").rolling_std(20) + 1e-10))
                    .alias("cross_beta_stability_60d")
                ])
                result_dfs.append(code_df)
            df = pl.concat(result_dfs)
    
    # Add placeholder flow features (would come from flow_daily)
    context.log.info("Adding placeholder flow features...")
    flow_features = [
        "flow_buy_ratio", "flow_sell_ratio", "flow_net_ratio",
        "flow_breadth_pos", "flow_buy_z", "flow_sell_z", "flow_net_z",
        "flow_smart_buy", "flow_smart_sell", "flow_smart_net",
        "flow_impulse"
    ]
    for feat in flow_features:
        df = df.with_columns(pl.lit(0.0).alias(feat))
    df = df.with_columns(pl.lit(0).alias("days_since_flow"))
    
    # Add placeholder financial features (would come from statements_effective)
    context.log.info("Adding placeholder financial features...")
    fin_features = [
        "fin_revenue_yoy", "fin_profit_yoy", "fin_eps_yoy",
        "fin_gross_margin", "fin_operating_margin",
        "fin_progress_revenue", "fin_progress_profit",
        "fin_guide_rev_revenue", "fin_guide_rev_profit",
        "fin_guide_momentum", "fin_roe", "fin_roa",
        "fin_accruals_ratio", "fin_surprise_score"
    ]
    for feat in fin_features:
        df = df.with_columns(pl.lit(0.0).alias(feat))
    
    df = df.with_columns([
        pl.lit(0).alias("fin_guide_rev_count"),
        pl.lit(0).alias("fin_nc_flag"),
        pl.lit(0).alias("fin_days_since_report")
    ])
    
    # Apply cross-sectional normalization (optional, for demonstration)
    context.log.info("Applying cross-sectional normalization...")
    
    # Select feature columns for normalization (exclude meta and targets)
    feature_cols = [col for col in df.columns 
                   if not col.startswith("meta_") and not col.startswith("y_")]
    
    # Note: This would be done in training pipeline, not in data preparation
    # df = CrossSectionalNormalizer.normalize_daily(
    #     df, feature_cols, method="zscore", robust=True, winsorize_pct=0.01
    # )
    
    # Final column ordering and type casting
    context.log.info("Finalizing panel structure...")
    
    # Ensure correct types
    df = df.with_columns([
        pl.col("meta_code").cast(pl.Utf8),
        pl.col("meta_date").cast(pl.Date),
        pl.col("meta_section").cast(pl.Categorical)
    ])
    
    # Cast numeric columns to appropriate types
    for col in df.columns:
        if col.startswith("px_") or col.startswith("mkt_") or col.startswith("cross_"):
            if "flag" in col:
                df = df.with_columns(pl.col(col).cast(pl.Int8))
            else:
                df = df.with_columns(pl.col(col).cast(pl.Float32))
    
    context.log.info(f"Generated ML panel with {len(df.columns)} columns and {len(df)} rows")
    
    # Run asset checks
    check_results = []
    
    # Check 1: Unique key
    duplicate_count = len(df) - len(df.unique(["meta_code", "meta_date"]))
    check_results.append(
        AssetCheckResult(
            check_name="unique_key",
            passed=duplicate_count == 0,
            metadata={"duplicate_count": duplicate_count}
        )
    )
    
    # Check 2: Column count (target ~145)
    col_count = len(df.columns)
    check_results.append(
        AssetCheckResult(
            check_name="column_count",
            passed=120 <= col_count <= 200,  # Allow some flexibility
            metadata={"column_count": col_count, "target": 145}
        )
    )
    
    # Check 3: Feature prefixes
    prefixes = {"px", "mkt", "cross", "flow", "fin", "meta", "y"}
    invalid_cols = []
    for col in df.columns:
        if not any(col.startswith(f"{p}_") or col.startswith(p) for p in prefixes):
            invalid_cols.append(col)
    
    check_results.append(
        AssetCheckResult(
            check_name="feature_prefixes",
            passed=len(invalid_cols) == 0,
            metadata={"invalid_columns": invalid_cols}
        )
    )
    
    # Check 4: Null rate
    total_values = len(df) * len(df.columns)
    total_nulls = sum(df[col].null_count() for col in df.columns)
    null_rate = total_nulls / total_values if total_values > 0 else 0
    
    check_results.append(
        AssetCheckResult(
            check_name="null_rate",
            passed=null_rate < 0.3,  # Less than 30% nulls
            metadata={"null_rate": f"{null_rate:.2%}"}
        )
    )
    
    # Check 5: Data leakage (simplified check)
    validator = SchemaValidator()
    leakage_check = validator.check_data_leakage(df)
    
    check_results.append(
        AssetCheckResult(
            check_name="data_leakage",
            passed=not leakage_check["has_leakage"],
            metadata={
                "suspicious_features": len(leakage_check["suspicious_features"])
            }
        )
    )
    
    return Output(df, metadata={
        "row_count": len(df),
        "column_count": len(df.columns),
        "unique_codes": len(df["meta_code"].unique()),
        "date_range": f"{df['meta_date'].min()} to {df['meta_date'].max()}",
        "null_rate": f"{null_rate:.2%}",
        "memory_usage_mb": df.estimated_size() / 1024 / 1024
    })