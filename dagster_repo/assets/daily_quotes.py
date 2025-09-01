"""
日次価格データ処理アセット
調整後OHLCV計算とテクニカル指標生成
"""

from dagster import asset, AssetIn, Output, AssetCheckResult, AssetCheckSpec
import polars as pl
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.gogooku3.features.ta_core import TechnicalIndicators
from src.gogooku3.contracts.schemas import DataSchemas, SchemaValidator


@asset(
    description="日次価格データの取得と調整後価格計算",
    compute_kind="polars",
    check_specs=[
        AssetCheckSpec(name="unique_key", description="(Code, Date)の一意性検証"),
        AssetCheckSpec(name="price_validity", description="価格データの妥当性検証"),
        AssetCheckSpec(name="adjustment_consistency", description="調整後価格の整合性検証"),
    ]
)
def daily_quotes(context) -> Output[pl.DataFrame]:
    """
    日次価格データ処理とテクニカル指標生成
    
    処理内容:
    1. 調整後価格計算（分割・配当調整）
    2. 80列のテクニカル指標生成
    3. 予測ターゲット生成
    """
    
    # Load raw data (placeholder - actual implementation would fetch from JQuants API)
    # For now, create sample data
    context.log.info("Loading daily quotes data...")
    
    # Sample data creation (replace with actual API call)
    df = pl.DataFrame({
        "Code": ["1301", "1301", "1301", "1301", "1301"] * 100,
        "Date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 5, 9), "1d").head(500),
        "Open": [100.0] * 500,
        "High": [105.0] * 500,
        "Low": [95.0] * 500,
        "Close": [102.0] * 500,
        "Volume": [1000000] * 500,
        "TurnoverValue": [102000000.0] * 500,
        "adj_factor": [1.0] * 500,
        "SharesOutstanding": [100000000] * 500,
    })
    
    # Add some variation to make it realistic
    import numpy as np
    np.random.seed(42)
    df = df.with_columns([
        (pl.col("Close") * (1 + np.random.randn(len(df)) * 0.02)).alias("Close"),
        (pl.col("Open") * (1 + np.random.randn(len(df)) * 0.02)).alias("Open"),
        (pl.col("High") * (1 + np.random.randn(len(df)) * 0.02 + 0.01)).alias("High"),
        (pl.col("Low") * (1 + np.random.randn(len(df)) * 0.02 - 0.01)).alias("Low"),
        (pl.col("Volume") * (1 + np.abs(np.random.randn(len(df))) * 0.5)).cast(pl.Int64).alias("Volume"),
    ])
    
    # Step 1: Apply adjustment factor
    context.log.info("Applying split/dividend adjustments...")
    for col in ["Open", "High", "Low", "Close"]:
        df = df.with_columns((pl.col(col) * pl.col("adj_factor")).alias(col))
    
    # Adjust volume for splits
    df = df.with_columns([
        (pl.col("Volume") / pl.col("adj_factor")).cast(pl.Int64).alias("Volume"),
        # Recalculate turnover for consistency
        (pl.col("Close") * pl.col("Volume")).alias("TurnoverValue")
    ])
    
    # Step 2: Add all technical indicators
    context.log.info("Generating technical indicators...")
    
    # Group by Code for proper rolling calculations
    df = df.sort(["Code", "Date"])
    
    # Apply indicators per stock
    result_dfs = []
    for code in df["Code"].unique():
        code_df = df.filter(pl.col("Code") == code)
        code_df = TechnicalIndicators.add_all_indicators(
            code_df,
            return_periods=[1, 5, 10, 20],
            ma_windows=[5, 20, 60, 200],
            vol_windows=[20, 60],
            realized_vol_method="parkinson",
            target_horizons=[1, 5, 10, 20]
        )
        result_dfs.append(code_df)
    
    df = pl.concat(result_dfs)
    
    # Add px_ prefix to feature columns
    feature_cols = [col for col in df.columns if col not in 
                   ["Code", "Date", "adj_factor", "SharesOutstanding"] and not col.startswith("y_")]
    
    rename_dict = {}
    for col in feature_cols:
        if not col.startswith("px_"):
            rename_dict[col] = f"px_{col.lower()}"
    
    if rename_dict:
        df = df.rename(rename_dict)
    
    context.log.info(f"Generated {len(df.columns)} columns for {len(df)} rows")
    
    # Run asset checks
    check_results = []
    
    # Check 1: Unique key
    duplicate_count = len(df) - len(df.unique(["Code", "Date"]))
    check_results.append(
        AssetCheckResult(
            check_name="unique_key",
            passed=duplicate_count == 0,
            metadata={"duplicate_count": duplicate_count}
        )
    )
    
    # Check 2: Price validity
    invalid_prices = df.filter(
        (pl.col("px_close") <= 0) |
        (pl.col("px_low") <= 0) |
        (pl.col("px_high") < pl.col("px_low"))
    )
    check_results.append(
        AssetCheckResult(
            check_name="price_validity",
            passed=len(invalid_prices) == 0,
            metadata={"invalid_count": len(invalid_prices)}
        )
    )
    
    # Check 3: Adjustment consistency
    turnover_diff = (df["px_turnovervalue"] - df["px_close"] * df["px_volume"]).abs()
    consistency_violations = df.filter(
        turnover_diff / (df["px_close"] * df["px_volume"] + 1e-10) > 0.001
    )
    check_results.append(
        AssetCheckResult(
            check_name="adjustment_consistency",
            passed=len(consistency_violations) == 0,
            metadata={"violation_count": len(consistency_violations)}
        )
    )
    
    return Output(df, metadata={
        "row_count": len(df),
        "column_count": len(df.columns),
        "unique_codes": len(df["Code"].unique()),
        "date_range": f"{df['Date'].min()} to {df['Date'].max()}"
    })