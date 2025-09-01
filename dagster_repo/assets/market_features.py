"""
市場特徴量アセット
TOPIX指数データから市場全体の特徴量を生成
"""

from dagster import asset, Output, AssetCheckResult, AssetCheckSpec
import polars as pl
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.gogooku3.contracts.schemas import DataSchemas


@asset(
    description="市場特徴量の生成（TOPIX基準）",
    compute_kind="polars",
    check_specs=[
        AssetCheckSpec(name="unique_date", description="日付の一意性検証"),
        AssetCheckSpec(name="bollinger_validity", description="ボリンジャーバンドの妥当性検証"),
        AssetCheckSpec(name="drawdown_range", description="ドローダウンの範囲検証"),
    ]
)
def market_features(context) -> Output[pl.DataFrame]:
    """
    市場特徴量の生成（26列）
    
    特徴量カテゴリ:
    - リターン: 1d, 5d, 10d, 20d
    - トレンド: EMA 5, 20, 60, 200
    - ボラティリティ: 20日ボラ, ATR, NATR
    - リスク: ドローダウン, ビッグムーブ
    - ボリンジャーバンド: 市場独自計算
    - Z-scores: 各指標の標準化
    - レジーム: ブル/ベア, トレンド, ボラティリティ
    """
    
    context.log.info("Loading market index data...")
    
    # Sample TOPIX data (replace with actual API call)
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2024, 12, 31), "1d")
    n = len(dates)
    
    # Generate realistic TOPIX data
    np.random.seed(42)
    returns = np.random.randn(n) * 0.01  # 1% daily volatility
    prices = 1000 * np.exp(np.cumsum(returns))
    
    df = pl.DataFrame({
        "Date": dates,
        "Close": prices,
        "High": prices * (1 + np.abs(np.random.randn(n) * 0.005)),
        "Low": prices * (1 - np.abs(np.random.randn(n) * 0.005))
    })
    
    # Calculate returns
    context.log.info("Calculating market returns...")
    for horizon in [1, 5, 10, 20]:
        df = df.with_columns(
            (pl.col("Close") / pl.col("Close").shift(horizon) - 1).alias(f"mkt_ret_{horizon}d")
        )
    
    # Calculate EMAs
    context.log.info("Calculating market EMAs...")
    for window in [5, 20, 60, 200]:
        df = df.with_columns(
            pl.col("Close").ewm_mean(span=window, adjust=False).alias(f"mkt_ema_{window}")
        )
    
    # Calculate deviations and gaps
    df = df.with_columns([
        ((pl.col("Close") - pl.col("mkt_ema_20")) / pl.col("mkt_ema_20")).alias("mkt_dev_20"),
        ((pl.col("mkt_ema_5") - pl.col("mkt_ema_20")) / pl.col("mkt_ema_20")).alias("mkt_gap_5_20"),
        # EMA slope (3-day change)
        ((pl.col("mkt_ema_20") - pl.col("mkt_ema_20").shift(3)) / pl.col("mkt_ema_20").shift(3)).alias("mkt_ema20_slope_3")
    ])
    
    # Calculate volatility
    context.log.info("Calculating market volatility...")
    df = df.with_columns(
        (pl.col("mkt_ret_1d").rolling_std(20) * np.sqrt(252)).alias("mkt_vol_20d")
    )
    
    # ATR calculation
    high_low = pl.col("High") - pl.col("Low")
    high_close = (pl.col("High") - pl.col("Close").shift(1)).abs()
    low_close = (pl.col("Low") - pl.col("Close").shift(1)).abs()
    
    true_range = pl.max_horizontal([high_low, high_close, low_close])
    df = df.with_columns([
        true_range.ewm_mean(span=14, adjust=False).alias("mkt_atr_14"),
        ((true_range.ewm_mean(span=14, adjust=False) / pl.col("Close")) * 100).alias("mkt_natr_14")
    ])
    
    # Market Bollinger Bands (独立計算)
    context.log.info("Calculating market Bollinger Bands...")
    bb_window = 20
    bb_std = 2.0
    
    bb_middle = pl.col("Close").rolling_mean(bb_window)
    bb_std_dev = pl.col("Close").rolling_std(bb_window)
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    bb_width = bb_upper - bb_lower
    bb_pct_b = (pl.col("Close") - bb_lower) / (bb_width + 1e-10)
    
    df = df.with_columns([
        bb_upper.alias("mkt_bb_upper"),
        bb_lower.alias("mkt_bb_lower"),
        bb_middle.alias("mkt_bb_middle"),
        bb_width.alias("mkt_bb_bw"),
        bb_pct_b.alias("mkt_bb_pct_b")
    ])
    
    # Calculate drawdown
    context.log.info("Calculating drawdown...")
    df = df.with_columns(
        pl.col("Close").cum_max().alias("_peak")
    )
    df = df.with_columns(
        ((pl.col("Close") - pl.col("_peak")) / pl.col("_peak")).alias("mkt_dd_from_peak")
    )
    df = df.drop("_peak")
    
    # Big move flag
    ret_std_60 = pl.col("mkt_ret_1d").rolling_std(60)
    df = df.with_columns(
        (pl.col("mkt_ret_1d").abs() >= (2 * ret_std_60)).cast(pl.Int8).alias("mkt_big_move_flag")
    )
    
    # Calculate Z-scores (rolling 60-day window)
    context.log.info("Calculating Z-scores...")
    z_window = 60
    for col, z_col in [
        ("mkt_ret_1d", "mkt_ret_1d_z"),
        ("mkt_vol_20d", "mkt_vol_20d_z"),
        ("mkt_bb_bw", "mkt_bb_bw_z"),
        ("mkt_dd_from_peak", "mkt_dd_from_peak_z")
    ]:
        mean = pl.col(col).rolling_mean(z_window)
        std = pl.col(col).rolling_std(z_window)
        df = df.with_columns(
            ((pl.col(col) - mean) / (std + 1e-10)).alias(z_col)
        )
    
    # Market regimes
    context.log.info("Calculating market regimes...")
    df = df.with_columns([
        # Bull market: price above 200 EMA
        (pl.col("Close") > pl.col("mkt_ema_200")).cast(pl.Int8).alias("mkt_bull_200"),
        
        # Trend up: 20 EMA > 60 EMA
        (pl.col("mkt_ema_20") > pl.col("mkt_ema_60")).cast(pl.Int8).alias("mkt_trend_up"),
        
        # High volatility: vol > 75th percentile
        (pl.col("mkt_vol_20d") > pl.col("mkt_vol_20d").quantile(0.75)).cast(pl.Int8).alias("mkt_high_vol"),
        
        # Squeeze: BB width < 25th percentile
        (pl.col("mkt_bb_bw") < pl.col("mkt_bb_bw").quantile(0.25)).cast(pl.Int8).alias("mkt_squeeze")
    ])
    
    # Clean up and select final columns
    schema = DataSchemas.get_market_features_schema()
    final_cols = list(schema.keys())
    
    # Ensure all columns exist
    for col in final_cols:
        if col not in df.columns:
            # Add missing columns with nulls
            df = df.with_columns(pl.lit(None).alias(col))
    
    df = df.select(final_cols)
    
    context.log.info(f"Generated {len(df.columns)} market features for {len(df)} days")
    
    # Run asset checks
    check_results = []
    
    # Check 1: Unique dates
    duplicate_dates = len(df) - len(df["Date"].unique())
    check_results.append(
        AssetCheckResult(
            check_name="unique_date",
            passed=duplicate_dates == 0,
            metadata={"duplicate_count": duplicate_dates}
        )
    )
    
    # Check 2: Bollinger Band validity
    bb_invalid = df.filter(
        (pl.col("mkt_bb_bw") <= 0) |
        (pl.col("mkt_bb_upper") <= pl.col("mkt_bb_lower"))
    )
    check_results.append(
        AssetCheckResult(
            check_name="bollinger_validity",
            passed=len(bb_invalid) == 0,
            metadata={"invalid_count": len(bb_invalid)}
        )
    )
    
    # Check 3: Drawdown range
    dd_invalid = df.filter(
        (pl.col("mkt_dd_from_peak") > 0) |
        (pl.col("mkt_dd_from_peak") < -1)
    )
    check_results.append(
        AssetCheckResult(
            check_name="drawdown_range",
            passed=len(dd_invalid) == 0,
            metadata={"invalid_count": len(dd_invalid)}
        )
    )
    
    return Output(df, metadata={
        "row_count": len(df),
        "column_count": len(df.columns),
        "date_range": f"{df['Date'].min()} to {df['Date'].max()}",
        "avg_volatility": float(df["mkt_vol_20d"].mean()),
        "max_drawdown": float(df["mkt_dd_from_peak"].min())
    })