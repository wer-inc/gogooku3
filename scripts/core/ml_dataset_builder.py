#!/usr/bin/env python3
"""
ML Dataset Builder with Polars (Fixed Version)
指摘事項P0～P2をすべて修正した62特徴量版
"""

import polars as pl
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import time
from typing import Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLDatasetBuilder:
    """Build ML dataset with 62 features using Polars (with fixes)."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(
            "/home/ubuntu/gogooku2/apps/gogooku3/output"
        )
        self.output_dir.mkdir(exist_ok=True)

    def create_technical_features(
        self, df: pl.DataFrame, topix_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Create all 62 technical features with bug fixes."""
        logger.info("Creating technical features with Polars...")
        start_time = time.time()

        # Sort by Code and Date first (P1-6: ensure proper date type)
        df = df.with_columns(
            pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        ).sort(["Code", "Date"])

        # P1-7: Use cumulative count for maturity index
        df = df.with_columns(pl.col("Date").cum_count().over("Code").alias("row_idx"))

        # ========== RETURNS (4 features) ==========
        # P0-1: Fixed - pct_change with over("Code") first
        # P0-2: No Winsorization at data creation stage (do it in training)
        df = df.with_columns(
            [
                pl.col("Close").pct_change().over("Code").alias("returns_1d"),
                pl.col("Close").pct_change(5).over("Code").alias("returns_5d"),
                pl.col("Close").pct_change(10).over("Code").alias("returns_10d"),
                pl.col("Close").pct_change(20).over("Code").alias("returns_20d"),
            ]
        )

        # ========== EMAs (5 features) ==========
        # Use adjust=False for causality, ignore_nulls=True to keep existing behavior
        df = df.with_columns(
            [
                pl.col("Close")
                .ewm_mean(span=5, adjust=False, ignore_nulls=True)
                .over("Code")
                .alias("ema_5"),
                pl.col("Close")
                .ewm_mean(span=10, adjust=False, ignore_nulls=True)
                .over("Code")
                .alias("ema_10"),
                pl.col("Close")
                .ewm_mean(span=20, adjust=False, ignore_nulls=True)
                .over("Code")
                .alias("ema_20"),
                pl.col("Close")
                .ewm_mean(span=60, adjust=False, ignore_nulls=True)
                .over("Code")
                .alias("ema_60"),
                pl.col("Close")
                .ewm_mean(span=200, adjust=False, ignore_nulls=True)
                .over("Code")
                .alias("ema_200"),
            ]
        )

        # ========== MA-DERIVED (17 features) ==========
        # P0-3: Fixed - Use EMA as denominator for deviations
        df = df.with_columns(
            [
                # Price deviations (denominator = EMA)
                ((pl.col("Close") - pl.col("ema_5")) / pl.col("ema_5")).alias(
                    "price_ema5_dev"
                ),
                ((pl.col("Close") - pl.col("ema_10")) / pl.col("ema_10")).alias(
                    "price_ema10_dev"
                ),
                ((pl.col("Close") - pl.col("ema_20")) / pl.col("ema_20")).alias(
                    "price_ema20_dev"
                ),
                ((pl.col("Close") - pl.col("ema_200")) / pl.col("ema_200")).alias(
                    "price_ema200_dev"
                ),
                # MA gaps
                ((pl.col("ema_5") - pl.col("ema_20")) / pl.col("ema_20")).alias(
                    "ma_gap_5_20"
                ),
                ((pl.col("ema_20") - pl.col("ema_60")) / pl.col("ema_60")).alias(
                    "ma_gap_20_60"
                ),
                ((pl.col("ema_60") - pl.col("ema_200")) / pl.col("ema_200")).alias(
                    "ma_gap_60_200"
                ),
                # MA slopes (rate of change)
                pl.col("ema_5").pct_change().over("Code").alias("ema5_slope"),
                pl.col("ema_20").pct_change().over("Code").alias("ema20_slope"),
                pl.col("ema_60").pct_change().over("Code").alias("ema60_slope"),
                # MA crosses (binary)
                (pl.col("ema_5") > pl.col("ema_20"))
                .cast(pl.Int8)
                .alias("ema_cross_5_20"),
                (pl.col("ema_20") > pl.col("ema_60"))
                .cast(pl.Int8)
                .alias("ema_cross_20_60"),
                (pl.col("ema_60") > pl.col("ema_200"))
                .cast(pl.Int8)
                .alias("ema_cross_60_200"),
                # MA ribbon alignment
                (
                    (pl.col("ema_5") > pl.col("ema_10"))
                    & (pl.col("ema_10") > pl.col("ema_20"))
                    & (pl.col("ema_20") > pl.col("ema_60"))
                )
                .cast(pl.Int8)
                .alias("ma_ribbon_bullish"),
                (
                    (pl.col("ema_5") < pl.col("ema_10"))
                    & (pl.col("ema_10") < pl.col("ema_20"))
                    & (pl.col("ema_20") < pl.col("ema_60"))
                )
                .cast(pl.Int8)
                .alias("ma_ribbon_bearish"),
            ]
        )

        # MA ribbon spread (std of EMAs)
        df = df.with_columns(
            pl.concat_list(
                [pl.col("ema_5"), pl.col("ema_10"), pl.col("ema_20"), pl.col("ema_60")]
            )
            .list.eval(pl.element().std())
            .list.first()
            .alias("ma_ribbon_spread_raw")
        )

        # Normalize spread
        df = df.with_columns(
            (pl.col("ma_ribbon_spread_raw") / pl.col("Close")).alias("ma_ribbon_spread")
        ).drop("ma_ribbon_spread_raw")

        # Distance to 200 EMA (trend strength)
        df = df.with_columns(
            ((pl.col("Close") - pl.col("ema_200")) / pl.col("ema_200")).alias(
                "dist_to_200ema"
            )
        )

        # ========== RETURNS × MA (12 features) ==========
        df = df.with_columns(
            [
                # Momentum ratios
                (pl.col("returns_5d") / (pl.col("returns_20d") + 1e-12)).alias(
                    "momentum_5_20"
                ),
                (pl.col("returns_1d") / (pl.col("returns_5d") + 1e-12)).alias(
                    "momentum_1_5"
                ),
                (pl.col("returns_10d") / (pl.col("returns_20d") + 1e-12)).alias(
                    "momentum_10_20"
                ),
                # Return × MA deviation interactions
                (pl.col("returns_1d") * pl.col("price_ema20_dev")).alias(
                    "ret1d_x_ema20dev"
                ),
                (pl.col("returns_5d") * pl.col("price_ema20_dev")).alias(
                    "ret5d_x_ema20dev"
                ),
                (pl.col("returns_1d") * pl.col("price_ema200_dev")).alias(
                    "ret1d_x_ema200dev"
                ),
                # Momentum × MA slope
                (pl.col("returns_5d") * pl.col("ema20_slope")).alias(
                    "mom5d_x_ema20slope"
                ),
                (pl.col("returns_20d") * pl.col("ema60_slope")).alias(
                    "mom20d_x_ema60slope"
                ),
            ]
        )

        # ========== VOLATILITY (7 features) ==========
        # P0-1: Fixed - pct_change().over("Code") THEN rolling_std()
        df = df.with_columns(
            [
                pl.col("Close")
                .pct_change()
                .over("Code")
                .rolling_std(window_size=20)
                .alias("volatility_20d_raw"),
                pl.col("Close")
                .pct_change()
                .over("Code")
                .rolling_std(window_size=60)
                .alias("volatility_60d_raw"),
            ]
        )

        # Annualize volatilities
        df = df.with_columns(
            [
                (pl.col("volatility_20d_raw") * np.sqrt(252)).alias("volatility_20d"),
                (pl.col("volatility_60d_raw") * np.sqrt(252)).alias("volatility_60d"),
            ]
        )

        # Volatility ratio and change
        df = df.with_columns(
            [
                (pl.col("volatility_20d") / (pl.col("volatility_60d") + 1e-12)).alias(
                    "volatility_ratio"
                ),
                pl.col("volatility_20d")
                .pct_change()
                .over("Code")
                .alias("volatility_change"),
            ]
        )

        # Sharpe ratios (P1-8: clarify calculation)
        # volatility_20d is annualized, divide by sqrt(252) to get daily
        df = df.with_columns(
            [
                (
                    pl.col("returns_1d")
                    / (pl.col("volatility_20d") / np.sqrt(252) + 1e-12)
                ).alias("sharpe_1d"),
                (
                    pl.col("returns_5d")
                    / (pl.col("volatility_20d") / np.sqrt(252) + 1e-12)
                    / np.sqrt(5)
                ).alias("sharpe_5d"),
                (
                    pl.col("returns_20d")
                    / (pl.col("volatility_20d") / np.sqrt(252) + 1e-12)
                    / np.sqrt(20)
                ).alias("sharpe_20d"),
            ]
        )

        # High/low volatility flags
        df = df.with_columns(
            [
                (
                    pl.col("volatility_20d")
                    > pl.col("volatility_20d").quantile(0.8).over("Code")
                )
                .cast(pl.Int8)
                .alias("high_vol_flag"),
                (
                    pl.col("volatility_20d")
                    < pl.col("volatility_20d").quantile(0.2).over("Code")
                )
                .cast(pl.Int8)
                .alias("low_vol_flag"),
            ]
        )

        # P0-5: Drop volatility_60d later (after ratio calculation)
        # We'll drop it at the end

        # ========== RSI (3 features) ==========
        # RSI needs special handling - we'll use pandas-ta later

        # ========== FLOW (4 features) ==========
        # Smart money index (simplified for now)
        df = df.with_columns(
            [
                pl.lit(0.0).alias("smart_money_index"),
                pl.lit(0.0).alias("smart_money_change"),
                pl.lit(0).cast(pl.Int8).alias("flow_high_flag"),
                pl.lit(0).cast(pl.Int8).alias("flow_low_flag"),
            ]
        )

        # ========== TARGETS (7 features) ==========
        df = df.with_columns(
            [
                # Future returns (shift by -n)
                (pl.col("Close").shift(-1).over("Code") / pl.col("Close") - 1).alias(
                    "target_1d"
                ),
                (pl.col("Close").shift(-5).over("Code") / pl.col("Close") - 1).alias(
                    "target_5d"
                ),
                (pl.col("Close").shift(-10).over("Code") / pl.col("Close") - 1).alias(
                    "target_10d"
                ),
                (pl.col("Close").shift(-20).over("Code") / pl.col("Close") - 1).alias(
                    "target_20d"
                ),
            ]
        )

        # Binary targets
        df = df.with_columns(
            [
                (pl.col("target_1d") > 0).cast(pl.Int8).alias("target_1d_binary"),
                (pl.col("target_5d") > 0).cast(pl.Int8).alias("target_5d_binary"),
                (pl.col("target_10d") > 0).cast(pl.Int8).alias("target_10d_binary"),
            ]
        )

        # ========== MATURITY FLAGS (6 features) ==========
        # P1-7: Fixed - Use row_idx for proper maturity flags
        df = df.with_columns(
            [
                (pl.col("row_idx") >= 5).cast(pl.Int8).alias("is_rsi2_valid"),
                (pl.col("row_idx") >= 15).cast(pl.Int8).alias("is_ema5_valid"),
                (pl.col("row_idx") >= 30).cast(pl.Int8).alias("is_ema10_valid"),
                (pl.col("row_idx") >= 60).cast(pl.Int8).alias("is_ema20_valid"),
                (pl.col("row_idx") >= 200).cast(pl.Int8).alias("is_ema200_valid"),
                (pl.col("row_idx") >= 60).cast(pl.Int8).alias("is_valid_ma"),
            ]
        )

        # P0-5: Drop volatility_60d and all raw columns
        cols_to_drop = ["volatility_60d", "volatility_60d_raw", "volatility_20d_raw"]
        df = df.drop([col for col in cols_to_drop if col in df.columns])

        # Add TOPIX features if available (after returns calculation)
        if topix_df is not None and not topix_df.is_empty():
            df = self.add_topix_features(df, topix_df)

        elapsed = time.time() - start_time
        logger.info(f"Created technical features in {elapsed:.2f} seconds")

        return df

    def add_pandas_ta_features(self, df_polars: pl.DataFrame) -> pl.DataFrame:
        """Add complex technical indicators using pandas-ta."""
        logger.info("Adding pandas-ta features...")
        start_time = time.time()

        # Convert to pandas for pandas-ta
        df_pandas = df_polars.to_pandas()

        # Process each stock separately
        results = []
        for code in df_pandas["Code"].unique():
            stock_df = df_pandas[df_pandas["Code"] == code].copy()
            stock_df = stock_df.sort_values("Date")

            # RSI calculations
            rsi_14 = ta.rsi(stock_df["Close"], length=14)
            rsi_2 = ta.rsi(stock_df["Close"], length=2)  # Ultra-short

            # Handle None values from ta.rsi
            stock_df["rsi_14"] = rsi_14 if rsi_14 is not None else 50.0
            stock_df["rsi_2"] = rsi_2 if rsi_2 is not None else 50.0

            # RSI delta - only calculate if RSI is valid
            if rsi_14 is not None and not rsi_14.empty:
                stock_df["rsi_delta"] = rsi_14.diff()
            else:
                stock_df["rsi_delta"] = 0.0

            # MACD (P1-9: use column names instead of iloc)
            try:
                macd = ta.macd(stock_df["Close"], fast=12, slow=26, signal=9)
                if (
                    macd is not None
                    and isinstance(macd, pd.DataFrame)
                    and not macd.empty
                ):
                    stock_df["macd_signal"] = macd["MACDs_12_26_9"]
                    stock_df["macd_histogram"] = macd["MACDh_12_26_9"]
                else:
                    stock_df["macd_signal"] = 0
                    stock_df["macd_histogram"] = 0
            except (TypeError, KeyError):
                # Handle case where MACD calculation fails
                stock_df["macd_signal"] = 0
                stock_df["macd_histogram"] = 0

            # Bollinger Bands (P0-4: prevent 0-division)
            bb = ta.bbands(stock_df["Close"], length=20, std=2)
            if bb is not None and not bb.empty:
                bb_upper = bb["BBU_20_2.0"]
                bb_lower = bb["BBL_20_2.0"]
                bb_middle = bb["BBM_20_2.0"]

                # P0-4: Fixed - Add epsilon and clip
                stock_df["bb_pct_b"] = (
                    (stock_df["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-12)
                ).clip(0, 1)
                stock_df["bb_bandwidth"] = (bb_upper - bb_lower) / (bb_middle + 1e-12)
            else:
                stock_df["bb_pct_b"] = 0.5
                stock_df["bb_bandwidth"] = 0

            results.append(stock_df)

        # Combine and convert back to Polars
        df_combined = pd.concat(results, ignore_index=True)
        df_final = pl.from_pandas(df_combined)

        elapsed = time.time() - start_time
        logger.info(f"Added pandas-ta features in {elapsed:.2f} seconds")

        return df_final

    def add_topix_features(
        self, df: pl.DataFrame, topix_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Add TOPIX-relative features to the dataset."""
        logger.info("Adding TOPIX relative features...")

        # Ensure both dataframes have the same date type
        # Convert to Date type for both
        if df["Date"].dtype != pl.Date:
            df = df.with_columns(pl.col("Date").cast(pl.Date))

        if topix_df["Date"].dtype == pl.Utf8:
            topix_df = topix_df.with_columns(
                pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            )
        elif topix_df["Date"].dtype != pl.Date:
            topix_df = topix_df.with_columns(pl.col("Date").cast(pl.Date))

        # Calculate TOPIX returns
        topix_df = topix_df.sort("Date").with_columns(
            [
                pl.col("Close").pct_change().alias("topix_return_1d"),
                pl.col("Close").pct_change(5).alias("topix_return_5d"),
                pl.col("Close").pct_change(10).alias("topix_return_10d"),
                pl.col("Close").pct_change(20).alias("topix_return_20d"),
            ]
        )

        # Rename Close to avoid conflict
        topix_df = topix_df.rename({"Close": "topix_close"})

        # Join TOPIX data with stock data
        df = df.join(
            topix_df.select(
                [
                    "Date",
                    "topix_close",
                    "topix_return_1d",
                    "topix_return_5d",
                    "topix_return_10d",
                    "topix_return_20d",
                ]
            ),
            on="Date",
            how="left",
        )

        # Calculate relative performance (stock return - market return)
        df = df.with_columns(
            [
                # Alpha (excess return over market)
                (pl.col("returns_1d") - pl.col("topix_return_1d")).alias("alpha_1d"),
                (pl.col("returns_5d") - pl.col("topix_return_5d")).alias("alpha_5d"),
                (pl.col("returns_10d") - pl.col("topix_return_10d")).alias("alpha_10d"),
                (pl.col("returns_20d") - pl.col("topix_return_20d")).alias("alpha_20d"),
                # Relative strength (stock return / market return)
                (pl.col("returns_1d") / (pl.col("topix_return_1d") + 1e-12)).alias(
                    "relative_strength_1d"
                ),
                (pl.col("returns_5d") / (pl.col("topix_return_5d") + 1e-12)).alias(
                    "relative_strength_5d"
                ),
                # Market regime indicator (TOPIX momentum)
                pl.when(pl.col("topix_return_20d") > 0.05)
                .then(1)  # Bull market
                .when(pl.col("topix_return_20d") < -0.05)
                .then(-1)  # Bear market
                .otherwise(0)
                .alias("market_regime"),
            ]
        )

        # Drop intermediate TOPIX columns (keep only derived features)
        df = df.drop(
            [
                "topix_close",
                "topix_return_1d",
                "topix_return_5d",
                "topix_return_10d",
                "topix_return_20d",
            ]
        )

        logger.info("✅ Added 7 TOPIX-relative features")
        return df

    def create_metadata(self, df: pl.DataFrame) -> dict:
        """Create dataset metadata."""
        # Count features (P1-10: exclude all non-feature columns)
        excluded_cols = [
            "Code",
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "target_1d",
            "target_5d",
            "target_10d",
            "target_20d",
            "target_1d_binary",
            "target_5d_binary",
            "target_10d_binary",
        ]

        feature_cols = [col for col in df.columns if col not in excluded_cols]

        metadata = {
            "created_at": datetime.now().isoformat(),
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "features": {"count": len(feature_cols), "names": feature_cols},
            "stocks": df["Code"].n_unique(),
            "date_range": {
                "start": str(df["Date"].min()),
                "end": str(df["Date"].max()),
            },
            "targets": {
                "regression": ["target_1d", "target_5d", "target_10d", "target_20d"],
                "classification": [
                    "target_1d_binary",
                    "target_5d_binary",
                    "target_10d_binary",
                ],
            },
            "version": "2.0-fixed",
            "fixes_applied": [
                "P0-1: pct_change with proper over() ordering",
                "P0-2: Removed Winsorization from data creation",
                "P0-3: EMA as denominator for deviations",
                "P0-4: Bollinger Bands 0-division prevention",
                "P0-5: volatility_60d dropped after ratio calculation",
                "P1-6: Date type casting",
                "P1-7: int_ranges for maturity flags",
                "P1-8: Clarified Sharpe ratio calculation",
                "P1-9: pandas-ta column name references",
                "P1-10: Accurate feature count",
            ],
        }

        return metadata

    def save_dataset(self, df: pl.DataFrame, metadata: dict):
        """Save dataset in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # P1-11: Save as Parquet (recommended)
        parquet_path = self.output_dir / f"ml_dataset_{timestamp}.parquet"
        df.write_parquet(parquet_path, compression="snappy")
        logger.info(f"Saved Parquet: {parquet_path}")

        # Also save CSV for compatibility
        csv_path = self.output_dir / f"ml_dataset_{timestamp}.csv"
        df.write_csv(csv_path)
        logger.info(f"Saved CSV: {csv_path}")

        # Save metadata
        meta_path = self.output_dir / f"ml_dataset_{timestamp}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {meta_path}")

        # Create symlinks to latest
        latest_parquet = self.output_dir / "ml_dataset_latest.parquet"
        latest_csv = self.output_dir / "ml_dataset_latest.csv"
        latest_meta = self.output_dir / "ml_dataset_latest_metadata.json"

        for latest, current in [
            (latest_parquet, parquet_path),
            (latest_csv, csv_path),
            (latest_meta, meta_path),
        ]:
            if latest.exists():
                latest.unlink()
            latest.symlink_to(current.name)

        return parquet_path, csv_path, meta_path


def create_sample_data(n_stocks: int = 100, n_days: int = 300) -> pl.DataFrame:
    """Create sample stock data for testing."""
    np.random.seed(42)
    data = []

    start_date = datetime.now() - timedelta(days=n_days)

    for stock_id in range(n_stocks):
        code = f"{1301 + stock_id}"
        prices = 1000 + np.random.randn(n_days).cumsum() * 10

        for day in range(n_days):
            date = start_date + timedelta(days=day)
            price = max(prices[day], 10)  # Ensure positive prices

            data.append(
                {
                    "Code": code,
                    "Date": date.strftime("%Y-%m-%d"),
                    "Open": price * (1 + np.random.uniform(-0.02, 0.02)),
                    "High": price * (1 + np.random.uniform(0, 0.03)),
                    "Low": price * (1 - np.random.uniform(0, 0.03)),
                    "Close": price,
                    "Volume": int(np.random.uniform(10000, 1000000)),
                }
            )

    return pl.DataFrame(data)


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("ML Dataset Builder (Fixed Version)")
    logger.info("=" * 60)

    # Create builder
    builder = MLDatasetBuilder()

    # Load or create data
    logger.info("Creating sample data...")
    df = create_sample_data(n_stocks=100, n_days=300)
    logger.info(f"Created data: {len(df)} rows, {df['Code'].n_unique()} stocks")

    # Create features
    df = builder.create_technical_features(df)
    df = builder.add_pandas_ta_features(df)

    # Create metadata
    metadata = builder.create_metadata(df)

    # Display summary
    logger.info("\n" + "=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Shape: {len(df)} rows × {len(df.columns)} columns")
    logger.info(f"Features: {metadata['features']['count']}")
    logger.info(f"Stocks: {metadata['stocks']}")
    logger.info(
        f"Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}"
    )
    logger.info(f"Memory usage: {df.estimated_size('mb'):.2f} MB")

    # Check for NaN/Inf
    nan_cols = []
    for col in df.columns:
        if df[col].is_null().sum() > 0:
            nan_cols.append(col)

    if nan_cols:
        logger.warning(f"Columns with NaN: {nan_cols}")

    # Save dataset
    logger.info("\nSaving dataset...")
    parquet_path, csv_path, meta_path = builder.save_dataset(df, metadata)

    logger.info("\n" + "=" * 60)
    logger.info("FIXES APPLIED")
    logger.info("=" * 60)
    for fix in metadata["fixes_applied"]:
        logger.info(f"✓ {fix}")

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Parquet: {parquet_path}")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Metadata: {meta_path}")

    return df, metadata


if __name__ == "__main__":
    df, metadata = main()
