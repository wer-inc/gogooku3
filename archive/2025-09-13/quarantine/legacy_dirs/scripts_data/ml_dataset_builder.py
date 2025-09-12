#!/usr/bin/env python3
"""
ML Dataset Builder with Polars (Enhanced Version for gogooku3-standalone)
gogooku3の構造をベースに、より多くの銘柄を含むMLデータセット作成処理
"""

import asyncio
import json
import logging
import os

# P0-1: Import feature validator for min_periods consistency
import sys
import time
from datetime import datetime, timedelta
from datetime import time as dt_time
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import pandas_ta as ta
import polars as pl

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
try:
    from features.feature_validator import FeatureValidator
except ImportError:
    # Logger may not be initialized yet here; avoid referencing it.
    FeatureValidator = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_data(n_stocks: int = 100, n_days: int = 300) -> pl.DataFrame:
    """サンプルデータを作成（テスト用）"""
    logger.info(f"Creating sample data: {n_stocks} stocks × {n_days} days")

    data = []
    base_date = datetime.now() - timedelta(days=n_days)

    for stock_idx in range(n_stocks):
        code = f"{1000 + stock_idx}"
        base_price = 1000 + stock_idx * 10

        for day_idx in range(n_days):
            date = base_date + timedelta(days=day_idx)
            if date.weekday() >= 5:  # Skip weekends
                continue

            # Generate random price movement
            change = np.random.randn() * 20
            close = base_price + change

            data.append({
                "Code": code,
                "Date": date.strftime("%Y-%m-%d"),
                "Open": close - abs(np.random.randn() * 5),
                "High": close + abs(np.random.randn() * 10),
                "Low": close - abs(np.random.randn() * 10),
                "Close": close,
                "Volume": int(np.random.uniform(100000, 1000000))
            })

    df = pl.DataFrame(data)
    logger.info(f"Created sample data: {len(df)} rows")
    return df


class MLDatasetBuilder:
    """Build ML dataset with enhanced stock coverage using gogooku2 batch data."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # gogooku2バッチデータの参照パス
        self.gogooku2_data_path = Path("/home/ubuntu/gogooku2/output/batch")

    def find_latest_ta_file(self) -> Path | None:
        """gogooku2のバッチ処理から最新のTAファイルを探す"""
        try:
            # 最新のTAファイルを探索
            ta_patterns = [
                "ta/parquet_*/tse_ta_*.parquet",
                "*/ta/parquet_*/tse_ta_*.parquet",
                "20*/ta/parquet_*/tse_ta_*.parquet"
            ]

            latest_file = None
            latest_time = 0

            for pattern in ta_patterns:
                for file_path in self.gogooku2_data_path.glob(pattern):
                    if file_path.is_file():
                        mtime = file_path.stat().st_mtime
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_file = file_path

            if latest_file:
                logger.info(f"Found latest TA file: {latest_file}")
                return latest_file
            else:
                logger.warning("No TA files found in gogooku2 output")
                return None

        except Exception as e:
            logger.error(f"Error finding TA file: {e}")
            return None

    def load_gogooku2_data(self) -> pl.DataFrame | None:
        """gogooku2のバッチデータを読み込み"""
        try:
            ta_file = self.find_latest_ta_file()
            if not ta_file:
                return None

            logger.info(f"Loading gogooku2 data from: {ta_file}")
            df = pl.read_parquet(ta_file)

            logger.info(f"Loaded gogooku2 data: {len(df):,} rows, {df['code'].n_unique():,} stocks")
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

            # 必要なカラム名の標準化
            rename_mapping = {
                'code': 'Code',
                'date': 'Date',
                'adjustment_close': 'Close',
                'adjustment_open': 'Open',
                'adjustment_high': 'High',
                'adjustment_low': 'Low',
                'adjustment_volume': 'Volume'
            }

            # 存在するカラムのみリネーム
            existing_renames = {k: v for k, v in rename_mapping.items() if k in df.columns}
            df = df.rename(existing_renames)

            # 日付を文字列形式に変換（ml_dataset_builderの期待形式）
            df = df.with_columns(pl.col("Date").dt.strftime("%Y-%m-%d"))

            return df

        except Exception as e:
            logger.error(f"Error loading gogooku2 data: {e}")
            return None

    def filter_stocks_by_quality(self, df: pl.DataFrame, min_data_points: int = 200) -> pl.DataFrame:
        """株式を品質基準でフィルタリング（より多くの銘柄を保持）"""
        logger.info("Applying quality filters to preserve more stocks...")

        initial_stocks = df['Code'].n_unique()

        # データ点数による基本的なフィルタリング（緩めの基準）
        stock_counts = df.group_by('Code').count()
        valid_stocks = stock_counts.filter(pl.col('count') >= min_data_points)['Code'].to_list()

        df_filtered = df.filter(pl.col('Code').is_in(valid_stocks))

        final_stocks = df_filtered['Code'].n_unique()

        logger.info(f"Stock filtering: {initial_stocks} → {final_stocks} stocks")
        logger.info(f"Preserved {final_stocks/initial_stocks*100:.1f}% of stocks")

        return df_filtered

    def create_technical_features(
        self, df: pl.DataFrame, topix_df: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        """Create all technical features with enhanced stock coverage."""
        logger.info("Creating technical features with Polars...")
        start_time = time.time()

        # Sort by Code and Date first
        df = df.with_columns(
            pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        ).sort(["Code", "Date"])

        # Use cumulative count for maturity index
        df = df.with_columns(pl.col("Date").cum_count().over("Code").alias("row_idx"))

        # ========== RETURNS (8 features) - MUST BE CALCULATED FIRST ==========
        df = df.with_columns(
            [
                pl.col("Close").pct_change().over("Code").alias("returns_1d"),
                pl.col("Close").pct_change(5).over("Code").alias("returns_5d"),
                pl.col("Close").pct_change(10).over("Code").alias("returns_10d"),
                pl.col("Close").pct_change(20).over("Code").alias("returns_20d"),
                # Log returns
                (pl.col("Close") / pl.col("Close").shift(1)).log().over("Code").alias("log_returns_1d"),
                (pl.col("Close") / pl.col("Close").shift(5)).log().over("Code").alias("log_returns_5d"),
                (pl.col("Close") / pl.col("Close").shift(10)).log().over("Code").alias("log_returns_10d"),
                (pl.col("Close") / pl.col("Close").shift(20)).log().over("Code").alias("log_returns_20d"),
            ]
        )

        # ========== EMAs (5 features) ==========
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
                # MA slopes
                pl.col("ema_5").pct_change().over("Code").alias("ema5_slope"),
                pl.col("ema_20").pct_change().over("Code").alias("ema20_slope"),
                pl.col("ema_60").pct_change().over("Code").alias("ema60_slope"),
                # MA crosses
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

        # MA ribbon spread
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

        # Distance to 200 EMA
        df = df.with_columns(
            ((pl.col("Close") - pl.col("ema_200")) / pl.col("ema_200")).alias(
                "dist_to_200ema"
            )
        )

        # ========== VOLUME FEATURES (6 features) ==========
        df = df.with_columns(
            [
                # Volume moving averages
                pl.col("Volume").rolling_mean(window_size=5, min_periods=5).over("Code").alias("volume_ma_5"),
                pl.col("Volume").rolling_mean(window_size=20, min_periods=20).over("Code").alias("volume_ma_20"),
            ]
        )

        df = df.with_columns(
            [
                # Volume ratios
                (pl.col("Volume") / (pl.col("volume_ma_5") + 1e-12)).alias("volume_ratio_5"),
                (pl.col("Volume") / (pl.col("volume_ma_20") + 1e-12)).alias("volume_ratio_20"),
                # Dollar volume (also as TurnoverValue for compatibility)
                (pl.col("Close") * pl.col("Volume")).alias("dollar_volume"),
                (pl.col("Close") * pl.col("Volume")).alias("TurnoverValue"),  # Alias for DATASET.md compatibility
            ]
        )

        # ========== PRICE POSITION INDICATORS (3 features) ==========
        df = df.with_columns(
            [
                # High-low ratio
                (pl.col("High") / (pl.col("Low") + 1e-12)).alias("high_low_ratio"),
                # Close position relative to high-low range
                ((pl.col("High") - pl.col("Close")) / (pl.col("High") - pl.col("Low") + 1e-12)).alias("close_to_high"),
                ((pl.col("Close") - pl.col("Low")) / (pl.col("High") - pl.col("Low") + 1e-12)).alias("close_to_low"),
            ]
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

        # ========== VOLATILITY (10 features) ==========
        # P0-1 & P0-4: Apply min_periods and add realized volatility
        df = df.with_columns(
            [
                pl.col("Close")
                .pct_change()
                .over("Code")
                .rolling_std(window_size=5, min_periods=5)
                .alias("volatility_5d_raw"),
                pl.col("Close")
                .pct_change()
                .over("Code")
                .rolling_std(window_size=10, min_periods=10)
                .alias("volatility_10d_raw"),
                pl.col("Close")
                .pct_change()
                .over("Code")
                .rolling_std(window_size=20, min_periods=20)  # P0-1: min_periods applied
                .alias("volatility_20d_raw"),
                pl.col("Close")
                .pct_change()
                .over("Code")
                .rolling_std(window_size=60, min_periods=60)  # P0-1: min_periods applied
                .alias("volatility_60d_raw"),
            ]
        )

        # Annualize volatilities
        df = df.with_columns(
            [
                (pl.col("volatility_5d_raw") * np.sqrt(252)).alias("volatility_5d"),
                (pl.col("volatility_10d_raw") * np.sqrt(252)).alias("volatility_10d"),
                (pl.col("volatility_20d_raw") * np.sqrt(252)).alias("volatility_20d"),
                (pl.col("volatility_60d_raw") * np.sqrt(252)).alias("volatility_60d"),
            ]
        )

        # P0-4: Add Parkinson realized volatility (correct implementation)
        if FeatureValidator:
            validator = FeatureValidator()
            df = validator.calculate_realized_volatility(df, window=20, annualize=True)
        else:
            # Fallback Parkinson implementation
            df = df.with_columns([
                ((pl.col("High") / pl.col("Low")).log() ** 2 / (4 * np.log(2)))
                .rolling_mean(window_size=20, min_periods=20)
                .over("Code")
                .pipe(lambda x: (x * 252).sqrt())
                .alias("realized_vol_20")
            ])

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

        # Sharpe ratios
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

        # ========== FLOW (Removed - will be replaced by actual flow features) ==========
        # Old placeholder flow features removed - see add_flow_features() for actual implementation

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

        # P0-1: Add comprehensive validity flags using FeatureValidator
        if FeatureValidator:
            validator = FeatureValidator()
            df = validator.add_validity_flags(df)
            # P1: Remove redundant features
            df = validator.remove_redundant_features(df)
        else:
            # Fallback: Basic maturity flags
            df = df.with_columns(
                [
                    (pl.col("row_idx") >= 2).cast(pl.Int8).alias("is_rsi2_valid"),  # RSI2 needs 2 days
                    (pl.col("row_idx") >= 15).cast(pl.Int8).alias("is_ema5_valid"),
                    (pl.col("row_idx") >= 30).cast(pl.Int8).alias("is_ema10_valid"),
                    (pl.col("row_idx") >= 60).cast(pl.Int8).alias("is_ema20_valid"),
                    (pl.col("row_idx") >= 200).cast(pl.Int8).alias("is_ema200_valid"),
                    (pl.col("row_idx") >= 60).cast(pl.Int8).alias("is_valid_ma"),
                ]
            )

        # Drop raw columns only
        cols_to_drop = ["volatility_5d_raw", "volatility_10d_raw", "volatility_20d_raw", "volatility_60d_raw"]
        df = df.drop([col for col in cols_to_drop if col in df.columns])

        # ウォームアップ期間のデータも保持（is_valid_*フラグで管理）
        # データを削除せず、全期間のデータを保持する

        elapsed = time.time() - start_time
        logger.info(f"Created technical features in {elapsed:.2f} seconds (all data preserved)")

        return df

    def add_pandas_ta_features(self, df_polars: pl.DataFrame) -> pl.DataFrame:
        """Add complex technical indicators using pandas-ta."""
        logger.info("Adding pandas-ta features...")
        start_time = time.time()

        # Convert to pandas for pandas-ta
        df_pandas = df_polars.to_pandas()

        # Process each stock separately
        results = []
        stock_codes = df_pandas["Code"].unique()

        logger.info(f"Processing {len(stock_codes)} stocks for pandas-ta features...")

        for i, code in enumerate(stock_codes):
            if i % 100 == 0:
                logger.info(f"Processing stock {i+1}/{len(stock_codes)}: {code}")

            stock_df = df_pandas[df_pandas["Code"] == code].copy()
            stock_df = stock_df.sort_values("Date")

            # RSI calculations
            rsi_14 = ta.rsi(stock_df["Close"], length=14)
            rsi_2 = ta.rsi(stock_df["Close"], length=2)

            # Handle None values from ta.rsi
            stock_df["rsi_14"] = rsi_14 if rsi_14 is not None else 50.0
            stock_df["rsi_2"] = rsi_2 if rsi_2 is not None else 50.0

            # RSI delta
            if rsi_14 is not None and not rsi_14.empty:
                stock_df["rsi_delta"] = rsi_14.diff()
            else:
                stock_df["rsi_delta"] = 0.0

            # MACD
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
                stock_df["macd_signal"] = 0
                stock_df["macd_histogram"] = 0

            # ADX (Average Directional Index)
            try:
                adx = ta.adx(stock_df["High"], stock_df["Low"], stock_df["Close"], length=14)
                if adx is not None and not adx.empty and "ADX_14" in adx.columns:
                    stock_df["adx_14"] = adx["ADX_14"]
                else:
                    stock_df["adx_14"] = np.nan
            except Exception:
                stock_df["adx_14"] = np.nan

            # Bollinger Bands
            bb = ta.bbands(stock_df["Close"], length=20, std=2)
            if bb is not None and not bb.empty:
                bb_upper = bb["BBU_20_2.0"]
                bb_lower = bb["BBL_20_2.0"]
                bb_middle = bb["BBM_20_2.0"]

                # Add raw Bollinger Band values
                stock_df["bb_upper"] = bb_upper
                stock_df["bb_lower"] = bb_lower
                stock_df["bb_middle"] = bb_middle

                stock_df["bb_pct_b"] = (
                    (stock_df["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-12)
                ).clip(0, 1)
                stock_df["bb_bandwidth"] = (bb_upper - bb_lower) / (bb_middle + 1e-12)
            else:
                stock_df["bb_pct_b"] = 0.5
                stock_df["bb_bandwidth"] = 0
                stock_df["bb_upper"] = np.nan
                stock_df["bb_lower"] = np.nan
                stock_df["bb_middle"] = np.nan

            results.append(stock_df)

        # Combine and convert back to Polars
        df_combined = pd.concat(results, ignore_index=True)
        # In sandboxed environments, polars.from_pandas may use multiprocessing
        # which can fail due to SemLock permissions. Use a safer conversion path.
        try:
            import pyarrow as pa  # type: ignore
            table = pa.Table.from_pandas(df_combined, preserve_index=False)
            df_final = pl.from_arrow(table)
        except Exception:
            # Fallback: round-trip via Parquet file to avoid multiprocessing path
            tmp_path = Path("output/_tmp_pandas_to_polars.parquet")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                df_combined.to_parquet(tmp_path, index=False)
            except Exception:
                # Last resort: CSV round-trip
                tmp_path = Path("output/_tmp_pandas_to_polars.csv")
                df_combined.to_csv(tmp_path, index=False)
                df_final = pl.read_csv(tmp_path)
            else:
                df_final = pl.read_parquet(tmp_path)
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        elapsed = time.time() - start_time
        logger.info(f"Added pandas-ta features in {elapsed:.2f} seconds")

        return df_final

    def add_sector_features(
        self,
        df: pl.DataFrame,
        listed_info_df: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """
        Attach sector and market attributes from listed_info to the daily dataset.

        - Adds `sector33` as the normalized Sector33Code
        - Also attaches `MarketCode` and `CompanyName` for reference
        - Uses as-of interval join so historical changes are respected

        Args:
            df: Daily stock dataset (must include Code, Date)
            listed_info_df: Listed info dataset including Code, Date, MarketCode, Sector33Code

        Returns:
            Dataset with sector and market attributes attached
        """
        try:
            if listed_info_df is None or listed_info_df.is_empty():
                logger.warning("No listed_info provided; skipping sector enrichment")
                return df

            # Ensure required columns exist; tolerate alternative casings
            cols = set(listed_info_df.columns)
            required = {"Code", "Date", "MarketCode", "Sector33Code"}
            missing = [c for c in required if c not in cols]
            if missing:
                logger.warning(f"listed_info missing columns {missing}; skipping sector enrichment")
                return df

            # Build validity intervals per Code for sector codes/names and MarketCode
            info = (
                listed_info_df
                .select([
                    pl.col("Code").cast(pl.Utf8),
                    pl.col("Date").cast(pl.Date).alias("valid_from"),
                    pl.col("MarketCode").cast(pl.Utf8),
                    pl.col("Sector33Code").cast(pl.Utf8),
                    # Optional names/codes if provided by API/pipeline
                    pl.col("Sector17Code").cast(pl.Utf8) if "Sector17Code" in cols else pl.lit(None).alias("Sector17Code"),
                    pl.col("Sector17Name").cast(pl.Utf8) if "Sector17Name" in cols else pl.lit(None).alias("Sector17Name"),
                    pl.col("Sector33Name").cast(pl.Utf8) if "Sector33Name" in cols else pl.lit(None).alias("Sector33Name"),
                    pl.col("CompanyName").cast(pl.Utf8) if "CompanyName" in cols else pl.lit(None).alias("CompanyName"),
                ])
                .sort(["Code", "valid_from"])
                .with_columns([
                    pl.col("valid_from").shift(-1).over("Code").alias("next_change"),
                ])
                .with_columns([
                    pl.when(pl.col("next_change").is_not_null())
                    .then(pl.col("next_change") - pl.duration(days=1))
                    .otherwise(pl.date(2999, 12, 31))
                    .alias("valid_to")
                ])
            )

            # Choose join key: prefer LocalCode (5-char) if present; else Code (4-digit)
            use_local = "LocalCode" in df.columns
            join_left = "LocalCode" if use_local else "Code"

            # Normalize target dataset dtypes
            base = df.sort([join_left, "Date"]).with_columns([
                pl.col(join_left).cast(pl.Utf8),
                pl.col("Date").cast(pl.Date),
            ])

            # Fallback: if listed_info is a snapshot (all valid_from after dataset start),
            # shift valid_from back to cover the dataset to enable joins.
            try:
                base_min = base.select(pl.col("Date").min().alias("min")).to_dicts()[0]["min"]
                info_min = info.select(pl.col("valid_from").min().alias("min")).to_dicts()[0]["min"]
                if info_min is not None and base_min is not None and info_min > base_min:
                    info = info.with_columns(pl.lit(base_min).cast(pl.Date).alias("valid_from"))
            except Exception:
                pass

            # Prepare right side key name to match left
            info_key = "Code"  # listed_info 'Code' is effectively LocalCode
            info_sorted = info.select([
                info_key, "valid_from", "valid_to",
                "Sector33Code", "Sector33Name", "Sector17Code", "Sector17Name",
                "MarketCode", "CompanyName"
            ]).sort([info_key, "valid_from"]).rename({info_key: join_left})

            # Determine if listed_info looks like a single snapshot (one row per Code)
            try:
                counts = info_sorted.group_by(join_left).count().select(pl.col("count").max()).to_series().item()
                snapshot_mode = counts == 1
            except Exception:
                snapshot_mode = False

            if snapshot_mode:
                # Simple left join by Code for snapshot data (prefer right-hand columns via suffix)
                joined = base.join(
                    info_sorted.drop([c for c in ["valid_from", "valid_to"] if c in info_sorted.columns]),
                    on=join_left,
                    how="left",
                    suffix="_info",
                )
            else:
                # As-of join on validity intervals using chosen key
                joined = base.join_asof(
                    info_sorted,
                    by=join_left,
                    left_on="Date",
                    right_on="valid_from",
                    strategy="backward",
                    suffix="_info",
                )
                # Keep only rows within validity interval
                joined = joined.filter(
                    (pl.col("valid_to").is_null()) | (pl.col("Date") <= pl.col("valid_to"))
                )

            # Standardize output columns and create IDs
            out = joined.drop([c for c in ["valid_from", "valid_to", "next_change"] if c in joined.columns])
            # Prefer right-hand columns if both sides carry same names
            def pick(name: str) -> pl.Expr:
                right = f"{name}_info"
                if right in out.columns:
                    return pl.coalesce([pl.col(right), pl.col(name)])
                return pl.col(name)

            out = out.with_columns([
                pick("Sector33Code").alias("sector33_code"),
                pick("Sector33Name").alias("sector33_name") if "Sector33Name" in out.columns or "Sector33Name_info" in out.columns else pl.lit(None).alias("sector33_name"),
                pick("Sector17Code").alias("sector17_code") if "Sector17Code" in out.columns or "Sector17Code_info" in out.columns else pl.lit(None).alias("sector17_code"),
                pick("Sector17Name").alias("sector17_name") if "Sector17Name" in out.columns or "Sector17Name_info" in out.columns else pl.lit(None).alias("sector17_name"),
                # Backward-compatible alias (expected by some training configs)
                pick("Sector33Code").alias("sector33"),
                # Also prefer MarketCode from info if available
                pick("MarketCode").alias("MarketCode"),
            ])

            # Build stable ID dictionaries from observed codes
            def _id_map(series: pl.Series) -> dict:
                vals = [v for v in series.to_list() if v is not None]
                uniq = sorted(set(vals))
                return {code: i for i, code in enumerate(uniq)}

            map33 = _id_map(out.select(pl.col("sector33_code")).to_series()) if "sector33_code" in out.columns else {}
            map17 = _id_map(out.select(pl.col("sector17_code")).to_series()) if "sector17_code" in out.columns else {}

            out = out.with_columns([
                pl.col("sector33_code").map_elements(lambda v: int(map33.get(v, -1)) if v is not None else -1, return_dtype=pl.Int32).alias("sector33_id"),
                pl.col("sector17_code").map_elements(lambda v: int(map17.get(v, -1)) if v is not None else -1, return_dtype=pl.Int32).alias("sector17_id"),
            ])

            # Log coverage
            has_sector = (out["sector33_code"].is_not_null().sum()) if "sector33_code" in out.columns else 0
            logger.info(f"Attached sector33 to {has_sector} records")

            return out
        except Exception as e:
            logger.error(f"Error adding sector features: {e}")
            logger.warning("Continuing without sector enrichment")
            return df

    def add_sector_series(
        self,
        df: pl.DataFrame,
        *,
        level: str = "33",
        windows: tuple[int, ...] = (1, 5, 20),
        method: str = "eq_median",
        series_mcap: str = "auto",
    ) -> pl.DataFrame:
        """
        Compute sector aggregate series (eq-median) and attach to rows.

        - Per-date median of returns_1d per sector: sec_ret_1d_eq
        - Rolling h-day sector return via log-sum: sec_ret_{h}d_eq
        """
        try:
            if method != "eq_median":
                logger.warning("Only eq_median supported; using eq_median")

            # Ensure returns_1d exists
            if "returns_1d" not in df.columns and "Close" in df.columns:
                df = df.with_columns(
                    pl.col("Close").pct_change().over("Code").alias("returns_1d")
                )

            id_col = "sector33_id" if level == "33" else "sector17_id"
            if id_col not in df.columns:
                logger.warning(f"{id_col} not present; skipping sector series")
                return df

            sec_daily = (
                df.select(["Date", id_col, "returns_1d"])  # type: ignore[list-item]
                .group_by(["Date", id_col])
                .agg(pl.col("returns_1d").median().alias("sec_ret_1d_eq"))
                .sort([id_col, "Date"])
            )

            def add_window(pdf: pl.DataFrame, h: int) -> pl.DataFrame:
                if h <= 1:
                    return pdf
                return pdf.with_columns(
                    (
                        (1.0 + pl.col("sec_ret_1d_eq")).log()
                        .rolling_sum(window_size=h, min_periods=h)
                        .over(id_col)
                        .exp()
                        - 1.0
                    ).alias(f"sec_ret_{h}d_eq")
                )

            for w in windows:
                if w > 1:
                    sec_daily = add_window(sec_daily, w)

            # Optional: market-cap weighted daily sector return series
            try:
                use_mcap = series_mcap in ("auto", "always") and "shares_outstanding" in df.columns
                if series_mcap == "always" and "shares_outstanding" not in df.columns:
                    logger.warning("series_mcap='always' requested but shares_outstanding is missing; skipping mcap series")
                if use_mcap:
                    tmp = df
                    if "returns_1d" not in tmp.columns and "Close" in tmp.columns:
                        tmp = tmp.with_columns(pl.col("Close").pct_change().over("Code").alias("returns_1d"))
                    tmp = tmp.with_columns((pl.col("Close") * pl.col("shares_outstanding")).alias("mcap"))
                    sec_mcap = (
                        tmp.select(["Date", id_col, "returns_1d", "mcap"])  # type: ignore[list-item]
                        .group_by(["Date", id_col])
                        .agg([
                            (pl.col("returns_1d") * pl.col("mcap")).sum().alias("num"),
                            pl.col("mcap").sum().alias("den"),
                        ])
                        .with_columns((pl.col("num") / (pl.col("den") + 1e-12)).alias("sec_ret_1d_mcap"))
                        .select(["Date", id_col, "sec_ret_1d_mcap"])  # keep only needed
                        .sort([id_col, "Date"])
                    )
                    sec_daily = sec_daily.join(sec_mcap, on=["Date", id_col], how="left")
                    # rolling windows for mcap series
                    def add_mcap_window(pdf: pl.DataFrame, h: int) -> pl.DataFrame:
                        if h <= 1 or "sec_ret_1d_mcap" not in pdf.columns:
                            return pdf
                        return pdf.with_columns(
                            (
                                (1.0 + pl.col("sec_ret_1d_mcap")).log()
                                .rolling_sum(window_size=h, min_periods=h)
                                .over(id_col)
                                .exp()
                                - 1.0
                            ).alias(f"sec_ret_{h}d_mcap")
                        )
                    for w in windows:
                        if w > 1:
                            sec_daily = add_mcap_window(sec_daily, w)
            except Exception as e:
                logger.warning(f"mcap sector series failed: {e}")

            # Momentum/EMA/Vol features on sector return series
            eps = 1e-12
            sec_daily = sec_daily.with_columns([
                # Momentum over 20d (sum of daily returns as approximation)
                pl.col("sec_ret_1d_eq").rolling_sum(window_size=20, min_periods=20).over(id_col).alias("sec_mom_20"),
                # EMAs over sector daily returns
                pl.col("sec_ret_1d_eq").ewm_mean(span=5, adjust=False).over(id_col).alias("sec_ema_5"),
                pl.col("sec_ret_1d_eq").ewm_mean(span=20, adjust=False).over(id_col).alias("sec_ema_20"),
            ]).with_columns([
                ((pl.col("sec_ema_5") - pl.col("sec_ema_20")) / (pl.col("sec_ema_20") + eps)).alias("sec_gap_5_20"),
            ]).with_columns([
                # Volatility of sector returns (20d)
                (pl.col("sec_ret_1d_eq").rolling_std(window_size=20, min_periods=20).over(id_col) * (252 ** 0.5)).alias("sec_vol_20"),
            ])

            # Time-series Z-score for sec_vol_20 (window 252)
            mu252 = pl.col("sec_vol_20").rolling_mean(window_size=252, min_periods=252).over(id_col)
            sd252 = pl.col("sec_vol_20").rolling_std(window_size=252, min_periods=252).over(id_col) + eps
            sec_daily = sec_daily.with_columns([
                ((pl.col("sec_vol_20") - mu252) / sd252).alias("sec_vol_20_z")
            ])

            # If attaching 17-level as well, prefix to avoid collisions with 33-level
            if level == "17":
                rename_map = {c: f"sec17_{c[4:]}" for c in sec_daily.columns if c.startswith("sec_")}
                if rename_map:
                    sec_daily = sec_daily.rename(rename_map)

            # Build keep lists
            keep = [
                "Date",
                id_col,
            ] + [c for c in sec_daily.columns if c.startswith("sec_ret_") or c.startswith("sec17_ret_")]
            # Also keep additional sector columns
            keep_extra = [
                col
                for col in [
                    "sec_mom_20",
                    "sec_ema_5",
                    "sec_ema_20",
                    "sec_gap_5_20",
                    "sec_vol_20",
                    "sec_vol_20_z",
                    "sec17_mom_20",
                    "sec17_ema_5",
                    "sec17_ema_20",
                    "sec17_gap_5_20",
                    "sec17_vol_20",
                    "sec17_vol_20_z",
                ]
                if col in sec_daily.columns
            ]
            out = df.join(sec_daily.select(keep + keep_extra), on=["Date", id_col], how="left")
            added_cols = [c for c in out.columns if c.startswith("sec_ret_") or c.startswith("sec_")]
            logger.info("Attached sector series columns: " + ", ".join(sorted(set(added_cols))))
            return out
        except Exception as e:
            logger.error(f"Error computing sector series: {e}")
            return df

    def add_sector_encodings(
        self,
        df: pl.DataFrame,
        *,
        onehot_17: bool = True,
        onehot_33: bool = False,
        freq_daily: bool = True,
        rare_threshold: float = 0.005,
    ) -> pl.DataFrame:
        """
        Add sector categorical encodings:
        - One-hot for 17-sector (rare categories -> Other bucket)
        - Daily frequency encodings for 17/33 sectors
        """
        try:
            out = df
            n_rows = len(out)
            # One-hot for 17-sector
            if onehot_17 and "sector17_id" in out.columns and n_rows > 0:
                counts = out.group_by("sector17_id").count().rename({"count": "cnt"})
                counts = counts.with_columns((pl.col("cnt") / float(n_rows)).alias("ratio"))
                keep_ids = set(counts.filter(pl.col("ratio") >= rare_threshold)["sector17_id"].to_list())

                # Build one-hot columns for kept categories
                for k in sorted(keep_ids):
                    col_name = f"sec17_onehot_{k}"
                    out = out.with_columns((pl.col("sector17_id") == pl.lit(k)).cast(pl.Int8).alias(col_name))

                # Other bucket (includes unknown -1 and rare categories)
                kept_list = list(keep_ids)
                out = out.with_columns(
                    (~pl.col("sector17_id").is_in(kept_list)).cast(pl.Int8).alias("sec17_onehot_other")
                )

            # One-hot for 33-sector (optional)
            if onehot_33 and "sector33_id" in out.columns and n_rows > 0:
                counts33 = out.group_by("sector33_id").count().rename({"count": "cnt"})
                counts33 = counts33.with_columns((pl.col("cnt") / float(n_rows)).alias("ratio"))
                keep33 = set(counts33.filter(pl.col("ratio") >= rare_threshold)["sector33_id"].to_list())
                for k in sorted(keep33):
                    out = out.with_columns((pl.col("sector33_id") == pl.lit(k)).cast(pl.Int8).alias(f"sec33_onehot_{k}"))
                out = out.with_columns((~pl.col("sector33_id").is_in(list(keep33))).cast(pl.Int8).alias("sec33_onehot_other"))

            # Daily frequency encodings for 17 & 33
            if freq_daily and "Date" in out.columns:
                # 17-sector daily freq via window counts (no join)
                if "sector17_id" in out.columns:
                    out = out.with_columns([
                        pl.count().over(["Date", "sector17_id"]).alias("_cnt17"),
                        pl.count().over("Date").alias("_tot17"),
                    ])
                    out = out.with_columns(
                        (pl.col("_cnt17") / (pl.col("_tot17") + 1e-12)).alias("sec17_daily_freq")
                    ).drop(["_cnt17", "_tot17"])

                # 33-sector daily freq via window counts (no join)
                if "sector33_id" in out.columns:
                    out = out.with_columns([
                        pl.count().over(["Date", "sector33_id"]).alias("_cnt33"),
                        pl.count().over("Date").alias("_tot33"),
                    ])
                    out = out.with_columns(
                        (pl.col("_cnt33") / (pl.col("_tot33") + 1e-12)).alias("sec33_daily_freq")
                    ).drop(["_cnt33", "_tot33"])

            logger.info("Added sector encodings (one-hot 17, daily freq 17/33 if available)")
            return out
        except Exception as e:
            logger.error(f"Error adding sector encodings: {e}")
            return df

    def add_relative_to_sector(
        self,
        df: pl.DataFrame,
        *,
        level: str = "33",
        x_cols: tuple[str, ...] = ("returns_5d", "ma_gap_5_20"),
        beta_window: int = 60,
        beta_min_periods: int = 20,
    ) -> pl.DataFrame:
        """
        Add relative-to-sector features (no leakage):
        - rel_to_sec_5d = returns_5d - sec_ret_5d_eq(sector_of(i,t), t)
        - alpha_vs_sec_1d = returns_1d - beta(i)*sec_ret_1d_eq where beta is rolling(Code-wise)
        - ret_1d_demeaned = returns_1d - mean_{sector at t}(returns_1d)
        - z_in_sec_{x} = cross-sectional Z within (Date×sector) for selected x_cols
        """
        try:
            out = df
            id_col = "sector33_id" if level == "33" else "sector17_id"
            if id_col not in out.columns:
                logger.warning(f"{id_col} not present; skipping relative-to-sector features")
                return out

            # Ensure returns columns exist
            if "returns_1d" not in out.columns and "Close" in out.columns:
                out = out.with_columns(
                    pl.col("Close").pct_change().over("Code").alias("returns_1d")
                )
            if "returns_5d" not in out.columns and "Close" in out.columns:
                out = out.with_columns(
                    pl.col("Close").pct_change(5).over("Code").alias("returns_5d")
                )

            # rel_to_sec_5d
            if "sec_ret_5d_eq" in out.columns and "returns_5d" in out.columns:
                out = out.with_columns(
                    (pl.col("returns_5d") - pl.col("sec_ret_5d_eq")).alias("rel_to_sec_5d")
                )

            # alpha_vs_sec_1d (rolling beta per Code vs sector daily return)
            eps = 1e-12
            if "sec_ret_1d_eq" in out.columns and "returns_1d" in out.columns:
                out = out.with_columns([
                    pl.col("returns_1d").rolling_mean(beta_window, min_periods=beta_min_periods).over("Code").alias("x_mean"),
                    pl.col("sec_ret_1d_eq").rolling_mean(beta_window, min_periods=beta_min_periods).over("Code").alias("y_mean"),
                    (pl.col("returns_1d") * pl.col("sec_ret_1d_eq")).rolling_mean(beta_window, min_periods=beta_min_periods).over("Code").alias("xy_mean"),
                    (pl.col("sec_ret_1d_eq") ** 2).rolling_mean(beta_window, min_periods=beta_min_periods).over("Code").alias("y2_mean"),
                ]).with_columns([
                    (pl.col("xy_mean") - pl.col("x_mean") * pl.col("y_mean")).alias("cov_xy"),
                    (pl.col("y2_mean") - pl.col("y_mean")**2).alias("var_y"),
                ]).with_columns([
                    (pl.col("cov_xy") / (pl.col("var_y") + eps)).alias("beta_sec"),
                ]).with_columns([
                    (pl.col("returns_1d") - pl.col("beta_sec") * pl.col("sec_ret_1d_eq")).alias("alpha_vs_sec_1d"),
                ]).drop(["x_mean","y_mean","xy_mean","y2_mean","cov_xy","var_y"])

            # ret_1d_demeaned within (Date×sector)
            if "returns_1d" in out.columns:
                out = out.with_columns([
                    pl.col("returns_1d").mean().over(["Date", id_col]).alias("ret_1d_mean_sec")
                ]).with_columns([
                    (pl.col("returns_1d") - pl.col("ret_1d_mean_sec")).alias("ret_1d_demeaned")
                ])

            # z_in_sec for selected columns
            for x in x_cols:
                if x in out.columns:
                    mu = pl.col(x).mean().over(["Date", id_col])
                    sd = pl.col(x).std(ddof=0).over(["Date", id_col]) + eps
                    out = out.with_columns(((pl.col(x) - mu) / sd).alias(f"z_in_sec_{x}"))

            return out
        except Exception as e:
            logger.error(f"Error adding relative-to-sector features: {e}")
            return df

    def add_sector_target_encoding(
        self,
        df: pl.DataFrame,
        *,
        target_col: str = "target_5d",
        level: str = "33",
        k_folds: int = 5,
        lag_days: int = 1,
        m: float = 100.0,
    ) -> pl.DataFrame:
        """
        Target encoding by sector with cross-fit and time lag (no leakage).

        TE_g(t) = ( n_g(t;¬f) * μ_g(t;¬f) + m * μ_all(t;¬f) ) / ( n_g(t;¬f) + m )

        where statistics only use dates u ≤ t-Δ and exclude current fold f.

        Args:
            df: input dataset
            target_col: e.g., 'target_5d'
            level: '33' or '17' (sector level)
            k_folds: number of cross-fit folds
            lag_days: Δ in days (>=1)
            m: smoothing hyperparameter
        """
        try:
            if k_folds <= 1:
                k_folds = 2
            id_col = "sector33_id" if level == "33" else "sector17_id"
            if id_col not in df.columns:
                logger.warning(f"{id_col} not present; skipping sector target encoding")
                return df

            x = df
            # Ensure types
            x = x.with_columns([
                pl.col("Date").cast(pl.Date),
                pl.col("Code").cast(pl.Utf8) if "Code" in x.columns else pl.lit("").alias("Code"),
            ])

            # Ensure target exists (fallback: compute from Close if available)
            if target_col not in x.columns and "Close" in x.columns:
                if target_col == "target_5d":
                    x = x.with_columns((pl.col("Close").shift(-5).over("Code") / pl.col("Close") - 1).alias("target_5d"))
                elif target_col == "target_1d":
                    x = x.with_columns((pl.col("Close").shift(-1).over("Code") / pl.col("Close") - 1).alias("target_1d"))
                else:
                    logger.warning(f"{target_col} missing and cannot be derived; skipping TE")
                    return df

            # Build fold assignments (stable by Code)
            x = x.with_columns(
                (pl.col("Code").hash().cast(pl.UInt64) % pl.lit(k_folds)).cast(pl.Int32).alias("fold")
            )

            # Prepare keys: unique Date×sector×fold, with lookback_date = Date - lag_days
            keys = x.select(["Date", id_col, "fold"]).unique().with_columns([
                (pl.col("Date") - pl.duration(days=lag_days)).alias("lookback_date"),
                (pl.col(id_col).cast(pl.Utf8) + pl.lit("_") + pl.col("fold").cast(pl.Utf8)).alias("pair")
            ]).sort(["lookback_date"])

            # Filter valid targets
            xt = x.filter(pl.col(target_col).is_not_null())

            # Per (Date, sector, fold): sums and counts
            fold_daily = xt.group_by(["Date", id_col, "fold"]).agg([
                pl.col(target_col).sum().alias("sum_f"),
                pl.len().alias("cnt_f"),
            ]).sort([id_col, "fold", "Date"]).with_columns([
                pl.col("sum_f").cumsum().over([id_col, "fold"]).alias("cum_sum_f"),
                pl.col("cnt_f").cumsum().over([id_col, "fold"]).alias("cum_cnt_f"),
                # Pair key for join_asof by single key
                (pl.col(id_col).cast(pl.Utf8) + pl.lit("_") + pl.col("fold").cast(pl.Utf8)).alias("pair"),
            ])

            # Per (Date, sector): sums and counts across all folds
            all_daily = xt.group_by(["Date", id_col]).agg([
                pl.col(target_col).sum().alias("sum_all"),
                pl.len().alias("cnt_all"),
            ]).sort([id_col, "Date"]).with_columns([
                pl.col("sum_all").cumsum().over([id_col]).alias("cum_sum_all"),
                pl.col("cnt_all").cumsum().over([id_col]).alias("cum_cnt_all"),
            ])

            # Global per-date (all sectors, all folds)
            glob_daily = xt.group_by(["Date"]).agg([
                pl.col(target_col).sum().alias("sum_glob"),
                pl.len().alias("cnt_glob"),
            ]).sort(["Date"]).with_columns([
                pl.col("sum_glob").cumsum().alias("cum_sum_glob"),
                pl.col("cnt_glob").cumsum().alias("cum_cnt_glob"),
            ])

            # Global per-date per fold (to exclude current fold from μ_all)
            glob_fold_daily = xt.group_by(["Date", "fold"]).agg([
                pl.col(target_col).sum().alias("sum_gf"),
                pl.len().alias("cnt_gf"),
            ]).sort(["fold", "Date"]).with_columns([
                pl.col("sum_gf").cumsum().over(["fold"]).alias("cum_sum_gf"),
                pl.col("cnt_gf").cumsum().over(["fold"]).alias("cum_cnt_gf"),
            ])

            # Decide output column name with level prefix to avoid collisions
            te_col = f"te{level}_sec_{target_col}"

            # As-of join at lookback_date
            # 1) Sector-level all folds cumulative at lookback
            keys_all = keys.join_asof(
                all_daily.select([id_col, "Date", "cum_sum_all", "cum_cnt_all"]).sort([id_col, "Date"]),
                by=id_col,
                left_on="lookback_date",
                right_on="Date",
                strategy="backward",
            ).rename({"cum_sum_all": "all_sum_lag", "cum_cnt_all": "all_cnt_lag"})
            try:
                keys_all = keys_all.drop("Date_right")
            except Exception:
                pass

            # 2) Sector-level fold cumulative at lookback
            keys_all = keys_all.join_asof(
                fold_daily.select(["pair", "Date", "cum_sum_f", "cum_cnt_f"]).sort(["pair", "Date"]),
                by="pair",
                left_on="lookback_date",
                right_on="Date",
                strategy="backward",
            ).rename({"cum_sum_f": "f_sum_lag", "cum_cnt_f": "f_cnt_lag"})
            try:
                keys_all = keys_all.drop("Date_right")
            except Exception:
                pass

            # 3) Global all folds cumulative at lookback
            keys_all = keys_all.join_asof(
                glob_daily.select(["Date", "cum_sum_glob", "cum_cnt_glob"]).sort(["Date"]),
                left_on="lookback_date",
                right_on="Date",
                strategy="backward",
            ).rename({"cum_sum_glob": "glob_sum_lag", "cum_cnt_glob": "glob_cnt_lag"})
            try:
                keys_all = keys_all.drop("Date_right")
            except Exception:
                pass

            # 4) Global per fold cumulative at lookback
            keys_all = keys_all.join_asof(
                glob_fold_daily.select(["fold", "Date", "cum_sum_gf", "cum_cnt_gf"]).sort(["fold", "Date"]),
                by=["fold"],
                left_on="lookback_date",
                right_on="Date",
                strategy="backward",
            ).rename({"cum_sum_gf": "gf_sum_lag", "cum_cnt_gf": "gf_cnt_lag"})
            try:
                keys_all = keys_all.drop("Date_right")
            except Exception:
                pass

            # Compute excl-fold sector and global stats
            eps = 1e-12
            keys_all = keys_all.with_columns([
                # Sector excl fold
                (pl.col("all_sum_lag").fill_null(0.0) - pl.col("f_sum_lag").fill_null(0.0)).alias("sec_sum_excl"),
                (pl.col("all_cnt_lag").fill_null(0) - pl.col("f_cnt_lag").fill_null(0)).alias("sec_cnt_excl"),
                # Global excl fold
                (pl.col("glob_sum_lag").fill_null(0.0) - pl.col("gf_sum_lag").fill_null(0.0)).alias("glob_sum_excl"),
                (pl.col("glob_cnt_lag").fill_null(0) - pl.col("gf_cnt_lag").fill_null(0)).alias("glob_cnt_excl"),
            ]).with_columns([
                # Means with guards
                (pl.when(pl.col("sec_cnt_excl") > 0).then(pl.col("sec_sum_excl") / (pl.col("sec_cnt_excl") + eps)).otherwise(None)).alias("mu_sec_excl"),
                (pl.when(pl.col("glob_cnt_excl") > 0).then(pl.col("glob_sum_excl") / (pl.col("glob_cnt_excl") + eps)).otherwise(0.0)).alias("mu_glob_excl"),
            ])

            # Bayesian smoothing
            m_lit = pl.lit(float(m))
            keys_all = keys_all.with_columns([
                (
                    (pl.col("sec_cnt_excl").cast(pl.Float64) * pl.col("mu_sec_excl").fill_null(pl.col("mu_glob_excl")) + m_lit * pl.col("mu_glob_excl"))
                    / (pl.col("sec_cnt_excl").cast(pl.Float64) + m_lit + eps)
                ).alias(te_col)
            ])

            # Attach TE to rows via (Date, sector, fold)
            te_cols = ["Date", id_col, "fold", te_col]
            out = x.join(keys_all.select(te_cols), on=["Date", id_col, "fold"], how="left")

            # If join produced no TE (e.g., missing keys), try simple (no-fold) fallback
            need_fallback = False
            if te_col not in out.columns:
                need_fallback = True
            else:
                try:
                    nn = out.select(pl.col(te_col).is_not_null().sum()).item()
                    if nn == 0:
                        need_fallback = True
                except Exception:
                    need_fallback = True

            if not need_fallback:
                return out

            # Simple no-fold fallback: per Date×sector cumulative up to t-Δ with global smoothing
            try:
                id_col = "sector33_id" if level == "33" else "sector17_id"
                xs = x
                # keys per Date×sector
                keys = xs.select(["Date", id_col]).unique().with_columns(
                    (pl.col("Date") - pl.duration(days=lag_days)).alias("lookback_date")
                ).sort([id_col, "lookback_date"])
                # Sector daily stats and cumulative
                xt = xs.filter(pl.col(target_col).is_not_null())
                sec_daily = xt.group_by(["Date", id_col]).agg([
                    pl.col(target_col).sum().alias("sum_s"),
                    pl.len().alias("cnt_s"),
                ]).sort([id_col, "Date"]).with_columns([
                    pl.col("sum_s").cumsum().over([id_col]).alias("cum_sum_s"),
                    pl.col("cnt_s").cumsum().over([id_col]).alias("cum_cnt_s"),
                ])
                glob_daily = xt.group_by(["Date"]).agg([
                    pl.col(target_col).sum().alias("sum_g"),
                    pl.len().alias("cnt_g"),
                ]).sort(["Date"]).with_columns([
                    pl.col("sum_g").cumsum().alias("cum_sum_g"),
                    pl.col("cnt_g").cumsum().alias("cum_cnt_g"),
                ])

                keys = keys.join_asof(
                    sec_daily.select([id_col, "Date", "cum_sum_s", "cum_cnt_s"]).sort([id_col, "Date"]),
                    by=id_col,
                    left_on="lookback_date",
                    right_on="Date",
                    strategy="backward",
                ).rename({"cum_sum_s": "sec_sum_lag", "cum_cnt_s": "sec_cnt_lag"})
                try:
                    keys = keys.drop("Date_right")
                except Exception:
                    pass
                keys = keys.join_asof(
                    glob_daily.select(["Date", "cum_sum_g", "cum_cnt_g"]).sort(["Date"]),
                    left_on="lookback_date",
                    right_on="Date",
                    strategy="backward",
                ).rename({"cum_sum_g": "glob_sum_lag", "cum_cnt_g": "glob_cnt_lag"})
                try:
                    keys = keys.drop("Date_right")
                except Exception:
                    pass
                eps = 1e-12
                keys = keys.with_columns([
                    (pl.when(pl.col("sec_cnt_lag") > 0).then(pl.col("sec_sum_lag") / (pl.col("sec_cnt_lag") + eps)).otherwise(None)).alias("mu_sec_lag"),
                    (pl.when(pl.col("glob_cnt_lag") > 0).then(pl.col("glob_sum_lag") / (pl.col("glob_cnt_lag") + eps)).otherwise(0.0)).alias("mu_glob_lag"),
                ])
                m_lit = pl.lit(float(m))
                keys = keys.with_columns(
                    (
                        (pl.col("sec_cnt_lag").cast(pl.Float64).fill_null(0.0) * pl.col("mu_sec_lag").fill_null(pl.col("mu_glob_lag")) + m_lit * pl.col("mu_glob_lag"))
                        / (pl.col("sec_cnt_lag").cast(pl.Float64).fill_null(0.0) + m_lit + eps)
                    ).alias(te_col)
                )
                out2 = xs.join(keys.select(["Date", id_col, te_col]), on=["Date", id_col], how="left")
                return out2
            except Exception:
                # If even the fallback fails, return original (no TE)
                return out
        except Exception as e:
            logger.warning(f"Sector TE (cross-fit) failed: {e}; applying simple no-fold fallback")
            try:
                id_col = "sector33_id" if level == "33" else "sector17_id"
                x = df.with_columns([pl.col("Date").cast(pl.Date)])
                # ensure target exists
                if target_col not in x.columns and "Close" in x.columns:
                    if target_col == "target_5d":
                        x = x.with_columns((pl.col("Close").shift(-5).over("Code") / pl.col("Close") - 1).alias("target_5d"))
                    elif target_col == "target_1d":
                        x = x.with_columns((pl.col("Close").shift(-1).over("Code") / pl.col("Close") - 1).alias("target_1d"))
                # keys per Date×sector
                keys = x.select(["Date", id_col]).unique().with_columns(
                    (pl.col("Date") - pl.duration(days=lag_days)).alias("lookback_date")
                ).sort(["lookback_date"])
                xt = x.filter(pl.col(target_col).is_not_null())
                all_daily = xt.group_by(["Date", id_col]).agg([
                    pl.col(target_col).sum().alias("sum_s"),
                    pl.len().alias("cnt_s"),
                ]).sort([id_col, "Date"]).with_columns([
                    pl.col("sum_s").cumsum().over([id_col]).alias("cum_sum_s"),
                    pl.col("cnt_s").cumsum().over([id_col]).alias("cum_cnt_s"),
                ])
                glob_daily = xt.group_by(["Date"]).agg([
                    pl.col(target_col).sum().alias("sum_g"),
                    pl.len().alias("cnt_g"),
                ]).sort(["Date"]).with_columns([
                    pl.col("sum_g").cumsum().alias("cum_sum_g"),
                    pl.col("cnt_g").cumsum().alias("cum_cnt_g"),
                ])
                keys_all = keys.join_asof(
                    all_daily.select([id_col, "Date", "cum_sum_s", "cum_cnt_s"]).sort([id_col, "Date"]),
                    by=id_col,
                    left_on="lookback_date",
                    right_on="Date",
                    strategy="backward",
                ).rename({"cum_sum_s": "sec_sum_lag", "cum_cnt_s": "sec_cnt_lag"})
                try:
                    keys_all = keys_all.drop("Date_right")
                except Exception:
                    pass
                keys_all = keys_all.join_asof(
                    glob_daily.select(["Date", "cum_sum_g", "cum_cnt_g"]).sort(["Date"]),
                    left_on="lookback_date",
                    right_on="Date",
                    strategy="backward",
                ).rename({"cum_sum_g": "glob_sum_lag", "cum_cnt_g": "glob_cnt_lag"})
                try:
                    keys_all = keys_all.drop("Date_right")
                except Exception:
                    pass
                eps = 1e-12
                keys_all = keys_all.with_columns([
                    (pl.when(pl.col("sec_cnt_lag") > 0).then(pl.col("sec_sum_lag") / (pl.col("sec_cnt_lag") + eps)).otherwise(None)).alias("mu_sec"),
                    (pl.when(pl.col("glob_cnt_lag") > 0).then(pl.col("glob_sum_lag") / (pl.col("glob_cnt_lag") + eps)).otherwise(0.0)).alias("mu_glob"),
                ])
                m_lit = pl.lit(float(m))
                keys_all = keys_all.with_columns([
                    (
                        (pl.col("sec_cnt_lag").cast(pl.Float64) * pl.col("mu_sec").fill_null(pl.col("mu_glob")) + m_lit * pl.col("mu_glob"))
                        / (pl.col("sec_cnt_lag").cast(pl.Float64) + m_lit + eps)
                    ).alias(te_col)
                ])
                out = x.join(keys_all.select(["Date", id_col, te_col]), on=["Date", id_col], how="left")
                return out
            except Exception as e2:
                logger.error(f"Sector TE fallback failed: {e2}")
                return df

    def create_metadata(self, df: pl.DataFrame) -> dict:
        """Create dataset metadata."""
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
            "version": "gogooku3-standalone-enhanced",
            "data_source": "gogooku2-batch-ta-data",
        }

        return metadata

    def save_dataset(self, df: pl.DataFrame, metadata: dict):
        """Save dataset in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as Parquet (recommended)
        parquet_path = self.output_dir / f"ml_dataset_{timestamp}.parquet"
        df.write_parquet(parquet_path, compression="snappy")
        logger.info(f"Saved Parquet: {parquet_path}")

        # Save metadata
        meta_path = self.output_dir / f"ml_dataset_{timestamp}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {meta_path}")

        # Create symlinks to latest
        latest_parquet = self.output_dir / "ml_dataset_latest.parquet"
        latest_meta = self.output_dir / "ml_dataset_latest_metadata.json"

        for latest, current in [
            (latest_parquet, parquet_path),
            (latest_meta, meta_path),
        ]:
            if latest.exists():
                latest.unlink()
            latest.symlink_to(current.name)

        return parquet_path, meta_path

    # ===== Spec finalization =====
    def finalize_for_spec(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply final, non-breaking transformations so the output matches
        docs/DATASET.md: drop helper columns, normalize flag names, and
        attach 5-digit LocalCode while retaining 4-digit Code.
        """
        # 1) Drop helper beta columns not in the canonical list
        helper_betas = ["beta_60d_raw", "beta_20d_raw", "beta_rolling"]
        to_drop = [c for c in helper_betas if c in df.columns]
        if to_drop:
            df = df.drop(to_drop)

        # 2) Normalize validity flag names (align with DATASET.md)
        try:
            rename_map = {
                "is_ema_5_valid": "is_ema5_valid",
                "is_ema_10_valid": "is_ema10_valid",
                "is_ema_20_valid": "is_ema20_valid",
                "is_ema_60_valid": "is_ema60_valid",
                "is_ema_200_valid": "is_ema200_valid",
                "is_rsi_2_valid": "is_rsi2_valid",
            }
            present = {k: v for k, v in rename_map.items() if k in df.columns}
            if present:
                df = df.rename(present)
        except Exception:
            pass

        # 3) Ensure LocalCode (5-digit) exists
        if "LocalCode" not in df.columns and "Code" in df.columns:
            df = df.with_columns(
                pl.col("Code").cast(pl.Utf8).str.zfill(5).alias("LocalCode")
            )

        return df

    def build_enhanced_dataset(self) -> dict | None:
        """Build enhanced ML dataset with more stocks from gogooku2 data."""
        logger.info("🚀 Building enhanced ML dataset with gogooku2 data...")

        # 1. Load gogooku2 data
        df = self.load_gogooku2_data()
        if df is None:
            logger.error("Failed to load gogooku2 data")
            return None

        # 2. Apply quality filters (preserve more stocks)
        df = self.filter_stocks_by_quality(df, min_data_points=100)  # Lower threshold

        # 3. Create technical features
        df = self.create_technical_features(df)

        # 4. Add pandas-ta features
        df = self.add_pandas_ta_features(df)

        # 5. Create metadata
        metadata = self.create_metadata(df)

        # 6. Save dataset
        parquet_path, meta_path = self.save_dataset(df, metadata)

        logger.info("✅ Enhanced ML dataset creation completed!")
        logger.info(f"Final dataset: {len(df):,} rows, {metadata['stocks']} stocks")
        logger.info(f"Features: {metadata['features']['count']}")
        logger.info(f"Files: {parquet_path}, {meta_path}")

        return {
            "df": df,
            "metadata": metadata,
            "parquet_path": parquet_path,
            "meta_path": meta_path
        }

    # ===== TOPIX関連メソッド =====

    async def fetch_topix_data_async(self, start_date: str, end_date: str) -> pl.DataFrame:
        """
        JQuants APIからTOPIXデータを非同期で取得

        Args:
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)

        Returns:
            TOPIXデータ (Date, Close, Volume列)
        """
        try:
            # JQuants API設定（環境変数から取得）
            base_url = "https://api.jquants.com/v1"
            refresh_token = os.getenv("JQUANTS_REFRESH_TOKEN")

            if not refresh_token:
                logger.warning("JQUANTS_REFRESH_TOKEN not found, using sample TOPIX data")
                return self._create_sample_topix_data(start_date, end_date)

            # IDトークン取得
            id_token = await self._get_jquants_token(base_url, refresh_token)

            # TOPIXデータ取得
            url = f"{base_url}/indices/topix"
            headers = {"Authorization": f"Bearer {id_token}"}

            async with aiohttp.ClientSession() as session:
                all_data = []
                pagination_key = None

                while True:
                    params = {"from": start_date, "to": end_date}
                    if pagination_key:
                        params["pagination_key"] = pagination_key

                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status != 200:
                            logger.error(f"TOPIX API error: {response.status}")
                            break

                        data = await response.json()
                        topix_data = data.get("topix", [])

                        if topix_data:
                            all_data.extend(topix_data)

                        # ページネーション確認
                        pagination_key = data.get("pagination_key")
                        if not pagination_key:
                            break

                if all_data:
                    df = pl.DataFrame(all_data)
                    logger.info(f"Fetched {len(df)} TOPIX records from {start_date} to {end_date}")
                    return df
                else:
                    logger.warning("No TOPIX data fetched, using sample data")
                    return self._create_sample_topix_data(start_date, end_date)

        except Exception as e:
            logger.error(f"Error fetching TOPIX data: {e}")
            return self._create_sample_topix_data(start_date, end_date)

    async def _get_jquants_token(self, base_url: str, refresh_token: str) -> str:
        """JQuants APIのIDトークンを取得"""
        url = f"{base_url}/token/auth_refresh"
        headers = {"Authorization": f"Bearer {refresh_token}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("idToken", "")
                else:
                    raise Exception(f"Token refresh failed: {response.status}")

    def _create_sample_topix_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """サンプルTOPIXデータを作成（APIが利用できない場合のフォールバック）"""
        logger.info("Creating sample TOPIX data")

        # 日付範囲を作成
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='D')

        # TOPIXのサンプルデータ生成（実際のTOPIXを近似）
        np.random.seed(42)
        n_days = len(dates)
        base_price = 2000.0

        # ランダムウォークで価格生成
        returns = np.random.normal(0.0002, 0.015, n_days)  # 平均0.02%、ボラ15%
        prices = base_price * np.exp(np.cumsum(returns))

        # サンプルデータ作成
        sample_data = []
        for i, date in enumerate(dates):
            if date.weekday() < 5:  # 月-金のみ
                sample_data.append({
                    "Date": date.strftime("%Y-%m-%d"),
                    "Close": round(prices[i], 2),
                    "Volume": np.random.randint(1000000, 5000000)
                })

        return pl.DataFrame(sample_data)

    def add_topix_features(self, df: pl.DataFrame, topix_df: pl.DataFrame | None = None, *, beta_lag: int = 1) -> pl.DataFrame:
        """
        TOPIX市場特徴量をデータセットに追加

        Args:
            df: 銘柄データ (Code, date, Close, return_1d等を含む)
            topix_df: TOPIXデータ (オプション、省略時は自動取得)

        Returns:
            TOPIX特徴量を追加したDataFrame
        """
        try:
            # Try different import paths
            try:
                from src.features.market_features import (
                    CrossMarketFeaturesGenerator,
                    MarketFeaturesGenerator,
                )
            except ImportError:
                # Add parent directories to path and retry
                import sys
                from pathlib import Path
                root_dir = Path(__file__).parent.parent.parent
                if str(root_dir) not in sys.path:
                    sys.path.insert(0, str(root_dir))
                from src.features.market_features import (
                    CrossMarketFeaturesGenerator,
                    MarketFeaturesGenerator,
                )

            logger.info("Adding TOPIX market features to dataset")

            # TOPIXデータが提供されていない場合は取得
            if topix_df is None:
                logger.info("TOPIX data not provided, fetching from API...")
                # 非同期関数を同期的に実行
                async def fetch_and_process():
                    # データセットの日付範囲を取得（列名の大小差異に対応）
                    date_col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
                    if date_col is None:
                        raise KeyError("Date column not found in dataset")
                    date_range = df.select([
                        pl.col(date_col).min().alias("start"),
                        pl.col(date_col).max().alias("end")
                    ]).to_dicts()[0]

                    start_date = date_range["start"].strftime("%Y-%m-%d")
                    end_date = date_range["end"].strftime("%Y-%m-%d")

                    topix_data = await self.fetch_topix_data_async(start_date, end_date)
                    return topix_data

                # 非同期実行
                topix_df = asyncio.run(fetch_and_process())

            if topix_df is None or len(topix_df) == 0:
                logger.warning("No TOPIX data available, skipping market features")
                return df

            # TOPIX市場特徴量の生成
            market_generator = MarketFeaturesGenerator()
            market_features_df = market_generator.build_topix_features(topix_df)

            # 銘柄データと市場特徴量の統合
            cross_generator = CrossMarketFeaturesGenerator(beta_lag=beta_lag)
            enhanced_df = cross_generator.attach_market_and_cross(df, market_features_df)

            # 統合結果の検証
            market_cols = [col for col in enhanced_df.columns if col.startswith(('mkt_', 'beta_', 'alpha_', 'rel_'))]
            logger.info(f"Added {len(market_cols)} TOPIX market features: {market_cols[:5]}...")

            return enhanced_df

        except ImportError as e:
            logger.warning(f"Market features module not available: {e}")
            logger.warning("Skipping TOPIX features addition")
            return df
        except Exception as e:
            logger.error(f"Error adding TOPIX features: {e}")
            return df

    def build_enhanced_dataset_with_market(self) -> dict | None:
        """
        TOPIX市場特徴量とtrade-specフロー特徴量を含む拡張データセットを作成

        Returns:
            拡張データセットの情報
        """
        logger.info("Building enhanced dataset with TOPIX market features and trade-spec flow features")

        try:
            # 1. 基本データセット作成
            base_result = self.build_enhanced_dataset()
            if not base_result:
                return None

            df = base_result["df"]

            # 2. TOPIX特徴量の追加
            enhanced_df = self.add_topix_features(df)

            # 3. Trade-spec（フロー）特徴量の追加
            # Trade-specデータの読み込み（gogooku2から）
            trades_spec_path = self.gogooku2_data_path / "20250824/trades_spec/tse_trades_spec_20250824_175808.parquet"
            if trades_spec_path.exists():
                logger.info(f"Loading trades_spec from: {trades_spec_path}")
                trades_spec_df = pl.read_parquet(trades_spec_path)

                # Listed info の読み込み（必要に応じて）
                listed_info_df = None  # TODO: listed_info の実装時に更新

                # フロー特徴量の追加
                enhanced_df = self.add_flow_features(enhanced_df, trades_spec_df, listed_info_df)
            else:
                logger.warning(f"Trade-spec file not found: {trades_spec_path}")

            # 4. 財務諸表特徴量の追加
            # 財務諸表データの読み込み（保存済みまたはAPI取得）
            statements_path = self.output_dir / "event_raw_statements.parquet"
            if not statements_path.exists():
                # 最新の保存ファイルを探す
                statements_files = list(self.output_dir.glob("event_raw_statements_*.parquet"))
                if statements_files:
                    statements_path = max(statements_files, key=lambda x: x.stat().st_mtime)

            if statements_path.exists():
                logger.info(f"Loading financial statements from: {statements_path}")
                statements_df = pl.read_parquet(statements_path)

                # 財務諸表特徴量の追加
                enhanced_df = self.add_statements_features(enhanced_df, statements_df)
            else:
                logger.warning(f"Financial statements file not found: {statements_path}")

            # 4. メタデータの更新
            metadata = base_result["metadata"]
            metadata["features"]["market_features"] = {
                "count": len([col for col in enhanced_df.columns if col.startswith(('mkt_', 'beta_', 'alpha_', 'rel_'))]),
                "categories": ["market_trend", "market_volatility", "cross_market", "regime_flags"]
            }

            # フロー特徴量のメタデータ追加
            flow_feature_count = len([col for col in enhanced_df.columns if col.startswith((
                'foreigners_', 'individuals_', 'smart_money', 'activity_z',
                'foreign_share', 'breadth_', 'flow_', 'days_since'
            ))])
            metadata["features"]["flow_features"] = {
                "count": flow_feature_count,
                "categories": ["investor_flow", "smart_money", "market_breadth", "flow_timing"]
            }

            # 財務諸表特徴量のメタデータ追加
            stmt_feature_count = len([col for col in enhanced_df.columns if col.startswith("stmt_")])
            metadata["features"]["statement_features"] = {
                "count": stmt_feature_count,
                "categories": ["yoy_growth", "margins", "guidance_progress", "guidance_revision"]
            }

            # 4. 拡張データセットの保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parquet_path = self.output_dir / f"ml_dataset_enhanced_with_market_{timestamp}.parquet"
            meta_path = self.output_dir / f"ml_dataset_enhanced_with_market_{timestamp}_meta.json"

            # Parquet保存
            enhanced_df.write_parquet(parquet_path)

            # メタデータ保存
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info("✅ Enhanced dataset with market features completed!")
            logger.info(f"Market features added: {metadata['features']['market_features']['count']}")
            logger.info(f"Files: {parquet_path}, {meta_path}")

            return {
                "df": enhanced_df,
                "metadata": metadata,
                "parquet_path": parquet_path,
                "meta_path": meta_path
            }

        except Exception as e:
            logger.error(f"Error building enhanced dataset with market features: {e}")
            return None

    def add_statements_features(
        self,
        df: pl.DataFrame,
        statements_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        財務諸表特徴量を追加（T+1 as-of結合）

        Args:
            df: 価格データ
            statements_df: 財務諸表データ

        Returns:
            財務諸表特徴量を含むデータフレーム
        """
        try:
            # Add src/features to path for imports
            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            from features.safe_joiner import SafeJoiner
            from utils.dtypes import ensure_date

            if statements_df is None or statements_df.is_empty():
                logger.warning("No statements data provided, skipping statement features")
                return df

            logger.info("Adding financial statement features with T+1 as-of join...")

            # 型変換：Date列を確実に日付型に
            if df.schema.get("Date") == pl.Utf8:
                df = df.with_columns([
                    pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
                ])
            else:
                df = ensure_date(df, "Date")

            # SafeJoinerを使用して財務諸表を結合
            joiner = SafeJoiner()
            df_with_statements = joiner.join_statements_asof(
                base_df=df,
                statements_df=statements_df,
                use_time_cutoff=True,  # 15:00判定を使用
                cutoff_time=dt_time(15, 0)
            )

            # カバレッジ統計
            if "is_stmt_valid" in df_with_statements.columns:
                coverage = (df_with_statements["is_stmt_valid"] == 1).sum() / len(df_with_statements) if len(df_with_statements) > 0 else 0
                logger.info(f"✅ Statement feature coverage: {coverage:.1%}")

            # 追加された特徴量のログ
            stmt_cols = [
                col for col in df_with_statements.columns
                if col.startswith("stmt_")
            ]
            logger.info(f"✅ Added {len(stmt_cols)} statement features: {stmt_cols[:8]}...")

            return df_with_statements

        except Exception as e:
            logger.error(f"Error adding statement features: {e}")
            logger.warning("Continuing without statement features")
            return df

    def add_flow_features(
        self,
        df: pl.DataFrame,
        trades_spec_df: pl.DataFrame | None = None,
        listed_info_df: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        """
        週次フローイベント特徴量を追加（改良版: Section×Date結合）

        Args:
            df: 銘柄日次データ
            trades_spec_df: trades_spec（売買内訳）データ
            listed_info_df: listed_info（銘柄情報）データ

        Returns:
            フロー特徴量を含むデータフレーム
        """
        try:
            # Add src/features to path for imports
            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            # NOTE: TradingCalendarFetcher is not required here; we use a simple
            # next business day helper and local business-day generation.

            from features.flow_joiner import add_flow_features as generate_flow_features
            from features.flow_joiner import (
                attach_flow_to_quotes,
                build_flow_intervals,
                expand_flow_daily,
            )
            from features.section_mapper import SectionMapper

            if trades_spec_df is None or trades_spec_df.is_empty():
                logger.warning("No trades_spec data provided, skipping flow features")
                return df

            logger.info("Adding flow features using Section×Date join strategy...")

            # 1) Section付与（listed_info → MarketCode → Section）: as-of backward + 区間ガード
            if listed_info_df is not None and not listed_info_df.is_empty():
                section_mapper = SectionMapper()
                section_intervals = section_mapper.create_section_mapping(listed_info_df)
                # 型とソート
                q = df.sort(["Code", "Date"]).with_columns([
                    pl.col("Code").cast(pl.Utf8),
                    pl.col("Date").cast(pl.Date),
                ])
                sec = section_intervals.sort(["Code", "valid_from"]).with_columns([
                    pl.col("Code").cast(pl.Utf8),
                    pl.col("valid_from").cast(pl.Date),
                    pl.col("valid_to").cast(pl.Date),
                    pl.col("Section").map_elements(
                        lambda s: (
                            "TSEPrime" if s in ("TSE1st","Prime","Prime Market","TSE Prime","東証プライム") else
                            "TSEStandard" if s in ("TSE2nd","Standard","Standard Market","TSE Standard","JASDAQ","JASDAQ Standard","Other","東証スタンダード") else
                            "TSEGrowth" if s in ("Mothers","Growth","Growth Market","TSE Growth","JASDAQ Growth","東証グロース") else s
                        ), return_dtype=pl.Utf8
                    ).alias("Section"),
                ])
                # as-of backward で区間開始に対して結合
                q = q.join_asof(
                    sec.select(["Code", "Section", "valid_from", "valid_to"]),
                    by="Code",
                    left_on="Date",
                    right_on="valid_from",
                    strategy="backward",
                )
                # 区間内ガード
                q = q.filter((pl.col("valid_to").is_null()) | (pl.col("Date") <= pl.col("valid_to")))
                df = q.drop([c for c in ["valid_from", "valid_to"] if c in q.columns])
                logger.info(f"Attached Section (as-of) to {df['Section'].is_not_null().sum()} records")
            else:
                # listed_infoがない場合はデフォルトSectionを付与
                df = df.with_columns(
                    pl.lit("AllMarket").alias("Section")
                )
                logger.warning("No listed_info provided, using 'AllMarket' as default Section")

            # 2) 営業日リストを取得（データの日付範囲から）
            date_min = df["Date"].min()
            date_max = df["Date"].max()

            # next_bd関数の定義（簡易版）
            def next_bd(d):
                """翌営業日を返す（週末をスキップ）"""
                from datetime import timedelta
                next_day = d + timedelta(days=1)
                while next_day.weekday() >= 5:  # 土日をスキップ
                    next_day += timedelta(days=1)
                return next_day

            # 3) trades_spec → 有効区間テーブル
            flow_intervals = build_flow_intervals(trades_spec_df, next_bd)
            logger.info(f"Built {len(flow_intervals)} flow intervals")

            # 4) フロー特徴量生成
            flow_feat = generate_flow_features(flow_intervals)
            logger.info(f"Generated flow features: {flow_feat.columns}")

            # 5) 区間→日次展開
            # 営業日リストの生成（簡易版: 週末を除く）
            business_days = []
            current = date_min
            while current <= date_max:
                if current.weekday() < 5:  # 平日のみ
                    business_days.append(current)
                current += timedelta(days=1)

            flow_daily = expand_flow_daily(flow_feat, business_days)
            logger.info(f"Expanded to {len(flow_daily)} daily flow records")

            # 6) 価格×フロー結合（Section一致）
            df_with_flow = attach_flow_to_quotes(df, flow_daily, "Section")

            # カバレッジ統計
            coverage = (df_with_flow["is_flow_valid"] == 1).sum() / len(df_with_flow) if len(df_with_flow) > 0 else 0
            logger.info(f"✅ Flow feature coverage: {coverage:.1%}")

            # フォールバック: Section一致でカバレッジが極端に低い場合は、AllMarketで再結合
            if coverage < 0.01:
                logger.warning("Flow coverage < 1%. Falling back to AllMarket aggregation join (Date-only)")
                # すべてのsectionをAllMarketに正規化
                flow_daily_all = flow_daily.with_columns([
                    pl.lit("AllMarket").alias("section")
                ])
                df_with_flow = attach_flow_to_quotes(
                    df.with_columns(pl.lit("AllMarket").alias("Section")),
                    flow_daily_all,
                    "Section",
                )
                coverage2 = (df_with_flow["is_flow_valid"] == 1).sum() / len(df_with_flow) if len(df_with_flow) > 0 else 0
                logger.info(f"✅ Flow feature coverage (fallback): {coverage2:.1%}")

            # 追加された特徴量のログ
            flow_cols = [
                col for col in df_with_flow.columns
                if col.startswith(("foreigners_", "individuals_", "smart_money",
                                 "activity_z", "foreign_share", "breadth_", "flow_"))
            ]
            logger.info(f"✅ Added {len(flow_cols)} flow features: {flow_cols[:8]}...")

            return df_with_flow

        except Exception as e:
            logger.error(f"Error adding flow features: {e}")
            logger.warning("Continuing without flow features")
            return df


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Enhanced ML Dataset Builder for gogooku3-standalone")
    logger.info("=" * 60)

    # Create builder
    builder = MLDatasetBuilder()

    # Build enhanced dataset with TOPIX market features
    result = builder.build_enhanced_dataset_with_market()

    # Fallback to basic dataset if market features fail
    if not result:
        logger.info("Falling back to basic dataset...")
        result = builder.build_enhanced_dataset()

    if result:
        logger.info("\n" + "=" * 60)
        logger.info("ENHANCED DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Shape: {len(result['df'])} rows × {len(result['df'].columns)} columns")
        logger.info(f"Features: {result['metadata']['features']['count']}")
        logger.info(f"Stocks: {result['metadata']['stocks']}")
        logger.info(f"Date range: {result['metadata']['date_range']['start']} to {result['metadata']['date_range']['end']}")
        logger.info(f"Memory usage: {result['df'].estimated_size('mb'):.2f} MB")
        logger.info("=" * 60)
        logger.info("COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        return result
    else:
        logger.error("Failed to create enhanced dataset")
        return None


if __name__ == "__main__":
    result = main()
