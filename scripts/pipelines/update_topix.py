#!/usr/bin/env python3
"""
TOPIX Feature Updater
既存のデータセットにTOPIX相対特徴量を追加/更新するスクリプト
"""

import os
import sys
import asyncio
import aiohttp
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import logging
import argparse
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

sys.path.append(str(Path(__file__).parent.parent))
from pipelines.run_pipeline import JQuantsAsyncFetcher  # noqa: E402
from core.ml_dataset_builder import MLDatasetBuilder  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TopixUpdater:
    """TOPIX特徴量の更新専用クラス"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(
            "/home/ubuntu/gogooku2/apps/gogooku3/output"
        )
        self.builder = MLDatasetBuilder(output_dir=self.output_dir)

    async def fetch_topix_only(self, start_date: str, end_date: str) -> pl.DataFrame:
        """TOPIXデータのみを取得"""
        email = os.getenv("JQUANTS_AUTH_EMAIL", "")
        password = os.getenv("JQUANTS_AUTH_PASSWORD", "")

        if not email or not password:
            logger.error("JQuants credentials not found")
            return pl.DataFrame()

        fetcher = JQuantsAsyncFetcher(email, password)

        async with aiohttp.ClientSession() as session:
            # Authenticate
            await fetcher.authenticate(session)

            # Fetch TOPIX data only
            logger.info(f"Fetching TOPIX data from {start_date} to {end_date}")
            topix_df = await fetcher.fetch_topix_data(session, start_date, end_date)

            if not topix_df.is_empty():
                logger.info(f"✅ Fetched {len(topix_df)} TOPIX records")

            return topix_df

    def update_dataset_with_topix(
        self, dataset_path: str, topix_df: pl.DataFrame
    ) -> pl.DataFrame:
        """既存データセットにTOPIX特徴量を追加"""
        logger.info(f"Loading existing dataset: {dataset_path}")

        # Load existing dataset
        if dataset_path.endswith(".parquet"):
            df = pl.read_parquet(dataset_path)
        elif dataset_path.endswith(".csv"):
            df = pl.read_csv(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

        logger.info(f"Dataset shape before: {df.shape}")

        # Remove existing TOPIX features if present
        topix_features = [
            "alpha_1d",
            "alpha_5d",
            "alpha_10d",
            "alpha_20d",
            "relative_strength_1d",
            "relative_strength_5d",
            "market_regime",
        ]
        existing_topix = [col for col in topix_features if col in df.columns]

        if existing_topix:
            logger.info(f"Removing existing TOPIX features: {existing_topix}")
            df = df.drop(existing_topix)

        # Add TOPIX features
        df = self.builder.add_topix_features(df, topix_df)
        # Finalize to match dataset_new.md spec
        try:
            df = self.builder.finalize_for_spec(df)
        except Exception:
            pass

        logger.info(f"Dataset shape after: {df.shape}")
        return df

    def save_updated_dataset(
        self, df: pl.DataFrame, original_path: str, suffix: str = "_with_topix"
    ):
        """更新されたデータセットを保存"""
        # Generate new filename
        orig_path = Path(original_path)
        new_name = orig_path.stem + suffix + orig_path.suffix
        new_path = orig_path.parent / new_name

        # Save based on format
        if orig_path.suffix == ".parquet":
            df.write_parquet(new_path)
            logger.info(f"Saved updated dataset: {new_path}")
        elif orig_path.suffix == ".csv":
            df.write_csv(new_path)
            logger.info(f"Saved updated dataset: {new_path}")

        # Also save as latest
        latest_parquet = self.output_dir / "ml_dataset_latest_with_topix.parquet"
        df.write_parquet(latest_parquet)
        logger.info(f"Saved as latest: {latest_parquet}")

        # Create metadata
        metadata = self.builder.create_metadata(df)
        metadata["topix_update"] = {
            "updated_at": datetime.now().isoformat(),
            "original_dataset": str(original_path),
            "topix_features_added": True,
        }

        import json

        meta_path = self.output_dir / "ml_dataset_latest_with_topix_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata: {meta_path}")

        return new_path


async def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Update existing dataset with TOPIX features"
    )

    parser.add_argument(
        "--dataset", type=str, help="Path to existing dataset (parquet or csv)"
    )
    parser.add_argument(
        "--latest", action="store_true", help="Use latest dataset from output directory"
    )
    parser.add_argument(
        "--start-date", type=str, help="Start date for TOPIX data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, help="End date for TOPIX data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days to fetch (if dates not specified)",
    )

    args = parser.parse_args()

    # Determine dataset path
    updater = TopixUpdater()

    if args.latest:
        # Find latest dataset
        dataset_path = updater.output_dir / "ml_dataset_latest.parquet"
        if not dataset_path.exists():
            # Try to find any recent dataset
            parquet_files = list(updater.output_dir.glob("ml_dataset_*.parquet"))
            if parquet_files:
                dataset_path = max(parquet_files, key=lambda x: x.stat().st_mtime)
            else:
                logger.error("No dataset found in output directory")
                return
    elif args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            return
    else:
        logger.error("Please specify --dataset or --latest")
        return

    logger.info(f"Using dataset: {dataset_path}")

    # Determine date range
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        # Use default date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    # Fetch TOPIX data
    logger.info("=" * 60)
    logger.info("TOPIX FEATURE UPDATE PIPELINE")
    logger.info("=" * 60)

    topix_df = await updater.fetch_topix_only(start_date, end_date)

    if topix_df.is_empty():
        logger.error("Failed to fetch TOPIX data")
        return

    # Update dataset
    updated_df = updater.update_dataset_with_topix(str(dataset_path), topix_df)

    # Save updated dataset
    new_path = updater.save_updated_dataset(updated_df, str(dataset_path))

    logger.info("=" * 60)
    logger.info("UPDATE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Original dataset: {dataset_path}")
    logger.info(f"Updated dataset: {new_path}")
    logger.info("TOPIX features added: 7")

    # Display sample of new features
    if "alpha_1d" in updated_df.columns:
        logger.info("\nSample of TOPIX features:")
        topix_cols = [
            "Date",
            "Code",
            "alpha_1d",
            "alpha_5d",
            "relative_strength_1d",
            "market_regime",
        ]
        available_cols = [col for col in topix_cols if col in updated_df.columns]
        sample = updated_df.select(available_cols).head(5)
        logger.info(f"\n{sample}")


if __name__ == "__main__":
    asyncio.run(main())
