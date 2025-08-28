#!/usr/bin/env python3
"""
Modular Data Updater for JQuants API
各APIエンドポイントを独立して更新可能なモジュール化システム
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
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

sys.path.append(str(Path(__file__).parent.parent))
from pipelines.run_pipeline import JQuantsAsyncFetcher  # noqa: E402
from core.ml_dataset_builder import MLDatasetBuilder  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataComponent:
    """Base class for data components"""

    def __init__(self, name: str, endpoint: str, required_cols: List[str]):
        self.name = name
        self.endpoint = endpoint
        self.required_cols = required_cols

    async def fetch(
        self,
        fetcher: JQuantsAsyncFetcher,
        session: aiohttp.ClientSession,
        params: Dict[str, Any],
    ) -> pl.DataFrame:
        """Fetch data from API endpoint"""
        raise NotImplementedError

    def process(self, df: pl.DataFrame, component_df: pl.DataFrame) -> pl.DataFrame:
        """Process and merge component data with main dataframe"""
        raise NotImplementedError

    def validate(self, df: pl.DataFrame) -> bool:
        """Validate component data"""
        return all(col in df.columns for col in self.required_cols)


class PriceDataComponent(DataComponent):
    """Daily price data component"""

    def __init__(self):
        super().__init__(
            name="prices",
            endpoint="/prices/daily_quotes",
            required_cols=["Code", "Date", "Open", "High", "Low", "Close", "Volume"],
        )

    async def fetch(
        self,
        fetcher: JQuantsAsyncFetcher,
        session: aiohttp.ClientSession,
        params: Dict[str, Any],
    ) -> pl.DataFrame:
        """Fetch price data for specified stocks"""
        codes = params.get("codes", [])
        from_date = params.get("from", "")
        to_date = params.get("to", "")

        if not codes:
            logger.error("No stock codes provided")
            return pl.DataFrame()

        return await fetcher.fetch_all_prices(session, codes, from_date, to_date)


class TopixComponent(DataComponent):
    """TOPIX index data component"""

    def __init__(self):
        super().__init__(
            name="topix", endpoint="/indices/topix", required_cols=["Date", "Close"]
        )

    async def fetch(
        self,
        fetcher: JQuantsAsyncFetcher,
        session: aiohttp.ClientSession,
        params: Dict[str, Any],
    ) -> pl.DataFrame:
        """Fetch TOPIX index data"""
        from_date = params.get("from", "")
        to_date = params.get("to", "")

        return await fetcher.fetch_topix_data(session, from_date, to_date)

    def process(self, df: pl.DataFrame, topix_df: pl.DataFrame) -> pl.DataFrame:
        """Add TOPIX-relative features"""
        builder = MLDatasetBuilder()
        return builder.add_topix_features(df, topix_df)


class TradesSpecComponent(DataComponent):
    """Trades specification component"""

    def __init__(self):
        super().__init__(
            name="trades_spec",
            endpoint="/markets/trades_spec",
            required_cols=["Code", "Date", "Section", "TradingUnit"],
        )

    async def fetch(
        self,
        fetcher: JQuantsAsyncFetcher,
        session: aiohttp.ClientSession,
        params: Dict[str, Any],
    ) -> pl.DataFrame:
        """Fetch trades specification data"""
        from_date = params.get("from", "")
        to_date = params.get("to", "")

        return await fetcher.get_trades_spec(session, from_date, to_date)

    def process(self, df: pl.DataFrame, spec_df: pl.DataFrame) -> pl.DataFrame:
        """Add trades specification features"""
        # Join trades spec data
        if not spec_df.is_empty():
            # Ensure date types match
            if df["Date"].dtype != spec_df["Date"].dtype:
                spec_df = spec_df.with_columns(pl.col("Date").cast(df["Date"].dtype))

            # Join on Code and Date
            df = df.join(
                spec_df.select(["Code", "Date", "Section", "TradingUnit"]),
                on=["Code", "Date"],
                how="left",
            )

            # Create features
            df = df.with_columns(
                [
                    # Section as categorical
                    pl.col("Section").cast(pl.Categorical).alias("market_section"),
                    # Trading unit as feature
                    pl.col("TradingUnit").alias("trading_unit"),
                    # Flag for unit changes
                    (
                        pl.col("TradingUnit").shift(1).over("Code")
                        != pl.col("TradingUnit")
                    )
                    .cast(pl.Int8)
                    .alias("trading_unit_changed"),
                ]
            )

            logger.info("✅ Added trades specification features")

        return df


class ListedInfoComponent(DataComponent):
    """Listed company information component"""

    def __init__(self):
        super().__init__(
            name="listed_info",
            endpoint="/listed/info",
            required_cols=["Code", "CompanyName", "MarketCode", "Sector33Code"],
        )

    async def fetch(
        self,
        fetcher: JQuantsAsyncFetcher,
        session: aiohttp.ClientSession,
        params: Dict[str, Any],
    ) -> pl.DataFrame:
        """Fetch listed company information"""
        return await fetcher.get_listed_info(session)

    def process(self, df: pl.DataFrame, info_df: pl.DataFrame) -> pl.DataFrame:
        """Add company information features"""
        if not info_df.is_empty():
            # Join company info
            df = df.join(
                info_df.select(["Code", "MarketCode", "Sector33Code"]),
                on="Code",
                how="left",
            )

            # Create features
            df = df.with_columns(
                [
                    # Market type encoding
                    pl.col("MarketCode").cast(pl.Categorical).alias("market_type"),
                    # Sector encoding
                    pl.col("Sector33Code").cast(pl.Categorical).alias("sector_code"),
                    # Is TSE Prime
                    (pl.col("MarketCode") == "111").cast(pl.Int8).alias("is_tse_prime"),
                ]
            )

            logger.info("✅ Added company information features")

        return df


class ModularDataUpdater:
    """Modular data update system"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(
            "/home/ubuntu/gogooku2/apps/gogooku3/output"
        )
        self.output_dir.mkdir(exist_ok=True)

        # Register components
        self.components = {
            "prices": PriceDataComponent(),
            "topix": TopixComponent(),
            "trades_spec": TradesSpecComponent(),
            "listed_info": ListedInfoComponent(),
        }

        self.builder = MLDatasetBuilder(output_dir=self.output_dir)

    async def fetch_component(
        self, component_name: str, params: Dict[str, Any]
    ) -> pl.DataFrame:
        """Fetch data for a specific component"""
        if component_name not in self.components:
            logger.error(f"Unknown component: {component_name}")
            return pl.DataFrame()

        component = self.components[component_name]

        # Get credentials
        email = os.getenv("JQUANTS_AUTH_EMAIL", "")
        password = os.getenv("JQUANTS_AUTH_PASSWORD", "")

        if not email or not password:
            logger.error("JQuants credentials not found")
            return pl.DataFrame()

        fetcher = JQuantsAsyncFetcher(email, password)

        async with aiohttp.ClientSession() as session:
            await fetcher.authenticate(session)

            logger.info(f"Fetching {component_name} data...")
            data = await component.fetch(fetcher, session, params)

            if not data.is_empty():
                logger.info(f"✅ Fetched {len(data)} {component_name} records")

            return data

    def update_dataset(
        self, base_df: pl.DataFrame, updates: Dict[str, pl.DataFrame]
    ) -> pl.DataFrame:
        """Update dataset with multiple components"""
        df = base_df

        for component_name, component_df in updates.items():
            if component_df.is_empty():
                continue

            if component_name not in self.components:
                logger.warning(f"Skipping unknown component: {component_name}")
                continue

            component = self.components[component_name]
            logger.info(f"Processing {component_name} component...")

            # Apply component-specific processing
            df = component.process(df, component_df)

        return df

    def save_dataset(self, df: pl.DataFrame, tag: str = None):
        """Save updated dataset with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = tag or "updated"

        # Save parquet
        parquet_path = self.output_dir / f"ml_dataset_{tag}_{timestamp}.parquet"
        df.write_parquet(parquet_path)
        logger.info(f"Saved: {parquet_path}")

        # Save CSV
        csv_path = self.output_dir / f"ml_dataset_{tag}_{timestamp}.csv"
        df.write_csv(csv_path)
        logger.info(f"Saved: {csv_path}")

        # Create metadata
        metadata = self.builder.create_metadata(df)
        metadata["update_info"] = {
            "timestamp": timestamp,
            "tag": tag,
            "components_updated": [],  # Will be set by caller
        }

        meta_path = self.output_dir / f"ml_dataset_{tag}_{timestamp}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata: {meta_path}")

        return parquet_path, csv_path, meta_path


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Modular JQuants data updater")

    parser.add_argument("--dataset", type=str, help="Base dataset path")
    parser.add_argument(
        "--update",
        nargs="+",
        choices=["prices", "topix", "trades_spec", "listed_info"],
        help="Components to update",
    )
    parser.add_argument("--codes", nargs="+", help="Stock codes (for price updates)")
    parser.add_argument("--from-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days if dates not specified"
    )
    parser.add_argument("--tag", type=str, help="Tag for output files")

    args = parser.parse_args()

    # Initialize updater
    updater = ModularDataUpdater()

    # Load base dataset if provided
    base_df = pl.DataFrame()
    if args.dataset:
        dataset_path = Path(args.dataset)
        if dataset_path.exists():
            if dataset_path.suffix == ".parquet":
                base_df = pl.read_parquet(dataset_path)
            elif dataset_path.suffix == ".csv":
                base_df = pl.read_csv(dataset_path)
            logger.info(f"Loaded base dataset: {base_df.shape}")

    # Determine date range
    if args.from_date and args.to_date:
        from_date = args.from_date
        to_date = args.to_date
    else:
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    # Prepare parameters
    params = {"from": from_date, "to": to_date, "codes": args.codes or []}

    # Fetch updates
    updates = {}
    if args.update:
        for component_name in args.update:
            logger.info(f"\nFetching {component_name}...")
            component_data = await updater.fetch_component(component_name, params)
            if not component_data.is_empty():
                updates[component_name] = component_data

    # Update dataset
    if updates and not base_df.is_empty():
        logger.info("\nUpdating dataset...")
        updated_df = updater.update_dataset(base_df, updates)

        # Save
        updater.save_dataset(updated_df, args.tag)

        logger.info("\n" + "=" * 60)
        logger.info("UPDATE COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Components updated: {list(updates.keys())}")
        logger.info(f"Final shape: {updated_df.shape}")
    else:
        logger.warning("No updates to apply or no base dataset")


if __name__ == "__main__":
    asyncio.run(main())
