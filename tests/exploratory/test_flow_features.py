#!/usr/bin/env python3
"""
Test script for flow features integration
"""

import logging
import sys
from pathlib import Path

import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from features.flow_features import FlowFeaturesGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_trades_data():
    """Create sample trades_spec data for testing"""

    # Sample data with multiple sections and dates
    data = []
    sections = ["TSEPrime", "TSEStandard", "TSEGrowth"]

    for week in range(10):  # 10 weeks of data
        for section in sections:
            base_date = f"2024-01-{(week * 7 + 1):02d}"

            data.append(
                {
                    "PublishedDate": base_date,
                    "Section": section,
                    "ForeignersBalance": 100000000 * (1 + week * 0.1),
                    "ForeignersTotal": 500000000,
                    "IndividualsBalance": -50000000 * (1 + week * 0.05),
                    "IndividualsTotal": 300000000,
                    "TotalTotal": 1000000000,
                    "ProprietaryBalance": 10000000,
                    "BrokerageBalance": -5000000,
                    "SecuritiesCosBalance": 15000000,
                    "InvestmentTrustsBalance": 20000000,
                    "BusinessCosBalance": -10000000,
                    "InsuranceCosBalance": 5000000,
                    "TrustBanksBalance": 25000000,
                    "CityBKsRegionalBKsEtcBalance": -15000000,
                    "OtherFinancialInstitutionsBalance": 10000000,
                }
            )

    return pl.DataFrame(data)


def create_sample_stock_data():
    """Create sample stock daily data"""

    data = []
    codes = ["1301", "2802", "6501", "7203", "8306"]

    for code in codes:
        for day in range(30):  # 30 days of data
            data.append(
                {
                    "Code": code,
                    "Date": f"2024-01-{(day + 1):02d}",
                    "Close": 1000 + day * 10,
                    "Volume": 1000000,
                }
            )

    df = pl.DataFrame(data)
    df = df.with_columns(
        pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
    )
    return df


def test_flow_features():
    """Test flow features generation and attachment"""

    logger.info("=" * 60)
    logger.info("Testing Flow Features Integration")
    logger.info("=" * 60)

    # Create sample data
    logger.info("Creating sample data...")
    trades_df = create_sample_trades_data()
    stock_df = create_sample_stock_data()

    logger.info(f"  Trades data: {trades_df.shape}")
    logger.info(f"  Stock data: {stock_df.shape}")

    # Initialize generator
    logger.info("\nInitializing FlowFeaturesGenerator...")
    flow_gen = FlowFeaturesGenerator(z_score_window=4)  # Small window for test

    # Build flow event table
    logger.info("\nBuilding flow event table...")
    flow_event_df = flow_gen.build_flow_event_table(trades_df)

    if flow_event_df.is_empty():
        logger.error("Flow event table is empty!")
        return

    logger.info(f"✅ Flow event table created: {flow_event_df.shape}")
    logger.info(f"  Columns: {flow_event_df.columns}")

    # Show sample of flow event table
    logger.info("\nSample flow event data:")
    print(flow_event_df.head(3))

    # Attach to daily panel
    logger.info("\nAttaching flow features to daily panel...")
    result_df = flow_gen.attach_flow_to_daily(stock_df, flow_event_df)

    logger.info(f"✅ Result shape: {result_df.shape}")

    # Check added features
    flow_cols = [
        "flow_foreign_net_ratio",
        "flow_individual_net_ratio",
        "flow_smart_money_idx",
        "flow_activity_z",
        "flow_foreign_share_activity",
        "flow_breadth_pos",
        "flow_smart_money_mom4",
        "flow_shock_flag",
        "flow_impulse",
        "days_since_flow",
    ]

    added_cols = [col for col in flow_cols if col in result_df.columns]
    logger.info(f"\n✅ Successfully added {len(added_cols)} flow features:")
    for col in added_cols:
        logger.info(f"  - {col}")

    # Show sample of result
    logger.info("\nSample result with flow features:")
    print(result_df.select(["Code", "Date"] + added_cols[:4]).head(5))

    # Statistics
    logger.info("\nFlow feature statistics:")
    for col in added_cols[:5]:
        if col in result_df.columns:
            non_null = result_df[col].is_not_null().sum()
            mean_val = result_df[col].mean() if non_null > 0 else 0
            logger.info(f"  {col}: non-null={non_null}, mean={mean_val:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Flow Features Test Completed Successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_flow_features()
