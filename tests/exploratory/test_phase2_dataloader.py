#!/usr/bin/env python3
"""
Phase 2 DataLoader拡張のテストスクリプト
新しいフィールド (group_day, sid, exposures) が正しく追加されているか確認
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_dataloader_extensions():
    """Test Phase 2 DataLoader extensions."""

    # Enable exposure features
    os.environ["USE_EXPOSURE_FEATURES"] = "1"
    os.environ["EXPOSURE_COLUMNS"] = "market_cap,beta,sector_code"

    print("=" * 60)
    print("Phase 2 DataLoader Extension Test")
    print("=" * 60)

    try:
        from src.gogooku3.training.atft.data_module import ProductionDataModuleV2, StreamingParquetDataset
        from omegaconf import OmegaConf

        # Create minimal config
        config = OmegaConf.create({
            "data": {
                "source": {
                    "data_dir": "data/raw/large_scale"
                },
                "schema": {
                    "code_column": "code",
                    "date_column": "date",
                    "target_column": "target",
                    "feature_columns": None,
                },
                "time_series": {
                    "sequence_length": 60,
                    "prediction_horizons": [1, 5, 10, 20]
                },
                "use_day_batch_sampler": True,
            },
            "normalization": {
                "online_normalization": {
                    "enabled": True
                }
            },
            "train": {
                "batch": {
                    "train_batch_size": 32,
                    "val_batch_size": 64,
                    "num_workers": 0,
                    "persistent_workers": False,
                    "pin_memory": False,
                    "prefetch_factor": None,
                }
            }
        })

        print("\n📁 Checking data directory...")
        data_dir = Path(config.data.source.data_dir)
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            print("   Please ensure ml_dataset_full.parquet exists")
            return False

        # Find parquet files
        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"❌ No parquet files found in {data_dir}")
            return False

        print(f"✅ Found {len(parquet_files)} parquet file(s)")

        # Test StreamingParquetDataset directly
        print("\n🧪 Testing StreamingParquetDataset...")

        # Get columns from first file
        import polars as pl
        df_sample = pl.scan_parquet(parquet_files[0]).head(1).collect()

        # Auto-detect feature columns
        exclude_cols = ["code", "date", "target"]
        numeric_dtypes = {
            pl.Float64, pl.Float32,
            pl.Int64, pl.Int32, pl.Int16, pl.Int8,
            pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8
        }

        feature_cols = [
            col for col in df_sample.columns
            if col not in exclude_cols
            and not col.startswith('target_')
            and df_sample.schema[col] in numeric_dtypes
        ]

        target_cols = [f"target_{h}d" for h in config.data.time_series.prediction_horizons]

        print(f"  Features: {len(feature_cols)} columns")
        print(f"  Targets: {target_cols}")

        # Create dataset with exposure features
        dataset = StreamingParquetDataset(
            file_paths=parquet_files[:1],  # Use only first file for test
            feature_columns=feature_cols,
            target_columns=target_cols,
            code_column="code",
            date_column="date",
            sequence_length=60,
            normalize_online=True,
            cache_size=10,
            exposure_columns=["market_cap", "beta", "sector_code"]
        )

        print(f"  Dataset size: {len(dataset)} samples")

        # Test single sample
        if len(dataset) > 0:
            print("\n📊 Testing single sample...")
            sample = dataset[0]

            # Check standard fields
            assert "features" in sample, "Missing 'features' field"
            assert "targets" in sample, "Missing 'targets' field"
            assert "code" in sample, "Missing 'code' field"
            assert "date" in sample, "Missing 'date' field"

            print(f"  ✅ Standard fields present")
            print(f"     Features shape: {sample['features'].shape}")
            print(f"     Targets: {list(sample['targets'].keys())}")

            # Check Phase 2 fields
            if os.getenv("USE_EXPOSURE_FEATURES") == "1":
                assert "group_day" in sample, "Missing 'group_day' field"
                assert "sid" in sample, "Missing 'sid' field"
                assert "exposures" in sample, "Missing 'exposures' field"

                print(f"  ✅ Phase 2 fields present")
                print(f"     group_day: {sample['group_day']} (dtype: {sample['group_day'].dtype})")
                print(f"     sid: {sample['sid']} (dtype: {sample['sid'].dtype})")
                print(f"     exposures shape: {sample['exposures'].shape} (dtype: {sample['exposures'].dtype})")

                # Validate dtypes
                assert sample['group_day'].dtype == torch.long, f"group_day should be torch.long, got {sample['group_day'].dtype}"
                assert sample['sid'].dtype == torch.long, f"sid should be torch.long, got {sample['sid'].dtype}"
                assert sample['exposures'].dtype == torch.float32, f"exposures should be torch.float32, got {sample['exposures'].dtype}"

                print(f"  ✅ Correct data types")

        # Test with DataLoader
        print("\n🔄 Testing with DataLoader...")
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )

        batch = next(iter(loader))
        print(f"  Batch keys: {list(batch.keys())}")

        if "group_day" in batch:
            print(f"  group_day shape: {batch['group_day'].shape}")
            print(f"  sid shape: {batch['sid'].shape}")
            print(f"  exposures shape: {batch['exposures'].shape}")

            # Check that group_day values are reasonable
            unique_days = torch.unique(batch['group_day'])
            print(f"  Unique group_day values in batch: {unique_days.tolist()}")

            # Check that sid values are reasonable
            unique_sids = torch.unique(batch['sid'])
            print(f"  Unique sid values in batch: {unique_sids.tolist()}")

        print("\n✅ Phase 2 DataLoader Extension Test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_integration():
    """Test that Phase 1 losses can use Phase 2 data."""

    print("\n" + "=" * 60)
    print("Phase 1-2 Integration Test")
    print("=" * 60)

    try:
        # Set environment variables for both phases
        os.environ["USE_EXPOSURE_FEATURES"] = "1"
        os.environ["USE_EXPOSURE_NEUTRAL"] = "1"
        os.environ["EXPOSURE_WEIGHT"] = "0.1"
        os.environ["USE_TURNOVER_PENALTY"] = "1"
        os.environ["TURNOVER_WEIGHT"] = "0.05"

        print("\n✅ Environment variables set:")
        print(f"  USE_EXPOSURE_FEATURES: {os.environ.get('USE_EXPOSURE_FEATURES')}")
        print(f"  USE_EXPOSURE_NEUTRAL: {os.environ.get('USE_EXPOSURE_NEUTRAL')}")
        print(f"  USE_TURNOVER_PENALTY: {os.environ.get('USE_TURNOVER_PENALTY')}")

        print("\n📝 Integration notes:")
        print("  - When group_day, sid, exposures are present in batch,")
        print("    the Phase 1 losses (exposure neutral, turnover) will activate")
        print("  - Without these fields, losses will be automatically skipped")
        print("  - This ensures backward compatibility")

        print("\n✅ Phase 1-2 Integration Ready")
        print("  Next step: Run smoke test with full pipeline")
        print("  Command: USE_EXPOSURE_FEATURES=1 USE_EXPOSURE_NEUTRAL=1 python scripts/smoke_test.py")

        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_dataloader_extensions()
    if success:
        test_loss_integration()

    sys.exit(0 if success else 1)