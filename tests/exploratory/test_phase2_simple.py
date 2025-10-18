#!/usr/bin/env python3
"""
Phase 2 DataLoader拡張の簡易テスト
実際のデータを使って新しいフィールドを確認
"""

import os
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Test Phase 2 DataLoader extensions with real data."""

    # Enable exposure features
    os.environ["USE_EXPOSURE_FEATURES"] = "1"
    os.environ["EXPOSURE_COLUMNS"] = "market_cap,beta,sector_code"
    os.environ["ALLOW_UNSAFE_DATALOADER"] = "0"  # Force single-process for safety

    print("=" * 60)
    print("Phase 2 DataLoader Extension Simple Test")
    print("=" * 60)

    try:
        import polars as pl

        from src.gogooku3.training.atft.data_module import StreamingParquetDataset

        # Use existing ML dataset - try multiple possible paths
        possible_paths = [
            Path(
                "/home/ubuntu/gogooku3-standalone/output/ml_dataset_latest_full.parquet"
            ),
            Path("/home/ubuntu/gogooku3-standalone/tmp_train.parquet"),
            Path(
                "/home/ubuntu/gogooku3-standalone/output/ml_dataset_20250928_084147_full.parquet"
            ),
        ]

        dataset_path = None
        for path in possible_paths:
            if path.exists():
                dataset_path = path
                break

        if dataset_path is None:
            print(f"❌ Dataset not found: {dataset_path}")
            return False

        print(f"\n✅ Found dataset: {dataset_path}")

        # Get column names
        df_sample = pl.scan_parquet(dataset_path).head(1).collect()
        print(f"   Dataset shape preview: {df_sample.shape}")

        # Auto-detect columns
        all_columns = df_sample.columns
        print(f"   Total columns: {len(all_columns)}")

        # Look for exposure-related columns
        exposure_cols = []
        if "market_cap" in all_columns:
            exposure_cols.append("market_cap")
            print("   ✓ Found market_cap")
        if "beta" in all_columns:
            exposure_cols.append("beta")
            print("   ✓ Found beta")
        if "sector_code" in all_columns:
            exposure_cols.append("sector_code")
            print("   ✓ Found sector_code")
        if "sector" in all_columns and "sector_code" not in all_columns:
            exposure_cols.append("sector")
            print("   ✓ Found sector (as alternative)")

        print("\n📊 Testing StreamingParquetDataset with exposure features...")

        # Feature columns (exclude special columns)
        exclude_cols = ["code", "date", "datetime", "Code", "Date"]
        feature_cols = []

        for col in all_columns:
            if (
                col not in exclude_cols
                and not col.startswith("target_")
                and not col.startswith("Target_")
            ):
                feature_cols.append(col)

        # Target columns
        target_cols = [
            col
            for col in all_columns
            if col.startswith("target_") or col.startswith("Target_")
        ]
        if not target_cols:
            # If no explicit target columns, try to find return-like columns
            target_cols = ["returns_1d"] if "returns_1d" in all_columns else []

        print(f"   Features: {len(feature_cols)} columns")
        print(f"   Targets: {target_cols[:5]}...")  # Show first 5 targets

        # Create dataset
        dataset = StreamingParquetDataset(
            file_paths=[dataset_path],
            feature_columns=feature_cols[:50],  # Use subset for testing
            target_columns=target_cols[:4] if target_cols else ["returns_1d"],
            code_column="code" if "code" in all_columns else "Code",
            date_column="date" if "date" in all_columns else "Date",
            sequence_length=60,
            normalize_online=False,  # Disable for testing
            cache_size=10,
            exposure_columns=exposure_cols if exposure_cols else None,
        )

        print(f"\n📦 Dataset created: {len(dataset)} samples")

        # Test single sample
        if len(dataset) > 0:
            print("\n🧪 Testing single sample...")
            sample = dataset[0]

            # Check standard fields
            print("   Standard fields:")
            for key in ["features", "targets", "code", "date"]:
                if key in sample:
                    if key == "features":
                        print(f"     ✓ {key}: shape {sample[key].shape}")
                    elif key == "targets":
                        print(f"     ✓ {key}: {len(sample[key])} horizons")
                    else:
                        print(f"     ✓ {key}: {sample[key]}")

            # Check Phase 2 fields
            print("\n   Phase 2 fields (if enabled):")
            phase2_fields = ["group_day", "sid", "exposures"]
            for field in phase2_fields:
                if field in sample:
                    if field == "exposures":
                        print(
                            f"     ✓ {field}: shape {sample[field].shape}, dtype {sample[field].dtype}"
                        )
                    else:
                        print(
                            f"     ✓ {field}: value {sample[field].item()}, dtype {sample[field].dtype}"
                        )
                else:
                    print(
                        f"     ✗ {field}: not found (check USE_EXPOSURE_FEATURES={os.getenv('USE_EXPOSURE_FEATURES')})"
                    )

            # Verify data types
            if "group_day" in sample:
                assert (
                    sample["group_day"].dtype == torch.long
                ), f"group_day wrong dtype: {sample['group_day'].dtype}"
            if "sid" in sample:
                assert (
                    sample["sid"].dtype == torch.long
                ), f"sid wrong dtype: {sample['sid'].dtype}"
            if "exposures" in sample:
                assert (
                    sample["exposures"].dtype == torch.float32
                ), f"exposures wrong dtype: {sample['exposures'].dtype}"

            print("\n✅ Phase 2 DataLoader Extension Test PASSED")

            # Show integration instructions
            print("\n📝 Integration Instructions:")
            print("   To enable new loss functions in training:")
            print("   ```bash")
            print("   # Enable DataLoader extensions")
            print("   export USE_EXPOSURE_FEATURES=1")
            print("   export EXPOSURE_COLUMNS=market_cap,beta,sector_code")
            print()
            print("   # Enable Phase 1 loss functions")
            print("   export USE_SOFT_SPEARMAN=1")
            print("   export SPEARMAN_WEIGHT=0.1")
            print("   export USE_EXPOSURE_NEUTRAL=1")
            print("   export EXPOSURE_WEIGHT=0.05")
            print("   export USE_TURNOVER_PENALTY=1")
            print("   export TURNOVER_WEIGHT=0.02")
            print()
            print("   # Run training")
            print("   python scripts/train_atft.py")
            print("   ```")

            return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
