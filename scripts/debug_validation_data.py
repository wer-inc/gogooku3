"""Debug validation data availability."""
import sys

sys.path.insert(0, "/workspace/gogooku3")

import glob
from pathlib import Path

import torch

print("=" * 70)
print("VALIDATION DATA DEBUG")
print("=" * 70)
print()

# Check validation files
val_pattern = "output/atft_data/val/*.parquet"
val_files = sorted(glob.glob(val_pattern))

print("📂 Validation Files Check")
print(f"Pattern: {val_pattern}")
print(f"Found: {len(val_files)} files")
print()

if val_files:
    print("First 5 files:")
    for f in val_files[:5]:
        size = Path(f).stat().st_size / (1024**2)  # MB
        print(f"  {Path(f).name}: {size:.2f} MB")
    print()

    # Load one file to check content
    print("📊 Sample File Analysis")
    import polars as pl

    sample_file = val_files[0]
    df = pl.read_parquet(sample_file)
    print(f"File: {Path(sample_file).name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Unique codes: {df['Code'].n_unique()}")
    print()

    # Check for NaN in targets
    target_cols = [c for c in df.columns if c.startswith("target_")]
    if target_cols:
        print("🎯 Target Columns NaN Analysis")
        for col in target_cols[:4]:  # First 4 horizons
            nan_count = df[col].is_null().sum()
            nan_pct = 100 * nan_count / len(df)
            print(f"  {col}: {nan_count:,} NaNs ({nan_pct:.1f}%)")
        print()
else:
    print("❌ No validation files found!")
    print()
    print("Expected location: output/atft_data/val/")
    print("Check if dataset was split correctly.")
    print()

# Try loading with DataModule
print("🔧 DataModule Test")
try:
    from hydra import compose, initialize_config_dir

    config_dir = Path("/workspace/gogooku3/configs/atft").absolute()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config_production_optimized")

    # Override to use val files
    cfg.data.source.val_files = val_pattern

    print(f"Config val_files: {cfg.data.source.val_files}")

    # Try creating dataloader
    from src.gogooku3.training.atft.data_module import ProductionDataModuleV2

    data_module = ProductionDataModuleV2(cfg)
    data_module.setup(stage="fit")

    val_loader = data_module.val_dataloader()
    print("✅ Val dataloader created")
    print(f"Loader type: {type(val_loader)}")

    # Try fetching first batch
    print("\n📦 First Batch Test")
    try:
        batch = next(iter(val_loader))
        print("✅ Batch fetched successfully")
        print(f"Batch keys: {list(batch.keys())}")
        if "features" in batch:
            print(f"Features shape: {batch['features'].shape}")
        if "targets" in batch:
            print(f"Targets type: {type(batch['targets'])}")
            if isinstance(batch["targets"], dict):
                print(f"Target horizons: {list(batch['targets'].keys())}")
                for k, v in list(batch["targets"].items())[:2]:
                    if torch.is_tensor(v):
                        nan_count = torch.isnan(v).sum().item()
                        print(f"  {k}: shape={v.shape}, NaNs={nan_count}")
    except StopIteration:
        print("❌ Dataloader is empty (StopIteration)")
    except Exception as e:
        print(f"❌ Error fetching batch: {e}")
        import traceback

        traceback.print_exc()

except Exception as e:
    print(f"❌ DataModule test failed: {e}")
    import traceback

    traceback.print_exc()

print()
print("=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
