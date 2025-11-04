#!/usr/bin/env python3
"""
ATFT P0 State Diagnostic Script

Purpose: Identify actual state before applying P0 fixes
- Which dataset is being used?
- How many features does it have?
- Are CS-Z columns present?
- What does the config expect?
"""

import sys
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def analyze_dataset(path: str) -> dict:
    """Analyze a single parquet dataset."""
    p = Path(path)
    if not p.exists():
        return None

    try:
        # Load first row to inspect schema
        df = pl.scan_parquet(str(p)).head(1).collect()
        cols = df.columns

        # Identify column types
        metadata_cols = ["Code", "Date", "Section", "MarketCode", "LocalCode",
                        "section_norm", "row_idx"]
        target_cols = [c for c in cols if c.startswith("target_")]
        cs_z_cols = [c for c in cols if c.endswith("_cs_z")]

        feature_cols = [c for c in cols
                       if c not in metadata_cols and c not in target_cols]

        return {
            "path": path,
            "exists": True,
            "total_columns": len(cols),
            "metadata_columns": len([c for c in cols if c in metadata_cols]),
            "target_columns": len(target_cols),
            "feature_columns": len(feature_cols),
            "cs_z_columns": len(cs_z_cols),
            "sample_cs_z": cs_z_cols[:5] if cs_z_cols else None,
            "sample_features": feature_cols[:10],
        }
    except Exception as e:
        return {
            "path": path,
            "exists": True,
            "error": str(e)
        }

def analyze_config() -> dict:
    """Analyze ATFT config expectations."""
    config_path = Path("configs/atft/config_production_optimized.yaml")

    if not config_path.exists():
        return {"error": "Config not found"}

    try:
        cfg = OmegaConf.load(config_path)

        return {
            "path": str(config_path),
            "total_features": OmegaConf.select(cfg, "model.input_dims.total_features"),
            "expected_features": OmegaConf.select(cfg, "model.expected_features"),
            "data_path": OmegaConf.select(cfg, "data.path"),
            "manifest_path": OmegaConf.select(cfg, "features.manifest_path"),
        }
    except Exception as e:
        return {"error": str(e)}

def analyze_training_logs() -> list:
    """Extract feature mismatch errors from training logs."""
    log_path = Path("_logs/training/ml_training.log")

    if not log_path.exists():
        return []

    errors = []
    try:
        with open(log_path) as f:
            for line in f:
                line_lower = line.lower()
                if ("expected" in line_lower and "got" in line_lower) or \
                   ("dimension mismatch" in line_lower) or \
                   ("unable to find column" in line_lower and "cs_z" in line_lower):
                    errors.append(line.strip())
    except Exception as e:
        errors.append(f"Error reading log: {e}")

    # Return last 10 errors only
    return errors[-10:] if errors else []

def main():
    print_header("ATFT P0 DIAGNOSTIC REPORT")
    print(f"Generated: {Path.cwd()}")

    # 1. Dataset Analysis
    print_header("1. Dataset Analysis")

    datasets = [
        "output/ml_dataset_latest_clean.parquet",
        "output/ml_dataset_latest_full.parquet",
        "output/atft_data/test/data_0.parquet",
        "output/atft_data/train/data_0.parquet",
    ]

    found_datasets = []
    for ds_path in datasets:
        result = analyze_dataset(ds_path)
        if result and result.get("exists"):
            found_datasets.append(result)

            if "error" in result:
                print(f"\n❌ {result['path']}")
                print(f"   Error: {result['error']}")
            else:
                print(f"\n📊 {result['path']}")
                print(f"   Total columns: {result['total_columns']}")
                print(f"   Metadata: {result['metadata_columns']}")
                print(f"   Targets: {result['target_columns']}")
                print(f"   Features: {result['feature_columns']}")
                print(f"   CS-Z features: {result['cs_z_columns']}")

                if result['cs_z_columns'] > 0:
                    print(f"   ✅ CS-Z present: {result['sample_cs_z']}")
                else:
                    print("   ⚠️  CS-Z absent (no *_cs_z columns)")

                print(f"   Sample features: {result['sample_features'][:5]}")

    if not found_datasets:
        print("\n❌ No datasets found!")
        sys.exit(1)

    # 2. Config Analysis
    print_header("2. Config Analysis")

    config = analyze_config()
    if "error" in config:
        print(f"❌ Error: {config['error']}")
    else:
        print(f"\n⚙️  Config: {config['path']}")
        print(f"   Expected features (input_dims.total_features): {config['total_features']}")
        print(f"   Expected features (model.expected_features): {config['expected_features']}")
        print(f"   Data path: {config['data_path']}")
        print(f"   Manifest path: {config['manifest_path']}")

    # 3. Training Log Analysis
    print_header("3. Recent Training Errors")

    errors = analyze_training_logs()
    if errors:
        print(f"\nFound {len(errors)} recent feature-related errors:")
        for i, error in enumerate(errors, 1):
            print(f"\n{i}. {error}")
    else:
        print("\n✅ No recent feature-related errors found")

    # 4. Summary & Recommendations
    print_header("4. Summary & Recommendations")

    # Find dataset with most features
    datasets_with_data = [d for d in found_datasets if "feature_columns" in d]
    if datasets_with_data:
        max_features = max(d["feature_columns"] for d in datasets_with_data)
        primary_dataset = [d for d in datasets_with_data
                          if d["feature_columns"] == max_features][0]

        print("\n📌 Primary Dataset (most features):")
        print(f"   Path: {primary_dataset['path']}")
        print(f"   Features: {primary_dataset['feature_columns']}")
        print(f"   CS-Z: {'✅ Present' if primary_dataset['cs_z_columns'] > 0 else '❌ Absent'}")

        # Check config mismatch
        if config.get("total_features"):
            config_features = config["total_features"]
            actual_features = primary_dataset["feature_columns"]

            if config_features != actual_features:
                print("\n⚠️  MISMATCH DETECTED:")
                print(f"   Config expects: {config_features} features")
                print(f"   Dataset has: {actual_features} features")
                print(f"   Difference: {actual_features - config_features:+d}")

                print("\n🔧 Recommended Fix:")
                if primary_dataset['cs_z_columns'] > 0:
                    print(f"   P0-2: Update config to {actual_features} features")
                    print("   P0-3: CS-Z present, no generation needed")
                else:
                    print(f"   P0-2: Update config to {actual_features} features")
                    print("   P0-3: CS-Z absent, may need runtime generation")
            else:
                print(f"\n✅ Config matches dataset: {config_features} features")

    print("\n" + "=" * 80)
    print("Diagnostic complete. Save this output to P0_DIAGNOSTIC_REPORT.txt")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
