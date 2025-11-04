"""
RFI-1: Collect Hydra/Config files for execution recipe
"""
import json
import os
import shutil
import subprocess
from pathlib import Path


def main():
    output_dir = Path("output/reports/diag_bundle/configs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Key config files to copy
    config_files = [
        "configs/atft/config_production_optimized.yaml",
        "configs/atft/feature_categories.yaml",
        "configs/atft/feature_groups.yaml",
        "configs/atft/data/io.yaml",
        "configs/atft/train/production_improved.yaml",
        "configs/atft/train/phase0_baseline.yaml",
    ]

    copied = []
    missing = []

    for config_path in config_files:
        if os.path.exists(config_path):
            dest = output_dir / Path(config_path).name
            shutil.copy2(config_path, dest)
            copied.append(config_path)
            print(f"✓ Copied: {config_path} → {dest}")
        else:
            missing.append(config_path)
            print(f"✗ Missing: {config_path}")

    # Git info for execution recipe
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
        branch = "unknown"

    # Execution recipe
    recipe = {
        "git_commit": commit,
        "git_branch": branch,
        "copied_configs": copied,
        "missing_configs": missing,
        "example_commands": {
            "train_optimized": "make train EPOCHS=120 BATCH_SIZE=2048",
            "train_safe": "make train-safe EPOCHS=120",
            "train_quick": "make train-quick EPOCHS=3",
            "direct_call": "python scripts/integrated_ml_training_pipeline.py --data-path output/ml_dataset_latest_clean_final.parquet --max-epochs 120 --batch-size 2048 --lr 5e-4"
        },
        "env_vars_optimized": {
            "ALLOW_UNSAFE_DATALOADER": "1",
            "NUM_WORKERS": "4",
            "USE_RANKIC": "1",
            "RANKIC_WEIGHT": "0.2",
            "CS_IC_WEIGHT": "0.15",
            "SHARPE_WEIGHT": "0.3",
        },
        "env_vars_safe": {
            "FORCE_SINGLE_PROCESS": "1",
        },
        "torch_seed": "42 (default in scripts/train_atft.py)",
        "numpy_seed": "42 (default)",
        "random_seed": "42 (default)",
    }

    recipe_path = output_dir / "execution_recipe.json"
    with open(recipe_path, "w") as f:
        json.dump(recipe, f, indent=2)

    print(f"\n{'='*80}")
    print("RFI-1: Config Collection Complete")
    print(f"{'='*80}")
    print(f"Copied configs: {len(copied)}")
    print(f"Missing configs: {len(missing)}")
    print(f"Saved: {recipe_path}")


if __name__ == "__main__":
    main()
