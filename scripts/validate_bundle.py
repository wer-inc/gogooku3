#!/usr/bin/env python
"""
Bundle Validation Script - 4-Way Verification

Validates compatibility between:
1. Checkpoint (model weights and metadata)
2. Config (YAML configuration)
3. Dataset (parquet file schema)
4. Manifest (MANIFEST.lock metadata)

Usage:
    python scripts/validate_bundle.py \
        --bundle bundles/apex_ranker_v0.1.0_prod \
        --dataset output/ml_dataset_latest_clean_with_adv.parquet

Exit Codes:
    0: All validations passed
    1: Validation failed (incompatibility detected)
    2: Missing required files
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl
import torch
import yaml


class BundleValidator:
    """Validates production bundle compatibility"""

    def __init__(self, bundle_path: Path, dataset_path: Path = None):
        self.bundle_path = Path(bundle_path)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.errors = []
        self.warnings = []

    def validate(self) -> bool:
        """Run all validation checks"""
        print("=" * 80)
        print("APEX Ranker Bundle Validation")
        print("=" * 80)
        print(f"Bundle: {self.bundle_path}")
        if self.dataset_path:
            print(f"Dataset: {self.dataset_path}")
        print()

        # Check required files exist
        if not self._check_files_exist():
            return False

        # Load all components
        manifest = self._load_manifest()
        checkpoint = self._load_checkpoint()
        config = self._load_config()
        dataset = self._load_dataset() if self.dataset_path else None

        # Run validation checks
        self._validate_checkpoint(checkpoint, manifest)
        self._validate_config(config, manifest)
        self._validate_checkpoint_config_compatibility(checkpoint, config, manifest)
        if dataset is not None:
            self._validate_dataset(dataset, manifest)

        # Report results
        return self._report_results()

    def _check_files_exist(self) -> bool:
        """Check all required files exist"""
        required = {
            "MANIFEST.lock": self.bundle_path / "MANIFEST.lock",
            "Model checkpoint": self.bundle_path / "models/apex_ranker_v0_enhanced.pt",
            "Config": self.bundle_path / "configs/v0_base_89_cleanADV.yaml",
        }

        missing = []
        for name, path in required.items():
            if not path.exists():
                missing.append(f"{name}: {path}")

        if missing:
            print("❌ Missing required files:")
            for item in missing:
                print(f"   - {item}")
            return False

        print("✅ All required files present")
        return True

    def _load_manifest(self) -> dict[str, Any]:
        """Load and validate MANIFEST.lock"""
        manifest_path = self.bundle_path / "MANIFEST.lock"
        with open(manifest_path) as f:
            manifest = json.load(f)

        print(f"✅ Loaded manifest: {manifest['bundle_version']}")
        return manifest

    def _load_checkpoint(self) -> dict[str, Any]:
        """Load model checkpoint"""
        ckpt_path = self.bundle_path / "models/apex_ranker_v0_enhanced.pt"
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # Extract effective dimension from weight shape
        # Checkpoint stores weights directly (no 'state_dict' wrapper)
        conv_weight = checkpoint['encoder.patch_embed.conv.weight']
        effective_dim = conv_weight.shape[0]

        print(f"✅ Loaded checkpoint: effective_dim={effective_dim}")
        return {
            'weights': checkpoint,
            'effective_dim': effective_dim,
            'conv_weight_shape': conv_weight.shape,
        }

    def _load_config(self) -> dict[str, Any]:
        """Load YAML config"""
        config_path = self.bundle_path / "configs/v0_base_89_cleanADV.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        patch_mult = config.get('model', {}).get('patch_multiplier')
        print(f"✅ Loaded config: patch_multiplier={patch_mult if patch_mult else 'auto (unset)'}")
        return config

    def _load_dataset(self) -> pl.DataFrame:
        """Load and validate dataset schema"""
        if not self.dataset_path.exists():
            self.warnings.append(f"Dataset not found: {self.dataset_path}")
            return None

        df = pl.read_parquet(self.dataset_path)
        print(f"✅ Loaded dataset: {len(df):,} rows, {len(df.columns)} columns")
        return df

    def _validate_checkpoint(self, checkpoint: dict, manifest: dict):
        """Validate checkpoint against manifest"""
        print("\n" + "─" * 80)
        print("Checkpoint Validation")
        print("─" * 80)

        expected_dim = manifest['checkpoint']['effective_dim']
        actual_dim = checkpoint['effective_dim']

        if actual_dim != expected_dim:
            self.errors.append(
                f"Checkpoint dimension mismatch: "
                f"expected={expected_dim}, actual={actual_dim}"
            )
            print(f"❌ Dimension mismatch: {actual_dim} != {expected_dim}")
        else:
            print(f"✅ Effective dimension: {actual_dim}")

        # Validate expected shape
        expected_shape = (expected_dim, 1, 16)
        actual_shape = checkpoint['conv_weight_shape']
        if tuple(actual_shape) != expected_shape:
            self.warnings.append(
                f"Unexpected weight shape: {actual_shape} (expected {expected_shape})"
            )
        else:
            print(f"✅ Weight shape: {actual_shape}")

    def _validate_config(self, config: dict, manifest: dict):
        """Validate config against manifest"""
        print("\n" + "─" * 80)
        print("Config Validation")
        print("─" * 80)

        # Check patch_multiplier
        patch_mult = config.get('model', {}).get('patch_multiplier')
        expected_mult = manifest['config']['patch_multiplier']

        if patch_mult is not None:
            # Config has explicit value
            if manifest['checkpoint']['csz_mode'] == 'trained_with_csz':
                self.errors.append(
                    f"CRITICAL: Config sets patch_multiplier={patch_mult}, "
                    f"but checkpoint was trained WITH CS-Z (178ch). "
                    f"This will cause dimension mismatch. "
                    f"Remove patch_multiplier line from config."
                )
                print("❌ patch_multiplier explicitly set (should be auto)")
            else:
                print(f"✅ patch_multiplier: {patch_mult}")
        else:
            print("✅ patch_multiplier: auto (unset)")

        # Verify SHA256
        config_path = self.bundle_path / "configs/v0_base_89_cleanADV.yaml"
        actual_sha = self._compute_sha256(config_path)
        expected_sha = manifest['config']['sha256']

        if actual_sha != expected_sha:
            self.errors.append(
                f"Config SHA256 mismatch: {actual_sha[:16]}... != {expected_sha[:16]}..."
            )
            print("❌ SHA256 mismatch")
        else:
            print(f"✅ SHA256: {actual_sha[:16]}...")

    def _validate_checkpoint_config_compatibility(
        self, checkpoint: dict, config: dict, manifest: dict
    ):
        """Validate checkpoint-config compatibility (critical check)"""
        print("\n" + "─" * 80)
        print("Checkpoint-Config Compatibility (CRITICAL)")
        print("─" * 80)

        effective_dim = checkpoint['effective_dim']
        patch_mult = config.get('model', {}).get('patch_multiplier')
        base_features = manifest['checkpoint']['base_features']

        # Determine expected behavior
        if patch_mult is None:
            # Auto mode: will detect from checkpoint
            expected_dim = effective_dim
            print(f"✅ Auto-detection mode: will load {effective_dim}ch from checkpoint")
        elif patch_mult == 1:
            # Forced CS-Z OFF: expects raw features
            expected_dim = base_features
            if effective_dim != base_features:
                self.errors.append(
                    f"DIMENSION MISMATCH: "
                    f"Config forces patch_multiplier=1 (expects {base_features}ch), "
                    f"but checkpoint has {effective_dim}ch. "
                    f"This will cause RuntimeError on model.load_state_dict(). "
                    f"FIX: Remove 'patch_multiplier' line from config."
                )
                print("❌ CRITICAL: Dimension mismatch detected")
                print(f"   Config expects: {expected_dim}ch (patch_multiplier={patch_mult})")
                print(f"   Checkpoint has: {effective_dim}ch")
                print("   → Model loading will FAIL")
            else:
                print(f"✅ Dimensions compatible: {expected_dim}ch")
        elif patch_mult == 2:
            # Forced CS-Z ON: expects doubled features
            expected_dim = base_features * 2
            if effective_dim != expected_dim:
                self.errors.append(
                    f"DIMENSION MISMATCH: patch_multiplier=2 expects {expected_dim}ch, "
                    f"checkpoint has {effective_dim}ch"
                )
                print("❌ CRITICAL: Dimension mismatch detected")
            else:
                print(f"✅ Dimensions compatible: {expected_dim}ch")
        else:
            self.warnings.append(f"Unknown patch_multiplier value: {patch_mult}")

    def _validate_dataset(self, dataset: pl.DataFrame, manifest: dict):
        """Validate dataset schema and statistics"""
        print("\n" + "─" * 80)
        print("Dataset Validation")
        print("─" * 80)

        # Check feature count
        expected_features = manifest['checkpoint']['base_features']

        # Count feature columns (exclude Date, Code, target_*)
        feature_cols = [
            col for col in dataset.columns
            if not col.startswith('target_') and col not in ['Date', 'Code']
        ]
        actual_features = len(feature_cols)

        if actual_features != expected_features:
            self.warnings.append(
                f"Feature count mismatch: expected {expected_features}, "
                f"found {actual_features}"
            )
            print(f"⚠️  Feature count: {actual_features} (expected {expected_features})")
        else:
            print(f"✅ Feature count: {actual_features}")

        # Check date range
        date_min = dataset['Date'].min()
        date_max = dataset['Date'].max()
        print(f"✅ Date range: {date_min} to {date_max}")

        # Check unique stocks
        n_stocks = dataset['Code'].n_unique()
        print(f"✅ Unique stocks: {n_stocks:,}")

        # Check for nulls in critical columns
        null_counts = {}
        for col in ['Date', 'Code']:
            null_count = dataset[col].null_count()
            if null_count > 0:
                null_counts[col] = null_count

        if null_counts:
            self.warnings.append(f"Null values detected: {null_counts}")
            print(f"⚠️  Null values: {null_counts}")
        else:
            print("✅ No nulls in critical columns")

    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _report_results(self) -> bool:
        """Print final validation report"""
        print("\n" + "=" * 80)
        print("Validation Report")
        print("=" * 80)

        if self.errors:
            print(f"\n❌ FAILED: {len(self.errors)} error(s) detected\n")
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")
            print()

        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} warning(s):\n")
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning}")
            print()

        if not self.errors and not self.warnings:
            print("\n✅ ALL CHECKS PASSED")
            print("\nBundle is ready for deployment.")
            print()
            return True
        elif not self.errors:
            print("\n✅ PASSED (with warnings)")
            print("\nBundle is usable but review warnings above.")
            print()
            return True
        else:
            print("\n❌ DEPLOYMENT BLOCKED")
            print("\nBundle has critical errors. Fix errors above before deployment.")
            print()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate APEX Ranker production bundle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate bundle only (no dataset check)
    python scripts/validate_bundle.py --bundle bundles/apex_ranker_v0.1.0_prod

    # Validate bundle + dataset compatibility
    python scripts/validate_bundle.py \\
        --bundle bundles/apex_ranker_v0.1.0_prod \\
        --dataset output/ml_dataset_latest_clean_with_adv.parquet
        """
    )
    parser.add_argument(
        '--bundle',
        type=str,
        required=True,
        help='Path to bundle directory'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to dataset parquet file (optional)'
    )

    args = parser.parse_args()

    validator = BundleValidator(
        bundle_path=args.bundle,
        dataset_path=args.dataset
    )

    success = validator.validate()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
