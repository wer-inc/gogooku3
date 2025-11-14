#!/usr/bin/env python3
"""
Direct asset materialization runner for gogooku5 Dagster assets.

This bypasses the dagster CLI's ANTLR 4.13 requirement by directly calling
the materialize() API, keeping everything in the Hydra-friendly ANTLR 4.9.3
runtime.

Usage:
    python materialize_asset.py --asset g5_dataset_chunks --config chunks.yaml
    python materialize_asset.py --asset g5_dataset_full --start 2023-01-01 --end 2023-12-31
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from dagster import AssetKey, DagsterInstance, materialize
from dagster._core.definitions.asset_selection import AssetSelection

# Ensure the dagster_gogooku5 module is importable
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "gogooku5" / "data" / "src"))

from dagster_gogooku5.defs import defs


def load_config(config_path: Path | None, cli_args: dict) -> dict:
    """
    Load configuration from YAML file or CLI arguments.

    Args:
        config_path: Path to YAML config file
        cli_args: CLI arguments to override/supplement config

    Returns:
        Dagster run config dict
    """
    config = {}

    if config_path and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # CLI args override YAML config
    if cli_args:
        if "ops" not in config:
            config["ops"] = {}

        # Merge CLI args into ops config
        for asset_name in ["build_dataset_chunks", "merge_latest_dataset"]:
            if asset_name not in config["ops"]:
                config["ops"][asset_name] = {"config": {}}

            asset_config = config["ops"][asset_name]["config"]
            for key, value in cli_args.items():
                if value is not None:
                    asset_config[key] = value

    return config


def materialize_asset_direct(
    asset_name: str,
    config: dict,
    instance: DagsterInstance | None = None,
) -> bool:
    """
    Materialize a Dagster asset directly without using the CLI.

    Args:
        asset_name: Name of the asset to materialize (e.g., "g5_dataset_chunks")
        config: Dagster run config dict
        instance: Dagster instance (uses default if None)

    Returns:
        True if successful, False otherwise
    """
    # Get the asset from definitions
    asset_key = AssetKey(asset_name)

    # Find matching asset
    matching_assets = [asset for asset in defs.assets if asset.key == asset_key]

    if not matching_assets:
        print(f"❌ Asset '{asset_name}' not found in definitions", file=sys.stderr)
        print(f"Available assets: {[str(asset.key) for asset in defs.assets]}", file=sys.stderr)
        return False

    # For assets with dependencies, include all upstream assets
    # This is necessary because Dagster needs the full dependency graph
    assets_to_materialize = list(defs.assets)  # Include all assets to satisfy dependencies

    # Use provided instance or get default
    if instance is None:
        instance = DagsterInstance.get()

    print(f"🚀 Materializing asset: {asset_name}")
    print(f"📋 Config: {json.dumps(config, indent=2)}")

    try:
        # Materialize with selection (only execute the requested asset)
        selection = AssetSelection.assets(asset_key)

        # Materialize the asset
        result = materialize(
            assets=assets_to_materialize,
            instance=instance,
            resources=defs.resources,
            run_config=config,
            selection=selection,
        )

        if result.success:
            print(f"✅ Successfully materialized {asset_name}")
            return True
        else:
            print(f"❌ Materialization failed for {asset_name}", file=sys.stderr)
            return False

    except Exception as e:
        print(f"❌ Error materializing {asset_name}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Directly materialize Dagster assets without CLI ANTLR conflicts"
    )
    parser.add_argument(
        "--asset",
        required=True,
        choices=["g5_dataset_chunks", "g5_dataset_full"],
        help="Asset to materialize",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file",
    )

    # Common config options that can be specified via CLI
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--chunk-months", type=int, help="Months per chunk (default: 3)")
    parser.add_argument("--latest-only", action="store_true", help="Only build latest chunk")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    parser.add_argument("--refresh-listed", action="store_true", help="Refresh listed metadata")

    # Merge-specific options
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial merge")
    parser.add_argument("--min-coverage", type=float, help="Minimum coverage for merge (0.0-1.0)")

    args = parser.parse_args()

    # Build CLI args dict (only non-None values)
    cli_config = {}
    if args.start:
        cli_config["start"] = args.start
    if args.end:
        cli_config["end"] = args.end
    if args.chunk_months is not None:
        cli_config["chunk_months"] = args.chunk_months
    if args.latest_only:
        cli_config["latest_only"] = True
    if args.resume:
        cli_config["resume"] = True
    if args.force:
        cli_config["force"] = True
    if args.refresh_listed:
        cli_config["refresh_listed"] = True
    if args.allow_partial:
        cli_config["allow_partial"] = True
    if args.min_coverage is not None:
        cli_config["min_coverage"] = args.min_coverage

    # Load and merge config
    config = load_config(args.config, cli_config)

    # Materialize the asset
    success = materialize_asset_direct(args.asset, config)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
