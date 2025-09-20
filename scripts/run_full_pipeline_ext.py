#!/usr/bin/env python
"""
End-to-end pipeline for feature preservation ML.

This script orchestrates the complete pipeline:
1. Dataset generation with feature extensions
2. Model training with cross-validation
3. Evaluation with ablation analysis
4. Performance reporting
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any
import subprocess
import sys
import logging

import polars as pl
import yaml


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run complete feature preservation ML pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline_ext.yaml"),
        help="Pipeline configuration file"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for dataset (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date for dataset (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pipeline_ext"),
        help="Output directory for all artifacts"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset generation (use existing)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing predictions)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load pipeline configuration."""
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            "dataset": {
                "adv_col": "dollar_volume_ma20",
                "winsorize_cols": ["returns_1d", "returns_5d", "rel_to_sec_5d"],
                "winsorize_k": 5.0,
            },
            "training": {
                "epochs": 10,
                "batch_size": 1024,
                "n_splits": 3,
                "embargo_days": 20,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
            },
            "evaluation": {
                "horizons": [1, 5, 10, 20],
                "ablation": True,
                "generate_html": True,
            },
            "performance": {
                "memory_limit_gb": 8,
                "cache_enabled": True,
            }
        }


def run_command(cmd: list[str], dry_run: bool = False, verbose: bool = False) -> int:
    """Execute a command with error handling."""
    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return 0

    if verbose:
        logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True
        )
        if verbose and result.stdout:
            print(result.stdout)
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return e.returncode


def generate_dataset(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    """Generate the base dataset."""
    logger.info(f"📊 Generating dataset from {args.start_date} to {args.end_date}")

    output_path = args.output_dir / "ml_dataset_full.parquet"

    if args.skip_dataset and output_path.exists():
        logger.info("Skipping dataset generation (using existing)")
        return output_path

    cmd = [
        "python", "scripts/pipelines/run_full_dataset.py",
        "--jquants",
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--output", str(output_path)
    ]

    if run_command(cmd, args.dry_run, args.verbose) != 0:
        raise RuntimeError("Dataset generation failed")

    return output_path


def extend_dataset(input_path: Path, args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    """Apply feature extensions to the dataset."""
    logger.info("🔧 Applying feature extensions")

    output_path = args.output_dir / "dataset_ext.parquet"

    cmd = [
        "python", "scripts/build_dataset_ext.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--adv-col", config["dataset"]["adv_col"]
    ]

    if run_command(cmd, args.dry_run, args.verbose) != 0:
        raise RuntimeError("Dataset extension failed")

    # Verify extensions were applied
    if not args.dry_run:
        df = pl.read_parquet(str(output_path))
        logger.info(f"Extended dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Check for key extended features
        expected_features = ["sec_ret_1d_eq_loo", "x_trend_intensity"]
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            logger.warning(f"Missing expected features: {missing}")

    return output_path


def train_model(data_path: Path, args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    """Train the multi-head model."""
    logger.info("🧠 Training multi-head model with feature groups")

    predictions_path = args.output_dir / "predictions.parquet"

    if args.skip_training and predictions_path.exists():
        logger.info("Skipping training (using existing predictions)")
        return predictions_path

    cmd = [
        "python", "scripts/train_multihead.py",
        "--data", str(data_path),
        "--epochs", str(config["training"]["epochs"]),
        "--batch-size", str(config["training"]["batch_size"]),
        "--feature-groups", "configs/feature_groups.yaml",
        "--pred-out", str(predictions_path)
    ]

    if run_command(cmd, args.dry_run, args.verbose) != 0:
        raise RuntimeError("Model training failed")

    return predictions_path


def evaluate_model(predictions_path: Path, args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    """Generate evaluation report with ablation analysis."""
    logger.info("📈 Generating evaluation report")

    report_path = args.output_dir / "evaluation_report.html"

    cmd = [
        "python", "scripts/eval_report.py",
        "--data", str(predictions_path),
        "--pred", "pred_1d",
        "--target", "target_1d"
    ]

    if config["evaluation"]["ablation"]:
        cmd.append("--ablation")

    horizons = ",".join(map(str, config["evaluation"]["horizons"]))
    cmd.extend(["--horizons", horizons])

    if config["evaluation"]["generate_html"]:
        cmd.extend(["--output", str(report_path)])

    if run_command(cmd, args.dry_run, args.verbose) != 0:
        raise RuntimeError("Evaluation failed")

    return report_path


def run_ci_tests(args: argparse.Namespace) -> bool:
    """Run CI tests to verify pipeline integrity."""
    logger.info("🧪 Running CI tests")

    test_commands = [
        ["python", "-m", "pytest", "tests/test_data_checks.py", "-v", "-q"],
        ["python", "-m", "pytest", "tests/test_cv_pipeline.py", "-v", "-q", "-m", "not slow"]
    ]

    for cmd in test_commands:
        if run_command(cmd, args.dry_run, args.verbose) != 0:
            logger.warning("Some CI tests failed")
            return False

    logger.info("✅ All CI tests passed")
    return True


def generate_summary(args: argparse.Namespace, config: Dict[str, Any],
                    start_time: float) -> None:
    """Generate pipeline execution summary."""
    elapsed = time.time() - start_time

    summary = {
        "pipeline": "Feature Preservation ML (全特徴量保持)",
        "period": f"{args.start_date} to {args.end_date}",
        "execution_time_seconds": elapsed,
        "configuration": config,
        "outputs": {
            "dataset": str(args.output_dir / "ml_dataset_full.parquet"),
            "extended_dataset": str(args.output_dir / "dataset_ext.parquet"),
            "predictions": str(args.output_dir / "predictions.parquet"),
            "report": str(args.output_dir / "evaluation_report.html"),
        }
    }

    summary_path = args.output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "="*60)
    logger.info("📊 PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"Period: {args.start_date} to {args.end_date}")
    logger.info(f"Execution time: {elapsed:.1f} seconds")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Summary saved to: {summary_path}")


def main() -> int:
    """Main pipeline orchestration."""
    args = parse_args()
    start_time = time.time()

    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    try:
        # Step 1: Generate base dataset
        dataset_path = generate_dataset(args, config)

        # Step 2: Apply feature extensions
        extended_path = extend_dataset(dataset_path, args, config)

        # Step 3: Train model
        predictions_path = train_model(extended_path, args, config)

        # Step 4: Evaluate and generate report
        report_path = evaluate_model(predictions_path, args, config)

        # Step 5: Run CI tests
        if not args.dry_run:
            tests_passed = run_ci_tests(args)
            if not tests_passed:
                logger.warning("⚠️ Some tests failed, but pipeline completed")

        # Generate summary
        generate_summary(args, config, start_time)

        logger.info("\n✅ Pipeline completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())