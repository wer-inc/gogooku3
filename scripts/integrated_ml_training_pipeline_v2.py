#!/usr/bin/env python3
"""
Integrated ML Training Pipeline V2 - Using FullTrainingPipeline
統合MLトレーニングパイプライン v2 - 改善版

PDFで提案された改善を実装:
- SafeTrainingPipelineとCompleteATFTTrainingPipelineの統合
- 設定の一元管理
- ベースライン結果のフィードバック
- Polars最適化
"""

import argparse
import logging
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gogooku3.training.full_training_pipeline import FullTrainingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the improved integrated training pipeline."""

    parser = argparse.ArgumentParser(
        description="Gogooku3 Integrated ML Training Pipeline V2 - 改善版統合パイプライン"
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to ML dataset (parquet file). Auto-detects if not specified.",
    )

    # Execution mode
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["baseline_only", "atft_only", "full"],
        default="full",
        help="Execution mode: baseline_only (SafeTrainingPipeline相当), "
             "atft_only (ATFT学習のみ), full (完全実行)",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/atft/unified_config.yaml",
        help="Path to unified configuration file",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/integrated_v2",
        help="Output directory for results",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="integrated_training_v2",
        help="Experiment name for tracking",
    )

    # Performance settings
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=8.0,
        help="Memory limit in GB",
    )
    parser.add_argument(
        "--use-polars",
        action="store_true",
        default=True,
        help="Use Polars for data processing (recommended)",
    )

    # Training settings
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of walk-forward validation splits",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=20,
        help="Embargo days between train and test",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Maximum training epochs (overrides config)",
    )

    # Advanced features
    parser.add_argument(
        "--use-baseline-feedback",
        action="store_true",
        default=True,
        help="Use baseline performance as feedback to ATFT training",
    )
    parser.add_argument(
        "--enable-optuna",
        action="store_true",
        help="Enable hyperparameter optimization with Optuna",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run inference after training",
    )

    # Debugging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actual training",
    )

    args = parser.parse_args()

    # Log configuration
    logger.info("=" * 80)
    logger.info("🚀 Gogooku3 Integrated ML Training Pipeline V2")
    logger.info("=" * 80)
    logger.info(f"📊 Run mode: {args.run_mode}")
    logger.info(f"⚙️ Config: {args.config}")
    logger.info(f"📁 Output: {args.output_dir}")

    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No actual training will be performed")
        return 0

    try:
        # Initialize the unified pipeline
        pipeline = FullTrainingPipeline(
            data_path=args.data_path,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            config_path=args.config,
            run_mode=args.run_mode,
            verbose=args.verbose,
        )

        # Override configuration if needed
        if args.max_epochs:
            pipeline.config["train"]["trainer"]["max_epochs"] = args.max_epochs

        if args.n_splits:
            pipeline.config["validation"]["n_splits"] = args.n_splits

        if args.embargo_days:
            pipeline.config["validation"]["embargo_days"] = args.embargo_days

        if args.memory_limit:
            pipeline.config["performance"]["memory_limit_gb"] = args.memory_limit

        if args.use_baseline_feedback:
            pipeline.config["baseline"]["feedback"]["enable"] = True

        if args.enable_optuna:
            pipeline.config["optuna"]["enable"] = True
            logger.info("🔬 Optuna hyperparameter optimization enabled")

        # Run the pipeline
        results = pipeline.run()

        # Post-training inference (if requested)
        if args.run_inference and results.get("success"):
            logger.info("\n🎯 Running inference on latest data...")
            results["inference"] = run_inference_pipeline(pipeline, results)

        # Print summary
        print_summary(results)

        return 0 if results.get("success") else 1

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Pipeline interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_inference_pipeline(pipeline: FullTrainingPipeline, training_results: dict) -> dict:
    """
    Run inference using the trained model.
    PDFで提案された改善: 推論パイプラインとの連携
    """
    inference_results = {
        "success": False,
        "predictions": None,
        "metrics": {},
    }

    try:
        # Find the best checkpoint
        checkpoint = None
        if "atft" in training_results:
            checkpoint = training_results["atft"].get("checkpoint")

        if not checkpoint:
            logger.warning("No checkpoint found, skipping inference")
            return inference_results

        # TODO: Implement actual inference
        # This would load the model and run predictions on recent data
        logger.info(f"Loading model from: {checkpoint}")
        logger.info("Inference implementation pending...")

        inference_results["success"] = True

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        inference_results["error"] = str(e)

    return inference_results


def print_summary(results: dict):
    """Print a comprehensive summary of the pipeline results."""

    print("\n" + "=" * 80)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("=" * 80)

    if not results.get("success"):
        print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
        return

    # Data preparation summary
    if "data_preparation" in results:
        data_info = results["data_preparation"]
        print("\n📂 Data Preparation:")
        print(f"  - Rows: {data_info.get('row_count', 'N/A'):,}")
        print(f"  - Loading time: {data_info.get('loading_time', 0):.1f}s")

        validation = data_info.get("validation", {})
        warnings = validation.get("warnings", [])
        if warnings:
            print(f"  - Warnings: {len(warnings)}")
            for w in warnings[:3]:  # Show first 3 warnings
                print(f"    ⚠️ {w}")

    # Baseline results
    if "baseline" in results:
        baseline = results["baseline"]
        if baseline.get("success"):
            metrics = baseline.get("metrics", {})
            print("\n🎯 Baseline Model (LightGBM):")
            print(f"  - IC: {metrics.get('mean_ic', 0):.3f}")
            print(f"  - RankIC: {metrics.get('mean_rank_ic', 0):.3f}")
            print(f"  - Training time: {metrics.get('training_time', 0):.1f}s")

    # ATFT results
    if "atft" in results:
        atft = results["atft"]
        if atft.get("success"):
            metrics = atft.get("metrics", {})
            print("\n🚄 ATFT-GAT-FAN Model:")
            print(f"  - Sharpe: {metrics.get('sharpe', 0):.3f}")
            print(f"  - Training time: {atft.get('training_time', 0):.1f}s")
            if atft.get("checkpoint"):
                print(f"  - Checkpoint: {atft['checkpoint']}")

    # Performance comparison
    if "comparison" in results:
        comp = results["comparison"]
        print("\n📈 Performance Comparison:")
        print(f"  - Improvement ratio: {comp.get('improvement_ratio', 0):.2f}x")
        print(f"  - Recommendation: {comp.get('recommendation', 'N/A')}")

    # Overall metrics
    if "metrics" in results:
        metrics = results["metrics"]
        print("\n⏱️ Execution Metrics:")
        print(f"  - Total time: {metrics.get('total_time', 0):.1f}s")
        print(f"  - Peak memory: {metrics.get('memory_peak', 0):.1f}GB")

    print("\n✅ Pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main())
