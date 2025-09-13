#!/usr/bin/env python3
"""
Thin wrapper to run the modern SafeTrainingPipeline from gogooku3.

This script preserves the legacy CLI while delegating to
gogooku3.training.safe_training_pipeline.SafeTrainingPipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import argparse


def _find_dataset_in_dir(data_dir: Path) -> Path:
    cands = sorted(data_dir.glob("*.parquet"))
    if not cands:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    # Use largest as primary
    return max(cands, key=lambda p: p.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SafeTrainingPipeline")
    parser.add_argument("--data-dir", default="output", help="Directory containing parquet dataset")
    parser.add_argument("--output-dir", default="output", help="Output root directory")
    parser.add_argument(
        "--experiment-name",
        default=f"safe_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Experiment name",
    )
    parser.add_argument("--memory-limit", type=float, default=8.0, help="Memory limit in GB")
    parser.add_argument("--n-splits", type=int, default=3, help="Walk-Forward splits")
    parser.add_argument("--embargo-days", type=int, default=20, help="Embargo days")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset_path = _find_dataset_in_dir(data_dir)

    try:
        from gogooku3.training.safe_training_pipeline import SafeTrainingPipeline
    except Exception as e:
        print(f"❌ Failed to import modern pipeline: {e}")
        print("Please install package (pip install -e .) and ensure gogooku3 is importable.")
        sys.exit(2)

    # Compose output directory under experiments/<experiment_name>
    output_dir = Path(args.output_dir) / "experiments" / args.experiment_name

    pipeline = SafeTrainingPipeline(
        data_path=dataset_path,
        output_dir=output_dir,
        experiment_name=args.experiment_name,
        verbose=args.verbose,
    )

    try:
        results = pipeline.run_pipeline(
            n_splits=args.n_splits,
            embargo_days=args.embargo_days,
            memory_limit_gb=args.memory_limit,
            save_results=True,
        )
        ok = bool(results)
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
