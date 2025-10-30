#!/usr/bin/env python3
"""
Thin wrapper to run the modern SafeTrainingPipeline from gogooku3.

This script preserves the legacy CLI while delegating to
gogooku3.training.safe_training_pipeline.SafeTrainingPipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure src/ is on sys.path so that 'gogooku3' can be imported without editable install
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _find_dataset_in_dir(data_dir: Path) -> Path:
    """
    Locate the ML dataset parquet file.

    Historically the training scripts expected datasets directly under
    OUTPUT_BASE (e.g., /home/ubuntu/gogooku3-standalone/output/batch), but in
    many environments the files live under repo-relative paths such as
    output/datasets/. To keep `make train-safe` working out-of-the-box we
    search a small set of common locations and fall back gracefully.
    """

    search_roots = []
    seen = set()

    def add_root(path: Path) -> None:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            # Allow non-existent paths so we can surface better error messages later
            resolved = path
        if resolved not in seen:
            seen.add(resolved)
            search_roots.append(path)

    add_root(data_dir)
    add_root(data_dir / "datasets")

    # Common repo-relative fallbacks
    repo_output = REPO_ROOT / "output"
    add_root(repo_output / "datasets")
    add_root(repo_output)
    add_root(REPO_ROOT / "data" / "processed")
    add_root(REPO_ROOT / "data")

    # Optional: honor explicit DATA_PATH if user exported it
    data_path_env = Path(os.getenv("DATA_PATH", "")).expanduser()
    if data_path_env:
        add_root(data_path_env.parent if data_path_env.is_file() else data_path_env)

    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue

        # Prefer ML datasets
        ml_candidates = sorted(root.glob("ml_dataset*.parquet"), key=lambda p: p.stat().st_size, reverse=True)
        if ml_candidates:
            chosen = ml_candidates[0]
            print(f"   📂 Using dataset: {chosen} (detected via {root})")
            return chosen

        # Fallback: any parquet file (pick largest)
        generic = sorted(root.glob("*.parquet"), key=lambda p: p.stat().st_size, reverse=True)
        if generic:
            chosen = generic[0]
            print(f"   📂 Using dataset: {chosen} (generic fallback in {root})")
            return chosen

    searched = "\n     ".join(str(r) for r in search_roots)
    raise FileNotFoundError(
        "No parquet datasets found in any of the expected locations.\n"
        "   Checked directories:\n"
        f"     {searched}\n"
        "💡 Generate a dataset with `make dataset-bg` or specify --data-dir manually."
    )


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
