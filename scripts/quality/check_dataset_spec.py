from __future__ import annotations

"""
Dataset spec validation used in Phase 1 refresh.

- 生成: Volatility / Volume / Momentum カテゴリのサンプルデータセット
- 検証: ParquetStockIterableDataset でストリーミング読み出しが可能かを確認
"""

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import importlib.util

import polars as pl

def _load_builder_class() -> type:
    module_path = SCRIPTS_DIR / "data" / "ml_dataset_builder.py"
    spec = importlib.util.spec_from_file_location("ml_dataset_builder", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load MLDatasetBuilder from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MLDatasetBuilder

MLDatasetBuilder = _load_builder_class()
from src.data.parquet_stock_dataset import OnlineRobustScaler, ParquetStockIterableDataset

LOG = logging.getLogger("dataset_spec")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate dataset specification coverage.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/ml_dataset_vol_volm_mom.parquet"),
        help="出力する Parquet のパス",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("output/ml_dataset_vol_volm_mom_metadata.json"),
        help="メタデータ出力パス",
    )
    parser.add_argument("--n-stocks", type=int, default=60)
    parser.add_argument("--n-days", type=int, default=180)
    parser.add_argument("--sequence-length", type=int, default=30)
    return parser


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    categories = ["Volatility", "Volume", "Momentum"]
    builder = MLDatasetBuilder(output_dir=args.output.parent)
    dataset_df, metadata = builder.build_category_dataset(
        categories,
        use_sample_data=True,
        sample_args={"n_stocks": args.n_stocks, "n_days": args.n_days},
    )

    ensure_output_dir(args.output)
    dataset_df.write_parquet(args.output)
    with open(args.metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    LOG.info("Saved dataset to %s (%d rows, %d columns)", args.output, len(dataset_df), len(dataset_df.columns))

    # Validate category coverage
    missing_cols = []
    for cat in categories:
        for col in builder.COLUMN_CATEGORY_MAP.get(cat, []):
            if col not in dataset_df.columns:
                missing_cols.append(col)
    if missing_cols:
        LOG.warning("Some expected columns are missing: %s", ", ".join(missing_cols))
    else:
        LOG.info("All category columns present")

    # Streaming dataset smoke test
    feature_cols = [c for c in dataset_df.columns if c not in ("Code", "Date")]
    target_cols = [c for c in feature_cols if c.startswith("feat_ret_")]
    scaler = OnlineRobustScaler(max_samples=100_000)
    iterable_ds = ParquetStockIterableDataset(
        file_paths=[args.output],
        feature_columns=feature_cols,
        target_columns=target_cols,
        code_column="Code",
        date_column="Date",
        sequence_length=args.sequence_length,
        scaler=scaler,
    )
    iterable_ds.fit()
    iterator = iter(iterable_ds)
    first_sample = next(iterator)

    LOG.info(
        "First sample -> features=%s, targets=%s, code=%s, date=%s",
        tuple(first_sample["features"].shape),
        list(first_sample["targets"].keys()),
        first_sample["code"],
        first_sample["date"],
    )

    LOG.info("Dataset spec validation completed successfully.")


if __name__ == "__main__":
    main()
