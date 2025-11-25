"""Storage cleanup utility for gogooku5 data module.

Retain only the latest N datasets and prune raw snapshots older than Y years.
Designed to keep disk usage stable on large machines while avoiding a rewrite
of the output path.
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

DATASET_PATTERN = re.compile(r"ml_dataset_(\d{8})_(\d{8})_.*\.(parquet|arrow)$")


def list_datasets(output_dir: Path) -> List[Path]:
    return sorted(
        [p for p in output_dir.glob("ml_dataset_*.*") if DATASET_PATTERN.match(p.name)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def prune_datasets(output_dir: Path, keep: int) -> List[Path]:
    datasets = list_datasets(output_dir)
    to_remove = datasets[keep:]
    removed: List[Path] = []
    for path in to_remove:
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            continue
    return removed


def _parse_dates_from_name(name: str) -> Tuple[datetime | None, datetime | None]:
    m = DATASET_PATTERN.match(name)
    if not m:
        return None, None
    start, end = m.group(1), m.group(2)
    try:
        return datetime.strptime(start, "%Y%m%d"), datetime.strptime(end, "%Y%m%d")
    except ValueError:
        return None, None


def prune_raw_source(source_dir: Path, keep_years: int) -> List[Path]:
    """Remove raw snapshots whose date ranges end before (today - keep_years)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=365 * keep_years)
    removed: List[Path] = []
    if not source_dir.exists():
        return removed

    for path in source_dir.glob("*.parquet"):
        _, end_dt = _parse_dates_from_name(path.name)
        if end_dt is None:
            # Best effort: if no dates in name, skip
            continue
        if end_dt < cutoff:
            try:
                path.unlink()
                removed.append(path)
            except OSError:
                continue
    return removed


def prune_raw(output_dir: Path, keep_years: int) -> List[Path]:
    removed: List[Path] = []
    raw_root = output_dir / "raw"
    if not raw_root.exists():
        return removed
    for source_dir in raw_root.iterdir():
        if not source_dir.is_dir():
            continue
        removed.extend(prune_raw_source(source_dir, keep_years))
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup datasets and raw snapshots.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Base output directory")
    parser.add_argument("--keep-datasets", type=int, default=3, help="Number of latest datasets to keep")
    parser.add_argument("--raw-years", type=int, default=2, help="Number of years of raw snapshots to keep")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir

    removed_ds = prune_datasets(output_dir, keep=args.keep_datasets)
    removed_raw = prune_raw(output_dir, keep_years=args.raw_years)

    print(f"[CLEANUP] Removed datasets: {len(removed_ds)}")
    for p in removed_ds[:10]:
        print(f"  - {p}")
    if len(removed_ds) > 10:
        print(f"  ... and {len(removed_ds) - 10} more")

    print(f"[CLEANUP] Removed raw snapshots: {len(removed_raw)}")
    for p in removed_raw[:10]:
        print(f"  - {p}")
    if len(removed_raw) > 10:
        print(f"  ... and {len(removed_raw) - 10} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
