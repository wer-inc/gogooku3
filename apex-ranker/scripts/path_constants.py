#!/usr/bin/env python3
"""
Path constants for APEX-Ranker quality management pipeline

Centralized path definitions to prevent CI/CD failures due to path mismatches.
All scripts should import from this module instead of hardcoding paths.

Usage:
    from scripts.path_constants import DATASET_RAW, DATASET_CLEAN

    python scripts/filter_dataset_quality.py \\
      --input {DATASET_RAW} \\
      --output {DATASET_CLEAN}
"""

from __future__ import annotations

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT.parent / "output_g5"  # SecId対応データセット配置先

# Dataset paths (raw + quality pipeline). Raw must include target_* columns.
DATASET_RAW = str(OUTPUT_DIR / "ml_dataset_full_with_targets.parquet")
DATASET_CLEAN = str(OUTPUT_DIR / "ml_dataset_full_with_targets_clean.parquet")

# Backtest output paths
BACKTEST_OUTPUT_DIR = OUTPUT_DIR / "backtest"
BACKTEST_JSON = str(BACKTEST_OUTPUT_DIR / "backtest_result.json")
BACKTEST_DAILY_CSV = str(BACKTEST_OUTPUT_DIR / "backtest_daily.csv")
BACKTEST_TRADES_CSV = str(BACKTEST_OUTPUT_DIR / "backtest_trades.csv")

# Quality report paths
REPORT_DIR = OUTPUT_DIR / "reports"
QUALITY_REPORT = str(REPORT_DIR / "quality_report.json")
BACKTEST_HEALTH_REPORT = str(REPORT_DIR / "backtest_health_report.json")

# Model paths
MODEL_DIR = PROJECT_ROOT.parent / "models"
MODEL_PRUNED = str(MODEL_DIR / "apex_ranker_v0_pruned.pt")
MODEL_ENHANCED = str(MODEL_DIR / "apex_ranker_v0_enhanced.pt")

# Config paths
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_PRUNED = str(CONFIG_DIR / "v0_pruned.yaml")
CONFIG_ENHANCED = str(CONFIG_DIR / "v0_base.yaml")

# Environment variable overrides (optional)
DATASET_RAW = os.getenv("DATASET_RAW", DATASET_RAW)
DATASET_CLEAN = os.getenv("DATASET_CLEAN", DATASET_CLEAN)


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for path in [OUTPUT_DIR, BACKTEST_OUTPUT_DIR, REPORT_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_all_paths() -> dict[str, str]:
    """Get all path constants as a dictionary (for debugging/logging)."""
    return {
        "DATASET_RAW": DATASET_RAW,
        "DATASET_CLEAN": DATASET_CLEAN,
        "BACKTEST_JSON": BACKTEST_JSON,
        "BACKTEST_DAILY_CSV": BACKTEST_DAILY_CSV,
        "BACKTEST_TRADES_CSV": BACKTEST_TRADES_CSV,
        "QUALITY_REPORT": QUALITY_REPORT,
        "BACKTEST_HEALTH_REPORT": BACKTEST_HEALTH_REPORT,
        "MODEL_PRUNED": MODEL_PRUNED,
        "MODEL_ENHANCED": MODEL_ENHANCED,
        "CONFIG_PRUNED": CONFIG_PRUNED,
        "CONFIG_ENHANCED": CONFIG_ENHANCED,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("📁 APEX-Ranker Path Constants")
    print("=" * 70)
    for name, path in get_all_paths().items():
        exists = "✅" if Path(path).exists() else "❌"
        print(f"{exists} {name:25} = {path}")
    print("=" * 70)
