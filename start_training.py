#!/usr/bin/env python3
"""
ATFT-GAT-FANモデルのトレーニング起動スクリプト
"""

import os
import sys
from pathlib import Path

# 環境変数の設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0を使用
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# データセットのパス
DATASET_PATH = "output/ml_dataset_20200906_20250906_20250906_215603_full.parquet"

print("=" * 80)
print("🚀 ATFT-GAT-FANモデルトレーニング")
print("=" * 80)
print(f"\n📊 データセット: {DATASET_PATH}")
print(f"🖥️  GPU: NVIDIA A100 80GB")
print(f"💾 メモリ: 216GB")
print("\n" + "=" * 80)

# トレーニングスクリプトの確認
training_scripts = [
    "scripts/integrated_ml_training_pipeline.py",
    "scripts/train_atft.py",
    "scripts/run_safe_training.py"
]

print("\n利用可能なトレーニングスクリプト:")
for script in training_scripts:
    if Path(script).exists():
        print(f"  ✅ {script}")
    else:
        print(f"  ❌ {script} (not found)")

print("\n" + "=" * 80)
print("\n推奨実行コマンド:")
print("\n1. 🧪 スモークテスト（1エポック、少量データ）:")
print(f"   python scripts/integrated_ml_training_pipeline.py \\")
print(f"     --data-path {DATASET_PATH} \\")
print(f"     --max-epochs 1 \\")
print(f"     --batch-size 256 \\")
print(f"     --sample-size 10000")

print("\n2. 🎯 本番トレーニング（フル）:")
print(f"   python scripts/integrated_ml_training_pipeline.py \\")
print(f"     --data-path {DATASET_PATH} \\")
print(f"     --max-epochs 50 \\")
print(f"     --batch-size 1024 \\")
print(f"     --early-stopping-patience 10")

print("\n3. 🛡️ Safe Training Pipeline（推奨）:")
print(f"   python scripts/run_safe_training.py \\")
print(f"     --data-dir output \\")
print(f"     --n-splits 3 \\")
print(f"     --memory-limit 32")

print("\n" + "=" * 80)