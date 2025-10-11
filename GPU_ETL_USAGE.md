# GPU-ETL 使用ガイド

## 概要
GPU-ETL機能により、データセット生成時にGPU（RAPIDS/cuDF）を使用してETL処理を高速化できます。
A100 80GB GPUとRAPIDSライブラリを使用して、大規模データの処理を効率化します。

## 🚀 GPU-ETLはデフォルトで有効

**v2.1.0以降、GPU-ETLはデフォルトで有効になっています。**

## 使用方法

### 1. 通常のデータセット生成（GPU-ETL自動有効）
```bash
# GPU-ETLが自動的に有効（デフォルト）
python scripts/pipelines/run_full_dataset.py \
  --jquants --start-date 2025-03-19 --end-date 2025-09-19
```

### 2. 明示的にGPU-ETLを無効化する場合
```bash
# --no-gpu-etlオプションでCPUのみ使用
python scripts/pipelines/run_full_dataset.py \
  --jquants --start-date 2025-03-19 --end-date 2025-09-19 \
  --no-gpu-etl
```

### 3. 専用ラッパースクリプト使用
```bash
# GPU設定を強制適用するラッパースクリプト
./scripts/run_dataset_gpu.sh --start-date 2025-03-19 --end-date 2025-09-19

# デフォルトで過去6ヶ月
./scripts/run_dataset_gpu.sh
```

### 4. Makefileターゲット使用
```bash
# GPU-ETL有効でデータセット生成
make dataset-full-gpu START=2025-03-19 END=2025-09-19

# 通常のデータセット生成（GPU-ETLはデフォルトで有効）
make dataset-full START=2025-03-19 END=2025-09-19
```

### 5. バックグラウンド実行（推奨）
```bash
# SSH切断にも安全。ログ/ PID / PGID を保存します。
make dataset-bg

# モニタ
tail -f _logs/dataset/*.log

# 停止
kill <PID>
# またはプロセスグループごとに停止
kill -TERM -<PGID>
```

## 設定ファイル

### .env ファイル
```bash
# GPU設定（既に設定済み）
FORCE_GPU=1
REQUIRE_GPU=1
USE_GPU_ETL=1
RMM_POOL_SIZE=70GB
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### configs/gpu/gpu_etl.yaml
GPU-ETLの詳細設定を管理。通常は変更不要。

## GPU使用状況の確認

### リアルタイムモニタリング
```bash
# 別ターミナルで実行
watch -n 1 nvidia-smi

# またはGPUメモリ使用量のみ
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

### ログでの確認
成功時のログメッセージ：
- `GPU-ETL: enabled (will use RAPIDS/cuDF if available)`
- `RMM initialized without pool allocator`
- GPU-ETL関数が呼ばれた場合、実際のGPU処理が実行

## GPU-ETLで高速化される処理

1. **Cross-sectional normalization**: 日次断面での正規化
2. **Rank computation**: ランク計算
3. **Z-score calculation**: Z-スコア計算
4. **Correlation matrix**: 相関行列計算

## トラブルシューティング

### RMM初期化警告が出る場合
```
RMM init failed: module 'rmm' has no attribute 'rmm_cupy_allocator'
RMM initialized without pool allocator
```
これは正常です。pool allocatorなしでもGPU-ETLは動作します。

### GPU-ETLを強制的に有効にする
```bash
export USE_GPU_ETL=1
export RMM_POOL_SIZE=70GB
python scripts/pipelines/run_full_dataset.py --jquants ...
```

### GPU-ETLが使用されているか確認
```python
# Pythonで確認
import os
print(f"USE_GPU_ETL: {os.getenv('USE_GPU_ETL')}")

from src.utils.gpu_etl import _has_cuda
print(f"CUDA available: {_has_cuda()}")
```

## パフォーマンス目安

- **小規模データ（1-2日）**: CPUの方が高速な場合あり
- **中規模データ（1週間-1ヶ月）**: GPU-ETLで1.5-3倍高速化
- **大規模データ（6ヶ月以上）**: GPU-ETLで3-5倍高速化

## 注意事項

1. **初回実行時**: JITコンパイルのため時間がかかることがあります
2. **メモリ制限**: A100 80GBのVRAMを超えるデータは自動的にバッチ処理
3. **フォールバック**: GPU処理に失敗した場合、自動的にCPUにフォールバック

## 推奨設定

### 6ヶ月データセット生成（本番用）
```bash
make dataset-full-gpu START=2025-03-19 END=2025-09-19
```

### 研究用データセット（インデックス付き）
```bash
python scripts/pipelines/run_full_dataset.py \
  --jquants --start-date 2025-03-19 --end-date 2025-09-19 \
  --config configs/pipeline/research_full_indices.yaml
# GPU-ETLはデフォルトで有効
```
