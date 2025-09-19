#!/bin/bash
# GPU-ETL対応データセット生成用ラッパースクリプト
#
# 使用方法:
#   ./scripts/run_dataset_gpu.sh --start-date 2025-03-19 --end-date 2025-09-19
#   ./scripts/run_dataset_gpu.sh  # デフォルト: 過去6ヶ月

set -e

# プロジェクトルートへ移動
cd "$(dirname "$0")/.."

# .envファイルを読み込み
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# GPU-ETL環境変数を強制設定（.envの値を上書き）
export REQUIRE_GPU=1
export USE_GPU_ETL=1
export RMM_POOL_SIZE=${RMM_POOL_SIZE:-70GB}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=src

# デフォルトの日付範囲（6ヶ月）
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "6 months ago" +%Y-%m-%d)

# コマンドライン引数を処理
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=== GPU-ETL Dataset Generation ==="
echo "GPU Device: $CUDA_VISIBLE_DEVICES"
echo "RMM Pool Size: $RMM_POOL_SIZE"
echo "Date Range: $START_DATE to $END_DATE"
echo "=================================="

# GPU使用状況を表示
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# run_full_dataset.pyを実行（--gpu-etlフラグを自動付与）
python scripts/pipelines/run_full_dataset.py \
    --jquants \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --gpu-etl \
    "${ARGS[@]}"

echo "=== Dataset generation completed ==="