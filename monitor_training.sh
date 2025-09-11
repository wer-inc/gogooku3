#!/bin/bash
# トレーニング進行状況をリアルタイムでモニタリング

echo "=========================================="
echo "トレーニング進行状況モニタリング"
echo "=========================================="

# プロセス情報
echo "📊 プロセス情報:"
ps aux | grep train_atft | grep -v grep | head -1

# GPU使用率
echo -e "\n🎮 GPU使用状況:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

# 最新のログ
echo -e "\n📈 最新のトレーニング状況:"
tail -20 logs/ml_training.log | grep -E "Epoch|Val Loss|Sharpe|Creating sequences" | tail -10

# ディスク使用量
echo -e "\n💾 ディスク使用量:"
df -h output/ | tail -1

# 推定完了時間
echo -e "\n⏰ 推定完了時間:"
PROGRESS=$(tail -1 logs/ml_training.log | grep -oP '\d+%' | tr -d '%' || echo "0")
if [ ! -z "$PROGRESS" ] && [ "$PROGRESS" -gt 0 ]; then
    ELAPSED=$(ps aux | grep train_atft | grep -v grep | awk '{print $10}' | head -1)
    echo "進捗: $PROGRESS%"
    echo "経過時間: $ELAPSED"
    # 簡単な推定
    if [ "$PROGRESS" -gt 0 ]; then
        TOTAL_MIN=$((100 * ${ELAPSED%%:*} / $PROGRESS))
        REMAIN_MIN=$(($TOTAL_MIN - ${ELAPSED%%:*}))
        echo "推定残り時間: 約${REMAIN_MIN}分"
    fi
fi

echo "=========================================="
