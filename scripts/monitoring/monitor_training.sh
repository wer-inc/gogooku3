#!/bin/bash
set -euo pipefail

clear
echo "=== GPU Training Monitor ==="

LOG_WRAP="_logs/train_gpu_latest/latest.log"
PID_FILE="_logs/train_gpu_latest/latest.pid"
ML_LOG="logs/ml_training.log"

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE" 2>/dev/null || true)
fi
if [ -z "${PID:-}" ]; then
  PID=$(pgrep -af 'python.*train_atft\.py' | head -1 | awk '{print $1}' || true)
fi

while true; do
  clear
  echo "=== GPU Training Monitor ==="
  date '+Time: %Y-%m-%d %H:%M:%S'
  echo "================================"

  echo -e "\n📊 GPU Status:"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
      --format=csv,noheader,nounits | \
      awk -F', ' '{printf "  GPU %s: %s\n    Util: %s%% | Mem: %s/%s MB | Temp: %s°C\n", $1, $2, $3, $4, $5, $6}'
    if [ -n "${PID:-}" ]; then
      echo -e "\n  Compute Apps (PID match):"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | \
        awk -v pid="$PID" -F', ' '$1==pid {printf "    PID %s | %s | %s\n", $1, $2, $3}' || true
    fi
  else
    echo "  nvidia-smi not available"
  fi

  echo -e "\n📦 Training Process:"
  if [ -n "${PID:-}" ] && ps -p "$PID" >/dev/null 2>&1; then
    ps -p "$PID" -o pid,pcpu,pmem,etime,comm --no-headers | \
      awk '{printf "  PID: %s | CPU: %s%% | MEM: %s%% | Time: %s | CMD: %s\n", $1, $2, $3, $4, $5}'
  else
    echo "  ⚠️ No active training process found"
  fi

  echo -e "\n📈 Latest Progress:"
  if [ -f "$ML_LOG" ]; then
    rg -n "Epoch|it/s|loss|Using device|GPU:" "$ML_LOG" | tail -n 3 | sed 's/^/  /' || true
  else
    echo "  (ml log not yet created)"
  fi

  echo -e "\n🧾 Wrapper Log (last 2 lines):"
  if [ -f "$LOG_WRAP" ]; then
    tail -n 2 "$LOG_WRAP" | sed 's/^/  /'
  else
    echo "  (wrapper log not found)"
  fi

  echo -e "\n================================"
  echo "Press Ctrl+C to exit"
  sleep 5
done
