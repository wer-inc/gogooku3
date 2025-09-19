#!/bin/bash
# Launches train_gpu_latest.sh in the background using nohup/setsid.
# Creates per-run log and pid files under _logs/train_gpu_latest/.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
LOG_ROOT="${REPO_ROOT}/_logs/train_gpu_latest"
mkdir -p "${LOG_ROOT}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_ROOT}/train_${STAMP}.log"
PID_FILE="${LOG_ROOT}/train_${STAMP}.pid"

CMD=("${SCRIPT_DIR}/train_gpu_latest.sh" "$@")

# Ensure unbuffered output so logs show progress promptly
export PYTHONUNBUFFERED=1

# Launch in background detached from terminal
nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
PID=$!

echo ${PID} >"${PID_FILE}"
ln -sf "${LOG_FILE}" "${LOG_ROOT}/latest.log"
ln -sf "${PID_FILE}" "${LOG_ROOT}/latest.pid"

cat <<EOF
Launched train_gpu_latest.sh (PID ${PID}).
Logs      : ${LOG_FILE}
PID file  : ${PID_FILE}
Tail logs : tail -f ${LOG_FILE}
Progress  : ./scripts/monitor_training_progress.py
To stop   : kill \
$(cat "${PID_FILE}")
EOF
