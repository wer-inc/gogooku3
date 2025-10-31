#!/bin/bash
# Automated training monitor for 50-epoch validation run
# Usage: ./scripts/monitor_training.sh [log_file]

set -euo pipefail

# Configuration
LOG_FILE="${1:-_logs/training/prod_validation_50ep_*.log}"
MONITOR_INTERVAL=30  # seconds
DASHBOARD_FILE="docs/TRAINING_DASHBOARD.md"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to get latest log file
get_latest_log() {
    ls -t ${LOG_FILE} 2>/dev/null | head -1 || echo ""
}

# Function to extract metrics from log
extract_metrics() {
    local logfile="$1"

    # Get PID from file
    local pid_file="_logs/training/prod_validation_50ep.pid"
    local pid=""
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file" 2>/dev/null || echo "")
    fi

    # Process status
    local process_status="Unknown"
    local cpu_usage="N/A"
    local mem_usage="N/A"
    local runtime="N/A"
    if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
        process_status="Running"
        cpu_usage=$(ps -p "$pid" -o %cpu --no-headers 2>/dev/null | xargs)
        mem_usage=$(ps -p "$pid" -o %mem --no-headers 2>/dev/null | xargs)
        runtime=$(ps -p "$pid" -o etime --no-headers 2>/dev/null | xargs)
    else
        process_status="Completed/Stopped"
    fi

    # Latest epoch
    local latest_epoch=$(grep -E "Epoch [0-9]+/50" "$logfile" 2>/dev/null | tail -1 | grep -oP "Epoch \K[0-9]+" || echo "0")

    # Latest Sharpe ratio
    local latest_sharpe=$(grep "Achieved Sharpe Ratio:" "$logfile" 2>/dev/null | tail -1 | grep -oP ": \K[0-9.]+" || echo "N/A")

    # Latest validation loss
    local latest_val_loss=$(grep -E "Val Loss=" "$logfile" 2>/dev/null | tail -1 | grep -oP "Val Loss=\K[0-9.]+" || echo "N/A")

    # Degeneracy reset count
    local reset_count=$(grep -c "DEGENERACY-GUARD.*reset applied" "$logfile" 2>/dev/null || echo "0")

    # Gradient warnings
    local grad_warnings=$(grep -c "GRAD-MONITOR.*< 1e-07" "$logfile" 2>/dev/null || echo "0")

    # GPU memory usage
    local gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
    local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")

    # Output metrics as JSON-like format
    echo "PID=$pid"
    echo "PROCESS_STATUS=$process_status"
    echo "CPU_USAGE=$cpu_usage"
    echo "MEM_USAGE=$mem_usage"
    echo "RUNTIME=$runtime"
    echo "LATEST_EPOCH=$latest_epoch"
    echo "LATEST_SHARPE=$latest_sharpe"
    echo "LATEST_VAL_LOSS=$latest_val_loss"
    echo "RESET_COUNT=$reset_count"
    echo "GRAD_WARNINGS=$grad_warnings"
    echo "GPU_MEM=$gpu_mem"
    echo "GPU_UTIL=$gpu_util"
}

# Function to generate dashboard markdown
generate_dashboard() {
    local logfile="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S UTC')

    # Extract metrics
    local metrics=$(extract_metrics "$logfile")

    # Parse metrics into variables
    local pid=$(echo "$metrics" | grep "PID=" | cut -d= -f2)
    local process_status=$(echo "$metrics" | grep "PROCESS_STATUS=" | cut -d= -f2)
    local cpu_usage=$(echo "$metrics" | grep "CPU_USAGE=" | cut -d= -f2)
    local mem_usage=$(echo "$metrics" | grep "MEM_USAGE=" | cut -d= -f2)
    local runtime=$(echo "$metrics" | grep "RUNTIME=" | cut -d= -f2)
    local latest_epoch=$(echo "$metrics" | grep "LATEST_EPOCH=" | cut -d= -f2)
    local latest_sharpe=$(echo "$metrics" | grep "LATEST_SHARPE=" | cut -d= -f2)
    local latest_val_loss=$(echo "$metrics" | grep "LATEST_VAL_LOSS=" | cut -d= -f2)
    local reset_count=$(echo "$metrics" | grep "RESET_COUNT=" | cut -d= -f2)
    local grad_warnings=$(echo "$metrics" | grep "GRAD_WARNINGS=" | cut -d= -f2)
    local gpu_mem=$(echo "$metrics" | grep "GPU_MEM=" | cut -d= -f2)
    local gpu_util=$(echo "$metrics" | grep "GPU_UTIL=" | cut -d= -f2)

    # Calculate progress percentage
    local progress=$((latest_epoch * 100 / 50))

    # Determine status emoji
    local status_emoji="🔄"
    if [[ "$process_status" == "Completed/Stopped" ]]; then
        if [[ $latest_epoch -eq 50 ]]; then
            status_emoji="✅"
        else
            status_emoji="⚠️"
        fi
    fi

    # Generate dashboard
    cat > "$DASHBOARD_FILE" <<EOF
# 50-Epoch Training Dashboard $status_emoji

**Last Updated**: $timestamp
**Status**: $process_status
**Progress**: Epoch $latest_epoch/50 (${progress}%)

---

## 📊 Current Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Latest Epoch** | $latest_epoch / 50 | ${progress}% complete |
| **Sharpe Ratio** | $latest_sharpe | $([ "$latest_sharpe" != "N/A" ] && echo "Baseline: 0.0818" || echo "Pending") |
| **Val Loss** | $latest_val_loss | $([ "$latest_val_loss" != "N/A" ] && echo "Active" || echo "Pending") |
| **Degeneracy Resets** | $reset_count | $([ $reset_count -lt 5 ] && echo "✅ Excellent" || echo "⚠️ Monitor") |
| **Gradient Warnings** | $grad_warnings | $([ $grad_warnings -eq 0 ] && echo "✅ Healthy" || echo "⚠️ Check logs") |

---

## 💻 System Status

| Resource | Usage | Details |
|----------|-------|---------|
| **Process (PID)** | $pid | $process_status |
| **CPU** | ${cpu_usage}% | Multi-threaded training |
| **Memory** | ${mem_usage}% | RAM usage |
| **Runtime** | $runtime | Elapsed time |
| **GPU Memory** | ${gpu_mem} MB | NVIDIA A100-SXM4-80GB |
| **GPU Utilization** | ${gpu_util}% | Training active |

---

## 🎯 Validation Objectives

- [$([ $latest_epoch -ge 1 ] && echo "x" || echo " ")] Training started successfully
- [$([ $latest_epoch -ge 10 ] && echo "x" || echo " ")] Reached epoch 10 (baseline stability)
- [$([ $latest_epoch -ge 25 ] && echo "x" || echo " ")] Reached epoch 25 (mid-point)
- [$([ $latest_epoch -ge 50 ] && echo "x" || echo " ")] Completed 50 epochs
- [$([ $grad_warnings -eq 0 ] && echo "x" || echo " ")] No gradient warnings
- [$([ $reset_count -lt 10 ] && echo "x" || echo " ")] Degeneracy resets under control (<10 total)

---

## 📈 Expected Progression

| Epoch Range | Expected Sharpe | Status |
|-------------|-----------------|--------|
| 1-5 | ~0.08 (baseline) | $([ $latest_epoch -ge 5 ] && echo "✅" || echo "Pending") |
| 6-20 | 0.10-0.15 | $([ $latest_epoch -ge 20 ] && echo "✅" || echo "Pending") |
| 21-50 | 0.15-0.30 | $([ $latest_epoch -ge 50 ] && echo "✅" || echo "Pending") |

**Target (120 epochs)**: 0.849

---

## 🔍 Quick Checks

### Check Training Progress
\`\`\`bash
tail -f _logs/training/prod_validation_50ep_*.log | grep -E "Epoch|Val Loss|Sharpe"
\`\`\`

### Check Gradient Health
\`\`\`bash
grep "GRAD-MONITOR" _logs/training/prod_validation_50ep_*.log | tail -10
\`\`\`

### Check Degeneracy Activity
\`\`\`bash
grep -c "DEGENERACY-GUARD.*reset applied" _logs/training/prod_validation_50ep_*.log
\`\`\`

### Monitor Process
\`\`\`bash
ps -p $pid -o pid,stat,%cpu,%mem,etime,cmd
\`\`\`

---

## ⚠️ Alerts

$(if [ $grad_warnings -gt 0 ]; then
    echo "- ⚠️ **Gradient warnings detected** ($grad_warnings). Check GRAD-MONITOR logs."
else
    echo "- ✅ No gradient warnings"
fi)

$(if [ $reset_count -gt 10 ]; then
    echo "- ⚠️ **High degeneracy reset count** ($reset_count). May need tuning."
else
    echo "- ✅ Degeneracy resets under control"
fi)

$(if [[ "$process_status" != "Running" ]] && [[ $latest_epoch -lt 50 ]]; then
    echo "- ⚠️ **Training stopped early** at epoch $latest_epoch. Check logs for errors."
fi)

---

**Log File**: \`$logfile\`
**Dashboard**: Auto-updated every 30 seconds
**Monitor Script**: \`scripts/monitor_training.sh\`
EOF

    echo "Dashboard updated: $DASHBOARD_FILE"
}

# Function to display terminal summary
display_summary() {
    local logfile="$1"
    local metrics=$(extract_metrics "$logfile")

    local pid=$(echo "$metrics" | grep "PID=" | cut -d= -f2)
    local process_status=$(echo "$metrics" | grep "PROCESS_STATUS=" | cut -d= -f2)
    local cpu_usage=$(echo "$metrics" | grep "CPU_USAGE=" | cut -d= -f2)
    local runtime=$(echo "$metrics" | grep "RUNTIME=" | cut -d= -f2)
    local latest_epoch=$(echo "$metrics" | grep "LATEST_EPOCH=" | cut -d= -f2)
    local latest_sharpe=$(echo "$metrics" | grep "LATEST_SHARPE=" | cut -d= -f2)
    local reset_count=$(echo "$metrics" | grep "RESET_COUNT=" | cut -d= -f2)
    local grad_warnings=$(echo "$metrics" | grep "GRAD_WARNINGS=" | cut -d= -f2)
    local gpu_util=$(echo "$metrics" | grep "GPU_UTIL=" | cut -d= -f2)

    clear
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║          50-Epoch Training Monitor - Live Dashboard          ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}Status:${NC} $process_status (PID: $pid)"
    echo -e "${GREEN}Runtime:${NC} $runtime | ${GREEN}CPU:${NC} ${cpu_usage}% | ${GREEN}GPU:${NC} ${gpu_util}%"
    echo ""
    echo -e "${YELLOW}═══ Progress ═══${NC}"
    echo -e "Epoch: ${BLUE}$latest_epoch${NC}/50 ($((latest_epoch * 100 / 50))%)"
    echo -e "Sharpe: ${BLUE}$latest_sharpe${NC}"
    echo ""
    echo -e "${YELLOW}═══ Health ═══${NC}"
    echo -e "Degeneracy Resets: ${BLUE}$reset_count${NC} $([ $reset_count -lt 5 ] && echo -e "${GREEN}✅${NC}" || echo -e "${RED}⚠️${NC}")"
    echo -e "Gradient Warnings: ${BLUE}$grad_warnings${NC} $([ $grad_warnings -eq 0 ] && echo -e "${GREEN}✅${NC}" || echo -e "${RED}⚠️${NC}")"
    echo ""
    echo -e "${YELLOW}═══ Controls ═══${NC}"
    echo -e "Press ${BLUE}Ctrl+C${NC} to exit monitor (training continues)"
    echo -e "Dashboard: ${BLUE}docs/TRAINING_DASHBOARD.md${NC}"
    echo ""
}

# Main monitoring loop
main() {
    echo "Starting training monitor..."

    # Get log file
    LOGFILE=$(get_latest_log)
    if [[ -z "$LOGFILE" ]]; then
        echo "Error: No log file found matching: $LOG_FILE"
        exit 1
    fi

    echo "Monitoring log: $LOGFILE"
    echo "Press Ctrl+C to exit"
    echo ""

    # Initial dashboard generation
    generate_dashboard "$LOGFILE"

    # Monitor loop
    while true; do
        display_summary "$LOGFILE"
        generate_dashboard "$LOGFILE"
        sleep $MONITOR_INTERVAL
    done
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
