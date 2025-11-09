#!/bin/bash
# APEX Ranker Day-1 Operations Script
# Usage:
#   apex-ranker/scripts/day1_operations.sh premarket    # T-30 to T-10 (before market open)
#   apex-ranker/scripts/day1_operations.sh order        # T time (generate rebalance orders)
#   apex-ranker/scripts/day1_operations.sh eod          # End of day (logging & monitoring)
#
# Cron setup (recommended):
#   30 8 * * * /workspace/gogooku3/apex-ranker/scripts/day1_operations.sh premarket
#   0 9 * * * /workspace/gogooku3/apex-ranker/scripts/day1_operations.sh order
#   0 16 * * * /workspace/gogooku3/apex-ranker/scripts/day1_operations.sh eod

set -euo pipefail

# Configuration
WORKSPACE="/workspace/gogooku3"
API_URL="http://localhost:8000"
LOG_DIR="/var/log/apex-ranker"
PREDICTIONS_DIR="${LOG_DIR}/predictions"
TRADE_LOG="${LOG_DIR}/trades.jsonl"
MONITORING_LOG="${LOG_DIR}/monitoring.jsonl"

# Dynamic parameters (overridable via environment variables)
DEFAULT_TOP_K="${APEX_TOP_K:-35}"
REQUEST_TOP_K="${APEX_REQUEST_TOP_K:-${DEFAULT_TOP_K}}"
DEFAULT_HORIZON="${APEX_PREDICTION_HORIZON:-20}"
SUPPLY_MIN_RATIO="${APEX_SUPPLY_MIN_RATIO:-1.5}"
SUPPLY_MIN_ABS="${APEX_SUPPLY_MIN_ABS:-0}"
SECTOR_LIMIT_RATIO="${APEX_SECTOR_LIMIT_RATIO:-0.4286}"
TURNOVER_ESTIMATE="${APEX_TURNOVER_ESTIMATE:-0.35}"
COST_PER_SIDE_BPS="${APEX_COST_PER_SIDE_BPS:-10}"
COST_ALERT_THRESHOLD_BPS="${APEX_COST_ALERT_THRESHOLD_BPS:-20}"
EOD_FINGERPRINT_COUNT="${APEX_EOD_FINGERPRINT_COUNT:-5}"

# Sanitize numeric parameters
if ! [[ "${DEFAULT_TOP_K}" =~ ^[0-9]+$ ]]; then
    DEFAULT_TOP_K=35
fi

if ! [[ "${REQUEST_TOP_K}" =~ ^[0-9]+$ ]]; then
    REQUEST_TOP_K="${DEFAULT_TOP_K}"
fi

if ! [[ "${DEFAULT_HORIZON}" =~ ^[0-9]+$ ]]; then
    DEFAULT_HORIZON=20
fi

if ! [[ "${SUPPLY_MIN_ABS}" =~ ^[0-9]+$ ]]; then
    SUPPLY_MIN_ABS=0
fi

if ! [[ "${EOD_FINGERPRINT_COUNT}" =~ ^[0-9]+$ ]]; then
    EOD_FINGERPRINT_COUNT=5
fi

if ! [[ "${SUPPLY_MIN_RATIO}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    SUPPLY_MIN_RATIO=1.5
fi

if ! [[ "${SECTOR_LIMIT_RATIO}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    SECTOR_LIMIT_RATIO=0.4286
fi

if ! [[ "${TURNOVER_ESTIMATE}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    TURNOVER_ESTIMATE=0.35
fi

if ! [[ "${COST_PER_SIDE_BPS}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    COST_PER_SIDE_BPS=10
fi

if ! [[ "${COST_ALERT_THRESHOLD_BPS}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    COST_ALERT_THRESHOLD_BPS=20
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

resolve_top_k_from_predictions() {
    local predictions_json=$1
    local resolved

    resolved=$(printf '%s\n' "${predictions_json}" | jq -r '.metadata.target_top_k // .metadata.top_k // empty' 2>/dev/null || true)

    if [[ -n "${resolved}" && "${resolved}" != "null" && "${resolved}" =~ ^[0-9]+$ ]]; then
        echo "${resolved}"
    else
        echo "${DEFAULT_TOP_K}"
    fi
}

resolve_top_k_from_file() {
    local predictions_file=$1
    local resolved

    if [ ! -f "${predictions_file}" ]; then
        echo "${DEFAULT_TOP_K}"
        return
    fi

    resolved=$(jq -r '.metadata.target_top_k // .metadata.top_k // empty' "${predictions_file}" 2>/dev/null || true)

    if [[ -n "${resolved}" && "${resolved}" != "null" && "${resolved}" =~ ^[0-9]+$ ]]; then
        echo "${resolved}"
    else
        echo "${DEFAULT_TOP_K}"
    fi
}

calculate_min_supply() {
    local top_k=$1

    python - <<PY
import math

top_k = int("${top_k}")
ratio = float("${SUPPLY_MIN_RATIO}")
absolute = int("${SUPPLY_MIN_ABS}")

minimum = math.ceil(top_k * ratio)
if absolute > 0:
    minimum = max(minimum, absolute)

print(minimum)
PY
}

calculate_sector_threshold() {
    local top_k=$1

    python - <<PY
import math

top_k = max(int("${top_k}"), 0)
ratio = float("${SECTOR_LIMIT_RATIO}")

if top_k == 0 or ratio <= 0:
    print(0)
else:
    threshold = math.ceil(top_k * ratio)
    threshold = max(threshold, 1)
    print(threshold)
PY
}

check_health() {
    log_info "Checking API health..."
    HEALTH=$(curl -s "${API_URL}/health" || echo '{"status":"error"}')
    STATUS=$(echo "${HEALTH}" | jq -r '.status')

    if [ "${STATUS}" != "ok" ]; then
        log_error "API health check failed: ${HEALTH}"
        return 1
    fi

    log_info "✅ API health: OK"
    return 0
}

get_predictions() {
    local date=$1
    local output_file=$2

    log_info "Requesting predictions for ${date}..."

    RESPONSE=$(curl -s -X POST "${API_URL}/predict" \
        -H "Content-Type: application/json" \
        -d "{\"date\": \"${date}\", \"top_k\": ${REQUEST_TOP_K}, \"horizon\": ${DEFAULT_HORIZON}}" \
        || echo '{"error":"request_failed"}')

    # Save response
    echo "${RESPONSE}" | jq '.' > "${output_file}"

    # Validate response
    if echo "${RESPONSE}" | jq -e '.predictions' > /dev/null 2>&1; then
        log_info "✅ Predictions received: $(echo "${RESPONSE}" | jq '.predictions | length') stocks"
        echo "${RESPONSE}"
    else
        log_error "Invalid prediction response"
        return 1
    fi
}

validate_supply() {
    local predictions=$1

    log_info "Validating supply constraints..."

    # Extract metadata
    SELECTED_COUNT=$(echo "${predictions}" | jq -r '.metadata.selected_count // 0')
    FALLBACK_USED=$(echo "${predictions}" | jq -r '.metadata.fallback_used // false')

    TOP_K=$(resolve_top_k_from_predictions "${predictions}")
    MINIMUM_SUPPLY=$(calculate_min_supply "${TOP_K}")

    # Check k_min constraint dynamically
    if [ "${SELECTED_COUNT}" -lt "${MINIMUM_SUPPLY}" ]; then
        log_error "Supply shortage: selected_count=${SELECTED_COUNT} (expected ≥${MINIMUM_SUPPLY} for top_k=${TOP_K})"
        return 1
    fi

    log_info "✅ Supply validation passed: selected_count=${SELECTED_COUNT}"

    if [ "${FALLBACK_USED}" = "true" ]; then
        log_warn "Fallback mode activated (expected for k_ratio=0.6)"
    fi

    return 0
}

check_sector_concentration() {
    local predictions=$1

    log_info "Checking sector concentration..."

    local top_k
    top_k=$(resolve_top_k_from_predictions "${predictions}")
    local sector_threshold
    sector_threshold=$(calculate_sector_threshold "${top_k}")

    # Extract top sectors (requires sector info in predictions)
    # This is a simplified check - actual implementation needs sector mapping
    SECTOR_COUNTS=$(echo "${predictions}" | jq -r --argjson top_k "${top_k}" '
        (.predictions[:$top_k] // [])
        | group_by(.sector // "UNKNOWN")
        | map({sector: .[0].sector, count: length})
        | sort_by(-.count)
        | .[0:3]
    ')

    echo "${SECTOR_COUNTS}" | jq -r '.[] | "\(.sector): \(.count) stocks"'

    # Warning if top sector exceeds configured threshold
    MAX_SECTOR_COUNT=$(echo "${SECTOR_COUNTS}" | jq -r '.[0].count // 0')
    if [ "${MAX_SECTOR_COUNT}" = "null" ] || [ -z "${MAX_SECTOR_COUNT}" ]; then
        MAX_SECTOR_COUNT=0
    fi

    if [ "${sector_threshold}" -gt 0 ] && [ "${MAX_SECTOR_COUNT}" -gt "${sector_threshold}" ]; then
        log_warn "High sector concentration: ${MAX_SECTOR_COUNT} stocks in top sector (threshold: ${sector_threshold})"
    fi
}

estimate_transaction_costs() {
    local predictions=$1

    log_info "Estimating transaction costs..."

    # Simplified cost estimation (configurable turnover and cost)
    ESTIMATED_COST_BPS=$(awk -v turnover="${TURNOVER_ESTIMATE}" -v bps="${COST_PER_SIDE_BPS}" 'BEGIN { printf "%.2f", turnover * bps * 2 }')

    log_info "Estimated transaction cost: ${ESTIMATED_COST_BPS} bps (threshold: ${COST_ALERT_THRESHOLD_BPS} bps)"

    if awk -v est="${ESTIMATED_COST_BPS}" -v thr="${COST_ALERT_THRESHOLD_BPS}" 'BEGIN { exit(est > thr ? 0 : 1) }'; then
        log_warn "Estimated costs exceed threshold"
    fi
}

# Main operation modes

premarket_checks() {
    log_info "=== PREMARKET CHECKS (T-30 to T-10) ==="

    # 1. Health check
    if ! check_health; then
        log_error "ABORT: API health check failed"
        exit 1
    fi

    # 2. Get predictions for today (first business day of month)
    TODAY=$(date +%Y-%m-%d)
    PREDICTIONS_FILE="${PREDICTIONS_DIR}/predictions_${TODAY}.json"
    mkdir -p "${PREDICTIONS_DIR}"

    PREDICTIONS=$(get_predictions "${TODAY}" "${PREDICTIONS_FILE}")

    # 3. Validate supply
    if ! validate_supply "${PREDICTIONS}"; then
        log_error "ABORT: Supply validation failed"
        exit 1
    fi

    # 4. Check sector concentration
    check_sector_concentration "${PREDICTIONS}"

    # 5. Estimate transaction costs
    estimate_transaction_costs "${PREDICTIONS}"

    log_info "✅ Premarket checks complete. Predictions saved to: ${PREDICTIONS_FILE}"
    log_info "Review predictions and proceed with order generation at T time."
}

generate_orders() {
    log_info "=== ORDER GENERATION (T time) ==="

    TODAY=$(date +%Y-%m-%d)
    PREDICTIONS_FILE="${PREDICTIONS_DIR}/predictions_${TODAY}.json"

    if [ ! -f "${PREDICTIONS_FILE}" ]; then
        log_error "Predictions file not found: ${PREDICTIONS_FILE}"
        log_error "Run premarket checks first"
        exit 1
    fi

    log_info "Generating rebalance orders from: ${PREDICTIONS_FILE}"

    local top_k
    top_k=$(resolve_top_k_from_file "${PREDICTIONS_FILE}")

    # Extract top-k stocks
    TOP_SELECTION=$(jq -r --argjson top_k "${top_k}" '(.predictions[:$top_k] // []) | .[] | "\(.code),\(.score),\(.rank)"' "${PREDICTIONS_FILE}")

    # Generate order template (simplified - actual implementation needs portfolio diff)
    ORDERS_FILE="${PREDICTIONS_DIR}/orders_${TODAY}.csv"

    echo "code,direction,target_weight,reason" > "${ORDERS_FILE}"

    # Equal weight for simplicity (actual: use optimizer with constraints)
    TARGET_WEIGHT=$(awk -v k="${top_k}" 'BEGIN { if (k > 0) printf "%.4f", 1.0 / k; else print "0.0000" }')

    printf '%s\n' "${TOP_SELECTION}" | while IFS=',' read -r code score rank; do
        echo "${code},BUY,${TARGET_WEIGHT},monthly_rebalance" >> "${ORDERS_FILE}"
    done

    log_info "✅ Orders generated: ${ORDERS_FILE}"
    ORDER_COUNT=$(awk 'NR>1 {count++} END {print count+0}' "${ORDERS_FILE}")
    log_info "Total orders: ${ORDER_COUNT}"
    log_info "Review orders and submit to execution system"

    # Log to trade log
    echo "{\"date\":\"${TODAY}\",\"event\":\"orders_generated\",\"count\":${ORDER_COUNT},\"file\":\"${ORDERS_FILE}\"}" >> "${TRADE_LOG}"
}

eod_monitoring() {
    log_info "=== END OF DAY MONITORING ==="

    TODAY=$(date +%Y-%m-%d)
    PREDICTIONS_FILE="${PREDICTIONS_DIR}/predictions_${TODAY}.json"

    if [ ! -f "${PREDICTIONS_FILE}" ]; then
        log_warn "No predictions file for today - was rebalance scheduled?"
        return 0
    fi

    log_info "Recording EOD metrics..."

    # Extract top 5 fingerprint
    TOP_N=$(jq -r --argjson fingerprint "${EOD_FINGERPRINT_COUNT}" '(.predictions[:$fingerprint] // []) | .[] | .code' "${PREDICTIONS_FILE}" | tr '\n' ',' | sed 's/,$//')

    # Calculate 30-day rolling Sharpe (simplified - needs historical returns)
    # This is placeholder - actual implementation needs return tracking
    SHARPE_30D="N/A (implement rolling calculation)"

    # Log monitoring metrics
    cat >> "${MONITORING_LOG}" <<EOF
{
  "date": "${TODAY}",
  "top_${EOD_FINGERPRINT_COUNT}_codes": "${TOP_N}",
  "sharpe_30d": "${SHARPE_30D}",
  "predictions_file": "${PREDICTIONS_FILE}",
  "timestamp": "$(date -Iseconds)"
}
EOF

    log_info "✅ EOD monitoring complete. Logged to: ${MONITORING_LOG}"
    log_info "Top-${EOD_FINGERPRINT_COUNT} codes: ${TOP_N}"
}

# Main execution
case "${1:-}" in
    premarket)
        premarket_checks
        ;;
    order)
        generate_orders
        ;;
    eod)
        eod_monitoring
        ;;
    *)
        echo "Usage: $0 {premarket|order|eod}"
        echo ""
        echo "Operations:"
        echo "  premarket  - Run T-30 to T-10 validation checks"
        echo "  order      - Generate rebalance orders at T time"
        echo "  eod        - End of day monitoring and logging"
        exit 1
        ;;
esac
