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
        -d "{\"date\": \"${date}\", \"top_k\": 35, \"horizon\": 20}" \
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

    # Check k_min constraint (53 = 1.5x target_top_k=35)
    if [ "${SELECTED_COUNT}" -lt 53 ]; then
        log_error "Supply shortage: selected_count=${SELECTED_COUNT} (expected ≥53)"
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

    # Extract top sectors (requires sector info in predictions)
    # This is a simplified check - actual implementation needs sector mapping
    SECTOR_COUNTS=$(echo "${predictions}" | jq -r '
        .predictions[:35]
        | group_by(.sector // "UNKNOWN")
        | map({sector: .[0].sector, count: length})
        | sort_by(-.count)
        | .[0:3]
    ')

    echo "${SECTOR_COUNTS}" | jq -r '.[] | "\(.sector): \(.count) stocks"'

    # Warning if top sector >15 stocks (42% concentration)
    MAX_SECTOR_COUNT=$(echo "${SECTOR_COUNTS}" | jq -r '.[0].count')
    if [ "${MAX_SECTOR_COUNT}" -gt 15 ]; then
        log_warn "High sector concentration: ${MAX_SECTOR_COUNT} stocks in top sector"
    fi
}

estimate_transaction_costs() {
    local predictions=$1

    log_info "Estimating transaction costs..."

    # Simplified cost estimation (assumes 35% turnover, 10bps cost per side)
    TURNOVER=0.35
    COST_PER_SIDE_BPS=10
    ESTIMATED_COST_BPS=$(echo "${TURNOVER} * ${COST_PER_SIDE_BPS} * 2" | bc -l)

    log_info "Estimated transaction cost: ${ESTIMATED_COST_BPS} bps (threshold: 20 bps)"

    if (( $(echo "${ESTIMATED_COST_BPS} > 20" | bc -l) )); then
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

    # Extract top 35 stocks
    TOP_35=$(jq -r '.predictions[:35] | .[] | "\(.code),\(.score),\(.rank)"' "${PREDICTIONS_FILE}")

    # Generate order template (simplified - actual implementation needs portfolio diff)
    ORDERS_FILE="${PREDICTIONS_DIR}/orders_${TODAY}.csv"

    echo "code,direction,target_weight,reason" > "${ORDERS_FILE}"

    # Equal weight for simplicity (actual: use optimizer with constraints)
    TARGET_WEIGHT=$(echo "scale=4; 1.0 / 35" | bc)

    echo "${TOP_35}" | while IFS=',' read -r code score rank; do
        echo "${code},BUY,${TARGET_WEIGHT},monthly_rebalance" >> "${ORDERS_FILE}"
    done

    log_info "✅ Orders generated: ${ORDERS_FILE}"
    log_info "Total orders: $(wc -l < "${ORDERS_FILE}")"
    log_info "Review orders and submit to execution system"

    # Log to trade log
    echo "{\"date\":\"${TODAY}\",\"event\":\"orders_generated\",\"count\":35,\"file\":\"${ORDERS_FILE}\"}" >> "${TRADE_LOG}"
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
    TOP_5=$(jq -r '.predictions[:5] | .[] | .code' "${PREDICTIONS_FILE}" | tr '\n' ',' | sed 's/,$//')

    # Calculate 30-day rolling Sharpe (simplified - needs historical returns)
    # This is placeholder - actual implementation needs return tracking
    SHARPE_30D="N/A (implement rolling calculation)"

    # Log monitoring metrics
    cat >> "${MONITORING_LOG}" <<EOF
{
  "date": "${TODAY}",
  "top_5_codes": "${TOP_5}",
  "sharpe_30d": "${SHARPE_30D}",
  "predictions_file": "${PREDICTIONS_FILE}",
  "timestamp": "$(date -Iseconds)"
}
EOF

    log_info "✅ EOD monitoring complete. Logged to: ${MONITORING_LOG}"
    log_info "Top-5 codes: ${TOP_5}"
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
