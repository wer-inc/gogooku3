#!/bin/bash
# Autonomous project monitoring and self-healing daemon
# Usage:
#   ./autonomous-monitor.sh                    # Run once and exit
#   ./autonomous-monitor.sh --daemon           # Run continuously (every hour)
#   ./autonomous-monitor.sh --watch            # Run continuously (every 5 minutes)
#   ./autonomous-monitor.sh --fix-now          # Fix issues immediately if found
#
# FEATURES:
#   - Periodic health checks
#   - Automatic issue detection
#   - Self-healing via Claude/Codex
#   - Alert notifications
#   - Training failure detection
#   - Disk space monitoring
#
# CRON SETUP:
#   # Check every hour and auto-fix critical issues
#   0 * * * * cd /workspace/gogooku3 && ./tools/autonomous-monitor.sh --fix-now >> _logs/autonomous-monitor.log 2>&1

set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

HEALTH_CHECK="${PROJECT_ROOT}/tools/project-health-check.sh"
CLAUDE_LAUNCHER="${PROJECT_ROOT}/tools/claude-code.sh"
LOG_DIR="_logs/autonomous-monitor"
ALERT_FILE="${LOG_DIR}/alerts.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Colors
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_GREEN='\033[0;32m'
COLOR_BLUE='\033[0;34m'
COLOR_RESET='\033[0m'

# Logging functions
log_info() {
    echo -e "${COLOR_BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${COLOR_RESET}"
}

log_warn() {
    echo -e "${COLOR_YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1${COLOR_RESET}"
}

log_error() {
    echo -e "${COLOR_RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${COLOR_RESET}"
}

log_success() {
    echo -e "${COLOR_GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1${COLOR_RESET}"
}

# Alert function
send_alert() {
    local severity=$1
    local message=$2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$severity] $message" >> "$ALERT_FILE"
    log_warn "ALERT: $message"
}

# Health check wrapper
run_health_check() {
    log_info "Running project health check..."

    if [ ! -x "$HEALTH_CHECK" ]; then
        log_error "Health check script not found or not executable: $HEALTH_CHECK"
        return 1
    fi

    set +e
    "$HEALTH_CHECK" > /dev/null 2>&1
    local exit_code=$?
    set -e

    return $exit_code
}

# Get health report summary
get_health_summary() {
    local report_file=$(ls -t _logs/health-checks/health-check-*.json 2>/dev/null | head -1)

    if [ -z "$report_file" ] || [ ! -f "$report_file" ]; then
        echo "No health report found"
        return 1
    fi

    if command -v jq &> /dev/null; then
        local critical=$(jq -r '.summary.critical_issues // 0' "$report_file")
        local warnings=$(jq -r '.summary.warnings // 0' "$report_file")
        local recommendations=$(jq -r '.summary.recommendations // 0' "$report_file")

        echo "Critical: $critical, Warnings: $warnings, Recommendations: $recommendations"
        echo "$report_file"
    else
        echo "jq not available - cannot parse report"
        echo "$report_file"
    fi
}

# Auto-fix function
auto_fix() {
    log_info "Triggering autonomous fix via Claude Code..."

    # Check if Claude is available
    if ! command -v claude &> /dev/null; then
        log_error "Claude not installed - cannot auto-fix"
        send_alert "ERROR" "Auto-fix failed: Claude not installed"
        return 1
    fi

    # Launch Claude in non-interactive mode with autonomous prompt
    local fix_log="${LOG_DIR}/auto-fix-$(date +%Y%m%d-%H%M%S).log"

    log_info "Fix log: $fix_log"

    # Use --print mode for non-interactive execution
    local autonomous_prompt="🚨 AUTONOMOUS MONITORING: Issues detected in periodic health check.

Your mission:
1. Run health check: tools/project-health-check.sh
2. Read the latest health report
3. Fix ALL critical issues (P0 priority)
4. Address warnings (P1 priority)
5. Verify fixes by re-running health check
6. Report completion status in JSON format

Execute autonomously and report results."

    set +e
    claude --print --permission-mode bypassPermissions "$autonomous_prompt" > "$fix_log" 2>&1
    local claude_exit=$?
    set -e

    if [ $claude_exit -eq 0 ]; then
        log_success "Auto-fix completed successfully"
        send_alert "SUCCESS" "Auto-fix completed - check $fix_log"
    else
        log_error "Auto-fix failed (exit code: $claude_exit)"
        send_alert "ERROR" "Auto-fix failed - check $fix_log"
    fi

    return $claude_exit
}

# Check for training failures
check_training_status() {
    log_info "Checking training status..."

    # Check for recent training logs with errors
    if [ -d "_logs/training" ]; then
        local recent_errors=$(find _logs/training -name "*.log" -mtime -1 -exec grep -l "ERROR\|CRITICAL\|OOM\|CUDA error" {} \; 2>/dev/null | wc -l)

        if [ "$recent_errors" -gt 0 ]; then
            log_warn "Found $recent_errors training log(s) with errors in last 24 hours"
            send_alert "WARNING" "Training errors detected in $recent_errors log file(s)"
            return 1
        fi
    fi

    return 0
}

# Check disk space
check_disk_space() {
    log_info "Checking disk space..."

    local usage=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $5}' | sed 's/%//')

    if [ "$usage" -gt 90 ]; then
        log_error "Disk usage critical: ${usage}%"
        send_alert "CRITICAL" "Disk usage at ${usage}% - cleanup required"
        return 2
    elif [ "$usage" -gt 80 ]; then
        log_warn "Disk usage high: ${usage}%"
        send_alert "WARNING" "Disk usage at ${usage}%"
        return 1
    else
        log_info "Disk usage OK: ${usage}%"
    fi

    return 0
}

# Main monitoring function
monitor_once() {
    log_info "========================================"
    log_info "Autonomous monitoring check started"
    log_info "========================================"

    # Run health check
    run_health_check
    local health_status=$?

    # Get summary
    local summary=$(get_health_summary)
    log_info "Health status: $summary"

    # Check training
    check_training_status || true

    # Check disk space
    check_disk_space || true

    # Determine action based on health status
    case $health_status in
        0)
            log_success "Project healthy - no action needed"
            ;;
        1)
            log_warn "Warnings detected - consider fixing"
            send_alert "INFO" "Project has warnings: $summary"

            if [ "${FIX_NOW:-false}" = true ]; then
                auto_fix
            fi
            ;;
        2)
            log_error "Critical issues detected - action required"
            send_alert "CRITICAL" "Project has critical issues: $summary"

            if [ "${FIX_NOW:-false}" = true ]; then
                auto_fix
            else
                log_info "Auto-fix not enabled - use --fix-now to enable"
            fi
            ;;
        *)
            log_error "Unknown health status: $health_status"
            ;;
    esac

    log_info "========================================"
    log_info "Monitoring check completed"
    log_info "========================================"
}

# Main execution
MODE="${1:-once}"

case "$MODE" in
    --daemon)
        log_info "Starting autonomous monitor daemon (hourly checks)"
        while true; do
            monitor_once
            log_info "Sleeping for 1 hour..."
            sleep 3600
        done
        ;;

    --watch)
        log_info "Starting autonomous monitor (5-minute checks)"
        while true; do
            monitor_once
            log_info "Sleeping for 5 minutes..."
            sleep 300
        done
        ;;

    --fix-now)
        FIX_NOW=true
        monitor_once
        ;;

    --help)
        cat << EOF
Autonomous Project Monitor

Usage:
  $0                    # Run once and exit
  $0 --daemon           # Run continuously (every hour)
  $0 --watch            # Run continuously (every 5 minutes)
  $0 --fix-now          # Fix issues immediately if found

Cron setup (hourly auto-fix):
  0 * * * * cd /workspace/gogooku3 && ./tools/autonomous-monitor.sh --fix-now >> _logs/autonomous-monitor.log 2>&1

Log files:
  ${LOG_DIR}/auto-fix-*.log       # Auto-fix execution logs
  ${ALERT_FILE}                   # Alert history

EOF
        ;;

    *)
        monitor_once
        ;;
esac
