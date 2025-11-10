#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="$ROOT/logs"
usage() {
  cat <<'HELP'
Usage: scripts/show_logs.sh [chunk|dataset|health] [--tail N]
  chunk           show chunk builder log (default latest)
  dataset         show dataset builder log (default latest)
  health          list health-check reports
Options:
  --tail N        pass -n N to tail (default -n 100)
HELP
}
if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi
mode=$1; shift
TAIL_ARGS=("-n" "100")
if [[ $# -ge 2 && $1 == "--tail" ]]; then
  TAIL_ARGS=("-n" "$2")
  shift 2
fi
case "$mode" in
  chunk)
    target="$LOG_ROOT/chunks/latest.log"
    ;;
  dataset)
    target="$LOG_ROOT/dataset/latest.log"
    ;;
  health)
    ls -1 "$LOG_ROOT/health-checks" | sort
    exit 0
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
if [[ ! -f "$target" ]]; then
  echo "Log file not found: $target" >&2
  exit 1
fi
tail "${TAIL_ARGS[@]}" "$target"
