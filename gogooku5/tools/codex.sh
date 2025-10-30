#!/usr/bin/env bash
set -euo pipefail

declare run_mode="interactive"
if [[ ${1:-} == "--status" ]]; then
  run_mode="status"
fi

case "$run_mode" in
  status)
    echo "[gogooku5] codex status placeholder"
    echo "- dataset package: pending implementation"
    echo "- models: atft_gat_fan, apex_ranker (skeletons)"
    ;;
  *)
    echo "Launch Codex CLI integration here (TBD)."
    ;;
 esac
